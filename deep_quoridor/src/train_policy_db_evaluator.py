"""Train a neural network evaluator from a policy DB.

The model approximates the minimax lookup: given a compact game state,
predict a value in [-1, 1] (P0-absolute: 1=P0 wins, -1=P1 wins, 0=draw).

Uses NNEvaluator (and its underlying network) so that the resulting model
is compatible with the rest of the AlphaZero infrastructure.

Two metrics are logged during training:
  - MSE loss on a held-out test set
  - Move accuracy: fraction of test states where the model picks the same
    best child state as the DB
"""

import argparse
import os
import random
import sys
from dataclasses import asdict

import numpy as np
import quoridor_rs
import torch
import wandb
from agents.alphazero.alphazero import AlphaZeroParams
from agents.alphazero.nn_evaluator import NNConfig, NNEvaluator
from quoridor import ActionEncoder, Board, Player, Quoridor
from utils.subargs import parse_subargs
from utils.timer import Timer, timer

DEBUG = False

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def policy_to_str(policy, action_encoder):
    """Print non-zero entries of a policy array as human-readable actions."""
    nonzero = np.nonzero(policy)[0]
    parts = []
    for idx in nonzero:
        action = action_encoder.index_to_action(idx)
        parts.append(f"{action}: {policy[idx]:.3f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@timer("compact_state_to_game")
def compact_state_to_game(state, board_size, max_walls, max_steps):
    """Convert a compact state (Python int) to a Quoridor game object."""
    grid, player_positions, walls_remaining, old_style_walls, current_player, completed_steps = (
        quoridor_rs.compact_state_to_game_state(state, board_size, max_walls, max_steps)
    )
    board = Board.from_arrays(
        board_size,
        max_walls,
        np.asarray(grid),
        np.asarray(player_positions),
        np.asarray(walls_remaining),
        np.asarray(old_style_walls),
    )
    return Quoridor(board, Player(current_player), completed_steps=completed_steps)


def child_action_index(row, col, action_type, board_size):
    """Map (row, col, action_type) from get_compact_child_states to an ActionEncoder index."""
    wall_size = board_size - 1
    if action_type == 2:  # pawn move
        return row * board_size + col
    elif action_type == 0:  # vertical wall
        return board_size**2 + row * wall_size + col
    elif action_type == 1:  # horizontal wall
        return board_size**2 + wall_size**2 + row * wall_size + col
    raise ValueError(f"Unknown action_type {action_type}")


@timer("build_policy_from_action_values")
def build_policy_from_action_values(db, state, board_size, num_actions):
    """Build a policy vector from DB action values.

    Uses lookup_action_values which correctly handles terminal child states.
    Assigns uniform probability across all actions with the best value
    (values are from acting player's perspective, so best = max).

    Returns the policy array, or None if no valid actions.
    """
    result = db.lookup_action_values(state)
    assert result is not None

    actions, values = result
    best_value = max(values)

    policy = np.zeros(num_actions, dtype=np.float32)
    for (row, col, action_type), value in zip(actions, values):
        if value == best_value:
            idx = child_action_index(row, col, action_type, board_size)
            policy[idx] = 1.0
    policy /= policy.sum()
    return policy


@timer("fetch_batch")
def fetch_batch(db, ids, evaluator: NNEvaluator, board_size, max_walls, max_steps):
    """Fetch rows by rowid from the policy table and compute features on the fly.

    Returns a list of sample dicts compatible with NNEvaluator.compute_losses.
    Each dict has keys: input_array, value, action_mask, mcts_policy, current_player.

    mcts_policy is derived from lookup_action_values: uniform over actions
    with the best value, zero for all others.
    """
    rows = db.fetch_states_by_rowid(ids)
    num_actions = evaluator.action_encoder.num_actions
    samples = []
    for state, value in rows:
        game = compact_state_to_game(state, board_size, max_walls, max_steps)
        current_player = int(game.current_player)
        if current_player == 1:
            # DB Values are always from P1 perspective
            value = -value

        mcts_policy = build_policy_from_action_values(
            db,
            state,
            board_size,
            num_actions,
        )
        db_policy_original = mcts_policy.copy()

        # Do the rotation, in the same way AlphaZeroAgent.store_training_data() does
        game, is_rotated = evaluator.rotate_if_needed_to_point_downwards(game)
        input_array = evaluator.game_to_input_array(game)
        action_mask = game.get_action_mask()
        if is_rotated:
            mcts_policy = evaluator.rotate_policy_from_original(mcts_policy)

        if DEBUG:
            print("")
            print(game)
            print(f"Value: {value}")
            print(f"Optimal actions: {policy_to_str(mcts_policy, evaluator.action_encoder)}")
            print(f"Valid actions: {policy_to_str(action_mask, evaluator.action_encoder)}")

        samples.append(
            {
                "input_array": input_array,
                "value": value,
                "action_mask": action_mask,
                "mcts_policy": mcts_policy,
                "current_player": current_player,
                "state": state,
                "db_policy_original": db_policy_original,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------


@timer("compute_accuracy")
def compute_accuracy(
    test_ids, db, evaluator: NNEvaluator, board_size, max_walls, max_steps, num_samples, test_player=None
):
    """Fraction of sampled states where model picks the same best action as DB.

    If test_player is 0 or 1, only states where the current player matches are counted.
    """
    assert len(test_ids) > 0
    assert num_samples > 0
    sample_ids = random.sample(test_ids, min(num_samples, len(test_ids)))

    # Fetch the state blobs and values for the sampled test IDs.
    rows = db.fetch_states_by_rowid(sample_ids)

    correct = 0
    total = 0
    inaccurate_printed = 0
    for state, _value in rows:
        game = compact_state_to_game(state, board_size, max_walls, max_steps)
        current_player = int(game.current_player)
        if test_player is not None and current_player != test_player:
            continue

        db_policy = build_policy_from_action_values(
            db,
            state,
            board_size,
            evaluator.action_encoder.num_actions,
        )

        _, model_policy = evaluator.evaluate(game)

        best_db_action_prob = max(db_policy)
        model_pick = int(np.argmax(model_policy))
        if db_policy[model_pick] == best_db_action_prob:
            correct += 1
        else:
            if DEBUG and inaccurate_printed < 3:
                inaccurate_printed += 1

                print("")
                print(game)
                print(f"DB optimal actions: {policy_to_str(db_policy, evaluator.action_encoder)}")
                print(f"Model optimal actions: {policy_to_str(model_policy, evaluator.action_encoder)}")

        Timer.log_totals()
        total += 1

    return correct / total


@timer("compute_test_metrics_batched")
def compute_test_metrics_batched(
    num_states, db, evaluator: NNEvaluator, board_size, max_walls, max_steps, batch_size, test_player=None
):
    """Compute test loss and accuracy over all states by iterating in batches.

    Returns (policy_loss, value_loss, total_loss, accuracy) as floats.
    """
    total_pol = total_val = total_tot = 0.0
    correct = total = 0

    evaluator.network.eval()
    with torch.no_grad():
        for start in range(1, num_states + 1, batch_size):
            batch_ids = list(range(start, min(start + batch_size, num_states + 1)))
            samples = fetch_batch(db, batch_ids, evaluator, board_size, max_walls, max_steps)
            if test_player is not None:
                samples = [s for s in samples if s["current_player"] == test_player]
            assert samples is not None, "fetch_batch should never return None"

            n = len(samples)
            pol, val, tot = evaluator.compute_losses(samples)
            total_pol += pol.item() * n
            total_val += val.item() * n
            total_tot += tot.item() * n

            for s in samples:
                original_game = compact_state_to_game(s["state"], board_size, max_walls, max_steps)
                _, model_policy = evaluator.evaluate(original_game)
                db_policy = s["db_policy_original"]
                best_db_prob = max(db_policy)
                model_pick = int(np.argmax(model_policy))
                if db_policy[model_pick] == best_db_prob:
                    correct += 1
                total += 1

    evaluator.network.train()

    assert total > 0, "No test samples found (check test_player filter?)"
    return total_pol / total, total_val / total, total_tot / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_TEST_SIZE = 10000


def parse_args():
    p = argparse.ArgumentParser(description="Train a neural network evaluator from a policy DB.")
    p.add_argument("db_path", help="Path to .sqlite policy DB")
    p.add_argument(
        "-p",
        "--params",
        type=str,
        default="",
        help="AlphaZero params in subargs form (e.g. nn_type=mlp,learning_rate=0.001)",
    )
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="If set, test on all states using sequential batches of this size (ignores --test-fraction and --exclude-test-set)",
    )
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--accuracy-states", type=int, default=200)
    p.add_argument("--output", default="evaluator.pt")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument(
        "--test-player",
        type=int,
        default=None,
        choices=[1, 2],
        help="Only evaluate test loss and accuracy on positions where this player is to move (1 or 2)",
    )
    p.add_argument(
        "--exclude-test-set",
        action="store_true",
        default=False,
        help="Exclude test set IDs from training batches (default: allow overlap)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debug output",
    )
    p.add_argument(
        "-w",
        "--wandb",
        nargs="?",
        const="",
        default=None,
        type=str,
        help="Enable wandb logging. Optionally pass project name (default: policydb_evaluator)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    global DEBUG
    DEBUG = args.debug

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    az_params = parse_subargs(args.params, AlphaZeroParams)
    nn_config = NNConfig.from_alphazero_params(az_params)
    print(f"AlphaZero params: {az_params}")

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    use_wandb = args.wandb is not None
    if use_wandb:
        wandb_project = args.wandb if args.wandb else "policydb_evaluator"
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "db_path": os.path.basename(args.db_path),
                "num_steps": args.num_steps,
                "test_fraction": args.test_fraction,
                "test_batch_size": args.test_batch_size,
                "test_player": args.test_player,
                "exclude_test_set": args.exclude_test_set,
                "output": args.output,
                **asdict(az_params),
            },
        )
        Timer.set_wandb_run(wandb_run)
        wandb.define_metric("step", hidden=True)
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("test/*", step_metric="step")

    # ------------------------------------------------------------------
    # Open DB and read metadata
    # ------------------------------------------------------------------
    db = quoridor_rs.PyPolicyDb(args.db_path)
    board_size, max_walls, max_steps, num_states = db.read_metadata()
    print(f"Board size: {board_size}, max_walls: {max_walls}, max_steps: {max_steps}")

    if num_states is None:
        # Fallback for DBs created before autoincrement ID was added.
        num_states = db.count_states()
        print(f"(num_states not in metadata, counted {num_states} rows)")
    print(f"Policy DB contains {num_states} states")

    # Convert 1-based player arg to 0-based internal representation.
    test_player = None
    if args.test_player is not None:
        test_player = args.test_player - 1
        print(f"Filtering test set to player {args.test_player} (internal: {test_player})")

    # ------------------------------------------------------------------
    # Train/test split by ID (IDs are 1-based, contiguous)
    # ------------------------------------------------------------------
    if args.test_batch_size is not None:
        test_id_set = None
        print(f"Full-DB test mode: {num_states} states, test batch size {args.test_batch_size}")
    else:
        test_size = min(max(1, int(num_states * args.test_fraction)), MAX_TEST_SIZE)
        test_id_set = set(random.sample(range(1, num_states + 1), test_size))
        test_ids = sorted(test_id_set)
        print(f"Train size: ~{num_states - test_size}, test size: {len(test_ids)}")

    # ------------------------------------------------------------------
    # Create NNEvaluator and set up optimizer
    # ------------------------------------------------------------------
    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, device, nn_config, max_cache_size=100000)
    evaluator.train_prepare(az_params.learning_rate, az_params.batch_size, args.num_steps, az_params.weight_decay)

    # ------------------------------------------------------------------
    # Pre-compute test samples (test set is small / capped)
    # ------------------------------------------------------------------
    if args.test_batch_size is None:
        print("Computing test features...")
        test_samples = fetch_batch(db, test_ids, evaluator, board_size, max_walls, max_steps)
        if test_player is not None:
            test_samples = [s for s in test_samples if s["current_player"] == test_player]
            print(f"Filtered test set to {len(test_samples)} states for player {args.test_player}")
        feature_dim = test_samples[0]["input_array"].shape[0]
    else:
        test_samples = None
        probe = fetch_batch(db, [1], evaluator, board_size, max_walls, max_steps)
        feature_dim = probe[0]["input_array"].shape[0]
    print(f"Feature dim: {feature_dim}")

    if use_wandb:
        wandb.config.update(
            {
                "board_size": board_size,
                "max_walls": max_walls,
                "max_steps": max_steps,
                "num_states": num_states,
                "train_size": num_states if args.test_batch_size is not None else num_states - test_size,
                "test_size": num_states if args.test_batch_size is not None else len(test_samples),
                "feature_dim": feature_dim,
            }
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    batch_size = az_params.batch_size

    learning_rate = az_params.learning_rate
    if use_wandb:
        wandb.log(
            {
                "learning_rate": learning_rate,
                "step": 0,
            }
        )

    for step in range(1, args.num_steps + 1):
        print(f"Step {step}/{args.num_steps}")
        if step % 100000 == 0:
            learning_rate = learning_rate / 2.0
            print(f"Lowering learning rate to {learning_rate}")
            evaluator.train_prepare(learning_rate, az_params.batch_size, args.num_steps, az_params.weight_decay)
            if use_wandb:
                wandb.log(
                    {
                        "learning_rate": learning_rate,
                        "step": step,
                    }
                )

        # Oversample to account for player filtering and states dropped by
        # build_policy_from_children (missing children in DB).
        oversample = 4 if test_player is None else 8
        batch_ids = random.sample(range(1, num_states + 1), min(batch_size * oversample, num_states))
        if args.exclude_test_set and test_id_set is not None:
            batch_ids = [i for i in batch_ids if i not in test_id_set]
        batch_samples = fetch_batch(db, batch_ids, evaluator, board_size, max_walls, max_steps)
        if test_player is not None:
            batch_samples = [s for s in batch_samples if s["current_player"] == test_player]
        batch_samples = batch_samples[:batch_size]
        train_policy_loss, train_value_loss, train_total_loss = evaluator.train_iteration_v2(batch_samples)

        if use_wandb:
            wandb.log(
                {
                    "train/value_loss": train_value_loss,
                    "train/policy_loss": train_policy_loss,
                    "train/total_loss": train_total_loss,
                    "step": step,
                }
            )

        if step % args.log_interval == 0 or step == 1:
            if args.test_batch_size is not None:
                test_policy_loss, test_value_loss, test_total_loss, acc = compute_test_metrics_batched(
                    num_states,
                    db,
                    evaluator,
                    board_size,
                    max_walls,
                    max_steps,
                    args.test_batch_size,
                    test_player=test_player,
                )
            else:
                evaluator.network.eval()
                with torch.no_grad():
                    test_policy_loss, test_value_loss, test_total_loss = evaluator.compute_losses(test_samples)
                    test_policy_loss = test_policy_loss.item()
                    test_value_loss = test_value_loss.item()
                    test_total_loss = test_total_loss.item()
                evaluator.network.train()

                Timer.log_totals()

                acc = compute_accuracy(
                    test_ids,
                    db,
                    evaluator,
                    board_size,
                    max_walls,
                    max_steps,
                    args.accuracy_states,
                    test_player=test_player,
                )

            print(
                f"step {step:6d} | "
                f"train: pol={train_policy_loss:.4f} val={train_value_loss:.4f} tot={train_total_loss:.4f} | "
                f"test: pol={test_policy_loss:.4f} val={test_value_loss:.4f} tot={test_total_loss:.4f} | "
                f"accuracy {acc:.3f}"
            )
            sys.stdout.flush()

            if use_wandb:
                wandb.log(
                    {
                        "test/value_loss": test_value_loss,
                        "test/policy_loss": test_policy_loss,
                        "test/total_loss": test_total_loss,
                        "test/accuracy": acc,
                        "step": step,
                    }
                )

            if test_total_loss < best_test_loss:
                best_test_loss = test_total_loss
                torch.save(
                    {
                        "network_state_dict": evaluator.network.state_dict(),
                        "board_size": board_size,
                        "max_walls": max_walls,
                        "max_steps": max_steps,
                        "params": asdict(az_params),
                    },
                    args.output,
                )

    print(f"Training complete. Best test loss: {best_test_loss:.4f}. Model saved to {args.output}")

    if use_wandb:
        wandb.summary["best_test_loss"] = best_test_loss
        wandb.finish()


if __name__ == "__main__":
    main()
