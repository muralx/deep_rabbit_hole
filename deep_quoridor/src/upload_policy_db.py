"""Upload a policy database Parquet file to wandb as an artifact.

The artifact carries the file's footer metadata (board_size, max_walls,
max_steps, num_states) so collaborators can find/select databases in the
wandb UI without downloading them. `train_policy_db_evaluator.py` can
consume the resulting artifact via a `wandb:<entity>/<project>/<name>:<alias>`
path in place of a local file.
"""

import argparse
import os
import sys

import quoridor_rs
import wandb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to a .parquet policy database file")
    p.add_argument(
        "--project",
        default="policydb",
        help="wandb project to upload to (default: %(default)s)",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Artifact name (default: file basename without .parquet)",
    )
    p.add_argument(
        "--description",
        default=None,
        help="Free-form description attached to the artifact",
    )
    p.add_argument(
        "--aliases",
        default=None,
        help="Comma-separated extra aliases (in addition to wandb's auto-added 'latest')",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.path):
        print(f"error: not a file: {args.path}", file=sys.stderr)
        sys.exit(1)
    if not args.path.endswith(".parquet"):
        print(f"error: expected a .parquet file, got: {args.path}", file=sys.stderr)
        sys.exit(1)

    name = args.name or os.path.splitext(os.path.basename(args.path))[0]

    # Read footer metadata for tagging. lazy=True avoids loading the full
    # dataset into memory just to inspect the header.
    db = quoridor_rs.PyPolicyDb(args.path, lazy=True)
    board_size, max_walls, max_steps, num_states = db.read_metadata()
    print(
        f"Read metadata: board_size={board_size}, max_walls={max_walls}, "
        f"max_steps={max_steps}, num_states={num_states}"
    )

    aliases = []
    if args.aliases:
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]

    run = wandb.init(project=args.project, job_type="upload_policy_db")
    try:
        artifact = wandb.Artifact(
            name=name,
            type="policy_db",
            description=args.description,
            metadata={
                "board_size": board_size,
                "max_walls": max_walls,
                "max_steps": max_steps,
                "num_states": num_states,
                "file_basename": os.path.basename(args.path),
            },
        )
        artifact.add_file(args.path)
        run.log_artifact(artifact, aliases=aliases or None)
        artifact.wait()  # block until the upload completes
    finally:
        wandb.finish()

    full_ref = f"{run.entity}/{args.project}/{name}:latest"
    print(f"Uploaded. Reference for train_policy_db_evaluator.py:\n  wandb:{full_ref}")


if __name__ == "__main__":
    main()
