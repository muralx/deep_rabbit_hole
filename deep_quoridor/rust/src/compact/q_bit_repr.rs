/// Compact bit-packed representation accessor for game states.
///
/// The state is split into two primitives so each field lives in one word:
///   - `walls: u128`  — wall bitmap, supports up to 128 wall positions (9x9 board).
///   - `scalars: u64` — positions, walls-remaining, current player, completed steps.
///
/// `QBitRepr` itself stores no game state — it only stores parameters and computed
/// offsets needed to interpret a `CompactState`.

// Wall orientations
pub const WALL_VERTICAL: usize = 0;
pub const WALL_HORIZONTAL: usize = 1;

/// Calculate the number of bits needed to represent a value up to max (inclusive)
const fn bits_needed(max: usize) -> usize {
    if max == 0 {
        1
    } else {
        (usize::BITS - max.leading_zeros()) as usize
    }
}

/// Bit-packed Quoridor game state.
///
/// Layout:
///   `walls`   — bit `i` set iff wall at `wall_index_to_position(i)` is placed.
///   `scalars` — packed: p1_pos | p2_pos | p1_walls_remaining | p2_walls_remaining
///               | current_player | completed_steps
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CompactState {
    pub walls: u128,
    pub scalars: u64,
}

impl CompactState {
    /// Serialize as 24 little-endian bytes: walls (16) ++ scalars (8).
    pub fn to_bytes(self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[..16].copy_from_slice(&self.walls.to_le_bytes());
        out[16..].copy_from_slice(&self.scalars.to_le_bytes());
        out
    }

    /// Deserialize from 24 little-endian bytes.
    pub fn from_bytes(b: [u8; 24]) -> Self {
        let mut walls = [0u8; 16];
        walls.copy_from_slice(&b[..16]);
        let mut scalars = [0u8; 8];
        scalars.copy_from_slice(&b[16..]);
        Self {
            walls: u128::from_le_bytes(walls),
            scalars: u64::from_le_bytes(scalars),
        }
    }
}

/// Accessor for bit-packed game states.
/// Stores parameters and computed offsets, but not the actual game state data.
#[derive(Clone, Debug)]
pub struct QBitRepr {
    // Parameters
    board_size: usize,
    max_walls: usize,
    max_steps: usize,

    // Computed values
    num_wall_positions: usize,
    num_player_positions: usize,
    position_bits: usize,
    walls_remaining_bits: usize,
    completed_steps_bits: usize,
    total_scalar_bits: usize,

    // Bit offsets within `scalars`
    player_pos_offsets: [usize; 2],
    walls_remaining_offsets: [usize; 2],
    current_player_offset: usize,
    completed_steps_offset: usize,
}

impl QBitRepr {
    /// Create a new QBitRepr state accessor with the given parameters
    pub fn new(board_size: usize, max_walls: usize, max_steps: usize) -> Self {
        let num_wall_positions = 2 * (board_size - 1) * (board_size - 1);
        let num_player_positions = board_size * board_size;
        let position_bits = bits_needed(num_player_positions - 1);
        let walls_remaining_bits = bits_needed(max_walls);
        let completed_steps_bits = bits_needed(max_steps);

        assert!(
            num_wall_positions <= 128,
            "QBitRepr: wall bitmap requires {num_wall_positions} bits, which exceeds u128 capacity"
        );

        let p1_pos_offset = 0;
        let p2_pos_offset = p1_pos_offset + position_bits;
        let player_pos_offsets = [p1_pos_offset, p2_pos_offset];
        let p1_walls_remaining_offset = p2_pos_offset + position_bits;
        let p2_walls_remaining_offset = p1_walls_remaining_offset + walls_remaining_bits;
        let walls_remaining_offsets = [p1_walls_remaining_offset, p2_walls_remaining_offset];
        let current_player_offset = p2_walls_remaining_offset + walls_remaining_bits;
        let completed_steps_offset = current_player_offset + 1;
        let total_scalar_bits = completed_steps_offset + completed_steps_bits;

        assert!(
            total_scalar_bits <= 64,
            "QBitRepr: scalar fields require {total_scalar_bits} bits, which exceeds u64 capacity"
        );

        Self {
            board_size,
            max_walls,
            max_steps,
            num_wall_positions,
            num_player_positions,
            position_bits,
            walls_remaining_bits,
            completed_steps_bits,
            total_scalar_bits,
            player_pos_offsets,
            walls_remaining_offsets,
            current_player_offset,
            completed_steps_offset,
        }
    }

    /// Total bits used across walls and scalars (for inspection / tests).
    #[allow(dead_code)]
    pub fn size_bits(&self) -> usize {
        self.num_wall_positions + self.total_scalar_bits
    }

    /// Get the number of wall positions
    pub fn num_wall_positions(&self) -> usize {
        self.num_wall_positions
    }

    /// Get the number of player positions
    #[allow(dead_code)]
    pub fn num_player_positions(&self) -> usize {
        self.num_player_positions
    }

    /// Create a zero-initialized packed state value.
    pub fn create_data(&self) -> CompactState {
        CompactState::default()
    }

    /// Read a bit from the wall bitmap.
    #[inline]
    fn get_wall_bit(&self, state: CompactState, wall_idx: usize) -> bool {
        debug_assert!(wall_idx < self.num_wall_positions);
        (state.walls >> wall_idx) & 1 == 1
    }

    /// Write a bit to the wall bitmap.
    #[inline]
    fn set_wall_bit(&self, state: &mut CompactState, wall_idx: usize, value: bool) {
        debug_assert!(wall_idx < self.num_wall_positions);
        if value {
            state.walls |= 1u128 << wall_idx;
        } else {
            state.walls &= !(1u128 << wall_idx);
        }
    }

    /// Read a multi-bit integer from the scalar field.
    #[inline]
    fn get_scalar_bits(&self, state: CompactState, bit_offset: usize, num_bits: usize) -> usize {
        debug_assert!(bit_offset + num_bits <= self.total_scalar_bits);
        ((state.scalars >> bit_offset) & ((1u64 << num_bits) - 1)) as usize
    }

    /// Write a multi-bit integer to the scalar field.
    #[inline]
    fn set_scalar_bits(
        &self,
        state: &mut CompactState,
        bit_offset: usize,
        num_bits: usize,
        value: usize,
    ) {
        debug_assert!(bit_offset + num_bits <= self.total_scalar_bits);
        let mask = (1u64 << num_bits) - 1;
        state.scalars =
            (state.scalars & !(mask << bit_offset)) | ((value as u64 & mask) << bit_offset);
    }

    /// Read a single scalar bit.
    #[inline]
    fn get_scalar_bit(&self, state: CompactState, bit_offset: usize) -> bool {
        debug_assert!(bit_offset < self.total_scalar_bits);
        (state.scalars >> bit_offset) & 1 == 1
    }

    /// Write a single scalar bit.
    #[inline]
    fn set_scalar_bit(&self, state: &mut CompactState, bit_offset: usize, value: bool) {
        debug_assert!(bit_offset < self.total_scalar_bits);
        if value {
            state.scalars |= 1u64 << bit_offset;
        } else {
            state.scalars &= !(1u64 << bit_offset);
        }
    }

    /// Check if a wall is present at the given position
    pub fn get_wall(
        &self,
        state: CompactState,
        row: usize,
        col: usize,
        orientation: usize,
    ) -> bool {
        let wall_index = self.wall_position_to_index(row, col, orientation);
        self.get_wall_bit(state, wall_index)
    }

    /// Set a wall at the given position
    pub fn set_wall(
        &self,
        state: &mut CompactState,
        row: usize,
        col: usize,
        orientation: usize,
        present: bool,
    ) {
        let wall_index = self.wall_position_to_index(row, col, orientation);
        self.set_wall_bit(state, wall_index, present);
    }

    /// Get a player's position
    pub fn get_player_position(&self, state: CompactState, player: usize) -> (usize, usize) {
        debug_assert!(player < 2);
        let index =
            self.get_scalar_bits(state, self.player_pos_offsets[player], self.position_bits);
        self.index_to_position(index)
    }

    /// Set a player's position
    pub fn set_player_position(
        &self,
        state: &mut CompactState,
        player: usize,
        row: usize,
        col: usize,
    ) {
        debug_assert!(player < 2);
        let new_index = self.position_to_index(row, col);
        debug_assert!(new_index < self.num_player_positions);
        self.set_scalar_bits(
            state,
            self.player_pos_offsets[player],
            self.position_bits,
            new_index,
        );
    }

    /// Get a player's remaining walls
    pub fn get_walls_remaining(&self, state: CompactState, player: usize) -> usize {
        debug_assert!(player < 2);
        self.get_scalar_bits(
            state,
            self.walls_remaining_offsets[player],
            self.walls_remaining_bits,
        )
    }

    /// Set a player's remaining walls
    pub fn set_walls_remaining(&self, state: &mut CompactState, player: usize, walls: usize) {
        debug_assert!(player < 2);
        debug_assert!(walls <= self.max_walls);
        self.set_scalar_bits(
            state,
            self.walls_remaining_offsets[player],
            self.walls_remaining_bits,
            walls,
        );
    }

    /// Get the current player (0 or 1)
    pub fn get_current_player(&self, state: CompactState) -> usize {
        if self.get_scalar_bit(state, self.current_player_offset) {
            1
        } else {
            0
        }
    }

    /// Set the current player
    pub fn set_current_player(&self, state: &mut CompactState, player: usize) {
        debug_assert!(player < 2);
        self.set_scalar_bit(state, self.current_player_offset, player == 1);
    }

    /// Get the number of completed steps
    pub fn get_completed_steps(&self, state: CompactState) -> usize {
        self.get_scalar_bits(
            state,
            self.completed_steps_offset,
            self.completed_steps_bits,
        )
    }

    /// Set the number of completed steps
    pub fn set_completed_steps(&self, state: &mut CompactState, steps: usize) {
        debug_assert!(steps <= self.max_steps);
        self.set_scalar_bits(
            state,
            self.completed_steps_offset,
            self.completed_steps_bits,
            steps,
        );
    }

    /// Convert a (row, col) position to a flat index
    pub fn position_to_index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.board_size && col < self.board_size);
        row * self.board_size + col
    }

    /// Convert a flat index to (row, col) position
    pub fn index_to_position(&self, index: usize) -> (usize, usize) {
        debug_assert!(index < self.num_player_positions);
        (index / self.board_size, index % self.board_size)
    }

    /// Get board size
    pub fn board_size(&self) -> usize {
        self.board_size
    }

    /// Get max walls
    #[allow(dead_code)]
    pub fn max_walls(&self) -> usize {
        self.max_walls
    }

    /// Get max steps
    #[allow(dead_code)]
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Convert (row, col, orientation) to wall_index
    /// orientation: 0 = vertical, 1 = horizontal
    pub fn wall_position_to_index(&self, row: usize, col: usize, orientation: usize) -> usize {
        debug_assert!(row < self.board_size - 1);
        debug_assert!(col < self.board_size - 1);
        debug_assert!(orientation < 2);
        orientation * (self.board_size - 1) * (self.board_size - 1)
            + row * (self.board_size - 1)
            + col
    }

    /// Convert wall_index to (row, col, orientation)
    /// Returns (row, col, orientation) where orientation is 0=vertical, 1=horizontal
    pub fn wall_index_to_position(&self, wall_index: usize) -> (usize, usize, usize) {
        debug_assert!(wall_index < self.num_wall_positions);
        let positions_per_orientation = (self.board_size - 1) * (self.board_size - 1);
        let orientation = wall_index / positions_per_orientation;
        let remainder = wall_index % positions_per_orientation;
        let row = remainder / (self.board_size - 1);
        let col = remainder % (self.board_size - 1);
        (row, col, orientation)
    }

    /// Display the board state as text art
    pub fn print(&self, state: CompactState) {
        println!("{}", self.display(state));
    }

    /// Create string with text art of the board
    ///
    /// Format:
    /// - '1' and '2' for player positions
    /// - '.' for empty cells
    /// - '|' for vertical walls
    /// - '-' for horizontal walls
    /// - Metadata (steps, walls) shown on the right side
    pub fn display(&self, state: CompactState) -> String {
        let mut output = String::new();

        // Get metadata
        let (p0_row, p0_col) = self.get_player_position(state, 0);
        let (p1_row, p1_col) = self.get_player_position(state, 1);
        let p0_walls = self.get_walls_remaining(state, 0);
        let p1_walls = self.get_walls_remaining(state, 1);
        let current_player = self.get_current_player(state);
        let steps = self.get_completed_steps(state);

        // Build metadata lines
        let meta_lines = vec![
            format!("Steps: {}", steps),
            format!("Current: P{}", current_player + 1),
            format!("P1 walls: {}", p0_walls),
            format!("P2 walls: {}", p1_walls),
            format!("Max steps: {}", self.max_steps),
        ];

        let mut line_idx = 0;

        // Print each row with walls between
        for row in 0..self.board_size {
            let mut cell_line = String::new();
            for col in 0..self.board_size {
                // Print cell content
                if p0_row == row && p0_col == col {
                    cell_line.push('1');
                } else if p1_row == row && p1_col == col {
                    cell_line.push('2');
                } else {
                    cell_line.push('.');
                }

                // Print vertical wall to the right (if not last column)
                if col < self.board_size - 1 {
                    if row < self.board_size - 1 && self.get_wall(state, row, col, WALL_VERTICAL) {
                        cell_line.push('|');
                    } else if row > 0 && self.get_wall(state, row - 1, col, WALL_VERTICAL) {
                        cell_line.push('|');
                    } else {
                        cell_line.push(' ');
                    }
                }
            }

            // Append metadata if available
            if line_idx < meta_lines.len() {
                output.push_str(&format!("{}  {}\n", cell_line, meta_lines[line_idx]));
            } else {
                output.push_str(&format!("{}\n", cell_line));
            }
            line_idx += 1;

            // Print horizontal wall row (if not last row)
            if row < self.board_size - 1 {
                let mut wall_line = String::new();
                for col in 0..self.board_size {
                    // Print horizontal wall below this cell
                    if col < self.board_size - 1 && self.get_wall(state, row, col, WALL_HORIZONTAL)
                    {
                        wall_line.push('-');
                    } else if col > 0 && self.get_wall(state, row, col - 1, WALL_HORIZONTAL) {
                        wall_line.push('-');
                    } else {
                        wall_line.push(' ');
                    }

                    // Print intersection
                    if col < self.board_size - 1 {
                        wall_line.push(' ');
                    }
                }

                // Append metadata if available
                if line_idx < meta_lines.len() {
                    output.push_str(&format!("{}  {}\n", wall_line, meta_lines[line_idx]));
                } else {
                    output.push_str(&format!("{}\n", wall_line));
                }
                line_idx += 1;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_calculation() {
        let q = QBitRepr::new(5, 10, 100);
        // 5x5 board: 2*(5-1)*(5-1) = 32 wall bits
        // Positions: ceil(log2(25)) = 5 bits each, 2 players = 10 bits
        // Walls remaining: ceil(log2(10)) = 4 bits each = 8 bits
        // Current player: 1 bit
        // Steps: ceil(log2(100)) = 7 bits
        // Scalars total: 10 + 8 + 1 + 7 = 26 bits
        // Walls + scalars: 32 + 26 = 58 bits
        assert_eq!(q.size_bits(), 58);
    }

    #[test]
    fn test_size_calculation_9x9_10w() {
        let q = QBitRepr::new(9, 10, 200);
        // 9x9 board: 2*8*8 = 128 wall bits
        // Positions: ceil(log2(81)) = 7 bits each, 2 players = 14 bits
        // Walls remaining: ceil(log2(10)) = 4 bits each = 8 bits
        // Current player: 1 bit
        // Steps: ceil(log2(200)) = 8 bits
        // Scalars total: 14 + 8 + 1 + 8 = 31 bits
        assert_eq!(q.num_wall_positions(), 128);
        assert_eq!(q.size_bits(), 128 + 31);
    }

    #[test]
    fn test_player_positions() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        q.set_player_position(&mut data, 0, 0, 2);
        q.set_player_position(&mut data, 1, 4, 2);

        assert_eq!(q.get_player_position(data, 0), (0, 2));
        assert_eq!(q.get_player_position(data, 1), (4, 2));
    }

    #[test]
    fn test_walls() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        // Set vertical wall at (0, 0)
        q.set_wall(&mut data, 0, 0, WALL_VERTICAL, true);
        // Set vertical wall at (1, 1)
        q.set_wall(&mut data, 1, 1, WALL_VERTICAL, true);
        assert!(q.get_wall(data, 0, 0, WALL_VERTICAL));
        assert!(q.get_wall(data, 1, 1, WALL_VERTICAL));
        assert!(!q.get_wall(data, 0, 1, WALL_VERTICAL));
    }

    #[test]
    fn test_walls_9x9() {
        let q = QBitRepr::new(9, 10, 200);
        let mut data = q.create_data();

        // Walls at all four corners of the wall grid, in both orientations.
        for &(r, c) in &[(0usize, 0usize), (0, 7), (7, 0), (7, 7)] {
            for &orient in &[WALL_VERTICAL, WALL_HORIZONTAL] {
                q.set_wall(&mut data, r, c, orient, true);
                assert!(q.get_wall(data, r, c, orient));
            }
        }
        // The wall bitmap should occupy the high bits too (bit 127 = last horizontal corner).
        assert_ne!(data.walls >> 64, 0);
    }

    #[test]
    fn test_walls_remaining() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        q.set_walls_remaining(&mut data, 0, 10);
        assert_eq!(q.get_walls_remaining(data, 0), 10);
    }

    #[test]
    fn test_walls_remaining_9x9() {
        let q = QBitRepr::new(9, 10, 200);
        let mut data = q.create_data();
        q.set_walls_remaining(&mut data, 0, 10);
        q.set_walls_remaining(&mut data, 1, 7);
        assert_eq!(q.get_walls_remaining(data, 0), 10);
        assert_eq!(q.get_walls_remaining(data, 1), 7);
    }

    #[test]
    fn test_current_player() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        assert_eq!(q.get_current_player(data), 0);

        q.set_current_player(&mut data, 1);
        assert_eq!(q.get_current_player(data), 1);

        q.set_current_player(&mut data, 0);
        assert_eq!(q.get_current_player(data), 0);
    }

    #[test]
    fn test_completed_steps() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        assert_eq!(q.get_completed_steps(data), 0);

        q.set_completed_steps(&mut data, 42);
        assert_eq!(q.get_completed_steps(data), 42);
    }

    #[test]
    fn test_wall_position_conversion() {
        let q = QBitRepr::new(5, 10, 100);

        // Test vertical wall at (1, 2)
        let idx = q.wall_position_to_index(1, 2, WALL_VERTICAL);
        let (row, col, orientation) = q.wall_index_to_position(idx);
        assert_eq!((row, col, orientation), (1, 2, WALL_VERTICAL));

        // Test horizontal wall at (3, 1)
        let idx = q.wall_position_to_index(3, 1, WALL_HORIZONTAL);
        let (row, col, orientation) = q.wall_index_to_position(idx);
        assert_eq!((row, col, orientation), (3, 1, WALL_HORIZONTAL));
    }

    #[test]
    fn test_compact_state_bytes_roundtrip() {
        let q = QBitRepr::new(9, 10, 200);
        let mut state = q.create_data();
        q.set_player_position(&mut state, 0, 0, 4);
        q.set_player_position(&mut state, 1, 8, 4);
        q.set_walls_remaining(&mut state, 0, 10);
        q.set_walls_remaining(&mut state, 1, 10);
        q.set_current_player(&mut state, 1);
        q.set_completed_steps(&mut state, 137);
        q.set_wall(&mut state, 7, 7, WALL_HORIZONTAL, true);

        let bytes = state.to_bytes();
        let restored = CompactState::from_bytes(bytes);
        assert_eq!(state, restored);
    }
}
