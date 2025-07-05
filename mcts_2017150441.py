import time
import random
import copy

from enum import Enum
from typing import Tuple, Union

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

# Define datatype for game result
class GameResult(Enum):
    """
    Enum to encode different states of the game. A game can be in progress (NOT_FINISHED), lost, won, or draw
    """
    NOT_FINISHED = 0
    NAUGHT_WIN = 1
    CROSS_WIN = 2
    DRAW = 3

# Define board and game hyperparameter
BOARD_DIM = 5  # type: int
BOARD_SIZE = BOARD_DIM * BOARD_DIM  # type: int
NUM_WIN = 3

# Board consists of cells
class CellState(Enum):
    EMPTY = 0  # type: int
    NAUGHT = 1  # type: int
    CROSS = 2  # type: int

class GomokuBoard:
    """
    The class to encode a gomoku board, including its current state of pieces.
    Also contains various utility methods.
    """

    WIN_CHECK_DIRS = [[(-1, 0), (1, 0)],
                      [(0, -1), (0, 1)],
                      [(-1, 1), (1, -1)],
                      [(-1, -1), (1, 1)]]

    def hash_value(self) -> int:
        """
        Encode the current state of the game (board positions) as an integer. Will be used for caching evaluations
        :return: A collision free hash value representing the current board state
        """
        res = 0
        for i in range(BOARD_SIZE):
            res *= BOARD_DIM
            res += self.state[i].value
        return res

    @staticmethod
    def other_side(side: CellState) -> CellState:
        """
        Utility method to return the value of the other player than the one passed as input
        :param side: The side we want to know the opposite of
        :return: The opposite side to the one passed as input
        """
        if side == CellState.EMPTY:
            raise ValueError("EMPTY has no 'other side'")
        if side == CellState.CROSS:
            return CellState.NAUGHT
        if side == CellState.NAUGHT:
            return CellState.CROSS
        raise ValueError("{} is not a valid side".format(side))

    def __init__(self):
        """
        Create a new Board. If a state is passed in, we use that otherwise we initialize with an empty board
        """
        self.state = np.ndarray(shape=(BOARD_SIZE,), dtype=CellState)
        self.reset()

    def coord_to_pos(self, coord: Tuple[int, int]) -> int:
        """
        Converts a 2D board position to a 1D board position.
        :param coord: A board position in 2D coordinates
        :return: The same board position in 1D coordinates
        """
        return coord[0] * BOARD_DIM + coord[1]

    def pos_to_coord(self, pos: int) -> Tuple[int, int]:
        """
        Converts a 1D board position to a 2D board position.
        :param pos: A board position in 1D coordinates
        :return: The same board position in 2D coordinates
        """
        return pos // BOARD_DIM, pos % BOARD_DIM

    def reset(self):
        """
        Resets the game board. All fields are set to be EMPTY.
        """
        self.state.fill(CellState.EMPTY)
        self.who_won = None

    def num_empty(self) -> int:
        """
        Counts and returns the number of empty fields on the board.
        :return: The number of empty fields on the board
        """
        return np.count_nonzero(self.state == CellState.EMPTY)

    def random_empty_spot(self) -> int:
        """
        Returns a random empty spot on the board in 1D coordinates
        :return: A random empty spot on the board in 1D coordinates
        """
        empty_spots = [i for i, cell in enumerate(self.state) if cell == CellState.EMPTY]
        if not empty_spots:
            return -1 # No empty spots
        return random.choice(empty_spots)

    def is_legal(self, pos: int) -> bool:
        """
        Tests whether a board position can be played, i.e. is currently empty
        :param pos: The board position in 1D that is to be checked
        :return: Whether the position can be played
        """
        return (0 <= pos < BOARD_SIZE) and (self.state[pos] == CellState.EMPTY)

    def move(self, position: int, side: CellState) -> Union[np.ndarray, GameResult, bool]:
        """
        Places a piece of side "side" at position "position". The position is to be provided as 1D.
        Throws a ValueError if the position is not EMPTY
        returns the new state of the board, the game result after this move, and whether this move has finished the game

        :param position: The position where we want to put a piece
        :param side: What piece we want to play (NAUGHT, or CROSS)
        :return: The game state after the move, The game result after the move, Whether the move finished the game
        """
        if self.state[position] != CellState.EMPTY:
            raise ValueError("Invalid move")

        self.state[position] = side

        if self.check_win(position, side):
            self.who_won = side
            return self.state, GameResult.CROSS_WIN if side == CellState.CROSS else GameResult.NAUGHT_WIN, True

        if self.num_empty() == 0:
            self.who_won = CellState.EMPTY
            return self.state, GameResult.DRAW, True

        return self.state, GameResult.NOT_FINISHED, False

    def check_win(self, pos: int, side: CellState):
        row = pos // BOARD_DIM
        col = pos % BOARD_DIM

        def check_direction(_dr, _dc):
            count = 0
            r, c = row + _dr, col + _dc
            while 0 <= r < BOARD_DIM and 0 <= c < BOARD_DIM:
                p = r * BOARD_DIM + c
                if self.state[p] == side:
                    count += 1
                    r += _dr
                    c += _dc
                else:
                    break
            return count

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            total_count = 1
            total_count += check_direction(dr, dc)
            total_count += check_direction(-dr, -dc)

            if total_count >= NUM_WIN:
                return True
        return False

    def state_to_char(self, pos):
        """
        Return 'X', 'O', or ' ' depending on what piece is on 1D position pos.
        """
        if (self.state[pos]) == CellState.NAUGHT:
            return 'O'
        elif (self.state[pos]) == CellState.CROSS:
            return 'X'
        else:
            return ' '

def draw_go_board(board: GomokuBoard):
    fig, ax = plt.subplots(figsize=(BOARD_DIM, BOARD_DIM))
    ax.set_xlim(0, BOARD_DIM)
    ax.set_ylim(0, BOARD_DIM)

    cell_size = 0.97
    rect_offset = (1.-cell_size) / 2
    offset = 1 / 2

    ax.add_patch(plt.Rectangle((0, 0), BOARD_DIM, BOARD_DIM, color='black', fill=True))

    for x in range(BOARD_DIM):
        for y in range(BOARD_DIM):
            ax.add_patch(plt.Rectangle((x + rect_offset, y + rect_offset), cell_size, cell_size, color='white', fill=True))

    for x in range(BOARD_DIM):
        for y in range(BOARD_DIM):
            pos = x * BOARD_DIM + y
            marker = board.state_to_char(pos)
            ax.text(x + offset, y + offset, marker, ha='center', va='center', fontsize=20)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.show()
    # time.sleep(0.1) # Reduced sleep time for faster visualization during development/debugging

from abc import ABC, abstractmethod

class GamePlayerInterface(ABC):
    """
    Interface for Gomoku player
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def move(self, board: GomokuBoard) -> Union[GameResult, bool]:
        pass

    @abstractmethod
    def new_game(self, side: CellState):
        pass

class RandomPlayer(GamePlayerInterface):
    def __init__(self):
        self.side = None
        super().__init__()

    def move(self, GomokuBoard: GomokuBoard) -> Union[GameResult, bool]:
        location = GomokuBoard.random_empty_spot()
        if location == -1:
            return GameResult.DRAW, True

        _, res, finished = GomokuBoard.move(int(location), self.side)
        return res, finished

    def new_game(self, side: CellState):
        self.side = side

def play_game(board: GomokuBoard, player1: GamePlayerInterface, player2: GamePlayerInterface, drawing_board: bool = True):
    player1.new_game(CellState.CROSS)
    player2.new_game(CellState.NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if drawing_board:
            draw_go_board(board)
        if finished:
            continue
        else:
            result, finished = player2.move(board)
            if drawing_board:
                draw_go_board(board)
    return result

def battle(player1: GamePlayerInterface, player2: GamePlayerInterface, num_games: int = 1000, silent: bool = False):
    board = GomokuBoard()
    draw_count = 0
    cross_count = 0
    naught_count = 0
    for _ in range(num_games):
        result = play_game(board, player1, player2, drawing_board=False)
        if result == GameResult.CROSS_WIN:
            cross_count += 1
        elif result == GameResult.NAUGHT_WIN:
            naught_count += 1
        else:
            draw_count += 1

    if not silent:
        print(f"After {num_games} games we have draws: {draw_count}, Player 1 (CROSS) wins: {cross_count}, and Player 2 (NAUGHT) wins: {naught_count}.")
        print(f"Which gives percentages of draws: {draw_count / num_games:.2%}, Player 1 (CROSS) wins: {cross_count / num_games:.2%}, and Player 2 (NAUGHT) wins:  {naught_count / num_games:.2%}")

    return cross_count, naught_count, draw_count

class MCTSAlgorithm(GamePlayerInterface):

    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1
    C = 1.4  # Exploration parameter for UCB1

    def __init__(self):
        self.side = None
        self.tree = nx.DiGraph()
        super().__init__()

    def new_game(self, side: CellState):
        """
        Set the side for the game to come.
        - Set the `side` attribute to the provided `side` argument.
        - Clear the game tree using `self.tree.clear()`.
        """
        self.side = side
        self.tree.clear()

    def simulate_game(self, board: GomokuBoard, first_side: CellState) -> int:
        """
        Simulate a game and evaluate the result.
        - Create instances of `RandomPlayer` for both players.
        - Initialize these players with the given `first_side`.
        - Simulate the game by having the players take turns until it's finished.
        - Evaluate the game result and return WIN_VALUE, LOSS_VALUE, or DRAW_VALUE accordingly.
        """
        sim_board = copy.deepcopy(board)
        player_cross = RandomPlayer()
        player_naught = RandomPlayer()

        player_cross.new_game(CellState.CROSS)
        player_naught.new_game(CellState.NAUGHT)

        current_player = player_cross if first_side == CellState.CROSS else player_naught
        opponent_player = player_naught if first_side == CellState.CROSS else player_cross

        finished = False
        result = GameResult.NOT_FINISHED

        while not finished:
            try:
                result, finished = current_player.move(sim_board)
            except ValueError:
                finished = True
                result = GameResult.DRAW

            if finished:
                break
            current_player, opponent_player = opponent_player, current_player

        if result == GameResult.CROSS_WIN:
            return self.WIN_VALUE if self.side == CellState.CROSS else self.LOSS_VALUE
        elif result == GameResult.NAUGHT_WIN:
            return self.WIN_VALUE if self.side == CellState.NAUGHT else self.LOSS_VALUE
        else:
            return self.DRAW_VALUE

    def find_optimal_move(self, current_node: int, find_max: bool):
        """
        Find the optimal move based on the current game state.
        - Retrieve the successors of the `current_node` from the game tree.
        - Initialize `best_score` as negative infinity for maximizing, or positive infinity for minimizing.
        - Iterate through the successors, considering their evaluation scores.
        - Select the best moves with the highest or lowest scores.
        - Randomly choose one of the best moves.
        - Return the best score and selected best move.
        """
        successors = list(self.tree.successors(current_node))
        if not successors:
            return 0, None

        best_score = -float('inf') if find_max else float('inf')
        best_moves = []

        for move_hash in successors:
            visits = self.tree[current_node][move_hash]['visits']
            wins = self.tree[current_node][move_hash]['wins']

            if visits == 0:
                score = float('inf') if find_max else -float('inf')
            else:
                score = wins / visits

            if find_max:
                if score > best_score:
                    best_score = score
                    best_moves = [self.tree[current_node][move_hash]['move']]
                elif score == best_score:
                    best_moves.append(self.tree[current_node][move_hash]['move'])
            else: # find_min
                if score < best_score:
                    best_score = score
                    best_moves = [self.tree[current_node][move_hash]['move']]
                elif score == best_score:
                    best_moves.append(self.tree[current_node][move_hash]['move'])

        if not best_moves:
            # Fallback to a random move if no clear best move (e.g., all unvisited or tied)
            return best_score, random.choice([self.tree[current_node][s]['move'] for s in successors])
            
        return best_score, random.choice(best_moves)

    def _min(self, board: GomokuBoard) -> Tuple[int, int]:
        """
        Evaluate the board position from the Minimizing player's point of view (opponent's turn).
        - Check if the board position is already in the game tree.
        - If not, expand the game tree with available moves.
        - Use UCB (Upper Confidence Bound) to select the best move.
        - Simulate games to evaluate the selected move.
        - Update the game tree with the result.
        - Return the best result and the selected move.
        """
        current_hash = board.hash_value()

        if current_hash not in self.tree:
            self.tree.add_node(current_hash, board_state=copy.deepcopy(board.state))
            for i in range(BOARD_SIZE):
                if board.state[i] == CellState.EMPTY:
                    new_board = copy.deepcopy(board)
                    try:
                        # Opponent's move (the minimizing player)
                        new_board.move(i, GomokuBoard.other_side(self.side))
                        new_hash = new_board.hash_value()
                        self.tree.add_edge(current_hash, new_hash, move=i, visits=0, wins=0)
                    except ValueError:
                        pass

        # Selection: Choose child node with lowest UCB1 for minimizing player (opponent)
        best_ucb_score = float('inf')
        selected_move_hash = None
        selected_move_pos = -1

        successors = list(self.tree.successors(current_hash))
        if not successors: # Game might be over or no legal moves
            return self.DRAW_VALUE, -1

        total_visits_parent = sum(self.tree[current_hash][succ]['visits'] for succ in successors)
        if total_visits_parent == 0: # If parent hasn't been visited, all children are equally good for initial exploration
            selected_move_hash = random.choice(successors)
            selected_move_pos = self.tree[current_hash][selected_move_hash]['move']
        else:
            for succ_hash in successors:
                visits = self.tree[current_hash][succ_hash]['visits']
                wins = self.tree[current_hash][succ_hash]['wins']
                move_pos = self.tree[current_hash][succ_hash]['move']

                if visits == 0:
                    ucb_score = -float('inf') # Prioritize unvisited nodes for opponent's exploration (they want to minimize our win)
                else:
                    # UCB1 for opponent (minimizing player): we are minimizing win value
                    ucb_score = (wins / visits) - self.C * np.sqrt(np.log(total_visits_parent) / visits)

                if ucb_score < best_ucb_score:
                    best_ucb_score = ucb_score
                    selected_move_hash = succ_hash
                    selected_move_pos = move_pos
                elif ucb_score == best_ucb_score:
                    # Randomly break ties
                    if random.random() < 0.5:
                        selected_move_hash = succ_hash
                        selected_move_pos = move_pos

        # Simulation: Simulate game from selected_move_hash
        sim_board = copy.deepcopy(board)
        # Apply the opponent's move (which is selected_move_pos)
        sim_board.move(selected_move_pos, GomokuBoard.other_side(self.side))
        
        # Simulate from the perspective of the *next* player in the simulation (which is our MCTS player)
        simulation_result = self.simulate_game(sim_board, self.side) # Our side makes the next move in simulation

        # Backpropagation: Update visits and wins for the edge representing the opponent's move
        self.tree[current_hash][selected_move_hash]['visits'] += 1
        self.tree[current_hash][selected_move_hash]['wins'] += simulation_result # Result is from our perspective

        return simulation_result, selected_move_pos


    def _max(self, board: GomokuBoard) -> Tuple[int, int]:
        """
        Evaluate the board position from the Maximizing player's point of view (our turn).
        - Check if the board position is already in the game tree.
        - If not, expand the game tree with available moves.
        - Use UCB (Upper Confidence Bound) to select the best move.
        - Simulate games to evaluate the selected move.
        - Update the game tree with the result.
        - Return the best result and the selected move.
        """
        current_hash = board.hash_value()

        if current_hash not in self.tree:
            self.tree.add_node(current_hash, board_state=copy.deepcopy(board.state))
            for i in range(BOARD_SIZE):
                if board.state[i] == CellState.EMPTY:
                    new_board = copy.deepcopy(board)
                    try:
                        # Our move (the maximizing player)
                        new_board.move(i, self.side)
                        new_hash = new_board.hash_value()
                        self.tree.add_edge(current_hash, new_hash, move=i, visits=0, wins=0)
                    except ValueError:
                        pass

        # Selection: Choose child node with highest UCB1 for maximizing player (us)
        best_ucb_score = -float('inf')
        selected_move_hash = None
        selected_move_pos = -1

        successors = list(self.tree.successors(current_hash))
        if not successors: # Game might be over or no legal moves
            return self.DRAW_VALUE, -1

        total_visits_parent = sum(self.tree[current_hash][succ]['visits'] for succ in successors)
        if total_visits_parent == 0: # If parent hasn't been visited, all children are equally good for initial exploration
            selected_move_hash = random.choice(successors)
            selected_move_pos = self.tree[current_hash][selected_move_hash]['move']
        else:
            for succ_hash in successors:
                visits = self.tree[current_hash][succ_hash]['visits']
                wins = self.tree[current_hash][succ_hash]['wins']
                move_pos = self.tree[current_hash][succ_hash]['move']

                if visits == 0:
                    ucb_score = float('inf') # Prioritize unvisited nodes for exploration
                else:
                    # UCB1 for maximizing player
                    ucb_score = (wins / visits) + self.C * np.sqrt(np.log(total_visits_parent) / visits)

                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score
                    selected_move_hash = succ_hash
                    selected_move_pos = move_pos
                elif ucb_score == best_ucb_score:
                    # Randomly break ties
                    if random.random() < 0.5:
                        selected_move_hash = succ_hash
                        selected_move_pos = move_pos

        # Simulation: Simulate game from selected_move_hash
        sim_board = copy.deepcopy(board)
        # Apply our move (which is selected_move_pos)
        sim_board.move(selected_move_pos, self.side)
        
        # Simulate from the perspective of the *next* player in the simulation (which is the opponent)
        simulation_result = self.simulate_game(sim_board, GomokuBoard.other_side(self.side)) # Opponent makes the next move in simulation

        # Backpropagation: Update visits and wins for the edge representing our move
        self.tree[current_hash][selected_move_hash]['visits'] += 1
        self.tree[current_hash][selected_move_hash]['wins'] += simulation_result # Result is from our perspective

        return simulation_result, selected_move_pos

    def move(self, board: GomokuBoard) -> Union[GameResult, bool]:
        """
        Make a move according to the MCTS algorithm.
        - Perform MCTS by iteratively calling _max and _min (implicitly handled by UCB in _max).
        - Update the game tree during the iterations.
        - Repeat the process for a certain number of iterations (e.g., 200 times).
        - Return the result of the move and whether the game has finished.
        """
        current_board_hash = board.hash_value()
        if current_board_hash not in self.tree:
            self.tree.add_node(current_board_hash, board_state=copy.deepcopy(board.state))
        
        num_iterations = 200 # As suggested in the problem description
        for _ in range(num_iterations):
            # In each iteration, we perform a selection, expansion (if new node), simulation, and backpropagation.
            # The _max method will handle this process for the current player.
            # The choice of whether to call _max or _min depends on whose turn it is in the simulation
            # and is handled within the selection part of MCTS, which here is encapsulated by _max/min.
            # For the main move decision, we always want to maximize our outcome.
            if self.side == CellState.CROSS: # If MCTS player is CROSS, they are maximizing
                self._max(board)
            else: # If MCTS player is NAUGHT, they are also maximizing their score
                # The logic inside _max handles the opponent's turn by simulating
                # from the opponent's perspective.
                self._max(board) # _max seeks to maximize our win, considering opponent's best response.

        # After iterations, select the best actual move to make based on the accumulated statistics.
        # We find the move that leads to the highest win rate from the current node.
        _, best_move_pos = self.find_optimal_move(current_board_hash, find_max=True)

        if best_move_pos is None or not board.is_legal(best_move_pos):
            # Fallback to random if no optimal move found or if the chosen move is somehow illegal
            # (should be rare with proper tree management).
            print("MCTS: Falling back to random move.")
            best_move_pos = board.random_empty_spot()
            if best_move_pos == -1:
                return GameResult.DRAW, True
            
        _, res, finished = board.move(best_move_pos, self.side)
        return res, finished

# This block will only execute when the script is run directly, not when imported.
if __name__ == "__main__":
    print("Running a demonstration of MCTS vs RandomPlayer...")
    
    # Demonstrate play_game with MCTS (drawing_board=True to visualize)
    print("\n--- MCTS Player (CROSS) vs Random Player (NAUGHT) (visualized) ---")
    board_demo = GomokuBoard()
    player_mcts_demo = MCTSAlgorithm()
    player_random_demo = RandomPlayer()
    
    # play_game function handles new_game calls internally
    result_demo = play_game(board_demo, player_mcts_demo, player_random_demo, drawing_board=True)
    print(f"\nDemonstration game result: {result_demo}")

    # Estimate win rate using battle function
    print("\n--- Estimating Win Rate (MCTS vs RandomPlayer over 1000 games) ---")
    player1_mcts = MCTSAlgorithm()
    player2_random = RandomPlayer()
    
    cross_wins_p1, naught_wins_p2, draws = battle(player1_mcts, player2_random, num_games=1000)

    print("\n--- Estimating Win Rate (RandomPlayer vs MCTS Player over 1000 games) ---")
    player1_random = RandomPlayer()
    player2_mcts = MCTSAlgorithm()

    cross_wins_p1_rev, naught_wins_p2_rev, draws_rev = battle(player1_random, player2_mcts, num_games=1000)

    print("\nSummary of Performance Requirements Check:")
    # Player 1 (MCTS) vs Player 2 (Random)
    mcts_win_rate_as_p1 = cross_wins_p1 / (cross_wins_p1 + naught_wins_p2) if (cross_wins_p1 + naught_wins_p2) > 0 else 0
    print(f"MCTS as Player 1 (CROSS) win rate against Random Player: {mcts_win_rate_as_p1:.2%}")
    print(f"Meets 60% win rate (excluding draws) as P1: {mcts_win_rate_as_p1 >= 0.60}")

    # Player 1 (Random) vs Player 2 (MCTS)
    mcts_loss_rate_as_p2 = cross_wins_p1_rev / (cross_wins_p1_rev + naught_wins_p2_rev) if (cross_wins_p1_rev + naught_wins_p2_rev) > 0 else 0
    print(f"MCTS as Player 2 (NAUGHT) loss rate against Random Player: {mcts_loss_rate_as_p2:.2%}")
    print(f"Meets 40% loss rate as P2: {mcts_loss_rate_as_p2 >= 0.40}")

    print("\nTo use your MCTS implementation in another script, you can import it like this:")
    print("```python")
    print("import mcts_2017150441")
    print("player_mcts = mcts_2017150441.MCTSAlgorithm()")
    print("```")