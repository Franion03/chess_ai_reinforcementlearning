"""UCI protocol wrapper for the Chess RL agent. Compatible with lichess-bot and any UCI GUI."""
import sys
import chess
import numpy as np
from real_chess.agent import Agent, GreedyAgent
from real_chess.environment import Board
from real_chess.learn import TD_search

ENGINE_NAME = "ChessRL"
ENGINE_AUTHOR = "Franion03"
DEFAULT_CHECKPOINT = "checkpoints/model.h5"
DEFAULT_NETWORK = "big"
DEFAULT_SEARCH_TIME = 1.0


class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.agent = None
        self.search_time = DEFAULT_SEARCH_TIME
        self.network = DEFAULT_NETWORK
        self.checkpoint = DEFAULT_CHECKPOINT

    def load_agent(self):
        self.agent = Agent(network=self.network)
        try:
            self.agent.model.load_weights(self.checkpoint)
        except (OSError, ValueError):
            pass  # Use untrained weights if no checkpoint
        self.agent.fix_model()

    def encode_board(self):
        """Encode current board into 8x8x8 layer representation."""
        mapper = {"p": 0, "r": 1, "n": 2, "b": 3, "q": 4, "k": 5,
                  "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5}
        layer_board = np.zeros((8, 8, 8))
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                sign = 1 if piece.symbol().isupper() else -1
                layer_board[mapper[piece.symbol()], i // 8, i % 8] = sign
        layer_board[6, :, :] = 1 / max(self.board.fullmove_number, 1)
        layer_board[6, 0, :] = 1 if self.board.turn else -1
        layer_board[7, :, :] = 1
        return layer_board

    def best_move(self):
        """Select best move using MCTS with the trained value network."""
        opponent = GreedyAgent()
        env = Board(opposing_agent=opponent, FEN=self.board.fen())
        searcher = TD_search(env, self.agent, search_time=self.search_time)
        self.agent.fix_model()

        from real_chess.tree import Node
        tree = Node(env.board, gamma=0.9)

        # Run MCTS from current position
        tree = searcher.mcts(tree)

        # Pick move with highest mean value
        best, best_val = None, -np.inf
        for move, child in tree.children.items():
            val = np.mean(child.values) if child.values else -np.inf
            if val > best_val:
                best_val = val
                best = move

        return best or next(iter(self.board.legal_moves))

    def best_move_fast(self):
        """Fast move selection without MCTS (single-ply evaluation)."""
        best, best_val = None, -np.inf
        for move in self.board.legal_moves:
            self.board.push(move)
            layer = self.encode_board()
            val = float(np.squeeze(self.agent.predict(np.expand_dims(layer, 0))))
            # Negate if we just moved as black (agent evaluates from white's perspective)
            if not self.board.turn:  # it's now opponent's turn, so we just moved
                val = val
            else:
                val = -val
            if val > best_val:
                best_val = val
                best = move
            self.board.pop()
        return best or next(iter(self.board.legal_moves))

    def run(self):
        """Main UCI loop."""
        use_mcts = True
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line == "uci":
                print(f"id name {ENGINE_NAME}")
                print(f"id author {ENGINE_AUTHOR}")
                print(f"option name SearchTime type spin default 1 min 0 max 30")
                print(f"option name Network type combo default big var big var simple var super_simple var alt")
                print(f"option name UseMCTS type check default true")
                print(f"option name Checkpoint type string default {DEFAULT_CHECKPOINT}")
                print("uciok")
                sys.stdout.flush()

            elif line == "isready":
                if self.agent is None:
                    self.load_agent()
                print("readyok")
                sys.stdout.flush()

            elif line.startswith("setoption"):
                parts = line.split(" ")
                if "SearchTime" in parts:
                    idx = parts.index("value") + 1
                    self.search_time = float(parts[idx])
                elif "Network" in parts:
                    idx = parts.index("value") + 1
                    self.network = parts[idx]
                    self.agent = None  # reload on next isready
                elif "UseMCTS" in parts:
                    idx = parts.index("value") + 1
                    use_mcts = parts[idx].lower() == "true"
                elif "Checkpoint" in parts:
                    idx = parts.index("value") + 1
                    self.checkpoint = parts[idx]
                    self.agent = None

            elif line == "ucinewgame":
                self.board = chess.Board()

            elif line.startswith("position"):
                parts = line.split(" ")
                if "startpos" in parts:
                    self.board = chess.Board()
                    if "moves" in parts:
                        moves_idx = parts.index("moves") + 1
                        for m in parts[moves_idx:]:
                            self.board.push_uci(m)
                elif "fen" in parts:
                    fen_idx = parts.index("fen") + 1
                    fen_parts = parts[fen_idx:]
                    if "moves" in fen_parts:
                        m_idx = fen_parts.index("moves")
                        fen = " ".join(fen_parts[:m_idx])
                        self.board = chess.Board(fen)
                        for m in fen_parts[m_idx + 1:]:
                            self.board.push_uci(m)
                    else:
                        self.board = chess.Board(" ".join(fen_parts))

            elif line.startswith("go"):
                if self.agent is None:
                    self.load_agent()
                if use_mcts:
                    move = self.best_move()
                else:
                    move = self.best_move_fast()
                print(f"bestmove {move}")
                sys.stdout.flush()

            elif line == "quit":
                break


if __name__ == "__main__":
    UCIEngine().run()
