"""Evaluate a trained chess RL agent against Random and Greedy baselines."""
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from real_chess.agent import Agent, GreedyAgent, RandomAgent
from real_chess.environment import Board


def play_game(agent_model, opponent, max_moves=80):
    """Play one game. Returns 1 (win), 0 (draw), -1 (loss) from white's perspective."""
    env = Board(opposing_agent=opponent)
    env.reset()

    for turn in range(max_moves):
        if env.board.is_game_over():
            break

        if env.board.turn:  # White (our agent)
            best_move, best_val = None, -np.inf
            for move in env.board.generate_legal_moves():
                env.step(move)
                val = np.squeeze(agent_model.predict(np.expand_dims(env.layer_board, axis=0), verbose=0))
                if val > best_val:
                    best_val = val
                    best_move = move
                env.board.pop()
                env.init_layer_board()
            env.step(best_move)
        else:  # Black (opponent)
            if isinstance(opponent, RandomAgent):
                move = opponent.select_move(env.board)
            else:
                best_move, best_val = None, -np.inf
                for move in env.board.generate_legal_moves():
                    env.step(move)
                    val = opponent.predict(np.expand_dims(env.layer_board, axis=0))
                    if val > best_val:
                        best_val = val
                        best_move = move
                    env.board.pop()
                    env.init_layer_board()
                move = best_move
            env.step(move)

    result = env.board.result()
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    return 0


def evaluate(agent_model, opponent, opponent_name, num_games):
    """Run N games and print statistics."""
    wins, draws, losses = 0, 0, 0
    for i in range(num_games):
        result = play_game(agent_model, opponent)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
        print(f"  Game {i+1}/{num_games}: {'Win' if result==1 else 'Draw' if result==0 else 'Loss'}")

    print(f"\n  vs {opponent_name}: W={wins} D={draws} L={losses} ({wins/num_games*100:.0f}% win rate)\n")
    return wins, draws, losses


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chess RL Agent")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.h5", help="Path to model file")
    parser.add_argument("--games", type=int, default=20, help="Games per opponent")
    parser.add_argument("--network", type=str, default="big", help="Network architecture (must match training)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    print(f"\n=== Evaluation ({args.games} games per opponent) ===\n")

    print("--- vs Random Agent ---")
    evaluate(model, RandomAgent(), "Random", args.games)

    print("--- vs Greedy Agent ---")
    evaluate(model, GreedyAgent(), "Greedy", args.games)


if __name__ == "__main__":
    main()
