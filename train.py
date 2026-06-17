"""Train the chess RL agent via self-play with TD learning + MCTS."""
import argparse
import os
from real_chess.agent import Agent, GreedyAgent
from real_chess.environment import Board
from real_chess.learn import TD_search


def main():
    parser = argparse.ArgumentParser(description="Train Chess RL Agent")
    parser.add_argument("--iterations", type=int, default=100, help="Number of self-play games")
    parser.add_argument("--search-time", type=float, default=1.0, help="MCTS search time per move (seconds)")
    parser.add_argument("--network", type=str, default="big", choices=["big", "simple", "super_simple", "alt", "default"], help="Network architecture")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"=== Chess RL Training ===")
    print(f"Iterations: {args.iterations}")
    print(f"Search time: {args.search_time}s")
    print(f"Network: {args.network}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print()

    network = args.network if args.network != "default" else None
    agent = Agent(lr=args.lr, network=network or "big")
    opponent = GreedyAgent()
    env = Board(opposing_agent=opponent)

    learner = TD_search(env, agent, gamma=args.gamma, search_time=args.search_time)

    print("Starting training...")
    learner.learn(iters=args.iterations)

    save_path = os.path.join(args.checkpoint_dir, "model.h5")
    agent.model.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Total reward trace length: {len(learner.reward_trace)}")


if __name__ == "__main__":
    main()
