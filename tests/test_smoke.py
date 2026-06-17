"""Smoke tests for chess RL project."""
import numpy as np
import pytest

from real_chess.agent import Agent, GreedyAgent, RandomAgent
from real_chess.environment import Board


class TestEnvironment:
    def test_board_init(self):
        env = Board(opposing_agent=GreedyAgent())
        assert env.layer_board.shape == (8, 8, 8)
        assert env.board.turn  # White to move

    def test_board_reset(self):
        env = Board(opposing_agent=GreedyAgent())
        env.step(list(env.board.generate_legal_moves())[0])
        env.reset()
        assert env.board.fullmove_number == 1

    def test_step_returns_reward(self):
        env = Board(opposing_agent=GreedyAgent())
        move = list(env.board.generate_legal_moves())[0]
        episode_end, reward = env.step(move)
        assert isinstance(episode_end, bool)
        assert isinstance(reward, (int, float))

    def test_layer_board_updates(self):
        env = Board(opposing_agent=GreedyAgent())
        initial = env.layer_board.copy()
        move = list(env.board.generate_legal_moves())[0]
        env.step(move)
        assert not np.array_equal(env.layer_board, initial)


class TestAgents:
    def test_random_agent_select_move(self):
        agent = RandomAgent()
        env = Board(opposing_agent=agent)
        move = agent.select_move(env.board)
        assert move in env.board.legal_moves

    def test_greedy_agent_predict(self):
        agent = GreedyAgent()
        env = Board(opposing_agent=agent)
        val = agent.predict(np.expand_dims(env.layer_board, axis=0))
        assert -1.0 <= val <= 1.0

    def test_agent_model_predict(self):
        agent = Agent(lr=0.003, network='simple')
        env = Board(opposing_agent=GreedyAgent())
        state = np.expand_dims(env.layer_board, axis=0)
        val = agent.model.predict(state, verbose=0)
        assert val.shape == (1, 1)


class TestTrainImport:
    def test_train_module_imports(self):
        import train
        assert hasattr(train, 'main')

    def test_evaluate_module_imports(self):
        import evaluate
        assert hasattr(evaluate, 'main')
        assert hasattr(evaluate, 'play_game')
