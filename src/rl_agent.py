"""Stage 7 — RL Decision Making / DQN agent (methodology flow box 7).

Genuine Deep Q-Network machinery (NumPy implementation, no new
dependencies): Q-network forward pass, epsilon-greedy action selection,
experience-replay buffer, target network, Bellman MSE update. Action
space matches the flow diagram: forward / turn_left / turn_right / stop.

TRUTHFUL STATUS: the network is RANDOMLY INITIALIZED (seed 42) and has
never been trained -- no CARLA/simulator exists in this offline
environment (see config/reward_config.json), so no episode, reward, or
policy update has ever run. ``decide()`` executes a real forward pass
and returns real Q-values, but they carry NO driving competence and are
labelled ``trained: False`` everywhere they surface. ``train_episode()``
refuses with ``SimulatorUnavailable`` instead of faking training.

The safety override (Stage 8) is independent of the DQN: an
emergency-stop recommendation from ``src.rl_state`` always wins over the
network output and is labelled as a rule, never as a learned behavior.
"""

from __future__ import annotations

import json
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dqn_config() -> Dict[str, Any]:
    try:
        return json.loads((PROJECT_ROOT / "config" / "dqn_config.json").read_text())
    except (OSError, ValueError):
        return {}


_DQN_CFG = _load_dqn_config()
ACTIONS: List[str] = list(_DQN_CFG.get("actions", ["forward", "turn_left", "turn_right", "stop"]))


class Action(str, Enum):
    """Stable action enum (Step 2). Values match config/dqn_config.json."""

    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
STATE_DIM = int(_DQN_CFG.get("state_dim", 13))
HIDDEN_DIM = int(_DQN_CFG.get("hidden_dim", 64))
_GAMMA = float(_DQN_CFG.get("gamma", 0.99))
_LR = float(_DQN_CFG.get("learning_rate", 0.001))
_EPS = float(_DQN_CFG.get("epsilon_inference", 0.0))
_BUFFER = int(_DQN_CFG.get("replay_capacity", 10_000))
_SEED = int(_DQN_CFG.get("seed", 42))
TRAINED = bool(_DQN_CFG.get("trained", False))


class SimulatorUnavailable(RuntimeError):
    """Raised whenever training is requested without a simulator."""


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class QNetwork:
    """Two-layer MLP Q-function: 13 -> 64 (ReLU) -> 4 (Q-values)."""

    def __init__(self, seed: int | None = None):
        seed = _SEED if seed is None else int(seed)
        rng = np.random.RandomState(seed)
        self.W1 = (rng.randn(STATE_DIM, HIDDEN_DIM) * np.sqrt(2.0 / STATE_DIM)).astype(np.float64)
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float64)
        self.W2 = (rng.randn(HIDDEN_DIM, len(ACTIONS)) * np.sqrt(2.0 / HIDDEN_DIM)).astype(np.float64)
        self.b2 = np.zeros(len(ACTIONS), dtype=np.float64)
        self.n_parameters = int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def forward(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        if s.shape[0] != STATE_DIM:
            raise ValueError(f"state must have dim {STATE_DIM}. Got {s.shape!r}")
        if not np.all(np.isfinite(s)):
            raise ValueError("state must be all-finite.")
        return _relu(s @ self.W1 + self.b1) @ self.W2 + self.b2

    def copy_from(self, other: "QNetwork") -> None:
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


class DQNAgent:
    """DQN with experience replay + target network (inference usable now)."""

    def __init__(self, epsilon: float | None = None, seed: int | None = None):
        self.q = QNetwork(seed=_SEED if seed is None else seed)
        self.target = QNetwork(seed=(_SEED if seed is None else seed) + 1)
        self.target.copy_from(self.q)
        self.epsilon = float(_EPS if epsilon is None else epsilon)
        from src.rl.replay_buffer import ReplayBuffer

        self.buffer: ReplayBuffer = ReplayBuffer(capacity=_BUFFER)
        self.updates = 0
        self.trained = False  # True only after load_policy() reads trained weights
        self.training_meta: Dict[str, Any] = {"episodes_run": 0}

    def save_policy(self, path) -> Dict[str, Any]:
        """Persist Q-network weights + training metadata (pickle dict)."""
        import pickle

        payload = {"W1": self.q.W1, "b1": self.q.b1, "W2": self.q.W2,
                   "b2": self.q.b2, "actions": list(ACTIONS),
                   "state_dim": STATE_DIM, "hidden_dim": HIDDEN_DIM,
                   "trained": self.trained, "training_meta": dict(self.training_meta),
                   "updates": self.updates}
        with open(str(path), "wb") as fh:
            pickle.dump(payload, fh)
        return {"path": str(path), "trained": self.trained,
                "updates": self.updates}

    def load_policy(self, path) -> Dict[str, Any]:
        """Load trained weights; marks this instance trained (verified shapes)."""
        import pickle

        with open(str(path), "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("state_dim") != STATE_DIM or payload.get("hidden_dim") != HIDDEN_DIM:
            raise ValueError("Weight shapes do not match this DQN architecture.")
        if list(payload.get("actions", [])) != list(ACTIONS):
            raise ValueError("Weight action space does not match.")
        self.q.W1 = np.asarray(payload["W1"], dtype=np.float64)
        self.q.b1 = np.asarray(payload["b1"], dtype=np.float64)
        self.q.W2 = np.asarray(payload["W2"], dtype=np.float64)
        self.q.b2 = np.asarray(payload["b2"], dtype=np.float64)
        self.target.copy_from(self.q)
        self.trained = bool(payload.get("trained", True))
        self.training_meta = dict(payload.get("training_meta", {}))
        return {"path": str(path), "trained": self.trained,
                "training_meta": self.training_meta}

    def act(self, state: np.ndarray) -> Dict[str, Any]:
        q = self.q.forward(state)
        if np.random.rand() < self.epsilon:
            action = int(np.random.randint(len(ACTIONS)))
            greedy = False
        else:
            action = int(np.argmax(q))
            greedy = True
        return {
            "action": ACTIONS[action],
            "action_index": action,
            "q_values": [round(float(v), 4) for v in q],
            "greedy": greedy,
            "epsilon": self.epsilon,
            "trained": TRAINED,
        }

    def remember(self, s, a: int, r: float, s2, done: bool) -> None:
        self.buffer.append(s, a, r, s2, done)

    def train_step(self, gamma: float | None = None, lr: float | None = None) -> Dict[str, Any]:
        """One genuine Bellman MSE gradient step on a sampled minibatch.

        Real update math on whatever transitions were stored -- but with no
        simulator, the buffer only ever holds manually supplied transitions,
        so this never constitutes policy training (labelled accordingly).
        """
        if len(self.buffer) < 32:
            return {"updated": False,
                    "reason": "fewer than 32 stored transitions (no simulator episodes)"}
        gamma = _GAMMA if gamma is None else float(gamma)
        lr = _LR if lr is None else float(lr)
        batch = self.buffer.sample(32)
        S = np.stack([b[0] for b in batch])
        A = np.array([b[1] for b in batch])
        R = np.array([b[2] for b in batch])
        S2 = np.stack([b[3] for b in batch])
        D = np.array([b[4] for b in batch], dtype=np.float64)
        q_next = _relu(S2 @ self.target.W1 + self.target.b1) @ self.target.W2 + self.target.b2
        target_q = R + (1.0 - D) * gamma * q_next.max(axis=1)
        H = _relu(S @ self.q.W1 + self.q.b1)
        Q = H @ self.q.W2 + self.q.b2
        pred = Q[np.arange(32), A]
        err = pred - target_q
        loss = float((err ** 2).mean())
        grad_out = np.zeros_like(H @ self.q.W2)
        grad_out[np.arange(32), A] = 2.0 * err / 32.0
        dW2 = H.T @ grad_out
        db2 = grad_out.sum(axis=0)
        dH = (grad_out @ self.q.W2.T) * (H > 0)
        dW1 = S.T @ dH
        db1 = dH.sum(axis=0)
        self.q.W2 -= lr * dW2
        self.q.b2 -= lr * db2
        self.q.W1 -= lr * dW1
        self.q.b1 -= lr * db1
        self.updates += 1
        return {"updated": True, "loss": round(loss, 6), "updates": self.updates,
                "note": "mechanical update on stored transitions only; NOT policy training"}

    def train_episode(self, *args, **kwargs) -> Dict[str, Any]:
        raise SimulatorUnavailable(
            "No CARLA/simulator installed in this offline environment; "
            "DQN training episodes cannot run. See config/reward_config.json.")


_agent: DQNAgent | None = None


def get_agent() -> DQNAgent:
    global _agent
    if _agent is None:
        _agent = DQNAgent()
        # Auto-load trained weights produced on a CARLA machine, if present.
        try:
            default_weights = PROJECT_ROOT / "models" / "rl" / "dqn_weights.pkl"
            if default_weights.is_file():
                _agent.load_policy(default_weights)
        except Exception:
            pass
    return _agent


def decide(state_vector: List[float], safety: Dict[str, Any]) -> Dict[str, Any]:
    """State -> DQN action with the Stage-8 safety override applied.

    The override is a deterministic rule (emergency stop when the safety
    check fires), never presented as learned behavior.
    """
    agent = get_agent()
    s = np.asarray(state_vector, dtype=np.float64)
    out = agent.act(s)
    overridden = bool(safety.get("emergency_stop", False))
    trained = bool(getattr(agent, "trained", False))
    return {
        "actions": list(ACTIONS),
        "q_values": out["q_values"],
        "network_action": out["action"],
        "final_action": "stop" if overridden else out["action"],
        "safety_override": overridden,
        "override_rule": safety.get("rule") if overridden else None,
        "trained": trained,
        "model": "DQN Q-network 13->64(ReLU)->4 (NumPy, random init seed 42, never trained)",
        "warning": ("Untrained network: Q-values carry no driving competence; "
                    "demonstrates the data path only. Do not use for control."
                    if not trained else
                    "Trained weights loaded -- Q-values reflect simulator training; "
                    "safety override still applies and hardware remains disabled."),
    }


def describe() -> Dict[str, Any]:
    agent = get_agent()
    trained = bool(getattr(agent, "trained", False))
    return {
        "actions": list(ACTIONS),
        "state_dim": STATE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_parameters": agent.q.n_parameters,
        "trained": trained,
        "episodes_run": agent.training_meta.get("episodes_run", 0),
        "rewards_optimized": agent.training_meta.get("mean_reward", None),
        "weights": "models/rl/dqn_weights.pkl" if trained else None,
        "status": ("trained weights loaded" if trained else
                   "implemented, untrained (no simulator); inference path live"),
    }
