"""
Diagnostic script: run this from the crl-atari project root.
    python diagnose.py

It will check every stage of the pipeline and print exactly
where things go wrong.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

# --- 1. Environment observation shape ---
print("=" * 60)
print("1. ENVIRONMENT DIAGNOSTICS")
print("=" * 60)

try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, TypeError):
    pass

env = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0)
env = gym.wrappers.AtariPreprocessing(
    env, noop_max=30, frame_skip=4, screen_size=84,
    terminal_on_life_loss=True, grayscale_obs=True, scale_obs=False,
)
print(f"After AtariPreprocessing: obs_space = {env.observation_space}")

try:
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    print("Using: FrameStackObservation")
except AttributeError:
    env = gym.wrappers.FrameStack(env, num_stack=4)
    print("Using: FrameStack (legacy)")

print(f"After FrameStack: obs_space = {env.observation_space}")
print(f"  shape = {env.observation_space.shape}")
print(f"  dtype = {env.observation_space.dtype}")

obs, info = env.reset(seed=42)
obs = np.array(obs)
print(f"\nRaw obs from reset:")
print(f"  type   = {type(obs)}")
print(f"  shape  = {obs.shape}")
print(f"  dtype  = {obs.dtype}")
print(f"  min    = {obs.min()}, max = {obs.max()}, mean = {obs.mean():.2f}")

# Check if shape is (4, 84, 84) or something else
expected = (4, 84, 84)
if obs.shape != expected:
    print(f"\n  *** SHAPE MISMATCH: expected {expected}, got {obs.shape} ***")
    print(f"  This is likely the root cause!")
    if obs.shape == (84, 84, 4):
        print(f"  FrameStack is channels-last. Need to transpose.")
else:
    print(f"  Shape OK: {obs.shape}")

# --- 2. Network forward pass ---
print("\n" + "=" * 60)
print("2. NETWORK FORWARD PASS")
print("=" * 60)

from agents.networks import NatureDQN

model = NatureDQN(num_actions=6, frame_stack=4)
model.eval()

# Test with the actual observation
state_t = torch.from_numpy(obs).unsqueeze(0)
print(f"Input tensor: shape={state_t.shape}, dtype={state_t.dtype}")

try:
    with torch.no_grad():
        q_vals = model(state_t)
    print(f"Q-values: {q_vals.numpy()[0]}")
    print(f"  shape = {q_vals.shape}")
    print(f"  mean  = {q_vals.mean().item():.6f}")
    print(f"  std   = {q_vals.std().item():.6f}")
    print(f"  range = [{q_vals.min().item():.6f}, {q_vals.max().item():.6f}]")
    if q_vals.std().item() < 1e-6:
        print("  *** Q-values are nearly identical — network can't differentiate actions ***")
except Exception as e:
    print(f"  *** FORWARD PASS FAILED: {e} ***")
    import traceback
    traceback.print_exc()

# --- 3. Replay buffer and training step ---
print("\n" + "=" * 60)
print("3. TRAINING STEP DIAGNOSTICS")
print("=" * 60)

from agents.dqn import ReplayBuffer, DQNAgent
import copy

obs_shape = env.observation_space.shape
print(f"Buffer obs_shape: {obs_shape}")

buffer = ReplayBuffer(10000, obs_shape)

# Collect some transitions
obs, _ = env.reset()
obs = np.array(obs)
for step in range(500):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_obs = np.array(next_obs)
    done = terminated or truncated
    buffer.push(obs, action, np.clip(reward, -1, 1), next_obs, done)
    if done:
        obs, _ = env.reset()
        obs = np.array(obs)
    else:
        obs = next_obs

print(f"Buffer size: {len(buffer)}")

batch = buffer.sample(32)
print(f"\nBatch shapes:")
for key, val in batch.items():
    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

# Check batch values
print(f"\nBatch states: min={batch['states'].min()}, max={batch['states'].max()}")
print(f"Batch rewards: min={batch['rewards'].min():.2f}, max={batch['rewards'].max():.2f}")
print(f"Batch dones: unique={np.unique(batch['dones'])}")
print(f"Batch actions: unique={np.unique(batch['actions'])}")

# Create agent and do training steps
model = NatureDQN(num_actions=6, frame_stack=4)
device = torch.device("cpu")
agent = DQNAgent(model, device, lr=0.00025, gamma=0.99, target_update_freq=1000, grad_clip=10.0)

print("\nRunning 10 training steps...")
for i in range(10):
    batch = buffer.sample(32)

    # Manual forward to inspect
    states = torch.from_numpy(batch["states"]).to(device)
    actions = torch.from_numpy(batch["actions"]).long().to(device)
    rewards = torch.from_numpy(batch["rewards"]).to(device)
    next_states = torch.from_numpy(batch["next_states"]).to(device)
    dones = torch.from_numpy(batch["dones"]).to(device)

    q_all = agent.online_net(states)
    q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = agent.target_net(next_states).max(1)[0]
        targets = rewards + 0.99 * (1.0 - dones) * next_q

    loss = F.smooth_l1_loss(q_values, targets)

    agent.optimizer.zero_grad()
    loss.backward()

    total_norm = sum(p.grad.norm().item() ** 2
                     for p in agent.online_net.parameters()
                     if p.grad is not None) ** 0.5

    agent.optimizer.step()

    print(f"  Step {i}: loss={loss.item():.6f}, "
          f"q_mean={q_values.mean().item():.6f}, "
          f"q_std={q_values.std().item():.6f}, "
          f"target_mean={targets.mean().item():.6f}, "
          f"grad_norm={total_norm:.6f}")

# Check if Q-values changed after training
print("\n--- Q-value evolution check ---")
test_obs = buffer.states[0]
state_t = torch.from_numpy(test_obs).unsqueeze(0)
with torch.no_grad():
    q_after = agent.online_net(state_t)
print(f"Q-values after 10 steps: {q_after.numpy()[0]}")

# --- 4. Action distribution during play ---
print("\n" + "=" * 60)
print("4. ACTION DISTRIBUTION (500 random steps)")
print("=" * 60)

action_mask = np.ones(6, dtype=np.float32)
action_mask[:6] = 1.0  # Pong has 6 valid actions
action_counts = np.zeros(6, dtype=int)

obs, _ = env.reset()
obs = np.array(obs)
for _ in range(500):
    action = agent.online_net.select_action(obs, action_mask, epsilon=0.0, device=device)
    action_counts[action] += 1
    next_obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        obs = np.array(obs)
    else:
        obs = np.array(next_obs)

action_names = ["NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE"]
print("Greedy action distribution (epsilon=0):")
for i, (name, count) in enumerate(zip(action_names, action_counts)):
    print(f"  {name:>10}: {count:>4} ({100*count/500:.1f}%)")

if action_counts.max() / 500 > 0.95:
    dominant = action_names[action_counts.argmax()]
    print(f"\n  *** Agent almost always picks {dominant} — Q-values are degenerate ***")

env.close()
print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
