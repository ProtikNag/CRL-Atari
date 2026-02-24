"""
Watch a trained agent play an Atari game with a graphical window.

Loads the best checkpoint for the selected game and renders the gameplay
in real-time using pygame.

Usage:
    python scripts/play.py --game Pong
    python scripts/play.py --game Breakout --speed 2
    python scripts/play.py --game SpaceInvaders --model-path path/to/model.pt
    python scripts/play.py --list-games

Controls:
    Q / ESC    - Quit
    R          - Restart episode
    P / SPACE  - Pause / Resume
    + / -      - Speed up / slow down
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn import DQNNetwork
from src.data.atari_wrappers import get_valid_actions
from src.utils.config import get_effective_config

# Lazy import: gymnasium and pygame loaded after arg parsing
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


# ── Supported games & their display names ────────────────────────────────────

GAME_ALIASES = {
    "pong": "PongNoFrameskip-v4",
    "breakout": "BreakoutNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4",
    "space_invaders": "SpaceInvadersNoFrameskip-v4",
}


def resolve_game(name: str) -> str:
    """Convert a friendly game name to the full env_id."""
    key = name.lower().replace("-", "").replace("_", "")
    if key in GAME_ALIASES:
        return GAME_ALIASES[key]
    # Try direct match
    if "NoFrameskip" in name:
        return name
    # Try appending suffix
    candidate = f"{name}NoFrameskip-v4"
    return candidate


def find_best_checkpoint(game_name: str, tag: str, checkpoint_dir: str) -> str:
    """Locate the best checkpoint for a given game.

    Args:
        game_name: Short game name (e.g., 'Pong').
        tag: Experiment tag.
        checkpoint_dir: Base checkpoint directory.

    Returns:
        Path to the best checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    short_name = game_name.replace("NoFrameskip-v4", "")
    candidates = [
        os.path.join(checkpoint_dir, tag, f"expert_{short_name}_best.pt"),
        os.path.join(checkpoint_dir, tag, f"expert_{short_name}_final.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No checkpoint found for '{short_name}' in {os.path.join(checkpoint_dir, tag)}/. "
        f"Tried: {[os.path.basename(c) for c in candidates]}"
    )


def build_model(config: dict, device: str) -> DQNNetwork:
    """Construct a DQN model from config."""
    model_cfg = config["model"]
    return DQNNetwork(
        in_channels=config["env"]["frame_stack"],
        conv_channels=model_cfg["conv_channels"],
        conv_kernels=model_cfg["conv_kernels"],
        conv_strides=model_cfg["conv_strides"],
        fc_hidden=model_cfg["fc_hidden"],
        unified_action_dim=model_cfg["unified_action_dim"],
        dueling=model_cfg.get("dueling", False),
    ).to(device)


def make_render_env(env_id: str, seed: int = 42) -> gym.Env:
    """Create a raw Atari env with rgb_array rendering (no wrappers)."""
    env = gym.make(env_id, render_mode="rgb_array")
    return env


def make_agent_env(env_id: str, config: dict) -> gym.Env:
    """Create the wrapped env for agent observation (same as training, no clip_reward)."""
    from src.data.atari_wrappers import make_atari_env
    env_cfg = config["env"]
    return make_atari_env(
        env_id=env_id,
        seed=config["seed"],
        frame_stack=env_cfg["frame_stack"],
        frame_skip=env_cfg["frame_skip"],
        screen_size=env_cfg["screen_size"],
        noop_max=env_cfg["noop_max"],
        episodic_life=False,
        clip_reward=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained agent play Atari.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/play.py --game Pong\n"
            "  python scripts/play.py --game Breakout --speed 2\n"
            "  python scripts/play.py --game SpaceInvaders --model-path results/checkpoints/default/consolidated_htcl.pt\n"
            "  python scripts/play.py --list-games\n"
        ),
    )
    parser.add_argument(
        "--game", type=str, default=None,
        help="Game to play (e.g., Pong, Breakout, SpaceInvaders).",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model checkpoint. If omitted, uses best expert checkpoint.",
    )
    parser.add_argument(
        "--tag", type=str, default="default",
        help="Experiment tag for finding checkpoints.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--episodes", type=int, default=0,
        help="Number of episodes to play (0 = unlimited).",
    )
    parser.add_argument(
        "--scale", type=int, default=3,
        help="Window scale factor (default: 3, gives 480x630 window).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect).",
    )
    parser.add_argument(
        "--list-games", action="store_true",
        help="List available games and exit.",
    )
    args = parser.parse_args()

    config = get_effective_config(args.config)

    if args.list_games:
        checkpoint_dir = config["logging"]["checkpoint_dir"]
        tag_dir = os.path.join(checkpoint_dir, args.tag)
        print("Available games in task sequence:")
        for env_id in config["task_sequence"]:
            short = env_id.replace("NoFrameskip-v4", "")
            ckpt = os.path.join(tag_dir, f"expert_{short}_best.pt")
            status = "checkpoint found" if os.path.exists(ckpt) else "NO CHECKPOINT"
            print(f"  {short:20s} ({env_id}) [{status}]")
        # Also check for consolidated models
        print("\nConsolidated models:")
        for method in ["ewc", "distillation", "htcl"]:
            ckpt = os.path.join(tag_dir, f"consolidated_{method}.pt")
            status = "found" if os.path.exists(ckpt) else "not found"
            print(f"  {method:20s} [{status}]")
        return

    if args.game is None:
        parser.error("--game is required (or use --list-games)")

    # Resolve game name
    env_id = resolve_game(args.game)
    game_short = env_id.replace("NoFrameskip-v4", "")
    print(f"Game: {game_short} ({env_id})")

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    model = build_model(config, device)
    if args.model_path:
        ckpt_path = args.model_path
    else:
        ckpt_path = find_best_checkpoint(
            env_id, args.tag, config["logging"]["checkpoint_dir"]
        )
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle both full agent checkpoints and bare state_dicts
    if "policy_net" in checkpoint:
        model.load_state_dict(checkpoint["policy_net"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    valid_actions = get_valid_actions(env_id)
    print(f"Valid actions: {valid_actions}")

    # ── Import pygame ─────────────────────────────────────────────────────
    try:
        import pygame
    except ImportError:
        print("pygame is required for the graphical viewer.")
        print("Install it with: pip install pygame")
        sys.exit(1)

    # ── Create environments ───────────────────────────────────────────────
    # Raw env for RGB rendering
    render_env = make_render_env(env_id, seed=config["seed"])
    # Wrapped env for agent observations
    agent_env = make_agent_env(env_id, config)

    # ── Pygame setup ──────────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption(f"CRL-Atari: {game_short} (Agent Playing)")

    # Get initial frame to determine dimensions
    raw_obs, _ = render_env.reset()
    frame = render_env.render()
    frame_h, frame_w = frame.shape[:2]

    # Scale window
    scale = args.scale
    win_w = frame_w * scale
    win_h = frame_h * scale + 60  # extra space for HUD

    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)
    big_font = pygame.font.SysFont("monospace", 24, bold=True)

    # ── Game loop ─────────────────────────────────────────────────────────
    base_fps = 30
    speed = args.speed
    paused = False
    episode = 0
    max_episodes = args.episodes

    def reset_both():
        """Reset both environments and sync."""
        nonlocal episode
        episode += 1
        r_obs, _ = render_env.reset(seed=config["seed"] + episode)
        a_obs, _ = agent_env.reset(seed=config["seed"] + episode)
        return r_obs, a_obs

    _, agent_obs = reset_both()
    total_reward = 0.0
    step_count = 0
    episode_rewards = []

    running = True
    while running:
        # ── Event handling ────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    _, agent_obs = reset_both()
                    total_reward = 0.0
                    step_count = 0
                elif event.key in (pygame.K_p, pygame.K_SPACE):
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed + 0.5, 10.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed - 0.5, 0.5)

        if paused:
            # Draw "PAUSED" overlay
            pause_text = big_font.render("PAUSED", True, (255, 255, 0))
            rect = pause_text.get_rect(center=(win_w // 2, win_h // 2))
            screen.blit(pause_text, rect)
            pygame.display.flip()
            clock.tick(10)
            continue

        # ── Agent selects action ──────────────────────────────────────
        with torch.no_grad():
            state_tensor = (
                torch.from_numpy(agent_obs).float().unsqueeze(0).to(device) / 255.0
            )
            q_values = model(state_tensor)
            mask = torch.full(
                (model.unified_action_dim,), float("-inf"), device=device
            )
            mask[valid_actions] = 0.0
            masked_q = q_values + mask.unsqueeze(0)
            action = masked_q.argmax(dim=1).item()

        # ── Step both environments ────────────────────────────────────
        # Step render env (raw, action needs to be mapped to local)
        # The agent env has UnifiedActionWrapper, but render env doesn't
        minimal_actions = render_env.unwrapped.ale.getMinimalActionSet()
        unified_to_local = {int(a): i for i, a in enumerate(minimal_actions)}
        local_action = unified_to_local.get(action, 0)
        raw_obs, raw_reward, raw_term, raw_trunc, _ = render_env.step(local_action)

        # Step agent env (wrapped)
        agent_obs, _, a_term, a_trunc, _ = agent_env.step(action)
        total_reward += raw_reward
        step_count += 1

        # ── Render frame ──────────────────────────────────────────────
        frame = render_env.render()
        # Convert to pygame surface
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (win_w, frame_h * scale))
        screen.fill((20, 20, 30))
        screen.blit(surf, (0, 0))

        # ── HUD ───────────────────────────────────────────────────────
        hud_y = frame_h * scale + 5
        reward_color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
        hud_lines = [
            (f"EP:{episode:3d}  REWARD:{total_reward:8.1f}  STEP:{step_count:5d}", reward_color),
            (f"SPD:{speed:.1f}x  [Q/ESC]Quit [R]Reset [P]Pause [+/-]Speed", (180, 180, 180)),
        ]
        for i, (text, color) in enumerate(hud_lines):
            rendered = font.render(text, True, color)
            screen.blit(rendered, (10, hud_y + i * 22))

        pygame.display.flip()
        clock.tick(int(base_fps * speed))

        # ── Episode end ───────────────────────────────────────────────
        if raw_term or raw_trunc:
            episode_rewards.append(total_reward)
            print(
                f"Episode {episode}: reward = {total_reward:.1f} "
                f"(steps = {step_count})"
            )

            if max_episodes > 0 and episode >= max_episodes:
                running = False
            else:
                # Brief pause to show final score
                time.sleep(0.5)
                _, agent_obs = reset_both()
                total_reward = 0.0
                step_count = 0

    # ── Cleanup ──────────────────────────────────────────────────────────
    pygame.quit()
    render_env.close()
    agent_env.close()

    if episode_rewards:
        print(f"\nSession summary: {len(episode_rewards)} episodes")
        print(f"  Mean reward: {np.mean(episode_rewards):.1f}")
        print(f"  Best reward: {np.max(episode_rewards):.1f}")
        print(f"  Worst reward: {np.min(episode_rewards):.1f}")


if __name__ == "__main__":
    main()
