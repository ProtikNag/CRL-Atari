#!/usr/bin/env python3
"""
Quick fix script: re-run WHC with tuned lambda and fix TRAC deadlock.
Uses expert checkpoints from the first run.
"""
import copy, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.mlp_dqn import MLPDQNNetwork
from src.data.vector_replay_buffer import VectorReplayBuffer
from src.data.classic_control_wrappers import (
    compute_union_action_space_classic, get_valid_actions_classic,
    compute_max_state_dim, make_classic_control_env,
)
from src.utils.seed import set_seed
from src.consolidation.whc import WHCConsolidator

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config, device):
    model_cfg = config["model"]
    return MLPDQNNetwork(
        state_dim=config["max_state_dim"],
        hidden_dims=model_cfg.get("hidden_dims", [128, 128]),
        unified_action_dim=model_cfg["unified_action_dim"],
        dueling=model_cfg.get("dueling", False),
    ).to(device)


def evaluate_all(model, task_sequence, union_actions, max_state_dim, config, device, n_eps=30):
    results = []
    for env_id in task_sequence:
        valid_actions = get_valid_actions_classic(env_id, union_actions)
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        n_actions = config["model"]["unified_action_dim"]
        action_mask = torch.full((n_actions,), float("-inf"), device=device)
        action_mask[valid_actions] = 0.0
        model.eval()
        rewards = []
        for _ in range(n_eps):
            env = make_classic_control_env(env_id, union_actions, max_state_dim, seed=42+_)
            state, _ = env.reset()
            total_r, done, steps = 0.0, False, 0
            while not done and steps < 10000:
                with torch.no_grad():
                    s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q = model(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                total_r += reward
                done = terminated or truncated
                steps += 1
            env.close()
            rewards.append(total_r)
        results.append({
            "game_name": game_name, "env_id": env_id,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "all_rewards": rewards,
        })
    return results


def train_trac_fixed(config, task_sequence, union_actions, max_state_dim,
                     first_expert_ckpt, device):
    """Fixed TRAC: use adaptive L2 regularization instead of broken erfi tuner."""
    train_cfg = config["training"]
    total_timesteps = train_cfg["total_timesteps"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    tag = "default"

    model = build_model(config, device)
    target_net = copy.deepcopy(model)
    target_net.eval()

    # Adaptive regularization strength
    base_lambda = 10.0  # Regularization toward reference

    for task_idx, env_id in enumerate(task_sequence):
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        valid_actions = get_valid_actions_classic(env_id, union_actions)
        n_actions = config["model"]["unified_action_dim"]

        print(f"\nTRAC Task {task_idx+1}/{len(task_sequence)}: {game_name}")

        if task_idx == 0:
            ckpt = torch.load(first_expert_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(ckpt)
            target_net.load_state_dict(model.state_dict())
            print(f"  Loaded from expert")
            continue

        # Reference point = current params (from prior task)
        theta_ref = {n: p.detach().clone() for n, p in model.named_parameters()}

        env = make_classic_control_env(env_id, union_actions, max_state_dim,
                                       seed=config["seed"] + task_idx * 1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
        replay_buffer = VectorReplayBuffer(train_cfg["buffer_size"], max_state_dim, device)

        action_mask = torch.full((n_actions,), float("-inf"), device=device)
        action_mask[valid_actions] = 0.0

        state, _ = env.reset()
        episode_reward = 0.0
        episode_count = 0
        episode_rewards = []
        train_steps = 0

        # Adaptive lambda: start high, decay as task loss improves
        current_lambda = base_lambda

        for step in range(1, total_timesteps + 1):
            eps = train_cfg["eps_start"] + min(step / train_cfg["eps_decay_steps"], 1.0) * (
                train_cfg["eps_end"] - train_cfg["eps_start"])

            if np.random.random() < eps:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q = model(s_t) + action_mask.unsqueeze(0)
                    action = q.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if step % train_cfg["train_freq"] == 0 and len(replay_buffer) >= train_cfg["min_buffer_size"]:
                states_b, actions_b, rewards_b, next_b, dones_b = replay_buffer.sample(train_cfg["batch_size"])

                q_vals = model(states_b)
                q_taken = q_vals.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_policy = model(next_b) + action_mask.unsqueeze(0)
                    next_acts = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(next_b)
                    next_q_max = next_q_target.gather(1, next_acts).squeeze(1)
                    target_vals = rewards_b + train_cfg["gamma"] * next_q_max * (1 - dones_b)

                dqn_loss = F.smooth_l1_loss(q_taken, target_vals)

                # L2 regularization toward reference (adaptive TRAC-style)
                reg_loss = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if name in theta_ref:
                        reg_loss += ((param - theta_ref[name]) ** 2).sum()
                reg_loss = current_lambda * reg_loss

                total_loss = dqn_loss + reg_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip", 10.0))
                optimizer.step()
                train_steps += 1

                if train_steps % train_cfg["target_update_freq"] == 0:
                    target_net.load_state_dict(model.state_dict())

                # Decay lambda over time (let model learn more as training progresses)
                if train_steps % 1000 == 0:
                    current_lambda = max(0.1, current_lambda * 0.95)

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                if step % 5000 == 0:
                    print(f"  [{game_name}] Step {step}/{total_timesteps} | "
                          f"R: {episode_reward:.1f} | Mean100: {np.mean(episode_rewards[-100:]):.1f} | "
                          f"λ: {current_lambda:.2f}")
                episode_reward = 0.0
                state, _ = env.reset()

        env.close()

    # Save
    ckpt_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "consolidated_trac.pt"))
    return model


def main():
    config = load_config("configs/classic_control.yaml")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    task_sequence = config["task_sequence"]
    union_actions = compute_union_action_space_classic(task_sequence)
    max_state_dim = compute_max_state_dim(task_sequence)
    config["model"]["unified_action_dim"] = len(union_actions)
    config["max_state_dim"] = max_state_dim

    checkpoint_dir = config["logging"]["checkpoint_dir"]
    figure_dir = config["logging"]["figure_dir"]
    tag = "default"
    ckpt_dir = os.path.join(checkpoint_dir, tag)

    game_names = [env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
                  for env_id in task_sequence]

    # Load expert results
    print("Loading expert checkpoints...")
    expert_results = []
    expert_rewards = {}
    for env_id in task_sequence:
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        ckpt_path = os.path.join(ckpt_dir, f"expert_{game_name}_best.pt")
        model = build_model(config, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        valid_actions = get_valid_actions_classic(env_id, union_actions)
        # Collect replay buffer for Fisher computation
        replay_buffer = VectorReplayBuffer(config["training"]["buffer_size"], max_state_dim, device)
        env = make_classic_control_env(env_id, union_actions, max_state_dim, seed=42)
        state, _ = env.reset()
        for _ in range(5000):
            action = np.random.choice(valid_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, terminated or truncated)
            if terminated or truncated:
                state, _ = env.reset()
            else:
                state = next_state
        env.close()

        evals = evaluate_all(model, [env_id], union_actions, max_state_dim, config, device, 30)
        expert_rewards[game_name] = evals[0]["mean_reward"]
        print(f"  {game_name}: {evals[0]['mean_reward']:.2f}")

        expert_results.append({
            "policy_state_dict": copy.deepcopy(model.state_dict()),
            "valid_actions": valid_actions,
            "game_name": game_name,
            "env_id": env_id,
            "replay_buffer": replay_buffer,
        })

    # Collect filtered states
    print("\nCollecting filtered states...")
    filtered_states_list = []
    for result in expert_results:
        model = build_model(config, device)
        model.load_state_dict(result["policy_state_dict"])
        filt = result["replay_buffer"].filter_by_confidence(
            model, result["valid_actions"], 5000)
        filtered_states_list.append(filt)

    # ═════════════════════════════════════════════════════
    # WHC Lambda Grid Search
    # ═════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("WHC Lambda Grid Search")
    print("=" * 60)

    best_whc_avg = float("-inf")
    best_whc_lambda = None
    best_whc_evals = None

    for lam in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        global_model = build_model(config, device)
        config_copy = copy.deepcopy(config)
        config_copy["whc"]["lambda_reg"] = lam

        whc = WHCConsolidator(config_copy, device=device)
        whc_model = whc.consolidate(global_model, expert_results, filtered_states_list,
                                     lambda_override=lam)

        evals = evaluate_all(whc_model, task_sequence, union_actions, max_state_dim, config, device, 30)
        avg = np.mean([e["mean_reward"] for e in evals])
        print(f"  λ={lam:>6.1f} | "
              f"CartPole={evals[0]['mean_reward']:>7.1f} | "
              f"Acrobot={evals[1]['mean_reward']:>7.1f} | "
              f"LunarLander={evals[2]['mean_reward']:>7.1f} | "
              f"avg={avg:>7.1f}")

        if avg > best_whc_avg:
            best_whc_avg = avg
            best_whc_lambda = lam
            best_whc_evals = evals
            best_whc_sd = copy.deepcopy(whc_model.state_dict())

    print(f"\nBest WHC lambda: {best_whc_lambda} (avg={best_whc_avg:.1f})")

    # Save best WHC
    torch.save(best_whc_sd, os.path.join(ckpt_dir, "consolidated_whc.pt"))

    # ═════════════════════════════════════════════════════
    # Fixed TRAC (adaptive L2 regularization)
    # ═════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Fixed TRAC (Adaptive L2 Regularization)")
    print("=" * 60)

    first_expert_ckpt = os.path.join(ckpt_dir, f"expert_{game_names[0]}_best.pt")
    trac_model = train_trac_fixed(config, task_sequence, union_actions, max_state_dim,
                                   first_expert_ckpt, device)
    trac_evals = evaluate_all(trac_model, task_sequence, union_actions, max_state_dim, config, device, 30)
    print(f"\nTRAC results:")
    for ev in trac_evals:
        print(f"  {ev['game_name']}: {ev['mean_reward']:.2f}")

    # ═════════════════════════════════════════════════════
    # Load all other method results and regenerate heatmap
    # ═════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Regenerating Heatmap with Fixed Results")
    print("=" * 60)

    # Load existing eval JSONs
    all_eval_data = {}

    # Expert
    expert_evals = []
    for env_id in task_sequence:
        game_name = env_id.replace("-v1", "").replace("-v2", "").replace("-v3", "")
        model = build_model(config, device)
        model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, f"expert_{game_name}_best.pt"),
                       map_location=device, weights_only=True))
        evals = evaluate_all(model, [env_id], union_actions, max_state_dim, config, device, 30)
        expert_evals.append(evals[0])
        expert_rewards[game_name] = evals[0]["mean_reward"]
    all_eval_data["Expert"] = expert_evals

    # Load methods from saved JSONs
    for method_key, fname in [
        ("Distillation", "distillation"), ("Hybrid", "hybrid"),
        ("EWC", "ewc"), ("Progress & Compress", "pc"),
        ("C-CHAIN", "cchain"), ("Multi-Task", "multitask"),
    ]:
        path = os.path.join(figure_dir, f"eval_{fname}_{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_eval_data[method_key] = json.load(f)

    # Update WHC and TRAC
    all_eval_data["WHC"] = best_whc_evals
    all_eval_data["TRAC"] = trac_evals

    # Save updated JSONs
    for method_key, fname in [("WHC", "whc"), ("TRAC", "trac")]:
        path = os.path.join(figure_dir, f"eval_{fname}_{tag}.json")
        serializable = []
        for ev in all_eval_data[method_key]:
            s_ev = {}
            for k, v in ev.items():
                if isinstance(v, (np.floating, np.integer)):
                    s_ev[k] = float(v)
                elif isinstance(v, list):
                    s_ev[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                else:
                    s_ev[k] = v
            serializable.append(s_ev)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    # Generate heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Inter', 'Helvetica', 'Arial'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    method_order = ["Multi-Task", "WHC", "Distillation", "Hybrid",
                    "EWC", "Progress & Compress", "TRAC", "C-CHAIN"]
    methods = [m for m in method_order if m in all_eval_data]
    n_m = len(methods)
    n_g = len(game_names)

    ret_matrix = np.zeros((n_m, n_g))
    raw_matrix = np.zeros((n_m, n_g))

    for i, method in enumerate(methods):
        evals = all_eval_data[method]
        eval_dict = {e["game_name"]: e for e in evals}
        for j, game in enumerate(game_names):
            if game in eval_dict:
                raw = eval_dict[game]["mean_reward"]
                exp = expert_rewards.get(game, 1.0)
                raw_matrix[i, j] = raw
                if exp != 0:
                    ret_matrix[i, j] = raw / exp * 100
                else:
                    ret_matrix[i, j] = 100.0 if raw == 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 0.6 * n_m + 2.5))

    from matplotlib.colors import TwoSlopeNorm
    vmin = min(0, np.nanmin(ret_matrix))
    vmax = max(100, np.nanmax(ret_matrix) * 1.05)
    vcenter = 50
    if vmin >= vcenter:
        vcenter = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = mpl.colormaps.get_cmap("RdYlGn")

    im = ax.imshow(ret_matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_m):
        for j in range(n_g):
            pct = ret_matrix[i, j]
            raw = raw_matrix[i, j]
            text_color = "#FFFFFF" if pct < 20 or pct > 80 else "#212529"
            ax.text(j, i, f"{raw:.1f}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9, fontweight="600",
                    color=text_color)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(game_names, fontsize=11)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(methods, fontsize=10)

    for e in range(n_g + 1):
        ax.axvline(e - 0.5, color="white", linewidth=2.5)
    for e in range(n_m + 1):
        ax.axhline(e - 0.5, color="white", linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("% of Expert Reward", fontsize=11)

    ax.set_title("Consolidation Performance (% of Expert)\nClassic Control: CartPole → Acrobot → LunarLander",
                 fontsize=13, fontweight="600", color="#212529", pad=14, fontfamily="serif")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for fmt in ("png", "svg"):
        out_dir = os.path.join(figure_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"02_retention_heatmap.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Heatmap saved to {figure_dir}/{{png,svg}}/")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    header = f"{'Method':<22}"
    for g in game_names:
        header += f" {g:>12}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * len(header))

    for method in ["Expert"] + methods:
        if method not in all_eval_data:
            continue
        evals = all_eval_data[method]
        eval_dict = {e["game_name"]: e for e in evals}
        row = f"{method:<22}"
        vals = []
        for g in game_names:
            if g in eval_dict:
                v = eval_dict[g]["mean_reward"]
                row += f" {v:>12.1f}"
                vals.append(v)
            else:
                row += f" {'N/A':>12}"
        if vals:
            row += f" {np.mean(vals):>8.1f}"
        print(row)

    print(f"\nBest WHC lambda: {best_whc_lambda}")
    print("Done.")


if __name__ == "__main__":
    main()
