"""
CRL-Atari: Continual Reinforcement Learning on Atari with HTCL Consolidation.

Entry point that orchestrates:
    1. Training local DQN agents per game (upper bound)
    2. Sequential fine-tuning baseline
    3. EWC baseline
    4. HTCL Taylor consolidation
    5. Evaluation on all games
    6. Publication-ready plots and result summaries
"""

import argparse
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch

from utils import set_seed, get_device, load_config, ResultsManager
from agents.networks import NatureDQN
from agents.dqn import train_dqn_on_env, ReplayBuffer
from envs.wrappers import make_atari_env, get_action_mask
from consolidation.taylor import (
    taylor_update, global_catchup, ConsolidationBuffer,
)
from consolidation.ewc import compute_fisher, train_dqn_with_ewc
from evaluation.metrics import (
    evaluate_agent, evaluate_on_all_games, compute_forgetting, aggregate_metrics,
)
from visualization.plots import (
    plot_per_game_rewards, plot_forgetting, plot_performance_heatmap,
    plot_training_curves, plot_forgetting_trajectory,
    plot_aggregate_summary, plot_radar_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CRL-Atari Experiment")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Override training steps per game")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with reduced steps (50k) for testing")
    parser.add_argument("--skip-individual", action="store_true",
                        help="Skip training individual agents (use checkpoints)")
    parser.add_argument("--methods", nargs="+",
                        default=["individual", "sequential", "ewc", "htcl"],
                        help="Methods to run")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def create_eval_envs(game_ids, unified_action_dim, frame_stack, seed):
    """Create separate evaluation environments (not shared with training)."""
    envs = []
    masks = []
    for gid in game_ids:
        env = make_atari_env(gid, unified_action_dim, frame_stack, seed + 1000)
        envs.append(env)
        masks.append(get_action_mask(gid, unified_action_dim))
    return envs, masks


def run_individual(game_ids, config, device, results_mgr):
    """Train one DQN per game independently. Serves as upper bound."""
    print("\n" + "=" * 60)
    print("PHASE: Individual DQN Agents (Upper Bound)")
    print("=" * 60)

    dqn_cfg = config["dqn"]
    unified_dim = config["unified_action_dim"]
    models = {}
    buffers = {}
    logs = {}

    for gid in game_ids:
        print(f"\n--- Training individual agent for {gid} ---")
        env = make_atari_env(gid, unified_dim, dqn_cfg["frame_stack"], config["seed"])
        mask = get_action_mask(gid, unified_dim)
        model = NatureDQN(unified_dim, dqn_cfg["frame_stack"]).to(device)

        trained_model, buffer, log = train_dqn_on_env(
            model, env, mask, device, dqn_cfg,
        )
        models[gid] = trained_model
        buffers[gid] = buffer
        logs[gid] = log

        ckpt_name = f"individual_{gid.replace('/', '_')}"
        results_mgr.save_checkpoint(trained_model.state_dict(), ckpt_name)
        env.close()

    return models, buffers, logs


def run_sequential(game_ids, config, device, results_mgr):
    """Sequential fine-tuning: single model trained on games in order."""
    print("\n" + "=" * 60)
    print("PHASE: Sequential Fine-Tuning (Naive Baseline)")
    print("=" * 60)

    dqn_cfg = config["dqn"]
    unified_dim = config["unified_action_dim"]

    model = NatureDQN(unified_dim, dqn_cfg["frame_stack"]).to(device)
    performance_trajectory = {}

    eval_envs, eval_masks = create_eval_envs(
        game_ids, unified_dim, dqn_cfg["frame_stack"], config["seed"]
    )

    for stage_idx, gid in enumerate(game_ids):
        print(f"\n--- Sequential: Training on {gid} ---")
        env = make_atari_env(gid, unified_dim, dqn_cfg["frame_stack"], config["seed"])
        mask = get_action_mask(gid, unified_dim)

        model, _, _ = train_dqn_on_env(model, env, mask, device, dqn_cfg)
        env.close()

        print(f"  Evaluating after training on {gid}...")
        stage_results = evaluate_on_all_games(
            model, eval_envs, eval_masks, game_ids, device,
            config["evaluation"]["eval_episodes"],
            config["evaluation"]["eval_epsilon"],
        )
        performance_trajectory[gid] = {
            g: stage_results[g]["mean_reward"] for g in game_ids
        }

    results_mgr.save_checkpoint(model.state_dict(), "sequential_final")

    for env in eval_envs:
        env.close()

    return model, performance_trajectory


def run_ewc(game_ids, config, device, results_mgr):
    """EWC: sequential training with Fisher-based regularization."""
    print("\n" + "=" * 60)
    print("PHASE: EWC Baseline")
    print("=" * 60)

    dqn_cfg = config["dqn"]
    ewc_cfg = config["consolidation"]["ewc"]
    unified_dim = config["unified_action_dim"]

    model = NatureDQN(unified_dim, dqn_cfg["frame_stack"]).to(device)
    fisher_list = []
    star_params_list = []
    performance_trajectory = {}

    eval_envs, eval_masks = create_eval_envs(
        game_ids, unified_dim, dqn_cfg["frame_stack"], config["seed"]
    )

    for stage_idx, gid in enumerate(game_ids):
        print(f"\n--- EWC: Training on {gid} ---")
        env = make_atari_env(gid, unified_dim, dqn_cfg["frame_stack"], config["seed"])
        mask = get_action_mask(gid, unified_dim)

        model, buffer, _ = train_dqn_with_ewc(
            model, env, mask, device, dqn_cfg,
            fisher_list, star_params_list,
            lambda_ewc=ewc_cfg["lambda_ewc"],
        )
        env.close()

        transitions = buffer.sample_all(ewc_cfg["fisher_samples"])
        fisher = compute_fisher(model, transitions, device, dqn_cfg["gamma"])
        fisher_list.append(fisher)
        star_params_list.append(
            {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        )

        print(f"  Evaluating after training on {gid}...")
        stage_results = evaluate_on_all_games(
            model, eval_envs, eval_masks, game_ids, device,
            config["evaluation"]["eval_episodes"],
            config["evaluation"]["eval_epsilon"],
        )
        performance_trajectory[gid] = {
            g: stage_results[g]["mean_reward"] for g in game_ids
        }

    results_mgr.save_checkpoint(model.state_dict(), "ewc_final")

    for env in eval_envs:
        env.close()

    return model, performance_trajectory


def run_htcl(game_ids, config, device, results_mgr,
             pretrained_models=None, pretrained_buffers=None):
    """
    HTCL: consolidate local agents into a global model via Taylor update.

    If pretrained_models and pretrained_buffers are provided (from
    run_individual), those are reused as local agents. Otherwise,
    local agents are trained from scratch per game.

    Args:
        pretrained_models: Dict[game_id -> nn.Module] from run_individual
        pretrained_buffers: Dict[game_id -> ReplayBuffer] from run_individual
    """
    reuse = pretrained_models is not None and pretrained_buffers is not None
    label = "HTCL Taylor Consolidation (reusing individual agents)" if reuse \
        else "HTCL Taylor Consolidation (training local agents)"

    print("\n" + "=" * 60)
    print(f"PHASE: {label}")
    print("=" * 60)

    dqn_cfg = config["dqn"]
    htcl_cfg = config["consolidation"]["htcl"]
    unified_dim = config["unified_action_dim"]

    global_model = NatureDQN(unified_dim, dqn_cfg["frame_stack"]).to(device)
    consolidation_buffer = ConsolidationBuffer(
        max_per_game=config["consolidation"]["buffer_per_game"]
    )
    performance_trajectory = {}

    eval_envs, eval_masks = create_eval_envs(
        game_ids, unified_dim, dqn_cfg["frame_stack"], config["seed"]
    )

    for stage_idx, gid in enumerate(game_ids):
        if reuse:
            print(f"\n--- HTCL: Reusing individual agent for {gid} ---")
            local_model = copy.deepcopy(pretrained_models[gid]).to(device)
            consolidation_buffer.add_game(gid, pretrained_buffers[gid])
        else:
            print(f"\n--- HTCL: Training local agent on {gid} ---")
            env = make_atari_env(gid, unified_dim, dqn_cfg["frame_stack"], config["seed"])
            mask = get_action_mask(gid, unified_dim)

            local_model = copy.deepcopy(global_model).to(device)
            local_model, buffer, _ = train_dqn_on_env(
                local_model, env, mask, device, dqn_cfg,
            )
            env.close()
            consolidation_buffer.add_game(gid, buffer)
            del buffer

        if stage_idx == 0:
            global_model.load_state_dict(local_model.state_dict())
            print("  First game: global model initialized from local.")
        else:
            print(f"  Taylor update: consolidating {gid} into global model...")
            combined_transitions = consolidation_buffer.get_combined()
            taylor_update(
                global_model, local_model, combined_transitions, device,
                gamma=dqn_cfg["gamma"],
                eta=htcl_cfg["eta"],
                max_norm=htcl_cfg["max_norm"],
                lambda_reg=htcl_cfg["lambda_reg"],
                verbose=True,
            )

            print("  Running catch-up phase...")
            global_catchup(
                global_model, local_model, combined_transitions, device,
                num_iterations=htcl_cfg["catchup_iterations"],
                gamma=dqn_cfg["gamma"],
                eta=htcl_cfg["eta"],
                max_norm=htcl_cfg["max_norm"],
                lambda_reg=htcl_cfg["catchup_lambda"],
                catchup_lr=htcl_cfg["catchup_lr"],
                patience=htcl_cfg.get("catchup_patience", 2),
                verbose=True,
            )

        del local_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"  Evaluating global model after {gid}...")
        stage_results = evaluate_on_all_games(
            global_model, eval_envs, eval_masks, game_ids, device,
            config["evaluation"]["eval_episodes"],
            config["evaluation"]["eval_epsilon"],
        )
        performance_trajectory[gid] = {
            g: stage_results[g]["mean_reward"] for g in game_ids
        }

    results_mgr.save_checkpoint(global_model.state_dict(), "htcl_final")

    for env in eval_envs:
        env.close()

    return global_model, performance_trajectory


def final_evaluation(models, game_ids, config, device):
    """Evaluate all method models on all games."""
    print("\n" + "=" * 60)
    print("PHASE: Final Evaluation")
    print("=" * 60)

    dqn_cfg = config["dqn"]
    unified_dim = config["unified_action_dim"]
    eval_envs, eval_masks = create_eval_envs(
        game_ids, unified_dim, dqn_cfg["frame_stack"], config["seed"]
    )
    eval_cfg = config["evaluation"]

    all_results = {}
    for method_name, model in models.items():
        if isinstance(model, dict):
            # Individual: per-game models
            results = {}
            for gid in game_ids:
                if gid in model:
                    idx = game_ids.index(gid)
                    results[gid] = evaluate_agent(
                        model[gid], eval_envs[idx], eval_masks[idx], device,
                        eval_cfg["eval_episodes"], eval_cfg["eval_epsilon"],
                    )
            all_results[method_name] = results
        else:
            all_results[method_name] = evaluate_on_all_games(
                model, eval_envs, eval_masks, game_ids, device,
                eval_cfg["eval_episodes"], eval_cfg["eval_epsilon"],
            )

    for env in eval_envs:
        env.close()

    return all_results


def generate_all_plots(
    final_results, trajectories, training_logs,
    game_ids, game_labels, results_mgr,
):
    """Generate all publication-ready plots."""
    print("\n" + "=" * 60)
    print("PHASE: Generating Plots")
    print("=" * 60)

    png, svg = results_mgr.get_plot_paths("per_game_rewards")
    plot_per_game_rewards(final_results, game_labels, png, svg)
    print(f"  Saved: per_game_rewards")

    png, svg = results_mgr.get_plot_paths("performance_heatmap")
    plot_performance_heatmap(final_results, game_labels, png, svg)
    print(f"  Saved: performance_heatmap")

    png, svg = results_mgr.get_plot_paths("aggregate_summary")
    agg = {m: aggregate_metrics(final_results[m], game_ids) for m in final_results}
    plot_aggregate_summary(agg, png, svg)
    print(f"  Saved: aggregate_summary")

    png, svg = results_mgr.get_plot_paths("radar_comparison")
    plot_radar_comparison(final_results, game_labels, png, svg)
    print(f"  Saved: radar_comparison")

    if training_logs:
        png, svg = results_mgr.get_plot_paths("training_curves")
        plot_training_curves(training_logs, game_labels, png, svg)
        print(f"  Saved: training_curves")

    # Forgetting plots for methods with trajectories
    cl_methods = {m: t for m, t in trajectories.items() if t}
    if cl_methods:
        forgetting_data = {}
        for method_name, traj in cl_methods.items():
            last_stage = game_ids[-1]
            forgetting_data[method_name] = {}
            for gid in game_ids:
                rewards_across_stages = [
                    traj[stage].get(gid, 0.0)
                    for stage in game_ids
                    if gid in traj.get(stage, {})
                ]
                if len(rewards_across_stages) >= 2:
                    forgetting_data[method_name][gid] = (
                        max(rewards_across_stages) - traj[last_stage].get(gid, 0.0)
                    )
                else:
                    forgetting_data[method_name][gid] = 0.0

        png, svg = results_mgr.get_plot_paths("forgetting")
        plot_forgetting(forgetting_data, game_labels, png, svg)
        print(f"  Saved: forgetting")

        # Build trajectory format for line plot
        trajectory_lines = {}
        for method_name, traj in cl_methods.items():
            trajectory_lines[method_name] = {}
            for gid in game_ids:
                trajectory_lines[method_name][gid] = [
                    traj[stage].get(gid, 0.0) for stage in game_ids
                    if stage in traj
                ]

        png, svg = results_mgr.get_plot_paths("forgetting_trajectory")
        plot_forgetting_trajectory(
            trajectory_lines, game_labels, game_ids, png, svg,
        )
        print(f"  Saved: forgetting_trajectory")


def save_all_results(final_results, trajectories, game_ids, game_labels, config, results_mgr):
    """Save all results to CSV and JSON."""
    rows = []
    for method, results in final_results.items():
        for gid in game_ids:
            if gid in results:
                rows.append({
                    "method": method,
                    "game": game_labels.get(gid, gid),
                    "game_id": gid,
                    "mean_reward": results[gid]["mean_reward"],
                    "std_reward": results[gid]["std_reward"],
                    "min_reward": results[gid]["min_reward"],
                    "max_reward": results[gid]["max_reward"],
                })

    df = pd.DataFrame(rows)
    csv_path = results_mgr.get_csv_path("final_results")
    df.to_csv(csv_path, index=False)
    print(f"\nResults CSV: {csv_path}")

    summary = {
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_results": {
            method: {
                gid: {
                    "mean_reward": results[gid]["mean_reward"],
                    "std_reward": results[gid]["std_reward"],
                }
                for gid in game_ids if gid in results
            }
            for method, results in final_results.items()
        },
        "trajectories": trajectories,
        "aggregate": {
            method: aggregate_metrics(final_results[method], game_ids)
            for method in final_results
        },
    }
    json_path = results_mgr.save_json(summary, "experiment_summary")
    print(f"Summary JSON: {json_path}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<15} {'Mean Reward':>12} {'Std':>8}")
    print("-" * 38)
    for method in final_results:
        agg = aggregate_metrics(final_results[method], game_ids)
        print(f"{method:<15} {agg['mean_reward']:>12.1f} {agg['std_reward']:>8.1f}")
    print()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.train_steps is not None:
        config["dqn"]["train_steps"] = args.train_steps
        # Scale epsilon decay to 50% of training budget
        config["dqn"]["epsilon_decay_steps"] = args.train_steps // 2
    if args.eval_episodes is not None:
        config["evaluation"]["eval_episodes"] = args.eval_episodes
    if args.device is not None:
        config["device"] = args.device
    if args.quick:
        config["dqn"]["train_steps"] = 50000
        config["dqn"]["epsilon_decay_steps"] = 40000
        config["dqn"]["learning_starts"] = 1000
        config["dqn"]["buffer_size"] = 10000
        config["consolidation"]["buffer_per_game"] = 2000
        config["evaluation"]["eval_episodes"] = 5

    set_seed(config["seed"])
    device = get_device(config["device"])
    results_mgr = ResultsManager(config["output_dir"])
    game_ids = config["games"]
    game_labels = config.get("game_labels", {g: g for g in game_ids})

    print(f"Device: {device}")
    print(f"Seed: {config['seed']}")
    print(f"Games: {game_ids}")
    print(f"Train steps per game: {config['dqn']['train_steps']}")
    print(f"Methods: {args.methods}")

    results_mgr.save_json(config, "config")

    models_to_eval = {}
    trajectories = {}
    training_logs = {}
    indiv_models = None
    indiv_buffers = None

    # --- Individual agents ---
    if "individual" in args.methods:
        indiv_models, indiv_buffers, indiv_logs = run_individual(
            game_ids, config, device, results_mgr,
        )
        models_to_eval["Individual"] = indiv_models
        training_logs = indiv_logs
        trajectories["Individual"] = {}

    # --- Sequential baseline ---
    if "sequential" in args.methods:
        seq_model, seq_traj = run_sequential(
            game_ids, config, device, results_mgr,
        )
        models_to_eval["Sequential"] = seq_model
        trajectories["Sequential"] = seq_traj

    # --- EWC ---
    if "ewc" in args.methods:
        ewc_model, ewc_traj = run_ewc(
            game_ids, config, device, results_mgr,
        )
        models_to_eval["EWC"] = ewc_model
        trajectories["EWC"] = ewc_traj

    # --- HTCL (reuses individual agents if available) ---
    if "htcl" in args.methods:
        htcl_model, htcl_traj = run_htcl(
            game_ids, config, device, results_mgr,
            pretrained_models=indiv_models,
            pretrained_buffers=indiv_buffers,
        )
        models_to_eval["HTCL"] = htcl_model
        trajectories["HTCL"] = htcl_traj

    # --- Final evaluation ---
    final_results = final_evaluation(models_to_eval, game_ids, config, device)

    # --- Plots ---
    generate_all_plots(
        final_results, trajectories, training_logs,
        game_ids, game_labels, results_mgr,
    )

    # --- Save ---
    save_all_results(
        final_results, trajectories, game_ids, game_labels, config, results_mgr,
    )


if __name__ == "__main__":
    main()
