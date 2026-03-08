import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import cvxpy as cp
import numpy as np

# Allow running this file directly: `python scripts/cluster_size_sensitivity.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.MA_UCMEC_dyna_noncoop import MA_UCMEC_dyna_noncoop


@dataclass
class StepMetrics:
    mean_reward: float
    mean_total_delay_ms: float
    mean_uplink_rate_mbps: float
    mean_cluster_aps_per_user: float
    offload_users: float
    mean_local_delay_ms: float
    mean_uplink_delay_ms: float
    mean_front_delay_ms: float
    mean_process_delay_ms: float
    mean_offload_delay_ms: float


def parse_cluster_sizes(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def fixed_action_onehot(m_sim: int, action_dim: int, action_idx: int) -> np.ndarray:
    idx = np.full((m_sim,), action_idx, dtype=int)
    return np.eye(action_dim, dtype=np.float32)[idx]


def reconstruct_delay_terms(
    env: MA_UCMEC_dyna_noncoop,
    task_size_prev: np.ndarray,
    task_density_prev: np.ndarray,
) -> Dict[str, np.ndarray]:
    omega = env.omega_last.copy()
    cluster_matrix = env.cluster_matrix.copy()
    uplink_rate_access = env.uplink_rate_access_b.copy()

    local_delay = np.zeros((env.M_sim, 1))
    uplink_delay = np.zeros((env.M_sim, 1))
    front_delay = np.zeros((env.M_sim, 1))
    task_mat = np.zeros((env.M_sim, env.K))

    for i in range(env.M_sim):
        if omega[i] == 0:
            local_delay[i, 0] = task_density_prev[0, i] * task_size_prev[0, i] / env.C_user[0, i]
        else:
            if uplink_rate_access[i, 0] > 0:
                uplink_delay[i, 0] = task_size_prev[0, i] / uplink_rate_access[i, 0]
            cpu_id = int(omega[i] - 1)
            task_mat[i, cpu_id] = task_size_prev[0, i] * task_density_prev[0, i]

    front_rate_user = env.front_rate_cal(omega, cluster_matrix)
    for i in range(env.M_sim):
        if omega[i] != 0:
            ap_idx = np.where(cluster_matrix[i, :] == 1)[0]
            delays = []
            for j in ap_idx:
                if front_rate_user[i, j] > 0:
                    delays.append(task_size_prev[0, i] / front_rate_user[i, j])
            if delays:
                front_delay[i, 0] = np.max(delays)

    actual_c = np.zeros((env.M_sim, env.K))
    for cpu in range(env.K):
        serve_user_id = []
        serve_user_task = []
        local_for_cpu = []
        uplink_for_cpu = []

        for u in range(env.M_sim):
            if task_mat[u, cpu] != 0:
                serve_user_id.append(u)
                serve_user_task.append(task_mat[u, cpu])
                local_for_cpu.append(local_delay[u, 0])
                uplink_for_cpu.append(uplink_delay[u, 0])

        if not serve_user_id:
            continue

        c = cp.Variable(len(serve_user_id))
        process_delay = cp.multiply(serve_user_task, cp.inv_pos(c))
        local_for_cpu = np.array(local_for_cpu)
        uplink_for_cpu = np.array(uplink_for_cpu)

        obj = cp.Minimize(cp.sum(cp.maximum(local_for_cpu, uplink_for_cpu + process_delay)))
        cons = [0 <= c, cp.sum(c) <= env.C_edge[cpu, 0]]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.SCS, verbose=False)

        c_val = c.value
        for k, uid in enumerate(serve_user_id):
            if c_val is not None:
                actual_c[uid, cpu] = c_val[k]
            else:
                actual_c[uid, cpu] = env.C_edge[cpu, 0] / len(serve_user_id)

    process_delay = np.zeros((env.M_sim, 1))
    for i in range(env.M_sim):
        if omega[i] != 0:
            cpu_id = int(omega[i] - 1)
            denom = np.sum(actual_c[i, :])
            if denom > 0:
                process_delay[i, 0] = task_mat[i, cpu_id] / denom

    offload_delay = uplink_delay + front_delay + process_delay
    total_delay = np.maximum(local_delay, offload_delay)
    total_delay = np.minimum(total_delay, 1.0)

    return {
        "local_delay": local_delay,
        "uplink_delay": uplink_delay,
        "front_delay": front_delay,
        "process_delay": process_delay,
        "offload_delay": offload_delay,
        "total_delay": total_delay,
    }


def to_step_metrics(
    env: MA_UCMEC_dyna_noncoop,
    reward: np.ndarray,
    delay_terms: Dict[str, np.ndarray],
) -> StepMetrics:
    offload_mask = env.omega_last != 0
    local_mask = ~offload_mask

    if np.any(offload_mask):
        mean_uplink_rate_mbps = float(np.mean(env.uplink_rate_access_b[offload_mask, 0]) / 1e6)
        mean_uplink_delay_ms = float(np.mean(delay_terms["uplink_delay"][offload_mask, 0]) * 1000)
        mean_front_delay_ms = float(np.mean(delay_terms["front_delay"][offload_mask, 0]) * 1000)
        mean_process_delay_ms = float(np.mean(delay_terms["process_delay"][offload_mask, 0]) * 1000)
        mean_offload_delay_ms = float(np.mean(delay_terms["offload_delay"][offload_mask, 0]) * 1000)
    else:
        mean_uplink_rate_mbps = 0.0
        mean_uplink_delay_ms = 0.0
        mean_front_delay_ms = 0.0
        mean_process_delay_ms = 0.0
        mean_offload_delay_ms = 0.0

    if np.any(local_mask):
        mean_local_delay_ms = float(np.mean(delay_terms["local_delay"][local_mask, 0]) * 1000)
    else:
        mean_local_delay_ms = 0.0

    return StepMetrics(
        mean_reward=float(np.mean(reward)),
        mean_total_delay_ms=float(np.mean(delay_terms["total_delay"]) * 1000),
        mean_uplink_rate_mbps=mean_uplink_rate_mbps,
        mean_cluster_aps_per_user=float(np.mean(np.sum(env.cluster_matrix, axis=1))),
        offload_users=float(np.count_nonzero(offload_mask)),
        mean_local_delay_ms=mean_local_delay_ms,
        mean_uplink_delay_ms=mean_uplink_delay_ms,
        mean_front_delay_ms=mean_front_delay_ms,
        mean_process_delay_ms=mean_process_delay_ms,
        mean_offload_delay_ms=mean_offload_delay_ms,
    )


def average_metrics(metrics: List[StepMetrics]) -> Dict[str, float]:
    keys = StepMetrics.__dataclass_fields__.keys()
    out = {}
    for k in keys:
        out[k] = float(np.mean([getattr(m, k) for m in metrics])) if metrics else 0.0
    return out


def run_one_cluster_size(
    cluster_size: int,
    steps: int,
    episodes: int,
    action_idx: int,
    seed: int,
) -> Dict[str, float]:
    np.random.seed(seed)
    env = MA_UCMEC_dyna_noncoop(render=False)
    env.seed(seed)
    env.action_space.seed(seed)
    env.cluster_size = cluster_size

    action = fixed_action_onehot(env.M_sim, env.action_dim, action_idx)
    obs = env.reset()
    _ = obs  # quiet lint

    metrics = []
    total_steps = steps * episodes
    for _ in range(total_steps):
        task_size_prev = env.Task_size.copy()
        task_density_prev = env.Task_density.copy()

        _, reward, done, _ = env.step(action)
        delay_terms = reconstruct_delay_terms(env, task_size_prev, task_density_prev)
        metrics.append(to_step_metrics(env, np.array(reward), delay_terms))

        if np.all(done):
            env.reset()

    summary = average_metrics(metrics)
    summary["cluster_size"] = cluster_size
    return summary


def print_table(results: List[Dict[str, float]]) -> None:
    headers = [
        "cluster_size",
        "mean_reward",
        "mean_total_delay_ms",
        "mean_uplink_rate_mbps",
        "mean_uplink_delay_ms",
        "mean_front_delay_ms",
        "mean_process_delay_ms",
        "mean_offload_delay_ms",
        "mean_local_delay_ms",
        "offload_users",
        "mean_cluster_aps_per_user",
    ]

    print(" | ".join(headers))
    print("-" * 160)
    for row in results:
        print(
            f"{int(row['cluster_size'])} | "
            f"{row['mean_reward']:.6f} | "
            f"{row['mean_total_delay_ms']:.3f} | "
            f"{row['mean_uplink_rate_mbps']:.3f} | "
            f"{row['mean_uplink_delay_ms']:.3f} | "
            f"{row['mean_front_delay_ms']:.3f} | "
            f"{row['mean_process_delay_ms']:.3f} | "
            f"{row['mean_offload_delay_ms']:.3f} | "
            f"{row['mean_local_delay_ms']:.3f} | "
            f"{row['offload_users']:.2f} | "
            f"{row['mean_cluster_aps_per_user']:.2f}"
        )


def save_csv(results: List[Dict[str, float]], path: str) -> None:
    if not results:
        return
    headers = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep cluster_size under fixed low-level actions and compare environment metrics."
    )
    parser.add_argument("--cluster-sizes", type=str, default="1,3,5,7,10")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-idx", type=int, default=3, help="Fixed discrete action index in [0, 9].")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    if args.action_idx < 0 or args.action_idx > 9:
        raise ValueError("--action-idx must be in [0, 9].")

    cluster_sizes = parse_cluster_sizes(args.cluster_sizes)
    results = []
    for cs in cluster_sizes:
        results.append(
            run_one_cluster_size(
                cluster_size=cs,
                steps=args.steps,
                episodes=args.episodes,
                action_idx=args.action_idx,
                seed=args.seed,
            )
        )

    print_table(results)
    if args.csv:
        save_csv(results, args.csv)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
