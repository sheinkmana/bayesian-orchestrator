from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss

from bayesian_orchestrator.config import output_dir
from bayesian_orchestrator.plotting import configure_matplotlib_cache
from bayesian_orchestrator.report import WorkflowResult, build_metadata, write_report

configure_matplotlib_cache()
import matplotlib.pyplot as plt


AGENTS = ("fast_llm", "retrieval_llm", "reasoning_llm")
AGENT_COSTS = np.asarray([0.01, 0.04, 0.12])
TRUE_AGENT_INTERCEPT = np.asarray([1.25, 1.75, 2.25])
TRUE_DIFFICULTY_SLOPE = np.asarray([-3.3, -2.25, -1.45])
TRUE_AMBIGUITY_SLOPE = np.asarray([-1.4, -0.7, -0.35])


def _require_numpyro():
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise RuntimeError(
            "The LLM routing example requires NumPyro. Install dependencies with `uv sync`."
        ) from exc
    return jax, jnp, numpyro, dist, MCMC, NUTS


def _simulate_routing_logs(
    rng: np.random.Generator,
    n: int,
    agent_prior: np.ndarray,
) -> dict[str, np.ndarray]:
    difficulty = rng.uniform(0.0, 1.0, size=n)
    ambiguity = rng.beta(2.0, 5.0, size=n)
    agent = rng.choice(len(AGENTS), size=n, p=agent_prior)
    oracle_matrix = _oracle_success_matrix(difficulty, ambiguity)
    success_prob = oracle_matrix[np.arange(n), agent]
    success = rng.binomial(1, success_prob)
    return {
        "difficulty": difficulty,
        "ambiguity": ambiguity,
        "agent": agent,
        "success": success,
        "oracle_success_prob": success_prob,
        "propensity": agent_prior[agent],
    }


def _oracle_success_matrix(difficulty: np.ndarray, ambiguity: np.ndarray) -> np.ndarray:
    logits = (
        TRUE_AGENT_INTERCEPT[None, :]
        + difficulty[:, None] * TRUE_DIFFICULTY_SLOPE[None, :]
        + ambiguity[:, None] * TRUE_AMBIGUITY_SLOPE[None, :]
    )
    return expit(logits)


def _subset_logs(logs: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[idx] for key, value in logs.items()}


def _make_model(jnp, numpyro, dist):
    def model(difficulty, ambiguity, agent, success=None):
        agent_count = len(AGENTS)
        mu_agent = numpyro.sample("mu_agent", dist.Normal(0.0, 1.5))
        tau_agent = numpyro.sample("tau_agent", dist.HalfNormal(1.0))
        with numpyro.plate("agent", agent_count):
            agent_intercept = numpyro.sample("agent_intercept", dist.Normal(mu_agent, tau_agent))
            difficulty_slope = numpyro.sample("difficulty_slope", dist.Normal(-2.0, 1.0))
            ambiguity_slope = numpyro.sample("ambiguity_slope", dist.Normal(-0.75, 0.75))
        logits = (
            agent_intercept[agent]
            + difficulty_slope[agent] * difficulty
            + ambiguity_slope[agent] * ambiguity
        )
        numpyro.sample("success", dist.Bernoulli(logits=logits), obs=success)

    return model


def _run_mcmc(key, logs: dict[str, np.ndarray], warmup: int, samples: int) -> dict[str, np.ndarray]:
    jax, jnp, numpyro, dist, MCMC, NUTS = _require_numpyro()
    model = _make_model(jnp, numpyro, dist)
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.85),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        key,
        jnp.asarray(logs["difficulty"]),
        jnp.asarray(logs["ambiguity"]),
        jnp.asarray(logs["agent"]),
        jnp.asarray(logs["success"]),
    )
    return {name: np.asarray(value) for name, value in mcmc.get_samples().items()}


def _posterior_success_probs(samples: dict[str, np.ndarray], logs: dict[str, np.ndarray]) -> np.ndarray:
    agent = logs["agent"]
    logits = (
        samples["agent_intercept"][:, agent]
        + samples["difficulty_slope"][:, agent] * logs["difficulty"][None, :]
        + samples["ambiguity_slope"][:, agent] * logs["ambiguity"][None, :]
    )
    return expit(logits).mean(axis=0)


def _candidate_logs(task_logs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    task_count = len(task_logs["difficulty"])
    return {
        "difficulty": np.repeat(task_logs["difficulty"], len(AGENTS)),
        "ambiguity": np.repeat(task_logs["ambiguity"], len(AGENTS)),
        "agent": np.tile(np.arange(len(AGENTS)), task_count),
    }


def _as_candidate_matrix(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1, len(AGENTS))


def _ece(y: np.ndarray, probs: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probs >= lower) & (probs < upper if upper < 1.0 else probs <= upper)
        if np.any(mask):
            out += np.mean(mask) * abs(float(np.mean(y[mask])) - float(np.mean(probs[mask])))
    return float(out)


def _metrics(y: np.ndarray, probs: np.ndarray, bins: int) -> dict[str, float]:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return {
        "log_score": float(log_loss(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "ece": _ece(y, probs, bins),
    }


def _expected_policy_value(
    oracle_success: np.ndarray,
    selected_agent: np.ndarray,
) -> float:
    return float(
        np.mean(oracle_success[np.arange(len(selected_agent)), selected_agent] - AGENT_COSTS[selected_agent])
    )


def _select_agents(candidate_probs: np.ndarray) -> np.ndarray:
    return np.argmax(candidate_probs - AGENT_COSTS[None, :], axis=1)


def _plot_reliability(path: Path, y: np.ndarray, predictions: dict[str, np.ndarray], bins: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="perfect calibration")
    colors = {"posterior_predictive": "#31688e"}
    edges = np.linspace(0.0, 1.0, bins + 1)
    for name, probs in predictions.items():
        xs, ys = [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (probs >= lower) & (probs < upper if upper < 1.0 else probs <= upper)
            if np.any(mask):
                xs.append(float(np.mean(probs[mask])))
                ys.append(float(np.mean(y[mask])))
        ax.plot(xs, ys, marker="o", linewidth=2, label=name, color=colors.get(name))
    ax.set_title("LLM Router Reliability Diagram")
    ax.set_xlabel("Predicted success probability")
    ax.set_ylabel("Observed route success rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_policy_values(path: Path, policy_values: dict[str, float]) -> None:
    labels = list(policy_values)
    values = [policy_values[label] for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, values, color=["#31688e", "#35a77c", "#b13f2d", "#6b4c9a"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("expected success minus cost")
    ax.set_title("Synthetic Counterfactual Policy Value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_policy_agent_mix(path: Path, policies: dict[str, np.ndarray]) -> None:
    labels = list(policies)
    counts = np.asarray(
        [np.bincount(policies[label], minlength=len(AGENTS)) / len(policies[label]) for label in labels]
    )
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(labels))
    colors = ["#31688e", "#35a77c", "#b13f2d"]
    for agent_idx, agent_name in enumerate(AGENTS):
        ax.bar(x, counts[:, agent_idx], bottom=bottom, color=colors[agent_idx], label=agent_name)
        bottom += counts[:, agent_idx]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("share of held-out tasks")
    ax.set_ylim(0, 1)
    ax.set_title("Candidate Agent Selected by Policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_bayesian_llm_orchestrator(config: dict[str, Any]) -> WorkflowResult:
    jax, _, _, _, _, _ = _require_numpyro()
    seed = int(config.get("seed", 11801))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    out_dir = output_dir(config)

    data_cfg = config.get("data", {})
    total_logs = int(data_cfg.get("log_count", 420))
    agent_prior = np.asarray(data_cfg.get("agent_prior", [0.48, 0.34, 0.18]), dtype=float)
    agent_prior = agent_prior / agent_prior.sum()

    mcmc_cfg = config.get("mcmc", {})
    warmup = int(mcmc_cfg.get("warmup", 120))
    samples = int(mcmc_cfg.get("samples", 180))

    bins = int(config.get("diagnostics", {}).get("calibration_bins", 10))

    logs = _simulate_routing_logs(rng, total_logs, agent_prior)
    order = rng.permutation(total_logs)
    train_end = int(total_logs * 0.7)
    train = _subset_logs(logs, order[:train_end])
    test = _subset_logs(logs, order[train_end:])

    key, subkey = jax.random.split(key)
    observed_samples = _run_mcmc(subkey, train, warmup, samples)

    test_logged_probs = _posterior_success_probs(observed_samples, test)

    predictions = {
        "posterior_predictive": np.clip(test_logged_probs, 1e-6, 1.0 - 1e-6),
    }
    metric_table = {name: _metrics(test["success"], probs, bins) for name, probs in predictions.items()}

    candidate_logs = _candidate_logs(test)
    candidate_base_probs = _posterior_success_probs(observed_samples, candidate_logs)
    candidate_predictions = {
        "posterior_predictive": _as_candidate_matrix(np.clip(candidate_base_probs, 1e-6, 1.0 - 1e-6)),
    }
    oracle_success = _oracle_success_matrix(test["difficulty"], test["ambiguity"])
    forecast_policies = {
        name: _select_agents(candidate_probs) for name, candidate_probs in candidate_predictions.items()
    }
    logged_policy = test["agent"]
    oracle_policy = _select_agents(oracle_success)
    policy_values = {
        "logged_policy": _expected_policy_value(oracle_success, logged_policy),
        **{
            f"{name}_policy": _expected_policy_value(oracle_success, selected_agent)
            for name, selected_agent in forecast_policies.items()
        },
        "oracle_policy": _expected_policy_value(oracle_success, oracle_policy),
    }
    selected_policy_value = policy_values["posterior_predictive_policy"]
    policy_utility_gate = selected_policy_value > policy_values["logged_policy"]

    reliability_plot = out_dir / "llm_router_reliability.png"
    policy_value_plot = out_dir / "llm_router_policy_value.png"
    policy_mix_plot = out_dir / "llm_router_policy_agent_mix.png"
    _plot_reliability(reliability_plot, test["success"], predictions, bins)
    _plot_policy_values(policy_value_plot, policy_values)
    _plot_policy_agent_mix(
        policy_mix_plot,
        {
            "logged_policy": logged_policy,
            "posterior_predictive_policy": forecast_policies["posterior_predictive"],
            "oracle_policy": oracle_policy,
        },
    )

    warnings: list[str] = [
        "Synthetic policy value uses known counterfactual outcomes for all candidate agents. Real routing logs need randomized exploration, propensities, A/B tests, or off-policy estimators."
    ]

    metrics: dict[str, Any] = {
        "train logs": len(train["success"]),
        "test logs": len(test["success"]),
        "logged policy expected true utility": round(policy_values["logged_policy"], 4),
        "selected policy expected true utility": round(selected_policy_value, 4),
        "oracle policy expected true utility": round(policy_values["oracle_policy"], 4),
        "selected policy improvement over logged": round(
            selected_policy_value - policy_values["logged_policy"], 4
        ),
        "policy gate utility beats logged": "pass" if policy_utility_gate else "review",
    }
    for name in forecast_policies:
        metrics[f"{name} policy expected true utility"] = round(policy_values[f"{name}_policy"], 4)
    for name, values in metric_table.items():
        for metric_name, value in values.items():
            metrics[f"{name} {metric_name}"] = round(value, 5)

    orchestrator_mode = config.get("mode") == "bayesian_llm_orchestrator"
    recommendation = (
        "Use posterior predictive success probability directly as the Bayesian routing signal, then score every "
        "candidate agent by posterior_success_probability - agent_cost. Treat the policy value comparison as "
        "synthetic-only; real deployment needs exploration propensities or randomized tests."
    )

    return write_report(
        output_dir=out_dir,
        title=(
            "Bayesian Orchestrator Report"
            if orchestrator_mode
            else "Bayesian LLM Router Report"
        ),
        context=(
            "A Bayesian LLM orchestrator treats routing as a decision under uncertainty. Synthetic routing logs "
            "cover three tools: a cheap fast LLM, a retrieval-augmented LLM, and a more expensive reasoning LLM. "
            "A Bayesian NumPyro controller estimates route success from task difficulty, ambiguity, and agent "
            "identity. The orchestrator converts posterior predictive success probabilities into actions by "
            "expected utility."
        ),
        diagnostics=[
            "Model the orchestrator state: outcome-labeled routing logs with task difficulty, ambiguity, selected agent, and success.",
            "Update beliefs: fit a non-conjugate Bayesian hierarchical logistic router with NumPyro NUTS.",
            "Make decisions: expand each held-out task to all candidate agents and select routes by posterior_success_probability - agent_cost.",
            "Validate policy impact: evaluate synthetic counterfactual policy value against logged and oracle policies.",
        ],
        metrics=metrics,
        plots=[
            {"label": "Router reliability diagram", "path": reliability_plot.name},
            {"label": "Synthetic policy value comparison", "path": policy_value_plot.name},
            {"label": "Candidate agent selected by policy", "path": policy_mix_plot.name},
        ],
        warnings=warnings,
        recommendation=recommendation,
        metadata=build_metadata(config, seed),
    )
