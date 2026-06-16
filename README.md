# Bayesian Orchestrator

Cost-aware Bayesian model routing evaluated as a reproducible Nebius Serverless AI Job.

This project asks a practical orchestration question: **when should an AI system use a cheap model, a stronger model, or pay for a second opinion?**
It builds a model-question call matrix on MMLU-Pro, fits hierarchical Bayesian reliability models, and compares adaptive routing with random, fixed-model, single-shot Bayesian, and oracle policies.

Built for the **Nebius Serverless AI Builders Challenge** in the AI & ML category. Challenge tag: `#NebiusServerlessChallenge`.

## Methodology

The router estimates each model's posterior probability of answering correctly, conditioned on model identity, subject, question length, and observed call features. It then selects the action with the highest posterior expected utility: $E[u\mid D,q,m]=P(y_{q,m}=1\mid D)-\lambda c_{q,m}.$

The adaptive policy can request a second model when its expected correction gain exceeds its token cost and a configurable minimum margin. When models disagree, their answers are adjudicated using posterior reliability. This is a cost-sensitive, myopic value-of-information policy, not full Bayesian experimental design.
Relevant references, full model specification, policy equations, diagnostics, assumptions, and claim boundaries are documented in [`MATH_APPROACH.pdf`](MATH_APPROACH.pdf).

## Quick Start

Requirements:

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Docker, only for the container workflow
- A Nebius Token Factory API key, only for live runs

Install and test:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run python -m unittest discover -s tests -v
```

Run the deterministic, network-free smoke test:

```bash
UV_CACHE_DIR=.uv-cache uv run bayesian-orchestrator run \
  --config examples/mmlu-bayesian-orchestrator/config.yaml
```

The smoke test uses synthetic questions and deterministic fake responses. It validates the workflow but (not evidence of routing usefulness).

## Why Serverless AI Job

The workload is a finite, restartable batch evaluation rather than an interactive service, so it runs as a Nebius Serverless AI Job. The container:

1. Loads a deterministic MMLU-Pro sample.
2. Calls each configured Token Factory model for every question.
3. Appends every completed call to a durable checkpoint.
4. Fits pre-call and post-call Bayesian reliability models.
5. Evaluates routing policies and writes reports, metrics, and plots.

Nebius Serverless AI provides disposable compute for the evaluation job, while Nebius Token Factory provides the OpenAI-compatible model API. A shared filesystem preserves checkpoints and results when a job exits or restarts.

## MMLU-PRO Benchmark

The canonical challenge configuration is [`examples/mmlu-bayesian-orchestrator/mmlu-pro-final.yaml`](examples/mmlu-bayesian-orchestrator/mmlu-pro-final.yaml). 
Architecture is

 ```mermaid
flowchart LR
    A["MMLU-Pro sample"] --> B["Nebius Serverless AI Job"]
    B --> C["LangGraph call runner"]
    C --> D["Nebius Token Factory models"]
    C --> E["Incremental call-matrix checkpoint"]
    E --> F["Hierarchical Bayesian reliability models"]
    F --> G["Random, fixed, Bayesian, adaptive, and oracle policies"]
    G --> H["Markdown report, JSON summary, and plots"]
```

It uses:

- MMLU-Pro test split ([TIGER-Lab MMLU-Pro Space](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro))
- 4,000 stratified questions
- Three Token Factory models
- A 30% exploration / 70% held-out split, stratified by category
- 500 HMC NUTS warmup steps and 1,000 posterior samples
- At most two calls per deployed adaptive decision

The config validates selected model IDs and dated pricing entries before making missing calls. Every completed question-model pair is checkpointed. For a less expensive trial, copy the final config and reduce `dataset.sample_count`, `mcmc.warmup`, `mcmc.samples`, and `evaluation.bootstrap_iterations`.

### Final benchmark results

The completed run is available at [`mmlu-pro-4000-stratified-seed48219/`](mmlu-pro-4000-stratified-seed48219/). It contains 12,000 calls: three models for each of 4,000 stratified MMLU-Pro questions. The Bayesian models were fit on 1,200 exploration questions and policies were evaluated on 2,800 held-out questions.

| Policy | Held-out accuracy | Observed cost per 1,000 questions | Utility |
|---|---:|---:|---:|
| Random model | 60.5% | $0.1651 | 0.5725 |
| Always small | 51.9% | $0.1727 | 0.4847 |
| Always middle | 56.4% | $0.0313 | 0.5577 |
| Always strong | 73.3% | $0.2912 | 0.6750 |
| Bayesian single shot | 73.9% | $0.2788 | 0.6832 |
| Adaptive Bayesian | **76.3%** | $0.2950 | **0.7042** |
| Retrospective oracle | 81.7% | $0.0871 | 0.7994 |

Utility is `accuracy - 200 x observed USD cost per question`. It is therefore specific to the configured cost scale, not a universal score. The oracle uses knowledge of all observed answers after the fact and is only an upper bound; its apparently low cost is not achievable by a deployed router.

For this one seed, MMLU-Pro sample, one prompt/configuration, three model versions, and one pricing snapshot dated June 14, 2026, findings support the claim that Bayesian adaptive policy improves accuracy. 

 The complete call matrix supports exact offline comparison for these 2,800 questions, but it does not establish production lift, a stronger claim requires repeated seeds or splits, refit-aware uncertainty, and randomized or propensity-logged production data (work to be continued...).

![Final MMLU accuracy and cost comparison](mmlu-pro-4000-stratified-seed48219/mmlu_accuracy_cost.png)

Model inference happens in Token Factory, so a GPU is not required. I used a regular CPU Serverless AI preset with 8 vCPUs and 32 GiB RAM (main consumption is for JAX/NumPyro fitting). The generated `summary.json` records actual token counts and model-call costs. This excludes Serverless AI compute and storage costs.

## Nebius Serverless Job

Build and test the image locally:

```bash
docker build -t bayesian-orchestrator:latest .
docker run --rm \
  -v "$PWD/outputs:/app/outputs" \
  bayesian-orchestrator:latest \
  run --config examples/mmlu-bayesian-orchestrator/config.yaml
```

Publish it to a registry Nebius can access:

```bash
docker tag bayesian-orchestrator:latest <registry>/<repo>/bayesian-orchestrator:latest
docker push <registry>/<repo>/bayesian-orchestrator:latest
```

For a private registry, use Nebius registry credentials or `--registry-secret`. 

First run the network-free smoke test in Serverless AI:

```bash
nebius ai job create \
  --name bayesian-routing-smoke \
  --image <registry>/<repo>/bayesian-orchestrator:latest \
  --args "run --config examples/mmlu-bayesian-orchestrator/config.yaml" \
  --timeout 1h \
  --platform <cpu-platform-id> \
  --preset <cpu-preset> \
  --subnet-id <subnet-id> \
  --volume '<filesystem-id>:/app/outputs:rw'
```

Run the final benchmark using a MysteryBox secret and shared filesystem:

```bash
nebius ai job create \
  --name bayesian-routing-mmlu-pro \
  --image <registry>/<repo>/bayesian-orchestrator:latest \
  --args "run --config examples/mmlu-bayesian-orchestrator/mmlu-pro-final.yaml" \
  --env-secret NEBIUS_API_KEY=<mysterybox-secret-selector> \
  --timeout 16 \
  --platform <cpu-platform-id> \
  --preset <8vcpu-32gb-or-larger-preset> \
  --subnet-id <subnet-id> \
  --volume '<filesystem-id>:/app/outputs:rw'
```

Mount a shared filesystem at `/app/outputs`. The checkpoint uses append and fsync semantics, so do not use an S3-backed mount for `call_matrix.jsonl`.

Long Token Factory (or your provider's) runs should configure application-level retries for API failures:

```yaml
provider:
  request_max_attempts: 8
  retry_initial_delay_seconds: 2
  retry_max_delay_seconds: 60
```

The workflow retries connection failures, timeouts, and HTTP `429`, `500`, `502`, `503`, and `504` responses with capped exponential backoff.
A result is checkpointed only after a successful response, so exhausted retries can be resumed from the same persistent `call_matrix.jsonl`.

To resume, create a replacement job with the same config and the same shared filesystem mounted at `/app/outputs`. With `provider.reuse_cache: true`, the workflow validates existing rows and calls only missing question-model pairs (saving on API costs).


## Result Validation

For [`examples/mmlu-bayesian-orchestrator/mmlu-pro-final.yaml`](examples/mmlu-bayesian-orchestrator/mmlu-pro-final.yaml), the completed output is available in:

```text
mmlu-pro-4000-stratified-seed48219/
```

Expected files:

- `call_matrix.jsonl`: append-only model-call checkpoint
- `report.md`: human-readable results and reproducibility metadata
- `summary.json`: machine-readable metrics, token usage, and costs
- `mmlu_accuracy_cost.png`: held-out accuracy and observed token-cost comparison


The final matrix contains 12,000 unique rows.

Policies evaluated on held-out questions are:

- `random`: exact expected utility from uniform model selection
- `always_<model>`: one fixed-model baseline per configured model
- `single_shot_bayesian`: one call selected by pre-call posterior utility
- `adaptive_bayesian`: optional second call and reliability-based adjudication
- `oracle`: best observed answer, used only as an upper bound

Reports also include Brier score, log score, AUROC, LOO ELPD, bootstrap intervals, token cost, latency, throughput, answer diversity, and dependence-group warnings.

Limitations are that:

- A full call matrix enables exact offline comparison on this sample, but production routing needs randomized exploration, logged propensities, or off-policy estimators.
- Self-reported model confidence is an input feature to audit, not a calibrated posterior.
- Similar models may have correlated errors, current approach doesn't have machinery to model this dependence.
- The adaptive policy is myopic and limited to the configured maximum number of calls.


## Repository Layout

```text
bayesian_orchestrator/                 Python package and workflows
examples/bayesian-orchestrator/       Synthetic Bayesian routing example
examples/mmlu-bayesian-orchestrator/  Smoke, calibration, and final configs
pricing/                              Versioned Token Factory price snapshots
tests/                                Unit tests for pricing and sampling
.github/workflows/                    GitHub Actions CI
Dockerfile                            Reproducible Serverless AI image
MATH_APPROACH.pdf                     Statistical design
```

## Other OpenAI-Compatible Providers

The implementation is not hard-coded to `NEBIUS_API_KEY`. `provider.api_key_env` names the environment variable to read, while `provider.base_url` can point to another OpenAI-compatible API:

```yaml
provider:
  type: openai_compatible
  base_url: https://provider.example/v1/
  api_key_env: OPENAI_COMPATIBLE_API_KEY
  temperature: 0.0
  max_tokens: 1024
  json_mode: true
  validate_models: false
  reuse_cache: true
  cache_file: call_matrix.jsonl
```

Set `validate_models: false` unless the provider supports the `/v1/models?verbose=true` response shape used by Token Factory. Replace model IDs and provide explicit prices or a compatible pricing catalog before making cost comparisons. The canonical challenge benchmark remains Nebius-specific.



## License

Released under the [MIT License](LICENSE).

Official references: [Serverless AI overview](https://docs.nebius.com/serverless/overview), [managing Serverless AI jobs](https://docs.nebius.com/serverless/jobs/manage), and [Token Factory API](https://docs.tokenfactory.nebius.com/api-reference/introduction).

Math references are in:  [`MATH_APPROACH.pdf`](MATH_APPROACH.pdf)
