# Math Behind the Bayesian Orchestrator

This note explains the mathematical model used by the repository and how it connects to the papers in `refs/`. The implementation has two workflows:

- `bayesian_llm_orchestrator`: a synthetic routing example with known counterfactual success probabilities.
- `mmlu_bayesian_orchestrator`: the main model-routing evaluation, using a full question-by-model call matrix and Bayesian reliability models.

The central idea is Bayesian decision theory at the orchestration layer: keep uncertainty over model reliability, update that uncertainty from exploration calls, and choose the model call or calls with the best expected utility. This follows the orchestration-level argument in Papamarkou et al., `refs/2605.00742v2.pdf`, and avoids assuming that the LLM itself performs coherent Bayesian updating.

## 1. Decision Problem

For each question `q` and candidate model `m`, define:

```text
y_{q,m} in {0, 1}       whether model m answers question q correctly
c_{q,m} >= 0            effective call cost
lambda >= 0             configured cost_scale
```

In LaTeX-style notation:

$$
y_{q,m} \in \{0,1\}, \qquad c_{q,m} \ge 0, \qquad \lambda \ge 0.
$$

The utility of making a call is:

```text
u(y_{q,m}, c_{q,m}) = y_{q,m} - lambda c_{q,m}.
```

$$
u(y_{q,m}, c_{q,m}) = y_{q,m} - \lambda c_{q,m}.
$$

Because `y_{q,m}` is unknown before the call, the orchestrator works with the posterior predictive probability:

```text
p_{q,m} = Pr(y_{q,m} = 1 | D),
```

$$
p_{q,m} = \Pr(y_{q,m}=1 \mid D).
$$

where `D` is the exploration data. The single-call expected utility is therefore:

```text
E[u | D, q, m] = p_{q,m} - lambda c_{q,m}.
```

$$
\mathbb{E}[u \mid D,q,m] = p_{q,m} - \lambda c_{q,m}.
$$

The single-shot Bayesian router chooses:

```text
pi(q) = argmax_m { p_{q,m} - lambda c_{q,m} }.
```

$$
\pi(q) = \arg\max_m \left\{p_{q,m} - \lambda c_{q,m}\right\}.
$$

This is the decision-theoretic version of the README rule:

```text
posterior_expected_utility = posterior_success_probability - model_cost * cost_scale.
```

## 2. Exploration Data and Information Boundary

The MMLU workflow first builds a full call matrix:

```text
{(q, m, answer, confidence, latency, tokens, cost, y_{q,m})}
```

for every selected question and configured model. This matrix is an offline evaluation device. The policy is fit only on an exploration split by question id and evaluated on held-out question ids:

```text
D_train = rows for exploration questions
D_test  = rows for held-out questions.
```

The held-out full matrix lets the code compute exact offline utilities for random, always-use-model, Bayesian, adaptive, and oracle policies. During simulated policy execution, however, the policy is only allowed to use model outputs for calls it has selected. This keeps the offline comparison exact while preserving the decision-time information boundary.

Equivalently, the split is:

$$
D_{\mathrm{train}} = \{(q,m,\ldots,y_{q,m}) : q \in Q_{\mathrm{explore}}\},
\qquad
D_{\mathrm{test}} = \{(q,m,\ldots,y_{q,m}) : q \in Q_{\mathrm{test}}\},
$$

with disjoint question sets:

$$
Q_{\mathrm{explore}} \cap Q_{\mathrm{test}} = \varnothing.
$$

## 3. Hierarchical Bayesian Reliability Model

The MMLU reliability model is a Bernoulli-logistic hierarchical model. Each row `i` corresponds to one question-model call. Let:

```text
model_i             model label index
subject_i           MMLU subject index
length_i            question length bucket
cost_i              configured cost weight
confidence_i        parsed model self-confidence, default 0.5 if absent
latency_i           log1p(latency_ms) / 10
tokens_i            log1p(total_tokens) / 10
effective_cost_i    token-price cost if available, else cost weight
y_i                 correctness indicator
```

The pre-call model uses only information known before seeing the answer:

```text
y_i ~ Bernoulli(sigmoid(eta_i))

eta_i =
    alpha
  + a_model[model_i]
  + a_subject[subject_i]
  + a_length[length_i]
  + beta_cost cost_i.
```

$$
y_i \sim \operatorname{Bernoulli}(\sigma(\eta_i)),
$$

$$
\eta_i =
\alpha
+ a^{\mathrm{model}}_{\mathrm{model}_i}
+ a^{\mathrm{subject}}_{\mathrm{subject}_i}
+ a^{\mathrm{length}}_{\mathrm{length}_i}
+ \beta_{\mathrm{cost}}\,\mathrm{cost}_i,
$$

where:

$$
\sigma(x) = \frac{1}{1+\exp(-x)}.
$$

The post-call model adds observed call features:

```text
eta_i =
    alpha
  + a_model[model_i]
  + a_subject[subject_i]
  + a_length[length_i]
  + beta_cost cost_i
  + beta_latency latency_i
  + beta_tokens tokens_i
  + beta_effective_cost effective_cost_i
  + beta_confidence (confidence_i - 0.5).
```

$$
\eta_i =
\alpha
+ a^{\mathrm{model}}_{\mathrm{model}_i}
+ a^{\mathrm{subject}}_{\mathrm{subject}_i}
+ a^{\mathrm{length}}_{\mathrm{length}_i}
+ \beta_{\mathrm{cost}}\,\mathrm{cost}_i
+ \beta_{\mathrm{latency}}\,\mathrm{latency}_i
+ \beta_{\mathrm{tokens}}\,\mathrm{tokens}_i
+ \beta_{\mathrm{effective\_cost}}\,\mathrm{effective\_cost}_i
+ \beta_{\mathrm{confidence}}(\mathrm{confidence}_i - 0.5).
$$

The priors in code are:

```text
alpha ~ Normal(0, 1.5)
sigma_model ~ HalfNormal(1)
sigma_subject ~ HalfNormal(1)
a_model[m] ~ Normal(0, sigma_model)
a_subject[s] ~ Normal(0, sigma_subject)
a_length[l] ~ Normal(0, 0.5)
beta_* ~ Normal(0, 1)
```

$$
\begin{aligned}
\alpha &\sim \mathcal{N}(0,1.5^2), \\
\sigma_{\mathrm{model}} &\sim \operatorname{HalfNormal}(1), \\
\sigma_{\mathrm{subject}} &\sim \operatorname{HalfNormal}(1), \\
a^{\mathrm{model}}_m &\sim \mathcal{N}(0,\sigma_{\mathrm{model}}^2), \\
a^{\mathrm{subject}}_s &\sim \mathcal{N}(0,\sigma_{\mathrm{subject}}^2), \\
a^{\mathrm{length}}_\ell &\sim \mathcal{N}(0,0.5^2), \\
\beta_j &\sim \mathcal{N}(0,1).
\end{aligned}
$$

The posterior is sampled with NUTS in NumPyro. For posterior samples `theta^(1), ..., theta^(S)`, the predictive reliability for row `i` is:

```text
Pr(y_i = 1 | D) ~= (1 / S) sum_s sigmoid(eta_i(theta^(s))).
```

$$
\Pr(y_i=1 \mid D)
\approx
\frac{1}{S}\sum_{s=1}^{S}\sigma\!\left(\eta_i(\theta^{(s)})\right).
$$

The pre-call posterior predictive probabilities drive the first routing decision. The post-call probabilities are used after a chosen model has produced an answer, because confidence, latency, token count, and parsing behavior are then observable.

## 4. Adaptive Calls and Myopic Decision Value

The adaptive policy is intentionally myopic. It is not the full Bayesian experimental design expected information gain objective from BED-LLM, `refs/2508.21184v3.pdf`. Instead, it asks a narrower decision question: is another model call expected to improve final task utility enough to pay for its cost?

For the `myopic_voi` strategy, let `C` be the models already called for question `q`, and let:

```text
b = max_{m in C} p_post(q, m)
```

$$
b = \max_{m \in C} p_{\mathrm{post}}(q,m).
$$

be the current best post-call correctness probability among called models. For an uncalled candidate model `m`, the code estimates the distribution of possible post-call reliability values from exploration rows with the same `(model, subject, length_bucket)` group. If that empirical set is `V_m`, the expected terminal reliability after trying `m` is:

```text
E[max(b, V_m)].
```

$$
\mathbb{E}_{v \sim V_m}\left[\max(b,v)\right].
$$

The myopic gain is:

```text
Delta(q, m) =
    E[max(b, V_m)] - b - lambda E[c_m].
```

$$
\Delta(q,m)
=
\mathbb{E}_{v \sim V_m}\left[\max(b,v)\right]
- b
- \lambda\,\mathbb{E}[c_m].
$$

The policy calls the candidate with the largest `Delta(q, m)` if:

```text
max_m Delta(q, m) > min_expected_gain
```

$$
\max_m \Delta(q,m) > \tau_{\mathrm{gain}},
$$

and `max_calls` has not been reached.

For the `adjudication` strategy, the first call is still chosen by pre-call expected utility. Additional calls are triggered by uncertainty, invalid first answers, and expected correction value.

Let $s_C$ be the current adjudicated answer score after calls $C$, and let $p_{\mathrm{pre}}(q,m)$ be the pre-call correctness probability of candidate model $m$. The implemented expected correction gain is:

$$
\Delta_{\mathrm{adj}}(q,m)
=
(1-s_C)\,p_{\mathrm{pre}}(q,m)
-
\lambda c_m.
$$

The second call is accepted only when:

$$
\max_m \Delta_{\mathrm{adj}}(q,m) > \tau_{\mathrm{gain}},
$$

unless the first answer is invalid and `invalid_answer_triggers_call` is enabled. In the calibrated MMLU-Pro configuration, $\lambda=200$ (`cost_scale: 200`) and $\tau_{\mathrm{gain}}=0.1$ (`min_expected_gain: 0.1`). Because $c_m$ is measured in US dollars, multiplying by 200 converts a typical $0.0002 call into a utility penalty of $0.04$:

$$
200 \times 0.0002 = 0.04.
$$

The candidate must therefore exceed both its monetary penalty and the additional $0.1$ expected-gain margin. On the 120-question calibration run, these values reduced the held-out policy from $2.0$ to $1.8167$ calls per question, stopped after one call on $18.33\%$ of questions, and retained $80.0\%$ accuracy versus $81.67\%$ for the always-two-call setting. The average simulated policy cost fell from approximately $0.00031$ to $0.00024$ per question. These values are an empirical operating point, not universal constants, and should be validated on a fresh question sample before being treated as final.

Called answers are scored as:

```text
score(a) = 1 - product_{m in C: answer_m = a} (1 - p_post(q, m)).
```

$$
\operatorname{score}(a)
=
1 -
\prod_{m \in C:\,\operatorname{answer}_m=a}
\left(1 - p_{\mathrm{post}}(q,m)\right).
$$

The selected final answer is the answer with the largest score. This multiplicative score is a useful aggregation heuristic, but it treats supporting model signals as if they add independent evidence. The code therefore reports a dependence warning when multiple called models share the same configured `dependence_group`.

## 5. Offline Policy Values

For a deterministic policy `pi`, the held-out empirical utility is:

```text
V_hat(pi) = (1 / |Q_test|) sum_{q in Q_test}
    [ y_{q,pi(q)} - lambda c_{q,pi(q)} ].
```

$$
\widehat{V}(\pi)
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q \in Q_{\mathrm{test}}}
\left[
y_{q,\pi(q)} - \lambda c_{q,\pi(q)}
\right].
$$

The evaluated baselines are:

```text
random:
  (1 / |Q_test|) sum_q (1 / M) sum_m [y_{q,m} - lambda c_{q,m}]

always_m:
  (1 / |Q_test|) sum_q [y_{q,m} - lambda c_{q,m}]

single_shot_bayesian:
  choose argmax_m {p_pre(q,m) - lambda c(q,m)}

adaptive_bayesian:
  choose first call by pre-call expected utility, then optionally call more models

oracle:
  (1 / |Q_test|) sum_q max_m [y_{q,m} - lambda c_{q,m}]
```

In LaTeX-style notation:

$$
\widehat{V}_{\mathrm{random}}
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q \in Q_{\mathrm{test}}}
\frac{1}{M}
\sum_{m=1}^{M}
\left[y_{q,m}-\lambda c_{q,m}\right],
$$

$$
\widehat{V}_{\mathrm{always}\,m}
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q \in Q_{\mathrm{test}}}
\left[y_{q,m}-\lambda c_{q,m}\right],
$$

$$
\pi_{\mathrm{single}}(q)
=
\arg\max_m
\left\{
p_{\mathrm{pre}}(q,m)-\lambda c_{q,m}
\right\},
$$

$$
\widehat{V}_{\mathrm{oracle}}
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q \in Q_{\mathrm{test}}}
\max_m
\left[y_{q,m}-\lambda c_{q,m}\right].
$$

The oracle is an upper bound for the sampled matrix, not a deployable policy, because it uses the held-out correctness of every model before deciding.

The code also computes a bootstrap confidence interval for:

```text
V_hat(adaptive_bayesian) - V_hat(random).
```

$$
\widehat{V}(\pi_{\mathrm{adaptive}})
-
\widehat{V}(\pi_{\mathrm{random}}).
$$

This is a finite-sample uncertainty diagnostic over held-out questions, not a proof of production lift.

## 6. Diagnostics

The posterior probabilities are evaluated with proper scoring and calibration diagnostics:

```text
Brier = (1 / n) sum_i (p_i - y_i)^2

Log score = -(1 / n) sum_i [
    y_i log p_i + (1 - y_i) log(1 - p_i)
]

ECE = sum_b (n_b / n) | mean_{i in b}(y_i) - mean_{i in b}(p_i) |
```

$$
\operatorname{Brier}
=
\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2,
$$

$$
\operatorname{LogScore}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log p_i + (1-y_i)\log(1-p_i)
\right],
$$

$$
\operatorname{ECE}
=
\sum_{b}
\frac{n_b}{n}
\left|
\frac{1}{n_b}\sum_{i \in b}y_i
-
\frac{1}{n_b}\sum_{i \in b}p_i
\right|.
$$

The workflow also reports AUROC and approximate leave-one-out expected log predictive density (LOO ELPD). These diagnostics test whether the reliability model is useful and calibrated; they do not replace the utility objective.

The self-confidence ablation fits a post-call model without `confidence_i` and compares Brier score, log score, and AUROC against the model with confidence. This is motivated by the belief-elicitation concerns in `refs/2602.06286v2.pdf`: a model's stated confidence should be treated as a feature to validate, not as a guaranteed true belief.

## 7. Synthetic Router

The synthetic workflow uses the same decision principle with simulated task features. For task `t` and agent `a`:

```text
success_t ~ Bernoulli(sigmoid(
    intercept[a]
  + difficulty_slope[a] difficulty_t
  + ambiguity_slope[a] ambiguity_t
))
```

$$
\operatorname{success}_t
\sim
\operatorname{Bernoulli}
\left(
\sigma\left(
\alpha_a
+ \beta^{\mathrm{difficulty}}_a\,\mathrm{difficulty}_t
+ \beta^{\mathrm{ambiguity}}_a\,\mathrm{ambiguity}_t
\right)
\right).
$$

The fitted Bayesian router uses hierarchical agent intercepts and agent-specific slopes:

```text
mu_agent ~ Normal(0, 1.5)
tau_agent ~ HalfNormal(1)
agent_intercept[a] ~ Normal(mu_agent, tau_agent)
difficulty_slope[a] ~ Normal(-2, 1)
ambiguity_slope[a] ~ Normal(-0.75, 0.75)
```

$$
\begin{aligned}
\mu_{\mathrm{agent}} &\sim \mathcal{N}(0,1.5^2), \\
\tau_{\mathrm{agent}} &\sim \operatorname{HalfNormal}(1), \\
\alpha_a &\sim \mathcal{N}(\mu_{\mathrm{agent}},\tau_{\mathrm{agent}}^2), \\
\beta^{\mathrm{difficulty}}_a &\sim \mathcal{N}(-2,1), \\
\beta^{\mathrm{ambiguity}}_a &\sim \mathcal{N}(-0.75,0.75^2).
\end{aligned}
$$

For each held-out task, the workflow expands the task across all candidate agents and chooses:

```text
argmax_a { Pr(success | task, agent=a, D) - cost_a }.
```

$$
\arg\max_a
\left\{
\Pr(\operatorname{success}=1 \mid \operatorname{task}, a, D)
- c_a
\right\}.
$$

Because the data are simulated, the code can compute true counterfactual expected utilities for all agents. The README correctly flags this as synthetic-only; real logs need randomized exploration, propensities, A/B tests, inverse-propensity scoring, doubly robust evaluation, or another off-policy evaluation method.

## 8. Relationship to the Reference Papers

- Papamarkou et al., "Position: agentic AI orchestration should be Bayes-consistent", `refs/2605.00742v2.pdf`: motivates the core design choice. The Bayesian machinery lives in the control layer that routes models and tools, not inside the LLM weights.
- Huang et al., "BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling", `refs/2406.05516v4.pdf`: supports the general pattern of combining agentic LLM workflows with explicit probabilistic graphical models and numerical Bayesian inference.
- Choudhury et al., "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design", `refs/2508.21184v3.pdf`: motivates adaptive information gathering. The repository uses a cheaper myopic decision-value approximation rather than full expected information gain.
- Ye et al., "Uncertainty Quantification for LLM Function-Calling", `refs/2604.22985v1.pdf`: motivates treating output validity, confidence, and call metadata as uncertainty signals around tool/model calls. The workflow records confidence, parsing validity, latency, and token counts as post-call reliability features.
- Yamin et al., "When Agents Say One Thing and Do Another: Validating Elicited Beliefs from LLMs", `refs/2602.06286v2.pdf`: motivates validating self-reported confidence against observed decisions and outcomes. The workflow tests whether confidence improves correctness prediction instead of trusting it directly.
- Qiu et al., "Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models", `refs/2503.17523v3.pdf`: provides background on Bayesian belief updating as a normative target for LLM agents. This project takes a complementary route by fitting an external Bayesian controller over observed model behavior.
- Chen et al., "LLMs are not (consistently) Bayesian: Quantifying internal (in)consistencies of LLMs' probabilistic beliefs", `refs/2605.06915v2.pdf`: reinforces the reason to separate orchestration-level Bayesian inference from claims about the LLM's internal belief updates.

## 9. Practical Interpretation

The implemented policy should be read as:

```text
Use exploration calls to learn calibrated model reliabilities.
Convert those reliabilities into cost-sensitive expected utility.
Call more models only when their expected marginal decision value beats cost.
Evaluate against exact held-out baselines before claiming routing lift.
```

This is a pragmatic Bayesian controller for model routing. It is not a claim that the underlying LLMs are Bayesian, and it is not full Bayesian experimental design. Its value comes from using explicit uncertainty and cost-sensitive decisions at the orchestration layer, where the assumptions are inspectable and the policy can be tested against held-out outcomes.
