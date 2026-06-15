# Mathematical Approach

This document describes the statistical model and decision rules implemented by the Bayesian Orchestrator. The main workflow evaluates model routing on a complete question-by-model call matrix. A second synthetic workflow demonstrates the same decision principle when counterfactual success probabilities are known.

The central design choice is to place Bayesian inference in the orchestration layer: estimate uncertainty about model reliability from observed outcomes, then choose calls by expected utility. This follows the control-layer argument of Papamarkou et al. [1] and avoids assuming that an LLM performs coherent internal Bayesian updating.

## 1. Decision Problem

For question $q$ and candidate model $m$, let $y_{q,m}$ indicate correctness, $c_{q,m}$ denote call cost, and $\lambda$ denote the configured cost scale:

$$
y_{q,m}\in\{0,1\},
\qquad
c_{q,m}\ge 0,
\qquad
\lambda\ge 0.
$$

The realized utility of a call is:

$$
u(y_{q,m},c_{q,m})=y_{q,m}-\lambda c_{q,m}.
$$

Before the call, correctness is unknown. Given exploration data $D$, the router uses the posterior predictive probability:

$$
p_{q,m}=\Pr(y_{q,m}=1\mid D).
$$

The expected utility of a single call is therefore:

$$
\mathbb{E}[u\mid D,q,m]=p_{q,m}-\lambda c_{q,m}.
$$

The single-shot Bayesian policy chooses:

$$
\pi_{\mathrm{single}}(q)
=
\arg\max_m\left\{p_{q,m}-\lambda c_{q,m}\right\}.
$$

## 2. Exploration Data And Information Boundary

For every selected question and model, the workflow records the question, answer, confidence, latency, token counts, cost, parsing status, and correctness. The resulting full call matrix is an offline evaluation device.

Questions, rather than individual calls, are divided into disjoint exploration and evaluation sets:

$$
D_{\mathrm{train}}
=
\{(q,m,\ldots,y_{q,m}):q\in Q_{\mathrm{explore}}\},
$$

$$
D_{\mathrm{test}}
=
\{(q,m,\ldots,y_{q,m}):q\in Q_{\mathrm{test}}\},
$$

$$
Q_{\mathrm{explore}}\cap Q_{\mathrm{test}}=\varnothing.
$$

The full held-out matrix permits exact offline comparison of routing policies on the sampled questions. During simulated policy execution, however, a policy may inspect only the outputs of calls it selected. Uncalled held-out responses remain hidden until scoring.

The final MMLU-Pro benchmark uses proportional stratified sampling. If category $k$ contains $N_k$ eligible questions and the requested sample size is $n$, its target allocation is:

$$
n_k\approx n\frac{N_k}{\sum_j N_j}.
$$

Deterministic largest-remainder allocation converts these targets to integer counts that sum to $n$. The exploration/evaluation split is stratified again inside the selected sample. With $n=4000$ and an exploration fraction of $0.3$, the intended split is 1,200 exploration questions and 2,800 evaluation questions.

## 3. Hierarchical Bayesian Reliability Model

Each row $i$ represents one question-model call. Its pre-call features are model identity, subject, question-length bucket, and configured cost weight. Its post-call features additionally include parsed confidence, latency, token count, and observed token-price cost.

Correctness follows a Bernoulli-logistic model:

$$
y_i\sim\operatorname{Bernoulli}(\sigma(\eta_i)),
\qquad
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

The pre-call linear predictor contains only information available before a response is observed:

$$
\eta_i^{\mathrm{pre}}
=
\alpha
+a^{\mathrm{model}}_{m_i}
+a^{\mathrm{subject}}_{s_i}
+a^{\mathrm{length}}_{\ell_i}
+\beta_{\mathrm{cost}}\,w_i.
$$

The post-call model adds observed response features:

$$
\begin{aligned}
\eta_i^{\mathrm{post}}={}&
\eta_i^{\mathrm{pre}}
+\beta_{\mathrm{latency}}\,\widetilde{t}_i
+\beta_{\mathrm{tokens}}\,\widetilde{n}_i \\
&+\beta_{\mathrm{observed\ cost}}\,\widetilde{c}_i
+\beta_{\mathrm{confidence}}(r_i-0.5).
\end{aligned}
$$

Here, $r_i$ is parsed self-confidence, $\widetilde{t}_i$ is transformed latency, $\widetilde{n}_i$ is transformed token count, and $\widetilde{c}_i$ is observed token-price cost when available.

The implemented priors are:

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

The posterior is sampled with NUTS in NumPyro. For posterior draws $\theta^{(1)},\ldots,\theta^{(S)}$, predictive reliability is approximated by:

$$
\Pr(y_i=1\mid D)
\approx
\frac{1}{S}
\sum_{s=1}^{S}
\sigma\!\left(\eta_i(\theta^{(s)})\right).
$$

Pre-call probabilities determine the first model. Post-call probabilities support stopping and answer adjudication after a response has been observed.

## 4. Adaptive Calls

The adaptive policy uses a myopic decision-value approximation inspired by adaptive information gathering [3]. It does not optimize the full expected information gain objective of sequential Bayesian experimental design.

Let $C$ be the set of models already called and let the current best post-call reliability be:

$$
b=\max_{m\in C}p_{\mathrm{post}}(q,m).
$$

For an uncalled model $m$, let $V_m$ be the empirical distribution of post-call reliability values observed in matching exploration groups. The estimated myopic gain is:

$$
\Delta(q,m)
=
\mathbb{E}_{v\sim V_m}[\max(b,v)]
-b
-\lambda\mathbb{E}[c_m].
$$

The policy buys another call only when:

$$
\max_m\Delta(q,m)>\tau_{\mathrm{gain}},
$$

subject to the configured maximum number of calls.

The challenge configuration uses the adjudication strategy. Let $s_C$ denote the current adjudicated answer score and $p_{\mathrm{pre}}(q,m)$ the pre-call correctness probability of candidate $m$. Its expected correction gain is:

$$
\Delta_{\mathrm{adj}}(q,m)
=
(1-s_C)p_{\mathrm{pre}}(q,m)
-\lambda c_m.
$$

A second call is accepted when:

$$
\max_m\Delta_{\mathrm{adj}}(q,m)>\tau_{\mathrm{gain}},
$$

unless an invalid first answer activates the explicit fallback rule.

The calibrated configuration uses $\lambda=200$ and $\tau_{\mathrm{gain}}=0.1$. A call costing $0.0002$ USD therefore receives a utility penalty of:

$$
200\times 0.0002=0.04.
$$

These values are an empirical operating point, not universal constants. They should remain frozen before evaluating a fresh confirmation sample.

## 5. Answer Adjudication

For answer option $a$, supporting calls are combined using:

$$
\operatorname{score}(a)
=
1-
\prod_{m\in C:\,\operatorname{answer}_m=a}
\left(1-p_{\mathrm{post}}(q,m)\right).
$$

The option with the largest score is selected. This aggregation is practical but treats supporting reliability signals as if they contribute independent evidence. Configured dependence groups therefore produce warnings when called models may share correlated errors.

## 6. Offline Policy Values

For deterministic policy $\pi$, held-out empirical utility is:

$$
\widehat{V}(\pi)
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q\in Q_{\mathrm{test}}}
\left[
y_{q,\pi(q)}-\lambda c_{q,\pi(q)}
\right].
$$

Uniform random routing has value:

$$
\widehat{V}_{\mathrm{random}}
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q\in Q_{\mathrm{test}}}
\frac{1}{M}
\sum_{m=1}^{M}
\left[y_{q,m}-\lambda c_{q,m}\right].
$$

The fixed-model baseline for model $m$ is:

$$
\widehat{V}_{\mathrm{always}\,m}
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q\in Q_{\mathrm{test}}}
\left[y_{q,m}-\lambda c_{q,m}\right].
$$

The adaptive policy pays for every call it actually makes:

$$
u_{\mathrm{adaptive}}(q)
=
\mathbf{1}\{\widehat{a}_q=a_q^*\}
-\lambda\sum_{m\in C_q}c_{q,m},
$$

where $C_q$ is the selected call set, $\widehat{a}_q$ is the adjudicated answer, and $a_q^*$ is the gold answer.

The oracle retrospectively selects the best observed model:

$$
\widehat{V}_{\mathrm{oracle}}
=
\frac{1}{|Q_{\mathrm{test}}|}
\sum_{q\in Q_{\mathrm{test}}}
\max_m\left[y_{q,m}-\lambda c_{q,m}\right].
$$

The oracle is an upper bound for the sampled matrix, not a deployable policy, because it uses every model's held-out correctness before selecting an action.

Question-level bootstrap resampling estimates uncertainty in policy differences, including:

$$
\widehat{V}(\pi_{\mathrm{adaptive}})
-
\widehat{V}(\pi_{\mathrm{random}}).
$$

This interval measures finite-sample uncertainty on the selected held-out questions; it is not proof of production lift.

## 7. Predictive Diagnostics

Posterior probabilities are assessed with proper scoring and calibration metrics. The Brier score is:

$$
\operatorname{Brier}
=
\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2.
$$

The mean negative log score is:

$$
\operatorname{LogScore}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

Expected calibration error over bins $b$ is:

$$
\operatorname{ECE}
=
\sum_b\frac{n_b}{n}
\left|
\frac{1}{n_b}\sum_{i\in b}y_i
-
\frac{1}{n_b}\sum_{i\in b}p_i
\right|.
$$

The workflow also reports AUROC and approximate leave-one-out expected log predictive density. These diagnose the reliability model; policy selection remains governed by expected utility.

Self-reported confidence is treated as a feature to validate rather than a trustworthy posterior. The confidence ablation compares post-call models with and without the confidence term, consistent with evidence that elicited LLM beliefs can be behaviorally incoherent [5] and that LLM belief updates are not consistently Bayesian [7]. Function-calling uncertainty work further motivates auditing output validity and call-specific uncertainty signals [4].

## 8. Operational Reproducibility

Each successful question-model pair is appended, flushed, and synced before the next call. The completed set is:

$$
P_{\mathrm{done}}
=
\{(q,m):(q,m)\text{ has a valid checkpoint row}\}.
$$

A resumed job calls only pairs outside $P_{\mathrm{done}}$. Failed requests are not recorded as completed results.

Transient provider failures use capped exponential backoff. With initial delay $d_0$, maximum delay $d_{\max}$, and retry index $r$:

$$
d_r=\min\left(d_0 2^{r-1},d_{\max}\right).
$$

The final configuration permits up to eight attempts with $d_0=2$ seconds and $d_{\max}=60$ seconds. Reproducibility additionally requires an immutable image, fixed configuration and seed, versioned pricing, and one writer per checkpoint.

## 9. Synthetic Router

The synthetic workflow uses task difficulty and ambiguity to generate outcome-labeled routing logs. For task $t$ assigned to agent $a$:

$$
\operatorname{success}_t
\sim
\operatorname{Bernoulli}
\left(
\sigma\left(
\alpha_a
+\beta_a^{\mathrm{difficulty}}\,\mathrm{difficulty}_t
+\beta_a^{\mathrm{ambiguity}}\,\mathrm{ambiguity}_t
\right)
\right).
$$

The fitted priors are:

$$
\begin{aligned}
\mu_{\mathrm{agent}} &\sim \mathcal{N}(0,1.5^2), \\
\tau_{\mathrm{agent}} &\sim \operatorname{HalfNormal}(1), \\
\alpha_a &\sim \mathcal{N}(\mu_{\mathrm{agent}},\tau_{\mathrm{agent}}^2), \\
\beta_a^{\mathrm{difficulty}} &\sim \mathcal{N}(-2,1), \\
\beta_a^{\mathrm{ambiguity}} &\sim \mathcal{N}(-0.75,0.75^2).
\end{aligned}
$$

For each held-out task, the selected agent is:

$$
\arg\max_a
\left\{
\Pr(\operatorname{success}=1\mid \operatorname{task},a,D)-c_a
\right\}.
$$

Because this workflow is simulated, true counterfactual expected utilities are available for every agent. Real routing logs do not provide those counterfactuals and require randomized exploration, logged propensities, controlled experiments, or valid off-policy estimators.

## 10. Interpretation And Limitations

The implementation is a pragmatic Bayesian controller for model routing. It combines explicit uncertainty estimates with cost-sensitive decisions at the orchestration layer. It does not claim that the underlying LLMs are Bayesian, that elicited confidence is intrinsically calibrated, or that the adaptive policy implements full Bayesian experimental design.

BayesAgent provides a related example of combining agentic LLM workflows with probabilistic graphical models and numerical Bayesian inference [2]. Bayesian Teaching studies Bayesian updating as a normative and trainable reasoning target inside language models [6]. This project instead fits an external controller over observed model behavior.

Conclusions remain specific to the selected questions, model versions, prompts, prices, and run date. Production claims require repeated evaluation and data collected under a policy that supports unbiased comparison.

## References

1. Theodore Papamarkou et al. "Position: Agentic AI Orchestration Should Be Bayes-Consistent." *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, 2026. [doi:10.48550/arXiv.2605.00742](https://doi.org/10.48550/arXiv.2605.00742).

2. Hengguan Huang, Xing Shen, Songtao Wang, Lingfa Meng, Dianbo Liu, David Alejandro Duchene, Hao Wang, and Samir Bhatt. "BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling." *AAAI Conference on Artificial Intelligence*, 2026. [doi:10.48550/arXiv.2406.05516](https://doi.org/10.48550/arXiv.2406.05516).

3. Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, and Tom Rainforth. "BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design." *International Conference on Learning Representations (ICLR)*, 2026. [doi:10.48550/arXiv.2508.21184](https://doi.org/10.48550/arXiv.2508.21184).

4. Zihuiwen Ye, Lukas Aichberger, Michael Kirchhof, Sinead Williamson, Luca Zappella, Yarin Gal, Arno Blaas, and Adam Goliński. "Uncertainty Quantification for LLM Function-Calling." arXiv preprint, 2026. [doi:10.48550/arXiv.2604.22985](https://doi.org/10.48550/arXiv.2604.22985).

5. Khurram Yamin, Jingjing Tang, Santiago Cortes-Gomez, Amit Sharma, Eric Horvitz, and Bryan Wilder. "When Agents Say One Thing and Do Another: Validating Elicited Beliefs from LLMs." arXiv preprint, 2026. [doi:10.48550/arXiv.2602.06286](https://doi.org/10.48550/arXiv.2602.06286).

6. Linlu Qiu, Fei Sha, Kelsey Allen, Yoon Kim, Tal Linzen, and Sjoerd van Steenkiste. "Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models." arXiv preprint, 2025. [doi:10.48550/arXiv.2503.17523](https://doi.org/10.48550/arXiv.2503.17523).

7. Chacha Chen, Matthew Jörke, Adam Goliński, Masha Fedzechkina, Guillermo Sapiro, Sinead Williamson, and Nicholas Foti. "LLMs Are Not (Consistently) Bayesian: Quantifying Internal (In)consistencies of LLMs' Probabilistic Beliefs." arXiv preprint, 2026. [doi:10.48550/arXiv.2605.06915](https://doi.org/10.48550/arXiv.2605.06915).
