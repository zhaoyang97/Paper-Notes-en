---
title: >-
  [Paper Note] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge
description: >-
  [ICML 2026][LLM Evaluation][LLM-as-a-Judge] RACER models the decision of "whether to invoke reasoning mode for judgment" as a distributionally robust constrained optimization problem with a KL uncertainty set. It utilize…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "reasoning model routing"
  - "KL uncertainty set"
  - "primal-dual"
  - "OOD robustness"
date: 2026-05-08
content_hash: d2da3910990b22e9
---

# Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge

**Conference**: ICML 2026  
**arXiv**: [2605.10805](https://arxiv.org/abs/2605.10805)  
**Code**: None  
**Area**: LLM Evaluation / Model Routing / Distributionally Robust Optimization  
**Keywords**: LLM-as-a-Judge, reasoning model routing, KL uncertainty set, primal-dual, OOD robustness

## TL;DR
RACER models the decision of "whether to invoke reasoning mode for judgment" as a distributionally robust constrained optimization problem with a KL uncertainty set. It utilizes a primal-dual algorithm to derive an optimal routing strategy that remains cost-effective under OOD conditions and provides the first theoretical guarantee of linear convergence for LLM router policies.

## Background & Motivation

**Background**: LLM-as-a-Judge increasingly employs reasoning models (e.g., o1, DeepSeek-R1, Qwen3 thinking). Although these models learn reasoning via RL on verifiable tasks, judgment tasks themselves are not explicitly optimized. Thus, whether "reasoning truly improves judgment accuracy" remains an open question. A natural intermediate solution is routing—dynamically selecting between reasoning or instruct modes based on query difficulty.

**Limitations of Prior Work**: Existing LLM routing research (FrugalGPT, P2L, RouteLLM, ThinkSwitcher) shares three common shortcomings. First, they focus almost exclusively on QA tasks, neglecting the judgment scenario. Second, they only optimize the "cost-accuracy trade-off under the training distribution"; once the query distribution shifts during deployment (e.g., changes in user base or domain ratios), cost constraints are violated and performance degrades. Third, most approaches are empirical or heuristic, lacking theoretical convergence guarantees. Empirical evidence also suggests that reasoning judges significantly improve accuracy in math/coding but provide negligible or even negative gains in safety/knowledge, while token costs increase several-fold on average—indiscriminate use of reasoning is both expensive and potentially detrimental.

**Key Challenge**: Reasoning mode is expensive and not universally beneficial (overthinking can be harmful), yet training data is static, leading to distorted reward estimation and cost budgets under OOD deployment.

**Goal**: To learn a routing policy $\pi(a | z)$ ($a \in \{0, 1\}$ denotes activating reasoning) under a fixed cost budget $C$, such that it (i) maximizes expected judge reward; (ii) is robust to query distribution shifts; and (iii) possesses theoretical convergence guarantees.

**Key Insight**: Distributionally Robust Optimization (DRO) with KL uncertainty sets combined with Lagrangian primal-dual. Reward and cost are both measured using worst-case metrics, treating "robustness in reward" and "robustness in cost" separately (the former prevents overestimating benefits under OOD, while the latter prevents budget overruns).

**Core Idea**: Reformulate the LLM-as-a-Judge routing problem as $\max_\pi \min_{\tilde{\rho} \in \mathcal{U}(\rho_n, \delta)} \mathbb{E}_{\tilde{\rho}}[r] \text{ s.t. } \max_{\tilde{\rho} \in \mathcal{U}} \mathbb{E}_{\tilde{\rho}}[c] \leq C$. It is proven that the worst-case distribution under a KL uncertainty set has a closed-form reweighting, allowing for efficient solving via primal-dual.

## Method

### Overall Architecture

Input: Preference dataset $\{(x_i, y_{i,1}, y_{i,2}, l_i)\}$ (containing ground-truth preference labels); a hybrid LLM providing a reasoning judge $\Phi_1$ and a non-reasoning judge $\Phi_0$.

Preprocessing: Run both modes for each instance, recording reward $r_i = \mathbb{I}(\Phi_{a_i}(z_i) = l_i)$ and cost $c_i$ (token count).

Router: A 4-layer NN taking embeddings (via bge-m3) of the concatenated prompt and response as input, outputting the reasoning probability.

Training: A primal-dual algorithm is used on each batch to (a) compute empirical means of reward/cost as baselines, (b) calculate worst-case distributions $\underline{\rho}, \bar{\rho}$ via closed-form reweighting, and (c) update policy $\pi_{t+1}$ and dual variable $\lambda_{t+1}$. The best iterate is selected based on the validation set.

### Key Designs

1.  **Dual Robustness in Distributionally Robust Constrained Optimization (Separate Robustness for Reward and Cost)**:

    - **Function**: Expresses router learning as $\max_\pi R_{\mathcal{U}(\rho_n, \delta)}(\pi) \text{ s.t. } C_{\mathcal{U}(\rho_n, \delta)}(\pi) \leq C$, where $R$ is the worst-case reward and $C$ is the worst-case cost. The uncertainty set is a KL ball centered at the empirical distribution $\rho_n$.
    - **Mechanism**: While traditional DRO robustifies only the objective, this work recognizes that reward and cost distortions under OOD follow different directions. OOD queries might be cheaper (making cost robustness less critical while requiring reward robustness to utilize budgets aggressively) or more expensive (making cost robustness vital to avoid overruns). Handling both cases separately ensures safety under various OOD modes.
    - **Design Motivation**: Figure 3 in the experiments validates this split: RACER-R (reward-only robustness) exceeds the budget in "expensive" OOD scenarios, while RACER-C (cost-only robustness) wastes budget in "cheap" OOD scenarios. Only dual robustness remains stable across both.

2.  **Closed-form Worst-case Reweighting for KL Uncertainty Sets (Theorem 3.1)**:

    - **Function**: Converts the abstract $\min/\max$ over $\mathcal{U}(\rho_n, \delta)$ into a closed-form reweighting of sample weights.
    - **Mechanism**: Defining $f_i = \mathbb{E}_{a \sim \pi(\cdot | z_i)}[f(z_i, a)]$, then $\underline{\rho}(i) \propto \rho_n(i) \exp(\frac{\underline{s} - f_i}{\tau})$ (minimization) and $\bar{\rho}(i) \propto \rho_n(i) \exp(\frac{f_i - \bar{s}}{\tau})$ (maximization). Intuitively, for reward, the worst-case distribution downweights samples with rewards higher than the baseline and upweights lower ones. For cost, it focuses optimization on "high-risk" high-cost areas. $\tau$ controls the intensity of reweighting.
    - **Design Motivation**: Alternating gradient on parameterized distributions is infeasible due to lack of samples; closed-form reweighting equates "worst-case over unknown distributions" to "weighting known samples," making engineering implementation trivial. This draws inspiration from distributionally robust RL (Gadot et al. 2024 / Xu et al. 2025).

3.  **Entropy-regularized Primal-Dual Algorithm + Linear Convergence Proof**:

    - **Function**: Solves the constrained min-max Lagrangian $L_\beta(\pi, \lambda) = R_{\underline{\rho}}(\pi) - \lambda C_{\bar{\rho}}(\pi) + \beta(\mathcal{H}(\pi) + \frac{1}{2}\lambda^2)$ to obtain $(\pi^*, \lambda^*)$.
    - **Mechanism**: Alternately updates $\pi_{t+1} = \arg\max_\pi \{R_{\underline{\rho}}(\pi) - \lambda_t C_{\bar{\rho}}(\pi) + \beta \mathcal{H}(\pi)\}$ and $\lambda_{t+1} = \arg\max_{\lambda \geq 0}\{-\lambda C_{\bar{\rho}}(\pi) + \frac{1}{2}\beta \lambda^2\}$. Theorem 4.1 proves the existence and uniqueness of the saddle point, while Theorem 4.2 provides a linear convergence rate: $\text{KL}(\pi_t \| \pi^*) \leq \frac{M^2 K^2}{2 \beta^2} (\frac{M^2 K^2}{M^2 K^2 + 2 \beta^2})^{2t} (\lambda_0 - \lambda^*)^2$.
    - **Design Motivation**: Entropy regularization $\mathcal{H}(\pi)$ prevents policy collapse into deterministic behavior and ensures last-iterate convergence; $\frac{1}{2}\lambda^2$ dual regularization ensures $\lambda$ is bounded. Together, they allow the model to converge to a unique saddle point using the last iterate, which is more practical than traditional ergodic averages.

### Loss & Training

The training cycle follows Algorithm 1: for each iteration (a) sample a batch; (b) enumerate $a \in \{0, 1\}$ to obtain reward $r$ and cost $c$; (c) calculate $\underline{\rho}(i) \propto \exp((\bar{r} - r_i)/\tau)$ and $\bar{\rho}(i) \propto \exp((c_i - \bar{c})/\tau)$ based on batch means $\bar{r}, \bar{c}$; (d) update $\pi$ and $\lambda$ via primal-dual; (e) select the best iterate on the validation set.

## Key Experimental Results

### Main Results

Data: Skywork Reward Preference subset + Math-Step-DPO-10K + Code-Preference-Pairs (40K total for training); evaluated on RewardBench / RewardBench-2 / JudgeBench. The judge pair consists of Qwen3-1.7B / 4B / 8B in reasoning vs. instruct modes. $C$ represents the cost ratio (reasoning/instruct token ratio).

| Model Scale | Method | Accuracy | Cost ratio |
|-------------|--------|----------|------------|
| 4B | All-Instruct | ~81.0 | 1.0 |
| 4B | All-Reasoning | ~85.5 | 11.2 (High) |
| 4B | Random | ~83.5 | 3.4 |
| 4B | **RACER (C=3.4)** | **~85.8** | 3.4 |
| 1.7B | RouterBench-KNN | 71.3 | 2.6 |
| 1.7B | RouteLLM-MF | 69.4 | 3.8 |
| 1.7B | M-IRT | 71.6 | 3.4 |
| 1.7B | **RACER (C=4)** | **72.2** | 3.6 |
| 8B | M-IRT | 88.9 | 3.4 |
| 8B | **RACER (C=4)** | **90.0** | 3.9 |

At approximately half the cost of All-Reasoning, RACER matches or exceeds its accuracy. Compared to SOTA router baselines, it achieves gains of 0.64, 1.10, and 1.06 points at the 1.7B, 4B, and 8B scales, respectively.

### Ablation Study

| Configuration | OOD Scenario | Conclusion |
|---------------|--------------|------------|
| ACER (Non-robust) | OOD gets expensive | Budget violated and reward drops |
| RACER-R only | OOD gets cheap | Highest reward (more aggressive budget use) |
| RACER-C only | OOD gets expensive | Cost safe (within budget) but lower reward |
| Full RACER | Both | Stable in both, optimal robustness |

Sensitivity of Entropy Regularization $\beta$ (Qwen3-4B):

| $\beta$ | $C=2$ Acc | $C=3$ Acc | $C=4$ Acc |
|---------|-----------|-----------|-----------|
| 0 | 85.2 | 86.7 | 86.8 |
| 0.005 | 85.5 | 86.7 | 86.7 |
| 0.01 | 85.5 | 86.7 | 86.7 |
| 0.05 | 84.8 | 86.0 | 86.2 |

Under tight budgets, $\beta = 0$ results in drops; $\beta \in [0.005, 0.01]$ is stable, while $\beta = 0.05$ is too strong and harms performance.

### Key Findings
- Reasoning judge gains are highly domain-dependent: significant improvements in math/coding (+10% or more), but negligible or negative gains in safety/knowledge; reasoning consumes $11.2\times$ tokens on average.
- Random routing approximates a linear interpolation between All-Instruct and All-Reasoning; RACER's curve is significantly concave towards the top-left, proving instance-level selection is far more effective.
- Distribution shifts indeed exist across benchmarks (training on Skywork, testing on RewardBench/JudgeBench). Non-robust ACER violates budgets or loses accuracy in certain settings.
- Cross-model family transfer: Trends remain consistent when Llama-3.1-8B is added (Appendix) despite training on Qwen3.

## Highlights & Insights
- **Framing "Reasoning is not free"**: Directly addresses a core pain point in the reasoning model era—vying for reasoning capabilities without considering the full cost-accuracy trade-off. Figure 2 visually demonstrates the highly unequal benefits of reasoning across benchmarks.
- **Split Robustness for Reward and Cost**: A clean design choice demonstrating that OOD distortions for reward and cost are independent; separate robustness is necessary for safety.
- **Closed-form KL Reweighting**: Implements DRO as "weighted sample gradients," requiring near-zero additional engineering overhead and facilitating adoption in production.
- **Theoretical Linear Last-Iterate Convergence**: Provides value for theoretical readers; deployment simply requires the final checkpoint rather than an ergodic average.
- The combination of entropy and dual regularization ensures a unique saddle point, providing a solid theoretical foundation.

## Limitations & Future Work
- Restricted to binary routing (reasoning vs. non-reasoning); extending to $K$ candidate judges requires moving to multi-classification, which is theoretically possible but underexplored in implementation.
- KL balls can be overly conservative under large distribution shifts, causing the router to default to always-instruct; alternative uncertainty sets (Wasserstein / $\chi^2$) are worth exploring.
- Assumes bounded cost (Assumption 2) and bounded density ratios (Assumption 3), which may not hold under severe OOD.
- Hyperparameter $\tau$ is not adaptively tuned (grid search only).
- Training requires executing reasoning judges for every instance in the preprocessing stage, which involves high initial token costs.
- Ground-truth labels for judging tasks rely on human-annotated preference datasets, assuming these labels are reliable.

## Related Work & Insights
- **vs. ThinkSwitcher (Liang 2025)**: Both handle mode switching for reasoning models, but ThinkSwitcher is heuristic and lacks robustness or theoretical guarantees; RACER provides a principled DRO framework.
- **vs. RouteLLM-MF (Ong 2024) / RouterBench (Hu 2024)**: Traditional multi-LLM routers for strong vs. weak model selection; RACER focuses on intra-model mode switching, though the framework is transferable.
- **vs. FrugalGPT (Chen 2023)**: A cascading strategy; RACER is single-shot, ensuring predictable latency.
- **vs. DRO Literature (Namkoong & Duchi 2016)**: Using $f$-divergence balls for DRO is classical; the novelty here lies in the "separate reward/cost robustness" and the "linear convergence for binary policies."

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "reasoning is not free for judge" + dual robustness + linear convergence is fresh in LLM router literature, though the underlying DRO mechanisms are well-established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive analysis across 3 benchmarks, 3 model scales, and 4 baselines, including ablation and sensitivity; however, testing was limited to Qwen and Llama.
- Writing Quality: ⭐⭐⭐⭐⭐ The controlled study in Section 2 builds a solid motivation; the derivations in Section 3 and theory in Section 4 are clear and concise.
- Value: ⭐⭐⭐⭐ Directly addresses the real-world cost pain points of reasoning models and OOD robustness; highly applicable to industrial LLM-as-a-judge pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas](../../ICLR2026/llm_evaluation/doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas.md)
- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](../../ACL2026/llm_evaluation/reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](../../ICLR2026/llm_evaluation/multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](../../ICLR2026/llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)

</div>

<!-- RELATED:END -->
