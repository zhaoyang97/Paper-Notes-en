---
title: >-
  [Paper Note] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge
description: >-
  [ICML 2026][LLM Evaluation][LLM-as-a-Judge] RACER formulates the problem of "deciding whether to invoke reasoning mode for each query in LLM-as-a-Judge" as a distributionally robust constrained optimization with a KL unc…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "Reasoning Model Routing"
  - "KL Uncertainty Set"
  - "Primal-Dual"
  - "OOD Robustness"
date: 2026-05-08
content_hash: 5d41c25afcd7b5c7
---

# Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge

**Conference**: ICML 2026  
**arXiv**: [2605.10805](https://arxiv.org/abs/2605.10805)  
**Code**: None  
**Area**: LLM Evaluation / Model Routing / Distributionally Robust Optimization  
**Keywords**: LLM-as-a-Judge, Reasoning Model Routing, KL Uncertainty Set, Primal-Dual, OOD Robustness

## TL;DR
RACER formulates the problem of "deciding whether to invoke reasoning mode for each query in LLM-as-a-Judge" as a distributionally robust constrained optimization with a KL uncertainty set. It solves for the optimal routing policy under OOD conditions that still satisfies the cost budget using a primal-dual algorithm, and for the first time provides a linear convergence guarantee for LLM router policies.

## Background & Motivation

**Background**: LLM-as-a-Judge increasingly leverages reasoning models (o1, DeepSeek-R1, Qwen3 thinking, etc.) for evaluation. These models learn reasoning via RL on verifiable tasks, but the judgment task itself is not explicitly optimized, leaving open the question of whether reasoning truly improves judgment accuracy. A natural intermediate solution is routing—dynamically selecting reasoning or instruct mode based on query difficulty.

**Limitations of Prior Work**: Existing LLM routing works (FrugalGPT, P2L, RouteLLM, ThinkSwitcher) share three main shortcomings. First, they almost exclusively focus on QA tasks, neglecting the judge scenario. Second, they only optimize the "cost-accuracy tradeoff under the training distribution," so when the query distribution shifts at deployment (e.g., user base or domain mix changes), cost constraints are violated and performance collapses. Third, most are empirical or heuristic, lacking theoretical convergence guarantees. The paper also empirically shows that reasoning judges significantly improve accuracy in math/coding but may have negative or negligible gains in safety/knowledge, with token costs increasing severalfold—indiscriminate use of reasoning is both expensive and potentially detrimental.

**Key Challenge**: Reasoning mode is costly and not universally beneficial (overthinking can be harmful), but training data is static, so both reward estimation and cost budgeting become inaccurate under OOD deployment.

**Goal**: Learn a routing policy $\pi(a | z)$ ($a \in \{0, 1\}$ indicates whether to activate reasoning) under a fixed cost budget $C$, such that (i) expected judge reward is maximized; (ii) robustness to query distribution shift is ensured; (iii) theoretical convergence is guaranteed.

**Key Insight**: Use distributionally robust optimization (DRO) with a KL uncertainty set and Lagrangian primal-dual approach. Both reward and cost are measured in the worst case, with "robustness on reward" and "robustness on cost" handled separately (the former prevents overestimating benefits under OOD, the latter prevents budget overruns under OOD).

**Core Idea**: Reformulate LLM-as-a-Judge routing as $\max_\pi \min_{\tilde{\rho} \in \mathcal{U}(\rho_n, \delta)} \mathbb{E}_{\tilde{\rho}}[r] \text{ s.t. } \max_{\tilde{\rho} \in \mathcal{U}} \mathbb{E}_{\tilde{\rho}}[c] \leq C$, and prove that under a KL uncertainty set, the worst-case distribution admits a closed-form reweighting, enabling efficient primal-dual optimization.

## Method

### Overall Architecture

Input: Preference dataset $\{(x_i, y_{i,1}, y_{i,2}, l_i)\}$ (with ground-truth preference labels); a hybrid LLM providing reasoning judge $\Phi_1$ and non-reasoning judge $\Phi_0$.

Preprocessing: For each instance, run both modes, record reward $r_i = \mathbb{I}(\Phi_{a_i}(z_i) = l_i)$ and cost $c_i$ (token count).

Router: 4-layer neural network; input is the embedding of prompt+response concatenated, obtained via bge-m3; output is the probability of reasoning.

Training: For each batch, the primal-dual algorithm (a) computes empirical means of reward/cost as baselines, (b) computes worst-case distributions $\underline{\rho}, \bar{\rho}$ via closed-form reweighting, (c) updates policy $\pi_{t+1}$ and dual $\lambda_{t+1}$. The best iterate is selected on validation.

### Key Designs

1. **Dual Robustness in Distributionally Robust Constrained Optimization (Separate Robustness for Reward and Cost)**:

    - **Function**: Formulate router learning as $\max_\pi R_{\mathcal{U}(\rho_n, \delta)}(\pi)$ s.t. $C_{\mathcal{U}(\rho_n, \delta)}(\pi) \leq C$, where $R$ is the worst-case reward and $C$ is the worst-case cost, with the uncertainty set being a KL ball centered at the empirical distribution $\rho_n$.
    - **Mechanism**: Traditional DRO only robustifies one objective. This work recognizes that reward and cost can be distorted in different directions under OOD—OOD queries may be cheaper (cost robustness less important, need to robustify reward to use budget more aggressively) or more expensive (cost robustness is key to prevent budget overrun). Thus, worst-case is taken separately for both, ensuring safety in both OOD scenarios.
    - **Design Motivation**: Figure 3 demonstrates the necessity of this split—RACER-R (robustifying only reward) exceeds budget in OOD costlier scenarios; RACER-C (robustifying only cost) wastes budget in OOD cheaper scenarios; only dual robustness is stable in both.

2. **Closed-form Worst-case Reweighting for KL Uncertainty Set (Theorem 3.1)**:

    - **Function**: Converts the abstract $\min/\max$ over $\mathcal{U}(\rho_n, \delta)$ into closed-form sample reweighting.
    - **Mechanism**: Define $f_i = \mathbb{E}_{a \sim \pi(\cdot | z_i)}[f(z_i, a)]$, then $\underline{\rho}(i) \propto \rho_n(i) \exp(\frac{\underline{s} - f_i}{\tau})$ (minimization), $\bar{\rho}(i) \propto \rho_n(i) \exp(\frac{f_i - \bar{s}}{\tau})$ (maximization). Intuitively, for reward, the worst-case distribution downweights samples with above-baseline reward and upweights those below baseline; for cost, it upweights high-cost samples, focusing optimization on "high-risk regions." $\tau$ controls the extremity of reweighting (smaller $\tau$ yields more extreme weights).
    - **Design Motivation**: Direct alternating gradient over parameterized distributions is infeasible (most distributions in the uncertainty set lack samples); closed-form reweighting turns "worst-case over unknown distributions" into "weighted known samples," requiring only per-sample weighting—simple in practice. The principle is inspired by Gadot et al. 2024 / Xu et al. 2025 on distributionally robust RL.

3. **Entropy-regularized Primal-Dual Algorithm + Linear Convergence Proof**:

    - **Function**: Solves the constrained min-max Lagrangian $L_\beta(\pi, \lambda) = R_{\underline{\rho}}(\pi) - \lambda C_{\bar{\rho}}(\pi) + \beta(\mathcal{H}(\pi) + \frac{1}{2}\lambda^2)$ to obtain $(\pi^*, \lambda^*)$.
    - **Mechanism**: Alternately update $\pi_{t+1} = \arg\max_\pi \{R_{\underline{\rho}}(\pi) - \lambda_t C_{\bar{\rho}}(\pi) + \beta \mathcal{H}(\pi)\}$ and $\lambda_{t+1} = \arg\max_{\lambda \geq 0}\{-\lambda C_{\bar{\rho}}(\pi) + \frac{1}{2}\beta \lambda^2\}$. Note that the $\pi$ update can be rewritten as a weighted objective over the original distribution $\rho$: $\mathbb{E}_{\rho, \pi}[\frac{p_{\underline{\rho}}}{p_\rho} r - \lambda_t \frac{p_{\bar{\rho}}}{p_\rho} c] + \beta \mathcal{H}$. Theorem 4.1 proves the saddle point is unique; Theorem 4.2 gives $\text{KL}(\pi_t \| \pi^*) \leq \frac{M^2 K^2}{2 \beta^2} (\frac{M^2 K^2}{M^2 K^2 + 2 \beta^2})^{2t} (\lambda_0 - \lambda^*)^2$, i.e., linear convergence.
    - **Design Motivation**: Entropy regularization $\mathcal{H}(\pi)$ is a classic RL technique (Cen et al. 2022, Ding et al. 2023), preventing the policy from degenerating to deterministic and encouraging exploration, while enabling last-iterate convergence in primal-dual. The $\frac{1}{2}\lambda^2$ regularization bounds the dual variable. Together, these ensure a unique saddle point and last-iterate convergence (rather than traditional ergodic average), making it practical to deploy the final model checkpoint. This is the first such guarantee for LLM routers.

### Loss & Training

Full training loop (Algorithm 1): Each iteration (a) sample a batch; (b) enumerate $a \in \{0, 1\}$ to obtain reward $r$ and cost $c$; (c) compute $\underline{\rho}(i) \propto \exp((\bar{r} - r_i)/\tau)$, $\bar{\rho}(i) \propto \exp((c_i - \bar{c})/\tau)$ using current batch means $\bar{r}, \bar{c}$; (d) primal-dual update of $\pi$ and $\lambda$; (e) select the best iterate on validation. Hyperparameter $\tau$ controls robustness strength, $\beta$ controls entropy regularization.

## Key Experimental Results

### Main Results

Data: Skywork Reward Preference subset + Math-Step-DPO-10K + Code-Preference-Pairs (total 40K training); evaluated on RewardBench / RewardBench-2 / JudgeBench; judge pairs are Qwen3-1.7B / 4B / 8B reasoning vs instruct modes. Budget $C$ is the cost ratio (reasoning/instruct token ratio).

| Model Size | Method | Accuracy | Cost ratio |
|------------|--------|----------|------------|
| 4B | All-Instruct | ~81.0 | 1.0 |
| 4B | All-Reasoning | ~85.5 | 11.2 (expensive) |
| 4B | Random | ~83.5 | 3.4 |
| 4B | **RACER (C=3.4)** | **~85.8** | 3.4 |
| 1.7B | RouterBench-KNN | 71.3 | 2.6 |
| 1.7B | RouteLLM-MF | 69.4 | 3.8 |
| 1.7B | M-IRT | 71.6 | 3.4 |
| 1.7B | **RACER (C=4)** | **72.2** | 3.6 |
| 8B | M-IRT | 88.9 | 3.4 |
| 8B | **RACER (C=4)** | **90.0** | 3.9 |

At roughly half the cost of All-Reasoning, RACER matches or exceeds All-Reasoning accuracy; compared to SOTA router baselines, RACER outperforms by 0.64, 1.10, and 1.06 points on 1.7B/4B/8B, respectively.

### Ablation Study

| Configuration | OOD Scenario | Conclusion |
|---------------|-------------|------------|
| ACER (non-robust) | OOD costlier | Exceeds budget and reward drops |
| RACER-R only | OOD cheaper | Highest reward (more aggressive budget use) |
| RACER-C only | OOD costlier | Cost safe (within budget) but lower reward |
| Full RACER | Both | Stable in both, best robustness |

Entropy regularization $\beta$ sensitivity (Qwen3-4B):

| $\beta$ | $C=2$ Acc | $C=3$ Acc | $C=4$ Acc |
|---------|-----------|-----------|-----------|
| 0 | 85.2 | 86.7 | 86.8 |
| 0.005 | 85.5 | 86.7 | 86.7 |
| 0.01 | 85.5 | 86.7 | 86.7 |
| 0.05 | 84.8 | 86.0 | 86.2 |

With tight budgets, $\beta = 0$ underperforms; $\beta \in [0.005, 0.01]$ is stable; $\beta = 0.05$ is too strong and hurts performance.

### Key Findings
- The gain from reasoning judges is highly domain-dependent: large improvements in math/coding (+10% or more), almost none or even negative in safety/knowledge; reasoning uses on average $11.2\times$ more tokens.
- Random routing is a near-linear interpolation between All-Instruct and All-Reasoning on the cost-accuracy curve, while RACER's curve is clearly concave towards the upper left, demonstrating that instance-level selection is much more effective than random activation at a fixed ratio.
- Distribution shift between real benchmarks is significant (training on Skywork, testing on RewardBench/JudgeBench); non-robust ACER violates budget or loses accuracy in some settings.
- Cross-model family transfer: training on Qwen3, adding Llama-3.1-8B (see appendix) shows consistent trends.

## Highlights & Insights
- The "reasoning is not free" framing directly addresses the core pain point of the reasoning model era—everyone is pursuing reasoning, but few consider the full cost-accuracy tradeoff. Figure 2 plots $\Delta$Accuracy vs cost ratio across benchmarks, making the uneven gains of reasoning immediately apparent.
- Separately robustifying reward and cost is a clean design—prior DRO works mostly use single robustness, but here it's clear that OOD distortions are independent, so separate robustness is correct.
- Closed-form reweighting for KL uncertainty set implements DRO as "weighted sample gradients," with almost zero engineering overhead, facilitating adoption in real systems.
- First work to prove linear last-iterate convergence for LLM routers, valuable for theory-oriented readers—deployment can simply use the last checkpoint, no need for ergodic averaging.
- The combination of entropy and dual regularization ensures a unique saddle point, a neat theoretical highlight.

## Limitations & Future Work
- Only binary routing (reasoning vs non-reasoning) is considered; extending to $K$ candidate judges (different model families/scales) would require changing $\pi: \mathcal{Z} \to \Delta(K)$ to multiclass. Theoretically, binary results can be extended, but engineering details are not covered.
- KL ball can be overly conservative under large distribution shifts (worst-case too pessimistic), causing routing to degenerate to always-instruct; the authors acknowledge that alternative uncertainty sets (Wasserstein / $\chi^2$) are worth exploring.
- Assumes bounded cost (Assumption 2) and bounded density ratio (Assumption 3), the latter may not hold under severe OOD.
- $\tau$ is not adaptively tuned, only grid searched.
- Training requires enumerating both judge modes, so each instance must be run through the reasoning judge, making preprocessing itself costly in tokens.
- The ground-truth labels for judging rely on human-annotated preference datasets, assuming these labels are reliable.

## Related Work & Insights
- **vs ThinkSwitcher (Liang 2025)**: Also switches modes in hybrid reasoning models, but ThinkSwitcher is heuristic, lacks distribution shift handling and theoretical guarantees; RACER formulates it as principled DRO.
- **vs RouteLLM-MF (Ong 2024) / RouterBench (Hu 2024)**: Traditional multi-LLM routers for strong vs weak model selection; RACER focuses on single-model mode switching, but the theoretical framework can transfer to multi-model.
- **vs FrugalGPT (Chen 2023)**: Cascading strategy, querying models in sequence until satisfied; RACER is single-shot mode selection, with controllable latency.
- **vs DRO literature (Namkoong & Duchi 2016, Duchi & Namkoong 2021)**: Using $f$-divergence balls for DRO is classic; this work's novelty lies in "separate robustness for reward/cost" and "binary policy + entropy regularization + linear convergence."

## Rating
- Novelty: ⭐⭐⭐⭐ "Reasoning is not free for judge" + dual robustness + linear convergence is a first in LLM router literature, though the underlying DRO approach is classic.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 benchmarks, 3 model scales, 4 baselines + ablation + sensitivity analysis; but only Qwen3 + Llama, and budget range is moderate.
- Writing Quality: ⭐⭐⭐⭐⭐ Section 2's controlled study strongly motivates the work, Section 3's method derivation is clear, Section 4's theory is concise.
- Value: ⭐⭐⭐⭐ Reasoning model deployment cost is a real pain point, OOD robustness is a real problem, and this is directly useful for industrial LLM-as-judge pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](../../ACL2025/llm_evaluation/yescieval_llm_judge_science.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](../../ICLR2026/llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ICML 2026\] Dual-branch Robust Unlearnable Examples](dual-branch_robust_unlearnable_examples.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](../../ICLR2026/llm_evaluation/multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)

</div>

<!-- RELATED:END -->
