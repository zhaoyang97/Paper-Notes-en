---
title: >-
  [Paper Note] LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation
description: >-
  [ICML 2026][Reinforcement Learning][Bayesian Optimization] This paper introduces LABO, which treats Large Language Models (LLMs) as "low-fidelity" evaluation sources integrated into the Bayesian Optimization loop. It uti…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Bayesian Optimization"
  - "LLM Prior"
  - "Multi-fidelity"
  - "KOH Model"
  - "Gating Criterion"
date: 2026-05-08
content_hash: d4958a9a12795a4b
---

# LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation

**Conference**: ICML 2026  
**arXiv**: [2605.22054](https://arxiv.org/abs/2605.22054)  
**Code**: Not disclosed  
**Area**: Bayesian Optimization / LLM Acceleration / Multi-fidelity / Scientific Discovery  
**Keywords**: Bayesian Optimization, LLM Prior, Multi-fidelity, KOH Model, Gating Criterion  

## TL;DR
This paper introduces LABO, which treats Large Language Models (LLMs) as "low-fidelity" evaluation sources integrated into the Bayesian Optimization loop. It utilizes a Kennedy–O'Hagan (KOH) joint Gaussian Process (GP) to decompose the real experiment $f_R$ into a scaled LLM prediction $\rho f_L$ and a residual process $\delta$. A "discrepancy dominance ratio" $p_\Delta = \sigma_\delta^2/(\rho^2\sigma_L^2 + \sigma_\delta^2)$ serves as a gating mechanism to decide whether to perform a real experiment for each candidate. By leveraging nearly free LLM queries for broad exploration and focusing expensive real experiments on regions where the LLM is untrustworthy, LABO significantly outperforms vanilla BO, LLAMBO, BOPRO, and CAKE across six scientific optimization tasks (e.g., COF, Fullerene) under the same real-world budget.

## Background & Motivation

**Background**: Scientific formulation optimization (drug discovery, catalyst design, molecular engineering) involves expensive experiments for each evaluation. Consequently, Bayesian Optimization (BO) is the mainstream approach—using GP surrogates to model targets, acquisition functions (EI, UCB) to balance exploration and exploitation, and iteratively suggesting the next set of candidates. Recent work has begun integrating LLMs into BO: LLAMBO uses LLMs for initialization and candidate suggestions, BOPRO performs BO in the latent space of LLM embeddings, and CAKE injects LLM priors into GP kernels.

**Limitations of Prior Work**: Existing LLM+BO methods treat the LLM as a "suggestion provider" for sampling, surrogates, or acquisition functions but **fail to fully exploit the fact that LLM evaluation costs are significantly lower than real experiments**. While an LLM inference costs cents, real chemical synthesis can take days and cost thousands. Current methods only call LLMs lightly for initialization or local decisions, rather than systematically using them as independently samplable "low-fidelity evaluation sources." Furthermore, BO itself faces two persistent issues: cold-start problems (lack of initial data) and exploration difficulties in high-dimensional search spaces.

**Key Challenge**: To fully utilize the low-cost, broad-coverage capabilities of LLMs, they must be integrated into the surrogate as an evaluation source. However, LLM predictions can systematically deviate from real experiments (due to flawed chemical intuition or reasoning hallucinations); blind trust can lead the surrogate astray. The core problem is how to dynamically balance "broad exploration via LLMs" and "saving real experiments only where the LLM is trustworthy."

**Goal**: Design a BO framework that simultaneously addresses two questions: (i) how to fuse heterogeneous LLM signals and real-world fidelity into a unified probabilistic surrogate; (ii) how to decide whether to expend a real experiment for a given candidate at each step.

**Key Insight**: The multi-fidelity simulation field has a mature Kennedy–O'Hagan (KOH) joint GP framework, which treats high-fidelity data as a linear transformation of low-fidelity data plus a residual process, modeling each with a GP. The authors treat the LLM directly as a low-fidelity evaluation source ("knowledge fidelity," distinct from traditional numerical simulation fidelity) within the KOH framework. They use the variance proportion of the residual GP as an interpretable indicator of uncertainty to trigger real experiments.

**Core Idea**: Use KOH to decompose the real target as $f_R(x) = \rho f_L(x) + \delta(x)$, where $f_L$ fits LLM predictions and $\delta$ fits the discrepancy between the LLM and reality. The discrepancy dominance ratio $p_\Delta(x) = \sigma_\delta^2(x)/(\rho^2\sigma_L^2(x) + \sigma_\delta^2(x))$ is compared against a threshold $\tau$. If $p_\Delta$ is large, it indicates uncertainty is primarily due to LLM untrustworthiness, necessitating a real experiment; if $p_\Delta$ is small, the LLM prediction is trusted, and only $f_L$ is updated.

## Method

### Overall Architecture
LABO consists of two stages: warm-start and an optimization loop. **Warm-start**: The LLM recommends a small number of high-potential points $\mathcal{X}_R$ for real experiments based on task priors $\mathcal{P}$ to obtain $\mathcal{D}_R$; simultaneously, Latin Hypercube Sampling (LHS) triggers a set of space-filling points $\mathcal{X}_L$ (ensuring $\mathcal{X}_R \subset \mathcal{X}_L$) for LLM predictions to obtain $\mathcal{D}_L$. **Optimization loop**: In each round, $f_L \sim \mathcal{GP}(0, k_L)$ is trained on $\mathcal{D}_L$, $\rho$ is estimated via least squares, and $\delta \sim \mathcal{GP}(0, k_\delta)$ is trained on the residuals $\{(x, y_R - \rho y_L)\}$ to synthesize $f_R = \rho f_L + \delta$. A set of candidates $\mathcal{X}_t$ is selected using the q-UCB acquisition function. Each $x \in \mathcal{X}_t$ must be queried via the LLM and added to $\mathcal{D}_L$, then $p_\Delta(x)$ is calculated to determine whether to trigger a real experiment to update $\mathcal{D}_R$, continuing until the real-world budget is exhausted.

### Key Designs

1.  **KOH-based Dual-Fidelity Joint GP Surrogate**:
    - **Function**: Treats the LLM as an independent low-fidelity source, fusing it with real experiments in a unified probabilistic framework. Predicted mean and variance are $\mu_R(x) = \rho\mu_L(x) + \mu_\delta(x)$ and $\sigma_R^2(x) = \rho^2\sigma_L^2(x) + \sigma_\delta^2(x)$, respectively.
    - **Mechanism**: Assumes $f_L(x) \sim \mathcal{GP}(0, k_L)$ is trained on all LLM evaluations, while $\delta(x) \sim \mathcal{GP}(0, k_\delta)$ is trained on residuals between matched $(x, y_R)$ pairs and LLM predictions. $\rho$ is calibrated via $\rho = \arg\min_\rho \sum_{(x, y_R) \in \mathcal{D}_R}(y_R - \rho y_L)^2$. Since the GPs are connected through $\rho$, increasing LLM queries (even without new experiments) updates $(\mu_L, \sigma_L^2)$ and subsequently improves $(\mu_R, \sigma_R^2)$.
    - **Design Motivation**: Treating the LLM and real experiments as independent GPs rather than a simple weighted average allows for **adaptive identification of systematic bias**. If the LLM is accurate, the residual GP variance is small; if inaccurate, the residual GP naturally absorbs the bias. Compared to treating LLMs as a prior mean (difficult to tune) or a kernel (CAKE approach, unstable), KOH is more interpretable and requires fewer hyperparameter adjustments.

2.  **Discrepancy Dominance Gating Criterion**:
    - **Function**: For a candidate $x$, $p_\Delta(x) = \sigma_\delta^2(x)/(\rho^2\sigma_L^2(x) + \sigma_\delta^2(x))$ is calculated as the proportion of total uncertainty contributed by the residual GP. A real experiment is triggered to update $\mathcal{D}_R$ only if $p_\Delta(x) > \tau$; otherwise, only the LLM is queried to update $\mathcal{D}_L$.
    - **Mechanism**: The intuition is that if uncertainty is primarily driven by the discrepancy $\delta$, the LLM signal is unreliable at $x$, necessitating an experiment to reduce the residual. If uncertainty is driven by the LLM variance $\sigma_L^2$, it suggests the LLM simply hasn't predicted near that point yet, and more LLM queries (which are low-cost) suffice. The authors theoretically prove that this gating causes the "real experiment region" to converge to a stable subset $\mathcal{X}_R^*$, yielding a cumulative regret bound $R_T \le C_1\sqrt{T_R^*\beta_T \Psi_T(\mathcal{X}_R^*)} + C_2\sqrt{T^\alpha \beta_T \Psi_T(\mathcal{X})} + C_3\sqrt{T_L\beta_T\Psi_T(\mathcal{X})}$, where $\Psi_T(\mathcal{X}_R^*) \ll \Psi_T(\mathcal{X})$.
    - **Design Motivation**: Traditional multi-fidelity BO relies on manually set cost/benefit ratios for query decisions, which are hard to tune. $p_\Delta$ provides an **information-theoretic criterion** derived from the GP's internal uncertainty decomposition, quantifying whether further LLM queries can reduce uncertainty and delegating the decision to the probabilistic model.

3.  **Prior-guided Warm-start with LHS Coverage**:
    - **Function**: At the start, the LLM performs two tasks: recommending $\mathcal{X}_R$ (high-potential points for real experiments) based on scientific priors (literature, constraints, semantics), and predicting a space-filling set of points $\mathcal{X}_L = \mathcal{X}_R \cup \mathcal{X}_{\text{LHS}}$ (50 points used in the paper) via LHS.
    - **Mechanism**: The first set of points addresses cold-start—the LLM translates prior knowledge into "plausible recipes" via in-context reasoning, providing initial real data points. The second set addresses high-dimensional exploration—LHS provides uniform coverage, and LLM predictions allow $f_L$ to fit the global structure early on, preventing the initial random-walk behavior of traditional BO. $\mathcal{X}_R \subset \mathcal{X}_L$ ensures paired data is available to train $\rho$ and $\delta$ immediately.
    - **Design Motivation**: Cold-start and high-dimensional exploration are independent pain points in BO. Traditional methods use either LHS for coverage or expert points for cold-start; LABO uses the LLM as both an "expert" and a "cheap simulator" to solve both simultaneously.

### Loss & Training
All main experiments use a fixed $\tau = 0.75$, batch size 2, 3 initial real points, 50 warm-up LLM evaluations, q-UCB acquisition function, and an RBF kernel, without per-task or per-LLM tuning. The primary LLM backend is Intern S1 241B, with ablations testing Intern-S1-mini 7B, Qwen3-235B (Instruct/Thinking), and DeepSeek V3.1 685B.

## Key Experimental Results

### Main Results

| Task (Dim) | Metric | LABO | Vanilla BO | LLAMBO | BOPRO | CAKE |
|--------------|----------|------|------------|--------|-------|------|
| COF (14D) | Final Target | **Best** | Lagging | Early gain, then stuck | Early gain, then stuck | High fluctuation |
| Sandwich (20D)| Final Target | **Best** | Lagging | Stuck | Stuck | High fluctuation |
| PCE10 (4D) | Rate+Final | **Best** | Fast conv, low value | — | — | — |
| Fullerene (3D)| Final 0.9512 | **Best** | Lagging | — | — | — |
| Flow Battery (3D)| Final | **Best** | — | — | — | — |
| P3HT (5D) | Final | **Best** | — | — | — | — |

LABO achieved the best performance in all 6 scientific tasks, with significantly lower variance than baselines (especially CAKE). The advantage was most pronounced in high-dimensional tasks (COF, Sandwich) because LLM-driven broad exploration is more efficient than pure BO.

### Ablation Study

| $\tau$ | COF Final | COF Iter to 90% | COF L/R Ratio | Fullerene Final | Fullerene L/R Ratio |
|--------|----------|-------------------|------------|----------------|------------------|
| 0.60 | 10.778±0.276 | 24.60±2.51 | 1.52±0.29 | 0.9490 | 1.54 |
| 0.70 | 11.070 | 15.83 | 2.00 | 0.9511 | 2.00 |
| **0.75** | **11.228** | **14.17** | **2.68** | **0.9512** | **3.87** |
| 0.80 | 11.134 | 14.80 | 3.44 | 0.9506 | 5.69 |
| 0.85 | 11.171 | 12.60 | 5.26 | 0.9499 | 14.60 |

### Key Findings
- Feeding the same LLM-initialized points to vanilla BO (isolating the contribution of the starting points) still showed LABO significantly ahead, indicating the performance gain **does not** solely come from initialization but from the entire dual-fidelity loop.
- Replacing LLM predictions with uniform random values (within the same range) caused LABO's performance to collapse, proving that LLM scientific priors provide real signals—not just any "broad coverage" works.
- Stronger LLM backends perform better, but the gap is not extreme: Qwen3-Thinking outperformed Qwen3-Instruct (reasoning helps), and DeepSeek 685B slightly outperformed Intern-S1 241B, which in turn outperformed Intern-S1-mini 7B. This suggests LABO is robust to LLM choice.
- $\tau = 0.75$ is the sweet spot: lower $\tau$ relies too much on real experiments, losing the acceleration advantage; higher $\tau$ over-trusts the LLM, leading to bias. High-dimensional tasks (COF) yielded a lower L/R ratio (2.68), while low-dimensional tasks (Fullerene) yielded a higher ratio (3.87)—LABO automatically allocates budget based on task complexity.
- Sampling trajectory visualization (COF task) showed: LLM query points covered the entire search space, while real experiments concentrated on a few high-uncertainty sub-regions, consistent with the theoretical prediction of $\mathcal{X}_R^* \subsetneq \mathcal{X}$.

## Highlights & Insights
- Reframing the LLM as a "knowledge-based low-fidelity evaluation source" instead of a "suggestion generator" is a key conceptual shift. While previous work treated LLMs as advisors, LABO treats them as cheap experimental instruments within a multi-fidelity framework, making the mature KOH toolkit immediately applicable.
- Using $p_\Delta$, a decomposable uncertainty ratio internal to the GP, as a gating mechanism provides both interpretability and theoretical guarantees (regret bounds). This is far more robust than manually tuning a "cost/benefit threshold" and is a transferable strategy for any active learning or Bayesian Experimental Design (BED) scenario.
- The framework is largely independent of LLM accuracy—the theoretical analysis explicitly states that "no structural assumptions are made about the LLM oracle, allowing for global inaccuracies." When the LLM is inaccurate, the residual GP automatically takes over, reflecting a pragmatic engineering attitude toward LLMs as "potentially inaccurate but cheap signal sources."

## Limitations & Future Work
- The authors include a "no special emphasis needed" statement regarding social impact, but the risk of "unreliable predictions" from LLMs in scientific optimization is significant. If LLMs are systematically biased toward compounds common in their training data, LABO might favor mainstream regions and miss novel discoveries.
- Observation: $\rho$ is estimated via global least squares, assuming a linear constant relationship between the LLM and reality across the entire space. In reality, LLM accuracy likely varies by chemical class, potentially requiring a localized $\rho(x)$ (e.g., piecewise or GP-modeled $\rho$).
- Experiments were conducted in small batches (batch=2, 3 initial points, 50 warm-up). Real-world wet-lab budgets might be even tighter. $\tau = 0.75$ is fixed, and dynamic scheduling (e.g., trusting the LLM more early on and experiments more later) was not explored.
- LLM query costs are treated as "effectively zero," but for GPT-4 level models and complex tasks, LLM inference is not negligible. Future work should weight LLM costs in the regret analysis.

## Related Work & Insights
- **vs LLAMBO**: LLAMBO has the LLM recommend initial points and candidates, but ultimate decision-making rests with traditional acquisition functions. LABO treats the LLM as an independent evaluation source, with decision power residing in the GP surrogate and gating criterion; the LLM enters the likelihood calculation directly.
- **vs CAKE**: CAKE injects LLM priors into the GP kernel. LABO builds a separate GP for the LLM and couples it with the real GP via KOH. CAKE's instability often stems from kernel updates disrupting the GP posterior; LABO's decoupled approach is more stable.
- **vs Traditional Multi-fidelity BO (e.g., BOCA, MF-MES)**: These target numerical simulation fidelities (same physical model, different precisions). LABO introduces "knowledge fidelity"—the LLM is a language model, not a physical simulation, yet the same KOH framework applies effectively.
- **vs ChemBOMAS (LLM Pseudo-experiments)**: ChemBOMAS treats LLM predictions as initial observations but only during initialization. LABO uses the LLM continuously throughout the loop with a gating mechanism to control trust levels.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Repositioning the LLM as a multi-fidelity source + KOH joint GP + $p_\Delta$ gating is a novel combination. The interpretability of $p_\Delta$ is superior to prior LLM+BO works.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 scientific tasks across dimensions and domains. Compares against multiple baselines with 5 random seeds and various LLM backends. AutoML and high-dimensional tasks are supplemented in the appendix.
- **Writing Quality**: ⭐⭐⭐⭐ Section 4 clearly explains KOH, gating, and the workflow. The regret decomposition in Theorem 5.1 clearly identifies $\Psi_T(\mathcal{X}_R^*) \ll \Psi_T(\mathcal{X})$ as the source of improvement. Figure 4 provides intuitive visualization.
- **Value**: ⭐⭐⭐⭐ Provides a concrete, reproducible framework for integrating LLMs into scientific optimization. The gating strategy is portable to other high-cost sampling scenarios and has direct application value for materials, chemistry, and drug discovery workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](../../ACL2026/reinforcement_learning/dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](../../ACL2026/reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](../../NeurIPS2025/reinforcement_learning/improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](../../ACL2026/reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Gradient-Variation Online Adaptivity for Accelerated Optimization with Hölder Smoothness](../../NeurIPS2025/reinforcement_learning/gradient-variation_online_adaptivity_for_accelerated_optimization_with_hölder_sm.md)

</div>

<!-- RELATED:END -->
