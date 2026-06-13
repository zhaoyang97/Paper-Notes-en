---
title: >-
  [Paper Note] RACO: Reward-free Alignment for Conflicting Objectives
description: >-
  [ICML 2026][Optimization][Multi-objective Alignment] RACO frames multi-objective LLM preference alignment as a multi-objective optimization problem—applying individual DPO losses for each objective and resolving gradient…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Multi-objective Alignment"
  - "Gradient Conflict"
  - "CAGrad-Clip"
  - "Pareto-critical point"
  - "DPO"
date: 2026-05-08
content_hash: fbe1af518ef5e4c5
---

# RACO: Reward-free Alignment for Conflicting Objectives

**Conference**: ICML 2026  
**arXiv**: [2602.02495](https://arxiv.org/abs/2602.02495)  
**Code**: TBD  
**Area**: Optimization / LLM Alignment / Multi-objective Optimization  
**Keywords**: Multi-objective Alignment, Gradient Conflict, CAGrad-Clip, Pareto-critical point, DPO

## TL;DR
RACO frames multi-objective LLM preference alignment as a multi-objective optimization problem—applying individual DPO losses for each objective and resolving gradient conflicts via clipped CAGrad (CAGrad with coefficients clipped by user weights). It provides theoretical guarantees for convergence to Pareto-critical points respecting user-specified weights (with strict acceleration in two-objective scenarios) and empirically achieves superior Pareto trade-offs across Qwen 3, Llama 3, and Gemma 3 model families.

## Background & Motivation

**Background**: Mainstream LLM alignment relies on RLHF (reward modeling + RL). Recently, reward-free routes (DPO, SimPO, IPO, KTO, etc.) optimize directly on preference pairs offline. however, most are single-objective, whereas human alignment is inherently multi-objective (helpful, harmless, faithful, concise).

**Limitations of Prior Work**: (1) Linear weighting for multi-objective aggregation fails to find directions that improve all objectives simultaneously when gradients conflict, inevitably sacrificing certain goals; (2) Existing multi-objective RL approaches (e.g., MODPO, Rame 2023) require training multiple reward models or weight-conditioned policies, which are complex and prone to reward model distortion; (3) AMoPO is reward-free but does not explicitly handle conflicts; (4) The "alignment tax" (drop in helpfulness as safety increases) and jailbreak phenomena reported by OpenAI are concrete manifestations of multi-objective conflicts.

**Key Challenge**: A solution that simultaneously satisfies reward-free pipeline simplification, explicit gradient conflict handling, and respect for user-specified weights does not exist. While CAGrad resolves conflicts in multi-task learning, its conflict-correction may be too aggressive in high-dimensional LLM fine-tuning, pushing updates toward less-preferred objectives.

**Goal**: (1) Reward-free multi-objective alignment; (2) Explicit handling of gradient conflicts; (3) Respect for user-specified weights; (4) Pareto convergence guarantees.

**Key Insight**: Treat multi-objective preference alignment as a multi-objective optimization problem—one DPO-style preference loss per objective, each with its own gradient. CAGrad serves as a natural primitive for a reward-free framework, but requires clipping to address over-correction in high-dimensional spaces.

**Core Idea**: CAGrad-Clip — The correction coefficients $p^*$ solved by CAGrad are clipped element-wise by the user weight $w$, such that $\tilde p = \min(p^*, w)$. This prevents the correction from pushing any objective weight beyond the user's specification, maintaining the user trade-off while benefiting from conflict mitigation.

## Method

### Overall Architecture

The DPO loss for each objective $i$ is: $\mathcal{L}_i(\theta) = -\mathbb{E}[\log \sigma(\beta(\log \pi_\theta(y_i^+|x)/\pi_{\text{ref}} - \log \pi_\theta(y_i^-|x)/\pi_{\text{ref}}))]$

Per step:
1. Compute $g_i = \nabla_\theta \mathcal{L}_i$ and weighted $g_0 = \sum_i w_i g_i$.
2. Solve $p^* \in \arg\min_p \{G_p^\top g_0 + c\|g_0\|\|G_p\|\}$ (CAGrad dual problem, $G_p = \sum_i p_i g_i$).
3. **Clip**: $\tilde p_i = \min(p_i^*, w_i)$.
4. $\tilde G_p = \sum_i \tilde p_i g_i$.
5. $G_0 = g_0 + c\|g_0\|\tilde G_p / \|\tilde G_p\|$ (if $\|\tilde G_p\| > 0$, else $G_0 = g_0$).
6. $\theta \leftarrow \theta - \eta G_0$.

### Key Designs

1. **CAGrad-Clip: User-Specified Weight Constraint on Correction**:
    - **Function**: Prevents CAGrad correction from pushing the update in a direction more biased than the user weight.
    - **Mechanism**: The $p^*$ solved by vanilla CAGrad may cause an objective's proportion to exceed $w_i$ (due to noise and over-correction in high dimensions). Clipping $\tilde p_i = \min(p_i^*, w_i)$ ensures the correction does not exceed user authorization.
    - **Design Motivation**: In the high-dimensional parameter space of LLM fine-tuning, CAGrad's trust-region search is noisy. Clipping acts as a trade-off-preserving hard constraint that is simple yet effective.

2. **Pareto Convergence Guarantee (Theorem 3.1)**:
    - **Function**: Theoretically proves that clipped updates still converge to Pareto-critical points.
    - **Mechanism**: Defines weighted loss $\mathcal{L}_w = \sum_i w_i \mathcal{L}_i$. Proves any limit point is simultaneously a critical point of $\mathcal{L}_w$ and a Pareto-critical point of $(\mathcal{L}_1, \dots, \mathcal{L}_m)$. Convergence rate: $\min_t \mathcal{M}(\theta_t)^2 \leq 2\mathcal{L}_w(\theta_0) / (\eta(1-c^2)T)$.
    - **Design Motivation**: Clipping alters the original CAGrad convergence analysis, necessitating a new proof to ensure theoretical completeness in converging to user-respecting Pareto points.

3. **Strict Acceleration in Two-Objective Scenarios (Theorem 3.2)**:
    - **Function**: Proves that clipping is strictly superior to no clipping in two-objective scenarios.
    - **Mechanism**: For two objectives, clipping allows the correction direction to more accurately reflect user weights, resulting in a strictly better convergence rate coefficient.
    - **Design Motivation**: Two-objective scenarios are the most common in LLM alignment (helpful vs harmless); a strict acceleration conclusion is highly persuasive.

## Key Experimental Results

### Multi-objective Summarization (Helpfulness vs Harmlessness)

| Method | Helpful (↑) | Harmless (↑) | Pareto Distance (↓) |
|------|--------|--------|----------|
| Weighted DPO (Linear) | 6.8 | 7.2 | 0.41 |
| MODPO (with reward model) | 7.1 | 7.4 | 0.32 |
| AMoPO (reward-free) | 7.3 | 7.6 | 0.28 |
| **RACO (CAGrad-Clip)** | **7.6** | **7.9** | **0.18** |

Ours consistently leads across multiple model families (Qwen 3-7B, Llama 3-8B, Gemma 3-9B).

### Safety Alignment (Safety vs Capability)

| Method | Capability MMLU | Safety Score | Tax (% Decrease) |
|------|----------|----------|----|
| Single-obj DPO (safety only) | 62.4 | 89.5 | -8.3% |
| Linear-weight multi-obj | 65.8 | 84.2 | -3.5% |
| AMoPO | 66.7 | 85.7 | -2.6% |
| **RACO** | **67.9** | **87.1** | **-1.4%** |

RACO significantly reduces the alignment tax (1.4% capability drop vs. 8.3% for single-objective DPO) while maintaining safety scores close to the single-objective safety baseline.

### Ablation Study

| Configuration | Helpful | Harmless | Pareto Distance |
|------|------|------|----|
| Full RACO (CAGrad-Clip) | 7.6 | 7.9 | 0.18 |
| Without clipping (vanilla CAGrad) | 7.4 | 7.5 | 0.27 |
| Without CAGrad (Pure weighted DPO) | 6.8 | 7.2 | 0.41 |
| MGDA Replacement | 6.9 | 7.3 | 0.36 |

The clipping component alone improves Pareto distance by +0.09, though CAGrad itself provides the largest contribution.

### Convergence Speed
In two-objective scenarios, CAGrad-Clip reaches the same Pareto distance approximately 25% faster than vanilla CAGrad (experimentally validating Theorem 3.2).

### Key Findings
- **Clipping is a critical fix for high-dimensional friendliness**: Vanilla CAGrad over-corrects on LLMs; clipping provides significant improvement.
- **Reward-free + Conflict Handling**: RACO is the first method to satisfy both requirements.
- **Significant reduction in alignment tax**: RACO preserves capability while achieving safety.
- **Generalization across model families**: Benefits seen in Qwen, Llama, and Gemma regardless of model architecture.

## Highlights & Insights
- **Reframing preference alignment as multi-objective optimization**: Instead of incremental changes to RLHF/DPO frameworks, this work brings tools from gradient conflict literature, offering a fresh perspective.
- **Clipping as a simple but key fix**: Vanilla CAGrad is unstable on LLMs; clipping stabilizes it. This combination of "simple engineering fix + rigorous theory" is highly practical.
- **Full cycle of theory and empirics**: Provides both convergence guarantees (Theorem 3.1) and acceleration conclusions (Theorem 3.2), validated across multiple model families.
- **Extensibility**: CAGrad-Clip is not limited to LLM alignment; it is applicable to any high-dimensional multi-objective optimization scenario (e.g., multi-task learning, multimodal training).

## Limitations & Future Work
- Only validated on 2-3 objectives; as dimensions of the CAGrad subproblem increase (5+), it may remain noisy.
- $c$ (trust region radius) is a manual hyperparameter; adaptive methods might be more robust.
- Evaluated only on summarization and safety; other alignment scenarios like code, math, and reasoning are untested.
- Clipping is a hard constraint $\tilde p = \min(p, w)$; soft clipping (e.g., sigmoid) might be smoother.
- Online settings (streaming new preference pairs) were not explored.

## Related Work & Insights
- **vs MODPO**: MODPO requires a reward model; RACO is reward-free.
- **vs AMoPO**: AMoPO is reward-free but does not handle conflicts; RACO does so explicitly.
- **vs MGDA / vanilla CAGrad**: MGDA ignores user weights; CAGrad over-corrects in LLMs; RACO addresses both.
- **Insight**: The clipping idea is applicable to all "multi-objective + high-dimensional + user preference" scenarios; the combination of reward-free + multi-objective is also relevant to many designs in RL.

## Rating
- Novelty: ⭐⭐⭐⭐ CAGrad-Clip is a simple but effective fix; innovative framing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Cross-model family × multi-task + detailed ablation + convergence verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical and algorithmic chain; intuitive comparison in the capability matrix.
- Value: ⭐⭐⭐⭐⭐ Alignment tax is a major pain point in LLM deployment; RACO provides an efficient reward-free solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)
- [\[ICLR 2026\] LCA: Local Classifier Alignment for Continual Learning](../../ICLR2026/optimization/lca_local_classifier_alignment_for_continual_learning.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)
- [\[ICLR 2026\] Celo2: Towards Learned Optimization Free Lunch](../../ICLR2026/optimization/celo2_towards_learned_optimization_free_lunch.md)

</div>

<!-- RELATED:END -->
