---
title: >-
  [Paper Note] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Policy Gradient] This paper proposes CAPO (Curvature-Aware Policy Optimization), which models second-order optimization geometry at the LM head's final layer to predict and filter token updates…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Policy Gradient"
  - "Curvature-Aware"
  - "Sample Efficiency"
  - "GRPO"
  - "Second-Order Optimization"
date: 2026-05-08
content_hash: 5bdb61839afab66c
---

# Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning

**Conference**: ICLR 2026  
**arXiv**: [2510.00819](https://arxiv.org/abs/2510.00819)  
**Code**: [https://github.com/luckeciano/stable-pg-llm](https://github.com/luckeciano/stable-pg-llm)  
**Area**: Video Understanding  
**Keywords**: Policy Gradient, Curvature-Aware, Sample Efficiency, GRPO, Second-Order Optimization

## TL;DR
This paper proposes CAPO (Curvature-Aware Policy Optimization), which models second-order optimization geometry at the LM head's final layer to predict and filter token updates that would cause policy collapse. Under aggressive hyperparameters (5× learning rate, 1/12 batch size), CAPO maintains training stability and achieves a 30× sample efficiency improvement over standard GRPO on MATH.

## Background & Motivation

**Background**: Policy gradient methods such as GRPO and PPO are central to LLM reasoning post-training (e.g., DeepSeek-R1). Current practice requires highly conservative hyperparameters—learning rates as low as $3 \times 10^{-6}$ and batch sizes in the thousands—to ensure training stability.

**Limitations of Prior Work**: Conservative settings imply enormous sample requirements and computational cost. However, increasing the learning rate or reducing batch size causes the variance of policy gradient estimates to surge dramatically, leading to catastrophic parameter updates and policy collapse—model performance drops below baseline and fails to recover.

**Key Challenge**: Policy gradients rely solely on first-order information and cannot sense curvature on non-convex RL objectives, potentially taking a large step along an apparently improving direction only to fall into a performance cliff. Meanwhile, the Hessian matrix is infeasible to compute or approximate directly at LLM scale (billions of parameters).

**Key Insight**: The authors observe that LLM logit outputs are produced solely by a final linear transformation $W \in \mathbb{R}^{K \times d_i}$, and that top-k sampling renders gradients naturally sparse (only $k < 100$ tokens have non-zero probability). This enables efficient approximation of the Hessian and Fisher information matrix within the final layer.

**Core Idea**: A last-layer curvature model is constructed to track the effect of each token update on the objective function and policy distribution, filtering out samples that violate trust-region constraints.

## Method

### Overall Architecture
CAPO augments GRPO with a lightweight data-filtering layer: before each gradient update, the curvature model predicts whether the update is safe, and only safe tokens participate in the actual gradient computation. The pipeline is: sample trajectories → compute last-layer gradients/curvature → evaluate token-level constraints → filter unsafe tokens → compute LLM policy gradient on filtered subset → update model parameters.

### Key Designs

1. **Last-Layer Curvature Model**:

    - LLM parameters are partitioned as $\bm{\theta} = (\bar{\bm{\theta}}, \bm{\psi})$, where $\bm{\psi} = \text{vec}(W)$ denotes only the LM head.
    - Closed-form expressions for the objective Hessian $\tilde{H}(\bm{\psi})$ and Fisher information matrix $\tilde{F}(\bm{\psi})$ are derived in this subspace.
    - Exploiting the sparsity of top-k sampling (typically $k < 100$), memory complexity is reduced from $\mathcal{O}((Kd_i)^2)$ to $\mathcal{O}(\tilde{k} \cdot d_i)$.

2. **Directional Curvature Computation**:

    - Objective shift: $m_H(\Delta\bm{\psi}) = \tilde{g}(\bm{\psi})^\top \Delta\bm{\psi} + \frac{1}{2} \Delta\bm{\psi}^\top \tilde{H}(\bm{\psi}) \Delta\bm{\psi}$
    - Policy shift: $m_F(\Delta\bm{\psi}) = \frac{1}{2} \Delta\bm{\psi}^\top \tilde{F}(\bm{\psi}) \Delta\bm{\psi}$
    - No large tensors need to be constructed; only sparse vector dot products are required.

3. **Token-Level Trust-Region Filtering**:

    - The batch is divided into token-level subsets; $m_H$ and $m_F$ are computed for each subset.
    - Acceptance condition: $\delta_H \leq m_H(\Delta\psi_i) \leq \delta_H^{high}$ and $m_F(\Delta\psi_i) \leq \delta_F$.
    - Rejected tokens do not participate in the actual policy gradient computation.

4. **Optimizer Modeling**: Adam's first- and second-moment estimates are used to simulate the actual update step $\Delta\bm{\psi}$, rather than a simple SGD step.

### Loss & Training
The same clipped surrogate objective and KL regularization as GRPO are used. CAPO's contribution lies in the data-filtering stage of gradient estimation, leaving the objective function itself unchanged. A monotonic improvement guarantee is theoretically established: choosing $\omega \geq C\sqrt{\delta_F}$ is sufficient to ensure $J(\pi_{\theta+\Delta\theta}) \geq J(\pi_\theta)$.

## Key Experimental Results

### Main Results (Qwen2.5-Math-7B, trained on MATH dataset)

| Method | Setting | MATH Accuracy | TEST-8 Avg. | Completions to Reach 70% |
|--------|---------|--------------|-------------|--------------------------|
| GRPO (conservative) | lr=3e-6, large batch | ~72% | ~65% | ~150K |
| GRPO (aggressive) | lr=1.5e-5, small batch | Collapse ❌ | Collapse ❌ | N/A |
| Dr.GRPO (aggressive) | same | Collapse ❌ | Collapse ❌ | N/A |
| REINFORCE (aggressive) | same | Collapse ❌ | Collapse ❌ | N/A |
| **CAPO (aggressive)** | **lr=1.5e-5, small batch** | **~72%** | **~66%** | **~5K (30×)** |

### Ablation Study

| Analysis Dimension | Result | Notes |
|-------------------|--------|-------|
| Token rejection rate | ~8% early, <2% thereafter | Minimal intervention suffices for stability |
| Generalizability | Dr.CAPO and ReinCAPO both effective | Compatible with arbitrary PG methods |
| Computational overhead | <5% additional time | Last-layer computation is extremely lightweight |
| $m_F$ tracking | Global $m_F$ spikes sharply for collapsing methods | Curvature model effectively warns of instability |
| $m_H$ tracking | $m_H$ curve is smooth for CAPO | Local constraints ensure global stability |

### Key Findings
- All baseline methods (GRPO, Dr.GRPO, REINFORCE) collapse under aggressive settings; only CAPO remains stable.
- CAPO achieves a 30× sample efficiency improvement on MATH and a 9× improvement on TEST (8 evaluation benchmarks).
- Policy shift $m_F$ is highly correlated with training instability—a spike in $m_F$ is a precursor to collapse.
- Curvature-aware filtering generalizes to different policy gradient objectives (Dr.GRPO→Dr.CAPO, REINFORCE→ReinCAPO), consistently preventing collapse.

## Highlights & Insights
- **Minimal intervention, maximal gain**: Computing curvature only at the final layer and rejecting fewer than 8% of tokens yields a 30× sample efficiency improvement. This suggests that training instability is concentrated in a small number of "toxic" samples, while the update directions of the vast majority of tokens are safe.
- **Diagnostic value of the curvature model**: Tracking $m_F$ and $m_H$ not only serves filtering purposes but also provides a window into understanding the optimization dynamics of RL-LLM training—spikes in $m_F$ precede collapse, a phenomenon that was previously nearly opaque.
- **Strong generalizability**: CAPO's token-filtering mechanism can be stacked on top of any policy gradient method, and the theoretical guarantees do not depend on the specific form of the advantage function—as validated by the success of Dr.CAPO and ReinCAPO.
- **Close correspondence between theory and practice**: The monotonic improvement guarantee of Theorem 5.1 is precisely verified experimentally—CAPO's $m_F$ remains consistently below the threshold.

## Limitations & Future Work
- Validation is limited to Qwen2.5-Math-7B (7B scale); larger models and longer training schedules remain to be tested.
- Thresholds $\delta_H, \delta_F, \delta_H^{high}$ require tuning for specific MDPs and base policies.
- The last-layer approximation may be informationally insufficient for deeper layers; extending to multi-layer curvature estimation is a natural direction.
- Evaluation is restricted to mathematical reasoning tasks; performance on code generation, agent tasks, and other domains is unknown.

## Related Work & Insights
- **vs. GRPO/PPO**: Their clipping mechanisms impose coarse-grained constraints in parameter space, whereas CAPO performs fine-grained filtering in sample space—the two approaches can be used in combination.
- **vs. K-FAC (Castanyer et al.)**: K-FAC computes a Kronecker approximation of the Fisher matrix in general deep RL, incurring high memory overhead that does not scale to LLM sizes; CAPO leverages LLM top-k sparsity and a last-layer approximation to substantially reduce complexity.
- **vs. Dr.GRPO**: Dr.GRPO reduces variance by modifying the advantage function but does not address curvature—it still collapses under aggressive settings; combined with CAPO (Dr.CAPO), it becomes stable.
- **Insight**: Instability in RL-LLM training may not require modifying the objective function; it may suffice to identify and filter "toxic samples"—analogous to data cleaning in machine learning.
- **Broader Insight**: The last-layer curvature modeling approach can be extended to other LLM optimization problems, such as detecting catastrophic forgetting in SFT.

## Rating
- Novelty: ⭐⭐⭐⭐ Practical application of second-order methods in LLM RL; the last-layer + sparsity technique is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ The 30× improvement is substantial and evaluated across 8 benchmarks, though model scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, experimental presentation is clear, and the logical chain from motivation to method is complete.
- Value: ⭐⭐⭐⭐⭐ Directly practical for LLM RL training efficiency, with potential to substantially reduce computational costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](../../ICML2026/llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)

</div>

<!-- RELATED:END -->
