---
title: >-
  [Paper Note] Mitigating Mismatch within Reference-based Preference Optimization
description: >-
  [ICLR 2026][LLM Alignment][DPO] This paper identifies the *premature satisfaction* problem in DPO — when the reference policy assigns lower probability to chosen than to rejected responses (~45% of pairs), DPO's gradient is unnecessarily attenuated by the pessimistic reference signal, even when the policy is still incorrect (i.e., $\Delta_\theta < 0$). The paper proposes HyPO (a one-line code change: clipping the reference margin via $\max(0, \Delta_{ref})$), achieving a 41.2% relative improvement over DPO on AlpacaEval 2.0.
tags:
  - ICLR 2026
  - LLM Alignment
  - DPO
  - reference policy
  - pessimistic bias
  - preference optimization
  - HyPO
  - premature satisfaction
date: 2026-05-08
content_hash: 682591a41aedab6c
---

# Mitigating Mismatch within Reference-based Preference Optimization

**Conference**: ICLR 2026
**arXiv**: [2602.11902](https://arxiv.org/abs/2602.11902)
**Code**: None
**Area**: LLM Alignment / Preference Optimization
**Keywords**: DPO, reference policy, pessimistic bias, preference optimization, HyPO, premature satisfaction

## TL;DR
This paper identifies the *premature satisfaction* problem in DPO — when the reference policy assigns lower probability to chosen than to rejected responses (~45% of pairs), DPO's gradient is unnecessarily attenuated by the pessimistic reference signal, even when the policy is still incorrect (i.e., $\Delta_\theta < 0$). The paper proposes HyPO (a one-line code change: clipping the reference margin via $\max(0, \Delta_{ref})$), achieving a 41.2% relative improvement over DPO on AlpacaEval 2.0.

## Background & Motivation
**State of the Field**: DPO optimizes preferences via the relative margin $\Delta_\theta - \Delta_{ref}$, where $\Delta_{ref}$ is the log-probability difference between chosen and rejected responses under the reference policy. This implements a proximal constraint for KL regularization, stabilizing training.

**Limitations of Prior Work**:
- **Train–inference mismatch**: DPO training optimizes the relative margin $\Delta_\theta - \Delta_{ref}$, whereas inference relies solely on the absolute margin $\Delta_\theta$. Studies show that after DPO training, the agreement rate between implicit reward ranking and likelihood ranking is only ~50%.
- **Two opposing remedies**: (a) Reference-free methods (SimPO, ORPO) remove the reference to resolve mismatch but sacrifice stability signals; (b) Stronger-reference methods (TR-DPO) reduce pessimistic cases but cannot eliminate them.
- **Pessimistic reference problem**: Even with the strongest reference (e.g., a SimPO-aligned model), ~45% of pairs still exhibit $\Delta_{ref} < 0$ (i.e., the reference favors rejected over chosen), representing an unavoidable structural upper bound.

**Root Cause**: The reference provides stability but introduces mismatch; removing the reference eliminates mismatch but sacrifices stability. Must these two objectives be mutually exclusive?

**Core Idea**: Conditionally use the reference — apply it normally when it is optimistic ($\Delta_{ref} \geq 0$, preserving stability), and treat it as neutral when it is pessimistic ($\Delta_{ref} < 0$, falling back to the absolute margin) — thus achieving the best of both worlds.

## Method

### Overall Architecture
Replace $\Delta_\theta - \Delta_{ref}$ in the DPO loss with $\Delta_\theta - \max(0, \Delta_{ref})$ → behaves identically to DPO on optimistic pairs and as a reference-free method on pessimistic pairs → preserves the DPO loss form and computational cost.

### Key Designs

1. **Formalization of Premature Satisfaction**

    - **Function**: Characterizes the systematic failure mode of DPO on pessimistic pairs.
    - **Core Analysis**: The gradient weight of DPO is $w_{DPO} = \sigma(-\beta(\Delta_\theta - \Delta_{ref}))$. When $\Delta_{ref} < 0$, even if $\Delta_\theta < 0$ (i.e., the policy is still incorrect), as long as $\Delta_\theta > \Delta_{ref}$ (the policy is "less wrong" than the reference), $w_{DPO}$ decays rapidly. For example, with $\Delta_{ref}=-3$ and $\Delta_\theta=-1$: the relative margin equals 2, yielding $w_{DPO} \approx 0.119$, attenuating the gradient to ~12%.
    - **Design Motivation**: This provides a principled explanation for the long-standing empirical observation that implicit reward rankings after DPO training have low agreement with likelihood rankings.

2. **HyPO Objective**

    - **Function**: Conditionally clips the reference margin.
    - **Formulation**: $\widetilde{\Delta}_{ref} = \max(\Delta_{ref}, \gamma)$ (default $\gamma=0$), with loss $\mathcal{L}_{HyPO} = \mathbb{E}[\log(1 + \exp(-\beta(\Delta_\theta - \widetilde{\Delta}_{ref})))]$.
    - **Behavioral Analysis**:
        - **Optimistic pairs** ($\Delta_{ref} \geq 0$): $\widetilde{\Delta}_{ref} = \Delta_{ref}$, equivalent to DPO, preserving the proximal constraint and training stability.
        - **Pessimistic pairs** ($\Delta_{ref} < 0$): $\widetilde{\Delta}_{ref} = 0$, reducing to an absolute margin update $\sigma(-\beta\Delta_\theta)$, eliminating interference from the pessimistic reference.
    - **Smooth variant**: The hard max can be replaced by a softplus: $\widetilde{\Delta}_{ref} = \gamma + \frac{1}{\alpha}\log(1+\exp(\alpha(\Delta_{ref}-\gamma)))$.
    - **Implementation**: A single-line code change — replace $\Delta_{ref}$ with $\max(0, \Delta_{ref})$.

3. **Theoretical Properties**

    - The gradient weight $w_{HyPO}$ satisfies $w_{HyPO} \geq w_{abs}$ (reference-free weight) for all pairs.
    - On non-pessimistic pairs, $w_{HyPO} = w_{DPO}$ (DPO behavior fully preserved).
    - On pessimistic pairs, $w_{HyPO} = w_{abs}$ (pessimistic bias fully eliminated).

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{HyPO}$ (direct replacement of the DPO loss, no additional terms).
- Hyperparameters: same $\beta$ as DPO; default $\gamma=0$.
- Computational cost: identical to DPO (one additional max operation).
- Orthogonally composable with other improvements (e.g., SquaredPO for probability shift, stronger reference models).

## Key Experimental Results

### Main Results

| Method | AlpacaEval 2.0 LC↑ | Arena-Hard↑ | Win Rate vs. DPO |
|--------|---------------------|-------------|------------------|
| DPO (Llama-3-8B) | 22.6% | 7.9% | — |
| SimPO (reference-free) | ~24% | ~9% | — |
| **HyPO** | **27.3%** | **11.2%** | **55.9%** |
| Relative Gain | **+41.2%** | **+41.8%** | — |

### Training Dynamics Analysis

| Metric | DPO | HyPO | Note |
|--------|-----|------|------|
| Absolute Agreement Rate | ~50% → ~55% | ~50% → **~62%** | Agreement between absolute ranking and preference |
| Absolute Margin on Pessimistic Subset | Low, stagnant | Continuously increasing | Directly confirms that premature satisfaction is resolved |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| DPO + stronger reference (SimPO-aligned) | Improved but limited | ~45% pessimistic pairs remain |
| Reference-free (SimPO) | Better than DPO | But sacrifices stability |
| HyPO ($\gamma=0$) | **Best** | Optimal balance of conditional reference usage |
| HyPO + softplus | Close to hard max | Optional smooth variant |

### Key Findings
- ~45% of preference pairs are pessimistic across all reference models — a structural issue that cannot be fully resolved by "using a stronger reference."
- The absolute margin on pessimistic pairs continues to increase under HyPO (while stagnating under DPO), directly validating the fix for premature satisfaction.
- HyPO maintains its advantage when scaled to larger models and different datasets.
- Performance on downstream tasks (e.g., MT-Bench) does not degrade, indicating that clipping does not harm general capabilities.

## Highlights & Insights
- **A profound improvement from a one-line change**: The minimalist modification $\max(0, \Delta_{ref})$ is grounded in a complete theoretical motivation and extensive empirical validation. The formalization of *premature satisfaction* is the most valuable contribution — it precisely explains a phenomenon that has long puzzled the community.
- **Unifying two opposing directions**: Rather than a binary choice of "use or discard the reference," HyPO adopts a conditional strategy of "when to use the reference." This perspective is more insightful than prior work.
- **Orthogonal to other improvements**: HyPO only modifies the handling of the reference margin and can be freely combined with other methods such as SquaredPO and TR-DPO.

## Limitations & Future Work
- The theoretical analysis is primarily intuitive (gradient weight attenuation analysis); no formal proof of convergence or optimality is provided.
- The threshold $\gamma=0$ is fixed and may not be optimal across all scenarios (some weakly pessimistic pairs may still benefit from the reference signal).
- Validation is limited to the off-policy setting; analogous issues in on-policy RLHF (e.g., PPO) remain unexplored.
- Experiments are conducted primarily on Llama/Mistral; performance on larger models (70B+) is not verified.

## Related Work & Insights
- **vs. SimPO / ORPO (reference-free)**: Completely removing the reference sacrifices stability; HyPO conditionally retains it, yielding superior results.
- **vs. TR-DPO (dynamic reference update)**: Reduces pessimistic pairs but does not eliminate them; HyPO directly addresses pessimistic pairs.
- **vs. SquaredPO**: Targets a different problem (probability shift vs. pessimistic reference); the two approaches are complementary and composable.
- **vs. RainbowPO**: Blends reference and constant margin; HyPO is simpler (a single max operation).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The discovery and formalization of *premature satisfaction* is substantive; the conditional reference perspective unifies two opposing directions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model, multi-benchmark evaluation with training dynamics analysis, ablation studies, and comparisons against existing methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from problem analysis → formalization → one-line fix is exceptionally clear, with intuitive figures.
- **Value**: ⭐⭐⭐⭐⭐ Offers direct practical improvement for DPO practitioners; a one-line code change yields a 41% performance gain.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](../../NeurIPS2025/llm_alignment/mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[CVPR 2026\] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](../../CVPR2026/llm_alignment/mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)

<!-- RELATED:END -->
