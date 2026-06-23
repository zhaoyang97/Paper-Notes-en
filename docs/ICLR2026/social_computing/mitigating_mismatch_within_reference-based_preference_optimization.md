---
title: >-
  [Paper Note] Mitigating Mismatch within Reference-based Preference Optimization
description: >-
  [ICLR 2026][Social Computing][DPO] Reveals the "premature satisfaction" issue in DPO—where the gradient is unnecessarily decayed by pessimistic signals from the reference policy when it assigns a lower probability to the chosen response than the rejected one (~45% of pairs), even if the policy remains incorrect ($\Delta_\theta < 0$). Proposes HyPO (a on
tags:
  - ICLR 2026
  - Social Computing
  - DPO
  - reference policy
  - pessimistic bias
  - preference optimization
  - HyPO
  - premature satisfaction
date: 2026-05-08
content_hash: 3b23a7d7daa176a6
---
# Mitigating Mismatch within Reference-based Preference Optimization

**Conference**: ICLR 2026  
**arXiv**: [2602.11902](https://arxiv.org/abs/2602.11902)  
**Code**: None  
**Area**: LLM Alignment / Preference Optimization  
**Keywords**: DPO, reference policy, pessimistic bias, preference optimization, HyPO, premature satisfaction

## TL;DR
Reveals the "premature satisfaction" issue in DPO—where the gradient is unnecessarily decayed by pessimistic signals from the reference policy when it assigns a lower probability to the chosen response than the rejected one (~45% of pairs), even if the policy remains incorrect ($\Delta_\theta < 0$). Proposes HyPO (a one-line change: $\max(0, \Delta_{ref})$ to clip the reference margin), achieving a 41.2% relative improvement over DPO on AlpacaEval 2.0.

## Background & Motivation
**Background**: DPO optimizes preferences via a relative margin $\Delta_\theta - \Delta_{ref}$, where $\Delta_{ref}$ represents the log-probability difference between chosen and rejected responses from a reference policy. This serves as a proximal constraint for KL regularization, stabilizing training.

**Limitations of Prior Work**:
   - **Training-Inference Mismatch**: DPO optimizes the relative margin $\Delta_\theta - \Delta_{ref}$ during training, yet inference relies solely on the absolute margin $\Delta_\theta$. Research indicates that the consistency between implicit reward ranking and likelihood ranking after DPO training is only ~50%.
   - **Two Opposing Directions**: (a) Reference-free methods (SimPO, ORPO) eliminate the mismatch by removing the reference but sacrifice stability; (b) Stronger reference methods (TR-DPO) reduce pessimistic instances but cannot eliminate them entirely.
   - **Pessimistic Reference Problem**: Even when utilizing the strongest references (e.g., SimPO-aligned models), ~45% of pairs exhibit $\Delta_{ref} < 0$ (where the reference prefers the rejected response), representing an unavoidable structural upper bound.

**Key Challenge**: The reference policy provides stability but introduces mismatch; removing it eliminates mismatch but loses stability. Can both benefits be achieved simultaneously?

**Key Insight**: Utilize the reference policy conditionally—use it normally when it is optimistic ($\Delta_{ref} \geq 0$) to provide stability, and treat it as neutral (degrading to an absolute margin) when it is pessimistic ($\Delta_{ref} < 0$).

## Method

### Overall Architecture
DPO relies on a reference policy to stabilize training for each preference pair, but this dependency introduces a hidden flaw: when the reference is "pessimistic" about a pair (deeming the rejected response better than the chosen one), the stabilizing force prematurely extinguished the gradient that should have continued optimization. This paper formalizes this as "premature satisfaction" by locating the decay mechanism in the DPO gradient weight. The proposed intervention is a minimal "surgery": replacing the reference term $\Delta_{ref}$ in the DPO loss with a bottom-clipped version $\max(0, \Delta_{ref})$. Consequently, for optimistic pairs ($\Delta_{ref} \ge 0$), the loss remains identical to DPO, preserving stability signals. For pessimistic pairs ($\Delta_{ref} < 0$), the reference term is clipped to 0, and the loss degrades to a reference-free form focusing only on the absolute margin. This modification requires no changes to network architecture or forward passes and shares the same computational cost as DPO, allowing for drop-in replacement.

### Key Designs

**1. Premature Satisfaction: Formalizing the cessation of learning in DPO on pessimistic pairs**

This explains a counter-intuitive phenomenon where the consistency between implicit reward ranking and likelihood ranking is only ~50% after DPO training. The authors analyze the gradient weight: $w_{DPO}=\sigma(-\beta(\Delta_\theta-\Delta_{ref}))$, where a smaller weight indicates the model "assumes" it has learned sufficiently. The issue arises in pessimistic pairs ($\Delta_{ref}<0$): even if the policy is incorrect (absolute margin $\Delta_\theta<0$), as long as it is "less wrong" than the reference ($\Delta_\theta>\Delta_{ref}$), the relative margin turns positive, causing $w_{DPO}$ to decay exponentially. For instance, if $\Delta_{ref}=-3$ and $\Delta_\theta=-1$, the relative margin is 2 and $w_{DPO}\approx 0.119$, leaving only 12% of the gradient—the model "satisfies" prematurely. Crucially, this cannot be solved by simply using a stronger reference, as even SimPO-aligned references still result in ~45% pessimistic pairs.

**2. HyPO Objective Function: Making the reference term conditional via a max operation**

After diagnosing the pessimistic reference, the fix is straightforward: clip the reference margin from below. Defining the clipped $\widetilde{\Delta}_{ref}=\max(\Delta_{ref},\gamma)$ (with a default threshold $\gamma=0$), the loss is:

$$\mathcal{L}_{HyPO}=\mathbb{E}\big[\log(1+\exp(-\beta(\Delta_\theta-\widetilde{\Delta}_{ref})))\big]$$

This operation allows the loss to switch behaviors automatically. In optimistic pairs, $\widetilde{\Delta}_{ref}=\Delta_{ref}$, preserving DPO behavior and stability. In pessimistic pairs, $\widetilde{\Delta}_{ref}=0$, removing the misleading signal. A softplus version $\widetilde{\Delta}_{ref}=\gamma+\frac{1}{\alpha}\log(1+\exp(\alpha(\Delta_{ref}-\gamma)))$ is also provided for smoothness.

**3. Gradient Weight Lower Bound: Proving the modification is "no-loss, more aggressive where needed"**

The authors characterize the relationship between the HyPO gradient weight $w_{HyPO}$, the DPO weight $w_{DPO}$, and the reference-free weight $w_{abs}=\sigma(-\beta\Delta_\theta)$. They conclude: $w_{HyPO}\ge w_{abs}$ across all pairs; $w_{HyPO}=w_{DPO}$ for non-pessimistic pairs; and $w_{HyPO}=w_{abs}$ for pessimistic pairs, releasing the decayed gradient. HyPO effectively combines the strengths of both DPO and reference-free methods.

### Loss & Training
The total loss is $\mathcal{L}=\mathcal{L}_{HyPO}$, replacing the DPO loss directly without extra terms. $\beta$ remains consistent with DPO, and the added threshold defaults to $\gamma=0$. The computational cost is identical to DPO. HyPO is orthogonal to other improvements like SquaredPO or TR-DPO.

## Key Experimental Results

### Main Results

| Method | AlpacaEval 2.0 LC↑ | Arena-Hard ↑ | Win Rate vs DPO |
|:-------|:-------------------|:-------------|:----------------|
| DPO (Llama-3-8B) | 22.6% | 7.9% | — |
| SimPO (reference-free) | ~24% | ~9% | — |
| **HyPO** | **27.3%** | **11.2%** | **55.9%** |
| Relative Gain | **+41.2%** | **+41.8%** | — |

### Training Dynamics Analysis

| Metric | DPO | HyPO | Description |
|:-------|:----|:-----|:------------|
| Absolute Agreement Rate | ~50% → ~55% | ~50% → **~62%** | Consistency between absolute ranking and preference |
| Pessimistic Subset Absolute Margin | Low, stagnant | Continual growth | Verifies the fix for premature satisfaction |

### Ablation Study

| Configuration | Effect | Description |
|:--------------|:-------|:------------|
| DPO + Optimized Reference (SimPO-aligned) | Limited improvement | Still contains ~45% pessimistic pairs |
| Reference-free (SimPO) | Better than DPO | But loses stability |
| HyPO ($\gamma=0$) | **Optimal** | Best balance for conditional reference |
| HyPO + softplus | Close to hard max | Optional smooth version |

### Key Findings
- ~45% of preference pairs are pessimistic for all reference models, representing a structural issue that "stronger references" cannot fully resolve.
- HyPO's absolute margin on the pessimistic subset continues to grow while DPO's stagnates, validating the fix for "premature satisfaction."
- HyPO maintains its advantage when scaled to larger models and different datasets.
- Performance on downstream tasks (e.g., MT-Bench) increases, indicating that clipping does not harm general capabilities.

## Highlights & Insights
- **Profound Improvement via One Line of Code**: The $\max(0, \Delta_{ref})$ modification is supported by robust theoretical motivation and empirical validation. The formalization of "premature satisfaction" is a significant contribution to explaining common community observations.
- **Unifying Opposing Directions**: Rather than a binary choice of whether to use a reference, HyPO introduces a conditional strategy, offering deeper insight than prior work.
- **Orthogonality**: Since HyPO only modifies the reference margin processing, it can be combined freely with other advancements like SquaredPO or TR-DPO.

## Limitations & Future Work
- The theoretical analysis is primarily intuitive (gradient weight decay); no formal proof of convergence or optimality is provided.
- The fixed threshold $\gamma=0$ may not be optimal for all scenarios, as some weakly pessimistic pairs might still benefit from reference signals.
- Validated only in off-policy settings; similar issues in on-policy RLHF (e.g., PPO) remain unexplored.
- Primary experiments focused on Llama/Mistral; effects on larger models (70B+) are not yet fully verified.

## Related Work & Insights
- **vs SimPO / ORPO (reference-free)**: Removing the reference entirely loses stability; HyPO retains it conditionally.
- **vs TR-DPO (dynamic reference update)**: Reduces but does not eliminate pessimistic pairs; HyPO addresses them directly.
- **vs SquaredPO**: Addresses different issues (probability shifting vs pessimistic references); these are complementary.
- **vs RainbowPO**: Mixes reference and constant margins; HyPO is simpler with only a max operation.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery and formalization of "premature satisfaction" are insightful; the "conditional reference" approach unifies two opposing directions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model and multi-benchmark coverage plus training dynamics and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical progression from problem analysis to formalization to the proposed change.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to improving DPO practices, with a 41% gain from a one-line change.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ICLR 2026\] Measuring and Mitigating Rapport Bias of Large Language Models under Multi-Agent Social Interactions](measuring_and_mitigating_rapport_bias_of_large_language_models_under_multi-agent.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)

</div>

<!-- RELATED:END -->
