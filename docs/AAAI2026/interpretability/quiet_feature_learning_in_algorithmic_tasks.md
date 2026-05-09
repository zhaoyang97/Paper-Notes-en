---
title: >-
  [Paper Note] Quiet Feature Learning in Algorithmic Tasks
description: >-
  [AAAI 2026][Phase transition] Across 10 algorithmic tasks (18,544 training runs, $10^9$–$10^{16}$ FLOPs), this work demonstrates that loss plateaus in Transformer training do not indicate stalled learning. During these plateaus, models quietly acquire "quiet features"—intermediate algorithmic subroutines that do not directly reduce output loss yet are causally necessary for final performance (ablating them reduces accuracy by 41–75%). This challenges the common practice of using loss curves to assess training progress.
tags:
  - AAAI 2026
  - Phase transition
  - implicit features
  - algorithmic tasks
  - loss plateau
  - Grokking
date: 2026-05-08
content_hash: dcd14a6eddf55f11
---

# Quiet Feature Learning in Algorithmic Tasks

**Conference**: AAAI 2026
**arXiv**: [2505.03997](https://arxiv.org/abs/2505.03997)
**Code**: [https://github.com/prudhvirajn/quiet-feature-learning-in-algorithmic-tasks](https://github.com/prudhvirajn/quiet-feature-learning-in-algorithmic-tasks)
**Area**: Deep Learning Theory / Emergence
**Keywords**: Phase transition, implicit features, algorithmic tasks, loss plateau, Grokking

## TL;DR
Across 10 algorithmic tasks (18,544 training runs, $10^9$–$10^{16}$ FLOPs), this work demonstrates that loss plateaus in Transformer training do not indicate stalled learning. During these plateaus, models quietly acquire "quiet features"—intermediate algorithmic subroutines that do not directly reduce output loss yet are causally necessary for final performance (ablating them reduces accuracy by 41–75%). This challenges the common practice of using loss curves to assess training progress.

## Background & Motivation

### State of the Field

Phase transitions (sudden drops) and extended plateaus in loss curves have been widely observed in LLM training. Scaling laws assume smooth loss reduction, but phase transitions on algorithmic tasks violate this assumption.

### Root Cause

**Key Challenge**: The plateau preceding a phase transition is assumed to represent "wasted computation"—but does the model actually learn nothing during this period?

### Limitations of Prior Work

**Limitations of Prior Work**: There is a lack of tools for probing what occurs inside the model during a loss plateau.

### Starting Point

Discussions of emergent abilities lack mechanistic evidence from internal model representations.

**Key Challenge**: Constant loss $\neq$ constant learning—models may be accumulating intermediate representations not directly reflected in the loss.

**Goal**: To demonstrate, via linear probing and ablation experiments, that substantive internal learning occurs during loss plateaus.

**Key Insight**: Linear probes are trained on the residual stream of 10 algorithmic tasks to detect "quiet features"—intermediate computational results learned before the loss drop occurs.

**Core Idea**: A loss plateau $\neq$ learning stagnation—models concurrently accumulate subroutines in the background, and the loss only drops once all subroutines are in place.

## Method

### Overall Architecture
10 algorithmic tasks (addition / multiplication / sorting / search, etc.) × Transformer++ architecture × multiple model scales → linear probes trained on the residual stream at each layer and position throughout training → detection of when intermediate computational features emerge → causal necessity validated via ablation.

### Key Designs

1. **Quiet Feature Probing**:

    - Function: Detect whether the model has already learned intermediate algorithmic steps during the loss plateau.
    - Mechanism: For the addition task, probe whether carry bits are encoded in the residual stream—even when the model cannot yet produce correct final outputs.
    - Finding: Carry bits are encoded before the loss drops (probe accuracy >90%), while the output loss remains on the plateau.

2. **Causal Ablation**:

    - Function: Demonstrate that quiet features are causally necessary for final performance.
    - Mechanism: After the model has learned the features, ablate (zero out) the representational dimensions encoding quiet features and observe performance degradation.
    - Result: Ablating carry bits causes accuracy to drop by 41.2%–75.1% ($p<0.001$), confirming these features are not redundant.

3. **Cross-Task Generality**:

    - Function: Verify that quiet features are not specific to the addition task.
    - A similar pattern is observed across all 10 tasks: feature learning occurs during the loss plateau, and the learned features are subsequently exploited after the phase transition.

### Loss & Training
- Standard cross-entropy loss
- AdamW + linear warmup + cosine annealing
- A total of 18,544 training runs

## Key Experimental Results

### Main Results

| Quiet Feature | Probe Accuracy (Plateau) | Accuracy Drop After Ablation |
|---|---|---|
| Addition carry bits | >90% | −41.2% ~ −75.1% |
| Sorting intermediate comparisons | >85% | Significant |
| Search intermediate pointers | >80% | Significant |

### Ablation Study: Phase Transition Timeline

| Stage | Output Loss | Quiet Feature Probe | Notes |
|---|---|---|---|
| Early plateau | High, flat | Random | Not yet learned |
| Late plateau | High, flat | **>90%** | Quiet learning |
| Post-transition | Sudden drop | >95% | Features utilized |

### Key Findings
- **Quiet features emerge before the loss drops**—models accumulate subroutines during the plateau.
- **Causal necessity**: The correlation is not incidental; ablation causes performance collapse.
- **The pattern holds across all 10 tasks**: strong generality.
- **Challenges scaling laws**: Power-law loss curves cannot predict the timing of phase transitions.

## Highlights & Insights
- The claim **"loss plateau $\neq$ learning stagnation"** challenges the standard practice of using loss curves for decisions such as early stopping and compute budget allocation.
- **A mechanistic explanation for emergent abilities**: capabilities do not appear abruptly; rather, subroutines accumulate in the background until a critical threshold is reached.
- Direct practical implication: training should not be halted during a loss plateau, as the model may be engaged in significant internal learning.

## Limitations & Future Work
- Validation is limited to algorithmic tasks; whether analogous quiet features exist in natural language tasks remains unknown.
- Linear probes may fail to capture features encoded nonlinearly.
- Experiments are conducted only on small models (<100M parameters).

## Related Work & Insights
- **vs. Grokking (Power et al.)**: Grokking refers to the sudden emergence of generalization after training. This paper analyzes the internal mechanisms preceding Grokking.
- **vs. Emergent Abilities (Wei et al.)**: That work provides macro-level observations of emergent abilities. This paper supplies micro-level mechanistic evidence.
- **vs. Lottery Ticket Hypothesis**: The lottery ticket hypothesis concerns the existence of sparse subnetworks, whereas this paper focuses on the dynamic trajectory of feature formation during training—the two perspectives are complementary.
- Implication for training monitoring: internal representation changes should be tracked rather than relying solely on the loss curve.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of "quiet features" is original and profound, revealing hidden dynamics of feature learning in the early stages of training.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10 tasks × 18K runs × causal ablation; the statistical design is rigorous.
- Writing Quality: ⭐⭐⭐⭐⭐ The findings are compelling, and the logical chain from observation to mechanistic explanation is well-constructed.
- Value: ⭐⭐⭐⭐⭐ Makes an important theoretical contribution to understanding deep learning training dynamics and challenges conventional interpretations of Grokking.

## Additional Notes
- The methodology and experimental design of this work offer reference value for related research areas.
- Future work may validate the generalizability and scalability of the approach across broader settings and larger scales.
- Integration with recent related work (e.g., intersections with RL/MCTS/multimodal methods) presents potential research opportunities.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Unsupervised Feature Selection Through Group Discovery](unsupervised_feature_selection_through_group_discovery.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[NeurIPS 2025\] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](../../NeurIPS2025/interpretability/tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)

<!-- RELATED:END -->
