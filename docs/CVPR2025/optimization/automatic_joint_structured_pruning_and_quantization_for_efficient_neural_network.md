---
title: >-
  [Paper Note] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression
description: >-
  [CVPR 2025][Optimization][structured pruning] Proposed the GETA framework to achieve automatic joint structured pruning and quantization-aware training: Quantization-Aware Dependency Graph (QADG) constructs a generic pruning search space + partially projected SGD guarantees layer-wise bit-width constraints + an interpretable joint learning strategy, achieving competitive or state-of-the-art compression performance on both CNNs and Transformers.
tags:
  - "CVPR 2025"
  - "Optimization"
  - "structured pruning"
  - "quantization-aware training"
  - "joint optimization"
  - "dependency graph"
  - "QADG"
date: 2026-05-08
content_hash: ec9795fc780fd7ef
---

# Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression

**Conference**: CVPR 2025  
**arXiv**: [2502.16638](https://arxiv.org/abs/2502.16638)  
**Code**: —  
**Area**: Optimization / Model Compression  
**Keywords**: structured pruning, quantization-aware training, joint optimization, dependency graph, QADG

## TL;DR
Proposed the GETA framework to achieve automatic joint structured pruning and quantization-aware training: Quantization-Aware Dependency Graph (QADG) constructs a generic pruning search space + partially projected SGD guarantees layer-wise bit-width constraints + an interpretable joint learning strategy, achieving competitive or state-of-the-art compression performance on both CNNs and Transformers.

## Background & Motivation

**Background**: Structured pruning and quantization are two fundamental DNN compression techniques, usually applied independently. Co-optimization has the potential to yield smaller, higher-quality models.

**Limitations of Prior Work**:
   - **Engineering Difficulty**: Existing joint schemes have complex workflows involving multiple stages (such as pruning followed by quantization, alternating optimization, etc.).
   - **Black-box Optimization**: A large amount of hyperparameter tuning is required to control the overall compression rate (such as searching for the pruning rate and bit-width of each layer).
   - **Insufficient Architectural Generalization**: Most methods are only applicable to specific network architectures (e.g., CNNs only) and cannot automatically handle arbitrary DNNs.

**Key Challenge**: Pruning changes the network structure (number of channels) while quantization changes numerical precision (bit-width). The interaction between them is complex, and independently optimizing their respective hyperparameters is already NP-hard, making joint optimization even more challenging.

**Key Insight**:
   - Use QADG to automatically construct the pruning search space for arbitrary quantization-aware networks.
   - Use partially projected SGD to transform discrete bit-width constraints into a continuous optimization problem.
   - Replace black-box search with white-box optimization.

**Core Idea**: QADG unified search space + projected SGD constraint satisfaction + interpretable pruning-quantization relationship = one-click joint compression.

## Method

### Overall Architecture
GETA takes an arbitrary DNN and a target compression rate as input, and outputs the jointly pruned and quantized model:
1. QADG analyzes the network structure and constructs the pruning search space.
2. Jointly optimize the pruning rate and bit-width of each layer within the search space.
3. Partially projected SGD ensures constraint satisfaction.
4. One-shot training, without post-processing.

### Key Designs

1. **Quantization-Aware Dependency Graph (QADG)**

    - **Function**: Automatically constructs the structured pruning search space for arbitrary quantization-aware DNNs.
    - **Mechanism**:
        - Extends traditional dependency graphs to consider quantization operations (such as fake quantization nodes).
        - Automatically identifies prunable channel groups and their dependencies.
        - Handles complex topologies such as skip connections and multi-branch structures.
    - **Novelty**: **Architecture-agnostic**, capable of handling arbitrary structures including CNNs, Transformers, and hybrid architectures.
    - **Implementation**: Based on static analysis of the computation graph.

2. **Partially Projected Stochastic Gradient Descent (Partially Projected SGD)**

    - **Function**: Guarantees that layer-wise bit-width constraints are always satisfied during training.
    - **Mechanism**:
        - Relaxes discrete bit-width constraints $b_l \in \{2, 4, 8, ...\}$ into continuous variables.
        - Projects onto the constraint set after each gradient update step.
        - Alternatingly updates weight parameters, bit-widths, and pruning rates.
    - **Mathematical Guarantee**: Converges to a stationary point within the constraint-feasible region.
    - **Novelty**: No outer-loop search (such as NAS, reinforcement learning) is required, ensuring white-box interpretability.

3. **Joint Learning Strategy**

    - **Function**: Establishes an interpretable relationship between pruning and quantization.
    - **Mechanism**:
        - Channel reduction after pruning $\rightarrow$ Allows higher-precision quantization for the same layer.
        - Decreased precision after quantization $\rightarrow$ Requires retaining more channels to compensate.
        - Automatically balances both using the Lagrangian multiplier method.
    - **Key Insight**: There is a **complementary relationship** between pruning rates and bit-widths.
    - **Implementation**: The joint optimization objective function includes both accuracy loss and compression rate constraints.

### Loss & Training
- End-to-end one-shot training without multiple stages.
- No need for the traditional pretraining-pruning-finetuning pipeline.
- Supports both training from scratch and starting from pretrained models.

## Key Experimental Results

### ResNet-18 / ImageNet

| Method | Top-1 Acc↑ | FLOPs↓ | Description |
|------|-----------|--------|------|
| Pruning-only baseline | Ref | Ref | Independent pruning |
| Quantization-only baseline | Ref | Ref | Independent quantization |
| Joint baseline | Ref | Ref | Two-stage |
| **GETA** | **Competitive/Best** | **Higher compression rate** | One-stage joint |

### Transformer Architectures (ViT / DeiT)

| Method | Top-1 Acc↑ | Compression Rate | Feature |
|------|-----------|--------|------|
| Independent pruning | Baseline | Medium | Pruning only |
| Independent quantization | Baseline | Medium | Quantization only |
| **GETA** | **Higher** | **Higher** | Joint optimization |

### Ablation Study

| Component | Impact on Accuracy |
|------|-----------|
| w/o QADG (Manual search space) | Accuracy drops, and fails to generalize |
| w/o Projected SGD (Unconstrained) | Constraint violation, bit-width uncontrollable |
| w/o Joint strategy (Independent optimization) | Worsened compression-accuracy trade-off |
| **Full GETA** | **Optimal trade-off** |

### Key Findings
- Joint optimization consistently outperforms simple combinations of independent optimizations.
- Automation by QADG eliminates the need to manually design search spaces.
- Projected SGD ensures that constraints are never violated during the training process.
- Validated on both CNNs and Transformers, demonstrating architecture-agnostic capability.

## Highlights & Insights
- **Fully Automated**: No manual design of pruning rates/bit-widths per layer is required.
- **White-box Optimization**: Offers high interpretability compared to black-box search methods like NAS.
- **One-shot Training**: Eliminates the engineering complexity of multi-stage pipelines.
- **Architecture Agnostic**: QADG automatically handles arbitrary network topologies.

## Limitations & Future Work
- The accuracy drop is still relatively large under extreme compression rates.
- QADG construction relies on manual static graph analysis, offering limited support for dynamic graphs.
- Currently only validated on classification tasks; downstream tasks like detection/segmentation remain to be verified.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel combination of QADG + Projected SGD + Joint Strategy
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both CNN and Transformer architectures
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation
- Value: ⭐⭐⭐⭐ Practical significance for model deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](../../ICML2025/optimization/the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [\[NeurIPS 2025\] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](../../NeurIPS2025/optimization/dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)
- [\[ICML 2025\] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](../../ICML2025/optimization/sdp-crown_efficient_bound_propagation_for_neural_network_verification_with_tight.md)
- [\[ICLR 2026\] NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization](../../ICLR2026/optimization/neuclip_efficient_large-scale_clip_training_with_neural_normalizer_optimization.md)
- [\[ICML 2025\] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](../../ICML2025/optimization/interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)

</div>

<!-- RELATED:END -->
