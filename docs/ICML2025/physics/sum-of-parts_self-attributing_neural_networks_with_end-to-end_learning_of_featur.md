---
title: >-
  [Paper Note] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups
description: >-
  [ICML 2025][Physics & Scientific Computing] SOP proposes a framework to transform any differentiable model into a group-based Self-Attributing Neural Network (SANN). It achieves state-of-the-art performance in SANNs through end-to-end learning of feature groups, and theoretically proves the error lower bound of feature-wise SANNs and the zero-error reachability of group-based SANNs.
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
date: 2026-05-08
content_hash: e4c0212e8cffa247
---

# Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups

**Conference**: ICML 2025  
**arXiv**: [2310.16316](https://arxiv.org/abs/2310.16316)  
**Area**: Explainability  

## TL;DR

SOP proposes a framework to transform any differentiable model into a group-based Self-Attributing Neural Network (SANN). It achieves state-of-the-art performance in SANNs through end-to-end learning of feature groups, and theoretically proves the error lower bound of feature-wise SANNs and the zero-error reachability of group-based SANNs.

## Background & Motivation

Self-Explaining Neural Networks (SENNs) provide guaranteed linear explanations by decomposing predictions into linear combinations of interpretable concepts. Self-Attributing Neural Networks (SANNs) are a prominent class of these methods that use feature subsets as interpretable concepts, where predictions can be faithfully decomposed into a linear combination of contributions from these feature subsets.

However, existing SANNs face a severe **performance-interpretability trade-off**:
- **NAM** (Neural Additive Models): Uses single features and fails to capture feature correlations.
- **BagNet**: Relies on fixed-size image patches, lacking flexibility.
- **FRESH**: Uses attention to select a single subset, limiting the number of groups.

What is the root cause of this trade-off, and can it be overcome through a better grouping strategy?

## Method

### Theoretical Foundation

#### Error Lower Bound of Feature-wise SANNs

The paper formally proves that feature-wise SANNs suffer from an unavoidable error lower bound when processing data with highly correlated features.

**Theorem 2.3 (Insertion Error Lower Bound for Binomials)**: For a $d$-dimensional multilinear binomial polynomial $p(x) = \prod_{i \in S_1 \cup S_2} x_i + \prod_{j \in S_2 \cup S_3} x_j$:

$$\sum_{S \subseteq [d]} \text{InsErr}(G, \alpha, S) \geq D_{ins}(\hat{\lambda})$$

where $D_{ins}(\hat{\lambda})$ is the lower bound computed via dual feasible points of a linear program. **This lower bound grows exponentially with the dimension $d$**.

#### Zero-Error Reachability of Group-based SANNs

**Theorem 2.4 (Informal)**: For any $m$-term polynomial $p$, a SANN using at most $m$ groups can achieve zero insertion and deletion errors.

### SOP Framework

SOP consists of three components:

$$f(x) = \sum_{i=1}^m \underbrace{\theta(\Gamma(x), x)_i}_{\text{分组选择器}} \cdot \underbrace{h(g_i \odot x)}_{\text{骨干预测器}}$$

where $\underbrace{g_i \in \Gamma(x)}_{\text{分组生成器}}$.

#### Group Generator $\Gamma$

A multi-head self-attention module is used to assign scores to features, and threshold truncation is applied to each attention distribution (retaining the top $\tau=20\%$ features):

$$\Gamma(x) = (g_1, \ldots, g_m) = \text{SoftSelfAttn}_{\tau=20\%}(h_e(x))$$

where $h_e$ is the encoder (typically the penultimate layer of the backbone model).

#### Backbone Predictor $h$

A pretrained high-performance model with **frozen parameters** is utilized to make predictions on the masked input for each group:

$$y_i = h(g_i \odot x), \quad i = 1, \ldots, m$$

#### Group Selector $\theta$

A sparse cross-attention module is used to assign weights to each group:

$$\theta(\Gamma(x), x) = (c_1, \ldots, c_m) = \text{SparseCrossAttn}(C_h, z)$$

where $C_h$ is initialized with the weights of the target class, and $z$ represents the final hidden states of all groups. **Sparsemax** is used instead of softmax to yield sparse group weights.

### Key Designs

- **Binarized Grouping**: Using $\{0,1\}$ masks to prevent unfaithful explanations caused by information leakage.
- **Frozen Backbone**: Keeping the high performance of the pretrained model while training only the group generator and selector.
- **Model-Agnostic**: Compatible with any differentiable model (e.g., ViT, CNN, BERT).

## Experiments

### Main Results

| Category | Method | ImageNet-S ViT Err.↓ | IOU↑ | CosmoGrid CNN MSE↓ | MultiRC BERT Err.↓ |
|---|---|---|---|---|---|
| Backbone | Backbone | 0.097 | - | 0.009 | 0.318 |
| Post-hoc | SHAP-F | 0.306 | 0.391 | 0.028 | 0.455 |
| Post-hoc | FG-F | 0.448 | 0.511 | 0.036 | 0.396 |
| SANN | BagNet | 0.501 | 0.314 | - | - |
| SANN | FRESH | 0.537 | 0.464 | - | 0.386 |
| **SANN** | **SOP** | **0.267** | **0.548** | **0.015** | **0.356** |

SOP achieves the best performance among all SANNs, and even outperforms most post-hoc methods on ImageNet-S.

### Application in Scientific Discovery

On the cosmological CosmoGrid dataset, the groups and scores of SOP reveal new insights into galaxy formation. Researchers can inspect these groups to understand the specific feature coordinates (such as galaxy density, morphology, etc.) that the model focuses on.

### Model Debugging

SOP can be used to detect whether the model relies on correct/incorrect features (such as object vs. background), aiding in model debugging.

## Highlights & Insights

- **Solid Theoretical Contributions**: Formally proving the fundamental limitations of feature-wise SANNs and the zero-error reachability of group-based SANNs.
- **Model-Agnostic Framework**: Standardized framework capable of converting any pretrained model into a SANN without needing specific architectures.
- **End-to-End Group Learning**: No supervision from group labels is required; group assignments automatically adapt to data correlations.
- **Cross-Modal Validation**: Exemplary performance demonstrated across vision (ViT), scientific (CNN), and language (BERT) tasks.
- **Practical Value**: Demonstrated practical utility in both model debugging and scientific discovery.

## Limitations & Future Work

- Each group requires an independent forward pass of the backbone model, causing the inference cost to scale linearly with the number of groups $m$.
- The group size is fixed at 20%, which may not be optimal for all types of data.
- Binarized grouping requires special handling (scale factors) during gradient propagation.
- The multi-head attention in the group generator increases the parameter size.
- The explanation accuracy of SOP is lower than certain post-hoc methods on chemistry datasets such as Mutag.

## Rating

⭐⭐⭐⭐ (4/5)

An excellent combination of theory and practice. The paper proves the fundamental importance of feature grouping for SANNs and introduces an elegant model-agnostic framework. Experimental validation across multiple domains demonstrates the versatility and practicality of the proposed method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[ICLR 2026\] ComPhy: Composing Physical Models with end-to-end Alignment](../../ICLR2026/physics/comphy_composing_physical_models_with_end-to-end_alignment.md)
- [\[NeurIPS 2025\] Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](../../NeurIPS2025/physics/scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)
- [\[ICML 2025\] PAC Learning with Improvements](pac_learning_with_improvements.md)
- [\[ICML 2025\] Compact Matrix Quantum Group Equivariant Neural Networks](compact_matrix_quantum_group_equivariant_neural_networks.md)

</div>

<!-- RELATED:END -->
