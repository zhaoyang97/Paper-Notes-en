---
title: >-
  [Paper Note] FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks
description: >-
  [ICML2025][Interpretability][Concept Activation Vectors] FastCAV is proposed to replace SVM training with the normalized mean difference vector of concept activation samples. This approach is theoretically equivalent to a simplified form of Fisher Discriminant Analysis. It achieves up to 63.6$\times$ (average 46.4$\times$) acceleration while maintaining comparable classification accuracy and downstream explanation quality to SVM-CAV.
tags:
  - "ICML2025"
  - "Interpretability"
  - "Concept Activation Vectors"
  - "Explainability"
  - "CAV"
  - "TCAV"
  - "Model Explainability"
  - "Acceleration"
date: 2026-05-08
content_hash: 9fc82b47d2d525fc
---

# FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks

**Conference**: ICML2025  
**arXiv**: [2505.17883](https://arxiv.org/abs/2505.17883)  
**Code**: [fastcav.github.io](https://fastcav.github.io/)  
**Area**: Video Understanding  
**Keywords**: Concept Activation Vectors, Explainability, CAV, TCAV, Model Explainability, Acceleration  

## TL;DR

FastCAV is proposed to replace SVM training with the normalized mean difference vector of concept activation samples. This approach is theoretically equivalent to a simplified form of Fisher Discriminant Analysis. It achieves up to 63.6$\times$ (average 46.4$\times$) acceleration while maintaining comparable classification accuracy and downstream explanation quality to SVM-CAV.

## Background & Motivation

### Background

Concept Activation Vectors (CAV) are crucial tools in the field of model explainability. The core idea is to find a directional vector in the activation space of a neural network that corresponds to a human-understandable concept (such as "stripes", "wheels", etc.), thereby quantifying the concept's influence on the model's prediction. The classic approach, TCAV (Kim et al., 2018), trains a linear SVM classifier to separate the activations of concept images from random images, using the SVM's normal vector as the CAV.

### Limitations of Prior Work

**High computational cost**: The complexity of training an SVM is $\mathcal{O}(\max(n,d)\min(n,d)^2)$. For modern large models (e.g., EVA-02-L/14 with an activation dimension of 1,049,600), training a single CAV takes several minutes.

**Needs high repetition**: To ensure statistical significance, TCAV requires computing multiple CAVs for the same concept using different random sets, which multiplies the computational cost.

**Infeasibility for modern architectures**: For large models like ConvNeXt-XXL and EVA-02, SVM-CAV computations can take over 4 days to complete the analysis of all concepts.

**Limiting downstream applications**: Tasks such as cross-layer concept tracking and analyzing concept evolution during training are rendered infeasible due to the massive computational overhead.

## Method

### Core Idea

By leveraging the near-orthogonality (superposition) of features in the activation space of neural networks, a simple mean difference vector can replace the SVM optimization process to extract concept directions.

### FastCAV Mechanism

**Step 1: Compute the Global Mean**

Compute the global mean of all activations from the concept image set $D_c$ and the random image set $D_r$:

$$\hat{\mu}_{D_c \cup D_r} = \frac{1}{|D_c| + |D_r|} \sum_{x \in D_c \cup D_r} g_l(x)$$

where $g_l(x)$ represents the activation vector of input $x$ at layer $l$.

**Step 2: Compute the Concept Direction**

Use the zero-centered mean of the concept samples as the CAV direction:

$$v_c^l \propto \frac{1}{|D_c|} \sum_{x \in D_c} (g_l(x) - \hat{\mu}_{D_c \cup D_r})$$

That is, $v_c^l$ points from the global mean $\hat{\mu}_{D_c \cup D_r}$ to the concept mean $\hat{\mu}_{D_c}$, which is then normalized to a unit vector.

**Step 3: Compute the Intercept**

The intercept of the decision boundary is: $b = -v_c^l \cdot \hat{\mu}_{D_c \cup D_r}$.

### Theoretical Foundation: Connection with Fisher Discriminant Analysis and SVM

The paper provides a detailed proof of the equivalence relationship between FastCAV and classic linear methods:

1. **Connection with LDA**: Assuming the concept and random samples both follow multivariate Gaussian distributions and are mixed in equal proportions, the expected solution of FastCAV is $\mathbb{E}[v_c^l] \propto \frac{\mu_c - \mu_r}{2}$. This is the solution of Fisher Discriminant Analysis under the assumption of **isotropic within-class covariance**.
2. The **Fisher LDA solution** is $\hat{\Sigma}^{-1}(\hat{\mu}_c - \hat{\mu}_r)$, which degenerates to FastCAV when $\Sigma^{-1}$ is proportional to the identity matrix.
3. **Connection with SVM**: Shashua (1999) proved that the solution of Fisher Discriminant Analysis on the support vector set is equivalent to the solution of a linear SVM. When the activation dimension $d \gg n$ (number of samples), almost all samples become support vectors, making their solutions converge.

### Complexity Comparison

| Method | Training Complexity | Inference Complexity |
|------|-----------|-----------|
| SVM-CAV | $\mathcal{O}(\max(n,d)\min(n,d)^2)$ | $\mathcal{O}(d)$ |
| SGD-SVM | $\mathcal{O}(Tnd)$ | $\mathcal{O}(d)$ |
| **FastCAV** | $\mathcal{O}(nd)$ (extremely small constant) | $\mathcal{O}(d)$ |

FastCAV only requires a single mean calculation and normalization, completely bypassing iterative optimization.

## Key Experimental Results

### Experimental Setup

- **Dataset**: Models trained on ImageNet, with concept images obtained from the Broden dataset.
- **Concept/Random Set Size**: 60 images each.
- **Statistics**: Averaged across 30 resampled random sets, covering all Broden concepts and network layers.
- **Evaluation Dimensions**: Computation time, classification accuracy, inter-method similarity, and intra-method robustness.
- **Tested Architectures**: Inception-v3, ResNet50, ConvNeXt-XXL, InceptionNeXt, ViT-L/16, EVA-02-L/14, EVA-02-L/14+.

### Main Results (Table 1 Summary)

| Model | Avg Dimension | FastCAV Time(s) | SVM-CAV Time(s) | FastCAV Acc | SVM Acc | Inter-method Similarity |
|------|---------|----------------|-----------------|-------------|---------|------------|
| Inception-v3 | 206K | **0.4** | 44.7 | **0.95** | 0.93 | 0.898 |
| ResNet50 | 341K | **1.1** | 135.4 | **0.89** | 0.87 | 0.837 |
| ConvNeXt-XXL | 754K | **5.5** | N/A (>4 days) | — | — | — |

Key findings:

- **Speed**: FastCAV achieves an average speedup of **46.4$\times$** and up to **63.6$\times$**. On Inception-v3, it requires only 0.4s compared to 44.7s for SVM.
- **Accuracy**: FastCAV's accuracy is comparable to or even better than SVM-CAV across most models (Inception-v3: 0.95 vs 0.93).
- **Robustness**: FastCAV's intra-method similarity is significantly higher than that of SVM-CAV (e.g., Inception-v3: 0.795 vs 0.338), showing that FastCAV is more stable against the choice of random sets.
- **Feasibility on Large Models**: For large models like ConvNeXt-XXL, SVM-CAV computation cannot be completed within 4 days, whereas FastCAV takes only a few seconds.

### Downstream Tasks Validation

- **TCAV Experiments**: Replacing SVM-CAV with FastCAV for TCAV tests yields the same concept importance rankings as the SVM method.
- **ACE Experiments**: In the Automatic Concept Explanation (ACE) task, FastCAV produces equivalent explanation results.
- **Concept Evolution Tracking**: Thanks to FastCAV's high efficiency, it is used to track the evolution of concepts across layers during ResNet50 training for the first time—an analysis previously impossible with SVM-CAV due to computational constraints.

### Medical Imaging Experiments

The paper also validates the applicability of FastCAV in the medical imaging field, proving its effectiveness on professional concepts (such as medical diagnostic features).

## Highlights & Insights

1. **Simple yet Effective**: The core operation only involves computing the mean and normalization, avoiding any optimization solver, yet it performs on par with SVM. This indicates that the near-orthogonality of features in high-dimensional activation spaces indeed holds.
2. **Complete Theoretical Chain**: It builds a step-by-step equivalence relation: FastCAV $\rightarrow$ Fisher LDA (isotropic assumption) $\rightarrow$ SVM (full support vector set assumption).
3. **Better Robustness**: SVM is highly sensitive to the choice of the random set (similarity is only 0.338), whereas FastCAV is much more stable (0.795), which is an unexpected but significant advantage.
4. **Unlocks New Analyses**: Concept evolution tracking during training was previously an intractable application, demonstrating the practical research value enabled by the acceleration.

## Limitations & Future Work

1. **Strong Theoretical Assumptions**: The equivalence relies on the "isotropic within-class covariance" and "Gaussian distribution" assumptions. Real activation spaces might not strictly satisfy these, which the paper acknowledges as a simplification.
2. **Incorrect Domain Classification**: This paper belongs to the explainability/XAI domain and has nothing to do with video understanding (the original stub's domain classification is incorrect).
3. **Fixed Concept Set Size**: The experiments use a fixed size of 60 concept images; a systematic study on the performance across different concept set sizes is lacking.
4. **Limited to Linear Concepts**: Like SVM-CAV, FastCAV assumes concepts are linearly separable in the activation space and does not apply to non-linear concepts (interactive or hierarchical concepts).
5. **No Exploration of Non-Visual Models**: Experiments focus heavily on vision models, leaving performance on NLP or multimodal models unverified.
6. **Failure Case Discussion**: The paper mentions that failure cases are discussed in Appendix B.2.2, but these boundary conditions should have been presented more prominently.

## Related Work & Insights

- **CAV/TCAV (Kim et al., 2018)**: The direct baseline that this work improves upon, which defines concept directions using the normal vectors of SVMs.
- **ACE (Ghorbani et al., 2019)**: Automatic Concept Extraction, which relies heavily on massive CAV computations and can directly benefit from FastCAV.
- **Superposition (Elhage et al., 2022)**: Features are encoded in near-orthogonal directions in the activation space, serving as the theoretical foundation for FastCAV.
- **Fisher-SVM Equivalence (Shashua, 1999)**: Fisher discriminant analysis is equivalent to the SVM solution on the support vector set.
- **Insight**: In high-dimensional, low-sample-size regimes, a simple mean difference direction can be the optimal solution for linear classification tasks. This idea can be transferred to other scenarios requiring rapid linear probing.

## Rating

- **Novelty**: ⭐⭐⭐ — The method itself is extremely simple (mean difference vector); the novelty lies in establishing the theoretical connections and validating that "simplicity is key".
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers over 7 architectures, multiple downstream tasks, medical imaging, concept evolution tracking, etc., offering a comprehensive four-dimensional evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear theoretical derivations, reasonable experimental design, and well-structured paper.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, no extra dependencies, 46$\times$ speedup, more stable, offering direct value to the XAI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hidden Monotonicity: Explaining Deep Neural Networks via their DC Decomposition](../../CVPR2026/interpretability/hidden_monotonicity_explaining_deep_neural_networks_via_their_dc_decomposition.md)
- [\[NeurIPS 2025\] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](../../NeurIPS2025/interpretability/fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)
- [\[NeurIPS 2025\] Deep Modularity Networks with Diversity-Preserving Regularization](../../NeurIPS2025/interpretability/deep_modularity_networks_with_diversity-preserving_regularization.md)
- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [\[ICCV 2025\] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps](../../ICCV2025/interpretability/ce-fam_concept-based_explanation_via_fusion_of_activation_maps.md)

</div>

<!-- RELATED:END -->
