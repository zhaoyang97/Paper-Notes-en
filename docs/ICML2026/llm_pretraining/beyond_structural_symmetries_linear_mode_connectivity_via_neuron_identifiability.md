---
title: >-
  [Paper Note] Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability
description: >-
  [ICML 2026][LLM Pretraining][Parameter symmetries] This paper proposes a theoretical framework of "effective function classes" and "neuron identifiability…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Parameter symmetries"
  - "neuron identifiability"
  - "linear mode connectivity"
  - "model merging"
  - "loss landscapes"
date: 2026-05-08
content_hash: dc50e73f8e2b04e3
---

# Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability

**Conference**: ICML 2026  
**arXiv**: [2606.04754](https://arxiv.org/abs/2606.04754)  
**Code**: https://github.com/vuenc/neuron-identifiability  
**Area**: Optimization & Theory  
**Keywords**: Parameter symmetries, neuron identifiability, linear mode connectivity, model merging, loss landscapes  

## TL;DR

This paper proposes a theoretical framework of "effective function classes" and "neuron identifiability," revealing that breaking structural symmetry does not equate to breaking effective symmetry. Even when permutation symmetry in the parameter space is eliminated, data-dependent approximate symmetries can result in extremely low costs for neuron swapping. Based on this, the paper provides sufficient conditions for achieving linear mode connectivity (LMC) without alignment.

## Background & Motivation

**Background**: Modern deep networks are typically overparameterized and exhibit extensive parameter symmetries (particularly permutation symmetry of hidden units), where functionally equivalent models correspond to large equivalence classes in the weight space. Linear mode connectivity (LMC) describes the phenomenon where the loss remains low during linear interpolation between two independently trained models in weight space. Prior work has shown that LMC holds in many scenarios after permutation alignment.

**Limitations of Prior Work**: Recent symmetry-breaking methods (such as $\mathbf{W}$-asymmetric networks, SYRE, etc.) eliminate structural permutation symmetry by introducing fixed biases or diagonal scaling in weight matrices. However, breaking structural symmetry does not always lead to alignment-free LMC—some interventions are effective while others are not, and the underlying mechanism remains unclear.

**Key Challenge**: Breaking structural symmetry does not mean breaking effective symmetry. When data or representations lie in low-dimensional subspaces, neurons may still realize similar functions over the input support even if their parameters are no longer interchangeable, resulting in low swapping costs. The core issue is that the effectiveness of symmetry breaking depends on the observability of architectural perturbations on the input support rather than just the structure of the parameter space.

**Goal**: (1) Formalize the effective function classes of neurons on input support and their realization costs; (2) Provide conditions for neuron identifiability (consistent feature assignment to neurons across training seeds); (3) Characterize sufficient conditions for alignment-free LMC.

**Key Insight**: Treat each neuron as an operator realizing a specific function on the input support, characterize realization costs using the Mahalanobis semi-norm, and measure symmetry-breaking effectiveness via permutation sensitivity. Under the subspace support model, all analyses are converted into explicit linear algebraic forms.

**Core Idea**: Replace the "parameter space symmetry group" with "effective function classes + realization costs" to analyze symmetry breaking, revealing the decisive impact of data geometry (intrinsic dimension, subspace coherence) on LMC and model merging.

## Method

### Overall Architecture

Consider a single-layer network with non-asymmetric intervention: $\bm{H}(\bm{W};\bm{x}) = \eta((\mathbf{F} + \mathbf{D} \odot \bm{W})\bm{x})$, where $\mathbf{F}$ is a fixed bias matrix, $\mathbf{D}$ is a fixed diagonal scaling matrix, and $\bm{W}$ is the trainable weight. This unified formula covers various symmetry-breaking schemes such as $\mathbf{W}$-asymmetric networks, SYRE, linear residual connections, and sparse networks. Under the assumption of low-dimensional subspace support (where inputs reside in a $k$-dimensional subspace $\mathcal{U}$), the function sets realizable by each neuron on the input support and their realization costs are analyzed to derive upper bounds for the LMC chord deviation.

### Key Designs

1.  **Effective Function Classes and Realization Cost**:
    - **Function**: Quantify the minimum weight norm required for each neuron to achieve a target function.
    - **Mechanism**: Under the subspace support model, the function class $\mathcal{H}_i(\mathcal{X})$ realizable by the $i$-th neuron is represented as an affine subspace $\mathbf{v}_i + \mathrm{im}(\mathbf{M}_i) \subseteq \mathbb{R}^k$, where $\mathbf{v}_i = \bm{U}^\top \mathbf{f}_i$ is the projection center and $\mathbf{M}_i = \bm{U}^\top \mathrm{Diag}(\mathbf{d}_i)$ is the projection operator. The realization cost is defined as the Mahalanobis semi-norm: $\|\bm{h}\|_{\mathcal{H}_i} = \|\bm{a} - \mathbf{v}_i\|_{\mathbf{S}_i^\dagger}$, where the anisotropy of the Gram matrix $\mathbf{S}_i = \mathbf{M}_i \mathbf{M}_i^\top$ determines the cost variance across directions.
    - **Design Motivation**: Even if function classes are identical (when $\mathbf{M}_i$ is full rank), the costs of realizing the same function can differ significantly across neurons. Optimizers prefer low-norm solutions, thereby consistently assigning features to "cheaper" neurons across different seeds.

2.  **Permutation Sensitivity**:
    - **Function**: Measure the change in realization cost caused by reassigning features to different neurons.
    - **Mechanism**: For a permutation $\pi$, cost sensitivity $\Delta_\pi^{\mathrm{out}}$ is defined. In the center-dominated regime, it is approximately $\mu_\mathbf{D}^{-1} \|\bm{\delta}_\pi^{\mathrm{out}}\|_F^2$, primarily determined by the projection center spacing $\gamma_{\mathrm{out}} = \sqrt{2} \min_{i \neq j} \|\mathbf{v}_j - \mathbf{v}_i\|_2$. in high-dimensional spaces, $\gamma_{\mathrm{out}} = \Theta(\sigma_\mathbf{F} \sqrt{k} m^{-2/k})$, which decays slowly when the intrinsic dimension $k$ is sufficiently large, ensuring effective symmetry breaking.
    - **Design Motivation**: If sensitivity to all non-identity permutations is high, a unique minimum-complexity feature-neuron assignment exists (neuron identifiability). Sensitivity to input permutations $\tau$ is controlled by $\gamma_{\mathrm{in}} = \Theta(\sigma_\mathbf{F} \sqrt{m}) \cdot \min_{a \neq b} \|\bm{U}_{b,:]} - \bm{U}_{a,:]}\|_2$, which critically depends on subspace coherence $\nu(\mathcal{U})$.

3.  **LMC Chord Deviation Bound**:
    - **Function**: Build a bridge from neuron identifiability to alignment-free LMC.
    - **Mechanism**: For ReLU networks, the chord deviation along the linear interpolation path satisfies $\sup_\lambda \|\xi_{\bm{H}}(\lambda;\cdot)\|_{L^2} = \mathcal{O}(\beta^{3/2}) \|\mathbf{F} \bm{\Sigma}^{1/2}\|_F$, where $\beta$ measures the ratio of the trainable part to the fixed part. Combined with the Lipschitz property of the loss, the chord deviation directly bounds the loss barrier.
    - **Design Motivation**: Reduce the LMC problem from global loss landscape analysis to layer-wise chord deviation analysis, using center-dominance conditions to propagate properties through the interpolation segment under convexity.

## Key Experimental Results

### Main Results: Alignment/Alignment-free LMC

| Architecture | Dataset | $\sigma_\mathbf{F}$ | Alignment-free LMC Barrier | Aligned LMC Barrier |
| :--- | :--- | :--- | :--- | :--- |
| MLP | MNIST | 0 | High (~80% accuracy drop) | Low |
| $\mathbf{W}$-MLP | MNIST | 0 | High | Low |
| $\mathbf{W}$-MLP | MNIST | 1 | **Low (Near zero)** | Low |
| SYRE-MLP | MNIST | - | Low | Low |
| ResNet | CIFAR-10 | 0 | High | Low |
| $\mathbf{W}$-ResNet | CIFAR-10 | 2 | **Moderately Low** | Low |
| SYRE-ResNet | CIFAR-10 | - | Medium | Low (Alignment still helps) |

### Ablation Study: Impact of Intrinsic Dimension $k$ on Neuron Swapping Cost

| Intrinsic Dimension $k$ | $\sigma_\mathbf{F}$ | Alignment-free LMC Acc. Drop | Swapping Cost Distribution |
| :--- | :--- | :--- | :--- |
| $k=2$ | 1 | 46.4 pp | Many low-cost swaps |
| $k=8$ | 1 | 15.7 pp | Medium |
| $k=32$ | 1 | **6.1 pp** | Only diagonal is low-cost |
| $k=2$ | 0 | High | Near zero or negative cost |

### Key Findings
- **Structure $\neq$ Effective**: Although $\mathbf{W}$-asymmetric networks break structural symmetry at $\sigma_\mathbf{F}=0$, the alignment-free LMC barrier remains high. After alignment, they perform similarly to standard networks.
- **Intrinsic Dimension is a Key Control Variable**: Larger $k$ leads to larger projection center spacing $\gamma_{\mathrm{out}}$, higher neuron swapping costs, and better alignment-free LMC.
- **Subspace Coherence Affects Diagonal Scaling Effectiveness**: With high coherence ($\nu \approx 1$), LMC can be achieved via diagonal masking $\mathbf{D}$ alone even when $\mathbf{F}=0$; this fails with low coherence.
- **Activation Matching Validation**: When $\sigma_\mathbf{F}$ is sufficiently large, the activation matching objective for the identity permutation approaches the optimal permutation, confirming neuron identifiability.

## Highlights & Insights
- **Elegance of Theoretical Insight**: Shifts symmetry breaking from "algebraic properties of parameter space" to "functional geometry on input support." The Mahalanobis semi-norm framework provides an intuitive and computable explanation for empirical phenomena, such as why biases fail to break symmetry (biases only introduce differences along a one-dimensional direction).
- **Central Role of Data Geometry**: Reveals the decisive influence of data intrinsic dimension and subspace coherence on LMC—a transferable insight for model merging, federated learning, and weight space learning.
- **Unification of Multiple Schemes**: The formula $W_{\mathrm{eff}} = \mathbf{F} + \mathbf{D} \odot \bm{W}$ unifies seemingly different symmetry-breaking approaches like W-asymmetric, SYRE, sparse networks, and linear residuals.

## Limitations & Future Work
- Theoretical analysis is limited to **linear subspace support models**; manifolds of real-world data may be more complex (nonlinear, curved).
- The current framework focuses on **layer-wise analysis** and does not fully characterize the interaction effects of cross-layer symmetries in deep networks.
- **Tradeoff between Identifiability and Feature Learning**: Strong symmetry breaking restricts the range of learnable features; in the center-dominated regime, the network degenerates toward a random feature model. Maximizing feature learning capacity while maintaining identifiability remains an open problem.
- Future directions: Extend the framework to broader symmetry groups (e.g., attention head symmetries), integrate exact training dynamics analysis, and develop quantitative diagnostic tools for predicting LMC.

## Related Work & Insights
- **LMC and Model Merging**: LMC conjecture by Entezari et al. (2022), Git Re-Basin method by Ainsworth et al. (2023), activation matching by Singh & Jaggi (2020).
- **Symmetry Breaking**: W-asymmetric networks by Lim et al. (2024b), SYRE by Ziyin et al. (2025).
- **Weight Space Learning**: Treating model weights as data objects is an emerging direction; this paper's identifiability analysis directly contributes to understanding "data symmetry" in this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Annotations Mitigate Post-Training Mode Collapse](annotations_mitigate_post-training_mode_collapse.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[NeurIPS 2025\] Beyond Benign Overfitting in Nadaraya-Watson Interpolators](../../NeurIPS2025/llm_pretraining/beyond_benign_overfitting_in_nadaraya-watson_interpolators.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)

</div>

<!-- RELATED:END -->
