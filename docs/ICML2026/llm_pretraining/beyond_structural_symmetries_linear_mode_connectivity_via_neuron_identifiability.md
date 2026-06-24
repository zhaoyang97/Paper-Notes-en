---
title: >-
  [Paper Note] Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability
description: >-
  [ICML 2026][LLM Pretraining][Parameter Symmetry] This paper proposes a theoretical framework of "effective function classes" and "neuron identifiability," revealing that breaking structural symmetry does not equate to breaking effective symmetry—even if permutation symmetry in the parameter space is eliminated, data-dependent approximate symmetries may still make neuron swapping costs extremely low. Based on this, it provides sufficient conditions for achieving Linear Mode Co…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Parameter Symmetry"
  - "Neuron Identifiability"
  - "Linear Mode Connectivity"
  - "Model Merging"
  - "Loss Landscape"
date: 2026-05-08
content_hash: 9b360171065d4f35
---

# Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability

**Conference**: ICML 2026  
**arXiv**: [2606.04754](https://arxiv.org/abs/2606.04754)  
**Code**: https://github.com/vuenc/neuron-identifiability  
**Area**: Optimization & Theory  
**Keywords**: Parameter Symmetry, Neuron Identifiability, Linear Mode Connectivity, Model Merging, Loss Landscape  

## TL;DR

This paper proposes a theoretical framework of "effective function classes" and "neuron identifiability," revealing that breaking structural symmetry does not equate to breaking effective symmetry—even if permutation symmetry in the parameter space is eliminated, data-dependent approximate symmetries may still make neuron swapping costs extremely low. Based on this, it provides sufficient conditions for achieving Linear Mode Connectivity (LMC) without the need for alignment.

## Background & Motivation

**Background**: Modern deep networks are typically over-parameterized and possess numerous parameter symmetries (especially permutation symmetry of hidden units), causing functionally identical models in the weight space to correspond to large equivalence classes. Linear Mode Connectivity (LMC) describes the phenomenon where the loss remains low during linear interpolation between two independently trained models in the weight space. Existing work has shown that LMC holds in many scenarios after aligning permutations.

**Limitations of Prior Work**: Recently, several symmetry-breaking methods (such as $\mathbf{W}$-asymmetric networks, SYRE, etc.) have been introduced to eliminate structural permutation symmetry by incorporating fixed biases or diagonal scaling in the weight matrix. However, breaking structural symmetry does not always result in alignment-free LMC—some interventions are effective while others are not, and the underlying mechanism remains unclear.

**Key Challenge**: Breaking structural symmetry is not equivalent to breaking effective symmetry. When data or representations reside in a low-dimensional subspace, even if the parameters of different neurons are no longer interchangeable, the functions they can implement on the input support may still be identical, and the cost of swapping them may be very low. The problem lies in the fact that the effectiveness of symmetry breaking depends on the observability of architectural perturbations on the input support, rather than solely on the structure of the parameter space.

**Goal**: (1) Formalize the effective function classes of neurons on the input support and their implementation costs; (2) Provide conditions for neuron identifiability (consistent assignment of features to neurons across training seeds); (3) Characterize sufficient conditions for the validity of alignment-free LMC.

**Key Insight**: Treat each neuron as an operator implementing a specific function on the input support, characterize the implementation cost using a Mahalanobis semi-norm, and measure the effectiveness of symmetry breaking through permutation sensitivity. Under a subspace support model, all analyses can be transformed into explicit linear algebraic forms.

**Core Idea**: Use "effective function classes + implementation costs" instead of "parameter space symmetry groups" to analyze symmetry breaking, revealing the decisive impact of data geometry (intrinsic dimension, subspace coherence) on LMC and model merging.

## Method

### Overall Architecture

The question this paper aims to answer is: why do some "symmetry-breaking" interventions yield alignment-free LMC while others do not? To this end, the authors view each neuron as an operator implementing a certain function on the input support, using a single-layer network with asymmetric interventions as a unified carrier: $\bm{H}(\bm{W};\bm{x}) = \eta((\mathbf{F} + \mathbf{D} \odot \bm{W})\bm{x})$, where $\mathbf{F}$ is a fixed bias matrix, $\mathbf{D}$ is a fixed diagonal scaling matrix, and $\bm{W}$ is the trainable weight—$\mathbf{W}$-asymmetric networks, SYRE, linear residuals, and sparse networks are all special cases of this formula. Under the low-dimensional support assumption that "inputs actually fall into a $k$-dimensional subspace $\mathcal{U}$," the entire analytical chain is: first characterize which functions each neuron can implement and at what cost (effective function classes), then use permutation sensitivity to judge whether these neurons are identifiable, and finally connect identifiability to the chordal deviation upper bound of LMC to provide sufficient conditions for linear connectivity without alignment.

### Key Designs

**1. Effective Function Classes and Implementation Cost: Replacing "parameter interchangeability" with "weight norm required to implement a function"**

Whether symmetry breaking is effective should not depend on whether neurons remain interchangeable in the parameter space, but rather on what they can implement on the actual input support and how much their respective implementation costs differ. Under the subspace support model, the function class $\mathcal{H}_i(\mathcal{X})$ that the $i$-th neuron can implement is an affine subspace $\mathbf{v}_i + \mathrm{im}(\mathbf{M}_i) \subseteq \mathbb{R}^k$, where the projection center $\mathbf{v}_i = \bm{U}^\top \mathbf{f}_i$ is determined by fixed biases, and the projection operator $\mathbf{M}_i = \bm{U}^\top \mathrm{Diag}(\mathbf{d}_i)$ is determined by diagonal scaling. The minimum weight norm to implement a target function is transformed into a Mahalanobis semi-norm $\|\bm{h}\|_{\mathcal{H}_i} = \|\bm{a} - \mathbf{v}_i\|_{\mathbf{S}_i^\dagger}$, where the anisotropy of the Gram matrix $\mathbf{S}_i = \mathbf{M}_i \mathbf{M}_i^\top$ determines which directions are "cheap" and which are "expensive." The key point is: even if the function classes of all neurons are identical (when $\mathbf{M}_i$ is full rank), the cost of implementing the same function can differ vastly; since optimizers prefer low-norm solutions, they will consistently assign a feature to the neuron where it is "cheapest to implement" across training seeds—this is precisely the source of identifiability.

**2. Permutation Sensitivity: Quantifying "neuron swappability" as the additional cost paid for a swap**

With implementation costs defined, one can ask: how much would the total cost increase if features were redistributed to other neurons? For an output-side permutation $\pi$, define the cost sensitivity $\Delta_\pi^{\mathrm{out}}$, which is approximately $\mu_\mathbf{D}^{-1} \|\bm{\delta}_\pi^{\mathrm{out}}\|_F^2$ in the center-dominated regime, with its magnitude primarily determined by the minimum spacing between projection centers $\gamma_{\mathrm{out}} = \sqrt{2} \min_{i \neq j} \|\mathbf{v}_j - \mathbf{v}_i\|_2$. If the sensitivity for every non-identity permutation is sufficiently large, then the "minimum complexity feature-neuron assignment" is unique, and neurons are thus identifiable. The dependence of this quantity on dimensionality is interesting: in high dimensions $\gamma_{\mathrm{out}} = \Theta(\sigma_\mathbf{F} \sqrt{k} m^{-2/k})$, and the larger the intrinsic dimension $k$, the slower it decays, so high-dimensional data naturally makes symmetry breaking "more effective." There is also a corresponding sensitivity for input-side permutations $\tau$, controlled by $\gamma_{\mathrm{in}} = \Theta(\sigma_\mathbf{F} \sqrt{m}) \cdot \min_{a \neq b} \|\bm{U}_{b,:} - \bm{U}_{a,:}\|_2$, which additionally depends on subspace coherence $\nu(\mathcal{U})$—explaining why the effectiveness of diagonal masks $\mathbf{D}$ alone varies with data geometry.

**3. LMC Chordal Deviation Upper Bound: Connecting identifiability to "no increase in loss during interpolation"**

The final step is translating neuron identifiability into the existence of LMC. For ReLU networks, the authors prove that the chordal deviation along the linear interpolation path satisfies $\sup_\lambda \|\xi_{\bm{H}}(\lambda;\cdot)\|_{L^2} = \mathcal{O}(\beta^{3/2}) \|\mathbf{F} \bm{\Sigma}^{1/2}\|_F$, where $\beta$ measures the ratio of the trainable part to the fixed part. By leveraging the Lipschitz continuity of the loss, the chordal deviation directly becomes an upper bound on the loss barrier; thus, as long as the chordal deviation is small, LMC holds. The value of this step lies in reducing the analysis of LMC from "the entire high-dimensional loss landscape" to "layer-wise chordal deviations" and using the center-dominant condition to allow local convexity to propagate along the entire interpolation path.

## Key Experimental Results

### Main Results: Aligned/Unaligned LMC

| Architecture | Dataset | $\sigma_\mathbf{F}$ | Unaligned LMC Barrier | Aligned LMC Barrier |
|------|--------|---------------------|-----------------|---------------|
| MLP | MNIST | 0 | High (~80% accuracy drop) | Low |
| $\mathbf{W}$-MLP | MNIST | 0 | High | Low |
| $\mathbf{W}$-MLP | MNIST | 1 | **Low (near zero)** | Low |
| SYRE-MLP | MNIST | - | Low | Low |
| ResNet | CIFAR-10 | 0 | High | Low |
| $\mathbf{W}$-ResNet | CIFAR-10 | 2 | **Low-to-Moderate** | Low |
| SYRE-ResNet | CIFAR-10 | - | Medium | Low (alignment still helps) |

### Ablation Study: Impact of Intrinsic Dimension $k$ on Neuron Swapping Cost

| Intrinsic Dimension $k$ | $\sigma_\mathbf{F}$ | Unaligned LMC Accuracy Drop | Swap Cost Distribution |
|-------------|---------------------|---------------------|-------------|
| $k=2$ | 1 | 46.4 pp | Many low-cost swaps |
| $k=8$ | 1 | 15.7 pp | Moderate |
| $k=32$ | 1 | **6.1 pp** | Only diagonal is low-cost |
| $k=2$ | 0 | High | Near zero or negative cost |

### Key Findings
- **Structure $\neq$ Effectiveness**: $\mathbf{W}$-asymmetric networks fail to achieve unaligned LMC when $\sigma_\mathbf{F}=0$ despite breaking structural symmetry, yet perform similarly to standard networks after alignment.
- **Intrinsic Dimension is a Key Control Variable**: The larger $k$ is, the larger the projection center spacing $\gamma_{\mathrm{out}}$ becomes, leading to higher neuron swap costs and superior unaligned LMC.
- **Subspace Coherence Affects Diagonal Scaling Effectiveness**: In cases of high coherence ($\nu \approx 1$), LMC can be achieved with diagonal masks $\mathbf{D}$ alone even if $\mathbf{F}=0$; this fails under low coherence.
- **Activation Matching Experiments**: When $\sigma_\mathbf{F}$ is sufficiently large, the activation matching objective for the identity permutation approaches the optimal permutation, confirming neuron identifiability.

## Highlights & Insights
- **Elegance of Theoretical Insight**: Shifts symmetry breaking from "algebraic properties of parameter space" to "functional geometry on input support." The Mahalanobis semi-norm framework makes the analysis both intuitive and computable, perfectly explaining empirical phenomena such as "why bias alone is ineffective at breaking symmetry" (biases only introduce differences in a one-dimensional direction).
- **Core Role of Data Geometry**: Reveals the decisive influence of data intrinsic dimension and subspace coherence on LMC—an important insight transferable to model merging, federated learning, and weight space learning.
- **Unification of Multiple Schemes**: A single formula $W_{\mathrm{eff}} = \mathbf{F} + \mathbf{D} \odot \bm{W}$ covers seemingly different symmetry-breaking schemes like W-asymmetric, SYRE, sparse networks, and linear residuals.

## Limitations & Future Work
- Theoretical analysis is restricted to **linear subspace support models**; actual data manifolds may be more complex (nonlinear, curved).
- The current framework focuses on **layer-wise analysis** and has not yet fully characterized the interaction effects of cross-layer symmetries in deep networks.
- **Tradeoff between Identifiability and Feature Learning**: Strong symmetry breaking limits the range of learnable features; in the center-dominated regime, the network degrades into a random feature model—how to maintain identifiability while maximizing feature learning capacity remains an open question.
- Future Directions: Extending the framework to richer symmetry groups (such as attention head symmetries), incorporating precise training dynamics analysis, and developing quantitative diagnostic tools to predict whether LMC will hold.

## Related Work & Insights
- **LMC and Model Merging**: LMC conjecture by Entezari et al. (2022), Git Re-Basin method by Ainsworth et al. (2023), activation matching by Singh & Jaggi (2020).
- **Symmetry Breaking**: W-asymmetric networks by Lim et al. (2024b), SYRE by Ziyin et al. (2025).
- **Weight Space Learning**: Treating model weights as data objects is becoming an emerging direction; the identifiability analysis in this paper directly serves the understanding of "data symmetry" in this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Annotations Mitigate Post-Training Mode Collapse](annotations_mitigate_post-training_mode_collapse.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[ICLR 2026\] Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](../../ICLR2026/llm_pretraining/beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)
- [\[ICLR 2026\] Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining](../../ICLR2026/llm_pretraining/beyond_urls_metadata_diversity_and_position_for_efficient_llm_pretraining.md)

</div>

<!-- RELATED:END -->
