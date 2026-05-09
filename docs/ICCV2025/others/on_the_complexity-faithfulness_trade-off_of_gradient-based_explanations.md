---
title: >-
  [Paper Note] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations
description: >-
  [ICCV 2025][Explainability] This paper proposes a unified spectral framework to systematically analyze and quantify the trade-off between the smoothness (complexity) and faithfulness of gradient-based explanations. It introduces Expected Frequency (EF) to measure a network's reliance on high-frequency information, controls explanation complexity by convolving ReLU with a Gaussian function, and defines an "explanation gap" to quantify the faithfulness loss induced by surrogate models.
tags:
  - ICCV 2025
  - Explainability
  - Gradient-Based Explanations
  - Spectral Analysis
  - Faithfulness-Complexity Trade-off
  - ReLU Networks
date: 2026-05-08
content_hash: 2a411c6d4a6e3395
---

# On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations

**Conference**: ICCV 2025  
**arXiv**: [2508.10490](https://arxiv.org/abs/2508.10490)  
**Code**: [github.com/Amir-Mehrpanah/On-the-Complexity-Faithfulness-Trade-off-of-Gradient-Based-Explanations-ICCV25](https://github.com/Amir-Mehrpanah/On-the-Complexity-Faithfulness-Trade-off-of-Gradient-Based-Explanations-ICCV25/)  
**Area**: Other  
**Keywords**: Explainability, Gradient-Based Explanations, Spectral Analysis, Faithfulness-Complexity Trade-off, ReLU Networks

## TL;DR

This paper proposes a unified spectral framework to systematically analyze and quantify the trade-off between the smoothness (complexity) and faithfulness of gradient-based explanations. It introduces Expected Frequency (EF) to measure a network's reliance on high-frequency information, controls explanation complexity by convolving ReLU with a Gaussian function, and defines an "explanation gap" to quantify the faithfulness loss induced by surrogate models.

## Background & Motivation

### Problem Definition

Gradient-based explanation methods (e.g., VanillaGrad) are widely used interpretability tools in computer vision, yet they exhibit two fundamental tensions:

**Explanation Complexity**: ReLU networks exhibit sharp transition characteristics and sometimes rely on individual pixels for prediction, causing gradient explanations to appear "noisy" and difficult for humans to interpret.

**Explanation Faithfulness**: To reduce complexity, post-hoc methods (e.g., GradCAM, SmoothGrad) smooth explanations by constructing surrogate models, at the cost of deviating from the true behavior of the original model.

### Limitations of Prior Work

**Fragmented Metrics**: Existing metrics such as entropy-based methods and pixel removal scores each measure only one dimension—either complexity or faithfulness—and cannot jointly analyze their trade-off.

**Confounding External Factors**: Pixel removal scores are sensitive to baseline selection and removal ordering, impeding principled analysis of the trade-off.

**Implicit Surrogate Models**: Existing explanation methods design smoothing strategies via trial and error, producing surrogate models that are typically implicit and inaccessible, making it difficult to directly measure the explanation gap.

**Lack of Architecture-Level Understanding**: Prior work has not established a formal connection between network architecture (particularly the choice of activation function) and explanation complexity.

### Core Motivation

**Key Insight**: The "noise" in VanillaGrad explanations is not genuine noise but rather a structural property arising from the network architecture—specifically, the sharp transitions introduced by ReLU. By establishing a formal link in the frequency domain between the tail behavior of a network's power spectrum and that of the gradient's spatial power spectrum, one can simultaneously understand and control explanation complexity while quantifying the faithfulness loss introduced by post-hoc processing.

## Method

### Overall Architecture

The method constructs a unified analytical framework from a spectral perspective, comprising three core components:
1. Measuring explanation complexity via the Tail of the Spatial Power Spectrum (TSPS)
2. Establishing a formal connection between the network's Tail of the Power Spectrum (TPS) and the gradient's TSPS
3. Defining the explanation gap in the Fourier domain to quantify faithfulness

### Key Designs

#### 1. **Expected Frequency (EF)**: Measuring Explanation Complexity

- **Function**: Quantifies the proportion of high-frequency components in an explanation via a weighted integral of the spatial power spectrum.
- **Mechanism**: The Expected Frequency is defined as:

$$\operatorname{EF}(e_f(x)) \coloneq \int \omega \operatorname{S}_{e_f(x)}(\omega) \, d\omega$$

where $\operatorname{S}$ denotes the spatial power spectrum of explanation method $e_f$. A lower EF indicates a smoother and simpler explanation in the spatial domain. For image data, a one-dimensional power spectrum is obtained via radial averaging in the frequency domain.

- **Design Motivation**: The more "noisy" an explanation, the heavier the tail of its spatial power spectrum. EF provides a concise and effective statistic for capturing tail behavior, and is jointly influenced by both the model and the explanation method.

#### 2. **Formal Connection Between Network TPS and Gradient TSPS**

- **Function**: Establishes a theoretical relationship between the tail behavior of the network's power spectrum and that of the input gradient's spatial power spectrum.
- **Mechanism**:

**Theorem 1 (Informal)**: In data domains with high input feature correlation (e.g., image data), given a trained neural network $f(x)$, the tail behavior of the power spectrum of $f(x)$ is proportional to the tail behavior of the spatial power spectrum of $\nabla f(x)$.

A key corollary follows from **Lemma 1**: convolving ReLU with a Gaussian yields a Smooth Parameterization (SP):

$$\xi = \phi * g_\beta$$

where $\beta$ is the Gaussian precision parameter, and as $\beta \to \infty$, standard ReLU is recovered. In practice, SoftPlus is used as an efficient approximation:

$$\text{SoftPlus}(x;\beta) = \frac{1}{\beta} \ln(1 + e^{\beta x}) \approx \text{ReLU} * g_\beta(x)$$

- **Design Motivation**: A heavier power spectrum tail implies stronger network dependence on high-frequency information, leading to more complex gradient explanations. Controlling the smoothness of the activation function directly controls explanation complexity while maintaining a zero explanation gap.

#### 3. **Explanation Gap**: Quantifying Faithfulness

- **Function**: Measures the degree to which a post-hoc explanation method deviates from the original model by introducing a surrogate model.
- **Mechanism**: The explanation gap is defined as the $L^2$ norm of the difference between the gradients of the original and surrogate models:

$$\mathcal{G}(f, \tilde{f}) = \int_{x \in \mathcal{X}} \|\nabla f(x) - \nabla \tilde{f}(x)\|_2^2 \, dx$$

Applying Parseval's theorem converts this to the Fourier domain:

$$\mathcal{G}(f, \tilde{f}) \approx \int_{\omega \in \mathcal{F}_{\text{high}}} \omega^2 \|\hat{f}(\omega) - \hat{\tilde{f}}(\omega)\|^2 \, d\omega$$

Since surrogate models suppress high-frequency components, the gap is dominated by the high-frequency portion. The EF difference is ultimately adopted as a proxy measure:

$$\Delta \operatorname{EF}(e_f) \coloneq |\operatorname{EF}(\nabla f) - \operatorname{EF}(e_f)|$$

- **Design Motivation**: Post-hoc methods are essentially low-pass filters that suppress high frequencies to achieve visual smoothness. The explanation gap quantifies this degree of "engineering." VanillaGrad has a zero gap (as it creates no surrogate model), whereas methods such as GradCAM incur the largest gap.

### Loss & Training

This paper introduces no new loss function design. The core technical contributions are:
- Replacing ReLU with SoftPlus($\beta$) during training, with $\beta$ controlling explanation complexity.
- Applying early stopping with a validation accuracy upper bound to ensure comparable training budgets across different smoothing parameters.
- Normalizing gradient magnitudes per pixel via rank normalization using the inverse transform method.

## Key Experimental Results

### Main Results

**EF and Explanation Gap of Different Explanation Methods on Imagenette-CNN**:

| Method | ReLU: EF + ΔEF | SP(β=0.9): EF + ΔEF |
|--------|----------------|---------------------|
| VanillaGrad | .390 + Δ.000 | .202 + Δ.000 |
| SmoothGrad | .286 + Δ.104 | .196 + Δ.005 |
| IntGrad | .396 + Δ.007 | .205 + Δ.003 |
| GuidedBP | .300 + Δ.090 | .202 + Δ.000 |
| DeepLift | .394 + Δ.005 | .204 + Δ.002 |
| GradCAM | .293 + Δ.097 | .177 + Δ.025 |

**EF and Explanation Gap on ImageNet (×10⁴)**:

| Method | ResNet50: EF + ΔEF | ViT-B16: EF + ΔEF |
|--------|--------------------|--------------------|
| VanillaGrad | .263 + Δ.000 | .222 + Δ.000 |
| SmoothGrad | .247 + Δ.017 | .221 + Δ.001 |
| GradCAM | .133 + Δ.130 | .181 + Δ.041 |

### Ablation Study

| Configuration | Observation | Note |
|---------------|-------------|------|
| Increasing β (→ReLU) | EF increases monotonically | Confirms ReLU leads to heavier power spectrum tails |
| SP(β=0.9) + VG | EF=.202, ΔEF=0 | Low complexity achievable at zero gap |
| ReLU + SmoothGrad | EF=.286, ΔEF=.104 | Post-hoc processing reduces complexity but introduces large gap |
| ViT + GELU | Low EF and low variance | ViT architecture more influential than activation function |
| Varying network depth | Spectral decay rate nearly unchanged | Depth has limited effect on explanation complexity |
| Varying learning rate | Curve shape varies but tail behavior unchanged | Learning rate affects details but not overall trends |

### Key Findings

1. **ReLU is the root cause of explanation complexity**: EF increases monotonically with $\beta$, confirming that the sharp transitions introduced by ReLU are the fundamental source of "noisy" gradient explanations.
2. **GradCAM incurs the largest gap**: Across all architectures, GradCAM consistently introduces the largest explanation gap, representing a significant faithfulness risk.
3. **SP provides a better trade-off**: Using SP(β=0.9), EF can be reduced from .390 to .202 at zero explanation gap, whereas SmoothGrad can only reduce it to .286 with a gap of .104.
4. **ViT is inherently smoother**: ViT employs GELU activations and its attention mechanism provides a global receptive field, resulting in lower variance across post-hoc methods.

## Highlights & Insights

1. **Unified Framework**: For the first time, explanation complexity and faithfulness are jointly quantified within a unified spectral framework, with both metrics sharing a consistent definition.
2. **Root Cause Analysis and Control**: Beyond identifying ReLU as the root cause of explanation complexity, the paper provides a practical approach to controlling complexity via smooth parameterization.
3. **Explanation Gap Concept**: A formal definition of the "explanation gap" is introduced, providing a tool for assessing the implicit faithfulness risks of post-hoc methods.
4. **Hyperparameter-Free Metrics**: EF and ΔEF do not depend on external factors such as baseline selection or removal ordering, offering cleaner intuition.
5. **Cross-Architecture Validation**: Theoretical predictions are validated consistently across CNN, ResNet, and ViT architectures.

## Limitations & Future Work

1. **Theoretical Reliance on Kernel Methods**: The connection to kernel methods may break down in deep networks, where the kernel perspective may not be intuitive along the depth dimension.
2. **Spatial Frequency Only**: The spectral analysis is conducted solely in the spatial domain, potentially overlooking information along other dimensions.
3. **Accuracy-Complexity Trade-off**: Smooth parameterization of ReLU may sacrifice classification accuracy, though the paper mitigates this by imposing a validation accuracy upper bound.
4. **Restricted to Classifiers**: The analytical framework is built on scalar classifiers $f: \mathbb{R}^n \to \mathbb{R}$; generalization to other tasks is not discussed.
5. **Theoretical Scope**: Theorem 1 requires high input feature correlation and may not hold for low-correlation data.

## Related Work & Insights

- **Distinction from SmoothGrad, GradCAM, etc.**: This paper does not propose a new explanation method but rather an analytical framework for evaluating the complexity-faithfulness trade-off of existing methods.
- **Relationship to Inherently Interpretable Models (e.g., B-cos Networks)**: SP can be viewed as an alternative "by-design" interpretability strategy that directly reduces a network's high-frequency dependence.
- **Uniqueness of the Spectral Perspective**: Unlike prior work that imposes spectral properties via optimization, this paper uses the spectral perspective purely for measurement and analysis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified analytical framework from a spectral perspective is creative, and the definitions of EF and the explanation gap are concise and powerful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validation is conducted across multiple datasets and architectures, with ablation studies covering depth, learning rate, and other factors.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous, the narrative is clear, and theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐ — Provides an important analytical tool for the interpretability community, though practical application scenarios warrant further exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training](failure_cases_are_better_learned_but_boundary_says_sorry_facilitating_smooth_per.md)
- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](../../AAAI2026/others/forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](../../NeurIPS2025/others/the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)

</div>

<!-- RELATED:END -->
