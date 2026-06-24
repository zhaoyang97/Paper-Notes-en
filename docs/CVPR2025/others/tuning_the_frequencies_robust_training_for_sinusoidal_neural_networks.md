---
title: >-
  [Paper Note] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks
description: >-
  [CVPR 2025][Implicit Neural Representations] TUNER is proposed, a sinusoidal MLP training scheme based on the amplitude-phase expansion theory of Bessel functions. By expanding hidden neurons into Fourier series of integer combinations of input frequencies, robust frequency initialization and in-training band-limiting control are achieved, significantly improving the convergence stability and reconstruction quality of implicit neural representations.
tags:
  - "CVPR 2025"
  - "Implicit Neural Representations"
  - "Sinusoidal Networks"
  - "Frequency Control"
  - "Band-limiting"
  - "Fourier Series"
date: 2026-05-08
content_hash: cc1eb4f1e2a36230
---

# Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks

**Conference**: CVPR 2025  
**arXiv**: [2407.21121](https://arxiv.org/abs/2407.21121)  
**Code**: None  
**Area**: Signal & Communications  
**Keywords**: Implicit Neural Representations, Sinusoidal Networks, Frequency Control, Band-limiting, Fourier Series

## TL;DR
TUNER is proposed, a sinusoidal MLP training scheme based on the amplitude-phase expansion theory of Bessel functions. By expanding hidden neurons into Fourier series of integer combinations of input frequencies, robust frequency initialization and in-training band-limiting control are achieved, significantly improving the convergence stability and reconstruction quality of implicit neural representations.

## Background & Motivation

**Background**: Sinusoidal MLPs (such as SIREN) have become the mainstream implicit neural representation (INR) methods for low-dimensional signals due to their smoothness and high representation capability, and are widely used to encode signals such as images, audio, SDFs, and displacement fields.

**Limitations of Prior Work**: The initialization and training of sinusoidal MLPs remain empirical. SIREN randomly initializes input frequencies within a range, which may introduce unwanted high frequencies, leading to overfitting and noisy reconstructions. BACON uses multiplicative filter networks (MFNs) to hard-truncate the spectrum for band-limiting, but this causes ringing artifacts, and MFNs lack non-linear activations, making it difficult to efficiently represent fine details.

**Key Challenge**: The composition of sinusoidal layers generates a large number of new frequencies, but existing methods lack a theoretical understanding of this frequency generation process—it is unclear how layer compositions generate frequencies, how input frequencies determine the network spectrum, and how to control the frequency range during training.

**Goal**: To establish a frequency generation theory for sinusoidal MLPs and, based on this, design a robust initialization scheme and an in-training band-limiting control mechanism.

**Key Insight**: Leveraging a generalization of the Jacobi-Anger identity, the composition of sinusoidal layers is expanded into a Fourier-like series, revealing that hidden neurons can be represented as sums of sines of integer combinations of input frequencies, with amplitudes given by Bessel functions.

**Core Idea**: The spectrum of a sinusoidal MLP is entirely determined by the input frequency $\omega$ (frequency = integer linear combination of input frequencies), and the amplitude is controlled by hidden weights through Bessel functions. Therefore, initializing input frequencies is equivalent to spectral sampling, and constraining hidden weights is equivalent to band-limiting control.

## Method

### Overall Architecture
TUNER targets a three-layer sinusoidal MLP $f(\mathbf{x}) = \mathbf{C} \circ \mathbf{S} \circ \mathbf{D}(\mathbf{x}) + e$, where $\mathbf{D}$ is the input layer (projecting coordinates into a list of sinusoids), $\mathbf{S}$ is the hidden sinusoidal layer, and $\mathbf{C}$ is the linear output layer. TUNER consists of two components: (1) input frequency initialization based on Fourier series theory, which selects appropriate integer frequencies within the band-limit; and (2) hidden weight constraints based on the amplitude upper bounds of Bessel functions, which clip hidden weights during training to control high-frequency amplitudes.

### Key Designs

1. **Amplitude-Phase Expansion (Theorem 1)**:

    - Function: Precisely expands hidden neurons into a sine series of integer combinations of input frequencies.
    - Mechanism: Proves that each hidden neuron $h_i(\mathbf{x}) = \sin(\sum_j W_{ij} \sin(\omega_j \mathbf{x} + \varphi_j) + b_i)$ can be expanded as $h_i(\mathbf{x}) = \sum_{\mathbf{k} \in \mathbb{Z}^m} \alpha_\mathbf{k} \sin(\beta_\mathbf{k} \mathbf{x} + \lambda_\mathbf{k})$, where the frequencies are $\beta_\mathbf{k} = \langle \mathbf{k}, \omega \rangle$ (integer linear combinations of input frequencies) and amplitudes are $\alpha_\mathbf{k} = \prod_j J_{k_j}(W_{ij})$ (products of Bessel functions).
    - Design Motivation: This expansion rigorously explains for the first time why the composition of sinusoidal layers can significantly increase representation capacity—$m$ input frequencies can generate $(2B+1)^m - 1)/2$ non-zero frequencies under a truncation order $B$.

2. **Spectral Sampling Initialization**:

    - Function: Initializes input layer frequencies so that the frequencies generated by the network cover the complete spectrum of the target signal.
    - Mechanism: Restricts input frequencies to integer frequencies $\omega_j \in \frac{2\pi}{p}\mathbb{Z}^d$ (ensuring periodicity) and freezes them during training. A hybrid sampling strategy is adopted: dense sampling in low-frequency regions (since signal energy is concentrated there) and sparse sampling in high-frequency regions (leveraging layer compositions to fill in). This is equivalent to strategic sampling of the target spectrum.
    - Design Motivation: Random initialization may generate excessive high frequencies, leading to overfitting, or miss key frequencies, causing reconstruction failure. Integer frequencies guarantee the orthogonality of the Fourier series.

3. **In-Training Band-Limiting Control (Theorem 2)**:

    - Function: Constrains the network spectrum within a specified band-limit during training.
    - Mechanism: Proves the amplitude upper bound $|\alpha_\mathbf{k}| \leq \prod_j (|W_{ij}|/2)^{|k_j|} / |k_j|!$. When $|W_{ij}| < 2$, the amplitude corresponding to higher-order $k_j$ decays exponentially; the smaller $|W_{ij}|$ is, the faster the decay. Thus, by clipping hidden weights $|W_{ij}| \leq c$ ($c < 2$) during training, high-frequency components can be effectively suppressed, achieving soft band-limiting filtering.
    - Design Motivation: The hard truncation of BACON leads to ringing artifacts. The soft decay based on Bessel functions provides smoother spectral control, avoiding the Gibbs phenomenon.

### Loss & Training
Training is conducted using standard MSE loss. Input frequencies are frozen and not trained; only the hidden layer weights $W$, biases $b$, and the output layer $C$ are optimized. After each gradient update, $W$ is clipped to maintain the band-limit. The Adam optimizer is used, with typical training for 3000 epochs.

## Key Experimental Results

### Main Results

| Method | Dataset | PSNR | Characteristics |
|------|--------|------|------|
| SIREN | Kodak | Lower | Uniformly random initialization, prone to noise overfitting |
| FFM | Kodak | Medium | Fourier Feature Mapping |
| BACON | Kodak | Medium | Hard-truncated band-limiting, suffers from ringing artifacts |
| TUNER | Kodak | Optimal | Fast convergence, no noise/artifacts |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Odd-only frequency initialization | Poor reconstruction quality | Missing even frequencies leads to period-doubling |
| Adding (1,0),(0,1) frequencies | Significant improvement | Fundamental frequencies guarantee complete spectral coverage |
| Uniformly random initialization | Noisy gradients | High-frequency overfitting |
| TUNER initialization | Clean gradients | Smooth gradients obtained without gradient supervision |

### Key Findings
- The choice of input frequencies is critical to the representation capability of the network—missing key fundamental frequencies prevents learning the full spectrum.
- Soft band-limiting provided by hidden weight clipping outperforms the hard truncation of BACON, avoiding ringing artifacts.
- The network under TUNER initialization yields clean (noise-free) gradients of the reconstructed signal even without gradient supervision, indicating that frequency control effectively prevents overfitting.
- The hybrid sampling strategy of dense-at-low-frequencies and sparse-at-high-frequencies aligns with the energy distribution patterns of natural signals.

## Highlights & Insights
- **Elegant Theoretical Framework**: By generalizing the Jacobi-Anger identity to multivariate and multi-layer scenarios, a rigorous connection between sinusoidal MLPs and Fourier series is established. This theoretical contribution is valuable independent of the practical application itself.
- **Initialization as Spectral Sampling**: The perspective of transforming the network initialization problem into a frequency-domain sampling problem is highly elegant, providing a new conceptual framework for understanding sinusoidal networks.
- **Practical Upper Bound for Bessel Functions**: The amplitude upper bound provided by Theorem 2 is concise and practical (relying only on the absolute values of the weights), directly translatable into in-training weight-clipping rules.

## Limitations & Future Work
- The theoretical analysis primarily targets three-layer networks; although generalizable, the expansion complexity of deeper networks grows exponentially.
- Input frequencies are frozen during training, limiting the network's capability to adaptively adjust its spectrum.
- Experiments are mainly validated on 2D image reconstruction, leaving high-dimensional applications such as 3D scenes (NeRF, SDF) insufficiently explored.
- Weight clipping is a heuristic soft band-limit rather than a precise spectral truncation, which may not be sufficiently strict for scenarios requiring extreme band-limiting.

## Related Work & Insights
- **vs SIREN**: SIREN randomly initializes input frequencies and does not control frequency growth during training, which easily generates high-frequency noise. TUNER fundamentally addresses this issue through integer frequency initialization and weight clipping.
- **vs BACON**: BACON uses MFNs for hard-truncated band-limiting, which has the advantage of theoretically precise truncation but suffers from ringing artifacts and lacks non-linear activations. TUNER preserves the non-linear representing power of sinusoidal activations, replacing hard truncation with soft decay.
- **vs FFM (Fourier Feature Mapping)**: FFM uses random Fourier features for input mapping without using sinusoidal activations. TUNER's theory demonstrates that the composition of layers with sinusoidal activations inherently generates rich frequencies, providing a unified perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Outstanding theoretical contribution; the amplitude-phase expansion and the Bessel upper bound serve as entirely new mathematical tools.
- Experimental Thoroughness: ⭐⭐⭐ Sufficient image reconstruction experiments, but lacks performance on more diverse application domains like 3D or video.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations, intuitive visualizations, and overall clear organization.
- Value: ⭐⭐⭐⭐ Provides the first comprehensive theoretical framework for the field of sinusoidal INRs, holding profound academic and practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)

</div>

<!-- RELATED:END -->
