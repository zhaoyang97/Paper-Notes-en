---
title: >-
  [Paper Note] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations
description: >-
  [ICLR 2026][Image Generation][Implicit neural representations] Through harmonic distortion analysis and Chebyshev polynomial approximation, this paper rigorously proves that odd/even symmetric activation functions exhibit systematic attenuation in post-activation spectra. It proposes modulating activation functions with complex sinusoidal terms $e^{j\zeta x}$ to preserve full spectral support, and introduces the COSMO-RC activation function alongside a regularized prior embedder architecture. The method achieves an average PSNR gain of +5.67 dB over the strongest baseline on Kodak image reconstruction and +3.45 dB on NeRF.
tags:
  - ICLR 2026
  - Image Generation
  - Implicit neural representations
  - activation function design
  - spectral bias
  - Chebyshev polynomials
  - complex sinusoidal modulation
date: 2026-05-08
content_hash: 46933abc85f3d25d
---

# COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations

**Conference**: ICLR 2026
**arXiv**: [2505.11640](https://arxiv.org/abs/2505.11640)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Implicit neural representations, activation function design, spectral bias, Chebyshev polynomials, complex sinusoidal modulation

## TL;DR

Through harmonic distortion analysis and Chebyshev polynomial approximation, this paper rigorously proves that odd/even symmetric activation functions exhibit systematic attenuation in post-activation spectra. It proposes modulating activation functions with complex sinusoidal terms $e^{j\zeta x}$ to preserve full spectral support, and introduces the COSMO-RC activation function alongside a regularized prior embedder architecture. The method achieves an average PSNR gain of +5.67 dB over the strongest baseline on Kodak image reconstruction and +3.45 dB on NeRF.

## Background & Motivation

**Background**: Implicit neural representations (INRs) use MLPs to map continuous coordinates to signal values (e.g., image pixels, 3D occupancy), with the choice of activation function being the core design degree of freedom. SIREN uses sinusoidal activations, WIRE uses wavelets, and Gaussian-based methods use Gaussian functions; further variants include FINER (variable-frequency sinusoids) and INCODE (prior embedders). While these methods each excel on different tasks, a unified theoretical explanation of why certain activations outperform others—and where their effectiveness breaks down—has been lacking.

**Limitations of Prior Work**: INRs face three core challenges: (1) spectral bias, whereby networks are inherently insensitive to high-frequency signal components, leading to blurry reconstructions; (2) poor noise robustness, causing overfitting to noise during denoising; and (3) difficulty in simultaneously capturing local details and global structure. Most existing activation function designs are based on empirical comparisons, lacking a systematic analytical framework grounded in the spectral domain.

**Key Challenge**: Nonlinear activation functions broaden the input spectrum (blueshift effect), enabling networks to represent high-frequency components. However, if an activation function possesses odd or even symmetry—as nearly all commonly used activations do—half of the coefficients in its Chebyshev polynomial expansion are identically zero, resulting in systematic attenuation of the post-activation spectrum. The network's expressive capacity is thus needlessly halved.

**Goal**: (1) Reveal the theoretical root cause of spectral attenuation in existing INR activation functions; (2) propose a general remedy—complex sinusoidal modulation; (3) design a concrete optimal activation function, COSMO-RC, and validate its superiority.

**Key Insight**: The authors begin from harmonic distortion analysis, expanding activation functions in Chebyshev polynomials and observing that the alternating vanishing of odd/even coefficients is an inevitable consequence of symmetry—and the precise mathematical source of spectral attenuation. This analytical perspective had not previously been examined.

**Core Idea**: Modulating the activation function with a complex exponential $e^{j\zeta x}$ breaks odd/even symmetry, ensuring that the real and imaginary parts of the Chebyshev coefficients cannot vanish simultaneously, thereby preserving full frequency support in the post-activation spectrum.

## Method

### Overall Architecture

The input consists of signal coordinates (e.g., 2D pixel coordinates $(x,y)$), which pass through a 5-layer MLP with 256 neurons per layer, each using the COSMO-RC complex-valued activation function. Layer outputs are normalized onto the unit circle in the complex plane to maintain training stability; the final layer extracts the real part to produce signal values (e.g., RGB pixel values). Additionally, a prior embedder based on the first five layers of ResNet-34 extracts features from the input signal and maps them to the activation hyperparameters $(T, \zeta)$, with sigmoid regularization constraining the parameter range. The entire system is trained end-to-end with a standard MSE loss.

### Key Designs

1. **Theoretical Discovery of Spectral Attenuation and Chebyshev Analysis**:

    - Function: Reveals the mathematical root cause of limited expressive capacity in existing activation functions.
    - Mechanism: Any activation function $\phi(x)$ is expanded in Chebyshev polynomials as $\phi(x) = \sum_{n=0}^{\infty} a_n T_n(x)$, where $T_n$ are Chebyshev polynomials of the first kind. Based on the action of nonlinear layers on the spectrum (formula $z' = \sum_{i=0}^{K} \alpha_i \bigotimes_{l=0}^{i} z$), the magnitude of each coefficient $\alpha_i$ directly determines the blueshift effect at the corresponding order. The authors rigorously prove: for even-symmetric functions $f(x) = f(-x)$, all odd-order coefficients $a_n = 0$ (odd $n$); for odd-symmetric functions $f(x) = -f(-x)$, all even-order coefficients $a_n = 0$ (even $n$). Consequently, raised cosine (even-symmetric) and sinusoidal (odd-symmetric) activations each have half their spectral contributions reduced to zero.
    - Design Motivation: Prior analyses of the blueshift effect focused solely on the rate of coefficient magnitude decay, never noticing the symmetry-induced systematic zeroing. This finding explains why all symmetric activation functions share a ceiling on expressive capacity.

2. **Complex Sinusoidal Modulation (COSMO)**:

    - Function: Breaks the odd/even symmetry of activation functions to restore full spectral support.
    - Mechanism: The activation function is modulated as $g(x) = \phi(x) \cdot e^{j\zeta x}$. Expanding the complex exponential yields $g(x) = \phi(x)(\cos\zeta x + j\sin\zeta x)$, where the Chebyshev coefficients of the real part $g_r(x) = \phi(x)\cos\zeta x$ and imaginary part $g_i(x) = \phi(x)\sin\zeta x$ are nonzero at complementary odd/even orders. Key theorem: when the real-part coefficients $a_n = 0$, the imaginary-part coefficients $b_n \neq 0$, and vice versa, so the complex coefficient $a_n + jb_n$ is never identically zero. This guarantees that every frequency order contributes to the post-activation spectrum.
    - Design Motivation: The complex exponential is neither odd nor even, and multiplying by it immediately breaks the symmetry of the original activation. This constitutes a minimally invasive fix—leaving the basic shape of the activation unchanged and merely adding a phase rotation term.

3. **COSMO-RC Activation Function**:

    - Function: A concrete activation function implementation derived from theoretically optimal choices.
    - Mechanism: Among all candidate activations, the raised cosine function exhibits the slowest Chebyshev coefficient decay, implying the strongest blueshift effect. Combining the raised cosine with complex sinusoidal modulation yields COSMO-RC: $\phi(x) = \frac{1}{T}\text{sinc}(\frac{x}{T}) \frac{\cos(\pi\beta x/T)}{1-(2\beta x/T)^2} \cdot e^{2\pi\zeta x j}$. The roll-off factor $\beta = 0.05$ is fixed, while the bandwidth parameter $T$ and frequency-shift parameter $\zeta$ are learnable. Each layer produces complex-valued outputs normalized to the unit circle for training stability (phase preserved, magnitude normalized); the final layer takes the real part as output.
    - Design Motivation: The raised cosine originates from pulse-shaping filters in communications, possessing compact support and slowly decaying sidelobes, which implies retention of higher-order components under the Chebyshev basis. Combined with complex sinusoidal modulation, it is the optimal choice both theoretically and empirically.

### Loss & Training

Training uses a standard MSE loss $L = \mathbb{E}_{x \in X} \|f_\theta(x) - \hat{S}_x\|^2$, with the Adam optimizer at a learning rate of 0.01 and decay rate of 0.01. The prior embedder uses the first five layers of ResNet-34 for 2D image tasks and ResNet3D-18 for 3D occupancy tasks; outputs are mapped via an MLP to a latent variable in $(2,4)$, then projected to predefined ranges $T \in [0,10]$, $\zeta \in [0,3]$ via sigmoid regularization $\theta = a + (b-a) \cdot \sigma(\hat{\theta})$. This mechanism adaptively adjusts activation parameters at each iteration, eliminating the need for manual grid search. The authors note that equivalent performance can be achieved without the prior embedder, but requires more rigorous parameter grid search.

## Key Experimental Results

### Main Results

| Task | Dataset | COSMO-RC | Prev. SOTA | Gain |
|------|---------|----------|------------|------|
| Image reconstruction | Kodak (24 images) | **41.24 dB** | INCODE 35.57 dB | **+5.67 dB** |
| Image denoising | DIV2K (Poisson noise) | Best | INCODE | **+0.46 dB** |
| Super-resolution 2× | DIV2K | **34.03 dB** / 0.96 SSIM | FINER 32.94 / 0.91 | +1.09 dB |
| Super-resolution 4× | DIV2K | **30.42 dB** / 0.95 SSIM | INCODE 29.96 / 0.85 | +0.46 dB |
| Super-resolution 6× | DIV2K | **27.66 dB** / 0.93 SSIM | FINER 27.02 / 0.80 | +0.64 dB |
| NeRF novel view synthesis | Lego (200 test views) | **29.50 dB** | INCODE 26.05 dB | **+3.45 dB** |
| Image inpainting | Celtic spiral (20% sampling) | Marginally above SOTA | — | Slight lead |
| 3D occupancy | Lucy (Stanford) | Highest IOU | — | Slight lead |

### Ablation Study

| Configuration (Kodak 22, 1000 epochs) | PSNR (dB) | Notes |
|---------------------------------------|-----------|-------|
| Width 256 × 3 layers (full model) | 39.57 | Default config, balanced efficiency and accuracy |
| Width 512 × 4 layers | **52.00** | Strongest config, validates scalability |
| Width 64 × 2 layers | 28.52 | Minimal config, significant performance drop |
| Raised cosine w/o complex modulation | ~35 dB (Fig. 2b) | Large degradation without modulation, validates core contribution |
| COSMO-RC w/o prior embedder | Equivalent (requires grid search) | Embedder does not affect the upper bound but greatly simplifies tuning |

### Computational Efficiency Comparison

| Method | Params (K) | Forward GFLOPs | Training time (s/it) | PSNR (dB) |
|--------|-----------|----------------|----------------------|-----------|
| SIREN | 199 | 25.9 | 0.222 | 32.9 |
| FINER | 199 | 25.9 | 0.270 | 36.4 |
| INCODE | 437 | 38.7 | 0.435 | 36.2 |
| WIRE | 100 | 13.0 | 0.645 | 32.5 |
| **COSMO-RC** | 437 | 38.7 | **3.500** | **45.1** |

### Key Findings

- **Complex sinusoidal modulation is the core contribution**: Removing the complex modulation causes a large PSNR drop in the raised cosine activation (~−6 dB), confirming that the theoretical analysis of spectral completeness is not merely academic but operationally decisive.
- **Raised cosine is the optimal basis**: It exhibits the slowest Chebyshev coefficient decay among all candidate activations, providing the strongest blueshift—validated both theoretically and empirically.
- **Excellent network scalability**: A width-512 × 4-layer configuration achieves 52 dB reconstruction accuracy, indicating that the expressive capacity ceiling of COSMO-RC is far from reached.
- **Computational cost is the primary trade-off**: COSMO-RC trains approximately 8× slower than INCODE (3.5 s vs. 0.435 s/it), attributable to complex arithmetic and the additional overhead of the prior embedder. Given the +8.9 dB performance gain, however, this trade-off is fully acceptable in offline scenarios.
- **Reduced advantage on structurally simple tasks**: Only marginal gains are observed on image inpainting and 3D occupancy, suggesting that spectral attenuation is less problematic for low-frequency-dominated signals.

## Highlights & Insights

- **Theory-driven activation function design paradigm**: The chain from Chebyshev analysis → discovery of symmetry-induced spectral attenuation → repair via complex modulation forms a complete theory–design–validation pipeline. This paradigm is transferable to other network designs requiring spectral modeling (e.g., PDE solvers, audio synthesis).
- **Complex exponential modulation as a minimally invasive fix**: Without altering the fundamental shape of the activation function, adding a single phase rotation term suffices to break symmetry—making it plug-and-play applicable to any existing INR activation.
- **Regularization strategy of the prior embedder**: Projecting parameters to a bounded interval via sigmoid is more stable than the unconstrained optimization in INCODE, while preserving end-to-end trainability.

## Limitations & Future Work

- **Computational efficiency is the primary weakness**: COSMO-RC's training throughput is only 1/10 of SIREN's (33 vs. 350 GFLOPs/s), with both complex arithmetic and the prior embedder contributing overhead. Distillation into a real-valued network or approximate real-valued modulation schemes merit investigation.
- **Discarding the imaginary part at the final layer may lose information**: The entire network operates in the complex domain, yet only the real part is used as output. The imaginary part encodes meaningful phase information (e.g., image edges), and discarding it outright seems wasteful. An output strategy that exploits both real and imaginary parts warrants exploration.
- **Fixing $\beta = 0.05$ limits flexibility**: The roll-off factor of the raised cosine is fixed, but signals of varying frequency complexity may require different roll-off characteristics. Making $\beta$ a learnable parameter is a natural extension.
- **Marginal advantage on inpainting and 3D occupancy**: Spectral attenuation is inherently less severe for low-frequency-dominated tasks, so the gains from complex modulation are limited in these settings. The paper does not adequately discuss when complex modulation is unnecessary.
- **Prior embedder introduces task-specific dependencies**: ResNet-34 is used for images and ResNet3D-18 for 3D data; switching to a new modality requires selecting a new prior network, limiting generalizability.

## Related Work & Insights

- **vs. SIREN**: The sinusoidal activation is an odd-symmetric function, causing all even-order Chebyshev coefficients to vanish and attenuating the post-activation spectrum at every other order. COSMO-RC completely resolves this via complex modulation, achieving +8.3 dB on Kodak.
- **vs. WIRE**: Wavelet activations address SIREN's global artifact problem, but their Chebyshev coefficients decay rapidly (compact support leads to small high-order coefficients), resulting in weak blueshift. The raised cosine basis of COSMO-RC significantly outperforms wavelets in coefficient decay.
- **vs. INCODE**: The prior embedder concept is inherited from INCODE; COSMO-RC adds sigmoid regularization to constrain parameter ranges and replaces the activation function. At the same architectural scale, PSNR improves by approximately +5.6 dB, indicating that activation function improvement is more impactful than architectural improvement.
- **vs. FINER**: FINER introduces learnable frequency parameters to increase the flexibility of sinusoidal activations, but does not address spectral attenuation induced by odd symmetry. COSMO-RC comprehensively outperforms FINER on super-resolution.

## Rating

- Novelty: ⭐⭐⭐⭐ The symmetry-induced root cause of spectral attenuation is a genuinely new theoretical finding; the complex modulation scheme is backed by rigorous mathematical proof.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers six task categories—image reconstruction, denoising, super-resolution, inpainting, 3D occupancy, and NeRF—with ablations on computational efficiency and network scale.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, and the logic chain from analysis to design to validation is complete.
- Value: ⭐⭐⭐⭐ Provides a generalizable spectral analysis framework for INR activation function design; the complex modulation scheme is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[ICLR 2026\] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)
- [\[ICLR 2026\] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)
- [\[ICLR 2026\] Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](verification_of_the_implicit_world_model_in_a_generative_model_via_adversarial_s.md)
- [\[ICLR 2026\] Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)](amortising_inference_and_meta-learning_priors_in_neural_networks.md)

</div>

<!-- RELATED:END -->
