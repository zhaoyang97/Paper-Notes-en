---
title: >-
  [Paper Note] SONIC: Spectral Oriented Neural Invariant Convolutions
description: >-
  [ICLR 2026][Spectral convolution] SONIC transfers the core idea of state space models to the multi-dimensional frequency domain…
tags:
  - "ICLR 2026"
  - "Spectral convolution"
  - "orientation invariance"
  - "continuous parameterization"
  - "global receptive field"
  - "resolution adaptability"
date: 2026-05-08
content_hash: 12778bb828df2ade
---

# SONIC: Spectral Oriented Neural Invariant Convolutions

**Conference**: ICLR 2026
**arXiv**: [2601.19884](https://arxiv.org/abs/2601.19884)  
**Code**: N/A  
**Area**: Medical Imaging / Computer Vision
**Keywords**: Spectral convolution, orientation invariance, continuous parameterization, global receptive field, resolution adaptability

## TL;DR

SONIC transfers the core idea of state space models to the multi-dimensional frequency domain, defining a set of orientation-selective spectral transfer functions using 6 continuous parameters (amplitude, orientation, damping, oscillation, etc.), and mixing across channels via low-rank matrices $B$ and $C$. This yields a drop-in convolutional replacement operator that inherently possesses a global receptive field and resolution invariance. On 3D medical segmentation, it matches nnU-Net with nearly two orders of magnitude fewer parameters, and is also competitive on ImageNet.

## Background & Motivation

**Background**: The two dominant paradigms for image feature extraction are CNNs and ViTs. CNNs scan local patches with fixed-size kernels and require very deep networks to indirectly obtain global context; ViTs provide global connectivity via self-attention but lack structured spatial inductive biases, rely on explicit positional encodings, and incur quadratic computational complexity with respect to resolution. Spectral methods such as GFNet and FNO attempt to operate directly in the Fourier domain but suffer from notable shortcomings.

**Limitations of Prior Work**: GFNet's frequency-domain filters are tied to the discrete FFT grid—filter size equals the input spatial resolution, requiring retraining or interpolation whenever the resolution changes. FNO, while capable of handling continuous functions, lacks orientation awareness, treating all frequency directions uniformly and thus struggling to efficiently capture edges and textures in natural images. Parameter counts in existing spectral methods are also typically proportional to the frequency-domain dimensionality, which is particularly problematic for high-resolution 3D medical imaging.

**Key Challenge**: A fundamental tension exists between global receptive fields and resolution independence—traditional spatial convolutions are local but resolution-friendly, whereas frequency-domain methods are global but constrained to discrete grids. Furthermore, orientation selectivity is critical for visual tasks (analogous to orientation-selective neurons in the V1 cortex), yet existing spectral methods broadly neglect this property.

**Goal**: (1) How to design truly continuous, grid-independent convolutional parameterizations in the frequency domain? (2) How to introduce orientation-aware priors in the frequency domain while maintaining an extremely low parameter count? (3) How to enable a single architecture to seamlessly operate across 2D/3D inputs and varying resolutions?

**Key Insight**: The authors observe that the core mechanism of state space models (e.g., S4, Mamba)—generating global convolutional kernels from a small number of continuous parameters—can be generalized from 1D sequences to the multi-dimensional frequency domain. Each "mode" defines an orientation-selective transfer function in frequency space via a directional analytic resolvent function, and a small number of modes combined through low-rank matrices can cover a rich frequency-domain response.

**Core Idea**: Parameterize orientation-selective global convolutional kernels in the frequency domain using SSM-style continuous analytic functions, achieving an extremely parameter-efficient global receptive field via low-rank decomposition.

## Method

### Overall Architecture

The SONIC operator pipeline is as follows: given an input feature map $X \in \mathbb{R}^{C \times H \times W}$ (or a 3D volume), a multi-dimensional FFT is first applied to obtain $\hat{X}$; a continuously parameterized transfer function $\hat{K}(\omega)$ then performs pointwise multiplication in the frequency domain (i.e., frequency-domain convolution); finally, an IFFT maps the result back to the spatial domain. The key distinction of the transfer function is that it is not a learnable tensor bound to the grid resolution, but rather a continuous function evaluated at arbitrary frequency coordinates from a small set of analytic functions. The entire SONIC block serves as a direct replacement for spatial convolutional layers in standard ResNet / U-Net architectures.

### Key Designs

1. **Orientation-Selective Spectral Modes**:

    - Function: Each mode defines an orientation-selective transfer function in the frequency domain, selectively amplifying or suppressing frequency components along specific orientations.
    - Mechanism: Each mode is governed by 6 continuous parameters—amplitude $a$, decay rate $\sigma$, oscillation frequency $\omega_0$, orientation angle $\theta$ (2D) or orientation vector (3D), and phase offset. Together, these define an analytic response function (in resolvent form) in the frequency-orientation space: $H_k(\omega) = a_k / (\sigma_k + i(\omega \cdot \hat{n}_k - \omega_{0,k}))$, where $\hat{n}_k$ is the unit direction vector. Since $H_k$ is a continuous function of the frequency coordinate $\omega$, it can be evaluated directly on FFT grids of arbitrary resolution.
    - Design Motivation: The energy of natural images in the frequency domain is distributed non-uniformly across orientations (edges correspond to high-frequency components along specific directions). Orientation-selective modes can more efficiently encode these anisotropic structures, while the resolvent parameterization ensures continuity across resolutions.

2. **Low-Rank Channel Mixing Matrices $B$ and $C$**:

    - Function: Map $K$ shared spectral modes to $C$ input/output channels, enabling cross-channel feature mixing in the frequency domain.
    - Mechanism: On the input side, matrix $B \in \mathbb{R}^{K \times C_{in}}$ projects $C_{in}$ channels into the $K$-mode space; after frequency-domain multiplication by the transfer function, matrix $C \in \mathbb{R}^{C_{out} \times K}$ maps the result back to the output channels. The resulting frequency-domain transfer function is $\hat{K}(\omega) = C \cdot \text{diag}(H_1(\omega), \ldots, H_K(\omega)) \cdot B$. Since $K \ll C$ in general, this is a low-rank decomposition with parameter count $O(K \cdot (C_{in} + C_{out}) + 6K)$, far below the $O(C_{in} \cdot C_{out} \cdot k^d)$ of traditional convolutions.
    - Design Motivation: Spectral modes are largely shared across channels (e.g., the need for "horizontal edge detection" recurs across multiple channels), and the low-rank decomposition naturally captures this shared structure.

3. **Continuous Resolution Invariance**:

    - Function: The same set of parameters can be applied directly to inputs of different spatial resolutions without fine-tuning or interpolation.
    - Mechanism: Since the transfer function $H_k(\omega)$ is a continuous function of the frequency coordinate, when the input resolution changes (i.e., the FFT grid becomes denser or sparser), it suffices to re-evaluate the function at the new frequency coordinates. This stands in sharp contrast to GFNet, whose filters are learnable tensors of the same size as the FFT grid and require explicit handling of dimension mismatches upon resolution change.
    - Design Motivation: In medical imaging, data acquired with the same protocol but different scanners often exhibit significant resolution variation (e.g., MRI slice thickness ranging from 1 mm to 5 mm), making resolution invariance critical for deployment.

### Loss & Training

- Standard cross-entropy loss is used for classification; Dice + CE joint loss is used for 3D medical segmentation.
- SONIC blocks serve as direct replacements for convolutional layers in ResNet / U-Net, and the training strategy is compatible with the original architectures without requiring special initialization or learning rate schedules.
- Medical segmentation experiments follow the standard nnU-Net training protocol to ensure fair comparison.
- For ImageNet experiments, due to computational constraints, the authors trained for only 200k steps (rather than the full 300 epochs), which is nonetheless sufficient to demonstrate the method's competitiveness.

## Key Experimental Results

### Main Results — 3D Medical Image Segmentation

SonicNet (nnU-Net with spatial convolutions replaced by SONIC blocks) is compared against standard methods on multiple 3D medical segmentation benchmarks:

| Method | Dataset | Dice Score | Parameters | Notes |
|--------|---------|-----------|------------|-------|
| nnU-Net (3×3×3 conv) | PROMIS / Prostate158 | Baseline | ~31M | De facto standard for medical segmentation |
| SonicNet | PROMIS / Prostate158 | Matches or slightly exceeds nnU-Net | **~0.4M** | Nearly **80× fewer** parameters |
| ViT baseline | PROMIS / Prostate158 | Below nnU-Net | ~25M | Lacks spatial prior |
| SonicNet | Additional Benchmark 1 (high variability) | Competitive with SOTA | ~0.4M | nnU-Net Revisited recommended dataset |
| SonicNet | Additional Benchmark 2 (high variability) | Competitive with SOTA | ~0.4M | Multi-center high-variability scenario |

### Synthetic Benchmarks & ImageNet

| Experiment | Method | Key Result | Notes |
|-----------|--------|-----------|-------|
| SynthShape (geometric robustness) | CNN / ViT / SONIC | SONIC shows the smallest performance degradation under rotation and noise perturbation | Deterministic reproducible dataset |
| HalliGalli (global receptive field validation) | CNN / ViT / GFNet / SONIC | **Only SONIC completes the task correctly**, remaining robust under noise | Requires simultaneous perception of distant corner shapes |
| ImageNet (200k steps) | ResNet / ViT / GFNet / FNO / SONIC | SONIC is competitive with an order of magnitude fewer parameters | Comparison under limited training budget |
| ImageNet resolution downsampling | All methods from 224→lower resolutions | SONIC shows the flattest performance degradation curve, validating resolution invariance | Same model applied directly at different resolutions |

### Ablation Study

| Configuration | Key Change | Notes |
|--------------|-----------|-------|
| Full SonicNet | Baseline | Complete model |
| Remove orientation selectivity (isotropic modes) | Significant performance drop | Orientation awareness is a core contribution |
| Replace continuous parameterization with discrete learnable spectrum (≈GFNet) | Loss of resolution generalization | Continuous parameterization is the foundation of resolution invariance |
| Varying number of modes $K$ | Too few $K$ sacrifices expressiveness; too many yields diminishing returns | An optimal $K$ trade-off exists |
| Varying model scale (parameter scaling) | SONIC maintains strong performance at extremely small parameter counts | Parameter efficiency consistently outperforms spatial convolutions |

### Key Findings

- **Orientation selectivity is critical**: Removing orientation parameters leads to a significant performance drop, indicating that anisotropic frequency-domain priors are more effective than isotropic global filtering.
- **The HalliGalli experiment is the most compelling**: CNNs' local receptive fields fundamentally preclude tasks requiring global perception; ViT and GFNet have global receptive fields in theory but collapse under noise; only SONIC remains robust—demonstrating that its global receptive field is *effective* rather than merely *theoretical*.
- **Remarkable parameter efficiency**: Matching a 31M-parameter nnU-Net with ~0.4M parameters in the 3D medical setting implies substantial redundancy in traditional 3D convolutional kernels.
- **Verifiable resolution invariance**: In the ImageNet downsampling experiment, SONIC's performance degradation curve is noticeably flatter than all competing methods.

## Highlights & Insights

- **Bridge from SSMs to multi-dimensional frequency domain**: SONIC essentially extends the idea of "generating global convolutional kernels from a small number of continuous parameters" in S4/Mamba from 1D sequences to the frequency domain of multi-dimensional signals. This cross-domain transfer is highly natural—the core SSM formulation is itself a Laplace transform / resolvent, which directly corresponds to a frequency-domain transfer function. This opens a new channel for importing advances in sequence modeling into visual tasks.
- **HalliGalli as a litmus test for effective global receptive fields**: Many methods claim to have global receptive fields, but in practice the effective receptive field after deep stacking is far smaller than the theoretical value. The HalliGalli task is a cleverly designed litmus test—only models that can genuinely exploit long-range information pass it. This experimental design can be reused to evaluate other architectures that claim global capabilities.
- **Deployment advantages of continuous parameterization**: A single trained model can be deployed directly on inputs of varying resolutions without retraining or fine-tuning, which is highly practical in medical imaging where different devices and scanning protocols produce data at substantially different resolutions.

## Limitations & Future Work

- **SONIC blocks are purely linear**: Frequency-domain multiplication is fundamentally a linear operation, requiring IFFT → nonlinear activation → FFT between consecutive SONIC blocks. The overhead of dual FFT/IFFT passes is acceptable in shallow networks but may become a bottleneck in very deep architectures. Nonlinearity in the frequency domain remains an open problem.
- **Insufficient ImageNet validation**: Due to computational constraints, the authors trained for only 200k steps (far fewer than the standard 300 epochs), so ImageNet results can only claim "competitiveness" rather than "superiority." Conclusions require comparison under a full training budget.
- **Hybrid architectures unexplored**: The paper deliberately preserves the "purity" of SONIC without mixing it with spatial convolutions. In practice, using spatial convolutions in lower layers to capture local texture and SONIC in upper layers to capture global structure may yield a superior design.
- **No validation on detection / dense prediction**: Evaluation is limited to classification and segmentation, without covering object detection, instance segmentation, or other tasks requiring precise localization.
- **Selection of mode count $K$**: Currently relies on manual tuning without an automated method for determining the optimal $K$.

## Related Work & Insights

- **vs. GFNet**: GFNet also operates in the frequency domain, but uses learnable tensors of the same size as the FFT grid, requiring interpolation or fine-tuning upon resolution change. SONIC resolves this entirely via continuous parameterization and reduces parameter count from $O(HW)$ to $O(K)$.
- **vs. FNO (Fourier Neural Operator)**: FNO retains a fixed number of low-frequency components to approximate frequency-domain filtering, but has no orientation selectivity whatsoever. SONIC's resolvent modes provide anisotropic frequency responses, which are substantially more effective for orientation-sensitive visual tasks.
- **vs. nnU-Net**: nnU-Net is the de facto standard for 3D medical segmentation, relying on stacked 3×3×3 spatial convolutions. SONIC matches its performance with ~1/80th the parameters, suggesting substantial compressible redundancy in 3D spatial convolutions.
- **vs. S4ND / Mamba**: SONIC's theoretical foundations derive directly from the SSM family, but further incorporate directional decomposition and low-rank decomposition, making the same framework applicable to 2D/3D vision rather than 1D sequences alone.
- **Broader Implications**: The orientation-selective resolvent approach in SONIC can be extended to video (joint spatiotemporal frequency orientation), point clouds (spherical harmonic directional decomposition), and weather forecasting (spherical frequency-domain filtering).

## Rating

- Novelty: ⭐⭐⭐⭐ The transfer of SSM ideas to multi-dimensional frequency-domain orientation selectivity is creative, but the approach remains fundamentally a parameterization variant of frequency-domain multiplication.
- Experimental Thoroughness: ⭐⭐⭐ Medical segmentation validation is solid, but ImageNet training is incomplete, detection tasks are absent, and ablations are not sufficiently systematic.
- Writing Quality: ⭐⭐⭐ The core idea is clearly articulated, but the initial submission was criticized by multiple reviewers for poor readability; significant revisions have improved this.
- Value: ⭐⭐⭐⭐ Directly practical for multi-resolution deployment in medical imaging; the parameter efficiency advantage is particularly attractive in resource-constrained 3D settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](../../ICML2026/others/continual_learning_of_domain-invariant_representations.md)
- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](../../ICML2026/others/learning_permutation-invariant_macroscopic_dynamics.md)
- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/others/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)

</div>

<!-- RELATED:END -->
