---
title: >-
  [Paper Note] Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling
description: >-
  [ICML 2026][AI Safety][Panoramic Watermarking] TRIAD treats 360° panoramas as spherical signals and utilizes the tensor product of third-order Spherical Harmonic (SH) coefficients projected onto a trivial representation…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Panoramic Watermarking"
  - "SO(3) Equivariance"
  - "Spherical Harmonics"
  - "Bispectral Invariants"
  - "Tensor Product Coupling"
date: 2026-05-08
content_hash: 9338399589adfc88
---

# Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling

**Conference**: ICML 2026  
**arXiv**: [2605.26702](https://arxiv.org/abs/2605.26702)  
**Code**: The paper states "Code is available here," arXiv page pending confirmation  
**Area**: AI Security / Digital Watermarking / Spherical Signal Processing  
**Keywords**: Panoramic Watermarking, SO(3) Equivariance, Spherical Harmonics, Bispectral Invariants, Tensor Product Coupling

## TL;DR
TRIAD treats 360° panoramas as spherical signals and utilizes the tensor product of third-order Spherical Harmonic (SH) coefficients projected onto a trivial representation to obtain a **provably SO(3)-invariant** bispectral scalar. This allows the watermark to be embedded in high-order SH coefficients and retrieved from this invariant, maintaining near 100% bit accuracy under arbitrary 3D rotations without relying on data augmentation.

## Background & Motivation

**Background**: AIGC has made 360° panoramas (VR / Metaverse / World Model training data) highly accessible, creating an urgent need for digital watermarking for copyright tracking. Existing deep watermarking methods (StegaStamp / TrustMark / VINE / Robust-Wide, etc.) are almost entirely built on the translation equivariance assumption of CNNs, relying on data augmentation to resist geometric attacks.

**Limitations of Prior Work**: Panoramas are inherently signals defined on the sphere $\mathbb{S}^2$; when a user views them through an HMD, head rotation applies an SO(3) rotation to the signal. When panoramas are represented in Equirectangular Projection (ERP), a 3D rotation on the sphere becomes a highly non-linear pixel displacement with severe latitude-dependent distortion (polar stretching + large-scale texture shifts), which planar CNNs cannot align.

**Key Challenge**: SO(3) is a continuous infinite group with infinitely many rotations, which finite augmentation samples cannot fully cover. Robustness gained through "memorization" lacks theoretical guarantees, incurs high training costs, and collapses to random guessing when encountering unseen rotation angles. Furthermore, using the naturally rotation-invariant zero-order $c_0$ (DC component) of SH coefficients as a carrier would directly alter global brightness or color temperature, making the watermark immediately visible.

**Goal**: To construct a **provably SO(3)-invariant** watermarking framework for spherical signals—embedding the watermark in high-order SH subspaces (large capacity, visually hidden) while being able to retrieve it from a scalar that is **strictly invariant** to rotation.

**Key Insight**: Leveraging SO(3) representation theory. SH coefficients $c_l$ transform block-wise under rotation via Wigner-D matrices $c_l' = D^l(R) c_l$, without mixing different $l$. While the second-order power spectrum is invariant, it is "phase-blind" and loses directional information. Only the **third-order** tensor product $\mathcal{V}_{l_1} \otimes \mathcal{V}_{l_2} \otimes \mathcal{V}_{l_3}$ projected onto the trivial representation $\mathcal{V}_0$ is both strictly invariant and phase-retaining (i.e., the classical bispectrum).

**Core Idea**: Utilizing an asymmetric structure of "high-order SH embedding + third-order bispectrum extraction" to mathematically decouple "information capacity" from "rotation invariance"—embedding in the equivariant high-order subspace and extracting from the invariant zero-order scalar.

## Method

### Overall Architecture
TRIAD is an end-to-end encoder-decoder framework. The input is an $H \times 2H$ ERP panorama $x$ and a $k=32$ bit watermark $w \in \{0,1\}^k$; the output is a visually imperceptible watermarked image $\tilde{x}$. The decoder requires no alignment and recovers $\hat{w}$ by extracting bispectral invariant scalars directly in the SH domain.

The pipeline consists of four steps: (1) **Spherical Harmonic Transform (SHT)**: Lifting $x$ to SH coefficients $c = \{c_l\}_{l=0}^{l_{\max}}$, with a default $l_{\max}=16$; (2) **Equivariant Embedding**: Using an e3nn-based SO(3)-equivariant backbone $\Phi_{eq}$ to map $c$ to selected high-order subspaces $\mathcal{V}_{embed} = \mathcal{V}_6 \oplus \mathcal{V}_8 \oplus \mathcal{V}_{14}$, then integrating the watermark into these equivariant features through a parameterized tensor product $\text{TP}_{\vartheta_1}(u, w)$. The result is transformed back to the spatial domain via inverse SHT to obtain the residual $\Delta x$, which is modulated by a perceptual mask $M_{perc}$ and a geometric mask $M_{geo}$ before being added to the original: $\tilde{x} = x + M_{perc}(x) \odot M_{geo} \odot \Delta x$; (3) **Decoder SHT** and projection to the same $\mathcal{V}_{embed}$ to obtain $\hat{u}$; (4) **Third-order Invariant Projection**: Calculating learnable bispectrum scalars using a two-step tensor product $h = \text{TP}_{\vartheta_2}(\hat{u}, \hat{u})$ and $z = \text{TP}_{\vartheta_3}(h, \hat{u})|_{\mathcal{V}_0}$, followed by a lightweight MLP to decode $\hat{w}$.

### Key Designs

1. **Provably Invariant Spherical Bispectral Invariant via Third-order SH Tensor Product**:
    - **Function**: Extracts watermark information hidden in high-order (rotation-sensitive) SH coefficients into a scalar signal that is **strictly invariant** to any SO(3) rotation.
    - **Mechanism**: Decomposing the tensor product of three SH irreducible representations $\mathcal{V}_{l_1} \otimes \mathcal{V}_{l_2} \otimes \mathcal{V}_{l_3} = \bigoplus_l \mathcal{V}_l$ and selecting the $l=0$ trivial component yields $I = \sum_{l_i, m_i} C^{0,0}_{l_1 m_1\, l_2 m_2\, l_3 m_3}\, c_{l_1}^{m_1} c_{l_2}^{m_2} c_{l_3}^{m_3}$, where $C^{0,0}_{\dots}$ are Clebsch-Gordan coefficients derived from Wigner 3-j symbols. Theorem 4.1 proves that this scalar satisfies $I(R \cdot f) = I(f)$ and yields non-zero responses to perturbations in high-order coefficients (allowing information recovery), with the proof in Appendix A.1.
    - **Design Motivation**: While the zero-order $c_0$ is naturally invariant, it represents the global DC component; altering it causes visible shifts in brightness. The second-order power spectrum is invariant but is a phase-blind many-to-one mapping with very low capacity. Third-order coupling is the lowest-order construction that **simultaneously** preserves phase, maintains strict invariance, and avoids the DC component, effectively decoupling "high capacity" from "provable invariance."

2. **Higher-Order Spectral Embedding via Equivariant Backbone**:
    - **Function**: Embeds the watermark in high-frequency harmonic subspaces that are perceptually less noticeable yet provide sufficient energy for third-order coupling, ensuring visual fidelity.
    - **Mechanism**: Operations are restricted to $\mathcal{V}_{embed} = \bigoplus_{l \in \mathcal{L}_{embed}} \mathcal{V}_l$ ($l > 0$), where $\mathcal{L}_{embed} = \{6, 8, 14\}$ by default. Structured spectral features $u = \Phi_{eq}(c)$ are extracted using a 2-layer Gated Block equivariant backbone. The 32-bit watermark (as $\mathcal{V}_0$ scalar features) is injected into the irreducible subspaces of $u$ via an SO(3)-equivariant tensor product $\Delta u = \text{TP}_{\vartheta_1}(u, w)|_{\mathcal{V}_{embed}}$, relying on SH basis orthogonality to prevent leakage into other bands. Finally, it is written back to the spatial domain through inverse SHT and the masks $\tilde{x} = x + M_{perc}(x) \odot M_{geo} \odot \Delta x$.
    - **Design Motivation**: Lower $l$ orders (e.g., $\mathcal{V}_4$) provide high robustness but create visible artifacts; high $l$ orders (e.g., $\mathcal{V}_{16}$) are stealthy but susceptible to compression or aliasing. Experiments identify the middle-frequency set $\{6, 8, 14\}$ as the fidelity-robustness "sweet spot"—providing sufficient phase diversity for third-order coupling while avoiding ultra-high bands prone to attack attenuation. The ERP geometric mask $M_{geo}$ specifically suppresses modification amplitudes in polar regions.

3. **Decoupled Embed-Extract via Equivariant Tensor Product Chain**:
    - **Function**: The decoder does not require alignment, inverse rotation, or prior angle estimation; it extracts invariant scalars numerically identical to those at training time from any rotated panorama.
    - **Mechanism**: The decoder approximates the third-order bispectrum using a two-step chain: $h = \text{TP}_{\vartheta_2}(\hat{u}, \hat{u})|_{\mathcal{V}_{embed}}$ performs second-order coupling in the equivariant subspace, and $z = \text{TP}_{\vartheta_3}(h, \hat{u})|_{\mathcal{V}_0}$ completes the coupling and forces projection to the trivial representation. This is equivalent to a parameterized $\text{TP}(\hat{u}, \hat{u}, \hat{u}) \to \mathcal{V}_0$, where $\vartheta_2, \vartheta_3$ learn to select the CG channels most sensitive to the watermark. The resulting invariant vector is decoded by an MLP. The system is trained end-to-end with $\mathcal{L}_{total} = \lambda_m \mathcal{L}_{MSE}(x, \tilde{x}) + \lambda_{bce} \mathcal{L}_{BCE}(w, \hat{w})$, where $\lambda_{bce}=10$ and $\lambda_m$ linearly increases from 1 to 20.
    - **Design Motivation**: Unlike traditional methods that place "robustness" on the data/training side (augmentation), this work places it on the **architecture and representation side**. By strictly projecting to $\mathcal{V}_0$ at the decoder's final step, the output is strictly invariant to SO(3) regardless of network complexity. This is the fundamental reason for near-perfect resistance to arbitrary rotation with "zero augmentation."

## Key Experimental Results

### Main Results: General Distortion and Rotation Robustness
Trained on 10,000 panoContext + SUN360 panoramas, tested on 2,000 images at $512 \times 1024$ resolution with a 32-bit watermark. Comparison with 6 SOTA baselines:

| Method | Capacity (bit) | PSNR ↑ | SSIM ↑ | JPEG | Resize | Gauss Noise | Median | Mixed |
|------|-----------|--------|--------|------|--------|-------------|--------|-------|
| StegaStamp | 100 | 27.96 | 0.8986 | 0.973 | 0.812 | 0.961 | 0.879 | 0.978 |
| TrustMark | 100 | 40.83 | 0.9968 | 0.993 | 1.000 | 0.986 | 0.984 | 0.979 |
| Robust-Wide | 64 | 41.65 | 0.9921 | 0.997 | 0.998 | 0.989 | **1.000** | 0.992 |
| VINE | 100 | 36.33 | 0.9865 | **1.000** | 1.000 | **1.000** | 0.965 | 0.986 |
| **TRIAD (Ours)** | 32 | 39.22 | 0.9946 | 0.978 | **1.000** | 0.975 | **1.000** | 0.984 |

Rotation robustness (sampling axes uniformly from 0°–180° geodesic distance): All baselines without rotation augmentation collapsed to near 50% bit accuracy (random guess). Even with heavy augmentation, performance fluctuated wildly across angles. **TRIAD, with zero augmentation, maintained near 100% bit accuracy across the entire rotation range.**

### Ablation Study

| Configuration | PSNR / Bit Acc | Key Findings |
|------|---------------|----------|
| $\mathcal{V}_{embed} = \mathcal{V}_4$ (Single LF) | High Acc / Poor Visuals | Noticeable low-freq artifacts, significant PSNR drop |
| $\mathcal{V}_{embed} = \mathcal{V}_{16}$ (Single HF) | High PSNR / Low Acc | HF susceptible to compression/aliasing, retrieval degraded |
| $\mathcal{V}_{embed} = \mathcal{V}_6 \oplus \mathcal{V}_8 \oplus \mathcal{V}_{14}$ (Ours) | 39.22 / ≈100% | Multi-scale mid-freq direct sum balances robustness and stealth |
| $l_{\max}=16$, $\{6,8,14\}$ | 39.22 / 1.000 | Default config, rotation bit acc=100% |
| $l_{\max}=24$, $\{6,8,14, 16, 20\}$ | 38.46 / 1.000 | Capacity expansion; slight PSNR drop but invariance holds |
| $l_{\max}=28$, $\{6,8,14, 16, 20, 22\}$ | 37.19 / 1.000 | Excessively wide band hurts fidelity; $l_{\max}=16$ is the sweet spot |
| Power Spec, 2nd-order, 16 bit | — / 92.4% | Power spectrum is phase-blind and many-to-one |
| Power Spec, 2nd-order, 32 bit | — / 61.3% | Fails to converge beyond 16-bit capacity |
| **Bispectrum, 3rd-order, 64 bit** | — / **100%** | Phase retention is key to high-capacity invariant watermarking |

### Key Findings
- The advantage of the third-order bispectrum over the second-order power spectrum is **structural**: the power spectrum is a "phase-blind" many-to-one mapping that fails training for capacities $\ge 16$ bits. The bispectrum maintains 100% recovery even at 64 bits due to phase preservation.
- Rotation robustness is **architecturally guaranteed**, not learned through training. Forcing the decoder to project to $\mathcal{V}_0$ mathematically eliminates "unseen rotation angles."
- Robustness to untrained non-rotational distortions (JPEG / Resize / Noise / Blur / Median Filter) is unexpectedly strong. The authors provide an algebraic explanation: low-pass attenuation only smoothly scales frequency bands, while bispectral values are products of multiple coefficients, providing intrinsic robustness.

## Highlights & Insights
- **Shifting Robustness from Data to Representation**: Augmentation is empirical and finite. Through an asymmetric design of "embedding in the equivariant subspace, extracting in the invariant subspace," provable invariance to the continuous infinite group SO(3) is achieved. This approach is directly transferable to point cloud watermarking, 3D Gaussian Splatting, mesh watermarking, or any geometric signal with SO(3) symmetry.
- **Hardcore Argument for Third-order > Second-order**: While the power spectrum has long been the default for rotation-invariant geometric descriptors, this paper challenges it by highlighting "phase blindness leading to low capacity limits," introducing the bispectrum from molecular dynamics/physics modeling into the information hiding domain.
- **Provable Theory + e3nn Implementation**: The method is not a theoretical toy. The entire equivariant tensor product chain is implemented using e3nn, with learnable CG coefficients, making it highly practical for engineering.

## Limitations & Future Work
- **Capacity Constraints**: To maintain stability, only the mid-frequency bands $\{6, 8, 14\}$ are used, capping the payload at 64 bits. Incorporating higher bands to expand to hundreds of bits is a challenge due to their fragility.
- **Global SO(3) Only**: The method does not yet handle **local** cropping or partial spherical signals. Once part of the sphere is removed, the third-order coupling coefficients $I$ are corrupted. The authors propose using local equivariant subgroups in the future.
- **Vulnerability to AI Editing**: The experiments focus on traditional distortions and rotations, lacking evaluation against AIGC editing attacks like Diffusion-based inpainting or regeneration.
- Computational costs for training and GPU memory usage of SHT at $l_{\max}=16$ (alongside e3nn tensor products) were not disclosed, and deployment costs remain researchers' concerns.

## Related Work & Insights
- **vs. Planar CNN Watermarking (StegaStamp / TrustMark / VINE)**: These rely on CNNs + data augmentation on ERP planes. TRIAD operates in the spherical SH domain, using SO(3) equivariance to guarantee rotation invariance with zero augmentation.
- **vs. 360° VR Watermarking (Liu et al. 2021, Spherical Wavelets)**: While recognizing the need for the spherical domain, earlier works relied on empirical synchronization rather than theoretical invariance.
- **vs. 3D Data Watermarking (GuardSplat, etc.)**: Others use geometric invariants but often rely on heuristics like salient points or SVD. TRIAD provides a unified framework based on SO(3) representation theory.
- **vs. Classical Power Spectrum SH Descriptors (Kazhdan 2003)**: Widely used for 3D shape retrieval, but their phase blindness limits watermarking capacity. This paper is the first to systematically argue that "watermarking tasks require the third-order bispectrum."
- **Insight**: The asymmetric pattern of "embedding in equivariant subspaces and extracting in trivial subspaces" can be extended to E(3)-equivariant molecular design, point cloud steganography, or graph signal watermarking for any scenario with compact group symmetries.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Introducing the bispectrum from representation theory to digital watermarking is a significant first, particularly the asymmetric "equivariant-to-trivial" architecture.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison with 6 SOTA methods across all rotation angles, 8 distortion types, and 3 ablation categories; lacks AI editing and training cost data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative on theory and methodology; Theorem 4.1 provides a rigorous proof. Method sections are precise in algebraic language.
- **Value**: ⭐⭐⭐⭐⭐ Provides the first "provably invariant" framework for SO(3)-symmetric geometric signals, with direct value for content provenance in VR, World Models, and Embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](../../CVPR2026/ai_safety/tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](prism_gauge-invariant_tangent-space_differentially_private_lora.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)

</div>

<!-- RELATED:END -->
