---
title: >-
  [Paper Note] Where, What, Why: Toward Explainable 3D-GS Watermarking
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] A representation-native 3D-GS watermarking framework that answers three key questions: Trio-Experts for carrier selection (where)…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Digital Watermarking"
  - "Copyright Protection"
  - "Explainability"
  - "Robust Embedding"
date: 2026-05-08
content_hash: 1bc309cf9964afd7
---

# Where, What, Why: Toward Explainable 3D-GS Watermarking

**Conference**: CVPR2026
**arXiv**: [2603.08809](https://arxiv.org/abs/2603.08809)  
**Authors**: Mingshu Cai (Waseda University), Jiajun Li (Southeast University), Osamu Yoshie, Yuya Ieiri, Yixuan Li (NTU)
**Code**: Not released  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Digital Watermarking, Copyright Protection, Explainability, Robust Embedding

## TL;DR

A representation-native 3D-GS watermarking framework that answers three key questions: Trio-Experts for carrier selection (where), Channel-wise Group Mask for gradient control (what), and decoupled fine-tuning for auditable attribution (why). Surpasses SOTA on both rendering quality (PSNR +0.83 dB) and bit accuracy (+1.24%).

## Background & Motivation

3D-GS has become the dominant paradigm for 3D content creation—thanks to its explicit parameterization, real-time rendering, and high fidelity—with broad applications in film, gaming, autonomous driving, and digital humans. However, its core advantage—directly editable Gaussian parameters—also introduces serious security risks: adversaries can trivially copy models, tamper with content, strip authorship information, and redistribute illegally.

Existing radiance field watermarking methods (WateRF, 3DGSW, GuardSplat) exhibit two critical gaps when applied to explicit discrete Gaussian representations:

1. **Carrier selection**: How to select watermark carriers from large-scale heterogeneous Gaussian primitives by jointly considering multi-view visibility, frequency-domain cues, and geometric/appearance stability.
2. **Robust and imperceptible embedding**: How to embed robust watermarks without degrading visual or rendering quality, while remaining extractable after common perturbations such as cropping, compression, and format conversion.

## Core Problem

Unified answers to three key questions: **Where** (on which Gaussians to embed the watermark), **What** (what to embed and how to control update magnitude), and **Why** (why these carriers were selected—auditable attribution).

## Method

### Overall Architecture

The framework consists of three stages:

1. **Initialization**: Prune redundant Gaussians by rendering contribution → Trio-Experts extract priors → SBAG selects carriers and performs densification.
2. **Decoupled Fine-tuning**: Channel-wise Group Mask routes gradients → watermark carriers and visual compensators are optimized independently.
3. **Inference/Verification**: Extract watermark bits from rendered views via a frozen decoder.

### 3.1 Pruning by Contribution

Pre-trained 3D-GS models typically contain substantial redundant Gaussians. Following 3D-GSW's contribution-based pruning strategy, a temporary color parameter $C'$ is introduced, and gradients $V_\pi = \partial L_\pi^{aux}/\partial C'$ are computed via an auxiliary loss as contribution scores. Gaussians with $V_\pi < 10^{-8}$ are pruned.

### 3.2 Trio-Experts: Three-Expert Carrier Prior Extraction

Unlike prior methods that rely on image-domain gradients or high-frequency heuristics, this paper proposes a **representation-native** Trio-Experts system, where all decision evidence is anchored in the 3D-GS parameter space, ensuring view-consistent evaluation.

Native Gaussian parameters are grouped by semantics:

- $\mathcal{C}_{geo} = \{\mathbf{x}, \mathbf{s}, \mathbf{q}\}$ (position, scale, rotation)
- $\mathcal{C}_{app} = \{\alpha, \mathbf{h}^{(0)}, \mathbf{h}^{(\geq 1)}\}$ (opacity, SH coefficients)
- $\mathcal{C}_{red} = \{\mathbf{x}, \mathbf{s}, \mathbf{q}, \mathbf{h}^{(0)}\}$ (for redundancy estimation)

After constructing a $k$-NN neighborhood $\mathcal{N}_k(i)$ in 3D position space, three experts compute the following:

**Geometry Expert $z_1$**: Based on $\mathcal{C}_{geo}$; captures structural decomposition and boundary cues to assess geometric stability. Computes scale isotropy $\text{Iso}_i = \min(\mathbf{s}_i)/\max(\mathbf{s}_i)$, neighborhood quaternion consistency $\text{RotCons}_i$, and compact footprint $\overline{fp}_i$ (geometric mean). High isotropy + high rotation consistency + small footprint = geometrically stable carrier candidate.

**Appearance Expert $z_2$**: Based on $\mathcal{C}_{app}$; measures cross-view appearance consistency via DC bandpass, opacity gating, and high-frequency suppression. Includes AC high-frequency energy ratio $\rho_i^{hf}$ (lower is better), bilateral opacity gate $g(\alpha_i)$ (moderate opacity preferred), and DC intensity bandpass filtering $c_i$.

**Redundancy Expert $z_3$**: Based on $\mathcal{C}_{red}$; characterizes inter-Gaussian distribution density to estimate substitutability. Measured via overlap-weighted neighborhood similarity: $r_{ij}$ combines color and shape similarity, while $w_{ij}$ approximates projected overlap. Highly redundant Gaussians are preferred as carriers, as surrounding Gaussians can compensate for perturbations.

Each expert $k$ maps feature $z_k(i)$ into an **evidence bundle** $E_k(i) = [U_k(i), S_k(i)]$:

- $U_k \in [0,1]$: uncertainty (from neighborhood dispersion + expert-specific penalties)
- $S_k \in [0,1]$: quality score

This quality–confidence decoupling enables confidence-aware gating downstream.

### 3.3 Safety and Budget-Aware Gate (SBAG)

SBAG decouples carrier selection into **ranking** and **budget allocation**.

**Ranking**: Evidence bundles are mapped to proxy scores $R_k(i) = \text{clip}(S_k(i) - \beta U_k(i), 0, 1)$, where $R_1$ captures geometric stability, $R_2$ appearance safety, and $R_3$ redundancy certainty. The symmetric point-level utility score is:

$$u_i = (R_1(i) \cdot R_2(i) \cdot R_3(i))^{1/3}$$

**Single-pass rendering estimation**: All training views are rendered once using DC+opacity to obtain view-corrected visibility $v_i$ and scene crowding factor $\eta$.

**Adaptive budget**: Given message length $M$ bits, the required number of carriers is adaptively computed as:

$$\kappa_{eff} = \kappa_0 \cdot \bar{v} \cdot \eta, \quad B = \lceil M / \kappa_{eff} \rceil$$

**Feasible set and selection**: A quantile-constrained feasible set $\mathcal{F}$ is defined (four-dimensional thresholds: geometry, appearance, redundancy, visibility), from which the top-$B$ by $u_i$ form the initial carrier set $\mathcal{WM}_0$.

**Prototype neighbor expansion**: To bridge view-coverage gaps, a compact evidence vector $\mathbf{e}_i$ is constructed, the prototype mean $\boldsymbol{\mu}$ of $\mathcal{WM}_0$ is computed, and neighbors are recruited via cosine similarity to expand the set to $\mathcal{WM}_{parent}$.

**Densification splitting**: Each parent Gaussian is split into $N_s$ visually equivalent child Gaussians—one routed to the watermark branch, the rest serving as visual compensators. This yields $\mathcal{WM}_\star$ (watermark carrier set) and $\mathcal{VIS}$ (visual compensator set).

### 3.4 Channel-wise Group Mask

To avoid visible degradation, per-channel masks are assigned to carriers and compensators. Five parameter channel groups are defined: $g \in \{\boldsymbol{\delta}_{dc}, \boldsymbol{\rho}_{rest}, \boldsymbol{\omega}_{opa}, \boldsymbol{\theta}_{rot}, \boldsymbol{\sigma}_{sca}\}$.

The two mask types are computed differently:

- **VIS mask** $m_g^{vis}$: Mean of channel weights over compensators, clipped with a minimum update floor $\text{floor}_g$.
- **WM mask** $m_g^{wm}$: Median of channel weights over carriers, clipped accordingly.

Gradient routing rules:

$$\nabla_{\theta_i^g} \mathcal{L} = \begin{cases} m_g^{wm}(i) \nabla_{\theta_i^g} \mathcal{L}_{wm}, & i \in \mathcal{WM}_\star \\ m_g^{vis}(i) \nabla_{\theta_i^g} \mathcal{L}_{vis}, & i \in \mathcal{VIS} \end{cases}$$

Two forward/backward passes ensure that $\mathcal{WM}_\star$ and $\mathcal{VIS}$ receive gradients orthogonally, completely eliminating optimization interference.

### 3.5 Decoupled Watermark Fine-tuning

**Visual objective** (applied to VIS only):

$$\mathcal{L}_{vis} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{wav}^{high}\mathcal{L}_{wav}^{high}$$

where $\mathcal{L}_{wav}^{high}$ penalizes multi-level DWT high-frequency subbands (LH/HL/HH).

**Watermark objective** (applied to $\mathcal{WM}_\star$ only): Employs EOT (Expectation Over Transformation) training, jointly optimizing over clean and transformed renderings:

$$\mathcal{L}_{wm} = \lambda_{wm}^{clean}\mathcal{L}_{wm}^{clean} + \lambda_{wm}^{eot}\mathcal{L}_{wm}^{eot} + \lambda_{wav}^{low}\mathcal{L}_{wav}^{low}$$

The watermark is embedded exclusively in the DWT low-frequency (LL) subband; $\mathcal{L}_{wav}^{low}$ regularizes low-frequency distortion. The EOT transformation family covers blur, rotation, scaling, cropping, noise, JPEG compression, and more.

**Key design**: VIS points are entirely excluded from the watermark loss. If VIS points were coupled into the watermark loss, they would oppose WM updates, simultaneously degrading visual quality and destabilizing extraction.

## Key Experimental Results

### Datasets and Setup

Evaluated on 25 scenes across three standard benchmarks: Blender, LLFF, and Mip-NeRF 360. Training uses a single NVIDIA A800 GPU for 2–10 epochs, with a frozen HiDDeN decoder (32/48/64 bits).

### Rendering Quality and Bit Accuracy

| Method | 32-bit Acc↑ | PSNR↑ | SSIM↑ | 48-bit Acc↑ | PSNR↑ | 64-bit Acc↑ | PSNR↑ |
|---|---|---|---|---|---|---|---|
| WateRF+3D-GS | 93.28 | 30.57 | 0.954 | 84.39 | 30.06 | 74.92 | 25.73 |
| GuardSplat | 95.58 | 35.32 | 0.978 | 93.29 | 33.36 | 90.14 | 32.25 |
| 3D-GSW | 97.22 | 35.15 | 0.977 | 93.59 | 33.26 | 91.31 | 32.52 |
| **Ours** | **98.46** | **35.98** | **0.982** | **94.29** | **33.45** | **91.65** | **32.71** |

At 32-bit: PSNR +0.83 dB (vs. 3D-GSW), bit accuracy +1.24%. Advantages become more pronounced as message length increases.

### Image-Level Perturbation Robustness (32-bit)

| Attack Type | WateRF | GuardSplat | 3D-GSW | **Ours** |
|---|---|---|---|---|
| None | 93.28 | 95.58 | 97.22 | **98.46** |
| Gaussian Noise | 78.12 | 90.11 | 83.71 | **91.22** |
| Rotation | 81.47 | 95.87 | 88.05 | **96.18** |
| Scaling 75% | 84.63 | 94.93 | 94.58 | **95.06** |
| Gaussian Blur | 87.09 | 97.16 | 95.94 | **97.75** |
| Cropping 40% | 84.58 | 95.05 | 92.73 | **95.88** |
| JPEG 50% | 82.03 | 89.92 | 92.54 | **92.95** |
| Combined | 64.73 | 88.64 | 90.96 | **91.30** |

Best performance across all attack types, with particularly notable margins under Gaussian noise and rotation attacks.

### Model-Level Perturbation Robustness

Under malicious manipulation of the 3D-GS representation (parameter noise $\sigma=0.1$, random removal/cloning of 20% of Gaussians), the proposed method consistently leads across all perturbations, demonstrating that the watermark information does not rely on fragile subsets.

### Ablation Study

All three components—SBAG, Group Mask, and Decoupled Finetuning—are individually necessary. Removing all components reduces bit accuracy to 94.70% and PSNR to 30.00 dB; enabling all components achieves 97.80%/35.20 dB. The adaptive budget mechanism (vs. fixed 1%/10%) achieves optimal balance between accuracy and storage (98.46%/178 MB).

## Highlights & Insights

1. **Representation-native design**: All decisions are made in the 3D-GS parameter space without relying on pixel-domain gradients, ensuring view-consistent evaluation.
2. **Three-tier decoupled architecture**: Expert scoring → gated selection → gradient routing, each stage with clear interpretability.
3. **Auditable attribution**: Per-Gaussian attribution reveals where the watermark is embedded and why those carriers were selected.
4. **Adaptive budget mechanism**: Scene-aware carrier count estimation ($\kappa_{eff}$) avoids over- or under-allocation.
5. **Near-lossless rendering quality**: PSNR of 35.98 dB and SSIM of 0.982 at 32-bit, approaching unwatermarked model quality.

## Limitations & Future Work

1. **Hyperparameter sensitivity**: Frequency-domain decoupled training requires careful tuning of loss weights to balance quality and robustness.
2. **Decoder dependency**: Reliance on the pre-trained HiDDeN decoder means its robustness ceiling constrains overall system performance.
3. **Computational overhead not quantified**: Additional costs from Trio-Experts' k-NN computation and two-pass backpropagation are not quantified.
4. **Dynamic scenes**: The paper mentions extensibility to dynamic scenes in the conclusion but provides no experimental validation.
5. **Adversarial attacks**: Only standard image perturbations are considered; robustness against targeted adversarial attacks is not evaluated.

## Related Work & Insights

- **WateRF**: Transfers NeRF frequency-domain watermarking to 3D-GS but lacks EOT adversarial training, leading to significant accuracy degradation under perturbations (combined attack: only 64.73%).
- **GuardSplat**: CLIP-guided + SH-space embedding with EOT, but heavily dependent on the CLIP decoder, limiting performance under complex perturbations.
- **3D-GSW**: Frequency-domain regularization + rendering contribution constraints; relatively comprehensive but lacks carrier-compensator decoupling.
- **Ours**: The only method performing carrier selection in 3D parameter space with gradient-routing decoupling, systematically addressing where/what/why.

The three-expert scoring + uncertainty-weighted fusion paradigm is transferable to other 3D-GS editing tasks (e.g., stylization region selection). The decoupled fine-tuning strategy for eliminating gradient conflicts offers insights for multi-objective 3D-GS optimization (e.g., simultaneous geometric refinement and semantic editing). The scene-aware adaptive budget design ($\kappa_{eff} = \kappa_0 \cdot \bar{v} \cdot \eta$) merits broader adoption.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The representation-native Trio-Experts system and gradient-decoupled routing are novel designs; the where/what/why framework is well-structured.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, three message lengths, image-level and model-level attacks, and complete ablations; however, dynamic scene and adversarial attack evaluations are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical presentation, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — An important contribution to 3D-GS copyright protection; explainability is a key selling point.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?](../../AAAI2026/3d_vision/can_protective_watermarking_safeguard_the_copyright_of_3d_gaussian_splatting.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)

</div>

<!-- RELATED:END -->
