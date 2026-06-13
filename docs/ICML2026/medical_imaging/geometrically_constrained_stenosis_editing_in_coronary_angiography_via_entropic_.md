---
title: >-
  [Paper Note] OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport
description: >-
  [ICML 2026][Medical Imaging][Coronary Angiography] OT-Bridge Editor reformulates "editing a vessel stenosis in coronary angiography" as a constrained entropic OT problem in a vessel-structure composite domain. By integra…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Coronary Angiography"
  - "Stenosis Editing"
  - "Schrödinger Bridge"
  - "Entropic Optimal Transport"
  - "Path-level Geometric Supervision"
date: 2026-05-08
content_hash: 61ca93449ad51caa
---

# OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport

**Conference**: ICML 2026  
**arXiv**: [2605.08851](https://arxiv.org/abs/2605.08851)  
**Code**: Not publicly released  
**Area**: Medical Imaging / Diffusion Models / Data Augmentation  
**Keywords**: Coronary Angiography, Stenosis Editing, Schrödinger Bridge, Entropic Optimal Transport, Path-level Geometric Supervision

## TL;DR
OT-Bridge Editor reformulates "editing a vessel stenosis in coronary angiography" as a constrained entropic OT problem in a vessel-structure composite domain. By integrating path-level geometric projection supervision into the Schrödinger Bridge, it achieves pixel-level shape/position controllable synthetic angiography, yielding a relative improvement of 27.8% in downstream stenosis detection mAP@0.5 on the ARCADE public dataset.

## Background & Motivation

**Background**: Coronary Angiography (CAG) stenosis detection suffers from a severe lack of annotations and significant cross-center domain gaps. The community typically employs diffusion models for "conditional generation" to augment data—using masks, text, or semantic maps as conditions to reconstruct images from noise.

**Limitations of Prior Work**: Existing diffusion editors exhibit two primary weaknesses: (i) Conditions are injected as "soft constraints" via guidance or cross-attention, which fails to precisely lock geometric boundaries (a single missing branch can affect subsequent detection); (ii) Reconstructing from pure noise unnecessarily redraws non-edited anatomical regions, leading to poor structural preservation.

**Key Challenge**: The core requirement for medical synthesis is the **localized minimal transport** from an existing image to a target image, rather than global reconstruction from noise. However, existing diffusion paradigms lack an interface for "path-level geometric hard constraints."

**Goal**: (1) Formalize "geometrically precise local editing" as a constrained optimal transport problem; (2) Design a generative path that can be supervised at the pixel level; (3) Significantly improve the performance of downstream stenosis detectors on public and multi-center data via synthetic angiography.

**Key Insight**: The authors observe that the Schrödinger Bridge (SB) naturally handles diffusion problems with "known endpoints + controllable paths," which matches the image translation requirement where the source is the original image and the target is the edited image. Furthermore, entropic regularization makes OT numerically solvable in high-dimensional pixel spaces.

**Core Idea**: The editing mask and target geometric description are incorporated into the boundary conditions and feasible path set of the SB. A Geometric Projection Guidance (GPG) is applied after each step of the bridge process to ensure the generative path remains within the "geometrically feasible corridor" throughout the process.

## Method

### Overall Architecture
The pipeline consists of three stages: (a) Constructing the binary mask $\mathbf{M}$, protected region $\bar{\mathbf{M}}=\mathbf{1}-\mathbf{M}$, and target geometry $\mathbf{S}^\star=\mathcal{S}(\cdot)$, while setting the starting point $S^\ast$ in a "vessel-structure composite domain" (original image edges + vessel mask); (b) Formulating the edit as an entropic OT problem with endpoints $\mu_0=\delta_{\mathbf{x}_0}$ and $\mu_1$ subject to geometric feasibility constraints, solved via Diffusion SB to obtain the bridge process; (c) Applying geometric projection $\Pi_\mathcal{F}$ to the bridge state every $K$ steps—pulling the edited area toward the target geometry while pulling the non-edited area toward the original image, ultimately outputting pixel-precise synthetic angiography.

### Key Designs

1.  **Geometric Constraints via Feasible Sets + Mask-aware Transport Cost**:
    - **Function**: Injects the strong constraints of "strict preservation of non-edited areas + target geometry matching in edited areas" into the OT formulation.
    - **Mechanism**: Defines $\mathcal{F}=\{\mathbf{x}\mid \mathbf{x}\odot\bar{\mathbf{M}}=\mathbf{x}_0\odot\bar{\mathbf{M}},\ \mathcal{S}(\mathbf{x}\odot\mathbf{M})=\mathbf{S}^\star\}$ as the hard feasible set. The transport cost is formulated in a mask-aware manner as $c(\mathbf{x},\mathbf{y})=\|(\mathbf{x}-\mathbf{y})\odot\bar{\mathbf{M}}\|_2^2+\lambda_M\|(\mathbf{x}-\mathbf{y})\odot\mathbf{M}\|_2^2$. The final objective is $\min_\pi\langle\mathbf{C},\pi\rangle+\varepsilon\,\mathrm{KL}(\pi\|\mathbf{K})$ s.t. $\pi\in\mathcal{Q}$.
    - **Design Motivation**: Classic diffusion editing utilizes soft guidance (e.g., ControlNet/SDEdit), which cannot guarantee bit-identical preservation of non-edited regions. Explicit feasible sets and mask-aware costs mathematically define "what can move and what cannot" at the source.

2.  **Schrödinger Bridge as a Dynamic Solver**:
    - **Function**: Converts entropic OT in high-dimensional pixel space into a bridge process that can be sampled step-by-step via a diffusion chain.
    - **Mechanism**: Minimizes $P^\star=\arg\min_{P:P_0=\mu_0,P_1=\mu_1}\mathrm{KL}(P\|R)$ in the trajectory space, where $R$ is the reference diffusion. Starting from $\mathbf{s}_0=\mathbf{x}_0$, the bridge evolves for $T=50$ steps via $\tilde{\mathbf{s}}_{k+1}\sim p^\star(\mathbf{s}_{k+1}\mid\mathbf{s}_k)$, exposing the generative path for external supervision.
    - **Design Motivation**: Solving entropic OT directly in pixel space is computationally expensive and numerically unstable. SB transforms endpoint matching into path density matching, allowing gradual learning and the insertion of supervision signals along the trajectory.

3.  **GPG: Path-level Geometric Projection Supervision**:
    - **Function**: "Pushes" the state back into the geometric feasible set at every step of the bridge process to prevent trajectory drift.
    - **Mechanism**: Each step samples $\tilde{\mathbf{x}}_{k+1}\sim p^\star$, followed by the projection $\mathbf{x}_{k+1}\leftarrow\Pi_\mathcal{F}(\tilde{\mathbf{x}}_{k+1})$, where $\Pi_\mathcal{F}(\mathbf{x})=\arg\min_\mathbf{y}\mathcal{L}_{\text{geo}}(\mathbf{y})+\lambda_{\text{out}}\mathcal{L}_{\text{out}}(\mathbf{y})$. The geometric term $\mathcal{L}_{\text{geo}}=\mathcal{D}_{\text{geo}}(\mathcal{S}(\mathbf{y}\odot\mathbf{M}),\mathbf{S}^\star)$ uses boundary distance based on Signed Distance Transform (SDT): $\mathcal{D}_{\text{geo}}=\frac{1}{|B_M|}\sum_\mathbf{u}|\phi^\star(\mathbf{u})|$. The external term $\mathcal{L}_{\text{out}}=\|(\mathbf{y}-\mathbf{x}_0)\odot\bar{\mathbf{M}}\|_2^2$ suppresses drift in non-edited regions. Projections occur every 5 steps with $\lambda_{\text{out}}=10, \lambda_{\text{geo}}=1$.
    - **Design Motivation**: Applying geometric constraints only at the endpoint is "soft command"; if the bridge trajectory has already deviated mid-way, correcting it is costly. Pixel-level precision requires staying within the geometric corridor at every step. SDT provides smooth, differentiable geometric distances to avoid gradient explosions from hard boundaries.

### Loss & Training
The SB reference process $R$ uses a standard discrete diffusion chain; $\varepsilon=10^{-2}$ controls transport smoothness. GPG projections are solved via gradient optimization and inserted into the sampling path as a logits-processor, requiring no retraining of the diffusion backbone and remaining decoupled from the editing space (pixel or latent).

## Key Experimental Results

### Main Results
Evaluated on the ARCADE public set and a multi-center internal set, comparing against GANs (Pix2PixHD, SPADE) and diffusion editors (SDEdit, SDM, SiameseDiff, DiGDA).

| Metric | Pix2PixHD | SDEdit | SiameseDiff | **OT-Bridge** |
|------|----------:|-------:|------------:|--------------:|
| Edit Dice ↑ | 0.621 | 0.645 | 0.722 | **0.774** |
| Edit mIoU ↑ | 0.801 | 0.781 | 0.837 | **0.892** |
| FID ↓ | 52.9 | 46.9 | 34.2 | **16.7** |
| SSIM ↑ | 0.676 | 0.705 | 0.790 | **0.878** |

Downstream YOLOv8 mAP@0.5 on ARCADE: Real-only 0.525 → Real+Synth **0.727** (+38.5% relative; average 27.8% across 4 detectors). On multi-center data: Real-only 0.654 → Real+Synth **0.731** (+11.8% relative; average 23.0%).

### Ablation Study

| Configuration | Edit Dice ↑ | Outside SSIM ↑ | Description |
|------|------------:|---------------:|------|
| Edge Domain | 0.592 | 0.533 | Editing on edge maps only; poor structure |
| Seg Domain | 0.767 | 0.694 | Mask domain only; coarse boundaries |
| **Composite (Default)** | **0.892** | **0.878** | Edges + Vessel Mask composite domain |
| Composite w/o Protection | 0.802 | 0.786 | Significant degradation without non-edited consistency |
| Endpoint-only GPG | bDice 0.765 | $\mathcal{E}_T=2.8$ | Endpoint OK but intermediate path drifted |
| **Path GPG (Default)** | **bDice 0.895** | $\mathcal{E}_T=1.1$ | Geometrically stable throughout |
| w/o Boundary Supervision $\partial m$ | bDice 0.582 | $\mathcal{E}_T=12.4$ | Nearly fails without SDT |

Scanning synthetic data scales shows saturation of gains at $r=1.0$ (Synth:Real Ratio = 1:1).

### Key Findings
- **GPG is the Core Contribution**: Removing path-level geometric supervision causes bDice to drop from 0.895 to 0.765, verifying that "step-by-step projection" is more critical for pixel-level editing than "endpoint constraints."
- **Composite Domain > Single Domain**: Using both edges and vessel masks as starting points maintains boundaries in edited areas while stabilizing structure in non-edited areas; both are indispensable.
- **Detector Domain Transfer**: Detectors trained with OT-Bridge synthetic images on ARCADE maintain an 11-13% gain when transferred to multi-center sets, suggesting that synthetic images provide "positional/morphological diversity" rather than simple noise.
- **Robustness to Mask Noise**: Boundary jitter and erosion/dilation have minor impacts on Dice, but spatial displacement (mask misalignment) significantly affects downstream mAP—indicating that localization precision is more important than boundary fine-tuning during deployment.

## Highlights & Insights
- **Reframing "Local Editing" as "Constrained Transport"**: Shifting from redrawing the entire image from noise to transporting minor changes from the original image better aligns with the essence of medical editing, leading to more stable outputs. This mindset is directly transferable to tasks like lesion inpainting, attribute editing, and object insertion.
- **Path-level Hard Supervision as a Natural Interface for SB**: While traditional diffusion only allows guidance at the "two ends," the SB bridge process makes "per-step constraints" a natural operation. The GPG technique could be adapted for other geometry-sensitive tasks (hands, fonts, circuit diagrams).
- **Composite Domain Starting Points** significantly improve the solvability of dual objectives (boundary + structure), suggesting that medical editors should avoid relying on a single representation.
- **Downstream Task-Driven Evaluation**: Evaluating synthetic data value via detector mAP is more persuasive than reporting FID/SSIM alone, sets a good example for medical generation research.

## Limitations & Future Work
- Validated only on coronary angiography stenosis editing; more complex scenarios like multi-branching, 3D CTA, or super-resolution were not tested.
- GPG projection requires gradient-based optimization at each step, increasing single-image inference time by approximately 1.5× compared to pure SB (not explicitly emphasized but inferable).
- Geometric supervision relies on accurate vessel masks; spatial displacement experiments show that mask misalignment leads to significant mAP drops, implying a need for reliable segmentation pre-processing.
- Hyperparameters ($\varepsilon, \lambda_M, \lambda_{\text{out}}$) are fixed without an adaptive mechanism; they may require re-tuning for different centers or image qualities.

## Related Work & Insights
- **vs. SDEdit / SDM / SiameseDiff**: These rely on noise-start + soft guidance. OT-Bridge explicitly defines "what cannot move" in the feasible set, achieving order-of-magnitude improvements in structural preservation (FID 16.7 vs 34-79).
- **vs. ControlNet / T2I-Adapter**: These emphasize structural semantic guidance without hard geometric constraints; OT-Bridge's GPG uses SDT projection on geometric descriptors, marking a difference between pixel-level and feature-level control.
- **vs. I²SB / BBDM / DSB**: While these are SB-style image translation models, this work specifically designs path-level supervision and composite domain starts for medical geometry editing, turning a general bridge model into a "geometry-aware editor."
- **Transferable Insights**: The GPG strategy of upgrading "soft guidance" to "path projection" is highly relevant for constrained visual generation (3D shapes, fonts, vector graphics, CAD). The OT perspective also inspires the redefinition of image editing tasks from "reconstruction" to "transport."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Uses constrained entropic OT + SB bridge process for geometry-sensitive medical editing; path-level projection supervision provides a genuine "per-step" control interface.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage with 4 detectors × 2 datasets, 5 baselines, and 4 major ablations (domain/GPG/scale/mask noise).
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical formulation and algorithmic pseudocode; effective visual comparisons in ROI figures; some minor technical details in appendices are slightly dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides substantial evidence of performance gains in multi-center scenarios for detectors trained with synthetic data, offering real help for clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](../../AAAI2026/medical_imaging/mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[AAAI 2026\] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing](../../AAAI2026/medical_imaging/fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)
- [\[AAAI 2026\] DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities](../../AAAI2026/medical_imaging/dia-gnostic_vlvae_disentangled_alignment-constrained_vision_language_variational.md)

</div>

<!-- RELATED:END -->
