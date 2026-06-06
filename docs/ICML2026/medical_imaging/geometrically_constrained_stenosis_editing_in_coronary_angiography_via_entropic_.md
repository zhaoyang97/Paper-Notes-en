---
title: >-
  [Paper Note] OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport
description: >-
  [ICML 2026][Medical Imaging][Coronary Angiography] OT-Bridge Editor reformulates "editing a vessel stenosis in coronary angiography" as a "constrained entropic OT problem in the vessel-structure composite domain…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Coronary Angiography"
  - "Stenosis Editing"
  - "Schrödinger Bridge"
  - "Entropic Optimal Transport"
  - "Path-level Geometric Supervision"
date: 2026-05-08
content_hash: 2d76c7984154f69e
---

# OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport

**Conference**: ICML 2026  
**arXiv**: [2605.08851](https://arxiv.org/abs/2605.08851)  
**Code**: Not publicly released  
**Area**: Medical Imaging / Diffusion Models / Data Augmentation  
**Keywords**: Coronary Angiography, Stenosis Editing, Schrödinger Bridge, Entropic Optimal Transport, Path-level Geometric Supervision

## TL;DR
OT-Bridge Editor reformulates "editing a vessel stenosis in coronary angiography" as a "constrained entropic OT problem in the vessel-structure composite domain," leveraging Schrödinger Bridge with path-level geometric projection supervision to achieve pixel-level controllable synthetic angiograms. On the ARCADE public dataset, it achieves a 27.8% relative improvement in downstream stenosis detection mAP@0.5.

## Background & Motivation

**Background**: Coronary angiography (CAG) stenosis detection suffers from severe annotation scarcity and large cross-center domain gaps. The community typically uses diffusion models for "conditional generation" to augment data—providing a mask, text, or semantic map for the diffusion model to reconstruct from noise.

**Limitations of Prior Work**: Existing diffusion editors have two major weaknesses: (i) Conditions are injected via guidance/cross-attention as "soft constraints," making it impossible to precisely lock geometric boundaries (missing a branch can affect downstream detection); (ii) Starting reverse diffusion from pure noise unnecessarily perturbs non-edited anatomical regions, leading to poor structural preservation.

**Key Challenge**: The medical synthesis requirement is "locally modifying a stenosis in an existing angiogram," essentially a **locally minimal transport** from source to target image, not reconstructing the entire image from noise. However, current diffusion paradigms lack interfaces for "path-level geometric hard constraints."

**Goal**: (1) Formalize "geometrically precise local editing" as a constrained optimal transport problem; (2) Design a generative path that can be supervised at the pixel level; (3) Use synthetic angiograms to significantly improve downstream stenosis detectors on public and multi-center data.

**Key Insight**: The authors observe that Schrödinger Bridge naturally handles "known endpoints + controllable paths" in diffusion, aligning well with "source is original image, target is edited image" image translation tasks; entropic regularization makes OT numerically tractable in high-dimensional pixel space.

**Core Idea**: Encode the editing mask and target geometry into the SB boundary conditions and path feasibility set, then apply geometric projection (GPG) at each bridge step, ensuring the generative path remains "sandwiched within the geometric feasible corridor" throughout.

## Method

### Overall Architecture
The pipeline consists of three stages: (a) Construct binary editing mask $\mathbf{M}$, protection region $\bar{\mathbf{M}}=\mathbf{1}-\mathbf{M}$, and target geometry $\mathbf{S}^\star=\mathcal{S}(\cdot)$ from editing specifications, and set the starting point $S^\ast$ in the "vessel-structure composite domain" (original image edges + vessel mask); (b) Formulate editing as an entropic OT with endpoints $\mu_0=\delta_{\mathbf{x}_0}$ and $\mu_1$ constrained by the geometric feasible set, solved via Diffusion SB to obtain the bridge process; (c) Every $K$ steps, apply geometric projection $\Pi_\mathcal{F}$ to the bridge state—pulling the editing region toward the target geometry and the non-editing region toward the original image, ultimately outputting pixel-level precise synthetic angiograms.

### Key Designs

1. **Geometric Constraints as Feasible Set + Mask-aware Transport Cost**:

    - **Function**: Encodes the two hard constraints—"strictly preserve non-editing region + match target geometry in editing region"—into the OT formulation.
    - **Mechanism**: Defines $\mathcal{F}=\{\mathbf{x}\mid \mathbf{x}\odot\bar{\mathbf{M}}=\mathbf{x}_0\odot\bar{\mathbf{M}},\ \mathcal{S}(\mathbf{x}\odot\mathbf{M})=\mathbf{S}^\star\}$ as the hard feasible set; the transport cost is mask-aware: $c(\mathbf{x},\mathbf{y})=\|(\mathbf{x}-\mathbf{y})\odot\bar{\mathbf{M}}\|_2^2+\lambda_M\|(\mathbf{x}-\mathbf{y})\odot\mathbf{M}\|_2^2$; the final objective is $\min_\pi\langle\mathbf{C},\pi\rangle+\varepsilon\,\mathrm{KL}(\pi\|\mathbf{K})$ s.t. $\pi\in\mathcal{Q}$.
    - **Design Motivation**: Classic diffusion editing applies constraints via soft guidance (ControlNet/SDEdit), which cannot guarantee invariance in non-editing regions. Explicit feasible sets and mask-aware costs mathematically define "where can/cannot change," eliminating "leakage" at the source.

2. **Schrödinger Bridge as Dynamic Solver**:

    - **Function**: Converts fragile OT in high-dimensional pixel space into a bridge process that can be sampled stepwise via diffusion chains.
    - **Mechanism**: Minimizes in trajectory space $P^\star=\arg\min_{P:P_0=\mu_0,P_1=\mu_1}\mathrm{KL}(P\|R)$, where $R$ is the reference diffusion; starting from $\mathbf{s}_0=\mathbf{x}_0$, rolls out $T=50$ steps via $\tilde{\mathbf{s}}_{k+1}\sim p^\star(\mathbf{s}_{k+1}\mid\mathbf{s}_k)$, exposing the entire generative path for subsequent supervision.
    - **Design Motivation**: Directly solving entropic OT in pixel space is computationally expensive and numerically unstable; SB's advantage is "matching endpoints via path density matching," enabling stepwise neural network learning and convenient path-level supervision.

3. **GPG: Path-level Geometric Projection Supervision**:

    - **Function**: At each bridge step, projects the state back into the geometric feasible set, preventing trajectory drift to incorrect shapes.
    - **Mechanism**: Each step samples $\tilde{\mathbf{x}}_{k+1}\sim p^\star$, then projects $\mathbf{x}_{k+1}\leftarrow\Pi_\mathcal{F}(\tilde{\mathbf{x}}_{k+1})$, where $\Pi_\mathcal{F}(\mathbf{x})=\arg\min_\mathbf{y}\mathcal{L}_{\text{geo}}(\mathbf{y})+\lambda_{\text{out}}\mathcal{L}_{\text{out}}(\mathbf{y})$. The geometric term $\mathcal{L}_{\text{geo}}=\mathcal{D}_{\text{geo}}(\mathcal{S}(\mathbf{y}\odot\mathbf{M}),\mathbf{S}^\star)$ uses a signed distance transform (SDT)-based boundary distance $\mathcal{D}_{\text{geo}}=\frac{1}{|B_M|}\sum_\mathbf{u}|\phi^\star(\mathbf{u})|$; the external term $\mathcal{L}_{\text{out}}=\|(\mathbf{y}-\mathbf{x}_0)\odot\bar{\mathbf{M}}\|_2^2$ suppresses drift in non-editing regions. In code, projection is performed every 5 steps, with $\lambda_{\text{out}}=10,\lambda_{\text{geo}}=1$.
    - **Design Motivation**: Applying geometric constraints only at the endpoint is "soft guidance"; if the bridge path drifts mid-way, pulling it back incurs high cost. Pixel-level precision requires staying within the geometric corridor at every step; SDT provides a smooth, differentiable geometric distance, avoiding gradient explosion from hard boundaries.

### Loss & Training
The SB reference process $R$ uses a standard discrete diffusion chain; $\varepsilon=10^{-2}$ controls transport smoothness. GPG projection is solved via gradient optimization, inserted as a logits-processor along the sampling path, requiring no retraining of the diffusion backbone and decoupled from the editing space (pixel or latent).

## Key Experimental Results

### Main Results
On the ARCADE public dataset and a multi-center internal dataset, compared with GANs (Pix2PixHD, SPADE) and diffusion editors (SDEdit, SDM, SiameseDiff, DiGDA):

| Metric | Pix2PixHD | SDEdit | SiameseDiff | **OT-Bridge** |
|--------|----------:|-------:|------------:|--------------:|
| Edit Dice ↑ | 0.621 | 0.645 | 0.722 | **0.774** |
| Edit mIoU ↑ | 0.801 | 0.781 | 0.837 | **0.892** |
| FID ↓ | 52.9 | 46.9 | 34.2 | **16.7** |
| SSIM ↑ | 0.676 | 0.705 | 0.790 | **0.878** |

Downstream YOLOv8 on ARCADE mAP@0.5: Real-only 0.525 → Real+Synth **0.727** (+38.5% relative; paper reports 27.8% average across 4 detectors); on multi-center dataset: Real-only 0.654 → Real+Synth **0.731** (+11.8%, average 23.0%).

### Ablation Study

| Setting | Edit Dice ↑ | Outside SSIM ↑ | Notes |
|---------|------------:|---------------:|-------|
| Edge domain | 0.592 | 0.533 | Editing only on edge map, poor structure |
| Seg domain | 0.767 | 0.694 | Only mask domain, coarse boundaries |
| **Composite (default)** | **0.892** | **0.878** | Edge + vessel mask composite domain |
| Composite w/o protection constraint | 0.802 | 0.786 | Removing non-editing consistency degrades performance |
| Endpoint GPG only | bDice 0.765 | $\mathcal{E}_T=2.8$ | Endpoint OK, but path drifts |
| **Path GPG (default)** | **bDice 0.895** | $\mathcal{E}_T=1.1$ | Geometric stability throughout |
| w/o boundary supervision $\partial m$ | bDice 0.582 | $\mathcal{E}_T=12.4$ | Without SDT, almost collapses |

Synthetic data scaling shows $r=1.0$ (synthetic:real=1:1) saturates the gain.

### Key Findings
- **GPG is the core contribution**: Removing path-level geometric supervision drops bDice from 0.895 to 0.765, confirming that "stepwise projection" is crucial for pixel-level editing.
- **Composite domain > single domain**: Using both edge and vessel mask as starting points preserves boundaries in the editing region and structure in the non-editing region; both are indispensable.
- **Detector cross-domain transfer**: Detectors trained on ARCADE with OT-Bridge synthetic images retain 11–13% gain when transferred to multi-center datasets, indicating synthetic images provide "positional/morphological diversity" rather than simple noise.
- **Mask noise robustness**: Boundary jitter/dilation-erosion has little effect on Dice, but spatial displacement (mask misalignment) most impacts downstream mAP—suggesting deployment should prioritize localization accuracy over boundary refinement.

## Highlights & Insights
- **Reframing "local editing" as "constrained transport"**: From redrawing the entire image from noise to transporting minimal changes from the original, better aligning with the essence of medical editing and yielding more stable outputs; this approach is directly transferable to any "local retouching" task (lesion inpainting, attribute editing, object insertion).
- **Path-level hard supervision is a natural interface for SB**: Traditional diffusion only allows guidance at endpoints; SB's bridge process makes "stepwise constraint" a natural operation. GPG can be adapted to other geometry-sensitive tasks (hands, fonts, circuit diagrams).
- **Composite domain starting point** significantly improves the solvability of the dual objectives "boundary + structure," suggesting that medical editors should avoid single representations.
- **Downstream task-driven evaluation**: Using detector mAP to directly assess the value of synthetic data is more convincing than reporting FID/SSIM alone, and is a practice worth emulating in medical generative research.

## Limitations & Future Work
- Only validated on coronary angiography stenosis editing; not tested on more complex multi-branch, 3D CTA, or super-resolution scenarios.
- GPG projection requires gradient optimization at each step, making single-image inference about 1.5× slower than pure SB (not emphasized in the paper but inferable).
- Geometric supervision depends on accurate vessel masks; spatial displacement experiments show that mask misalignment significantly degrades downstream mAP, indicating deployment requires reliable segmentation pre-processing.
- Hyperparameters such as $\varepsilon,\lambda_M,\lambda_{\text{out}}$ are fixed, lacking adaptive mechanisms; may require retuning for different centers or angiography qualities.

## Related Work & Insights
- **vs SDEdit / SDM / SiameseDiff**: All start from noise with soft guidance; OT-Bridge explicitly encodes "what cannot change" in the feasible set, achieving an order-of-magnitude improvement in structural preservation (FID 16.7 vs 34–79).
- **vs ControlNet / T2I-Adapter**: Emphasize "semantic structure guidance" without geometric hard constraints; OT-Bridge's GPG directly projects geometric descriptors via SDT, representing a pixel-level vs feature-level distinction.
- **vs I²SB / BBDM / DSB**: Also SB-style image translation, but this work designs path-level supervision and composite domain starting points specifically for medical geometric editing, turning a general bridge model into a "geometry-sensitive editor."
- **Transferable insights**: Upgrading "soft guidance" to "path projection" via GPG is valuable for constrained visual generation (3D shapes, fonts, vector graphics, CAD); the OT perspective also inspires redefining more image editing tasks as "transport" rather than "reconstruction."

## Rating
- Novelty: ⭐⭐⭐⭐ Uses constrained entropic OT + SB bridge process for geometry-sensitive medical editing; path-level projection supervision is a genuinely new "stepwise control" interface.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 detectors × 2 datasets + 5 baselines + 4 major ablations (domain/GPG/scale/mask noise), comprehensively covered.
- Writing Quality: ⭐⭐⭐⭐ Clear formula layout, well-presented algorithm pseudocode; Figures 4–6 provide intuitive ROI visual comparisons; a few appendix details are somewhat obscure.
- Value: ⭐⭐⭐⭐⭐ Directly demonstrates substantial multi-center gains for detectors trained on "synthetic + real" images, providing tangible clinical impact.

## Related Papers

- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[NeurIPS 2025\] Fractional Diffusion Bridge Models](../../NeurIPS2025/medical_imaging/fractional_diffusion_bridge_models.md)
- [\[AAAI 2026\] Constrained Best Arm Identification with Tests for Feasibility](../../AAAI2026/medical_imaging/constrained_best_arm_identification_with_tests_for_feasibility.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](../../ICLR2026/medical_imaging/unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](../../ICLR2026/medical_imaging/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](../../AAAI2026/medical_imaging/mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](../../ACL2026/medical_imaging/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)

</div>

<!-- RELATED:END -->
