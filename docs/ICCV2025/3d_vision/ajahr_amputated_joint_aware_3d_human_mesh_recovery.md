---
title: >-
  [Paper Note] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery
description: >-
  [ICCV 2025][3D Vision][Human Mesh Recovery] The first 3D human mesh recovery framework for amputees — by synthesizing 1M+ amputee images (A3D)…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Amputees"
  - "SMPL"
  - "Synthetic Dataset"
  - "Body Part Classifier"
date: 2026-05-08
content_hash: 853aec8f4d03f647
---

# AJAHR: Amputated Joint Aware 3D Human Mesh Recovery

**Conference**: ICCV 2025
**arXiv**: [2509.19939](https://arxiv.org/abs/2509.19939)  
**Code**: [chojinie/AJAHR_official](https://github.com/chojinie/AJAHR_official)  
**Area**: 3D Vision / Human Body Reconstruction
**Keywords**: Human Mesh Recovery, Amputees, SMPL, Synthetic Dataset, Body Part Classifier

## TL;DR
The first 3D human mesh recovery framework for amputees — by synthesizing 1M+ amputee images (A3D), designing the BPAC-Net amputation classifier to distinguish amputation from occlusion, and employing a dual-tokenizer switching strategy to encode amputation/normal pose priors separately. The method achieves substantial improvements on amputee data (MVE 16.87 lower than TokenHMR on ITW-amputee) while remaining competitive on non-amputee benchmarks.

## Background & Motivation
Existing HMR methods universally assume a standard human body structure, causing models to hallucinate non-existent limbs when processing amputees — yielding both inaccurate and non-inclusive results. Data availability is nearly absent: among 704 public human body datasets, none include individuals with physical disabilities. At the technical level, amputation and occlusion appear nearly identical in images — missing limbs versus occluded limbs — making discrimination difficult. Ethically, collecting real amputee data is costly and raises significant privacy concerns.

## Core Problem
How to build an inclusive human mesh recovery model capable of accurately handling both amputees and non-amputees without real amputee training data? Sub-problems include: (1) how to synthesize realistic amputee training data; (2) how to enable the model to distinguish amputation from occlusion; and (3) how to provide distinct pose priors for amputees and non-amputees.

## Method

### Overall Architecture
An end-to-end ViT-based architecture: input images are encoded by a ViT into embedding tokens, while ViTPose extracts 2D keypoints. BPAC-Net takes RGB images and keypoint heatmaps as input to classify limb amputation status; its features are injected into a Transformer decoder via cross-attention to guide pose regression. The decoder output tokens are mapped to codebook dimensions via a Bridge MLP, and the appropriate codebook (containing amputation priors vs. standard priors) is selected based on BPAC-Net predictions, with weighted aggregation yielding SMPL pose parameters $\theta$. Three additional branches regress global rotation $g$, body shape $\beta$, and camera translation $t$ respectively, which are fed into SMPL to produce the final mesh.

### Key Designs
1. **A3D Synthetic Dataset**: The core idea leverages the SMPL kinematic tree to represent amputation — setting the pose parameters of the amputated parent joint and all its child joints to zero matrices, causing child joint vertices to naturally converge to the parent joint position. The pipeline proceeds as follows: extract SMPL parameters from H36M/MPII/COCO → refine with ScoreHMR → select amputation indices (covering 12 types: hand/forearm/full arm/ankle/knee/full leg) → apply BEDLAM textures (balanced across 2 genders × 7 ethnicities) → segment backgrounds with SAM and inpaint with LaMa → composite via weak perspective projection → quality filtering (SSIM > 0.5 + no residual person detection). The pipeline yields 1M+ high-quality annotated images with SMPL parameters, 2D/3D joints, and amputation region labels.
2. **BPAC-Net (Body Part Amputation Classifier)**: RGB images ($H \times W \times 3$) and keypoint heatmaps ($H \times W \times J$) are channel-concatenated and fed into ResNet-32 with CBAM to extract spatial/semantic features $F$. Four parallel classification heads $\mathcal{H}_p$ predict the amputation status of each limb (4 classes per limb: intact + 3 amputation types). A feature alignment head produces a 1280-dimensional global vector used directly as cross-attention input to the Transformer decoder, guiding pose regression with amputation semantics. BPAC-Net serves three roles: (1) adjusting loss weights to emphasize learning of amputated regions; (2) implicitly assisting pose estimation via cross-attention features; and (3) driving amputated region visualization (zeroing SMPL parameters) at inference.
3. **Dual-Tokenizer Switching Strategy**: Two VQ-VAE tokenizers are pre-trained — $C_\text{amp}$ trained on AMASS + MOYO + A3D (containing amputation pose priors), and $C_\text{non\_amp}$ trained on AMASS + MOYO only (standard priors). Codebook size is $256 \times 2048$ with 320 pose tokens. At inference, the 4-dimensional binary vector $\hat{y}$ predicted by BPAC-Net determines codebook selection: if any limb is classified as amputated ($\|\hat{y}\|_1 > 0$), $C_\text{amp}$ is used; otherwise $C_\text{non\_amp}$ is used. Ablation experiments confirm that the dual-tokenizer strategy outperforms a single unified tokenizer.

### Loss & Training
- Tokenizer training loss: $\mathcal{L}_\text{total} = 100 \cdot \mathcal{L}_\text{mix} + 1.0 \cdot \mathcal{L}_\text{codebook} + 1.0 \cdot \mathcal{L}_\text{commitment}$ (VQ-VAE paradigm; $\mathcal{L}_\text{mix}$ includes $\ell_2$ distances over vertices, 3D joints, and pose parameters)
- Overall AJAHR loss: $\mathcal{L}_\text{overall} = 10^{-3} \cdot \mathcal{L}_\theta + 5 \times 10^{-4} \cdot \mathcal{L}_\beta + 10^{-2} \cdot \mathcal{L}_\text{2D} + 5 \times 10^{-2} \cdot \mathcal{L}_\text{3D} + 10^{-2} \cdot \mathcal{L}_\text{cls}$
- Equal sampling of amputee/non-amputee data (0.5:0.5) to prevent class imbalance
- AdamW optimizer, lr = $5 \times 10^{-6}$, weight_decay = $10^{-4}$, batch = 64, 150K iterations, 2× A100
- Partial fine-tuning: most of the ViTPose backbone is frozen; only the last 4 blocks, patch embedding, pose embedding, and the last 2 blocks of the Transformer decoder are updated
- Pose parameters use 6D continuous rotation representation to avoid numerical instability caused by zero matrices

## Key Experimental Results

### Amputee Data

| Dataset | Metric | AJAHR (Ours) | TokenHMR | Gain |
|--------|------|------------|----------|------|
| A3D | MPJPE↓ | **73.42** | 76.01 | −2.59 |
| A3D | MVE↓ | **73.19** | 74.70 | −1.51 |
| A3D | PA-MPJPE↓ | **49.42** | 49.94 | −0.52 |
| ITW-amputee | MPJPE↓ | **116.42** | 136.52 | −20.10 |
| ITW-amputee | MVE↓ | **129.25** | 146.12 | −16.87 |
| ITW-amputee | PA-MPJPE↓ | **77.18** | 91.00 | −13.82 |

### Non-Amputee Data

| Dataset | Metric | AJAHR (Ours) | TokenHMR | Gain |
|--------|------|------------|----------|------|
| 3DPW | MPJPE↓ | 95.26 | 90.23 | +5.03 |
| 3DPW | PA-MPJPE↓ | **44.94** | 47.17 | −2.23 |
| EMDB | MPJPE↓ | **112.83** | 113.26 | −0.43 |
| EMDB | PA-MPJPE↓ | **58.62** | 58.98 | −0.36 |

### BPAC-Net Classification Performance

| Dataset | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| A3D (amputee) | 0.881 | 0.756 | 0.922 | 0.820 |
| 3DOH50K (occlusion) | 0.956 | 0.956 | 1.000 | 0.977 |

### Ablation Study
- **Keypoint noise robustness**: Performance degrades only marginally under 25% noise ratio (3DPW PA-MPJPE: 44.94 → 45.08), indicating reasonable tolerance to ViTPose detection errors.
- **Multi-modal input outperforms uni-modal**: Image + keypoints jointly > keypoints only (PA-MPJPE: 44.98 vs. 46.91) > image only (PA-MPJPE: 44.98 vs. 59.54).
- **Dual tokenizer > single tokenizer**: The unified model with dual-tokenizer switching outperforms single tokenizers (amputation-only or non-amputation-only) across all datasets.
- **320 tokens is optimal**: 160 tokens yield insufficient capacity; 640 tokens introduce redundancy and interference.
- **BPAC-Net requires a strong baseline**: Attaching BPAC-Net to weaker baselines such as HMR2.0 or BEDLAM-CLIFF yields no gain; a TokenHMR-level backbone is required.

## Highlights & Insights
- **Pioneer problem formulation**: The first work to systematically define and address 3D human mesh recovery for amputees, forming a complete loop from dataset construction to methodology to evaluation.
- **SMPL zero-pose amputation representation**: The method cleverly exploits the hierarchical structure of the SMPL kinematic tree — setting joint parameters to zero matrices causes child joint vertices to naturally converge, representing amputation without modifying the model.
- **Amputation vs. occlusion disambiguation**: BPAC-Net achieves an F1 of 0.977 on the 3DOH50K occlusion dataset, demonstrating effective discrimination between "limb absent" and "limb occluded."
- **Mature synthetic data pipeline**: Multi-ethnicity balancing, quality filtering, and background diversification yield an LPIPS of only 0.155, making the pipeline transferable to other data-scarce human body tasks.

## Limitations & Future Work
- Only supports joint-level amputation types defined by the SMPL kinematic tree (12 types); finger-level loss and non-standard truncation positions are not supported.
- A3D simulates amputation only and does not include prosthetic limb scenarios.
- A synthetic-to-real domain gap persists (LPIPS = 0.155), and real amputee data remains scarce (ITW-amputee contains only 640 test images).
- MPJPE on 3DPW is slightly higher than TokenHMR (95.26 vs. 90.23), indicating a minor trade-off on non-amputee scenarios.
- Future directions include extending to prosthetic limb modeling, non-joint-boundary amputation, Paralympic sports analysis, and inclusive AR/VR applications.

## Related Work & Insights
- **vs. TokenHMR**: AJAHR augments TokenHMR with BPAC-Net and a dual-tokenizer switching strategy, substantially reducing MVE on ITW-amputee (146.12 → 129.25) while also improving PA-MPJPE on non-amputees.
- **vs. WheelPose**: WheelPose focuses on synthetic pipelines for 2D pose estimation of wheelchair users; this work is the first to address 3D mesh recovery with amputation modeling, operating at a higher-dimensional level.
- **vs. Zhou et al.**: Zhou et al. use diffusion models to reconstruct prosthetic limbs as complete limbs to assist pose estimation, sidestepping the core challenge of amputation modeling; AJAHR directly models amputation at the SMPL level.
- **vs. HMR2.0/BEDLAM-CLIFF**: The performance gap widens substantially on amputee data (HMR2.0 MVE = 154.43 vs. AJAHR = 129.25 on ITW-amputee).
- Inclusive AI and fairness-aware AI represent an important direction in social computing; this work demonstrates a systematic approach to addressing technical bias against minority populations.
- The synthetic data pipeline (SMPL manipulation + texturing + background compositing + quality filtering) is transferable to any human-related task lacking annotated data.
- The flexible use of the SMPL model is instructive — the zero-pose trick encodes structural variation without modifying the model architecture.
- The dual-tokenizer/multi-codebook strategy generalizes to other scenarios requiring conditional pose priors (e.g., varying body shapes or motion types).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneer problem formulation for amputee HMR with a complete loop spanning data, method, and evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-scenario evaluation across amputee/non-amputee/occlusion settings with extensive ablations, though real amputee data remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is well-articulated, methodology is clearly described, and the paper is well-structured.
- **Value**: ⭐⭐⭐⭐ — A pioneering contribution to inclusive AI; the synthetic data pipeline and dual-tokenizer approach offer broadly applicable insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/3d_vision/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] Fall Risk and Gait Analysis using World-Spaced 3D Human Mesh Recovery](../../CVPR2026/3d_vision/fall_risk_gait_analysis_hmr.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[ICCV 2025\] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting](contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)

</div>

<!-- RELATED:END -->
