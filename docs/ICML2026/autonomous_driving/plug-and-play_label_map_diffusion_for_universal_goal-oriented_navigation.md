---
title: >-
  [Paper Note] Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation
description: >-
  [ICML 2026][Autonomous Driving][Goal-Oriented Navigation] This paper proposes PLMD: it merges BEV semantic maps and obstacle maps into a Label Map and uses DDPM to complete semantics and obstacle labels in unexplored are…
tags:
  - "ICML 2026"
  - "Autonomous Driving"
  - "Goal-Oriented Navigation"
  - "Label Map"
  - "DDPM"
  - "SPADE Modulation"
  - "HM3D"
  - "MP3D"
date: 2026-05-08
content_hash: 6500b4e6291eb671
---

# Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation

**Conference**: ICML 2026  
**arXiv**: [2605.05960](https://arxiv.org/abs/2605.05960)  
**Code**: Not yet released  
**Area**: Embodied Navigation / BEV Map Completion / Diffusion Models  
**Keywords**: Goal-Oriented Navigation, Label Map, DDPM, SPADE Modulation, HM3D, MP3D

## TL;DR
This paper proposes PLMD: it merges BEV semantic maps and obstacle maps into a Label Map and uses DDPM to complete semantics and obstacle labels in unexplored areas under obstacle prior modulation. As a plug-and-play module for any GON policy, it consistently achieves new SOTA results on HM3D/MP3D across ON, IIN, and MRON tasks.

## Background & Motivation
**Background**: Goal-Oriented Navigation (GON) includes three major subtasks: ObjectNav (ON), Instance ImageNav (IIN), and Multi-Robot ObjectNav (MRON). Prevailing modular methods construct egocentric semantic BEV maps and use RL/LLM to plan long-term goals on these maps (e.g., SemExp, IEVE, 3D-Mem, Co-NavGPT).

**Limitations of Prior Work**: The bottleneck of modular methods is that "only observed areas have semantics"—robots must traverse the entire room to locate targets, which is highly inefficient. Previous works (Ji et al. 2024, Li et al. 2025) attempted to use diffusion models for completion on semantic BEVs but only learned "statistical correlations between semantics" (e.g., tables and chairs often co-occur), ignoring the more rigid structural prior of "obstacle layout." This results in semantic hallucinations such as room boundary drift, wall penetration, or objects growing on walls in unobserved areas.

**Key Challenge**: BEV differs from natural images—large areas are free space and object pixels are sparse; pure semantic diffusion fails to obtain a stable geometric skeleton. However, the "geometric structure of obstacles/walls" follows strong learnable statistical patterns within houses (closed walls, connected doorways). Completing obstacles first and then using them to modulate semantics avoids "imagining objects through walls."

**Goal**: (1) Provide plausible semantic and obstacle predictions for unobserved areas without retraining any GON policies; (2) Explicitly utilize obstacle priors to resolve semantic hallucinations; (3) Maintain compatibility with RL, SSL, and LLM navigation paradigms.

**Key Insight**: Merge semantic and obstacle maps into a Label Map visualized with a unified palette. Employ two serial diffusion networks—stabilize the obstacle map prior first, then modulate semantic map diffusion via SPADE residual blocks, ensuring semantic generation is constrained by the geometric skeleton at each denoising step.

**Core Idea**: A "cascaded diffusion completion" where obstacles lead and semantics follow, combined with HDBSCAN clustering to extract candidate goals from the predicted Label Map. The entire module is completely plug-and-play.

## Method

### Overall Architecture
PLMD is integrated behind any semantic map-based GON policy in four steps: (I) The robot performs normal navigation to build an egocentric semantic and obstacle BEV $M_t\in\mathbb R^{(n+4)\times H\times W}$ ($n$ semantic channels + occupancy/free space/position); (II) Render semantics and obstacles into a visual Label Map $L_{t}=[S_{vt},C_{vt}]$ using a fixed palette, with a mask $m$ for unobserved areas; (III) The obstacle network $\mathcal G_\phi$ performs reverse SDE denoising on the obstacle map to obtain $c_t^0$, while the semantic network $\tilde{\mathcal G}_\phi$ performs semantic denoising via SPADE residual blocks modulated by $c_t^{\tau-1}$ to obtain $s_t^0$, forming the predicted Label Map $L_t^P=[S_t^P,C_t^P]$; (IV) Extract the core of the largest cluster of the target color using HDBSCAN on $L_t^P$ as the long-term goal and navigate via FMM local planning. If no reliable cluster exists, fall back to the original policy; the prediction refreshes every 50 steps after the first 100 steps.

### Key Designs

1.  **Obstacle Prior Modulated Cascaded Diffusion**:
    *   **Function**: Generate the obstacle map first as a geometric skeleton, then use it to modulate features at each timestep of semantic diffusion to ensure semantics do not penetrate walls or "grow" in free space.
    *   **Mechanism**: The obstacle map $c_\tau$ evolves along the SDE $\mathrm dc=\theta_\tau(\mu_c-c)\mathrm d\tau+\delta_\tau\mathrm dw$. Reverse denoising is done via the conditional network $\mathcal G_\phi(c_\tau,\mu_c,\tau)$ minimizing $\mathcal L_\alpha=\sum_\tau\alpha_\tau\mathbb E[\|c_\tau-(\mathrm dc_\tau)_{\mathcal G_\phi}-c_{\tau-1}^*\|_p]$. The semantic network $\tilde{\mathcal G}_\phi(s_\tau,c_{\tau-1},\tau)$ applies SPADE on the $k$-th layer feature $f_\tau^k$: $\hat f_\tau^k=\mathbf W_\gamma^{(k)}(c_{\tau-1})f_\tau^k+\mathbf b_\beta^{(k)}(c_{\tau-1})$, where scale and bias are determined by $c_{\tau-1}$. $\mathcal G_\phi$ is pre-trained individually, then frozen to train $\tilde{\mathcal G}_\phi$.
    *   **Design Motivation**: Semantic pixels are sparse in BEV. Without a geometric skeleton, diffusion draws randomly in early steps. Obstacle structures (wall lines, doorways) are denser and more stable in data. Generating obstacles before semantics aligns with human intuition—"sketch the room layout before placing furniture." SPADE modulation is a proven "layout-driven semantic" mechanism from GauGAN.

2.  **Unified Label Map Representation**:
    *   **Function**: Compress multi-channel semantic maps ($n$ classes) and obstacle maps (2 classes) into a single three-channel color image so diffusion models can directly utilize image generation backbones.
    *   **Mechanism**: Map $n+2$ class labels to different RGB values using a fixed palette; fill unobserved areas with white as a mask. On the output side, reverse-lookup the palette to get predicted semantic vectors $S_t^P\in\mathbb R^{n\times H\times W}$ and obstacle vectors $C_t^P\in\mathbb R^{2\times H\times W}$, then concatenate into $L_t^P=[S_t^P,C_t^P]$.
    *   **Design Motivation**: Direct diffusion on $n+4$ channels would require redesigning backbones. Visualizing as a Label Map allows direct reuse of mature DDPM/U-Net image inpainting experience. The palette "discretizes" semantic categories, reducing confusion between adjacent classes.

3.  **Density Clustering-based Candidate Goal Extraction**:
    *   **Function**: Locate reliable target category positions from the predicted Label Map as long-term goals to avoid being misled by isolated noise.
    *   **Mechanism**: Collect all pixel coordinates $X=\{x_1,\dots,x_n\}$ of the target color. Use HDBSCAN to extract clusters $Z=\text{HDBSCAN}(X,N)$ with $N=5$. Rank clusters by a composite score: "50% density + 40% cluster size + 10% distance to start." Select the highest-scored cluster core as the long-term goal for FMM planning; if no cluster meets the threshold, continue with the original strategy.
    *   **Design Motivation**: Diffusion models inevitably scatter some noise, and the single point with maximum probability is prone to error. HDBSCAN naturally handles noise without requiring a pre-defined number of clusters. The composite score balances "target reliability" and "exploration efficiency."

### Loss & Training
Two stages: (1) Train the obstacle network $\mathcal G_\phi$ with $\mathcal L_\alpha$; (2) Freeze $\mathcal G_\phi$ and train the semantic network $\tilde{\mathcal G}_\phi$ using $\mathcal L_\zeta(\phi)=\sum_\tau\zeta_\tau\mathbb E[\|s_\tau-(\mathrm ds_\tau)_{\tilde{\mathcal G}_\phi}s_{\tau-1}-s_{\tau-1}^*\|_p]$. Data: HM3D_v0.1 + MP3D run with FBE for $\mathcal N=2000$ episodes, saving a mask pair every 25 steps against the final complete map as GT. RedNet for semantic segmentation, $n=40$ classes, $480\times 480$ resolution resized to $256\times 256$ input. Adam optimizer with $\beta_1=0.9, \beta_2=0.99$, and $T=100$ denoising steps.

## Key Experimental Results

### Main Results
Three tasks across multiple datasets (HM3D_v0.1/v0.2, MP3D). PLMD combined with OpenFMNav (ON), FBE/IEVE (IIN), and MCoCoNav (MRON):

| Task | Dataset | Prev. SOTA | PLMD (Ours) | Gain |
|------|--------|---------|-------------|------|
| ON | HM3D_v0.1 | SGM 0.602 / 0.308 | **0.656 / 0.333** | +5.4% / +2.5% SR/SPL |
| ON | MP3D | UniGoal 0.410 / 0.164 | **0.426 / 0.164** | +1.6% SR |
| IIN | HM3D_v0.2 | IEVE 0.702 / 0.252 | **0.776 / 0.283** | +7.4% / +3.1% |
| MRON | HM3D_v0.2 | MCoCoNav 0.716 / 0.387 | **0.762 / 0.406** | +4.6% / +1.9% |
| MRON | MP3D | MCoCoNav 0.568 / 0.334 | **0.591 / 0.382** | +2.3% / +4.8% |

The largest gain is in IIN (+7.4% SR) because IIN relies heavily on complete semantic maps for instance matching, thus benefiting the most from map completion.

### Ablation Study
**Comparison with other diffusion completion methods (MRON HM3D_v0.2):**

| Method | SR | SPL | PSNR |
|------|------|------|------|
| IR-SDE | 0.698 | 0.370 | 29.895 |
| StrDiffusion | 0.729 | 0.374 | 31.486 |
| **PLMD** | **0.762** | **0.406** | **34.284** |

**Key component ablation (HM3D_v0.2):**

| Config | ON SR | IIN SR | MRON SR | PSNR |
|------|-------|--------|---------|------|
| Full | 0.665 | 0.776 | 0.762 | 34.284 |
| w/o $\mathcal G_\phi$ (No Obstacle Prior) | 0.636 | 0.730 | 0.714 | 30.437 |
| w/o Obstacle Map | 0.626 | 0.727 | 0.717 | 34.284 |
| w/o HDBSCAN Clustering | 0.657 | 0.757 | 0.748 | 34.284 |
| Replace observed with predicted (Extreme) | 0.640 | – | 0.731 | 34.284 |

Removing the obstacle prior $\mathcal G_\phi$ drops PSNR by 3.85 and IIN SR by 4.6%; clustering affects IIN most (−1.9%), as long-term goals rely on a single reliable location.

### Key Findings
- **Obstacle prior is the quality ceiling**: PSNR confirms the decisive role of the geometric skeleton in diffusion generation quality; this holds for all BEV/layout generation tasks.
- **Refresh frequency "every 50 steps starting from 100" is universal**: Starting too early lacks map information (high input noise), and refreshing too frequently slows down inference; task-level dynamic scheduling is not required.
- **Open-vocabulary generalization**: Using Grounded SAM for segmentation (PLMD†), SR=0.354 on unseen classes (lamp, toy car, microwave) outperforms MCoCoNav at 0.327, proving PLMD learns "category-agnostic geometric-semantic associations."
- **Gap with GT label map**: Feeding GT Label Map to OpenFMNav yields SR 0.742 vs PLMD's 0.665—an 8% gap remains for future improvement.

## Highlights & Insights
- **"Structure first, semantics follow" cascade**: Using the obstacle map as a skeleton via SPADE at each layer is a clever migration of SPADE GauGAN to BEV inpainting, reusable in autonomous driving OccNet/HD-Map completion.
- **True plug-and-play**: Consistently improves performance without touching RL/LLM navigation policies—a great example of the deployment value of "orthogonal completion modules."
- **Goal selection with HDBSCAN + composite score**: Solves the engineering problem of diffusion noise, representing a key step from research to deployment.
- **Unified Label Map representation**: Compressing multi-channels to RGB allows immediate use of the SOTA image inpainting toolchain for DDPM backbones.

## Limitations & Future Work
- Training data is collected via FBE, which may introduce exploration bias (unvisited area distribution may differ from real deployment); there is no quantitative analysis of generalization to unfamiliar furniture layouts.
- Obstacle prediction errors cascade into semantic errors (SPADE modulation is unidirectional); failure cases are not discussed.
- 100-step DDPM plus triggering every 50 steps introduces latency (Open-vocabulary Grounded SAM PLMD† takes ~1500s per call), necessitating latent/consistency acceleration.
- Fixed palette for three-channel colorization might cause category confusion as $n$ increases; learnable token embeddings could be considered.
- Map fusion between multiple robots is still independently predicted; shared latents could potentially accelerate consistency.

## Related Work & Insights
- **vs SGM (zhang2024) / T-Diff (yu2024)**: These use pure semantic correlation for completion without obstacle skeletons; PLMD outperforms SGM by +5.4% SR on ON HM3D_v0.1.
- **vs StrDiffusion (liu2024)**: Uses structural sparsity to reduce hallucinations but remains at the semantic level; PLMD explicitly models obstacles → SPADE modulates semantics, resulting in +2.8 PSNR.
- **vs IEVE (lei2024)**: IIN SOTA; adding PLMD still yields +7.4% SR, proving PLMD is orthogonal to strong existing IIN policies.
- **vs AD BEV Occupancy Prediction (OccWorld, SurroundOcc)**: Both complete unobserved occupancy, but PLMD handles free/occupied/semantic jointly, making it more tailored for embodied navigation semantics.
- **Inspiration**: The obstacle → semantic cascade can be applied to joint OccNet + HDMap prediction in autonomous driving or post-processing in indoor SLAM.

## Rating
- Novelty: ⭐⭐⭐⭐ Obstacle-modulated semantic diffusion is a clear novelty, though the SPADE concept is borrowed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 tasks × multiple datasets × multiple backbones + frequency sweep + open-vocabulary + full ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, rigorous formulas, and sufficient details in the appendix.
- Value: ⭐⭐⭐⭐ High industrial value as a plug-and-play BEV completion module, with spillover insights for autonomous driving BEV tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](../../AAAI2026/autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[ICCV 2025\] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation](../../ICCV2025/autonomous_driving/igl-nav_incremental_3d_gaussian_localization_for_image-goal_navigation.md)
- [\[ICML 2026\] Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering](mitigating_error_accumulation_in_continuous_navigation_via_memory-augmented_kalm.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[ICLR 2026\] SPACeR: Self-Play Anchoring with Centralized Reference Models](../../ICLR2026/autonomous_driving/spacer_self-play_anchoring_with_centralized_reference_models.md)

</div>

<!-- RELATED:END -->
