---
title: >-
  [Paper Note] Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation
description: >-
  [ICML 2026][Robotics][Goal-Oriented Navigation] This paper proposes PLMD: merging BEV semantic and obstacle maps into a Label Map, using DDPM to complete unexplored regions’ semantic + obstacle labels under obstacle prio…
tags:
  - "ICML 2026"
  - "Robotics"
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
**Code**: Not released  
**Area**: Embodied Navigation / BEV Map Completion / Diffusion Models  
**Keywords**: Goal-Oriented Navigation, Label Map, DDPM, SPADE Modulation, HM3D, MP3D

## TL;DR
This paper proposes PLMD: merging BEV semantic and obstacle maps into a Label Map, using DDPM to complete unexplored regions’ semantic + obstacle labels under obstacle priors, serving as a plug-and-play module for any GON policy. It consistently sets new SOTA on ON / IIN / MRON tasks across HM3D/MP3D.

## Background & Motivation
**Background**: Goal-oriented navigation (GON) includes three sub-tasks: ObjectNav, Instance Image Navigation (IIN), and Multi-Robot ObjectNav (MRON). Mainstream modular approaches build egocentric semantic BEV maps, then use RL/LLM to plan long-term goals on the map (e.g., SemExp, IEVE, 3D-Mem, Co-NavGPT).

**Limitations of Prior Work**: The Achilles’ heel of modular methods is “semantics only in observed regions”—robots must traverse the entire room to locate targets, leading to inefficiency. Prior works (Ji et al. 2024, Li et al. 2025) tried using diffusion models to complete unknown regions on semantic BEV, but only learned “statistical correlations between semantics” (e.g., tables and chairs co-occur), ignoring the more rigid “obstacle layout” prior. This leads to semantic hallucinations such as room boundary drift, wall penetration, and objects growing on walls.

**Key Challenge**: BEV differs from natural images—large free space, sparse object pixels; pure semantic diffusion cannot yield stable geometric skeletons. However, “geometric structure of obstacles/walls” within houses has strong statistical regularities (walls must be closed, doors connected). If obstacles are completed first and then used to modulate semantics, “objects imagined through walls” can be avoided.

**Goal**: (1) Provide reasonable semantic + obstacle predictions for unobserved regions without retraining any GON policy; (2) Explicitly leverage obstacle priors to address semantic hallucinations; (3) Compatible with RL/SSL/LLM navigation paradigms.

**Key Insight**: Merge semantic and obstacle maps into a Label Map visualized with a unified palette, use two serial diffusion networks—obstacle prior stabilizes first, then semantic diffusion is modulated by SPADE residuals at each denoising step, constraining semantics with geometric skeletons.

**Core Idea**: “Obstacle first + semantic follow” cascaded diffusion completion, combined with HDBSCAN clustering to extract candidate goals from predicted Label Maps, forming a fully plug-and-play module.

## Method

### Overall Architecture
PLMD attaches to any semantic map-based GON policy, in four steps: (I) The robot executes navigation, building egocentric semantic + obstacle BEV $M_t\in\mathbb R^{(n+4)\times H\times W}$ ($n$ semantic channels + occupancy/free space/position); (II) Use a fixed palette to render semantics and obstacles into a visual Label Map $L_{t}=[S_{vt},C_{vt}]$, generate mask $m$ for unobserved regions; (III) Obstacle network $\mathcal G_\phi$ first denoises the obstacle map via reverse SDE to get $c_t^0$, semantic network $\tilde{\mathcal G}_\phi$ uses $c_t^{\tau-1}$ to modulate features at each layer via SPADE residual blocks for semantic denoising to get $s_t^0$, then combine into predicted Label Map $L_t^P=[S_t^P,C_t^P]$; (IV) Use HDBSCAN clustering on $L_t^P$ to find the largest cluster core of the target color as the long-term goal, and plan locally with FMM. If no reliable cluster is found, revert to the original policy; after 100 navigation steps, refresh prediction every 50 steps.

### Key Designs

1. **Cascaded Diffusion with Obstacle Prior Modulation**:

    - **Function**: Generate the obstacle map as geometric skeleton first, then use it to modulate features at each semantic diffusion step, ensuring semantics do not cross walls or appear in free space.
    - **Mechanism**: Obstacle map $c_\tau$ evolves along SDE $\mathrm dc=\theta_\tau(\mu_c-c)\mathrm d\tau+\delta_\tau\mathrm dw$, reverse denoising via conditional network $\mathcal G_\phi(c_\tau,\mu_c,\tau)$ minimizes $\mathcal L_\alpha=\sum_\tau\alpha_\tau\mathbb E[\|c_\tau-(\mathrm dc_\tau)_{\mathcal G_\phi}-c_{\tau-1}^*\|_p]$. Semantic network $\tilde{\mathcal G}_\phi(s_\tau,c_{\tau-1},\tau)$ uses SPADE at the $k$-th layer: $\hat f_\tau^k=\mathbf W_\gamma^{(k)}(c_{\tau-1})f_\tau^k+\mathbf b_\beta^{(k)}(c_{\tau-1})$, with $c_{\tau-1}$ determining scale/bias. Pretrain $\mathcal G_\phi$ alone, then freeze it to train $\tilde{\mathcal G}_\phi$.
    - **Design Motivation**: Semantic pixels are sparse on BEV; without geometric skeleton, early diffusion steps scatter noise. Obstacle structures (walls, doors) are denser and more stable, so generating obstacles first then semantics matches human intuition (“draw the room layout before placing objects”). SPADE modulation, as validated in GauGAN, naturally fits as a “layout-driven semantics” mechanism.

2. **Unified Label Map Representation**:

    - **Function**: Compress multi-channel semantic map ($n$ classes) + obstacle map (2 classes) into a single 3-channel color image, enabling direct use of image generation backbones in diffusion models.
    - **Mechanism**: Use a fixed palette to map $n+2$ label classes to distinct RGB; unobserved regions are filled white as mask; at output, reverse-map using the same palette to obtain predicted semantic vector $S_t^P\in\mathbb R^{n\times H\times W}$ and obstacle vector $C_t^P\in\mathbb R^{2\times H\times W}$, then combine into $L_t^P=[S_t^P,C_t^P]$.
    - **Design Motivation**: Direct diffusion on $n+4$ channels requires redesigning the backbone; visualized Label Map allows direct reuse of mature DDPM/U-Net image inpainting techniques; the palette naturally discretizes semantic classes, reducing confusion between adjacent classes.

3. **Density Clustering-based Candidate Goal Extraction**:

    - **Function**: Identify reliable locations of target classes from the predicted Label Map as long-term goals, avoiding being misled by isolated noise points.
    - **Mechanism**: Collect all pixel coordinates $X=\{x_1,\dots,x_n\}$ of the target color, use HDBSCAN to extract clusters $Z=\text{HDBSCAN}(X,N)$, $N=5$. Rank clusters by a composite score: “density 50% + cluster size 40% + distance from start 10%”, select the highest-scoring cluster core as the long-term goal, and plan with FMM; if no cluster meets the threshold, continue with the original navigation policy.
    - **Design Motivation**: Diffusion models inevitably scatter some noise points; picking the single highest-probability point is risky. HDBSCAN naturally handles noise and does not require predefining the number of clusters; the composite score balances “goal reliability” and “exploration efficiency”.

### Loss & Training
Two stages: (1) Train obstacle network $\mathcal G_\phi$ alone with $\mathcal L_\alpha$; (2) Freeze $\mathcal G_\phi$, train semantic network $\tilde{\mathcal G}_\phi$ with $\mathcal L_\zeta(\phi)=\sum_\tau\zeta_\tau\mathbb E[\|s_\tau-(\mathrm ds_\tau)_{\tilde{\mathcal G}_\phi}s_{\tau-1}-s_{\tau-1}^*\|_p]$. Data: HM3D_v0.1 + MP3D, using FBE to collect $\mathcal N=2000$ episodes, saving a mask pair every 25 steps, with the final complete map as GT; RedNet for semantic segmentation, $n=40$ classes, resolution $480\times 480$ → model input $256\times 256$. Adam $\beta_1=0.9,\beta_2=0.99$, $T=100$ denoising steps.

## Key Experimental Results

### Main Results
Three tasks × multiple datasets (HM3D_v0.1/v0.2, MP3D). PLMD combined with OpenFMNav (ON), FBE/IEVE (IIN), MCoCoNav (MRON):

| Task | Dataset | Prev. SOTA | PLMD (Ours) | Gain |
|------|--------|---------|-------------|------|
| ON | HM3D_v0.1 | SGM 0.602 / 0.308 | **0.656 / 0.333** | +5.4% / +2.5% SR/SPL |
| ON | MP3D | UniGoal 0.410 / 0.164 | **0.426 / 0.164** | +1.6% SR |
| IIN | HM3D_v0.2 | IEVE 0.702 / 0.252 | **0.776 / 0.283** | +7.4% / +3.1% |
| MRON | HM3D_v0.2 | MCoCoNav 0.716 / 0.387 | **0.762 / 0.406** | +4.6% / +1.9% |
| MRON | MP3D | MCoCoNav 0.568 / 0.334 | **0.591 / 0.382** | +2.3% / +4.8% |

IIN shows the largest improvement (+7.4% SR), as IIN heavily relies on complete semantic maps for instance matching, maximizing the benefit from map completion.

### Ablation Study
**Comparison with other diffusion completion methods (MRON HM3D_v0.2):**

| Method | SR | SPL | PSNR |
|------|------|------|------|
| IR-SDE | 0.698 | 0.370 | 29.895 |
| StrDiffusion | 0.729 | 0.374 | 31.486 |
| **PLMD** | **0.762** | **0.406** | **34.284** |

**Key component ablation (HM3D_v0.2):**

| Configuration | ON SR | IIN SR | MRON SR | PSNR |
|------|-------|--------|---------|------|
| Full | 0.665 | 0.776 | 0.762 | 34.284 |
| w/o $\mathcal G_\phi$ (no obstacle prior) | 0.636 | 0.730 | 0.714 | 30.437 |
| w/o obstacle map | 0.626 | 0.727 | 0.717 | 34.284 |
| w/o HDBSCAN clustering | 0.657 | 0.757 | 0.748 | 34.284 |
| Replace observed map with predicted (extreme) | 0.640 | – | 0.731 | 34.284 |

Removing obstacle prior $\mathcal G_\phi$ drops PSNR by 3.85 and IIN SR by 4.6%; clustering impacts IIN most (−1.9%), as long-term goals rely more on a single reliable location.

### Key Findings
- Obstacle map prior is the quality ceiling: PSNR alone proves the decisive role of geometric skeleton in diffusion generation quality; this holds for all BEV/layout generation tasks.
- Execution frequency “refresh every 50 steps after 100 steps” is a universal setting: starting too early means insufficient map info (more input noise), refreshing too often slows inference; no need for task-level dynamic scheduling.
- Open-vocabulary generalization: Using Grounded SAM for semantic segmentation (PLMD†), achieves SR=0.354 on unseen classes (lamp, toy car, microwave), outperforming MCoCoNav 0.327, proving PLMD learns “geometry-semantic associations independent of specific classes”.
- Gap with GT label map: Feeding GT Label Map to OpenFMNav yields SR 0.742 vs PLMD 0.665—an 8% gap remains for future improvement.

## Highlights & Insights
- **“Draw structure first, then fill semantics” cascade**: Using obstacle map as geometric skeleton to modulate semantics at each SPADE layer is a neat transfer of GauGAN’s idea to BEV inpainting, and is directly reusable in autonomous driving OccNet/HD-Map completion.
- **Truly plug-and-play**: No modification to RL/LLM navigation policies, just map completion yields consistent gains—sets a good example for deploying such orthogonal “completion modules”.
- **HDBSCAN + composite score for goal selection**: Solves the practical engineering issue of “diffusion noise”, a key step from research to deployment.
- **Unified Label Map representation**: Compressing multi-channel into RGB enables direct reuse of DDPM backbones, unlocking the full SOTA image inpainting toolchain.

## Limitations & Future Work
- Training data is collected via FBE policy, possibly introducing exploration bias (distribution of unreached regions may differ from real deployment); no quantitative analysis for generalization to unfamiliar furniture layouts.
- Obstacle prediction errors can cascade to amplify semantic errors (SPADE modulation is unidirectional); failure cases are not discussed.
- 100-step DDPM plus ON/MRON triggering every 50 steps introduces significant latency for real-time robots (open-vocabulary Grounded SAM PLMD† takes over 1500 seconds per run), requiring latent/consistency acceleration.
- Three-channel coloring uses a fixed palette; as the number of classes $n$ increases, color crowding may cause class confusion; learnable token embedding could be considered.
- Map fusion among multiple robots remains independent prediction; shared latent acceleration for consistency is not exploited.

## Related Work & Insights
- **vs SGM (zhang2024) / T-Diff (yu2024)**: Pure semantic correlation completion, no obstacle skeleton; PLMD achieves +5.4% SR on ON HM3D_v0.1.
- **vs StrDiffusion (liu2024)**: Uses structural sparsity to reduce hallucination, but still at semantic level; PLMD explicitly models obstacles → SPADE modulates semantics, PSNR +2.8.
- **vs IEVE (lei2024)**: IIN SOTA, PLMD adds +7.4% SR on top, proving orthogonality with strong existing IIN policies.
- **vs Autonomous Driving BEV Occupancy Prediction (OccWorld, SurroundOcc)**: All complete unobserved occupancy, but PLMD jointly handles free/occupied/semantic, more aligned with embodied navigation semantics.
- **Insights**: Obstacle → semantic cascaded modulation can be applied to autonomous driving OccNet + HDMap joint prediction, indoor SLAM post-processing, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ Obstacle prior modulated semantic diffusion is a clear contribution, though SPADE itself is borrowed
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 tasks × multiple datasets × multiple backbones + frequency sweep + open vocabulary + full ablation
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, rigorous formulas, detailed appendix
- Value: ⭐⭐⭐⭐ Plug-and-play BEV completion module, high industrial value, with spillover insights for autonomous driving BEV tasks

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
