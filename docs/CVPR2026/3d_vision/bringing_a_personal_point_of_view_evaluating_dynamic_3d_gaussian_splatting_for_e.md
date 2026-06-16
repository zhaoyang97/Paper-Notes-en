---
title: >-
  [Paper Note] Bringing a Personal Point of View: Evaluating Dynamic 3D Gaussian Splatting for Egocentric Scene Reconstruction
description: >-
  [CVPR 2026][3D Vision][Paper Note] This is an evaluation study: the authors utilize **paired egocentric (ego) and exocentric (exo) recordings of the same scene** from EgoExo4D to systematically compare four monocular dynamic 3D Gaussian Splatting models. They find that ego-view reconstruction quality is consistently inferior, and this performance gap pr
tags:
  - CVPR 2026
  - 3D Vision
date: 2026-05-08
content_hash: 9e29e36ce0dda649
---
# Bringing a Personal Point of View: Evaluating Dynamic 3D Gaussian Splatting for Egocentric Scene Reconstruction

**Conference**: CVPR 2026 (EgoVis Workshop)  
**arXiv**: [2604.23803](https://arxiv.org/abs/2604.23803)  
**Code**: https://github.com/Jaswar/evaluating-3dgs-egocentric (Available)  
**Area**: 3D Vision  
**Keywords**: Dynamic 3DGS, Egocentric Video, Novel View Synthesis, Benchmarking, Static/Dynamic Decoupled Evaluation

## TL;DR
This is an evaluation study: the authors utilize **paired egocentric (ego) and exocentric (exo) recordings of the same scene** from EgoExo4D to systematically compare four monocular dynamic 3D Gaussian Splatting models. They find that ego-view reconstruction quality is consistently inferior, and this performance gap primarily stems from the reconstruction of the **static background** (rather than dynamic objects). Consequently, they argue that existing methods are not generalizable to egocentric scenarios and necessitate specialized egocentric approaches.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the State-of-the-Art (SOTA) for high-quality and efficient novel view synthesis, spawning variants (Deformable-3DGS, 4DGS, RTGS, etc.) that reconstruct **dynamic scenes** from monocular videos. Egocentric video (head-mounted camera) is highly valuable for AR, robotics, and assistive technologies as it directly reflects the visual input received by an agent acting in the world.

**Limitations of Prior Work**: These monocular dynamic 3DGS models are almost exclusively evaluated on **exocentric** scenes (datasets like D-NeRF, Nerfies, HyperNeRF, DyCheck). They have never been rigorously tested on egocentric videos. Egocentric data presents two inherent difficulties—**fast, unpredictable camera motion** driven by the head/body, and **highly dynamic scenes** (frequent hand-object interactions). It remains unknown whether and to what extent these challenges degrade reconstruction quality.

**Key Challenge**: The only two dynamic 3DGS models specifically targeting egocentric data are EgoGaussian and DeGauss. The former claims to outperform general baselines on ego data, but the authors **cannot reproduce** this gain; the latter lacks quantitative evaluation entirely. Whether specialized egocentric models are actually needed remains a pending question due to the lack of fair benchmarks.

**Goal**: Decompose this research gap into three measurable questions: (1) Is ego reconstruction truly harder than exo? (2) Does the difficulty lie in static or dynamic regions? (3) Is camera motion correlated with reconstruction quality?

**Key Insight**: Leverage the **paired ego/exo recordings of the same scenes** provided by EgoExo4D. The exo view serves as a "natural opposite" to the ego view. Furthermore, the dataset provides ground truth intrinsic/extrinsic parameters and semi-dense point clouds, allowing baseline performance to be **isolated** from SfM errors (e.g., COLMAP), which is a crucial prerequisite for fair comparison.

**Core Idea**: Instead of proposing a new model, this paper introduces an evaluation protocol comprising "paired ego-exo + static/dynamic partitioning + camera motion correlation" to quantify where the "egocentric difficulty" lies.

## Method

### Overall Architecture
As an **evaluation/research-oriented paper**, no new model is proposed. The "Method" refers to the designed evaluation protocol. The pipeline involves: sampling 25 paired scenes from EgoExo4D $\rightarrow$ undistorting fisheye frames and generating valid pixel masks $\rightarrow$ training four dynamic 3DGS models on both ego and exo paths using uniform train/val/test splits (4-hour random hyperparameter search per scene, 3 re-runs) $\rightarrow$ measuring **masked PSNR/SSIM/LPIPS** at three levels: overall, static/dynamic partitions, and correlation with camera velocity. The tested models include one egocentric specialized model, EgoGaussian, and three general baselines: Deformable-3DGS, 4DGS, and RTGS.

As this is a benchmark study without a multi-module serial algorithm pipeline, no architecture diagram is provided. The "Key Designs" below detail the core methodological decisions.

### Key Designs

**1. Paired Ego-Exo Evaluation Protocol: Isolating "Viewpoint" as the Sole Variable**

To determine if ego is harder, confounding factors like scene content and calibration errors must be excluded. The authors chose EgoExo4D (1,286 hours of paired footage) because it provides GT poses and point clouds, which can directly initialize Gaussians, thereby **bypassing COLMAP errors**. Sampling: 20 random clips of 300 frames (10s) each (aligned with existing dynamic monocular 3DGS tasks), plus 5 hand-picked clips meeting EgoGaussian's strict data requirements (rigid motion, divisible into passive/active segments), totaling 25 scenes. For training splits: ego uses even frames for training, $i\equiv1\pmod 4$ for validation, and $i\equiv3\pmod 4$ for testing. Since exo cameras are **stationary** and a single view lacks multi-view information for 3DGS, the authors **randomly select one exo camera** at each time index $i$ to maintain monocular input while providing 3D cues. Fisheye frames are undistorted (creating black borders), and a synchronized binary mask is fed into the 3DGS pipeline to filter invalid pixels.

**2. Static/Dynamic Partitioned Evaluation: Pinpointing the Difficulty**

Overall PSNR only indicates that "ego is worse" without specifying why. The authors use **SAM 2** to manually annotate masks for dynamic objects, defining an object as dynamic at frame $i$ only if it has **actually moved** relative to frame $i-1$. This yields frame-wise dynamic masks (union of moving objects) and static masks (background). These are intersected with the undistortion masks to calculate **mPSNR / mSSIM / mLPIPS (masked versions)**. This design is highly informative: it reveals that the ego/exo mPSNR gap stems from **static regions**, while PSNR values in dynamic regions are similar—a conclusion invisible when looking only at overall metrics.

**3. EgoGaussian Replication Check: Identifying Conflicts via "Corrected Metrics + Original Data Backtesting"**

Finding that they could not reproduce EgoGaussian’s reported superiority over baselines, the authors backtested on EgoGaussian’s original data (Epic-Kitchens, HOI4D). They discovered the original metric algorithm was biased: masked areas were **zeroed out** rather than **ignored** (polluting the metric), and LPIPS was calculated without normalizing images to $[-1,1]$. The authors present two sets of results: `ours*` (original metrics) and `ours†` (corrected metrics). While `ours*` reproduces the original numbers, general baselines outperform EgoGaussian under `ours†`. This proves that the performance inversion is due to **biased metric calculation** in the original paper, serving as a methodological warning.

**4. Camera Motion Correlation Analysis: Testing the "More Motion = Better Reconstruction" Hypothesis**

The signature challenge of egocentric video is rapid camera motion. The authors analyze two metrics. **Camera Velocity**: Linear velocity $v_t\in\mathbb{R}^3$ is derived from translation between adjacent frames, taking the maximum component $\hat v_t=\max_{1\le i\le 3}|v_{ti}|$, normalized to $[0,1]$ per scene, and log-scaled (Eq. 1):

$$\bar v_t = \ln\!\Big(\frac{\hat v_t - \min_t(\hat v_t)}{\max_t(\hat v_t) - \min_t(\hat v_t)}\Big)$$

Plotting $\bar v_t$ against the mLPIPS of static regions at the same timestamp reveals that for $\bar v_t>0.5$, higher velocity correlates with higher LPIPS (worse reconstruction), with Pearson/Spearman correlations $\approx 0.5, p\ll0.05$. **Camera Baseline**: Defined as the maximum distance between any two points on the trajectory (log scale), it shows **no significant correlation** with mLPIPS. Together, these refute the prior assumption that "more camera motion $\rightarrow$ more multi-view info $\rightarrow$ better reconstruction"; in egocentric settings, motion may actually degrade quality.

### Loss & Training
No new models are trained; target loss functions of the tested models are retained. Protocol highlights: Each model undergoes a **fixed 4-hour random hyperparameter search** per scene (searching configuration ranges for 4DGS/RTGS, and deformation network width/depth/iterations for Deformable-3DGS), selecting the config with the highest validation PSNR. Each scene is **retrained 3 times** to report mean ± standard deviation. All experiments were conducted on a single NVIDIA A40. EgoGaussian bypassed hyperparameter searching as single-scene training already exceeded 4 hours.

## Key Experimental Results

### Main Results: Ego vs. Exo (mPSNR on Random Scenes, Higher is Better)

| Model | Ego mPSNR | Exo mPSNR | Exo Gain |
|------|-----------|-----------|----------|
| Deformable-3DGS | 29.75 | **33.95** | +4.20 |
| 4DGS | 28.96 | **32.05** | +3.09 |
| RTGS | 29.27 | **31.30** | +2.03 |
| EgoGaussian | — (Only measurable on 5 EgoGaussian scenes) | — | — |

- Exo outperforms ego across almost all models/metrics with very low variance. The sole exception is RTGS mPSNR on 5 EgoGaussian scenes (ego 29.26 vs. exo 28.05), yet SSIM/LPIPS remain better for exo.
- EgoGaussian (ego, EgoGaussian scenes) scored only 26.97 mPSNR, **performing worse** than all three general baselines—contradicting the original paper's claims.

### Static vs. Dynamic Partition (mPSNR on Random Scenes)

| Model | Region | Ego | Exo | Better |
|------|------|-----|-----|--------|
| Deformable-3DGS | Dynamic | **23.84** | 22.17 | Ego |
| Deformable-3DGS | Static | 30.86 | **37.23** | Exo |
| 4DGS | Dynamic | **22.61** | 22.49 | Similar/Ego |
| 4DGS | Static | 30.24 | **33.74** | Exo |
| RTGS | Dynamic | 23.52 | **23.62** | Similar |
| RTGS | Static | 30.34 | **32.47** | Exo |

- **Core Conclusion**: mPSNR for dynamic regions is similar between ego and exo (sometimes higher for ego), but exo leads significantly in static regions $\rightarrow$ **The overall ego/exo performance gap is driven by static background reconstruction.**
- Across both ego and exo, **static regions are reconstructed better than dynamic ones** (e.g., Def3DGS-Ego Static 30.86 vs. Dynamic 23.84), indicating dynamic modeling remains a universal weakness—countering Liang et al. [19]'s conclusion that performance changes little when masking the static background.

### EgoGaussian Original Data Backtesting (Table 3, Epic-Kitchens / HOI4D, mPSNR)

| Config | EK-Passive | EK-Active | Explanation |
|------|-----------|-----------|------|
| EgoGaussian (original) | 28.33 | 28.34 | Original figures |
| EgoGaussian (ours\*, original metrics) | 28.76 | 30.55 | Replication successful |
| Def3DGS (ours†, corrected metrics) | **37.54** | **32.94** | Baseline surpasses EgoGaussian |
| 4DGS (ours†, corrected metrics) | 34.40 | 29.61 | Baseline surpasses EgoGaussian |

- `ours*` (zeroed-out masks, unnormalized LPIPS) replicates the original results. Switching to corrected metrics `ours†` (ignoring masked areas, LPIPS normalized to $[-1,1]$) results in general baselines outperforming EgoGaussian $\rightarrow$ The original advantage stemmed from **biased metric calculation** rather than modeling capability.

### Key Findings
- **Difficulty lies in the static background**: This is the most counter-intuitive finding—the difficulty of egocentric reconstruction is not "moving objects" but the static background (likely due to unstable geometric cues from constant camera motion).
- **More motion $\neq$ better reconstruction**: Camera velocity is positively correlated with mLPIPS ($\approx 0.5, p \ll 0.05$), and there is no significant correlation with camera baseline—conflicting with the assumption that motion provides better multi-view information.
- **Metric algorithms can flip conclusions**: Zeroing out vs. ignoring masked areas, and LPIPS normalization, are sufficient to reverse rankings, highlighting the need for a unified masked metric standard.

## Highlights & Insights
- **Paired ego/exo comparison** is the core of this evaluation: By making "viewpoint" the sole independent variable, the claim "ego is harder" becomes a credible conclusion rather than being contaminated by dataset differences. This design is transferable to any "Modality A vs. Modality B" difficulty comparison.
- **Partitioned evaluation** redirects the field's attention: By showing the gap lies in the background, it suggests that egocentric reconstruction research should not focus solely on dynamic objects; the static background is the hidden bottleneck.
- **Replication + Correction paradigm**: Replicating original numbers first ensures the pipeline is correct, then applying corrected metrics to reach the opposite conclusion cleanly attributes the conflict to metric algorithms rather than implementation variances.

## Limitations & Future Work
- **Asymmetric evaluation**: Exo cameras are stationary, so the test viewpoint has been "seen" at different times during training, allowing the model to "memorize" rather than generalize. Ego views naturally force spatio-temporal generalization. This asymmetry requires cautious interpretation; future work needs dense or moving exo cameras.
- **Sample size (25 scenes)**: While larger than D-NeRF (8) or Nerfies (4), it may still be sensitive to outliers. The authors mitigated this with scene-wise comparisons.
- **Unrepresentative EgoGaussian scenes**: Its segments ignore hands and assume rigid motion, making these dynamic sections easier to reconstruct than real-world egocentric video.
- The correlation between camera motion and quality **is not necessarily causal** and may involve latent factors like body movement.
- DeGauss was not included for quantitative comparison as its implementation was released late.

## Related Work & Insights
- **vs. EgoGaussian [49]**: A specialized egocentric model requiring manual segmentation and object masks. This study finds it **underperforms general baselines** under corrected metrics, questioning the "specialized is better" narrative.
- **vs. Liang et al. [19]**: [19] concluded that masking static backgrounds has little impact on performance. This paper finds the opposite for egocentric data—**static is reconstructed significantly better than dynamic**, and the ego-exo gap resides in the static region—showing these conclusions do not generalize across viewpoints.
- **vs. Deformable-3DGS / 4DGS / RTGS**: These are general monocular dynamic 3DGS baselines. The paper treats them as "non-egocentric specialized" references to quantify the inherent difficulty of the egocentric setting.
- **Insight**: Applying the "paired control + region decomposition + metric correction" trio to any NVS benchmark can reveal the true bottlenecks hidden by aggregate metrics.

## Rating
- Novelty: ⭐⭐⭐⭐ No new model, but the combination of paired ego-exo control, region partitioning, and metric verification provides a unique evaluation of 3DGS.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models × 25 scenes × 3 re-runs + backtesting + significance tests; exceeds most dynamic 3DGS evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem breakdown, high information density in tables, and honest acknowledgment of limitations.
- Value: ⭐⭐⭐⭐ Establishes a fair benchmark and methodological standard for egocentric dynamic reconstruction, correcting a previously accepted "specialized model" claim.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ClipGStream: Clip-Stream Gaussian Splatting for Any Length and Any Motion Multi-View Dynamic Scene Reconstruction](clipgstream_clip-stream_gaussian_splatting_for_any_length_and_any_motion_multi-v.md)
- [\[CVPR 2026\] AeroGS: Scale-Aware Gaussian Splatting for Pose-Free Dynamic UAV Scene Reconstruction](aerogs_scale-aware_gaussian_splatting_for_pose-free_dynamic_uav_scene_reconstruc.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] $L^{2}DGS$: Low-Light Dynamic Gaussian Splatting](l2dgs_low-light_dynamic_gaussian_splatting.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)

</div>

<!-- RELATED:END -->
