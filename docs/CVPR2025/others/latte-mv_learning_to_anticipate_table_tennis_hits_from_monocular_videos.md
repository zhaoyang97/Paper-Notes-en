---
title: >-
  [Paper Note] LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos
description: >-
  [CVPR 2025][Table tennis prediction] LATTE-MV proposes a scalable system to reconstruct 3D match data from monocular table tennis match videos and trains a Transformer model to anticipate the opponent's striking intention. Combined with conformal prediction for uncertainty-aware anticipatory control, it improves the robot's return success rate in simulation from 49.9% to 59.0%.
tags:
  - "CVPR 2025"
  - "Table tennis prediction"
  - "Monocular 3D reconstruction"
  - "Anticipatory control"
  - "Transformer"
  - "Conformal prediction"
date: 2026-05-08
content_hash: 58675d8e7ad6c3af
---

# LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos

**Conference**: CVPR 2025  
**arXiv**: [2503.20936](https://arxiv.org/abs/2503.20936)  
**Code**: [https://sastry-group.github.io/LATTE-MV/](https://sastry-group.github.io/LATTE-MV/) (Project Page)  
**Area**: Other  
**Keywords**: Table tennis prediction, Monocular 3D reconstruction, Anticipatory control, Transformer, Conformal prediction

## TL;DR
LATTE-MV proposes a scalable system to reconstruct 3D match data from monocular table tennis match videos and trains a Transformer model to anticipate the opponent's striking intention. Combined with conformal prediction for uncertainty-aware anticipatory control, it improves the robot's return success rate in simulation from 49.9% to 59.0%.

## Background & Motivation

1. **Background**: Table tennis robots serve as a classic testbed for humanoid robotics research. In recent years, deep learning-driven systems have achieved human-level cooperative or competitive play. However, existing systems perform poorly against high-speed hits, primarily due to the lack of ability to anticipate the opponent's intention.

2. **Limitations of Prior Work**: Prior studies leveraging anticipation (e.g., IDDM or LSTM to predict ball placement) are limited by small dataset sizes (all <1,000 table tennis rallies) and fail to fully learn complex match dynamics. Meanwhile, publicly available large-scale 3D competitive table tennis datasets are virtually non-existent.

3. **Key Challenge**: Anticipation capabilities require vast amounts of professional match data to learn. However, collecting 3D data of professional matches traditionally relies on dedicated recording setups like multi-camera systems or RGB-D devices, which are extremely expensive and difficult to scale.

4. **Goal**: (1) How to reconstruct 3D table tennis match data at scale from low-cost monocular videos? (2) How to leverage large-scale data to learn an opponent intention anticipation model with uncertainty estimation?

5. **Key Insight**: The authors observe that a massive number of public table tennis match videos are available online. If 3D information can be automatically extracted from these monocular videos, the bottleneck of proprietary equipment can be bypassed. A fully automated 3D reconstruction pipeline can be constructed by combining various pre-trained models (YOLO segmentation, HMR human reconstruction, TrackNetV3 ball tracking).

6. **Core Idea**: Large-scale reconstruction of 3D match data (73,222 rallies) from public monocular table tennis videos to train a Transformer for anticipating opponent strikes, and utilization of conformal prediction to quantify prediction uncertainty to guide robot pre-positioning.

## Method

### Overall Architecture
The system is divided into two primary modules: (1) **3D Reconstruction Pipeline**—filters ~50 hours of actual play footage from ~800 hours of raw table tennis videos, and extracts 3D data of 73,222 rallies through entity tracking (table, racket, players, ball) and global localization (camera calibration, player SMPL reconstruction, ball 3D trajectory reconstruction); (2) **Anticipatory Control**—uses a Transformer to learn match dynamics and constructs confidence intervals via ensembling + conformal prediction to guide simulated robot pre-positioning before the opponent strikes.

### Key Designs

1. **Monocular Video 3D Reconstruction System**:

    - **Function**: Automatically extracts 3D player poses and 3D ball trajectories from monocular table tennis videos.
    - **Mechanism**: Completed in three steps—(a) Video clipping: a CNN classifier is trained to filter actual play footage (filtering 800h down to 50h); (b) Entity tracking: YOLOv8 is used to segment the table/racket surfaces, a custom UNet detects 6 table keypoints (4 corners + 2 net post intersections), HMR 2.0 reconstructs the player's SMPL mesh, and TrackNetV3 tracks the 2D ball position; (c) Global localization: external and internal camera parameters are estimated using standard ITTF table dimensions and the 10 detected calibration points, projecting players and ball into the world coordinate system. The 3D ball trajectory is reconstructed by detecting striking and bouncing points, fitting a parabolic model with Stokes' drag $x_k(t), y_k(t), z_k(t)$ between these points, and optimizing the drag coefficient $k$ by minimizing reprojection error.
    - **Design Motivation**: Achieves full automation by combining multiple pre-trained vision models without requiring specialized hardware, allowing scalability to massive public match videos on the internet.

2. **Transformer Anticipation Model**:

    - **Function**: Predicts future opponent stroke trajectories based on historical match sequences.
    - **Mechanism**: Tokenizes reconstructed data from each frame (opponent SMPL joint positions, player's root position, ball position) and feeds it into a decoder-only Transformer ($d=256$, $L=4$ layers, 16 heads) to autoregressively model $p(t) = \prod p(t_i | t_{i-1}, ..., t_1)$. Assuming a Gaussian distribution, the training objective degenerates to the MSE loss $\mathcal{L} = \sum \|\hat{t} - t\|_2^2$. The model has only 3.2M parameters and achieves inference in <10ms.
    - **Design Motivation**: Transformers effectively capture temporal dependencies and match dynamics patterns on large-scale data while remaining fast enough during inference to satisfy real-time constraints.

3. **Conformal Prediction for Uncertainty Quantification**:

    - **Function**: Constructs confidence intervals with theoretical coverage guarantees for predictions.
    - **Mechanism**: Trains an ensemble of 5 Transformers (each trained on non-overlapping data subsets), taking the ensemble mean $\hat{f}(X)$ and standard deviation $\hat{\sigma}(X)$ for each sample. Normalized residuals $R_i = |Y_i - \hat{f}(X_i)| / \hat{\sigma}(X_i)$ are computed on a calibration set to obtain the quantile $\hat{q}_\alpha$. During prediction, the confidence interval is constructed as $\mathcal{C}_\alpha(X) = [\hat{f}(X) \pm \hat{q}_\alpha \hat{\sigma}(X)]$. Taking the Cartesian product of these intervals constructed separately for the x, y, and z axes theoretically guarantees $\Pr(b_t \in \mathcal{C}_\alpha) \geq 1-3\alpha$.
    - **Design Motivation**: Ensemble standard deviation alone cannot guarantee coverage; conformal prediction provides finite-sample coverage guarantees without distribution assumptions and can filter out predictions with overly high uncertainty.

### Loss & Training
- The Transformer is trained using MSE loss (equivalent to Gaussian NLL with fixed variance) for 200 epochs.
- Dataset split: 5 training subsets + 2,500 calibration set + 1,000 test set.
- Reprojection error (Eq. 5) is utilized when optimizing the drag coefficient $k$ for ball trajectory reconstruction.

## Key Experimental Results

### Main Results

| Pre-positioning Strategy | Return Rate | Return Accuracy (m) | Pose Accuracy (m / °) |
|-----------|--------|-------------|-----------------|
| Baseline (No Pre-positioning) | 49.9% | 0.497 | 0.25 / 13.3° |
| Anticipatory (Ours) | **59.0%** | **0.463** | **0.19 / 9.86°** |
| Oracle (Ground Truth Trajectory) | 64.5% | 0.453 | 0.15 / 6.26° |

In the KUKA robot simulation, anticipatory control improves the return rate by 9.1 percentage points (+18.2%), bridging approximately 62% of the gap to the Oracle.

### Conformal Prediction Coverage

| $1-\alpha$ | $\mathcal{C}_{\alpha,x}$ | $\mathcal{C}_{\alpha,y}$ | $\mathcal{C}_{\alpha,z}$ | Theoretical $1-3\alpha$ | Empirical $\mathcal{C}_\alpha$ |
|-----------|------|------|------|----------|---------|
| 0.90 | 0.885 | 0.906 | 0.905 | 0.70 | 0.763 |
| 0.85 | 0.830 | 0.858 | 0.862 | 0.55 | 0.652 |
| 0.80 | 0.782 | 0.796 | 0.822 | 0.40 | 0.532 |

### Key Findings
- Although confidence intervals over-cover in most cases for extreme hits (y-value > 0.75m), in biased subsets, more than 85% lean in the correct direction.
- The average ball speed in the dataset is 11.25 m/s, with a hit interval of 0.56 seconds. The data distribution is biased: players tend to strike towards the right side (bimodal y distribution).
- Reconstruction accuracy: Average ball position error is 8.9 cm, and human joint error is 28 cm (at 640x360 resolution).
- The dataset fails to reconstruct the final segment of scoring points (moments when the ball is not returned), introducing significant distribution bias.

## Highlights & Insights
- **The approach of leveraging public videos to build a large-scale 3D dataset is highly inspiring**: Automatically extracting data via a combined pipeline of pre-trained models (YOLO + HMR + TrackNet + camera calibration) presents a generalized workflow transferrable to other sports domains (e.g., basketball, tennis).
- **The integration of conformal prediction and ensemble models** provides uncertainty quantification with theoretical guarantees. This is more reliable than standard ensemble deviation alone, which is particularly crucial for safety-critical robotic applications.
- **The generalized pipeline for anticipatory control** (prediction $\rightarrow$ confidence intervals $\rightarrow$ reachable set check $\rightarrow$ pre-positioning target selection) can be transferred to other fast-response human-robot interaction scenarios.

## Limitations & Future Work
- Racket poses are not reconstructed, failing to capture spin information, which is critical in professional table tennis.
- Reconstruction accuracy is limited (8.9 cm for the ball, 28 cm for the human body), with low-resolution video being the main bottleneck.
- Obvious dataset bias: It fails to capture the final strike of winning points (the most critical strategic information) and exhibits cohort-level preferences in stroke direction.
- Verified only in simulation, without deployment to real-world hardware.
- The controller employs a simple blocking strategy (no swinging) and does not optimize the return strategy itself.

## Related Work & Insights
- **vs i-sim2real (Abeyruwan et al.)**: Achieved human-robot cooperative table tennis, but did not involve anticipation. This work complements the anticipation dimension.
- **vs IDDM (Intent-Driven Dynamics Model)**: Modeled human intentions using latent variable models, but used a dataset of <1,000 rallies. This work scales up by 73 times with a large-scale dataset + Transformer replacement.
- The reconstruction pipeline methodology of this work can serve as a reference template for other motion analysis research.

## Rating
- Novelty: ⭐⭐⭐⭐ First to reconstruct a 3D table tennis dataset from large-scale monocular videos and apply it to anticipatory control.
- Experimental Thoroughness: ⭐⭐⭐ Thoroughly evaluated in simulation but lacks real-world hardware experiments, with limited analysis on reconstruction accuracy.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed method description, though some details (e.g., controller design) are deferred to the appendix.
- Value: ⭐⭐⭐⭐ High value of the dataset and pipeline to the community; the anticipation framework is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](../../ACL2025/others/tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)
- [\[CVPR 2026\] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos](../../CVPR2026/others/videoworld_2_learning_transferable_knowledge_from_real-world_videos.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](../../ACL2025/others/mapmake_schema_guided_text_to_table_generation.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
