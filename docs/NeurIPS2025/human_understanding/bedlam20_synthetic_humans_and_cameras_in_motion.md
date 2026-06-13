---
title: >-
  [Paper Note] BEDLAM2.0: Synthetic Humans and Cameras in Motion
description: >-
  [NeurIPS 2025][Human Understanding][synthetic data] BEDLAM2.0 is a comprehensive upgrade over BEDLAM, introducing diverse camera motions (synthetic translation/tracking/orbit + handheld/head-mounted capture)…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "synthetic data"
  - "SMPL-X"
  - "camera motion"
  - "HPS estimation"
  - "world coordinates"
  - "BEDLAM"
date: 2026-05-08
content_hash: fe2c0dc16f324e0c
---

# BEDLAM2.0: Synthetic Humans and Cameras in Motion

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2511.14394](https://arxiv.org/abs/2511.14394)  
**Code/Data**: [bedlam2.is.tuebingen.mpg.de](https://bedlam2.is.tuebingen.mpg.de/)
**Area**: Human Pose Estimation / Synthetic Data
**Keywords**: synthetic data, SMPL-X, camera motion, HPS estimation, world coordinates, BEDLAM

## TL;DR

BEDLAM2.0 is a comprehensive upgrade over BEDLAM, introducing diverse camera motions (synthetic translation/tracking/orbit + handheld/head-mounted capture), broader body shape coverage (BMI 18–41), strand-based hair, shoes, size-graded clothing, and more 3D environments. The resulting dataset comprises 27K+ sequences and 8M+ frames; models trained exclusively on this synthetic data surpass the state of the art in world-coordinate human motion estimation.

## Background & Motivation

**Background**: BEDLAM was the first synthetic dataset capable of training competitive 3D human pose regressors without any real images, and has since become a standard training resource for HPS (Human Pose and Shape) methods. However, world-coordinate human motion estimation—which must account for camera motion and zoom—has emerged as an active research direction, and BEDLAM's diversity in camera motion and focal length remains severely limited.

**Limitations of Prior Work**: (1) The majority of BEDLAM sequences use a static camera, with only a negligible number of moving-camera clips, resulting in insufficient camera motion diversity. (2) Focal length coverage is narrow (HFOV concentrated at 52° or 65°), inconsistent with the wide focal length distribution observed in real-world videos. (3) Body shape diversity is inadequate, with a notable absence of high-BMI bodies. (4) All characters are barefoot, hairstyles are unrealistic, and clothing is available only in a single size unsuitable for varied body shapes.

**Key Challenge**: World-coordinate human motion estimation requires large quantities of training data with ground-truth camera motion and 3D body parameters. Such data are difficult to obtain from real captures, making synthetic data a critical path—yet BEDLAM's synthetic diversity falls short of what is needed.

**Goal**: To construct a synthetic dataset that substantially surpasses BEDLAM across camera motion, body shape, clothing, hair, and scene dimensions, with explicit support for end-to-end training of world-coordinate human pose estimation methods.

**Key Insight**: A dataset-engineering perspective that systematically improves every dimension of BEDLAM—camera (focal length + motion), body (shape + motion + hands), appearance (hair + shoes + clothing), and scene/rendering.

**Core Idea**: Through systematic improvements including synthetic-and-captured diverse camera motions, size-graded clothing, strand-based hair, and SMPL-X shoes, synthetic data alone can match or exceed the state of the art previously achievable only with real data.

## Method

### Overall Architecture

Sample from the AMASS motion library (4,643 motions) → retarget motions to diverse body shapes (BMI 18–41) → dress characters with size-graded clothing, strand-based hair, and shoes → place in 15 3D environments → apply synthetic/captured camera motions with diverse focal lengths → render in Unreal Engine 5.3 (1280×720 @ 30 fps) → output images, depth maps, SMPL-X ground truth, and camera parameters.

### Key Designs

1. **Diverse Camera System**
    - Focal length range: 14 mm–400 mm (16:9 DSLR sensor); 9% of videos include zoom during capture.
    - Synthetic camera motions: static, translation, tracking, dolly, orbit, zoom, and combinations, with differentiable Perlin noise superimposed to simulate hand shake.
    - Captured camera motions: real camera trajectories recorded using smartphones/tablets and an Apple Vision Pro headset within virtual scenes (static, orbit, approach/retreat), comprising 86.4% synthetic and 13.6% captured.
    - **Design Motivation**: Real-world camera motion is highly diverse; BEDLAM's static cameras cause end-to-end training of world-coordinate methods to underperform.

2. **Body Shape, Clothing, and Appearance Diversity**
    - Body shapes: 1,615 SMPL-X shapes spanning BMI 18–41, with oversampling of high-BMI bodies.
    - Clothing: 187 3D garments (76 more than BEDLAM), of which 50 are graded XS–6XL and matched to BMI.
    - Hair: 40 strand-based 3D hair grooms (50K–100K strands per groom), fitted to individual head shapes, with 9 color presets.
    - Shoes: 182 shoe models (Google Scanned Objects) mapped onto the SMPL-X "sock-foot" mesh via displacement maps, with height adjusted according to sole thickness.
    - **Design Motivation**: To close the domain gap between synthetic data and real images caused by bare feet, absent hair, and single-size clothing.

3. **Scene, Rendering, and Occlusion**
    - 15 high-quality 3D environments (vs. 5 in BEDLAM), including 9 indoor scenes (vs. 1 in BEDLAM).
    - Time-of-day and weather randomization (daylight, sunset, overcast, night).
    - A custom UE5 C++ plugin ensures deterministic consistency of camera shake between image and depth renders.
    - 12.7% of frames exhibit >20% occlusion; the top 10% most-occluded bodies have an average occlusion rate of 61.1%.

### Dataset Scale

27,480 video sequences; 8,048,411 PNG frames; 12.5M training bounding boxes; 862K test bounding boxes; 4,643 motions; 1,615 body shapes; 187 garments; 40 hairstyles; 182 shoe models; 15 3D environments.

## Key Experimental Results

### Single-Frame Method (CameraHMR)

| Training Data | 3DPW PA-MPJPE↓ | 3DPW MPJPE↓ | 3DPW PVE↓ | EMDB PA-MPJPE↓ | EMDB MPJPE↓ | RICH PA-MPJPE↓ | RICH MPJPE↓ |
|---|---|---|---|---|---|---|---|
| B1 | 43.2 | 68.0 | 80.7 | 50.0 | 88.7 | 42.1 | 75.2 |
| B2 | 41.1 | 64.8 | 76.3 | 46.5 | 74.6 | 36.8 | 70.8 |
| B1+B2 | 41.0 | 65.2 | 77.7 | 46.4 | 75.5 | 36.4 | 68.0 |

### Video Method (World-Coordinate Evaluation)

| Method | Training Data | RICH WA-MPJPE↓ | RICH W-MPJPE↓ | EMDB WA-MPJPE↓ | EMDB W-MPJPE↓ | RICH Jitter↓ | RICH Foot-Sliding↓ |
|---|---|---|---|---|---|---|---|
| GVHMR | B1 | 87.3 | 140.0 | 112.4 | 284.6 | 13.5 | 2.9 |
| GVHMR | B2 | 75.5 | 120.6 | 113.7 | 284.4 | 12.3 | 2.7 |
| GVHMR | B1+B2 | 75.8 | 121.3 | 109.7 | 273.1 | 11.3 | 2.6 |
| PromptHMR | B1 | 85.7 | 139.4 | 77.6 | 211.1 | 12.7 | 4.0 |
| PromptHMR | B2 | 75.3 | 122.4 | 71.9 | 197.7 | 11.7 | 2.8 |
| PromptHMR | B1+B2 | **72.5** | **116.6** | **70.5** | **193.7** | **10.2** | **2.6** |

### Key Findings

- Single-frame methods: training on B2 alone significantly outperforms B1 on all benchmarks, with a ~20% improvement in shape accuracy.
- Video methods: the B1+B2 combination achieves the best results, surpassing the original SOTA—which relied on real data—using synthetic data exclusively.
- B1 and B2 are complementary: B1 retains motions such as sitting and stair climbing that were removed from B2, while B2 offers stronger camera motion and appearance diversity.
- Training PromptHMR on B1+B2 reduces RICH WA-MPJPE from 85.7 to 72.5 (a 15.4% reduction).

## Highlights & Insights

- Synthetic data *alone* surpassing SOTA trained on real data marks an important milestone, indicating that sufficiently high-quality synthetic data can substitute for costly real-data annotation.
- The inclusion of shoes, though seemingly minor, has far-reaching consequences: it closes the domain gap between SMPL-X's bare feet and real-world footwear, affecting height estimation and ground-contact prediction.
- Capturing first-person camera motion via Apple Vision Pro represents an innovative data acquisition strategy.
- Size-graded clothing (XS–6XL) paired with diverse body shapes is a practically impactful yet easily overlooked engineering contribution.
- A custom UE5 C++ plugin that fixes a camera pose recording bug in the Movie Render Pipeline demonstrates how low-level engineering details can be critical to dataset quality.

## Limitations & Future Work

- Only human–ground interaction is supported; human–object interaction (e.g., sitting on a chair) and human–human interaction (e.g., handshakes) are not modeled.
- Motions lack semantic consistency with the scene (e.g., dancing in a kitchen), limiting applicability to semantic tasks.
- Children, amputees, and individuals with body shapes significantly deviating from the population mean are not represented.
- The absence of facial expressions and audio precludes reasoning about interpersonal communication.
- A visual domain gap between synthetic and real video remains.
- Only flat-soled shoes are considered; high heels would require changes to foot topology and pose.

## Related Work & Insights

- **vs. BEDLAM (B1)**: B2 is a comprehensive upgrade—camera (static → diverse motion), body shape (limited → BMI 18–41), hair (card-based → strand-based), shoes (none → 182 models), clothing (111 single-size garments → 187 size-graded garments), and scenes (5 → 15).
- **vs. PDHuman / BEDLAM-CC**: These works address focal length diversity but do not tackle camera motion.
- **vs. HumanVid / WHAC-A-Mole**: Camera motion is included, but synthetic data lacks realism or dataset scale is limited.
- **vs. EgoGen**: BEDLAM assets are reused for an egocentric viewpoint, whereas B2 provides a richer variety of camera motion types.
- **Implications for future work**: Every engineering detail in synthetic data construction—sole thickness, hair fitting to individual head shapes, motion retargeting—can influence final model performance.

## Rating

- Novelty: ⭐⭐⭐ — Systematic engineering improvements over BEDLAM; no methodological novelty, but substantial engineering depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comparisons against multiple SOTA methods on several standard benchmarks, with thorough B1 vs. B2 vs. B1+B2 ablations.
- Writing Quality: ⭐⭐⭐⭐ — Each improvement dimension is described with clear motivation; dataset-documentation style but executed at high quality.
- Value: ⭐⭐⭐⭐ — As an upgrade to a community-standard training dataset, it has direct and broad impact on the HPS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ICCV 2025\] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](../../ICCV2025/human_understanding/udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](../../CVPR2026/human_understanding/efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[CVPR 2026\] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](../../CVPR2026/human_understanding/idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)

</div>

<!-- RELATED:END -->
