---
title: >-
  [Paper Note] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes EmbodiedSplat, a complete pipeline that captures real environments via iPhone video → reconstructs 3D Gaussian Splat meshes → fine-tunes navigation polici…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Embodied Navigation"
  - "Sim-to-Real Transfer"
  - "Scene Reconstruction"
  - "Personalized Policy Training"
  - "ImageNav"
date: 2026-05-08
content_hash: 1331d0e4459d8321
---

# EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device

**Conference**: ICCV 2025
**arXiv**: 2509.17430
**Code**: [https://gchhablani.github.io/embodied-splat](https://gchhablani.github.io/embodied-splat) (Project Page)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Embodied Navigation, Sim-to-Real Transfer, Scene Reconstruction, Personalized Policy Training, ImageNav

## TL;DR

This paper proposes EmbodiedSplat, a complete pipeline that captures real environments via iPhone video → reconstructs 3D Gaussian Splat meshes → fine-tunes navigation policies in Habitat-Sim → deploys to the real world. The approach achieves 20%–40% absolute success rate improvement over zero-shot baselines on real-scene ImageNav tasks, with a sim-vs-real Spearman rank correlation coefficient of 0.87–0.97.

## Background & Motivation

Training and evaluation of embodied AI primarily rely on simulated environments, yet three major challenges persist:

**Synthetic environments lack realism**: Synthetic datasets such as HSSD differ substantially from the real world in both style and scene complexity, leading to poor sim-to-real transfer.

**High cost of real-scene capture**: High-fidelity scans from datasets like HM3D and Matterport3D require expensive specialized equipment and labor-intensive collection processes.

**Insufficient coverage of deployment diversity**: Pre-training datasets cannot anticipate all possible deployment scenarios; policy performance degrades significantly when robots are deployed in novel environments such as university buildings or shopping malls.

**Core Problem**: Can low-cost mobile video capture be used to reconstruct deployment environments, generate sufficiently high-quality 3D meshes for policy fine-tuning, and achieve effective sim-to-real transfer?

**The key insight of this work** is not to pursue the highest reconstruction quality, but rather to study the relationship between reconstruction quality and navigation performance—i.e., *"how good does the mesh need to be?"*

## Method

### Overall Architecture (Four-Stage Pipeline)

1. **Scene Capture**: An iPhone 13 Pro Max with the Polycam app records RGB-D video (20–30 minutes per scene); Nerfstudio processes the footage and samples 1,000 aligned RGB-depth frames with poses.
2. **Mesh Reconstruction**: DN-Splatter is trained for 30,000 iterations to produce 3D Gaussian Splats, with meshes generated via Poisson reconstruction; Polycam-exported meshes are also evaluated for comparison.
3. **Sim Training**: Meshes are converted to `.glb` format, loaded into Habitat-Sim, and ImageNav episodes are generated for policy training and fine-tuning.
4. **Real Deployment**: The trained policy is deployed on a Stretch robot for real-world navigation.

### Key Design Choices

- **DN-Splatter**: Employs depth-normal regularization to improve mesh quality. Sensor depth weight $\lambda_d = 0.2$, with depth smoothness and normal loss enabled.
- **Normal Encoder**: Metric3D-V2 is empirically selected over Omnidata due to its higher mesh quality output.
- **Episode Generation**: HM3D/HSSD datasets generate 10,000 episodes per training scene; self-captured scenes generate only 1,000 training and 100 evaluation episodes.
- **Evaluation Metric**: Success Rate (SR)—the agent stops within 1 m of the goal within the maximum number of steps.

### Training Strategy

- **Zero-Shot**: Pre-trained directly on HM3D (800 training scenes) or HSSD (134 training scenes) for 600M–1200M steps.
- **Fine-Tuning**: Starting from a pre-trained policy, fine-tuning is performed on a single reconstructed scene for only 20M steps (learning rate 2.5e-6 for the LSTM policy, 6e-7 for the visual encoder).
- **Overfitting**: Training from scratch on a single scene for ~100M steps, used to assess whether large-scale pre-training is necessary.

## Key Experimental Results

### Main Results: Real-World Navigation Success Rate in the Lounge Scene

| Policy | Pre-training Data | Mesh Type | Success Rate (10 episodes) |
|---|---|---|---|
| Zero-Shot | HM3D (real) | — | 50% |
| Zero-Shot | HSSD (synthetic) | — | 10% |
| Fine-Tuned | HM3D → DN mesh | DN-Splatter | **70%** |
| Fine-Tuned | HM3D → Polycam | Polycam | **70%** |
| Fine-Tuned | HSSD → DN mesh | DN-Splatter | 40% |
| Fine-Tuned | HSSD → Polycam | Polycam | 50% |
| Overfit (no pre-training) | — → Polycam | Polycam | 50% |
| Overfit (no pre-training) | — → DN mesh | DN-Splatter | 10% |

**Key Finding**: HM3D pre-training + fine-tuning improves success rate from 50% → 70% (**+20%**); HSSD pre-training + fine-tuning improves from 10% → 50% (**+40%**).

### Ablation Study: Fine-Tuning Effectiveness in Simulation

| Scene | HM3D Zero-Shot SR | HM3D Fine-Tuned SR |
|---|---|---|
| conf_a (DN mesh) | 85% | 95%+ |
| conf_b (DN mesh) | 88% | 95%+ |
| classroom (DN mesh) | 53% | 90%+ |
| lounge (DN mesh) | 50% | 90%+ |
| classroom (Polycam) | 42% | 90%+ |
| lounge (Polycam) | 76% | 90%+ |

After fine-tuning, simulation success rates reach 90%+ across all scenes, **requiring only an additional 20M steps compared to 600M steps of pre-training**.

### Analysis: Sim-to-Real Correlation

- DN mesh Spearman Rank Correlation Coefficient (SRCC) between simulation and real-world performance: **0.87–0.97**
- This indicates that performance improvements in simulation reliably predict real-world improvements.
- Scene scale (average shortest path distance) is negatively correlated with zero-shot success rate.
- PSNR is positively correlated with success rate.

### Key Findings

1. **Non-trivial real-world success is achievable without large-scale pre-training**: A policy overfitted solely on a Polycam mesh achieves 50% real-world success rate.
2. **Real-data pre-training substantially outperforms synthetic**: HM3D zero-shot achieves 50% vs. HSSD zero-shot at 10%.
3. **Continued training of HM3D beyond 400M steps leads to stagnation or degradation of zero-shot performance on self-captured scenes**.
4. **Polycam meshes offer higher visual fidelity** (using original captured images directly), while DN-Splatter meshes (open-source) remain competitive.

## Highlights & Insights

1. **Low-cost, scalable pipeline**: 20–30 minutes of iPhone capture + 1–2 hours of DN-Splatter training yields a usable simulation scene at a fraction of the cost of Matterport-level scanning.
2. **Paradigm for personalized deployment**: Rather than pursuing a general-purpose universal policy, the approach advocates rapidly capturing a specific deployment scene and fine-tuning within it—a strategy likely more practical than scaling pre-training data in real deployments.
3. **Systematic analysis**: Beyond demonstrating the method, the paper provides in-depth analysis across multiple dimensions, including reconstruction quality vs. navigation performance and pre-training data vs. transfer performance, offering valuable insights for future research.
4. **Handheld capture suffices (no gimbal required)**: In contrast to the MuSHRoom dataset which uses a gimbal, this work demonstrates that handheld mobile capture is sufficient.

## Limitations & Future Work

1. **Small-scale real-world evaluation**: Real-world testing is limited to 10 episodes in a single lounge scene, raising concerns about statistical reliability.
2. **Only ImageNav is evaluated**: The pipeline has not been extended to more complex tasks such as ObjectNav or mobile manipulation.
3. **Limited scene scale**: Self-captured scenes cover 1–3 rooms; reconstruction quality and navigation performance for building-scale environments remain unknown.
4. **Visual fidelity of DN meshes**: The learned Gaussian colors differ from real photographs; policies overfitted on DN meshes achieve only 10% real-world success rate (vs. 50% for Polycam).
5. **No direct comparison with related methods such as Phone2Proc**.

## Related Work & Insights

- **Phone2Proc**: Uses the iPhone RoomPlan API to generate layouts followed by procedural scene generation, but requires post-processing and multi-variant generation; this work directly reconstructs the entire scene.
- **GaussNav / SplatNav**: Employ Gaussian Splatting for navigation but are not end-to-end or have not been validated on real robots.
- **Implications for future embodied AI**: Low-cost 3D reconstruction + rapid fine-tuning may become a standard deployment pipeline for robots—scan first, then deploy.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — First systematic validation of the complete personalized navigation pipeline: GS → Habitat → Real.
- **Experimental Thoroughness**: ⭐⭐⭐ — Real-world evaluation is small in scale, but the analysis covers multiple informative dimensions.
- **Value**: ⭐⭐⭐⭐⭐ — Simple pipeline, low cost, immediately actionable.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with clearly articulated research questions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HouseTour: A Virtual Real Estate A(I)gent](housetour_a_virtual_real_estate_aigent.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](../../ICLR2026/3d_vision/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

</div>

<!-- RELATED:END -->
