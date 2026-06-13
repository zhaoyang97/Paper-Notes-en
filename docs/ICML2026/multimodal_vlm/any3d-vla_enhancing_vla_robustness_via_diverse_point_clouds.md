---
title: >-
  [Paper Note] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds
description: >-
  [ICML 2026][Multimodal VLM][Point Cloud Fusion] Through a pilot study, the authors find that "explicitly lifting vision to point clouds and then fusing with 2D patches" is the most effective way to inject 3D information…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Point Cloud Fusion"
  - "Sim-to-Real"
  - "Domain Generalization"
  - "Data Augmentation"
  - "Grasping"
date: 2026-05-08
content_hash: 50b93d43915d2b99
---

# Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds

**Conference**: ICML 2026  
**arXiv**: [2602.00807](https://arxiv.org/abs/2602.00807)  
**Code**: https://xianzhefan.github.io/Any3D-VLA.github.io  
**Area**: Robotics / VLA / Multimodal 3D Representation  
**Keywords**: Point Cloud Fusion, Sim-to-Real, Domain Generalization, Data Augmentation, Grasping

## TL;DR
Through a pilot study, the authors find that "explicitly lifting vision to point clouds and then fusing with 2D patches" is the most effective way to inject 3D information into VLA. To address the scarcity of 3D data and domain gaps among different point cloud sources (simulation/sensor/monocular estimation), they propose Any3D-VLA: using hybrid point cloud training to learn source-agnostic geometric representations, achieving a 29.2% improvement (62.5% vs 33.3%) over the strongest baseline in real-world zero-shot grasping tasks.

## Background & Motivation
**Background**: Current mainstream VLAs (e.g., π0.5, GraspVLA) use 2D images as visual input, leveraging VLM backbones for unified language-vision-action modeling. The community has explored 3D injection: depth-pretrained encoders (DepthVLA), spatial foundation models (VGGT), depth-as-channel (3D-CAVLA), and point cloud branches (PointVLA / 3DS-VLA).

**Limitations of Prior Work**: (1) Pure 2D VLAs are fragile with small objects, viewpoint changes, and occlusions. (2) Existing 3D injection methods have issues: implicit depth/3D (VGGT-like) rely on reconstruction loss for geometry, lacking metric precision and prone to "spatial hallucination"; depth-as-channel treats depth as a 2D image, destroying 3D topology; point cloud branches either use non-pretrained encoders or process point clouds independently without 2D alignment. (3) 3D data scarcity and cross-environment (simulation vs sensor vs estimation) noise/scale/geometric bias cause severe domain gaps, making 3D VLA sim-to-real difficult.

**Key Challenge**: To obtain precise 3D geometric signals, one must rely on expensive metric depth hardware (strong dependency, large cross-environment differences) or model-estimated depth (with scale drift noise). Truly "deployable industrial-grade VLA" must work under any depth source—this is a robustness issue, not just an accuracy issue.

**Goal**: (1) Use a pilot study to select the optimal 3D injection paradigm; (2) Design a plug-in module to integrate 3D information into existing VLA backbones; (3) Explicitly model depth source heterogeneity via "hybrid point cloud training," making the model source-agnostic at deployment.

**Key Insight**: The authors first conduct a clean pilot experiment, fairly comparing five paradigms: 2D-only, implicit-depth RGB, implicit-3D RGB, RGBD-image-plane, and point-cloud+2D-patch fusion (on the same simulation benchmark and ground-truth metric depth). They find point-cloud+2D-patch fusion significantly outperforms others—forming the basis for Any3D-VLA.

**Core Idea**: Lift RGB+depth to point clouds, encode with 3D grid compression and pretrained point cloud encoder, align with ViT patches via scatter-mean, then fuse back to 2D representation using gated residuals; during training, mix simulator/sensor/model-estimated point cloud sources so the 3D encoder learns source-agnostic geometric features.

## Method
Any3D-VLA is a plug-in visual observation module attachable to any VLA backbone. The pipeline: RGB+optional depth → lift to point cloud → 3D compression → point cloud encoder → patch alignment → 2D-3D gated fusion → VLA backbone.

### Overall Architecture
- **Data Preparation**: Synthesize RGBD datasets in Isaac Sim (Objaverse LVIS subset, 290 classes, 10,680 instances, single view, camera parameters matched to RealSense D435). For each timestep, export (1) ground-truth metric depth from Isaac rendering pipeline, (2) metric depth estimated by monocular depth models; both are used.
- **VLA Backbone**: InternLM2-1.8B as VLM backbone + conditional flow-matching action expert, connected via PAG (Progressive Action Generation). The visual observation module is the core of this work.
- **Visual Module Steps**: (1) Point Cloud Construction: unproject each valid depth pixel to camera coordinates using intrinsics; (2) 3D Compression: grid sampling (Sonata) compresses point cloud from 30k-60k to 3k-8k; (3) Vision Encoder: DINOv2+SigLIP for 2D, Concerto (pretrained on 2D+3D data) for 3D; (4) Patch-Wise Alignment + 2D-3D Fusion: project 3D points back to image patch grid, scatter-mean to aggregate patch-level 3D features, then gated residual fusion with 2D patch tokens.
- **Output**: Fused token sequence → fed with language and proprioceptive tokens to VLA backbone → autoregressive generation of bbox token + grasp pose token; finally, flow-matching expert generates continuous end-effector action chunks.

### Key Designs

1. **Point-cloud–2D patch fusion as the optimal 3D injection paradigm (based on pilot study)**:

    - **Function**: Explicitly injects metric-accurate 3D geometric signals while retaining 2D backbone pretraining.
    - **Mechanism**: The pilot study fairly compares five 3D injection methods (see Table 2), finding only point-cloud provides native 3D topology and explicit spatial alignment with 2D patches, enabling stable VLA improvement (Single-Trial SR from 45.3 → 61.1). VGGT-like implicit methods, though with reconstruction priors, often suffer spatial hallucination in fine-grained manipulation; depth-as-channel compresses 3D into 2D, losing topology. Any3D-VLA thus selects point cloud + 2D patch fusion as its foundation.
    - **Design Motivation**: Injecting 3D is not just about "providing depth," but "how to represent" is crucial. Point clouds preserve native 3D topology and, via patch alignment, retain 2D backbone semantic priors—achieving both.

2. **Patch-Wise Alignment + Gated Residual Fusion**:

    - **Function**: Align unordered point cloud features to ViT's regular patch grid, injecting as "minor corrections" to 2D representations.
    - **Mechanism**: Each 3D point $\mathbf{x}_i$ is projected to the image plane via camera projection $(u_i, v_i) = \pi(\mathbf{x}_i)$, locating patch index $a_i$. Points within the same patch are scatter-mean aggregated to $\mathbf{g}_j^\text{3D}$; if no points, a learnable empty token $\mathbf{e}^\text{3D}$ is used. Linear projection to token dim $\mathbf{h}_j^\text{3D} = W_\text{3D}\mathbf{g}_j^\text{3D}$, concatenated with $\mathbf{h}_j^\text{2D}$, passed through MLP to get residual $\delta_j$. Fusion uses gated residual: $\mathbf{h}_j^\text{fused} = \mathbf{h}_j^\text{2D} + \sigma(g) \cdot \text{LayerNorm}(\delta_j)$, with gating $g$ initialized at -2.1972 so $\sigma(g)$ is small at training start, preserving pretrained 2D representations and gradually opening up during training.
    - **Design Motivation**: "Making minor corrections to the 2D backbone" rather than "replacing 2D representations" preserves strong semantic priors from DINOv2+SigLIP, while allowing 3D signals to intervene as needed. Gated init addresses the common issue of "destroying original representations in early epochs" when injecting new modalities.

3. **Hybrid Point Cloud Training (key sim-to-real enabler)**:

    - **Function**: Mixes multiple point cloud sources during training, enabling the 3D encoder to learn source-agnostic geometric patterns, removing reliance on specific depth hardware at deployment.
    - **Mechanism**: Three training settings—Setting 1: simulator GT point cloud only; Setting 2: hybrid (each trajectory randomly selects simulator/sensor or single-frame RGB-estimated metric point cloud); Setting 3: sensor only. Setting 2 is key: the model sees various point cloud sources (with noise, scale bias, geometric imperfections) throughout training, forcing the 3D encoder and fusion layer to learn source-invariant features. The mix: 30% RealSense + various monocular estimation models (UniDepthV2 / DA3 / MapAnything) each 20%.
    - **Design Motivation**: This is the paper's most critical engineering insight—the main deployment barrier for 3D VLA is not accuracy per se, but the huge differences between depth sources in different environments. Directly injecting this heterogeneity into training data is equivalent to making "robustness" part of the optimization objective.

### Loss & Training
Jointly train VLM head + flow-matching action expert: use grounding data from GRIT to supervise VLM autoregressive bbox token prediction; use synthetic RGBD data to additionally supervise grasp pose token + end-effector action (flow matching loss). **No depth/point cloud reconstruction loss is used**—the authors deliberately verify that performance gains come from representation design, not auxiliary supervision.

## Key Experimental Results

### Main Results (Real-world Zero-shot)
On four challenge categories (Standard / Scale&Shape / Viewpoint / Appearance-Deprived), compared with π0.5, GraspVLA (2D baselines), and SpatialVLA (3D baseline). 47 real objects, 120 trials, up to 3 grasps per trial.

| Method | Training Setting | Inference Point Cloud | Overall SR (%) |
|--------|------------------|----------------------|----------------|
| π0.5 (2D) | – | – | ≈ 26 |
| GraspVLA (2D) | – | – | ≈ 30 |
| SpatialVLA (3D) | – | – | 33.3 (strongest baseline) |
| Any3D-VLA | Setting 1 (sim only) | RealSense | Improved |
| Any3D-VLA | Setting 2 (hybrid) | RealSense | Further improved |
| **Any3D-VLA** | **Setting 2 (hybrid)** | **DA3 estimated** | **62.5 (+29.2)** |

### Post-training (Few-shot Real Demonstration Fine-tune)
Two challenge tasks: Task1—place pink tulip in vase / Task2—place transparent seasoning cup in fixed slot. 100 real demonstrations each.

| Model | Training Setting | Inference Point Cloud | Task1 SR (%) | Task2 SR (%) |
|-------|------------------|----------------------|--------------|--------------|
| π0.5 | – | – | 33.3 | 26.7 |
| GraspVLA | – | – | 33.3 | 53.3 |
| SpatialVLA | – | – | 13.3 | 6.7 |
| Any3D-VLA | RealSense only | RealSense | 73.3 | 60.0 |
| Any3D-VLA | RealSense only | DA3 | 80.0 | 60.0 |
| Any3D-VLA | Hybrid | RealSense | 80.0 | 66.7 |
| **Any3D-VLA** | **Hybrid** | **DA3** | **93.3** | **86.7** |

### Key Findings
- Hybrid training outperforms single-source training under any inference point cloud source, proving it learns truly source-agnostic geometry rather than simple multi-task overfitting.
- DA3-estimated point clouds often perform as well as or better than RealSense sensor point clouds at inference, indicating modern monocular depth estimation models can produce more accurate point clouds than consumer-grade depth cameras—suggesting future 3D VLA deployment may eliminate depth hardware dependency.
- Pilot study counterintuitively shows: under perfect simulated depth, using depth as channel input gives only 11-point improvement (45.3 → 56.8), while point-cloud fusion gives 16-point improvement (45.3 → 61.1). This shows "how geometry is represented" matters more than "whether geometry is present."
- Inference latency is 1.7~2.0 FPS (DA3 route); with action chunking (chunk size=4), practical for desktop manipulation.

## Highlights & Insights
- **Clean pilot study setup**: Strict variable control across all methods (same backbone, training strategy, simulated ground-truth depth), using SR data to demonstrate point-cloud+2D-patch as the optimal paradigm—this "pilot before commit" experimental methodology is highly instructive.
- **Gated residual fusion initialization trick**: Initializing gating to $\sigma^{-1}(\text{very small})$ so the new modality starts training with "almost no effect," avoiding catastrophic forgetting; this "cold-to-hot" injection strategy is applicable to any new modality.
- **Hybrid training as a sim-to-real panacea**: Rather than tuning for one depth source's accuracy, expose the model to all depths—this "data diversity > single-source accuracy" philosophy has been validated in LLM data mixing and autonomous driving sensor fusion, and is precisely applied here for VLA 3D injection.

## Limitations & Future Work
- Object categories capped at 290 (Objaverse LVIS subset), still far from open vocabulary.
- Single-view input; multi-view fusion may further improve occlusion scenarios, but at the cost of increased latency.
- Inference still depends on an estimated depth model (DA3), shifting the latency bottleneck from the 3D encoder to the depth model.
- Main validation is on desktop manipulation; not yet tested on mobile platforms or long-horizon tasks (loco-manipulation).
- Transparent/reflective objects remain challenging (though the paper demonstrates a transparent seasoning cup, SR is not very high).

## Related Work & Insights
- **vs PointVLA (Li et al. 2025a)**: PointVLA injects point cloud features into the action expert, but point cloud and 2D are relatively independent; this work's patch-level alignment enables one-to-one correspondence between 3D signals and 2D tokens, with finer granularity.
- **vs SpatialVLA**: SpatialVLA is the strongest 3D baseline but still image-plane-centric; this work uses native 3D topology + hybrid training to nearly double SR.
- **vs VGGT / Spatial Forcing**: Those use implicit 3D priors; this work empirically shows explicit 3D geometry is more reliable for fine-grained manipulation.
- **vs DepthVLA / 3D-CAVLA**: Those use depth as an extra channel or depth expert; this work lifts depth to point cloud space and reprojects, achieving both geometric precision and topology.

## Rating
- Novelty: ⭐⭐⭐⭐ Pilot study + gated patch fusion + hybrid training is solid; individual techniques have prior work but the integration is new
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Simulation + real + zero-shot + post-training + multiple depth sources + multiple baselines, textbook-level experimental design
- Writing Quality: ⭐⭐⭐⭐ Pilot study section is very clear, logical chain "why choose point cloud → how to fuse → how to sim2real" is coherent
- Value: ⭐⭐⭐⭐⭐ Directly addresses practical needs, and the hybrid training paradigm is transferable to any "multi-source sensor heterogeneity" scenario

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks](../../AAAI2026/multimodal_vlm/phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)
- [\[ICML 2026\] VLANeXt: A Recipe for Building Robust VLA Models](vlanext_recipes_for_building_strong_vla_models.md)
- [\[ICML 2026\] TRAP: Hijacking CoT Reasoning of VLA with Adversarial Patches for Targeted Behavior Attacks](trap_hijacking_vla_cot-reasoning_via_adversarial_patches.md)
- [\[ICML 2026\] VLA-Arena: An Open-Source Framework for Evaluating Vision-Language-Action Models](vla-arena_an_open-source_framework_for_benchmarking_vision-language-action_model.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)

</div>

<!-- RELATED:END -->
