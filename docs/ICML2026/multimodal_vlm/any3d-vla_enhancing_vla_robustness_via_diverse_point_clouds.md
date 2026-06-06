---
title: >-
  [Paper Note] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds
description: >-
  [ICML 2026][Multimodal VLM][Point cloud fusion] Through a pilot study, the authors discovered that "explicitly lifting vision to point clouds and then fusing them with 2D patches" is the most effective way to inject 3D i…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Point cloud fusion"
  - "sim-to-real"
  - "domain generalization"
  - "data augmentation"
  - "grasping"
date: 2026-05-08
content_hash: a69f891a64262425
---

# Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds

**Conference**: ICML 2026  
**arXiv**: [2602.00807](https://arxiv.org/abs/2602.00807)  
**Code**: https://xianzhefan.github.io/Any3D-VLA.github.io  
**Area**: Robotics / VLA / Multimodal 3D Representation  
**Keywords**: Point cloud fusion, sim-to-real, domain generalization, data augmentation, grasping

## TL;DR
Through a pilot study, the authors discovered that "explicitly lifting vision to point clouds and then fusing them with 2D patches" is the most effective way to inject 3D information into Vision-Language-Action (VLA) models. To address 3D data scarcity and domain gaps across various point cloud sources (simulation, sensors, or monocular estimation), Any3D-VLA is proposed. It employs hybrid point cloud training to learn source-agnostic geometric representations, achieving a 29.2% zero-shot improvement over the strongest baseline (62.5% vs. 33.3%) in real-world grasping tasks.

## Background & Motivation
**Background**: Current mainstream VLAs (e.g., π0.5, GraspVLA) use 2D images as visual inputs and leverage Vision-Language Model (VLM) backbones for unified language-vision-action modeling. The community has explored injecting 3D information through depth-pretrained encoders (DepthVLA), spatial foundation models (VGGT), depth-as-channel (3D-CAVLA), and point cloud branches (PointVLA / 3DS-VLA).

**Limitations of Prior Work**: (1) Pure 2D VLAs are fragile when dealing with small objects, viewpoint variations, and occluded scenes. (2) Existing 3D injection methods face several issues: implicit depth/3D approaches (like VGGT) rely on reconstruction loss for geometry learning, which lacks metric precision and is prone to "spatial hallucinations"; depth-as-channel methods treat depth as 2D images, destroying 3D topology; point cloud branches either use non-pretrained encoders or process point clouds independently without alignment to 2D features. (3) 3D data scarcity and domain gaps in noise, scale, and geometric bias across different environments (simulation vs. sensor vs. estimation) hinder effective sim-to-real transfer for 3D VLAs.

**Key Challenge**: To obtain precise 3D geometric signals, models must rely either on expensive metric depth hardware (high dependency, large cross-environment variance) or model-estimated depth (noisy with scale drift). A truly "industrial-grade deployable VLA" must function effectively regardless of the depth source—this is a robustness issue rather than a simple precision problem.

**Goal**: (1) Select the optimal 3D injection paradigm through a pilot study; (2) Design a plug-in module to integrate 3D information into existing VLA backbones; (3) Explicitly model depth source heterogeneity via "hybrid point cloud training" to make the model source-agnostic during deployment.

**Key Insight**: The authors first conducted a clean pilot experiment to fairly compare five paradigms: 2D-only, implicit-depth RGB, implicit-3D RGB, RGBD-image-plane, and point-cloud+2D-patch fusion (under the same simulation benchmark and ground-truth metric depth). They found that point-cloud+2D-patch significantly outperformed other configurations, forming the basis for Any3D-VLA.

**Core Idea**: RGB+depth is lifted into a point cloud, which is then processed via 3D grid compression and a pretrained point cloud encoder. Features are aligned with ViT patches using scatter-mean and fused back into 2D representations via gated residuals. During training, the model is exposed to a mix of simulator, sensor, and model-estimated point cloud sources to learn source-agnostic geometric features.

## Method
Any3D-VLA is a plug-in visual observation module that can be attached to any VLA backbone. The pipeline follows: RGB+optional depth → lift to point cloud → 3D compression → point cloud encoder → patch alignment → 2D-3D gated fusion → VLA backbone.

### Overall Architecture
- **Data Preparation**: An RGBD dataset was synthesized in Isaac Sim (Objaverse LVIS subset, 290 classes, 10,680 instances, single-view, with camera parameters matching a RealSense D435). For each timestep, both (1) ground-truth metric depth from the Isaac rendering pipeline and (2) metric depth estimated by monocular depth models were exported.
- **VLA Backbone**: InternLM2-1.8B serves as the VLM backbone, combined with a conditional flow-matching action expert, connected via Progressive Action Generation (PAG). The visual observation module is the core contribution.
- **Visual Module Steps**: (1) Point Cloud Construction: Each valid depth pixel is unprojected to the camera coordinate system using camera intrinsics; (2) 3D Compression: Point clouds are compressed from 30k-60k to 3k-8k points using Sonata-style grid sampling; (3) Vision Encoder: DINOv2+SigLIP for 2D and Concerto (a point cloud encoder pretrained on 2D+3D data) for 3D; (4) Patch-Wise Alignment + 2D-3D Fusion: 3D points are projected back to image patch grids, aggregated into patch-level 3D features via scatter-mean, and fused with 2D patch tokens using gated residuals.
- **Output**: The fused token sequence, along with language and proprioception tokens, is fed into the VLA backbone to autoregressively generate bbox and grasp pose tokens. Finally, the flow-matching expert generates continuous end-effector action chunks.

### Key Designs

1. **Point-cloud–2D patch fusion as the Optimal 3D Injection Paradigm**:
    - **Function**: Explicitly injects metric-accurate 3D geometric signals while preserving the pretrained knowledge of the 2D backbone.
    - **Mechanism**: Based on a pilot study (see Table 2) comparing five 3D injection methods, VLA performance improved stably (Single-Trial SR from 45.3 to 61.1) only when using point clouds with native 3D topology and explicit spatial alignment with 2D patches. Implicit methods like VGGT often suffer from spatial hallucinations in fine-grained manipulation, and depth-as-channel loses topology.
    - **Design Motivation**: Injecting 3D is not just about "providing depth"; the "representation" determines success. Point clouds preserve native 3D topology while patch alignment maintains the semantic priors of the 2D backbone.

2. **Patch-Wise Alignment + Gated Residual Fusion**:
    - **Function**: Aligns unordered point cloud features to the regular patch grid of the ViT and injects them as "minor corrections" to the 2D representation.
    - **Mechanism**: Each 3D point $\mathbf{x}_i$ is mapped back to the image plane via the projection function $(u_i, v_i) = \pi(\mathbf{x}_i)$ to find its patch index $a_i$. Points within the same patch are aggregated via scatter-mean to obtain $\mathbf{g}_j^\text{3D}$; if no points exist in a patch, a learnable empty token $\mathbf{e}^\text{3D}$ is used. After linear projection to token dimension $\mathbf{h}_j^\text{3D} = W_\text{3D}\mathbf{g}_j^\text{3D}$, it is concatenated with $\mathbf{h}_j^\text{2D}$ and passed through an MLP to obtain residual $\delta_j$. Fusion uses gated residuals: $\mathbf{h}_j^\text{fused} = \mathbf{h}_j^\text{2D} + \sigma(g) \cdot \text{LayerNorm}(\delta_j)$, where the gating $g$ is initialized to -2.1972 so that $\sigma(g)$ is very small initially, preventing the destruction of pretrained 2D representations.
    - **Design Motivation**: Applying "minor corrections" rather than "replacing representations" preserves the strong semantic priors of DINOv2+SigLIP while allowing 3D signals to intervene when necessary.

3. **Hybrid Point Cloud Training (Key Sim-to-Real Lever)**:
    - **Function**: Lessons learned from diverse point cloud sources during training allow the 3D encoder to acquire source-agnostic geometric patterns, removing dependency on specific depth hardware.
    - **Mechanism**: Three training settings were defined—Setting 1 (Simulator GT only), Setting 2 (Hybrid: simulator, sensor, or monocular-estimated metric PC selected with fixed probabilities), and Setting 3 (Sensor only). Setting 2 is critical: the model encounters various point cloud sources (with noise, scale bias, and geometric imperfections) throughout training, forcing the 3D encoder and fusion layers to learn source-agnostic features. The mixture ratio includes 30% RealSense and 20% each for various monocular estimation models (UniDepthV2, DA3, MapAnything).
    - **Design Motivation**: This is the paper's most vital engineering insight—the primary obstacle for 3D VLA deployment is the massive discrepancy between depth sources across environments. Treating this heterogeneity as part of the optimization goal directly addresses robustness.

### Loss & Training
The VLM head and flow-matching action expert are trained jointly. Grounding data from GRIT is used to supervise the VLM's autoregressive prediction of bbox tokens, while synthetic RGBD data supervises grasp pose tokens and end-effector actions (flow matching loss). **No depth or point cloud reconstruction loss is added**, intentionally verifying that performance gains stem from representation design rather than auxiliary supervision.

## Key Experimental Results

### Main Results (Real-world zero-shot)
Comparison against π0.5, GraspVLA (2D baseline), and SpatialVLA (3D baseline) across 4 challenge categories (Standard, Scale&Shape, Viewpoint, Appearance-Deprived), involving 47 real objects and 120 trials.

| Method | Training Setting | Inference PC | Overall SR (%) |
| :--- | :--- | :--- | :--- |
| π0.5 (2D) | – | – | ≈ 26 |
| GraspVLA (2D) | – | – | ≈ 30 |
| SpatialVLA (3D) | – | – | 33.3 (Prev. SOTA) |
| Any3D-VLA | Setting 1 (sim only) | RealSense | Gain |
| Any3D-VLA | Setting 2 (hybrid) | RealSense | Further Gain |
| **Any3D-VLA** | **Setting 2 (hybrid)** | **DA3 estimated** | **62.5 (+29.2)** |

### Post-training (Fine-tuning with limited real demonstrations)
Evaluated on two challenging tasks (Task 1: Placing a pink tulip in a vase; Task 2: Placing a transparent sauce cup into a slot) with 100 real demonstrations each.

| Model | Training Setting | Inference PC | Task 1 SR (%) | Task 2 SR (%) |
| :--- | :--- | :--- | :--- | :--- |
| π0.5 | – | – | 33.3 | 26.7 |
| GraspVLA | – | – | 33.3 | 53.3 |
| SpatialVLA | – | – | 13.3 | 6.7 |
| Any3D-VLA | RealSense only | RealSense | 73.3 | 60.0 |
| Any3D-VLA | RealSense only | DA3 | 80.0 | 60.0 |
| Any3D-VLA | Hybrid | RealSense | 80.0 | 66.7 |
| **Any3D-VLA** | **Hybrid** | **DA3** | **93.3** | **86.7** |

### Key Findings
- Hybrid training performance $\ge$ single-source training across any inference point cloud source, proving it learns source-agnostic geometry rather than simple multi-task overfitting.
- Point clouds estimated by DA3 outperformed RealSense sensor data in most cases, indicating that modern monocular depth models can produce more accurate geometry than consumer-grade depth cameras, suggesting future 3D VLAs could bypass depth hardware entirely.
- Pilot study results were counter-intuitive: under perfect simulation depth, depth-as-channel provided only an 11-point gain (45.3 to 56.8), whereas point-cloud fusion provided a 16-point gain (45.3 to 61.1), proving "how geometry is represented" is more important than "if geometry is present."
- Inference latency is 1.7~2.0 FPS (DA3 route), which is practical for tabletop manipulation when amortized via action chunking (chunk size=4).

## Highlights & Insights
- **Clean Pilot Study Design**: By strictly controlling variables (same backbone, training strategy, and simulation ground-truth depth), the authors used SR data to prove point-cloud+2D-patch as the optimal paradigm—a "pilot before commit" methodology that is highly commendable.
- **Gated Residual Fusion Initialization**: Initializing gating to a very small value ensures the new modality begins with "zero influence," avoiding catastrophic forgetting of pretrained priors.
- **Hybrid Training as a Sim-to-Real Panacea**: Rather than struggling to tune a single depth source's precision, exposing the model to all depth sources allows it to treat robustness as part of the objective. This philosophy (Diversity > Single-source Precision) mirrors successes in LLM mixing and autonomous driving sensor fusion.

## Limitations & Future Work
- Object categories are capped at 290 (Objaverse LVIS subset), leaving a gap toward open-vocabulary capabilities.
- Relies on single-view input; multi-view fusion might improve occlusion scenarios but would increase latency.
- Inference still depends on an estimated depth model (DA3), shifting the latency bottleneck from the 3D encoder to the depth model.
- Primarily validated on tabletop manipulation; not yet tested on mobile platforms or long-horizon loco-manipulation tasks.
- Transparent and reflective objects remain difficult.

## Related Work & Insights
- **vs. PointVLA (Li et al. 2025a)**: PointVLA injects point cloud features into the action expert, but 3D and 2D processing remain relatively independent; Any3D-VLA's patch-level alignment makes 3D signals and 2D tokens correspond more precisely.
- **vs. SpatialVLA**: SpatialVLA is the strongest 3D baseline but remains anchored in the image plane; Any3D-VLA nearly doubles its SR using native 3D topology and hybrid training.
- **vs. VGGT / Spatial Forcing**: While those use implicit 3D priors, this work demonstrates that explicit 3D geometry is more reliable for fine-grained manipulation.
- **vs. DepthVLA / 3D-CAVLA**: Instead of treating depth as a channel, Any3D-VLA lifts it to point cloud space and re-projects, gaining both geometric precision and topology.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Solid combination of pilot study, gated patch fusion, and hybrid training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Exemplary design including simulation, real-world, zero-shot, and post-training evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ High logical clarity; the chain of reasoning is very persuasive.
- **Value**: ⭐⭐⭐⭐⭐ High impact for real-world deployment where multi-source sensor heterogeneity is a standard challenge.

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
