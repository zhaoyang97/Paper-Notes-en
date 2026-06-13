---
title: >-
  [Paper Note] DexVLG: Dexterous Vision-Language-Grasp Model at Scale
description: >-
  [ICCV 2025][Robotics][Dexterous grasping] This paper presents DexVLG — the first large-scale vision-language-dexterous-grasp model. It introduces DexGraspNet 3.0…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Dexterous grasping"
  - "vision-language model"
  - "Flow Matching"
  - "semantic part grasping"
  - "large-scale dataset"
date: 2026-05-08
content_hash: 88637b9d65838028
---

# DexVLG: Dexterous Vision-Language-Grasp Model at Scale

**Conference**: ICCV 2025
**arXiv**: [2507.02747](https://arxiv.org/abs/2507.02747)  
**Code**: None  
**Area**: Robotics / Dexterous Grasping
**Keywords**: Dexterous grasping, vision-language model, Flow Matching, semantic part grasping, large-scale dataset

## TL;DR
This paper presents DexVLG — the first large-scale vision-language-dexterous-grasp model. It introduces DexGraspNet 3.0, a dataset comprising 174K objects and 170M grasp poses with part-level semantic annotations. By combining a VLM encoder with a Flow Matching pose prediction head, DexVLG achieves over 76% zero-shot execution success in simulation and demonstrates semantically aligned dexterous grasping in the real world.

## Background & Motivation
- **Background**: Vision-Language-Action (VLA) models are advancing rapidly in robotics, but progress has been largely confined to simple parallel-jaw grippers due to the difficulty of data collection.
- **Limitations of Prior Work**: Functional grasping with human-like dexterous hands — i.e., grasping specific object parts according to semantic instructions — remains severely understudied, lacking both large-scale training data and effective model architectures.
- **Key Challenge**: Dexterous hands have high degrees of freedom (>20 DoF), resulting in an enormous grasp pose space that makes it difficult for conventional methods to cover semantically aligned grasps across diverse objects and parts.
- **Key Insight**: A data-driven approach: first generate large-scale, high-quality, semantically aligned dexterous grasp data, then train a large model to learn language-guided grasp pose prediction.
- **Core Idea**: Construct an ultra-large-scale part-level dexterous grasp dataset (DexGraspNet 3.0), leverage a VLM to interpret natural language instructions and RGBD inputs, and predict dexterous hand grasp poses via Flow Matching.

## Method

### Overall Architecture
DexVLG consists of two major components: (1) the DexGraspNet 3.0 data construction pipeline — filtering objects from Objaverse, semantic segmentation via SAMesh, part name and object size annotation via GPT-4o, and energy-optimization-based grasp pose synthesis; and (2) the DexVLG model — a VLM encoder that processes RGBD images and language instructions, followed by a Flow Matching pose prediction head that generates dexterous hand grasp pose parameters.

### Key Designs
1. **DexGraspNet 3.0 Dataset Construction**:

    - **Function**: Filters and annotates 174K objects from Objaverse's 800K+ collection, synthesizing grasp poses for every semantic part of each object, totaling 170M grasp poses.
    - **Core Pipeline**:
        - GPT-4o six-view querying to filter low-quality or unsuitable objects
        - Trimesh mesh extraction → ManifoldPlus watertightening → CoACD convex decomposition
        - SAMesh semantic segmentation → GPT-4o Set-of-Marks annotation of part names
        - GPT-4o estimation of plausible object sizes with rescaling (diagonal 20–50 cm)
        - Part-geometry-based initial hand pose alignment → gradient optimization for grasp synthesis
    - **Design Motivation**: Data scale is fundamental to generalization. Part-level annotations enable the model to understand semantic instructions such as "grasp the handle" or "grasp the cap."

2. **LP-based Differentiable Force Closure (LP-DFC)**:

    - **Function**: Improves upon the original DFC energy optimization objective to synthesize more natural grasp poses.
    - **Mechanism**: At each timestep, the hand pose is fixed and linear programming is used to solve for optimal contact force magnitudes:
      $\min_{\mathbf{f}} \|G(\mathbf{f} \odot c)\|_2$, s.t. $\max_i(\mathbf{f})_i = 1$, $(\mathbf{f})_i \geq 0$
      The DFC energy is then rescaled based on the net torque $P$ and contact forces $\mathbf{f}$.
    - **Design Motivation**: The original DFC assumes equal contact forces, leading to artifacts such as finger tilting during thumb opposition. LP-DFC models variable contact force magnitudes, producing more natural poses that conform better to object geometry.

3. **Part-Aligned Hand Pose Initialization**:

    - **Function**: Semantically aligns the initial dexterous hand pose according to the geometric properties of the target object part.
    - **Mechanism**: Object parts are classified into four categories — lid-like, disk-like, L-shaped, and shaft-like — each with a dedicated palm position and orientation alignment strategy. Two grasp modes are defined: Wrap grasp (7 contact points: 5 fingertips + palm) and Pinch grasp (4 contact points: thumb + index + middle finger + palm).
    - **Design Motivation**: Gradient-based optimization is highly sensitive to initialization (as noted in the DexGraspNet paper). Part-aligned initialization injects strong geometric priors, yielding more natural and semantically distinguishable optimized poses.

4. **VLM + Flow Matching Pose Prediction**:

    - **Function**: Accepts RGBD images and language instructions to predict dexterous hand grasp pose parameters.
    - **Mechanism**: A VLM encodes visual and language inputs; Flow Matching serves as the denoising module to generate grasp poses, replacing DDPM/DDIM diffusion.
    - **Design Motivation**: The Flow Matching paradigm is more amenable to learning pose generation than conventional diffusion models, as validated by experiments showing significant improvements over DDPM and DDIM.

### Loss & Training
- Regularization energy for grasp synthesis: $E_{reg} = \omega_{limit}E_{limit} + \omega_{pen}E_{pen} + \omega_{spen}E_{spen} + \omega_{dir}E_{dir}$
    - Components include joint limit energy, hand-object penetration energy, self-penetration energy, and contact direction alignment energy (cosine similarity).
- 2×5,000 initial poses are generated per object part (5,000 for Wrap and 5,000 for Pinch modes).
- Simulated penetration checks further filter low-quality poses.
- Tabletop scene camera setup: 80 cm from the table center, uniformly sampled at a 45° downward angle.

## Key Experimental Results

### Main Results (Simulation Benchmark)

| Data | LVIS-Seen | | Unseen | | SamPart3D | |
|------|-----------|---|--------|---|-----------|---|
| | Suc↑ | PGA↑ | Suc↑ | PGA↑ | Suc↑ | PGA↑ |
| Wrap grasp | 87.7 | 62.1 | 79.1 | 36.6 | 76.3 | 52.0 |
| Pinch grasp | 71.8 | 20.2 | 54.8 | 15.2 | 50.6 | 21.3 |

DexVLG maintains a 79.1% Wrap grasp success rate on unseen objects and 76.3% zero-shot generalization on SamPart3D.

### Ablation Study

**Denoising Paradigm Ablation**:

| Method | LVIS-Seen | | Unseen | | SamPart3D | |
|------|-----------|---|--------|---|-----------|---|
| | Suc↑ | PGA↑ | Suc↑ | PGA↑ | Suc↑ | PGA↑ |
| DDPM | 51.9 | 7.8 | 34.1 | 10.9 | 40.7 | 5.5 |
| DDIM | 57.7 | 12.5 | 39.6 | 10.4 | 35.2 | 8.5 |
| **Flow Matching** | **75.3** | **39.1** | **54.0** | **18.3** | **53.4** | **27.0** |

**Dataset Quality Evaluation**:

| Method | Scale↑ | Penetration↓ (mm) | Self-Penetration↓ (mm) | Q1↑ |
|------|-------|----------|------------|-----|
| DexGraspNet | 1.32M | 13.5 | 0.93 | 0.114 |
| Multi-GraspLLM | 120k | 7.1 | - | 0.091 |
| **Ours-Wrap** | **103M** | **1.75** | **0.19** | 0.085 |
| **Ours-Pinch** | **67M** | **1.42** | **0.22** | 0.067 |

### Key Findings
- Flow Matching substantially outperforms DDPM/DDIM: success rate improvements of 23.4%/17.6% on LVIS-Seen, and PGA improvements of 31.3%/26.6%.
- Wrap grasping consistently outperforms Pinch grasping across all datasets (15.9–24.3% higher success rate), indicating that whole-hand wrapping yields greater stability.
- DexGraspNet 3.0 is two orders of magnitude larger than its predecessor (170M vs. 1.32M poses) while achieving substantially lower penetration depth (1.75 mm vs. 13.5 mm).
- Part-aligned initialization produces more natural and semantically distinguishable grasp poses compared to random initialization.
- The Q1 stability metric is slightly lower than that of pure power grasp datasets, as part-aligned grasping does not optimize for stability as its sole objective.

## Highlights & Insights
- A quintessential demonstration of the data-driven approach: 174K objects, 170M poses, and part-level annotations — the sheer scale of the dataset constitutes a core contribution in its own right.
- The multi-faceted application of GPT-4o throughout the data pipeline (object filtering, part annotation, size estimation) illustrates the potential of LLMs in robotic data construction.
- The LP-DFC improvement over DFC is modest but critical — transitioning from equal-force to variable-force modeling markedly improves pose quality in thumb-opposition scenarios.
- The four-category part geometry classification (lid/disk/L-shaped/shaft), while not exhaustive, covers the majority of functionally relevant part shapes.
- The complete pipeline — from dataset construction to model training to simulation evaluation — carries substantial engineering value.

## Limitations & Future Work
- Only the Appendix was available in the cached version; details of the main method and experiments are missing — the VLM encoder architecture and training strategy remain unknown.
- Real-world experimental descriptions are insufficient, with only qualitative references to "successful part-aligned grasps" and no quantitative results.
- Pinch grasp success rates are relatively low (50–72%), limiting applicability to fine manipulation scenarios.
- Data synthesis is conducted entirely in simulation; the sim-to-real gap may hinder real-world deployment.
- Evaluation is limited to the LEAP Hand; generalization to other dexterous hand designs (e.g., Shadow Hand, Allegro Hand) has not been verified.
- Object size clipping to 20–50 cm restricts applicability to very small or very large objects.

## Related Work & Insights
- The DexGraspNet series (1.0→2.0→3.0) demonstrates an iterative development trajectory for dexterous grasp datasets.
- Part-geometry relational priors from SoFar and OmniSpatial are effectively leveraged.
- The superiority of Flow Matching as an action generation paradigm is validated in dexterous manipulation, potentially establishing it as the standard choice for VLA systems.
- The part-aligned initialization strategy can be generalized to other scenarios requiring semantically guided pose synthesis.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First large-scale language-guided dexterous grasping model and dataset; however, the methodology primarily combines existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ Only appendix ablations are accessible in the cached version; the main experiments cannot be fully assessed. The simulation benchmark is well-designed, but real-world validation is insufficient.
- **Writing Quality**: ⭐⭐⭐ The appendix is detailed and thorough, but the absence of the main body limits a complete evaluation.
- **Value**: ⭐⭐⭐⭐⭐ The scale and quality of the dataset alone represent an extremely high contribution to the community and will drive future research in dexterous grasping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment](evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)
- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[ICCV 2025\] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](../../ICML2026/robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](../../NeurIPS2025/robotics/safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)

</div>

<!-- RELATED:END -->
