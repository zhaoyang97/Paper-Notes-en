---
title: >-
  [Paper Note] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation
description: >-
  [CVPR 2025][Segmentation][Video Object Segmentation] This paper introduces the physical concept of "phase" to the video object segmentation task, constructing the M3-VOS benchmark containing 479 videos, 205K masks, covering 6 physical phases and 23 transition types. It also proposes a plug-and-play method, ReVOS, to improve the performance of phase-transitioning object segmentation through reverse propagation refinement.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Video Object Segmentation"
  - "Physical Phase Transition"
  - "Reverse Propagation"
  - "Multi-phase"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: 448b4e79950e9992
---

# M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2412.13803](https://arxiv.org/abs/2412.13803)  
**Code**: [https://zixuan-chen.github.io/M-cube-VOS.github.io/](https://zixuan-chen.github.io/M-cube-VOS.github.io/)  
**Area**: Video Segmentation / Video Object Segmentation  
**Keywords**: Video Object Segmentation, Physical Phase Transition, Reverse Propagation, Multi-phase, Benchmark Dataset

## TL;DR
This paper introduces the physical concept of "phase" to the video object segmentation task, constructing the M3-VOS benchmark containing 479 videos, 205K masks, covering 6 physical phases and 23 transition types. It also proposes a plug-and-play method, ReVOS, to improve the performance of phase-transitioning object segmentation through reverse propagation refinement.

## Background & Motivation

**Background**: Video Object Segmentation (VOS) has made significant progress, driven by benchmarks like DAVIS and YouTubeVOS, which have promoted high-performance methods like XMem, Cutie, and SAM2. Dominant methods rely on appearance matching and memory bank mechanisms, performing exceptionally well in standard scenarios.

**Limitations of Prior Work**: Existing VOS benchmarks and methods almost exclusively focus on objects in a single phase (typically solid), leading to severe limitations when segmenting objects undergoing phase transitions. For example, during processes such as ice melting into water, dry ice sublimating into gas, or liquids boiling and evaporating, the appearance, shape, and boundaries of objects change drastically. Methods based on appearance matching fail severely in these scenarios—they cannot track an object whose morphology changes completely. Although VOST and VSCOS address object appearance changes, they are still limited to variations within a single physical phase.

**Key Challenge**: Phase transitions are highly common in the natural world (cooking, industrial manufacturing, laboratory operations, etc.), but the computer vision community has almost completely neglected the understanding of physical phase transitions. The root causes are: (1) the lack of annotated datasets covering diverse phase transition processes; (2) current model architectures based on appearance priors are inherently unsuited for handling scenarios with fundamental appearance changes.

**Goal**: (1) Systematically define phase classification and transition categories; (2) construct a VOS benchmark covering multiple phases, phase transitions, and scenarios; (3) propose a method to improve the segmentation of phase-transitioning objects.

**Key Insight**: The authors borrow the concept of "physical phase" from physics/chemistry but define phases using macroscopic visual features rather than microscopic molecular spacing. A key observation is that during a phase transition, the "disorder" (entropy/disorder) of an object tends to increase—meaning masks in the first half of a video are more regular than those in the second half. If propagation is performed in reverse starting from the end of the video, it becomes a process from high disorder to low disorder, which is likely easier to segment.

**Core Idea**: Utilizing the physical prior of increasing disorder during phase transitions, this paper proposes a reverse propagation mechanism to refine the segmentation results of forward propagation.

## Method

### Overall Architecture
M3-VOS consists of two parts: a dataset and a method. The dataset systematically defines three major phases (solid (granular/rigid/deformable), liquid (viscous/non-viscous), and aerosol/gas), 6 subcategories, and 23 phase transition types, collecting 479 high-resolution videos with 30fps dense mask annotations. The method part proposes ReVOS, a plug-and-play framework that can be stacked on top of any mask propagation-based VOS backbone (such as Cutie). After forward propagation is completed, ReVOS performs reverse propagation from the end of the video to predict masks, and then fuses the forward and reverse features using a Readout Fusion module to generate the final masks.

### Key Designs

1. **Phase Taxonomy**:

    - Function: Provides a systematic description framework of object states for the VOS task.
    - Mechanism: Based on macroscopic visual features, daily objects are classified into three major phases: solid (further divided into granular, rigid, and deformable), liquid (viscous/non-viscous), and aerosol/gas. Phase transitions are categorized into intra-phase transitions (e.g., liquid flowing, solid fracturing, gas diffusing) and cross-phase transitions (e.g., solidifying, melting, sublimating, evaporating, dissolving, etc., totaling 10 types). This yields 23 specific transition types.
    - Design Motivation: Existing VOS benchmarks lack descriptions of the phase attributes of objects, making it impossible to evaluate a model's ability to understand phase transitions. This systematic taxonomy provides a structured framework for benchmark design and performance analysis.

2. **Reverse Memory**:

    - Function: Harnesses the reverse process of decreasing information entropy to improve segmentation accuracy.
    - Mechanism: The authors calculate the texture entropy $h_{LBP}$ of masks using LBP (Local Binary Patterns) and find that the mask entropy in the second half of videos is higher than that in the first half in most VOS datasets (e.g., 4.72 vs 4.68 in M3-VOS). This validates the hypothesis that "forward propagation becomes progressively harder, while reverse propagation becomes progressively easier." In practice, after forward propagation is completed, the predicted mask of the last frame is taken as the starting point, and reverse propagation is executed within a sliding window of size $T$, maintaining a dedicated reverse working memory to store high-resolution features.
    - Design Motivation: During forward propagation, dramatic appearance changes caused by phase transitions deteriorate mask predictions. Reverse propagation undergoes an entropy-reduction process, which can compensate for the information loss of forward propagation.

3. **Booster**:

    - Function: Expands the coverage of the forward propagation's final-frame mask to provide a better starting point for reverse propagation.
    - Mechanism: During forward propagation, decoded logits are scaled by introducing an amplification factor $\alpha$: $M = \sigma(\alpha \cdot X_{decode})$, where $\sigma$ is the sigmoid function. This effectively lowers the threshold for mask prediction, allowing more potentially target-matching regions to be included and reducing false negatives.
    - Design Motivation: If forward propagation severely loses target information in the final frame, reverse propagation cannot recover it. The Booster ensures the final frame mask coverage is as complete as possible by relaxing detection criteria, under the premise that a few false positives are preferable to losing large target areas.

### Loss & Training
Freeze all parameters of the Cutie backbone and only train the Readout Fusion module. Use the AdamW optimizer with a learning rate of 1e-5. Train for 75K iterations in total, with the learning rate decaying by 10x at 60K and 67.5K iterations. Training takes about 10 hours on 4 A100 GPUs. The training data comprises five datasets: YouTubeVOS, DAVIS, BURST, OVIS, and MOSE.

## Key Experimental Results

### Main Results

| Method | M3-VOS Full $\mathcal{J}$ | M3-VOS Core $\mathcal{J}$ | VOST $\mathcal{J}$ | DAVIS'17 $\mathcal{J}$ |
|------|--------------------------|--------------------------|-------------------|---------------------|
| DeAOT | 72.5 | 65.2 | 82.7 | 86.2 |
| XMem | 70.4 | 61.5 | 82.9 | 85.4 |
| SAM2 | 69.5 | 57.8 | 85.5 | 85.2 |
| Cutie-base | 74.6 | 64.6 | 85.6 | 86.8 |
| **ReVOS (Ours)** | **75.6** | **66.5** | **86.0** | 86.8 |

### Ablation Study

| Configuration | $\mathcal{J}$ | $\mathcal{J}_{tr}$ | $\mathcal{J}_{cc}$ | Description |
|------|------|---------|---------|------|
| Forward only (Cutie) | 74.7 | 64.6 | 64.5 | Baseline |
| Reverse only | 75.4 | 66.4 | 64.9 | Reverse alone already shows improvement |
| +Readout Fusion | 75.7 | 66.4 | 65.3 | Fusing forward and reverse is superior |
| w/o Booster | 74.2 | 64.2 | 63.9 | Removing Booster makes it worse than baseline |
| +Booster | 75.7 | 66.4 | 65.3 | Booster makes a critical contribution |

### Key Findings
- All existing methods perform significantly worse on M3-VOS compared to traditional benchmarks like DAVIS, demonstrating that phase-transitioning object segmentation is a genuinely unsolved challenge.
- All methods perform even worse on the M3-VOS Core subset (which has a balanced distribution), showing that existing models suffer from scene bias and phase bias.
- ReVOS's improvement is most pronounced in cross-phase transition scenarios, proving that reverse propagation is most effective for handling drastic appearance changes.
- Pure reverse propagation already outperforms pure forward propagation, but combining both yields the best results.
- The Booster module is critical; without it, the starting mask for reverse propagation is incomplete, which drags down the overall performance.
- Setting the reverse interval to $L=60$ yields the best performance and the highest FPS (21.6), because step-skipping covers a larger temporal range while saving compute.

## Highlights & Insights
- **Introducing the concept of physical phase into VOS** is a highly novel perspective, re-examining the challenges of VOS through a physics lens. This taxonomy is generalizable and can guide the design of future datasets and methods.
- **The entropy-reduction motivation for reverse propagation** stems from physical intuition but is empirically validated. This idea can be transferred to other sequential prediction tasks—when forward propagation difficulty increases, a reverse process can be used to compensate.
- **The plug-and-play design** allows ReVOS to be layered on top of any mask propagation backbone, offering high practicality.

## Limitations & Future Work
- The paper acknowledges that annotation bias cannot be completely avoided, especially for gaseous/liquid objects with blurry boundaries.
- ReVOS shows negligible improvement on small objects and can occasionally degrade under certain conditions—since the Booster for small objects might introduce too many false positives.
- Reverse propagation incurs additional computational overhead, reducing FPS from Cutie's 30+ to 15-22.
- Currently, all models achieve relatively low performance on M3-VOS; introducing physical knowledge from multimodal large language models to assist in understanding phase transition processes could be a valuable future direction.
- The dataset scale (479 videos) is relatively limited; scaling up the dataset and including more extreme phase transition scenarios (e.g., explosions, chemical reactions) is a worthwhile direction.

## Related Work & Insights
- **vs VOST**: VOST focuses on changes in object appearance (such as state changes after object use) but is restricted to solid objects without covering cross-phase transitions. M3-VOS is a superset of VOST in the phase dimension.
- **vs SAM2**: SAM2 performs the best on VOST but is inferior to Cutie on M3-VOS, indicating that while SAM2 is trained on massive data, its training distribution is biased towards solid objects, limiting its generalization to liquid/gaseous phases.
- **vs DyeNet (bidirectional propagation)**: DyeNet requires high-confidence masks to perform reverse propagation. ReVOS relaxes this requirement using the Booster module and is more flexible as a plug-and-play module.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing the physical phase concept to VOS is a completely new perspective, and the entropy-reduction reverse propagation motivation is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The dataset is detailed, experiments cover multiple benchmarks, ablation studies are comprehensive, and detailed phase transition analysis is provided.
- Writing Quality: ⭐⭐⭐⭐ Concepts are defined clearly, but the paper is long and somewhat redundant in certain parts.
- Value: ⭐⭐⭐⭐⭐ M3-VOS fills the gap in VOS regarding phase transitions, and ReVOS provides an effective baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)
- [\[ICML 2025\] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning](../../ICML2025/segmentation/unmore_unsupervised_multi-object_segmentation_via_center-boundary_reasoning.md)
- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)
- [\[ECCV 2024\] OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](../../ECCV2024/segmentation/olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)

</div>

<!-- RELATED:END -->
