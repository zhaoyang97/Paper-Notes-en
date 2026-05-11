---
title: >-
  [Paper Note] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video
description: >-
  [CVPR 2026][Segmentation][Robot Segmentation] This paper presents RobotSeg, the first foundation model supporting both image and video robot segmentation. Built upon SAM 2…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Robot Segmentation"
  - "SAM2"
  - "Structure-Aware"
  - "Automatic Segmentation"
  - "Label-Efficient Learning"
date: 2026-05-08
content_hash: 3c6d66af9a60681c
---

# RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video

**Conference**: CVPR 2026
**arXiv**: [2511.22950](https://arxiv.org/abs/2511.22950)
**Code**: [https://github.com/showlab/RobotSeg](https://github.com/showlab/RobotSeg)
**Area**: Segmentation
**Keywords**: Robot Segmentation, SAM2, Structure-Aware, Automatic Segmentation, Label-Efficient Learning

## TL;DR

This paper presents RobotSeg, the first foundation model supporting both image and video robot segmentation. Built upon SAM 2, it introduces a Structure-Enhanced Memory Associator (SEMA), a Robot Prompt Generator (RPG), and a label-efficient training strategy requiring only first-frame annotations. In automatic mode, it achieves 85.1 J&F on Whole Robot segmentation, surpassing the fine-tuned SAM 2.1 by 4.9 points, with only 41.3M parameters — far fewer than existing 638M+ solutions.

## Background & Motivation

1. **Background**: Robot segmentation is a fundamental capability in robotic perception, with applications in visual servoing (VLA systems), cross-embodiment data augmentation, sim-to-real transfer, and safety monitoring. Existing approaches either rely on language-conditioned segmentation (CLIPSeg/LISA/EVF-SAM) or employ SAM 2 with general-purpose prompted segmentation.
2. **Limitations of Prior Work**: (a) Robots exhibit highly diverse morphologies (Franka/Fanuc/Sawyer/UR5, etc.) and their appearance can be easily confused with the background; (b) articulated structures are complex, causing existing models to frequently produce fragmented segmentation; (c) drastic shape changes during manipulation lead to temporal inconsistency. While SAM 2 offers strong general capabilities, it lacks structural priors for articulated robots, depends on manual prompts, and requires per-frame annotation.
3. **Key Challenge**: Robot segmentation simultaneously demands structure-awareness (articulated geometry), autonomy (no manual prompts), and annotation efficiency (high cost of large-scale video labeling) — three requirements that existing methods fail to address jointly.
4. **Goal**: To build a dedicated model and dataset enabling structure-aware, automatic, and label-efficient video robot segmentation.
5. **Key Insight**: Targeted enhancement of SAM 2 — injecting structural priors via Canny edges and multi-scale perception, generating automatic prompts via a learnable token bank with historical clustering, and achieving first-frame-only supervision via cyclic, semantic, and patch consistency.
6. **Core Idea**: Graft three modules specifically tailored to robot characteristics (structural awareness, automatic prompting, and label efficiency) onto SAM 2, achieving state-of-the-art robot segmentation with only 41M parameters.

## Method

### Overall Architecture

The input is a video sequence, from which a backbone extracts per-frame visual features. The SEMA module constructs memory from historical frame features and segmentation results, then enhances current-frame features with edge-based structural information. The RPG module generates semantic priors and temporal cues from a learnable token bank and historical memory as segmentation prompts, replacing manual click/box inputs. The enhanced features and prompts are fed into SAM 2's mask decoder to produce robot segmentation masks. During training, only the first-frame GT annotation is required; end-to-end learning is achieved via a three-level consistency loss.

### Key Designs

1. **Structure-Enhanced Memory Associator (SEMA)**:

    - **Function**: Injects historical frame information and structural priors into current-frame features for temporally consistent articulated robot segmentation.
    - **Mechanism**: Dual-branch design. **Upper branch** (temporal association): historical frame features and masks are encoded into memory $M_t$; the current-frame feature $F_t$ sequentially passes through self-attention, cross-attention with $M_t$ as key-value, and an MLP to produce temporally enhanced feature $F_t'$. **Lower branch** (structural enhancement): a Canny filter extracts the edge map $E_t$ from the current frame; edge modulation is applied as $F_t^{edge} = F_t \odot (1 + E_t)$; a multi-scale feature extractor followed by cross-attention with $M_t$ as key-value yields a structural map $S_t = \sigma(\text{CrossAttn}(F_t^{ms}, M_t))$; the two branches are fused via $F_t'' = F_t' \odot (1 + \alpha S_t)$, where $\alpha$ is a learnable weight.
    - **Design Motivation**: Articulated joints are the principal challenge in robot segmentation. Edge information provides strong structural priors, Canny is a zero-cost structural detector, and multi-scale perception simultaneously captures coarse joints and fine end-effectors.

2. **Robot Prompt Generator (RPG)**:

    - **Function**: Automatically generates segmentation prompts without manual input (clicks/boxes).
    - **Mechanism**: Two types of robot tokens are generated. **Class tokens** are retrieved from a learnable token bank according to the target category (robot arm/gripper/whole robot), providing class-level semantic priors. **Object tokens** are extracted from historical memory via hierarchical clustering — Farthest Point Sampling first initializes $R$ macro-region centers, K-Means clustering produces region masks, and within each region a further $S$ micro-level sub-clusters are formed; all prototype vectors are concatenated as object tokens. Both token types are jointly fed into the mask decoder to guide segmentation.
    - **Design Motivation**: Class tokens provide the semantic prior of "what to segment," while object tokens supply appearance cues of "how the robot looked in the previous frame." Hierarchical clustering (macro–micro) captures both coarse contours and fine end-effector details.

3. **Label-Efficient Training (LET)**:

    - **Function**: Enables training of video segmentation models using only the first-frame GT mask.
    - **Mechanism**: Three-level consistency losses. (a) **Cyclic consistency** $\mathcal{L}_{cyc}$: propagation proceeds from frame 0 to frame $t$ and back to frame 0; both the initial and recovered first-frame predictions are supervised against the first-frame GT using focal + dice loss. (b) **Semantic consistency** $\mathcal{L}_{sem}$: the mean feature within the predicted mask of an intermediate frame should remain semantically consistent with the first-frame object (maximized cosine similarity), preventing the model from learning a trivial "constant mask" shortcut. (c) **Patch consistency** $\mathcal{L}_{patch}$: DINOv3 patch similarities propagate the first-frame GT to intermediate frames as pseudo-labels, supervised with IoU loss (masks are downsampled 16× to match patch granularity). The total loss is $\mathcal{L}_{mask} = w_{cyc}\mathcal{L}_{cyc} + w_{sem}\mathcal{L}_{sem} + w_{patch}\mathcal{L}_{patch}$.
    - **Design Motivation**: Cyclic consistency exploits temporal symmetry for self-supervision using the sole available annotation; semantic consistency prevents intermediate frames from drifting away from the target semantics; patch consistency provides pixel-level pseudo-supervision. Together, the three losses form a complete hierarchical supervision covering video, object, and patch levels.

### Loss & Training

The model is jointly trained on RoboEngine-Train (3,532 images) and VRS-Train (2,707 videos, 131K frames) for 25 epochs. AdamW optimizer is used with lr=3×10⁻⁴ for the image encoder and lr=6×10⁻⁵ for other components, with cosine decay. The structural map $S_t$ receives additional supervision. Training runs for 15 hours on 8 × NVIDIA A5000 GPUs.

## Key Experimental Results

### Main Results

**VRS Video Dataset (Whole Robot J&F)**

| Method | Params | Auto (AU) | 1-click | 3-click | BBox | Interactive (OI) |
|--------|--------|-----------|---------|---------|------|-----------------|
| RoboEngine (fine-tuned) | 898.4M | - | - | - | - | - |
| SAM 2.1 (original) | 39.0M | - | 38.2 | 69.0 | 60.4 | 73.6 |
| SAM 2.1 (fine-tuned) | 39.0M | - | 73.6 | 82.1 | 82.5 | 85.1 |
| **RobotSeg** | **41.3M** | **85.1** | **85.1** | **86.3** | **85.8** | **86.7** |

**RoboEngine Image Dataset (Whole Robot J&F)**

| Method | Params | Auto (AU) | 1-click | 3-click | BBox |
|--------|--------|-----------|---------|---------|------|
| RoboEngine (fine-tuned) | 898.4M | 86.6 | - | - | - |
| SAM 2.1 (fine-tuned) | 39.0M | - | 78.0 | 90.2 | 86.0 |
| **RobotSeg** | **41.3M** | **87.9** | **88.8** | **93.5** | **89.4** |

### Ablation Study

| Configuration | AU | 1C | Added Components |
|---------------|----|----|-----------------|
| (a) SAM 2.1 original | - | 38.2 | - |
| (b) Fine-tuned | - | 73.6 | Robot data |
| (e) +LET full | - | 77.4 | Cyclic+Semantic+Patch |
| (g) +RPG full | 83.1 | 83.3 | Class+Object token |
| (i) +SEMA full (complete) | **85.1** | **85.1** | Multi-scale+Memory-guided |

### Key Findings
- **Automatic segmentation capability**: RobotSeg is the only high-accuracy method capable of automatic segmentation without any prompt (85.1 J&F), with virtually no gap between automatic and 1-click modes (85.1 vs. 85.1), demonstrating that RPG's automatic prompts are sufficient.
- **Parameter efficiency**: At 41.3M parameters, RobotSeg is substantially more compact than RoboEngine (898.4M) and LISA (13,993M), making it suitable for on-robot deployment.
- **Label efficiency**: LET improves performance from 73.6 to 77.4 (+3.8) using only first-frame annotations, substantially reducing the burden of per-frame labeling.
- **Value of structural enhancement**: SEMA yields an additional 2.0-point gain over the RPG-only variant (83.1→85.1), confirming that edge-awareness and multi-scale modeling are critical for articulated robot segmentation.
- **Fine-grained segmentation**: RobotSeg segments not only the whole robot but also individual components — robot arm (75.6 AU) and gripper (76.0 AU) — supporting part-level data augmentation and motion analysis.

## Highlights & Insights
- **The VRS dataset** is the first video-level robot segmentation benchmark (2,812 videos, 138K frames, 10 robot types), 38× larger than RoboEngine, representing a significant infrastructure contribution to the field.
- **The three-level label-efficient training loss design** is elegantly constructed: cyclic consistency leverages temporal symmetry for self-supervision, semantic consistency prevents degenerate solutions, and DINOv3 patch propagation supplies pseudo-supervision — this strategy is directly transferable to other video segmentation settings with only first-frame annotations.
- **The structural enhancement implementation** is practically motivated: Canny edges + multi-scale perception + memory-guided modulation introduce negligible computational overhead while consistently improving segmentation quality at articulated regions.
- **The hierarchical clustering strategy in RPG** (FPS initialization → macro-level K-Means → micro-level K-Means) provides a coarse-to-fine object representation that is particularly effective for articulated robots with drastic appearance variations.

## Limitations & Future Work
- The work focuses exclusively on robot segmentation and does not explore generalization to other articulated objects (e.g., human hands, tools).
- Canny edge detection is a fixed operator and may fail under low-texture conditions or motion blur; learned edge detectors could be considered as a replacement.
- First-frame annotation still requires human effort; integrating VLMs to enable fully zero-annotation automatic segmentation is a promising direction.
- The VRS test set contains only 105 videos; expanding the evaluation set would allow more reliable assessment.

## Related Work & Insights
- **vs. RoboEngine**: RoboEngine handles only image-level segmentation and relies on EVF-SAM (898M); RobotSeg achieves competitive or superior performance in both image and video settings with 41.3M parameters, outperforming by 0.7–5.0 points in AU mode.
- **vs. SAM 2.1**: Even after fine-tuning, general-purpose SAM 2.1 is unavailable in automatic mode (requiring manual prompts), whereas RobotSeg achieves fully automatic segmentation through RPG.
- **vs. LISA/EVF-SAM**: Language-conditioned segmentation models carry enormous parameter counts (14B/898M) and achieve only 42–64 J&F in automatic mode, far below RobotSeg's 85.1.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The three individual enhancements over SAM 2 are not individually groundbreaking, but their combination is well-motivated; the dataset contribution is significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five evaluation settings, image + video benchmarks, fine-grained part segmentation, comprehensive ablations, and multi-method comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rich illustrations, though descriptions of certain modules are somewhat verbose.
- **Value**: ⭐⭐⭐⭐⭐ Both the dataset and model have direct practical value for the robot perception community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events](a_dataset_of_medication_images_with_instance_segme.md)
- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)
- [\[CVPR 2026\] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)
- [\[CVPR 2026\] Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)
- [\[AAAI 2026\] Tracking and Segmenting Anything in Any Modality](../../AAAI2026/segmentation/tracking_and_segmenting_anything_in_any_modality.md)

</div>

<!-- RELATED:END -->
