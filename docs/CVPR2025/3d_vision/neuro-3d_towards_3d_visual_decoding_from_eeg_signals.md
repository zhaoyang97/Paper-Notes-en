---
title: >-
  [Paper Note] Neuro-3D: Towards 3D Visual Decoding from EEG Signals
description: >-
  [CVPR 2025][3D Vision][EEG decoding] Neuro-3D is the first work to reconstruct colored 3D point clouds from electroencephalography (EEG) signals. It introduces the EEG-3D dataset (12 subjects, 72 Objaverse object categories, dynamic video + static image stimuli) and achieves cross-modal 3D visual decoding through a dynamic-static EEG fusion encoder, CLIP-aligned contrastive learning, and diffusion-based point cloud generation with color prediction.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "EEG decoding"
  - "Brain-Computer Interface (BCI)"
  - "3D point cloud reconstruction"
  - "Dynamic-static fusion"
  - "CLIP alignment"
date: 2026-05-08
content_hash: 4cec724c8b1427e8
---

# Neuro-3D: Towards 3D Visual Decoding from EEG Signals

**Conference**: CVPR 2025  
**arXiv**: [2411.12248](https://arxiv.org/abs/2411.12248)  
**Code**: [https://github.com/gzq17/neuro-3D](https://github.com/gzq17/neuro-3D)  
**Area**: 3D Vision  
**Keywords**: EEG decoding, Brain-Computer Interface (BCI), 3D point cloud reconstruction, Dynamic-static fusion, CLIP alignment

## TL;DR

Neuro-3D is the first work to reconstruct colored 3D point clouds from electroencephalography (EEG) signals. It introduces the EEG-3D dataset (12 subjects, 72 Objaverse object categories, dynamic video + static image stimuli) and achieves cross-modal 3D visual decoding through a dynamic-static EEG fusion encoder, CLIP-aligned contrastive learning, and diffusion-based point cloud generation with color prediction.

## Background & Motivation

1. **Background**: Brain signal visual decoding originated from fMRI, and 2D image reconstruction has already been achieved (e.g., MindEye, Brain-Diffuser). EEG has attracted attention due to its portability and high temporal resolution, but existing EEG decoding is limited to 2D images or category classification.
2. **Limitations of Prior Work**: (1) There is no prior work on decoding from EEG to 3D—3D reconstruction requires understanding the shape and appearance of objects, while EEG signals are highly noisy; (2) There is a lack of datasets containing both EEG recordings and 3D ground truths; (3) Existing EEG datasets (such as Things-EEG, GOD) lack 3D annotations and dynamic video stimuli.
3. **Key Challenge**: The signal-to-noise ratio of EEG is extremely low (due to non-invasive acquisition), whereas 3D reconstruction requires detailed shape and color information—creating a significant gap between signal quality and target complexity.
4. **Goal**: Establish the EEG-3D dataset and design a complete decoding pipeline from EEG to 3D point clouds.
5. **Key Insight**: Dynamic video stimuli (object rotation) provide 3D viewpoint variation information, while static image stimuli provide stable appearance information—the fusion of both allows EEG signals to capture a more complete 3D perception.
6. **Core Idea**: Dynamic-static EEG fusion $\rightarrow$ CLIP alignment (contrastive learning) $\rightarrow$ shape generation (diffusion point cloud) + color prediction (single-step coloring).

## Method

### Overall Architecture

Dynamic EEG $e_d$ (viewing rotating videos) + static EEG $e_s$ (viewing images) $\rightarrow$ dynamic-static fusion encoder (adaptive aggregation via cross-attention) $\rightarrow$ decoupling into geometric features $f_g$ and appearance features $f_a$ $\rightarrow$ CLIP-aligned contrastive learning $\rightarrow$ $f_g$-conditioned diffusion generation of an 8192-point 3D point cloud $\rightarrow$ $f_a$-conditioned single-step color prediction $\rightarrow$ colored 3D point cloud.

### Key Designs

1. **Dynamic-Static EEG Fusion Encoder**

    - **Function**: Adaptively fuse dynamic (temporally rich) and static (high signal-to-noise ratio) EEG signals.
    - **Mechanism**: Static EEG is encoded as $z_s = E_s(e_s)$, and dynamic EEG is encoded as $z_d = E_d(e_d)$ (incorporating temporal self-attention). Adaptive neural aggregator: $z_{sd} = \text{Softmax}(QK^T/\sqrt{d})V$, where $Q$ is derived from the static representation, and $K/V$ are derived from the dynamic representation.
    - **Design Motivation**: Dynamic videos provide multi-angle information but elicit complex EEG responses; static images have a higher signal-to-noise ratio but lack 3D perspectives. Cross-attention allows the static representation to guide ($Q$) while the dynamic representation complements ($K/V$).

2. **Geometry-Appearance Decoupled Learning**

    - **Function**: Decompose the EEG representation into separate shape and color branches.
    - **Mechanism**: The fused features are mapped via two MLPs into $f_g$ (geometry) and $f_a$ (appearance), which are individually aligned with CLIP visual features: $\mathcal{L}_{align} = \alpha \cdot \text{CLIP}(f, f_v) + (1-\alpha) \cdot \text{MSE}(f, f_v)$, coupled with a category classification loss $\mathcal{L}_c$.
    - **Design Motivation**: 3D shape and color are independent attributes—different colors can exist for the same shape; decoupling them enables more efficient learning for each branch.

3. **Diffusion Point Cloud Generation + Majority Voting Coloring**

    - **Function**: Generate 3D shapes from EEG features and apply color.
    - **Mechanism**: A Point-Voxel Network (PVN) is used as a denoiser, generating 8192 3D points via a Markov diffusion process conditioned on $f_g$. Coloring is simplified using majority voting—predicting the dominant color of the object rather than point-wise colors to reduce prediction complexity.
    - **Design Motivation**: EEG signals are too noisy to support accurate point-wise color prediction; majority voting provides a reasonable holistic color.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{align}(f_g, f_v) + \mathcal{L}_{align}(f_a, f_v) + \gamma \mathcal{L}_c$. The feature dimension is 1024, and videos are downsampled to $n=4$ frames for CLIP alignment.

## Key Experimental Results

### Main Results

| Task | Metric | Neuro-3D | Baseline |
|------|------|----------|------|
| Object Category Classification (72 classes) | top-1 | Significantly outperforms | DeepNet 3.70%, Random 1.39% |
| Color Category Classification (6 classes) | top-1 | Significantly outperforms | DeepNet 20.95%, Random 16.67% |
| 3D Reconstruction | 2-way top-1 | Effectively discriminates | - |
| 3D Reconstruction | Chamfer Distance | Reasonably generates | - |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Static EEG only | Classification drops | Lacks 3D perspective information |
| Dynamic EEG only | Classification drops | Insufficient signal-to-noise ratio |
| Dynamic + static fusion | Optimal | Complementary information |
| w/o CLIP alignment | Reconstruction degrades | Semantic alignment serves as a bridge |
| w/o Decoupling | Shape/color confusion | Decoupling assists individual learning |

### Key Findings

- Dynamic-static fusion performs better than using either modality alone—confirming that the two stimuli provide complementary information.
- Although 3D point cloud reconstruction from EEG is coarse, it is recognizable at the category level—marking the first step in this direction.
- The EEG-3D dataset is the first benchmark that simultaneously contains EEG recordings, 3D ground truths, and color information.

## Highlights & Insights

- **Pioneering Problem Definition**: Proposing the visual decoding task from EEG to 3D for the first time.
- **Long-term Value of the EEG-3D Dataset**: 12 subjects $\times$ 72 classes $\times$ multi-modal annotations, which can support various future works.
- **Innovation of Dynamic Video Stimuli**: Previous EEG datasets only featured static images—rotating videos provide crucial cues for 3D perception.

## Limitations & Future Work

- The low signal-to-noise ratio of EEG results in coarse reconstruction quality—future work could consider higher-quality signals such as fNIRS or Electrocorticography (ECoG).
- Color prediction is simplified to majority voting—point-wise color prediction requires more effective signal decoding.
- Only 12 subjects are included; generalization to a larger population requires validation.
- The 3D reconstruction is mainly recognizable at the category level, with limited ability for fine-grained discrimination within the same category.

## Related Work & Insights

- **vs MindEye/Brain-Diffuser**: Reconstruction of 2D images from fMRI. Neuro-3D extends this to 3D and utilizes more portable EEG.
- **vs Mind-3D**: Also features 3D annotations but lacks color information. EEG-3D includes color annotations for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First 3D point cloud reconstruction from EEG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across datasets, methods, classification, and reconstruction.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Pioneering work and dataset contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3D-VCD: Hallucination Mitigation in 3D-LLM Embodied Agents through Visual Contrastive Decoding](../../CVPR2026/3d_vision/3d-vcd_hallucination_mitigation_in_3d-llm_embodied_agents_through_visual_contras.md)
- [\[CVPR 2025\] Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](doppelgangers_improved_visual_disambiguation_with_geometric_3d_features.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
