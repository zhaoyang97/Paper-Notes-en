---
title: >-
  [Paper Note] Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach
description: >-
  [ECCV 2024][Remote Sensing][Cross-Platform Person Re-Identification] This work constructs G2A-VReID, the first ground-to-aerial cross-platform video person re-identification dataset, and proposes the VSLA-CLIP method, which adapts CLIP to video ReID tasks through visual-semantic alignment and a parameter-efficient Video Set-Level-Adapter.
tags:
  - "ECCV 2024"
  - "Remote Sensing"
  - "Cross-Platform Person Re-Identification"
  - "Video ReID"
  - "CLIP Adaptation"
  - "Visual-Semantic Alignment"
  - "UAV"
date: 2026-05-08
content_hash: 148973fd0b85dace
---

# Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach

**Conference**: ECCV 2024  
**arXiv**: [2408.07500](https://arxiv.org/abs/2408.07500)  
**Code**: [https://github.com/FHR-L/VSLA-CLIP](https://github.com/FHR-L/VSLA-CLIP)  
**Area**: Remote Sensing  
**Keywords**: Cross-Platform Person Re-Identification, Video ReID, CLIP Adaptation, Visual-Semantic Alignment, UAV

## TL;DR

This work constructs G2A-VReID, the first ground-to-aerial cross-platform video person re-identification dataset, and proposes the VSLA-CLIP method, which adapts CLIP to video ReID tasks through visual-semantic alignment and a parameter-efficient Video Set-Level-Adapter.

## Background & Motivation

Video person re-identification (VReID) has recently received widespread attention because videos provide richer appearance and temporal information than single images. However, **existing VReID datasets and methods are almost entirely based on a single platform (ground-level surveillance cameras)**. Under limited camera perspective changes, visual part alignment between query and gallery videos is typically achieved using simple striping-based splitting.

In practical applications, a critical scenario exists: when a suspect flees a city into a remote area where no ground-level surveillance is deployed, a cross-platform search via unmanned aerial vehicle (UAV) platforms is required. This scenario introduces **unprecedented challenges**:

**Drastic perspective changes**: Ground cameras are placed at approximately 2 meters high, while UAVs fly at heights of 20-60 meters, causing extreme differences in pitch angles.

**Drastic resolution differences**: Pedestrian widths are around 10-70 pixels in ground cameras, but only 5-35 pixels in UAV observations.

**Severe self-occlusion**: Visual appearance information is heavily lost under aerial top-down perspectives, causing traditional temporal modeling methods to degrade in performance.

**Difficult visual alignment**: Cross-platform visual feature alignment is far more challenging than single-platform alignment.

**Key Challenge**: Traditional methods realize ReID by directly aligning visual part features. However, due to extreme differences in viewpoints and resolutions in cross-platform scenarios, direct alignment in the visual space is almost impractical.

**Key Insight**: Given the difficulty of direct visual spatial alignment, can vision-language models like CLIP be utilized to **reformulate the cross-platform visual alignment problem into a visual-semantic alignment problem**? By learning semantic descriptions for each ID, visual features from different platforms can be aligned onto a unified semantic space, thereby bypassing direct cross-platform visual matching.

**Core Idea**: Leverage the visual-semantic alignment capability of CLIP to address cross-platform feature alignment, while proposing a parameter-efficient Video Set-Level-Adapter (VSLA) that treats videos as unordered sets of frames rather than temporal sequences to aggregate complementary information across frames.

## Method

### Overall Architecture

The method consists of two main parts: (1) an FT-CLIP baseline, which achieves visual-semantic alignment by fine-tuning the CLIP image encoder; and (2) VSLA-CLIP, which replaces full-parameter fine-tuning with a parameter-efficient adapter while introducing Platform-Bridge Prompts (PBP) to further narrow the cross-platform gap. A two-stage training strategy is adopted: the first stage learns ID-specific description tokens and shared text prompts, and the second stage trains the image encoder (or adapters) while freezing the text side.

### Key Designs

1. **Visual-Semantic Alignment**:

   Function: Reformulate cross-platform visual alignment into visual-semantic alignment.

   Mechanism: Learn a set of trainable description tokens $[\mathbf{S}]_i$ and shared text prompts $[\mathbf{P}]_i$ for each identity ID, which are fed into the CLIP Text Encoder to generate semantic features $\mathbf{T}$. Then, a visual-to-semantic cross-entropy loss $\mathcal{L}_{v2sce}$ is used to align the video visual embedding $\mathbf{V}_i$ with the semantic features:

    $\mathbf{V}_i = \frac{1}{T}\sum_{j}^{T}\mathbf{E}_i(\mathcal{V}_{ij})$

    $\mathcal{L}_{v2sce}(i) = \sum_{k=1}^{N}-q_k\log\frac{\exp(s(\mathbf{V}_i, \mathbf{T}_{y_k}))}{\sum_{y_j=1}^{N}\exp(s(\mathbf{V}_i, \mathbf{T}_{y_j}))}$

   Design Motivation: Visual features from different platforms vary drastically in the visual space, but the description of the same individual should remain consistent in the semantic space. Using the semantic space as a bridge allows indirect cross-platform feature alignment.

2. **Video Set-Level-Adapter (VSLA)**:

   Function: Adapt pre-trained image foundation models to video ReID tasks in a parameter-efficient manner.

   VSLA consists of two components:

    - **Intra-Frame Adapter (IFA)**: A bottleneck structure running in parallel with the MLP of each ViT layer to provide intra-frame appearance representation adaptation:
    $\text{IFA}(\mathbf{x}_i') = \sigma(\mathbf{x}_i'\mathbf{W}_{down})\mathbf{W}_{up}$
      It accounts for only 5.5% of the total Image Encoder parameters (when $\alpha=256$).

    - **Cross-Frame Attention Adapter (CFAA)**: A cross-frame attention adapter that reshapes the input $\mathbf{x} \in \mathbb{R}^{T\times(N+1)\times\alpha}$ into $\mathbb{R}^{(N+1)\times T\times\alpha}$ to perform attention along the frame dimension, thereby aggregating complementary information.

   Core Idea: **Treat videos as unordered sets of frames instead of temporal sequences**. Under aerial perspectives, temporal information is highly limited (due to severe self-occlusion), making complementary information more vital than temporal modeling. The model is invariant to frame permutation: $\mathbf{M}(\{\mathcal{V}_{ij}\}) = \mathbf{M}(\{\mathcal{V}_{i\pi(j)}\})$.

3. **Platform-Bridge Prompt (PBP)**:

   Function: Further bridge the semantic gap between the surveillance ground platform and the aerial platform.

   Mechanism: Insert platform-specific learnable prompts into the Multi-Head Self-Attention (MSA) of the first $d$ layers of the Image Encoder depending on the source of the input:

    $f_k(\mathbf{h}, \mathbf{p}_k) = \begin{cases} \text{MSA}_k([\mathbf{h}:\mathbf{p}_k^{ground}]) & \text{if } \mathbf{h} \in Set^{ground} \\ \text{MSA}_k([\mathbf{h}:\mathbf{p}_k^{uav}]) & \text{if } \mathbf{h} \in Set^{uav} \end{cases}$

   Design Motivation: Prompts act as explicit instructions that guide the pre-trained model to focus on learning platform-invariant features, automatically abstracting platform-agnostic representations.

### Loss & Training

Two-stage training:
- **Stage 1**: Freeze the Image/Text Encoder, optimize the ID description tokens and shared text prompts using $\mathcal{L}_{i2t} + \mathcal{L}_{t2i}$.
- **Stage 2**: Freeze the Text Encoder and description tokens. The overall loss is:

$$\mathcal{L}_{stage2} = \mathcal{L}_{v2sce} + \beta\mathcal{L}_{tri} + \gamma\mathcal{L}_{id} + \delta\mathcal{L}_{i2t} + \epsilon\mathcal{L}_{t2i}$$

where $\beta=1.0, \gamma=0.25, \delta=1.0, \epsilon=1.0$. Adam optimizer is used, with ViT-Base-16 as the Image Encoder, and weights initialized from ViFi-CLIP.

## Key Experimental Results

### Main Results

| Dataset | Metric | VSLA-CLIP‡ | Prev. SOTA | Gain |
|--------|------|------------|----------|------|
| MARS | mAP | 88.60 | 87.0 (DenseIL) | +1.60 |
| MARS | Rank-1 | 91.82 | 91.6 (LSTRL) | +0.22 |
| LS-VID | mAP | 85.20 | 82.4 (LSTRL) | +2.80 |
| LS-VID | Rank-1 | 91.66 | 89.8 (LSTRL) | +1.86 |
| iLIDS | Rank-1 | 95.33 | 92.5 (SINet) | +2.83 |
| G2A-VReID | mAP | 79.70 | 76.7 (MGH) | +3.00 |
| G2A-VReID | Rank-1 | 72.55 | 69.9 (MGH) | +2.65 |

### Ablation Study

**Component contributions (on LS-VID and G2A-VReID datasets):**

| Configuration | Tunable Params (M) | LS-VID mAP | G2A mAP | Description |
|------|-----------|------------|---------|------|
| baseline | 86.1 | 76.10 | 72.80 | Fine-tuned with triplet + ID loss only |
| baseline+VSA (FT-CLIP‡) | 88.0 | 84.07 | 78.11 | Visual-Semantic Alignment + Full fine-tuning |
| IFA | 4.7 | 77.31 | 73.82 | Intra-frame adapter only |
| IFA+VSA | 6.6 | 84.16 | 79.01 | Adapter + Semantic Alignment |
| IFA+VSA+CFAA (VSLA-CLIP‡) | 14.5 | 85.20 | 79.70 | Complete adapter |
| IFA+VSA+CFAA+PBP | 14.5 | - | 81.29 | + Platform-Bridge Prompt |

**Dimension $\alpha$ ablation of IFA/CFAA (on LS-VID):**

| α | Tunable Params (M) | mAP | Rank-1 |
|---|-----------|-----|--------|
| 64 | 2.6 | 79.58 | 86.71 |
| 128 | 5.6 | 83.64 | 90.00 |
| 256 | 12.6 | 85.20 | 91.66 |

### Key Findings

1. **Visual-Semantic Alignment is the most critical component**: Regardless of using full fine-tuning or adapters, adding VSA yields the most significant performance improvement (+7.97 / +6.85 on LS-VID mAP).
2. **VSLA-CLIP outperforms FT-CLIP with only 16.5% of its parameters**: With 14.5M vs. 88.0M tunable parameters, VSLA-CLIP achieves higher mAP on both LS-VID (85.20 vs. 84.07) and G2A-VReID (79.70 vs. 78.11).
3. **Temporal modeling performs poorly in cross-platform scenarios**: Temporal models like STMN perform worse on G2A-VReID compared to part-alignment methods like MGH.
4. **PBP brings an additional 1.59% mAP improvement on the cross-platform dataset** (81.29 vs. 79.70) but is of no use on single-platform datasets.

## Highlights & Insights

- **Elegant paradigm shift**: Transforming visual alignment into semantic alignment is a generic and generalized approach to handling cross-modal/cross-domain alignment problems.
- **The perspective of treating videos as unordered sets is noteworthy**: Under special perspectives like aerial imaging, traditional temporal modeling assumptions are no longer suitable.
- **Highly parameter-efficient**: VSLA adapts the foundation model with only 5.5% of its parameter size, and set-level attention is naturally invariant to frame order.
- **The dataset itself is highly valuable**: G2A-VReID is the first ground-to-aerial cross-platform video ReID dataset, containing 2,788 IDs and 185,907 images.

## Limitations & Future Work

- Although large in scale, the G2A-VReID dataset features limited scene types (9 categories) and lacks coverage of night scenes and adverse weather conditions.
- It only considers two platforms (ground-level + UAV) and has not extended to multi-platform settings (such as vehicle-mounted or wearable cameras).
- PBP requires knowing the source platform of the samples during training, and platform-specific labels are also needed during deployment.
- Integrating VSLA with larger-scale vision foundation models (such as ViT-Large or EVA) is a potential future study.

## Related Work & Insights

- CLIP-ReID [Li et al.] first applied CLIP to image ReID, whereas this work extends it to video ReID and cross-platform scenarios.
- The principles of parameter-efficient fine-tuning methods like LoRA are borrowed for the VSLA design, but IFA runs in parallel only to the MLP, yielding fewer parameters.
- The concept of set-level video representations can inspire other video understanding tasks (e.g., unordered frame aggregation in action recognition).

## Rating

- Novelty: ⭐⭐⭐⭐ Triple contributions including a cross-platform dataset, semantic-alignment mechanism, and set-level adapter.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation on four datasets with thorough ablation, achieving SOTA results on all datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and fluid logical flow.
- Value: ⭐⭐⭐⭐ The dataset and the methodology provide significant advancements to the cross-platform ReID domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[CVPR 2026\] WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification](../../CVPR2026/remote_sensing/whu-mars_a_multispectral_aerial-ground_benchmark_towards_any-scenario_person_re-.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](../../CVPR2026/remote_sensing/cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)
- [\[CVPR 2026\] YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction](../../CVPR2026/remote_sensing/yieldsat_a_multimodal_benchmark_dataset_for_high-resolution_crop_yield_predictio.md)
- [\[ECCV 2024\] Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)

</div>

<!-- RELATED:END -->
