---
title: >-
  [Paper Note] Referring Atomic Video Action Recognition
description: >-
  [ECCV 2024][Video Understanding][atomic action recognition] This paper proposes a new task, "Referring Atomic Video Action Recognition" (RAVAR), and the RefAVA dataset (containing 36,630 instances). It also introduces RefAtomNet, which fuses visual, textual, and location-semantic tri-stream tokens through cross-stream agent attention, improving mAP by 3.85%/3.17% over the best baseline BLIPv2.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "atomic action recognition"
  - "referring expression"
  - "multi-stream fusion"
  - "agent attention"
date: 2026-05-08
content_hash: 170668256f738ca5
---

# Referring Atomic Video Action Recognition

**Conference**: ECCV 2024  
**arXiv**: [2407.01872](https://arxiv.org/abs/2407.01872)  
**Code**: [https://ravar-dataset.github.io/](https://ravar-dataset.github.io/)  
**Area**: Video Understanding / Human Understanding  
**Keywords**: atomic action recognition, referring expression, multi-stream fusion, agent attention, video understanding

## TL;DR
This paper proposes a new task, "Referring Atomic Video Action Recognition" (RAVAR), and the RefAVA dataset (containing 36,630 instances). It also introduces RefAtomNet, which fuses visual, textual, and location-semantic tri-stream tokens through cross-stream agent attention, improving mAP by 3.85%/3.17% over the best baseline BLIPv2.

## Background & Motivation
**Background**: Atomic Video Action Recognition focuses on the most fundamental, indivisible actions of humans. Existing works (such as I3D, X3D, MViTv2, Hiera) in multi-person scenarios either manually crop specific person regions or automatically detect all individuals and predict their actions separately, which requires extensive pre- and post-processing.

**Limitations of Prior Work**: In practical applications (such as assistive systems or human-computer interaction), users often only care about the actions of specific individuals. Existing methods either predict for everyone (inefficient) or require manual cropping (impracticable). There is a lack of mechanism to specify target individuals using natural language descriptions.

**Key Challenge**: Videos contain a large amount of irrelevant visual information that distracts the model from focusing on the target individual. How to suppress irrelevant information based on the referring expressions is the core challenge.

**Goal**: To define the RAVAR task: given a video and a text describing a specific individual (e.g., "the woman in a red shirt"), recognize the individual's atomic actions and output their location.

**Key Insight**: A tri-stream architecture (visual + textual + location-semantic) that suppresses irrelevant information via cross-stream fusion with agent attention.

**Core Idea**: Introducing a location-semantic aware stream (fusing detection box coordinates and target category semantics) combined with cross-stream agent attention fusion to precisely locate the specific individual described by text and recognize their atomic actions.

## Method

### Overall Architecture
RefAtomNet comprises three token streams: (1) Visual stream—ViT encodes video frames, and QFormer extracts visual tokens; (2) Textual referring stream—BERT encodes the referring text, and QFormer extracts textual tokens; (3) Location-semantic stream—DETR detects objects, then merges bounding box coordinates and category semantic embeddings. The tri-stream tokens are fused via agent-based cross-stream attention, and then an MLP head predicts actions and bounding boxes.

### Key Designs

1. **Location-Semantic Aware Tokens**

    - **Function**: To encode the physical locations and semantic information of detected objects in the scene into tokens, assisting in localization.
    - **Mechanism**: Freezing DETR to detect $N_o$ objects from keyframes, obtaining bounding boxes $\mathbf{r}_{boxes} \in \mathbb{R}^{N_o \times 4}$ and category labels $\mathbf{r}_{cats}$. Category labels are encoded via BERT to obtain semantic embeddings, which are then concatenated with bounding box coordinates and projected:
    $\mathbf{t}^{LS} = \mathbf{P}_{LS}(\text{Concat}[\mathcal{V}_{RT}(\mathbf{r}_{cats}), \mathbf{r}_{boxes}])$
    - **Design Motivation**: Referring descriptions often contain spatial cues (e.g., "the person on the left"), which are difficult to extract accurately from visual features alone. Object detection results naturally provide location and semantics, serving as auxiliary cues for localization.

2. **Cross-Stream Agent Attention Fusion**

    - **Function**: To leverage agent tokens to suppress irrelevant visual information across streams.
    - **Mechanism**:
        - Calculate the Q, K, V, and agent token $\mathbf{A}^\phi$ for each stream $\phi \in \{RT, VT, LS\}$.
        - Compute agent attention for the textual and location-semantic streams: $\mathbf{M}_{QA}^\pi = \sigma_c(\text{MatMul}[\alpha \cdot \mathbf{A}_*^\pi, \mathbf{Q}^\pi])$.
        - Perform cross-stream fusion of agent query attention with the visual stream:
    $\hat{\mathbf{M}}_{QA}^\gamma = \text{AVG}[\mathbf{M}_{QA}^\gamma, \sigma_c(\sum_\pi \mathbf{M}_{QA}^\pi) \cdot \mathbf{M}_{QA}^\gamma, \sigma_t(\sum_\pi \mathbf{M}_{QA}^\pi) \cdot \mathbf{M}_{QA}^\gamma]$
        - Similarly compute cross-stream agent token fusion, and ultimately aggregate all streams: $\mathbf{t}_{agg} = \sum_\phi \mathbf{t}_*^\phi / N_s$.
    - **Design Motivation**: Standard attention struggles to effectively distinguish the relative importance of different streams. Agent attention aggregates critical information through intermediate agent tokens while filtering out redundancies. Applying this across streams allows textual and location-semantic cues to guide the visual stream to attend to the correct regions.

3. **1D Sequential Agent Token Adaptation**

    - **Function**: To adapt the agent attention, originally designed for 2D images, to a 1D sequential format.
    - **Mechanism**: Replacing 2D pooling with fully-connected layers to acquire agent tokens, and removing the depthwise convolution branch as well as 2D positional encodings.
    - **Design Motivation**: The tri-stream tokens stem from different modalities and are all in 1D sequence format, making the original 2D design of agent attention inapplicable.

### Loss & Training
- BCE Loss (multi-label action classification): $L_{BCE} = -\frac{1}{N_c}\sum_{i=1}^{N_c}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- MSE Loss (bounding box regression): $L_{MSE} = \sum_{j=1}^{4}(b_j - \hat{b}_j)^2$, with a weight of 5.
- BertAdam optimizer, lr=1e-4, batch=128, 40 epochs.
- Text encoder is frozen, with 214M trainable parameters.

## Key Experimental Results

### RefAVA Dataset

| Attribute | Value |
|------|------|
| Video Clips | 17,946 (from 127 movies) |
| Annotated instances | 36,630 |
| Train / Val / Test | 22,658 / 10,916 / 3,056 |
| Atomic Action Categories | 80 classes (object manipulation + person interaction + body movement) |
| Total Frames | 1,615,140 |

### Main Results

| Method | Category | Val mIOU | Val mAP | Val AUROC | Test mIOU | Test mAP | Test AUROC |
|------|------|----------|---------|-----------|-----------|----------|------------|
| I3D | AAL | 0.00 | 44.04 | 57.77 | 0.00 | 44.64 | 62.71 |
| X3D | AAL | 0.26 | 44.45 | 59.09 | 0.27 | 46.34 | 64.51 |
| AskAnything | VQA | 20.09 | 51.42 | 66.12 | 22.35 | 52.25 | 69.35 |
| BLIPv2 | VTR | 32.99 | 52.13 | 66.56 | 32.75 | 53.19 | 69.92 |
| Su et al. | VOS | 23.71 | 52.17 | 66.67 | 26.02 | 53.20 | 70.19 |
| **RefAtomNet** | **Ours** | **38.22** | **55.98** | **69.73** | **36.42** | **57.52** | **73.95** |

### Ablation Study

| Configuration | Val mIOU | Val mAP | Val AUROC | Description |
|------|----------|---------|-----------|------|
| w/o ALSAF (simple addition fusion) | 27.30 | 50.70 | 65.31 | mIOU decreases by 10.92 |
| w/o LSAS (w/o location-semantic stream) | 31.90 | 55.21 | 69.47 | Degraded localization capability |
| w/o CAAF (w/o cross-stream attention fusion) | 36.21 | 55.43 | 69.66 | Minor drop |
| w/o CATF (w/o cross-stream token fusion) | 35.01 | 53.83 | 67.71 | AUROC decreases by 2.02 |
| **RefAtomNet (Full)** | **38.22** | **55.98** | **69.73** | — |

### Comparison with other fusion mechanisms

| Fusion Method | Val mIOU | Val mAP | Val AUROC |
|----------|----------|---------|-----------|
| Addition | 27.30 | 50.70 | 65.31 |
| Concatenation | 18.64 | 52.23 | 66.45 |
| AttentionBottleneck | 33.47 | 50.97 | 65.07 |
| **Ours (Agent Fusion)** | **38.22** | **55.98** | **69.73** |

### Key Findings
- AAL methods (I3D, X3D, etc.) score close to 0 on mIOU, indicating they are completely unable to localize the text-specified individual.
- VQA and VTR baselines benefit from text-aware pre-training, yet fine-grained predictions for atomic actions remain insufficient.
- Simple addition fusion of the three streams causes the mIOU to plunge (from 38.22 to 27.30), demonstrating that indiscriminate fusion introduces significant irrelevant visual distractions.
- The location-semantic stream provides the most substantial improvement to mIOU (from 31.90 to 38.22), which indicates that bounding box coordinates combined with category semantics are essential for localizing the referred individual.

## Highlights & Insights
- **Meaningful New Task Definition**: RAVAR directly links natural language references with atomic action recognition, addressing the practical demand of identifying "who did what" in multi-person scenarios. The dataset is extended from AVA, with high-quality referring descriptions provided by 7 annotators.
- **Cross-Stream Redefinition of Agent Attention**: Adapting agent attention from 2D images to 1D multi-stream sequence fusion is highly innovative. By passing and filtering information across modalities via agent tokens, this approach is far more effective than simple concatenation or attention bottlenecks.

## Limitations & Future Work
- RefAVA is based on the AVA dataset; hence, its coverage of 80 atomic action classes remains limited.
- Textual descriptions do not include action descriptions (only describing appearance and position); practical scenarios might require more flexible referring patterns.
- The volume of 214M trainable parameters is relatively large, and the inference efficiency is not discussed.
- Object detection is performed only on keyframes, lacking object tracking across the temporal dimension.

## Related Work & Insights
- **vs BLIPv2**: As the strongest VTR baseline, BLIPv2 achieves a mAP of 53.19 on RAVAR. RefAtomNet's score of 57.52 demonstrates that the specifically designed location-semantic stream and cross-stream fusion are crucial for fine-grained action recognition.
- **vs RVOS (Referring Video Object Segmentation)**: In RVOS, the input referring expression contains action terms. In contrast, RAVAR's text only describes appearance/location without actions, representing a different task formulation.

## Rating
- Novelty: ⭐⭐⭐⭐ New task + new dataset + new method, with a highly practical task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 baseline methods covering 5 domains (AAL/VQA/VTR/SF/VOS).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive task comparison diagrams.
- Value: ⭐⭐⭐⭐ Fills the gap in referring atomic action recognition, with the dataset and benchmark poised to drive subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)
- [\[ECCV 2024\] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](on_the_utility_of_3d_hand_poses_for_action_recognition.md)

</div>

<!-- RELATED:END -->
