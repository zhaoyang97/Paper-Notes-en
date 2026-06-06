---
title: >-
  [Paper Note] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance
description: >-
  [ICCV 2025][Medical Imaging][Interactive Segmentation] This paper proposes MultiverSeg, a progressive interactive segmentation system in which each image annotated by the user reduces the number of interactions required…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Interactive Segmentation"
  - "In-Context Learning"
  - "Biomedical Imaging"
  - "Dataset Annotation"
  - "Progressive Segmentation"
date: 2026-05-08
content_hash: 3b9e1f0077c3f29a
---

# MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance

**Conference**: ICCV 2025
**arXiv**: [2412.15058](https://arxiv.org/abs/2412.15058)  
**Code**: [multiverseg.csail.mit.edu](https://multiverseg.csail.mit.edu)  
**Area**: Medical Imaging
**Keywords**: Interactive Segmentation, In-Context Learning, Biomedical Imaging, Dataset Annotation, Progressive Segmentation

## TL;DR

This paper proposes MultiverSeg, a progressive interactive segmentation system in which each image annotated by the user reduces the number of interactions required for subsequent images. By incorporating previously segmented images as in-context inputs, the system improves with use. On 12 unseen datasets, it reduces click counts by 36% and scribble steps by 25% compared to ScribblePrompt.

## Background & Motivation

### Problem Definition

Biomedical researchers and clinicians frequently need to perform new segmentation tasks on sets of related images. Existing methods are either interactive (requiring substantial human effort per image) or depend on pre-existing annotated datasets. For a new dataset of tens to hundreds of images, an ideal system should "learn" from annotation experience and progressively reduce the required human interaction as more images are labeled.

### Limitations of Prior Work

**Interactive segmentation methods** (SAM, ScribblePrompt, etc.): Although generalizable to new tasks, these methods annotate each image independently and cannot leverage completed annotations to accelerate subsequent segmentation. The total annotation effort scales linearly with the number of images.

**In-context learning methods** (UniverSeg, etc.): These can exploit labeled context sets for inference but typically require large context sets to achieve good performance and lack a correction mechanism — if a prediction is inaccurate, the user has no way to interactively fix it.

**Continual learning / fine-tuning methods** (MonaiLabel, nnUNet, etc.): These require manually annotating a substantial training set before training an automatic segmentation model, demanding machine learning expertise and computational resources that are not available to typical biomedical researchers.

### Core Motivation

**Key insight**: When annotating a new dataset, completed segmentations should help with subsequent images. Combining interactive segmentation with in-context learning enables *progressive segmentation* — the first image is segmented purely through interaction, and with each subsequent annotated image, the interaction required for the next decreases, potentially reaching zero. This decouples the total annotation effort from linear growth with dataset size.

## Method

### Overall Architecture

MultiverSeg is an interactive in-context segmentation framework with the following core workflow:

1. The user segments the first image interactively (clicks, bounding boxes, scribbles).
2. The completed image–segmentation pair is added to the context set.
3. For each subsequent image, the model jointly uses user interactions and the context set to produce predictions.
4. As the context set grows, the number of interactions required per new image decreases.

### Key Designs

#### 1. **Interactive In-Context Segmentation Network**

- **Function**: Accepts a target image, user interactions (clicks / bounding boxes / scribbles), and a variable-size context set (previously segmented image–mask pairs), and outputs a segmentation prediction.
- **Mechanism**:

  The target image input $q_i$ contains 5 channels: the target image $x_i$, an interaction mask $u_{i,j}$ (one channel each for positive/negative scribbles and clicks), and the previous-step prediction $\hat{y}_{i,j-1}$. Each entry in the context set $\{(x_l, y_l)\}_{l=1}^{m}$ consists of a stacked image–segmentation pair.

  The network follows a UNet encoder–decoder architecture. At each scale, a CrossBlock mechanism fuses target image features with context set features:

  $$z_i = \text{LN}(A(\text{Conv}(q \| v_i; \theta_z)))$$
  $$q' = \text{LN}(A(\text{Conv}(\frac{1}{n}\sum_{i=1}^{n} z_i; \theta_q)))$$
  $$v_i' = \text{LN}(A(\text{Conv}(z_i; \theta_v)))$$

  where $q$ denotes target features and $V = \{v_i\}$ denotes context features. CrossBlock is applied at every resolution level, enabling information exchange across all scales.

- **Design Motivation**: CrossBlock allows the target image to retrieve relevant information from the context set at each scale, while context features adapt to the current target. The addition of LayerNorm improves training stability compared to the original CrossBlock in UniverSeg.

#### 2. **Progressive Dataset Segmentation Protocol**

- **Function**: Defines how to segment a new dataset from scratch in practice.
- **Mechanism**:

  For the first image (empty context set), a pretrained ScribblePrompt-UNet $g_\phi$ is used for purely interactive segmentation. For subsequent images $i > 0$, the context-aware model $f_\theta$ is used:

  $$\hat{y}_{i,j} = f_\theta(x_i^t, u_{i,j}^t, \hat{y}_{i,j-1}^t; S_i^t)$$

  where $S_i^t = \{(x_l^t, \hat{y}_{l,k_l}^t)\}_{l=0}^{i-1}$ is the context set formed by all previously segmented images.

  The optimization objective is to maximize segmentation quality across the entire dataset while minimizing total interaction:

  $$\min \sum_{i=1}^{N} \mathcal{L}_{seg}(y_i^t, \hat{y}_{i,k_i}^t), \quad \text{while minimizing} \sum_{i=1}^{N} \sum_{j=1}^{k_i} u_{i,j}^t$$

- **Design Motivation**: Framing dataset segmentation as a global optimization problem rather than independent per-image processing allows the annotation effort on earlier images to be amortized over all subsequent images.

#### 3. **Multi-Task Training and Data Augmentation Strategy**

- **Function**: Trains a unified model on 79 diverse biomedical datasets.
- **Mechanism**:

  Training simulates the full interactive segmentation workflow. At each step, a task $t$, a target sample $(x_i^t, y_i^t)$, and a context set $S_i^t$ of random size $n \sim U[0, 64]$ are sampled. The loss sums over $k$ iterative predictions:

  $$\mathcal{L}(\theta; \mathcal{T}) = \mathbb{E}_{t \in \mathcal{T}} \left[ \mathbb{E}_{(x_i^t, y_i^t; S^t) \in t} \left[ \sum_{j=1}^{k} \mathcal{L}_{seg}(y_i^t, \hat{y}_{i,j}^t) \right] \right]$$

  Interaction simulation strategy: the first step randomly combines bounding boxes, clicks, and scribbles (1–3 positive interactions, 0–3 negative interactions); subsequent steps sample corrective interactions from the error region $\varepsilon_{i-1}^t$.

  Data augmentation operates at two levels: **task augmentation** (applying the same transformation to both target and context, thereby changing the segmentation task itself) and **sample augmentation** (applying independent augmentations to target and context separately, increasing within-task variability).

  Synthetic data generation (probability $p_{synth}=0.5$): synthetic labels are generated from a single image using a superpixel algorithm, and a set of synthetic "tasks" is constructed via copying and strong augmentation.

- **Design Motivation**: Training with variable-size context sets enables the model to handle any scenario from 0 to 64 context samples. Synthetic task generation substantially expands the diversity of training tasks, helping the model learn to perform in-context reasoning across diverse structures.

### Loss & Training

- **Segmentation loss**: Soft Dice Loss + Focal Loss ($\gamma=20$), computed separately for each of $k=3$ iterative predictions
- **Optimizer**: Adam, learning rate $\eta = 10^{-4}$
- **Batch size**: 2; context set size $m \sim U[0, 64]$
- **Prediction post-processing**: At inference, predictions are thresholded to 0/1 before being added to the context set, improving the quality of subsequent predictions

## Key Experimental Results

### Main Results

Evaluated on 12 unseen datasets (covering 8 modalities, 187 segmentation tasks), with a target Dice of 90% (fully supervised nnUNet achieves an average Dice of 88.67% on the same data).

**Total interactions per image (Center Clicks) required to reach 90% Dice**:

| Method | Total Clicks/Image | Reduction | Uses Context |
|---|---|---|---|
| SAM | ~8.5 | — | No |
| MedSAM | avg. Dice 65.93% | Cannot reach target | No |
| ScribblePrompt | ~5.2 | Baseline | No |
| SP+UVS | ~4.5 | ↓13.5% | Yes (UniverSeg) |
| **MultiverSeg** | **~3.3** | **↓36.4%** | **Yes** |

**Total steps per image (Centerline Scribbles) required to reach 90% Dice**:

| Method | Scribble Steps/Image | Reduction |
|---|---|---|
| ScribblePrompt | ~3.8 | Baseline |
| SP+UVS | ~3.2 | ↓15.8% |
| **MultiverSeg** | **~2.8** | **↓25.3%** |

### Ablation Study

**Effect of context set size on segmentation quality** (no interaction; pure context inference using GT context labels):

| Context Set Size | MultiverSeg Dice | UniverSeg Dice |
|---|---|---|
| 1 | ~62% | ~45% |
| 4 | ~72% | ~60% |
| 16 | ~78% | ~68% |
| 64 | ~80% | ~72% |

**Synergistic effect of interaction and context** (Dice after interaction steps at varying context set sizes):

| Context Set Size | 0 Steps | 1 Click | 3 Clicks | 5 Clicks |
|---|---|---|---|---|
| 0 | ~0% | ~65% | ~80% | ~88% |
| 4 | ~72% | ~80% | ~88% | ~92% |
| 16 | ~78% | ~85% | ~90% | ~93% |
| 64 | ~80% | ~86% | ~91% | ~93% |

### Key Findings

1. **Interaction demand decreases as the dataset grows**: The second image requires ~5 clicks to reach 90% Dice, while the 18th requires only ~1, approaching automatic segmentation.
2. **Two information sources are complementary**: Context provides task priors; interactions provide precise corrections. Their combined effect far exceeds either alone.
3. **No retraining at inference time**: Inference takes <0.15 seconds (with 64-sample context), far faster than fine-tuning approaches (~20 minutes on an A100 per task).
4. **Robustness to prediction quality**: Even when predicted (rather than GT) segmentations are used in the context set, interaction requirements are substantially reduced.
5. **Larger datasets yield greater gains**: As more images are annotated, the marginal interaction cost per image approaches zero.

## Highlights & Insights

1. **Shift in problem framing**: The work reframes segmentation from "independent per-image annotation" to "dataset-level progressive annotation," introducing an amortized cost analysis perspective on annotation effort.
2. **Unified framework for in-context learning and interaction**: This is the first approach to integrate both paradigms into a single model, elegantly handling the full range from 0 to 64 context samples through variable-size context set training.
3. **High practical utility**: No machine learning expertise, GPU training, or pre-existing labeled data is required, making the system directly accessible to clinical researchers.
4. **Ingenuity of the training strategy**: Synthetic task generation (constructing "tasks" from a single image) substantially expands training task diversity.

## Limitations & Future Work

1. **Performance degrades with high intra-dataset diversity**: When image composition varies substantially within a dataset (e.g., BUID breast ultrasound), small context sets may underperform purely interactive methods.
2. **2D segmentation only**: 3D modalities must be processed as 2D slices, discarding volumetric context information.
3. **Context set ordering not optimized**: Images are currently added in random order; actively selecting the most informative images could further reduce total interaction.
4. **Binary segmentation constraint**: Only one foreground class is handled per run; multi-label simultaneous segmentation requires multiple passes.

## Related Work & Insights

- **Relationship to UniverSeg**: MultiverSeg can be viewed as an "interactively enhanced" UniverSeg, with improvements in architecture (larger CrossBlock feature dimensions: 256 vs. 64, plus LayerNorm) and training data (67 vs. 53 datasets).
- **Relationship to ScribblePrompt**: The first image is segmented directly using ScribblePrompt; in-context capability is layered on top for all subsequent images.
- **Insight**: Annotation tools could record user annotation history and feed it as context input for the next image, enabling an annotation experience that becomes progressively faster with use.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The fusion of interactive segmentation and in-context learning is conceptually clean yet previously unachieved; the CrossBlock design and training strategy are technically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 unseen datasets, 187 tasks, multiple interaction modalities, comprehensive ablations and comparisons, including an upper-bound analysis (nnUNet).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem definition is clear, figures are intuitive, and experimental design is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses a pain point for biomedical researchers; the tool is open-source and deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](../../NeurIPS2025/medical_imaging/pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[ICCV 2025\] RadGPT: Constructing 3D Image-Text Tumor Datasets](radgpt_constructing_3d_image-text_tumor_datasets.md)
- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/medical_imaging/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[NeurIPS 2025\] Orochi: Versatile Biomedical Image Processor](../../NeurIPS2025/medical_imaging/orochi_versatile_biomedical_image_processor.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](../../ICML2026/medical_imaging/factored_classifier-free_guidance.md)

</div>

<!-- RELATED:END -->
