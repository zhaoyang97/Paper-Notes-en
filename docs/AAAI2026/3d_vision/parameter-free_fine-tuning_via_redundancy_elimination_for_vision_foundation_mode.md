---
title: >-
  [Paper Note] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models
description: >-
  [AAAI 2026][3D Vision][Vision Foundation Models] This work identifies a significant number of redundant channels in vision foundation models (such as SAM, SAM2, and DINOv2) and proposes a parameter-free fine-tuning method. By employing an output-difference-based channel selection algorithm to locate optimal replacement pairs, redundant channels are replaced with active ones to enhance feature representations for downstream tasks, achieving an average mIoU improvement of 5 to…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Vision Foundation Models"
  - "Parameter-Free Fine-Tuning"
  - "Channel Redundancy"
  - "SAM"
  - "Feature Selection"
date: 2026-05-08
content_hash: 3cd53532546eb9a6
---

# Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models

**Conference**: AAAI 2026  
**arXiv**: [2504.08915](https://arxiv.org/abs/2504.08915)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Vision Foundation Models, Parameter-Free Fine-Tuning, Channel Redundancy, SAM, Feature Selection

## TL;DR

This work identifies a significant number of redundant channels in vision foundation models (such as SAM, SAM2, and DINOv2) and proposes a parameter-free fine-tuning method. By employing an output-difference-based channel selection algorithm to locate optimal replacement pairs, redundant channels are replaced with active ones to enhance feature representations for downstream tasks, achieving an average mIoU improvement of 5 to 11 points.

## Background & Motivation

Vision Foundation Models (VFMs) such as SAM and DINOv2 exhibit powerful general visual representation capabilities after being trained on large-scale datasets. Adapting them to downstream tasks typically requires parameter fine-tuning:

- **Full-parameter Fine-tuning**: Updates all parameters, which incurs high computational costs.
- **Parameter-Efficient Fine-Tuning** (PEFT/LoRA/Adapter): Updates a small number of parameters (thousands to millions), but still requires backpropagation and computation graph maintenance.

**Key Observation** (Control experiments in Table 1): On SAM's PerSeg dataset, setting the activation values of certain channels to 0 yielded the following:
- Setting Channel 6 to 0: mIoU remains unchanged (50.6 → 50.6), indicating this channel is redundant.
- Setting Channel 216 to 0: mIoU actually increases (50.6 → 52.7), indicating this channel is even harmful.
- Setting Channel 175/19/189 to 0: mIoU decreases, indicating these channels are beneficial for the task.

**Root Cause Analysis**: Among the general features learned by VFMs on large-scale datasets, many are irrelevant or even detrimental to specific downstream tasks. This redundancy occurs because the model needs to generalize across a wide variety of tasks.

**Core Problem**: Is it possible to adapt to downstream tasks without modifying any model parameters, solely by selecting, reusing, and enhancing existing features?

## Method

### Overall Architecture

In sharp contrast to traditional fine-tuning methods:
- **(a) Traditional Decoder Fine-tuning**: Updates decoder parameters to adapt pre-trained features to the task.
- **(b) Traditional Encoder Fine-tuning**: Updates encoder parameters to modify pre-trained features.
- **(c) Ours**: Updates zero parameters, only replacing redundant channels with more effective ones.

Workflow: Search dataset → Encoder extracts features → Pairwise channel replacement → Compare output differences → Construct dictionary → Search for optimal combination → Apply replacement

### Key Designs

1. **Problem Formulation**

   Goal: Find the optimal set of replacement pairs $P^*$ that maximizes performance on the downstream dataset $S$:
   $$P^* = \arg\max_P \text{mIoU}(S, P)$$
   where $P = \{(i,j)_1, (i,j)_2, ..., (i,j)_k\}$, and $(i,j)$ denotes replacing channel $i$ with channel $j$.

   Directly enumerating all combinations is intractable: it would require $2^{C^2}$ inferences when $C=256$.

2. **Channel Selection Algorithm**

   **Three Strategies to Reduce Search Overhead**:

   **(1) Search Based on Output Difference**:
   Given the search dataset $\mathbf{S}$, the encoder outputs features $X \in \mathbb{R}^{D \times C \times W \times H}$. For each replacement pair $(i,j)$, compute:
   $$\Delta\text{Acc}_{(i \to j)} = D(X') - D(X)$$
   where $D(X)$ and $D(X')$ denote the outputs of the decoder with the original and replaced features, respectively.

   Construct a dictionary $\mathcal{D} = \{(i,j): \Delta\text{Acc}_{(i \to j)}\}$, and select the top $N$ pairs to form $\mathcal{D}_{topN}$.

   Then, traverse all combinations in $\mathcal{D}_{topN}$ ($2^N - 1$ combinations) to find the optimal combination $P^*$.

   **Complexity Reduction**: Reduced from $2^{C^2}$ to $C^2 + 2^N - 1$ (which requires only $\sim$65,536 + 1,023 inferences when $N=10$).

   **(2) Sample Reduction**: Uses only 50 images as the search dataset.

   **(3) Feature Caching**: Pre-stores encoder features, modifying only the cached features and feeding them to the decoder during each inference, thereby preventing redundant encoding.

   **Design Motivation**: The output difference of a single-pair replacement predicts its contribution within a combination. This select-then-combine strategy significantly reduces computational overhead while maintaining search effectiveness. Since it only requires inference and no backpropagation, the GPU memory overhead is minimal.

3. **Implementation of Channel Replacement**

   Given a replacement pair $(i,j)$, the feature transformation is formulated as:
   $$X'_{d,c,w,h} = X_{d, f_{i \to j}(c), w,h}$$
   where $f_{i \to j}(\cdot)$ is a mapping function that maps channel $i$ to channel $j$.

   This is not random shuffling; rather, it selectively replaces redundant channels with effective ones in a completely deterministic process.

### Loss & Training

In the search phase, the same Dice + CE loss as the baseline is utilized for output evaluation. Note: The search process only involves model inference, requiring **no gradient computation or backpropagation**.

Implementation Details:
- Search dataset: 50 randomly sampled images
- $N = 10$ (top-N replacement pairs)
- Baseline fine-tuning comparison experiments use 25 epochs, Adam optimizer, with an initial learning rate of $10^{-4}$

## Key Experimental Results

### Main Results

**Parameter-free Fine-tuning Performance across SAM Versions** (Average mIoU on 9 datasets):

| Model | Backbone | Params | Baseline Avg | +Ours Avg | Gain Δ |
|------|------|--------|----------|-----------|--------|
| SAM | ViT-B | 91M | 49.14 | 58.08 | **+8.94** |
| SAM | ViT-L | 308M | 56.15 | 67.61 | **+11.46** |
| SAM | ViT-H | 636M | 55.54 | 60.68 | +5.14 |
| SAM2 | Hiera-T | 39M | 57.29 | 65.63 | **+8.34** |
| SAM2 | Hiera-S | 46M | 61.04 | 68.69 | +7.65 |
| SAM2 | Hiera-B+ | 81M | 61.62 | 66.94 | +5.32 |
| SAM2 | Hiera-L | 224M | 67.77 | 73.53 | +5.76 |

Remarkable performance is achieved with a 5-11 point mIoU improvement without updating any parameters.

**Performance when Combined with Existing Fine-tuning Methods**:

| Fine-tuning Method | Baseline Avg | +Ours Avg | Extra Gain |
|----------|----------|-----------|----------|
| Decoder-only | 73.61 | 74.62 | +1.01 |
| SAMed (LoRA) | 78.56 | 79.72 | **+1.16** |
| SAM-COBOT | 78.73 | 79.32 | +0.59 |
| SAM-Adapter | 72.89 | 73.80 | +0.91 |
| SAM-PARSER | 60.96 | 65.39 | **+4.43** |
| DoRA | 79.12 | 79.92 | +0.80 |

This demonstrates that even after parameter fine-tuning, channel redundancy still exists in the model, and ours can serve as a plug-and-play module for further improvement.

### Ablation Study

**Computational Overhead Comparison**:

| Method | GPU Memory (GB) | Trainable Params (K) |
|------|--------------|---------------|
| Encoder-only | 34.6 | 89,670 |
| Decoder-only | 13.7 | 4,057 |
| MedSAM | 34.7 | 93,735 |
| SAMed (LoRA) | 28.9 | 147 |
| SAM-PARSER | 15.9 | 0.5 |
| **Ours** | **11.1** | **0** |

Ours achieves the lowest GPU memory consumption (11.1 GB vs. 13.7-34.7 GB for other methods) with zero parameters.

**Impact of the Number of Replacement Pairs**: Increasing the number of replacement pairs generally improves performance, peaking at 6 pairs on the COCO dataset.

**Generalization to Other Vision Tasks**:

| Model | Backbone | NYUv2 MSE↓ / AbsRel↓ / δ₁↑ | CIFAR Acc↑ |
|------|------|---------------------------|-----------|
| DINOv2 | ViT-S | 0.225/0.126/0.893 | 80.41 |
| +Ours | ViT-S | **0.209/0.112/0.907** | **80.81** |
| DINOv2 | ViT-B | 0.210/0.110/0.900 | 88.08 |
| +Ours | ViT-B | **0.193/0.095/0.916** | **88.49** |

It is equally effective for depth estimation and image classification.

### Key Findings

- Feature maps of effective channels exhibit clearer structures, edges, and textures, whereas redundant channels are blurry and noisy (see Figure 5 for visualization).
- Certain channels display cross-domain consistency: for example, Channel 19 is effective across natural, medical, and camouflaged scenarios, while Channels 20, 98, 162, and 226 are consistently redundant.
- Larger models (ViT-H, Hiera-L) exhibit slightly smaller improvements, likely due to their relatively lower redundancy.
- Improvements on in-distribution datasets (natural images) are greater than on out-of-distribution datasets (medical images), which aligns with SAM's training data distribution.

## Highlights & Insights

1. **Disruptive Paradigm Innovation**: This work is the first to demonstrate that VFMs can undergo completely parameter-free fine-tuning—requiring no gradients, no backpropagation, and no additional parameters. Downstream performance is significantly enhanced solely through "channel exchange."
2. **Extremely Low Computational Barrier**: Requiring only 11.1 GB of VRAM and model inference, the method can run on consumer-grade GPUs, substantially lowering the barrier for VFM adaptation.
3. **Orthogonal and Complementary to PEFT Methods**: Can be integrated as a plug-and-play post-processing step, yielding an additional 0.5-4.4 mIoU points on top of already fine-tuned models.
4. **Deep Insight into Channel Redundancy**: Reveals the pervasive phenomenon of feature redundancy in foundation models, offering a fresh perspective on understanding feature utilization efficiency in large models.
5. **Cross-Task Generalizability**: Extends from segmentation to depth estimation and classification, and from SAM to DINOv2, validating the universality of the proposed method.

## Limitations & Future Work

- The search process still requires traversing $C^2$ (~65,536) pairs and $2^N - 1$ combinations, which, although inference-only, still incurs time costs on large-scale datasets.
- The choice of search dataset may affect the optimal replacement pairs, as 50 images might be insufficiently representative of different datasets.
- The selection of $N=10$ is relatively fixed; adaptive methods for determining $N$ have not been explored.
- Channel replacement is a "hard replacement"; softer schemes for channel weight adjustment remain unexplored.
- The operation is limited to features in the final encoder layer; multi-layer channel replacement has not been investigated.
- Currently evaluated only on SAM/SAM2/DINOv2, leaving its generalizability to other VFMs like CLIP and MAE unknown.

## Related Work & Insights

- **SAM-PARSER (2024)**: Compresses trainable parameters to just 512. Ours goes a step further to zero parameters.
- **ShuffleNet**: Channel shuffling is used for cross-group information fusion during training, which is fundamentally different from the objective and method of ours.
- **Channel-Exchanging Network**: Involves channel exchange in multi-modal fusion, whereas ours performs redundancy elimination within a single modality.
- **Network Pruning**: Pruning removes redundancy but typically requires retraining, which is not required by ours.
- **Insight**: Feature redundancy in foundation models is a universal phenomenon. "Subtraction" (removing redundancy) can sometimes be more effective than "addition" (adding parameters). This approach could be generalized to the adaptation of large NLP models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The parameter-free fine-tuning paradigm is proposed for the first time in the VFM domain, representing a bold and effective approach.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (9 datasets $\times$ 7 backbones $\times$ 6 fine-tuning method combinations + depth/classification extensions.)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, comprehensive experimental design, and insightful visualization analysis.)
- Value: ⭐⭐⭐⭐⭐ (Highly practical, extremely low computational barrier, and orthogonally complementary to existing methods.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](../../NeurIPS2025/3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)
- [\[ECCV 2024\] Sapiens: Foundation for Human Vision Models](../../ECCV2024/3d_vision/sapiens_foundation_for_human_vision_models.md)

</div>

<!-- RELATED:END -->
