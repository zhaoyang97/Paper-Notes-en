---
title: >-
  [Paper Note] SummDiff: Generative Modeling of Video Summarization with Diffusion
description: >-
  [ICCV 2025][Image Generation][Video Summarization] SummDiff is the first work to introduce diffusion models into video summarization, formulating the task as a conditional generation problem. By learning the distribution…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Video Summarization"
  - "Diffusion Models"
  - "Conditional Generation"
  - "Subjectivity Modeling"
  - "Knapsack Problem"
date: 2026-05-08
content_hash: 2bb9d0dfa08284c3
---

# SummDiff: Generative Modeling of Video Summarization with Diffusion

**Conference**: ICCV 2025
**arXiv**: [2510.08458](https://arxiv.org/abs/2510.08458)  
**Area**: Image Generation
**Keywords**: Video Summarization, Diffusion Models, Conditional Generation, Subjectivity Modeling, Knapsack Problem

## TL;DR

SummDiff is the first work to introduce diffusion models into video summarization, formulating the task as a conditional generation problem. By learning the distribution of "good summaries," the model generates diverse plausible summaries that better reflect the inherent subjectivity of the video summarization task.

## Background & Motivation

Video summarization aims to select keyframes from long videos to preserve core content. However, the task is inherently **highly subjective**: different annotators often disagree on what constitutes a "good summary."

Key limitations of existing methods include:

**Ignoring annotation diversity**: Most methods average frame-level importance scores from multiple annotators as the training target. This regression-based approach discards the individual perspectives of different annotators. For example, if half the annotators select clips from the first quarter of a video and the other half select clips from the last quarter, simple averaging assigns similar intermediate scores to both segments, failing to distinguish between two equally valid summarization strategies.

**Deterministic output**: Given a video, only a single summary can be generated, failing to reflect the plurality of reasonable summarization possibilities.

**Imperfect evaluation metrics**: Insufficient analysis of the knapsack step, and F1 scores that are overly sensitive to video segmentation.

The authors propose a **generative perspective**: treating video summarization as a conditional generation task, where the model learns a probability distribution over good summaries and generates diverse plausible summaries through sampling.

## Method

### Overall Architecture

SummDiff consists of three stages: (1) video encoding; (2) diffusion-based importance score denoising; and (3) knapsack optimization to produce the final summary.

### 1. Video Encoding

A pretrained image encoder extracts per-frame features $\mathbf{z}_i \in \mathbb{R}^D$, which are then contextualized via self-attention to yield the global visual feature matrix $\mathbf{Z} \in \mathbb{R}^{N \times D}$.

### 2. Video Importance Score Denoiser (Core)

**Forward process**: For a single annotator's importance scores $\mathbf{s}_0 \in [0,1]^N$, a logit transformation is first applied: $\mathbf{u}_0 = \log \frac{\mathbf{s}_0}{1-\mathbf{s}_0}$. Gaussian noise is then added in logit space:

$$\mathbf{s}_t = \sqrt{\bar{\alpha}_t} \mathbf{s}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_t$$

**Codebook quantization**: The noised logits $\mathbf{u}_t$ are mapped back to $[0,1]$ via sigmoid, then uniformly divided into $K$ bins, each corresponding to a learnable $D$-dimensional embedding. This maps the scalar scores $\mathbf{u}_t \in \mathbb{R}^N$ to $\mathcal{C}(\mathbf{u}_t) \in \mathbb{R}^{N \times D}$, enabling compatibility with cross-attention.

**Transformer cross-attention denoising**: The quantized embeddings $\mathcal{C}(\mathbf{u}_t)$ serve as queries, and the visual features $\mathbf{Z}$ serve as keys/values. AdaLN-Zero is employed to inject timestep and positional conditioning, avoiding information aliasing:

$$\mathbf{X}_1 = \mathbf{A}_1 \odot \text{softmax}(\mathbf{Q}_t' \mathbf{K}'^{\top}) \mathbf{V}' + \mathbf{Q}_t'$$

### 3. Training and Inference

**Training loss**: MSE is computed against each individual annotator's scores:

$$\mathcal{L}(\mathbf{s}_0, \hat{\mathbf{s}}_0) = \|\mathbf{s}_0 - \sigma(\text{FC}(f_\theta(\mathcal{C}(\mathbf{u}_t), t, \mathbf{Z})))\|_2^2$$

**Inference**: Starting from random noise $\mathbf{u}_T \sim \mathcal{N}(0, \mathbf{I})$, the model performs iterative denoising via DDIM reverse process, followed by a sigmoid to obtain $\hat{\mathbf{s}}_0 \in (0,1)^N$.

### 4. Summary Generation (KTS + Knapsack)

KTS is used to segment the video into semantic shots. Shot-level importance is computed as the mean frame score, and dynamic programming solves the knapsack problem under a budget constraint (e.g., $\rho=0.15$) to select the optimal subset of shots.

## Key Experimental Results

### Main Results

| Method | SumMe τ | SumMe ρ | TVSum τ | TVSum ρ |
|--------|---------|---------|---------|---------|
| Random | 0.000 | 0.000 | 0.000 | 0.000 |
| Human | 0.205 | 0.213 | 0.177 | 0.204 |
| CSTA | 0.108 | 0.120 | 0.168 | 0.221 |
| **SummDiff** | **0.133** | **0.148** | **0.173** | **0.226** |

Under the TVT setting, SummDiff achieves a 23% improvement in τ on SumMe and outperforms all baselines on TVSum.

### Large-Scale Evaluation on Mr. HiSum

| Method | τ | ρ | MAP@50% | MAP@15% |
|--------|---|---|---------|---------|
| CSTA | 0.128 | 0.185 | 63.38 | 30.42 |
| **SummDiff** | **0.175** | **0.238** | **65.44** | **33.83** |

On the large-scale Mr. HiSum dataset (31,892 videos), SummDiff substantially outperforms the strongest baseline CSTA across all metrics, demonstrating strong scalability.

### Ablation Study

| Configuration | SumMe τ | TVSum τ |
|---------------|---------|---------|
| w/o codebook (scalar input) | 0.105 | 0.152 |
| Simple additive conditioning | 0.118 | 0.161 |
| **AdaLN-Zero + codebook** | **0.133** | **0.173** |

Both codebook quantization and AdaLN-Zero conditioning injection are critical to performance gains.

## Highlights & Insights

1. **First application of diffusion models to video summarization**, transforming a deterministic regression problem into a conditional generation problem that naturally accommodates task subjectivity.
2. **Training on individual annotator scores** rather than averaged scores effectively preserves diverse summarization perspectives.
3. **Codebook quantization** elegantly resolves the dimensionality mismatch between scalar importance scores and high-dimensional cross-attention.
4. **New evaluation metrics** are proposed, offering deeper assessment through knapsack-level analysis.

## Limitations & Future Work

- Evaluation reliability on conventional small-scale datasets (SumMe/TVSum) is limited, with only 25 and 50 videos respectively.
- Training requires per-annotator individual scores; on large-scale datasets such as Mr. HiSum, only aggregated annotations are available, limiting the full advantage of the generative formulation.
- The logit-space transformation requires clipping, which may introduce numerical bias.

## Related Work & Insights

- Traditional video summarization: VASNet, PGL-SUM, CSTA, and others predict averaged scores via regression.
- Generative approaches: GAN-based summarization (SUM-GAN) uses adversarial loss but targets a different objective.
- Diffusion models: This work represents the first application to video summarization, drawing on the conditional generation mechanism of DiT.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](../../NeurIPS2025/image_generation/diffusion_generative_modeling_on_lie_group_representations.md)

</div>

<!-- RELATED:END -->
