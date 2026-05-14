---
title: >-
  [Paper Note] What If: Understanding Motion Through Sparse Interactions
description: >-
  [ICCV 2025][Segmentation][motion understanding] This paper proposes the Flow Poke Transformer (FPT), which directly predicts **multimodal probability distributions** over object motion in a scene (rather than a single de…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "motion understanding"
  - "optical flow distribution prediction"
  - "sparse interaction"
  - "moving part segmentation"
  - "Transformer"
date: 2026-05-08
content_hash: 49ab3f4b81ddd5a0
---

# What If: Understanding Motion Through Sparse Interactions

**Conference**: ICCV 2025
**arXiv**: [2510.12777](https://arxiv.org/abs/2510.12777)
**Code**: [compvis.github.io/flow-poke-transformer](https://compvis.github.io/flow-poke-transformer)
**Area**: Image Segmentation
**Keywords**: motion understanding, optical flow distribution prediction, sparse interaction, moving part segmentation, Transformer

## TL;DR

This paper proposes the Flow Poke Transformer (FPT), which directly predicts **multimodal probability distributions** over object motion in a scene (rather than a single deterministic outcome), conditioned on sparse "poke" interactions, enabling interpretable motion understanding and moving part segmentation.

## Background & Motivation

The central challenge in understanding scene dynamics is that **real-world motion is inherently uncertain and multimodal**. Human visual intelligence does not predict a single deterministic future; rather, it infers the multiple ways in which objects might move.

Limitations of existing motion prediction methods:

**Dense video prediction** (e.g., diffusion-based video generation) must commit to *one* trajectory, ignoring the multimodality of motion — even if the generated frames are photorealistic, this does not imply an understanding of the underlying physical process.

**Deterministic optical flow prediction** (e.g., RAFT) estimates motion between two given frames, requiring future frames as input and thus incapable of forecasting future motion.

**Existing conditional motion generation** (e.g., DragAPart, PuppetMaster) synthesizes results directly in RGB space, making the underlying motion representation inaccessible and providing no uncertainty estimates.

Core motivation: to design a model that **directly outputs probability distributions over motion** rather than merely sampling from them. This enables:
- Direct quantification of uncertainty
- Identification of multimodal motion
- Exploration of physical interactions in a scene via sparse pokes

## Method

### Overall Architecture

Given an image $\mathcal{I}$, the model learns the conditional distribution $p(\mathbf{f}(\mathbf{q})|\mathcal{P}, \mathcal{I})$: the distribution of motion $\mathbf{f}(\mathbf{q}) \in \mathbb{R}^2$ at query point $\mathbf{q}$, conditioned on a set of pokes $\mathcal{P} = \{(\mathbf{p}_i, \mathbf{f}(\mathbf{p}_i))\}_{i=1}^{N_p}$ and the image. The architecture consists of an **image encoder** (ViT-Base initialized with DINOv2-R) and the **Flow Poke Transformer** (ViT-Base), totaling 220M parameters.

### Key Designs

1. **Sparse Kinematic Modeling**: Each poke $(\mathbf{p}_i, \mathbf{f}(\mathbf{p}_i))$ and query point $\mathbf{q}_j$ are represented as independent tokens. Poke motion is encoded via Fourier embeddings, and positions are represented using RoPE relative positional encodings to support **arbitrary-precision, off-grid locations**. Query tokens attend only to themselves and the pokes (not to other queries), enabling **efficient parallel prediction** over multiple queries.

2. **Gaussian Mixture Model (GMM) Output Distribution**: The projection head at the Transformer output directly predicts an $N$-component GMM:
$$p_\theta = \sum_{n=1}^N \pi^{(n)} \cdot \mathcal{N}(\boldsymbol{\mu}^{(n)}, \boldsymbol{\Sigma}^{(n)})$$
A key improvement over GIVT is the use of **full covariance matrices** $\boldsymbol{\Sigma}^{(n)} \in \mathbb{R}^{2 \times 2}$ (guaranteed positive definite by predicting the lower-triangular Cholesky factor $\mathbf{L}^{(n)}$) rather than diagonal covariances, substantially increasing modeling expressiveness.

3. **Query-Causal Attention**: During training, a causal attention mask is applied over pokes, and each query attends only to its corresponding poke subset. This reduces computational complexity from $\mathcal{O}(N_p^2 \cdot N_q^2)$ to $\mathcal{O}(N_p^2 + N_p \cdot N_q)$, enabling efficient training.

4. **Camera Motion Adaptation**: Adaptive normalization layers (AdaIN) are used to condition the model on whether the camera is static, preventing camera motion from dominating the learned motion distribution.

5. **Moving Part Segmentation**: The KL divergence is used to measure the influence of a poke on the motion distribution at a query point:
$$D_{KL}(p_\theta(\mathbf{f}(\mathbf{q})|(\mathbf{p}, \mathbf{f}(\mathbf{p})), \mathcal{I}) \parallel p_\theta(\mathbf{f}(\mathbf{q})|\mathcal{I}))$$
A KL divergence of zero indicates motion independence, while a nonzero value indicates influence from the poke — directly quantifying part-level motion correlation.

### Loss & Training

The model is trained by minimizing the **negative log-likelihood** of ground-truth optical flow:
$$\mathcal{L} = -\log p_\theta(\mathbf{f}(\mathbf{q})|\mathcal{P}, \mathcal{I}) = -\log\left(\sum_{n=1}^N \pi^{(n)} \mathcal{N}(\mathbf{f}(\mathbf{q})|\boldsymbol{\mu}^{(n)}_\theta, \boldsymbol{\Sigma}^{(n)}_\theta)\right)$$

Training details:
- Dataset: WebVid 3.8M subset (general-purpose pretraining); an alternative variant uses 5M open-domain videos
- Ground-truth optical flow: dense tracking on a $48^2$ grid using CoTracker3 / TAPNext
- Optimizer: AdamW, lr=5e-5, batch size 32→128, 800k steps
- Per image: 0–128 pokes and 15 random query points sampled
- Training time: 7 days on 2×H200 (or 24h on 8×H200 with an optimized configuration)

## Key Experimental Results

### Main Results: Talking Face Motion Generation (TalkingHead-1KH)

| Method | Training Data | 1 Poke EPE↓ | 10 Pokes EPE↓ | 100 Pokes EPE↓ |
|--------|--------------|-------------|--------------|----------------|
| InstantDrag | Face-specific | 9.24 | 8.39 | 7.29 |
| Motion-I2V | General (Zero-Shot) | 29.08 | 20.90 | n/a |
| **FPT (Ours)** | **General (Zero-Shot)** | **7.64** | **4.20** | **2.51** |

The zero-shot general-purpose model surpasses the face-specific model InstantDrag, with performance gains increasing markedly as the number of pokes grows.

### Articulated Object Motion Estimation (Drag-A-Move)

| Method | Training Set | EPE↓ | PCK↑ | Seg. mIoU↑ |
|--------|-------------|------|------|-----------|
| DragAPart | DAM (specialized) | 9.69 | 0.514 | 0.273 |
| PuppetMaster | DAM+OAHQ (specialized) | 9.62 | 0.472 | 0.112 |
| FPT (zero-shot) | General | 12.74 | 0.191 | 0.287 |
| **FPT (fine-tuned)** | **General→DAM** | **3.57** | **0.834** | **0.572** |

After fine-tuning, EPE is **reduced by 63%**, and moving part segmentation mIoU reaches 0.572 (vs. 0.273 for DragAPart).

### Ablation Study

| Predictive Uncertainty Calibration | Description |
|-----------------------------------|-------------|
| Pearson $\rho$ = 0.66 (sampled) | Predicted uncertainty strongly correlates with actual error |
| Pearson $\rho$ = 0.64 (mean) | Mean prediction is equally accurate |
| Pearson $\rho$ = 0.62 (highest-confidence mode) | Higher confidence correlates with higher accuracy |

| Multimodal Analysis | Description |
|--------------------|-------------|
| High mode diversity | Mode variation covers a large portion of the poke magnitude |
| Mode closest to GT has above-average confidence | Confidence predictions are semantically meaningful |
| More pokes → more unimodal | Distribution naturally converges as conditioning information increases |

### Key Findings

- The **uncertainty predicted by FPT correlates strongly with actual error** (Pearson $\rho > 0.6$), validating the reliability of probabilistic distribution prediction
- **General-purpose pretraining generalizes effectively**: zero-shot performance on talking-face motion surpasses specialized models
- Moving part segmentation **requires no dedicated training** and emerges naturally from distribution comparison
- Single-H200 inference latency is <25ms with throughput >160k predictions/s, suitable for real-time applications

## Highlights & Insights

- **Direct access to the probability distribution**: Unlike diffusion or GAN models that only support sampling, FPT's GMM output allows direct reading of probability densities, computation of KL divergences, and mode identification
- **Sparse representation → efficiency**: Sparse token modeling avoids the computational overhead of dense prediction while preserving rich motion semantics
- **Moving part segmentation emerges naturally from motion understanding**: No additional annotations or modules are required; part-level correlation is quantified solely via KL divergence — an elegant conceptual contribution
- The query-causal attention design substantially reduces training cost and is transferable to other sparse conditional prediction tasks

## Limitations & Future Work

- Generalization to cartoon or animated imagery is limited (training data consists primarily of real-world video)
- The model occasionally erroneously couples object shadows with object motion
- The current formulation models only 2D motion distributions; extending to 3D motion is a natural future direction
- Autoregressive dense sampling can generate globally consistent motion but is comparatively slow

## Related Work & Insights

- The GMM output head design builds upon GIVT but extends it to full covariance matrices, enhancing modeling capacity
- The poke concept originates from iPoke (Blattmann et al. 2021) but is elevated from RGB synthesis to probabilistic distribution modeling
- RoPE positional encodings enable the model to support arbitrary-precision off-grid positions, contributing to strong generalization

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Direct prediction of motion probability distributions + emergent moving part segmentation
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-domain evaluation + uncertainty analysis + segmentation
- Value: ⭐⭐⭐⭐ — Real-time inference + general-purpose pretraining + multiple downstream tasks
- Overall: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
