---
title: >-
  [Paper Note] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)
description: >-
  [ICCV 2025][Image Restoration][Generic Event Boundary Detection] DiffGEBD is the first work to introduce diffusion models into Generic Event Boundary Detection (GEBD). It frames boundary prediction as an iterative denoising process from random noise to a plausible boundary distribution, leverages Classifier-Free Guidance to control prediction diversity, and proposes two new evaluation metrics—Symmetric F1 and Diversity Score—to measure quality and diversity in multi-prediction scenarios.
tags:
  - ICCV 2025
  - Image Restoration
  - Generic Event Boundary Detection
  - Diffusion Models
  - Classifier-Free Guidance
  - Diversity Evaluation
  - Temporal Self-Similarity
date: 2026-05-08
content_hash: bcb8195f0c5cb2cd
---

# Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)

**Conference**: ICCV 2025
**arXiv**: [2508.12084](https://arxiv.org/abs/2508.12084)
**Code**: [https://cvlab.postech.ac.kr/research/DiffGEBD](https://cvlab.postech.ac.kr/research/DiffGEBD)
**Area**: Image Restoration
**Keywords**: Generic Event Boundary Detection, Diffusion Models, Classifier-Free Guidance, Diversity Evaluation, Temporal Self-Similarity

## TL;DR

DiffGEBD is the first work to introduce diffusion models into Generic Event Boundary Detection (GEBD). It frames boundary prediction as an iterative denoising process from random noise to a plausible boundary distribution, leverages Classifier-Free Guidance to control prediction diversity, and proposes two new evaluation metrics—Symmetric F1 and Diversity Score—to measure quality and diversity in multi-prediction scenarios.

## Background & Motivation

**State of the Field**: Generic Event Boundary Detection (GEBD) aims to segment videos into semantically meaningful event segments by identifying their turning points. Unlike conventional action recognition and temporal action detection, GEBD focuses on class-agnostic universal event boundaries. Kinetics-GEBD serves as the standard benchmark, providing annotations from multiple annotators per video.

**Limitations of Prior Work**:
- The definition of event boundaries in GEBD is inherently **subjective and diverse**—different annotators may perceive boundaries differently for the same video.
- Nevertheless, all existing methods (UBoCo, DDM-Net, LCVS, SC-Transformer, BasicGEBD, EfficientGEBD, etc.) adopt **deterministic models** that produce only a **single prediction** per video, ignoring boundary diversity.
- The conventional F1 metric evaluates alignment between a single prediction and multiple annotations, and is ill-suited for many-to-many alignment evaluation.

**Root Cause**: The ground truth in GEBD is inherently diverse (multiple annotators provide different answers), yet models are constrained to produce a single deterministic output. This asymmetry between diverse annotations and deterministic predictions prevents models from faithfully reflecting the variability in human judgment.

**Starting Point**: Diffusion models are naturally suited to this challenge—different outputs can be sampled simply by varying the initial noise. DiffGEBD reframes GEBD as a generative problem: given video-conditioned features, event boundaries are iteratively denoised from random noise.

## Method

### Overall Architecture

DiffGEBD consists of three core components:

- **Backbone** (ResNet-50): Extracts visual features $\bm{F} \in \mathbb{R}^{L \times D}$ from the input video $\bm{V} \in \mathbb{R}^{L \times H \times W \times 3}$.
- **Temporal Self-Similarity Encoder** (Encoder $f$): Captures dynamic changes between adjacent frames via a sliding-window temporal self-similarity module, producing temporal embeddings $\bm{E} \in \mathbb{R}^{L \times C}$.
- **Denoising Decoder** (Decoder $h$): A Transformer Encoder-based module that denoises the noisy boundary label $\bm{y}_t$ conditioned on the temporal embedding $\bm{E}$ to produce boundary predictions.

**Training**: At each step, a diffusion timestep $t$ is sampled, Gaussian noise is added to the ground-truth boundary label $\bm{y}_0 \in \{0,1\}^L$ to obtain $\bm{y}_t$, and the decoder learns to reconstruct $\bm{y}_0$ from $\bm{y}_t$. Annotations from different annotators are cycled across epochs.

**Inference**: Starting from random Gaussian noise $\hat{\bm{y}}_T$, DDIM sampling iteratively denoises to $\hat{\bm{y}}_0$. By initializing with different random noise, **a single model can generate multiple distinct plausible predictions**.

### Key Designs

1. **Temporal Self-Similarity Encoder**:

    - **Function**: Processes visual features using 1D convolution, sliding-window temporal self-similarity matrices, FCN, and 2D pooling.
    - **Mechanism**: The self-similarity matrix measures changes in semantic consistency across neighboring frames; boundary locations correspond to peaks of inconsistency.
    - **Design Motivation**: Temporal self-similarity features $\bm{E}$ better capture fine-grained inter-frame changes than raw backbone features $\bm{F}$. Ablation results (Table 2) confirm this: conditioning on $\bm{F}$ yields F1$_{\text{sym}}$ of 68.5, while conditioning on $\bm{E}$ yields 74.0.

2. **Classifier-Free Guidance (CFG)**:

    - **Training**: With probability $p=0.1$, the conditioning feature $\bm{E}$ is randomly replaced with a zero matrix, jointly training conditional and unconditional modes.
    - **Inference**: $\hat{\bm{y}}_t = (1+w)\hat{\bm{y}}_t^c - w\hat{\bm{y}}_t^u$, where $w$ is the guidance weight.
    - Large $w$ → more deterministic outputs strongly conditioned on visual features; small $w$ → more diverse predictions.
    - Optimal $w=0.6$ (maximizes F1$_{\text{sym}}$); $w=4.0$ is used for deterministic evaluation.

3. **Diversity-Aware Evaluation Metrics**:

    - **Symmetric F1 (F1$_{\text{sym}}$)**: Combines the Pred-to-GT alignment score (each prediction matched to its best GT) and the GT-to-Pred alignment score (each GT covered by some prediction), taking their harmonic mean.
    - **Diversity Score**: Computes the average pairwise dissimilarity among $N_P$ predictions: $\frac{1}{N_P^2}\sum_{i,j}(1 - \text{F1}(\hat{Y}_i, \hat{Y}_j))$.

### Loss & Training

- **Loss Function**: Mean squared error $\mathcal{L} = \frac{1}{L}\sum_{l=1}^{L}(\bm{y}_{0,l} - \hat{\bm{y}}_{t,l})^2$
- Videos are uniformly sampled to 100 frames.
- Backbone: ImageNet-pretrained ResNet-50 (frozen); Encoder: BasicGEBD-L4; Decoder: 6-layer Transformer.
- Diffusion steps: $T=1000$ (training); DDIM sampling with 32 steps (inference).
- CFG dropout probability $p=0.1$.
- The top 4 annotations (by F1 consistency score) are selected from 5 annotators per video in Kinetics-GEBD.

## Key Experimental Results

### Diversity-Aware Evaluation (Core Experiment)

Kinetics-GEBD dataset; each model generates $N_P=5$ predictions:

| Method | F1$_{\text{sym}}$ | F1$_{\text{p2g}}$ | F1$_{\text{g2p}}$ | Diversity |
|------|:---:|:---:|:---:|:---:|
| Temporal Perceiver | 69.4 | 72.2 | 67.4 | 14.6 |
| SC-Transformer | 72.9 | 74.9 | 71.6 | 18.9 |
| BasicGEBD | 72.2 | 74.5 | 70.6 | 18.6 |
| EfficientGEBD | 72.6 | 76.0 | 70.2 | 14.9 |
| **DiffGEBD** | **74.0** | 75.6 | **72.9** | **20.4** |

Note: Competing methods are deterministic models; their multiple predictions are obtained via 5 independent training runs with different random seeds. DiffGEBD's diverse predictions are produced by a single model with different initial noise.

### Conventional Evaluation

| Method | Kinetics-GEBD F1@0.05 | TAPOS F1@0.05 |
|------|:---:|:---:|
| EfficientGEBD | 78.3 | 63.1 |
| DyBDet | **79.6** | 62.5 |
| DiffGEBD ($w=4.0$) | 78.4 | **65.8** |

### Ablation Study

**Effect of conditioning features**: $\bm{E}$ (temporal self-similarity) vs. $\bm{F}$ (raw visual features) → F1$_{\text{sym}}$ 74.0 vs. 68.5

**Number of inference steps**: 1 step → F1$_{\text{sym}}$ 64.0; 2 steps → 72.3; 8 steps → 73.7; 32 steps → 74.0 (best)

**Number of annotations**: Performance improves monotonically from 1 to 4 annotations; using all 5 introduces noise from low-consistency annotations, causing a slight drop.

**CFG weight $w$**: F1$_{\text{sym}}$ is highest at $w=0.6$; as $w$ increases, F1$_{\text{p2g}}$ rises then falls, while Diversity decreases monotonically.

### Key Findings

- DiffGEBD achieves state-of-the-art performance on F1$_{\text{sym}}$, F1$_{\text{g2p}}$, and Diversity simultaneously.
- EfficientGEBD achieves the highest F1$_{\text{p2g}}$ but exhibits very low diversity (14.9), indicating precise yet insufficiently diverse predictions.
- High diversity does not necessarily imply high F1$_{\text{g2p}}$—diversity must be accompanied by plausibility to be meaningful.
- DiffGEBD substantially outperforms prior SOTA on TAPOS (65.8 vs. 63.1), suggesting that generative approaches yield greater advantages on more challenging datasets.
- The diffusion model requires only a single training run, unlike deterministic models that require multiple training runs to obtain diverse predictions.

## Highlights & Insights

- **Novel problem formulation**: DiffGEBD reframes GEBD from a discriminative problem ("which locations are boundaries?") to a generative one ("sample plausible boundary distributions from noise"). This perspective shift is elegant and directly leverages the generative diversity of diffusion models to match the annotation diversity inherent in GEBD.
- **Rigorous evaluation protocol design**: The combination of Symmetric F1 and Diversity Score fills a critical gap in evaluating multi-prediction scenarios. Symmetric F1 considers both Pred→GT and GT→Pred directions, making it more balanced than unidirectional F1.
- **Effective use of CFG**: Classifier-Free Guidance, originally developed for balancing quality and diversity in image generation, is precisely transferred to boundary detection. A single continuous parameter $w$ enables smooth interpolation between deterministic and diverse prediction regimes.
- **Fair experimental design**: Comparing multiple training runs of deterministic methods against single-model multi-sample outputs of DiffGEBD, while also reporting conventional evaluation results, ensures comprehensive and fair comparison.

## Limitations & Future Work

- **Inference efficiency**: DDIM requires 32 iterative denoising steps, making inference slower than deterministic methods—a bottleneck for real-time boundary detection scenarios.
- **Conventional F1 on Kinetics-GEBD does not surpass SOTA**: DyBDet (79.6) still outperforms DiffGEBD (78.4), indicating room for improvement in single best-prediction scenarios.
- **Generalization**: Experiments are conducted only on Kinetics-GEBD and TAPOS; cross-domain generalization remains unverified.
- **Coarse-grained diversity control**: The CFG weight $w$ is a global parameter and cannot control diversity at a local level. Unambiguous boundaries may require no diversity, while uncertain regions may benefit from higher diversity.
- **Computational overhead**: Compared to lightweight deterministic models (e.g., EfficientGEBD), the diffusion framework incurs substantially higher computational cost due to the encoder, decoder, and multi-step inference.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness](../../ACL2026/image_restoration/purging_the_gray_zone_latent-geometric_denoising_for_precise_knowledge_boundary_.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [\[ICCV 2025\] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention](consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)

<!-- RELATED:END -->
