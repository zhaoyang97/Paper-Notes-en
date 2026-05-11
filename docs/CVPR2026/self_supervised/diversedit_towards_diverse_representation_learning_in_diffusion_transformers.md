---
title: >-
  [Paper Note] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers
description: >-
  [CVPR 2026][Self-Supervised Learning][Diffusion Transformer] Through systematic analysis, this work identifies inter-block representation diversity as a key factor for effective learning in DiTs…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Diffusion Transformer"
  - "representation diversity"
  - "long residual connections"
  - "diversity loss"
  - "image generation"
date: 2026-05-08
content_hash: 29108a9715fa3483
---

# DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers

**Conference**: CVPR 2026
**arXiv**: [2603.04239](https://arxiv.org/abs/2603.04239)
**Code**: [Available](https://github.com/kobeshegu/DiverseDiT)
**Area**: Self-Supervised / Representation Learning
**Keywords**: Diffusion Transformer, representation diversity, long residual connections, diversity loss, image generation

## TL;DR

Through systematic analysis, this work identifies inter-block representation diversity as a key factor for effective learning in DiTs, and proposes DiverseDiT: long residual connections to diversify inputs combined with a representation diversity loss to explicitly promote feature differentiation across blocks—accelerating convergence and improving generation quality without any external guidance model.

## Background & Motivation

### 1. State of the Field

Diffusion Transformers (DiT) have achieved breakthroughs in visual generation owing to their excellent scalability. Recent studies have found that high-performing diffusion models can capture more discriminative internal representations, inspiring methods such as REPA—which aligns intermediate DiT representations with features from pretrained visual encoders (e.g., DINOv2)—with subsequent works like REPA-E and REG further extending this direction.

### 2. Limitations of Prior Work

- **Dependence on large external models**: REPA-family methods require strong pretrained visual encoders (DINOv2, MAE, etc.), which are themselves costly to train.
- **Unclear mechanisms**: Fundamental questions—how DiTs learn meaningful representations and why external alignment helps—remain poorly understood.
- **Blind alignment is harmful**: Aligning more blocks with more encoders can actually degrade performance.

### 3. Root Cause

A disconnect exists between "using external models to provide guidance" and "understanding the intrinsic representation learning mechanism"—practitioners apply REPA without understanding why it works, precluding principled improvements.

### 4. Paper Goals

Reveal the intrinsic mechanisms of representation learning in DiTs, and on that basis design an efficient representation learning framework that requires no external guidance.

### 5. Starting Point

CKA (Centered Kernel Alignment) is used to systematically measure the evolution of inter-block representation similarity during training, offering a novel perspective—inter-block representation diversity—for understanding and improving DiTs.

### 6. Core Idea

**Representation diversity hypothesis**: The greater the representational difference across blocks in a DiT, the better the model learns. REPA is effective precisely because it increases the representational difference between the aligned block and the others. Based on this insight, one can directly design mechanisms to promote diversity without relying on external encoders.

## Method

### Overall Architecture

DiverseDiT comprises two complementary components:

1. **Long Residual Connections**: Inject shallow-block outputs into their symmetric deep counterparts, breaking input homogenization.
2. **Representation Diversity Loss**: Composed of an orthogonality loss, a mutual information minimization loss, and a feature dispersion loss, explicitly constraining each block to learn distinct features.

The two components promote representation diversity from the perspectives of input diversification and output differentiation, respectively, without requiring any external pretrained model.

### Key Designs

#### Design 1: Long Residual Connections

- **Function**: Connect the output of block $i$ to block $(L-i)$ (where $L$ is the total number of layers), forming symmetric skip connections.
- **Mechanism**: $f_l = \text{Linear}(\text{Norm}(f_i \oplus f_{l-1}))$, where $\oplus$ denotes concatenation; the concatenated $2D$-dimensional features are projected back to $D$ dimensions via LayerNorm + Linear.
- **Design Motivation**: In vanilla DiT, each block receives input only from the previous layer's output, leading to highly homogeneous inputs. Injecting shallow-layer features provides differentiated input signals to different blocks, encouraging feature reuse and preventing representation collapse.

#### Design 2: Orthogonality Loss $\mathcal{L}_{\text{orth}}$

- **Function**: Penalizes high cosine similarity between the mean representations of different blocks.
- **Mechanism**: The token-level representation of each block is averaged over the batch and token dimensions to obtain $\mu_l \in \mathbb{R}^D$; cosine similarity between selected block pairs is then minimized.
- **Design Motivation**: Encourages different blocks to learn features with orthogonal directions, avoiding redundancy.

#### Design 3: Mutual Information Minimization Loss $\mathcal{L}_{\text{MI}}$

- **Function**: Minimizes statistical dependencies between representations of different blocks.
- **Mechanism**: The average cosine similarity between normalized token vectors serves as an efficient proxy for mutual information, avoiding direct computation of high-dimensional covariance matrices.
- **Design Motivation**: Ensures statistical independence among inter-block representations so that each block captures complementary information.

#### Design 4: Feature Dispersion Loss $\mathcal{L}_{\text{disp}}$

- **Function**: Maximizes the variance of feature activations across the channel dimension.
- **Mechanism**: After flattening and normalizing representations of each block, the mean activation per dimension is computed; its variance is then maximized (negated as the loss).
- **Design Motivation**: Encourages the model to fully utilize all feature channels, preventing activations from concentrating on a few dimensions.

### Loss & Training

Total diversity loss: $\mathcal{L}_{\text{div}} = 0.33 \cdot \mathcal{L}_{\text{orth}} + 0.33 \cdot \mathcal{L}_{\text{MI}} + 0.33 \cdot \mathcal{L}_{\text{disp}}$

**Adaptive weighting mechanism**: When $\mathcal{L}_{\text{div}}$ approaches zero, the model diverges (excessive separation impedes learning of shared semantic representations). A piecewise linear weight is therefore applied:

- $\mathcal{L}_{\text{div}} > 0.5$: weight $w=1$ (normal optimization)
- $0.1 < \mathcal{L}_{\text{div}} \le 0.5$: weight $w = (\mathcal{L}_{\text{div}} - 0.1) / 0.5$ (gradually reduced)
- $\mathcal{L}_{\text{div}} \le 0.1$: weight $w=0$ (diversity optimization halted)

Training configuration: AdamW, lr=1e-4, batch size=256, 8×H800 GPUs. Only a small number of additional parameters (Linear layers for long residual connections) are introduced.

## Key Experimental Results

### Main Results

**Table 1: Results at different model scales on ImageNet 256×256 (no CFG, 400K iterations)**

| Model | FID↓ | sFID↓ | IS↑ | Prec.↑ | Rec.↑ |
|-------|------|-------|-----|--------|-------|
| SiT-B | 36.80 | 6.77 | 40.09 | 0.51 | 0.63 |
| **+ Ours** | **28.05** | **6.04** | **50.66** | **0.57** | 0.63 |
| REPA-B | 22.99 | 6.70 | 64.73 | 0.59 | 0.65 |
| **+ Ours** | **17.29** | **6.56** | **79.92** | **0.62** | 0.65 |
| SiT-XL | 17.43 | 5.11 | 76.00 | 0.64 | 0.64 |
| **+ Ours** | **12.42** | **4.85** | **95.01** | **0.68** | 0.63 |
| REPA-XL | 8.73 | 5.21 | 118.68 | 0.69 | 0.65 |
| **+ Ours** | **8.09** | **5.02** | **123.23** | **0.70** | 0.65 |

**Table 2: Comparison with SOTA methods on ImageNet 256×256 (with CFG)**

| Method | Epochs | FID↓ | IS↑ | Rec.↑ |
|--------|--------|------|-----|-------|
| DiT-XL/2 | 1400 | 2.27 | 278.20 | 0.57 |
| SiT-XL/2 | 1400 | 2.06 | 270.30 | 0.59 |
| REPA | 200 | 1.96 | 264.00 | 0.60 |
| REG | 800 | 1.36 | 299.40 | 0.66 |
| SRA | 800 | 1.58 | 311.40 | 0.63 |
| **DiverseDiT (Ours)** | **80** | **1.89** | **276.85** | **0.66** |
| **DiverseDiT (Ours)** | **200** | **1.52** | **282.72** | **0.66** |

**Single-step generation SOTA (ImageNet 256×256, with CFG)**: MeanFlow-XL/2 + Ours achieves FID=**2.99**, surpassing all existing single-step methods.

### Ablation Study

**Component ablation (SiT-B / REPA-B, 400K iter)**:

| Configuration | SiT-B FID↓ | REPA-B FID↓ |
|---------------|-----------|------------|
| Full (complete method) | 28.05 | 17.29 |
| w/o diversity loss | 32.77 | 20.66 |
| w/o residual connections | 33.72 | 18.18 |

**Loss variant ablation (REPA-B)**:

| Configuration | FID↓ | IS↑ |
|---------------|------|-----|
| Full | 17.29 | 79.92 |
| only $\mathcal{L}_{\text{orth}}$ | 18.97 | 75.44 |
| only $\mathcal{L}_{\text{MI}}$ | 17.70 | 78.34 |
| only $\mathcal{L}_{\text{disp}}$ | 20.85 | 68.74 |

**Adaptive range ablation**: A constant weight causes divergence; the range [0.1, 0.5] is optimal (FID 28.05), outperforming [0.2, 0.7] (30.59) and [0.3, 0.9] (31.85).

### Key Findings

1. **Consistent improvement**: The method yields stable gains across three baselines (SiT, REPA, MeanFlow) and three scales (B/L/XL).
2. **Cross-scale competitiveness**: REPA-B + Ours (17.29) outperforms vanilla SiT-L (18.77); REPA-L + Ours (8.47) outperforms REPA-XL (8.73).
3. **Training efficiency**: FID 1.89 is achieved with only 80 epochs, surpassing REPA's 1.96 at 200 epochs.
4. **Complementarity with existing methods**: SiT-B + Ours + DispLoss + SRA = FID 21.95, outperforming REPA-B (22.99) without requiring an external encoder.

## Highlights & Insights

- **Analysis-driven design**: Systematic CKA analysis first reveals "representation diversity" as the key factor, which then motivates the method design—the logical chain is complete and principled.
- **New explanation for REPA**: REPA is effective not because of external knowledge per se, but because it increases the representational difference between the target block and the others—a highly illuminating insight.
- **Simplicity and efficiency**: Both components are conceptually simple and lightweight, introducing only a small number of additional parameters (Linear layers for long residual connections), with broad applicability.
- **No external models required**: Eliminates dependence on large pretrained encoders such as DINOv2 and MAE.

## Limitations & Future Work

- The adaptive weighting mechanism (piecewise linear function) is somewhat ad hoc; the thresholds 0.1/0.5 lack theoretical justification.
- The selection strategy for block pairs (subset $\mathcal{P}$) is not thoroughly discussed; the optimal choice may depend on model scale and depth.
- Validation is limited to ImageNet; more complex scenarios such as text-to-image and video generation remain untested.
- A gap remains compared to REG (FID 1.36@800ep) vs. Ours (FID 1.52@200ep); performance under extended training is not fully explored.
- The generalizability of long residual connections to asymmetric architectures (e.g., U-ViT) awaits verification.

## Related Work & Insights

- **REPA [Yu et al.]**: Aligns intermediate hidden states with external encoders; this work explains the root cause of its effectiveness.
- **DispLoss [Wang et al.]**: Dispersion loss encourages representations to spread across the embedding space; DiverseDiT takes a more systematic approach (input diversity + output diversity simultaneously).
- **SRA [Li et al.]**: A self-alignment method where low-noise layers guide high-noise layers; complementary to and stackable with DiverseDiT.
- **MeanFlow [Liu et al.]**: A single-step generation method; DiverseDiT applies seamlessly and sets a new SOTA.
- **Insights**: The inter-block diversity perspective generalizes to other Transformer architectures (ViT, LLMs)—the question of "inter-layer collaboration vs. inter-layer redundancy" in deep networks is a broadly relevant research direction worth further investigation.

## Rating

⭐⭐⭐⭐ A solid, analysis-driven work. The logical chain from CKA observations to method design is self-consistent; the two components are simple, effective, and complementary to existing methods; experiments comprehensively cover multiple baselines and scales. Minor weaknesses include the somewhat empirical nature of the adaptive weighting design and the absence of validation on broader generative scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](../../ICLR2026/self_supervised/no_other_representation_component_is_needed_diffusion_transformers_can_provide_r.md)
- [\[CVPR 2026\] Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)
- [\[CVPR 2026\] TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)

</div>

<!-- RELATED:END -->
