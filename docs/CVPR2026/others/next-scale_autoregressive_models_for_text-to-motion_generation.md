---
title: >-
  [Paper Note] Next-Scale Autoregressive Models for Text-to-Motion Generation
description: >-
  [CVPR 2026][text-to-motion generation] MoScale proposes a next-scale autoregressive motion generation framework that replaces conventional next-token prediction. By performing hierarchical causal generation from coarse to fine, the model captures global semantic structure and introduces cross-scale hierarchical refinement and in-scale temporal refinement, achieving state-of-the-art performance on HumanML3D and KIT-ML (Top-1 0.540, FID 0.046).
tags:
  - CVPR 2026
  - text-to-motion generation
  - autoregressive models
  - multi-scale prediction
  - hierarchical generation
  - motion synthesis
date: 2026-05-08
content_hash: b5973b899aab0190
---

# Next-Scale Autoregressive Models for Text-to-Motion Generation

**Conference**: CVPR 2026
**arXiv**: [2604.03799](https://arxiv.org/abs/2604.03799)
**Code**: See project homepage
**Area**: Other
**Keywords**: text-to-motion generation, autoregressive models, multi-scale prediction, hierarchical generation, motion synthesis

## TL;DR

MoScale proposes a next-scale autoregressive motion generation framework that replaces conventional next-token prediction. By performing hierarchical causal generation from coarse to fine, the model captures global semantic structure and introduces cross-scale hierarchical refinement and in-scale temporal refinement, achieving state-of-the-art performance on HumanML3D and KIT-ML (Top-1 0.540, FID 0.046).

## Background & Motivation

1. **State of the Field**: Text-to-motion generation aims to synthesize human motion sequences that faithfully reflect the intent of textual descriptions. Current approaches fall into three main categories: next-token autoregressive models (T2M-GPT, AttT2M), diffusion models (MDM, ReMoDiffuse), and masked Transformers (MoMask, MoMask++).

2. **Limitations of Prior Work**:
    - **Diffusion models and masked Transformers**: These methods generate a full-resolution sequence draft and then iteratively refine it. The initial global semantics are often inaccurate, and subsequent refinement primarily improves local consistency rather than global structure.
    - **Next-token AR**: Human motion exhibits strong short-term predictability (future poses can be inferred from a brief history), causing AR models to exploit short-range shortcuts during training to minimize loss without learning long-range semantic structure. Temporal convolutions in VQ-VAE further amplify local correlations.
    - **Shared limitation**: Both paradigms struggle to capture repetition counts (e.g., "two jumping jacks") and sequence-level action patterns (e.g., "turn around, pick something up, then turn back").

3. **Root Cause**: The causal direction of next-token prediction (frame-by-frame along the temporal axis) coincides with the high short-term predictability of human motion, creating a shortcut that impedes global semantic reasoning.

4. **Paper Goals**: Design a causal hierarchical structure that commits to a global semantic layout at the earliest stage of generation, thereby eliminating short-range shortcuts.

5. **Starting Point**: Inspired by next-scale modeling in image generation (VAR), the method organizes motion sequences into hierarchical discrete token groups at increasing temporal resolutions, generating autoregressively from the coarsest scale (global semantics) to the finest scale (local details).

6. **Core Idea**: Replace next-token with next-scale prediction — determine the global motion structure at the coarsest scale and progressively refine it toward high temporal resolution.

## Method

### Overall Architecture

(1) **Multi-scale motion representation**: A residual VQ-VAE encodes motion into discrete token groups at $K$ scales (e.g., 6→12→24→49), where coarse scales capture global structure and fine scales encode residual details. (2) **Next-scale causal Transformer**: Starting from a text condition, tokens are generated autoregressively scale by scale, with bidirectional attention applied within each scale. (3) **Two refinement mechanisms**: cross-scale hierarchical refinement to correct error accumulation, and in-scale temporal refinement to improve local consistency.

### Key Designs

1. **Hierarchical Motion Representation (Multi-Scale Residual Quantization)**:

    - **Function**: Encodes the motion sequence into multi-level discrete tokens from coarse to fine.
    - **Mechanism**: The encoder maps motion $\mathbf{m} \in \mathbb{R}^{T \times D_m}$ to a latent representation $\mathbf{f}$, which is then quantized layer by layer at $K$ increasing lengths $(L_1, ..., L_K)$. At scale $k$, the residual $\mathbf{f} - \hat{\mathbf{f}}_{:k-1}$ not captured by the preceding $k{-}1$ scales is downsampled to length $L_k$ and vector-quantized. All scales share a single codebook $\mathbf{Z} \in \mathbb{R}^{V \times D_e}$.
    - **Design Motivation**: Unlike conventional residual VQ, which quantizes residuals at the same resolution across layers, each layer here corresponds to a distinct temporal resolution. This naturally enables coarse scales to capture global structure and fine scales to capture local details.

2. **Cross-Scale Hierarchical Refinement**:

    - **Function**: Strengthens the model's ability to correct prediction errors propagated from coarser scales.
    - **Mechanism**: During training, a random subset of tokens from scale $k{-}1$ is replaced with randomly sampled codebook entries (corruption rate $\gamma_k \sim U[0, \gamma_{max}]$), and scale $k$ learns to predict the correct residual targets from the corrupted input. Crucially, the corruption affects only the input to scale $k$ and does not alter the learning target of scale $k{-}1$. The optimal corruption rate is $\gamma_{max} = 0.6$.
    - **Design Motivation**: Under standard teacher forcing, each scale receives perfect inputs during training, so errors accumulate at inference time. By exposing the model to perturbed intermediate states during training, it learns to recover correct outputs from imperfect conditions, thereby reducing exposure bias.

3. **In-Scale Temporal Refinement**:

    - **Function**: Leverages bidirectional context to improve temporal consistency within each scale.
    - **Mechanism**: Within each scale, a mask-and-repredict operation is applied to low-confidence tokens. A binary mask $\mathbf{m}_k^i$ is constructed to replace uncertain tokens with [MASK]; these are concatenated with accumulated features from preceding scales and fed into the Transformer for re-prediction. A cosine re-masking schedule is adopted, with refinement steps set to $(1, 2, 5, 10)$ per scale.
    - **Design Motivation**: Text-motion datasets are far smaller than language corpora, and prior work has shown that diffusion-style iterative refinement and bidirectional context benefit low-data regimes. This mechanism also enables MoScale to natively support zero-shot tasks such as motion editing, completion, and continuation.

### Loss & Training

- VQ-VAE training: reconstruction loss + joint position loss + commitment loss
- Transformer training: teacher forcing + cross-entropy loss; text condition dropped with 10% probability (CFG)
- HumanML3D: 120 training epochs, learning rate $3 \times 10^{-4}$; KIT-ML: 60 training epochs
- Inference CFG scale: 5 (HumanML3D) / 3 (KIT-ML)
- Codebook size: 512; 4 hierarchical scales; sequence lengths: (6, 12, 24, 49)

## Key Experimental Results

### Main Results

HumanML3D:

| Method | Type | Top-1↑ | FID↓ | MM-Dist↓ | Diversity |
|--------|------|--------|------|----------|-----------|
| T2M-GPT | Next-token | 0.492 | 0.141 | 3.121 | 9.722 |
| ParCo | Next-token | 0.515 | 0.109 | 2.927 | 9.576 |
| ReMoDiffuse | Diffusion | 0.510 | 0.103 | 2.974 | 9.018 |
| MoMask++ | MaskedTrans | 0.528 | 0.072 | 2.912 | — |
| **MoScale (S=18)** | **Next-scale** | **0.540** | **0.046** | **2.830** | 9.525 |

KIT-ML:

| Method | Top-1↑ | FID↓ | MM-Dist↓ |
|--------|--------|------|----------|
| ParCo | 0.430 | 0.453 | 2.820 |
| MoMask | 0.433 | 0.204 | 2.779 |
| **MoScale (S=18)** | **0.442** | 0.173 | **2.717** |

### Ablation Study

| Configuration | Top-1↑ | FID↓ | MM-Dist↓ |
|---------------|--------|------|----------|
| Base (no refinement) | 0.481 | 0.176 | 3.136 |
| + Hierarchical Refinement (HR) | 0.534 | 0.090 | 2.853 |
| + Temporal Refinement (TR) | 0.497 | 0.129 | 3.043 |
| + HR & TR (full model) | **0.540** | **0.046** | **2.830** |

Text complexity analysis (Top-3):

| Method | FULL | MEDIUM+HIGH | HIGH |
|--------|------|-------------|------|
| ParCo | 0.801 | 0.778 | 0.709 |
| MoMask++ | 0.811 | 0.802 | 0.762 |
| **MoScale** | **0.817** | **0.812** | **0.775** |

### Key Findings

- **Hierarchical refinement is the primary driver of text alignment improvement**: HR accounts for the substantial Top-1 gain from 0.481 to 0.534, while TR primarily improves local temporal consistency.
- **MoScale's advantage grows with text complexity**: On the high-complexity subset, MoScale outperforms ParCo by 0.066 (6.6 percentage points), far exceeding the overall margin of 0.016.
- **Optimal corruption rate $\gamma_{max} = 0.6$**: A lower value insufficiently exposes the model to errors; a higher value introduces excessive noise.
- **Scalable model capacity**: Performance improves consistently as the Transformer depth increases from 4 to 16 layers, with high training efficiency.

## Highlights & Insights

- **The short-range shortcut of next-token prediction** is a profound insight: the high short-term predictability of human motion allows AR models to "take the easy path," learning local dynamics while ignoring global semantics. Next-scale prediction breaks this shortcut by enforcing global structure encoding at the coarsest scale.
- **Cross-scale refinement** is elegantly designed: by corrupting the preceding scale's input to train the current scale, the method simulates inference-time error accumulation while maintaining single-pass forward training efficiency.
- **Unified zero-shot capability**: The same mask-and-repredict mechanism supports motion editing, completion, continuation, and other tasks, achieving 78–82% user preference in a user study.
- The method exhibits superior training efficiency compared to baselines and demonstrates scalable model capacity, making it practically valuable.

## Limitations & Future Work

- Quantization error in the VQ-VAE remains a bottleneck, potentially causing information loss for fine-grained motions.
- The current design uses only T5 text features; the potential of stronger text representations (e.g., LLM embeddings) for further semantic alignment has not been explored.
- The MModality score (0.873) is relatively low, indicating reduced generation diversity.
- The choice of scale count and sequence lengths relies on empirical tuning rather than an adaptive mechanism.
- Inference speed at S=18 (0.28s) is acceptable but significantly slower than at S=4 (0.08s); the trade-off between refinement steps and quality warrants further investigation.

## Related Work & Insights

- **vs. MoMask++**: MoMask++ employs a shared codebook, a unified Transformer, and random token perturbation, but the perturbation breaks hierarchical causality. MoScale strictly maintains coarse-to-fine causal structure and achieves a Top-1 improvement of 0.012.
- **vs. ParCo**: ParCo extends the standard next-token AR paradigm with additional modules but yields limited gains. MoScale changes the generation direction (scale vs. time), achieving greater improvements under more efficient training.
- **vs. VAR (image generation)**: MoScale borrows the next-scale concept from VAR but introduces key adaptations — cross-scale hierarchical refinement and in-scale temporal refinement — to address the low-data challenges unique to motion data.
- This work demonstrates that **generation order** is critical in sequence modeling, offering insights applicable to other temporal generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introduces next-scale prediction to motion generation and provides deep analysis of the short-range shortcut problem in next-token AR models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks, a user study, text complexity analysis, and detailed ablations — comprehensive and convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, method presentation is fluent, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ Significant contribution to the motion generation field; the cross-scale autoregressive paradigm has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)
- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_largescale_multimodal_dataset_for_cad.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](../../ICCV2025/others/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)

<!-- RELATED:END -->
