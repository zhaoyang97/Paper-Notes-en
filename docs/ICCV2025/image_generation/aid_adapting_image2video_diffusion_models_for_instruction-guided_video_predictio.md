---
title: >-
  [Paper Note] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction
description: >-
  [ICCV 2025][Image Generation][Video Prediction] This paper proposes AID, a framework that transfers a pretrained Image2Video diffusion model (SVD) to text-guided video prediction (TVP) tasks. Through MLLM-assisted video…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Video Prediction"
  - "Diffusion Models"
  - "Text-Guided"
  - "Stable Video Diffusion"
  - "Adapter"
date: 2026-05-08
content_hash: 6db571c368b19271
---

# AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction

**Conference**: ICCV 2025
**arXiv**: [2406.06465](https://arxiv.org/abs/2406.06465)
**Code**: [https://chenhsing.github.io/AID](https://chenhsing.github.io/AID)
**Area**: Image Generation
**Keywords**: Video Prediction, Diffusion Models, Text-Guided, Stable Video Diffusion, Adapter

## TL;DR

This paper proposes AID, a framework that transfers a pretrained Image2Video diffusion model (SVD) to text-guided video prediction (TVP) tasks. Through MLLM-assisted video state prediction, a Dual-Query Transformer for condition injection, and spatiotemporal adapters, AID surpasses the previous state-of-the-art FVD scores by over 50% across multiple datasets.

## Background & Motivation

Text-guided video prediction (TVP) aims to predict future video frames given an initial frame and a textual instruction, with broad applications in VR, robotic manipulation, and content creation. Existing TVP methods face two core challenges:

**Data scarcity in target domains**: Training data for specific domains (e.g., robotic arm manipulation, egocentric cooking videos) is limited in scale. Directly fine-tuning video models extended from text-to-image models tends to produce poor inter-frame consistency and temporal instability.

**Lack of controllability in SVD**: Large-scale pretrained video models such as Stable Video Diffusion have learned strong video dynamics priors, but only support image-to-video generation without text control.

The core idea is to transfer SVD's video prior to TVP while injecting text controllability—rather than training video understanding from scratch. This entails two technical challenges: how to design and inject text conditioning, and how to adapt to target domains at low training cost.

## Method

### Overall Architecture

AID builds upon SVD and consists of three main components: (1) **MLLM-assisted video prediction prompting**—using a multimodal large language model to predict future video states; (2) **Dual-Query Transformer (DQFormer)**—integrating multimodal conditioning information; and (3) **spatiotemporal adapters**—enabling efficient transfer to target datasets with minimal parameters. During training, all parameters of SVD's VAE and 3D UNet are frozen; only the newly introduced DQFormer, adapters, and cross-attention layers are trained.

### Key Designs

1. **MLLM-Assisted Video State Prediction (Video Prediction Prompting)**:

    - A single-sentence instruction is insufficient to fully describe the temporal dynamics of a video.
    - The initial frame and text instruction are fed into a multimodal LLM (e.g., LLaVA) to predict multiple development stages of the video.
    - For example, "lifting up one end of a tablet box, then letting it drop down" is decomposed into 4 states: initial state → lift one end → release one end → final state.
    - Design motivation: explicitly decomposing temporal changes into discrete state descriptions provides a semantic basis for subsequent frame-level conditional control.

2. **Dual-Query Transformer (DQFormer)**:

    - **Upper branch (multimodal embedding)**: text instruction features $\bm{t_1}$ are processed by self-attention and then cross-attended with visual features $\bm{v}$ extracted by a CLIP Visual Encoder:
    $\text{multimodal emb} = \text{Softmax}\Big(\frac{(W_1^Q \text{SelfAttn}(\bm{t_1}))(W_1^K \bm{v})^T}{\sqrt{d_1}}\Big)(W_1^V \bm{v})$
    - **Lower branch (decomposed embedding)**: learnable queries $\bm{Q} \in \mathbb{R}^{(N \cdot N_t) \times C}$ ($N$ frames × $N_t$ queries per frame) first cross-attend with instruction features $\bm{t_1}$ for frame-level decomposition, then cross-attend with MLLM state features $\bm{t_2}$ to fuse multi-state information.
    - The outputs of both branches are concatenated into MCondition and injected into the UNet via cross-attention.
    - Design motivation: the upper branch handles global multimodal alignment (understanding *what to do*), while the lower branch handles frame-level temporal decomposition (understanding *how each frame should change*).

3. **Three Adapter Types (Spatiotemporal Transfer)**:

    - **Spatial Adapter**: placed alongside the spatial self-attention layer, consisting of a down-projection linear layer + GELU + up-projection linear layer, with the up-projection initialized to zero to preserve the original model:
    $\text{S-Adapter}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{up}(\text{GELU}(\mathbf{W}_{down}(\mathbf{X})))$
    - **Short-Term Temporal Adapter**: inserts a depthwise separable 3D convolution between linear layers to model short-term temporal relations between adjacent frames:
    $\text{ST-Adapter}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{up}(\text{3D-Conv}(\mathbf{W}_{down}(\mathbf{X})))$
    - **Long-Term Temporal Adapter**: replaces convolution with temporal self-attention to capture global temporal dependencies:
    $\text{LT-Adapter}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{up}(\text{Self-Attn}(\mathbf{W}_{down}(\mathbf{X})))$
    - Design motivation: freezing the original UNet weights while adding lightweight adapters preserves the video prior, enables low-cost domain transfer, and avoids overfitting.

### Inference Strategy

Dual classifier-free guidance is applied to independently control the frame condition and the text condition:
$$\tilde{e_\theta}(z_t, c_T, c_V) = e_\theta(z_t, \emptyset, \emptyset) + s_V \cdot (\ldots) + s_T \cdot (\ldots)$$

## Key Experimental Results

### Main Results

| Method | Text | SSv2 FVD↓ | Bridge FVD↓ | Epic100 FVD↓ |
|--------|------|-----------|-------------|--------------|
| SimVP | No | 537.2 | 681.6 | 1991 |
| PVDM | No | 502.4 | 490.4 | 482.3 |
| VideoFusion | Yes | 163.2 | 501.2 | 349.9 |
| Tune-A-Video | Yes | 291.4 | 515.7 | 365.0 |
| Seer | Yes | 112.9 | 246.3 | 271.4 |
| **AID (Ours)** | **Yes** | **50.23** | **21.57** | **52.78** |

FVD improvements: 55.5% on SSv2, **91.2%** on Bridge, and 80.6% on Epic100.

### Ablation Study

**Condition component ablation (SSv2)**:

| Configuration | FVD↓ | KVD↓ | Notes |
|---------------|------|------|-------|
| No MCondition | 152.4 | 0.14 | No text guidance; prediction from initial frame only |
| Multimodal embedding only | 74.98 | 0.03 | Missing frame-level decomposition |
| Decomposed embedding only | 70.16 | 0.04 | Missing multimodal alignment |
| Full model without LLaVA | 64.48 | 0.03 | Missing state prediction |
| **Full AID** | **50.23** | **0.02** | All components synergize optimally |

**Adapter ablation (SSv2)**:

| Configuration | FVD↓ | KVD↓ | Notes |
|---------------|------|------|-------|
| No adapters | 279.42 | 0.71 | Only DQFormer trained; transfer fails |
| No spatial adapter | 68.12 | 0.05 | Spatial distribution not adapted |
| No temporal adapters | 76.32 | 0.03 | Temporal dynamics not transferred |
| No short-term temporal | 59.62 | 0.03 | Local temporal modeling absent |
| No long-term temporal | 58.16 | 0.02 | Global temporal modeling absent |
| **Full model** | **50.23** | **0.02** | All three adapters contribute |

### Key Findings

- FVD of 21.57 on Bridge Data indicates generation quality approaching real video.
- Qualitative comparisons reveal that Seer suffers from "ghosting" artifacts (cumulative inter-frame errors cause later frames to deviate), while commercial models such as Gen-2 and Pika fail to follow specific manipulation instructions.
- In robotic manipulation scenarios (e.g., "put corn in pot which is in sink distractors"), AID correctly localizes target objects and executes the appropriate placement action.
- Training only DQFormer without adapters yields FVD of 279.42, demonstrating that the condition injection module alone cannot accomplish domain transfer and that fine-tuning of the video prior is necessary.

## Highlights & Insights

- **Transfer learning paradigm**: AID is the first to successfully transfer a large-scale Image2Video model to domain-specific TVP tasks rather than training from scratch, leveraging the video dynamics prior acquired by SVD from large-scale data.
- **MLLM as "video director"**: using a multimodal LLM to predict video development states is an elegant solution for mapping single-sentence instructions to multi-frame temporal changes.
- **High efficiency**: freezing the UNet backbone and training only adapters and condition injection modules substantially reduces trainable parameters and memory consumption.
- Dual classifier-free guidance provides independent control over frame and text conditions.

## Limitations & Future Work

- The quality of MLLM state prediction depends on LLaVA's capabilities; complex scenes may yield inaccurate state descriptions.
- Experiments are conducted only at 256×256 resolution; high-resolution video prediction remains to be validated.
- Autoregressive long-video generation may suffer from error accumulation, which the paper does not address.
- The adapter design is relatively standard; exploring more parameter-efficient methods (e.g., LoRA) could further reduce training cost.
- No comparison with recent closed-source models such as Sora is provided.

## Related Work & Insights

- AID shares conceptual similarity with AnimateDiff and Tune-A-Video in using adapters for transfer, but focuses specifically on TVP and introduces a more complete condition injection framework.
- The dual-branch design of DQFormer draws inspiration from BLIP-2's Q-Former, with the addition of a frame-level decomposition branch.
- The results demonstrate the substantial potential of pretrained video models—high-quality video priors can significantly alleviate the difficulty of training on small datasets.

## Rating
- Novelty: ⭐⭐⭐⭐ First effective transfer of an I2V model to TVP; MLLM-based state prediction and DQFormer design are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets with comprehensive ablations, though comparisons with more recent methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Method description is logically clear with well-formatted equations.
- Value: ⭐⭐⭐⭐ Practical applicability to robotic manipulation and egocentric video prediction; FVD improvements are remarkable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)

</div>

<!-- RELATED:END -->
