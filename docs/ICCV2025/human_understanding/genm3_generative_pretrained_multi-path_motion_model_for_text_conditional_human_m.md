---
title: >-
  [Paper Note] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation
description: >-
  [ICCV 2025][Human Understanding][human motion generation] This paper proposes GenM3, a framework that learns unified discrete motion representations via a Multi-Expert VQ-VAE (MEVQ-VAE) and employs a Multi-path Motion Transformer (MMT) to handle intra-modal variation and cross-modal alignment. By integrating 11 motion datasets (~220 hours), GenM3 achieves state-of-the-art FID of 0.035 on HumanML3D.
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "human motion generation"
  - "VQ-VAE"
  - "multi-path Transformer"
  - "large-scale dataset"
  - "text-conditional generation"
date: 2026-05-08
content_hash: dec269d100f2fdb2
---

# GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation

**Conference**: ICCV 2025  
**arXiv**: [2503.14919](https://arxiv.org/abs/2503.14919)  
**Code**: N/A  
**Area**: Human Understanding  
**Keywords**: human motion generation, VQ-VAE, multi-path Transformer, large-scale dataset, text-conditional generation

## TL;DR

This paper proposes GenM3, a framework that learns unified discrete motion representations via a Multi-Expert VQ-VAE (MEVQ-VAE) and employs a Multi-path Motion Transformer (MMT) to handle intra-modal variation and cross-modal alignment. By integrating 11 motion datasets (~220 hours), GenM3 achieves state-of-the-art FID of 0.035 on HumanML3D.

## Background & Motivation

Generating diverse and accurate human motions from textual descriptions is a key research direction in computer vision. Despite the remarkable success of large pretrained models in text and image domains, motion generation still faces three major challenges:

**Heterogeneity of motion data distributions**: Datasets differ substantially in motion types and capture devices, such that joint training may improve performance on one dataset at the expense of another.

**Lack of dedicated motion pretraining backbones**: Existing methods adapt pretrained language models to motion tasks (e.g., MotionGPT), yet fundamental structural and contextual differences exist between motion and language.

**Absence of high-quality large-scale unified motion datasets**: Motion capture data is costly to acquire, and existing datasets are each limited to specific motion types.

A representative failure case: MotionGPT, trained on HumanML3D, fails to generate correct motion even for the simple text prompt "a person places their hands on their hips."

## Method

### Overall Architecture

GenM3 consists of two core components and a three-stage training pipeline:
- **MEVQ-VAE**: Discretizes continuous motion sequences into tokens.
- **MMT (Multi-path Motion Transformer)**: Performs cross-modal modeling of text and motion.
- Training pipeline: Stage 1 trains MEVQ-VAE → Stage 2 pretrains MMT (motion only) → Stage 3 trains with text conditioning.

### Key Designs

1. **Multi-Expert VQ-VAE (MEVQ-VAE)**:

    - Introduces multi-expert 1D convolutional layers into the encoder and decoder of a standard VQ-VAE.
    - **Each block contains 3 standard 1D convolutional layers + 1 multi-expert convolutional layer.**
    - All $e_q$ experts are activated simultaneously; the output is a weighted combination: $y = \sum_{i=1}^{e_q} w_i \cdot \text{Conv}_i(x)$
    - $w_i$ are learnable weights that adaptively modulate each expert's contribution.
    - Uses a shared codebook (8192 codewords of dimension 32) with a downsampling rate of 4.
    - Loss function: $\mathcal{L}_q = \mathcal{L}_{rec} + \beta \mathcal{L}_{commit}$

2. **Motion Descriptor**:

    - A CLIP encoder produces text embeddings $\mathbf{E}_t$ and a global text feature $\mathbf{e}_t$.
    - Motion tokens are passed through a motion embedder to obtain $\mathbf{E}_m$.
    - A text-guided attention aggregation generates a contextual summary: $\mathbf{E}_{ctx} = \text{mean}(\text{softmax}(\mathbf{E}_m \mathbf{E}_t) \mathbf{E}_t)$
    - The context token provides high-level motion semantics and enhances cross-modal alignment.

3. **Multi-path Motion Transformer (MMT)**:

    - First half (9 layers): standard Transformer self-attention + FFN.
    - Second half (9 layers): multi-path Transformer, introducing three parallel paths within the FFN layers:
        - **Motion path**: processes motion tokens; each path contains multiple densely activated experts.
        - **Text path**: processes text tokens.
        - **Cross-modal shared path**: processes both motion and text tokens simultaneously to promote cross-modal alignment.
    - Expert outputs within each path are weighted by a gating function: $\mathbb{E}_p(x) = \sum_i g_{p,i}(x) \mathbb{E}_{p,i}(x)$
    - Outputs from the three paths are concatenated and passed through a projection layer: $\text{Output} = \mathbf{W}_{proj}([\mathbb{E}_{motion}; \mathbb{E}_{text}; \mathbb{E}_{cross-modal}]) + b_{proj}$

### Loss & Training

- **Stage 1 (MEVQ-VAE)**: Reconstruction loss + commitment loss; codebook updated via moving averages.
- **Stage 2 (Pretraining)**: Motion data only; masked modeling objective; all paths take motion tokens as input. Loss: $\mathcal{L} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\setminus \mathcal{M}})$
- **Stage 3 (Text-conditional training)**: Motion–text pairs; all three paths activated; masked modeling framework conditioned on text and visible motion tokens.
- Optimizer: AdamW, learning rate $2\times10^{-4}$, warmup + cosine annealing.
- Batch size 160, trained for 120K iterations.
- Inference uses parallel decoding: all tokens initialized as [Mask], iteratively replacing low-confidence tokens.

## Key Experimental Results

### Main Results

**HumanML3D Benchmark (30FPS evaluator)**:

| Method | FID↓ | R-Precision Top3↑ | MMDist↓ | Diversity↑ |
|------|------|-------------------|---------|-----------|
| Real | 0.002 | 0.785 | 2.982 | 9.458 |
| T2M-GPT | 0.160 | 0.770 | 3.083 | 9.653 |
| MMM | 0.110 | 0.784 | 2.951 | 9.484 |
| **GenM3** | **0.046** | **0.804** | **2.852** | 9.675 |

**HumanML3D Benchmark (20FPS evaluator, comparison with more methods)**:

| Method | FID↓ | R-Precision Top3↑ | Diversity↑ |
|------|------|-------------------|-----------|
| MoMask | 0.045 | 0.807 | - |
| MotionGPT | 0.232 | 0.778 | 9.528 |
| OMG | 0.381 | 0.784 | 9.657 |
| **GenM3** | **0.035** | 0.795 | 9.341 |

**IDEA400 Zero-shot Generalization**:

| Method | FID↓ | R-Precision Top3↑ | MMDist↓ |
|------|------|-------------------|---------|
| T2M-GPT | 7.947 | 0.301 | 5.488 |
| MMM | 6.001 | 0.307 | 4.980 |
| GenM3 | 4.430 | 0.335 | 4.732 |
| **GenM3*** | **4.232** | **0.338** | **4.520** |

### Ablation Study

**Comparison of VQ methods**:

| Method | FID↓ |
|------|------|
| Standard VQ | 0.098 |
| **MEVQ-VAE** | **0.048** |
| RVQ | 0.043 |
| MERVQ (RVQ + Multi-Expert) | **0.032** |
| G-RVQ | 0.045 |
| FSQ | 0.057 |

**Multi-path Transformer ablation**:

| Motion Path | Text Path | Cross-modal Path | FID↓ | Diversity↑ |
|---------|---------|-----------|------|-----------|
| ✓ | - | - | 0.058 | 9.400 |
| ✓ | ✓ | - | 0.045 | 9.282 |
| ✓ | - | ✓ | 0.044 | 9.344 |
| ✓ | ✓ | ✓ | **0.035** | 9.341 |

**Dense MoE vs. Sparse MoE**:

| Type | FID↓ | R-Precision Top3↑ |
|------|------|-------------------|
| Sparse MoE | 0.058 | 0.799 |
| **Dense MoE** | **0.046** | **0.804** |

### Key Findings

- **GenM3's FID of 0.035 substantially outperforms** the second-best MoMask (0.045) by 22%.
- **Pretraining on the large-scale mixed dataset yields the largest gain for GenM3 (35.21%)**, markedly exceeding the gains observed for MMM and T2M-GPT, attributed to the multi-expert design's ability to adapt to heterogeneous data.
- The Multi-Expert design generalizes to RVQ: MERVQ reduces FID from 0.043 to 0.032.
- Dense MoE (all experts activated) outperforms Sparse MoE (partial activation), as full activation better captures cross-dataset commonalities.
- Using all three paths simultaneously yields the best results; the cross-modal path contributes more than the text path alone (0.044 vs. 0.045).
- GenM3* achieves stronger zero-shot performance on IDEA400, indicating that additional text–motion pairs enhance generalization.

## Highlights & Insights

- **Design philosophy of dense multi-expert activation**: Unlike mainstream sparse MoE, GenM3 adopts full expert activation with learned weighting, better handling data heterogeneity.
- **Progressive logic of the three-stage training strategy**: First learn good motion representations → then learn general motion patterns → finally align with text conditioning.
- **Compact design of the Motion Descriptor**: Text-guided attention compresses motion sequences into context tokens efficiently.
- **Unified integration of 11 datasets**: Covers single-person motion, dyadic interaction, and human–object interaction, totaling ~220 hours and surpassing existing largest motion datasets.

## Limitations & Future Work

- The Diversity metric is slightly lower than certain methods (e.g., MoMamba at 9.871), possibly because the multi-expert design tends to produce more stable outputs.
- Unifying 11 datasets requires substantial manual annotation and preprocessing (e.g., using ChatGLM to generate descriptions for BABEL; manual annotation for IMHD).
- No comparison with the latest diffusion-based motion generation methods (e.g., various-scale variants of OMG) under identical conditions.
- Evaluation is limited to the 22-joint SMPL skeleton; extension to the full body (including hands and face) is not explored.
- Inference speed and memory consumption are not reported.

## Related Work & Insights

- MoMask/MMM: masked completion paradigm for motion generation.
- MotionGPT: jointly models motion tokens as a new language with LLMs.
- OMG: large-scale diffusion model for motion generation.
- VLMo: source of inspiration for the multi-path Transformer design (vision–language domain).
- Insight: Scaling laws in motion generation are beginning to emerge; the combination of dedicated motion backbones and large-scale data holds substantial potential.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual multi-expert design of MEVQ-VAE and Multi-path Transformer is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluations on HumanML3D and IDEA400, VQ method comparisons, and component ablations are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and the training stage descriptions are well-organized.
- **Value**: ⭐⭐⭐⭐⭐ FID of 0.035 substantially advances the state of the art; the unified 11-dataset collection is a significant contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](../../ECCV2024/human_understanding/large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](../../CVPR2025/human_understanding/personabooth_personalized_text-to-motion_generation.md)
- [\[CVPR 2026\] Multi-level Causal LLM-based Text-to-Motion Generation with Human Alignment (MoTiGA)](../../CVPR2026/human_understanding/multi-level_causal_llm-based_text-to-motion_generation_with_human_alignment.md)

</div>

<!-- RELATED:END -->
