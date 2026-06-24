---
title: >-
  [Paper Note] Scaling Large Motion Models with Million-Level Human Motions
description: >-
  [ICML 2025][Human Understanding] This paper introduces MotionLib (the first million-level motion dataset, containing 1.2 million sequences), MotionBook (comprising lossless features and a 2D lookup-free motion tokenizer), and Being-M0 (a large motion model), demonstrating the scaling laws of both data and model size in the motion generation field for the first time.
tags:
  - "ICML 2025"
  - "Human Understanding"
date: 2026-05-08
content_hash: d3047a0cbc196937
---

# Scaling Large Motion Models with Million-Level Human Motions

**Conference**: ICML 2025  
**arXiv**: [2410.03311](https://arxiv.org/abs/2410.03311)  
**Area**: Human Understanding  

## TL;DR

This paper introduces MotionLib (the first million-level motion dataset, containing 1.2 million sequences), MotionBook (comprising lossless features and a 2D lookup-free motion tokenizer), and Being-M0 (a large motion model), demonstrating the scaling laws of both data and model size in the motion generation field for the first time.

## Background & Motivation

Text-to-motion (T2M) generation is an emerging field with widespread applications in games, movies, and robotics. However, current methods are constrained by the scale of available data:

- **Enormous Data Quantity Gap**: The largest existing motion dataset, Motion-X, contains only ~80K sequences, whereas vision-language datasets (e.g., ImageNet) are orders of magnitude larger.
- **Defects of Existing VQ Tokenizers**:
  1. **Information Loss**: Compressing complex motion states (containing joint positions, velocities, ground contacts, etc.) into a single 1D embedding.
  2. **Limited Codebook Capacity**: Small codebooks restrict the diversity of the generated motions.
- **Feature Representation Issues**: The commonly used H3D format omits raw rotation information, requiring time-consuming reconstruction methods.

## Method

### MotionLib Dataset

The first million-level motion dataset, containing **1.21 million motion sequences** and **2.48 million text descriptions**:

| Dataset | Sequences | Texts | Hours | Text Type |
|---|---|---|---|---|
| HumanML3D | 29.2K | 89K | 28.6 | body |
| Motion-X | 81.1K | 142K | 144.2 | body |
| **MotionLib** | **1.21M** | **2.48M** | **1456.4** | **Hierarchical** |

**Construction Pipeline:**
1. Collect over 20 million videos from public datasets and YouTube.
2. Extract SMPL parameters in the world coordinate system using WHAM.
3. Generate **hierarchical text annotations**: body-part-level descriptions (e.g., left arm) + full-body-level descriptions (1-3 sentences).
4. Refine raw motions using an RL policy $\pi_{\text{refine}}$ to ensure adherence to physical laws.

### MotionBook: Efficient Motion Encoding

#### Lossless Motion Features (SMPL-D135)

Each frame is encoded as $m \in \mathbb{R}^{135}$:
- **Root Node (9D)**: 6D rotation $\mathbf{r}_{rot} \in \mathbb{R}^6$, 2D XZ-plane velocity, and 1D height.
- **Body Joints (126D)**: 6D rotation vectors of 21 key joints $\mathbf{j}^r \in \mathbb{R}^{21 \times 6}$.

Compared to the H3D format (263D), SMPL-D135 is more compact and retains complete rotation information.

#### 2D Lookup-Free Quantization (2D-LFQ)

**Core Innovation:**
1. Treat the motion sequence as a single-channel image $\mathcal{M} \in \mathbb{R}^{T \times D \times 1}$.
2. Divide the feature dimension into $P$ components, which are encoded separately (e.g., root orientation, joint rotation, foot contact, etc.).
3. Represent the encoder output as $\mathbb{E}(\mathcal{M}) \in \mathbb{R}^{\lfloor T/\alpha \rfloor \times P \times d}$.

**Lookup-Free Quantization**: Replace codebooks with an integer set $\mathbb{C} = \times_{i=1}^d C_i$, where $C_i = \{-1, 1\}$:

$$Q(z_i) = -\mathbb{1}\{z_i \leq 0\} + \mathbb{1}\{z_i > 0\}$$

The token index is computed as $Index(z) = \sum_{i=1}^d 2^{i-1}\mathbb{1}\{z_i > 0\}$. This scales the codebook size by at least **two orders of magnitude** (from ~512 to ~65K+), while avoiding codebook collapse.

### Being-M0: Large Motion Model

An autoregressive motion generation model based on a pretrained LLM:

**Two-stage Training:**
1. **Motion-Text Alignment**: Pretraining on the entire MotionLib dataset to learn basic motion-text associations.
2. **Motion Instruction Tuning**: Fine-tuning using over 250 instruction templates and 900K instruction-following data points.

**Training Loss:**

$$\mathcal{L}(\Theta) = -\sum_{j=1}^{L}\log P_\Theta(y_j | desc, \hat{y}_{1:j-1})$$

## Key Experimental Results

### Scaling Law Experiments

| Decoder | Instructions | Parameters | MotionLib-eval FID ↓ |
|---|---|---|---|
| GPT-2 | 0.02M | 355M | 30.612 |
| GPT-2 | 1.2M | 355M | 6.936 |
| LLaMA-3 | 0.02M | 8B | 29.257 |
| LLaMA-3 | 0.08M | 8B | 21.295 |
| LLaMA-3 | 0.5M | 8B | 8.973 |
| LLaMA-3 | 1.2M | 8B | **6.029** |
| LLaMA-2 | 1.2M | 13B | 6.221 |

**Key Findings**:
- Increasing the data scale from 0.02M to 1.2M results in a significant drop in FID from ~30 to ~6, demonstrating a clear scaling behavior.
- Larger models yield consistent improvements (LLaMA-3 8B outperforms GPT-2 355M), but **the impact of data scale is far more significant than that of model scale**.

### Motion Tokenizer Comparison

2D-LFQ scales the codebook size from 512 in traditional VQ methods to 65536+ while keeping the MPJPE error low, achieving nearly 100% codebook utilization.

### Generalization Capability

While existing models struggle on out-of-domain concept tests within MotionLib, Being-M0 demonstrates significantly better generalization capabilities on unseen motion categories.

## Highlights & Insights

- **First Million-Level Motion Dataset**: 1.2M sequences and 2.48M text annotations, which is more than 15 times larger than existing datasets.
- **First Demonstration of Scaling Laws in Motion Generation**: Both data scale and model size are proven to effectively reduce generation errors.
- **Innovative 2D Lookup-Free Quantization**: Scales the codebook size by two orders of magnitude, fundamentally resolving the limitations of codebook capacity.
- **Lossless Feature Design**: SMPL-D135 is more compact than H3D while preserving complete rotation information.
- **Hierarchical Text Annotations**: Body-part-level + full-body-level descriptions provide unprecedented textual granularity.

## Limitations & Future Work

- The quality of motions extracted from millions of web videos is highly uneven, necessitating an additional RL refinement strategy.
- Some fast or intense motions still suffer from foot-sliding issues even after RL refinement.
- The dataset primarily focuses on single-person scenarios, offering limited support for multi-person interactions.
- The accuracy of SMPL parameters extracted from 2D videos remains limited compared to high-end MoCap data.
- The computational overhead and inference cost for large LLM backbones are relatively high.

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is a landmark work in the field of motion generation. The construction of the million-level dataset, the first demonstration of scaling laws, and the innovative 2D-LFQ tokenizer are all significant contributions. The paper demonstrates immense engineering effort and highly systematic experiments, laying a solid foundation for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](../../ICCV2025/human_understanding/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](../../CVPR2025/human_understanding/chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2026\] LLaMo: Scaling Pretrained Language Models for Unified Motion Understanding and Generation with Continuous Autoregressive Tokens](../../CVPR2026/human_understanding/llamo_scaling_pretrained_language_models_for_unified_motion_understanding_and_ge.md)
- [\[ICLR 2026\] EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?](../../ICLR2026/human_understanding/emoprefer_can_large_language_models_understand_human_emotion_preferences.md)

</div>

<!-- RELATED:END -->
