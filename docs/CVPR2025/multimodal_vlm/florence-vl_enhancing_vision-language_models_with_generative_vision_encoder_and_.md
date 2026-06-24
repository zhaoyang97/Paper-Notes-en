---
title: >-
  [Paper Note] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion
description: >-
  [CVPR 2025][Multimodal VLM][Generative Vision Encoder] This work replaces CLIP with the generative vision foundation model Florence-2 as the vision encoder for VLMs. Through "Depth-Breadth Fusion" (DBFusion), it integrates low-level DaViT features with high-level features from three task prompts (caption, OCR, and grounding), achieving performance that surpasses multi-encoder approaches using only a single encoder with 576 tokens.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Generative Vision Encoder"
  - "Florence-2"
  - "Depth-Breadth Fusion"
  - "Multi-Task Feature Extraction"
  - "OCR Enhancement"
date: 2026-05-08
content_hash: 0ed01be04322eb7e
---

# Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion

**Conference**: CVPR 2025  
**arXiv**: [2412.04424](https://arxiv.org/abs/2412.04424)  
**Code**: [https://github.com/JiuhaiChen/Florence-VL](https://github.com/JiuhaiChen/Florence-VL)  
**Area**: Multimodal VLM  
**Keywords**: Generative Vision Encoder, Florence-2, Depth-Breadth Fusion, Multi-Task Feature Extraction, OCR Enhancement

## TL;DR
This work replaces CLIP with the generative vision foundation model Florence-2 as the vision encoder for VLMs. Through "Depth-Breadth Fusion" (DBFusion), it integrates low-level DaViT features with high-level features from three task prompts (caption, OCR, and grounding), achieving performance that surpasses multi-encoder approaches using only a single encoder with 576 tokens.

## Background & Motivation

**Background**: Current mainstream VLMs (e.g., LLaVA, Cambrian) employ vision encoders pretrained via contrastive learning, such as CLIP or SigLIP. To compensate for the limitations of a single encoder, some methods (e.g., Cambrian) adopt multi-encoder fusion strategies.

**Limitations of Prior Work**: CLIP generates a single global feature via contrastive learning, lacking specialized capabilities for different visual tasks (such as OCR, grounding, and scene description). Although multi-encoder schemes mitigate this deficiency, they introduce substantial parameter overhead and inference costs.

**Key Challenge**: Vision encoders in VLMs must simultaneously possess various perceptual capabilities (e.g., text recognition, spatial localization, and global understanding), whereas traditional contrastive learning encoders only learn a single, unified vision-language alignment representation.

**Goal**: To replace CLIP with a single generative vision model that provides multiple task-level visual features, achieving the effect of "one encoder performing the work of multiple encoders."

**Key Insight**: Florence-2 is a generative vision foundation model capable of extracting visual features for different tasks (e.g., OCR, detailed description, and region localization) via different text prompts. This multi-task capability makes it naturally suited as a visual front-end for VLMs.

**Core Idea**: Leveraging the multi-prompt feature extraction capability of Florence-2, the proposed method concatenates low-level and multi-task high-level features along the channel dimension (DBFusion), achieving efficient and comprehensive visual encoding with only 576 tokens.

## Method

### Overall Architecture
Input image $\rightarrow$ DaViT backbone of Florence-2 extracts low-level features $V$ $\rightarrow$ Three task prompts are fed into the Florence-2 encoder to extract high-level features $V'_{t1}$ (detailed description), $V'_{t2}$ (OCR), and $V'_{t3}$ (dense region description) $\rightarrow$ Concatenate along the channel dimension as $[V, V'_{t1}, V'_{t2}, V'_{t3}]$ $\rightarrow$ MLP projection to the LLM input space $\rightarrow$ 576 visual tokens are fed into Phi 3.5 or LLaMA 3.

### Key Designs

1. **Breadth Dimension: Three-Task Prompt Extraction**:

    - **Function**: Extracting visual features with different perceptual focuses from the same image.
    - **Mechanism**: Utilizing the generative architecture of Florence-2 (which accepts text prompts to control the output), three prompts are designed: "describe what is shown in the image with a paragraph" (global semantics), "provide the text shown in the image" (textual information), and "locate the objects in the image, with their descriptions" (spatial relationships). Each prompt generates a distinct 576-token feature map through the Florence-2 encoder.
    - **Design Motivation**: Ablation studies show that removing OCR features causes the most significant performance drop (avg 57.3 vs. 58.3 full), validating the complementarity of multi-task features. This also explains why Florence-VL substantially outperforms prior works on document-related tasks.

2. **Depth Dimension: Low-Level to High-Level Feature Fusion**:

    - **Function**: Preserving low-level detailed details to compensate for the limitations of high-level semantic features.
    - **Mechanism**: The low-level features $V$ from the DaViT backbone capture low-level details such as textures and edges, which are complementary to the high-level semantic features output by the encoder. Direct concatenation preserves information from both levels.
    - **Design Motivation**: Ablation studies show that using only the low-level features $[V]$ yields an OCRBench score of only 31.2; incorporating the three high-level features boosts this score to 41.4 (+32.7%), and DocVQA increases from 27.9 to 44.5 (+59.5%).

3. **Channel-Dimension Concatenation Fusion Strategy**:

    - **Function**: Fusing multiple features while keeping the token count constant.
    - **Mechanism**: Three fusion strategies were compared: Token Concatenation (concatenation along the token dimension, resulting in 1728 tokens, which is slow), Average Pooling (576 tokens, leading to information loss), and Channel Concatenation (concatenation along the channel dimension, resulting in 576 tokens while preserving all information). Channel concatenation achieves the best performance under the same token count budget.
    - **Design Motivation**: Although token concatenation preserves all information, the 3$\times$ token count incurs heavy inference costs; average pooling discards task-specific details. Channel concatenation allows the LLM to learn how to integrate information from different channels via a linear projection layer.

### Loss & Training
Two-stage training is adopted. In the pretraining stage, full-parameter training (Florence-2 + projection + LLM) is conducted on 16.9M image-text caption data using 64$\times$H100 GPUs. In the instruction-tuning stage, only the projection layer and LLM are trained on 10M instruction data. A key difference: unlike LLaVA 1.5 which freezes the vision encoder during pretraining, Florence-VL fine-tunes the vision encoder throughout the entire training process.

## Key Experimental Results

### Main Results

| Method | Encoder | Tokens | MMBench | POPE | MM-Vet | TextVQA | OCRBench | DocVQA |
|------|--------|--------|---------|------|--------|---------|----------|--------|
| LLaVA-Next 8B | CLIP | 2880 | 72.2 | 86.6 | 41.7 | - | - | - |
| Cambrian 8B | Multi-encoder | 576 | 75.9 | 87.4 | 48.0 | 71.7 | 62.4 | 77.8 |
| **Florence-VL 8B** | Florence-2 | 576 | **76.2** | **89.9** | **56.3** | **74.2** | **63.4** | **84.9** |

### Ablation Study

| Configuration | MMBench | POPE | OCRBench | DocVQA | Description |
|------|---------|------|----------|--------|------|
| Low-level [V] only | 64.3 | 86.1 | 31.2 | 27.9 | Low-level features are insufficient |
| Full DBFusion | **66.1** | **89.4** | **41.4** | **44.5** | Depth + Breadth |
| Without OCR features | 65.6 | 88.8 | 35.2* | 42.1* | OCR has the greatest impact |
| Without description features | 64.9 | 89.3 | - | - | MM-Vet drops by 3.4 |
| Token Concatenation (1728) | 66.6 | 88.7 | 40.8 | 44.6 | 3x tokens offer no obvious advantage |

### Key Findings
- **Florence-2 achieves the best alignment with LLM**: Quantitative analysis indicates that Florence-2 yields a lower alignment loss than CLIP, SigLIP, DINOv2, and Stable Diffusion encoders.
- **Single encoder surpasses multi-encoder**: Florence-VL 8B (using a single Florence-2) comprehensively outperforms Cambrian 8B (multi-encoder fusion), demonstrating that a high-quality generative encoder can substitute for a combination of multiple encoders.
- **OCR features are key to document tasks**: Under the LLaVA 1.5 setting, Florence-VL improves DocVQA from 28.1 to 44.5 (+58%) and TextVQA from 58.2 to 62.8 (+8%), which is almost entirely attributable to the OCR prompt features.
- **Knowledge-oriented benchmarks depend on the LLM rather than vision**: Ablation studies show that different combinations of visual features have minimal impact on knowledge-based benchmarks such as MMMU, indicating that the bottleneck for these tasks lies within the LLM's internal knowledge.

## Highlights & Insights
- **A paradigm shift from contrastive to generative encoders**: The multi-prompt feature extraction capability of Florence-2 is unattainable for contrastive learning models, suggesting that the visual encoders of VLMs should pivot from contrastive learning to generative pretraining.
- **Simplicity and efficiency of channel-dimension concatenation**: Compared with Cambrian's complex SVA fusion, a straightforward channel concatenation followed by a linear projection achieves superior performance, proving that fusion strategies do not need to be overly engineered.
- **Importance of full-parameter pretraining**: In contrast to LLaVA 1.5's approach of freezing the encoder, Florence-VL's fine-tuning of the encoder throughout the training process yields substantial performance gains.

## Limitations & Future Work
- Florence-2 is a 0.23B encoder, which is smaller than CLIP ViT-L (0.3B). However, with three forward passes to extract high-level features, the practical inference overhead is approximately 4$\times$ that of a single encoding pass.
- The selection of the three prompts is manually designed, and whether more or alternative prompts could yield further improvements remains unexplored.
- The method is validated only on Phi 3.5 and LLaMA 3, leaving compatibility with other LLM backbones unknown.
- Head-to-head comparisons with the latest high-resolution VLMs (e.g., InternVL2, Qwen2-VL) under identical resolution settings are currently lacking.

## Related Work & Insights
- **vs Cambrian**: Cambrian achieves its performance only by combining multiple encoders such as CLIP, DINOv2, and SigLIP. In contrast, Florence-VL outperforms it using a single Florence-2 with the same token count (576).
- **vs LLaVA-Next**: LLaVA-Next requires 2880 tokens to match Florence-VL's 576 tokens on certain benchmarks. This 5$\times$ token reduction demonstrates that the quality of the encoder is more critical than the sheer quantity of tokens.
- **vs InternVL2**: InternVL2 utilizes InternViT (6B), which is much larger than Florence-2 (0.23B). Yet, the two exhibit comparable results across several benchmarks, suggesting that the pretraining paradigm (generative vs. contrastive) might be more crucial than model size.

## Rating
- **Novelty**: ⭐⭐⭐⭐ This work is the first to systematically apply a generative vision foundation model as a VLM encoder, and the DBFusion design is intuitive yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ It evaluates on more than 15 benchmarks, provides detailed depth/breadth/strategy ablation analyses, and conducts fair comparisons with multi-encoder schemes.
- **Writing Quality**: ⭐⭐⭐⭐ The methodological motivation is clearly articulated, and the ablation experiments are well-structured.
- **Value**: ⭐⭐⭐⭐⭐ It offers a new paradigm for vision encoder selection in VLMs; the Florence-2 + DBFusion combination can be directly applied to other VLM frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models](spa-vl_a_comprehensive_safety_preference_alignment_dataset_for_vision_language_m.md)
- [\[CVPR 2025\] VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge](vila-m3_enhancing_vision-language_models_with_medical_expert_knowledge.md)
- [\[ACL 2025\] Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](../../ACL2025/multimodal_vlm/exploring_how_generative_mllms_perceive_more.md)
- [\[ICCV 2025\] EVEv2: Improved Baselines for Encoder-Free Vision-Language Models](../../ICCV2025/multimodal_vlm/evev2_improved_baselines_for_encoderfree_visionlanguage_mode.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](../../NeurIPS2025/multimodal_vlm/sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)

</div>

<!-- RELATED:END -->
