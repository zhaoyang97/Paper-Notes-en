---
title: >-
  [Paper Note] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Multi-image Reasoning] This paper identifies a severe position bias in multi-image Large Vision-Language Models (LVLMs)—where open-source models place excessive emphasis on trailing images and closed-source models neglect middle images. It proposes a training-free SoFt Attention (SoFA) method that mitigates this bias by linearly interpolating between causal attention and bidirectional attention across images, improving average accuracy by 2~3% acro…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multi-image Reasoning"
  - "Position Bias"
  - "Attention Mechanism"
  - "Large Vision-Language Models"
  - "Training-free Method"
date: 2026-05-08
content_hash: 3ed30a7d32d34cb3
---

# Identifying and Mitigating Position Bias of Multi-image Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2503.13792](https://arxiv.org/abs/2503.13792)  
**Code**: [https://github.com/xytian1008/sofa](https://github.com/xytian1008/sofa)  
**Area**: Multimodal VLM  
**Keywords**: Multi-image Reasoning, Position Bias, Attention Mechanism, Large Vision-Language Models, Training-free Method

## TL;DR

This paper identifies a severe position bias in multi-image Large Vision-Language Models (LVLMs)—where open-source models place excessive emphasis on trailing images and closed-source models neglect middle images. It proposes a training-free SoFt Attention (SoFA) method that mitigates this bias by linearly interpolating between causal attention and bidirectional attention across images, improving average accuracy by 2~3% across multiple benchmarks.

## Background & Motivation

Multimodal large models have successfully expanded from single-image reasoning to multi-image reasoning, finding wide application in tasks such as difference detection, image counting, and video understanding. However, in the NLP domain, LLMs have been shown to suffer from "position bias" (such as the "lost in the middle" phenomenon) when processing multiple documents, meaning they tend to focus on the beginning and end of the input sequence while overlooking the middle information. The **Core Problem** of this paper is: **Does this position bias also exist in multi-image LVLMs?** Experiments show that simply swapping the order of input images can alter approximately 30% of predictions, with accuracy fluctuating by up to 10%. This severely undermines the robustness and reliability of the models. The authors' further analysis reveals that the **causal attention mechanism** among images is the root cause of this position bias—latter images can attend to all prior images, whereas the former images remain "isolated." **Core Idea**: Smooth the dependence of image tokens on position information by performing soft interpolation between causal attention and bidirectional attention.

## Method

### Overall Architecture

The authors first design a Position-wise Question Answering (PQA) task to quantitatively detect patterns of position bias, analyze the mechanistic origins of position bias (causal attention vs. positional encoding), and finally propose the SoFA method to mitigate this bias. SoFA is a plug-and-play training-free method that only requires modifying the attention masks between image tokens in the LVLM.

### Key Designs

1. **Position-wise Question Answering (PQA) Task**:
    - **Function**: Quantitatively evaluate the reasoning performance of LVLMs at each image position.
    - **Mechanism**: Instruct the model to independently answer the same question for each image (e.g., "How many cats are in each image?"), outputting a list formatted like $[3, 2, 0, ...]$, which allows calculating position-wise accuracy.
    - **Design Motivation**: Existing multi-image benchmarks only evaluate overall performance, failing to differentiate locations where models perform well or poorly. PQA achieves fine-grained positional analysis by constructing a position-neutral question set from VQAv2.

2. **Mechanistic Analysis of Position Bias**:
    - **Function**: Identify the root cause of position bias.
    - **Mechanism**: The two major factors affecting position information are positional encodings and causal attention. The authors compare three inter-image attention mechanisms: (A) causal attention (default, unidirectional), (B) isolated attention (no inter-image interaction), and (C) bidirectional attention (fully-connected across images). Experiments show that switching causal attention to either isolated or bidirectional attention significantly alleviates the bias, but leads to a drop in performance.
    - **Design Motivation**: Directly modifying positional encoding is too aggressive (as it ruins temporal understanding required in video understanding), whereas causal attention is a more moderate leverage point.

3. **SoFt Attention (SoFA)**:
    - **Function**: Alleviate multi-image position bias without retraining.
    - **Mechanism**: Perform linear interpolation on intermediate image attention masks: $\mathbf{M}_{\text{soft}} = (1-\sigma)\mathbb{1}_{\text{causal}} + \sigma\mathbb{1}_{\text{bidirectional}}$, where $\sigma$ controls the proportion of bidirectional attention. **Only physical attention among image tokens is modified, while causal attention between text tokens remains unchanged.**
    - **Design Motivation**: Completely switching to bidirectional attention deviates from the pre-training distribution and degrades performance, whereas interpolation achieves a balance between accuracy and robustness. SoFA is applied every two layers (instead of every layer) to better align with the training framework.

### Loss & Training

SoFA is a **training-free method** that involves no parameter updates. The hyperparameter $\sigma$ is optimized for each task using a 32-shot validation set. Models are run using FP16 precision and Flash Attention, with sub-image splitting disabled to ensure a fair comparison.

## Key Experimental Results

### Main Results

| Model | Benchmark | Inconsistent Prediction Rate w/o SoFA | Inconsistent Prediction Rate w/ SoFA | Reduction |
|------|------|-------------------|-------------------|---------|
| Idefics2 | BLINK | 41.55% | 12.36% | -29.19% |
| InternVL2 | MuirBench | 38.65% | 5.16% | -33.49% |
| LLaVA-NeXT | MIRB | 28.56% | 6.96% | -21.60% |

### Overall Performance Improvement

| Model | BLINK | Mantis-Eval | MuirBench | MIRB | NLVR2 | MVBench |
|------|-------|-------------|-----------|------|-------|---------|
| InternVL2 | 38.95 | 50.30 | 54.53 | 42.66 | 85.56 | 29.31 |
| InternVL2+SoFA | **43.26** | **51.11** | **57.14** | **46.19** | **88.19** | **32.77** |
| LLaVA-NeXT | 53.34 | 50.83 | 48.22 | 57.15 | 87.28 | 54.26 |
| LLaVA-NeXT+SoFA | **55.92** | **54.51** | **50.43** | **60.67** | **89.45** | **57.71** |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| Causal attention (default) | Severe bias, poor performance in front positions | Baseline |
| Isolated attention | Bias eliminated but performance drops drastically | Lack of inter-image interaction leads to OOD |
| Bidirectional attention | Alleviated bias but slight performance decline | Deviation from pre-training distribution |
| SoFA (interpolation) | Mitigated bias with performance gains | Optimal trade-off |
| 100-image long context | 49.19$\rightarrow$55.11% (+5.92%) | SoFA displays greater advantages in long context |

### Key Findings

- Open-source models exhibit **recency bias** (i.e., better performance on latter images, poorer performance on former images), whereas closed-source models (e.g., GPT-4o) exhibit a **U-shaped curve**, performing worst at the middle positions.
- As the number of images increases, the bias intensifies: with 20 images, the gap in accuracy between the front and end images for OpenFlamingo reaches 14%.
- SoFA yields the most significant improvements in visual retrieval and analogy tasks (+6.84% and +5.53%), where fully understanding the first reference image is crucial.
- In a 16-shot in-context learning scenario, SoFA improves VizWiz accuracy from 45.35% to 49.17%.

## Highlights & Insights

- **Clear Problem Definition**: This is the first systematic exploration of position bias in multi-image LVLMs, designing the PQA task for position-level evaluation.
- **Simple Yet Effective Method**: It only modifies the interpolation coefficient of the attention mask, which is training-free and plug-and-play.
- **In-depth Mechanistic Analysis**: Through visualization of attention distributions, it clearly demonstrates how SoFA disperses attention from being concentrated at the tail to the global scope.

## Limitations & Future Work

- $\sigma$ needs to be tuned on a validation set for each individual task, increasing operational cost.
- The bias is only alleviated rather than eradicated—some residual effects of causal attention remain.
- For tasks that inherently require chronological image sequences (e.g., temporal video understanding), the effectiveness of SoFA should be evaluated with caution.
- Joint application with modifications to positional encodings was not explored.

## Related Work & Insights

- Inspired by "lost in the middle" studies in NLP, this work transfers the position-bias problem to the multimodal domain.
- It shares similarities with position bias in LLM-as-a-judge (which tends to favor the first response).
- Insight from SoFA: For other inherent biases in LVLMs inherited from LLMs, similar "soft correction" strategies could be considered.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic study on the position bias of LVLMs with a proposed PQA evaluation framework, though the method itself (attention interpolation) is relatively simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 benchmarks, 5 open-source models + GPT-4o, and multiple scenarios (in-context learning, long context, task type analysis).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logically clear, progressing sequentially from phenomenon discovery to mechanistic analysis and then to the solution.
- **Value**: ⭐⭐⭐⭐ Unveils a significant flaw in LVLMs; the SoFA method is highly practical, though the 2~3% improvement is relatively modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)
- [\[CVPR 2025\] Joint Vision-Language Social Bias Removal for CLIP](joint_vision-language_social_bias_removal_for_clip.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](../../NeurIPS2025/multimodal_vlm/hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[CVPR 2025\] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models](calico_part-focused_semantic_co-segmentation_with_large_vision-language_models.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
