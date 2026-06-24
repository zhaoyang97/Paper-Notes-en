---
title: >-
  [Paper Note] Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder
description: >-
  [ACL 2025][Multimodal VLM][CLIP] This study systematically investigates why generative multimodal LLMs (e.g., LLaVA) drastically outperform CLIP on visual reasoning tasks despite using the exact same vision encoder, revealing that patch tokens, positional encodings, and prompt weighting are the key factors.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "CLIP"
  - "Multimodal LLM"
  - "Visual Reasoning"
  - "Spatial Reasoning"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 7d17a5f00ad02cc8
---

# Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder

**Conference**: ACL 2025  
**arXiv**: [2411.05195](https://arxiv.org/abs/2411.05195)  
**Code**: [GitHub](https://github.com/lst627/CLIP-Embeds)  
**Area**: Multimodal VLM  
**Keywords**: CLIP, Multimodal LLM, Visual Reasoning, Spatial Reasoning, Contrastive Learning

## TL;DR

This study systematically investigates why generative multimodal LLMs (e.g., LLaVA) drastically outperform CLIP on visual reasoning tasks despite using the exact same vision encoder, revealing that patch tokens, positional encodings, and prompt weighting are the key factors.

## Background & Motivation

**Background**: CLIP performs exceptionally well on zero-shot classification, yet struggles with compositional reasoning, spatial understanding, and fine-grained visual comprehension tasks.

**Limitations of Prior Work**: It is widely believed that the CLIP vision encoder discards critical information. However, generative MLLMs achieve excellent performance on these tasks using the same encoder, suggesting that the bottleneck does not lie within the encoder itself.

**Key Challenge**: With identical vision encoders and weights, details that CLIP fails to extract are successfully retrieved by generative MLLMs, indicating that the information extraction strategy is the determining factor.

**Goal**: Identify the core design choices that allow generative MLLMs to outperform CLIP, providing a roadmap for improving CLIP-style models.

**Key Insight**: Conduct controlled experiments on challenging benchmarks like What'sUp and Winoground to dissect factors such as training data, token usage, positional encodings, language models, and training objectives.

**Core Idea**: The superior performance of generative MLLMs stems from architectural designs (patch tokens, RoPE, and prompt weighting), rather than superior training data or more powerful text encoders.

## Method

### Overall Architecture

Using CLIP-ViT-L/14-336px and LLaVA-1.5-7B as the primary comparative subjects, this study systematically evaluates multiple visual reasoning benchmarks and performs ablation studies to analyze individual factors.

### Key Designs

1. **Token Usage Experiments**: Comparing [CLS] token vs. patch tokens. Utilizing patch token aggregation from PACL increases pair accuracy on What'sUp from 1.9% to 9.7%. Incorporating RoPE further boosts this to 22.3%.
2. **Training Data Experiments**: Fine-tuning CLIP/SigLIP/EVA-CLIP on LLaVA-1.5 training data, including hard negatives, still results in performance close to random guessing, demonstrating that data is not the primary driver.
3. **Text Encoder Experiments**: Using stronger LLM-translated text encoders (e.g., LLM2CLIP) is still insufficient to resolve the performance gap.
4. **Contrastive Fine-Tuning Experiments**: Converting LLaVA into a CLIP-like contrastive encoder through fine-tuning still yields better performance than native CLIP, proving that fine-grained visual reasoning does not rely on autoregressive loss.
5. **Prompting as Weighting**: Serving questions as prompts that thoroughly fuse with the image re-weights the image tokens, significantly boosting the extraction of relevant information.

### Evaluation Strategy

VQAScore is used as a unified evaluation protocol, defined as $P(\text{"Yes"} | \text{image}, \text{question})$, to ensure a fair comparison between generative MLLMs and CLIP. Individual accuracy and pair accuracy are both reported.

## Key Experimental Results

### Main Results (What'sUp Spatial Reasoning)

| Model | Left/Right Pairs | On/Under Pairs | Front/Behind Pairs |
|------|-----------------|----------------|-------------------|
| CLIP-ViT-L/14-336px | 1.9% | 23.3% | 7.8% |
| LLaVA-1.5-7B | 93.2% | 52.4% | 52.9% |
| Phi-3-V-3.8B | 95.1% | 58.3% | 26.5% |
| LLaMA-3-V-8B | 96.1% | 64.1% | 47.1% |
| Random Baseline | 25.0% | 25.0% | 25.0% |

### Ablation Study (CLIP + Various Improvements)

| Configuration | What'sUp A Pairs | What'sUp B Pairs |
|------|-----------------|-----------------|
| CLIP Original | 1.9% | 10.8% |
| + Patch Tokens | 9.7% | 9.8% |
| + PT + RoPE | 22.3% | 20.6% |
| + PT + RoPE + Multiple Text Tokens | 0.0% | 6.9% |
| + PT + RoPE + Stronger Text Encoder | 10.7% | 15.7% |

### Key Findings

- Patch tokens provide richer spatial information than the [CLS] token.
- RoPE positional encodings are crucial for spatial reasoning.
- Merely relying on a stronger text encoder or more training data is insufficient.
- MLLMs maintain their advantage even when converted to a contrastive setup, demonstrating that the architecture itself inherently holds the advantage.

## Highlights & Insights

- Groundbreaking discovery: The CLIP vision encoder actually retains sufficient visual information; the bottleneck lies in how the information is extracted.
- Carefully designed control experiments systematically eliminate potential factors, presenting a methodology highly worth studying.
- Fine-grained reasoning is not unique to generative models; contrastive models can also acquire this capability through architectural improvements.

## Limitations & Future Work

- The investigation primarily focuses on What'sUp spatial reasoning; generalization to other visual reasoning tasks remains to be verified.
- The vision encoder was frozen during control experiments without fully exploring the effects of unfreezing.
- The improved CLIP still exhibits a gap in absolute performance compared to full MLLMs.

## Related Work & Insights

- Complementary to existing works that improve CLIP, such as NegCLIP and SPARC.
- Offers direct guiding principles for VLM design, such as next-generation CLIP models.
- Insight: Architectural choices may be more critical than the scale of training data.

## Technical Details

- VQAScore definition: $P(\text{"Yes"} | \text{image}, \text{"Does this figure show 'text'? Please answer yes or no."})$
- CLIP utilizes the [CLS] token combined with cosine similarity for image-text matching.
- LLaVA architecture: CLIP vision encoder patch tokens → 2-layer MLP connector → generative language model.
- Contrastive fine-tuning experiment: projecting LLaVA's hidden states onto a contrastive embedding space via a projection layer.
- SigLIP shows no improvement even when fine-tuned with an unfrozen vision encoder, further eliminating the encoder itself as the bottleneck.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel perspective, systematically elucidating for the first time the root causes of the gap between CLIP and MLLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rigorously designed control experiments covering multiple benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exceptionally clear logic with a roadmap-style experimental showcase that is easy to follow.
- **Value**: ⭐⭐⭐⭐ Offers practical guiding value for VLM architectural design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](../../ICCV2025/multimodal_vlm/are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)
- [\[NeurIPS 2025\] FlySearch: Exploring how vision-language models explore](../../NeurIPS2025/multimodal_vlm/flysearch_exploring_how_vision-language_models_explore.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[CVPR 2025\] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](../../CVPR2025/multimodal_vlm/florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)
- [\[ACL 2025\] Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging](exploring_compositional_generalization_of_multimodal_llms_for_medical_imaging.md)

</div>

<!-- RELATED:END -->
