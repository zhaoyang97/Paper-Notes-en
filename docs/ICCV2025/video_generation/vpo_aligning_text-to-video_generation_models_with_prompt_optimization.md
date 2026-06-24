---
title: >-
  [Paper Note] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization
description: >-
  [Video Generation] > This paper proposes the VPO framework, which systematically optimizes text prompts for video generation based on three core principles (Harmless, Accurate, Helpful). Through principle-guided SFT and multi-feedback preference optimization, VPO significantly improves the safety, alignment, and quality of generated videos.
tags:
  - "Video Generation"
date: 2026-05-08
content_hash: ebaf1bedc4f79416
---

# VPO: Aligning Text-to-Video Generation Models with Prompt Optimization

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.20491](https://arxiv.org/abs/2503.20491)
- **Authors**: Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, et al. (Tsinghua University CoAI, Zhipu AI, Tsinghua KEG)
- **Code**: [GitHub](https://github.com/thu-coai/VPO)
- **Area**: Video Understanding / Video Generation
- **Keywords**: Prompt Optimization, Video Generation, Text Alignment, RLHF, DPO, Safety, CogVideoX

## TL;DR

> This paper proposes the VPO framework, which systematically optimizes text prompts for video generation based on three core principles (Harmless, Accurate, Helpful). Through principle-guided SFT and multi-feedback preference optimization, VPO significantly improves the safety, alignment, and quality of generated videos.

## Background & Motivation

### Problem Definition
Video generation models are trained on carefully annotated detailed descriptions, yet user inputs at inference time are often short, vague, or poorly structured. Prompt optimization aims to bridge this gap by transforming user inputs into high-quality prompts for video generation.

### Limitations of Prior Work

**Safety risks**: Existing LLM in-context learning approaches do not explicitly ensure the safety of optimized prompts, potentially leading to harmful content generation.

**Intent distortion**: Prompt rewriting may inadvertently alter user intent or introduce undesired biases.

**Neglect of video quality**: Prior methods optimize the semantic richness of prompts without considering the impact on final video quality.

**LLM refusal problem**: LLMs may refuse to process queries containing sensitive keywords, including abstractly phrased inputs (e.g., "20 - 11 coins = 9 coins").

### Core Principles (Analogous to HHH in LLM Alignment)
- **Harmless**: Optimized prompts must not contain harmful content such as gore or violence.
- **Accurate**: User intent must be faithfully preserved; the original meaning should not be altered except for safety concerns.
- **Helpful**: Prompts should be detailed and descriptive to assist the model in generating high-quality videos.

## Method

### Overall Architecture
VPO consists of two stages:
1. **Principle-Based SFT**: Constructing a high-quality SFT dataset and fine-tuning a base model.
2. **Multi-Feedback Preference Optimization**: DPO training incorporating both text-level and video-level feedback.

### Stage 1: Principle-Based SFT

#### Query Filtering
- Source: VidProM dataset (1M+ real text-to-video queries)
- Rule-based filtering: keywords, special characters, length constraints
- Diversity filtering: self-BLEU deduplication
- Safety queries: extraction of queries with unsafe labels + LLM re-evaluation
- Final set: ~18k general queries + 2k safety-related queries

#### Initial Optimized Prompt Generation
GPT-4o with carefully designed few-shot examples is used to generate initial optimized prompts.

#### Principle-Driven Refinement
- LLM-as-a-judge evaluates prompts according to the three core principles
- Issues are identified (harmful content, missing key information, vague descriptions) and a critique $c$ is generated
- Refinement is performed based on the critique: $(x, p) \rightarrow (x, p_{refined})$

#### SFT Training
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(s|x, s_{<i})$$

### Stage 2: Multi-Feedback Preference Optimization

#### Text-Level Preference Data Construction
- For each query $x$, $K$ optimized prompts are sampled from the SFT model
- LLM-as-a-judge checks for principle violations
- Flawed prompts are refined to form preference pairs $(x, p_j < p_{j_{refined}})$

#### Video-Level Preference Data Construction
- Prompts passing the text-level check are used to generate videos
- VisionReward evaluates video quality scores $r_m$
- Preference pairs $(x, p_m < p_{m+1})$ are constructed based on score differences exceeding 0.5

#### DPO Training
$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,p_w,p_l) \sim D_{dpo}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(p_w|x)}{\pi_{ref}(p_w|x)} - \beta \log \frac{\pi_\theta(p_l|x)}{\pi_{ref}(p_l|x)} \right) \right]$$

- Training data: $D_{dpo} = D_{text} \cup D_{video}$
- Base model: LLaMA3-8B-Instruct

## Key Experimental Results

### Main Results — MonetBench & VBench

| Method | MonetBench Overall | Human Action | Scene | Multiple Objects | Appear. Style |
|------|------|------|------|------|------|
| **CogVideoX-2B** |
| Original Query | 3.27 | 80.00 | 28.34 | 40.17 | 22.60 |
| GLM-4 Few-Shot | 3.57 | 96.20 | 55.51 | 68.40 | 23.47 |
| GPT-4o Few-Shot | 3.58 | 98.20 | 52.53 | 63.63 | 23.73 |
| VPO-SFT | 3.59 | 97.00 | 55.04 | 68.98 | 24.13 |
| **VPO** | **3.76** | **99.00** | **55.83** | **70.17** | **24.20** |
| **CogVideoX-5B** |
| Original Query | 3.77 | 88.00 | 41.32 | 45.67 | 23.37 |
| GLM-4 Few-Shot | 3.98 | 98.40 | 55.60 | 72.38 | 24.39 |
| GPT-4o Few-Shot | 4.03 | 99.20 | 53.13 | 72.21 | 24.20 |
| VPO-SFT | 4.01 | 97.20 | 58.40 | 73.70 | 24.55 |
| **VPO** | **4.15** | **99.60** | **55.68** | **75.73** | **24.57** |

### Text Alignment Evaluation

| Method | Aligned↑ | Unsafe↓ | Imprecise↓ | Refusal↓ |
|------|----------|---------|------------|----------|
| GLM-4 Few-Shot | 83.4 | 5.4 | 10.0 | 1.2 |
| GPT-4o Few-Shot | 86.4 | 2.4 | 8.6 | 2.6 |
| VPO-SFT | 93.8 | 0.8 | 5.4 | 0.0 |
| **VPO (2B)** | **94.6** | **0.6** | **4.8** | **0.0** |
| **VPO (5B)** | **94.8** | **0.4** | **4.8** | **0.0** |

### Cross-Model Generalization — Open-Sora 1.2

| Method | Human Action | Scene | Multiple Objects |
|------|------|------|------|
| Original Query | 88.80 | 44.08 | 55.99 |
| GPT-4o Few-Shot | 92.40 | 53.21 | 65.02 |
| **VPO** | **97.00** | **53.58** | **67.88** |

### Key Findings
1. **VPO achieves substantial improvements**: On CogVideoX-5B, human evaluators prefer VPO over original queries by 37.5% and over the official prompt optimizer by 14%.
2. **Safety is greatly enhanced**: The unsafe rate drops from 5.4% (GLM-4) to 0.4% (VPO), with a marked increase in Level-1 full-safety rate.
3. **Text-level feedback is critical**: Removing text-level feedback not only reduces safety but also degrades general video generation quality.
4. **VPO outperforms Diffusion DPO**: As an RLHF-based approach, VPO surpasses Diffusion DPO, and the two methods are complementary.
5. **Cross-model generalization**: VPO trained on CogVideoX-2B directly improves performance on Open-Sora 1.2.
6. **Stable iterative optimization**: Iterative refinement stabilizes after three rounds without over-optimization or quality degradation.

## Highlights & Insights

1. **Principle-driven systematic methodology**: Rather than simple prompt rewriting, VPO introduces the HHH principles from LLM alignment into video prompt optimization, resulting in a methodologically rigorous framework.
2. **Multi-level feedback**: By jointly considering text-level feedback (safety + accuracy) and video-level feedback (quality), VPO forms a complete optimization loop.
3. **Prompt Optimization ≈ RLHF**: The paper reveals an important insight — optimizing prompts and optimizing the model are orthogonal and complementary strategies for aligning video generation models.
4. **Strong practicality**: VPO incurs low training overhead (based on LLaMA3-8B), is straightforward to deploy, and generalizes well across models.
5. **Safety must not be overlooked**: Case studies vividly demonstrate the risk of harmful content generation when using few-shot methods.

## Limitations & Future Work

1. **Dependence on a specific video generation model**: Video-level preference data relies on the target model to generate videos; switching models requires reconstructing the dataset.
2. **Reward model bias**: Video-level feedback depends on VisionReward and is subject to the biases inherent in that reward model.
3. **Reliance on GPT-4o**: SFT data construction and text-level evaluation both depend on GPT-4o.
4. **Prompt optimization only**: The video generation model itself is not optimized, so improvements are bounded by the model's inherent capabilities.

## Related Work & Insights

### Related Research
- **Video Generation**: CogVideoX, Open-Sora, Stable Video Diffusion, HunyuanVideo
- **Prompt Optimization**: AutoPrompt, Promptist, Prompt-A-Video
- **RLHF on Diffusion**: Diffusion DPO
- **Video Evaluation**: VBench, VisionReward, MonetBench

### Insights
- The training–inference data distribution gap is a common challenge across all generative models; prompt optimization offers a lightweight solution.
- Principle-based refinement (critique + revision) is an effective paradigm for constructing high-quality training data.
- Safety and quality are not at odds — improving safety can also enhance general performance.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
