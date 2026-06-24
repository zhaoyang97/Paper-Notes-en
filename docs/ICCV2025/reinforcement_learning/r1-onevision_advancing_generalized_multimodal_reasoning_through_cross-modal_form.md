---
title: >-
  [Paper Note] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization
description: >-
  [ICCV 2025][Reinforcement Learning][multimodal reasoning] This paper proposes R1-Onevision, a framework that converts images into formalized textual representations via a cross-modal reasoning pipeline, combined with a two-stage post-training strategy of SFT followed by rule-based reinforcement learning (GRPO), to significantly enhance multimodal reasoning in vision-language models, surpassing GPT-4o on multiple mathematical reasoning benchmarks.
tags:
  - "ICCV 2025"
  - "Reinforcement Learning"
  - "multimodal reasoning"
  - "cross-modal formalization"
  - "vision-language model"
  - "reasoning benchmark"
date: 2026-05-08
content_hash: e6a18b6a9557d849
---

# R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization

**Conference**: ICCV 2025
**arXiv**: [2503.10615](https://arxiv.org/abs/2503.10615)  
**Code**: None  
**Area**: Reinforcement Learning / Multimodal Reasoning
**Keywords**: multimodal reasoning, cross-modal formalization, reinforcement learning, vision-language model, reasoning benchmark

## TL;DR

This paper proposes R1-Onevision, a framework that converts images into formalized textual representations via a cross-modal reasoning pipeline, combined with a two-stage post-training strategy of SFT followed by rule-based reinforcement learning (GRPO), to significantly enhance multimodal reasoning in vision-language models, surpassing GPT-4o on multiple mathematical reasoning benchmarks.

## Background & Motivation

While large language models have achieved remarkable progress in textual reasoning (e.g., DeepSeek-R1), multimodal reasoning remains a significant challenge. Existing vision-language models exhibit the following limitations when handling complex reasoning tasks:

**Perception errors**: DeepSeek-R1, for instance, relies on incomplete image descriptions from GPT-4o, leading to flawed reasoning foundations.

**Insufficient reasoning depth**: Qwen2.5-VL, despite strong multimodal capabilities, lacks deep reasoning and ultimately fails to solve problems.

**Limitations of templated reasoning**: Methods such as LLaVA-CoT employ predefined reasoning structures, constraining flexibility and creativity.

**Generalization issues from direct imitation**: Methods such as MAmmoTH-VL directly imitate ground-truth answers, lacking a trial-and-error process.

Furthermore, existing multimodal reasoning benchmarks (e.g., MathVision, MathVista) primarily focus on mathematical problems and lack comprehensive evaluation covering multiple disciplines and difficulty levels.

## Method

### Overall Architecture

The R1-Onevision framework consists of three components: (1) a cross-modal reasoning pipeline for dataset construction; (2) a two-stage post-training strategy of SFT + RL; and (3) R1-Onevision-Bench, a comprehensive reasoning benchmark.

### Key Designs

1. **Cross-Modal Reasoning Pipeline**: Image content is converted into formalized textual representations, enabling language reasoning models to precisely process visual information. Differentiated strategies are applied according to image type:

    - **Diagrams/flowcharts**: GPT-4o generates structured representations (SPICE circuits, PlantUML flowcharts, HTML layouts, CSV/JSON tables).
    - **Natural scenes**: Grounding DINO extracts bounding boxes + GPT-4o generates descriptive captions.
    - **Text-dominant images**: EasyOCR extracts text and positions + GPT-4o reconstructs the document.
    - **Mathematical images**: GPT-4o provides reasoning strategies.
    - **Reasoning process generation**: A role-playing strategy is adopted to iteratively review images and refine understanding; DeepSeek R1 is used on LlaVA-OneVision to generate reasoning chains.
    - **Quality assurance**: GPT-4o filters inaccurate or inconsistent CoT steps.

2. **R1-Onevision Dataset**: A total of 155K carefully curated samples spanning science, mathematics, charts, and general scenarios, each annotated with detailed step-by-step reasoning.

3. **Two-Stage Post-Training Strategy**:

    - **SFT stage**: Qwen2.5-VL is fine-tuned on the R1-Onevision dataset to cultivate coherent reasoning patterns and standardized output format (`<think>...</think>` structure).
    - **RL stage**: GRPO (Group Relative Policy Optimization) is applied on the CLEVR dataset with two defined rewards:
        - **Accuracy reward**: The final answer is extracted via regular expressions and compared against the ground truth.
        - **Format reward**: Ensures the reasoning process is correctly enclosed within `<think>` tags.
    - GRPO loss function: $\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}[\min(\text{ratio}_t \cdot \text{Adv}_t, \text{clipped\_ratio}_t \cdot \text{Adv}_t) - \beta \cdot \text{KL}(\pi_\theta(y|x), \pi_{\text{ref}}(y|x))]$

### Loss & Training

- SFT: batch size 128, learning rate 1e-5, trained for 1 epoch.
- RL: trained for 1 epoch on a 10K subset of CLEVR.
- Base models: Qwen2.5-VL-7B and Qwen2.5-VL-3B.

## Key Experimental Results

### Main Results

**Performance on mathematical reasoning benchmarks**:

| Model | MathVision | MathVerse(ALL) | MathVerse(Vision Only) | MathVista | WeMath |
|------|-----------|----------------|----------------------|-----------|--------|
| Qwen2.5-VL-7B (base) | 25.4 | 43.6 | 38.2 | 63.7 | 61.0 |
| GPT-4o | 30.6 | 41.2 | 34.5 | 60.0 | 69.0 |
| InternVL2.5-8B | 17.1 | 35.6 | 22.8 | 64.5 | 53.8 |
| LLaVA-CoT-11B | - | - | 22.6 | 52.5 | - |
| **R1-Onevision-7B** | **29.9** | **46.4** | **40.0** | **64.1** | **61.8** |

**R1-Onevision-Bench results (partial)**:

| Model | Avg. | Middle | High | College | Social | Math | Physics | Chemistry | Biology | Deduction |
|------|-----|------|------|------|------|------|------|------|------|------|
| GPT-4o | 49.6 | 51.3 | 56.2 | 45.3 | 26.5 | 41.3 | 52.5 | 71.4 | 63.4 | 26.5 |
| Gemini-2.0-Flash | 59.1 | 56.0 | 65.9 | 61.2 | 39.8 | 52.3 | 64.4 | 74.3 | 67.2 | 39.8 |
| Qwen2.5-VL-7B | 32.1 | 33.8 | 37.1 | 25.3 | 19.4 | 31.5 | 27.3 | 39.0 | 47.0 | 19.4 |
| **R1-Onevision-7B** | **36.2** | **40.1** | **39.5** | **27.6** | **26.5** | **33.0** | **30.2** | **49.5** | **53.0** | **26.5** |
| Qwen2.5-VL-72B | 52.0 | 54.3 | 56.7 | 54.1 | 23.5 | 48.9 | 55.8 | 63.8 | 63.4 | 23.5 |

### Ablation Study

**Training strategy ablation (based on Qwen2.5-VL-7B)**:

| Strategy | MathVision | MathVerse | MathVerse (Vision Only) |
|------|-----------|-----------|----------------------|
| Base | 25.4 | 43.6 | 38.2 |
| +SFT | 26.3 | 43.4 | 39.7 |
| +SFT+RL | **29.9** | **46.4** | **40.0** |
| RL only (w/o SFT) | 28.0 | - | - |

**Model scale ablation (Qwen2.5-VL-3B)**:

| Model | MathVision | MathVerse | MathVerse (Vision Only) |
|------|-----------|-----------|----------------------|
| Qwen2.5-VL-3B | 21.7 | 34.7 | 31.2 |
| R1-Onevision-3B | **23.7** | **38.6** | **35.5** |

### Key Findings

- R1-Onevision-7B surpasses GPT-4o by 5.2% and 4.1% on MathVerse and MathVista, respectively.
- SFT serves as an essential foundation for RL: SFT+RL outperforms RL-only by 1.9% on MathVision.
- All models perform poorly on deduction-type problems, with no model exceeding 40%.
- The 7B model after post-training substantially narrows the gap with closed-source large models.
- The method is effective at both 3B and 7B scales, validating scalability.

## Highlights & Insights

- **Core Idea of cross-modal formalization**: Converting images into structured textual representations (e.g., SPICE, PlantUML) enables language reasoning models to effectively "perceive" visual content, elegantly transforming visual reasoning into textual reasoning.
- **Role-playing reasoning strategy**: Iteratively reviewing images to simulate the human comprehension process yields more accurate results than single-pass description.
- **Complementarity of SFT and RL**: SFT establishes reasoning format and foundational capability, while RL further enhances generalization.
- **Education-level design of R1-Onevision-Bench**: Graded from middle school → high school → college → social examinations, providing an intuitive dimension for capability assessment.

## Limitations & Future Work

- The reasoning process generation relies on closed-source models such as GPT-4o and DeepSeek R1, resulting in high data construction costs.
- The RL stage is trained only on the CLEVR dataset (10K), limiting its scale.
- All models perform poorly on deduction-type problems, indicating that logical reasoning remains a bottleneck.
- 83.1% of benchmark questions are multiple-choice, providing insufficient evaluation of open-ended reasoning.
- Formalized descriptions depend on the accuracy of OCR and detection models, which may introduce errors.

## Related Work & Insights

- DeepSeek-R1: Demonstrates the powerful effect of RL on enhancing textual reasoning capability.
- LLaVA-CoT/LlamaV-o1: Pioneers of predefined reasoning structures.
- MAmmoTH-VL: Large-scale multimodal reasoning data construction.
- Insight: The key to visual reasoning may lie not in better visual encoders, but in transforming visual information into forms that language models can reason over efficiently.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cross-modal formalization pipeline and education-level benchmark design are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-benchmark evaluation, training strategy ablation, and model scale ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Unified contribution of dataset, model, and benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](../../ICML2025/reinforcement_learning/t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [\[ICLR 2026\] R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](../../ICLR2026/reinforcement_learning/r1-reward_training_multimodal_reward_model_through_stable_reinforcement_learning.md)
- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](../../ICLR2026/reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[ICML 2025\] Diving into Self-Evolving Training for Multimodal Reasoning](../../ICML2025/reinforcement_learning/diving_into_self-evolving_training_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
