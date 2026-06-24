---
title: >-
  [Paper Note] Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems
description: >-
  [ICML 2025][VLM Reasoning][Bongard Problems] This paper systematically evaluates the abstract visual reasoning capabilities of 4 closed-source and 4 open-source MLLMs on three datasets: the classic synthetic Bongard Problems (BPs), Bongard HOI, and Bongard-OpenWorld. Seven problem-solving strategies and a new dataset, Bongard-RWR (which represents synthetic BP concepts using real-world images), are proposed, revealing that the extremely poor performance of MLLMs on synthetic…
tags:
  - "ICML 2025"
  - "VLM Reasoning"
  - "Bongard Problems"
  - "abstract visual reasoning"
  - "MLLM evaluation"
  - "Bongard-RWR"
  - "few-shot concept learning"
date: 2026-05-08
content_hash: eb29615f0f29c953
---

# Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems

**Conference**: ICML 2025  
**arXiv**: [2411.01173](https://arxiv.org/abs/2411.01173)  
**Code**: [GitHub](https://github.com/pavonism/bongard-rwr)  
**Area**: Multimodal Reasoning Evaluation / Abstract Visual Reasoning  
**Keywords**: Bongard Problems, abstract visual reasoning, MLLM evaluation, Bongard-RWR, few-shot concept learning

## TL;DR

This paper systematically evaluates the abstract visual reasoning capabilities of 4 closed-source and 4 open-source MLLMs on three datasets: the classic synthetic Bongard Problems (BPs), Bongard HOI, and Bongard-OpenWorld. Seven problem-solving strategies and a new dataset, Bongard-RWR (which represents synthetic BP concepts using real-world images), are proposed, revealing that the extremely poor performance of MLLMs on synthetic BPs is not due to domain shift but rather an inherent limitation in abstract reasoning.

## Background & Motivation

**Background**: Bongard Problems (BPs) are classic abstract visual reasoning tasks—given 6 images on the left and 6 on the right, the goal is to discover the common concept that distinguishes the two sides and describe it in natural language. This requires joint perception and reasoning, serving as a benchmark test for evaluating the abstract capabilities of AI.

**Limitations of Prior Work**:

1. Traditional deep learning approaches simplify BPs into binary classification tasks, bypassing the challenge of natural language answer generation.

2. Existing BP datasets containing real-world images (e.g., Bongard HOI/OpenWorld) feature different concepts than the classic synthetic BPs, making it impossible to directly compare the impact of domain shift.

3. The abstract reasoning capabilities of MLLMs have not been systematically evaluated—is it limited by the out-of-domain nature of synthetic images, or by a fundamental deficiency in reasoning capability itself?

**Core Problem**: Is the poor performance of MLLMs on synthetic BPs due to domain shift in the training data or inherent deficiencies in abstract reasoning?

## Method

### Overall Architecture

Seven problem-solving strategies tailored for MLLMs are designed (3 direct generation + 4 step-by-step reasoning steps) to evaluate 8 MLLMs across 4 BP datasets, with the newly constructed Bongard-RWR dataset used to control the domain shift variable.

### Key Designs

1. **7 Problem-Solving Strategies**

    - **Direct**: Directly inputs the complete BP matrix image and requires the model to output the answer in one go.
    - **Descriptive**: Describes each panel one by one, and then reasons for the answer based on the textual descriptions.
    - **Descriptive-iterative**: Iterates through descriptions of images on the same side within the same context window, progressively refining the descriptions.
    - **Descriptive-direct**: Descriptive + appending the complete matrix image to assist reasoning.
    - **Contrastive**: Compares the differences between corresponding images on the left and right sides, pair by pair.
    - **Contrastive-iterative**: Compares iteratively in the same context to gradually build understanding.
    - **Contrastive-direct**: Contrastive + appending the complete matrix image.
    - Key Discovery: The Descriptive strategy performs the best across all datasets, whereas the Contrastive strategy performs worse.

2. **Bongard-RWR Dataset Construction**

    - Goal: Represent the same abstract concepts as synthetic BPs (e.g., "convex vs. non-convex") using real-world images.
    - Pipeline: GPT-4o generates 10 real-world text descriptions $\rightarrow$ Pexels API image search $\rightarrow$ GPT-4o filtering $\rightarrow$ Manual refinement.
    - Final 60 problems: 12 fully automatically generated, 24 semi-automatically generated + manually adjusted, and 24 fully manually constructed.
    - Four variants provided: Original, RWR-S (square cropping), RWR-G (grayscale), and RWR-SG (square grayscale).

3. **Automated Answer Evaluation**

    - An ensemble of 4 closed-source MLLMs is utilized to judge whether the generated answer and the ground truth describe the same concept.
    - The answer is deemed correct if at least 2 models agree.
    - This avoids evaluation difficulties caused by the diversity of natural language answers.

### Evaluation Settings

- Closed-source models: GPT-4o, GPT-4 Turbo, Gemini 1.5 Pro, Claude 3.5 Sonnet
- Open-source models: InternVL2-8B, LLaVA-1.6 Mistral-7B, Phi-3.5-Vision, Pixtral 12B
- Datasets: 100 synthetic BPs + 100 Bongard HOIs + 100 Bongard-OpenWorld + 60 Bongard-RWR
- Human baseline: Average correct score of 39.2/60 (65%) on Bongard-RWR

## Key Experimental Results

### Main Results

**Number of Correct Natural Language Generations (Total dataset sizes in column headers)**

| Model | Synthetic BP /100 | HOI /100 | OpenWorld /100 | RWR /60 |
|------|-----------|---------|---------------|---------|
| GPT-4o (Descriptive) | 17 | **42** | **46** | 8 (13.3%) |
| GPT-4 Turbo (Desc.) | 15 | **45** | **57** | 5 (8.3%) |
| Claude 3.5 Sonnet (Desc.) | 19 | 44 | 53 | **13 (21.7%)** |
| Gemini 1.5 Pro (Desc.) | **21** | 40 | 32 | 7 (11.7%) |
| Pixtral 12B (Desc.) | 4 | 27 | 34 | 1 (1.7%) |
| InternVL2-8B | 0 | 2 | 18 | 0 (0%) |
| Human Baseline | - | - | - | 39.2 (65%) |

### Ablation Study

**Strategy Comparison (The number of unique problems solved collectively by all models)**

| Dataset | Descriptive Series | Contrastive Series |
|--------|-------------|--------------|
| Synthetic BP | 44 | 44 |
| Bongard HOI | **82** | 63 |
| Bongard-OpenWorld | **90** | 76 |
| Bongard-RWR | **20** | 11 |

### Key Findings

- **All MLLMs perform extremely poorly on synthetic BPs**: The best model (Gemini 1.5 Pro, Descriptive) only solves 21/100.
- **Bongard-RWR confirms that domain shift is not the main reason**: GPT-4o solves 42 problems on real-world Bongard HOI but only 8 on real-world Bongard-RWR, because Bongard-RWR retains the abstract concepts of synthetic BPs.
- **Descriptive >> Contrastive**: MLLMs are better at describing images one-by-one and then synthesizing reasoning rather than directly contrasting them—which is opposite to human reasoning patterns.
- **Closed-source >> Open-source**: Closed-source models lead in 35 out of 40 (dataset, strategy) combinations.
- **Iterative reasoning yields worse results**: Descriptive-iterative performs worse than Descriptive, indicating that current models struggle to effectively utilize historical information within the context window.
- Humans achieve a 65% accuracy rate on Bongard-RWR, whereas the best MLLM only achieves 21.7%.

## Highlights & Insights

- By using Bongard-RWR to elegantly control the "same concept, different domain" contrast variable, this work confirms that aggregate deficiencies in MLLM abstract reasoning are inherent.
- The systematic comparison of 7 problem-solving strategies reveals the inference mode preference of MLLMs: they are better suited for a pipeline of textual description $\rightarrow$ synthesized reasoning.
- The discovery that "contrastive reasoning is actually worse" highlights a fundamental difference between MLLMs and humans in analogical reasoning.
- Ablations on dataset variants (square/grayscale) show that removing color and cropping margins can improve classification accuracy.

## Limitations & Future Work

- Bongard-RWR contains only 60 problems, which is relatively small in scale.
- Automated evaluation relies on an MLLM ensemble to judge semantic equivalence, which may introduce bias.
- Finetuned MLLMs were not evaluated (e.g., whether finetuning on a subset of BPs improves reasoning).
- Although the 8 models cover the mainstream, more recent powerful models (such as GPT-4o-mini and Claude 3.5 Opus) are missing.
- Only the zero-shot setting is considered; few-shot in-context learning has not been explored.

## Related Work & Insights

- **vs Bongard-LOGO**: Bongard-LOGO uses synthetic data for binary classification, whereas ours uses MLLMs for natural language answer generation, which aligns more closely with the original definition of BPs.
- **vs Wüst et al. (2024)**: Concurrent work also evaluates MLLM performance on synthetic BPs, while ours additionally introduces Bongard-RWR to conduct domain-shift-controlled experiments.
- **vs ARC/RPM**: These tasks can be solved by LLMs after textualization, whereas BPs require joint visual perception and abstract reasoning.
- Insights: The "reasoning" capability of current MLLMs may rely more on pattern matching rather than genuine abstract concept formation.

## Rating

- Novelty: ⭐⭐⭐⭐ The system design of the Bongard-RWR dataset and 7 strategies constitutes a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale evaluation across 8 models $\times$ 4 datasets $\times$ 7 strategies.
- Writing Quality: ⭐⭐⭐⭐ Rigorous experimental design and in-depth analysis.
- Value: ⭐⭐⭐⭐ Reveals the inherent limitations of MLLMs in abstract reasoning, providing valuable guidance for future model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](../../NeurIPS2025/vlm_reasoning/can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)
- [\[ICML 2026\] MET-Bench: Multimodal Entity Tracking for Evaluating the Limitations of Vision-Language and Reasoning Models](../../ICML2026/vlm_reasoning/met-bench_multimodal_entity_tracking_for_evaluating_the_limitations_of_vision-la.md)
- [\[NeurIPS 2025\] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](../../NeurIPS2025/vlm_reasoning/flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](../../CVPR2025/vlm_reasoning/insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](../../NeurIPS2025/vlm_reasoning/affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
