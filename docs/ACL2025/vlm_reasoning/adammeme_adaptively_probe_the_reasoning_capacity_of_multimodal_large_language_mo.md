---
title: >-
  [Paper Note] AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness
description: >-
  [ACL 2025][VLM Reasoning][meme understanding] This work proposes AdamMeme—an adaptive evaluation framework based on multi-agent collaboration, which probes the reasoning capabilities and specific weaknesses of Multimodal Large Language Models (mLLMs) in harmful content understanding by iteratively generating more challenging meme samples.
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "meme understanding"
  - "harmfulness detection"
  - "multimodal evaluation"
  - "multi-agent"
  - "adaptive probing"
date: 2026-05-08
content_hash: 929c0f2a58b567fd
---

# AdamMeme: Adaptively Probe the Reasoning Capacity of Multimodal Large Language Models on Harmfulness

**Conference**: ACL 2025  
**arXiv**: [2507.01702](https://arxiv.org/abs/2507.01702)  
**Code**: [https://github.com/viczxchen/AdamMeme](https://github.com/viczxchen/AdamMeme)  
**Area**: Multimodal / VLM  
**Keywords**: meme understanding, harmfulness detection, multimodal evaluation, multi-agent, adaptive probing

## TL;DR

This work proposes AdamMeme—an adaptive evaluation framework based on multi-agent collaboration, which probes the reasoning capabilities and specific weaknesses of Multimodal Large Language Models (mLLMs) in harmful content understanding by iteratively generating more challenging meme samples.

## Background & Motivation

In the era of social media, multimodal memes have become an important medium for online communication. Many memes contain implicit hate, discrimination, or misleading content, requiring AI systems to accurately understand and identify their harmfulness. Multimodal Large Language Models (mLLMs) such as GPT-4V and LLaVA have made significant progress in vision-language understanding. However, their ability to understand the **implicit harmfulness** of memes requires systematic evaluation.

Existing benchmarks for meme harmfulness evaluation have several key limitations:

**Static Datasets**: Evaluations are based on fixed datasets (such as Hateful Memes) and cannot keep up with the dynamic evolution of internet memes.

**Model Agnosticism**: All models use the same test set, preventing in-depth probing tailored to specific models.

**Accuracy-centric**: Focusing only on correct/incorrect ratios fails to provide a fine-grained analysis of model weaknesses.

**Lack of Challenge**: Simple samples constitute the majority, making it difficult to differentiate the true capabilities of different models.

Key Challenge: Judging the harmfulness of memes requires complex multimodal reasoning (understanding image-text interactions, cultural metaphors, sarcasm, irony, etc.), whereas existing evaluation methods fail to fully uncover model deficiencies in these aspects.

The Key Insight of this work is **adaptive evaluation**: instead of testing all models with a fixed set of questions, challenging test samples are dynamically generated based on each model's weaknesses, akin to an "adaptive testing" concept. Core Idea: Through a multi-agent collaboration framework, one agent is responsible for generating challenging memes, while another agent evaluates the performance of the target model, working iteratively to gradually expose the reasoning blind spots of the model.

## Method

### Overall Architecture

AdamMeme is a three-stage iterative pipeline:
1. **Harmfulness Mining**: Mining memes with specific harmful types from the seed dataset and utilizing an LLM to generate "misbelief statements" to construct finer-grained evaluation dimensions.
2. **Model Scoring**: Having the target mLLM judge the harmfulness of mined memes, evaluating model performance by comparing the outputs with reference answers.
3. **Iterative Refinement**: Adaptively generating more targeted challenging samples based on the feedback from the model's performance to expose specific weaknesses of the model.

The entire process can be iteratively performed across multiple rounds, with each round feeding the model's error patterns back to the mining agent to generate more targeted test data.

### Key Designs

1. **Harmfulness Mining Agent**:

    - Function: Identifying harmful attributes from raw meme data and systematically decomposing the fine-grained dimensions of harmfulness.
    - Mechanism: First, the OCR-SAM tool is applied to erase the text inside the meme, obtaining a "pure image" version. Then, the LLM is leveraged to analyze the source of harmfulness in the original meme—whether it stems from the image itself, the text itself, or the image-text interaction. For each harmful dimension, a misbelief statement (e.g., "women should not work in tech") is generated to serve as an evaluation reference standard.
    - Design Motivation: The harmfulness of memes is often implicit rather than explicit, conveyed through the implicit relationships between image and text as well as cultural contexts. Separating image and text helps identify the source of harmfulness, while the misbelief statements provide a clear reference benchmark for evaluation.

2. **Model Scoring**:

    - Function: Evaluating the target mLLM's understanding of meme harmfulness from multiple perspectives.
    - Mechanism: Multi-level evaluation questions are designed: (a) binary classification—is it harmful? (b) harmfulness category—which type of harmfulness does it belong to? (c) reasoning explanation—why is it harmful? By comparing model outputs with reference answers (including misbelief statements), the evaluation agent scores the performance. Scoring not only considers the correctness of the final decision but also evaluates the rationality of the reasoning chain.
    - Design Motivation: Merely looking at the final judgment cannot reflect the model's depth of understanding. A model might "guess" the correct answer while having an entirely incorrect reasoning chain. Multi-dimensional scoring more accurately captures the true reasoning capabilities of the model.

3. **Iterative Refinement Agent**:

    - Function: Adaptively adjusting the difficulty and distribution of evaluation data based on the model's error patterns.
    - Mechanism: Analyzing the model's error cases from the previous evaluation round, extracting error patterns (e.g., "unable to identify sarcasm," "ignoring image-text contrast," "insensitive to specific cultural metaphors"), and then target-mining or generating more similar challenging samples from the seed data pool. As iterations progress, the evaluation set becomes increasingly "difficult" for the target model.
    - Design Motivation: Static evaluations only provide a "snapshot," whereas adaptive iterations can continuously probe the boundaries of model capabilities. Drawing inspiration from penetration testing, continuous pressure is applied to find the weak points of the system.

### Loss & Training

AdamMeme is an **evaluation framework** and does not involve model training or loss functions. Its core strategy lies in the information flow between multiple agents:
- Mining Agent → Scoring Agent: Passes mined memes and reference answers.
- Scoring Agent → Refinement Agent: Passes error pattern analyses of the model.
- Refinement Agent → Mining Agent: Passes harmfulness dimensions that require focused probing.

## Key Experimental Results

### Datasets

Three publicly available meme datasets are used:
- **MAMI**: Multimodal Misogyny Identification dataset.
- **HarM**: Harmful memes data from the MOMENTA project.
- **FHM**: Facebook Hateful Memes Challenge dataset.

### Main Results

| Target Model | Initial Accuracy | Post-iteration Accuracy | No. of Weakness Dimensions | Main Weaknesses Exposed |
|----------|-----------|-------------|-----------|---------------|
| GPT-4V | High | Significant Drop | Multiple | Understanding of cultural metaphors |
| LLaVA-1.5 | Medium | Obvious Drop | Multiple | Sarcasm and irony recognition |
| InstructBLIP | Low | Further Drop | Multiple | Image-text interaction reasoning |
| MiniGPT-4 | Low | Continuous Drop | Multiple | Implicit bias identification |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Without Iterative Refinement | Baseline | Equivalent to static evaluation, with limited discrimination power. |
| 1 Iteration Round | Significantly improves discrimination power | Begins to expose differences in capabilities between models. |
| 3 Iteration Rounds | Optimal | Model weaknesses are fully exposed; further iterations yield diminishing marginal returns. |
| Without Text Erasure | Decreased | Fails to distinguish the sources of harmfulness (image vs. text vs. interaction). |
| Without Misbelief Statement | Decreased | Evaluation reference is less precise, leading to decreased scoring consistency. |
| Single-agent vs. Multi-agent | Multi-agent is superior | Specialized division of labor improves the quality of mining and evaluation. |

### Key Findings
- **Different mLLMs exhibit different weakness patterns**: GPT-4V performs worse on culturally sensitive content, while LLaVA shows insufficient reasoning capacity on sarcastic memes.
- **Iterative refinement is effective**: After 3 rounds of iteration, AdamMeme can significantly decrease the models' effective accuracy, exposing weaknesses that static evaluations fail to uncover.
- **Image-text interaction is the greatest challenge**: All models perform the worst on memes that require understanding image-text contrast/contradiction relationships.
- **Model scale is not everything**: Medium-sized open-source models may outperform large-scale closed-source models in certain harmfulness dimensions.

## Highlights & Insights

- **Adaptive Evaluation Paradigm**: Breaking through the limitations of static benchmarks, this work introduces dynamic and personalized thinking into mLLM evaluation. This "test-adaptive" evaluation paradigm can be transferred to the evaluation of other NLP/CV tasks.
- **Multi-agent Division of Labor**: The three agents (Mining, Scoring, and Refinement) each perform their duties, maintaining a clear information flow. This serves as a reusable template for multi-agent collaborative frameworks.
- **Misbelief Statement Design**: Explicitly defining implicit harmfulness as "misbelief statements" not only provides evaluation reference benchmarks but also helps understand the nature of harmfulness.
- **Image-text Separation Analysis**: Erasing text using OCR-SAM to create text-free image versions provides an effective method for analyzing the harmfulness generated by image-text interaction.

## Limitations & Future Work

- **Biases in the Evaluation Agent**: Utilizing an LLM (such as GPT-4) as the evaluation agent can introduce inherent biases, potentially affecting the objectivity of scoring.
- **Cultural Limitations**: Based primarily on English meme datasets, there is insufficient coverage of memes in other languages and cultures, such as Chinese or Japanese.
- **Generation Quality**: Iterative refinement relies on the generation/selection capabilities of the agent; if the agent itself cannot generate sufficiently high-quality challenging samples, the bounds of the evaluation will be constrained.
- **Computational Overhead**: Multiple rounds of iteration require numerous API calls (involving interactions among multiple LLMs), resulting in higher costs.
- **Lack of Human Validation**: The reliability of automated evaluation needs further validation through manual annotations.

## Related Work & Insights

- **vs. Hateful Memes Challenge (Facebook)**: HMC represents a static benchmark combined with human-annotated baselines, whereas AdamMeme acts as a dynamic adaptive evaluation framework that can continuously mining model weaknesses.
- **vs. MM-SafetyBench**: MM-SafetyBench focuses on safety alignment and jailbreak attacks, whereas AdamMeme concentrates on evaluating harmfulness *understanding* capabilities, highlighting a difference in evaluation dimensions.
- **vs. Adversarial Attack Methods**: Adversarial attacks focus solely on "fooling the model," whereas AdamMeme not only identifies weak-point samples but also provides fine-grained capability analyses.

## Rating

- Novelty: ⭐⭐⭐⭐ The adaptive evaluation paradigm is a first in the meme domain, and the multi-agent design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across multiple models and multiple datasets, with detailed iteration analyses.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and supported by rich explanations.
- Value: ⭐⭐⭐⭐ Provides new insights into mLLM evaluation with highly transferable methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[ACL 2025\] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?](judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)
- [\[NeurIPS 2025\] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](../../NeurIPS2025/vlm_reasoning/flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)
- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](../../CVPR2025/vlm_reasoning/insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
