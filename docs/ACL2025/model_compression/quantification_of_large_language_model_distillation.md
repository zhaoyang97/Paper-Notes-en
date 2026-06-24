---
title: >-
  [Paper Note] Quantification of Large Language Model Distillation
description: >-
  [ACL 2025][Model Compression][Knowledge Distillation Quantification] This paper proposes two complementary LLM distillation quantification methods: Identity Consistency Evaluation (ICE) and Response Similarity Evaluation (RSE). By utilizing jailbreak attacks to uncover identity leakage and multi-granular response similarity, these methods measure the degree of model distillation. The results show that most well-known LLMs (except Claude, Doubao…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Knowledge Distillation Quantification"
  - "Identity Consistency"
  - "Response Similarity"
  - "LLM Homogenization"
  - "Jailbreak Attack"
date: 2026-05-08
content_hash: 02924eecacd0d5e0
---

# Quantification of Large Language Model Distillation

**Conference**: ACL 2025  
**arXiv**: [2501.12619](https://arxiv.org/abs/2501.12619)  
**Code**: [https://github.com/Aegis1863/LLMs-Distillation-Quantification](https://github.com/Aegis1863/LLMs-Distillation-Quantification)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation Quantification, Identity Consistency, Response Similarity, LLM Homogenization, Jailbreak Attack

## TL;DR
This paper proposes two complementary LLM distillation quantification methods: Identity Consistency Evaluation (ICE) and Response Similarity Evaluation (RSE). By utilizing jailbreak attacks to uncover identity leakage and multi-granular response similarity, these methods measure the degree of model distillation. The results show that most well-known LLMs (except Claude, Doubao, and Gemini) exhibit a high degree of distillation.

## Background & Motivation
Model distillation has become a foundational technique for constructing LLMs, significantly reducing costs through knowledge transfer from strong teacher models to weaker student models. However, distillation also introduces the risk of model homogenization—models developed by different teams become increasingly similar, reducing diversity and weakening their capacity to handle complex or novel tasks.

The Key Challenge currently faced is: the distillation process is opaque, lacking standardized benchmark datasets, and distilled knowledge may be embedded within representations in abstract forms that are difficult to interpret directly. More critically, while the academic community widely uses distilled data, it lacks a critical examination of its associated issues. Our Key Insight is: **systematically quantifying the degree of LLM distillation from two perspectives—identity cognitive contradiction and response similarity**, providing tools for the transparency and independence of LLM development.

## Method

### Overall Architecture
The paper proposes two complementary evaluation metrics: ICE, which detects whether the model accidentally inherits the teacher model's identity information during distillation, and RSE, which measures the similarity between the target model's and reference model's responses. Combining both provides a comprehensive distillation quantification assessment.

### Key Designs
1. **Identity Consistency Evaluation (ICE)**:

    - Mechanism: If model A is distilled from model B, A may accidentally learn B's identity information (e.g., name, developer).
    - Utilizing the open-source GPTFuzz jailbreak framework to iteratively generate adversarial prompts to bypass the models' self-awareness constraints.
    - Defining a fact set F containing the identity descriptions of each source model (e.g., "I am Claude, developed by Anthropic").
    - Three-level evaluation metrics:
        - **Loose Score**: Any identity contradiction is considered a successful attack.
        - **Strict Score**: Only counted when the model incorrectly identifies itself as another known entity.
        - **Hard Score**: The most stringent metric, requiring that the prompt contains no identity keywords while the response does (filtering out context-induced prompting).
    - Attack prompts cover 5 areas: team affiliation, partnerships, industry involvement, technical expertise, and geographical information.

2. **Response Similarity Evaluation (RSE)**:

    - Mechanism: The response style, logical structure, and content details of the distilled model will be similar to those of the teacher model.
    - Utilizing GPT-4o-0806 as the reference model (as the GPT series is the most common distillation source).
    - Three evaluation datasets: ArenaHard (general reasoning), Numina (mathematical reasoning), and ShareGPT (instruction following).
    - Adopting the LLM-as-a-judge approach, rating similarity on a 1-5 scale.
    - Assessing from three dimensions: style, logic, and content.
    - Compared to traditional n-gram similarity and BERTScore, RSE can capture logic-level information.

3. **Validation and Comparative Analysis**:

    - Validating RSE effectiveness on Qwen2.5-7B-Instruct using SFT: as the SFT epoch increases, the RSE score consistently rises.
    - Base model vs. Instruct model comparison.
    - Evaluation of reasoning models (such as DeepSeek-R1).

### Loss & Training
This paper presents an evaluation method and does not involve training. ICE utilizes the MCTS algorithm of GPTFuzz to iteratively optimize attack prompts, starting with 50 seed prompts and selecting a subset for optimization in each step. RSE uses LLM judges for scoring.

## Key Experimental Results

### Main Results - ICE Results

| Model | Loose Score | Strict Score | Hard Score | Distillation Degree |
|------|------------|-------------|------------|---------|
| Claude3.5-Sonnet | Very Low | Very Low | Very Low | Low |
| Doubao-Pro-32k | Very Low | Very Low | Very Low | Low |
| Gemini-2.0-Flash | Low | Low | Low | Low |
| GLM4-Plus | High | High | Medium | High |
| Qwen-Max-0919 | High | High | Relatively High | High |
| DeepSeek-V3 | High | 0.25 | 0.07 | High |

### Main Results - RSE Results (with GPT-4o-0806 as Reference)

| Model | RSE Score | 2-gram | BERTScore | Distillation Degree |
|------|---------|--------|-----------|---------|
| Llama3.1-70B-Instruct | 3.628 | 0.213 | 0.828 | Low |
| Doubao-Pro-32k | 3.720 | 0.216 | 0.823 | Low|
| Claude3.5-Sonnet | 3.740 | 0.189 | 0.823 | Low |
| DeepSeek-V3 | 4.102 | 0.220 | 0.837 | High |
| Qwen-Max-0919 | 4.174 | 0.252 | 0.838 | High |
| GPT4o-0513 | 4.240 | 0.269 | 0.841 | High |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Base vs Instruct (Qwen Series) | Base Strict Score is higher | Base models are more likely to leak distillation traces. |
| DeepSeek-V3 vs R1 | No significant difference | R1 is trained on V3, featuring limited identity fine-tuning. |
| RSE Validation (SFT 1-3 epoch) | Score consistently increases (3.554 -> 4.222 on ArenaHard) | Demonstrates that RSE can effectively detect distillation. |
| Qwen-Max citation of Claude | Qwen-Max contains Claude references | Suggests that Qwen-Max might have been distilled from Claude. |

### Key Findings
- Claude, Doubao, and Gemini exhibit a low degree of distillation in both ICE and RSE, suggesting these models are more likely to be independently developed.
- Most open-source and closed-source LLMs (including GLM4-Plus, Qwen-Max, and DeepSeek-V3) exhibit a high degree of distillation.
- The distillation degree of Base models is generally higher than that of aligned Instruct models.
- Qwen-Max's responses frequently contain references to Claude3.5-Sonnet, while the Qwen 2.5 series primarily refers to GPT.
- ICE shows that LLMs are more easily jailbroken in "team", "industry", and "tech" categories.

## Highlights & Insights
- This work systematically proposes an LLM distillation quantification framework for the first time, filling an important research gap.
- The ICE method is highly ingenious, leveraging jailbreak attacks to uncover the "identity fingerprints" left by distillation, presenting a novel approach.
- RSE assesses through multiple dimensions (style, logic, content), which is more informative than simple text similarity metrics.
- The experiments cover mainstream closed-source and open-source models, providing valuable industry insights.
- The finding that Qwen-Max references Claude is intriguing, hinting at complex distillation chains.

## Limitations & Future Work
- ICE depends on the success rate of jailbreak attacks; if the model's safety alignment is robustly executed, occurrences of distillation might go undetected.
- RSE uses GPT-4o as the reference model, but if the target model is distilled from other models (such as Claude), the distillation degree might be underestimated.
- The true positive rate of Loose Score is only 0.78-0.90, necessitating attention to the false positive issue.
- It cannot distinguish between "direct distillation" and "indirect distillation" (such as training on data generated by GPT).
- Research idea: Incorporating internal representations of models (such as attention pattern analysis) could provide finer-grained traceability of distillation chains.
- There is a lack of in-depth analysis regarding the relationship between the distillation degree and actual performance.

## Related Work & Insights
- There is a connection with data contamination detection methods (e.g., LM Contamination Index), but this paper focuses on knowledge transfer between models rather than training set leakage.
- Transforming the perspective of jailbreak attacks (GPTFuzz) into a detection tool rather than an attack tool is highly inspiring.
- Raising calls for transparency in LLM development has significant social implications, especially in the context of the widespread academic use of distilled data.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the LLM distillation quantification problem and provides an actionable framework for the first time, though ICE is based on the existing GPTFuzz.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, metrics, and manual verification, but lacks validation on ground-truth distillation relationships.
- Writing Quality: ⭐⭐⭐⭐ Clarifies the problem definition and organizes the experiments reasonably, though some symbols and formulas could be more concise.
- Value: ⭐⭐⭐⭐ Provides a significant boost to LLM development transparency, though the reliability of the quantitative results still requires more validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](drpruning_robust_pruning.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)
- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)

</div>

<!-- RELATED:END -->
