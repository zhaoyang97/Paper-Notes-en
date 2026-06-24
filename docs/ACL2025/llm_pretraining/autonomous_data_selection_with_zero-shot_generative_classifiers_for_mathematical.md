---
title: >-
  [Paper Note] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts
description: >-
  [ACL 2025][LLM Pretraining][Autonomous Data Selection] AutoDS is proposed, which uses the base language model itself as a zero-shot generative classifier to automatically evaluate mathematical text quality by calculating a continuous LM-Score from YES/NO token logits. It filters high-quality corpora for continual pre-training, achieving an approximately 2x token efficiency improvement on MATH, GSM8K, and BBH.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Autonomous Data Selection"
  - "Zero-shot Generative Classifiers"
  - "Mathematical Texts"
  - "Continual Pre-training"
  - "LM-Score"
date: 2026-05-08
content_hash: 85a5c8d01f3f5f68
---

# AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts

**Conference**: ACL 2025  
**arXiv**: [2402.07625](https://arxiv.org/abs/2402.07625)  
**Code**: [GitHub](https://github.com/yifanzhang-pro/AutoMathText)  
**Area**: Data Selection / Mathematical Reasoning  
**Keywords**: Autonomous Data Selection, Zero-shot Generative Classifiers, Mathematical Texts, Continual Pre-training, LM-Score

## TL;DR

AutoDS is proposed, which uses the base language model itself as a zero-shot generative classifier to automatically evaluate mathematical text quality by calculating a continuous LM-Score from YES/NO token logits. It filters high-quality corpora for continual pre-training, achieving an approximately 2x token efficiency improvement on MATH, GSM8K, and BBH.

## Background & Motivation

**Mathematical reasoning is one of the core challenges for LLMs, yet high-quality mathematical corpora are scarce and vary greatly in quality.** Mathematical text contains symbolic formulas, multi-step derivations, and rigorous proof structures, which differ dramatically from general language tasks. Although the community is highly enthusiastic about building LLMs to master mathematics, the lack of well-curated mathematical corpora has consistently been a key bottleneck.

**Existing data selection methods have obvious limitations.** The Phi-1/Phi-2 methodology uses GPT-4 to annotate the educational value of code snippets and then trains a random forest classifier for data filtering. This requires expensive GPT-4 API calls and only yields discrete labels ("good"/"bad"), discarding fine-grained quality information. In mathematical contexts, treating texts with educational values of 0.95 and 0.001 identically is clearly sub-optimal. Keyword-based heuristics (such as counting the number of LaTeX symbols) fail to capture deep mathematical reasoning.

**Key Insight: The base LLM itself can act as a data quality judge.** Inspired by the Bradley-Terry model in DPO, the authors propose directly calculating continuous scores using the YES/NO logits of the base model, requiring no human annotations, SFT, RLHF, or secondary classifiers. This autonomous learning paradigm of "letting the model decide what to learn" is both highly efficient and scalable.

## Method

### Overall Architecture

The core pipeline of AutoDS consists of: (1) designing meta-prompts to ask the base model two YES/NO diagnostic questions; (2) calculating continuous LM-Scores from the model's logits; (3) filtering high-scoring documents based on thresholding to construct the AutoMathText dataset; and (4) performing continual pre-training on the selected data. The entire process requires zero annotation, zero additional models, and is fully automated.

### Key Designs

1. **Zero-shot Generative Classifier—Logits-based Continuous Scoring**:
    - Function: Computes continuous quality scores for each mathematical text using the YES/NO token logits of a base LLM.
    - Mechanism: Designs a meta-prompt featuring two diagnostic questions—Q1: "Does this text exhibit mathematical intelligence?" Q2: "Is it useful for future mathematics learning?". It extracts logits from the model's output to calculate scores for each question: $\text{LM-Score}(Q) = \frac{\exp(\text{logit}(\text{YES}))}{\exp(\text{logit}(\text{YES})) + \exp(\text{logit}(\text{NO}))}$. The final score is the product of the scores of both questions: $\text{LM-Score}(Q_1, Q_2) = \text{LM-Score}(Q_1) \times \text{LM-Score}(Q_2)$, ensuring that texts must score highly on both mathematical intelligence and educational value.
    - Design Motivation: Continuous scoring preserves fine-grained quality details compared to binary classification; thus, texts with scores of 0.95 and 0.50 can be treated differently, improving token efficiency. The scoring formula is essentially a softmax normalization, which shares the same form as the Bradley-Terry reward model in RLHF, but operates without any supervised signals.

2. **Autonomous Data Selection Pipeline**:
    - Function: Selects high-quality mathematical texts from three major data sources (OpenWebMath, arXiv, and Algebraic Stack).
    - Mechanism: Uses the Qwen-72B base model to compute LM-Scores, processing 11.26 million documents (200+ GB). Documents are retained based on score thresholds (e.g., 0.50-1.00 or 0.75-1.00). High-scoring documents mostly originate from math-intensive websites such as math.stackexchange.com.
    - Design Motivation: Human annotation of 11.26 million documents would cost over \$10 million. AutoDS utilizes 4 A100-80G GPUs for approximately 750 GPU hours, costing less than \$10,000 under typical cloud pricing, a 1000x cost reduction.

3. **Autonomous Continual Pre-training**:
    - Function: Performs continual pre-training on training data selected by the base model itself.
    - Mechanism: Models not only learn from data, but also actively decide "what data to learn from," achieving self-directed learning. It allows for continuous evaluation and dynamic curation as new data arrives.
    - Design Motivation: In specialized domains like mathematics, human annotation is scarce and expensive, and keyword heuristics are unreliable (e.g., OpenWebMath's classifier heavily relies on LaTeX symbol density). Allowing the model to judge autonomously is more accurate and scalable.

### Loss & Training

Continual pre-training utilizes a standard causal LM objective (next-token prediction), where the selection strategy indirectly affects training effectiveness via data quality. Scores are generated using Qwen-72B, and continual pre-training is validated on Gemma-2B, LLaMA2-7B, and Mistral-7B.

## Key Experimental Results

### Main Results—Comparison of Continual Pre-training Across Models

| Model + Data Selection Method | MATH (5-shot) | GSM8K (5-shot) | BBH (3-shot) |
|-------------------|---------------|----------------|--------------|
| Mistral-7B Base | 12.88 | 38.82 | 55.92 |
| + Uniform (OpenWebMath) | 14.26 | 44.12 | 56.50 |
| + DSIR | 12.30 | 42.00 | 55.97 |
| + QuRating | 12.90 | 36.32 | 55.63 |
| **+ AutoDS** | **16.14** | **45.41** | **58.61** |
| LLaMA2-7B Base | 2.94 | 12.51 | 39.89 |
| + Uniform (OpenWebMath) | 5.14 | 19.79 | 41.53 |
| **+ AutoDS** | **7.74** | **21.99** | **42.76** |
| Gemma-2B Base | 10.96 | 17.29 | 34.19 |
| **+ AutoDS** | **11.02** | **18.88** | **34.88** |

### Ablation Study—AutoDS vs. Binary Filter

| Method | Data Volume (M tokens) | MATH CPT (%) | MATH SFT (%) |
|------|-------------------|-------------|-------------|
| No Pre-training | 0 | 12.88 | 27.20 |
| OpenWebMath (Full) | 328.9 | 10.50 | 26.98 |
| AutoDS (0.75-1.00) | 328.9 | **13.68** | **28.06** |

### Key Findings

- AutoDS consistently outperforms Uniform, DSIR, and QuRating across all base models (Gemma-2B, LLaMA2-7B, Mistral-7B) and benchmarks (MATH, GSM8K, BBH).
- Given the same number of tokens, the pre-training efficiency of AutoDS is approximately twice that of Uniform, achieving "2x token efficiency."
- Under certain configurations, DSIR and QuRating perform worse than even the base models, indicating that inappropriate data selection can be detrimental.
- The OpenWebMath classifier primarily depends on LaTeX symbol density, making it prone to selecting texts with high digit density but no genuine mathematical substance (e.g., package tracking pages).
- StackExchange sites (especially math.stackexchange.com) contribute a significant share of high-quality mathematical texts in high-score tiers.

## Highlights & Insights

- **Autonomous developmental paradigm of "models selecting their own data"**: Eliminates the need for external annotations, classifiers, or alignment steps, achieving true self-directed data curation.
- **Continuous Scoring > Binary Classification**: Retains fine-grained quality details, improving token efficiency by approximately twofold.
- **Theoretical connection with Bradley-Terry / DPO**: The softmax form of the LM-Score aligns with the form of the RLHF reward model, suggesting that data selection and preference optimization might be theoretically unified.
- **Highly Cost-Effective**: Processes 11.26 million documents using 4 A100 GPUs for about 750 GPU hours with a cost of under \$10,000 (compared to over \$10 million for human annotation).

## Limitations & Future Work

- Scoring quality depends heavily on the capabilities of the base model; weaker models might not accurately assess the value of mathematical content.
- Only evaluated in the mathematical domain; meta-prompts for other specialized fields (e.g., medicine, law, code) would need redesigning.
- The binary YES/NO assumption might not suffice to capture richer quality dimensions (such as novelty, rigor, complexity, etc.).
- Only validated on English mathematical texts; the effectiveness of scoring multilingual mathematical corpora remains unexplored.
- The scoring model (Qwen-72B) differs from the target models (Mistral-7B, etc.). Is there an optimal strategy for selecting scoring models?

## Related Work & Insights

- **vs. Phi-1/Phi-2 Data Selection**: Phi-1/2 requires GPT-4 annotation and training a random forest classifier, which is expensive and yields only discrete labels. AutoDS requires zero annotations, zero secondary classifiers, and produces continuous scores.
- **vs. DSIR (Importance Resampling)**: DSIR performs distribution matching based on n-gram statistical features. AutoDS relies on LLM semantic understanding, which is more accurate in mathematical scenarios.
- **vs. QuRating**: QuRating also uses LLMs to evaluate data "quality" but in a different manner; the logits-based approach in AutoDS is simpler and more efficient.
- **Insights**: The principle that continuous quality scoring outperforms binary classification should be universally applicable to all data selection scenarios—not just mathematics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The zero-shot generative classifier concept is highly novel, the continuous LM-Score design is elegant, and the autonomous data selection paradigm holds profound significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Conducts experiments across three models on three benchmarks, providing comprehensive ablations and comparisons against DSIR, QuRating, and Uniform.
- Writing Quality: ⭐⭐⭐⭐ Features clear motivation, concise methodology, and intuitive data visualizations (treemaps, distribution charts).
- Value: ⭐⭐⭐⭐⭐ The AutoMathText dataset is open-sourced, the methodology is directly reusable, and the 2x efficiency gain is of practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[ACL 2025\] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)

</div>

<!-- RELATED:END -->
