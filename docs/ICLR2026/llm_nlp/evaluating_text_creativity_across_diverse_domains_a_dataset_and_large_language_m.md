---
title: >-
  [Paper Note] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator
description: >-
  [ICLR 2026][LLM/NLP][creativity evaluation] This paper proposes a context-aware pairwise comparison framework for evaluating text creativity…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "creativity evaluation"
  - "LLM-as-a-judge"
  - "pairwise comparison"
  - "text creativity"
  - "dataset construction"
  - "CrEval"
  - "cross-domain evaluation"
date: 2026-05-08
content_hash: 78687d3bb5e60dc2
---

# Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator

**Conference**: ICLR 2026
**arXiv**: [2505.19236](https://arxiv.org/abs/2505.19236)  
**Code**: [Project Page](https://creval-creative-evaluation.github.io)  
**Area**: LLM/NLP
**Keywords**: creativity evaluation, LLM-as-a-judge, pairwise comparison, text creativity, dataset construction, CrEval, cross-domain evaluation

## TL;DR

This paper proposes a context-aware pairwise comparison framework for evaluating text creativity, constructs the CreataSet dataset comprising 100K+ human-annotated and 1M+ synthetic samples, and trains the CrEval evaluator, which surpasses GPT-4o by 18.7% in alignment with human judgments.

## Background & Motivation

- **Creativity evaluation is a frontier challenge in LLM assessment**: As LLMs demonstrate creativity in domains such as creative writing, literature, and humor, accurately evaluating their creative outputs becomes increasingly important.
- **Three major limitations of existing methods**:
  1. **Poor cross-domain generalizability**: Most methods evaluate only a single domain (e.g., problem-solving, humor, metaphor), where creativity is entangled with other concepts, making generalization difficult.
  2. **Insufficient granularity**: Most methods evaluate at the model/subject level and cannot distinguish which of two responses to the same prompt is more creative (text-level creativity).
  3. **Unreliable automation**: Directly prompting LLMs to evaluate creativity yields unreliable and inconsistent judgments at high cost.
- **Annotation consistency issues**: Without contextual guidance, human annotators show inconsistent understanding of creativity (ICC = 0.59); providing shared instructions raises ICC to 0.75.
- **Scarcity of creative data**: Training reliable evaluators requires large-scale data, yet data with creativity labels is extremely scarce.

## Method

### Overall Architecture

The overall pipeline consists of three steps: (1) cross-domain creative dataset initialization; (2) context-aware response augmentation; and (3) hybrid-strategy label construction. The CrEval evaluator is then trained on the resulting data.

### Key Designs

**1. Cross-Domain Dataset Initialization (CreataSet-Base)**

Initial data are collected from 8 sources and unified into the $(I, R)$ format (instruction–response pairs):
- **Type A**: Naturally creative datasets (e.g., Oogiri-GO humor, Ruozhiba).
- **Type B**: Standalone creative texts (poetry, lyrics, prose), where missing instructions are generated via instruction inversion fine-tuning.
- **Type C**: General instruction-tuning datasets (Infinity-Instruct), covering diverse domains.

The resulting CreataSet-Base contains 113K+ creative samples spanning 87 sub-domains and 17 core domains. Data are primarily in Simplified Chinese (reflecting the cultural-contextual dependency of creativity), though the framework is extensible to other languages.

**2. Context-Aware Response Augmentation (CreataSet-Ext)**

Multiple responses at varying creativity levels are generated for each instruction, forming $(I, R_1, \ldots, R_k)$ with $k=5$:
- Models of different capability levels are used (Qwen2.5-14B-Instruct vs. MiniCPM-2B-SFT).
- Two prompting modes are applied per model: a standard prompt $\text{Prompt}_o$ and a creativity-oriented prompt $\text{Prompt}_c$.
- For Type C data, GPT-4o additionally generates more creative responses.

**3. Hybrid-Strategy Label Construction**

- **Test set** (3K samples): 30 human annotators rate responses on a 4-point Likert scale; ICC(2k) = 0.92 (high consistency). Pairs with score difference > 0.3 are labeled as distinguishable; those with difference < 0.1 are labeled as ties.
- **Training set** (weakly supervised pseudo-labels): Based on two assumptions — (1) stronger models produce more creative responses (86.6% accuracy), and (2) creativity-oriented prompts outperform standard prompts (81.4% accuracy).

**4. CrEval Training**

CrEval receives a triple $(I, R_1, R_2)$ and predicts which response is more creative (or a tie). Training loss:

$$\mathcal{L} = -\sum_{(I,R_1,R_2) \in \mathcal{D}} \log P(y|I, R_1, R_2)$$

Key training techniques:
- **Position bias mitigation**: Data augmentation by swapping $R_1$ and $R_2$ with adjusted labels.
- **Negative sampling**: A randomly selected response serves as the least creative response, enhancing the model's sensitivity to the instruction context $I$.

### Loss & Training

Standard cross-entropy loss over classification labels expressed as text output, following the pairwise comparison paradigm of LLM-as-a-judge.

## Key Experimental Results

### Main Results

**CrEval vs. Baselines (CreataSet Test Set)**

| Method | F1 | Kappa | Agreement |
|------|-----|-------|-----------|
| PPL (Perplexity) | 0.357 | -0.042 | 0.430 |
| DSI (Semantic Divergence) | 0.480 | 0.175 | 0.457 |
| Creativity Index | 0.531 | 0.231 | 0.568 |
| GPT-4o | 0.703 | 0.519 | 0.642 |
| Claude-3.5-Sonnet | 0.727 | 0.609 | 0.740 |
| o3 | 0.721 | 0.578 | 0.725 |
| DeepSeek-R1 | 0.653 | 0.457 | 0.547 |
| CrEval-7B | **0.732** | **0.601** | **0.745** |
| CrEval-14B | **0.735** | **0.613** | **0.762** |

CrEval-7B already surpasses all general-purpose LLMs of comparable and larger scale. CrEval-14B achieves a Kappa 9.7% higher than DeepSeek-V3 and an Agreement 12.6% higher than GPT-4o.

**CrEval Gains over Base Models**

| Metric | CrEval-7B vs. Qwen2.5-7B |
|------|---------------------------|
| F1 | +19.2% |
| Kappa | +49.1% |
| Agreement | +29.8% |

### Ablation Study

**Data Source Ablation**

| Data Configuration | F1 | Kappa |
|----------|-----|-------|
| Synthetic only | 0.689 | 0.513 |
| Human only | 0.701 | 0.548 |
| Synthetic + Human (mixed) | **0.732** | **0.601** |

The combination of human-annotated and synthetic data is indispensable for training a robust evaluator.

**Cross-Domain Generalization**

Traditional metrics (PPL, DSI) perform extremely poorly across domains (Kappa near 0). Gemma-series models outperform others on short texts and lyrics but generalize poorly to humor domains (Oogiri-GO, Ruozhiba) and classical styles. CrEval performs consistently and robustly across all creative domains.

### Key Findings

1. **Traditional metrics completely fail**: PPL's Kappa approaches 0, indicating that perplexity is nearly uncorrelated with human creativity judgments.
2. **Reasoning models are not necessarily better**: DeepSeek-R1's F1 (0.653) is substantially lower than GPT-4o's (0.703), suggesting that reasoning chains offer limited benefit for creativity evaluation.
3. **CrEval can improve LLM creativity**: Using CrEval as a preference signal during training can enhance the creative output of LLMs themselves.

## Highlights & Insights

1. **Context-aware annotation protocol**: Providing shared instructions as context raises human annotation ICC from 0.59 to 0.75 — a simple yet methodologically impactful innovation.
2. **Weakly supervised large-scale data construction**: Pseudo-labels derived from model capability differences and prompting strategies (86.6% and 81.4% accuracy) elegantly address the scarcity of creative data.
3. **7B model surpasses frontier LLMs**: CrEval-7B outperforms GPT-4o, o3, and others on creativity evaluation, demonstrating the value of domain-specialized training.
4. **Broad cross-domain coverage**: The dataset spanning 87 sub-domains far exceeds the domain coverage of existing creativity datasets.
5. **Practical closed loop**: CrEval not only evaluates creativity but can also serve as a preference signal to improve LLM creativity.

## Limitations & Future Work

1. **Language limitation**: The dataset is primarily in Simplified Chinese; cross-lingual generalizability has not been verified.
2. **Subjectivity of creativity**: Creativity is inherently subjective and culturally dependent; the taxonomy of 87 domains and the rating criteria may carry systematic biases.
3. **Weak supervision label noise**: The assumption that "stronger models produce more creative responses" does not always hold (13.4% error rate), and this noise propagates into training.
4. **Scalability of pairwise comparison**: Large-scale ranking requires $O(n^2)$ pairwise comparisons; absolute creativity scoring would be more practical but is not addressed in this work.

## Related Work & Insights

- **Distinction from traditional creativity tests**: RAT and TTCT assess human divergent thinking at the subject level; this paper is the first to systematically evaluate creativity at the text level across domains.
- **Relationship to LLM-as-a-judge**: The work inherits the Arena-style pairwise comparison paradigm but specializes it for creativity — one of the hardest dimensions to evaluate.
- **Distinction from LitBench**: LitBench covers only the creative writing domain, whereas this work spans 87 domains and additionally addresses annotation consistency.
- **Implications for evaluation infrastructure**: A creativity evaluator can be deployed in parallel with safety and helpfulness evaluators to form a more comprehensive LLM evaluation ecosystem.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First cross-domain text-level creativity evaluation framework with an elegant context-aware annotation protocol.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comparisons with 25+ baselines (including frontier LLMs); ablation studies cover data sources, domain generalization, and model scale.
- **Value**: ⭐⭐⭐⭐ Both the dataset and evaluator are open-sourced, with direct applicability to improving LLM creativity.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich figures and tables, and well-motivated method descriptions.
- **Overall**: ⭐⭐⭐⭐ A solid and systematic contribution that fills the gap in automatic cross-domain text-level creativity evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)
- [\[ICLR 2026\] First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation](first_is_not_really_better_than_last_evaluating_layer_choice_and_aggregation_str.md)
- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](../../ACL2026/llm_nlp/how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](../../ACL2026/llm_nlp/why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](../../ACL2026/llm_nlp/personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)

</div>

<!-- RELATED:END -->
