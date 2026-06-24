---
title: >-
  [Paper Note] Disentangling Language and Culture for Evaluating Multilingual Large Language Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual Evaluation] The Dual Evaluation Framework is proposed to decouple multilingual LLM evaluation along two dimensions: "linguistic medium" and "cultural context." This reveals a "Cultural-Linguistic Synergy" phenomenon—where models perform better when the cultural context aligns with the querying language—and explains this behavior from an interpretability perspective using FFN neuron activation analysis.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Evaluation"
  - "Cultural-Linguistic Synergy"
  - "Neuron Interpretability"
  - "Cross-Cultural Understanding"
  - "BLEnD"
date: 2026-05-08
content_hash: 8d2df033a940d762
---

# Disentangling Language and Culture for Evaluating Multilingual Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.24635](https://arxiv.org/abs/2505.24635)  
**Code**: [https://yingjiahao14.github.io/Dual-Evaluation/](https://yingjiahao14.github.io/Dual-Evaluation/)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Evaluation, Cultural-Linguistic Synergy, Neuron Interpretability, Cross-Cultural Understanding, BLEnD  

## TL;DR

The Dual Evaluation Framework is proposed to decouple multilingual LLM evaluation along two dimensions: "linguistic medium" and "cultural context." This reveals a "Cultural-Linguistic Synergy" phenomenon—where models perform better when the cultural context aligns with the querying language—and explains this behavior from an interpretability perspective using FFN neuron activation analysis.

## Background & Motivation

1. Existing multilingual evaluations primarily translate English benchmarks (e.g., MMLU $\rightarrow$ MMMLU) to target languages. However, the testing content remains rooted in Western/English cultural contexts, failing to reflect genuine cross-cultural usage scenarios.
2. Culture-specific benchmarks (e.g., M3Exam, BLEnD) source material from genuine local scenarios, but neglect the common real-world need of multilingual users to query across cultures (e.g., a Spanish speaker asking about Chinese tea culture in Spanish).
3. Prior evaluations treat language and culture as inseparable dimensions, limiting fine-grained analysis of LLMs' distinct cross-lingual and cross-cultural capabilities.
4. There is a lack of a systematic framework to simultaneously evaluate the models' native cultural-linguistic alignment, cross-lingual understanding, and cross-cultural capabilities.
5. While models achieve peak performance in English on standard multilingual benchmarks (MMMLU, MGSM), it remains unclear whether this holds true for culturally relevant questions.
6. There is a critical need to understand the internal mechanisms behind models' multilingual capabilities from an interpretability perspective, rather than relying solely on empirical observations.

## Method

### Overall Architecture: Dual Evaluation Framework

The evaluation questions are represented as $Q_{i,j}$, where $i$ denotes the cultural context and $j$ represents the linguistic medium. The framework generates four types of evaluation scenarios from the same template question: native alignment $Q_{i,i}$ (language matches culture), cross-lingual $Q_{i,j}$ (same culture, different languages), and cross-cultural $Q_{j,i}$ (same language, different cultures), thereby enabling quantitative cross-dimensional comparison.

### Module 1: Dataset Construction

- Based on the BLEnD dataset, template questions are locally adapted ($Adapt_i$): replacing references to countries/regions, adjusting linguistic paradigms, and compiling culture-specific answer sets.
- The native alignment set $Q_{i,i}$ is directly sourced from BLEnD; the cross-lingual set $Q_{i,en}$ is obtained from the English translated version of BLEnD; other language pairs $Q_{i,j}$ are constructed using GPT-4o translations.
- The final dataset covers 7 languages (English, Chinese, Spanish, Indonesian, Korean, Persian, Sundanese) $\times$ corresponding cultural regions, totaling 9,500 samples. Human evaluation of translation quality yielded a 97.8% perfect score rate.

### Module 2: Multilingual Ability Evaluation

- Finding 1: Models perform best on English cultural contexts, and this advantage persists across different languages (e.g., asking about American culture in Spanish yields higher scores than asking about Spanish culture in Spanish).
- Finding 2 (Cultural-Linguistic Synergy): Culturally relevant questions are answered better when asked in their corresponding language rather than in English (e.g., answering Chinese cultural questions in Chinese scored an average of 8.8 points higher than in English, and 15.7 points higher for Indonesian), despite the models being predominately trained on English data.

### Module 3: Interpretability Analysis (Neuron Probing)

- The $i$-th neuron of the $l$-th layer is defined as the $i$-th element of $Activation(W_{up}^l \cdot h^l)$ in the FFN layers.
- Through a top-k ($k=5$) threshold, a key neuron set $N_q$ is extracted for each question. "Language-specific neurons" are defined as those activated only when responding in the target language (and not in English).
- Comparing the proportion of language-specific neurons $P_{i,i}$ vs. $P_{en,i}$ reveals that $P_{i,i} > P_{en,i}$ when Cultural-Linguistic Synergy occurs (e.g., this holds true for Llama-3-8B in Chinese, Indonesian, Persian, and Korean).

### Loss & Training

- Neuron probing is conducted using Qwen2.5-7B-Instruct and Llama-3-8B-Instruct.
- Hypotheses are validated through cross-series comparison between Llama-3 and Llama-3.1: models with stronger multilingual capabilities activate a higher proportion of language-specific neurons (Llama-3.1: 67% vs. Llama-3: 57%).
- Ablation studies validate the threshold selection by masking key neurons: target task performance drops significantly under the selected threshold, while the OOD task (ARC) remains largely unaffected.

## Key Experimental Results

### Table 1: Cross-cultural Evaluation (Queries in Spanish)

| Model | Spanish Culture | US Culture |
|------|-----------|---------|
| Claude-3.5-Sonnet | 81.0 | 82.0 |
| GPT-4o | 76.5 | 77.6 |
| Llama-3-70b | 72.0 | 79.6 |
| Qwen2.5-7b | 62.0 | 70.5 |
| Llama-3-8b | 58.9 | 74.5 |

**Findings**: Even when queried in Spanish, most models still perform better on US cultural questions than on Spanish ones, indicating that training data dominance in English cultural knowledge propagates across languages.

### Table 2: Quantifying Cultural-Linguistic Synergy (Chinese/Indonesian/Persian)

| Cultural Context | Query in Native Language vs. English (Average Difference) |
|---------|-------------------------------|
| Chinese Culture | +8.8 |
| Indonesian Culture | +15.7 |
| Persian Culture | -0.95 (limited by the performance of low-resource language models like Bloomz) |

**Findings**: Cultural-Linguistic Synergy is highly pronounced in high-to-mid-resource languages; the number of activated language-specific neurons is highly correlated with model performance (Pearson r=0.95).

## Highlights & Insights

- **Innovative Evaluation Paradigm**: First to decouple the two dimensions of language and culture, building a $Q_{i,j}$ four-quadrant evaluation framework that covers three real-world usage scenarios: native alignment, cross-lingual, and cross-cultural.
- **Discovery of Cultural-Linguistic Synergy**: Counter-intuitively demonstrates that models predominantly trained on English perform better when the culture aligns with the querying language, challenging the notion of "English omnipotence."
- **Closed-Loop Interpretability**: Moves from empirical observation to neuron activation analysis, providing an internal mechanism explanation for the Synergy phenomenon. The proportion of language-specific neurons can serve as a potential indicator of multilingual capabilities during training phases.

## Limitations & Future Work

- Only one representative cultural region was selected for each language, failing to cover cultural variations of the same language across different regions (e.g., Latin American Spanish vs. European Spanish).
- Cross-lingual pairs are limited to those paired with English (where $i$ or $j$ in $Q_{i,j}$ must be English), leaving cross-pairings between non-English languages unexplored.
- Neuron probing was only validated on 7B/8B models and not scaled to larger models (e.g., 70B+) due to computational resource constraints.
- The cross-lingual dataset constructed via translation relies on GPT-4o, which may introduce translation bias.

## Related Work & Insights

- **Multilingual Evaluation Benchmarks**: MMMLU, MGSM (translation-based); M3Exam, BLEnD, CulturalBench (culture-specific). This work is the first to unify both evaluation categories under a decoupled language-culture framework.
- **Multilingual Interpretability**: Tang et al. (2024) found that language-specific neurons are key to multilingual capabilities; Wendler et al. (2024) investigated latent languages through latent space projections; Zhao et al. (2024) proposed multilingual workflows. This work extends neuron analysis from a purely linguistic dimension to a cultural dimension.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The language-culture decoupled dual-axis evaluation framework is novel, and the concept of Cultural-Linguistic Synergy is insightful.
- **Effectiveness**: ⭐⭐⭐⭐ Supported by extensive experiments across 8 models $\times$ 7 languages, with neuron probing providing internal mechanism evidence and ablation studies validating the threshold selection.
- **Significance**: ⭐⭐⭐⭐ Challenges the "English dominance" assumption, offering direct guiding value for multilingual model training and evaluation.
- **Clarity**: ⭐⭐⭐⭐ The framework diagram and four-quadrant examples are intuitive, and the notation system is clear, though some formula layouts are dense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_llm_english_accent.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)

</div>

<!-- RELATED:END -->
