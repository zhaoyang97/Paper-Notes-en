---
title: >-
  [Paper Note] MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Metaphor] Proposes MultiMM, the first cross-cultural multimodal metaphor dataset containing 8,461 Chinese-English advertisement image-text pairs with fine-grained annotations, and designs the SEMD model integrating sentiment features to enhance metaphor detection.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Metaphor"
  - "Cultural Bias"
  - "Cross-Cultural"
  - "Sentiment Analysis"
  - "Metaphor Detection"
date: 2026-05-08
content_hash: d080cc7e43f5c680
---

# MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors

**Conference**: ACL 2025  
**arXiv**: [2506.06987](https://arxiv.org/abs/2506.06987)  
**Code**: [GitHub](https://github.com/DUTIR-YSQ/MultiMM)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Metaphor, Cultural Bias, Cross-Cultural, Sentiment Analysis, Metaphor Detection

## TL;DR

Proposes MultiMM, the first cross-cultural multimodal metaphor dataset containing 8,461 Chinese-English advertisement image-text pairs with fine-grained annotations, and designs the SEMD model integrating sentiment features to enhance metaphor detection.

## Background & Motivation

**Background**: Metaphors are ubiquitous in communication, with approximately one-third of sentences containing metaphors. Multimodal metaphors are more expressive than unimodal ones, yet existing datasets primarily originate from Western cultural backgrounds.

**Limitations of Prior Work**: Cultural bias in training data leads to overestimated model performance and poor generalization in non-Western cultural scenarios. There is a lack of benchmark datasets for cross-cultural multimodal metaphor research.

**Key Challenge**: Although conceptual metaphor mapping has universality, specific linguistic and visual expressions heavily depend on cultural backgrounds (e.g., "dinosaur" refers to "outdated" in English, but "ugly" in Chinese).

**Goal**: Construct the first cross-cultural multimodal metaphor dataset to reveal the impact of cultural bias on metaphor processing.

**Key Insight**: Collect image-text pairs from Chinese and English advertisements, and annotate three dimensions: metaphor occurrence, source/target domain relationships, and sentiment categories.

**Core Idea**: The influence of cultural bias on multimodal metaphor processing has been severely underestimated, and sentiment information can serve as a bridge for cross-cultural metaphor understanding.

## Method

### Overall Architecture

A dataset of 8,461 Chinese-English advertisement image-text pairs (4,397 Chinese, 4,064 English) was constructed, and a three-branch SEMD model was designed: ViT extracts image features, BERT extracts text features, and sentiment analysis extracts sentiment features, which are then concatenated and fused for classification.

### Key Designs

1. **Data Collection and Annotation**: Chinese samples were collected through Baidu search, and English samples were derived from public advertisement datasets. The annotation model includes: metaphor occurrence (literal/metaphorical), source/target domain vocabulary, and sentiment category (positive/neutral/negative). Annotated by 8 experts with Fleiss' $\kappa = 0.73$ (metaphor) and $0.82$ (sentiment).
2. **SEMD Model**: A three-branch architecture—ViT encodes images, mBERT encodes text, and a sentiment analysis module extracts sentiment features (NRCLex emotion + VADER sentiment scores), which are fused via concat and passed through a feed-forward network to obtain the final prediction.
3. **Cross-Cultural Analysis**: Analysis of cultural differences in source domain vocabulary distribution between Chinese and English advertisements—English favors lions/eagles (strength and freedom), while Chinese favors dragons/pandas (power and national pride). The same source domain may express different sentiments in different cultures.

### Evaluation Strategy

Evaluated on two tasks: metaphor detection and sentiment analysis, reporting Accuracy, Macro Precision, and Macro F1. Compared with 18 baseline models (8 text-based, 3 vision-based, 7 multimodal).

## Key Experimental Results

### Main Results (Metaphor Detection F1%)

| Model | English | Chinese |
|------|------|------|
| mBERT (Text) | 64.00 | 65.52 |
| ViT (Image) | 71.67 | 69.04 |
| CMGCN (Multimodal) | 79.04 | 74.91 |
| GPT-4o (Multimodal LLM) | 64.00 | 67.00 |
| LLaVA | 73.31 | 69.84 |
| Qwen2.5-VL-72B | 59.12 | 67.66 |
| **SEMD (Ours)** | **80.16** | **77.79** |

### Ablation Study (Fusion Method + Sentiment Features)

| Sentiment Features | Fusion Method | English F1% | Chinese F1% |
|---------|---------|---------|---------|
| None | concat | 78.64 | 74.84 |
| Yes | add | 77.76 | 74.37 |
| Yes | max | 78.59 | 74.37 |
| Yes | concat | **80.88** | **77.39** |

### Key Findings

- The visual modality generally outperforms the textual modality, indicating that advertising images contain rich metaphorical features.
- Multimodal models significantly outperform unimodal ones, whereas large multimodal LLMs like GPT-4o perform poorly in comparison.
- Positive sentiments dominate in metaphorical content (74.75% in English, 56.42% in Chinese), and English advertisements present more extreme sentiment.
- Translation degrades metaphor detection performance, confirming the importance of cultural context in metaphor understanding.

## Highlights & Insights

- The first cross-cultural multimodal metaphor dataset, filling the gap of Eastern culture in metaphor research.
- Discovered that sentiment, as a cross-cultural universal feature, can effectively enhance metaphor understanding.
- Large multimodal LLMs do not dominate in metaphor detection, indicating that metaphor understanding requires specialized designs.

## Limitations & Future Work

- Only covers Chinese and English cultures, without extending to more diverse cultural backgrounds.
- The SEMD model architecture is simple (concat fusion); more complex cross-modal interactions can be explored.
- Sentiment features rely on existing tools (NRCLex/VADER), potentially introducing tool bias.

## Related Work & Insights

- Complementary to prior work such as MultiMET, expanding from monocultural to cross-cultural analysis.
- Prompts researchers to attend to cultural bias issues in NLP systems.
- The sentiment-metaphor correlation is worth exploring in broader tasks.

## Additional Technical Details

- SEMD metaphor detection prediction: $P_{Meta} = \text{Sigmoid}(\text{Fusion}(\text{concat}(I_i, T_i, S_i)))$
- Sentiment analysis prediction (excluding sentiment feature input): $P_{Senti} = \text{Sigmoid}(\text{Fusion}(\text{concat}(I_i, T_i)))$
- Model hyperparameters: embedding dimension of 768, dropout of 0.3, maximum text length of 30 tokens, batch size of 64, and learning rate of $3e-5 \sim 5e-4$.
- Dataset split: train/validation/test = 80%/10%/10% (Chinese: 3517/440/440, English: 3251/406/407).
- Fleiss' $\kappa$: Metaphor 0.73, Target Domain 0.70, Source Domain 0.66, Sentiment 0.82.
- Cultural difference findings: English metaphorical advertisements have almost no negative sentiment (0.69%), whereas Chinese metaphorical advertisements have 3.17%.
- Translation experiments: SEMD F1 drops from 80.16 to 78.57 after EN $\rightarrow$ CN translation, and from 77.79 to 75.53 after CN $\rightarrow$ EN translation.

## Rating

- Novelty: ⭐⭐⭐⭐ The first cross-cultural multimodal metaphor benchmark, offering a valuable task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis across various dimensions with over 18 baselines.
- Writing Quality: ⭐⭐⭐⭐ Rich in data analysis, though the model design section is relatively straightforward.
- Value: ⭐⭐⭐⭐ Advances awareness of cultural bias and promotes more equitable NLP systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](../../ACL2026/multimodal_vlm/vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](../../ACL2026/multimodal_vlm/cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)

</div>

<!-- RELATED:END -->
