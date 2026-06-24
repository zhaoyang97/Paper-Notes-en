---
title: >-
  [Paper Note] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis
description: >-
  [ACL 2025][AIGC Detection][LLM-generated text detection] This paper constructs the first Korean LLM-generated text detection benchmark, KatFish (covering three genres and four LLMs). By analyzing three types of Korean linguistic features—word spacing, POS diversity, and comma usage—the authors propose the KatFishNet detection method, achieving an average AUROC 19.78% higher than the best baseline under the OOD (unseen LLM) setting.
tags:
  - "ACL 2025"
  - "AIGC Detection"
  - "LLM-generated text detection"
  - "Korean linguistic features"
  - "comma usage patterns"
  - "word spacing"
  - "benchmark dataset"
date: 2026-05-08
content_hash: 03508e58ebbefe6e
---

# KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis

**Conference**: ACL 2025  
**arXiv**: [2503.00032](https://arxiv.org/abs/2503.00032)  
**Code**: [GitHub](https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis)  
**Area**: AIGC Detection / Korean NLP  
**Keywords**: LLM-generated text detection, Korean linguistic features, comma usage patterns, word spacing, benchmark dataset

## TL;DR

This paper constructs the first Korean LLM-generated text detection benchmark, KatFish (covering three genres and four LLMs). By analyzing three types of Korean linguistic features—word spacing, POS diversity, and comma usage—the authors propose the KatFishNet detection method, achieving an average AUROC 19.78% higher than the best baseline under the OOD (unseen LLM) setting.

## Background & Motivation

**Background**: LLM-generated text detection is crucial for maintaining academic integrity, preventing plagiarism, and protecting copyright. Existing detection methods (DetectGPT, LRR, etc.) are primarily designed for English, leveraging statistical properties like log probability and perturbation.

**Limitations of Prior Work**: Languages with unique morphological and syntactic characteristics require specialized detection methods. Korean has three distinct characteristics: (1) relatively flexible word spacing rules (with 70% of spacing errors originating from dependent nouns); (2) a rich morphological system (diverse postpositions and endings); (3) a significantly lower frequency of comma usage compared to English. These characteristics severely degrade the performance of English-centric detection methods on Korean text. Additionally, there is a lack of benchmark datasets for Korean LLM-generated text.

**Key Challenge**: General statistical methods (such as log probability) rely on specific language models, yielding poor generalization across languages and models. Conversely, the morphological complexity of Korean provides distinctive signals that can be exploited to differentiate humans from LLMs.

**Goal**: (1) To create the first benchmark for detecting Korean LLM-generated texts; (2) To identify systematic differences between humans and LLMs in Korean linguistic features; (3) To design a lightweight and interpretable detection method based on these differences.

**Key Insight**: The authors hypothesize that LLMs exhibit linguistic patterns distinct from humans when generating Korean text—particularly in word spacing, POS combinations, and punctuation usage—as LLMs trained on multilingual data might transfer English punctuation habits to Korean.

**Core Idea**: Construct feature vectors leveraging Korean-specific linguistic characteristics (word spacing rules, POS n-gram diversity, and comma usage patterns) to train traditional ML classifiers for lightweight and efficient detection.

## Method

### Overall Architecture

The workflow of KatFishNet: (1) Perform morphological analysis on the input Korean text using Korean POS taggers such as Bareun and Kkma; (2) Extract quantitative metrics for three categories of linguistic features; (3) Construct feature vectors to input into Logistic Regression, Random Forest, or SVM classifiers; (4) Output binary classification results of human-written vs. LLM-generated text.

### Key Designs

1. **Word Spacing Patterns**:

    - **Function**: Capture the difference between humans and LLMs in adherence to Korean spacing rules.
    - **Mechanism**: Define three metrics—MMN-BN Space Ratio (spacing ratio between numeral determiners and dependent nouns), BN Space Ratio (spacing ratio before dependent nouns), and VX Space Ratio (spacing ratio before auxiliary verbs). LLMs strictly adhere to spacing rules, whereas humans frequently omit spaces due to readability, habit, or unfamiliarity with the rules.
    - **Design Motivation**: The flexibility of Korean spacing rules serves as a natural signal for human-machine distinction. This difference is most pronounced in essays and poetry, and minimal in academic abstracts (where writing is inherently more standardized).

2. **POS N-gram Diversity**:

    - **Function**: Measure the diversity of syntactic structures.
    - **Mechanism**: Extract POS sequences using the Kkma tagger and calculate diversity scores for 1-gram to 5-gram (number of unique n-grams / total number of n-grams). Human writing exhibits higher diversity in POS combinations due to flexible use of syntactical structures, whereas LLMs generate text based on statistical patterns and tend to repeat common structures.
    - **Design Motivation**: LLM text generation is based on the most probable word combinations in training data, which inherently leads to repetition, whereas human writing is more flexible and diverse.

3. **Comma Usage Patterns**:

    - **Function**: Provide the strongest detection signal.
    - **Mechanism**: Define five metrics—comma occurrence rate (proportion of sentences containing commas), average comma usage (number of commas / number of morphemes), average relative position of commas, average segment length, and POS diversity scores before and after commas. LLMs use commas significantly more frequently than humans (human 26.31% vs. LLM 61.03% in essays), place commas much later in sentences, and show higher POS combination diversity around commas.
    - **Design Motivation**: Comma usage reflects contextual and stylistic factors, highly depending on writer intent, which is challenging for LLMs to learn precisely from training data. In particular, LLMs might transfer English comma usage conventions to Korean (e.g., inserting commas after conjunctive adverbs), which is unnatural in Korean.

### Loss & Training

Standard Logistic Regression, Random Forest, and SVM classifiers are deployed. The dataset is split 8:2, with text generated by GPT-4o serving as the LLM portion of the training set, and texts generated by Solar, Qwen2, and LLaMA-3.1 as the OOD test sets.

## Key Experimental Results

### Main Results

Essay detection results:

| Method | →Solar | →Qwen2 | →Llama3.1 | Average |
|------|--------|--------|-----------|------|
| Best Baseline (LLM Para.) | 92.08 | 79.74 | 72.00 | 81.27 |
| KatFishNet (Word Spacing) | 86.00 | 80.63 | 71.91 | 79.51 |
| KatFishNet (POS Comb.) | 92.26 | 83.10 | 73.63 | 82.99 |
| **KatFishNet (Punctuation)** | **97.57** | **94.63** | **92.45** | **94.88** |

Gain of the best KatFishNet (punctuation features) across three genres compared to the best baseline: Essay +16.74% (vs. LLM Paraphrasing), Poetry +10.72% (vs. DetectGPT), Academic Abstract +31.90% (vs. LLM Paraphrasing).

### Ablation Study

| Feature Type | Logistic Reg. | Random Forest | SVM | Description |
|---------|--------------|---------------|-----|------|
| Word Spacing (Essay) | 79.51 | 75.14 | 76.61 | Minor difference among the three models |
| POS Combinations (Essay) | 82.99 | 82.33 | 79.55 | LR performs slightly better |
| **Punctuation (Essay)** | **94.88** | **93.82** | **94.36** | Punctuation features consistently perform the best |

### Key Findings

- **Comma usage patterns are the most effective detection feature**, consistently performing the best across all genres and ML models. This is because comma usage reflects complex contextual and stylistic factors that are highly difficult for LLMs to mimic.
- The LLM Prompting baseline (directly prompting the LLM to judge) performs near-random (~50%), indicating that even LLMs struggle to identify their own generated text.
- The fine-tuned RoBERTa baseline performs poorly in Korean (~65%), showing that detectors pre-trained on English/Chinese cannot be directly migrated to Korean.
- Performance for all methods is relatively lower on academic abstracts, as the academic writing style itself is highly standardized and uniform, compressing the differences between humans and machines.

## Highlights & Insights

- **Linguistically-Driven Detection Approach**: Instead of relying on black-box deep models, this work identifies interpretable detection signals from the perspective of Korean linguistics. This methodology can be transferred to other non-English languages by identifying characteristic writing habits (e.g., honorific usage in Japanese, tokenization patterns in Arabic) and leveraging human-machine disparities to construct detectors.
- **Cross-Lingual Punctuation Transfer Hypothesis**: The discovery that LLMs transfer English punctuation habits to Korean is highly insightful—multilingual training leads to stylistic homogenization across languages, which presents an exploitable systematic vulnerability.
- **Lightweight and Practical Solution**: KatFishNet can be trained and run for inference entirely on CPUs, making it ideal for resource-constrained scenarios.

## Limitations & Future Work

- **Limited Genre Coverage**: Only three genres (essays, poetry, academic abstracts) were tested, leaving out domains like news, social media, and legal documents.
- **Handling Only Purely Human vs. Purely Machine Text**: Real-world scenarios often involve hybrid texts (e.g., human drafts polished by LLMs), which are not considered in this work.
- **Limitations of Morphological Analyzers**: Korean morphological analyzers still have room for improvement, and parsing errors can affect the accuracy of feature extraction.
- **Adversarial Robustness**: If LLMs are prompted to mimic human spacing and comma habits, the detection performance may decline.

## Related Work & Insights

- **vs. DetectGPT (Mitchell et al., 2023)**: DetectGPT, based on perturbation-driven log-probability variations, performs best on poetry (66.02%) but is significantly outperformed by KatFishNet on essays and abstracts, illustrating that statistical approaches hold an advantage in structurally simple texts but fall short in scenarios requiring linguistic understanding.
- **vs. LLM Paraphrasing (Zhu et al., 2023)**: This baseline paraphrases the source text using an LLM and compares semantic similarity. While it represents the strongest baseline in essays (81.27%), it is significantly outperformed by KatFishNet's punctuation features (94.88%), suggesting that linguistic features capture signals that paraphrasing methods cannot.
- **vs. Fine-tuning RoBERTa**: RoBERTa pre-trained on English-Chinese bilingual data yields poor results in Korean (~65%), further supporting the argument for language-specific detection methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first Korean LLM detection dataset coupled with a linguistically-driven detection method, offering a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple genres, various LLMs, numerous baselines, OOD evaluations, human evaluations, and ablation studies make it highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, in-depth linguistic analysis, and abundant tables/illustrations.
- **Value**: ⭐⭐⭐⭐ Pioneers research into non-English LLM detection, with a methodology showcasing strong transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Empirical Study on Detecting AI-Generated Text in Financial Reports](an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)
- [\[ACL 2025\] Cognitive Framework for Detecting AI-Generated Fiction](cognitive_framework_for_detecting_ai-generated_fiction.md)
- [\[ICLR 2026\] Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text](../../ICLR2026/aigc_detection/learn-to-distance_distance_learning_for_detecting_llm-generated_text.md)
- [\[ICLR 2026\] HLD: Approximate Hierarchical Linguistic Distribution Modeling for LLM-Generated Text Detection](../../ICLR2026/aigc_detection/hld_approximate_hierarchical_linguistic_distribution_modeling_for_llm-generated_.md)
- [\[ACL 2025\] Learning to Rewrite: Generalized LLM-Generated Text Detection](learning_to_rewrite_generalized_llm-generated_text_detection.md)

</div>

<!-- RELATED:END -->
