---
title: >-
  [Paper Note] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual LLMs] This paper reveals that multilingual LLMs exhibit an "English accent" when generating non-English text—biasing toward English patterns lexically and syntactically. It proposes corpus-level naturalness metrics based on JSD (for lexical distribution) and WL graph kernel + MMD (for syntactic dependency trees), and demonstrates that the naturalness of target languages can be effectively improved using DPO alignment…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual LLMs"
  - "Naturalness Evaluation"
  - "English Bias"
  - "Lexical Distribution"
  - "Syntactic Naturalness"
date: 2026-05-08
content_hash: c89c8838452f809d
---

# Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs

**Conference**: ACL 2025  
**arXiv**: [2410.15956](https://arxiv.org/abs/2410.15956)  
**Code**: None  
**Area**: Multilingual LLM / NLP  
**Keywords**: Multilingual LLMs, Naturalness Evaluation, English Bias, Lexical Distribution, Syntactic Naturalness

## TL;DR
This paper reveals that multilingual LLMs exhibit an "English accent" when generating non-English text—biasing toward English patterns lexically and syntactically. It proposes corpus-level naturalness metrics based on JSD (for lexical distribution) and WL graph kernel + MMD (for syntactic dependency trees), and demonstrates that the naturalness of target languages can be effectively improved using DPO alignment.

## Background & Motivation

**Background**: Currently, multilingual LLMs are dominated by English (e.g., Llama 3.1 contains only 8% non-English training data), and even models claiming multilingual support exhibit a strong English-centric bias.

**Limitations of Prior Work**:
   - Multilingual LLM evaluations primarily focus on task performance (e.g., MMLU, BLEU), neglecting linguistic naturalness.
   - LLMs generate unnatural expressions in non-English languages—exhibiting an "accent" similar to native English speakers learning a foreign language.
   - A large amount of translated text in training data contains "translationese," which is learned and amplified by the models.
   - There is a lack of systematic evaluation metrics for multilingual naturalness.

**Key Challenge**: Multilingual LLMs may achieve high scores on task benchmarks, but the generated non-English texts do not read naturally to native speakers—displaying traces of English in both lexical choice and syntactic structure.

**Goal**: Devise automated metrics to quantify the naturalness of multilingual LLMs and propose methods for improvement.

**Key Insight**: Decouple the naturalness problem into two dimensions—lexical naturalness and syntactic naturalness—and perform statistical comparisons at the corpus level (rather than the sentence level).

**Core Idea**: Quantify the degree of the "English accent" by comparing the lexical and dependency tree distributions of LLM-generated texts with those of native speaker texts.

## Method

### Overall Architecture
Evaluation framework: Construct a topic-aligned multilingual dataset (Wikipedia entries) $\rightarrow$ Prompt LLMs to generate descriptions in each language $\rightarrow$ Compare distribution differences between LLM outputs and human texts using lexical and syntactic metrics. Improvement framework: Construct preference data (human text vs. artificially distorted/unnatural text) $\rightarrow$ DPO training.

### Key Designs

1. **Lexical Naturalness**:

    - **Function**: Compare vocabulary distribution differences between LLM-generated texts and human texts.
    - **Mechanism**: Compute the Jensen-Shannon Divergence (JSD) between the lexical distributions of the two corpora: $\text{JSD}(P||Q) = \frac{1}{2}(D_{KL}(P||M) + D_{KL}(Q||M))$, where $M = \frac{1}{2}(P+Q)$. This is calculated at the word level (not sub-word level); a lower JSD indicates more natural vocabulary choice.
    - **Design Motivation**: Direct comparison of word frequency distributions without relying on external embedding models (avoiding the introduction of the English bias of those embedding models), which implicitly captures statistical characteristics such as the type-token ratio.

2. **Syntactic Naturalness**:

    - **Function**: Compare the distribution differences of dependency tree structures between LLM-generated texts and human texts.
    - **Mechanism**: Parse each sentence into a dependency tree using Universal Dependencies $\rightarrow$ Calculate structural similarity between pairs of trees using the Weisfeiler-Lehman (WL) graph kernel $\rightarrow$ Quantify the difference between the two tree distributions using Maximum Mean Discrepancy (MMD).
    - **Design Motivation**: The UD framework is cross-linguistically consistent, the WL graph kernel captures hierarchical subtree patterns, and MMD is a classic method for distribution comparison. The entire pipeline is transparent, interpretable, and independent of language model embeddings.

3. **DPO Naturalness Alignment**:

    - **Function**: Improve target language naturalness via preference learning.
    - **Mechanism**: Construct preference data where positive instances are texts written by native human speakers, and negative instances are synonymous but artificially distorted, unnatural texts (e.g., introducing translationese or substituting with Anglicized expressions). Train the model with DPO to prefer natural expressions.
    - **Design Motivation**: Eliminates the need for a pretrained naturalness classifier (avoiding overfitting), directly constructs preference data using manipulated negative examples, which is simple yet effective.

### Evaluation Settings
- Models: Qwen1.5-7B, Qwen2-7B, Mistral-v0.3-7B, Mistral-Nemo-12B, Llama-3-8B, Llama-3.1-8B
- Languages: English, French, Chinese
- Data: 3,722 topic-aligned Wikipedia entries

## Key Experimental Results

### Main Results

| Model | English Lexical↓ | English Syntactic↓ | Chinese Lexical↓ | Chinese Syntactic↓ | French Lexical↓ | French Syntactic↓ |
|------|----------|----------|----------|----------|----------|----------|
| Human Baseline | 23.07 | 3.53 | 25.91 | 2.93 | 24.25 | 3.22 |
| Qwen1.5-7B | 30.36 | 22.19 | 41.00 | 23.33 | 38.35 | 24.21 |
| Llama-3.1-8B | 26.79 | 16.80 | 33.29 | 10.32 | 31.52 | 11.27 |
| **Mistral-Nemo-12B** | **25.12** | **14.77** | **34.78** | **12.84** | **31.34** | **14.72** |

### Ablation Study

DPO naturalness alignment performance (Chinese):

| Metric | Before Alignment | After Alignment | Change |
|------|--------|-------|------|
| Lexical JSD | 33.29 | Decrease | Improved |
| Syntactic MMD | 10.32 | Decrease | Improved |
| General Task Performance | baseline | Comparable | No detriment |

### Key Findings
- All LLMs show significantly higher lexical and syntactic divergence in non-English languages compared to English, confirming the existence of an "English accent."
- Chinese exhibits the largest naturalness gap (lexical JSD: human 25.91 vs. worst 41.00), suggesting that languages more structurally distant from English suffer more from English bias.
- Llama-3.1 shows improved naturalness over Llama-3 in all languages, demonstrating the effectiveness of increasing the multilingual data proportion.
- The syntactic gap is larger than the lexical gap (relative to the human baseline), indicating that the Anglicization of syntactic structures is a more prominent issue in LLMs.
- DPO alignment significantly improves naturalness without degrading performance on general task benchmarks.

## Highlights & Insights
- **The "English accent" analogy is vivid and accurate**: Comparing multilingual LLM generation to a native English speaker's "accent" when speaking foreign languages is highly intuitive. This perspective successfully imports linguistic concepts (translationese) into LLM evaluation.
- **Metric design independent of external language models**: This avoids the circular problem of "using an English-biased model to evaluate English bias." Both JSD and the WL graph kernel + MMD are classical statistical methods that are transparent and interpretable.
- **Corpus-level rather than sentence-level evaluation**: Issues might not be obvious in single sentences, but statistical patterns manifest clearly at the corpus level—a significant methodological contribution.

## Limitations & Future Work
- Validated only on English, French, and Chinese; generalizability to other languages (especially low-resource ones) remains unknown.
- DPO alignment requires manual construction of unnatural negative examples, meaning construction quality heavily impacts performance.
- The metrics only focus on linguistic forms (lexicon and syntax) without covering semantic naturalness.
- Domain-specificity of Wikipedia data—performance may vary in other domains such as dialogue or news.

## Related Work & Insights
- **vs. Translationese Detection**: Prior translationese studies trained classifiers to distinguish original from translated text; this work proposes statistics-based, training-free metrics that are more generalizable.
- **vs. MAUVE**: MAUVE uses language model embeddings to compute distributional divergences; this work avoids embedding models entirely to bypass bias, directly comparing word frequencies and syntactic trees.
- **vs. Multilingual LLM Analysis (Wendler et al.)**: While they demonstrated that the internal conceptual space of LLMs is biased toward English, this work quantifies how this bias affects generation quality at the output level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "English accent" perspective is highly novel, and corpus-level naturalness metrics fill a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 6 models across 3 languages, but the DPO improvement is only fully validated on Chinese.
- Writing Quality: ⭐⭐⭐⭐⭐ Clever analogy, clear design motivation for the metrics, and excellent interdisciplinary integration (linguistics and NLP).
- Value: ⭐⭐⭐⭐⭐ Highly valuable reference for evaluating and improving multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[ACL 2025\] M-RewardBench: Evaluating Reward Models in Multilingual Settings](m_rewardbench.md)
- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)

</div>

<!-- RELATED:END -->
