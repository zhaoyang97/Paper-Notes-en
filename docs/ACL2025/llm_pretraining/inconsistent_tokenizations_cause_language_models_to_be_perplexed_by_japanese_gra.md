---
title: >-
  [Paper Note] Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar
description: >-
  [ACL 2025][LLM Pretraining][Tokenization consistency] Reveals that inconsistent tokenization of the tokenizer is the root cause of LLMs failing to adhere to subtle grammatical rules such as the Japanese "first-person psych-predicate constraint"—when restricting test sentences to consistent tokenization, the perplexity gap of Llama 3 improves by 28 times.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Tokenization consistency"
  - "Japanese grammar"
  - "Psych-predicate constraint"
  - "Perplexity"
  - "Byte fallback"
date: 2026-05-08
content_hash: 06c90f496f1a4889
---

# Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar

**Conference**: ACL 2025  
**arXiv**: [2505.19599](https://arxiv.org/abs/2505.19599)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Tokenization consistency, Japanese grammar, Psych-predicate constraint, Perplexity, Byte fallback

## TL;DR

Reveals that inconsistent tokenization of the tokenizer is the root cause of LLMs failing to adhere to subtle grammatical rules such as the Japanese "first-person psych-predicate constraint"—when restricting test sentences to consistent tokenization, the perplexity gap of Llama 3 improves by 28 times.

## Background & Motivation

Standard LLM evaluation benchmarks (e.g., MMLU, llm-jp-eval) primarily test "high-level" capabilities (memorization, reasoning) while ignoring language-specific subtle capabilities. This is particularly problematic for Japanese:

### Japanese First-Person Psych-Predicate Constraint

Japanese has a unique grammatical phenomenon: predicates describing inner states (e.g., "寒い/cold", "悲しい/sad", etc.) can only describe the first person when used directly:

- ✅ 私は寒い (I feel cold) — First person + psych-predicate, grammatical
- ❌ 母は寒い (My mother feels cold) — Third person + psych-predicate in direct form, **ungrammatical**
- ✅ 母は寒がっている (My mother seems to feel cold) — Third person + psych-predicate + **evidential marker**, grammatical
- ✅ 母は寒そうだ (My mother looks cold) — Another grammatical form similar to the above

Native speakers naturally adhere to this rule (even without knowing the rule name), but L2 learners and **even state-of-the-art models like GPT-4o frequently violate this rule**.

Research Question: Why do LLMs violate this rule? Is it due to insufficient data or other systemic reasons?

## Method

### Overall Architecture

Investigates the mastery of the Japanese psych-predicate constraint by LLMs through two types of experiments:

1. **Perplexity Experiments**: Construct minimal pairs (grammatical vs. ungrammatical) and compare model perplexity.
2. **Machine Translation Experiments**: Ask models to translate English sentences containing third-person psych-predicates and observe whether evidential markers are utilized.

### Key Designs

**Model Selection**: Six models within the 7-10B parameter range are selected:
- Multilingual models: Mistral 0.1-7B, LLaMA 2-7B, LLaMA 3-8B
- Japanese-tailored models: Weblab-10B, Swallow-7B, Swallow-MS-7b

**Minimal Pair Design**:
Construct four types of sentences based on linguistic templates:
- (a) First person + psych-predicate (grammatical)
- (b) Third person + non-psych-predicate (grammatical)
- (c) Third person + psych-predicate + evidential marker (grammatical)
- (#) Third person + psych-predicate + direct form (**ungrammatical**)

Ideally, an LLM should yield lower perplexity for (a), (b), and (c) than for (#).

**Tokenization Consistency Analysis Metrics**:
- **fertility score**: Ratio of the number of tokens to the number of characters, measuring the "bloatness" of the tokenizer.
- **byte fallback rate**: Frequency of degrading to byte encoding when the tokenizer encounters unknown characters.

| Model | Fertility | Byte Fallback Rate |
|------|----------|-----------------|
| Llama 3 | 0.85 | 0.08 |
| Swallow | 1.00 | 0.19 |
| Weblab | 1.23 | **0.66** |
| Llama 2 | 1.58 | 0.49 |

### Loss & Training

This paper does not involve model training. The core analytical tool is **sentence-level perplexity**:

$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(t_i | t_{<i})\right)$$

Perplexity is reported using the median (instead of the mean) because token probability scales vary significantly across different models, and the median is more robust to outliers.

## Key Experimental Results

### Main Results

**Median perplexity of each model across four types of sentences**:

| Sentence Type | Mistral | Llama 2 | Llama 3 | Weblab | Swallow | Swallow-MS |
|------|---------|---------|---------|--------|---------|------------|
| (#) Ungrammatical | 2.0e+04 | 3.3e+04 | 6.9e+03 | 2.0e+06 | 1.2e+03 | 1.9e+03 |
| (a) Grammatical-1st | 3.6e+04 | 1.2e+05 | 9.1e+04 | 6.1e+05✅ | 1.9e+03❌ | 3.2e+03❌ |
| (b) Grammatical-Non-psych | 1.8e+03✅ | 5.9e+03✅ | 4.5e+03✅ | 7.3e+05✅ | 1.2e+03✅ | 2.9e+03❌ |
| (c) Grammatical-Evidential | 2.0e+04✅ | 4.9e+04❌ | 3.7e+04❌ | 1.3e+06✅ | 4.1e+03❌ | 3.3e+03❌ |

🔑 Core Finding: **Only Weblab yields lower perplexity on all three grammatical sentence types than on the ungrammatical type**.

**Counterintuitive Success of Weblab**:
Weblab utilizes an **unmodified English tokenizer**, leading to:
- Almost every Japanese character triggers byte fallback (rate 0.66)
- Even basic words learned in second grade of elementary school like "食べる" (eat) or "買う" (buy) cannot be tokenized correctly
- However, precisely because the tokenization is **consistently poor**, it avoids grammatical-specific tokenization inconsistency!

### Ablation Study

**Tokenization Consistency Experiments on Llama 3**:

In Llama 3, psych-predicate adjectives ending in "しい" (e.g., "悲しい/kanashii", "寂しい/sabishii") trigger byte fallback when combined with evidential expressions, resulting in extremely low probabilities. However, adjectives ending in "い" (e.g., "痛い/itai", "寒い/samui") do not.

When restricting test sentences to **consistently well-tokenized sentences**:
- Perplexity of grammatical type (c): 3.7e+04 → **1.3e+03** (reduced by approximately **28 times**)
- Perplexity of ungrammatical type (#): 6.9e+03 → 3.9e+03 (reduced by only approximately 1.8 times)

Conclusion: The model weights of Llama 3 **have already learned** the psych-predicate constraint, but inconsistent tokenization masks this capability.

**Machine Translation Experiments** (translating "My mother is {psych-predicate}"):

| Psych-predicate | Weblab-Evidential✅ | Weblab-Grammatical✅ | Llama3-Evidential✅ | Llama3-Ungrammatical❌ |
|---------|------------|------------|------------|-------------|
| cold | 47% | 53% | 0% | 69% |
| embarrassed | 90% | 10% | 32% | 39% |
| lonely | 0% | 0% | 0% | 100% |
| pain | 0% | 6% | 0% | 100% |

- Weblab consistently generates evidential markers on "cold" and "embarrassed"
- Llama 3 almost never uses evidential markers, producing ungrammatical expressions 100% of the time on "lonely" and "pain"
- Llama 3 also produces mistranslations (29%) and grammatical errors (31%), whereas Weblab does not

### Key Findings

1. **Inconsistent tokenization is the root cause**: Different tokenization behaviors for the same grammatical structure across different vocabulary items make model perplexity fail to reflect true grammatical knowledge.
2. **Consistently poor > inconsistently good**: Weblab performs best by using an English tokenizer fully unsuited for Japanese.
3. **When restricting to consistent tokenization, the model demonstrates learned grammatical knowledge**: Llama 3's performance improves by 28 times on the consistently tokenized subset.
4. **Byte fallback is a key confounding factor**: It reduces token probabilities of specific characters by several orders of magnitude.
5. **Instruction tuning cannot fix this problem**: The instructions-tuned version of Llama 3 still generates a large number of ungrammatical expressions in machine translation tasks.

## Highlights & Insights

1. **Linguistics-driven AI Analysis**: Rarely applies rigorous linguistic minimal pair methods to diagnostic LLM capability assessment.
2. **"Consistently bad is better than inconsistently good"**: This counterintuitive finding has profound implications for tokenizer design.
3. **Attributing surface-level performance failures to underlying engineering decisions**: It is not that "LLMs do not understand Japanese grammar", but rather that "the tokenizer prevents them from demonstrating this knowledge".
4. **GPT-4o makes the same mistakes**: Even state-of-the-art models are affected by this issue, suggesting this is not a model scale problem.
5. **A warning for tokenizer design in multilingual LLMs**: Pursuing a more efficient Japanese tokenizer might accidentally introduce grammar-specific tokenization inconsistency.

## Limitations & Future Work

1. **Focus on only one grammatical phenomenon**: Although the psych-predicate constraint is highly representative, Japanese contains many other subtle grammatical rules.
2. **Model scale limitations**: Only 7-10B models are investigated; whether larger models exhibit the same problem remains unknown.
3. **Confounding variables**: The volume of Japanese training data, proportion of Japanese data, and the interaction effect of tokenizers with training data are difficult to disentangle.
4. **Lack of clear solutions**: Points out the problem but does not propose specific tokenizer improvement schemes.
5. **Korean exhibits similar phenomena**, but cross-lingual comparative validation is not performed.
6. Byte fallback acts as both an analytical tool and a confounding factor, making causal relationships difficult to establish fully.

## Related Work & Insights

- **Hasegawa & Hirose (2005)**: Linguistics foundations of first-person psych-predicate constraints.
- **Rust et al. (2021)**: Proposed tokenizer fertility score, utilized in this paper to quantify tokenization quality.
- **Fujii et al. (2024) (Swallow)**: Japanese continually pre-trained LLMs; this paper analyzes its tokenizer characteristics.
- **Cool-Fusion (2407.19807)**: Interesting comparison—Cool-Fusion resolves cross-tokenizer issues at the paragraph/chunk level, whereas this paper reveals another impact of tokenizer inconsistency.
- Insight: Subtle failures in NLP systems are often not due to "lack of intelligence" but rather "engineering design flaws"—the tokenizer, as the lowest-level component of an LLM, has design decisions that cascade into all upper-level language capabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Precisely pinpoints the causal relationship between tokenizer inconsistency and grammatical capability, highly original.
- **Practicality**: ⭐⭐⭐ — Reveals an important problem but does not provide a complete solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Double validation via perplexity and translation; the consistent-tokenization ablation study is exquisitely designed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear introduction of linguistic background, rigorous experimental narrative logic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)

</div>

<!-- RELATED:END -->
