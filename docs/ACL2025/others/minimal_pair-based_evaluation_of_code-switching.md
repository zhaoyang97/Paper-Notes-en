---
title: >-
  [Paper Note] Minimal Pair-Based Evaluation of Code-Switching
description: >-
  [ACL 2025][Code-switching] This paper proposes a minimal pair-based evaluation method for code-switching (CS), collecting up to 1000 minimal pairs for each of 11 language pairs. It is found that both bilinguals and large-scale LLMs prefer naturally occurring CS sentences. Furthermore, larger models show more consistent preferences, and the manipulation of closed-class words produces the largest probability differences.
tags:
  - "ACL 2025"
  - "Code-switching"
  - "minimal pairs"
  - "bilingual evaluation"
  - "language models"
  - "linguistic theory"
date: 2026-05-08
content_hash: e0522c2d6fc61169
---

# Minimal Pair-Based Evaluation of Code-Switching

**Conference**: ACL 2025  
**arXiv**: [2506.01840](https://arxiv.org/abs/2506.01840)  
**Code**: None  
**Area**: Others  
**Keywords**: Code-switching, minimal pairs, bilingual evaluation, language models, linguistic theory

## TL;DR

This paper proposes a minimal pair-based evaluation method for code-switching (CS), collecting up to 1000 minimal pairs for each of 11 language pairs. It is found that both bilinguals and large-scale LLMs prefer naturally occurring CS sentences. Furthermore, larger models show more consistent preferences, and the manipulation of closed-class words produces the largest probability differences.

## Background & Motivation

**Background**: Code-switching (CS) is the phenomenon where bilingual speakers alternate between two languages within a single utterance, serving as an important research subject in multilingual NLP. Evaluating whether LLMs can understand and process CS like real bilinguals is a crucial dimension for measuring the multilingual capabilities of LLMs.

**Limitations of Prior Work**: Existing CS evaluation methods suffer from three main drawbacks: (1) narrow language coverage, typically targeting only a few language pairs like English-Spanish; (2) failure to cover diverse CS phenomena (such as syntactic constraints and different switching patterns of closed-class versus open-class words); (3) lack of scalability—manual annotation of CS quality is extremely costly and highly subjective. There currently lacks a scalable evaluation framework with broad language coverage and linguistic theoretical support.

**Key Challenge**: High-quality CS evaluation requires bilingual annotation, which is not scalable, whereas automatic evaluation methods lack linguistic theoretical backing. How can we strike a balance between scalability and evaluation validity?

**Goal**: Design a minimal pair-based CS evaluation method to automatically construct evaluation pairs through minimal manipulation, while validating its effectiveness through human experiments.

**Key Insight**: Inspired by the minimal pair methodology in linguistics, which observes the effect of changing only one factor. Minimal modifications (such as switching a word at a specific code-switch point back to the matrix language) are applied to natural CS sentences to form "natural vs. manipulated" contrastive pairs.

**Core Idea**: If a CS sentence is natural, both bilinguals and good-performing language models should prefer the original version after minimally disrupting a specific switching pattern. This degree of preference can serve as an indicator of CS comprehension performance.

## Method

### Overall Architecture

The proposed method consists of three steps: (1) collecting natural CS corpora from multilingual social media (Twitter/X); (2) automatically generating minimal pair variants for each CS sentence—replacing a word at a code-switch point with its translation equivalent in the matrix language, so that switching no longer occurs; (3) validating the effectiveness of the minimal pairs via bilingual human experiments, and then performing probability evaluation using LLMs. The input is a minimal pair (natural version + manipulated version) of a CS sentence, and the output is the model's computed probability (or perplexity) for both.

### Key Designs

1. **Minimal Pair Construction Strategy**:

    - **Function**: Generate a minimally modified variant for each natural CS sentence.
    - **Mechanism**: Identify the code-switch points in a sentence and replace words from the embedded language with translation equivalents from the matrix language. For example, in the English-Spanish CS sentence "I went to the tienda yesterday", "tienda" is replaced back with "store". The manipulation types cover switches of both open-class words (nouns, verbs, etc.) and closed-class words (articles, prepositions, etc.).
    - **Design Motivation**: The essence of the minimal pair method lies in changing only one variable, so that preference differences can be directly attributed to the naturalness of CS rather than other semantic factors.

2. **Multilingual Pair Coverage and Data Collection**:

    - **Function**: Ensure that the evaluation is not limited to a few language pairs and achieves broad linguistic coverage.
    - **Mechanism**: Eleven language pairs are covered, including combinations of English with Spanish, Hindi, Tagalog, Arabic, etc., with up to 1000 minimal pairs collected for each pair. The data is sourced from natural bilingual user tweets on Twitter/X, filtered using language identification tools to extract tweets containing CS.
    - **Design Motivation**: CS patterns vary across language pairs (influenced by grammatical structures), and broad coverage is required to draw generalized conclusions.

3. **Human Validation + LLM Probability Evaluation**:

    - **Function**: Establish ground truth for the evaluation method and test the CS comprehension ability of LLMs.
    - **Mechanism**: In human experiments, bilinguals of respective language pairs are invited to make preference judgments on the minimal pairs. In LLM evaluations, token-level probabilities of the models on both natural CS sentences and manipulated variants are calculated, comparing the sum of log probabilities of the two. If a model assigns a higher probability to the natural CS sentence, it indicates that the model captures natural CS patterns.
    - **Design Motivation**: Human judgment serves as the gold standard to validate the effectiveness of the minimal pair setup itself, while LLM probability-based evaluation avoids requiring models to generate CS text (bypassing generation quality issues) and directly measures the language knowledge of models.

### Loss & Training

As this study focuses on an evaluation framework, it does not involve model training. The core evaluation metric is as follows: for each minimal pair, compare the log probabilities assigned by the model to the natural CS sentence and the manipulated variant, i.e., $\log P(s_{natural})$ vs $\log P(s_{manipulated})$, counting the ratio of instances where the model prefers the natural version.

## Key Experimental Results

### Human Preference Experiments

| Language Pair | Bilingual Preference Ratio for Natural CS | Sample Size |
|--------|----------------------|--------|
| en-es | >70% | ~1000 |
| en-hi | >65% | ~1000 |
| en-tl | >65% | ~800 |
| en-ar | >60% | ~600 |
| Average across pairs | **Consistent preference for natural CS** | 11 pairs |

### LLM Preference Experiments (by Model Scale)

| Model | Scale | Preference Ratio for Natural CS | Closed-class Word Difference |
|------|------|---------------|-------------|
| Small Models (~1B) | Small | ~55% | Low |
| Medium Models (~7B) | Medium | ~62% | Medium |
| Large Models (~70B+) | Large | ~70%+ | **Largest** |

### Key Findings

- Across **all 11 language pairs**, bilinguals consistently preferred natural CS sentences, validating the effectiveness of the minimal pair method.
- **Larger models show higher consistency in preferring natural CS**, demonstrating a clear scaling law. This indicates that CS understanding capabilities scale with model size.
- **Manipulation of closed-class words yields the largest probability differences**—which aligns with linguistic theory (Myers-Scotton's Matrix Language Frame model): closed-class words (articles, prepositions, etc.) in CS typically come from the matrix language, and changing them severely violates CS syntactic constraints.
- Switching open-class words (nouns, etc.) is more flexible, hence resulting in relatively smaller probability differences after manipulation.

## Highlights & Insights

- **Methodological Innovation**: The classic linguistic methodology of minimal pairs is systematically applied to CS evaluation, cleverly dodging the difficulty of having models generate CS text (by simply comparing probabilities) while remaining theoretically grounded. This concept can be extended to evaluate other linguistic phenomena.
- **Discovery of Scaling Law**: CS comprehension ability scales with model size, suggesting that CS knowledge is acquired naturally during large-scale pre-training rather than requiring dedicated training. This offers crucial insights for designing multilingual models.
- **Alignment with Linguistic Theory**: The results on closed-class words validate the Matrix Language Frame theory, showing that LLMs have subtly learned CS syntactic constraints, rather than relying solely on superficial statistical superficialities.

## Limitations & Future Work

- **Data Source Biased towards Social Media**: CS on Twitter/X might not represent the full spectrum of spoken CS, lacking phonetic and spoken switching cues.
- **Minimal Pair Construction Relies on Translation Quality**: Automatic translation replacement may introduce unnatural expressions, especially in language pairs with significant syntactic differences.
- **Evaluates Only Comprehension/Discrimination Ability**: A model's ability to recognize natural CS does not guarantee its capability to generate natural CS; evaluation at the generation level remains an open challenge.
- **11 Language Pairs are Still Limited**: Evaluating CS in low-resource language pairs (such as African language combinations) is an important future direction.
- Future work can combine minimal pair approaches to assess the CS generation capabilities of LLMs, or investigate the impact of different pre-training data ratios on CS comprehension.

## Related Work & Insights

- **vs. LinCE (Aguilar et al. 2020)**: LinCE is the most commonly used CS benchmark, but it primarily bases on sequence labeling tasks and does not directly measure CS preferences. This paper's minimal pair method offers an evaluation perspective closer to linguistic testing.
- **vs. BLI/Dictionary-based Methods**: These methods evaluate cross-lingual capability through word-level translation pairs, neglecting the syntactic level of CS. This paper operates at the sentence level, capturing richer linguistic phenomena.
- This minimal-pair-based evaluation concept can be adjusted to assess LLM's understanding of dialects, register-switching, and other sociolinguistic phenomena.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically applies linguistic methodology to LLM evaluation, offering a novel and theoretically deep framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of 11 language pairs, rigorous validation via human experiments, and comparison across multiple model scales.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, balancing both linguistic and NLP perspectives.
- Value: ⭐⭐⭐⭐ Fills a methodological gap in CS evaluation, providing a new paradigm for multilingual model evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Code-Switching and Syntax: A Large-Scale Experiment](code-switching_and_syntax_a_large-scale_experiment.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Towards Better Evaluation for Generated Patent Claims](patclaimeval_patent_evaluation.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] CiteEval: Principle-Driven Citation Evaluation for Source Attribution](citeeval_principle-driven_citation_evaluation_for_source_attribution.md)

</div>

<!-- RELATED:END -->
