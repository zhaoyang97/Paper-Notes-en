---
title: >-
  [Paper Note] Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs
description: >-
  [ACL 2025][Impossible Languages] By training GPT-2 small on 12 languages, this study systematically tests whether language models (LMs) can distinguish possible languages (natural languages) from impossible ones (scrambled word orders, etc.). The findings reveal that LMs exhibit partial human-like learning biases but are not perfect—they can differentiate within a single language but fail to achieve complete separation cross-linguistically. Furthermore…
tags:
  - "ACL 2025"
  - "Impossible Languages"
  - "Linguistic Typology"
  - "Cognitive Modeling"
  - "GPT-2"
  - "Greenberg Universal 20"
date: 2026-05-08
content_hash: 003cd28bc67db20d
---

# Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs

**Conference**: ACL 2025  
**arXiv**: [2502.18795](https://arxiv.org/abs/2502.18795)  
**Code**: [GitHub](https://github.com/picol-georgetown/multilingual-LM)  
**Area**: Computational Linguistics / Cognitive Science  
**Keywords**: Impossible Languages, Linguistic Typology, Cognitive Modeling, GPT-2, Greenberg Universal 20

## TL;DR

By training GPT-2 small on 12 languages, this study systematically tests whether language models (LMs) can distinguish possible languages (natural languages) from impossible ones (scrambled word orders, etc.). The findings reveal that LMs exhibit partial human-like learning biases but are not perfect—they can differentiate within a single language but fail to achieve complete separation cross-linguistically. Furthermore, in noun phrase (NP) word order experiments, generalization testing (rather than perplexity) is found to reflect typological preferences.

## Background & Motivation

**Background**: Debate strictly continues regarding whether LLMs can serve as cognitive models for human language acquisition. Proponents argue LLMs' linguistic capabilities reflect human language theory, whereas opponents (e.g., Chomsky et al.) contend that LLMs use fundamentally different cognitive mechanisms from humans and can learn any arbitrary input, rendering their success meaningless for understanding human language.

**Limitations of Prior Work**: Kallini et al. (2024) first demonstrated on English that GPT-2 can distinguish possible from impossible languages. However, their study was limited to English, leaving the cross-linguistic generalizability of these findings unknown. Additionally, learning behaviors on typologically unattested (yet theoretically possible) languages have not been explored.

**Key Challenge**: The "anything goes" hypothesis (where LMs treat all languages equally) vs. whether LMs' actual preferences align with those of humans.

**Goal**: (1) Can LMs distinguish possible vs. impossible languages cross-linguistically? (2) Do LMs exhibit typological preferences, i.e., are they more likely to learn word orders that are typologically common?

**Key Insight**: Build two parallel corpora (OPUS12: 12 languages, 10M tokens; OPUS30: 30 languages, 0.7M tokens) to ensure cross-linguistic comparability, while introducing NP word order tests based on Greenberg's Universal 20.

## Method

### Overall Architecture

Three sets of experiments: (1) Within-language comparison: each natural language vs. its impossible variants; (2) Cross-linguistic comparison: whether all natural languages are completely separable from all impossible languages; (3) Learning differences between attested vs. unattested NP word orders.

### Key Designs

1. **Parallel Corpus Construction (OPUS12/OPUS30)**:
    - **Function**: Build sentence-aligned multilingual parallel corpora from 5 OPUS sources.
    - **Mechanism**: OPUS12 contains 12 languages (from 4 language families), with the English portion having ~10M tokens (equivalent to the input of a child aged 2-5). OPUS30 contains 30 languages and is used for test sets. Parallel corpora ensure consistent content (information content) across different languages, thereby isolating the effect of formal linguistic features on learnability.
    - **Design Motivation**: Most existing studies use non-parallel corpora, where differences in information content across different language texts confound the comparison of learnability.

2. **Impossible Language Construction**:
    - **Function**: Generate various "impossible" variants for each natural language—deterministic scrambling (3 seeds), local window scrambling ($w=2,3,5,10$), complete reversal, and parity reordering.
    - **Mechanism**: The scrambling operations are deterministic, meaning the original language can be recovered via an inverse transformation. If LMs were merely general pattern matchers (as opponents claim), they should be able to learn these variants equally well.
    - **Design Motivation**: Scrambling operations were chosen because (a) they were identified by Kallini et al. as "most impossible" language types; (b) human studies also show that humans have a strong preference for regularization.

3. **NP Word Order Generalization Testing ($\Delta\text{GenScore}$)**:
    - **Function**: Propose the $\Delta\text{GenScore}$ metric to test the generalization ability of models trained on different NP word orders.
    - **Mechanism**: $\Delta\text{GenScore} = \text{GenScore}_{\checkmark} - \text{GenScore}_{\times}$, comparing whether a model trained on a natural language or on an unnatural word order generalizes better to the other's test data. A $\Delta\text{GenScore} > 0$ indicates that the natural language model has superior generalization ability.
    - **Design Motivation**: Perplexity cannot distinguish natural vs. unnatural in NP word order experiments (since standardized NPs actually possess greater regularity and lower entropy), but generalization tests can reveal the model's intrinsic preferences.

### Loss & Training

GPT-2 small is trained independently for each language using a pre-trained BPE tokenizer for each language (vocabulary size ~50k). Each configuration is run with 3 random seeds, up to a maximum of 1,200 training steps, with 120 warmup steps. Evaluation uses geometric mean perplexity (on a parallel test set of 10K sentences).

## Key Experimental Results

### Main Results

Experiment 1 (Within-language): Across all 12 languages except Italian, natural languages exhibit lower perplexity than their impossible variants.

| Language | Natural Language Perplexity | Closest Impossible Variant Perplexity | Statistically Significant? |
|------|-------------|-------------------|----------|
| English | ~15 | ~17 (shuffle_local $w=2$) | Yes |
| Chinese | ~8 | ~10 (shuffle_local $w=2$) | Yes |
| Arabic | ~35 | ~37 (shuffle_local $w=2$) | Yes |
| Italian | ~20 | ~19.5 (shuffle_local $w=2$) | No ($p=0.353$) |

Experiment 2 (Cross-linguistic): The macro F1 of the linear SVM classifier is 0.75, showing that complete separation is not possible.

### Ablation Study

Experiment 3 (NP Word Order)—$\Delta\text{GenScore}$ Analysis:

| NP Word Order Variant | Typological Status | English $\Delta\text{GenScore}$ | Chinese $\Delta\text{GenScore}$ |
|-----------|-----------|--------------|--------------|
| Nnda | Unattested | + Positive (Consistent) | + Positive (Consistent) |
| anNd | Unattested | + Positive (Consistent) | + Positive (Consistent) |
| daNn | Rarely Attested | + Positive | + Positive |
| dnaN ($\approx$English) | Frequently Attested | + Positive | + Positive |
| dnNa ($\approx$Italian) | Frequently Attested | Mixed | + Positive |

### Key Findings

1. **Local scrambling (small window) is harder to distinguish than global scrambling**: The perplexity of shuffle_local ($w=2$) is closest to that of natural language; for some languages (Italian), they are even indistinguishable.
2. **Complete separation is impossible cross-linguistically**: Some impossible languages (e.g., English shuffle_local $w=3$) have lower perplexity than certain natural languages (e.g., Russian, Arabic).
3. **Correlation between perplexity and TCW (tokens per word) is not significant** ($\rho=0.564, p=0.076$), indicating morphological complexity is not the main driver of perplexity differences.
4. **$\Delta\text{GenScore}$ distinguishes typological preferences**: Unattested word orders (Nnda, anNd) consistently yield positive $\Delta\text{GenScore}$ values, demonstrating that the model generalizes better when trained on natural word orders.

## Highlights & Insights

- **Empirical support for a moderate position**: LMs are neither "anything goes" universal learners nor do they hold purely human-like learning biases; rather, they sit somewhere along a continuum.
- **Decoupling of perplexity vs. generalization testing**: Perplexity as a metric may lack sensitivity (due to influence from textual entropy), whereas generalization tests (using minimal pairs) are more capable of revealing intrinsic preferences.
- **Constituency preservation hypothesis**: Scrambling that preserves phrase structure is easier to learn than scrambling that disrupts phrase structure, explaining why perplexity for internal NP scrambling is lower than for global scrambling.
- **Multilingual parallel corpora methodology**: Controlling for information content by constructing a content-consistent parallel corpus represents a key methodological contribution to cross-linguistic learnability comparisons.

## Limitations & Future Work

- Only GPT-2 small is evaluated; larger models might display different behaviors.
- The training data size is limited to 10M tokens, which is relatively small.
- The NP word order experiments only cover 4 languages that have constituency parsers.
- The parallel corpus may contain noise and has not been manually verified.
- Other types of impossible languages, such as count-based grammars, are not explored.

## Related Work & Insights

- **Kallini et al. (2024)**: The direct predecessor of this work, which first proved that LMs can distinguish possible/impossible languages using English.
- **Culbertson & Newport (2015)**: Humans show a learning bias toward harmonic NP word orders (like dnaN); this study finds that LMs exhibit similar, though weaker, preferences.
- **Insight**: The learning bias of LMs may stem from the autoregressive architecture's intrinsic sensitivity to local dependencies, rather than any deep understanding of "language."

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to extend impossible language research to 12 languages, introducing NP word order tests and the $\Delta\text{GenScore}$ metric.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Explores 12 languages, multiple scrambling methods, and several analytical dimensions, though limited by the use of GPT-2 small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear research questions, rigorous methodology, and depth of discussion.
- **Value**: ⭐⭐⭐⭐ Provides crucial empirical evidence to the debate on LLMs as cognitive models; the $\Delta\text{GenScore}$ methodology is a useful contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)
- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)

</div>

<!-- RELATED:END -->
