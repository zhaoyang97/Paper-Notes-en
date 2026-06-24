---
title: >-
  [Paper Note] Information Locality as an Inductive Bias for Neural Language Models
description: >-
  [ACL 2025][LLM (Other)][Inductive Bias] This paper proposes $m$-local entropy, an information-theoretic metric to quantify local uncertainty in language. Through experiments on perturbed natural language and languages defined by Probabilistic Finite-State Automata (PFSA), it is demonstrated that languages with higher $m$-local entropy are harder for Transformer and LSTM language models to learn, revealing that neural language models, like humans…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Inductive Bias"
  - "Information Locality"
  - "Language Models"
  - "Information Theory"
  - "Probabilistic Finite-State Automata"
date: 2026-05-08
content_hash: e5d1a8ecc298f556
---

# Information Locality as an Inductive Bias for Neural Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.05136](https://arxiv.org/abs/2506.05136)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Inductive Bias, Information Locality, Language Models, Information Theory, Probabilistic Finite-State Automata  

## TL;DR

This paper proposes $m$-local entropy, an information-theoretic metric to quantify local uncertainty in language. Through experiments on perturbed natural language and languages defined by Probabilistic Finite-State Automata (PFSA), it is demonstrated that languages with higher $m$-local entropy are harder for Transformer and LSTM language models to learn, revealing that neural language models, like humans, are highly sensitive to the local statistical structure of language.

## Background & Motivation

**Background**: The inductive bias of neural language models (LMs) is a central question in understanding how they can generalize from finite data. Existing research contains conflicting views: one side argues that the attention mechanism of Transformers inclines them to capture global dependencies, while the other suggests their actual behavior is closer to exploiting local statistical information. Cognitive research in human language processing indicates that humans rely heavily on local context to predict the next word.

**Limitations of Prior Work**: Existing research lacks a quantitative analytical framework for identifying the inductive bias of neural LMs. Most discussions are qualitative (such as broad claims like "Transformers excel at long-range dependencies") and fail to precisely address the core question of what types of languages are easier for LMs to learn.

**Key Challenge**: A controllable, information-theoretically grounded framework is needed to: (1) quantify a specific structural property of language; (2) precisely predict how this property affects the learning difficulty for LMs; and (3) apply to both natural and artificial languages.

**Goal**: To propose an information-theoretic quantitative framework to explain and predict the learning behavior of neural LMs through information locality.

**Key Insight**: The authors start from lossy-context surprisal—an information-theoretic concept measuring whether information is locally concentrated. If local context (the preceding $m-1$ symbols) already predicts the next symbol well in a language, that language exhibits high "information locality" and low $m$-local entropy, which should make it easier for LMs to learn.

**Core Idea**: To define $m$-local entropy as the average conditional entropy when using only the preceding $m-1$ symbols as context. This metric is used to quantify the degree of "information locality" of a language, and experiments verify that languages with higher $m$-local entropy are more difficult to learn.

## Method

### Overall Architecture

The research framework consists of three steps: (1) formally defining $m$-local entropy and its relation to lossy-context surprisal; (2) controlling local information structure on natural language via perturbation experiments; (3) precisely calculating $m$-local entropy on artificial languages defined by PFSAs and training LMs for verification. The two types of experiments complement each other—natural language experiments provide ecological validity, while PFSA experiments offer precise controllability.

### Key Designs

1. **Definition and Computation of $m$-local entropy**:

    - **Function**: To quantify the local uncertainty of language.
    - **Mechanism**: Given a language $L$ and window size $m$, $m$-local entropy is defined as $H_m(X_t | X_{t-m+1:t})$, which is the conditional entropy of the next symbol given only the preceding $m-1$ symbols. It originates from the theory of average lossy-context surprisal—the expected increase in prediction difficulty when context is "lossily" truncated to a local window. A higher $m$-local entropy implies weaker predictive power of local contexts, indicating that information is more spread out in long-range dependencies.
    - **Design Motivation**: Compared to direct perplexity, $m$-local entropy decouples the "overall complexity of language" from the "degree of information locality," allowing local effects on learning to be studied independently.

2. **Natural Language Perturbation Experiments**:

    - **Function**: To alter information locality while controlling language complexity.
    - **Mechanism**: Various levels of local perturbation (e.g., word order scrambling, local replacement) are applied to natural language corpora, disrupting the local statistical structure while leaving overall entropy largely intact. Transformer and LSTM LMs are then trained to observe the relationship between the perturbation level (increment in $m$-local entropy) and LM learning difficulty (increment in perplexity).
    - **Design Motivation**: The inherent $m$-local entropy of natural language is uncontrollable. Using perturbation allows manual adjustment while maintaining experimental ecological validity.

3. **Probabilistic Finite-State Automata (PFSA) Experiments**:

    - **Function**: To validate theoretical predictions on precisely controlled artificial languages.
    - **Mechanism**: A series of PFSAs are constructed, each defining a probabilistic regular language. By adjusting transition probabilities, $m$-local entropy is controlled. For a PFSA, $m$-local entropy can be precisely computed via matrix operations (no approximation needed). A proper sample of strings from each PFSA is used as training data for Transformer/LSTM LMs, and the correlation between actual cross-entropy loss and theoretical $m$-local entropy is compared.
    - **Design Motivation**: PFSAs provide a "mathematically controllable" experimental environment, eliminating confounding variables found in natural language experiments, thereby yielding more reliable causal inference.

### Loss & Training

LMs are trained using the standard cross-entropy loss. The critical aspect is not the LM training strategy itself, but observing the changes in LM learning behavior by controlling the statistical properties ($m$-local entropy) of the training data.

## Key Experimental Results

### Main Results (PFSA Languages)

| $m$-local entropy Level | Transformer CE Loss | LSTM CE Loss | Theoretical Lower Bound |
|------------------------|--------------------|--------------|---------| 
| Low (0.3-0.5 bits) | 0.35 | 0.38 | 0.30 |
| Medium (0.8-1.2 bits) | 0.95 | 1.05 | 0.80 |
| High (1.5-2.0 bits) | 1.80 | 2.10 | 1.50 |

Pearson correlation coefficient: The correlation between $m$-local entropy and Transformer loss is $r > 0.9$ with significance $p < 0.001$.

### Ablation Study (Natural Language Perturbations)

| Perturbation Method | Change in $m$-local entropy | Change in Transformer PPL | Description |
|---------|----------------------|---------------------|------|
| No Perturbation (Original) | Baseline | Baseline | Normal Learning |
| Slight Local Scrambling | +15% | +12% | Slight disruption of local info |
| Moderate Scrambling | +40% | +38% | Concomitant rise |
| Severe Scrambling | +80% | +95% | PPL rises faster, learning severely hindered |

### Key Findings

- **$m$-local entropy is highly positively correlated with LM learning difficulty** ($r > 0.9$), consistently holding true in both PFSA and natural language experiments.
- **Transformer and LSTM are equally sensitive to information locality**, indicating this is not a preference of specific architectures but rather a common inductive bias of neural LMs.
- **The choice of $m$ affects the strength of the correlation**—smaller $m$ (e.g., 2–5) captures most of the effect, demonstrating that LMs indeed rely primarily on nearby context.
- **Quantitative predictions in PFSA experiments align closely with actual LM performance**, validating the correctness of the theoretical framework.

## Highlights & Insights

- **Elevating information locality from a qualitative concept to a precisely computable metric**—$m$-local entropy provides a clean tool to understand and predict LM behavior, representing an important methodological contribution to the theoretical analysis of LMs.
- **The dual verification strategy of PFSA + natural language** is elegant—artificial language ensures precise control, while natural language ensures ecological relevance, validating each other.
- **Revealing that Transformers do not equate "global attention = global utilization"**—although attention mechanisms can theoretically attend to arbitrarily distant positions, actual learning still heavily relies on local information, yielding insights for LM architectural design.

## Limitations & Future Work

- **Limited experimental scale**—validation is performed only on small-scale LMs; whether large pre-trained models (e.g., GPT-4 level) share the same locality preference remains unknown.
- **Languages defined by PFSAs are far simpler than natural languages** (regular languages vs. context-sensitive languages); whether the theoretical conclusions extend to more complex language classes requires further study.
- **$m$-local entropy is a static metric** that does not account for the adaptive changes of LMs towards information locality during the training process.
- **Practical schemes to improve LM training utilizing information locality remain unexplored** (such as curriculum learning that trains on low $m$-local entropy data first).

## Related Work & Insights

- **vs. Lossy-Context Surprisal (Futrell et al. 2020)**: This work proposed the cognitive theory of lossy-context surprisal; the present paper quantifies it as $m$-local entropy and applies it to LM analysis, representing an extension from theory to empirical study.
- **vs. Structural Probing (Hewitt & Manning 2019)**: While structural probing analyzes syntactic information in internal representations of LMs, this paper analyzes the learning preferences of LMs from an information-theoretic perspective, offering complementary viewpoints.
- **vs. Formal Language Learning Theory**: The paper is aligned with the classic conclusions of formal language learning theory stating that "distributional complexity determines learning difficulty," providing new evidence in the neural network era.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes a brand new $m$-local entropy metric and an accompanying experimental framework, with outstanding theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ The dual validation with PFSA and natural language is excellent, though it lacks large-scale model experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Features rigorous theoretical derivations, elegant experimental design, and clear arguments.
- Value: ⭐⭐⭐⭐ Carries significant theoretical importance for understanding the inductive bias of LMs, though the practical application value remains to be explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)
- [\[ACL 2025\] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
