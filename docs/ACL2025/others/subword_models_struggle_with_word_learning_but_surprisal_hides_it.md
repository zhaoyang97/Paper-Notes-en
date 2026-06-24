---
title: >-
  [Paper Note] Subword Models Struggle with Word Learning, but Surprisal Hides It
description: >-
  [ACL 2025][Word learning] Through the psycholinguistic lexical decision task, this paper reveals that subword (BPE) language models are far inferior to character-level models in isolated word recognition, and that the commonly used surprisal metric masks this deficiency by introducing syntactic context.
tags:
  - "ACL 2025"
  - "Word learning"
  - "subword tokenization"
  - "character-level models"
  - "lexical decision"
  - "surprisal"
date: 2026-05-08
content_hash: 286b281a257a16a8
---

# Subword Models Struggle with Word Learning, but Surprisal Hides It

**Conference**: ACL 2025  
**arXiv**: [2502.12835](https://arxiv.org/abs/2502.12835)  
**Code**: Yes  
**Area**: NLP / Cognitive Linguistics  
**Keywords**: Word learning, subword tokenization, character-level models, lexical decision, surprisal

## TL;DR

Through the psycholinguistic lexical decision task, this paper reveals that subword (BPE) language models are far inferior to character-level models in isolated word recognition, and that the commonly used surprisal metric masks this deficiency by introducing syntactic context.

## Background & Motivation

When acquiring language, humans first learn to recognize words and then understand grammar. However, existing studies using LMs as language acquisition models focus heavily on the syntactic level, paying insufficient attention to the implicit "word learning" process. Prior research on word learning primarily uses **surprisal (negative log-probability)** to measure whether a model has "learned" a word. However, surprisal essentially measures "the predictability of a word given a context," directly corresponding to the training objective, and fails to reveal whether the model possesses independent lexical knowledge.

Furthermore, subword tokenization methods such as BPE split words into linguistically implausible sub-units, which is cognitively implausible. In contrast, character-level models avoid this a priori segmentation and theoretically should align more closely with human word discovery.

The core problem of this paper is: **Do LMs "know" which strings are valid words?** This is more fundamental than "Can LMs predict the occurrence of words in context?"

## Method

### Overall Architecture

The authors design three experimental paradigms to probe the lexical knowledge of models, progressing from context-free to context-dependent:

1. **Lexical Decision** — Context-free
2. **Surprisal** — Plausible context
3. **Anti-Surprisal** — Implausible context

### Key Designs

1. **Lexical Decision Task**
    - **Function**: Given a real-word/pseudo-word pair (e.g., sending / monding), determine which is the real word.
    - **Mechanism**: Calculate the average surprisal of both words solely following a space character (the most neutral starting token) and compare them.
    - **Design Motivation**: Simulate the forced-choice lexical decision in psycholinguistics, stripping away syntactic/semantic contextual interference.
    - Generate 1000 minimal pairs of high-frequency words and 1000 minimal pairs of low-frequency words using the Wuggy tool.

2. **Surprisal Experiment**
    - **Function**: Measure the surprisal of real words vs. pseudo-words in plausible syntactic contexts.
    - **Mechanism**: Sample sentences containing target words from OpenSubtitles and replace the target words with matched pseudo-words.
    - **Design Motivation**: Test whether the model can better distinguish real words from pseudo-words when "syntactic context is available."

3. **Anti-Surprisal Experiment**
    - **Function**: Insert real words/pseudo-words into mismatched contexts.
    - **Mechanism**: Select sentences that do not contain the target word and insert it randomly at index $\ge 3$.
    - **Design Motivation**: Provide lexical context without semantic/syntactic cues to test whether the mere "presence of other words" helps in judgment.

4. **Learning Trajectory Analysis**
    - Save 19 intermediate checkpoints logarithmically.
    - Evaluate simultaneously on the BLiMP (syntactic benchmark) and the lexical decision task.
    - Compare the temporal relationship between "lexical learning" and "syntactic learning" in character-level vs. subword models.

### Model Configuration

| Model | Tokenization | Parameters | Training Data |
|------|----------|--------|----------|
| Llama (×3) | Character/BPE | 0.49M-30M | BabyLM 10M |
| GPT-2 (×2) | Character/BPE | 85-97.5M | 100M words |
| Pythia (×6) | BPE | 14M-1.4B | 825GB |

## Key Experimental Results

### Main Results — Lexical Decision vs Surprisal (Table 1 Summary)

| Model | Tokenization | Lexical Decision (High/Low Freq) | Surprisal (High/Low Freq) | Anti-Surprisal (High/Low Freq) |
|------|------|---------------------|---------------------|-------------------------|
| Llama-0.49M | Character | 97.6/83.0 | 98.2/84.3 | 98.0/83.1 |
| Llama-21.9M | Character | 99.0/93.3 | 99.8/94.7 | 99.0/92.5 |
| GPT-2 | Character | 98.7/97.3 | 99.8/99.4 | 98.0/96.3 |
| Llama-30M | BPE | 83.6/68.6 | 92.7/81.1 | 83.7/76.1 |
| Pythia-1.4B | BPE | 87.8/81.6 | 97.9/97.9 | 76.5/84.7 |
| GPT-2 | BPE | 35.6/79.1 | 99.0/99.2 | 84.7/86.9 |

### Key Findings

1. **Character-level models are near-perfect in lexical decision** (97-99%), whereas even the largest BPE models achieve only around 88%.
2. **Surprisal masks the gap**: In the presence of context, BPE models catch up (>90%), but this relies heavily on syntactic signals.
3. **Anti-Surprisal reveals the struggles of BPE**: BPE models instead prefer pseudo-words in implausible contexts, indicating that their lexical and syntactic knowledge are inseparable.
4. **Learning trajectories differ significantly**:
    - Character-level models: Lexical learning precedes syntactic learning, with the two curves clearly separated.
    - BPE models: Lexical and syntactic learning trajectories are highly correlated and occur simultaneously, showing an S-shaped curve.
5. BPE models exhibit a persistent performance gap between high-frequency and low-frequency words, which cannot be bridged by scaling up the model.

### Ablation Study

- Results are consistent across different model architectures (Llama/GPT-2/Pythia).
- The benefit of scaling model size is limited for BPE models.
- Differences in training data volume (10M vs. 825GB) do not alter the overall trend of character-level vs. BPE models.

## Highlights & Insights

1. **Methodological Innovation**: Introduces the psycholinguistic lexical decision paradigm to LM evaluation, filling the gap of "context-independent lexical probing."
2. **Profound Insight**: Surprisal has a fundamental flaw as a metric for evaluating word learning—it directly aligns with the training objective and fails to truly probe abstract lexical knowledge.
3. **Cognitive Implication**: The learning trajectory of character-level models (lexical before syntactic) aligns better with the language acquisition order of human children.
4. **Empirical Finding**: The a priori segmentation of BPE tokenization effectively "skips" the word discovery phase, causing lexical and syntactic learning to become entangled.

## Limitations & Future Work

- Experiments are conducted only in English; different writing or phonetic systems may exhibit different patterns.
- Character-level models are tested only at small scales, lacking validation on large-scale character-level models.
- The semantic/referential dimensions of word learning (such as object naming) are not covered.
- The performance of morphology-aware tokenizers has not been explored.

## Related Work & Insights

- While intuitive, the surprisal threshold method of Chang & Bergen (2022) identifies frequent function words as "earliest learned," which contradicts actual child production.
- Le Godais et al. (2017) previously observed ~95% lexical decision accuracy in character-level LSTMs.
- The choice of tokenization method should be handled with greater caution in BabyLM and language acquisition simulations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic use of the lexical decision paradigm in benchmarking Transformer LMs, offering a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive analysis across multiple models, architectures, tokenization methods, and learning trajectories.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Highly logical, tight integration between motivation and experiments, with clear illustrations.
- **Value**: ⭐⭐⭐⭐ — Statistically significant for understanding internal lexical representations in LMs, particularly crucial for the BabyLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Support Samples of Next Word Prediction](on_support_samples_of_next_word_prediction.md)
- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ACL 2025\] MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](../../NeurIPS2025/others/aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)

</div>

<!-- RELATED:END -->
