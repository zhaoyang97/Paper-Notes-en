---
title: >-
  [Paper Note] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization
description: >-
  [NeurIPS 2025][LLM Pretraining][non-canonical tokenization] This paper reveals that LLMs can secretly handle non-canonical tokenizations (e.g., splitting "Hello" into "He"+"llo" instead of the canonical whole-word token)—even when the input token sequence differs from training, models exhibit surprising robustness. This capability stems from the property that sub-word embeddings in the embedding space can linearly combine to approximate whole-word embeddings.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - non-canonical tokenization
  - character-level
  - robustness
  - embedding space
  - vocabulary attack
date: 2026-05-08
content_hash: 5bfbcfe23077c0e8
---

# Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization

**Conference**: NeurIPS 2025
**arXiv**: [2506.19004](https://arxiv.org/abs/2506.19004)
**Code**: Available
**Area**: LLM Pretraining
**Keywords**: non-canonical tokenization, character-level, robustness, embedding space, vocabulary attack

## TL;DR
This paper reveals that LLMs can secretly handle non-canonical tokenizations (e.g., splitting "Hello" into "He"+"llo" instead of the canonical whole-word token)—even when the input token sequence differs from training, models exhibit surprising robustness. This capability stems from the property that sub-word embeddings in the embedding space can linearly combine to approximate whole-word embeddings.

## Background & Motivation

**State of the Field**: Modern LLMs employ sub-word tokenizers such as BPE/WordPiece, with identical canonical tokenization used during both training and inference. However, adversarial attacks, multilingual mixing, and OCR noise may produce non-canonical tokenizations.

**Limitations of Prior Work**:
- It is assumed that LLMs can only process canonical token sequences seen during training.
- Non-canonical tokenizations (e.g., decomposing a word into finer fragments) are believed to cause catastrophic performance degradation.
- A systematic understanding of LLM robustness to tokenization variations is lacking.

**Root Cause**: Are LLMs truly this fragile—requiring precisely canonical token sequences to function? If not, where does this robustness originate?

**Paper Goals**: Systematically test and explain LLM behavior under non-canonical tokenizations.

**Starting Point**: Canonical tokens are randomly split into sub-tokens according to various strategies; model performance is evaluated across diverse tasks; and the geometric structure of the embedding space is examined to explain the source of robustness.

**Core Idea**: The LLM embedding space exhibits "sub-word linear additivity"—the embedding sequence of split tokens can approximately reconstruct the representation of the canonical token after a few Transformer layers.

## Method

### Overall Architecture
**Experimental design**: Canonical tokenized input → split into non-canonical token sequences under different strategies → fed into an unmodified LLM → measure the degree of output quality degradation. **Analysis**: Examine the representational distance between split tokens and original tokens in the embedding space.

### Key Designs

1. **Non-Canonical Tokenization Strategies**:

    - **Function**: Systematically generate different types of non-canonical tokenizations.
    - **Mechanism**: (a) Random split—split each token into two sub-tokens at a random position; (b) Character-level—decompose all words to individual characters; (c) Maximal/minimal sub-word—split using different greedy strategies.
    - **Design Motivation**: Different splitting strategies test varying degrees of deviation, ranging from mild to extreme.

2. **Embedding Space Analysis**:

    - **Function**: Explain the source of robustness.
    - **Mechanism**: Measure the cosine similarity between the hidden states of split sub-tokens after $k$ Transformer layers and those of canonical tokens. High similarity is observed as early as intermediate layers.
    - **Design Motivation**: If a split sequence can be "repaired" back to the canonical representation within a few layers, this explains why the final output is largely unaffected.

3. **Cross-Task Evaluation**:

    - **Function**: Validate robustness across diverse NLP tasks.
    - **Mechanism**: Evaluate on tasks including text completion, QA, classification, and translation, using models such as GPT-2, LLaMA, and Mistral.
    - **Design Motivation**: Confirm that robustness is not an artifact of a specific task or model.

### Loss & Training
- No training is required—this is a purely inference-time analysis.

## Key Experimental Results

### Main Results
Performance retention rates of various models under random splitting:

| Model | Canonical | Random Split (50%) | Character-Level | Retention Rate |
|---|---|---|---|---|
| GPT-2 | 100% | ~85–90% | ~70–80% | High |
| LLaMA-7B | 100% | ~90–95% | ~75–85% | Higher |
| Mistral-7B | 100% | ~90–95% | ~80–85% | Higher |

### Ablation Study: Embedding Space Alignment

| Layer Depth | Cosine Similarity: Split Token vs. Canonical Token |
|---|---|
| Layer 0 (input embedding) | ~0.6 |
| Layer 4 | ~0.85 |
| Layer 8 | ~0.92 |
| Final layer | ~0.95 |

### Key Findings
- **LLMs exhibit surprising robustness to non-canonical tokenizations**: Randomly splitting 50% of tokens leads to only ~5–15% performance degradation.
- **Larger models are more robust**: LLaMA-7B retains more performance than GPT-2 (124M).
- **Embedding repair occurs in early layers**: As few as 4–8 Transformer layers suffice for the representations of split tokens to align closely with canonical representations.
- **Extreme character-level splitting remains acceptable**: Even when fully decomposed to the character level, models can still complete most tasks.
- **Security implication**: Adversarial attacks based on tokenization manipulation may be less effective than previously assumed.

## Highlights & Insights
- **"Sub-word linear additivity in the embedding space"** is a theoretically valuable finding—it suggests that the early layers of a Transformer act as an implicit "re-tokenization" mechanism.
- This has direct practical implications for **robust LLM deployment**: even when the tokenizer makes errors (e.g., due to OCR noise or malicious input), the model will not fail catastrophically.
- It prompts a **reassessment of adversarial robustness research**: many token-manipulation-based attacks may have been overestimated.

## Limitations & Future Work
- Only token splitting (decomposing larger tokens into smaller ones) is tested; token merging (combining adjacent tokens) is not examined.
- Tasks with stronger dependencies on precise tokenization, such as mathematical reasoning, may behave differently.
- The embedding alignment analysis is observational and lacks a rigorous theoretical explanation.
- The effect of non-canonical tokenization on syntax-sensitive tasks such as code generation is not evaluated.
- Analysis is limited to English and Latin scripts; robustness for non-Latin scripts (e.g., Chinese, Arabic) may differ.

## Related Work & Insights
- **vs. adversarial NLP attack research**: Such work assumes that tokenization manipulation can effectively attack LLMs; this paper challenges that assumption.
- **vs. character-level LLMs**: Models such as ByT5 are trained from scratch at the character level; this paper demonstrates that standard BPE-trained models can "secretly" also process character-level input.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A counter-intuitive finding that challenges the assumption that tokenization is critical to LLM functioning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models, tasks, splitting strategies, and embedding analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear and intuitive.
- Value: ⭐⭐⭐⭐ Important implications for LLM robustness and security research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](lm_behavioral_phases.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](differentiable_hierarchical_visual_tokenization.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](../../ICLR2026/llm_pretraining/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)

<!-- RELATED:END -->
