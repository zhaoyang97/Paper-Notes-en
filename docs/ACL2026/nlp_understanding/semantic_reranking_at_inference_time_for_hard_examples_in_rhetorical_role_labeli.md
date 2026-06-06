---
title: >-
  [Paper Note] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling
description: >-
  [ACL 2026][NLP Understanding][Rhetorical Role Labeling] Ours proposes RiSE, an inference-time semantic reranking framework that automatically identifies low-confidence hard examples and reranks model outputs using label…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Rhetorical Role Labeling"
  - "Inference-time Reranking"
  - "Label Semantics"
  - "Contrastive Learning"
  - "Hard Examples"
date: 2026-05-08
content_hash: fea0d08b5c0dfd2a
---

# Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling

**Conference**: ACL 2026  
**arXiv**: [2605.18007](https://arxiv.org/abs/2605.18007)  
**Code**: [GitHub](https://github.com/AnasBelfathi/rise-framework)  
**Area**: medical_imaging  
**Keywords**: Rhetorical Role Labeling, Inference-time Reranking, Label Semantics, Contrastive Learning, Hard Examples

## TL;DR
Ours proposes RiSE, an inference-time semantic reranking framework that automatically identifies low-confidence hard examples and reranks model outputs using label semantic representations learned through contrastive learning, achieving an average improvement of +9.15 macro-F1 on hard examples across 8 rhetorical role labeling datasets.

## Background & Motivation
**Background**: Rhetorical Role Labeling (RRL) assigns functional roles to each sentence in a document, widely applied in legal, medical, and scientific domains. Language models perform well on average performance but remain unreliable on hard examples.

**Limitations of Prior Work**: (1) Existing methods treat labels as discrete identifiers, ignoring semantic information encoded in label names; (2) The processing of hard examples (low-confidence predictions) is usually implicit and lacks specialized mechanisms; (3) Confusion frequently occurs between semantically similar labels, where the macro-F1 gap between Top-1 and Top-3 suggests that correct labels are often ranked highly but not selected.

**Key Challenge**: Standard classifiers use one-hot vectors to represent labels, which cannot exploit semantic relationships between labels; meanwhile, pure similarity-based methods utilize label semantics but lose the discriminative power of the classifier.

**Goal**: Design an inference-time method that utilizes label semantics to improve hard example prediction while maintaining the discriminative behavior of the classifier.

**Key Insight**: Inference-time intervention—automatically detecting low-confidence samples and reweighting classifier logits using text-label semantic similarity derived from contrastive learning.

**Core Idea**: Confusion-weighted contrastive learning + Adaptive hard example detection + Logit semantic reranking = Inference-time hard example recovery without retraining.

## Method

### Overall Architecture
RiSE operates at inference time: it first generates logits for the input sentence using a base classifier; then, it automatically identifies hard examples based on logit variance; for hard examples, it calculates semantic similarity using text-label representations from contrastive learning and reranks logits via element-wise multiplication. Easy samples directly use the original output.

### Key Designs
1. **Adaptive Hard Example Detection**:

    - **Function**: Automatically identifies samples with uncertain predictions based on model confidence.
    - **Mechanism**: Calculates the variance of the logit vector as a confidence metric—low variance implies multiple labels have close scores (strong label competition). The adaptive threshold is defined as the mean logit variance of misclassified samples on the validation set: $$\sigma^2_{\text{mis}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \text{Var}(\mathbf{z}_i)$$.
    - **Design Motivation**: Fixed thresholds are not suitable for different combinations of models and datasets; variance is used instead of entropy because it directly captures score dispersion in the original decision space and is less sensitive to calibration.

2. **Confusion-Weighted Contrastive Learning (Label Semantic Learning)**:

    - **Function**: Learns a shared text-label embedding space that encodes classifier confusion patterns.
    - **Mechanism**: Constructs label affinity weights $w_{y'} = P(y, y')$ (normalized label confusion probabilities) using the classifier's predictive behavior on the validation set; in the weighted InfoNCE loss $\mathcal{L}_{\text{CW}}$, negative samples from highly confused pairs receive higher weights, forcing the model to work harder to distinguish semantically similar labels.
    - **Design Motivation**: Confusion patterns between labels are domain-specific (e.g., "Analysis" and "Argument" are easily confused in legal texts); weighting by confusion probability is more effective than uniform weighting.

3. **Inference-Time Semantic Reranking**:

    - **Function**: Incorporates label semantic similarity into classifier outputs to correct hard example predictions.
    - **Mechanism**: For a hard example $x$, the cosine similarity vector $\mathbf{s}_x \in \mathbb{R}^C$ between the input embedding $\mathbf{e}_x$ and each label embedding $\mathbf{e}_y$ is calculated, and reranking is performed via element-wise multiplication: $\tilde{\mathbf{z}}_x = \mathbf{s}_x \odot \mathbf{z}_x$.
    - **Design Motivation**: Intervening only on hard examples preserves the classifier's discriminative ability on easy samples, while element-wise multiplication elegantly fuses discriminative and semantic signals.

## Key Experimental Results

### Main Results (Mean macro-F1 / weighted-F1 across 7 LMs × 8 Datasets)

| Model | Baseline mF1 | + RiSE mF1 | Baseline wF1 | + RiSE wF1 |
|------|-------------|-----------|-------------|-----------|
| LLaMA-3-8B | 67.98 | **68.51†** | 74.88 | **75.65†** |
| Mistral-7B | 67.66 | **69.50†** | 74.61 | **75.75†** |
| Qwen3-8B | 67.59 | **68.82†** | 74.28 | **75.23†** |
| ALBERT-base | 66.45 | **66.99** | 72.98 | **73.39** |
| BERT-base | 66.17 | **67.25†** | 73.49 | **73.86** |
| DeBERTa-base | 68.02 | **68.50** | 74.44 | **74.86** |
| RoBERTa-base | 67.21 | **68.11†** | 74.24 | **74.77** |

### Gain on Hard Examples (Average across models and datasets)

| Metric | Gain |
|------|---------|
| Hard Example macro-F1 | **+9.15** |
| Full Set macro-F1 | +0.5 ~ +1.8 |

### Key Findings
- RiSE improves hard example performance by +9.15 macro-F1 on average while maintaining or slightly improving full-set performance.
- It is consistently effective across 7 models (including encoder and causal architectures) and 8 datasets (Legal/Medical/Scientific), demonstrating strong generalization.
- The Cohen's κ between human difficulty annotations and model difficulty is 0.40 (moderate agreement), indicating that model-perceived difficulty and human-perceived difficulty overlap but are not identical.
- Absolute gains for causal LMs (LLaMA-3/Mistral/Qwen3) are slightly larger than for encoder models, possibly because causal LMs are more prone to label confusion.

## Highlights & Insights
- Zero-cost intervention at inference time: A plug-and-play universal framework that does not modify models, require retraining, or change architectures.
- Confusion-weighted contrastive learning cleverly utilizes the classifier's own error patterns to guide label semantic learning—learning from failure.
- Variance as a difficulty metric is simple and effective, outperforming alternatives like entropy.
- The introduction of manual difficulty annotations provides an interpretable perspective for understanding model behavior.

## Limitations & Future Work
- The label semantic embedder requires training a contrastive model on each dataset using training data; although lightweight, it is not entirely zero-overhead.
- Variance thresholds depend on misclassified sample statistics from the validation set, which may be unstable under extreme class imbalance.
- The element-wise multiplication fusion method is relatively simple and might lose more complex signal interactions.
- Validated only on sentence-level classification tasks; applicability to token-level or finer-grained labeling remains to be explored.

## Related Work & Insights
- HiCuLR handles hard examples via curriculum learning during training, while RiSE handles them at inference time—the two are complementary.
- Label semantics are widely used in zero-shot/few-shot text classification; RiSE applies them to inference-time hard example recovery from a new perspective.
- The idea of combining contrastive learning with confusion matrix weighting can be extended to post-processing improvements in other multi-class classification tasks.
- Inference-time intervention paradigms (such as self-consistency, reranking, etc.) are increasingly important in the LLM era.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing confusion-weighted label semantic reranking at inference time is a clear new contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models × 8 datasets × 3 domains + manual annotation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, coherent methodology motivation, and design logic.
- Value: ⭐⭐⭐⭐ A plug-and-play universal framework with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?](filling_the_gap_is_commonsense_knowledge_generation_useful_for_natural_language_.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Accurate and Efficient Statistical Testing for Word Semantic Breadth](accurate_and_efficient_statistical_testing_for_word_semantic_breadth.md)
- [\[ACL 2026\] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](test-time_reasoners_are_strategic_multiple-choice_test-takers.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)

</div>

<!-- RELATED:END -->
