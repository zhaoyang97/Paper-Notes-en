---
title: >-
  [Paper Note] AutoJudge: Judge Decoding Without Manual Annotation
description: >-
  [NeurIPS 2025][Model Compression][Speculative Decoding] AutoJudge automates the annotation of "critical tokens" in Judge Decoding — by using a semi-greedy search to replace mismatched tokens and checking whether the fina…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Speculative Decoding"
  - "Judge Decoding"
  - "Critical Tokens"
  - "Automatic Annotation"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: dc90f7bfc78ad9a8
---

# AutoJudge: Judge Decoding Without Manual Annotation

**Conference**: NeurIPS 2025
**arXiv**: [2504.20039](https://arxiv.org/abs/2504.20039)
**Code**: [https://github.com/garipovroma/autojudge](https://github.com/garipovroma/autojudge)
**Area**: Model Compression
**Keywords**: Speculative Decoding, Judge Decoding, Critical Tokens, Automatic Annotation, Inference Acceleration

## TL;DR
AutoJudge automates the annotation of "critical tokens" in Judge Decoding — by using a semi-greedy search to replace mismatched tokens and checking whether the final answer changes, it labels token importance, trains a logistic regression classifier to predict importance at inference time, enabling speculative decoding to accept 40+ tokens per round (vs. ~20 in standard methods), achieving 1.5× speedup on GSM8K with less than 1% accuracy loss.

## Background & Motivation

**Background**: Speculative decoding uses a small draft model to generate token sequences that are then verified by a large target model. Standard approaches strictly verify every token, accepting only ~20 tokens per round. Judge Decoding relaxes verification by allowing deviations in non-critical tokens, but requires manual annotation of which tokens are "critical."

**Limitations of Prior Work**: (a) Manual annotation of critical tokens demands domain expertise and does not scale; (b) strict verification of all tokens is overly conservative — deviations in many tokens (e.g., formatting, newlines) do not affect the final answer; (c) the speedup of speculative decoding is bottlenecked by low acceptance rates.

**Key Challenge**: Relaxing verification can substantially increase acceptance rates, but requires knowing which tokens can be relaxed — a determination that previously required human annotation.

**Goal**: Automatically annotate token importance and train a classifier for real-time prediction.

**Key Insight**: If replacing a token does not change the final answer, it is "non-critical." A semi-greedy search algorithm automatically identifies such tokens.

**Core Idea**: Semi-greedy search replaces tokens → checks whether the answer changes → automatically labels tokens as critical/non-critical → logistic regression predicts importance → non-critical tokens bypass verification → 40+ tokens per round.

## Method

### Overall Architecture
1. **Offline Annotation**: For training data, draft model generates sequences → mismatched tokens are identified → each is replaced and the answer is checked for changes → tokens are labeled critical/non-critical.
2. **Classifier Training**: Concatenated hidden states from draft and target models serve as features; a logistic regression binary classifier is trained with a threshold tuned to achieve ≥90% recall.
3. **Online Inference**: Draft model generates → classifier predicts importance → target model verifies only critical tokens → non-critical tokens are accepted directly → significantly more tokens accepted per round.

### Key Designs

1. **Semi-Greedy Search Annotation (Algorithm 1)**:

    - Function: Automatically identifies tokens whose replacement does not affect the final answer.
    - Mechanism: Extract answer $\alpha$; generate draft response $\tilde{y}$; for each mismatched position $t$: replace the target token with the draft token → re-run the model → if the new answer is equivalent, mark as non-critical; otherwise mark as critical and retain the target token.
    - Design Motivation: Exhaustive search is prohibitively expensive ($2^n$ combinations); semi-greedy search provides a practical approximation.

2. **Hidden-State Classifier**:

    - Function: Predicts token importance in real time during inference.
    - Mechanism: Input = concatenation of hidden states from both draft and target models at mismatched token positions; logistic regression outputs an importance probability; threshold is tuned to ≥90% recall (preferring over-verification to missing critical tokens).
    - Design Motivation: Logistic regression is extremely lightweight (millisecond-level inference); hidden states encode token-level semantic information.

3. **Stacking with EAGLE-2**:

    - Function: Applies AutoJudge on top of a stronger draft model.
    - Mechanism: EAGLE-2 already provides a higher-quality draft; stacking AutoJudge yields an additional 1.01–1.20× speedup.
    - Design Motivation: The two techniques are orthogonal — EAGLE improves draft quality while AutoJudge relaxes verification.

### Loss & Training
- Logistic regression: standard cross-entropy loss.
- Threshold selection: tuned on a validation set to achieve ≥90% recall.
- Numerical precision: bfloat16 introduces ~10% embedding variance; float32 is more stable.

## Key Experimental Results

### Main Results

| Task | Tokens Accepted / Round | Accuracy | Speedup |
|------|------------------------|----------|---------|
| GSM8K 0-shot (8B/70B) | **40+** | 92% (≤1% loss) | **1.5×** |
| GSM8K 8-shot (8B/70B) | **40+** | 95.4% (<1% loss) | — |
| LiveCodeBench | **22+** | ~2% loss | **2×** |
| vLLM Integration | — | — | 1.5–2× (A100/H100) |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| Draft + Target hidden states vs. Draft only | Concatenation is better |
| Token embeddings vs. previous token | Token embeddings are better |
| Rule-based method (math tokens only) | 65–80% (notably weaker than learned) |
| + EAGLE-2 | Additional 1.01–1.20× speedup |
| GSM8K classifier → LiveCodeBench | Fails (task-specific training required) |

### Key Findings
- Accepted tokens per round double from ~20 to 40+, directly translating to 1.5–2× speedup.
- Accuracy loss is minimal (<1–2%), confirming that "non-critical" tokens can indeed be skipped.
- The approach is task-specific — the GSM8K classifier does not transfer to LiveCodeBench and requires retraining.
- Numerical precision has a significant impact — bfloat16 embedding approximations introduce noise.

## Highlights & Insights
- **Automating the notion of "what is critical"**: The key transition from manual to automatic annotation makes Judge Decoding practical.
- **Semi-greedy search as a practical approximation**: Though not optimal, it is effective and computationally feasible.
- **Hidden states carry sufficient signal**: A simple logistic regression can predict token importance, suggesting that LLMs internally "know" which tokens are critical.

## Limitations & Future Work
- Requires task-specific training data — the definition of "critical tokens" varies across tasks.
- Open-ended tasks (e.g., creative writing) make it difficult to define answer equivalence.
- Semi-greedy search is suboptimal — exhaustive tree search may be more accurate but is computationally prohibitive.
- Sensitivity to numerical precision limits deployment flexibility.

## Related Work & Insights
- **vs. Standard Speculative Decoding**: Standard methods accept ~20 tokens per round; AutoJudge achieves 40+, doubling the acceptance rate.
- **vs. Judge Decoding (manual)**: Manual annotation does not scale; AutoJudge automates the process.
- **vs. EAGLE/Medusa**: These methods improve draft quality and are orthogonal to AutoJudge, which relaxes verification.

## Rating
- Novelty: ⭐⭐⭐⭐ Automating token importance annotation is a practical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers GSM8K, LiveCodeBench, EAGLE stacking, and vLLM integration.
- Writing Quality: ⭐⭐⭐⭐ Algorithm descriptions are clear.
- Value: ⭐⭐⭐⭐ Advances Judge Decoding from theory to practice.

### Supplementary Notes on the Method
- **Complexity of semi-greedy search**: For a sequence of length $L$ with $M$ mismatched tokens, the search complexity is $O(M \times T_{gen})$ (where $T_{gen}$ is the time for a single generation pass). In practice, $M \approx L/3$ (approximately one-third of tokens are mismatched), with each search taking ~1–2 minutes per sample.
- **Classifier feature selection**: Concatenating hidden states from both draft and target models (rather than using either alone) yields better performance — the disagreement between the two models is itself a strong signal of token importance.
- **Relationship to speculative decoding theory**: Standard speculative decoding guarantees exact alignment with the target distribution (lossless); AutoJudge permits <2% accuracy loss in exchange for 2× speedup — an explicit accuracy–speed trade-off.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](traversal_verification_for_speculative_tree_decoding.md)
- [\[NeurIPS 2025\] Elastic ViTs from Pretrained Models without Retraining](elastic_vits_from_pretrained_models_without_retraining.md)
- [\[NeurIPS 2025\] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding](reject_only_critical_tokens_pivot-aware_speculative_decoding.md)
- [\[NeurIPS 2025\] Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency](robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)

</div>

<!-- RELATED:END -->
