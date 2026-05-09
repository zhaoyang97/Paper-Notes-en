---
title: >-
  [Paper Note] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines
description: >-
  [ACL 2026][Tsetlin Machine] This paper proposes an LLM-guided semantic bootstrapping framework that leverages LLMs to generate sub-intents and trains a Non-Negated Tsetlin Machine (NTM) via three-stage curriculum synthetic data generation. High-confidence symbolic features extracted by the NTM are injected into real data representations, enabling a standard TM to approach BERT-level classification performance while maintaining full interpretability.
tags:
  - ACL 2026
  - Tsetlin Machine
  - semantic guidance
  - symbolic learning
  - sub-intent discovery
  - interpretable classification
date: 2026-05-08
content_hash: b383e7f7a13b0547
---

# LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines

**Conference**: ACL 2026
**arXiv**: [2604.12223](https://arxiv.org/abs/2604.12223)
**Code**: None
**Area**: Interpretability / Text Classification
**Keywords**: Tsetlin Machine, semantic guidance, symbolic learning, sub-intent discovery, interpretable classification

## TL;DR

This paper proposes an LLM-guided semantic bootstrapping framework that leverages LLMs to generate sub-intents and trains a Non-Negated Tsetlin Machine (NTM) via three-stage curriculum synthetic data generation. High-confidence symbolic features extracted by the NTM are injected into real data representations, enabling a standard TM to approach BERT-level classification performance while maintaining full interpretability.

## Background & Motivation

**Background**: The Tsetlin Machine (TM) has attracted attention in interpretable NLP due to its clause-level transparency, and has been applied to document classification, sentiment analysis, and related tasks. Pre-trained language models such as BERT provide powerful semantic representations but at high computational cost and with limited transparency.

**Limitations of Prior Work**: (1) TMs rely on Boolean bag-of-words (BoW) representations and cannot generalize across semantically similar but lexically distinct expressions unless they appear explicitly in training data; (2) augmenting TM inputs with Word2Vec or GloVe provides only limited semantic alignment; (3) BERT achieves strong performance but lacks decision traceability in high-stakes domains such as law and medicine.

**Key Challenge**: There is a fundamental tension between symbolic interpretability and semantic generalization — BoW representations guarantee transparency but sacrifice semantic understanding, whereas embedding-based representations capture semantics but lose interpretability.

**Goal**: To transfer LLM semantic knowledge into TMs in symbolic form, without introducing embedding layers or runtime LLM calls.

**Key Insight**: LLMs are used to generate interpretable sub-intents (e.g., *positive_due_to_plot*) and corresponding synthetic data, bridging the semantic gap through symbolic augmentation rather than embedding augmentation.

**Core Idea**: The LLM does not participate in classification inference; instead, it acts as a "semantic teacher" during offline training, providing symbolic semantic priors to the TM via sub-intent decomposition and curriculum-based data generation.

## Method

### Overall Architecture

The framework consists of three stages: (1) LLM-guided sub-intent discovery and three-stage synthetic data generation (Seed → Core → Enriched); (2) pre-training a Non-Negated TM (NTM) on synthetic data to extract high-confidence symbolic features; (3) injecting the semantic features extracted by the NTM into the BoW representations of real data, followed by fine-tuning a standard TM on the augmented representations. At inference time, the pipeline is entirely symbolic — no LLM or embeddings are required.

### Key Designs

1. **LLM-Guided Sub-Intent Discovery and Three-Stage Data Generation**

    - **Function**: Decompose class labels into interpretable semantic factors and generate diverse training data.
    - **Mechanism**: The LLM decomposes each class into fine-grained sub-intents (e.g., *positive* → *positive_due_to_plot*, *positive_due_to_acting*). Synthetic data are then generated through a three-stage curriculum: the Seed stage produces canonical 15–20-word expressions as anchors; the Core stage preserves lexical stability while varying syntactic structure; the Enriched stage introduces synonyms and compound phrases to expand the lexical space.
    - **Design Motivation**: Single-step LLM generation tends to collapse into high-probability patterns or overly generic phrases. The three-stage strategy follows curriculum learning principles, ensuring coverage, lexical diversity, and semantic fidelity — each of which is critical for clause formation in Boolean symbolic models.

2. **Non-Negated Tsetlin Machine (NTM)**

    - **Function**: Extract stable, high-confidence semantic symbolic features from synthetic data.
    - **Mechanism**: The NTM modifies the standard TM in two respects: (1) negated literals are eliminated, reducing clauses to purely monotone conjunctions $C_\iota^\kappa = \bigwedge_{k \in I_\iota^\kappa} x_k$; (2) Type I feedback is strengthened ($P_{\text{reward}}=1.0$, $P_{\text{penalty}}=0.0$), enabling Tsetlin Automata (TA) to converge rapidly to high-confidence literal sets. Literals corresponding to the deepest TA states are extracted as semantic indicators.
    - **Design Motivation**: Removing negated literals ensures monotone semantic interpretability of clauses — all learned rules reflect positively associated lexical patterns. Strengthened feedback ensures rapid and stable convergence on synthetic data.

3. **Semantic Feature Injection and TM Fine-Tuning**

    - **Function**: Inject LLM-derived symbolic semantic knowledge into real data.
    - **Mechanism**: Real samples are passed through the NTM to predict sub-intents; high-confidence literals from activated clauses are collected, and binary presence indicators for these literals are appended to the original BoW representation. A standard TM is then fine-tuned on this hybrid representation.
    - **Design Motivation**: The augmentation occurs offline and introduces no new components at inference time — the final model remains purely symbolic and efficient. The semantic features provide cross-lexical associations absent from the original BoW.

### Loss & Training

The NTM is trained using modified Type I/II feedback (150 clauses per sub-intent, $T=5000$, $s=5$). The standard TM is fine-tuned on augmented data using an integer-weighted variant. All synthetic data are generated by GPT-4o (nucleus sampling, $p=0.9$, temperature $=0.7$).

## Key Experimental Results

### Main Results

**Performance comparison across six classification benchmarks**

| Method | AG-News | R8 | R52 | IMDB | SST2 | HoC |
|---|---|---|---|---|---|---|
| TM | 88.34 | 96.16 | 84.62 | 90.62 | 75.61 | 77.42 |
| TM (GloVe) | 90.12 | 97.50 | 89.14 | 90.88 | 76.38 | 78.78 |
| BERT | 94.75 | 97.49 | 94.26 | 93.46 | 94.00 | 82.90 |
| **LLM-Guided TM** | **93.10** | **97.88** | **94.45** | **92.10** | **85.24** | **81.90** |

### Ablation Study

**Performance gains of TM variants across datasets**

| Dataset | TM → LLM-TM Gain | Gap vs. BERT |
|---|---|---|
| AG-News | +4.76% | −1.65% |
| R8 | +1.72% | +0.39% |
| R52 | +9.83% | +0.19% |
| SST2 | +9.63% | −8.76% |
| HoC | +4.48% | −1.00% |

### Key Findings

- LLM-Guided TM surpasses BERT on R8 and R52 while maintaining full symbolic interpretability.
- SST2 shows the largest absolute gain (+9.63%) yet also the largest remaining gap versus BERT (−8.76%), indicating that short-text sentiment analysis still requires contextual understanding.
- On the biomedical HoC dataset, the proposed method approaches BERT (81.90% vs. 82.90%); semantic decomposition effectively recovers compound word semantics (e.g., *immunosuppression* → *immune* + *suppression*).
- Symbolic feature groups are semantically coherent — e.g., the *politics* sub-intent extracts {*parliament*, *election*, *results*}.
- The entire inference pipeline remains purely symbolic — no embeddings and no runtime LLM calls are required.

## Highlights & Insights

- The paradigm of "LLM as semantic teacher rather than classifier" is elegant — it leverages LLM world knowledge while entirely avoiding runtime overhead.
- Sub-intent decomposition makes the augmented features inherently interpretable, unlike embedding-based augmentation which introduces black-box components.
- The three-stage curriculum generation strategy is particularly important for clause learning in Boolean symbolic models — balancing lexical stability and diversity is the critical design consideration.

## Limitations & Future Work

- The framework depends on LLM generation quality — sub-intents may be inaccurate in complex domains or when class boundaries overlap.
- Removing negated literals improves interpretability but reduces expressive power, precluding the capture of negation logic.
- No systematic hyperparameter ablation is conducted (number of clauses, synthetic sample size, weighting schemes, etc.).
- The remaining gap versus BERT on SST2 suggests that contextual understanding of short texts remains a bottleneck.

## Related Work & Insights

- **vs. TM (GloVe)**: GloVe augmentation provides static word vector alignment, whereas the proposed sub-intent guidance provides structured semantic associations, yielding a +5.31% improvement on R52.
- **vs. BERT**: BERT retains an advantage on most tasks (except R8/R52) at the cost of interpretability. The proposed method closes most of the performance gap while preserving symbolic transparency.
- **vs. symbolic distillation methods**: Existing approaches typically distill models into decision trees or linear rules; this work is the first to distill into clause logic.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of symbolically transferring LLM semantic knowledge into Tsetlin Machines is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Six datasets spanning multiple domains; ablation experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — The framework is described clearly and case analyses are convincing.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for high-stakes scenarios requiring interpretability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](../../CVPR2026/interpretability/text-guided_fine-grained_video_anomaly_understanding.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](../../CVPR2026/interpretability/dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](../../NeurIPS2025/interpretability/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)
- [\[ACL 2026\] LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)

<!-- RELATED:END -->
