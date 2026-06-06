---
title: >-
  [Paper Note] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning
description: >-
  [ACL 2026][Multilingual & Machine Translation][Curriculum Learning] This paper proposes CLewR (Curriculum Learning with Restarts), a strategy that sorts training data from easy to hard and restarts the curriculum at each…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Curriculum Learning"
  - "Preference Optimization"
  - "Machine Translation"
  - "Catastrophic Forgetting"
  - "DPO"
date: 2026-05-08
content_hash: 70b1b61f218f1203
---

# CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning

**Conference**: ACL 2026  
**arXiv**: [2601.05858](https://arxiv.org/abs/2601.05858)  
**Code**: [https://github.com/alexandra-dragomir/CLewR](https://github.com/alexandra-dragomir/CLewR)  
**Area**: Optimization  
**Keywords**: Curriculum Learning, Preference Optimization, Machine Translation, Catastrophic Forgetting, DPO

## TL;DR

This paper proposes CLewR (Curriculum Learning with Restarts), a strategy that sorts training data from easy to hard and restarts the curriculum at each epoch. This approach effectively mitigates catastrophic forgetting and consistently improves machine translation performance across multiple model families (Gemma2, Qwen2.5, Llama3.1) and various preference optimization algorithms (DPO, CPO, ARPO).

## Background & Motivation

**Background**: Large Language Models (LLMs) perform exceptionally well in zero-shot multilingual machine translation. Subsequent work utilizes preference optimization (e.g., DPO, CPO, ARPO) to further enhance translation quality by performing contrastive learning between high-quality and low-quality translations.

**Limitations of Prior Work**: Preference optimization methods often ignore the presentation order of data samples during training—a factor that significantly impacts training effectiveness. Existing curriculum learning works (e.g., CurriDPO) simply sort data by difficulty but fail to address catastrophic forgetting: simple samples learned early in training are forgotten in later stages.

**Key Challenge**: Traditional curriculum learning traverses data from easy to hard only once. When the model concentrates on difficult samples in the late stages of training, it forgets the knowledge of simple samples learned previously. However, without sequential learning, the benefits of curriculum learning cannot be realized.

**Goal**: To propose a data-level curriculum strategy that simultaneously captures the benefits of curriculum learning and mitigates catastrophic forgetting, specifically for preference optimization in machine translation.

**Key Insight**: Sort data from easy to hard within each epoch, but restart the sorting at the end of each epoch—meaning each epoch completely traverses all samples from easy to hard.

**Core Idea**: By restarting the easy-to-hard curriculum sorting each epoch, CLewR inherently addresses catastrophic forgetting because every epoch begins by revisiting all samples from the start.

## Method

### Overall Architecture

CLewR consists of two stages: (1) Sorting phase—all training triplets are sorted based on the similarity difference between the chosen and rejected translations; (2) Training phase—all data is traversed in a fixed easy-to-hard order within each epoch without random shuffling. Sorting is performed once before training begins, and the same order is repeated in each epoch.

### Key Designs

1.  **Difficulty Scoring based on Multiple Metrics**:
    - **Function**: Calculates a difficulty score for each preference triplet $(x, y_w, y_l)$.
    - **Mechanism**: Computes BLEU, COMET-22, and METEOR metrics between the chosen translation $y_w$ and rejected translation $y_l$. The normalized average of these metrics serves as the similarity score $s$. High-similarity samples (small difference between chosen and rejected) are considered hard, while low-similarity samples are easy.
    - **Design Motivation**: A single metric may be biased; synthesizing three complementary translation evaluation metrics provides a more robust difficulty estimation.

2.  **Epoch-level Restart Mechanism**:
    - **Function**: Restarts the easy-to-hard curriculum sorting at the beginning of each epoch.
    - **Mechanism**: Data sorting is completed once before training. During training, each epoch traverses all data in the same easy-to-hard order (no shuffling), ensuring the model reviews simple samples at the start of every round.
    - **Design Motivation**: Traditional curriculum learning often leads to catastrophic forgetting due to single-pass traversal. Restarting the curriculum every epoch allows the model to periodically consolidate knowledge of simple samples.

3.  **CLewR-z Variant (Distance-based Sorting via ARPO)**:
    - **Function**: Uses ARPO's adaptive distance function $z_\theta$ instead of external metrics for sorting.
    - **Mechanism**: The $z_\theta(y_w, y_l)$ in ARPO encodes the log-likelihood difference between chosen and rejected responses. $s = -z_\theta$ is used as the curriculum score. An enhanced ARPO is also proposed, combining $z_\theta$ with BLEU and COMET distances via weighted summation.
    - **Design Motivation**: Utilizing the model's internal distance signals (rather than external metrics) for sorting aligns the curriculum more closely with the optimization objective.

### Loss & Training

CLewR is compatible with three preference optimization algorithms: DPOP (an enhanced DPO), CPO (Contrastive Preference Optimization with behavior cloning), and ARPO (Adaptive Rejection Preference Optimization). During training, the native loss functions of each algorithm are used; CLewR only modifies the data presentation order.

## Key Experimental Results

### Main Results

**BLEU Scores for Gemma2-9B on 6 Romance Languages (en→xx)**

| Method | DPOP | +CurriDPO | +CLewR | CPO | +CLewR | ARPO | +CLewR |
|--------|------|-----------|--------|-----|--------|------|--------|
| BLEU | 23.26 | 21.81 | 22.35 | 33.53 | **36.24** | 35.37 | **36.63** |

**BLEU Scores for Qwen2.5-7B on 6 Romance Languages (en→xx)**

| Method | DPOP | +CLewR | CPO | +CLewR | ARPO | +CLewR |
|--------|------|--------|-----|--------|------|--------|
| BLEU | 24.43 | 23.59 | 27.68 | **30.05** | 30.41 | **31.56** |

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| CLewR (Multi-metric) | Integrated sorting via BLEU+COMET+METEOR | Optimal |
| CLewR-z (Model-distance) | Sorting using ARPO internal distance | Near-optimal |
| ARPO-z'-V1/V2 (Enhanced dist.) | Enhanced distance functions | Further improves baseline ARPO |
| CurriDPO | Competing method | Less effective than CLewR |

### Key Findings

- CLewR consistently improves performance for CPO and ARPO, while the effect on DPOP varies by model.
- On Gemma2, the best configuration (CLewR + ARPO-z'-V2) achieved a BLEU of 37.45 (en→xx), a Gain of 2.08 over baseline ARPO.
- CLewR outperforms CurriDPO across all model families and most preference optimization algorithms.
- The enhanced version of ARPO (combining distance functions with external metrics) further boosts baseline performance.

## Highlights & Insights

- The method is remarkably simple—by only changing the data presentation order without modifying model architecture or loss functions, consistent performance gains are achieved.
- The "Restart" mechanism is an elegant solution to the contradiction between curriculum learning and catastrophic forgetting.
- CLewR exhibits strong universality and can be seamlessly integrated into various preference optimization algorithms like DPO, CPO, and ARPO.

## Limitations & Future Work

- Performance on DPOP is less stable than on CPO/ARPO, potentially due to DPOP's dependence on reference models.
- Difficulty sorting is based on static pre-training evaluation and does not dynamically adjust the curriculum during training.
- Validation is limited to machine translation; effectiveness in other preference optimization scenarios (e.g., dialogue, summarization) remains to be verified.

## Related Work & Insights

- **vs CurriDPO**: CurriDPO uses iterative curricula but fails to solve catastrophic forgetting; CLewR is more effective via epoch-level restarts.
- **vs X-ALMA**: X-ALMA is a strong baseline based on ARPO; CLewR further enhances performance on top of ARPO.

## Rating

- Novelty: ⭐⭐⭐⭐ The restart mechanism is simple but effectively addresses the conflict between curriculum learning and catastrophic forgetting.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across three model families, three preference optimization algorithms, and multilingual evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, concise methodology, and detailed experiments.
- Value: ⭐⭐⭐⭐ Provides a simple and effective curriculum learning strategy for preference optimization training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)
- [\[ACL 2026\] Selective Contrastive Learning For Gloss Free Sign Language Translation](selective_contrastive_learning_for_gloss_free_sign_language_translation.md)
- [\[ACL 2026\] SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](serm_self-evolving_relevance_model_with_agent-driven_learning_from_massive_query.md)
- [\[ACL 2026\] From Fragments to Facts: A Curriculum-Driven DPO Approach for Generating Hindi News Veracity Explanations](from_fragments_to_facts_a_curriculum-driven_dpo_approach_for_generating_hindi_ne.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)

</div>

<!-- RELATED:END -->
