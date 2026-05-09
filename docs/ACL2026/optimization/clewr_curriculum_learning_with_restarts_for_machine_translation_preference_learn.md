---
title: >-
  [Paper Note] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning
description: >-
  [ACL 2026][Optimization][Curriculum Learning] The paper proposes CLewR (Curriculum Learning with Restarts), a strategy that sorts data from easy to hard and restarts the curriculum at each epoch during preference optimization training, effectively mitigating catastrophic forgetting and consistently improving machine translation performance across multiple model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization algorithms (DPO, CPO, ARPO).
tags:
  - ACL 2026
  - Optimization
  - Curriculum Learning
  - Preference Optimization
  - Machine Translation
  - Catastrophic Forgetting
  - DPO
date: 2025-05-08
content_hash: 0efaa1b46be7004d
---

# CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning

**Conference**: ACL 2026  
**arXiv**: [2601.05858](https://arxiv.org/abs/2601.05858)  
**Code**: [https://github.com/alexandra-dragomir/CLewR](https://github.com/alexandra-dragomir/CLewR)  
**Area**: Optimization & Theory  
**Keywords**: Curriculum Learning, Preference Optimization, Machine Translation, Catastrophic Forgetting, DPO

## TL;DR

The paper proposes CLewR (Curriculum Learning with Restarts), a strategy that sorts data from easy to hard and restarts the curriculum at each epoch during preference optimization training, effectively mitigating catastrophic forgetting and consistently improving machine translation performance across multiple model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization algorithms (DPO, CPO, ARPO).

## Background & Motivation

**Background**: Large language models demonstrate strong performance in zero-shot multilingual machine translation. Subsequent work further improves translation quality through preference optimization (e.g., DPO, CPO, ARPO), contrasting high-quality translations with low-quality ones.

**Limitations of Prior Work**: Preference optimization methods ignore the presentation order of data samples during training—a factor that significantly impacts training effectiveness. Existing curriculum learning work (e.g., CurriDPO) simply sorts data by difficulty but does not address catastrophic forgetting during training: easy samples learned early are forgotten in later stages.

**Key Challenge**: Traditional curriculum learning arranges data from easy to hard and traverses it only once. The model forgets knowledge from easy samples when it concentrates on hard samples later in training, but not following a curriculum means losing curriculum learning benefits.

**Goal**: Propose a data-level curriculum strategy that simultaneously enjoys curriculum learning benefits and mitigates catastrophic forgetting, applicable to preference optimization training for machine translation.

**Key Insight**: Sort data from easy to hard within each epoch, but restart the sorting after each epoch—meaning every epoch traverses all samples completely, from easy to hard.

**Core Idea**: By restarting the easy-to-hard curriculum ordering at each epoch, CLewR natively solves catastrophic forgetting because every epoch traverses all samples from the beginning.

## Method

### Overall Architecture

CLewR consists of two phases: (1) Sorting phase—ranks all training triplets based on the similarity difference between chosen and rejected translations; (2) Training phase—traverses all data in the fixed easy-to-hard order within each epoch without random shuffling. Sorting is completed once before training begins, and each epoch repeats the same ordering.

### Key Designs

1. **Multi-Metric Difficulty Scoring**:

    - Function: Computes a difficulty score for each preference triplet $(x, y_w, y_l)$
    - Mechanism: Computes BLEU, COMET-22, and METEOR between chosen translation $y_w$ and rejected translation $y_l$, normalizes them, and averages to obtain a similarity score $s$. High-similarity samples (small difference between chosen and rejected) are hard samples; low-similarity ones are easy
    - Design Motivation: A single metric may be biased; combining three complementary translation evaluation metrics provides a more robust difficulty estimate

2. **Epoch-Level Restart Mechanism**:

    - Function: Restarts the easy-to-hard curriculum ordering at each epoch
    - Mechanism: Data sorting is completed once before training. During training, each epoch traverses all data in the same easy-to-hard order (no shuffling), ensuring the model can review from easy samples at the start of every epoch
    - Design Motivation: Traditional curriculum learning traversing data only once easily leads to catastrophic forgetting. By restarting the curriculum each epoch, the model reinforces easy sample knowledge every round

3. **CLewR-z Variant (ARPO Distance-Based Sorting)**:

    - Function: Uses ARPO's adaptive distance function $z_\theta$ instead of external metrics for sorting
    - Mechanism: ARPO's $z_\theta(y_w, y_l)$ encodes the log-likelihood difference between chosen and rejected responses, using $s = -z_\theta$ as the curriculum score. An enhanced ARPO version is also proposed that combines $z_\theta$ with weighted BLEU and COMET distances
    - Design Motivation: Using the model's internal distance signal (rather than external metrics) for sorting aligns the curriculum ordering more closely with the optimization objective

### Loss & Training

CLewR is compatible with three preference optimization algorithms: DPOP (enhanced DPO), CPO (contrastive preference optimization with behavior cloning), and ARPO (adaptive rejection preference optimization). Training uses each algorithm's native loss function; CLewR only changes the data presentation order.

## Key Experimental Results

### Main Results

**Gemma2-9B BLEU scores on 6 Romance languages (en→xx direction)**

| Method | DPOP | +CurriDPO | +CLewR | CPO | +CLewR | ARPO | +CLewR |
|--------|------|-----------|--------|-----|--------|------|--------|
| BLEU | 23.26 | 21.81 | 22.35 | 33.53 | **36.24** | 35.37 | **36.63** |

**Qwen2.5-7B BLEU scores on 6 Romance languages (en→xx direction)**

| Method | DPOP | +CLewR | CPO | +CLewR | ARPO | +CLewR |
|--------|------|--------|-----|--------|------|--------|
| BLEU | 24.43 | 23.59 | 27.68 | **30.05** | 30.41 | **31.56** |

### Ablation Study

| Config | Note | Effect |
|--------|------|--------|
| CLewR (multi-metric sorting) | BLEU+COMET+METEOR combined sorting | Best |
| CLewR-z (model distance sorting) | Using ARPO internal distance sorting | Near-best |
| ARPO-z'-V1/V2 (enhanced distance) | Enhanced distance function | Further improves baseline ARPO |
| CurriDPO | Competing method | Worse than CLewR |

### Key Findings

- CLewR consistently improves performance on CPO and ARPO; effects on DPOP vary by model
- On Gemma2, the best CLewR + ARPO-z'-V2 configuration reaches BLEU 37.45 (en→xx), a +2.08 improvement over baseline ARPO
- CLewR outperforms CurriDPO across all model families and most preference optimization algorithms
- Enhanced ARPO (combining external metrics with distance function) further improves baseline ARPO performance

## Highlights & Insights

- The method is extremely simple—only changing data presentation order without modifying model architecture or loss function yields consistent performance improvements
- The "restart" mechanism is an elegant solution to the contradiction between curriculum learning and catastrophic forgetting
- CLewR has strong universality and can be seamlessly integrated into DPO, CPO, ARPO, and other preference optimization algorithms

## Limitations & Future Work

- Effects on DPOP are less stable than on CPO/ARPO, possibly related to DPOP's reference model dependency
- Difficulty sorting is based on static pre-training evaluation, unable to dynamically adjust the curriculum during training
- Only validated on machine translation tasks; effectiveness on other preference optimization scenarios (e.g., dialogue, summarization) remains to be verified

## Related Work & Insights

- **vs CurriDPO**: CurriDPO uses iterative curriculum but does not address catastrophic forgetting; CLewR is more effective through epoch-level restarts
- **vs X-ALMA**: X-ALMA is a strong ARPO-based baseline; CLewR further improves upon ARPO

## Rating

- Novelty: ⭐⭐⭐⭐ The restart mechanism, though simple, effectively resolves the curriculum learning + catastrophic forgetting contradiction
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three model families, three preference optimization algorithms, multilingual evaluation, comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, concise method, detailed experiments
- Recommendation: ⭐⭐⭐⭐ Provides a simple and effective curriculum learning strategy for preference optimization training

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SCRAPL: Scattering Transform with Random Paths for Machine Learning](../../ICLR2026/optimization/scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](../../NeurIPS2025/optimization/preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](../../ICLR2026/optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](../../ICLR2026/optimization/when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)

<!-- RELATED:END -->
