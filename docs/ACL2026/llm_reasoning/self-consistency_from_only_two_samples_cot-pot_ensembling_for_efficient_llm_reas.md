---
title: >-
  [Paper Note] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning
description: >-
  [ACL 2026][LLM Reasoning][To be supplemented] To be supplemented after thorough reading.
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "To be supplemented"
date: 2026-05-08
content_hash: 988223a7051cc72d
---

# Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.17433](https://arxiv.org/abs/2604.17433)
**Code**: None
**Area**: LLM Reasoning Efficiency
**Keywords**: Self-Consistency, Chain-of-Thought, Program-of-Thought, Cross-Modal Ensembling, Bayesian Early Stopping

## TL;DR

This paper proposes CoT-PoT, a cross-modal ensembling method that exploits the complementarity between chain-of-thought (CoT) and program-of-thought (PoT) reasoning modalities to reduce the number of samples required for self-consistency by 9.3×, resolving 78.6% of problems with only 2 samples.

## Background & Motivation

**State of the Field**: Self-Consistency (SC) improves LLM reasoning accuracy by sampling multiple reasoning paths and selecting the most frequent answer via majority vote, but requires a large number of samples (typically 40), incurring prohibitive computational costs. Existing adaptive consistency methods reduce the average sample count but remain far from efficient.

**Limitations of Prior Work**: Standard SC relies on high-temperature sampling to increase diversity among reasoning paths. However, empirical observation reveals that multiple samples from the same modality often differ only in surface wording rather than in substantive semantic content, resulting in severe information redundancy across samples.

**Root Cause**: The core assumption of SC is that convergence of diverse reasoning paths to a single answer constitutes a strong signal of correctness, making path diversity—not quantity—the key factor. Yet existing methods achieve diversity only through temperature sampling, which is of limited effectiveness.

**Paper Goals**: To maximize reasoning diversity by combining two fundamentally different reasoning modalities, thereby achieving high accuracy with a minimal number of samples.

**Starting Point**: CoT (natural language step-by-step reasoning) and PoT (program-based computation) represent two essentially distinct reasoning paradigms—CoT is more flexible and expressive but prone to arithmetic errors, while PoT is computationally robust but susceptible to symbolic formulation errors. Critically, the error patterns of the two modalities are highly uncorrelated.

**Core Idea**: When CoT and PoT produce the same answer to a given problem, this cross-modal agreement constitutes an exceptionally strong correctness signal, precisely because their error modes are nearly independent. Based on this insight, a Bayesian early-stopping strategy is designed such that most problems require only one CoT sample plus one PoT sample.

## Method

### Overall Architecture

The framework comprises two families of strategies: (1) **full-budget sampling strategies**—alternating between CoT and PoT samples and aggregating answers via CPMaj, CPMax, or CPAgr; and (2) **early-stopping strategies**—based on a Bayesian model that terminates sampling upon observing cross-modal agreement, with both data-driven and data-free variants.

### Key Designs

1. **Cross-Modal Full-Budget Aggregation**:

    - **Function**: Maximize accuracy under a fixed sampling budget.
    - **Mechanism**: CoT and PoT samples are drawn in alternation (each modality receiving half the budget). Three aggregation strategies are proposed: CPMaj (cross-modal majority vote), CPMax (select the answer from the more confident modality), and CPAgr (prioritize answers appearing in both modalities). All strategies outperform single-modality SC.
    - **Design Motivation**: The complementarity of the two modalities in terms of logical structure and computation yields higher-quality diversity than repeated sampling within a single modality.

2. **Bayesian Cross-Modal Early Stopping**:

    - **Function**: Minimize the number of samples while preserving high accuracy.
    - **Mechanism**: Early stopping is formalized as Bayesian hypothesis testing. CoT and PoT samples are drawn alternately; after PoT produces an answer $y$, subsequent CoT samples are monitored for agreement with $y$. Three core probabilities are defined: $c$ (probability that an answer is safe/correct), $a_1$ (probability that a CoT sample agrees with the anchor answer), and $a_2$ (conditional probability that the answer is safe given agreement). Sampling stops when the posterior $P(C|k,t)$ exceeds a threshold $\rho$. A key empirical finding is that $a_2 \approx 1$—agreement almost certainly implies correctness—providing theoretical justification for the simplest strategy of stopping after a single cross-modal agreement.
    - **Design Motivation**: Cross-modal agreement carries far more information than same-modality agreement, as the error patterns of the two modalities are nearly independent.

3. **Data-Free Minimal Strategy (CPFF)**:

    - **Function**: The most efficient strategy, applicable without any training data.
    - **Mechanism**: Motivated by the extreme parameterization $a_2 \approx 1$, CPFF compares only the first CoT and the first PoT answer, both generated at temperature 0—stopping if they agree (2 samples total) and continuing with alternating sampling otherwise. A parallel adaptive-consistency Beta test serves as a fallback.
    - **Design Motivation**: Since $a_2$ is consistently close to 0.99 across all evaluated models, cross-modal agreement is an extremely reliable correctness signal.

### Loss & Training

No training is involved. Data-driven variants infer Bayesian parameters from 100 questions drawn from each benchmark's training split. Evaluation spans 5 benchmarks (GSM8K, MATH, FinQA, SVAMP, TabMWP) and 5 LLMs. The first sample is generated at temperature 0; subsequent samples use temperature 0.7.

## Key Experimental Results

### Main Results

| Method | Avg. Accuracy | Avg. Samples | 2-Sample Resolution Rate |
|---|---|---|---|
| SCCoT (40 samples) | 84.6% | 40 | 0% |
| SCPoT (40 samples) | 82.9% | 40 | 0% |
| CPMax (full budget) | **85.7%** | 40 | 0% |
| Adaptive SC | ~84% | ~10 | 0% |
| CPFF (early stopping) | ~85% | **4.3** | **78.6%** |

### Ablation Study

| Configuration | Accuracy | Samples | Note |
|---|---|---|---|
| Single-modality SC | 84.6% | 40 | Baseline |
| CPMaj (full budget) | 85.6% | 40 | Cross-modal aggregation |
| CPAA (any agreement) | ~85% | ~4 | Efficient |
| CPFA (first + any) | ~85% | ~4.5 | Slightly conservative |
| CPFF (first + first) | ~85% | ~4.3 | Most efficient |

### Key Findings

- Full-budget cross-modal ensembling outperforms single-modality SC (85.7% vs. 84.6%), achieving higher accuracy under the same budget.
- Early-stopping strategies reduce sampling by 9.3× on average, with 78.6% of problems requiring only 2 samples.
- The finding that $a_2 \approx 1$ is pivotal—cross-modal agreement is an almost certain indicator of correctness.
- Stronger reasoning models such as DeepSeek R1 benefit more from cross-modal consistency, exhibiting higher 2-sample resolution rates.
- On certain benchmarks (e.g., SVAMP), the 2-sample resolution rate exceeds 90%.

## Highlights & Insights

- **The insight that diversity matters more than quantity is profound**: 40 same-modality samples may carry less information than 2 cross-modal samples. This principle generalizes to other settings requiring repeated reasoning.
- **The Bayesian formalization is elegant**: it translates the intuition—cross-modal agreement implies high confidence—into a provable probabilistic model, with the key parameter $a_2 \approx 1$ strongly validated experimentally.
- **Practical significance for reasoning-oriented models is substantial**: as o1/R1-style models become widespread, 2-sample SC can dramatically reduce inference costs.

## Limitations & Future Work

- PoT requires a code execution environment, which may be unavailable in certain deployment contexts.
- The applicability of the PoT modality is limited for non-mathematical or non-computational reasoning tasks (e.g., commonsense reasoning).
- When both modalities commit systematic errors on the same problem, cross-modal agreement can produce spuriously high confidence in an incorrect answer.
- The fallback mechanism of the early-stopping strategy (adaptive consistency) still requires a non-trivial number of additional samples.

## Related Work & Insights

- **vs. Standard Self-Consistency**: Standard SC pursues quantitative diversity within a single modality; CoT-PoT pursues modality-level diversity, which is substantially more efficient.
- **vs. Adaptive Consistency**: Adaptive consistency relies on statistical majority voting for early stopping and requires at least 4 samples. The cross-modal agreement signal in CoT-PoT is stronger, enabling stopping at 2 samples.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The cross-modal consistency insight is concise and profound; the Bayesian early-stopping framework is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 benchmarks × 5 LLMs, with full-budget, early-stopping, and multiple variant evaluations—exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, theoretical derivations are rigorous, and experimental organization is excellent.
**Code**: To be confirmed
**Area**: llm_reasoning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)

</div>

<!-- RELATED:END -->
