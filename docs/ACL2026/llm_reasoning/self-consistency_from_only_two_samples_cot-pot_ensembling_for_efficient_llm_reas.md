---
title: >-
  [Paper Note] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Self-Consistency] Ours proposes the CoT-PoT cross-modal ensemble method, leveraging the complementarity between Chain-of-Thought (CoT) and Program-of-Thought (PoT) reasoning modalities to reduce…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Self-Consistency"
  - "Chain-of-Thought"
  - "Program-of-Thought"
  - "Cross-Modal Ensemble"
  - "Bayesian Early Stopping"
date: 2026-05-08
content_hash: 6fabf4e93acfe7ff
---

# Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.17433](https://arxiv.org/abs/2604.17433)  
**Code**: None  
**Area**: LLM Reasoning Efficiency  
**Keywords**: Self-Consistency, Chain-of-Thought, Program-of-Thought, Cross-Modal Ensemble, Bayesian Early Stopping

## TL;DR

Ours proposes the CoT-PoT cross-modal ensemble method, leveraging the complementarity between Chain-of-Thought (CoT) and Program-of-Thought (PoT) reasoning modalities to reduce the number of samples required for self-consistency by 9.3x, solving 78.6% of problems with only 2 samples.

## Background & Motivation

**Background**: Self-Consistency (SC) improves LLM reasoning accuracy by sampling multiple reasoning paths and voting for the most frequent answer, but it typically requires a large number of samples (often 40), leading to extremely high computational costs. Existing adaptive consistency methods reduce the average number of samples but remain insufficiently efficient.

**Limitations of Prior Work**: Standard SC utilizes high-temperature sampling to increase diversity in reasoning paths, but empirical observations show that multiple samples within the same modality often exhibit superficial phrasing differences rather than substantive semantic diversity. This results in significant information redundancy within large sample sets.

**Key Challenge**: The core assumption of SC is that "different reasoning paths converging to the same answer is a strong signal of correctness." The priority is the diversity of reasoning paths rather than their quantity. However, existing methods only increase diversity through temperature sampling, which has limited effectiveness.

**Goal**: Maximize reasoning diversity by combining two fundamentally different reasoning modalities to achieve high accuracy with a minimal number of samples.

**Key Insight**: CoT (natural language step-by-step reasoning) and PoT (writing programs for calculation) are inherently different. CoT is flexible but prone to calculation errors, while PoT is computationally robust but prone to symbolic representation errors. Their error patterns are highly uncorrelated.

**Core Idea**: If CoT and PoT yield the same answer for the same problem, this cross-modal consistency serves as an extremely strong signal of correctness due to their nearly uncorrelated error patterns. Based on this, a Bayesian early stopping strategy is designed, where most problems require only 1 CoT + 1 PoT sample.

## Method

### Overall Architecture

The framework consists of two categories of strategies: (1) Full sampling strategies—alternating between CoT and PoT sampling and using various aggregation methods (CPMaj/CPMax/CPAgr) to vote for answers; (2) Early stopping strategies—terminating sampling once cross-modal consistency is observed based on a Bayesian model, including both data-driven and data-independent variants.

### Key Designs

1.  **Cross-modal full sampling aggregation**:
    - **Function**: Maximize accuracy under a fixed sampling budget.
    - **Mechanism**: Alternates between sampling CoT and PoT (each taking half the budget) and proposes three strategies: CPMaj (Cross-Modal Majority vote), CPMax (selecting the answer from the most confident modality), and CPAgr (prioritizing answers appearing in both modalities). All strategies surpass single-modal SC.
    - **Design Motivation**: Leverages the complementarity between the two modalities in logical framework and computation, providing higher quality diversity than repeated sampling within a single modality.

2.  **Bayesian cross-modal consistency early stopping**:
    - **Function**: Minimize the number of samples while maintaining high accuracy.
    - **Mechanism**: Formalizes early stopping as a Bayesian hypothesis test. By alternating CoT and PoT sampling, once a PoT produces an answer $y$, it tracks how many subsequent CoT samples yield the same $y$. Three core probabilities are defined: $c$ (probability of answer safety), $a_1$ (probability of CoT agreeing with the anchor answer), and $a_2$ (conditional probability of answer safety given agreement). Sampling stops when the posterior $P(C|k,t)$ exceeds a threshold $\rho$. A key empirical finding is $a_2 \approx 1$ (agreement almost certainly implies safety), providing theoretical support for the simplest strategy (stopping at the first agreement).
    - **Design Motivation**: The information content of cross-modal consistency is much higher than intra-modal consistency because error patterns are nearly uncorrelated.

3.  **Data-independent minimalist strategy (CPFF)**:
    - **Function**: The most efficient strategy applicable without any training data.
    - **Mechanism**: Based on the extreme parameterization where $a_2 \approx 1$, CPFF compares the first CoT and first PoT answers at temperature 0—if they agree, it stops (2 samples total). If they disagree, it continues alternating samples. A Beta test for adaptive consistency is run in parallel as a fallback.
    - **Design Motivation**: $a_2$ is close to 0.99 across all models, indicating that cross-modal consistency is an extremely reliable signal of correctness.

### Loss & Training

A training-free method. Data-driven variants infer Bayesian parameters from 100 problems in each benchmark's training set. Evaluated across 5 benchmarks (GSM8K, MATH, FinQA, SVAMP, TabMWP) and 5 LLMs. The first sample uses temperature 0, and subsequent samples use temperature 0.7.

## Key Experimental Results

### Main Results

| Method | Mean Accuracy | Mean Samples | 2-Sample Solve Rate |
| :--- | :--- | :--- | :--- |
| SCCoT (40 samples) | 84.6% | 40 | 0% |
| SCPoT (40 samples) | 82.9% | 40 | 0% |
| CPMax (Full) | **85.7%** | 40 | 0% |
| Adaptive SC | ~84% | ~10 | 0% |
| CPFF (Early Stop) | ~85% | **4.3** | **78.6%** |

### Ablation Study

| Configuration | Accuracy | Samples | Description |
| :--- | :--- | :--- | :--- |
| Single-modal SC | 84.6% | 40 | Baseline |
| CPMaj (Full) | 85.6% | 40 | Cross-modal aggregation |
| CPAA (Any-Agree) | ~85% | ~4 | Highly efficient |
| CPFA (First+Any) | ~85% | ~4.5 | Slightly conservative |
| CPFF (First+First) | ~85% | ~4.3 | Most efficient |

### Key Findings

- Cross-modal full sampling consistency outperforms single-modal methods (85.7% vs 84.6%), providing higher accuracy under the same budget.
- Early stopping strategies reduce samples by 9.3x on average, with 78.6% of problems requiring only 2 samples.
- The finding $a_2 \approx 1$ is pivotal—cross-modal consistency almost certainly indicates a correct answer.
- Stronger reasoning models like DeepSeek R1 benefit more from cross-modal consistency (higher 2-sample solve rate).
- On certain benchmarks (e.g., SVAMP), the 2-sample solve rate exceeds 90%.

## Highlights & Insights

- **Insight that "diversity is more important than quantity"**: The information content of 40 intra-modal samples may be less than that of 2 cross-modal samples. This perspective is generalizable to other scenarios requiring multiple reasoning steps.
- **Elegant Bayesian formalization**: Translates the intuition (cross-modal agreement = high confidence) into a provable probabilistic model, with the key parameter $a_2 \approx 1$ strongly validated by experiments.
- **Significant practical value for reasoning models**: As o1/R1-style models become more common, 2-sample SC can significantly reduce inference costs.

## Limitations & Future Work

- PoT depends on a code execution environment, which may not be available in all deployment scenarios.
- For non-mathematical/non-computational reasoning tasks (e.g., commonsense reasoning), PoT is less applicable.
- When both modalities make systematic errors, cross-modal consistency may yield false high confidence.
- The fallback mechanism (adaptive consistency) for the early stopping strategy still requires some additional samples.

## Related Work & Insights

- **vs Standard Self-Consistency**: Standard SC pursues quantitative diversity within the same modality; CoT-PoT pursues modal diversity, which is more efficient.
- **vs Adaptive Consistency**: Adaptive consistency uses statistical majorities for early stopping and still typically requires at least 4 samples. The cross-modal consistency signal in CoT-PoT is stronger, allowing for termination at 2 samples.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The insight into cross-modal consistency is simple and profound; the Bayesian early stopping framework is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, covering 5 benchmarks × 5 LLMs with various full and early stopping variants.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous theoretical derivation, and excellent experimental organization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)
- [\[ACL 2026\] Revisiting the Uniform Information Density Hypothesis in LLM Reasoning](revisiting_the_uniform_information_density_hypothesis_in_llm_reasoning.md)
- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ICML 2026\] Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain](../../ICML2026/llm_reasoning/self-play_only_evolves_when_self-synthetic_pipeline_ensures_learnable_informatio.md)

</div>

<!-- RELATED:END -->
