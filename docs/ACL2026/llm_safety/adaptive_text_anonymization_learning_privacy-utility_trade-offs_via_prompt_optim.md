---
title: >-
  [Paper Note] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization
description: >-
  [ACL 2026][LLM Safety][text anonymization] This paper proposes an adaptive text anonymization framework that employs evolutionary prompt optimization to automatically discover task-specific anonymization instructions for LLMs, outperforming manually designed strategies across multiple privacy-utility trade-off scenarios while operating entirely on open-source models.
tags:
  - ACL 2026
  - LLM Safety
  - text anonymization
  - privacy protection
  - prompt optimization
  - evolutionary algorithms
  - privacy-utility trade-off
date: 2026-05-08
content_hash: 283d867b794b054b
---

# Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization

**Conference**: ACL 2026
**arXiv**: [2602.20743](https://arxiv.org/abs/2602.20743)
**Code**: [https://github.com/gabrielloiseau/adaptive-text-anonymization](https://github.com/gabrielloiseau/adaptive-text-anonymization)
**Area**: AI Safety
**Keywords**: text anonymization, privacy protection, prompt optimization, evolutionary algorithms, privacy-utility trade-off

## TL;DR

This paper proposes an adaptive text anonymization framework that employs evolutionary prompt optimization to automatically discover task-specific anonymization instructions for LLMs, outperforming manually designed strategies across multiple privacy-utility trade-off scenarios while operating entirely on open-source models.

## Background & Motivation

**Background**: Text anonymization is a foundational technique for enabling the sharing and analysis of sensitive data. Existing approaches fall into two main categories: traditional sequence labeling (detecting and masking PII entities) and LLM-based adversarial-collaborative pipelines (e.g., the AF method, which uses an attacker LLM to guide anonymization decisions).

**Limitations of Prior Work**: Current LLM-based anonymization pipelines suffer from three key limitations: (1) a fixed trade-off paradigm—each scenario requires a manually designed strategy that cannot flexibly adapt to new requirements; (2) reliance on manual prompt engineering, which is subjective, labor-intensive, and often suboptimal; and (3) dependence on closed-source API models (e.g., GPT-4/5), whose use for sensitive data through external APIs is inherently at odds with privacy goals.

**Key Challenge**: Anonymization is fundamentally context-dependent—the appropriate strategy for medical reports differs drastically from that for social media comments, and no one-size-fits-all solution exists. Yet existing methods lack the ability to adaptively adjust strategies.

**Goal**: To design an adaptive framework that (1) automatically discovers anonymization prompts tailored to specific privacy-utility requirements, (2) operates on open-source models, and (3) identifies multiple Pareto-optimal strategies within a single optimization run.

**Key Insight**: The anonymization problem is reframed as a "string discovery" problem—rather than modifying model parameters, the framework searches for optimal natural language instructions to guide model behavior.

**Core Idea**: An evolutionary prompt optimization algorithm (GEPA) is used to automatically search the space of anonymization prompts. Starting from a general seed prompt, the algorithm evolves task-adapted instructions, enabling adaptive privacy-utility trade-offs.

## Method

### Overall Architecture

The framework takes as input a text to be anonymized and a privacy-utility task specification $(p, u)$, and outputs the anonymized text. Within a fixed computational budget, it searches for the optimal anonymization instruction $\Pi^*$ via evolutionary prompt optimization. The process comprises three stages: initialization, warm-up with basic feedback, and refinement with rich feedback.

### Key Designs

1. **Two-Stage GEPA Evolutionary Optimization**:

    - *Function*: Starting from a general seed prompt, discovers task-specific anonymization instructions through evolutionary search.
    - *Mechanism*: A prompt pool $P$ is maintained; at each iteration, high-performing and diverse prompts are selected via Pareto ranking. A proposer agent analyzes execution trajectories and feedback to propose mutations. New candidates are evaluated on a validation set and incorporated into the pool via Pareto pruning. Stage 2 uses a simple scalar aggregation signal $\mu$; when performance stagnates, the process transitions to Stage 3.
    - *Design Motivation*: Evolutionary search naturally supports multi-objective optimization (privacy vs. utility) and can identify multiple Pareto-optimal solutions within a single run, rather than converging to a single fixed trade-off point.

2. **Rich Feedback Generation Mechanism**:

    - *Function*: Upgrades coarse scalar feedback into structured feedback augmented with natural language explanations.
    - *Mechanism*: A dedicated rich feedback agent (an independent LLM) decomposes the aggregated metric $\mu$ into $\mu_{rich}$, supplying the proposer with interpretable, structured improvement signals that enable more substantial and directed behavioral updates.
    - *Design Motivation*: Scalar feedback is too coarse for the proposer to understand what is suboptimal and how to improve. Rich feedback enables more precise prompt optimization within the remaining budget using fewer evaluations.

3. **Adaptive Validation Sampling**:

    - *Function*: Uses sampled subsets to evaluate candidate prompts during the refinement stage, conserving the computational budget.
    - *Mechanism*: A round-robin strategy prioritizes less-evaluated samples when selecting $D'_{valid} \subset D_{valid}$ (sampling ratio $\alpha = 0.3$); the full validation set is used for final selection to ensure fairness.
    - *Design Motivation*: Evaluating on the full validation set at every step is computationally expensive; sampling maintains coverage diversity while improving budget efficiency.

### Loss & Training

No gradient-based training is involved. The optimization objective is the aggregation of privacy and utility scores (e.g., their mean), with multi-objective trade-offs handled via Pareto selection. The evolutionary budget is $B = 1500$ LLM forward passes, with early-stopping patience $n = 5$.

## Key Experimental Results

### Main Results

| Benchmark | Method | Privacy ↑ | Utility ↑ |
|-----------|--------|-----------|-----------|
| DB-Bio | Optimized Qwen3 | 65.5 | 100 |
| DB-Bio | AF (GPT-5) | 78.0 | 92.1 |
| TAB | Optimized Qwen3 | 92.3 | 56.2 |
| TAB | AF (GPT-5) | 59.9 | 42.5 |
| PUPA | Optimized Qwen3 | 98.0 | 79.3 |
| PUPA | AF (GPT-5) | 94.2 | 46.0 |
| MedQA | Optimized Qwen3 | 24.6 | 45.9 |
| MedQA | AF (GPT-5) | 24.4 | 45.8 |

### Ablation Study

| Configuration | Privacy-Utility Performance | Notes |
|---------------|-----------------------------|-------|
| Seed Prompt | Baseline | General seed prompt, no optimization |
| Task-Specific Prompt | Moderate | Manually designed task-specific prompt |
| Optimized Prompt | Best | Automatically optimized prompt |
| OpenPII (entity detection) | High utility, low privacy | Detects PII entities only; insufficient privacy protection |
| DP-Prompt ($\epsilon = 100$) | High privacy, low utility | Differential privacy noise severely degrades utility |

### Key Findings
- The optimized open-source Qwen3-30B is competitive with or superior to the GPT-5 baseline on most tasks, particularly in terms of utility preservation.
- Different models exhibit distinct optimization characteristics: Mistral tends toward aggressive privacy improvement (potentially at the cost of utility), Gemma makes conservative improvements, and Qwen is the most robust.
- A single optimization run can discover multiple Pareto-optimal strategies spanning the full spectrum from privacy-first to utility-first.

## Highlights & Insights
- Reformulating anonymization as a "string search" problem is an elegant abstraction; each Pareto solution is simply a natural language string, incurring minimal storage and deployment overhead.
- Evolutionary optimization naturally supports multi-objective discovery, identifying multiple distinct trade-off points in a single run—far more efficient than traditional approaches that require separately designed strategies for each trade-off point.
- The rich feedback mechanism—decomposing scalar metrics into structured natural language explanations—is transferable to any scenario requiring LLM self-improvement.

## Limitations & Future Work
- Evaluation of privacy and utility metrics still relies on closed-source LLMs (e.g., Gemini-2.5-flash), which contradicts the goal of fully local deployment.
- Each task still requires a small amount of labeled data (111 training + 111 validation samples), precluding a fully zero-shot setup.
- The anonymization capabilities of reasoning-oriented models (e.g., CoT models) remain unexplored and may represent a complementary direction.

## Related Work & Insights
- **vs. AF (Staab et al.)**: AF employs a fixed adversarial-collaborative strategy and depends on GPT-5, whereas the proposed framework automatically searches for strategies via evolutionary optimization and operates on open-source models.
- **vs. DP-Prompt**: Differential privacy methods offer theoretical guarantees but substantially degrade utility; the proposed approach significantly outperforms DP-Prompt on practical privacy-utility trade-offs.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframing anonymization as a prompt optimization problem is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, three open-source models, multiple baselines and ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; method description is systematic.
- Value: ⭐⭐⭐⭐ Directly applicable to sensitive data processing scenarios.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](../../CVPR2026/llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)

</div>

<!-- RELATED:END -->
