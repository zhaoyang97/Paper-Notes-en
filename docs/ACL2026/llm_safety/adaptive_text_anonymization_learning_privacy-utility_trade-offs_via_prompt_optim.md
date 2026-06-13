---
title: >-
  [Paper Note] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization
description: >-
  [ACL 2026][LLM Safety][Text anonymization] This paper proposes an adaptive text anonymization framework that automatically discovers task-specific anonymization instructions for LLMs through evolutionary prompt optimizat…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Text anonymization"
  - "Privacy protection"
  - "Prompt optimization"
  - "Evolutionary algorithms"
  - "Privacy-utility trade-off"
date: 2026-05-08
content_hash: dbb5f536077601e0
---

# Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization

**Conference**: ACL 2026 Findings  
**arXiv**: [2602.20743](https://arxiv.org/abs/2602.20743)  
**Code**: [https://github.com/gabrielloiseau/adaptive-text-anonymization](https://github.com/gabrielloiseau/adaptive-text-anonymization)  
**Area**: AI Safety  
**Keywords**: Text anonymization, Privacy protection, Prompt optimization, Evolutionary algorithms, Privacy-utility trade-off

## TL;DR

This paper proposes an adaptive text anonymization framework that automatically discovers task-specific anonymization instructions for LLMs through evolutionary prompt optimization. It outperforms manually designed strategies across various privacy-utility trade-off scenarios and is executable on open-source models.

## Background & Motivation

**Background**: Text anonymization is a fundamental technology for enabling sensitive data sharing and analysis. Current methods are primarily divided into traditional sequence labeling (detecting and masking PII entities) and LLM-based adversary-collaboration pipelines (e.g., using an attacker LLM to guide anonymization decisions in AF methods).

**Limitations of Prior Work**: Existing LLM anonymization pipelines face three major limitations: (1) Fixed trade-off paradigms—designing a strategy manually for each scenario lacks flexibility for new requirements; (2) Reliance on manual prompt engineering, which is subjective, laborious, and often suboptimal; (3) Dependency on closed-source API models (like GPT-4/5), where sending sensitive data to external APIs contradicts the privacy objective itself.

**Key Challenge**: Anonymization is inherently context-dependent—strategies for medical reports and social media comments differ significantly. No "one-size-fits-all" solution exists, yet current methods cannot adaptively adjust strategies.

**Goal**: To design an adaptive framework capable of (1) automatically discovering anonymization prompts tailored to specific privacy-utility requirements, (2) running on open-source models, and (3) identifying multiple Pareto-optimal strategies in a single optimization run.

**Key Insight**: Reframe the anonymization problem as a "string discovery" problem—rather than modifying model parameters, search for the optimal natural language instructions to guide model behavior.

**Core Idea**: Utilize a Generalized Evolutionary Prompt Optimization (GEPA) algorithm to automatically search the anonymization prompt space, evolving task-adapted instructions from a general seed prompt to achieve an adaptive privacy-utility trade-off.

## Method

### Overall Architecture

The input consists of the text to be anonymized and the privacy-utility task specification $(p, u)$, and the output is the anonymized text. The framework searches for the optimal anonymization instruction $\Pi^*$ within a fixed computational budget via evolutionary prompt optimization. The process is divided into three stages: Initialization, Basic Feedback Warm-up, and Rich Feedback Refinement.

### Key Designs

1.  **Two-stage GEPA Evolutionary Optimization**:
    - **Function**: Discovers task-specific anonymization instructions from a general seed prompt through evolutionary search.
    - **Mechanism**: Maintains a prompt pool $P$. In each iteration, high-performance and diverse prompts are selected via Pareto sorting. A proposer agent analyzes execution trajectories and feedback to suggest mutations. New candidates are evaluated on a validation set and incorporated into the pool via Pareto pruning. Stage 2 uses simple scalar aggregated feedback $\mu$, transitioning to Stage 3 when performance plateaus.
    - **Design Motivation**: Evolutionary search naturally supports multi-objective optimization (privacy vs. utility), enabling the discovery of multiple Pareto-optimal solutions in a single run rather than converging to a single fixed trade-off point.

2.  **Rich Feedback Generation Mechanism**:
    - **Function**: Upgrades coarse-grained scalar feedback into structured feedback containing natural language explanations.
    - **Mechanism**: A specialized rich feedback agent (an independent LLM) decomposes the aggregated metric $\mu$ into $\mu_{rich}$, providing the proposer with interpretable, structured improvement signals for more substantial directional behavior updates.
    - **Design Motivation**: Scalar feedback is too coarse for the proposer to understand "what is wrong and how to improve." Rich feedback enables more precise prompt optimization using fewer evaluations within the remaining budget.

3.  **Adaptive Validation Sampling**:
    - **Function**: Evaluates candidate prompts using a sampled subset during the refinement stage to save computational budget.
    - **Mechanism**: Employs a round-robin strategy to prioritize samples with fewer evaluations as $D'_{valid} \subset D_{valid}$ (sampling ratio $\alpha=0.3$). The full validation set is used for final selection to ensure fairness.
    - **Design Motivation**: Evaluating on the full validation set every time consumes too much budget. Sampling improves budget efficiency while maintaining coverage diversity.

### Loss & Training

No gradient training is involved. The optimization objective is the aggregation of privacy and utility scores (e.g., mean), with multi-objective trade-offs achieved through Pareto selection. The evolutionary budget $B=1500$ LLM forward passes, with an early stopping patience $n=5$.

## Key Experimental Results

### Main Results

| Benchmark | Method | Privacy↑ | Utility↑ |
|-----------|--------|----------|----------|
| DB-Bio    | Optimized Qwen3 | 65.5 | 100 |
| DB-Bio    | AF (GPT-5)      | 78.0 | 92.1 |
| TAB       | Optimized Qwen3 | 92.3 | 56.2 |
| TAB       | AF (GPT-5)      | 59.9 | 42.5 |
| PUPA      | Optimized Qwen3 | 98.0 | 79.3 |
| PUPA      | AF (GPT-5)      | 94.2 | 46.0 |
| MedQA     | Optimized Qwen3 | 24.6 | 45.9 |
| MedQA     | AF (GPT-5)      | 24.4 | 45.8 |

### Ablation Study

| Configuration | Privacy-Utility Performance | Description |
|---------------|----------------------------|-------------|
| Seed Prompt   | Baseline                   | General seed prompt without optimization |
| Task-Specific Prompt | Medium              | Manually designed task-specific prompt |
| Optimized Prompt | Optimal                 | Automatically optimized prompt |
| OpenPII (Entity Detection) | High Utility, Low Privacy | Only detects PII entities, insufficient privacy protection |
| DP-Prompt ($\epsilon=100$) | High Privacy, Low Utility | Differential privacy noise severely damages utility |

### Key Findings
- The optimized open-source Qwen3-30B is competitive with or even outperforms the GPT-5 baseline on most tasks, especially in utility preservation.
- Different models exhibit distinct optimization characteristics: Mistral tends toward aggressive privacy enhancement (potentially sacrificing utility), Gemma improves conservatively, and Qwen is the most robust.
- A single optimization run can discover multiple Pareto-optimal strategies covering the full spectrum from privacy-first to utility-first.

## Highlights & Insights
- Abstracting the anonymization problem as a "string search" is a clever maneuver; each Pareto solution is simply a natural language string, making storage and deployment costs extremely low.
- Evolutionary optimization naturally supports multi-objective discovery. Finding multiple trade-offs in one run is far more efficient than traditional methods that require separate strategy designs for each point.
- The concept of the rich feedback mechanism—decomposing scalar metrics into structured natural language explanations—is transferable to any scenario requiring LLM self-improvement.

## Limitations & Future Work
- Evaluation of privacy and utility metrics still relies on closed-source LLMs (e.g., Gemini-2.5-flash), contradicting the goal of fully localized deployment.
- Each task still requires a small amount of labeled data (111 training + 111 validation), making it not entirely zero-shot.
- The anonymization capabilities of reasoning models (like CoT models) have not been considered, which could be a complementary direction.

## Related Work & Insights
- **vs AF (Staab et al.)**: AF uses fixed adversary-collaboration strategies and relies on GPT-5, while Ours uses evolutionary optimization to search for strategies automatically and can run on open-source models.
- **vs DP-Prompt**: Differential privacy methods provide theoretical guarantees but severely hurt utility. Ours significantly outperforms DP-Prompt in practical privacy-utility trade-offs.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframing anonymization as a prompt optimization problem is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets, 3 open-source models, multiple baselines, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions and systematic method descriptions.
- Value: ⭐⭐⭐⭐ Directly applicable value for sensitive data processing scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](../../CVPR2026/llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)

</div>

<!-- RELATED:END -->
