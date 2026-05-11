---
title: >-
  [Paper Note] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning
description: >-
  [NeurIPS 2025][LLM Reasoning][Chain-of-Thought] This paper provides the first systematic formalization of the "Thought Leap" phenomenon in CoT reasoning chains, and proposes CoT-Bridge…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Thought Leap"
  - "Reasoning Completeness"
  - "Data Augmentation"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 59168e72d76c9ffc
---

# Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.14684](https://arxiv.org/abs/2505.14684)
**Code**: [Project Page](https://zju-real.github.io/CoT-Bridge)
**Area**: LLM Reasoning
**Keywords**: Chain-of-Thought, Thought Leap, Reasoning Completeness, Data Augmentation, Mathematical Reasoning

## TL;DR
This paper provides the first systematic formalization of the "Thought Leap" phenomenon in CoT reasoning chains, and proposes CoT-Bridge, a model that automatically detects and fills omitted intermediate steps. It achieves up to +5.87% improvement on NuminaMath and can serve as a plug-and-play module to enhance distillation and RL pipelines.

## Background & Motivation
**Background**: LLMs have achieved remarkable progress on mathematical tasks via Chain-of-Thought reasoning, and the quality of CoT datasets directly determines the performance ceiling of trained models.

**Limitations of Prior Work**: Existing CoT datasets (e.g., MetaMathQA, NuminaMath) widely exhibit the Thought Leap phenomenon — human experts tend to omit "obvious" intermediate steps when writing reasoning chains due to their background knowledge, resulting in incomplete reasoning traces.

**Key Challenge**: Steps omitted by human experts are trivial to them but constitute fatal cognitive gaps for LLMs — models cannot bridge these reasoning gaps via implicit knowledge, severely impairing generalization ability.

**Goal**: (a) How to automatically detect leap positions in reasoning chains? (b) How to generate high-quality intermediate bridging steps? (c) Can the completed data consistently improve downstream model performance?

**Key Insight**: Through controlled experiments on MetaMathQA, the authors find that artificially introducing varying degrees of step omissions causes accuracy drops of up to 27.83% and significantly slower convergence, demonstrating that incomplete reasoning chains are more harmful than factual errors.

**Core Idea**: Train a dedicated Bridge model to detect reasoning leaps in CoT chains and automatically insert missing steps, thereby improving training data quality and downstream model reasoning capability.

## Method

### Overall Architecture
Given a potentially incomplete reasoning chain $C=(s_0, s_1, \ldots, s_n)$, CoT-Bridge jointly outputs: (1) a predicted set of leap positions $\hat{\mathcal{L}}$ and (2) the corresponding sequence of missing steps $\hat{\mathcal{M}}$. The generated steps are then inserted at the corresponding positions in the original chain to produce the completed chain $C_{bridged}$. The overall pipeline consists of three stages: training data construction → Bridge model training → application to existing datasets for augmentation.

### Key Designs

1. **Task Formalization — Definition of Thought Leap**:

    - Function: Define a completeness function $V(s_i, s_{i+1})$ to judge whether the reasoning between adjacent steps is sufficient.
    - Mechanism: If $V(s_k, s_{k+1}) = \text{False}$ for some $k$, a Thought Leap exists between $s_k$ and $s_{k+1}$, and a missing step sequence $S'_{miss}$ must be generated such that every pair of adjacent steps in the completed chain satisfies the completeness condition.
    - Design Motivation: Unlike prior work focusing on factual accuracy, this paper addresses the overlooked dimension of reasoning *structural completeness*.

2. **ScaleQM+ Dataset Construction**:

    - Function: Systematically remove intermediate steps from the structurally complete ScaleQuestMath dataset to construct "incomplete→complete" training pairs.
    - Mechanism: For a chain of length $m$, short chains ($m \leq 10$) have 1–2 steps removed, and long chains ($m > 10$) have 1–3 steps removed; the final step (containing the answer) is always retained; complete chains are preserved with probability 0.2 to teach the model to recognize cases requiring no completion.
    - Design Motivation: The "forward deletion + reverse completion" construction avoids costly manual annotation while naturally guaranteeing ground-truth quality, yielding 588k training samples in total.

3. **CoT-Bridge Model**:

    - Function: Fine-tuned on Qwen2.5-Math-7B, the model learns the joint mapping $f: C \rightarrow (\hat{\mathcal{L}}, \hat{\mathcal{M}})$ for detecting leap positions and generating bridging steps.
    - Mechanism: Contrasted against CoT-Bridge-Random (which generates content given random positions), CoT-Bridge must jointly learn *where* to bridge and *what* to insert.
    - Design Motivation: Experiments demonstrate that accurate leap localization is critical — inserting steps at random positions can disrupt reasoning coherence.

### Data Augmentation Application
CoT-Bridge is applied to MetaMathQA and NuminaMath-CoT, adapting to each dataset's step delimiter ("\n" or "\n\n") to generate augmented versions MetaMath-Bridge and NuminaMath-Bridge.

## Key Experimental Results

### Main Results (Meta-Llama3.1-8B + NuminaMath, average over 6 benchmarks)

| Method | GSM8K | MATH500 | GaoKao2023EN | AMC23 | Avg. |
|--------|-------|---------|-------------|-------|------|
| Direct SFT | 84.86 | 51.45 | 49.03 | 20.00 | 43.87 |
| QwenBridger-72B | 85.25 | 54.20 | 51.62 | 35.00 | 48.41 |
| CoT-Bridge-Random | 84.82 | 54.20 | 51.88 | 33.75 | 48.50 |
| **CoT-Bridge** | **85.97** | **56.80** | **54.42** | **35.63** | **49.74 (+5.87)** |

### Plug-and-Play Augmentation (Qwen2.5-Math-1.5B + Distill/Reject Sampling)

| Configuration | GSM8K | MATH500 | Avg. | Note |
|---------------|-------|---------|------|------|
| Distill - Direct SFT | 81.86 | 68.15 | 55.23 | Distillation data, direct training |
| Distill - CoT-Bridge | 82.52 | 71.50 | **58.25 (+3.02)** | Completion applied after distillation |
| Reject Sampling - Direct SFT | 83.36 | 74.90 | 60.44 | Rejection sampling data, direct training |
| Reject Sampling - CoT-Bridge | 83.74 | 75.25 | **61.81 (+1.37)** | Completion applied after sampling |

### Key Findings
- **Leap localization accuracy is critical**: CoT-Bridge-Random causes performance degradation on multiple benchmarks (e.g., GaoKao -1.56% and MathOdyssey -3.68% on Qwen2.5-Math-1.5B + NuminaMath), while CoT-Bridge yields consistent gains.
- **Largest gains on competition-level problems**: AMC23 sees a +15.63% improvement for LLaMA, indicating that harder problems benefit most from complete reasoning chains.
- **Improved OOD generalization**: Across 5 out-of-domain logical reasoning datasets, LLaMA achieves an average improvement of +2.99%, with a concurrent reduction in invalid response rate.
- **Enhanced RL cold-start**: Using Bridge-augmented data for SFT cold-start followed by GRPO yields a final RL accuracy of 63.98% vs. 60.88% (+3.1%).

## Highlights & Insights
- **Novel problem formulation**: Unlike prior work focusing on factual errors and answer accuracy, this paper is the first to systematically study the *structural completeness* of CoT chains, formalizing Thought Leap as a detectable and repairable task — an insightful and underexplored perspective.
- **Elegant data construction**: Constructing training pairs by deleting steps from complete data and learning to restore them avoids annotation costs while naturally guaranteeing ground-truth quality, since the supervision signal derives from the original complete chains.
- **Practical plug-and-play design**: CoT-Bridge can be seamlessly integrated on top of distillation, rejection sampling, and RL pipelines as a general-purpose data quality enhancement module, with low barriers to adaptation in other settings.

## Limitations & Future Work
- **Reliance on the completeness assumption of ScaleQuestMath**: The approach treats ScaleQuestMath as an approximately ideal complete CoT source; however, this dataset may itself contain Thought Leaps, which limits the quality ceiling of the Bridge model's completions.
- **Restricted to the mathematical domain**: Although OOD gains are observed on logical reasoning tasks, generalization to code generation, scientific reasoning, and other domains remains unvalidated.
- **Fixed Bridge model scale at 7B**: The scaling behavior of larger or smaller Bridge models is unexplored, as is the feasibility of distilling Bridge capabilities into smaller models.
- **Lack of fine-grained completion quality evaluation**: Completion quality is assessed primarily via downstream task accuracy; direct evaluation of the mathematical correctness and logical coherence of generated steps is absent.

## Rating
- Novelty: ⭐⭐⭐⭐ First formalization of the Thought Leap problem with a unique perspective, though the core method (deletion then recovery) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple models, datasets, and settings (distillation/RL/OOD) with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and experimental organization is well-structured.
- Value: ⭐⭐⭐⭐ Offers a practical plug-and-play data quality enhancement tool with concrete guidance for CoT dataset construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] On Learning Verifiers and Implications to Chain-of-Thought Reasoning](on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)
- [\[NeurIPS 2025\] Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)

</div>

<!-- RELATED:END -->
