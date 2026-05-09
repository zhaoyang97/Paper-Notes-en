---
title: >-
  [Paper Note] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation
description: >-
  [ACL 2026][LLM/NLP][recipe generation] This paper proposes a topological loss function based on Sinkhorn divergence, representing ingredient lists as point clouds in embedding space and minimizing the geometric discrepancy between predicted and reference ingredients. The approach significantly improves ingredient recall and quantity precision in structured recipe generation, with generated outputs preferred by human evaluators in 62% of cases.
tags:
  - ACL 2026
  - LLM/NLP
  - recipe generation
  - topological loss
  - optimal transport
  - structured text generation
  - composite loss function
date: 2026-05-08
content_hash: 21c0a610f31832e7
---

# Losses that Cook: Topological Optimal Transport for Structured Recipe Generation

**Conference**: ACL 2026
**arXiv**: [2601.02531](https://arxiv.org/abs/2601.02531)
**Code**: [GitHub](https://github.com/DarthReca/losses-cook)
**Area**: Text Generation
**Keywords**: recipe generation, topological loss, optimal transport, structured text generation, composite loss function

## TL;DR
This paper proposes a topological loss function based on Sinkhorn divergence, representing ingredient lists as point clouds in embedding space and minimizing the geometric discrepancy between predicted and reference ingredients. The approach significantly improves ingredient recall and quantity precision in structured recipe generation, with generated outputs preferred by human evaluators in 62% of cases.

## Background & Motivation

**State of the Field**: Recipe generation requires not only fluent text but also precise ingredients, quantities, timing, temperatures, and procedural consistency across steps. Prevailing approaches fine-tune language models using cross-entropy (CE) loss.

**Limitations of Prior Work**: CE treats all tokens as equally important, yet recipes exhibit strong asymmetry—high-impact tokens (ingredients, quantities, times, temperatures, key actions) differ substantially from low-impact tokens (connectives). This leads to common failure modes: low ingredient recall, inaccurate quantities, and steps that are grammatically correct but procedurally infeasible.

**Root Cause**: Token-level training objectives cannot capture the holistic structural properties of an ingredient set. Omitting a critical ingredient (e.g., eggs in Carbonara) or doubling a temperature renders the entire recipe unusable, regardless of textual fluency.

**Paper Goals**: Design loss functions that directly optimize ingredient set completeness and numerical accuracy while preserving textual fluency.

**Starting Point**: Drawing from optimal transport theory, the paper treats ingredient lists as point clouds in embedding space and uses geometric distance to measure the correspondence between predicted and reference ingredients.

**Core Idea**: Minimize the transport distance between predicted and reference ingredient point clouds via Sinkhorn divergence, explicitly encoding ingredient-level structural constraints into the training loss.

## Method

### Overall Architecture
Given a natural language prompt (e.g., "Generate a recipe for Pasta Carbonara"), the model outputs a structured JSON containing an ingredient list and a sequence of step instructions. The approach fine-tunes Qwen3-4B with LoRA, with the key contribution being a composite loss function that replaces the standard CE objective.

### Key Designs

1. **Topological Loss**:

    - Function: Aligns the point cloud distributions of predicted and reference ingredients in embedding space.
    - Mechanism: For tokens corresponding to the ingredient portion of the predicted sequence, logits are converted to probability distributions via softmax, and a soft embedding is computed as $emb_{soft} = P \cdot E$ (where $E$ is the token embedding matrix), forming the predicted point cloud. The reference point cloud is constructed by directly looking up embeddings from the ground-truth sequence. The Sinkhorn divergence $\mathcal{L}_{Topo} = \mathcal{S}_\epsilon(PC_{pred}, PC_{target})$ then measures the geometric dissimilarity between the two point clouds.
    - Design Motivation: CE penalizes all substitutions uniformly, whereas the topological loss captures semantic proximity—predicting "salt" instead of "pepper" should incur a smaller penalty than predicting "egg," as the former pair is geometrically closer in embedding space.

2. **Dice Loss**:

    - Function: Optimizes set-level token overlap.
    - Mechanism: A differentiable Dice coefficient measures the overlap between the predicted and reference token sets, encouraging the model to generate the correct token inventory.
    - Design Motivation: Compared to CE and Focal Loss, Dice loss more effectively handles coverage of critical tokens, particularly excelling in time and temperature precision.

3. **Mixed Loss**:

    - Function: Combines the complementary strengths of the topological and Dice losses.
    - Mechanism: $L = 0.6 L_{CE} + 0.2 L_{Dice} + 0.2 L_{Topo}$, where CE maintains linguistic fluency, Dice improves numerical precision, and the topological loss reinforces ingredient structural consistency.
    - Design Motivation: Individual custom losses each have distinct strengths (Topo excels at ingredient recall; Dice excels at time and temperature accuracy); their combination yields complementary gains.

### Loss & Training
All composite losses are combined with CE in the form $L = 0.6 L_{CE} + 0.4 L_{custom}$. Training is conducted on a 5,000-sample subset of the RECIPE-NLG dataset (covering pasta, rice, and sandwiches), augmented with 235 manually curated culinary questions addressing ingredient identification, substitution, scaling, and quantity reasoning.

## Key Experimental Results

### Main Results

| Model | R1↑ | BS↑ | AP↑ | QP↑ | IR↑ | TeP↑ | TiP↑ | AD↓ | SD↓ |
|------|-----|-----|-----|-----|-----|------|------|-----|-----|
| Gemini 2.0 (No-FT) | 15.08 | 88.50 | 43.80 | 44.51 | 37.47 | 76.88 | 36.92 | 36.21 | 48.60 |
| Qwen3-4B (CE) | 27.30 | 88.78 | 45.09 | 50.94 | 35.98 | 61.93 | 52.09 | 37.83 | 39.48 |
| Qwen3-4B (Topo) | 30.40 | 90.97 | 59.68 | 63.93 | 48.59 | 65.59 | 55.55 | 30.49 | 34.09 |
| Qwen3-4B (Topo+Dice) | **31.90** | **90.99** | 57.59 | **65.09** | 47.09 | 67.89 | **61.95** | **30.49** | **34.09** |

### Ablation Study

| Configuration | IR↑ | QP↑ | Notes |
|------|-----|-----|------|
| CE only | 35.98 | 50.94 | Baseline |
| CE + Focal | 43.09 | 54.94 | Marginal improvement; underperforms other losses |
| CE + Dice | 44.90 | 57.44 | Strong numerical precision |
| CE + Topo | 48.59 | 63.93 | Best ingredient recall |
| CE + Topo + Dice | 47.09 | 65.09 | Best overall |

### Key Findings
- The topological loss yields the largest improvement in ingredient recall (+12.6% vs. CE), validating the effectiveness of point cloud alignment in embedding space.
- Dice loss achieves the strongest temperature precision (74.58% vs. 61.93% for CE), demonstrating its suitability for numerical constraints.
- The mixed Topo+Dice configuration produces synergistic gains on QP and TiP, surpassing either loss used individually.
- In human evaluation, Topo+Dice substantially outperforms CE on overall quality (62% vs. 11% preference), with a 67.5% reduction in generation errors.

## Highlights & Insights
- Modeling ingredient lists as point clouds and aligning them via optimal transport is a particularly elegant formulation, recasting the set-matching problem as a geometric one that naturally supports partial matching and semantic proximity. This paradigm is transferable to any generation task requiring set-level matching (e.g., entity list generation, keyword extraction).
- The soft embedding design renders the topological loss fully differentiable, enabling end-to-end training without additional decoding steps.
- The complementarity of different loss components in the composite formulation (structural vs. numerical) offers a compelling design paradigm for task-specific loss engineering.

## Limitations & Future Work
- Training data covers only three food categories (pasta, rice, sandwiches); generalization to other cuisines remains unverified.
- The data augmentation comprises only 235 manually curated questions, representing a limited scale.
- Evaluation metrics rely on automated extraction pipelines, which may introduce noise for non-standard formats or rare culinary terminology.
- The topological loss depends on the geometric properties of the embedding space and incurs additional computational overhead.
- Future work may extend the approach to broader cuisines and incorporate allergen and nutritional constraints.

## Related Work & Insights
- **vs. CE-only fine-tuning**: CE assigns uniform weight to all tokens; this work demonstrates that loss functions targeting critical tokens substantially improve structured output quality.
- **vs. Focal Loss**: Focal loss reweights difficult samples but cannot capture set-level structure, underperforming Dice and Topo on recipe-specific metrics.
- **vs. constrained decoding methods**: This paper addresses the problem at training time, incurring no additional inference complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing optimal transport into recipe generation loss design is a novel idea, though the application scope is relatively narrow.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-model comparisons, ablation studies, and human evaluation, though data scale and domain coverage are limited.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with clear method exposition.
- Value: ⭐⭐⭐ The technical approach is intellectually stimulating, but the application domain is narrow and broader transferability remains to be validated.
To be supplemented after a thorough reading.

## Background & Motivation
To be supplemented after a thorough reading.

## Method
To be supplemented after a thorough reading.

## Key Experimental Results
To be supplemented after a thorough reading.

## Highlights & Insights
To be supplemented after a thorough reading.

## Limitations & Future Work
To be supplemented after a thorough reading.

## Related Work & Insights
To be supplemented after a thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Near-Optimal Online Deployment and Routing for Streaming LLMs](../../ICLR2026/llm_nlp/near-optimal_online_deployment_and_routing_for_streaming_llms.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)

<!-- RELATED:END -->
