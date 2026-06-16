---
title: >-
  [Paper Note] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation
description: >-
  [ACL 2026][Text Generation][Paper Note] This paper proposes a topological loss function based on Sinkhorn divergence that represents ingredient lists as point clouds in embedding space. By minimizing the geometric discrepancy between predicted and ground-truth ingredients, it significantly improves ingredient recall and quantity accuracy in structured recipe
tags:
  - ACL 2026
  - Text Generation
date: 2026-05-08
content_hash: a8ab480e93fc7350
---
# Losses that Cook: Topological Optimal Transport for Structured Recipe Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.02531](https://arxiv.org/abs/2601.02531)  
**Code**: [GitHub](https://github.com/DarthReca/losses-cook)  
**Area**: Text Generation  
**Keywords**: Recipe Generation, Topological Loss, Optimal Transport, Structured Text Generation, Composite Loss Functions

## TL;DR
This paper proposes a topological loss function based on Sinkhorn divergence that represents ingredient lists as point clouds in embedding space. By minimizing the geometric discrepancy between predicted and ground-truth ingredients, it significantly improves ingredient recall and quantity accuracy in structured recipe generation, being preferred in 62% of human evaluations.

## Background & Motivation

**Background**: Recipe generation demands not only fluent text but also precision in ingredients, dosages, timing, temperature, and procedural consistency. Current mainstream methods involve fine-tuning language models using Cross-Entropy (CE) loss.

**Limitations of Prior Work**: CE treats all tokens as equally important, yet recipes exhibit strong inherent asymmetry—huge differences exist between high-impact tokens (ingredients, quantities, time, temperature, core actions) and low-impact tokens (connectives). This leads to common failure modes: low ingredient recall, inaccurate quantities, and steps that are grammatically correct but procedurally unexecutable.

**Key Challenge**: Token-level training objectives fail to capture the holistic structural properties of ingredient sets—omitting a key ingredient (e.g., eggs in pasta) or doubling a temperature makes the recipe unusable even if the text is fluent.

**Goal**: To design loss functions capable of directly optimizing ingredient set integrity and numerical accuracy while maintaining textual fluency.

**Key Insight**: Starting from Optimal Transport theory, ingredient lists are treated as point clouds in embedding space, utilizing geometric distance to measure the matching degree between predicted and ground-truth ingredients.

**Core Idea**: Use Sinkhorn divergence to minimize the transport distance between predicted and reference ingredient point clouds, explicitly encoding ingredient-level structural constraints into the training loss.

## Method

### Overall Architecture
The input is a natural language prompt (e.g., "Generate a recipe for Spaghetti Carbonara"), and the output is a structured JSON containing ingredient lists and instruction lists. Based on the Qwen3-4B model, it is fine-tuned using LoRA. The core innovation lies in the design of composite loss functions as alternatives to single CE loss.

### Key Designs

**1. Topological Loss: Explicitly incorporating the geometric structure of ingredient sets into the training objective**

CE treats all token substitutions equally; misidentifying "salt" as "pepper" is penalized as heavily as misidentifying it as "egg," even though the former is much closer in embedding space. The topological loss treats the entire ingredient list as a point cloud in embedding space. For ingredients in the predicted sequence, logits are converted into a probability distribution via softmax, and weighted embeddings $emb_{soft} = P \cdot E$ (where $E$ is the embedding matrix) are calculated to construct a predicted point cloud. For the ground truth, reference point clouds are built via embedding lookup. Geometric dissimilarity is measured using Sinkhorn divergence: $\mathcal{L}_{Topo} = \mathcal{S}_\epsilon(PC_{pred}, PC_{target})$. Because penalties follow embedding distances, the model is guided toward aligning "semantically proximal" ingredient sets rather than optimizing token-by-token—omitting a key ingredient or swapping it for an unrelated one will immediately increase the geometric distance.

**2. Dice Loss: Monitoring key token coverage from a set-overlap perspective**

CE is a token-level objective and is insensitive to how well the actual set of necessary tokens is covered. Dice loss employs a differentiable Dice coefficient to measure the overlap between the predicted and reference token sets, directly encouraging the model to generate all required key tokens. Compared to CE and Focal Loss, it is more targeted toward key token coverage, particularly excelling in numerical accuracy metrics for time and temperature.

**3. Mixed Loss: Allowing multiple custom losses to compensate for each other's weaknesses**

Experiments showed that Topo excels in ingredient recall while Dice excels in time and temperature precision. To achieve a comprehensive result, the authors weighted these with CE: $L = 0.6 L_{CE} + 0.2 L_{Dice} + 0.2 L_{Topo}$. CE maintains foundational linguistic fluency, Dice enhances numerical precision, and Topological loss strengthens ingredient structural consistency. The combination yields complementary gains, with mixed QP and TiP scores exceeding any single loss function.

### Loss & Training
All composite losses are combined with CE in the form $L = 0.6 L_{CE} + 0.4 L_{custom}$. Training is based on a 5,000-sample subset of the RECIPE-NLG dataset (pasta, rice, sandwiches), augmented with 235 human-curated cooking questions covering ingredient identification, substitution, scaling, and quantitative reasoning.

## Key Experimental Results

### Main Results

| Model | R1↑ | BS↑ | AP↑ | QP↑ | IR↑ | TeP↑ | TiP↑ | AD↓ | SD↓ |
|------|-----|-----|-----|-----|-----|------|------|-----|-----|
| Gemini 2.0 (No-FT) | 15.08 | 88.50 | 43.80 | 44.51 | 37.47 | 76.88 | 36.92 | 36.21 | 48.60 |
| Qwen3-4B (CE) | 27.30 | 88.78 | 45.09 | 50.94 | 35.98 | 61.93 | 52.09 | 37.83 | 39.48 |
| Qwen3-4B (Topo) | 30.40 | 90.97 | 59.68 | 63.93 | 48.59 | 65.59 | 55.55 | 30.49 | 34.09 |
| Qwen3-4B (Topo+Dice) | **31.90** | **90.99** | 57.59 | **65.09** | 47.09 | 67.89 | **61.95** | **30.49** | **34.09** |

### Ablation Study

| Configuration | IR↑ | QP↑ | Description |
|------|-----|-----|------|
| CE only | 35.98 | 50.94 | Baseline |
| CE + Focal | 43.09 | 54.94 | Slight improvement, but inferior to other losses |
| CE + Dice | 44.90 | 57.44 | Better numerical precision |
| CE + Topo | 48.59 | 63.93 | Best ingredient recall |
| CE + Topo + Dice | 47.09 | 65.09 | Best overall performance |

### Key Findings
- Topological loss provides the largest Gain in ingredient recall (+12.6% vs CE), proving the effectiveness of point cloud alignment in embedding space.
- Dice loss is strongest in temperature accuracy (74.58% vs 61.93% for CE), demonstrating targeted numerical constraint capability.
- The hybrid Topo+Dice approach produces synergistic gains in QP and TiP, surpassing the performance of either loss used in isolation.
- In human evaluations, Topo+Dice outperformed CE in overall quality by 62% vs 11%, reducing generation errors by 67.5%.

## Highlights & Insights
- Modeling ingredient lists as point clouds and aligning them via Optimal Transport is a clever approach. It transforms a set-matching problem into a geometric one, naturally supporting partial matching and semantic proximity. This logic can be transferred to any generation task requiring set-level matching (e.g., entity list generation, keyword extraction).
- The use of soft embeddings makes the topological loss differentiable, allowing for end-to-end training without additional decoding steps.
- The complementarity of the composite loss components (structural vs. numerical) provides an excellent paradigm for loss function design.

## Limitations & Future Work
- Training data only covers three categories (pasta, rice, sandwiches); generalization to other cuisines remains unverified.
- Data augmentation relied on only 235 human questions, which is relatively small in scale.
- Evaluation metrics depend on automated extraction processes, which may be noisy for non-standard formats or rare culinary terms.
- Topological loss increases computational overhead due to its reliance on embedding space geometric properties.
- Future work could extend to a broader range of cuisines and consider allergens or nutritional constraints.

## Related Work & Insights
- **vs CE-only fine-tuning**: While CE weights all tokens equally, this work demonstrates that loss designs targeting key tokens significantly enhance structured output quality.
- **vs Focal Loss**: Focal loss reweights difficult samples but fails to capture set-level structures, making it less effective for recipe-specific metrics compared to Dice and Topo.
- **vs Constrained decoding methods**: Ours solves the problem at the training stage, avoiding added inference complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Optimal Transport into the loss function design for recipe generation is innovative, though the application scenario is specific.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-model comparisons, ablations, and human evaluations, though data scale and domain coverage are limited.
- Writing Quality: ⭐⭐⭐⭐ The paper structure is clear, and the methodology is well-articulated.
- Value: ⭐⭐⭐ The technical approach is inspiring, but the application scenario is narrow; its broader transferability needs further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](../../ACL2025/nlp_generation/doc_level_mbr_optimal_transport.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)

</div>

<!-- RELATED:END -->
