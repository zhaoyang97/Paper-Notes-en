---
title: >-
  [Paper Note] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation
description: >-
  [ACL 2026][Text Generation][Recipe Generation] This paper proposes a topological loss function based on Sinkhorn divergence that represents ingredient lists as point clouds in an embedding space. By minimizing the geomet…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Recipe Generation"
  - "Topological Loss"
  - "Optimal Transport"
  - "Structured Text Generation"
  - "Composite Loss Functions"
date: 2026-05-08
content_hash: 8174fe92b4c9ef8c
---

# Losses that Cook: Topological Optimal Transport for Structured Recipe Generation

**Conference**: ACL 2026  
**arXiv**: [2601.02531](https://arxiv.org/abs/2601.02531)  
**Code**: [GitHub](https://github.com/DarthReca/losses-cook)  
**Area**: Text Generation  
**Keywords**: Recipe Generation, Topological Loss, Optimal Transport, Structured Text Generation, Composite Loss Functions

## TL;DR
This paper proposes a topological loss function based on Sinkhorn divergence that represents ingredient lists as point clouds in an embedding space. By minimizing the geometric discrepancy between predicted and ground-truth ingredients, the method significantly improves ingredient recall and quantity accuracy in structured recipe generation, being preferred in 62% of human evaluations.

## Background & Motivation

**Background**: Recipe generation requires not only fluent text but also precise consistency across ingredients, quantities, time, temperature, and procedural steps. Current mainstream methods fine-tune language models based on Cross-Entropy (CE) loss.

**Limitations of Prior Work**: CE treats all tokens as equally important, failing to account for the strong asymmetry in recipes—specifically the massive difference between high-impact tokens (ingredients, quantities, time, temperature, core actions) and low-impact tokens (connective words). This leads to common failure modes: low ingredient recall, inaccurate quantities, and steps that are grammatically correct but procedurally non-executable.

**Key Challenge**: Token-level training objectives cannot capture the global structural properties of an ingredient set. Omitting a key ingredient (e.g., eggs in Carbonara) or doubling a temperature renders the entire recipe unusable, regardless of text fluency.

**Goal**: Design a loss function that directly optimizes ingredient set integrity and numerical accuracy while maintaining text fluency.

**Key Insight**: Leveraging Optimal Transport theory, ingredient lists are treated as point clouds in an embedding space, using geometric distances to measure the alignment between predicted and reference ingredients.

**Core Idea**: Use Sinkhorn divergence to minimize the transportation distance between predicted and reference ingredient point clouds, explicitly encoding ingredient-level structural constraints into the training loss.

## Method

### Overall Architecture
The input is a natural language prompt (e.g., "Generate a recipe for Spaghetti Carbonara"), and the output is a structured JSON containing ingredient lists and procedural instructions. Based on the Qwen3-4B model, fine-tuning is performed using LoRA. The core innovation lies in designing a composite loss function to replace simple CE.

### Key Designs

1.  **Topological Loss**:
    - **Function**: Aligns point cloud distributions of predicted and reference ingredients in the embedding space.
    - **Mechanism**: For tokens in the ingredient portion of the predicted sequence, logits are converted into a probability distribution via softmax. Weighted embeddings are calculated as $emb_{soft} = P \cdot E$ (where $E$ is the word embedding matrix) to construct the predicted point cloud. The reference point cloud is constructed via direct embedding lookup. Sinkhorn divergence $\mathcal{L}_{Topo} = \mathcal{S}_\epsilon(PC_{pred}, PC_{target})$ is then used to measure geometric dissimilarity between the two clouds.
    - **Design Motivation**: While CE treats all substitutions equally, topological loss captures semantic proximity—penalizing the prediction of "salt" as "pepper" less than predicting it as "egg" because they are closer in the embedding space.

2.  **Dice Loss**:
    - **Function**: Optimizes token overlap at the set level.
    - **Mechanism**: A differentiable Dice coefficient measures the overlap between predicted and reference token sets, encouraging the model to generate the correct set of tokens.
    - **Design Motivation**: Compared to CE and Focal Loss, Dice loss is superior at handling key token coverage, particularly for time and temperature precision.

3.  **Mixed Loss**:
    - **Function**: Integrates the advantages of topological and Dice losses.
    - **Mechanism**: $L = 0.6 L_{CE} + 0.2 L_{Dice} + 0.2 L_{Topo}$. CE maintains linguistic fluency, Dice improves numerical precision, and topological loss strengthens ingredient structural consistency.
    - **Design Motivation**: Each custom loss has specific strengths (Topo for ingredient recall, Dice for time/temperature); the mixture yields complementary gains.

### Loss & Training
All composite losses are combined with CE in the form $L = 0.6 L_{CE} + 0.4 L_{custom}$. Training is based on a 5000-sample subset of the RECIPE-NLG dataset (Pasta, Rice, Sandwiches) and augmented with 235 human-curated cooking questions covering ingredient identification, substitution, scaling, and quantity reasoning.

## Key Experimental Results

### Main Results

| Model | R1↑ | BS↑ | AP↑ | QP↑ | IR↑ | TeP↑ | TiP↑ | AD↓ | SD↓ |
|-------|-----|-----|-----|-----|-----|------|------|-----|-----|
| Gemini 2.0 (No-FT) | 15.08 | 88.50 | 43.80 | 44.51 | 37.47 | 76.88 | 36.92 | 36.21 | 48.60 |
| Qwen3-4B (CE) | 27.30 | 88.78 | 45.09 | 50.94 | 35.98 | 61.93 | 52.09 | 37.83 | 39.48 |
| Qwen3-4B (Topo) | 30.40 | 90.97 | 59.68 | 63.93 | 48.59 | 65.59 | 55.55 | 30.49 | 34.09 |
| Qwen3-4B (Topo+Dice) | **31.90** | **90.99** | 57.59 | **65.09** | 47.09 | 67.89 | **61.95** | **30.49** | **34.09** |

### Ablation Study

| Configuration | IR↑ | QP↑ | Description |
|---------------|-----|-----|-------------|
| CE only | 35.98 | 50.94 | Baseline |
| CE + Focal | 43.09 | 54.94 | Slight improvement, but inferior to other losses |
| CE + Dice | 44.90 | 57.44 | Better numerical precision |
| CE + Topo | 48.59 | 63.93 | Best ingredient recall |
| CE + Topo + Dice | 47.09 | 65.09 | Best overall performance |

### Key Findings
- Topological loss provides the largest gain in ingredient recall (+12.6% vs CE), proving the effectiveness of point cloud alignment in embedding space.
- Dice loss is strongest for temperature precision (74.58% vs 61.93% for CE), excelling at numerical constraints.
- The hybrid Topo+Dice approach generates synergistic gains in QP and TiP, exceeding the use of either loss individually.
- In human evaluation, Topo+Dice outperformed CE in overall quality by 62% vs 11%, with a 67.5% reduction in generation errors.

## Highlights & Insights
- Modeling ingredient lists as point clouds aligned via optimal transport is an ingenious approach, transforming set matching into a geometric problem that naturally supports partial matching and semantic proximity. This logic can be transferred to any generation task requiring set-level matching (e.g., entity list generation, keyword extraction).
- The soft embedding design makes the topological loss differentiable, allowing for end-to-end training without additional decoding steps.
- The complementarity of different components in the composite loss (structural vs. numerical) provides a strong paradigm for loss function design.

## Limitations & Future Work
- Training data is limited to pasta, rice, and sandwiches; generalization to other cuisines remains unverified.
- Data augmentation utilized only 235 human-curated questions, which is small in scale.
- Evaluation metrics rely on automated extraction pipelines, which may be noisy for non-standard formats or rare culinary terms.
- Topological loss depends on the geometric properties of the embedding space and increases computational overhead.
- Future work could expand to broader cuisines and incorporate allergen or nutritional constraints.

## Related Work & Insights
- **vs CE-only fine-tuning**: CE weights all tokens equally; this work proves that loss designs targeting key tokens significantly improve structured output quality.
- **vs Focal Loss**: Focal re-weights difficult samples but fails to capture set-level structure, underperforming compared to Dice and Topo on recipe-specific metrics.
- **vs Constrained Decoding**: This study addresses the issue from the training side, avoiding increased inference complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative application of optimal transport to loss design for recipe generation, though the scenario is specialized.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multi-model comparisons, ablations, and human evaluations, but constrained by data scale and domain coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-articulated methodology.
- Value: ⭐⭐⭐ Inspiring technical approach, though the application scope is narrow; requires verification of broader transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ICML 2026\] Characterizing the Effect of Noise in Language Generation in the Limit](../../ICML2026/nlp_generation/characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)
- [\[ACL 2026\] Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)

</div>

<!-- RELATED:END -->
