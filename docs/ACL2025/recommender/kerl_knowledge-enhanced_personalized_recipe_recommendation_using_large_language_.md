---
title: >-
  [Paper Note] KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models
description: >-
  [ACL 2025][Recommender Systems][Recipe Recommendation] Proposes KERL, a unified food recommendation system that combines the FoodKG knowledge graph with Multi-LoRA fine-tuning of Phi-3-mini. It accomplishes three functions: personalized recipe recommendation (F1=0.973), recipe generation, and micronutrient estimation, performance-wise significantly outperforming baseline LLMs and traditional embedding methods.
tags:
  - "ACL 2025"
  - "Recommender Systems"
  - "Recipe Recommendation"
  - "Knowledge Graphs"
  - "LLM Fine-tuning"
  - "Multi-LoRA"
  - "Nutrition Generation"
date: 2026-05-08
content_hash: 5c020f38d9339676
---

# KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.14629](https://arxiv.org/abs/2505.14629)  
**Code**: [https://github.com/mohbattharani/KERL](https://github.com/mohbattharani/KERL)  
**Area**: Recommendation Systems / Knowledge Graphs  
**Keywords**: Recipe Recommendation, Knowledge Graphs, LLM Fine-tuning, Multi-LoRA, Nutrition Generation

## TL;DR
Proposes KERL, a unified food recommendation system that combines the FoodKG knowledge graph with Multi-LoRA fine-tuning of Phi-3-mini. It accomplishes three functions: personalized recipe recommendation (F1=0.973), recipe generation, and micronutrient estimation, performance-wise significantly outperforming baseline LLMs and traditional embedding methods.

## Background & Motivation
**Background**: Recipe recommendation systems must consider individual preferences, dietary constraints, and health guidelines. Existing methods follow two lines of work: (a) traditional knowledge-graph-based methods (such as pFoodReq), which use embedding similarity to match queries with recipes; (b) LLMs applied to recipe generation (e.g., FoodLMM, LLaVA-Chef), which primarily focus on a single capability.

**Limitations of Prior Work**: (a) There is a lack of a unified system that simultaneously performs recommendation, recipe generation, and nutritional analysis; (b) LLMs suffer from hallucinations and outdated information; (c) the integration of existing food KGs and LLMs is superficial, exhibiting poor performance especially when handling queries with complex constraints (ingredient preferences + nutritional restrictions).

**Key Challenge**: LLMs possess powerful language understanding capabilities but lack structured food knowledge, whereas KGs contain rich food relationships but lack flexible natural language understanding.

**Goal**: How to effectively inject structured knowledge from a food KG into LLMs to build a unified system for recommendation, recipe generation, and nutritional analysis that satisfies individual preference constraints.

**Key Insight**: Rely on SPARQL to retrieve subgraphs from the KG as context for the LLM, and train a lightweight Multi-LoRA adapter for each of the three subtasks.

**Core Idea**: KG subgraph retrieval + Multi-LoRA fine-tuning of Phi-3-mini = a unified personalized food recommendation system.

## Method

### Overall Architecture
KERL consists of three modules sharing a single Phi-3-mini backbone, each utilizing a distinct LoRA adapter: (1) **KERL-Recom**: receives user queries (including ingredient preferences and nutritional constraints), retrieves subgraphs from FoodKG as context, and recommends recipes satisfying the constraints; (2) **KERL-Recipe**: generates cooking instructions based on the recommended recipe name and ingredients; (3) **KERL-Nutri**: estimates micronutrient information (proteins, fiber, fats, cholesterol, etc.) based on the recipe name, ingredients, and cooking instructions.

### Key Designs

1. **KG Subgraph Retrieval and LLM-Enhanced Recommendation (KERL-Recom)**:

    - **Function**: Extract entities (tags like "vegetarian", ingredients like "spinach") from natural language queries, construct SPARQL queries to retrieve subgraphs from FoodKG, and serialize these subgraphs into text to serve as the LLM context.
    - **Mechanism**: During training, sample $K/2$ positive recipes satisfying constraints and $K/2$ negative recipes from the recipe set $R(t_j)$ associated with the tags to serve as the context. During inference, iteratively traverse all tag-related recipes (input to the LLM in batches) and merge results from multiple calls.
    - **Design Motivation**: Provide structured KG information directly in the LLM's context, forcing the LLM to make selections based on real data rather than generating them from scratch, thereby fundamentally reducing hallucinations.

2. **Multi-LoRA Architecture**:

    - **Function**: Train one LoRA adapter (r=64, α=16) for each of the three tasks using the same shared Phi-3-mini backbone model.
    - **Mechanism**: Activate the corresponding task's adapter during inference, while sharing the base model. The three adapters can be deployed simultaneously on a single GPU.
    - **Design Motivation**: Avoid waste of resource cost for training three independent models. LoRA enables efficient fine-tuning of the 3.8B parameter model on four A6000 GPUs.

3. **Automated Benchmark Construction**:

    - **Function**: Automatically construct 77,900 QA pairs with complex constraints based on FoodKG.
    - **Mechanism**: Use template questions + randomly sampled ingredient preferences (likes/dislikes) + nutritional constraints (<, >, ranges) to deterministically compute recipes that satisfy all constraints from the KG as ground truth.
    - **Design Motivation**: Existing datasets (such as pFoodReq with only 6,918 entries) are too small in scale and lack diverse constraint types.

### Loss & Training
Standard cross-entropy loss + LoRA (r=64, α=16, dropout=0.5). Learning rate is $2 \times 10^{-5}$ with a cosine scheduler, training each adapter for 2 epochs.

## Key Experimental Results

### Main Results

| Model | Parameters | Precision | Recall | F1 |
|------|--------|-----------|--------|----|
| KERL-Recom | 3.8B+LoRA | **0.978** | **0.969** | **0.973** |
| Llama-2 | 7B (zero-shot) | 0.825 | 0.627 | 0.713 |
| Phi-3-mini-128K | 3.8B (zero-shot) | 0.778 | 0.278 | 0.410 |
| Mistral | 7B (zero-shot) | 0.536 | 0.558 | 0.547 |
| pFoodReq (Embedding) | - | - | 0.618 | 0.637 |

KERL-Recom achieves an F1 of 0.973 on the KGQA benchmark, which is 26 points higher than zero-shot Llama-2 and 21.7 points higher than the traditional embedding method pFoodReq.

### Ablation Study

| Module/Configuration | Metric | Description |
|-----------|------|------|
| KERL-Recom (Full) | F1=0.973 | KG Subgraph + LoRA Fine-tuning |
| Phi-3-mini-128K (zero-shot) | F1=0.410 | No fine-tuning, leveraging long context only |
| Phi-3-mini-4K (zero-shot) | F1=0.071 | Insufficient context window |
| KERL-Recipe (title+ing) | BLEU-4=0.079 | Generating recipe instructions from title + ingredients |
| LLaVA-Chef (title+ing) | BLEU-4=0.065 | Comparison baseline |
| KERL-Nutri (title+ing+instr) | MAE=9.38| Nutritional estimation with the most complete input |

### Key Findings
- **KG injection is crucial**: Zero-shot Phi-3-mini F1 is only 0.071–0.410, whereas fine-tuning makes it reach 0.973, indicating that LLMs cannot handle complex food constraints based solely on their internal knowledge.
- **Excellent generalization across recipe types**: All tags except dairy-free (due to few training samples) and gluten-free achieve F1 > 0.93.
- **More complete inputs yield better nutritional estimation**: Under title+ingredients+instructions, MAE (9.38) outperforms using title only (13.13).
- **Small models are sufficient**: Phi-3-mini (3.8B) + LoRA significantly outperforms the zero-shot performance of 7B models.

## Highlights & Insights
- **RAG paradigm using KG subgraphs as LLM context**: Distinct from vector-retrieval RAG, this work directly uses SPARQL for precise KG subgraph retrieval. In scenarios requiring strict constraint satisfaction, structured retrieval is vastly superior to fuzzy vector retrieval.
- **Practicality of Multi-LoRA**: The three tasks share one 3.8B model + three small adapters, yielding extremely low deployment costs. This serves as an excellent example for multi-task LLM deployment.
- **Automated benchmark construction**: Programmatically generating QA pairs with constraints from the KG ensures the accuracy of the ground truth. This approach of "automatically generating training data from a knowledge base" has broad applicability.

## Limitations & Future Work
- The benchmark dataset is template-generated, lacking the natural language diversity of real user queries.
- Only FoodKG was utilized; generalization and transferability to other food knowledge bases have not been validated.
- The MAE of nutritional estimation is relatively high (9.38g), which may not be precise enough for scenarios requiring strict nutritional calculations.
- Inference requires iteratively traversing a large number of recipe subgraphs, which may pose efficiency challenges for large-scale recipe libraries.
- Multimodal inputs (e.g., food images) were not considered.

## Related Work & Insights
- **vs pFoodReq**: The embedding-based matching method obtains F1=0.637 while KERL achieves F1=0.854 (on their dataset), demonstrating that generative methods possess clear advantages in constraint reasoning.
- **vs LLaVA-Chef**: A multimodal model focused on recipe generation, whereas KERL shows advantages even in text-only recipe generation (BLEU-4: 0.079 vs 0.065) while covering recommendation + nutrition.
- **vs FoodGPT**: FoodGPT focuses purely on recipe generation, whereas KERL is a more integrated, all-in-one system for recommendation, recipe generation, and nutrition estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ A unified food recommendation system using KG+LLM Multi-LoRA, featuring a cohesive methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features a self-built benchmark, comparisons with multiple baselines, cross-tag analyses, and comparison against pFoodReq.
- Writing Quality: ⭐⭐⭐⭐ The system description is clear, though excessive mathematical notation slightly impacts readability.
- Value: ⭐⭐⭐⭐ Provides a comprehensive solution in the food recommendation domain; the Multi-LoRA paradigm is highly valuable for reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](../../NeurIPS2025/recommender/r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[ICLR 2026\] RPM: Reasoning-Level Personalization for Black-Box Large Language Models](../../ICLR2026/recommender/rpm_reasoning-level_personalization_for_black-box_large_language_models.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)

</div>

<!-- RELATED:END -->
