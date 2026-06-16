---
title: >-
  [Paper Note] Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning
description: >-
  [ICLR 2026][LLM/NLP][systematic generalization] This paper introduces the Compositional-ARC dataset to evaluate systematic generalization in abstract spatial reasoning—specifically…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "systematic generalization"
  - "meta-learning for compositionality"
  - "ARC"
  - "abstract reasoning"
  - "few-shot learning"
date: 2026-05-08
content_hash: e9bdba42ff54322b
---

# Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning

**Conference**: ICLR 2026
**arXiv**: [2504.01445](https://arxiv.org/abs/2504.01445)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: systematic generalization, meta-learning for compositionality, ARC, abstract reasoning, few-shot learning

## TL;DR
This paper introduces the Compositional-ARC dataset to evaluate systematic generalization in abstract spatial reasoning—specifically, whether models can generalize from known primitive geometric transformations (e.g., translation, rotation) to unseen combinations thereof. A 5.7M-parameter encoder-decoder model trained with MLC achieves 78.26% exact match on the systematicity task, matching the ARC Prize 2024 champion (8B model + TTT) while vastly outperforming GPT-4o, o3-mini, and similar models (<3%).

## Background & Motivation

**Background**: Systematic generalization—the ability to automatically extend understanding of known components to novel combinations—is a hallmark of human cognition. Despite broad progress, LLMs consistently underperform on compositional generalization benchmarks.

**Limitations of Prior Work**: Lake & Baroni (2023) demonstrated human-level systematic generalization via Meta-Learning for Compositionality (MLC) on pseudo-linguistic tasks, yet whether MLC generalizes to non-linguistic domains such as spatial reasoning remains unexplored.

**Key Challenge**: State-of-the-art LLMs (including o3-mini and GPT-4o) excel at standard reasoning tasks but fail systematically when required to recombine primitive components into novel compositions—not due to insufficient capacity, but due to the absence of a training paradigm that enforces compositional generalization.

**Goal**: (1) Design a benchmark for evaluating systematic generalization in spatial reasoning; (2) Validate that MLC can be extended beyond the linguistic domain.

**Key Insight**: Leveraging the closure property of geometric transformations (the composition of two valid transformations remains a valid transformation), the paper constructs ARC-style 2D grid tasks to test whether models can infer unseen level-2 combinations from primitive transformations and level-1 combinations.

**Core Idea**: Extend MLC from language to visual-spatial reasoning, demonstrating that a small model with the right training paradigm can substantially outperform LLMs with ~1,400× more parameters on compositional generalization.

## Method

### Overall Architecture

1. **Dataset (Compositional-ARC)**: 2D objects on 10×10 grids → 5 primitive geometric transformations (translation/rotation/reflection/scaling/recoloring) → 3 types of indicators (shape/color/neighbor relation) selecting transformation targets → level-1 combinations (2 indicators) → level-2 combinations (3 indicators, TEST SET)
2. **Model Training (MLC)**: Transformer encoder-decoder (5.7M parameters) trained over 100K episodes, each with a distinct "visual interpretation grammar"
3. **Evaluation**: Given study examples of primitive transformations and level-1 combinations, predict unseen level-2 combinations

### Key Designs

1. **Compositional-ARC Dataset**:

    - Function: Evaluate a model's ability to generalize from known transformations to novel transformation compositions
    - Mechanism:
        - 5 primitive transformations: translation (right/down by 1 cell), rotation (±90°), reflection (horizontal/vertical), scaling (left/upward), recoloring (red/orange)
        - 3 indicator types: shape (e.g., L-shaped object undergoes translation), color (e.g., green object undergoes reflection), neighbor (e.g., object adjacent to a specific object undergoes scaling)
        - Compositional hierarchy: level-1 = 2-indicator combinations (e.g., shape + color → translation + reflection); level-2 = 3-indicator combinations (shape + color + neighbor → translation + reflection + scaling)
        - Systematicity test: study examples present only primitive transformations and level-1 combinations; models must infer unseen level-2 combinations
    - Design Motivation: The closure property of geometric transformations and the 2D grid representation of ARC provide a clean, controllable testbed for compositional generalization.

2. **MLC Extended to Spatial Reasoning**:

    - Function: Train a small model to infer visual interpretation grammars from few-shot examples and compose them
    - Mechanism: Each training episode uses a distinct "visual interpretation grammar" (e.g., "yellow object translates" in one episode may become "yellow object rotates" in another), forcing the model to infer grammars from study examples rather than memorizing fixed mappings.
    - Encoding: The 10×10 grid is divided into 2×2 patches (25 patches per grid), with each patch encoded as an embedding vector. 1D positional encoding marks grid-pair order; 2D positional encoding captures spatial structure.
    - Auxiliary task: During training, the model is also required to reproduce the outputs of study examples (copy task), strengthening comprehension of study examples.
    - Architecture: 3-layer encoder + 3-layer decoder, 8 attention heads, dim=128, FFN=768, GELU activation, 5.7M parameters total.

### Loss & Training

- Standard cross-entropy loss (predicting the patch sequence of output grids)
- Auxiliary copy task loss (reproducing study example outputs)
- Training over 100K episodes, each with a unique visual interpretation grammar
- Level-2 combinations are disjoint between training and testing (OOD evaluation)

## Key Experimental Results

### Main Results

Exact Match Accuracy on the Systematicity task:

| Model | Parameters | Exact Match (%) | Notes |
|-------|-----------|----------------|-------|
| GPT-4o | ~hundreds of B | 0.99 | General LLM |
| Gemini 2.0 Flash | ~hundreds of B | 2.66 | General LLM |
| o3-mini (low) | ~hundreds of B | 0.53 | General LLM (reasoning-enhanced) |
| Llama-3.2-3B-ReARC | 3B | 0.87 | ARC-specialized |
| Llama-3.2-3B-ReARC + TTT | 3B | 73.70 | + Test-time training |
| Mistral-8B-Full + TTT | 8B | 78.20 | ARC Prize 2024 Champion |
| **MLC (Ours)** | **5.7M** | **78.26** | Matches champion! |

### Ablation Study

| Configuration | Exact Match (%) | Notes |
|--------------|----------------|-------|
| MLC (full) | 86.73 ± 6.03 | Mean over 4 splits |
| − Copy task | 69.05 ± 9.23 | Auxiliary task is important |
| − Primitive transformation examples | 75.27 ± 12.95 | Moderate degradation |
| − Level-1 combination examples | 21.01 ± 19.07 | Severe collapse |
| MLC (more complex dataset) | 88.10 | Remains effective with more transformation types |

### Key Findings
- **5.7M << 8B, yet comparable performance**: The MLC-trained micro-model matches the ARC Prize 2024 champion (8B + TTT + extensive engineering) on systematic generalization, despite a 1,400× parameter gap.
- **Near-zero systematic generalization in general LLMs**: GPT-4o (0.99%) and o3-mini (0.53%) achieve 22% and 64% on 3-shot tasks respectively, but nearly completely fail on the Systematicity task requiring compositional generalization.
- **Level-1 combination examples are critical**: Removing level-1 examples drops accuracy from ~87% to 21%, indicating that intermediate-level compositional examples are essential for inferring higher-level combinations.
- **Copy task as a hidden gain**: The auxiliary copy task contributes ~18 percentage points of improvement by forcing deeper understanding of study examples.
- **Distinct error patterns**: LLMs primarily err by predicting wrong shapes or applying only primitive transformations; MLC errors are mainly minor shape deviations with rare degradation to primitive or level-1 transformations.

## Highlights & Insights
- **A compelling case for "right training paradigm > large model"**: The core difference between 5.7M and 8B parameters lies not in scale but in the MLC training strategy—a strong counterexample to the "scaling is all you need" narrative.
- **Successful transfer of MLC from language to vision**: MLC is shown to be not a language-specific trick but a general compositional generalization training paradigm, compelling models to learn rules rather than memorize mappings via dynamically varied grammars.
- **Elegant experimental design**: The hierarchical level-0/1/2 task structure is both clean and deep, enabling precise isolation and measurement of compositional generalization.

## Limitations & Future Work
- **Relatively simple tasks**: Only 5 primitive transformations, 10×10 grids, and 2 objects—still far from real-world spatial reasoning complexity.
- **Compositional depth limited to level-2**: Whether generalization holds for deeper combinations (3+ transformation compositions) remains untested.
- **Fixed grid size**: Generalization to grids of varying sizes is not evaluated.
- **Divergence from the original ARC dataset**: Compositional-ARC applies more regularized transformation rules than the original ARC, which features greater diversity and abstraction.

## Related Work & Insights
- **vs. Lake & Baroni (2023, original MLC)**: The original paper validates MLC on pseudo-linguistic tasks; this work is the first to extend it to visual-spatial reasoning, demonstrating MLC's generality.
- **vs. ARC Prize 2024 Champion (Franzen et al.)**: The champion employs an 8B model with a custom tokenizer, data augmentation, TTT, DFS search, and extensive engineering; the MLC model achieves parity with only 5.7M parameters and straightforward training.
- **vs. GPT-4o / o3-mini**: Exposes a fundamental deficit in state-of-the-art LLMs regarding compositional generalization—these models can perform pattern matching but cannot engage in systematic composition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First extension of MLC to visual-spatial reasoning; elegant dataset design; high-impact experimental findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model comparisons, 4-split validation, detailed ablations, error analysis, and complexity scaling experiments are all included.
- Writing Quality: ⭐⭐⭐⭐⭐ Richly illustrated, logically structured, with concepts introduced in a well-paced progression.
- Value: ⭐⭐⭐⭐⭐ Makes an important contribution to understanding systematic generalization in AI; the result of a 5.7M-parameter model outperforming GPT-4o is highly illuminating.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](../../AAAI2026/llm_nlp/collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[ICML 2026\] Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](../../ICML2026/llm_nlp/reasoning_on_the_manifold_bidirectional_consistency_for_self-verification_in_dif.md)

</div>

<!-- RELATED:END -->
