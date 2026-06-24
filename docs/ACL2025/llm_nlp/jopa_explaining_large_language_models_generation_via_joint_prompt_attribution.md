---
title: >-
  [Paper Note] JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution
description: >-
  [ACL 2025][LLM (Other)][prompt attribution] This work proposes the JoPA (Joint Prompt Attribution) framework, which models prompt attribution for LLM generation tasks as a combinatorial optimization problem. It utilizes a probabilistic search algorithm to efficiently find combinations of input tokens that causally impact the output, thereby addressing the limitation of existing methods that ignore cooperative effects among tokens.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "prompt attribution"
  - "interpretability"
  - "counterfactual"
  - "combinatorial optimization"
  - "generation explanation"
date: 2026-05-08
content_hash: b6748bbf719c3e8a
---

# JoPA: Explaining Large Language Model's Generation via Joint Prompt Attribution

**Conference**: ACL 2025  
**arXiv**: [2405.20404](https://arxiv.org/abs/2405.20404)  
**Code**: [https://github.com/yuruic/JoPA](https://github.com/yuruic/JoPA)  
**Area**: LLM/NLP  
**Keywords**: prompt attribution, interpretability, counterfactual, combinatorial optimization, generation explanation

## TL;DR
This work proposes the JoPA (Joint Prompt Attribution) framework, which models prompt attribution for LLM generation tasks as a combinatorial optimization problem. It utilizes a probabilistic search algorithm to efficiently find combinations of input tokens that causally impact the output, thereby addressing the limitation of existing methods that ignore cooperative effects among tokens.

## Background & Motivation

**Background**: Interpretability research in LLM generation mostly focuses on classification tasks and next-token prediction, with little work explaining "how input prompts affect the entire generated sequence."

**Limitations of Prior Work**: Methods like Captum remove tokens individually to measure their impact, ignoring semantic interactions between tokens—for example, removing "doctor" and "patient" individually has minimal impact, but removing them together as a combination has a significant impact.

**Key Challenge**: Exhaustively searching all token combinations is computationally infeasible for long inputs due to the exponential search space. How can key combinations be found efficiently?

**Goal**: To design an efficient algorithm to search for prompt token combinations in a discrete space that have the maximum causal impact on generation.

**Key Insight**: Counterfactual explanation—"how would the generation change if these tokens were removed?" The combination that triggers the greatest change is considered the most significant.

**Core Idea**: To formulate prompt attribution as a combinatorial optimization problem of masking tokens, and to solve it efficiently in a discrete space using a search algorithm guided by gradients and probabilistic updates.

## Method

### Overall Architecture
Learn a binary mask over the input token sequence -> Optimization objective: the masked tokens should maximize the change in generation probability -> Iterative optimization of the mask with a probabilistic search algorithm -> Output the most significant token combinations as the explanation.

### Key Designs

1. **Counterfactual Objective Function**

    - Maximize: the change in generation probability after masking a subset of tokens
    - Constraint: the number of masked tokens should be as small as possible (sparsity)
    - **Design Motivation**: To identify the smallest yet most influential subset of tokens.

2. **Probabilistic Search Algorithm**

    - Maintain a masking probability $p_i$ for each token position
    - Gradient information guides the direction of probability updates
    - An iterative process of sampling, evaluation, and updating
    - **Design Motivation**: To utilize gradients for search direction, while employing a probabilistic mechanism to balance exploration and exploitation in the discrete space.

3. **Generation Shift Metrics**

    - Comprehensive consideration of: change in generation probability, changes in word frequency, and changes in semantic similarity
    - **Design Motivation**: Multi-dimensional measurement of "how much the generation differs."

### Evaluation Metrics

| Metric | Aspect Measured | Description |
|------|---------|------|
| Probability Faithfulness | Generation probability change after masking | The larger, the better |
| Word Frequency Faithfulness | Output word frequency change after masking | The larger, the better |
| Semantic Faithfulness | Semantic similarity drop after masking | The larger, the better |
| Sparsity | Ratio of masked tokens | The smaller, the better |

## Key Experimental Results

### Main Results — Faithfulness Comparison Across Three Tasks

| Method | Summarization Task Prob Faithfulness | QA Task Prob Faithfulness | General Instruction Prob Faithfulness | Average Sparsity |
|------|-------------------|-----------------|-------------------|-----------|
| Random | Low | Low | Low | 10% |
| Captum (per-token) | Medium | Medium | Medium | 10% |
| Gradient saliency | Medium | Medium-High | Medium | 10% |
| **JoPA** | **High** | **High** | **High** | **8%** |

### Ablation Study

| Configuration | Probability Faithfulness | Description |
|------|-----------|------|
| JoPA (Full) | Highest | Gradient + Probabilistic Search |
| W/o Gradient Guidance | -15% | Pure probabilistic search has low efficiency |
| W/o Probabilistic Update | -10% | Pure gradient greedy search easily falls into local optima |
| Per-token (Captum) | -25% | Ignores combinatorial effects |

### Key Findings
- **JoPA consistently outperforms baselines across all three tasks**, requiring only about 8% of the tokens to be masked.
- **Combinatorial effects indeed exist**: Per-token methods miss 20-30% of important information.
- **The integration of gradient guidance and probabilistic search is critical**: The absence of either significantly degrades performance.
- **Explanations are applicable to safety analysis**: Identifying prompt segments that trigger harmful generations.
- **Explanations can also improve efficiency**: Removing non-essential tokens while preserving generation quality.

## Highlights & Insights
- **Formulating generation explanation as combinatorial optimization** is an elegant formulation—elevating interpretability from "computing importance scores" to "identifying causal subsets."
- The hybrid algorithm combining **probabilistic search and gradient guidance** acts as a general tool for discrete space optimization, which can be transferred to other combinatorial optimization scenarios.
- **The "doctor and patient" example** intuitively demonstrates why joint attribution is more accurate than independent attribution.

## Limitations & Future Work
- The search algorithm still incurs certain computational costs.
- Scalability to long inputs (>2K tokens) remains to be verified.
- Directions for improvement: Hierarchical search (coarse-to-fine), and comparison with attention-based analysis.

## Related Work & Insights
- **vs Captum**: Captum performs per-token attribution, whereas JoPA considers combinatorial effects.
- **vs LIME/SHAP**: These are attribution methods for classification tasks, whereas JoPA extends attribution to generation tasks.
- **vs CoT Self-Explanation**: CoT might be unfaithful, whereas JoPA guarantees causal faithfulness through counterfactual reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of combinatorial optimization formulation and generation task attribution is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks, multiple metrics, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear formulation.
- Value: ⭐⭐⭐⭐ Possesses practical value for LLM interpretability and safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition](songcomposer_llm_lyric_melody_generation.md)
- [\[ACL 2025\] APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts](appl_a_prompt_programming_language_for_harmonious_integration_of_programs_and_la.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] Attribution Methods in NLP: Navigating a Fragmented Landscape](attribution_methods_in_nlp_navigating_a_fragmented_landscape.md)

</div>

<!-- RELATED:END -->
