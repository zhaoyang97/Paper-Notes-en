---
title: >-
  [Paper Note] Sample Efficient Demonstration Selection for In-Context Learning
description: >-
  [ICML 2025][LLM Evaluation][In-Context Learning] This paper proposes a sample-efficient demonstration selection method for in-context learning (ICL). Under limited annotation budgets, it efficiently selects the optimal combination of demonstrations, significantly improving the ICL performance of LLMs while dramatically reducing the required amount of labeled data.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "In-Context Learning"
  - "Demonstration Selection"
  - "Sample Efficiency"
  - "ICL"
date: 2026-05-08
content_hash: c505c731de66bb90
---

# Sample Efficient Demonstration Selection for In-Context Learning

**Conference**: ICML 2025  
**arXiv**: [2506.08607](https://arxiv.org/abs/2506.08607)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: In-Context Learning, Demonstration Selection, Sample Efficiency, ICL

## TL;DR
This paper proposes a sample-efficient demonstration selection method for in-context learning (ICL). Under limited annotation budgets, it efficiently selects the optimal combination of demonstrations, significantly improving the ICL performance of LLMs while dramatically reducing the required amount of labeled data.

## Background & Motivation
**Background**: In-context learning (ICL) is a core capability of LLMs, enabling them to perform new tasks given only a few examples in the prompt. However, ICL performance is highly sensitive to the choice of demonstrations, and selecting different examples can result in severe performance fluctuations.

**Limitations of Prior Work**: Existing demonstration selection methods often require large amounts of annotated data to evaluate candidate quality, or necessitate iterative testing of different combinations on a validation set, leading to high annotation costs.

**Key Challenge**: The core advantage of ICL lies in "few-shot" learning; however, selecting the optimal demonstrations themselves may require a large volume of annotated data, creating a paradox.

**Goal**: Efficiently select the optimal ICL demonstrations under an extremely limited annotation budget.

**Key Insight**: Utilize LLM internal signals (e.g., perplexity, confidence) to evaluate demonstration quality, reducing reliance on external annotations.

**Core Idea**: Model demonstration selection as a sample-efficient optimization problem, employing intelligent search strategies to locate high-quality demonstrations under a minimal evaluation budget.

## Method

### Overall Architecture
- **Input**: Candidate pool, small annotation budget, target task
- **Selection Process**: Efficient search for the optimal demonstration combination
- **Output**: Selected $k$-shot demonstration set

### Key Designs
1. **Sample-Efficient Evaluation Strategy**:
    - Utilizes internal signals of the model (e.g., token probability, perplexity) to estimate demonstration quality.
    - Avoids requiring full annotation-based evaluations for every candidate combination.
    - **Design Motivation**: Labeled data is a scarce resource, and its information utilization should be maximized.

2. **Intelligent Search Algorithm**:
    - Designs an efficient combinatorial search algorithm to avoid exhaustive search.
    - Potentially employs techniques such as Bayesian optimization and multi-armed bandit algorithms.
    - Approaches the optimal solution under an extremely small evaluation budget.
    - **Design Motivation**: The space of demonstration combinations is exponential, necessitating intelligent search.

3. **Cross-Task Generalization**:
    - The selection strategy potentially possesses cross-task transfer capabilities.
    - Selection preferences learned on one task can be generalized to similar tasks.
    - **Design Motivation**: To further reduce the annotation requirements for each new task.

### Loss & Training
No LLM training is involved; this is an inference-time demonstration selection method.

## Key Experimental Results

### Main Results

| Method | Annotation Budget | Average ICL Performance | Gain over Random |
|------|---------|-----------|-----------|
| Random Selection | 0 | Baseline | 0% |
| Full Evaluation | Large | Upper Bound | Highest |
| Ours | Small (~10%) | Close to Upper Bound | Significant Gain |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Without Internal Signals | Lower | Relies solely on external annotations |
| + Perplexity Signal | Improved | Leverages model feedback |
| + Intelligent Search | Further Improved | Efficient combinatorial optimization |

### Key Findings
- A minimal annotation budget (approx. 10% of the full set) is sufficient to achieve near-optimal demonstration selection performance.
- Model internal signals serve as effective indicators of demonstration quality.
- Combinatorial effects among demonstrations are crucial—individually high-quality examples do not necessarily form a high-quality combination.
- The method is consistently effective across various task types and model scales.

## Highlights & Insights
- Directly addresses the practical pain points of demonstration selection in ICL.
- Significantly reduces the application cost of ICL.
- Leveraging the model's own signals to guide selection is an elegant design.
- Offers direct practical value for deploying LLMs.

## Limitations & Future Work
- The quality of internal model signals may vary across different models.
- For highly novel tasks, the selection strategy might be less effective than on familiar domains.
- Can be integrated with Retrieval-Augmented Generation (RAG) for further extension.
- Dynamic demonstration selection (adapted to individual inputs) might yield better results.

## Related Work & Insights
- Directly compared with prior demonstration selection methods such as KATE and EPR.
- Shares conceptual similarities with active learning strategies.
- Insight: The "few-shot" advantage of ICL should be consistently maintained throughout the demonstration selection process.

## Rating
- Novelty: ⭐⭐⭐⭐ The sample-efficient selection strategy is a valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple tasks and models.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problem.
- Value: ⭐⭐⭐⭐ Directly beneficial for the practical application of ICL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In-Context Learning for Pure Exploration](../../ICLR2026/llm_evaluation/in-context_learning_for_pure_exploration.md)
- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](../../ICLR2026/llm_evaluation/detecting_data_contamination_in_llms_via_in-context_learning.md)
- [\[ICLR 2026\] DISCO: Diversifying Sample Condensation for Efficient Model Evaluation](../../ICLR2026/llm_evaluation/disco_diversifying_sample_condensation_for_efficient_model_evaluation.md)
- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](../../ICLR2026/llm_evaluation/in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)
- [\[ICML 2025\] Hyperband-based Bayesian Optimization for Black-box Prompt Selection](hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)

</div>

<!-- RELATED:END -->
