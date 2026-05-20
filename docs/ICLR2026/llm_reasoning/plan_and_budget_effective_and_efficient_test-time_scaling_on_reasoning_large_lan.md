---
title: >-
  [Paper Note] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs
description: >-
  [ICLR 2026][LLM Reasoning][Test-time scaling] This paper proposes the Plan-and-Budget framework, which decomposes complex queries into sub-problems and adaptively allocates token budgets based on estimated complexity…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Test-time scaling"
  - "reasoning efficiency"
  - "overthinking"
  - "token budget allocation"
  - "reasoning LLM"
date: 2026-05-08
content_hash: 3cd04c1e62b871bc
---

# Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs

**Conference**: ICLR 2026
**arXiv**: [2505.16122](https://arxiv.org/abs/2505.16122)  
**Code**: [github.com/junhongmit/P-and-B](https://github.com/junhongmit/P-and-B)  
**Area**: LLM Reasoning
**Keywords**: Test-time scaling, reasoning efficiency, overthinking, token budget allocation, reasoning LLM

## TL;DR

This paper proposes the Plan-and-Budget framework, which decomposes complex queries into sub-problems and adaptively allocates token budgets based on estimated complexity, achieving efficient test-time scaling for reasoning LLMs — with up to 70% accuracy improvement, 39% token reduction, and 193.8% gain on the E3 metric.

## Background & Motivation

Reasoning-oriented large language models (e.g., DeepSeek-R1, QwQ) have achieved remarkable success on complex tasks such as mathematical reasoning and code generation, yet computational efficiency during inference has become an increasingly pressing concern:

**Overthinking**: Many mainstream LLMs generate verbose and off-topic reasoning chains even for simple queries. Models "think too much," producing unnecessary intermediate steps that waste computational resources.

**Fixed-budget limitations**: Recent work attempts to mitigate overthinking by enforcing a fixed token budget, but this one-size-fits-all strategy leads to **underthinking** — for difficult problems, a fixed budget may be insufficient, resulting in incomplete reasoning.

**Query difficulty heterogeneity**: Real-world queries vary greatly in complexity. A simple arithmetic problem and a complex multi-step reasoning problem require vastly different computational resources, yet existing methods lack principled resource allocation mechanisms.

**Lack of theoretical foundation**: There is no formal theoretical framework for optimally allocating reasoning computation.

Through empirical analysis, the authors find that **inefficient reasoning typically stems from unclear problem-solving strategies** — models begin reasoning without an explicit plan, making them prone to drifting off course.

## Method

### Overall Architecture

Plan-and-Budget is a model-agnostic test-time framework consisting of three core steps:
1. **Plan**: Decompose the complex query into a series of sub-problems
2. **Estimate**: Assess the complexity of each sub-problem
3. **Budget**: Adaptively allocate token budgets to each sub-problem based on the complexity estimates

### Key Designs

1. **BAM: Budget Allocation Model**:

    - Function: Establishes a formal mathematical model of the reasoning process
    - Mechanism: Models the reasoning process as a sequence of sub-problems with varying uncertainty. Each sub-problem $q_i$ has an uncertainty parameter $u_i$, and the number of tokens required to resolve it is proportional to $u_i$
    - Under this model, the optimality of different budget allocation strategies is analyzed
    - Provides a theoretical result proving that **adaptive allocation is superior to uniform allocation**: sub-problems of greater difficulty should receive larger budgets
    - **Design Motivation**: Provides rigorous mathematical grounding for the intuitive principle of "think more on hard problems, less on easy ones"

2. **E3: Effective and Efficient Evaluation Metric**:

    - Function: Defines a composite metric that simultaneously measures correctness and computational efficiency
    - Mechanism: Captures the trade-off between accuracy and token consumption
    - $$E3 = \frac{\text{Accuracy}}{\text{Normalized Token Cost}}$$
    - A model that produces shorter and more precise reasoning chains achieves a higher E3 score
    - **Design Motivation**: Existing evaluations focus solely on accuracy or token count, lacking a unified measure

3. **Sub-problem Decomposition (Plan Stage)**:

    - Function: Uses the LLM itself or a lightweight auxiliary model to decompose the original query into multiple sub-problems
    - Mechanism: Prompts the model to "formulate a plan" before beginning reasoning, explicitly identifying which sub-tasks need to be resolved
    - Decomposition granularity is adaptively adjusted according to the structure of the original problem
    - Each sub-problem constitutes an independently solvable unit
    - **Design Motivation**: A clear plan prevents the reasoning process from drifting off course, effectively avoiding overthinking

4. **Adaptive Budget Scheduling (Budget Stage)**:

    - Function: Dynamically allocates token budgets based on the estimated complexity of each sub-problem
    - Mechanism: Employs an adaptive scheduling strategy; complexity estimates are derived from sub-problem features (length, keywords, problem type, etc.)
    - Simpler sub-problems receive smaller budgets; more complex ones receive larger budgets
    - A total budget cap can be set or determined automatically
    - In practice, the model is informed of the token limit for the current sub-problem during generation
    - **Design Motivation**: BAM theory demonstrates that adaptive allocation is optimal; in practice, this approach simultaneously avoids both overthinking and underthinking

### Loss & Training

- Plan-and-Budget is a **purely test-time** method and **requires no training or fine-tuning**
- The framework guides the LLM through carefully designed prompts for sub-problem decomposition and budget control
- It is model-architecture-agnostic and can be directly applied to any reasoning-oriented LLM
- Compatible with models of varying scales

## Key Experimental Results

### Main Results

| Task Type | Model | Method | Accuracy Change | Token Change | E3 Change |
|-----------|-------|--------|----------------|-------------|-----------|
| Math Reasoning | DS-Qwen-32B | Plan-and-Budget | +70% ↑ | −39% ↓ | +193.8% ↑ |
| Math Reasoning | DS-LLaMA-70B | Plan-and-Budget | Improved | Reduced | Significantly improved |
| Complex Reasoning | Multiple models | Plan-and-Budget | Consistent improvement | Consistent reduction | Comprehensive improvement |

**Key Finding Across Model Scales**: Plan-and-Budget enables a smaller model (DS-Qwen-32B) to reach the efficiency level of a larger model (DS-LLaMA-70B), demonstrating the ability to bridge performance gaps without retraining.

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Plan only (no budget control) | Accuracy improves; efficiency gain limited | Decomposition alone is beneficial |
| Budget only (no decomposition) | Efficiency improves; accuracy may decrease | Lacks structured guidance |
| Plan + uniform budget | Moderate improvement | Inferior to adaptive allocation |
| Plan + adaptive budget | Optimal | Full framework achieves best results |
| Different decomposition granularities | Medium granularity is optimal | Too fine increases overhead; too coarse loses structure |

### Key Findings

1. **Plan and Budget are both indispensable**: Decomposition addresses the direction of reasoning; budget allocation addresses resource efficiency; their combination yields the best outcome
2. **Small model + Plan-and-Budget ≈ Large model**: The framework can effectively compensate for differences in model scale
3. **Adaptive outperforms fixed**: Neither a large fixed budget nor a small fixed budget matches adaptive allocation
4. **Model-agnostic generalization**: The framework is effective across different reasoning LLMs

## Highlights & Insights

1. **Elegant integration of theory and practice**: The BAM theoretical model is established first; the optimality of adaptive allocation is derived theoretically before the Plan-and-Budget framework is designed — the approach is not purely heuristic
2. **Introduction of the E3 metric**: Fills the gap in reasoning efficiency evaluation and provides the community with a unified measurement standard
3. **Precise diagnosis of overthinking**: Empirical analysis identifies "lack of strategy" — rather than insufficient model capability — as the root cause of overthinking
4. **Efficiency pathway for smaller models**: Improves performance through computational efficiency rather than model scale, which is highly practical in resource-constrained scenarios
5. **Zero training cost**: A purely test-time method that is ready to use out of the box

## Limitations & Future Work

1. **Decomposition quality depends on LLM capability**: If the LLM's own decomposition ability is limited, the Plan stage may produce ill-formed sub-problems, degrading overall performance
2. **Accuracy of complexity estimation**: The effectiveness of adaptive budget allocation depends on the accuracy of complexity estimation, which is itself a difficult problem
3. **Additional prompt overhead**: Sub-problem decomposition in the Plan stage and guidance in the Budget stage require extra prompt tokens, which may not be worthwhile for very short, simple queries
4. **Inter-sub-problem dependencies**: The assumption of linear decomposition into independent sub-problems may be overly simplistic — in practice, sub-problems may exhibit complex interdependencies
5. **Integration with RL-based methods**: Plan-and-Budget could be combined with reinforcement learning-based reasoning optimization, but this direction is not explored in the paper

## Related Work & Insights

- **Test-time scaling**: Complementary to existing test-time methods such as Self-Consistency, Tree-of-Thought, and Best-of-N
- **Overthinking research**: Builds on pioneering work addressing reasoning redundancy, such as STILL and S1
- **Budget-aware reasoning**: Related techniques including token budget constraints and early stopping
- **Insights**: The BAM theoretical modeling approach can be generalized to other resource allocation scenarios (e.g., multimodal reasoning, multi-tool invocation)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The theoretical model makes a genuine contribution, though the engineering implementation of Plan-and-Budget is relatively straightforward
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple models and tasks, thorough ablations, and convincing results
- **Writing Quality**: ⭐⭐⭐⭐ — Theory and experiments are well integrated
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a practical efficiency bottleneck in reasoning LLMs; plug-and-play

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](../../ACL2026/llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)

</div>

<!-- RELATED:END -->
