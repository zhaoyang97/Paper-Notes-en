---
title: >-
  [Paper Note] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes
description: >-
  [ACL 2025][LLM (Other)][math reasoning] Proposes MathNeuro, a computationally efficient method requiring only forward passes, which isolates parameters exclusive to mathematical reasoning in LLMs by filtering out parameters that are also important for general language tasks. Pruning these parameters removes mathematical capability, while scaling them enhances mathematical performance by 4-35%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "math reasoning"
  - "parameter importance"
  - "skill localization"
  - "pruning"
  - "neuron isolation"
date: 2026-05-08
content_hash: 36752f3a9d91baf0
---

# Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes

**Conference**: ACL 2025  
**arXiv**: [2410.16930](https://arxiv.org/abs/2410.16930)  
**Code**: [https://github.com/bryanchrist/MathNeuro](https://github.com/bryanchrist/MathNeuro)  
**Area**: LLM / Interpretability / Mathematical Reasoning  
**Keywords**: math reasoning, parameter importance, skill localization, pruning, neuron isolation  

## TL;DR

Proposes MathNeuro, a computationally efficient method requiring only forward passes, which isolates parameters exclusive to mathematical reasoning in LLMs by filtering out parameters that are also important for general language tasks. Pruning these parameters removes mathematical capability, while scaling them enhances mathematical performance by 4-35%.

## Background & Motivation

**Background:** Mathematical reasoning is a core capability in LLM research. However, there is very little research on how mathematical reasoning is encoded in model parameters and whether it can be localized and isolated. Existing skill/knowledge localization methods primarily focus on language-specific parameters or factual knowledge, without specifically investigating mathematical reasoning.

**Limitations of Prior Work:** (1) Gradient-based parameter importance methods (e.g., Panigrahi et al. 2023) are computationally expensive and infeasible for large models; (2) Forward-pass-based methods like Wanda (Sun et al. 2023) can identify parameters important for mathematics but fail to isolate math-specific parameters, as these parameters highly overlap with those important for other tasks; (3) LAPE (Tang et al. 2024) exhibits inconsistent performance across different models.

**Key Challenge:** Mathematical reasoning is deeply intertwined with natural language understanding rather than just involving computation, making math-specific parameters difficult to distinguish from general language parameters.

## Method

### Overall Architecture

MathNeuro consists of three steps: (1) calculating the importance score of each parameter using mathematical and non-mathematical data separately; (2) taking the top-$K\%$ most important parameters for each; (3) taking the set difference between the mathematical top-$K$ and non-mathematical top-$K$ as the math-specific parameters.

### Key Designs

1. **Parameter Importance Calculation Based on Weight $\times$ Activation:** Following the core idea of Wanda, the importance score is calculated for each parameter $(i,j)$ as $S_{ij} = |W_{ij}| \cdot \|X_j\|_2$, taking into account both weight magnitude and activation strength. The scores are summed over $N$ samples to obtain a robust estimate. This requires no gradients, only forward passes.

2. **Task-Specific Filtering:** In both attention and MLP layers, importance scores are computed using mathematical data (GSM8K/MATH) and non-mathematical data (MMLU/RACE) separately. After taking their respective top-$K\%$ parameters, the set difference is computed: $T_{math} = \text{TopK}_{math} \setminus \text{TopK}_{non\text{-}math}$, filtering out parameters that are also crucial for general language.

3. **Data Efficiency:** Experiments show that using just a single mathematical sample and a single non-mathematical sample can effectively localize math-specific parameters. Although slightly less effective than using 500 samples, it still significantly outperforms the baseline.

## Key Experimental Results

### Pruning Experiments (Llama 3.2 1B IT, TopK=15%)

| Method | GSM8K Accuracy Change | RACE Accuracy Change | MMLU Accuracy Change |
|------|-----------------|----------------|----------------|
| MathNeuro (RACE) | Drops significantly to ~0% | Slight drop (≈ random pruning) | Slight drop |
| MathNeuro (MMLU) | Drops significantly to ~0% | Slight drop | Slight drop (≈ random pruning) |
| Wanda | Drops significantly | Drops significantly | Drops significantly |
| LAPE | Inconsistent | Inconsistent | Inconsistent |
| Random | Moderate drop | Moderate drop | Moderate drop |

### Scaling Experiments (Scaling Factor 1.1, TopK=5%)

| Model | Method | GSM8K Gain | Impact on Non-Math Tasks |
|------|------|-----------|---------------|
| Llama 3.2 1B IT | MathNeuro | +4-17% | No significant change |
| Gemma 2 2B IT | MathNeuro | +4-17% | No significant change |
| Llama 3.1 8B IT | MathNeuro (×1.01) | +4-17% | No significant change |
| Phi 1.5 (Pre-trained) | MathNeuro | +5-35% (MATH) | No significant change |

### Parameter Consistency Analysis

| Number of Samples | Overlap Rate of Parameters Identified in Two Independent Runs |
|--------|----------------------|
| 1 | ~70-80% |
| 10 | ~85-90% |
| 100 | ~95%+ |
| 500 | ~97%+ |

### Key Findings

- Parameters identified by MathNeuro account for only ~1.5-1.8% of the total model parameters, yet they carry almost all mathematical reasoning capability.
- Math-specific parameters are distributed relatively evenly across all decoder blocks, indicating that mathematical reasoning is encoded across the entire model rather than concentrated in specific layers.
- Mathematical parameters identified on GSM8K generalize well to unseen mathematical tasks such as MATH and EGSM.
- The model's degradation on non-mathematical tasks after pruning is comparable to random pruning, confirming the effectiveness of parameter isolation.

## Highlights & Insights

- The method is extremely simple, requiring only forward passes and set difference operations, without gradients or complex optimization.
- Highly data-efficient: even a single sample can localize math-specific parameters.
- Well-designed bidirectional verification: pruning deletes capabilities, while scaling enhances them, providing mutual validation.
- Discovers that mathematical reasoning parameters are uniformly distributed across all layers of the model, offering new insights into how LLMs encode skills.
- Consistently effective across five models of different sizes (1B-8B).

## Limitations & Future Work

- Validated only on 1B-8B scale models, without testing larger models (>8B).
- The scaling factor is empirically chosen (1.1 for small models, 1.01 for large models), lacking systematic hyperparameter search.
- The binary classification of "math vs. non-math" is oversimplified, as mathematical reasoning encompasses various sub-skills (arithmetic, algebra, geometry, etc.).
- Evaluation is primarily based on GSM8K/MATH, which may not represent all types of mathematical reasoning.
- The method is based on Wanda's weight $\times$ activation formula, thus lacking strong theoretical explanation.

## Related Work & Insights

- **Skill Localization:** Wanda (Sun et al. 2023) weight $\times$ activation-based pruning; LAPE (Tang et al. 2024) activation probability entropy-based language localization.
- **Mathematical Reasoning:** GSM8K (Cobbe et al. 2021), MATH (Hendrycks et al. 2021b) benchmarks; Hanna et al. 2023 analysis of addition and subtraction concept processing.
- **Knowledge Encoding:** Dai et al. 2022 gradient-based knowledge neurons; Suau et al. 2024 language-specific parameter intervention.
- **Model Pruning:** Chang et al. 2024 survey on parameter importance.

## Rating

| Metric | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Technical Depth | 7 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Practical Value | 7 |
| Overall Score | 7.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)

</div>

<!-- RELATED:END -->
