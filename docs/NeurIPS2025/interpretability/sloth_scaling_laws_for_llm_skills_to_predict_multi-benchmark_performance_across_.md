---
title: >-
  [Paper Note] Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families
description: >-
  [NeurIPS 2025][Interpretability][scaling laws] This paper proposes Skills Scaling Laws (Sloth), which assumes that LLM performance is driven by low-dimensional latent skills (e.g., reasoning, instruction following). By exploiting inter-benchmark correlations, Sloth constructs scaling laws that generalize across model families, enabling prediction of large-model performance on multiple benchmarks using only a small amount of family-specific data.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "scaling laws"
  - "LLM benchmarks"
  - "latent skills"
  - "factor analysis"
  - "performance prediction"
date: 2026-05-08
content_hash: 527612213c1bbeef
---

# Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families

**Conference**: NeurIPS 2025
**arXiv**: [2412.06540](https://arxiv.org/abs/2412.06540)  
**Code**: [https://github.com/felipemaiapolo/sloth](https://github.com/felipemaiapolo/sloth)  
**Area**: Interpretability
**Keywords**: scaling laws, LLM benchmarks, latent skills, factor analysis, performance prediction

## TL;DR
This paper proposes Skills Scaling Laws (Sloth), which assumes that LLM performance is driven by low-dimensional latent skills (e.g., reasoning, instruction following). By exploiting inter-benchmark correlations, Sloth constructs scaling laws that generalize across model families, enabling prediction of large-model performance on multiple benchmarks using only a small amount of family-specific data.

## Background & Motivation
**Background**: Traditional scaling laws (e.g., Chinchilla) predict loss rather than benchmark accuracy, and generalize poorly across model families.

**Limitations of Prior Work**: Intra-family scaling laws require training models of varying sizes (typically 3–5), which is costly; cross-family scaling laws suffer from poor accuracy.

**Key Challenge**: Ignoring family information leads to inaccuracy; incorporating family information introduces too many parameters, requiring large amounts of training data.

**Goal**: Predict the performance of larger models in a given family across multiple benchmarks using minimal family-specific data (even a single model).

**Key Insight**: Scores across different benchmarks are correlated, as they all reflect underlying "skills"; this correlation can be leveraged to reduce the number of parameters.

**Core Idea**: Scaling laws are formulated in a low-dimensional "skill space" rather than directly in benchmark space, with parameters shared via factor analysis.

## Method

### Overall Architecture
Input: scores of LLMs from multiple families on multiple benchmarks → extract $d$-dimensional latent skills via factor analysis → fit scaling laws for each skill as a function of model size $s$ and training token count $t$ → estimate efficiency parameters $\alpha$ for a new family → predict performance of larger models.

### Key Designs

1. **Latent Skill Decomposition**:

    - Function: Decompose scores on $J$ benchmarks into a linear combination of $d$ low-dimensional latent skills.
    - Mechanism: $\eta_i(s,t) = \Lambda \theta_i(s,t) + b$, where $\Lambda \in \mathbb{R}^{J \times d}$ is the factor loading matrix and $\theta_i$ is the skill vector for family $i$.
    - Design Motivation: Exploit inter-benchmark correlations to reduce parameter count and avoid overfitting.

2. **Cross-Family Skill Scaling Model**:

    - Function: Model how skills scale with computational resources.
    - Mechanism: $\theta_{ik}(s,t) = \alpha_{ik} + \beta_k^\top x(s,t)$, where $x = (\log s, \log t, \log s \cdot \log t)$; the slopes $\beta_k$ are **shared across families**, while the intercepts $\alpha_{ik}$ are **family-specific**.
    - Design Motivation: $\alpha_{ik}$ absorbs family-specific factors (data quality, post-training, etc.), while $\beta_k$ captures universal compute-to-skill relationships.

3. **Learnable Activation Function**:

    - Function: Replace the fixed sigmoid function with a monotone neural network.
    - Mechanism: $\sigma_j$ is a benchmark-specific monotonically increasing function with non-negative weight constraints.
    - Design Motivation: Different benchmarks exhibit different difficulty curve shapes, which a fixed sigmoid may fail to capture.

### Loss & Training
- Huber loss minimization is used to estimate the conditional median.
- Constrained optimization ensures $\gamma_j \in [0,1]$ (probability of a correct guess) and monotonicity of $\sigma_j$.
- The overall model is a lightweight neural network that can be fit in seconds on a laptop.

## Key Experimental Results

### Main Results — Prediction on 12 Benchmarks

| Method | MAE↓ | Note |
|--------|------|------|
| Owen et al. (no family info) | High | Family-agnostic |
| Ruan et al. (family-specific) | Medium | Requires existing large models |
| Sloth ($d=2$ skills) | **Lowest** | Requires only 1 small model |

### Ablation Study

| Configuration | Performance |
|---------------|-------------|
| $d=1$ (one-dimensional skill) | Better than baseline |
| $d=2$ (two-dimensional skill) | Best balance |
| $d=3+$ | Overfitting, no additional gain |
| Fixed sigmoid | Slightly worse than learned sigmoid |
| No interaction term | Reduced prediction accuracy |

### Key Findings
- Two latent skills suffice to capture variance across 12 benchmarks (analogous to the $g$-factor in IQ plus a second factor).
- Family efficiency parameters $\alpha$ vary substantially, explaining why models from different families with the same FLOPs exhibit large performance differences.
- The interaction term $\log s \cdot \log t$ is important, indicating that model size and data volume do not affect skills independently.
- The framework can predict the effects of test-time compute scaling.
- Compute-optimal skill scaling rules can be derived from the framework.

## Highlights & Insights
- **Elegant abstraction via skill space**: Reducing "model performance across benchmarks" to "which skills a model has mastered" not only improves prediction accuracy but also provides interpretable insights—e.g., revealing which benchmarks measure similar underlying skills.
- **Practical value of family efficiency parameters**: A single evaluated small model suffices to estimate the efficiency of a new family, enabling prediction of large-model performance—highly valuable for deciding whether to proceed with large-scale training.
- **Clever connection to economics**: The paper draws on the translog production function from stochastic frontier analysis, framing LLM training as "a production process that converts compute into skills."

## Limitations & Future Work
- There is no theoretical guidance for selecting the skill dimensionality $d$.
- The assumption that skill slopes are identical across families (differing only in intercepts) may be overly strong.
- Validation is limited to benchmarks from the Open LLM Leaderboard.
- Accuracy when extrapolating to very large models remains unknown.

## Related Work & Insights
- **vs. Chinchilla (Hoffmann et al.)**: That work predicts loss; this work predicts benchmark performance. That work requires training multiple models; this work leverages publicly available evaluation data.
- **vs. Ruan et al.**: Their approach assumes large models are already trained to predict performance on new benchmarks; this work assumes a small model is available to predict large-model performance.
- **Core Insight**: The low-rank structure of LLM evaluation suggests that a small number of carefully designed benchmarks may suffice for comprehensive model assessment.

## Rating
- Novelty: ⭐⭐⭐⭐ The latent-skill perspective on scaling laws constitutes a novel framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on 12 benchmarks across multiple families with comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; the connection to economic theory is elegant.
- Value: ⭐⭐⭐⭐⭐ Substantial practical value for LLM training decisions and evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](towards_scaling_laws_for_symbolic_regression.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[NeurIPS 2025\] AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)

</div>

<!-- RELATED:END -->
