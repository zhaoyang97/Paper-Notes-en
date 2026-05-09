---
title: >-
  [Paper Note] IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration
description: >-
  [ACL 2026][Interpretable Decision-Making] This paper proposes IDEA, a framework that extracts LLM decision knowledge into an interpretable parametric model over semantic factors. An EM algorithm jointly learns the mapping from verbal probability expressions to numeric values and the decision parameters, enabling calibrated, editable, and interpretable LLM decision-making. IDEA with Qwen-3-32B achieves 78.6% average F1 across five datasets, surpassing DeepSeek R1 (68.1%) and GPT-5.2 (77.9%).
tags:
  - ACL 2026
  - Interpretable Decision-Making
  - Verbal Probability Calibration
  - EM Algorithm
  - Parameter Editing
  - Human-AI Collaboration
date: 2026-05-08
content_hash: 80759fd07e220ef9
---

# IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration

**Conference**: ACL 2026
**arXiv**: [2604.12573](https://arxiv.org/abs/2604.12573)
**Code**: [https://github.com/leonbig/IDEA](https://github.com/leonbig/IDEA)
**Area**: Interpretability / LLM Decision-Making
**Keywords**: Interpretable Decision-Making, Verbal Probability Calibration, EM Algorithm, Parameter Editing, Human-AI Collaboration

## TL;DR

This paper proposes IDEA, a framework that extracts LLM decision knowledge into an interpretable parametric model over semantic factors. An EM algorithm jointly learns the mapping from verbal probability expressions to numeric values and the decision parameters, enabling calibrated, editable, and interpretable LLM decision-making. IDEA with Qwen-3-32B achieves 78.6% average F1 across five datasets, surpassing DeepSeek R1 (68.1%) and GPT-5.2 (77.9%).

## Background & Motivation

**Background**: LLMs are increasingly deployed in automated decision-making scenarios, yet their application in high-stakes domains such as financial investment and loan approval remains constrained by a fundamental "trust deficit"—stakeholders cannot reliably verify, audit, or intervene in the decision process.

**Limitations of Prior Work**: Existing methods fall short along three dimensions: (1) probability estimates produced by LLMs are overconfident and poorly calibrated; (2) generated explanations tend to be post-hoc rationalizations rather than faithful reflections of internal reasoning; and (3) no quantitative framework exists for precisely incorporating expert knowledge into decisions—prompt-based instructions alone cannot guarantee compliant behavior. For instance, ranking and scoring the same options can yield inconsistent orderings, and explicit instructions to exclude a factor do not reliably prevent it from influencing predictions.

**Key Challenge**: There is a fundamental internal-external misalignment in LLMs. Logit-based methods conflate next-token confidence with decision uncertainty and remain black-box; DeLLMa relies on LLMs to directly produce precise numeric values, precisely what LLMs are unreliable at; BIRD assumes factor independence and uses a fixed verbal-to-numeric mapping, sacrificing calibration accuracy and the natural correlations among factors.

**Goal**: To construct a decision framework that simultaneously satisfies three properties—calibrated probability estimates, semantic interpretability, and quantitative human-AI collaboration (i.e., precise parameter editing).

**Key Insight**: The authors identify two key observations: (i) although LLMs cannot reliably produce precise numeric probabilities, they can generate decision-relevant factors from broad world knowledge; and (ii) LLMs express verbal probability phrases (e.g., "likely," "unlikely") more consistently than precise numeric values, because such phrases appear far more frequently in pretraining corpora.

**Core Idea**: Rather than making the LLM's internal reasoning process transparent, IDEA extracts its knowledge into an inherently transparent form—an interpretable parametric model over a semantic factor space—and employs an EM algorithm to jointly learn the verbal-to-numeric mapping and the decision parameters.

## Method

### Overall Architecture

IDEA decomposes the target probability $P(O_i|Q)$ into two separable components: a decision model $P(O_i|\mathbf{f})$ (mapping factor configurations to outcomes) and factor inference $P(\mathbf{f}|C)$ (inferring factor values from context). The pipeline consists of an offline training phase (factor identification → behavioral probing → EM joint estimation) and an online inference phase (factor determination → joint sampling → marginalization), with additional support for expert parameter editing.

### Key Designs

1. **EM Joint Estimation (Verbal-to-Numeric Calibration)**:

    - *Function*: Jointly learns the mapping from verbal probability expressions to numeric values and the decision model parameters.
    - *Mechanism*: Addresses a chicken-and-egg circularity—learning the decision model requires numeric labels, yet determining what a verbal expression represents numerically requires the decision model. The E-step computes the posterior expectation of each latent probability as a precision-weighted combination of the model prediction and the verbal mapping; the M-step updates the model parameters (MSE + ranking consistency loss + elastic net regularization) and the verbal mapping (with monotonicity constraints). The decision model is a logistic regression with interaction terms, where main effects directly quantify each factor's contribution.
    - *Design Motivation*: BIRD's use of a fixed mapping derived from the psychology literature causes an average F1 drop of −6.8%; joint learning adapts to the specific task and to the particular LLM's verbal usage patterns.

2. **Correlated Sampling for Factor Inference**:

    - *Function*: Preserves natural inter-factor correlations when marginalizing over uncertain factors.
    - *Mechanism*: Factors are partitioned into observed and uncertain sets. For uncertain factors, $T=50$ joint configurations are sampled from the LLM conditioned on the context (high temperature ensures diversity), and the final probability is computed via Monte Carlo estimation. This estimator is unbiased with standard error $O(1/\sqrt{T})$.
    - *Design Motivation*: Prior methods such as BIRD assume conditional independence among factors, ignoring natural correlations such as those between "high income" and "stable employment," leading to inaccurate marginalization estimates.

3. **Quantitative Parameter Editing**:

    - *Function*: Enables experts to edit the relative importance of factors with mathematical precision.
    - *Mechanism*: Average Marginal Effects (AME) translate logistic regression coefficients from log-odds space into intuitive probability-space changes. The framework supports both structural editing (adding or removing factors) and quantitative editing (solving a constrained optimization via sequential quadratic programming to satisfy expert-specified importance ratios while minimizing perturbation to other factors). For example, excluding the credit history factor requires only zeroing the corresponding coefficient, precisely shifting the approval probability from 21.6% to 52.3%.
    - *Design Motivation*: Prompt-based factor exclusion is highly unreliable (ERR only 0.06–0.43), whereas parameter editing achieves perfect factor exclusion (ERR = 1.00) with zero relative error.

### Loss & Training

The M-step model parameter update uses a composite loss: MSE reconstruction loss + ranking consistency hinge loss (ensuring that the probability assigned to "likely" exceeds that of "unlikely") + elastic net regularization (applied only to interaction terms; L1 induces sparsity, L2 ensures numerical stability). The verbal mapping is initialized with values from the psychology literature and iterated until the change in the Q-function falls below $10^{-4}$.

## Key Experimental Results

### Main Results

Binary decision accuracy is evaluated across five datasets (complex decisions: BIGDATA22 stock prediction, German Credit loan approval; reasoning: COMMON2SENSE, PLASMA, TODAY):

| Model | Method | Avg. F1 (5 datasets) | 3-Class Ranking Macro F1 |
|---|---|---|---|
| Qwen-3-32B | IDEA | **78.6%** | **0.693** |
| Qwen-3-32B | CoT | 67.7% | 0.339 |
| Qwen-3-32B | BIRD | 71.4% | 0.521 |
| GPT-5.2 | CoT | 77.9% | 0.402 |
| DeepSeek R1 | CoT | 68.1% | 0.286 |
| Qwen-3-8B | IDEA | 73.2% | 0.697 |
| Qwen-3-4B | IDEA | 71.6% | 0.504 |

### Ablation Study

| Configuration (Qwen-3-32B) | Avg. F1 | Ranking Macro F1 | Notes |
|---|---|---|---|
| IDEA (full) | 78.6% | 0.693 | Full model |
| w/o EM | 71.8% (−6.8%) | 0.632 | Fixed verbal mapping |
| w/o Inter | 71.0% (−7.6%) | 0.644 | No interaction terms |
| w/o MC | 71.8% (−6.8%) | 0.617 | Deterministic factor assignment |

### Key Findings

- The three modules contribute comparably, each accounting for approximately 6–8% of performance gain, confirming that EM calibration, interaction terms, and correlated sampling are all indispensable.
- IDEA achieves perfect factor exclusion (ERR = 1.00) and zero calibration error, whereas the best prompt-based method reaches only ERR = 0.43.
- IDEA substantially outperforms direct prompting even on smaller models (Qwen-3-4B), indicating that the framework is not sensitive to model scale.
- IDEA's advantage is more pronounced on ranking tasks, particularly in identifying the "equivalent" category, far surpassing other methods.

## Highlights & Insights

- **Leveraging the Consistency of Verbal Probabilities**: The framework cleverly exploits the observation that LLMs produce verbal expressions such as "likely/unlikely" more consistently than precise numbers, converting unreliable numeric outputs into reliable ordinal signals and learning the optimal numeric mapping via EM. This idea generalizes to any scenario requiring numeric information extraction from LLMs.
- **Mathematical Guarantees for Parameter Editing**: By combining AME with constrained optimization, IDEA achieves precise, predictable, and reversible behavioral interventions—the first framework to provide mathematical guarantees for human intervention in LLM decision-making. The "extract–model–edit" paradigm is transferable to other human-AI decision intervention settings.
- **Elegance of the Decoupled Design**: Decomposing decision-making into factor inference and a decision model as independent components leverages the LLM's breadth of world knowledge (factor generation) while circumventing its numerical unreliability (parametric modeling)—a principled design that plays to the LLM's strengths while compensating for its weaknesses.

## Limitations & Future Work

- The current formulation is restricted to binary decisions and binary factors; extending to multi-class decisions and continuous factors requires more complex parameterization.
- The factor completeness assumption may be difficult to satisfy in open-domain decision settings; omitting important factors will systematically degrade performance.
- EM guarantees convergence only to a local optimum, and sensitivity to initialization is not thoroughly discussed.
- The behavioral probing stage requires a large number of LLM queries (up to 256 configurations), which may be prohibitive in API-cost-sensitive settings.

## Related Work & Insights

- **vs. BIRD**: BIRD assumes factor independence and uses a fixed mapping; IDEA comprehensively outperforms it via EM joint learning and correlated sampling, with an average F1 improvement of approximately 7%.
- **vs. DeLLMa**: DeLLMa relies on LLMs to directly produce numeric utilities, precisely where LLMs are unreliable; IDEA sidesteps this issue by using verbal probability expressions as an intermediary.
- **vs. Concept Bottleneck Models**: CBMs require task-specific training and assume concept independence; IDEA provides comparable interpretability with the additional capability of parameter editing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Unifies verbal probability calibration, EM joint learning, and parameter editing into a single framework with a novel and practical approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets, three model scales, and complete ablations; validation on larger models and broader decision domains is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly developed, progressing logically from the trust deficit, with rigorous formalization.
- Value: ⭐⭐⭐⭐ Provides a practical interpretable and editable solution for LLM decision-making in high-stakes domains with genuine deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](../../NeurIPS2025/interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[ACL 2026\] HistLens: Mapping Idea Change across Concepts and Corpora](histlens_mapping_idea_change_across_concepts_and_corpora.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)

</div>

<!-- RELATED:END -->
