---
title: >-
  [Paper Note] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models
description: >-
  [NEURIPS2025][Causal Inference][Value alignment] This paper proposes the COUPLE framework, which constructs a Structural Causal Model (SCM) to model the dependencies and priorities among multi-dimensional values…
tags:
  - "NEURIPS2025"
  - "Causal Inference"
  - "Value alignment"
  - "counterfactual reasoning"
  - "structural causal model"
  - "pluralistic values"
  - "LLM alignment"
date: 2026-05-08
content_hash: b4b200078148908b
---

# Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models

**Conference**: NEURIPS2025
**arXiv**: [2510.18526](https://arxiv.org/abs/2510.18526)
**Code**: To be confirmed
**Area**: Causal Inference
**Keywords**: Value alignment, counterfactual reasoning, structural causal model, pluralistic values, LLM alignment

## TL;DR

This paper proposes the COUPLE framework, which constructs a Structural Causal Model (SCM) to model the dependencies and priorities among multi-dimensional values, and leverages counterfactual reasoning to achieve steerable alignment of LLMs toward arbitrary fine-grained pluralistic value objectives.

## Background & Motivation

**Background**: As LLMs increasingly serve users from diverse cultures, communities, and groups, aligning only to averaged principles such as "Helpful, Honest, and Harmless" (HHH) is insufficient; alignment with pluralistic human values is required.

**Multi-dimensionality of Values**: Research in psychology and social science (e.g., Schwartz's theory of basic human values) demonstrates that human values consist of multiple dimensions and their relative priorities, and different individuals arrive at markedly different judgments on the same issue due to differing value priorities.

**Challenge 1 — Value Complexity**: Existing methods treat multiple value dimensions as independent and equally important, ignoring their interdependencies and relative priorities.

**Challenge 2 — Value Steerability**: Value priorities are continuous and fine-grained; prompt-based methods struggle to precisely steer subtle value differences, while fine-tuning methods fail to generalize to unseen value combinations due to data sparsity.

**Limitations of Prior Work**: Prompt-based methods (role-playing, value prompting) and fine-tuning-based methods (CultureLLM, VIM, etc.) either neglect the structural relationships among values or are constrained by training data coverage.

**Core Motivation**: A framework is needed that can both model complex causal relationships among value dimensions and respond sensitively to fine-grained value variations, enabling interpretable pluralistic value alignment.

## Method

### Overall Architecture

COUPLE (**CO**unterfactual reasoning for pl**U**ralistic val**U**e a**L**ignm**E**nt) is a three-step inference-time alignment framework: (1) **Value Attribution** — inferring the value profile underlying a response; (2) **Value Intervention** — applying a $\text{do}$-intervention on value dimensions that deviate from the target; (3) **Counterfactual Prediction** — generating a new response conforming to the target value profile.

The core idea is to construct a **Structural Causal Model (SCM)** $(X, \mathcal{F}, \epsilon)$, treating the question $q$, value dimensions $v$, value concepts $c_v$, and the final response $r$ as endogenous variables, and modeling the causal relationship $V \to R$.

### Key Designs

#### Module 1: Value Attribution

Given a question $q$ and a response $r$, the value profile $v' = [(v_1, s_1'), \ldots, (v_d, s_d')]$ underlying the response is inferred, with each dimension scored on a 1–5 priority scale. Key design choices:

- **Joint Evaluation**: All value dimensions are presented simultaneously to the LLM and scored on a 5-point Likert scale, encouraging cross-dimensional joint reasoning and trade-off analysis.
- **Value Concept Extraction**: Key value concepts $C_r = [c_r^1, c_r^2, \ldots]$ are extracted from the response to remove surface-level noise and improve evaluation robustness.
- **Calibration of Scoring Criteria**: An iterative calibration strategy is adopted, combining a small amount of human-annotated data with paraphrasing, rephrasing, and prompt augmentation to refine scoring standards.
- **Exogenous Variable Estimation**: Non-value-related text in the response serves as $\epsilon_2$, and the $v' \to C_r$ relationship serves as a proxy for $\epsilon_1$.

#### Module 2: Counterfactual Value Concept Generation

When the deviation $\Delta(v', v) = \sum_i |s_i' - s_i|$ between the inferred value $v'$ and the target value $v$ exceeds threshold $\theta$, the intervention $\text{do}(V = v)$ is applied to generate counterfactual value concepts:

$$C_v = \arg\max_C P(\mathcal{C} \mid \text{do}(V=v), q, (v' \to C_r))$$

Key mechanisms:
- **Relation graph $\mathcal{G}$**: Models relationships among value dimensions (consistent, opposing, or unrelated).
- **Covariance matrix $\Sigma$**: Captures the relative importance of each dimension.
- Each value concept is computed as $c_v^i = \mathcal{F}(v_i, \mathcal{G}_{v_i}, \Sigma_{v_i}, q, (v' \to C_r))$.

#### Module 3: Final Response Generation

The counterfactual value concepts $C_v$ are aggregated and combined with exogenous variable $\epsilon_2$ (preserving original style and fluency), and a strong LLM generates the final aligned response $r_v = \mathcal{F}_r(\text{Pa}(r_v), \epsilon_2)$.

### Loss & Training

COUPLE supports two modes:
- **Inference-time alignment (prompt-based)**: A strong LLM (GPT-4.1-mini, DeepSeek-R1) directly executes the three-step pipeline.
- **Fine-tuning alignment (tuning-based)**: COUPLE generates training data to support Naive SFT (direct $(v, q, r_v)$ triplet training) and Reasoning SFT (training on complete counterfactual records including intermediate reasoning steps).

Evaluation metrics: MAE (absolute deviation) and Spearman rank correlation (priority trend).

## Key Experimental Results

### Main Results: Closed-Source LLMs on Two Datasets

| Method | GPT-4.1-mini MAE↓ | GPT-4.1-mini Corr↑ | DeepSeek-R1 MAE↓ | DeepSeek-R1 Corr↑ |
|---|---|---|---|---|
| Raw Model | 3.791 / 0.891 | 0.147 / 0.156 | 2.753 / 0.876 | 0.300 / 0.160 |
| Value Prompt | 2.182 / 0.505 | 0.620 / 0.611 | 1.720 / 0.425 | 0.708 / 0.729 |
| Tree of Thought | 1.975 / 0.461 | 0.752 / 0.663 | 1.753 / 0.368 | 0.698 / 0.783 |
| Plan and Solve | 2.158 / 0.500 | 0.618 / 0.632 | 2.027 / 0.307 | 0.548 / 0.845 |
| **COUPLE** | **1.433 / 0.355** | **0.778 / 0.848** | **1.082 / 0.123** | **0.798 / 0.928** |

> Values are reported as Touché23-ValueEval / DailyDilemma. COUPLE significantly outperforms all baselines across all settings.

### Ablation Study (GPT-4.1-mini, Touché23-ValueEval)

| Variant | MAE↓ | Correlation↑ |
|---|---|---|
| **COUPLE (Full)** | **1.433** | **0.778** |
| w/o SCM | 1.873 | 0.752 |
| w/o Value Concepts | 1.812 | 0.761 |
| w/o Counterfactual | 1.546 | 0.779 |
| w/o SCM & Counterfactual | 2.182 | 0.620 |

### Key Findings

1. **SCM is the most critical component**: Removing SCM increases MAE from 1.433 to 1.873 (+31%), indicating that structured causal modeling is essential for fine-grained reasoning.
2. **Reasoning LLMs are stronger**: DeepSeek-R1 outperforms GPT-4.1-mini across all methods, validating the importance of reasoning capacity for pluralistic value alignment.
3. **Reasoning SFT is optimal**: On open-source LLMs, Reasoning SFT (MAE=2.039, Corr=0.578) significantly outperforms Naive SFT and all culture-specific fine-tuning baselines, demonstrating that exposing intermediate reasoning steps enhances value sensitivity.
4. **Advantage grows with more value dimensions**: As the number of value dimensions increases from 1 to 5, baseline methods degrade notably while COUPLE remains stable, demonstrating its capacity to handle value complexity.
5. **Human evaluation**: In a human evaluation on 200 samples, COUPLE achieves a win rate of approximately 60%+ against Value Prompt and approximately 55%+ against Plan and Solve.

## Highlights & Insights

1. **Causal perspective on value alignment**: This work is the first to introduce SCMs and counterfactual reasoning into pluralistic value alignment, establishing a novel paradigm for LLM alignment.
2. **Elegant three-step pipeline**: The attribution → intervention → prediction counterfactual workflow is grounded in causal theory while remaining practically implementable.
3. **Value concepts as a bridge**: Introducing value concepts as an intermediate representation between high-level values and behavioral responses enhances both interpretability and robustness.
4. **Data augmentation capability**: The framework can synthesize training data for unseen value objectives, mitigating the data sparsity problem in fine-tuning-based approaches.
5. **Interpretability analysis**: High-frequency word analysis within value concepts (e.g., Security → stability, safety; Power → control, dominance) intuitively demonstrates the mapping from values to behavior.

## Limitations & Future Work

1. **Dependence on strong LLMs**: Inference-time alignment requires GPT-4-level LLMs to construct the SCM and execute counterfactual reasoning, incurring high computational costs.
2. **Multi-step inference latency**: The three-step pipeline introduces additional latency compared to direct prompting, making it unsuitable for real-time scenarios.
3. **Accuracy of value evaluator**: Automated evaluation relies on LLM-as-judge, which still falls short of human evaluation (80% agreement rate).
4. **Limited to 10- and 50-dimensional value systems**: Performance on higher-dimensional value systems (e.g., fine-grained moral frameworks) remains unverified.
5. **Absence of direct comparison with mainstream alignment methods** such as RLHF and DPO.

## Related Work & Insights

- **vs. CultureLLM/CulturePark**: These methods collect or synthesize data for specific cultures to fine-tune models, lacking generalization to arbitrary value objectives.
- **vs. Value Prompt/Role Prompt**: These methods directly inject value information but ignore inter-value interactions and cannot precisely control fine-grained priorities.
- **vs. VIM**: Aligns to target values by retrieving training samples; limited by data sparsity.
- **Inspiration from causal AI**: This work extends causal reasoning from traditional statistical problems to LLM behavior control, opening a new application direction for causal AI in alignment research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First application of SCM + counterfactual reasoning to pluralistic value alignment; conceptually novel with solid theoretical grounding)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers two datasets, multiple LLMs, human evaluation, ablation studies, and steerability analysis)
- Writing Quality: ⭐⭐⭐⭐ (Motivation is clear and framework diagrams are intuitive, though notation is dense)
- Value: ⭐⭐⭐⭐⭐ (Addresses a core challenge in pluralistic value alignment with high practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[NeurIPS 2025\] From Black-box to Causal-box: Towards Building More Interpretable Models](from_black-box_to_causal-box_towards_building_more_interpretable_models.md)
- [\[ICLR 2026\] Copy-Paste to Mitigate Large Language Model Hallucinations](../../ICLR2026/causal_inference/copy-paste_to_mitigate_large_language_model_hallucinations.md)

</div>

<!-- RELATED:END -->
