---
title: >-
  [Paper Note] MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion
description: >-
  [ICML 2025 (AIW Workshop)][Causal Inference][Bias Mitigation] Proposes Multiperspective Fusion (MPF), a tuning-free, post-deployment alignment framework that guides LLMs to generate responses aligned with human baselines by decomposing baseline sentiment distributions into interpretable perspective components, thereby effectively mitigating model bias.
tags:
  - "ICML 2025 (AIW Workshop)"
  - "Causal Inference"
  - "Bias Mitigation"
  - "Post-training Alignment"
  - "Multi-perspective Fusion"
  - "Sentiment Distribution"
  - "KL Divergence"
  - "Calibration Error"
date: 2026-05-08
content_hash: fab8104f00cf3a16
---

# MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion

**Conference**: ICML 2025 (AIW Workshop)

**arXiv**: [2507.02595](https://arxiv.org/abs/2507.02595)

**Authors**: Xin Guan, PeiHsin Lin, Zekun Wu, Ze Wang, Ruibo Zhang, Emre Kazim, Adriano Koshiyama

**Area**: Causal Inference / LLM Alignment and Debiasing

**Keywords**: Bias Mitigation, Post-training Alignment, Multi-perspective Fusion, Sentiment Distribution, KL Divergence, Calibration Error

## TL;DR

Proposes Multiperspective Fusion (MPF), a tuning-free, post-deployment alignment framework that guides LLMs to generate responses aligned with human baselines by decomposing baseline sentiment distributions into interpretable perspective components, thereby effectively mitigating model bias.

## Background & Motivation

Large Language Models (LLMs) often exhibit systematic biases after deployment, such as displaying different sentiment tendencies toward candidates from specific university backgrounds in recruitment scenarios. Existing debiasing methods typically require:

**Fine-tuning**: Requires substantial computational resources and training data, and may introduce new biases.

**Complex prompt engineering**: Requires manual prompt design and is difficult to generalize.

**Retraining**: Infeasible for already deployed models.

The core motivation of MPF is: **Can post-processing achieve alignment with human expectation baselines without modifying model parameters?** The framework is built on the SAGED pipeline, an automated system used to construct bias benchmarks and extract interpretable baseline distributions.

## Method

### Overall Architecture

The MPF framework consists of three core modules:

1. **Baseline Distribution Extraction**: Obtains the baseline sentiment distribution from human experts (e.g., HR professionals) through the SAGED pipeline.
2. **Perspective Decomposition**: Decomposes the baseline distribution into multiple interpretable perspective components.
3. **Multi-perspective Fusion Generation**: Guides model generation through weighted sampling and balancing.

### Key Designs: Perspective Decomposition

Given the baseline distribution $P_{\text{baseline}}(s)$ (where $s$ represents sentiment), MPF decomposes it into $K$ perspective components:

$$P_{\text{baseline}}(s) = \sum_{k=1}^{K} \pi_k \cdot P_k(s)$$

where $\pi_k$ is the weight of the $k$-th perspective, satisfying $\sum_{k=1}^{K} \pi_k = 1$, and $P_k(s)$ is the sentiment distribution under the $k$-th perspective.

### Multi-perspective Fusion Generation

During the generation phase, MPF guides the output through the following steps:

1. Independently sample and generate response $r_k$ from each perspective $k$.
2. Perform weighted sampling based on the decomposed weights $\pi_k$.
3. The final output distribution is:

$$P_{\text{MPF}}(r) = \sum_{k=1}^{K} \pi_k \cdot P_{\text{LLM}}(r | \text{perspective}_k)$$

### Loss & Training

The optimization goal of MPF is to minimize the KL divergence between the output distribution and the target baseline:

$$\mathcal{L}_{\text{align}} = D_{\text{KL}}(P_{\text{MPF}} \| P_{\text{baseline}})$$

Simultaneously, a calibration error constraint is introduced:

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |{\text{acc}(b) - \text{conf}(b)}|$$

## Key Experimental Results

### Main Results: Alignment Performance with Different Baselines

The experiment was evaluated in a recruitment scenario, comparing the LLM's sentiment distributions toward candidates from different university backgrounds.

| Method | KL Divergence (↓) | Calibration Error ECE (↓) | Generalization to Unseen Questions |
|:---|:---:|:---:|:---:|
| Original LLM (No Intervention) | 0.85+ | High | N/A |
| Simple Prompt Debiasing | 0.40~0.60 | Medium | Poor |
| MPF + Counterfactual Baseline | **~0.05** | **Low** | **Yes** |
| MPF + HR Baseline | **~0.08** | **Low** | **Yes** |

### Alignment Results Across Different Baseline Types

| Baseline Type | Target Distribution Characteristics | KL after MPF Alignment | Sentiment Shift Correction |
|:---|:---|:---:|:---:|
| Counterfactual Baseline (Absolute Equality) | Uniform sentiment distribution across university backgrounds | ≤ 0.05 | Eliminate preference |
| HR Baseline (Top University Bias) | Retains real preferences of HR experts | ≤ 0.10 | Align with expert judgment |
| Random Baseline | Random distribution | ≤ 0.15 | Approach random |

### Key Findings

1. **Dual-baseline Validation**: MPF can simultaneously align with both the counterfactual baseline (absolute fairness) and the HR baseline (real-world preferences), demonstrating the framework's flexibility.
2. **Generalization to Unseen Questions**: On new questions outside the training set, MPF's alignment performance is preserved, with the KL divergence increase strictly under 0.03.
3. **No Fine-Tuning**: The entire process does not modify model parameters, achieved solely through multi-perspective sampling during inference.

## Highlights & Insights

- **Plug-and-Play**: Compatible with already deployed LLMs without requiring access to model weights or training data, achieving true post-deployment alignment.
- **Interpretability**: Perspective decomposition provides an interpretable analysis of bias—each perspective component corresponds to an identifiable opinion inclination.
- **Dual-Objective Design**: Supports both "absolute fairness" (counterfactual baseline) and "human-expected fairness" (HR baseline), offering flexibility in application scenarios.
- **SAGED Integration**: Utilizes an automated bias benchmark construction pipeline to reduce manual intervention.

## Limitations & Future Work

1. **Domain Limitation**: Experiments are validated only in recruitment scenarios (university bias). Generalization to other bias types (e.g., race, gender) remains to be explored.
2. **Workshop Paper**: As an AIW Workshop paper, the experimental scale and depth are relatively limited.
3. **Baseline Acquisition Cost**: Requires human experts to provide baseline distributions, which might incur high acquisition costs in practical applications.
4. **Computational Overhead**: Multi-perspective generation requires multiple sampling steps, which increases inference latency.
5. **Decomposition Quality**: The quality of perspective decomposition depends on the characteristics of the baseline distribution, which may perform poorly in complex multi-dimensional bias scenarios.

## Related Work & Insights

- **SAGED Pipeline**: Prior work of MPF that automatically builds bias benchmarks and extracts baseline distributions.
- **DPO / RLHF**: Alignment methods that require fine-tuning; MPF provides a parameter-free alternative.
- **Constitutional AI**: Constraints generation via rules; MPF achieves a similar goal through distribution alignment.
- **Inference-Time Intervention**: Similar to methods like Inference-Time Intervention (ITI), but MPF does not require access to internal representations.

**Insight**: Modeling bias mitigation as a distribution alignment problem, rather than a classification problem, may provide a new technological path for causal fairness research.

## Rating

| Dimension | Score (1-10) | Evaluation |
|:---|:---:|:---|
| Novelty | 6 | The idea of multi-perspective decomposition is novel, but the overall framework is relatively simple. |
| Technical Depth | 5 | As a Workshop paper, the theoretical analysis is not deep enough. |
| Experimental Thoroughness | 5 | Evaluated only on a single scenario, lacking large-scale comparative experiments. |
| Writing Quality | 7 | Clear expression with an intuitive description of the framework. |
| Value | 7 | High practical utility for a post-deployment solution without fine-tuning. |
| Overall Recommendation | 6 | Valuable research direction, but requires more validation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Social Debiasing for Fair Multi-modal LLMs](../../ICCV2025/causal_inference/social_debiasing_for_fair_multi-modal_llms.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)

</div>

<!-- RELATED:END -->
