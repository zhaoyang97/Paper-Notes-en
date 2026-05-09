---
title: >-
  [Paper Note] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models
description: >-
  [ACL 2026][Model-agnostic explanation] This paper proposes a proxy-model-based black-box interpretability framework that leverages cheap small models to approximate the local decision boundaries of expensive large models for generating LIME/SHAP explanations. A statistical screen-and-apply mechanism ensures reliability: proxy explanations maintain over 90% fidelity while reducing costs by 88.2%, and are successfully applied to downstream optimization tasks such as prompt compression and poisoned sample removal.
tags:
  - ACL 2026
  - Model-agnostic explanation
  - proxy models
  - black-box interpretability
  - prompt compression
  - feature attribution
date: 2026-05-08
content_hash: c809fdf773bad16e
---

# Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models

**Conference**: ACL 2026
**arXiv**: [2505.12509](https://arxiv.org/abs/2505.12509)
**Code**: [https://github.com/outerform/Large-Model-Explanation-Benchmark](https://github.com/outerform/Large-Model-Explanation-Benchmark)
**Area**: Interpretability / LLM Optimization
**Keywords**: Model-agnostic explanation, proxy models, black-box interpretability, prompt compression, feature attribution

## TL;DR

This paper proposes a proxy-model-based black-box interpretability framework that leverages cheap small models to approximate the local decision boundaries of expensive large models for generating LIME/SHAP explanations. A statistical screen-and-apply mechanism ensures reliability: proxy explanations maintain over 90% fidelity while reducing costs by 88.2%, and are successfully applied to downstream optimization tasks such as prompt compression and poisoned sample removal.

## Background & Motivation

**Background**: Post-hoc explanations serve not only as transparency tools but also as drivers of model optimization (e.g., prompt debugging and data cleaning). However, closed-source models (e.g., GPT-4o, Google Gemini) block access to internal representations, making model-agnostic methods (e.g., LIME, SHAP) the only viable option. These methods rely on extensive perturbation sampling—generating a single LIME explanation typically requires 1,000 queries, and generating explanations over a validation set of 50 samples demands 50,000 queries, costing over \$100.

**Limitations of Prior Work**: (1) A cost–utility dilemma: the upfront cost of generating explanations exceeds the potential gains of the optimization tasks they are meant to support, rendering these powerful tools practically unusable; (2) existing acceleration methods (e.g., amortized explainers, feature reduction) are orthogonal to the proposed approach but fail to exploit inter-LLM homogeneity; (3) white-box explanation methods require access to internal model representations and are thus inapplicable to closed-source models.

**Key Challenge**: Model-agnostic explanations can theoretically guide LLM optimization, but their reliance on large numbers of queries to expensive models makes them practically infeasible—creating a fundamental utility dilemma in which explanation costs exceed optimization benefits.

**Goal**: (1) Propose an economically viable proxy explanation framework that replaces expensive models with cheap models for explanation generation; (2) ensure the reliability of proxy explanations through a statistical validation mechanism; (3) demonstrate the practical utility of proxy explanations in downstream optimization tasks.

**Key Insight**: The framework is grounded in inter-LLM homogeneity—different LLMs tend to exhibit similar behaviors on similar inputs, implying that small models can locally approximate the decision boundaries of large models ("inferring the whole from a part").

**Core Idea**: Use cheap local/open-source models as proxies to generate perturbation-based explanations. A two-level statistical screening mechanism (task-level + instance-level) ensures that proxy explanations are deployed only when they are sufficiently reliable, with fallback to the expensive oracle model otherwise.

## Method

### Overall Architecture

The proxy explanation framework comprises two key steps: (1) a **screening step**, which statistically verifies before deployment whether the proxy model's local decision boundary is aligned with the target model's; and (2) an **application step**, in which proxy explanations are used in place of expensive oracle explanations for instances that pass screening. Screening operates at two levels: offline task-level screening and online instance-level screening.

### Key Designs

1. **Task-Level Screening (Offline)**:

    - **Function**: A one-time evaluation of whether proxy model $f'$ can provide sufficiently faithful explanations for target model $f$ across the entire task/dataset $\mathbb{D}$.
    - **Mechanism**: A sequential one-sided paired $t$-test is used to verify whether the fidelity of proxy explanations reaches $\tau$ times ($\tau = 0.9$) that of oracle explanations, at confidence level $1 - \delta = 0.99$. For paired differences $d_i = q_{\text{proxy}}(\mathbf{x}_i) - \tau \cdot q_{\text{oracle}}(\mathbf{x}_i)$, the test evaluates $H_0: \mu_d < 0$ vs. $H_1: \mu_d \geq 0$. A confidence interval is computed; if the entire interval exceeds zero, $H_1$ is accepted; otherwise, sampling continues up to a maximum of $N = 50$ samples.
    - **Design Motivation**: Blindly employing a small model carries the risk of poor alignment; a statistical safeguard is needed to ensure proxy explanations are deployed only when their average fidelity is sufficiently high.

2. **Instance-Level Screening (Online)**:

    - **Function**: Runtime per-instance verification of whether the proxy model and the target model agree on the current input.
    - **Mechanism**: $s_{\text{inst}}(\mathbf{x}; f, f') = \mathbf{1}[f(\mathbf{x}) = f'(\mathbf{x})]$—proxy explanations are used only when both models produce the same prediction.
    - **Design Motivation**: (a) Local explanations are designed for the model's current prediction; they are inapplicable when the two models disagree. (b) Prediction disagreement signals divergent local decision behavior near $\mathbf{x}$, making proxy explanations more likely to be unfaithful. Disagreements trigger fallback to the oracle, ensuring fidelity.

3. **Proxy Explanation Application and Cost Reduction**:

    - **Function**: Generate explanations via the proxy model to substantially reduce cost.
    - **Mechanism**: The Cost Reduction Ratio is defined as $\text{CRR} = \frac{C_{\text{oracle}}}{C_{\text{proxy}} + C_{\text{fallback}} + C_{\text{screen}}}$, where $C_{\text{proxy}}$ is the proxy explanation cost (for consistent instances only), $C_{\text{fallback}}$ is the cost of falling back to the oracle (for inconsistent instances), and $C_{\text{screen}}$ is the screening cost. Running open-source models locally further reduces $C_{\text{proxy}}$ to near zero.
    - **Design Motivation**: The hybrid proxy-plus-fallback strategy maximizes cost savings while guaranteeing fidelity.

### Loss & Training

This paper does not involve model training. The proxy explanation framework uses existing LIME and Kernel SHAP methods to generate explanations, each with 1,000 perturbation samples and default hyperparameters. The 12 evaluated LLMs include the GPT-4o series, DeepSeek V3, seven Qwen 2.5 models, and two Llama 3.1 models.

## Key Experimental Results

### Main Results

**Cost Reduction Ratio (CRR) — Explaining Expensive LLMs with Proxy Models**

| Target Model | CRR Type | SST | MMLU | NQ |
|---|---|---|---|---|
| GPT-4o | CRR_max (API) | 10.33 | 4.84 | 7.41 |
| GPT-4o | CRR_local | 14.17 | 5.62 | 10.53 |
| DeepSeek V3 | CRR_local | 13.31 | 5.32 | 8.33 |
| Qwen 2.5 72B | CRR_local | 16.25 | 6.07 | 9.09 |

**Screening Step Reliability**

| Metric | LIME (SST/MMLU/NQ) | Kernel SHAP (SST/MMLU/NQ) |
|---|---|---|
| Precision | 100.0 / 99.4 / 94.1 | 100.0 / 100.0 / 100.0 |
| Recall | 80.2 / 77.6 / 76.1 | 96.3 / 97.2 / 96.2 |
| F1 | 89.0 / 87.2 / 84.2 | 98.1 / 98.5 / 98.0 |

### Ablation Study

**Prompt Compression Comparison (Compression Rate % on GPT-4o for Different Methods)**

| Method | MMLU-Chem | MMLU-CS | HellaSwag | GSM8K | PIQA |
|---|---|---|---|---|---|
| Random | 29.0 | 35.6 | 58.8 | 25.3 | 54.3 |
| AttnComp | 34.5 | 39.1 | 64.3 | 30.2 | 60.2 |
| LLMLingua | 38.7 | 38.3 | 62.7 | 28.9 | 58.7 |
| Proxy Exp. | 41.0 | 43.0 | 70.1 | 35.5 | 64.5 |
| Oracle Exp. | 49.2 | 50.2 | 75.5 | 37.2 | 69.2 |

**Poisoned Sample Removal (GPT-4o Accuracy %)**

| Task | Oracle Exp. | Proxy Exp. | Random Deletion |
|---|---|---|---|
| SST | 94.2 | 94.0 | 87.1 |
| HellaSwag | 93.7 | 93.5 | 88.4 |
| PIQA | 91.5 | 90.7 | 79.6 |

### Key Findings

- For the most expensive target model, GPT-4o, proxy explanations achieve up to 88% cost savings (CRR_local reaching 14.17) while maintaining over 90% fidelity.
- The screening step achieves an average precision of 98.9%, rarely marking unfaithful proxy explanations as usable; even in the rare case of a false positive, the actual proxy fidelity still exceeds 89%.
- Proxy explanations achieve 91.7% of oracle performance in prompt compression, substantially outperforming random deletion and state-of-the-art methods such as LLMLingua and AttnComp.
- Proxy explanations accurately identify and remove poisoned samples, recovering GPT-4o accuracy from below 80% to 94%, nearly matching oracle-level performance.
- Cross-model explanation transferability remains consistent across tasks and datasets; Qwen 7B/14B achieves over 90% fidelity when used as a proxy for GPT-4o.

## Highlights & Insights

- The paper transforms LLM homogeneity from a passive observation into an actively exploitable resource—the similarity of local decision boundaries across different LLMs becomes the foundation for reducing explanation costs.
- The two-level screen-and-apply mechanism balances safety and cost efficiency: task-level screening filters out unsuitable proxies in a single pass, while instance-level screening provides a per-sample safety valve.
- The work repositions interpretability from a passive observation tool to an active optimization primitive (prompt compression, data cleaning), substantially broadening the practical scope of explanation methods.

## Limitations & Future Work

- The framework focuses on perturbation-based feature attribution methods (LIME, SHAP); its applicability to other explanation techniques remains unexplored.
- In scenarios requiring extreme reasoning capabilities (e.g., complex symbolic logic), alignment between a small proxy and a large oracle may weaken, causing the framework to fall back to the oracle and reducing cost savings.
- The possibility of aligning proxy models to oracle models via lightweight fine-tuning is not explored.
- The dual-use risk of explanations themselves—the same tools could be exploited for adversarial attacks or to generate misleading explanations—is not addressed.

## Related Work & Insights

- **vs. LLMLingua/AttnComp**: These are dedicated prompt compression methods; proxy explanations substantially outperform them in compression rate, indicating that explanation-guided optimization is more effective than specialized heuristics.
- **vs. Amortized Explanation Methods**: Amortized methods train a unified explainer to approximate the explanation distribution; this approach is orthogonal to the proposed proxy framework and could be combined with it to achieve further cost reductions.
- **vs. White-Box Explanations**: White-box methods require access to internal model representations and are inapplicable to closed-source models; this paper achieves comparable results using model-agnostic methods with proxy models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Leveraging inter-LLM homogeneity to construct a proxy explanation framework is a novel perspective; the statistical screening mechanism is rigorously designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 12 LLMs, 7 datasets, two explanation methods, and two downstream tasks—an exceptionally comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem motivation is clearly articulated, the statistical framework is rigorous, and the experimental organization is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ Transforms black-box interpretability from "theoretically feasible but practically unusable" to "economically viable and practically useful"; the open-source benchmark dataset offers long-term value.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Interpretability from the Ground Up](interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[NeurIPS 2025\] OrdShap: Feature Position Importance for Sequential Black-Box Models](../../NeurIPS2025/interpretability/ordshap_feature_position_importance_for_sequential_black-box_models.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)

<!-- RELATED:END -->
