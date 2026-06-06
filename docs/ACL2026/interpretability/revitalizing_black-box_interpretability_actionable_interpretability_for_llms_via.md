---
title: >-
  [Paper Note] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models
description: >-
  [ACL 2026][Interpretability][Model-agnostic explanation] This paper proposes a proxy-based black-box interpretability framework that utilizes inexpensive small models to approximate the local decision boundaries of costl…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Model-agnostic explanation"
  - "Proxy models"
  - "Black-box interpretability"
  - "Prompt compression"
  - "Feature attribution"
date: 2026-05-08
content_hash: c3f3112db12a2d6f
---

# Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models

**Conference**: ACL 2026  
**arXiv**: [2505.12509](https://arxiv.org/abs/2505.12509)  
**Code**: [https://github.com/outerform/Large-Model-Explanation-Benchmark](https://github.com/outerform/Large-Model-Explanation-Benchmark)  
**Area**: Interpretability / LLM Optimization  
**Keywords**: Model-agnostic explanation, Proxy models, Black-box interpretability, Prompt compression, Feature attribution

## TL;DR

This paper proposes a proxy-based black-box interpretability framework that utilizes inexpensive small models to approximate the local decision boundaries of costly large models for generating LIME/SHAP explanations. By employing a screen-and-apply mechanism to ensure reliability, proxy explanations achieve over 90% fidelity while reducing costs by 88.2%, and are successfully applied to downstream optimization tasks such as prompt compression and poisoned sample removal.

## Background & Motivation

**Background**: Post-hoc explanations serve not only as transparency tools but also as drivers for model optimization (e.g., prompt debugging and data cleaning). However, closed-source models (e.g., GPT-4o, Google Gemini) block access to internal representations, making model-agnostic methods (e.g., LIME, SHAP) the only viable option. Yet, these methods rely on intensive perturbation sampling—generating a single LIME explanation typically requires 1,000 queries; generating explanations for a validation set of 50 samples requires 50,000 queries, costing over \$100.

**Limitations of Prior Work**: (1) Cost-utility dilemma: The upfront cost of generating explanations exceeds the potential benefits of the optimization tasks, rendering these powerful tools impractical; (2) Existing acceleration methods (e.g., amortized explainers, feature reduction) are orthogonal to this work but fail to exploit the homogeneity among LLMs; (3) White-box methods require access to internal weights, which is infeasible for closed-source models.

**Key Challenge**: Model-agnostic explanations can theoretically guide LLM optimization, but their demand for massive queries to expensive models makes them practically unusable—creating a fundamental utility dilemma where the "cost of explanation exceeds the benefit of optimization."

**Goal**: (1) Propose an economically viable proxy explanation framework that replaces expensive models with cheap ones; (2) Ensure the reliability of proxy explanations via a statistical verification mechanism; (3) Demonstrate the practical utility of proxy explanations in downstream optimization tasks.

**Key Insight**: Based on LLM homogeneity—different LLMs tend to exhibit similar behaviors on similar inputs, implying that small models can approximate the local decision boundaries of large models ("seeing the large through the small").

**Core Idea**: Use inexpensive local/open-source models as proxies to generate perturbation-based explanations. A two-layer statistical screening (task-level + instance-level) ensures deployment only when proxy explanations are reliable, with a fallback to the expensive model otherwise.

## Method

### Overall Architecture

The proxy explanation framework consists of two key steps: (1) Screening—statistically verifying whether the proxy model's local decision boundary aligns with the target model's before deployment; (2) Application—generating explanations using the proxy model for instances that pass screening, replacing expensive oracle explanations. Screening is divided into offline task-level and online instance-level layers.

### Key Designs

1.  **Task-Level Screening (Offline)**:
    - **Function**: One-time evaluation of whether proxy model $f'$ can provide sufficiently faithful explanations for target model $f$ across an entire task/dataset $\mathbb{D}$.
    - **Mechanism**: Use a sequential one-sided paired $t$-test to verify if proxy fidelity reaches $\tau$ times the oracle fidelity ($\tau=0.9$) with confidence $1-\delta=0.99$. For paired differences $d_i = q_{\text{proxy}}(\mathbf{x}_i) - \tau \cdot q_{\text{oracle}}(\mathbf{x}_i)$, test $H_0: \mu_d < 0$ vs $H_1: \mu_d \geq 0$. If the confidence interval is entirely above zero, $H_1$ is accepted; otherwise, sampling continues up to $N=50$.
    - **Design Motivation**: Blindly using small models poses risks of poor alignment; a statistical safety valve is needed to ensure deployment only when the proxy is faithful on average.

2.  **Instance-Level Screening (Online)**:
    - **Function**: Per-instance verification during runtime to check if the proxy and target models are consistent for the current input.
    - **Mechanism**: $s_{\text{inst}}(\mathbf{x}; f, f') = \mathbf{1}[f(\mathbf{x}) = f'(\mathbf{x})]$, meaning proxy explanations are used only when both models yield the same prediction.
    - **Design Motivation**: (a) Local explanations are designed for the model's current prediction; they are invalid if predictions disagree; (b) Disagreement suggests differing local decision behaviors near $\mathbf{x}$. Falling back to the oracle during inconsistency guarantees fidelity.

3.  **Proxy Explanation Application & Cost Reduction**:
    - **Function**: Significant cost reduction by generating explanations via the proxy.
    - **Mechanism**: Cost Reduction Ratio $\text{CRR} = \frac{C_{\text{oracle}}}{C_{\text{proxy}} + C_{\text{fallback}} + C_{\text{screen}}}$, where $C_{\text{proxy}}$ is the cost for consistent instances, $C_{\text{fallback}}$ is the cost for oracle fallback, and $C_{\text{screen}}$ is the screening cost. Running open-source models locally reduces $C_{\text{proxy}}$ to nearly zero.
    - **Design Motivation**: A hybrid proxy-plus-fallback strategy maximizes savings while maintaining fidelity.

### Loss & Training

This paper does not involve model training. The framework uses existing LIME and Kernel SHAP methods to generate explanations, each using 1,000 perturbation samples with default hyperparameters. Evaluation involves 12 LLMs, including the GPT-4o series, DeepSeek V3, seven Qwen 2.5 models, and two Llama 3.1 models.

## Key Experimental Results

### Main Results

**Cost Reduction Ratio (CRR) — Explaining Expensive LLMs via Proxies**

| Target Model | CRR Type | SST | MMLU | NQ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | CRR_max (API) | 10.33 | 4.84 | 7.41 |
| GPT-4o | CRR_local | 14.17 | 5.62 | 10.53 |
| DeepSeek V3 | CRR_local | 13.31 | 5.32 | 8.33 |
| Qwen 2.5 72B | CRR_local | 16.25 | 6.07 | 9.09 |

**Reliability of Screening Steps**

| Metric | LIME (SST/MMLU/NQ) | Kernel SHAP (SST/MMLU/NQ) |
| :--- | :--- | :--- |
| Precision | 100.0 / 99.4 / 94.1 | 100.0 / 100.0 / 100.0 |
| Recall | 80.2 / 77.6 / 76.1 | 96.3 / 97.2 / 96.2 |
| F1 | 89.0 / 87.2 / 84.2 | 98.1 / 98.5 / 98.0 |

### Ablation Study

**Prompt Compression Comparison (Compression Rate % on GPT-4o)**

| Method | MMLU-Chem | MMLU-CS | HellaSwag | GSM8K | PIQA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random | 29.0 | 35.6 | 58.8 | 25.3 | 54.3 |
| AttnComp | 34.5 | 39.1 | 64.3 | 30.2 | 60.2 |
| LLMLingua | 38.7 | 38.3 | 62.7 | 28.9 | 58.7 |
| Proxy Exp. | 41.0 | 43.0 | 70.1 | 35.5 | 64.5 |
| Oracle Exp. | 49.2 | 50.2 | 75.5 | 37.2 | 69.2 |

**Poisoned Sample Removal (GPT-4o Accuracy %)**

| Task | Oracle Exp. | Proxy Exp. | Random deletion |
| :--- | :--- | :--- | :--- |
| SST | 94.2 | 94.0 | 87.1 |
| HellaSwag | 93.7 | 93.5 | 88.4 |
| PIQA | 91.5 | 90.7 | 79.6 |

### Key Findings

- For the most expensive model, GPT-4o, proxy explanations save up to 88% in costs (CRR_max reaches 14.17) while maintaining over 90% fidelity.
- The screening step achieves an average precision of 98.9%, rarely labeling unfaithful proxies as usable; even in rare false positives, actual proxy fidelity remains above 89%.
- Proxy explanations reach 91.7% of oracle performance in prompt compression, significantly outperforming random deletion and SOTA methods like LLMLingua/AttnComp.
- Proxy explanations accurately identify and remove poisoned samples, restoring GPT-4o accuracy from <80% to 94%, nearly matching oracle performance.
- Cross-model explanation transferability remains consistent across tasks; Qwen 7B/14B achieve over 90% fidelity for GPT-4o.

## Highlights & Insights

- Transforms LLM homogeneity from a passive observation into an active utility tool—leveraging the similarity in local decision boundaries to reduce explanation costs.
- The dual-layered "screen-and-apply" mechanism balances safety and cost-efficiency: task-level screening filters unqualified proxies once, while instance-level screening provides a per-sample safety valve.
- Shifts interpretability from a passive observation tool to an active optimization primitive (prompt compression, data cleaning), expanding the application boundaries of explanation methods.

## Limitations & Future Work

- Focused on perturbation-based feature attribution (LIME, SHAP); applicability to other interpretability techniques remains unexplored.
- In scenarios requiring extreme reasoning (e.g., complex symbolic logic), the alignment between small proxies and large oracles may weaken, causing the framework to fall back to the oracle and reduce cost savings.
- Does not explore the direction of using lightweight fine-tuning to align proxy models with oracles.
- Dual-use risks of explanations—the same tools could be used for adversarial attacks or generating misleading explanations.

## Related Work & Insights

- **vs LLMLingua/AttnComp**: These are specialized prompt compression methods; proxy explanations significantly outperform them in compression rates, indicating that explanation-guided optimization is more effective than specialized heuristics.
- **vs Amortized Explanations**: Amortized methods train a unified explainer to approximate the explanation distribution. This is orthogonal to the proxy approach and could be combined to further reduce costs.
- **vs White-box Interpretability**: White-box methods require internal representations and are inapplicable to closed-source models; this work achieves similar utility using model-agnostic methods via proxies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Building a proxy explanation framework via LLM homogeneity is a fresh perspective; the statistical screening design is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 LLMs, 7 datasets, two explanation methods, and two downstream tasks provide extremely comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, rigorous statistical framework, and well-organized experiments.
- **Value**: ⭐⭐⭐⭐⭐ Transitions black-box interpretability from "theoretically possible but practically unusable" to "economically viable and practically useful"; the open-source benchmark is of long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Interpretability Can Be Actionable](../../ICML2026/interpretability/interpretability_can_be_actionable.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)
- [\[ACL 2026\] Interpretability from the Ground Up](interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)

</div>

<!-- RELATED:END -->
