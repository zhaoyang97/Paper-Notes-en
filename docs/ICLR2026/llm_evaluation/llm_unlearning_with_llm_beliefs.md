---
title: >-
  [Paper Note] LLM Unlearning with LLM Beliefs
description: >-
  [ICLR 2026][LLM Evaluation][LLM unlearning] This paper reveals that LLM unlearning methods such as GA and NPO suffer from a *squeezing effect*—reducing the probability of a target response causes probability mass to redi…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "LLM unlearning"
  - "gradient ascent"
  - "squeezing effect"
  - "bootstrapping"
  - "model beliefs"
date: 2026-05-08
content_hash: 24004a0a086ed282
---

# LLM Unlearning with LLM Beliefs

**Conference**: ICLR 2026
**arXiv**: [2510.19422](https://arxiv.org/abs/2510.19422)
**Code**: [OpenUnlearning](https://github.com/locuslab/openunlearning)
**Area**: LLM Evaluation
**Keywords**: LLM unlearning, gradient ascent, squeezing effect, bootstrapping, model beliefs

## TL;DR

This paper reveals that LLM unlearning methods such as GA and NPO suffer from a *squeezing effect*—reducing the probability of a target response causes probability mass to redistribute toward semantically related high-likelihood regions, resulting in spurious unlearning. The authors propose a bootstrapping-based framework that leverages the model's own high-confidence predictions (model beliefs) as additional unlearning targets. Two instantiations, BS-T (token-level) and BS-S (sequence-level), achieve more thorough unlearning while preserving model utility across multiple benchmarks including TOFU, MUSE, and WMDP.

## Background & Motivation

**Necessity of LLM Unlearning**: Large language models trained on massive corpora inevitably memorize sensitive, harmful, or copyrighted content, posing risks of privacy leakage and harmful generation at deployment. LLM unlearning aims to remove such knowledge directly from model parameters via post-hoc adjustment, making it harder to circumvent than content filters or context-level defenses.

**Mainstream Methods and Their Limitations**: Current dominant approaches are based on gradient ascent (GA) and its variants (NPO, WGA, GradDiff), which minimize the log-likelihood of target responses to reduce their generation probability. GA, however, tends to severely degrade overall model performance. Subsequent improvements such as NPO (instance-level reweighting) and WGA (token-level reweighting) offer partial relief but do not address the fundamental problem.

**Spurious Unlearning**: The authors find that GA-based methods appear to successfully reduce the probability of target responses, yet the model continues to generate semantically related paraphrases that retain the original knowledge—a phenomenon termed *spurious unlearning*. For example, NPO achieves very low scores on TOFU (Probability 0.06, ROUGE-L 0.20), yet its generations still preserve key information such as "English."

**Misleading Evaluation Metrics**: Widely used metrics such as ROUGE, perplexity, and Truth Ratio cannot detect spurious unlearning and incorrectly report successful forgetting. This suggests that a substantial portion of "successful" unlearning reported in the literature may be spurious, exposing a systemic evaluation crisis in the field.

**Root Cause — The Squeezing Effect**: Softmax normalization constrains conditional probabilities to sum to one. When GA reduces $\pi_\theta(\mathbf{y}_u|\mathbf{x}_u)$, probability mass is inevitably redistributed to other candidate responses and concentrates in high-likelihood regions—which correspond precisely to semantically related paraphrases. This is the *squeezing effect*.

**Key Insight**: The locations to which probability mass "escapes" are exactly where the model itself is most confident—i.e., the model beliefs. Suppressing not only the original target but also these high-confidence predictions blocks the escape routes of probability mass and achieves genuine unlearning. This constitutes the core intuition behind the bootstrapping framework.

## Method

### Problem Formulation

Given a forget dataset $\mathcal{D}_u = \{(\mathbf{x}_u, \mathbf{y}_u)\}$ and a retain dataset $\mathcal{D}_r$, LLM unlearning pursues two objectives: (1) *Forgetting*: reduce the model's likelihood on $\mathcal{D}_u$ and its paraphrases; (2) *Retention*: keep the output distribution on $\mathcal{D}_r$ close to that of the original model.

### Diagnosing the Squeezing Effect

The authors verify the squeezing effect through two complementary experiments:

- **Semantic Similarity Distribution**: Beam search is used to sample from the original model; candidates are grouped into high/medium/low likelihood bins, and an LLM-as-Judge evaluates semantic similarity. The high-likelihood group is most semantically similar to the original target, and generations after NPO unlearning remain highly similar to this group.
- **Probability Dynamics Tracking**: The log-probability of each group is tracked throughout training. Both GA and NPO first increase the probability of the high-likelihood group before it slowly decreases (GA eventually collapses due to over-updating; NPO sustains the squeezing effect persistently).

### Bootstrapping Framework

#### BS-T (Token-Level Bootstrapping)

**Core Idea**: Suppress not only the target token but also its top-$k$ high-probability neighborhood. A soft target is constructed as:

$$\mathbf{t}_u^i = \lambda_{\text{BST}} \cdot \text{sg}\left[\pi_\theta(\cdot|\mathbf{x}_u, \mathbf{y}_u^{<i})\big|_{\mathcal{H}_k^{(i)}}\right] + (1-\lambda_{\text{BST}}) \cdot \mathbf{e}_{y_u^i}$$

where $\mathcal{H}_k^{(i)} = \text{Top-}k(\pi_\theta(\cdot|\mathbf{x}_u, \mathbf{y}_u^{<i}))$ is the set of top-$k$ high-likelihood tokens at position $i$, $\text{sg}$ denotes the stop-gradient operator, and $\lambda_{\text{BST}}$ controls the strength of neighborhood penalty.

The loss applies GA against this soft target:

$$\mathcal{L}_{\text{BST}}(\theta; \mathcal{D}_u) = \mathbb{E}_{\mathcal{D}_u}\left[\sum_{i=1}^{|\mathbf{y}_u|} \langle \mathbf{t}_u^i, \log\pi_\theta(\cdot|\mathbf{x}_u, \mathbf{y}_u^{<i})\rangle\right]$$

**Key Design Points**: (1) The stop-gradient operator prevents gradients from back-propagating through model predictions, ensuring training stability. (2) Although the mechanism resembles self-distillation, the objective is the opposite—erasing rather than reinforcing knowledge. (3) Temperature can be used to control the breadth of forgetting.

#### BS-S (Sequence-Level Bootstrapping)

**Core Idea**: Token-level suppression may not fully prevent the re-emergence of complete harmful continuations. BS-S samples high-confidence complete responses from the model and treats them as additional unlearning data:

$$\hat{\mathcal{D}}_u = \{(\mathbf{x}_u, \hat{\mathbf{y}}_u^{(j)})\}_{j=1}^N, \quad \hat{\mathbf{y}}_u^{(j)} \sim \pi_\theta(\cdot|\mathbf{x}_u)$$

The final objective is:

$$\mathcal{L}_{\text{BSS}} = (1-\lambda_{\text{BSS}}) \mathcal{L}(\theta; \mathcal{D}_u) + \lambda_{\text{BSS}} \mathcal{L}(\theta; \hat{\mathcal{D}}_u)$$

where $\mathcal{L}$ can be any unlearning loss (GA, NPO, or BS-T). BS-S can be applied off-policy (sampling once before training) or on-policy (periodic resampling during training).

#### Theoretical Analysis

Using the AKG learning dynamics framework, the authors prove:

- **GA residual**: $\mathcal{G}_{\text{GA}}^i = \pi^i - \mathbf{e}_{y_u^i}$, applying pressure only in the direction of the target token.
- **BS-T residual**: $\mathcal{G}_{\text{BST}}^i[v] = \mathcal{G}_{\text{GA}}^i[v] - \lambda \mathbf{q}^i[v]$, additionally applying repulsive force in the top-$k$ neighborhood directions.

Intuitively, GA's residual only "presses down" on the target token, allowing squeezed probability mass to form a new peak in the neighborhood; BS-T's residual simultaneously presses down on both the target and its neighborhood, preventing the formation of new peaks.

### Summary of Key Designs

| Design | Purpose | Mechanism |
|--------|---------|-----------|
| Soft target interpolation | Extend forgetting scope to high-probability neighborhood | One-hot + top-$k$ distribution interpolation |
| Stop-gradient | Training stability | Prevents gradients from back-propagating through model predictions |
| Sequence-level sampling | Cover complete harmful continuations | Sample high-confidence responses from the model |
| Compatibility with existing objectives | Generality | Can be combined with NPO/WGA/GradDiff |

## Key Experimental Results

### Table 1: TOFU Benchmark (Llama 3 Series, 10% Forget Setting, with Retain Regularization)

| Method | **1B** Agg.↑ | Mem.↑ | Util.↑ | **3B** Agg.↑ | Mem.↑ | Util.↑ | **8B** Agg.↑ | Mem.↑ | Util.↑ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Retrain | 0.64 | 0.58 | 0.71 | 0.65 | 0.57 | 0.75 | 0.65 | 0.57 | 0.75 |
| GradDiff | 0.52 | 0.49 | 0.56 | 0.49 | 0.47 | 0.52 | 0.50 | 0.45 | 0.55 |
| NPO | 0.58 | 0.58 | 0.58 | 0.62 | 0.58 | 0.66 | 0.63 | 0.57 | 0.70 |
| RMU | 0.58 | 0.59 | 0.57 | 0.55 | 0.44 | 0.74 | 0.62 | 0.55 | 0.72 |
| SimNPO | 0.47 | 0.35 | 0.70 | 0.41 | 0.28 | 0.74 | 0.29 | 0.18 | 0.72 |
| WGA | 0.53 | 0.47 | 0.62 | 0.51 | 0.42 | 0.66 | 0.52 | 0.41 | 0.70 |
| **BS-T** | 0.59 | 0.56 | 0.62 | 0.62 | 0.56 | 0.68 | 0.63 | 0.57 | 0.70 |
| **BS-S** | **0.61** | **0.59** | **0.63** | **0.63** | **0.58** | **0.70** | **0.64** | **0.58** | **0.71** |

### Table 2: WMDP Benchmark (Zephyr-7B-β)

| Method | Bio↓ | Cyber↓ | MMLU↑ |
|--------|:---:|:---:|:---:|
| Original | 0.64 | 0.45 | 0.58 |
| GradDiff | 0.27 | 0.28 | 0.43 |
| NPO | 0.27 | 0.30 | 0.44 |
| RMU | 0.29 | 0.27 | **0.55** |
| **BS-T** | **0.26** | 0.28 | 0.52 |
| **BS-S** | **0.26** | **0.27** | 0.54 |

### Table 3: MUSE-News Benchmark (Llama 2 7B-Chat)

| Method | VerbMem↓ | KnowMem↓ | UtilPres↑ |
|--------|:---:|:---:|:---:|
| Retrain | 0.2016 | 0.3170 | 0.5602 |
| NPO | 0.2914 | 0.3290 | 0.4651 |
| RMU | 0.3861 | 0.5088 | 0.4962 |
| **BS-T** | 0.2837 | 0.3278 | 0.4602 |
| **BS-S** | **0.2713** | **0.3250** | 0.4774 |

## Key Findings

1. **Spurious Unlearning Is Systemic, Not Incidental**: LLM-as-Judge evaluation reveals that NPO's high-likelihood generations on TOFU are semantically only slightly less similar to the original target than high-likelihood paraphrases, and far more similar than the medium-likelihood group, demonstrating that spurious unlearning is a systematic outcome of NPO rather than an occasional failure on specific samples.

2. **Probability Mass Is Persistently Squeezed Into High-Likelihood Regions**: During training, both GA and NPO cause the log-probability of the high-likelihood group to first rise before eventually declining (GA collapses catastrophically; NPO sustains the effect persistently), confirming that the squeezing effect not only exists but persists throughout training. GA "escapes" the problem by destroying the overall model; NPO continuously maintains a state of spurious unlearning.

3. **BS-T and BS-S Monotonically Suppress High-Likelihood Region Probabilities**: The probability dynamics of BS show that probabilities of both the target and its high-likelihood neighborhood decrease monotonically, directly confirming that the bootstrapping framework effectively mitigates the squeezing effect. LLM-as-Judge evaluation also shows that BS outperforms baselines on both Naturalness and Similarity dimensions.

4. **Sequence-Level Bootstrapping in BS-S Provides Significant Complementary Gains**: Ablation studies show that using BS-T as the underlying loss within BS-S achieves the best performance (Agg. 0.64), and the BS-S framework consistently improves over different underlying losses (GA/NPO/WGA), validating the general effectiveness of sequence-level bootstrapping.

5. **Unreliability of Conventional Evaluation Metrics**: GA causes model outputs to degenerate into meaningless repetition (e.g., repeated "always"), while ROUGE and similar metrics report scores near ~0 (appearing to indicate perfect unlearning), even though the model is completely unusable. This exposes a fundamental flaw in the evaluation framework that the LLM unlearning field has long relied upon.

## Highlights & Insights

- **Discovery and Naming of the Squeezing Effect**: The probability mass redistribution caused by softmax normalization is precisely characterized as the "squeezing effect," providing a unified mechanistic explanation for spurious unlearning in GA-based methods—concise, compelling, and empirically verifiable.
- **Elegant Exploitation of Model Beliefs**: The locations where probability mass escapes to are precisely where the model is most confident. "Using model beliefs against model beliefs" constitutes an elegant closed-loop design.
- **Introduction of LLM-as-Judge Evaluation**: The paper proposes an LLM-as-Judge evaluation framework along two dimensions—Naturalness and Similarity—which aligns more closely with human judgment than conventional metrics and has the potential to become a standard evaluation tool for the LLM unlearning field.
- **Unity of Theory and Practice**: The AKG learning dynamics framework formally demonstrates how BS-T reshapes the residual structure, with theoretical analysis closely consistent with empirical observations.

## Limitations & Future Work

- **Hyperparameter Sensitivity**: BS-T is sensitive to $\lambda_{\text{BST}}$ and BS-S to $\lambda_{\text{BSS}}$, requiring per-dataset and per-model tuning. Automated scheduling strategies remain undeveloped.
- **Computational Overhead of BS-S**: Sampling multiple complete responses from the model as additional unlearning data incurs non-trivial cost; with $N>5$, a single 80G GPU encounters OOM.
- **Lack of Theoretical Support for On-Policy BS-S**: The AKG framework is applicable only to the off-policy setting; in the on-policy case, the sampling distribution depends on the current parameters, preventing derivation under fixed-data assumptions.
- **Limited Model Scale**: Validation is restricted to models in the 1B–8B range; applicability to larger models (70B+) remains unknown.
- **Single-Turn Dialogue Setting**: The framework does not account for implicit knowledge leakage in multi-turn dialogues or circumvention of unlearning through prompt engineering.

## Related Work & Insights

| Dimension | **Ours (BS-T/BS-S)** | **NPO (Zhang et al., 2024)** | **WGA (Wang et al., 2025b)** |
|-----------|------|------|------|
| Core Mechanism | Suppress target + model high-confidence predictions | Instance-level DPO-style reweighted GA | Token-level weighted GA |
| Addresses Squeezing Effect | ✅ Directly counteracts high-likelihood neighborhood | ❌ Reweights only, does not expand forgetting scope | ❌ Only balances token contributions |
| Forgetting Granularity | Token-level + Sequence-level | Instance-level | Token-level |
| Spurious Unlearning | Effectively mitigated | Persistently present (Case 2) | Not explicitly addressed |
| Theoretical Basis | AKG residual analysis | DPO-inspired heuristic | GA conditional token formulation |
| Computational Cost | BS-T lightweight, BS-S moderate | Lightweight | Lightweight |
| TOFU-10% 1B Agg. | *0.61 (BS-S)* | 0.58 | 0.53 |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The discovery of the squeezing effect, the model beliefs perspective, and the bootstrapping framework design are all exceptionally insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — TOFU/MUSE/WMDP across three model families and multiple settings, dual evaluation with LLM-as-Judge and conventional metrics, comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain from motivation → diagnosis → solution → theory → experiments is tightly connected, with polished figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Code has been merged into OpenUnlearning; the framework is compatible with existing methods and has fundamental implications for both methodology and evaluation in the LLM unlearning field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](../../ACL2026/llm_evaluation/from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Model](predicting_llm_reasoning_performance_with_small_proxy_model.md)

</div>

<!-- RELATED:END -->
