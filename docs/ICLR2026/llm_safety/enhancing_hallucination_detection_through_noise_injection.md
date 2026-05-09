---
title: >-
  [Paper Note] Enhancing Hallucination Detection through Noise Injection
description: >-
  [ICLR 2026][LLM Safety][Hallucination detection] Injecting uniform noise into MLP activations of intermediate LLM layers to approximate the Bayesian posterior, capturing epistemic uncertainty that is complementary to the aleatoric uncertainty captured by sampling temperature. This raises hallucination detection AUROC on GSM8K from 71.56 to 76.14.
tags:
  - ICLR 2026
  - LLM Safety
  - Hallucination detection
  - noise injection
  - epistemic uncertainty
  - Bayesian approximation
  - intermediate representations
date: 2026-05-08
content_hash: e331b1b11a4e6ab5
---

# Enhancing Hallucination Detection through Noise Injection

**Conference**: ICLR 2026
**arXiv**: [2502.03799](https://arxiv.org/abs/2502.03799)
**Code**: Not released
**Area**: LLM Safety
**Keywords**: Hallucination detection, noise injection, epistemic uncertainty, Bayesian approximation, intermediate representations

## TL;DR
Injecting uniform noise into MLP activations of intermediate LLM layers to approximate the Bayesian posterior, capturing epistemic uncertainty that is complementary to the aleatoric uncertainty captured by sampling temperature. This raises hallucination detection AUROC on GSM8K from 71.56 to 76.14.

## Background & Motivation

**Background**: Mainstream hallucination detection methods estimate LLM uncertainty via semantic entropy or multi-sample consistency, but these approaches primarily capture aleatoric uncertainty (uncertainty inherent in the data).

**Limitations of Prior Work**: Epistemic uncertainty — the model's uncertainty about its own knowledge — is largely ignored in existing methods. Standard sampling only varies the randomness of token distributions without altering the model itself, and thus cannot capture signals reflecting "what the model does not know it knows."

**Key Challenge**: Full Bayesian inference requires sampling from the posterior distribution over model weights, which is computationally intractable for large models, while existing approximations such as MC-Dropout are insufficiently effective.

**Goal**: How can epistemic uncertainty in large language models be efficiently captured without retraining?

**Key Insight**: Injecting small-magnitude noise into intermediate representations as a proxy distribution for the weight posterior.

**Core Idea**: Adding uniform noise to MLP activations is equivalent to applying small perturbations to the weights; the resulting variance across multiple samples reflects epistemic uncertainty.

## Method

### Overall Architecture
For a given input, a sampling temperature of $T=0.5$ is maintained (to capture aleatoric uncertainty), while $U(0, \alpha)$ noise is injected into the MLP activations of the top one-third of layers (to capture epistemic uncertainty). $K$ candidate responses are generated, and response entropy is computed as the uncertainty score.

### Key Designs

1. **Proxy Posterior Distribution**:

    - **Function**: Approximate the Bayesian weight posterior with parameterized noise.
    - **Mechanism**: A proxy distribution $q(\omega)$ is defined; weights of non-target layers are fixed at pre-trained values (delta distribution), while weights of target layers receive bounded perturbations around pre-trained values. The same noise vector is applied across all selected layers to avoid cancellation effects through residual connections.
    - **Design Motivation**: The noise magnitude $\alpha$ controls the "width" of the posterior — excessively large $\alpha$ degrades generation quality, while excessively small $\alpha$ fails to capture uncertainty. The optimal $\alpha$ lies in the range $[0.01, 0.11]$.

2. **Noise Injection Location**:

    - **Function**: Inject noise exclusively into MLP activation layers (top one-third of layers).
    - **Mechanism**: Empirical comparison of attention layers versus MLP layers shows MLP injection yields substantially better performance (76.14 vs. 71.89 AUROC).
    - **Design Motivation**: MLP layers encode more factual knowledge; perturbing them more effectively probes the model's certainty about specific knowledge.

3. **Detection Pipeline**:

    - Generate $K$ sampled responses for each input.
    - Compute response entropy: $H_{\text{ans}} = -\sum(p(a_j) \cdot \log(p(a_j)))$
    - High entropy indicates high uncertainty, signaling a potential hallucination.

## Key Experimental Results

### Main Results

| Dataset | Model | Baseline AUROC | +Noise AUROC | Gain |
|---------|-------|---------------|-------------|------|
| GSM8K | Llama-2-7B | 71.56 | **76.14** | +4.58 |
| GSM8K | Llama-2-13B | 77.20 | **79.25** | +2.05 |
| TriviaQA | Mistral-7B | 75.86 | **77.76** | +1.90 |
| CSQA | Gemma-2B | 58.97 | **61.71** | +2.74 |

### Ablation Study

| Setting | AUROC (GSM8K) |
|---------|--------------|
| Aleatoric only ($T=0.5$, no noise) | 71.56 |
| Epistemic only ($T=0$, with noise) | 74.35 |
| **Combined** | **76.14** |
| Noise in attention layers | 71.89 |

### Key Findings
- Epistemic and aleatoric uncertainty are complementary; their combination outperforms either individually.
- MLP layers are more suitable for noise injection than attention layers (76.14 vs. 71.89).
- All uncertainty metrics (predictive entropy, semantic entropy, lexical similarity, EigenScore) improve with noise injection.
- Larger models (13B vs. 7B) exhibit stronger baselines but smaller absolute gains from noise injection.

## Highlights & Insights
- **Simplicity and Generality**: Noise injection requires no retraining and no additional parameters, making it plug-and-play for any LLM.
- **Practical Bayesian Inference**: The approach reduces the theoretically elegant but practically intractable Bayesian inference to the minimal operation of adding noise, while retaining its theoretical motivation.
- **MLP vs. Attention Finding**: Empirical evidence that MLP layers are more sensitive to knowledge encoding supports the hypothesis that "MLPs serve as knowledge stores."

## Limitations & Future Work
- The optimal noise magnitude $\alpha$ is a dataset-dependent hyperparameter requiring tuning on a validation set.
- Multiple forward passes ($K$ samples) are required, leading to linearly increasing inference cost.
- Gains on CSQA are modest (+0.97), possibly due to task-specific characteristics.
- Theoretical guarantees for noise injection (e.g., distance to the true Bayesian posterior) have not been established.

## Related Work & Insights
- **vs. Semantic Entropy**: Semantic entropy captures only aleatoric uncertainty; noise injection additionally captures epistemic uncertainty.
- **vs. MC-Dropout**: Dropout is another Bayesian approximation method, but it is rarely used in large models and offers limited effectiveness.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of noise injection for hallucination detection is novel, though the technical contribution is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and datasets in combination with various uncertainty metrics.
- Writing Quality: ⭐⭐⭐⭐ The Bayesian framework is articulated clearly.
- Value: ⭐⭐⭐⭐ A plug-and-play enhancement for hallucination detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[ICLR 2026\] VeriTrail: Closed-Domain Hallucination Detection with Traceability](veritrail_closed-domain_hallucination_detection_with_traceability.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](unlearning_evaluation_through_subset_statistical_independence.md)

</div>

<!-- RELATED:END -->
