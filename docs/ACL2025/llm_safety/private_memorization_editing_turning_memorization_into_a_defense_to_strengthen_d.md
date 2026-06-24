---
title: >-
  [Paper Note] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models
description: >-
  [ACL 2025][LLM Safety][Privacy Protection] This paper proposes PME (Private Memorization Editing), which transforms the memorization tendency of LLMs from a security vulnerability into a defense mechanism. By editing the parameters of Feed Forward layers, it removes memorized personally identifiable information (PII), achieving privacy protection without retraining.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Privacy Protection"
  - "Model Editing"
  - "Training Data Extraction Attack"
  - "PII"
  - "Memorization"
date: 2026-05-08
content_hash: 4846aff48e04a4b9
---

# Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.10024](https://arxiv.org/abs/2506.10024)  
**Code**: [GitHub](https://github.com/elenasofia98/PME)  
**Area**: AI Safety  
**Keywords**: Privacy Protection, Model Editing, Training Data Extraction Attack, PII, Memorization

## TL;DR

This paper proposes PME (Private Memorization Editing), which transforms the memorization tendency of LLMs from a security vulnerability into a defense mechanism. By editing the parameters of Feed Forward layers, it removes memorized personally identifiable information (PII), achieving privacy protection without retraining.

## Background & Motivation

### 1. Background
As the parameter scale of LLMs increases, their ability to perform verbatim memorization of training data also strengthens. Training datasets often contain uncontrolled personally identifiable information (PII), such as email addresses, phone numbers, and credit card numbers, which can be extracted during inference via Training Data Extraction (TDE) attacks.

### 2. Limitations of Prior Work
- **Retraining is infeasible**: Once private information is discovered in the training collection, retraining LLMs from scratch is cost-prohibitive.
- **Limitations of existing editing methods**: Private Association Editing (PAE) protects privacy by severing the association between usernames and private information, but it does not directly address the leakage caused by verbatim memorization.
- **Imprecise localization**: Traditional model editing methods (such as MEMIT) pre-locate key layers through causal tracing, but this localization technique has been shown to be unreliable in guiding the editing process.

### 3. Key Challenge
The memorization capability of LLMs is the root cause of PII leakage. However, it also provides a clue—since attackers exploit memorization to extract PII, defenders can leverage the same memorized knowledge to accurately locate and eliminate the leakage.

### 4. Goal
To propose an efficient parameter editing method that directly edits memorized training samples, replacing PII with virtual placeholders (e.g., mail@domain.com) while minimizing the impact on the model's general language modeling capabilities.

### 5. Key Insight
Utilizing the property that Transformer computation can be decomposed into the sum of outputs from individual components, a geometric approach (projection) is used to estimate the contribution of each layer to PII generation. The layers are then edited proportionally to their contribution.

### 6. Core Idea

**Estimate the contribution of each layer to PII generation using geometric projection, and edit Feed Forward weights across all layers proportionally to replace memorized true PII with privacy-safe virtual PII.**

## Method

### Overall Architecture

The core workflow of PME is as follows:
1. Identify the set of memorized PII $\mathcal{S} = \{(p, t) | \text{s.t. } M(p) = t\}$ in the model through TDE attacks.
2. Optimize the target representation $x^*$ at the last layer $L$ to make the model generate the virtual PII $t^*$.
3. Estimate the contribution coefficient $w^l$ of each layer using geometric projection.
4. Compute the weight updates $\Delta^l$ across all layers proportionally to their contributions.

### Key Designs

#### Module 1: Additive Decomposition of Transformer Outputs

The representation of the final layer in a Transformer can be decomposed into the sum of the outputs of individual sub-components:

$$x_n^L = x_n + \sum_{l=1}^{L} a_n^l + \sum_{l=1}^{L} h_n^l$$

PME prioritizes the output of Feed Forward blocks $h_n^l$, which have been widely demonstrated to store information:

$$h_n^l = f\left((a_n^l + x_n^{l-1}) W_{in}^l\right) W_{out}^l$$

#### Module 2: Target Representation Optimization

Optimize the offset $\delta^*$ for the $L$-th layer using gradient descent to maximize the probability of generating the virtual PII $t^*$:

$$\delta^* = \arg\max_{\hat{\delta}} \mathcal{P}\left(t^* \mid \sigma\left((x_n^L + \hat{\delta}) W_U\right)\right)$$

The privacy-safe value is $x^* = x_n^L + \delta^*$.

#### Module 3: Estimating Layer Contribution via Geometric Projection

The core innovation of PME lies in using projection to estimate the contribution of each layer. The projection of the truncated sum $x_n^l \simeq \sum_{i=1}^l h_n^i$ onto the direction of $x_n^L$ is defined as:

$$w_p^l = \frac{x_n^l \cdot x_n^L}{\|x_n^L\|^2}$$

Normalizing yields the contribution coefficients:

$$w^l = \frac{w_p^l}{\sum_{i=1}^{L-1} w_p^i}$$

#### Module 4: Computing Weight Updates

The new values are allocated based on layer contributions: $v^* = w^l \cdot x^*$

The weight update matrix is computed via a closed-form solution:

$$\Delta^l = (V^* - V_0^*) {K^*}^T (K_0 K_0^T + K^* {K^*}^T)^{-1}$$

Final update: $\hat{W}_{out}^l = W_{out}^l + \Delta^l$

### Loss & Training
- Editing is based solely on 200-token-long memorized prompts.
- A Wikipedia subset is used to estimate the covariance matrix $K_0 K_0^T$.
- Layer contributions are estimated using a single forward pass, eliminating the extra computational overhead of causal tracing.

## Key Experimental Results

### Main Results: PII Leakage Reduction of PME vs. Baselines

**GPT-Neo 1.3B (Email, 200-token Memorization Attack):**

| Method | Original Leakage | Post-edit Leakage | Accuracy Drop |
|------|---------|----------|----------|
| Pre-edit | 179 | - | - |
| **PME** | 179→**0** | 0 | **100%** |
| MEMIT | 179→1 | 1 | 99.44% |
| GRACE | 179→0 | 0 | 100% |
| DeMem | 179→88 | 88 | 50.84% |

**GPT-J 6B (Email, 200-token Memorization Attack):**

| Method | Leakage | Δ Acc % |
|------|------|--------|
| PME | 1 | **99.65%** |
| MEMIT | 1 | 99.65% |

PME reduces PII leakage to 0 or near 0 in most configurations.

### Comparison Across PII Types

**GPT-Neo 2.7B:**

| PII Type | Original Leakage (200) | Post-PME Edit | Accuracy Drop |
|---------|-------------|----------|----------|
| Email | 286 | 1 | 99.65% |
| Phone | 34→1 (1.3B) | 1 | 97.06% |
| URL | 75→16 (1.3B) | 16 | 78.67% |

The defense performance on URLs is relatively weaker, as URL structures are more complex and associated with broader contexts.

### Key Findings
1. **PME editing performed in the most informative 200-token scenario can also defend against shorter context attacks (50/100-token) and association attacks.**
2. **Larger models memorize more**: GPT-J 6B memorizes significantly more PII than GPT-Neo 1.3B.
3. **Editing does not hurt general capabilities**: Post-edited models perform comparably to original models on general language modeling benchmarks.
4. PME requires only one additional forward pass to estimate layer contributions, making it more efficient than MEMIT's causal tracing.

## Highlights & Insights

1. **The concept of "using memorization against memorization" is highly ingenious**—transforming the memorization of LLMs, which is typically viewed as a vulnerability, into a defensive utility.
2. **Estimating layer contribution via geometric projection** outperforms traditional causal tracing: it is more efficient (single forward pass) and estimates layers independently for each sample rather than using a predefined set of layers.
3. **Rigorous experimental design**: Evaluating on models with fully open-sourced training data (GPT-Neo/GPT-J with The Pile) allows for precise assessment of privacy leakage.
4. **Flexible editing strategy**: Editing with only a 200-token prompt is sufficient to defend against attacks of various lengths and types.

## Limitations & Future Work

1. **Limited defense efficacy on URLs**: The leakage reduction for URLs is only around 79%, far lower than the almost 100% reduction for emails.
2. The maximum scale of the evaluated models is 6B; performance on larger models (70B+) remains unknown.
3. Access to training data is required to detect memorized PII, which limits applicability in third-party model auditing scenarios.
4. The virtual PII replacement strategy (e.g., mail@domain.com), while preserving semantics, may cause inconsistencies in certain downstream tasks.

## Related Work & Insights

- **MEMIT** (Meng et al., 2023): Serves as the editing foundation for PME, but PME improves upon its layer localization method.
- **PAE** (Venditti et al., 2024): Obfuscates context by severing name-PII associations, whereas PME directly edits memorized sequences.
- **Training Data Extraction** (Carlini et al., 2021; Huang et al., 2022): Represents the attack evaluation frameworks used in PME.
- **Insight**: Model editing techniques hold immense potential for privacy preservation. Especially for large models that cannot be retrained, "precise editing" is far more practical than "complete/unselective forgetting."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The conceptual paradigm shift of "memorization as defense" is highly novel, and estimating layer contribution via geometric projection showcases strong originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across 3 models, 3 PII categories, multiple attack scenarios, and 5 baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is clearly formulated with a natural transition from intuition to formal equations.
- **Value**: ⭐⭐⭐⭐⭐ — Privacy protection is a critical issue in LLM deployment; the proposed method is highly efficient and bypasses the need for retraining.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](../../ICML2025/llm_safety/watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[ACL 2025\] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)
- [\[ACL 2025\] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)

</div>

<!-- RELATED:END -->
