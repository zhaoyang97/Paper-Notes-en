---
title: >-
  [Paper Note] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM
description: >-
  [NeurIPS 2025][LLM Safety][Data extraction attack] This paper reveals that even exact unlearning (retraining from scratch to remove data influence) is susceptible to privacy leakage. By exploiting the divergence between…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Data extraction attack"
  - "machine unlearning"
  - "LLM privacy"
  - "reversed model guidance"
  - "exact unlearning"
date: 2026-05-08
content_hash: 0adf1a3cfb8f23c4
---

# Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM

**Conference**: NeurIPS 2025
**arXiv**: [2505.24379](https://arxiv.org/abs/2505.24379)  
**Code**: [GitHub](https://github.com/Nicholas0228/unlearned_data_extraction_llm)  
**Area**: Medical Imaging
**Keywords**: Data extraction attack, machine unlearning, LLM privacy, reversed model guidance, exact unlearning

## TL;DR

This paper reveals that even exact unlearning (retraining from scratch to remove data influence) is susceptible to privacy leakage. By exploiting the divergence between model checkpoints before and after unlearning, an adversary can apply reversed model guidance with token filtering to substantially improve extraction success rates for deleted data—in some settings doubling the extraction rate.

## Background & Motivation

LLM training corpora may contain sensitive personal information (e.g., medical records). Regulations such as GDPR and CCPA grant users the right to be forgotten, making machine unlearning a practical necessity.

**Background**: Approximate unlearning methods such as gradient ascent and NPO are computationally efficient but have been shown to be vulnerable to attacks that recover unlearned data (Lucki et al.; Hu et al.) and provide no formal guarantees.

**Limitations of Prior Work**: Exact unlearning—retraining from scratch on the dataset with the target records removed—is widely regarded as the gold standard and is generally assumed to be immune to extraction and inversion attacks.

**Key Challenge**: In realistic deployment scenarios, model checkpoints or logits APIs from before unlearning are often accessible: snapshots of open-source models can be saved, and API users may have cached prior query results. This means an adversary can simultaneously access both the pre- and post-unlearning model versions.

The central finding is counterintuitive: **the unlearning operation itself increases the risk of information leakage**, because it provides the adversary with an additional signal—the behavioral difference between the two model versions precisely encodes distributional information about the deleted data.

## Method

### Overall Architecture

The adversary has access to checkpoints or logits APIs for both the pre-unlearning model $\theta$ and the post-unlearning model $\theta'$, along with a known prefix $x_{\leq i}$ for each target record (e.g., patient name, date of birth). The attack goal is to leverage the divergence between the two models to reconstruct the deleted data $X_0$. The core idea is to treat unlearning as the inverse of fine-tuning, and to construct a surrogate predictor that approximates the distribution of the deleted data via reversed model guidance.

### Key Designs

1. **Reversed Model Guidance**: The unlearning process is treated as the inverse of fine-tuning—$\theta$ can be viewed as $\theta'$ fine-tuned on $X_0$. Assuming a parametric approximation to re-learning the distribution of $X_0$ from $\theta'$, the guidance formulation is derived as:

    $$\log q(x_{i+1}|x_{\leq i}) = \log p_{\theta'}(x_{i+1}|x_{\leq i}) + w\bigl(\log p_\theta(x_{i+1}|x_{\leq i}) - \log p_{\theta'}(x_{i+1}|x_{\leq i})\bigr)$$

    where $w = 1/\lambda$ is the guidance scale. **Design Motivation**: Analogous to classifier guidance in diffusion models, the logits difference between the pre- and post-unlearning models serves as a guidance signal; tokens with larger divergence are more likely to belong to the distribution of the deleted data.

2. **Token Filtering Strategy**: Directly using the logits difference can produce incoherent generations. Borrowing from Contrastive Decoding, candidate tokens are restricted to those with sufficiently high probability under the pre-unlearning model $\theta$:

    $$V' = \{v \in V \mid p_\theta(v|x_{\leq i}) \geq \gamma \max_{v \in V} p_\theta(v|x_{\leq i})\}$$

    The token with the highest guided log-probability $\log q$ is then selected from $V'$. **Design Motivation**: The pre-unlearning model retains residual knowledge of $X_0$; constraining selection to its high-probability tokens suppresses low-frequency noise tokens and preserves generation coherence.

3. **Greedy Decoding Integration**: The final next-token selection is:

    $$x_{\text{next}} = \arg\max_{v \in V'} \log q(v|x_{\leq i})$$

    combining the directionality of the guidance signal with the quality control of token filtering.

### Loss & Training

- The attack method requires **no training**; only model inference is needed.
- Exact unlearning is implemented by fine-tuning on the full dataset to obtain $\theta$, then re-fine-tuning a pretrained model on the dataset with $X_0$ removed to obtain $\theta'$.
- Default configuration: forget set size 10%; guidance scale $w=2.0$ (Phi-1.5) / $w=1.4$ (Llama2-7B); filtering threshold $\gamma=10^{-5}$.
- Evaluation metrics: Rouge-L(R) and A-ESR (average extraction success rate, thresholds $\tau=1.0$ / $\tau=0.9$).

## Key Experimental Results

### Main Results

Extraction attack results on three standard benchmarks (10% forget set):

| Dataset | Model | Method | Rouge-L(R) | A-ESR$_{0.9}$ | A-ESR$_{1.0}$ |
|---|---|---|---|---|---|
| MUSE | Phi-1.5 | Pre-unlearning generation only | 0.473 | 0.114 | 0.101 |
| MUSE | Phi-1.5 | **Ours** | **0.606** | **0.249** (+118%) | **0.224** (+121%) |
| MUSE | Llama2-7b | Pre-unlearning generation only | 0.675 | 0.424 | 0.384 |
| MUSE | Llama2-7b | **Ours** | **0.744** | **0.496** (+17%) | **0.438** (+14%) |
| TOFU | Phi-1.5 | Pre-unlearning generation only | 0.566 | 0.100 | 0.070 |
| TOFU | Phi-1.5 | **Ours** | **0.643** | **0.202** (+102%) | **0.120** (+71%) |
| WMDP | Phi-1.5 | Pre-unlearning generation only | 0.429 | 0.079 | 0.069 |
| WMDP | Phi-1.5 | **Ours** | **0.567** | **0.218** (+175%) | **0.192** (+178%) |

### Ablation Study

| Configuration | Rouge-L(R) | A-ESR$_{1.0}$ | Note |
|---|---|---|---|
| Post-unlearning model generation only | 0.296 | 0.004 | Post-unlearning model yields nearly no extraction |
| Pre-unlearning model generation only | 0.473 | 0.101 | Baseline extraction capability |
| Guidance $w=1.0$ | ~0.52 | ~0.14 | Insufficient guidance |
| Guidance $w=2.0$ (optimal) | **0.606** | **0.224** | Best guidance strength |
| Guidance $w=4.0$ | ~0.55 | ~0.18 | Over-guidance causes degradation |
| $\gamma=10^{-1}$ (over-filtering) | ~0.50 | ~0.13 | Excessive filtering impairs guidance |
| $\gamma=10^{-5}$ (optimal) | **0.606** | **0.224** | Moderate filtering is optimal |
| $\gamma=0$ (no filtering) | ~0.56 | ~0.19 | No filtering slightly reduces performance |

### Key Findings

- **Medical Scenario Simulation**: On synthetic SOAP-format patient diagnostic records, the attack raises the exact-match extraction rate from 14% to **21%** (+50%), demonstrating a serious threat in real-world medical privacy settings.
- **Training Epoch Effect**: The higher the degree of memorization (more training epochs), the smaller the optimal guidance scale $w$—consistent with the theoretical derivation ($w = 1/\lambda$, where $\lambda$ increases with training).
- **Forget Set Size Has Limited Impact**: Extraction effectiveness is more dependent on instance-level memorization than on the overall size of the forgotten data.
- **Applicability to Approximate Unlearning**: The method is also effective against approximate unlearning methods such as GA and NPO, though the improvement diminishes as model utility degrades.

## Highlights & Insights

- **A Profound Paradox**: The unlearning operation itself becomes a source of privacy leakage signals, fundamentally challenging the security assumptions of the machine unlearning field.
- **Elegant Transfer of Diffusion Model Guidance**: The classifier-free guidance paradigm is creatively transposed from generative modeling to privacy attacks by treating unlearning as the inverse of fine-tuning.
- **Practical Threat Model**: The adversary requires only API-level access to logits (no model weights), substantially lowering the barrier to attack.
- **Warning for Unlearning Evaluation Standards**: Existing evaluations consider only the post-unlearning model; this work demonstrates that threat models must account for adversaries with access to historical checkpoints.

## Limitations & Future Work

- The attack assumes the adversary possesses data prefixes (e.g., patient names), which may not hold in all scenarios.
- Validation is currently limited to mid-sized models (Phi-1.5 and Llama2-7B); effectiveness on larger-scale models remains to be verified (preliminary results are provided in the appendix).
- The medical dataset is synthetic; memorization and unlearning behavior in real-world medical LLMs may differ.
- The guidance scale $w$ requires manual tuning; adaptive selection strategies remain to be developed.
- Effective defensive countermeasures are not thoroughly examined (only simple strategies such as data perturbation are briefly tested).

## Related Work & Insights

- **Relation to Carlini et al. (2021)**: This work extends data extraction attacks to the unlearning setting, exploiting the difference between two model versions as an additional signal.
- **Connection to Contrastive Decoding (Li et al., 2023)**: The constrained token selection strategy is borrowed from contrastive decoding, but the objective shifts from improving generation quality to improving attack efficacy.
- **Implications for LLM Privacy Governance**: Model version management requires stricter security policies; legacy model checkpoints should be treated as sensitive assets.
- **Warning for Medical AI**: In LLMs trained on patient data, naïve exact unlearning is insufficient to guarantee privacy.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reveals the privacy paradox of exact unlearning; the reversed guidance attack is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three standard benchmarks, medical simulation, ablation studies, and generalization analysis are included; defense experiments are somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, threat model is rigorously defined, and figures are effective (particularly Fig. 1's patient data extraction example).
- **Value**: ⭐⭐⭐⭐⭐ — Carries significant implications for LLM security and privacy governance, with particular urgency for the medical AI domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](../../ICLR2026/llm_safety/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[NeurIPS 2025\] On Optimal Steering to Achieve Exact Fairness](on_optimal_steering_to_achieve_exact_fairness.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)
- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[NeurIPS 2025\] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)

</div>

<!-- RELATED:END -->
