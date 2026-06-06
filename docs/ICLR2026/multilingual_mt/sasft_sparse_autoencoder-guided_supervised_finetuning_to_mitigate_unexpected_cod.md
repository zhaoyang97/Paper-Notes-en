---
title: >-
  [Paper Note] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs
description: >-
  [Multilingual & Machine Translation] This paper uses Sparse Autoencoders (SAEs) to identify that unexpected code-switching in LLMs is associated with abnormally high pre-activation values of target-language features…
tags:
  - "Multilingual & Machine Translation"
date: 2026-05-08
content_hash: 8bb2cefcb313b86a
---

# SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2507.14894](https://arxiv.org/abs/2507.14894)
- **Code**: [GitHub](https://github.com/Aatrox103/SASFT)
- **Area**: Natural Language Processing / Large Language Models
- **Keywords**: Code-Switching, Sparse Autoencoder, Multilingual LLMs, SFT, Language Features

## TL;DR

This paper uses Sparse Autoencoders (SAEs) to identify that unexpected code-switching in LLMs is associated with abnormally high pre-activation values of target-language features, and proposes SASFT, a method that constrains language feature pre-activation values during SFT training, reducing unexpected code-switching by over 50%.

## Background & Motivation

### State of the Field
Multilingual LLMs (e.g., Qwen-3, Llama-4, Gemma-3) frequently exhibit unexpected code-switching during generation—for instance, inserting Chinese or Korean text into responses to English queries—severely degrading user experience.

### Limitations of Prior Work

**The only known attempt**: Guo et al. (2025) address code-switching in DeepSeek-R1 using GRPO with a language consistency reward, but lack mechanistic analysis and achieve limited effectiveness.

**Lack of fundamental understanding**: Existing work has not deeply analyzed the internal mechanism of code-switching.

### Core Findings
SAE-based analysis reveals:
1. LLMs contain **language-specific features**—directions in the residual stream that exhibit large projection values only when processing tokens of a specific language;
2. When unexpected code-switching occurs, the **pre-activation values of target-language features increase abnormally**;
3. Ablation experiments confirm that suppressing these features reduces code-switching.

## Method

### Overall Architecture

SASFT proceeds in two steps: (1) identifying language-specific features in the LLM; (2) introducing an auxiliary loss during SFT training to constrain these features.

### 1. Sparse Autoencoder (SAE) Background

Given a residual stream activation $\mathbf{x} \in \mathbb{R}^N$, the SAE computes feature activations $\mathbf{a} \in \mathbb{R}^M$ ($M \gg N$):

$$
\mathbf{f(x)} = \mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}
$$

$$
\mathbf{a(x)} = \text{ReLU}(\mathbf{f(x)})
$$

This work focuses on the **pre-activation values** $\mathbf{f(x)}$ rather than the post-activation values $\mathbf{a(x)}$, as the latter discards meaningful negative pre-activation information.

### 2. Language-Specific Feature Identification

Following Deng et al. (2025), the monolinguality of each feature is measured. For feature $s$ and language $L$, the score is defined as:

$$
\nu_s^L = \mu_s^L - \gamma_s^L
$$

where $\mu_s^L$ is the average activation of the feature on language $L$, and $\gamma_s^L$ is its average activation on all other languages. Features with the highest $\nu$ values are treated as language-specific features.

### 3. Mechanistic Analysis of Code-Switching

**Key Finding 1**: Prior to a code-switching event, the pre-activation values of the target-language features gradually increase (Figure 3).

**Key Finding 2**: Directional ablation that subtracts the language feature direction reduces the code-switching rate (Figure 4):

$$
\mathbf{x}' \leftarrow \mathbf{x} - \lambda \mathbf{d}
$$

However, inference-time ablation has two drawbacks: (1) it requires substantially reducing pre-activation values, which may impair other capabilities; (2) it requires external intervention, increasing inference overhead.

### 4. SASFT Training Objective

An auxiliary loss is introduced during SFT to teach the LLM to autonomously maintain appropriate language feature pre-activation values:

$$
L_{\text{reduce}} = \mathbb{E}_{\mathcal{D}_j \sim \mathcal{D} \setminus \{\mathcal{D}_L\}}\left[\mathbb{E}_{\mathbf{x} \sim \mathcal{D}_j}\left[\sum_{s \in \mathcal{S}_L} \text{ReLU}(\mathbf{f}_s(\mathbf{x}) - \alpha_j)\right]\right]
$$

where $\mathcal{S}_L$ is the set of language-specific features for language $L$, and $\alpha_j$ is the estimated threshold of the mean pre-activation value.

The final training loss is:

$$
L_{\text{training}} = L_{\text{cross-entropy}} + \lambda L_{\text{reduce}}
$$

### Key Design Choices

- Target-language data $\mathcal{D}_L$ is excluded, as generating text in the target language does not constitute code-switching.
- The threshold $\alpha_j$ is not set to zero, since the mean pre-activation value may be negative.
- The loss can be applied across multiple layers for greater stability.

## Experiments

### Main Results: Code-Switching Rate Comparison

| Model | Method | CS→Chinese | CS→Russian | CS→Korean |
|-------|--------|-----------|-----------|----------|
| Gemma-2-2B | SFT (baseline) | 0.74% | 0.57% | 3.45% |
| | SFT+GRPO | 0.74 (0%) | 0.49 (-14%) | 3.44 (0%) |
| | SFT+Penalty | 0.67 (-10%) | 0.41 (-27%) | 1.18 (-66%) |
| | **SASFT** | **0.42 (-43%)** | **0.22 (-61%)** | **0.73 (-79%)** |
| Gemma-2-9B | SFT (baseline) | 0.78% | 0.12% | 0.81% |
| | **SASFT** | **0.41 (-47%)** | **0.01 (-94%)** | **0.13 (-84%)** |
| Llama-3.1-8B | SFT (baseline) | 1.16% | 0.67% | 0.57% |
| | **SASFT** | — | — | — |

### Ablation Study: Impact of Different Components

| Configuration | CS→Chinese (↓) | MMLU (↑) | HumanEval (↑) |
|--------------|---------------|---------|--------------|
| SFT baseline | 0.78% | 69.2 | 42.1 |
| SASFT (single layer) | 0.52% | 69.5 | 42.8 |
| SASFT (multi-layer) | **0.41%** | **69.8** | **43.2** |
| Inference-time ablation | 0.45% | 67.3 | 40.5 |

### Key Findings

1. **SASFT reduces code-switching by over 50% in most settings**, achieving complete elimination in Korean scenarios in some cases;
2. **Substantially outperforms GRPO**: GRPO yields near-zero improvement (0%) in most configurations, while SASFT consistently reduces code-switching;
3. **Multilingual capabilities are preserved**: Performance is maintained or improved across six benchmarks including MMLU, HumanEval, and Flores-200;
4. **Multi-layer application yields greater stability**: Cross-layer SASFT is more robust than its single-layer counterpart;
5. **Suppression is more effective than promotion**: Reducing non-target-language features is more effective than amplifying source-language features;
6. **Training-time constraint outperforms inference-time intervention**: SASFT modifies the model's internal behavior with no additional inference overhead.

## Highlights & Insights

- This is the first work to provide a rigorous mechanistic analysis of unexpected code-switching in LLMs, establishing a causal link to language feature pre-activation values.
- The translation from inference-time intervention to training-time constraint is elegant, directly addressing both drawbacks of the former approach.
- Strong generalizability is demonstrated across five models from three model families: Gemma-2, Llama-3.1, and Qwen-3.
- The auxiliary loss design is clean and principled, leveraging ReLU gating to penalize only pre-activation values that exceed the threshold.

## Limitations & Future Work

- The approach requires a pre-trained SAE for the target model (Qwen-3 necessitates training a custom SAE, and this overhead is not quantified).
- Language-specific feature identification depends on multilingual calibration data.
- The paper primarily evaluates on Chinese, Korean, and Russian; generalization to a broader set of languages remains to be verified.
- The code-switching definition relies on script-level detection, which may miss fine-grained lexical mixing.

## Related Work & Insights

- **Code-switching in LLMs**: Guo et al. (2025) identify and attempt to address code-switching in DeepSeek-R1 using GRPO.
- **SAE-based analysis**: Deng et al. (2025) discover language-specific features in LLMs.
- **Multilingual LLMs**: Qwen-3 (Yang et al., 2025), Llama-4 (Meta, 2025), Gemma-3 (Team et al., 2025).
- **Mechanistic interpretability**: Sparse autoencoders for understanding internal representations of LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to combine SAE-based interpretability with the code-switching problem, delivering a seamless pipeline from mechanism analysis to solution.
- Technical Depth: ⭐⭐⭐⭐ — A complete analysis–discovery–solution chain with a well-motivated pre-activation constraint design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage across 5 models × 3 languages × 6 benchmarks.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses a critical pain point in multilingual LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts](../../ACL2026/multilingual_mt/structure-guided_entity_resolution_fine-tuning_llms_for_robust_name_matching_in_.md)
- [\[ICLR 2026\] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](atlas_adaptive_transfer_scaling_laws_for_multilingual_pretraining_finetuning_and.md)
- [\[ACL 2026\] Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection](../../ACL2026/multilingual_mt/multilingual_steering_by_design_multilingual_sparse_autoencoders_and_principled_.md)
- [\[ICCV 2025\] SignRep: Enhancing Self-Supervised Sign Representations](../../ICCV2025/multilingual_mt/signrep_enhancing_self-supervised_sign_representations.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](../../NeurIPS2025/multilingual_mt/polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)

</div>

<!-- RELATED:END -->
