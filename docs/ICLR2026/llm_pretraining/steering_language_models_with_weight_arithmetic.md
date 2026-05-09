---
title: >-
  [Paper Note] Steering Language Models with Weight Arithmetic
description: >-
  [ICLR 2026][LLM Pretraining][weight steering] This paper proposes Contrastive Weight Steering, which extracts behavioral direction vectors from the weight difference between models fine-tuned on positive and negative behavioral data, and directly modifies model weights to control behavior. The method demonstrates superior generalization and consistency compared to Activation Steering across experiments on sycophancy, malicious behavior, and refusal.
tags:
  - ICLR 2026
  - LLM Pretraining
  - weight steering
  - activation steering
  - sycophancy
  - alignment
  - task vector
  - model safety
date: 2026-05-08
content_hash: 901645aa969d10a5
---

# Steering Language Models with Weight Arithmetic

**Conference**: ICLR 2026
**arXiv**: [2511.05408](https://arxiv.org/abs/2511.05408)
**Code**: [GitHub](https://github.com/safety-research/weight-steering)
**Area**: LLM Pretraining
**Keywords**: weight steering, activation steering, sycophancy, alignment, task vector, model safety

## TL;DR

This paper proposes Contrastive Weight Steering, which extracts behavioral direction vectors from the weight difference between models fine-tuned on positive and negative behavioral data, and directly modifies model weights to control behavior. The method demonstrates superior generalization and consistency compared to Activation Steering across experiments on sycophancy, malicious behavior, and refusal.

## Background & Motivation

The root cause of LLM alignment challenges is that RLHF and SFT require high-quality supervision over large-scale distributions to generalize reliably, while fine-tuning specific behaviors on narrow distributions risks catastrophic forgetting or the introduction of new alignment issues.

**Activation Steering** is the prevailing approach, controlling behavior by intervening on internal activations at inference time. However, it suffers from two limitations:
1. Limited generalization — performance degrades in out-of-distribution (OOD) settings
2. Limited expressivity — interventions are restricted to a single layer or a small subset of layers

The core problem this paper addresses is: **Can narrow-distribution data be used to reliably control LLM behavior via weight arithmetic?** Weight-space modifications are theoretically more expressive than activation-space interventions, as they can simultaneously affect behavior across all layers.

## Method

### Overall Architecture

Contrastive Weight Steering proceeds in three steps:

1. **Positive fine-tuning**: Fine-tune on data $D^+$ exhibiting the target behavior to obtain $\theta_{\text{positive}}$
2. **Negative fine-tuning**: Fine-tune on data $D^-$ exhibiting the opposite behavior to obtain $\theta_{\text{negative}}$
3. **Extract the weight steering vector** and apply it to the target model

### Key Designs

The **weight steering vector** is defined as the difference between the two fine-tuned models:

$$w_b = \tau^+ - \tau^- = \theta_{\text{positive}} - \theta_{\text{negative}}$$

where $\tau^+ = \theta_{\text{positive}} - \theta_{\text{pre}}$ and $\tau^- = \theta_{\text{negative}} - \theta_{\text{pre}}$.

The key intuition behind taking the difference is to **cancel out weight changes unrelated to the target behavior** (e.g., topic, style, length), thereby isolating the behavioral direction.

**Behavioral steering** is applied by scaling the vector:

$$\theta_{\text{steered}} = \theta_{\text{pre}} + k \cdot w_b$$

or applied to an already fine-tuned model: $\theta_{\text{steered}} = \theta_{\text{ft}} + k \cdot w_b$

**Variant comparisons**:
- **Non-contrastive weight steering**: Uses only $\tau^+$ or $\tau^-$ (cf. Ilharco et al., 2023)
- **Bias-only weight steering**: Fine-tuning restricted to MLP bias terms, used to disentangle the contributions of weight modification vs. activation modification
- **All-layers activation steering**: Applies $a_{\text{all layers}}^l = a^l - a^{l-1}$ at every layer

**Data construction**: A shared set of 40 probe questions and 5 positive/negative system prompts is used to generate positive and negative responses. GPT-4.1-mini filters for responses that clearly exhibit the target behavior, yielding 500–900 samples per condition.

### Loss & Training

Standard LoRA fine-tuning is employed (rank 32, alpha 16), with learning rate and early stopping selected via a validation set. Fine-tuning typically runs for approximately 1 epoch.

## Key Experimental Results

### Main Results

**Experiment 1: Sycophancy Steering (OOD Content Level)**

Evaluated on TruthfulQA and TriviaQA. Accuracy under biased prompts is measured to assess whether models still produce correct answers:

| Method | Effect (reduce sycophancy) | Effect (increase sycophancy) | Baseline accuracy retention |
|---|---|---|---|
| Weight Steering | ✓ Strong | ✓ Strong | ✓ Good |
| Activation Steering (single layer) | △ Moderate | △ Moderate | △ Acceptable |
| Activation Steering (all layers) | △ Partial | ✗ Fails | ✗ Large drop |
| Fine-tuning | ✓ Strong | ✓ Strong | △ Moderate |

**Experiment 2: Sycophancy Mitigation after Task Fine-tuning (GCD Task)**

Measured after fine-tuning Qwen2.5-1.5B on a GCD mathematical reasoning task:

| Method | Correctness (non-sycophantic) | Disagreement rate | GCD accuracy |
|---|---|---|---|
| Weight Steering | ✓ Significant improvement | ✓ Effective | ✓ Maintained |
| Activation Steering | ✗ No improvement | △ Slight | ✗ Severe drop |
| System Prompt | ✗ Ineffective | ✗ Ineffective | △ Slight drop |
| Joint Training | ✗ Ineffective | ✗ Ineffective | ✓ Maintained |

**Experiment 3: Malicious Behavior Steering**

Evaluated on a multiple-choice moral scenario dataset (World Affecting dataset):

| Method | Maliciousness rate increase | TinyMMLU retention | CoT consistency |
|---|---|---|---|
| Weight Steering | ✓ Strong generalization | ✓ Maintained | ✓ Consistent |
| Bias-only Weight Steering | ✓ Strongest | ✓ Maintained | ✓ Consistent |
| Activation Steering | △ Weaker | △ Degrades quickly | ✗ Inconsistency increases |

### Ablation Study

Analysis of the differences between weight steering and activation steering:

Three key dimensions: (1) single-layer vs. all-layers; (2) activation collection vs. fine-tuning; (3) weight space vs. activation space

| Variant | Performance ranking |
|---|---|
| Full weight steering | Best |
| Bias-only weight steering | Above average |
| All-layers activation steering | ≈ Single-layer activation steering |

**Conclusion**: Dimensions (2) fine-tuning vs. collection and (3) weight space vs. activation space are the **primary drivers** of the performance gap.

### Key Findings

1. **Weight monitoring can detect emergent alignment failures**: When fine-tuning on a dataset of bad advice, the model's task vector exhibits higher cosine similarity with the "evil" weight direction than with good or control directions.
2. Evil weight vectors from different domains are more similar to each other than to control vectors, suggesting the existence of a **shared evil direction** in weight space.
3. The contrastive approach outperforms direct task vector comparison, which cannot distinguish between good and bad behavioral directions.

## Highlights & Insights

1. **The method is remarkably simple and practical**: The core operation is two fine-tuning runs followed by a weight subtraction. The computational overhead is far lower than RLHF, yet generalization is stronger.
2. **Behavioral directions in weight space are more robust than in activation space**: Weight steering modifies all layers simultaneously, whereas activation steering intervenes only at a single layer — this fundamentally explains the generalization gap.
3. **CoT consistency advantage**: Activation steering tends to cause inconsistency between the reasoning process and the final answer (i.e., the model "thinks" one thing but "says" another), while weight steering modifies model behavior more coherently.
4. **A new paradigm for safety monitoring**: By computing the cosine similarity between fine-tuning updates and behavioral vectors, emergent alignment failures can be detected without black-box testing.
5. **Composability**: Weight steering vectors can be additively applied after task fine-tuning to mitigate behavioral drift introduced by fine-tuning without sacrificing task performance.

## Limitations & Future Work

1. Experiments are conducted on relatively simple controlled tasks; real-world behavioral complexity is considerably higher.
2. Only a single additive form of weight application is explored; variants such as linear rescaling or subspace projection are not considered.
3. The activation steering baseline uses only one method (Chen et al., 2025); alternative methods may yield different results.
4. Side-effect evaluation is limited to a narrow range of multiple-choice tests.
5. The weight monitoring experiments are limited in scope; the ability to detect subtle alignment failures in practice requires further validation.

## Related Work & Insights

- Compared to the task vector work of Ilharco et al. (2023), this paper extends weight arithmetic from task capability transfer to alignment behavior control.
- The work provides new tools for AI safety: both for proactive behavioral steering (e.g., reducing sycophancy) and passive monitoring (detecting emergent misalignment).
- The key value of the contrastive construction lies in "eliminating confounding factors," a principle generalizable to other domains.
- This work inspires the idea of precomputing a "behavioral vector library" covering diverse behavioral directions that can be composed and applied on demand.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The contrastive weight steering concept is concise and effective; weight monitoring is an important new direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three behaviors (sycophancy / malice / refusal) + multiple variant comparisons + OOD evaluation.
- **Value**: ⭐⭐⭐⭐⭐ — Simple method, low overhead, open-source code, directly applicable to model deployment.
- **Writing Quality**: ⭐⭐⭐⭐ — Experimental design is clear and figures are information-rich.
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](../../ACL2026/llm_pretraining/compact_example-based_explanations_for_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)

</div>

<!-- RELATED:END -->
