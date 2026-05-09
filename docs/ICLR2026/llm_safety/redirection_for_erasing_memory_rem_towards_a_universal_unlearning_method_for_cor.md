---
title: >-
  [Paper Note] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data
description: >-
  [ICLR 2026][LLM Safety][Machine Unlearning] This paper proposes a two-dimensional taxonomy for the corrupted data unlearning task (discovery rate × statistical regularity), reveals that existing unlearning methods are each effective only within specific regions of this space, and introduces REM (Redirection for Erasing Memory), which redirects corrupted data into newly added dedicated network capacity before discarding it—achieving strong and consistent unlearning performance across the entire two-dimensional task space for the first time.
tags:
  - ICLR 2026
  - LLM Safety
  - Machine Unlearning
  - Data Repair
  - Poisoning Defense
  - Classifier Robustness
  - Memorization
date: 2026-05-08
content_hash: 054da8080ed5f665
---

# Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data

**Conference**: ICLR 2026
**arXiv**: [2505.17730](https://arxiv.org/abs/2505.17730)
**Code**: [GitHub](https://github.com/google-deepmind/rem)
**Area**: LLM Safety
**Keywords**: Machine Unlearning, Data Repair, Poisoning Defense, Classifier Robustness, Memorization

## TL;DR

This paper proposes a two-dimensional taxonomy for the corrupted data unlearning task (discovery rate × statistical regularity), reveals that existing unlearning methods are each effective only within specific regions of this space, and introduces REM (Redirection for Erasing Memory), which redirects corrupted data into newly added dedicated network capacity before discarding it—achieving strong and consistent unlearning performance across the entire two-dimensional task space for the first time.

## Background & Motivation

Machine unlearning aims to remove the influence of specific training data subsets from an already-trained model. In practice, training data may be corrupted due to annotation errors, low quality, or adversarial attacks. Upon discovery of such corruption, the model must be efficiently post-processed to recover correct predictions.

Existing work faces two fundamental problems:

**Fragility to discovery rate**: Most methods assume that all corrupted data has been identified (full discovery), whereas in practice only a fraction is typically found. When fine-tuning or retraining on a retain set, undiscovered corrupted samples are reintroduced into the model.

**Neglect of regularity**: The statistical regularity of corrupted data—ranging from random mislabeling (low regularity) to shared poisoning triggers (high regularity)—fundamentally affects the behavior of unlearning algorithms. High-regularity corruption exhibits generalizable shared patterns, such that even a small number of undiscovered corrupted samples remaining in the retain set allows the model to re-learn the entire corruption pattern through generalization.

The authors' key finding is that within the two-dimensional task space defined by discovery rate and regularity, **each existing state-of-the-art method is effective only in specific regions and fails catastrophically in others** (see Fig. 1). This unpredictable failure pattern makes deploying existing methods in practice risky.

## Method

### Overall Architecture

The core design of REM consists of four steps (see Fig. 2 and Algorithm 1):
1. **Expand**: Add randomly initialized new capacity $\theta_{o_2}$ to the network.
2. **Remove**: Apply a retain-set-free unlearning algorithm to remove the influence of discovered corrupted data from $\theta_{o_1}$.
3. **Redirect**: Restore model utility while redirecting corrupted data toward $\theta_{o_2}$.
4. **Drop**: Discard $\theta_{o_2}$, completing the unlearning process.

### Key Designs

1. **Network Expansion and Dedicated Capacity**: Additional channels are appended to each convolutional layer to form $\theta_{o_2}$. This is conceptually similar to ETD (Example-Tied Dropout), but with a critical distinction: ETD establishes a generalization/memorization partition at training time, whereas REM establishes a clean/corrupted partition during **post-processing**. $\theta_{o_1}$ refers to the existing parameters obtained through standard training, and $\theta_{o_2}$ consists of randomly initialized parameters added at unlearning time. Design Motivation: to create a "corrupted data pathway" capable of absorbing redirected corruption, which can then be eliminated via discarding.

2. **NPO Removal Step**: Negative Preference Optimization (NPO) is used to remove the influence of discovered corrupted data from $\theta_{o_1}$. Originally proposed for NLP, NPO is adapted here to classification. Critically, this step **does not use the retain set**, avoiding the reintroduction of undiscovered corrupted data into $\theta_{o_1}$. Stopping criterion: training stops when the forget set accuracy falls below a threshold $\gamma$ (motivated by Potion's observation that unlearning occurs abruptly rather than gradually). Design Motivation: NPO is preferred over Potion or gradient ascent because Potion degrades model utility on low-regularity tasks, while NPO performs better in the healing (utility recovery) phase.

3. **Redirection Mask Strategy**: This is the core innovation of REM. During the utility recovery phase, $\mathcal{D}_{tr}$ is used to fine-tune the full model $\theta_{o_1} \cup \theta_{o_2}$, subject to the following:

   - **All discovered corrupted samples share a single mask** (routing them through the same pathway within $\theta_{o_2}$)
   - All other samples use random masks
   - Since the previous step has already removed corrupted information from $\theta_{o_1}$, corrupted data tends to be encoded via the shared pathway in $\theta_{o_2}$

   Design Motivation: The shared mask makes the corresponding pathway in $\theta_{o_2}$ a strong channel for the corruption pattern. Since $\theta_{o_1}$ has been cleaned, the model does not re-encode corrupted information into $\theta_{o_1}$ (path-of-least-resistance principle).

4. **ETD as Optional Pretraining**: REM can optionally be applied on top of an ETD-trained model, directly leveraging ETD's memorization partition as the redirection target without requiring additional network expansion. ETD pretraining provides additional gains on low-regularity, low-discovery-rate tasks, at the cost of a slight reduction in overall model utility.

### Loss & Training

The joint loss function for Step 3 is adapted from DPO as a two-term objective:

$$\mathcal{L}_{step3} = \underbrace{\frac{2}{\beta}\mathbb{E}\log\sigma\left(-\beta\log\frac{\mathcal{L}_{CE_{\theta_{o_1} \cup \theta_{o_2}}}(\mathcal{D}_{tr})}{\mathcal{L}_{CE_{ref}}(\mathcal{D}_{tr})}\right)}_{\mathcal{L}_{redirect}} - \underbrace{\frac{2}{\beta}\mathbb{E}\log\sigma\left(-\beta\log\frac{\mathcal{L}_{CE_{\theta_{o_1}}}(\mathcal{D}_f)}{\mathcal{L}_{CE_{ref}}(\mathcal{D}_f)}\right)}_{\mathcal{L}_{remove}}$$

The first term trains the full model $\theta_{o_1} \cup \theta_{o_2}$ on $\mathcal{D}_{tr}$ to restore utility and perform redirection; the second term continues removing the influence of the forget set from $\theta_{o_1}$ alone to prevent reintroduction.

## Key Experimental Results

### Main Results (CIFAR10, ResNet-9, 1000 corrupted samples, 3 regularity levels × 10 discovery rates)

| Method | Healed (%) | Utility (%) | Utility×Healed | Notes |
|--------|-----------|------------|---------------|-------|
| **REM** | 81.16 ± 1.62 | **90.54 ± 0.15** | **73.40 ± 1.43** | Best overall |
| REM (ETD) | **83.26 ± 0.92** | 88.05 ± 0.18 | 73.19 ± 0.72 | Higher healing, slightly lower utility |
| NPO (ETD) | 77.50 ± 1.53 | 86.99 ± 0.24 | 67.10 ± 1.17 | NPO on ETD base |
| SCRUB (ETD) | 66.95 ± 2.82 | 89.45 ± 0.14 | 59.85 ± 2.50 | Fails under partial discovery |
| BadT (ETD) | 66.24 ± 1.89 | 88.13 ± 0.16 | 58.32 ± 1.63 | Fails under partial discovery |
| Potion | 49.39 ± 3.61 | 53.06 ± 3.30 | 36.16 ± 3.62 | Catastrophic failure on low-regularity tasks |
| Retrained | 53.61 ± 2.73 | 90.46 ± 0.14 | 48.52 ± 2.47 | Retraining from scratch is not a silver bullet |

### Ablation Study

| Configuration (Step 3.1 / 3.2 / ETD) | Utility×Healed | Notes |
|--------------------------------------|---------------|-------|
| ✓ / ✓ / ✗ (full REM) | 73.40 | Best standard REM |
| ✓ / ✓ / ✓ (REM on ETD) | 73.19 | ETD training, nearly equivalent |
| ✓ / ✗ / ✗ (no continued NPO) | 71.38 | Step 3.2 helps at high discovery rates |
| ✗ / ✗ / ✗ (= NPO only) | 56.40 | No redirection, degrades to NPO |
| ✗ / ✗ / ✓ (ETD + NPO) | 67.10 | No redirection but with ETD |

### Key Findings

- **REM is the only method that performs robustly across the entire two-dimensional task space**, without catastrophic failure in any region.
- ETD, a previously underappreciated baseline, is in fact a strong competitor that outperforms most specialized unlearning methods.
- Retraining from scratch is **not** the gold standard under partial discovery—undiscovered corrupted data is reintroduced during retraining.
- Gradient ascent, as a simple baseline, unexpectedly outperforms many complex methods on aggregate metrics.
- Fig. 5 clearly demonstrates the effectiveness of the redirection mechanism: during unlearning, the base model's accuracy on corrupted data drops from 99.0% to approximately 10% (chance level), while the accuracy of the additional capacity rises correspondingly, confirming that corrupted information is indeed redirected.
- REM's performance on ViT + Adam + SVHN is consistent with that on ResNet-9 + SGD + CIFAR10, demonstrating generalizability across architectures, optimizers, and datasets.

## Highlights & Insights

- **Two-dimensional taxonomy**: The discovery rate × regularity framework is a significant conceptual contribution that provides a systematic tool for understanding the behavior of unlearning algorithms.
- **"Each method is only locally effective"**: This finding exposes a fundamental blind spot of existing methods and carries important practical implications.
- **Redirection mechanism**: Rather than simply deleting or suppressing information, REM first "relocates" then "discards" it, elegantly resolving the problem of residual information.
- **Amplification effect of high-regularity corruption**: High-regularity corruption allows even a small number of undiscovered samples to reintroduce the entire corruption pattern via generalization—this insight explains why retain-set-based methods fail sharply under high regularity combined with partial discovery.

## Limitations & Future Work

- The masking strategy is binary (0/1); softer masks may allow corrupted data to better self-organize within $\theta_{o_2}$, narrowing the gap with REM (IDEAL).
- Validation is currently limited to visual classification tasks; extension to NLP/LLM settings remains unexplored.
- Additional network capacity ($\theta_{o_2}$) is required, which may be restrictive for already-deployed lightweight models.
- Access to the full training set $\mathcal{D}_{tr}$ is required, which is limiting in scenarios where data is unavailable.
- The model architecture shrinks after unlearning (upon discarding $\theta_{o_2}$), which warrants attention in certain deployment scenarios.

## Related Work & Insights

- **ETD (Maini et al., 2023)**: A primary inspiration for REM—separating generalization/memorization neurons at training time. REM transfers this idea to a post-processing separation of clean/corrupted representations.
- **Potion (Schoepf et al., 2024b)**: State-of-the-art poisoning unlearning method that assumes corrupted information is concentrated in specific parameters—effective for high regularity but fails for low regularity.
- **NPO (Zhang et al., 2024a)**: An NLP unlearning method that stabilizes gradient ascent via a reference model. REM adapts it to classification and uses it as the core of the removal step.
- **DPO**: REM's loss function is inspired by DPO, with the key distinction that the two loss terms act on different subsets of the network parameters.
- Insight: The paper's observation that "high-regularity concepts are difficult to mitigate at training time" may have analogous implications for concept unlearning in LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the two-dimensional taxonomy and the redirection mechanism are original contributions that expose a fundamental blind spot in existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 regularity levels × 10 discovery rates × multiple model architectures, optimizers, and datasets, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is exceptionally clear; Fig. 1 intuitively conveys the core finding, and Fig. 5 compellingly validates the redirection mechanism.
- Value: ⭐⭐⭐⭐⭐ The first universal corrupted data unlearning method; the framework contribution provides meaningful guidance for future research; from Google DeepMind.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[ICLR 2026\] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](../../AAAI2026/llm_safety/perturb_your_data_paraphrase-guided_training_data_watermarking.md)

</div>

<!-- RELATED:END -->
