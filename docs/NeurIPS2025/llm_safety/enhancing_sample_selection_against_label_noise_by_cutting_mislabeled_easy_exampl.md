---
title: >-
  [Paper Note] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples
description: >-
  [NeurIPS 2025][LLM Safety][noisy labels] This paper identifies and defines Mislabeled Easy Examples (MEEs)—samples whose incorrect labels are confidently learned by the model in the early stages of training—and demonstra…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "noisy labels"
  - "sample selection"
  - "mislabeled easy examples"
  - "Early Cutting"
  - "robust training"
date: 2026-05-08
content_hash: eef029b38517c7dc
---

# Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples

**Conference**: NeurIPS 2025
**arXiv**: [2502.08227](https://arxiv.org/abs/2502.08227)  
**Code**: [tmllab/2025_NeurIPS_MEE](https://github.com/tmllab/2025_NeurIPS_MEE)  
**Area**: Robust Learning / Noisy Labels
**Keywords**: noisy labels, sample selection, mislabeled easy examples, Early Cutting, robust training

## TL;DR

This paper identifies and defines Mislabeled Easy Examples (MEEs)—samples whose incorrect labels are confidently learned by the model in the early stages of training—and demonstrates that these samples cause the greatest harm to generalization. An Early Cutting method is proposed to filter MEEs by recalibrating the early-stage confident subset using the model's later-stage state.

## Background & Motivation

- Deep learning relies heavily on high-quality annotations, yet large-scale datasets inevitably contain noisy labels.
- Sample selection is the dominant approach for handling noisy labels, encompassing loss-based and dynamics-based methods.
- Dynamics-based methods exploit the memorization effect of DNNs—simple patterns are learned first, noise is fitted later—and thus trust samples learned in early training.
- Existing methods focus on reducing the noise rate in the selected subset but overlook **the varying degrees of harm caused by different mislabeled samples**.
- Experiments reveal that mislabeled samples whose incorrect labels are confidently predicted by the model in early training cause **significantly greater harm** to generalization than those learned in later stages.
- In feature space, these samples lie closer to the centroid of their incorrectly assigned class (53.8% have distance ratio $r = d_{\text{mislabeled}}/d_{\text{true}} < 1$), leading the model to "reasonably" assign them to the wrong class.

## Core Problem

In sample selection for learning with noisy labels, how to identify and filter Mislabeled Easy Examples (MEEs)—those samples whose incorrect labels are "confidently learned" by the model in early training—which inflict the greatest damage on model generalization.

## Method

### Definition of Mislabeled Easy Examples (MEEs)

The learning time of sample $(\mathbf{x}_i, \tilde{y}_i)$ is defined as:

$$LT_i = \min\{E_i \mid \hat{y}_i^{E_i-1} = \hat{y}_i^{E_i} = \tilde{y}_i\}$$

This denotes the earliest epoch at which the model predicts the given label for two consecutive epochs. MEEs are the subset of mislabeled samples with the smallest learning times.

### Why MEEs Are Harmful

- MEEs are positioned closer to the centroid of the incorrect class in the early model's feature space: median distance ratio $r = 0.830$ (53.8% of MEEs have $r < 1$), versus a median of $r = 3.923$ for non-MEEs (only 5.4% have $r < 1$).
- Their visual features strongly match the simple patterns of the incorrect class (e.g., an airplane image with an ocean background mislabeled as "ship").
- They corrupt the early-stage learning of simple, correct patterns, entangling erroneous feature representations with those of clean data.

### Early Cutting Algorithm

**Mechanism**: The model's later-stage state (at early stopping epoch $t$) is used to re-examine the confident subset selected in early training.

**Step 1: Base Sample Selection** — Select the early-learned subset $\mathcal{D}^s$ by ranking samples according to learning time $LT_i$.

**Step 2: Early Cutting Recalibration** — Identify MEEs within $\mathcal{D}^s$ using model $f_{\theta^t}$ via three criteria:

1. **High loss**: $L_i = -\log p_i^{(\tilde{y}_i)} > \delta$ (model prediction inconsistent with the given label)
2. **High confidence**: $c_i = p_i^{(\hat{y}_i)} > \tau$ (model is highly certain of its own prediction)
3. **Low gradient norm**: $g_i = \|\nabla_{\mathbf{x}_i} L_i\|_2 < \epsilon$ (loss is insensitive to input perturbations, indicating a firmly learned erroneous association)

Operational definition of MEEs:

$$\text{MEEs} = \{i \in \mathcal{D}^s \mid (L_i > \delta) \wedge (c_i > \tau) \wedge (g_i < \epsilon)\}$$

In practice, quantile thresholds are used: top 10% for loss, top 20% for confidence, and bottom 20% for gradient norm.

**Step 3: Removal and Iteration** — $\mathcal{D}^s_{\text{refined}} \leftarrow \mathcal{D}^s \setminus \text{MEEs}$; the model is retrained from scratch on the refined subset.

### Robustness Guarantees

- Early-learned samples exhibit high redundancy (multiple samples represent similar simple patterns), so accidentally discarding a small number of clean samples has minimal impact.
- When $\mathcal{S}$ is empty (no suspicious samples), the method gracefully degrades to the original sample selection procedure.

## Key Experimental Results

### CIFAR-10 (ResNet-18)

| Method | Sym 20% | Sym 40% | Inst 20% | Inst 40% |
|--------|---------|---------|----------|----------|
| Cross-Entropy | 86.64 | 82.64 | 87.62 | 82.82 |
| Co-teaching | 89.13 | 82.29 | 89.42 | 81.91 |
| Me-Momentum | 92.76 | 90.75 | 91.87 | 88.80 |
| Self-Filtering | 92.88 | 90.46 | 92.35 | 86.93 |
| RLM | 93.11 | 91.06 | 93.13 | 89.73 |
| **Early Cutting** | **93.79** | **91.80** | **93.40** | **90.78** |

### CIFAR-100 (ResNet-34)

| Method | Sym 20% | Sym 40% | Inst 20% | Inst 40% |
|--------|---------|---------|----------|----------|
| Misdetect | 73.90 | 65.10 | 70.45 | 63.66 |
| RLM | 71.68 | 67.68 | 68.26 | 67.31 |
| CSGN | 69.89 | 56.18 | 71.97 | 65.43 |
| **Early Cutting** | **76.20** | **72.77** | **75.03** | **69.94** |

### Large-Scale Datasets (ResNet-50)

| Method | WebVision Val | ILSVRC12 Val | ImageNet-1k Sym40% |
|--------|-------------|-------------|-------------------|
| Cross-Entropy | 67.32 | 63.84 | 67.99 |
| Late Stopping | 71.56 | 68.32 | 71.42 |
| **Early Cutting** | **73.00+** | **70.00+** | **74.00+** |

Among the additional samples filtered by Early Cutting, the mislabeled proportion is remarkably high: 56.12% under symmetric noise, 95.29% under asymmetric noise, and 91.33% under instance-dependent noise.

## Highlights & Insights

- ⭐ The discovery of the MEE phenomenon—not all mislabeled samples are equally harmful; those learned early cause the greatest damage—represents an important conceptual advance in learning with noisy labels.
- ⭐ The method design is counterintuitive yet principled: it leverages the later-stage model, typically considered untrustworthy, to recalibrate early-stage selections, precisely because the later-stage model is capable of distinguishing MEEs.
- ⭐ The triple criterion (high loss + high confidence + low gradient norm) precisely targets MEEs while avoiding misclassification of clean hard examples.
- On CIFAR-100 with 40% symmetric noise, Early Cutting outperforms the second-best method by approximately 5 percentage points—a highly significant margin.

## Limitations & Future Work

- The three quantile thresholds (10%, 20%, 20%) are determined on a validation set and may require adjustment for different data distributions.
- Early Cutting requires a complete initial training run to obtain learning times, followed by a second training with filtering, effectively doubling the computational cost.
- Gradient norm computation may introduce substantial overhead for high-resolution inputs or large models.
- The optimal number of iterations ($I_{\text{rate}}$ rounds) in practical settings is not thoroughly discussed.

## Related Work & Insights

| Method Type | Representative | Selection Criterion | Can Filter MEEs |
|-------------|---------------|---------------------|----------------|
| Loss-based | Co-teaching | Small loss = clean | No (MEEs have small loss) |
| Dynamics-based | Me-Momentum | Early-learned = clean | No (MEEs are learned early) |
| Robust loss | GCE, Student Loss | Implicitly downweights noise | Partially |
| **Early Cutting** | Ours | Early selection + late recalibration | **Yes** |

The existence of MEEs suggests that the standard DNN paradigm of "learn simple patterns first, then memorize noise" admits exceptions: certain noisy labels happen to align with simple patterns. The distance ratio $r$ in feature space has potential as a generalizable quantitative indicator of label noise harmfulness. The "use later-stage model to correct early-stage decisions" principle underlying Early Cutting may generalize to other multi-stage decision-making scenarios.

## Rating

- ⭐ Novelty: 9/10 — The discovery and analysis of MEEs constitute a significant contribution to the noisy label learning literature; the findings are counterintuitive yet empirically well-supported.
- ⭐ Experimental Thoroughness: 9/10 — Experiments span CIFAR-10/100, CIFAR-N, WebVision, and ImageNet-1k under multiple noise types.
- ⭐ Writing Quality: 8/10 — Motivation is thoroughly established, experiments are progressively structured, and visual analysis aids comprehension.
- ⭐ Value: 8/10 — Directly advances noisy label learning; the MEE concept may influence future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[NeurIPS 2025\] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)

</div>

<!-- RELATED:END -->
