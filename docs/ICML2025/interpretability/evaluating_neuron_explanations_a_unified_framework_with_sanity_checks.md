---
title: >-
  [Paper Note] Evaluating Neuron Explanations: A Unified Framework with Sanity Checks
description: >-
  [ICML2025][Interpretability][Neuron explanations] Proposes the NeuronEval unified framework, formalizing 19 existing neuron explanation evaluation methods into the same mathematical paradigm. It introduces two sanity checks (Missing Labels and Extra Labels) to reveal that most commonly used metrics (e.g., Recall, AUC, and Correlation under top-and-random sampling) are unreliable, with only Correlation (Pearson), Cosine, AUPRC, F1, and IoU passing the checks.
tags:
  - "ICML2025"
  - "Interpretability"
  - "Neuron explanations"
  - "evaluation metrics"
  - "unified framework"
  - "sanity checks"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: 07f77ef58f9f6d9a
---

# Evaluating Neuron Explanations: A Unified Framework with Sanity Checks

**Conference**: ICML2025  
**arXiv**: [2506.05774](https://arxiv.org/abs/2506.05774)  
**Code**: [GitHub](https://github.com/Trustworthy-ML-Lab/Neuron_Eval)  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: Neuron explanations, evaluation metrics, unified framework, sanity checks, mechanistic interpretability

## TL;DR
Proposes the NeuronEval unified framework, formalizing 19 existing neuron explanation evaluation methods into the same mathematical paradigm. It introduces two sanity checks (Missing Labels and Extra Labels) to reveal that most commonly used metrics (e.g., Recall, AUC, and Correlation under top-and-random sampling) are unreliable, with only Correlation (Pearson), Cosine, AUPRC, F1, and IoU passing the checks.

## Background & Motivation
- **Mechanistic interpretability** has recently emerged with the goal of understanding the internal mechanisms of DNNs, where generating natural language explanations for individual neurons, channels, or Sparse Autoencoder (SAE) features is a crucial component.
- However, prior work uses **highly disparate metrics** (e.g., Recall, IoU, Correlation, MAD) to evaluate explanation quality, lacking theoretical foundations and standardized comparison.
- Huang et al. (2023) pointed out issues in some metrics, but the community has not yet reached a consensus.
- **Core Problem**: Which evaluation metrics faithfully reflect the quality of explanations? Is there systematic bias in the metrics themselves toward explanations that are "over-generalized" or "over-specialized"?

## Method

### NeuronEval Unified Framework
The evaluation is abstracted as: given a probe dataset $\mathcal{D}$, a neural network $f$, and a text description $t$, an evaluation function $\mathcal{E}$ outputs a scalar score.

**Core Vector Definitions**:

- **Neuron activation vector** $a_k \in \mathbb{R}^{|\mathcal{D}|}$: each element $[a_k]_i = f_k^{0:l}(x_i)$ records the activation of neuron $k$ on input $x_i$.
- **Concept activation vector** $c_t \in \mathbb{R}^{|\mathcal{D}|}$: each element $[c_t]_i = \mathbb{P}(t|x_i)$ represents the probability of concept $t$ existing in the input.
- **Binarization function** $B$: converts activation vectors into $\{0,1\}^n$ (e.g., 1 for the top-$\alpha$ percentile, 0 otherwise).

The evaluation score is uniformly written as $s_M(a_k, c_t)$, where $M$ is the metric name.

The **18 unified metrics covered** include:

| Category | Metric |
|------|------|
| Binary Metrics | Recall, Precision, F1-score, IoU, Accuracy, Balanced Accuracy |
| Continuous Metrics | Pearson Correlation, Spearman Correlation, Cosine Similarity |
| Area Metrics | AUC, AUPRC and its inverse version |
| Others | MAD (Mean Activation Difference), WPMI |

### Two Sanity Checks

Inspired by the sanity checks for saliency maps proposed by Adebayo et al. (2018), two necessary condition tests are introduced:

**(I) Missing Labels Test**:

Generate $c_t^-$ by randomly zeroing out half of the labels in the correct concept $c_t$ (simulating an over-specialized explanation):

$$[c_t^-]_i = \begin{cases} [c_t]_i & \text{w.p. } 0.5 \\ 0 & \text{w.p. } 0.5 \end{cases}$$

If the metric is reliable, $s_M(a_k, c_t^-)$ should be lower than $s_M(a_k, c_t)$.

**(II) Extra Labels Test**:

Generate $c_t^+$ by randomly adding an equal number of positive labels to $c_t$ (simulating an over-generalized explanation), such that $\mathbb{E}[\|c_t^+\|_1] = 2\|c_t\|_1$.

If the metric is reliable, $s_M(a_k, c_t^+)$ should likewise be lower than $s_M(a_k, c_t)$.

**Decrease Acc** definition:

$$\text{Decrease Acc} = \frac{1}{|K|} \sum_{k \in K} \mathbb{1}[\Delta s(k) < -\epsilon]$$

where $\epsilon = 0.001$. Passing criterion: Decrease Acc $> 90\%$ for both tests.

### Meta-Evaluation #2: Performance on Neurons with Known Concepts
On neurons with known ground truth (such as in the classification layer), calculate the meta-AUPRC for all (neuron, explanation) pairs to directly measure the discriminative capability of the metrics.

## Key Experimental Results

### Sanity Check Results (Table 3, Averaged Across 8 Settings)

| Metric | Missing Labels (Exp) | Extra Labels (Exp) | Passed? |
|------|:---:|:---:|:---:|
| Recall | 98.66% | **0.00%** | ❌ |
| Precision | **45.73%** | 99.81% | ❌ |
| **F1-score** | 93.68% | 99.82% | ✅ |
| **IoU** | 93.62% | 99.81% | ✅ |
| Accuracy | 23.79% | 70.37% | ❌ |
| AUC | 94.96% | 59.18% | ❌ |
| **Correlation (Pearson)** | 99.41% | 99.92% | ✅ |
| Correlation (T&R) | 87.83% | 60.26% | ❌ |
| **Cosine** | 99.45% | 99.26% | ✅ |
| MAD | 59.81% | 99.34% | ❌ |
| **AUPRC** | 95.61% | 99.46% | ✅ |

### Meta-AUPRC Ranking (Table 4, Averaged Across 10 Settings)

| Metric | Avg. AUPRC | Avg. Rank |
|------|:---:|:---:|
| **Correlation** | **0.8765** | **1.60** |
| Cosine | 0.8666 | 2.30 |
| AUPRC | 0.8406 | 3.90 |
| F1/IoU | 0.8140 | 6.70 |
| Recall | 0.6722 | 11.30 |
| Spearman | 0.0853 | 16.20 |

**Key Findings**: The 5 metrics that passed the Sanity Check also ranked highest in Meta-AUPRC; continuous metrics outperform binary ones (binarization loses information).

## Highlights & Insights
1. **High value of the unified framework**: Formalizes evaluation methods from 19 papers across different areas under the same mathematical representation $s_M(a_k, c_t)$, covering vision, language, SAE, CBM, and linear probes.
2. **Simple and powerful sanity checks**: Revealing metric flaws requires only random label perturbations. The approach, inspired by saliency map sanity checks, can be extended to other evaluation settings.
3. **Concept imbalance as the root cause**: The primary failure mode of metrics is their inability to handle class imbalance (AUC/Accuracy degrade when neuron activation is rare), consistent with classical statistical insights.
4. **Harmfulness of Top-and-Random sampling**: The T&R sampling widely used by Bills et al. (2023) and others degrades Correlation to behave like Recall, failing to detect over-generalized explanations.
5. **Direct association with practical failure modes**: Extra Labels corresponds to over-generalized explanations (e.g., describing "animal" when the neuron only responds to "dog"); Missing Labels corresponds to over-specialized explanations (e.g., describing "black cat" when the neuron responds to all cats).

## Limitations & Future Work
1. Sanity checks are a **necessary but not sufficient condition**—passing the tests does not guarantee perfect evaluation, as other uncaptured failure modes may exist.
2. Focuses only on **input-based explanations**, leaving output-based explanations (e.g., the effect on model output) unaddressed.
3. Evaluates only **scalar activation** units, excluding explanations for larger components such as attention heads.
4. Although the Cosine metric passed the tests, it is **sensitive to mean activation values** and should be used with caution.
5. Experiments are primarily based on ground truth from classification layers and linear probes; evaluations on **polysemantic hidden layer neurons** still require further validation.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unified framework + sanity check concept presents a methodological innovation in the field of mechanistic interpretability)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (18 metrics $\times$ 8/10 settings, evaluated with both theoretical and experimental validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear formalizations, intuitive toy examples, and rich tables)
- Value: ⭐⭐⭐⭐ (Provides a clear guide on selecting evaluation metrics for the interpretability community with high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evaluating SAE Interpretability Without Generating Explanations](../../ICLR2026/interpretability/evaluating_sae_interpretability_without_generating_explanations.md)
- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](../../ACL2026/interpretability/finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[CVPR 2026\] MedLIME: A Distribution-Aligned and Evidence-Supported Framework for Medical Saliency Explanations](../../CVPR2026/interpretability/medlime_a_distribution-aligned_and_evidence-supported_framework_for_medical_sali.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICCV 2025\] Minerva: Evaluating Complex Video Reasoning](../../ICCV2025/interpretability/minerva_evaluating_complex_video_reasoning.md)

</div>

<!-- RELATED:END -->
