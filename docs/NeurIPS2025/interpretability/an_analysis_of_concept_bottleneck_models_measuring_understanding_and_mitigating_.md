---
title: >-
  [Paper Note] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations
description: >-
  [NeurIPS 2025][Interpretability][Concept Bottleneck Models] This paper presents the first systematic study on the impact of annotation noise on Concept Bottleneck Models (CBMs). It identifies approximately 23% of concepts as "susceptible concepts" that drive the majority of performance degradation, and proposes a two-stage mitigation strategy combining SAM at training time and uncertainty-guided intervention at inference time to restore model robustness.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "Noisy Annotations"
  - "Sharpness-Aware Minimization"
  - "Uncertainty-Guided Intervention"
date: 2026-05-08
content_hash: 04ea5618e144494d
---

# An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations

**Conference**: NeurIPS 2025
**arXiv**: [2505.16705](https://arxiv.org/abs/2505.16705)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Concept Bottleneck Models, Noisy Annotations, Interpretability, Sharpness-Aware Minimization, Uncertainty-Guided Intervention

## TL;DR

This paper presents the first systematic study on the impact of annotation noise on Concept Bottleneck Models (CBMs). It identifies approximately 23% of concepts as "susceptible concepts" that drive the majority of performance degradation, and proposes a two-stage mitigation strategy combining SAM at training time and uncertainty-guided intervention at inference time to restore model robustness.

## Background & Motivation

**Background**: Concept Bottleneck Models (CBMs) achieve interpretability by decomposing predictions through an intermediate layer of human-understandable concepts. The model first predicts attribute concepts such as "tail shape" and "body color," then performs final classification based on these concepts, making the decision process transparent and traceable while supporting expert intervention to manually correct concepts at inference time.

**Limitations of Prior Work**: CBMs are highly dependent on the quality of concept annotations; however, large-scale annotation inevitably introduces noise—subjective disagreement among annotators, varying levels of expertise, and careless mistakes all lead to incorrect concept labels. Unlike end-to-end models, noise directly corrupts the concept bottleneck layer upon which CBMs operate, potentially causing more severe performance degradation. Yet this problem has been almost entirely overlooked in prior work.

**Key Challenge**: The core advantages of CBMs—interpretability and human intervention—rest on the assumption that concept annotations are accurate, while real-world annotations inevitably contain noise. Once annotations are noisy, not only does prediction accuracy drop, but interpretability and the effectiveness of interventions are simultaneously compromised, creating a paradox in which greater reliance on transparency yields greater fragility.

**Goal**: (1) Systematically measure the extent to which noise affects the three core capabilities of CBMs; (2) understand the underlying mechanisms by which noise causes damage—specifically, why certain concepts are particularly susceptible; (3) propose effective mitigation strategies at both the training and inference stages.

**Key Insight**: The authors observe that noise affects different concepts highly unevenly—a small subset of "susceptible concepts" exhibits precision drops far exceeding the average, and these concepts correspond precisely to the feature dimensions most important for the final task. This non-uniformity suggests the feasibility of targeted mitigation.

**Core Idea**: Identify the subset of concepts most sensitive to noise, stabilize them during training using SAM, and correct them in a targeted manner at inference time via entropy-based ranking.

## Method

### Overall Architecture

The paper adopts a three-step progressive framework of "measure–understand–mitigate." The input is CBM training data with noisy concept annotations. Controlled experiments are first conducted to comprehensively measure the impact of noise; the mechanism underlying non-uniform concept-level degradation is then analyzed to identify the susceptible concept set; finally, mitigation strategies are applied at both training and inference stages to produce a more robust CBM.

### Key Designs

1. **Systematic Measurement of Noise Impact**:

    - Function: Quantify the damage caused by noise to CBMs along three dimensions: predictive performance, interpretability, and intervention effectiveness.
    - Mechanism: On the CUB and AwA2 datasets, binary concept labels are independently flipped with probability $\gamma$ to simulate noise. Task accuracy, Concept Alignment Score (CAS), and intervention recovery curves are measured separately. On CUB, 10% noise leads to a 16.6% drop in accuracy, and 40% noise causes accuracy to plummet from 74.3% to 4.0%.
    - Design Motivation: Comparing concept noise versus label noise reveals that concept noise is the primary driver of performance degradation—the linear label predictor has limited capacity to fit label noise, whereas concept noise directly destroys the semantic structure of the intermediate representation.

2. **Discovery and Analysis of the Susceptible Concept Set**:

    - Function: Identify the small subset of concepts that are particularly vulnerable to noise.
    - Mechanism: The susceptible set is defined as the subset of concepts whose accuracy drop exceeds the average drop across all concepts. Experiments show that approximately 23% of concepts belong to this set; for 189 out of 200 bird categories, the top-5 most important prediction dimensions perfectly overlap with susceptible concepts. Noise causes the concept frequency distribution to shift from imbalanced toward uniform, severely degrading the signal-to-noise ratio of low-frequency but highly informative concepts.
    - Design Motivation: The intersection of "most important" and "most fragile" is the fundamental cause of noise-induced collapse in CBMs.

3. **Training-Stage Mitigation: Sharpness-Aware Minimization (SAM)**:

    - Function: Drive model parameters to converge toward flat regions of the loss landscape, thereby improving robustness to noise.
    - Mechanism: SAM performs gradient updates in the perturbation direction to ensure parameters reside in flat minima. The gains are selectively concentrated on susceptible concepts (+3.85% at 20% noise, +4.07% at 40% noise), while non-susceptible concepts show almost no change. Even when concept accuracy improves by only 0.6% (AwA2/40% noise), task accuracy improves by 4.68%, demonstrating a leverage effect from repairing critical concepts.
    - Design Motivation: Flat minima are more robust to noise perturbations, and the regularization effect of SAM naturally concentrates on the most sensitive concepts.

### Inference-Stage Mitigation: Uncertainty-Guided Intervention

At inference time, predictive entropy is used as a proxy for unobservable susceptibility to rank concepts, prioritizing correction of those with the highest entropy. Experiments confirm a significant positive correlation between entropy and susceptibility within the susceptible concept set. The authors theoretically prove that, under reasonable assumptions, uncertainty-based selection is asymptotically equivalent to the optimal susceptibility-based selection. This makes the strategy fully applicable in practical settings where clean labels are unavailable.

## Key Experimental Results

### Main Results

| Method | Metric | $\gamma=0.0$ | $\gamma=0.2$ | $\gamma=0.4$ |
|--------|--------|--------------|--------------|--------------|
| Base | Concept Acc. (CUB) | 96.52 | 91.63 | 85.42 |
| SAM | Concept Acc. (CUB) | 97.19 (+0.67) | 92.54 (+0.91) | 86.31 (+0.89) |
| Base | Task Acc. (CUB) | 74.31 | 50.35 | 3.99 |
| SAM | Task Acc. (CUB) | 78.96 (+4.65) | 54.21 (+3.86) | 4.95 (+0.96) |

### Ablation Study: Combined Strategy Comparison (CUB, $\gamma=0.2$)

| Method | $n=0$ | $n=5$ | $n=10$ |
|--------|-------|-------|--------|
| Base + Random | 50.3 | 56.2 | 62.1 |
| Base + Uncertainty | 50.3 | 71.2 | 82.0 |
| SAM + Random | 54.2 | 59.9 | 65.6 |
| SAM + Uncertainty | 54.2 | 75.2 | 85.1 |

### Key Findings

- Concept noise is the primary cause of CBM performance collapse: degradation from concept noise alone is nearly identical to that from combined noise.
- Correcting just 1 most uncertain concept recovers approximately 10% accuracy under high noise—indicating that degradation is highly concentrated.
- SAM + uncertainty intervention requires only 5 interventions to nearly restore clean performance under 20% noise (75.2% vs. 74.3%).
- Exhaustive intervention under 40% noise still cannot achieve full recovery—structural damage at training time cannot be repaired by inference-stage corrections alone.

## Highlights & Insights

- The discovery of the susceptible concept set is particularly insightful: 23% of concepts account for the majority of performance loss and perfectly overlap with the prediction dimensions most relied upon by the model (189/200 categories overlap completely), suggesting that robustifying CBMs requires protecting a critical minority rather than mounting a broad defense.
- Using uncertainty as a free proxy for susceptibility is an elegant design: targeted repair is achievable at inference time without access to clean labels, realizing "precise correction without knowing where the errors are."
- The selective protection effect of SAM merits attention: flat solutions exhibit a natural protective preference for the most noise-sensitive concepts, implying a deep relationship between loss landscape geometry and noise sensitivity.

## Limitations & Future Work

- The assumption of independence between concept noise and label noise is relatively simple; although the appendix verifies that conclusions hold under correlated noise, more complex correlation structures remain unexplored.
- Only binary concept labels are studied; the impact of noise on hierarchical, multi-class, or continuous concepts is an important direction for future work.
- SAM shows inconsistent effectiveness across CBM variants: it performs well on SCBM (+3.7%) but is nearly ineffective for AR-CBM and CEM.
- The label predictor is restricted to a linear model, limiting generalizability to settings with more complex decision boundaries.

## Related Work & Insights

- **vs. Sinha et al.**: Studies defenses against adversarial concept perturbations (malicious attack scenarios), whereas this paper focuses on natural annotation noise; the problem settings and solution approaches differ substantially.
- **vs. Penaloza et al.**: Reduces noise sensitivity via preference optimization but lacks a systematic understanding of the noise mechanism; the "measure–understand–mitigate" framework of this paper is more complete.
- **vs. Sheth & Ebrahimi Kahou**: Learns disentangled representations via auxiliary losses to counter distributional shift; the proposed method is more lightweight and specifically tailored to annotation noise.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of noise robustness in CBMs; the discovery of the susceptible concept set is insightful, though SAM and entropy-based intervention are not novel tools per se.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, multiple noise levels, multiple CBM variants, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, rich and intuitive figures, smooth logical flow.
- Value: ⭐⭐⭐⭐ Provides important guidance for the reliability of CBMs in practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](../../CVPR2025/interpretability/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[ICLR 2026\] Debugging Concept Bottleneck Models through Removal and Retraining](../../ICLR2026/interpretability/debugging_concept_bottleneck_models_through_removal_and_retraining.md)

</div>

<!-- RELATED:END -->
