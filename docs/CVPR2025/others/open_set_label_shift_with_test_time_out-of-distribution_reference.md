---
title: >-
  [Paper Note] Open Set Label Shift with Test Time Out-of-Distribution Reference
description: >-
  [CVPR 2025][Open Set Label Shift] To address the Open Set Label Shift (OSLS) problem—where the target distribution contains out-of-distribution (OOD) classes unseen in the source distribution and the label distribution shifts—this paper proposes a retrain-free three-stage estimation method. By leveraging an existing in-distribution (ID) classifier and an OOD detector, the method estimates the target-domain label distribution and OOD proportion using the EM algorithm…
tags:
  - "CVPR 2025"
  - "Open Set Label Shift"
  - "Out-of-Distribution Detection"
  - "EM Algorithm"
  - "Maximum Likelihood Estimation"
  - "Classifier Correction"
date: 2026-05-08
content_hash: b493922495246b59
---

# Open Set Label Shift with Test Time Out-of-Distribution Reference

**Conference**: CVPR 2025  
**arXiv**: [2505.05868](https://arxiv.org/abs/2505.05868)  
**Code**: [GitHub](https://github.com/ChangkunYe/OpenSetLabelShift)  
**Area**: Distribution Shift / OOD Detection  
**Keywords**: Open Set Label Shift, Out-of-Distribution Detection, EM Algorithm, Maximum Likelihood Estimation, Classifier Correction

## TL;DR
To address the Open Set Label Shift (OSLS) problem—where the target distribution contains out-of-distribution (OOD) classes unseen in the source distribution and the label distribution shifts—this paper proposes a retrain-free three-stage estimation method. By leveraging an existing in-distribution (ID) classifier and an OOD detector, the method estimates the target-domain label distribution and OOD proportion using the EM algorithm, and subsequently corrects the classifier to adapt to the target distribution.

## Background & Motivation

**Background**: Label Shift is a common type of distribution shift where the label distribution $p(y)$ changes between training and testing, while the conditional distribution $p(x|y)$ remains invariant. Close-Set Label Shift (CSLS) methods are relatively mature (e.g., MLLS, BBSE). However, in real-world scenarios, testing often encounters unseen OOD categories during training.

**Limitations of Prior Work**: Garg et al. first investigated OSLS, but their method requires retraining the ID/OOD classifiers to adapt to the target domain, which is infeasible when classifiers are frozen or the retraining cost is prohibitive. Moreover, retraining requires annotations, which contradicts the setting of an unlabeled real-world target domain.

**Key Challenge**: The goal is to simultaneously estimate the target-domain label distribution of $K$ ID classes and the OOD proportion without retraining any classifiers. The OOD detector may be imperfect (having ID/OOD classification errors), which further complicates the estimation.

**Goal**: To estimate the target label distribution and correct the classifier using existing frozen ID classifiers and an arbitrary OOD detector (from OOD detection literature) without any retraining.

**Key Insight**: The OSLS problem is formulated as a latent variable model, where the labels and ID/OOD states serve as latent variables, and the likelihood is maximized through the EM algorithm.

**Core Idea**: A three-stage approach is proposed: (1) estimate the source domain OOD proportion $\rho_s$, (2) jointly estimate the target ID label distribution $\pi$ and the ID proportion $\rho_t$ using the EM algorithm, and (3) correct the estimation of $\rho_t$ after relaxing the assumption of a perfect OOD detector. The estimation results are then used to adjust the ID classifier via importance weighting.

## Method

### Overall Architecture
Given: Source domain ID labeled data, target domain unlabeled data, a frozen $K$-class ID classifier $f$, and a frozen ID/OOD detector $h$. An OOD reference dataset can optionally be provided. Output: Target domain label distribution estimates and the corrected classifier.

### Key Designs

1. **Source Domain OOD Proportion Estimation**:

    - **Function**: To obtain $\rho_s = p_s(b=1)$, representing the proportion of ID data in the source domain.
    - **Mechanism**: Utilizing the OOD reference dataset and source domain data, an estimator for $\rho_s$ is constructed using the prediction values of the OOD detector $h$. A sampling error upper bound based on concentration inequalities is provided.
    - **Design Motivation**: The EM algorithm requires $\rho_s$ as input; the OOD reference dataset serves as an anchor to calibrate the OOD detector.

2. **EM Algorithm for Target Distribution Estimation**:

    - **Function**: To jointly estimate the target ID label distribution $\pi$ and the ID proportion $\rho_t$.
    - **Mechanism**: Treating the labels of the target domain data as latent variables, the complete-data likelihood is constructed using the soft predictions of the ID classifier $f$ and the ID probabilities from the OOD detector $h$. The E-step calculates the label posteriors, and the M-step updates $\pi$ and $\rho_t$. A Dirichlet prior can optionally be incorporated to obtain a MAP estimate.
    - **Design Motivation**: The EM algorithm is a classic method for CSLS (e.g., MLLS). This paper extends it to the open-set setting that incorporates OOD classes.

3. **Correction for Imperfect OOD Detectors**:

    - **Function**: To relax the assumption of a perfect OOD detector.
    - **Mechanism**: When $h(x)$ does not represent the ground truth $p_s(b=1|x)$ (i.e., Assumption 3.3B does not hold), the estimate of $\rho_t$ becomes biased. A brand-new correction term is constructed using the statistics of $h$ on the OOD reference dataset and ID data to provide an error upper bound.
    - **Design Motivation**: In practice, OOD detectors are based on heuristics and cannot be perfect.

### Loss & Training
Training-free: The method runs entirely at inference time, utilizing the outputs of the pre-existing classifiers.

## Key Experimental Results

### Main Results (CIFAR10/100, ImageNet-200)

| Method | Label Estimation Error ↓ | Calibration Accuracy ↑ |
|------|------------|-----------|
| Garg et al. (Requires Retraining) | Moderate | Moderate |
| BBSE-OVA | Large | Low |
| **Ours (No Retraining)** | **Smallest** | **Highest** |

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| Perfect OOD Detector | Highest estimation accuracy |
| Imperfect OOD Detector + Correction | Performance close to perfect |
| Different OOD datasets as reference | Insensitive to reference data choice |
| MAP vs MLE | MAP is more stable with small sample sizes |

### Key Findings
- Outperforms retraining-based baselines without requiring any classifier retraining.
- Flexible selection of OOD detectors—any existing method in the literature can be directly applied.
- The correction step effectively mitigates the impact of an imperfect OOD detector.
- Consistent superiority is demonstrated across CIFAR10/100 and ImageNet-200 datasets.

## Highlights & Insights
- The "no-retraining" design significantly enhances practical utility, making it highly applicable in scenarios with frozen models, privacy constraints, or limited computational resources.
- Elegantly generalises classic CSLS methods (EM/MLE) to the open-set setting with rigorous theoretical derivations.
- The sampling error upper bound derived from concentration inequalities provides reliability guarantees.

## Limitations & Future Work
- Assumes that $p(x|y)$ remains invariant between the source and target domains (label shift assumption); covariate shift is out of scope.
- Requires an OOD reference dataset (though the choice is flexible).
- Estimation quality degrades if the original ID classifier performs poorly.
- Extension to scenarios that simultaneously handle label shift and covariate shift is a potential future direction.

## Related Work & Insights
- **vs Garg et al. (OSLS)**: Requires retraining the OOD classifier, while the proposed method directly utilizes existing OOD detectors.
- **vs MLLS/MAPLS (CSLS)**: The proposed method extends the closed-set EM methods to the open-set setting containing OOD classes.
- **vs BBSE**: BBSE solves via a linear system and does not handle OOD; the proposed EM method naturally resolves the OOD proportion.

## Rating
- Novelty: ⭐⭐⭐⭐ The extension from CSLS to OSLS is elegant, and the training-free design is a key advantage.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-OOD detectors, with rigorous theoretical guarantees.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear problem definition.
- Value: ⭐⭐⭐⭐ Highly practical for distribution shift issues in real-world deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sufficient Invariant Learning for Distribution Shift](sufficient_invariant_learning_for_distribution_shift.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](../../NeurIPS2025/others/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](../../ECCV2024/others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)

</div>

<!-- RELATED:END -->
