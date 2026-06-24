---
title: >-
  [Paper Note] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach
description: >-
  [AAAI 2026 Best Paper Oral][Medical Imaging][Developmental dysplasia of the hip] This paper proposes an "ultrasound-first, radiation-preserving" cross-modal selective imaging strategy. By combining a self-supervised pretrained frozen encoder, a measurement-faithful lightweight head network, and a conformal-prediction-calibrated one-sided lower bound, the framework provides principled decisions on when ultrasound alone suffices and when additional X-ray imaging is warranted fo…
tags:
  - "AAAI 2026 Best Paper Oral"
  - "Medical Imaging"
  - "Developmental dysplasia of the hip"
  - "ultrasound–X-ray cross-modal learning"
  - "conformal prediction"
  - "selective imaging"
  - "self-supervised learning"
date: 2026-05-08
content_hash: a153b43f5282eb71
---

# Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach

**Conference**: AAAI 2026 Best Paper Oral  
**arXiv**: [2511.18457](https://arxiv.org/abs/2511.18457)  
**Code**: None  
**Area**: Medical Imaging / Cross-Modal Learning / Selective Imaging
**Keywords**: Developmental dysplasia of the hip, ultrasound–X-ray cross-modal learning, conformal prediction, selective imaging, self-supervised learning

## TL;DR

This paper proposes an "ultrasound-first, radiation-preserving" cross-modal selective imaging strategy. By combining a self-supervised pretrained frozen encoder, a measurement-faithful lightweight head network, and a conformal-prediction-calibrated one-sided lower bound, the framework provides principled decisions on when ultrasound alone suffices and when additional X-ray imaging is warranted for diagnosing developmental dysplasia of the hip (DDH).

## Background & Motivation

**Developmental dysplasia of the hip (DDH)** is a common pediatric orthopedic condition encompassing a spectrum of pathologies ranging from insufficient acetabular coverage to complete hip dislocation. Two complementary imaging modalities are used clinically:

**Ultrasound (US)**: Used for early screening and management, reporting Graf α/β angles and femoral head coverage. Advantages include no ionizing radiation and broad accessibility.

**X-ray (XR)**: More appropriate for evaluating acetabular development and surgical planning after ossification has progressed, measuring the acetabular index (AI), center-edge angle (CE), and IHDI grade. The key disadvantage is exposure to ionizing radiation.

**Key Challenge**: In pediatric care, minimizing ionizing radiation exposure is a fundamental principle, yet relying solely on ultrasound risks missing abnormalities detectable only on X-ray. Clinically, a **clearly normal ultrasound** (high α angle, adequate coverage) generally implies low marginal diagnostic value from X-ray; an abnormal or borderline ultrasound raises the value of proceeding to X-ray.

**Limitations of Prior Work**:
- Automated ultrasound and automated X-ray analysis have each advanced independently (automatic standard plane detection, Graf measurement prediction, AI/CE angle estimation, etc.), but **cross-modal US–XR learning remains extremely scarce**.
- No practical, measurement-grounded strategy exists for trading off radiation risk against diagnostic gain with explicit statistical guarantees.
- Existing methods are largely black-box classifiers, lacking direct alignment with clinical decision thresholds.

The paper's motivation is to address this gap: **constructing a calibratable, interpretable, and tunable "ultrasound-first" strategy that uses conformal prediction to provide finite-sample, distribution-free coverage guarantees, enabling clinical teams to make transparent trade-offs between radiation exposure and the risk of missed findings.**

## Method

### Overall Architecture

The paper proposes a four-stage pipeline:

1. **Self-supervised pretraining**: Modality-specific ResNet-18 encoders are pretrained independently with SimSiam on large-scale unannotated registry data (37,186 ultrasound images + 19,546 X-ray images), then frozen.
2. **Measurement-faithful head networks**: Lightweight MLP heads are trained on top of the frozen encoders to directly predict clinically used named measurements (α/β angles, coverage, AI, CE, IHDI, etc.).
3. **Conformal calibration**: An affine bias correction is fitted on a calibration set, and one-sided conformal residual quantiles are computed, providing finite-sample marginal coverage guarantees for ultrasound predictions.
4. **Selective imaging strategy**: Calibrated ultrasound lower bounds are compared against clinical thresholds to decide "ultrasound only" versus "refer to X-ray," with decision curve analysis used to trade off radiation cost against missed-finding penalty.

### Key Designs

1. **SimSiam self-supervised pretraining**: An encoder $f_\phi$ is trained independently for each modality. Let the projection MLP be $h_\phi$ and the prediction MLP be $q_\phi$. Two augmented views $v_1, v_2$ are generated from image $x$, and the loss is the stop-gradient negative cosine similarity:
$$\mathcal{L}_{SSL} = -\frac{1}{2}\left[\frac{\langle p_1, \text{sg}(z_2)\rangle}{\|p_1\| \|\text{sg}(z_2)\|} + \frac{\langle p_2, \text{sg}(z_1)\rangle}{\|p_2\| \|\text{sg}(z_1)\|}\right]$$
After 10 epochs of training, the projection and prediction heads are discarded and the encoder is frozen. **Design Motivation**: To leverage large volumes of unannotated imaging data for learning general visual representations, compensating for the scarcity of annotated samples.

2. **Measurement-faithful head networks**: A 512-dimensional global average-pooled feature $u = f_\phi(x)$ is extracted from the frozen encoder, and a small MLP is trained on top:

    - **Ultrasound head**: Single hidden layer (128 units), 3 outputs predicting $\hat{\alpha}, \hat{\beta}, \widehat{\text{cov}}$, with loss $\mathcal{L}_{US} = \lambda_\alpha |\hat{\alpha} - y^\alpha| + \lambda_\beta |\hat{\beta} - y^\beta| + \lambda_{\text{cov}} |\widehat{\text{cov}} - y^{\text{cov}}|$
    - **X-ray head**: Predicts AI and CE angles (MAE loss) with optional IHDI classification (cross-entropy)

   The core design philosophy is **measurement faithfulness** — outputs are the named measurements used in daily clinical practice rather than black-box classifications, making the decision process fully traceable.

3. **Conformal-calibrated one-sided lower bounds**: Constructed in two steps:

    - (i) **Affine bias correction**: $\tilde{y}^t = a_t \hat{y}^t + b_t$ is fitted on the calibration set to reduce systematic bias without retraining the head.
    - (ii) **One-sided residual quantile**: Residuals $r_i^t = y_i^t - \tilde{y}_i^t$ are computed, and a conformal radius $q_t^+(\delta_t)$ is calculated at miscoverage level $\delta_t$, ensuring that under exchangeability $y^t \geq \text{LB}_t(x; \delta_t)$ holds with probability $\geq 1 - \delta_t$. The calibrated lower bound is $\text{LB}_t(x; \delta_t) = \tilde{y}^t(x) - q_t^+(\delta_t)$.

4. **Three selective imaging rules**: Based on comparing calibrated lower bounds against clinical thresholds $T_\alpha = 60°$ and $T_{\text{cov}} = 50\%$:

    - **Alpha-only**: $d_\alpha(x) = \mathbb{I}[\text{LB}_\alpha(x) \geq 60°]$
    - **Alpha OR Coverage**: Either lower bound exceeding its threshold is sufficient to proceed with ultrasound alone
    - **Alpha AND Coverage**: Both lower bounds must exceed their thresholds to proceed with ultrasound alone

   Sweeping over a grid of $(\delta_\alpha, \delta_{\text{cov}})$ values generates a family of policies. Conservative settings (small $\delta$) provide high coverage but low ultrasound pass rates; lenient settings (large $\delta$) increase the pass rate at the cost of higher missed-finding risk.

### Loss & Training

- Self-supervised stage: SimSiam negative cosine similarity loss, 10 epochs per modality
- Supervised head networks: MAE loss for ultrasound (equal weights $\lambda_\alpha = \lambda_\beta = \lambda_{\text{cov}} = 1$), MAE + cross-entropy for X-ray
- Head networks are trained exclusively on the post-train set (30 subjects, 136 images); the encoder remains frozen throughout
- The calibration set (7 subjects, 28 images) is used only for bias correction and conformal calibration, not for training
- Data splits are strictly performed at the subject level to prevent leakage

## Key Experimental Results

### Main Results

Dataset: 75 paired subjects and 321 images drawn from a large registry. The evaluation set consists of 38 subjects (157 images), yielding N=77 strictly matched hip pairs after pairing.

| Modality | Measurement | MAE | Notes |
|----------|-------------|:---:|-------|
| Ultrasound | α angle | 9.69° | Frozen encoder + lightweight head |
| Ultrasound | β angle | 11.25° | Same |
| Ultrasound | Femoral head coverage | 13.97 pp | Percentage points |
| X-ray | Acetabular index (AI) | 7.60° | Frozen encoder + lightweight head |
| X-ray | Center-edge angle (CE) | 8.93° | Same |

**Selective imaging strategy results** (N=77 strictly matched pairs):

| Rule | Miscoverage $\delta_\alpha / \delta_{\text{cov}}$ | US Pass Rate | X-ray Usage Rate |
|------|:---:|:---:|:---:|
| AND | 0.10 / 0.10 | 0.00 | 1.00 |
| AND | 0.20 / 0.20 | 0.00 | 1.00 |
| OR | 0.35 / 0.35 | 0.43 | 0.57 |
| OR | 0.40 / 0.40 | 0.55 | 0.45 |

### Ablation Study

| Configuration | Key Result | Notes |
|---------------|-----------|-------|
| Conservative AND 0.10/0.10 | Coverage ~0.90 (α), ~0.94 (cov) | Safe but nearly all cases referred to X-ray |
| Lenient OR 0.40/0.40 | US pass rate 55% | Substantially reduces X-ray use but increases missed-finding risk |
| Calibration radius (δ=0.10) | $q_\alpha^+ = 10.75°$, $q_{\text{cov}}^+ = 28.74$ pp | Conservative lower bounds |
| Decision curve analysis | OR strategy optimal at high radiation cost λ | Explicit radiation vs. safety trade-off |

### Key Findings

1. **Frozen encoder + lightweight head is competitive**: A small MLP head trained on only 30 subjects atop a frozen self-supervised encoder achieves measurement accuracy comparable to single-modality methods using larger networks.
2. **Conformal coverage follows theoretical expectations**: Empirical coverage varies monotonically with the miscoverage level $\delta$, validating the finite-sample guarantees in practice.
3. **Three rule families provide flexibility**: The AND rule is most conservative (nearly all cases referred to X-ray), the OR rule is most lenient (roughly half of X-rays can be saved), and the Alpha-only rule falls in between.
4. **Borderline cases are automatically referred to X-ray**: Many borderline hips have α angles within a few degrees of the 60° threshold; the system does not force an ultrasound-only decision for these cases but instead refers them to X-ray, consistent with the behavior of experienced clinicians when uncertain.

## Highlights & Insights

- **Measurement-faithful rather than black-box**: The entire reasoning chain is visible — landmark and point annotations → derived measurements → calibrated lower bounds → rule evaluation → policy outcome. There are no hidden logits or opaque multi-class outputs controlling safety-critical decisions, which is essential for clinical deployment.
- **Label-efficient pipeline**: Self-supervised pretraining leverages large volumes of unannotated data to learn representations; with the encoder frozen, only minimal annotations (30 subjects) are needed to train the head, perfectly suited to the reality of expensive annotation in medical imaging.
- **Conformal prediction provides distribution-free guarantees**: No assumptions about the data distribution are required; finite-sample coverage control is achieved under the exchangeability condition alone, making this more robust than conventional confidence intervals.
- **Decision curves explicitly expose trade-offs**: By parameterizing radiation cost $\lambda$ and missed-finding penalty $\mu$, modeling is decoupled from policy, placing radiation management decisions in the hands of the clinical team.
- **Fail-safe design**: Uncertain cases are automatically referred to X-ray rather than forced through an ultrasound-only decision; the one-sided conformal lower bound guarantees coverage control.

## Limitations & Future Work

1. **Small data scale**: Annotated data comprise only 75 subjects, and the calibration set only 7 subjects (26 ultrasound images), which may result in wide conformal radii. Larger-scale, multi-center validation is needed.
2. **Trainee annotators**: Measurement annotations were produced by trainees rather than senior experts, potentially introducing annotation noise.
3. **Age- and ossification-aware thresholds not implemented**: Although an ossification nucleus landmark interface is provisioned, age-dependent differentiated thresholds are not actually implemented.
4. **No prospective clinical study**: Validation in actual clinical workflows — including radiation savings per 100 infants and changes in recall rates — is still needed.
5. **Cross-modal proxy model not completed**: The paper mentions the possibility of predicting X-ray AI or IHDI risk from ultrasound measurements as a proxy, but this extension is not implemented.
6. **High MAE for femoral head coverage**: The MAE for femoral head coverage is approximately 14 percentage points, leaving room for improvement.

## Related Work & Insights

This paper is an excellent example of applying **statistical decision theory** (conformal prediction, decision curve analysis) to **clinical imaging workflow optimization**. Rather than pursuing peak accuracy on a single modality, it integrates cross-modal information into an **actionable selective decision framework**. This methodology generalizes to other clinical scenarios involving the question of "is this expensive test worth performing?" — for instance, performing a low-cost screening test first (blood panel, simple imaging) before deciding whether to proceed with an expensive or risky follow-up examination (contrast imaging, biopsy, etc.). The application of conformal prediction in medical AI is an emerging area, and this paper demonstrates a practical path for integrating it with clinical decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐ — First cross-modal selective imaging strategy grounded in conformal prediction
- Experimental Thoroughness: ⭐⭐⭐ — Small data scale; lacks external validation and prospective studies
- Value: ⭐⭐⭐⭐ — Highly actionable in clinical workflows; decisions are transparent and auditable
- Writing Quality: ⭐⭐⭐⭐⭐ — Methodology is articulated with exceptional clarity; clinical motivation and technical solution are tightly integrated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](learning_with_preserving_for_continual_multitask_learning.md)
- [\[ICML 2025\] Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing](../../ICML2025/medical_imaging/enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza.md)
- [\[ICLR 2026\] sleep2vec: Unified Cross-Modal Alignment for Heterogeneous Nocturnal Biosignals](../../ICLR2026/medical_imaging/sleep2vec_unified_cross-modal_alignment_for_heterogeneous_nocturnal_biosignals.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)

</div>

<!-- RELATED:END -->
