---
title: >-
  [Paper Note] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis
description: >-
  [NeurIPS 2025 (TS4H Workshop)][Medical Imaging][Survival Analysis] This paper proposes a Dual Mixture-of-Experts (Dual MoE) framework for discrete-time survival analysis…
tags:
  - "NeurIPS 2025 (TS4H Workshop)"
  - "Medical Imaging"
  - "Survival Analysis"
  - "Mixture-of-Experts"
  - "Discrete Time"
  - "Patient Heterogeneity"
  - "Breast Cancer"
date: 2026-05-08
content_hash: d1712ecbcdaeb657
---

# Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis

**Conference**: NeurIPS 2025 (TS4H Workshop)
**arXiv**: [2510.26014](https://arxiv.org/abs/2510.26014)  
**Code**: N/A  
**Area**: Medical Imaging
**Keywords**: Survival Analysis, Mixture-of-Experts, Discrete Time, Patient Heterogeneity, Breast Cancer

## TL;DR
This paper proposes a Dual Mixture-of-Experts (Dual MoE) framework for discrete-time survival analysis, combining a feature encoder MoE (for modeling patient subgroup heterogeneity) with a hazard network MoE (for capturing temporal dynamics). The framework achieves improvements of up to 0.04 in time-dependent C-index on the METABRIC and GBSG breast cancer datasets.

## Background & Motivation

**Background**: Survival analysis aims to predict the time-to-event and is widely used in clinical and biomedical research. The classical Cox Proportional Hazards (CPH) model assumes a time-constant hazard ratio. Deep learning approaches (e.g., DeepHit, ConSurv) replace the CPH constraint with flexible neural networks to model non-proportional hazards.

**Limitations of Prior Work**: Most deep survival models still rely on a single shared feature encoder and a single hazard network. However, patient populations exhibit heterogeneous subgroups (e.g., ER+/ER−, HER2+/HER2−), and a single encoder tends to be biased toward dominant patterns. Moreover, hazards vary not only across patients but also over time—different patients may exhibit fundamentally different risk trajectories at the same time point.

**Key Challenge**: A single network binds all patients and all time intervals to one shared functional form, which is insufficient to capture the interplay between patient heterogeneity and temporal dynamics.

**Goal**: How can patient subgroup differences and temporal variation be modeled simultaneously at both the feature encoding and risk prediction levels?

**Key Insight**: Introduce Mixture-of-Experts (MoE) into survival analysis, employing multiple expert networks at both the feature encoding and hazard prediction stages, with soft routing enabling subgroup-aware and time-aware modeling.

**Core Idea**: Apply MoE independently at the feature encoding and hazard prediction stages, using patient-feature-driven routing and joint patient-plus-time-driven routing to achieve fine-grained survival modeling.

## Method

### Overall Architecture
The inputs consist of patient covariates $x_i$, observed time $\tau_i$, and event indicator $\delta_i$. The model operates in two stages: (1) **Feature Encoder MoE** — an initial encoder extracts a base representation, which is then assigned via soft routing to $K$ expert encoders; the weighted combination yields a subgroup-aware patient representation; (2) **Hazard Network MoE** — $L$ hazard experts are jointly routed based on patient features and time embeddings, producing per-interval hazard estimates. Both MoE stages are trained end-to-end with a negative log-likelihood loss and load-balancing regularization.

### Key Designs

1. **Feature Encoder MoE**:

    - **Function**: Learns subgroup-aware patient representations via multiple expert encoders.
    - **Mechanism**: An initial encoder $g(\cdot)$ extracts base features; a router generates softmax routing probabilities $\pi_k^{feat}$ from patient features; the final representation is the weighted sum of all expert outputs:
    $z(x) = \sum_{k=1}^{K} \pi_k^{feat} \cdot f_k(g(x))$
    - **Design Motivation**: Different subgroups (e.g., ER+/− or HER2+/−) exhibit distinct risk feature distributions; a single encoder cannot simultaneously optimize representations for all subgroups.

2. **Hazard Network MoE**:

    - **Function**: Captures fine-grained patient–time interaction risk patterns via multiple expert hazard networks.
    - **Mechanism**: Each hazard expert $h_l$ predicts hazards across all time intervals; the router jointly receives patient features and learnable time embeddings $e_t$ to determine routing probabilities $\pi_{t,l}^{haz}$:
    $\lambda(t|x) = \sum_{l=1}^{L} \pi_{t,l}^{haz} \cdot h_l(z(x), e_t)$
    - **Design Motivation**: This enables experts to specialize along both the time and patient dimensions simultaneously—for instance, certain experts may focus on subgroups with elevated early-stage risk, while others handle late-stage risk subgroups.

3. **Load-Balancing Regularization**:

    - **Function**: Prevents routing collapse onto a small number of experts.
    - **Mechanism**: Feature MoE balancing loss $\mathcal{L}_{LB}^{feat} = \alpha(K \sum_k \bar{\pi}_k^{feat^2} - 1)$; hazard MoE balancing loss $\mathcal{L}_{LB}^{haz} = \beta(T_{max} \sum_t L \sum_l \bar{\pi}_{t,l}^{haz^2} - 1)$.
    - **Design Motivation**: Uneven expert utilization is a common failure mode in MoE; without constraints, the router tends to converge to using only 1–2 experts.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{NLL} + \mathcal{L}_{LB}^{feat} + \mathcal{L}_{LB}^{haz}$
- Negative log-likelihood: $\mathcal{L}_{NLL} = -\sum_{i=1}^{N}[\delta_i \log \hat{p}(\tau_i|x_i) + (1-\delta_i) \log \hat{S}(\tau_i|x_i)]$
- Where $\hat{p}(t|x) = \lambda(t|x) S(t-1|x)$, $S(t|x) = \prod_{t' \leq t}(1-\lambda(t'|x))$
- METABRIC uses $(K=4, L=4)$ experts; GBSG uses $(K=6, L=3)$ experts.
- Results are averaged over 10 random seeds.

## Key Experimental Results

### Main Results (C-index)

| Dataset | Method | Dual MoE | C-index | Td-C 10% | Td-C 50% | Td-C 90% |
|--------|------|---------|---------|----------|----------|----------|
| METABRIC | Naïve impl. | ✗ | 0.646 | 0.670 | 0.638 | 0.606 |
| METABRIC | Naïve impl. | ✓ | **0.654** | 0.669 | **0.646** | **0.623** |
| METABRIC | ConSurv | ✗ | 0.657 | 0.656 | 0.649 | 0.617 |
| METABRIC | ConSurv | ✓ | **0.668** | **0.696** | **0.657** | **0.634** |
| GBSG | Naïve impl. | ✗ | 0.662 | 0.744 | 0.669 | 0.652 |
| GBSG | Naïve impl. | ✓ | **0.667** | **0.751** | **0.677** | **0.659** |
| GBSG | ConSurv | ✗ | 0.665 | 0.742 | 0.674 | 0.658 |
| GBSG | ConSurv | ✓ | **0.668** | **0.752** | **0.677** | **0.659** |

### Ablation Study (Inferred from Visualizations)

| Analysis Dimension | Key Findings |
|---------|---------|
| Feature routing visualization | ER+/ER− and HER2+/HER2− subgroups exhibit clearly distinct expert preference distributions, validating that the router can automatically discover subgroup structure. |
| Hazard routing trajectories | Hazard expert assignments vary substantially over time across different patients, with dominant experts differing between early and late time points, demonstrating that the model captures temporal dynamics. |
| Baseline comparison | CoxPH (0.663) outperforms Naïve impl. (0.646), indicating that deep models are prone to overfitting on small datasets; Dual MoE mitigates this issue. |

### Key Findings
- Dual MoE **consistently improves** performance across all settings, with a maximum gain of +0.04 in time-dependent C-index (METABRIC ConSurv at the 10% time point).
- The improvement is more pronounced when combined with ConSurv, indicating that Dual MoE functions as a general plug-in module that can be flexibly integrated into existing deep survival analysis pipelines.
- The feature router automatically discovers clinically meaningful subgroups (ER/HER2 status differences) without requiring explicit subgroup labels.
- The hazard router exhibits time-dependent expert specialization—different time intervals are dominated by different experts—consistent with the intuition of non-proportional hazards.
- On GBSG, the Naïve impl. without MoE already approaches CoxPH, suggesting that simple deep models are competitive on this dataset.

## Highlights & Insights
- **Dual-level MoE design philosophy**: The feature MoE captures *who differs* (inter-patient heterogeneity), while the hazard MoE captures *when it differs* (temporal dynamics); the two are orthogonal and complementary.
- **Strong interpretability**: Visualization of routing probabilities directly reveals subgroup structure and temporal dynamics, offering greater clinical interpretability than black-box models.
- **Plug-and-play**: The framework can directly replace the encoder and hazard head of existing deep survival models without modifying the loss function or training procedure.
- **Time embeddings as router input**: Incorporating learnable time embeddings into the hazard router is an elegant design that enables expert specialization along the time axis.

## Limitations & Future Work
- Validation is limited to two breast cancer datasets; evaluation on additional disease types and larger-scale datasets is needed.
- As a workshop paper, ablation experiments are insufficiently detailed—the independent contributions of the feature MoE and hazard MoE are not disentangled.
- The choice of expert counts $(K, L)$ appears to require per-dataset tuning, and no adaptive mechanism is provided.
- No comparison is made against other heterogeneity-modeling approaches (e.g., mixture cure models, random effects models).
- The framework could be extended to multimodal (e.g., imaging + clinical) survival analysis scenarios.

## Related Work & Insights
- **vs. CoxPH**: The classical method assumes proportional hazards and cannot handle non-proportional hazard scenarios; this paper addresses this limitation through a flexible MoE hazard network.
- **vs. DeepHit / ConSurv**: These methods use a single encoder and hazard head; this paper introduces MoE at both levels to enhance modeling capacity, and can additionally serve as a plug-in for ConSurv.
- **vs. Standard MoE (Shazeer et al.)**: Standard MoE is applied at a single level; this paper innovatively employs MoE at both the encoding and prediction stages, with the hazard MoE router additionally conditioned on time embeddings.
- The approach has direct relevance to survival prediction in pathology and radiomics (e.g., prognosis from whole-slide images).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Simultaneously introducing dual-level MoE into survival analysis is a novel combination; the time-conditioned routing design in the hazard MoE is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ As a workshop paper, evaluation is limited to two datasets with insufficient ablation studies and baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with complete formulations; visualizations effectively support the motivation.
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play general module with direct applicability to clinical survival prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICML 2026\] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](../../ICML2026/medical_imaging/eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)
- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)

</div>

<!-- RELATED:END -->
