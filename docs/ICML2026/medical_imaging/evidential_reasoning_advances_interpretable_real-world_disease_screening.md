---
title: >-
  [Paper Note] Evidential Reasoning Advances Interpretable Real-World Disease Screening
description: >-
  [ICML 2026][Medical Imaging][Dual Knowledge Banks] EviScreen utilizes "normal + pathological" dual knowledge banks for region-level evidence retrieval…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Dual Knowledge Banks"
  - "Evidential Reasoning"
  - "Coreset Memory Bank"
  - "Contrastive Retrieval"
  - "Clinical-Oriented Evaluation"
date: 2026-05-08
content_hash: f9df814b42b96c07
---

# Evidential Reasoning Advances Interpretable Real-World Disease Screening

**Conference**: ICML 2026  
**arXiv**: [2605.15171](https://arxiv.org/abs/2605.15171)  
**Code**: https://github.com/DopamineLcy/EviScreen (Available)  
**Area**: Medical Imaging / Explainable AI / Anomaly Detection  
**Keywords**: Dual Knowledge Banks, Evidential Reasoning, Coreset Memory Bank, Contrastive Retrieval, Clinical-Oriented Evaluation

## TL;DR
EviScreen utilizes "normal + pathological" dual knowledge banks for region-level evidence retrieval, followed by evidential reasoning between the current case and evidence using cross-attention and self-attention. It provides both **retrospective interpretability** (identifying which historical cases support the current judgment) and **localization interpretability** (anomaly maps obtained through contrastive retrieval), improving specificity at high recall across four real-world external test sets to SOTA levels.

## Background & Motivation

**Background**: Current disease screening in medical imaging follows two mainstreams: (a) Deviation-based prediction (unsupervised anomaly detection, e.g., PatchCore, SimpleNet), which models only normal samples and alerts upon deviation; (b) Direct prediction (full supervision for binary classification), using Grad-CAM for post-hoc localization.

**Limitations of Prior Work**: (a) These methods fail to utilize the rich information in pathological samples and have limited capability for complex modalities (e.g., chest X-rays, dermoscopy); (b) Post-hoc maps like Grad-CAM have been proven by multiple studies to have poor localization quality and fail to explain "why this region looks like a lesion"—lacking **evidential reasoning**. Prototype methods (e.g., ProtoPNet) use a fixed number of prototypes to represent preset classes, which lacks scalability and fails to cover the diverse morphologies in real clinical settings.

**Key Challenge**: Clinicians make decisions by "retrieving and comparing similar past cases." Existing models either ignore historical cases or rely on "learned abstract prototypes," which is disconnected from the actual diagnostic process.

**Goal**: (1) Design a screening framework capable of "retrieving region-level evidence from a scalable case bank" like a doctor; (2) Establish real-world clinical evaluations (external testing + specificity at high recall).

**Key Insight**: Anomaly screening is reformulated into a two-stage "retrieval + reasoning" process. A foundation model extracts region features to construct two coreset knowledge banks (normal and pathological). Each patch of the query image undergoes $k$-NN retrieval within both banks, and the model subsequently performs attention-based reasoning using these evidence tokens.

**Core Idea**: By using "normal vs. pathological" dual knowledge banks for **contrastive retrieval**, the model provides both localization (anomaly map $\mathbf M = \text{ReLU}(\mathbf M_N - \mathbf M_P)$) and reasoning (cross-attention integrating evidence into query features), transforming "post-hoc Grad-CAM" into an "inherent evidence flow."

## Method

### Overall Architecture
The framework consists of two stages. **Stage 1: Dual Knowledge Bank Construction**: The training set is split into a "bank subset" $\mathcal X^B_{N/P}$ and a "training subset" $\mathcal X^R_{N/P}$. A frozen foundation model $F_\theta$ extracts intermediate patch features, which are locally aggregated via $\mathcal G_{agg}$ to form region feature sets $\mathcal S_{N/P}$. Greedy coreset subsampling (an NP-hard approximation minimizing $\max_m\min_n\|m-n\|_2$) is used to compress these into compact knowledge banks $\mathcal K_N, \mathcal K_P$.

**Stage 2: Reasoning**: For a query image $\mathbf x$, the feature map $\mathbf Z=\mathcal G_{agg}(F_\theta(\mathbf x))\in\mathbb R^{h\times w\times d}$ is extracted. Each patch query $\mathbf Z(i,j)$ performs $k$-NN retrieval in $\mathcal K_N$ and $\mathcal K_P$ to obtain evidence $\mathbf E_N, \mathbf E_P\in\mathbb R^{h\times w\times k\times d}$. These enter the "Evidence-Aware Reasoning" module for layer-wise cross-attention (query $\leftrightarrow$ evidence) and self-attention (inter-patch refinement). Finally, the [CLS] tokens of both branches are concatenated and fed into an MLP for prediction $\hat y$. A **training-free variant** is also provided, where scores are directly pooled from the contrastive retrieval map $\mathbf M = \text{ReLU}(\mathbf M_N - \mathbf M_P)$.

### Key Designs

1.  **Dual Knowledge Bank**:
    *   **Function**: Represents the diverse morphologies of "normal regions" and "pathological regions" in a compact, scalable manner.
    *   **Mechanism**: Patch features are extracted from $\mathcal X^B_N, \mathcal X^B_P$, aggregated locally, and subjected to greedy coreset subsampling to obtain $\mathcal K_N, \mathcal K_P$. The optimization objective $\mathcal K^*=\arg\min_{\mathcal K\subset\mathcal S}\max_{m\in\mathcal S}\min_{n\in\mathcal K}\|m-n\|_2$ is a classic NP-hard problem, solved via iterative greedy approximation to obtain a representative subset.
    *   **Design Motivation**: Compared to prototype-based methods with a fixed $K$, the coreset capacity scales with data, covering varied clinical lesions. Coreset handles "dirty" pathological data (mixture of normal and lesion tissue) better than class-predefined prototypes.

2.  **Evidence-Aware Reasoning**:
    *   **Function**: Integrates $k$ retrieved evidence vectors into query features to output new representations informed by historical comparisons.
    *   **Mechanism**: Each layer first performs cross-attention: $\mathbf T^\ell_N(i,j)=\operatorname{softmax}\big(\mathbf Z^\ell_N(i,j)\mathbf E_N(i,j)^\top/\sqrt d\big)\mathbf E_N(i,j)$; then reshapes for inter-patch self-attention. The normal branch $\mathbf Z_N$ and pathological branch $\mathbf Z_P$ run in parallel, followed by $\hat y=\text{MLP}([\mathbf Z_N^{\text{CLS}};\mathbf Z_P^{\text{CLS}}])$.
    *   **Design Motivation**: Cross-attention explicitly writes "external evidence" into feature maps, providing traceable nearest-neighbor (NN) evidence (retrospective interpretability). Dual branches preserve two-dimensional signals (similarity to normal vs. pathological), aligning with clinical "differential diagnosis" intuition.

3.  **Contrastive Retrieval Anomaly Map (Training-Free Variant)**:
    *   **Function**: Provides pixel-level anomaly localization without parameter training, serving as a strong baseline and interpretability foundation.
    *   **Mechanism**: Average distances to $k$-NN in both banks are computed as $\mathbf M_N, \mathbf M_P\in\mathbb R^{h\times w}$. The anomaly map is $\mathbf M(i,j)=\text{ReLU}(\mathbf M_N(i,j)-\mathbf M_P(i,j))$—meaning only regions "far from normal but close to pathological" are highlighted.
    *   **Design Motivation**: PatchCore-like methods using only $\mathbf M_N$ often misclassify rare normal variations. Contrastive differencing filters out non-pathological deviations and provides an inherent localization map mechanically distinct from Grad-CAM.

### Loss & Training
Training: Only the "Evidence-Aware Reasoning" module (cross/self-attention + MLP) is trained; the foundation model remains frozen. The loss is binary Cross-Entropy (BCE). The training-free variant has no learnable parameters. Primary hyperparameters include $k$, bank size, and number of cross-attention layers $L$.

## Key Experimental Results

### Main Results
Evaluated on 10 public datasets across three modalities (fundus, chest X-ray, dermoscopy), focusing on four **external** test sets: JSIEC, RIADD, CheXpert, Derm12345. Clinical metrics: AUROC, AP, Spe@95%R (specificity at 95% recall), Spe@99%R. Results (percentage):

| Metric | EviScreen | EviScreen-TF (Training-free) | FM | PatchCore* | PatchCore | NFM-DRA | DRA | SCRD4AD | EDC | SimpleNet | CIPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUROC | **98.06** | 96.76 | 95.84 | 94.96 | 92.12 | 95.53 | 92.53 | 94.88 | 79.12 | 73.73 | 94.83 |
| AP | **96.10** | 94.20 | 94.24 | 89.61 | 86.62 | 93.23 | 89.53 | 89.85 | 71.44 | 57.66 | 91.36 |
| Spe@95%R | **94.74** | 91.48 | 87.95 | 87.26 | 81.09 | 90.37 | 80.12 | 88.50 | 51.45 | 53.81 | 87.33 |
| Spe@99%R | **91.62** | 87.74 | 79.29 | 83.31 | — | — | — | — | — | — | — |

### Ablation Study

| Configuration | Key Effect | Description |
|---|---|---|
| Full EviScreen | Best across all metrics | Dual banks + reasoning module |
| Remove Pathological Bank (Use $\mathcal K_N$ only) | Drop in Spe@95%R | Validates dual bank necessity |
| Use Fixed Prototypes (ProtoPNet style) | Performance drop | Coreset offers larger capacity than fixed prototypes |
| Cross-attention only (Remove self-attention) | Slight drop | Inter-patch refinement provides context consistency |
| Training-free variant | Outperforms PatchCore | Contrastive retrieval alone is a strong baseline |

### Key Findings
- **Larger Gaps in Clinical Metrics**: Compared to AUROC, the proposed method shows more significant leads in Spe@95%R and Spe@99%R (e.g., Spe@99%R is 8.3 points higher than PatchCore*), highlighting the advantages of evidential reasoning in high-recall zones critical for clinical use.
- **Training-Free Variant Beats PatchCore**: Contrastive retrieval alone outperforms traditional anomaly detection baselines, validating the "dual bank + contrast" concept.
- **Scalability**: Coreset size can expand linearly, facilitating the inclusion of new cases—a feat unreachable for prototype methods.

## Highlights & Insights
- **Dual-Track Interpretability**: Provides both retrospection and localization explanations—closer to a radiologist's diagnostic workflow than a single Grad-CAM heatmap.
- **Clinical-Oriented Evaluation Framework**: centered on 10 datasets, external testing, and Spe@high-Recall, this is one of the few designs truly aimed at clinical deployment.
- **Transferable Paradigm**: The "Contrastive Retrieval + Attention Fusion" paradigm can be migrated to other domains (e.g., industrial defect detection, satellite change detection) requiring "normal vs. abnormal" differentiation.

## Limitations & Future Work
- Constructing dual banks requires sufficient and representative pathological samples; "sparse pathology banks" for rare diseases may cause contrastive retrieval to fail.
- The foundation model remains frozen, creating a performance ceiling; updating with domain-specific foundation models may yield gains.
- Performing $k$-NN for every patch introduces latency with large banks; combining coreset growth with approximate NN acceleration is a potential expansion.
- Current loss does not explicitly supervise "evidence-prediction consistency"; future work could add contrastive terms to ensure retrieved evidence dominates prediction.

## Related Work & Insights
- **vs. PatchCore / SimpleNet**: Also uses coreset memory banks but only for normal data; EviScreen introduces the pathological bank for more precise contrastive localization.
- **vs. ProtoPNet / Prototype Methods**: Fixed prototype counts are limited by predefined classes; coreset capacity is flexible and provides broader coverage.
- **vs. Grad-CAM Post-hoc Interpretation**: Interpretations in this work are "inherent" (from retrieval + cross-attention), not dependent on gradient visualization tricks, resulting in higher stability and quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual coresets, contrastive retrieval, and dual-track interpretability is a novel and complete solution for medical screening.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 10 datasets, 3 modalities, external testing, clinical metrics, and training-free ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with listed limitations and contributions; Figure 1 effectively compares the three paradigms.
- Value: ⭐⭐⭐⭐⭐ Provides a deployable clinical-oriented pipeline and evaluation protocol, contributing both a methodology and a benchmark to the medical imaging community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Experience with Single Domain Generalization in Real World Medical Imaging Deployments](../../AAAI2026/medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)
- [\[ICML 2026\] Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[AAAI 2026\] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](../../AAAI2026/medical_imaging/deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)
- [\[ICML 2026\] PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)

</div>

<!-- RELATED:END -->
