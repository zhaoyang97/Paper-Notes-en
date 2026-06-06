---
title: >-
  [Paper Note] Evidential Reasoning Advances Interpretable Real-World Disease Screening
description: >-
  [ICML 2026][Medical Imaging][Dual Knowledge Bank] EviScreen employs a "normal + pathological" dual knowledge bank for region-level evidence retrieval…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Dual Knowledge Bank"
  - "Evidential Reasoning"
  - "Coreset Memory Bank"
  - "Contrastive Retrieval"
  - "Clinically-Oriented Evaluation"
date: 2026-05-08
content_hash: 91e72f4598c1b203
---

# Evidential Reasoning Advances Interpretable Real-World Disease Screening

**Conference**: ICML 2026  
**arXiv**: [2605.15171](https://arxiv.org/abs/2605.15171)  
**Code**: https://github.com/DopamineLcy/EviScreen (available)  
**Area**: Medical Imaging / Interpretable AI / Anomaly Detection  
**Keywords**: Dual Knowledge Bank, Evidential Reasoning, Coreset Memory Bank, Contrastive Retrieval, Clinically-Oriented Evaluation

## TL;DR
EviScreen employs a "normal + pathological" dual knowledge bank for region-level evidence retrieval, then performs evidential reasoning between the current case and retrieved evidence via cross-attention and self-attention. This provides both **retrospective interpretability** (which historical cases support the current decision) and **localization interpretability** (abnormality maps from contrastive retrieval). On four real-world external test sets, it achieves SOTA specificity at high recall.

## Background & Motivation

**Background**: There are two mainstream approaches for disease screening in medical imaging: (a) Deviation-based prediction (unsupervised anomaly detection, e.g., PatchCore, SimpleNet) models only normal samples and flags deviations; (b) Direct prediction (binary classification, fully supervised), with post-hoc localization via Grad-CAM.

**Limitations of Prior Work**: (a) Pathological samples' rich information is not utilized, limiting performance on complex modalities (e.g., chest X-ray, dermoscopy); (b) Grad-CAM post-hoc maps have been shown to have poor localization quality and cannot explain "why this region appears abnormal"—lacking **evidential reasoning**. Prototype-based methods (e.g., ProtoPNet) use a fixed number of prototypes per class, which lack scalability and cannot cover the diversity of real clinical cases.

**Key Challenge**: Clinicians make decisions by "retrieving similar historical cases for comparison"; existing models either ignore historical cases or rely on "dozens of learned abstract prototypes"—disconnected from real diagnostic workflows.

**Goal**: (1) Design a screening framework that, like clinicians, can "retrieve similar region-level evidence from a scalable case bank"; (2) Establish clinically-oriented evaluation (external testing + specificity at high recall).

**Key Insight**: Reframe anomaly screening as a two-stage "retrieval + reasoning" process—extract region features using a foundation model, construct normal and pathological coreset knowledge banks; for each patch in the query image, perform $k$-NN retrieval in both banks, then use attention-based reasoning over these evidence tokens.

**Core Idea**: Using "normal vs pathological" dual knowledge banks for **contrastive retrieval** enables both localization (abnormality map $\mathbf M = \text{ReLU}(\mathbf M_N - \mathbf M_P)$) and reasoning (cross-attention integrates evidence into the query feature), turning "post-hoc Grad-CAM" into "built-in evidence flow".

## Method

### Overall Architecture
Two stages. **Stage 1: Dual Knowledge Bank Construction**: Split the training set into a "bank subset" $\mathcal X^B_{N/P}$ and a "training subset" $\mathcal X^R_{N/P}$; extract intermediate patch features using a frozen foundation model $F_\theta$, then aggregate locally via $\mathcal G_{agg}$ to form region feature sets $\mathcal S_{N/P}$; apply greedy coreset subsampling (approximating the NP-hard problem of minimizing $\max_m\min_n\|m-n\|_2$) to obtain compact knowledge banks $\mathcal K_N,\mathcal K_P$.

**Stage 2: Reasoning**: For a query image $\mathbf x$, extract feature map $\mathbf Z=\mathcal G_{agg}(F_\theta(\mathbf x))\in\mathbb R^{h\times w\times d}$; for each patch query $\mathbf Z(i,j)$, perform $k$-NN in $\mathcal K_N,\mathcal K_P$ to obtain evidence $\mathbf E_N,\mathbf E_P\in\mathbb R^{h\times w\times k\times d}$; feed into the "evidence-aware reasoning" module with layered cross-attention (query ↔ evidence) + self-attention (inter-patch refinement); finally, concatenate the [CLS] tokens from both branches and send to MLP for prediction $\hat y$. Also provides a **training-free variant**: directly pool scores from contrastive retrieval $\mathbf M = \text{ReLU}(\mathbf M_N - \mathbf M_P)$.

### Key Designs

1. **Dual Coreset Knowledge Bank**:

    - **Function**: Compactly and scalably represent the diverse morphologies of both "normal regions" and "pathological regions".
    - **Mechanism**: Extract patch features from $\mathcal X^B_N,\mathcal X^B_P$ → local aggregation → greedy coreset subsampling to obtain $\mathcal K_N,\mathcal K_P$. The optimization objective $\mathcal K^*=\arg\min_{\mathcal K\subset\mathcal S}\max_{m\in\mathcal S}\min_{n\in\mathcal K}\|m-n\|_2$ is a classic NP-hard problem, approximated via iterative greedy selection for good coverage.
    - **Design Motivation**: Compared to ProtoPNet's "fixed $K$ learned prototypes", coreset capacity can scale freely with data, covering the vast diversity of real clinical lesions; and since pathological images are inherently "normal + lesion mixed", only coreset (not class-specific prototypes) can accommodate such noisy data.

2. **Evidence-Aware Reasoning**:

    - **Function**: Truly integrate the $k$ evidence vectors retrieved for each patch into the query feature, outputting new representations that consider "comparison with similar historical cases".
    - **Mechanism**: Each layer first performs cross-attention: $\mathbf T^\ell_N(i,j)=\operatorname{softmax}\big(\mathbf Z^\ell_N(i,j)\mathbf E_N(i,j)^\top/\sqrt d\big)\mathbf E_N(i,j)$; then reshapes and applies inter-patch self-attention for refinement; normal branch $\mathbf Z_N$ and pathological branch $\mathbf Z_P$ run in parallel, final prediction $\hat y=\text{MLP}([\mathbf Z_N^{\text{CLS}};\mathbf Z_P^{\text{CLS}}])$.
    - **Design Motivation**: Cross-attention explicitly writes "external evidence" into the feature map, ensuring each prediction is traceable to NN evidence (retrospective interpretability); dual branches preserve the two-dimensional signal of "similar to normal" vs "similar to pathological", better aligning with clinical intuition for differential diagnosis.

3. **Contrastive Retrieval Abnormality Map (Training-Free Variant)**:

    - **Function**: Provides pixel-level abnormality localization without training any parameters, serving as a strong baseline and interpretability foundation.
    - **Mechanism**: For each patch, compute average $k$-NN distances to both banks $\mathbf M_N,\mathbf M_P\in\mathbb R^{h\times w}$; the abnormality map is $\mathbf M(i,j)=\text{ReLU}(\mathbf M_N(i,j)-\mathbf M_P(i,j))$—only regions "far from normal but close to pathological" are highlighted. The final score is pooled from $\mathbf M$.
    - **Design Motivation**: Pure PatchCore uses only $\mathbf M_N$, mislabeling all rare normal variants; the "normal–pathological" difference filters out non-pathological deviations, yielding more focused localization; also provides an inherently different, endogenous localization map compared to Grad-CAM.

### Loss & Training
Training: Only the "evidence-aware reasoning module" (cross/self-attention + MLP) is trained; the foundation model is frozen; loss is binary BCE. The training-free variant has no learnable parameters. $k$, knowledge bank size, and number of cross-attention layers $L$ are main hyperparameters.

## Key Experimental Results

### Main Results
10 public datasets, three major modalities (fundus, chest X-ray, dermoscopy), with focus on four **external** test sets: JSIEC, RIADD, CheXpert, Derm12345. Clinically-oriented metrics: AUROC, AP, Spe@95%R (specificity at 95% recall), Spe@99%R (specificity at 99% recall). Results (percent):

| Metric | EviScreen | EviScreen-TF (training-free) | FM | PatchCore* | PatchCore | NFM-DRA | DRA | SCRD4AD | EDC | SimpleNet | CIPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUROC | **98.06** | 96.76 | 95.84 | 94.96 | 92.12 | 95.53 | 92.53 | 94.88 | 79.12 | 73.73 | 94.83 |
| AP | **96.10** | 94.20 | 94.24 | 89.61 | 86.62 | 93.23 | 89.53 | 89.85 | 71.44 | 57.66 | 91.36 |
| Spe@95%R | **94.74** | 91.48 | 87.95 | 87.26 | 81.09 | 90.37 | 80.12 | 88.50 | 51.45 | 53.81 | 87.33 |
| Spe@99%R | **91.62** | 87.74 | 79.29 | 83.31 | — | — | — | — | — | — | — |

### Ablation Study

| Configuration | Key Effect | Notes |
|---|---|---|
| Full EviScreen | Best on all metrics | Dual bank + reasoning module |
| Remove pathological bank (only $\mathcal K_N$) | Spe@95%R drops | Validates irreplaceability of dual bank |
| Use fixed prototypes (ProtoPNet style) | Performance drops | Coreset has larger capacity than fixed prototypes |
| Cross-attention only (no self) | Slight drop | Inter-patch refinement provides contextual consistency |
| Training-free variant | Outperforms PatchCore | Contrastive retrieval alone is a strong baseline |

### Key Findings
- **Greater clinical metric gap**: Compared to AUROC, this method's lead over baselines is more pronounced on Spe@95%R and Spe@99%R (e.g., Spe@99%R is 8.3 points higher than PatchCore*), highlighting the advantage of evidential reasoning in clinically critical high-recall regions.
- **Training-free variant surpasses PatchCore**: Contrastive retrieval alone outperforms traditional anomaly detection baselines, validating the effectiveness of the "dual bank + contrastive" approach.
- **Scalability**: Coreset size can scale linearly, facilitating continual addition of new cases—something prototype methods cannot achieve.

## Highlights & Insights
- **Dual-track interpretability**: Provides both retrospection and localization explanations—far closer to radiologists' real diagnostic workflow than a single Grad-CAM heatmap.
- **Clinically-oriented evaluation framework**: 10 datasets + external testing + Spe@high-Recall as core, among the few truly deployment-oriented medical imaging evaluation designs; can serve as a community reference.
- **Transferable "contrastive retrieval + attention fusion"**: This paradigm can be directly applied to other domains (industrial defect detection, satellite change detection) requiring "normal vs abnormal" discrimination.

## Limitations & Future Work
- Dual bank construction requires sufficient and representative pathological samples; for rare or emerging diseases, "sparse pathological bank" may render contrastive retrieval ineffective.
- The foundation model is fully frozen, so performance ceiling is limited by it; continual replacement with medical domain foundation models can bring gains.
- Each patch performs $k$-NN during inference, causing significant latency for large banks; combining coreset growth strategies with approximate NN acceleration is a clear extension.
- Current reward/loss does not explicitly supervise "evidence–prediction consistency"; future work can add contrastive terms to ensure retrieved evidence truly guides prediction.

## Related Work & Insights
- **vs PatchCore / SimpleNet**: Both use coreset memory banks, but only normal bank; EviScreen introduces a pathological bank for contrast, enabling more precise localization via contrastive retrieval.
- **vs ProtoPNet / Prototype methods**: Fixed prototype count is limited by preset classes; coreset capacity is flexible and covers a broader range.
- **vs Grad-CAM post-hoc explanations**: This work's explanations are "endogenous" (from retrieval + cross-attention), not reliant on gradient visualization tricks, yielding higher and more stable explanation quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual coreset, contrastive retrieval, and dual-track interpretability is a novel and complete solution for medical screening tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 datasets, 3 modalities, external testing + clinical metrics + comprehensive training-free ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with three key limitations and three main contributions; Figure 1's comparison of three paradigms is concise and effective.
- Value: ⭐⭐⭐⭐⭐ Provides a deployable clinically-oriented pipeline and evaluation protocol, contributing both methodology and benchmark to the medical imaging community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Experience with Single Domain Generalization in Real World Medical Imaging Deployments](../../AAAI2026/medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)
- [\[ICML 2026\] Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[AAAI 2026\] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](../../AAAI2026/medical_imaging/deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)

</div>

<!-- RELATED:END -->
