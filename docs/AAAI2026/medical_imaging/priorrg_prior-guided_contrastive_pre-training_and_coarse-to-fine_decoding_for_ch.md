---
title: >-
  [Paper Note] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation
description: >-
  [AAAI 2026][Medical Imaging][Chest X-ray report generation] PriorRG proposes a two-stage chest X-ray report generation framework that aligns clinical context with spatiotemporal visual features via prior-guided contrasti…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Chest X-ray report generation"
  - "prior knowledge"
  - "contrastive pre-training"
  - "coarse-to-fine decoding"
  - "spatiotemporal fusion"
date: 2026-05-08
content_hash: ba91f1473f0d913d
---

# PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation

**Conference**: AAAI 2026
**arXiv**: [2508.05353](https://arxiv.org/abs/2508.05353)  
**Code**: [GitHub](https://github.com/mk-runner/PriorRG)  
**Area**: Medical Imaging / Radiology Report Generation
**Keywords**: Chest X-ray report generation, prior knowledge, contrastive pre-training, coarse-to-fine decoding, spatiotemporal fusion

## TL;DR

PriorRG proposes a two-stage chest X-ray report generation framework that aligns clinical context with spatiotemporal visual features via prior-guided contrastive pre-training, then progressively integrates clinical context, disease progression, and multi-level visual cues through prior-aware coarse-to-fine decoding, achieving a 3.6% improvement in BLEU-4 and a 3.8% improvement in F1 on MIMIC-CXR.

## Background & Motivation

Automated radiology report generation (RRG) aims to reduce radiologists' workload by enabling AI systems to interpret medical images and produce structured textual descriptions. Most existing methods generate reports from a single image (see Figure 1(a) in the paper), overlooking **patient-specific prior knowledge** that radiologists routinely rely on in clinical practice, including:

**Clinical Context (CC)**: indications and medical history, reflecting the diagnostic intent for a patient

**Prior Image (PI)**: the most recent prior scan used to track disease progression

Several prior works have attempted to incorporate partial prior information but exhibit notable limitations:
- Methods such as SEI leverage indication information but ignore longitudinal data, making the generated reports prone to hallucinations when describing disease progression.
- Methods such as HERGen introduce prior images to model temporal changes but neglect clinical context, lacking personalized reasoning capability.

This motivates the central research question: **Can temporal visual changes and clinical context be jointly modeled to improve cross-modal alignment and report generation?**

## Method

### Overall Architecture

PriorRG adopts a two-stage training pipeline (see Figure 2 in the paper) that mirrors real-world clinical workflows:
- **Visual encoder**: RAD-DINO (frozen)
- **Text encoder**: CXR-BERT (trainable)
- **Report generator**: DistilGPT2 (trainable)

Inputs include the current image $x_i^{cur}$, prior image $x_i^{pri}$ (possibly absent), indication $z_i$ (possibly absent), and medical history $h_i$ (possibly absent).

### Key Designs

#### 1. Stage 1: Prior-Guided Contrastive Pre-training

**Objective**: Leverage clinical context to guide spatiotemporal feature extraction and enhance cross-modal alignment.

**Visual feature extraction**: After extracting features with RAD-DINO, learnable **view-position embeddings** are introduced and fused into the visual features to handle appearance variation across different projection views (e.g., AP/PA), yielding $\boldsymbol{V} \in \mathbb{R}^{M \times s \times d}$.

**Text feature extraction**: CXR-BERT encodes text with special tokens `[INDICATION]`, `[HISTORY]`, and `[FINDINGS]` prepended to indications, medical history, and reports respectively, enabling type-aware unified encoding while gracefully handling missing fields.

**Spatiotemporal Fusion Network (STF)**: A ViT-style cross-attention fusion module models disease progression between the current and prior images:

$$\boldsymbol{V}^{st}_{ca} = \text{LN}(\boldsymbol{V}^{cur} + \text{CA}(\text{LN}(\boldsymbol{V}^{cur}), \text{LN}(\boldsymbol{V}^{pri})))$$

$$\boldsymbol{V}^{st} = \text{LN}(\boldsymbol{V}^{st}_{ca} + \text{FFN}(\boldsymbol{V}^{st}_{ca}))$$

where CA denotes cross-attention. The STF uses 3 layers. When no prior image is available, the current image features are used directly.

**Instance-level cross-modal alignment**: Simulating the clinical diagnostic process, a Perceiver architecture progressively fuses clinical context with spatiotemporal features:

$$\boldsymbol{\bar{T}}^c = \text{Perceiver}(\boldsymbol{E}^{lat}, \boldsymbol{T}^c)$$

$$\boldsymbol{\bar{V}}^{st} = \text{Perceiver}(\boldsymbol{\bar{T}}^c, \boldsymbol{V}^{st})$$

Global visual features $\boldsymbol{V}^g$ are obtained via global average pooling and L2 normalization. Image–report similarity is computed and optimized using a cross-entropy alignment loss $\mathcal{L}_{align}$ that supports multiple positive pairs.

#### 2. Stage 2: Prior-Aware Coarse-to-Fine Decoding

**Attention-enhanced Layer Fusion Network (ALF)**: CBAM-based channel and spatial attention is applied to features from each encoder layer to highlight diagnostically relevant information. A Conv2D projector then fuses these into a **multi-level visual representation** $\boldsymbol{V}^{hier}$, addressing the limitation of existing methods that rely solely on the last hidden state and overlook low-level details such as lesion morphology.

**Coarse-to-fine decoding**: Inspired by principles of visual cognition, prior knowledge and multi-level visual semantics are progressively integrated:
- **Coarse-grained priors**: $\boldsymbol{\bar{T}}^c$ (clinical context) and $\boldsymbol{\bar{V}}^{st}$ (spatiotemporal features) supply high-level clinical background and disease progression cues.
- **Fine-grained enhancement**: $\boldsymbol{\bar{V}}^{hier} = \text{Perceiver}(\boldsymbol{\bar{V}}^{st}, \boldsymbol{V}^{hier})$, where spatiotemporal features serve as queries to extract fine-grained information from multi-level features.
- **Final concatenation**: $\boldsymbol{\bar{T}}^c$, $\boldsymbol{\bar{V}}^{st}$, and $\boldsymbol{\bar{V}}^{hier}$ are concatenated along the sequence dimension and fed into DistilGPT2 for report generation.

### Loss & Training

- **Stage 1**: Contrastive alignment loss $\mathcal{L}_{align}$ (cross-entropy), supporting multi-view positive pairs.
- **Stage 2**: Cross-entropy loss $\mathcal{L}_{CE}$ for training autoregressive report generation.
- Unified feature dimension $d=768$, number of latent variables $N=128$, maximum generation length $K=100$, beam size 3.
- AdamW optimizer with ReduceLROnPlateau scheduler and early stopping (patience = 15).
- On MIMIC-CXR: 30 epochs for Stage 1 pre-training and 30 epochs for Stage 2 fine-tuning.

## Key Experimental Results

### Main Results

| Dataset | Metric | PriorRG | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| MIMIC-CXR | B-1 | 0.412 | 0.416 (MPO) | -0.4% |
| MIMIC-CXR | B-4 | 0.175 | 0.139 (MPO) | +3.6% |
| MIMIC-CXR | MTR | 0.189 | 0.176 (BioViL-T) | +1.3% |
| MIMIC-CXR | R-L | 0.324 | 0.309 (MPO) | +1.5% |
| MIMIC-CXR | F1 | 0.511 | 0.473 (R2-LLM) | +3.8% |
| MIMIC-ABN | B-1 | 0.326 | 0.267 (SEI) | +5.9% |
| MIMIC-ABN | B-4 | 0.102 | 0.073 (SEI) | +2.9% |
| MIMIC-ABN | F1 | 0.471 | 0.460 (CMN) | +1.1% |

PriorRG achieves comprehensive superiority on long n-gram matching (B-4) and clinical accuracy (F1), outperforming SEI on 13 of 14 CheXpert observation categories in terms of F1.

### Ablation Study

| Configuration | B-4 | F1 | Notes |
|------|-----|-----|------|
| (a) No CC, No PI, No Hidden | 0.108 | 0.472 | Baseline: Stage 1 only, no prior knowledge |
| (c) CC, No PI, No Hidden | 0.170 | 0.487 | Adding clinical context yields significant gains |
| (e) CC, PI, No Hidden | 0.171 | 0.499 | Prior image provides further improvement |
| (d) CC, No PI, Hidden | 0.173 | 0.507 | Multi-level features improve clinical accuracy |
| (f) No Stage 1, Stage 2 only | 0.165 | 0.459 | Omitting pre-training causes substantial degradation |
| **PriorRG (full)** | **0.175** | **0.511** | All components synergistically optimal |

### Key Findings

1. **Clinical context has the largest impact**: As CC availability increases from 0% to 100%, B-2 rises from 0.139 to 0.294 (Table 5), demonstrating that indications and medical history are critical for report generation.
2. **Prior image contribution is more pronounced at the study level**: It enhances temporal cue modeling, yielding significant improvements on the Stu-P@K retrieval metric.
3. **Coarse-to-fine outperforms fine-to-coarse**: PriorRG achieves better NLG metrics than the Fine2coarse variant, confirming that progressively integrating high-level semantics before fine-grained details is more effective.
4. **Zero-shot report generation capability**: In an unsupervised setting, PriorRG achieves B-4 = 0.178 and MTR = 0.211, substantially outperforming R2GenGPT and Med-LLM.

## Highlights & Insights

1. **Complete simulation of clinical workflow**: The pipeline — from initial clinical assessment to spatiotemporal contrast to multi-level refinement — closely mirrors radiologists' diagnostic reasoning process.
2. **Graceful handling of missing inputs**: Special tokens and the Perceiver architecture naturally accommodate the absence of prior images, indications, or medical history, making the framework practically deployable.
3. **Disease progression description capability**: Qualitative analysis shows that PriorRG correctly describes lesion changes (e.g., "cardiac silhouette remains enlarged but unchanged"), reducing hallucinations when characterizing temporal variations.
4. **Leading GREEN scores**: Evaluated using the pretrained GREEN-RadLlama2-7B model, PriorRG significantly outperforms all baselines on both the number of matched findings and the overall GREEN composite score.

## Limitations & Future Work

1. The use of DistilGPT2 as the report generator constrains capacity due to its relatively small model scale; future work may explore more powerful LLMs such as LLaMA.
2. Validation is limited to the MIMIC-CXR dataset, with no exploration of report generation for other imaging modalities (e.g., CT, MRI).
3. Organ-level localization information is not incorporated; the authors note in the conclusion that an organ-aware diagnostic framework will be explored in future work.
4. The two-stage training process is sequential; end-to-end joint optimization may yield further performance improvements.

## Related Work & Insights

- **BioViL-T / MLRG**: Pioneering works in longitudinal data modeling, but lacking clinical context.
- **SEI**: A representative method utilizing indication information, but neglecting disease progression.
- **Perceiver architecture**: Effectively employed for progressive cross-modal fusion, serving as a versatile tool for connecting information at different levels of granularity.
- Insight: The impact of clinical prior knowledge — especially indications — on report generation far exceeds expectations; visual information alone is insufficient.

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 4 | First work to fully exploit the combination of clinical context, prior images, and multi-level visual features |
| Technical Depth | 4 | Two-stage design is well-motivated; Perceiver-based progressive fusion is creative |
| Experimental Thoroughness | 5 | Covers main experiments, ablations, retrieval, qualitative analysis, and LLM-based evaluation |
| Value | 4 | Directly aligned with clinical workflows; handles missing inputs effectively |
| Writing Quality | 4 | Well-structured, richly illustrated, with clearly articulated motivation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[AAAI 2026\] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)

</div>

<!-- RELATED:END -->
