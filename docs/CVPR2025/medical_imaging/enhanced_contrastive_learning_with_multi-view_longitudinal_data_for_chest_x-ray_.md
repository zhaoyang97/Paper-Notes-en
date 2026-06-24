---
title: >-
  [Paper Note] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation
description: >-
  [CVPR 2025][Medical Imaging][Radiology Report Generation] This paper proposes MLRG, a two-stage framework that integrates the spatial information of current multi-view images and the temporal information of historical longitudinal data during vision-language pre-training via multi-view longitudinal contrastive learning. It flexibly handles missing patient prior knowledge using tokenized absence encoding, achieving a 2.3% improvement in BLEU-4 on MIMIC-CXR and a 5.5% improveme…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Radiology Report Generation"
  - "Multi-view Longitudinal Data"
  - "Contrastive Learning"
  - "Missing Data Handling"
  - "Chest X-ray"
date: 2026-05-08
content_hash: 1a5742be79245dd0
---

# Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation

**Conference**: CVPR 2025  
**arXiv**: [2502.20056](https://arxiv.org/abs/2502.20056)  
**Code**: [https://github.com/mk-runner/MLRG](https://github.com/mk-runner/MLRG)  
**Area**: Medical Image  
**Keywords**: Radiology Report Generation, Multi-view Longitudinal Data, Contrastive Learning, Missing Data Handling, Chest X-ray

## TL;DR
This paper proposes MLRG, a two-stage framework that integrates the spatial information of current multi-view images and the temporal information of historical longitudinal data during vision-language pre-training via multi-view longitudinal contrastive learning. It flexibly handles missing patient prior knowledge using tokenized absence encoding, achieving a 2.3% improvement in BLEU-4 on MIMIC-CXR and a 5.5% improvement in F1 on MIMIC-ABN.

## Background & Motivation

**Background**: Automated radiology report generation (RRG) automatically generates clinical reports from X-ray images using a visual encoder + text generator, which can effectively reduce the workload of radiologists. Most existing methods only utilize a single image or fixed dual-view images to generate reports.

**Limitations of Prior Work**: In clinical practice, radiologists typically make diagnoses by synthesising multi-view images (different views like PA, AP, lateral have different geometric characteristics; for instance, AP may magnify the cardiac silhouette), historical images of patients, and prior reports. However, existing methods suffer from three main limitations: (1) using only a single image fails to differentiate between the geometric differences of various views; (2) ignoring the longitudinal information of disease progression, which can lead to model hallucinations (generating groundless disease descriptions); (3) some patients lack "INDICATION", "previous report", or "previous image" (due to being first-time visits or data storage issues), which existing methods struggle to handle flexibly.

**Key Challenge**: The flexible fusion of multi-source heterogeneous information (multi-view $\times$ multi-temporal points $\times$ optional textual priors) coupled with the inconsistent availability of data across different patients.

**Goal**: (1) How to flexibly fuse a variable number of current multi-view images and potentially missing historical images? (2) How to leverage the spatiotemporal information inherent in radiology reports to guide vision-language pre-training? (3) How to handle the absence of patient-specific prior knowledge (INDICATION, previous report)?

**Key Insight**: The diagnostic process of radiologists is modeled into two stages: first, pre-training visual representations using multi-view longitudinal contrastive learning (utilizing the spatiotemporal semantics of reports as supervisory signals), and second, using tokenized absence encoding during the generation stage to adapt flexibly to data availability.

**Core Idea**: Pre-train multi-view longitudinal visual representations using multi-positive contrastive learning and cross-modal alignment, and handle missing prior knowledge with tokenized absence encoding to achieve flexible and accurate report generation.

## Method

### Overall Architecture
A two-stage architecture. Stage 1 (Pre-training): The RAD-DINO visual encoder extracts features of multi-view images, adds learnable view position embeddings and temporal position embeddings, and fuses spatiotemporal visual features via a Multi-view Longitudinal Fusion network (MLF) to perform multi-granularity contrastive learning with textual features extracted by CXR-BERT. Stage 2 (Report Generation): Tokenized absence encoding is used to handle missing INDICATION and previous reports. These features, along with pre-trained visual features, are fed into the multimodal fusion network, and DistilGPT2 generates the reports.

### Key Designs

1. **Multi-Positive Contrastive Learning**:

    - **Function**: Enhances the consistency of visual features among multi-view images from the same study.
    - **Mechanism**: Each current image is treated as an anchor, other view images from the same study are positive samples, and images from different studies are negative samples. A learnable view position embedding $E_v$ is added to distinguish geometric variations across views such as PA/AP/lateral. A multi-positive cross-entropy loss $\mathcal{L}_{MPC}$ is used to maximize the similarity of images from the same study. Studies with only a single image (no positive sample pairs) are skipped.
    - **Design Motivation**: Different views provide complementary information (e.g., PA and lateral observe different anatomical structures), but current methods only simplistically distinguish "frontal" and "lateral", ignoring fine-grained differences like PA vs. AP. Learnable view embeddings allow the model to automatically capture geometric discrepancies across views.

2. **Multi-view Longitudinal Fusion Network (MLF) + Instance-wise Cross-modal Alignment**:

    - **Function**: Flexibly fuses spatiotemporal features of current multi-view images and historical images, and aligns them with textual reports.
    - **Mechanism**: MLF utilizes a cross-attention mechanism, using the features of the anchor scan as the query, and the features of auxiliary views and historical images as the key/value. Since the number of views and historical availability vary for each study, instance-wise processing ensures flexibility. The fused spatiotemporal visual features $V^{st}$ and the report text features undergo instance-wise cross-modal alignment (CLIP-style), using a cross-entropy loss $\mathcal{L}_G$ to align global visual and textual representations. Crucially, reports naturally contain temporal comparison information (such as "compared to prior study..."), making alignment using multi-view longitudinal fused features more reasonable than focusing only on current images.
    - **Design Motivation**: Radiology reports not only describe the current state but also compare it with historical changes. Relying solely on current images for cross-modal alignment makes it difficult to learn temporal variation information and can even lead to hallucinations. The cross-attention mechanism of MLF naturally supports a variable number of key/value inputs.

3. **Tokenized Absence Encoding**:

    - **Function**: Flexibly handles missing patient-specific prior knowledge (INDICATION and previous report) during the generation phase.
    - **Mechanism**: When a certain item of prior knowledge is missing, instead of simply discarding it or padding with zeros, it is replaced with a dedicated learnable absence token. The text generator (DistilGPT2) receives the output of the multimodal fusion network, which flexibly fuses visual features with available or missing prior knowledge. The absence token enables the model to learn to distinguish between "this information is not present" and "this information is empty."
    - **Design Motivation**: In real-world clinical data, about 35-50% of patients lack INDICATION or previous reports/images (as statisticalized in Table 1). Simply ignoring this decreases the model's capacity to leverage available information, whereas the proposed absence token optimizes report generation regardless of whether prior knowledge is available or absent.

### Loss & Training
- Stage 1 Pre-training: $\mathcal{L}_{pretrain} = \mathcal{L}_{MPC} + \mathcal{L}_G$
- Stage 2 Fine-tuning: Standard auto-regressive cross-entropy loss to train the DistilGPT2 generator.
- Visual Encoder: RAD-DINO (ViT, frozen); Text Encoder: CXR-BERT; Text Generator: DistilGPT2.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|---------|------|
| MIMIC-CXR | BLEU-4 ↑ | **0.158** | 0.135 (SEI) | +2.3% |
| MIMIC-CXR | F1 (CE) ↑ | **0.505** | 0.473 (B-LLM) | +3.2% |
| MIMIC-CXR | RadGraph ↑ | **0.291** | 0.249 (SEI) | +4.2% |
| MIMIC-CXR | ROUGE-L ↑ | **0.320** | 0.304 (CoFE) | +1.6% |
| MIMIC-ABN | F1 (CE) ↑ | **0.515** | 0.460 (CMN) | +5.5% |
| Two-view CXR | F1 RadGraph ↑ | **0.254** | 0.227 (CXRMate) | +2.7% |

### Ablation Study

| Configuration | BLEU-4 | F1 | RadGraph | Note |
|------|--------|-----|----------|------|
| Full MLRG | **0.158** | **0.505** | **0.291** | Complete model |
| w/o MPC | Decrease | Decrease | Decrease | Without multi-positive contrastive learning |
| w/o MLF | Decrease | Decrease | Decrease | Without longitudinal fusion network |
| w/o Absence Encoding | Decrease | Decrease | Decrease | Without absence encoding |
| Single-view only | Significant decrease | Significant decrease | Significant decrease | Using only a single image |

### Key Findings
- The joint usage of multi-view longitudinal data (MVL input) comprehensively outperforms single-image (SI), multi-view (MVD), and longitudinal-only (Long) schemes on all datasets.
- Compared to LLM-based methods on MIMIC-CXR (such as B-LLM which uses MiniGPT-4), MLRG achieves better results with a much smaller DistilGPT2, demonstrating that representation learning is more critical than model scale.
- Using the spatiotemporal features after longitudinal fusion for cross-modal alignment yields significantly better performance than using only the current image features.
- The improvement is more pronounced in clinical accuracy metrics (F1, RadGraph) than in NLG metrics (BLEU), indicating that the proposed method yields the most significant enhancements in medical substance.

## Highlights & Insights
- **Leveraging the spatiotemporal semantics of reports as supervision**: Radiology reports naturally contain history comparisons ("unchanged...", "new opacity..."). Aligning report text with multi-view longitudinal fused features enables learning temporal variation sequence patterns that cannot be learned in traditional single-image alignment. This insight is highly worth borrowing in other longitudinal medical tasks.
- **Generality of tokenized absence encoding**: Utilizing a dedicated absence token to handle missing inputs is more semantic than zero padding or dropout. This technique can be transferred to any generation task where inputs are multi-source and some sources can be missing.
- **Cross-attention naturally handles variable inputs**: MLF uses query-KV cross-attention to process different numbers of views and historical images without requiring padding or fixed sequence depths, providing a simple and elegant design.

## Limitations & Future Work
- Only the most recent historical study was utilized; longer longitudinal sequences (multiple historical records) could provide richer disease progression insights.
- The scale of the report generator DistilGPT2 is small; replacing it with a larger LLM (e.g., LLaMA) may further improve generation quality.
- The visual encoder RAD-DINO is frozen and not fine-tuned, which may limit the adaptation of visual features to specific tasks.
- The generalization of the model has not been validated on multi-center datasets.
- Currently designed only for chest X-ray; extending to other modalities like CT or MRI requires further adjustments.

## Related Work & Insights
- **vs CXRMate**: CXRMate also leverages multi-view longitudinal data but introduces noise by synthesizing historical reports and fails to distinguish between view differences. MLRG directly uses real data and accounts for view geometry differences, outperforming it.
- **vs SEI**: SEI uses only single-image + INDICATION, whereas MLRG uses richer multi-view longitudinal inputs and outperforms it across all metrics.
- **vs BioViL-T**: BioViL-T utilizes longitudinal data for visual representation learning but does not apply it to report generation; MLRG unifies longitudinal pre-training and report generation into a single framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of multi-view longitudinal contrastive learning and absence encoding is innovative, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on three datasets, with multi-dimensional metrics, rich comparisons, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem analysis and systematic method description.
- **Value**: ⭐⭐⭐⭐ A practical approach aligned with real clinical requirements; the direction of multi-view longitudinal fusion is highly worth exploring further.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[ICLR 2026\] Learning Self-Critiquing Mechanisms for Region-Guided Chest X-Ray Report Generation](../../ICLR2026/medical_imaging/learning_self-critiquing_mechanisms_for_region-guided_chest_x-ray_report_generat.md)
- [\[CVPR 2026\] Phrase-grounded APO for Improving Chest X-ray Report Generation](../../CVPR2026/medical_imaging/phrase-grounded_apo_for_improving_chest_x-ray_report_generation.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[CVPR 2025\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)

</div>

<!-- RELATED:END -->
