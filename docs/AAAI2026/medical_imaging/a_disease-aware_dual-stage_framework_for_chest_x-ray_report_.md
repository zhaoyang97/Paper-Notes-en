---
title: >-
  [Paper Note] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation
description: >-
  [AAAI 2026][Medical Imaging][chest X-ray report generation] This paper proposes a two-stage disease-aware framework that learns 14 Disease-Aware Semantic Tokens (DASTs) corresponding to pathology categories for explicit disease representation. It further employs a Disease-Visual Attention Fusion (DVAF) module and a Dual-Modal Similarity Retrieval (DMSR) mechanism to assist an LLM in generating clinically accurate chest X-ray reports, achieving state-of-the-art performance on three datasets: CheXpert Plus, IU X-Ray, and MIMIC-CXR.
tags:
  - AAAI 2026
  - Medical Imaging
  - chest X-ray report generation
  - disease-aware semantic tokens
  - visual-language alignment
  - retrieval-augmented generation
  - state space models
date: 2026-05-08
content_hash: 6bd86bbb950d6133
---

# A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation

**Conference**: AAAI 2026
**arXiv**: [2511.12259](https://arxiv.org/abs/2511.12259)
**Code**: None
**Area**: Medical Imaging / Report Generation
**Keywords**: chest X-ray report generation, disease-aware semantic tokens, visual-language alignment, retrieval-augmented generation, state space models

## TL;DR

This paper proposes a two-stage disease-aware framework that learns 14 Disease-Aware Semantic Tokens (DASTs) corresponding to pathology categories for explicit disease representation. It further employs a Disease-Visual Attention Fusion (DVAF) module and a Dual-Modal Similarity Retrieval (DMSR) mechanism to assist an LLM in generating clinically accurate chest X-ray reports, achieving state-of-the-art performance on three datasets: CheXpert Plus, IU X-Ray, and MIMIC-CXR.

## Background & Motivation

Automated chest X-ray report generation is an important task in medical AI, capable of reducing radiologists' workload and shortening patient waiting times. Current mainstream methods are based on encoder-decoder architectures (e.g., the R2Gen series), with some incorporating LLMs (e.g., R2GenGPT). Despite notable progress, several critical limitations remain:

1. **Lack of disease-aware capability**: Visual features extracted by existing encoders tend to be generic image representations without explicitly encoding the presence or absence of individual diseases, causing models to overlook subtle pathological findings and produce overly generic reports.
2. **Insufficient visual-language alignment**: A substantial semantic gap exists between image features and medical text, making implicit alignment via encoder-decoder architectures unreliable.
3. **Absence of contextual references**: Most methods generate reports solely from the current image without leveraging information from similar historical cases, particularly limiting performance on rare diseases or atypical presentations.

## Core Problem

How to introduce explicit disease-aware capability into chest X-ray report generation, enabling the model not only to "see" the image but also to "understand" the presence of each disease, while incorporating references from similar cases to produce more complete and clinically accurate reports?

## Method

### Overall Architecture

The framework consists of two stages:

- **Input**: Chest X-ray image
- **Stage 1**: A VMamba visual encoder extracts image patch tokens, while 14 learnable DASTs are trained via a cross-attention mechanism to attend to pathology-relevant regions. A multi-label classification task supervises DAST learning of disease semantics. Contrastive learning is applied to align visual and textual representations.
- **Stage 2**: The LLM (Phi-4) parameters are frozen. A DVAF module fuses DASTs with visual features, and a DMSR mechanism retrieves the most relevant case reports from the training set as in-context examples. The fused features and retrieved reports are fed together to the LLM to generate the final report.
- **Output**: Structured radiology report

### Key Designs

1. **Disease-Aware Semantic Tokens (DASTs)**: The central innovation of this paper. Fourteen learnable tokens are defined, each corresponding to a pathology category in the CheXpert ontology. These tokens serve as queries in a cross-attention mechanism over visual patch tokens (as keys/values), selectively aggregating visual region information relevant to each pathology. Each DAST is followed by an independent classification head for multi-label classification. This design elegantly embeds the explicit task of disease recognition into feature learning, compelling the visual encoder to learn discriminative, disease-relevant representations rather than generic image features.

2. **Disease-Visual Attention Fusion (DVAF)**: The core module for fusing DASTs with visual features in Stage 2. The pipeline proceeds as follows: cross-attention allows DASTs to attend again to visual patch tokens to incorporate spatial context → self-attention models inter-disease relationships → attention pooling produces a unified disease representation → a gating mechanism fuses this disease representation with the global visual mean feature → the fused token is appended to the original patch sequence and fed to the LLM. Compared to simple concatenation or mean pooling, DVAF achieves finer focus on lesion regions while accounting for disease co-occurrence relationships.

3. **Dual-Modal Similarity Retrieval (DMSR)**: A composite query vector is constructed by jointly considering visual feature similarity and disease classification logit similarity to retrieve the most relevant cases from the training set. Retrieved reports are inserted as in-context examples into the LLM prompt. This design is more precise than purely visual retrieval by incorporating the disease semantic dimension; for instance, two X-ray images may appear visually similar yet differ in cardiomegaly presence—pure visual retrieval may conflate them, whereas the addition of disease logits enables disambiguation.

### Loss & Training

**Stage 1 Losses**:
- $\mathcal{L}_{CLS}$: Binary cross-entropy multi-label classification loss supervising DASTs to learn disease semantics
- $\mathcal{L}_{CTL}$: Contrastive learning loss aligning VMamba-extracted visual features (global average pooling) with Bio-ClinicalBERT-extracted text features
- $\mathcal{L}_{total} = \mathcal{L}_{CLS} + \mathcal{L}_{CTL}$

**Stage 2 Loss**:
- $\mathcal{L}_{LM}$: Standard autoregressive language modeling loss; only projection layers and normalization parameters are optimized while Stage 1 components and the LLM are frozen

**Training Details**:
- Stage 1 uses 480K image-text pairs aggregated from the training sets of MIMIC-CXR, CheXpert Plus, and IU X-Ray; 14-class pathology labels are extracted automatically via CheXpert labeler
- VMamba encoder is initialized with ImageNet pre-trained weights; input images are 224×224 grayscale
- AdamW optimizer with learning rate 1e-4, 500-step linear warm-up followed by cosine decay
- Stage 2 fine-tunes separately per dataset with Phi-4 as the LLM

## Key Experimental Results

| Dataset | Metric | Ours | Prev. SOTA (MambaXray) | Gain |
|--------|------|------|----------|------|
| CheXpert Plus | BLEU-4 | 0.133 | 0.112 | +18.8% |
| CheXpert Plus | ROUGE-L | 0.291 | 0.276 | +5.4% |
| CheXpert Plus | CIDEr | 0.227 | 0.139 | +63.3% |
| CheXpert Plus | Precision | 0.394 | 0.377 | +4.5% |
| CheXpert Plus | Recall | 0.356 | 0.319 | +11.6% |
| CheXpert Plus | F1 | 0.361 | 0.335 | +7.8% |
| IU X-Ray | BLEU-4 | 0.187 | 0.185 | +1.1% |
| IU X-Ray | ROUGE-L | 0.384 | 0.371 | +3.5% |
| IU X-Ray | CIDEr | 0.634 | 0.524 | +21.0% |
| MIMIC-CXR | BLEU-4 | 0.131 | 0.133 | -1.5% |
| MIMIC-CXR | ROUGE-L | 0.291 | 0.289 | +0.7% |

### Ablation Study

- **Contribution of DASTs**: On CheXpert Plus, incorporating DASTs + DVAF improves BLEU-4 from 0.114 to 0.122 and CIDEr from 0.193 to 0.206, demonstrating meaningful gains from disease-aware tokens.
- **DVAF vs. simple fusion**: DVAF (0.122 / 0.288 / 0.206) substantially outperforms simple concatenation (0.110 / 0.279 / 0.188) and mean pooling (0.112 / 0.285 / 0.190), validating the effectiveness of cascaded attention fusion.
- **Contribution of DMSR**: Adding DMSR on top of DVAF raises CIDEr on CheXpert Plus from 0.206 to 0.227 and on IU X-Ray from 0.597 to 0.634, demonstrating the substantial benefit of retrieval augmentation for report quality.
- Overall ablation results confirm that all three components contribute independently, with the best performance achieved when combined.

## Highlights & Insights

- **The DAST design is particularly elegant**: Learnable tokens combined with cross-attention and multi-label classification explicitly encode disease semantics, producing an intermediate representation that simultaneously serves as a by-product of disease detection and a semantic bridge for report generation.
- **Dual-modal retrieval is more principled than purely visual retrieval**: Jointly considering visual similarity and disease classification logit similarity avoids retrieval errors caused by visually similar but pathologically distinct cases.
- **The two-stage training strategy is well-motivated and clearly delineated**: Stage 1 focuses on learning disease-aware representations and visual-language alignment, while Stage 2 leverages these representations for report generation.
- **VMamba as the visual encoder**: Linear complexity $O(N)$ versus Transformer's $O(N^2)$ offers greater efficiency for high-resolution medical images.
- **CIDEr improvement of 63.3% on CheXpert Plus** indicates substantial advances in terminological precision and specificity of generated reports.

## Limitations & Future Work

- **DASTs are fixed to the CheXpert ontology**: The design is coupled to CheXpert's 14 label categories and cannot accommodate additional disease classes or alternative annotation schemes, raising concerns about generalizability.
- **Limited gains on MIMIC-CXR**: On this largest and most challenging dataset, BLEU-4 is marginally below MambaXray and CIDEr remains low. The authors attribute this to the diverse writing styles in MIMIC-CXR, though it may also reveal limitations in handling stylistic variation.
- **Cost of building and maintaining the retrieval database**: DMSR requires pre-computing visual features and disease logits for all training samples; storage and retrieval efficiency warrants attention in large-scale deployment.
- **Stage 1 training data is aggregated from three datasets**: The potential impact of heterogeneous data sources on learning quality is not analyzed.
- **No comparison with alternative multi-disease labeling schemes**: Finer-grained annotations such as RadGraph may provide richer disease semantics than CheXpert's 14-class labels.

## Related Work & Insights

- **vs. MambaXray (CVPR'25)**: Both employ Mamba-series encoders, but MambaXray directly connects Vim to an LLM without explicit disease-aware mechanisms. The proposed DASTs introduce an intermediate semantic layer, yielding a substantial lead on CheXpert Plus (CIDEr 0.227 vs. 0.139), though the advantage on MIMIC-CXR is negligible.
- **vs. R2GenGPT**: R2GenGPT also uses an LLM decoder (Llama2) but relies on simple visual feature projection without disease awareness or retrieval augmentation. The proposed method outperforms R2GenGPT on nearly all metrics.
- **vs. PromptMRG (AAAI'24)**: PromptMRG uses prompt-guided diagnosis results, sharing conceptual similarities with the disease-aware paradigm proposed here, but differs in granularity and implementation. The proposed method leads on most metrics.
- **Core distinction**: The unique contribution lies in decoupling disease recognition and report generation into two stages and establishing an explicit semantic bridge via DASTs, as opposed to end-to-end implicit learning.

## Inspirations

- The DAST design can be transferred to other tasks requiring category-aware features, e.g., using learnable tokens to query visual features in multi-label scenarios as a form of soft categorical prompting.
- The dual-modal retrieval strategy offers a generalizable principle for any RAG system: combining task-relevant semantic signals with a single modality can substantially improve retrieval quality.
- The two-stage paradigm of "first learn disease recognition and alignment, then learn generation" is transferable to other medical report generation tasks (e.g., pathology reports, ultrasound reports).
- Future work could explore dynamically determining the number of DASTs or organizing tokens using a hierarchical disease ontology.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The DAST design is innovative and dual-modal retrieval is well-motivated, though the two-stage framework and retrieval augmentation are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, ablation studies, and visualizations are provided, but limited gains on MIMIC-CXR are not thoroughly analyzed.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and method descriptions are detailed, though certain formulations (e.g., the contrastive loss) are presented too briefly.
- **Value**: ⭐⭐⭐⭐ Substantial improvements on CheXpert Plus, with practical value in the DAST and dual-modal retrieval designs, though generalizability requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[AAAI 2026\] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)

</div>

<!-- RELATED:END -->
