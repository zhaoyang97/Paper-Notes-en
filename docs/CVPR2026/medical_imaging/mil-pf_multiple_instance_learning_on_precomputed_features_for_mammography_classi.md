---
title: >-
  [Paper Note] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification
description: >-
  [CVPR 2026][Medical Imaging][Multiple Instance Learning] This paper proposes MIL-PF, a framework that leverages frozen foundation vision encoders (DINOv2/MedSigLIP) to precompute features, followed by a lightweight MIL head of approximately 40K parameters for mammography classification. The method achieves state-of-the-art performance on the large-scale EMBED dataset while substantially reducing training cost.
tags:
  - CVPR 2026
  - Medical Imaging
  - Multiple Instance Learning
  - Mammography
  - Precomputed Features
  - Foundation Models
  - Attention Aggregation
date: 2026-05-08
content_hash: 41e7cdfc073beab4
---

# MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**Conference**: CVPR 2026
**arXiv**: [2603.09374](https://arxiv.org/abs/2603.09374)
**Code**: Available (promised by authors)
**Area**: Medical Imaging
**Keywords**: Multiple Instance Learning, Mammography, Precomputed Features, Foundation Models, Attention Aggregation

## TL;DR

This paper proposes MIL-PF, a framework that leverages frozen foundation vision encoders (DINOv2/MedSigLIP) to precompute features, followed by a lightweight MIL head of approximately 40K parameters for mammography classification. The method achieves state-of-the-art performance on the large-scale EMBED dataset while substantially reducing training cost.

## Background & Motivation

Mammography classification presents three fundamental challenges: (1) **extremely high image resolution** (up to 4708×5844 pixels), making end-to-end fine-tuning of large models computationally prohibitive; (2) **annotation scarcity**—labels are typically available only at the breast level as BI-RADS reports (weak supervision), without pixel- or region-level annotations; (3) **multi-view nature**—CC and MLO views of each breast require joint inference. Existing methods such as GMIC and FPN-AbMIL either require end-to-end training of large backbone networks (1.7–22.9M parameters) or exhibit instability on large-scale, diverse datasets.

The paper's core insight is that recent general-purpose vision foundation models (DINOv2-g, MedSigLIP), even when completely frozen, produce strong representations for the out-of-distribution mammography domain—thus obviating the need for encoder fine-tuning and requiring only a minimal task-specific aggregation module.

## Method

### Overall Architecture

MIL-PF operates in two stages:

**Stage 1: Feature Precomputation**—For each breast bag $\mathcal{S}_i$, images are passed through the frozen encoder $\mathcal{F}$ to extract two types of embeddings: (a) global embeddings $\mathcal{G}_i = \{\mathcal{F}(I_i^{(n)})\}_n$—one encoding per full image; (b) local tile embeddings $\mathcal{T}_i = \bigcup_n \bigcup_k \{\mathcal{F}(C_i^{(n)(k)})\}$—each image is partitioned into a grid of tiles matching the encoder input size (448×448 / 518×518), with pure-background tiles discarded. This yields an embedding dataset $\mathcal{E} = \{(\mathcal{G}_i, \mathcal{T}_i, y_i)\}_i$.

**Stage 2: MIL Head Training**—A lightweight aggregation head is trained on the precomputed $\mathcal{E}$. Global and local streams are processed by their respective aggregators, concatenated, and passed through a final classification layer.

### Key Designs

1. **Dual-Stream Architecture (Global + Local Stream)**: The global stream $\mathcal{A}_\psi^G$ aggregates image-level features to capture low-frequency signals such as global breast tissue structure; the local stream $\mathcal{A}_\omega^T$ aggregates tile-level features to capture high-frequency signals such as sparsely distributed lesions. At inference, $\hat{y}_i = h_\theta(\text{concat}(\mathcal{A}_\psi^G(\mathcal{G}_i), \mathcal{A}_\omega^T(\mathcal{T}_i)))$. The MLPs in each stream project embeddings from the encoder dimension down to 16 and then to 8 dimensions. This late-fusion strategy, while simple, proves sufficient.

2. **Perceiver Cross-Attention Local Aggregator**: The key challenge for the local stream is the extreme sparsity of lesion tiles—the vast majority correspond to normal tissue. Mean pooling dilutes the signal with background; max pooling captures only a single most salient tile; self-attention focuses on modeling inter-tile dependencies rather than selecting relevant tiles. MIL-PF adopts **Perceiver-style cross-attention**: a single trainable latent vector $z$ serves as a query over all local tile embeddings (Keys/Values) via $\text{softmax}(zK^T)V$, distilling the most relevant information into a single summary vector. A single latent query suffices, without requiring the multiple latents used in the standard Perceiver.

3. **Frozen Encoder + Precomputation Paradigm**: The encoder $\mathcal{F}$ is kept entirely frozen, and all bag embeddings are precomputed and stored. The MIL head has only approximately 40K trainable parameters; the entire EMBED training set (~500K images) fits in a single batch on an A100 40GB GPU, with each training run taking only 5–7 minutes. This enables: (a) 36 independent runs for model selection; (b) flexible encoder substitution; (c) minimal computational resource requirements.

### Loss & Training

- Binary Cross-Entropy loss
- Data split: 70%/10%/20% (train/val/test), stratified by BI-RADS score with strict prevention of patient leakage
- 36 independent runs per experiment; the model with the best validation AUC is selected
- Non-overlapping tiles for classification; 75% overlapping tiles for attention map generation
- BI-RADS 1 is treated as negative; BI-RADS 4/5/6 as positive

## Key Experimental Results

### Main Results

| Dataset | Metric | MIL-PF (Best) | Prev. SOTA | Gain |
|--------|------|--------------|---------|------|
| EMBED (BI-RADS) | AUC | **0.918** (MedSigLIP, max) | 0.875 (SILIL GMIC) | +4.3% |
| EMBED (BI-RADS) | Spec@Sens=0.9 | **0.762** (DINOv2, attn) | 0.566 (SILIL GMIC) | +19.6% |
| VinDr (BI-RADS) | AUC | 0.911 (MedSigLIP, attn) | **0.920** (FPN-AbMIL) | −0.9% |
| VinDr (BI-RADS) | Spec@Sens=0.9 | **0.792** (DINOv2, attn) | 0.720 (FPN-AbMIL) | +7.2% |
| RSNA (Cancer) | AUC | **0.925** (MedSigLIP, max) | 0.914 (FPN-AbMIL mean) | +1.1% |
| VinDr (Calcification) | AUC | **0.967** (both encoders) | 0.962 (FPN-AbMIL mean) | +0.5% |

### Ablation Study

| Configuration | EMBED AUC | Notes |
|------|-----------|------|
| DINOv2-g (attn) | 0.916 | General-purpose encoder performs remarkably well |
| MedSigLIP (attn) | 0.914 | Medical encoder slightly lower but comparable |
| MammoCLIP (mammography-specific) | Lower | Domain-specific encoder generalizes worse |
| $\mathcal{A}^T$: max pooling | 0.905/0.918 | Simple yet effective |
| $\mathcal{A}^T$: attention | 0.916/0.914 | Attention aggregation slightly better and more stable |
| Trainable parameters | ~40K/50K | 1/35 to 1/450 of competing methods |

### Key Findings

- **General foundation models are surprisingly strong**: DINOv2 and MedSigLIP, when fully frozen, generalize to the out-of-distribution mammography domain better than MammoCLIP, which was pretrained specifically on mammography data.
- **Advantages most pronounced on the largest dataset**: MIL-PF shows the clearest gains on EMBED (~500K images, the largest and most diverse public mammography dataset), indicating that its robustness stems from representation and aggregation design rather than data volume alone.
- **Substantial gains on Spec@Sens=0.9**: The clinically most relevant metric—specificity at high sensitivity—shows the largest improvements, carrying direct clinical significance.
- **Detection limitations**: Attention maps are competitive for medium and large lesions, but small-lesion IoU@0.25 is poor due to the coarse tile granularity (448/518px).

## Highlights & Insights

- **Minimalist philosophy**: 40K parameters suffice to achieve SOTA—when the encoder is strong enough, complex end-to-end fine-tuning is unnecessary.
- **Revival of the precomputation paradigm**: By exploiting the powerful representations of modern foundation models, this work brings the classic "precomputed features + lightweight head" paradigm back to the frontier, with particular relevance for research groups with limited computational resources.
- **MIL problem formalization**: The paper provides a clear formal definition of the nested hierarchical structure of the mammography MIL problem (bag → image → tiles → ROIs) and the dual-stream signal decomposition (global tissue vs. local lesion).
- **Engineering efficiency**: The combination of 5–7 minute training and 36-run model selection makes hyperparameter search and model selection highly practical.

## Limitations & Future Work

- Detection accuracy is constrained by tile size (448/518px); localization of small lesions is imprecise—multiscale tiling or finer-grained segmentation may help.
- Global and local streams are fused via simple concatenation (late fusion), with no explicit modeling of their interaction; early or intermediate fusion could yield further improvements.
- The framework depends on the representational quality of the pretrained encoder—if the encoder is insensitive to certain anomaly types, the MIL head cannot compensate.
- A single latent query in the Perceiver may limit the model's ability to simultaneously attend to multiple independent lesions.
- Joint modeling with radiology report text remains unexplored.

## Related Work & Insights

- GMIC [Shen et al.] trains end-to-end with 14.1M parameters; FPN-AbMIL [Mourão et al.] uses 1.76M—MIL-PF matches or surpasses both with 40K parameters, demonstrating that representation quality outweighs downstream head complexity.
- DINOv2's self-supervised pretraining on large-scale curated data establishes a new standard for general-purpose visual feature extraction.
- Key takeaway: For high-resolution medical images, the "frozen foundation model + lightweight task-specific head" paradigm may be more practical than end-to-end fine-tuning.

## Rating

- Novelty: ⭐⭐⭐ Individual components are not entirely new (MIL + frozen encoders + Perceiver), but their combination and the "less is more" philosophy are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets (EMBED/VinDr/RSNA), multiple encoder comparisons, multiple baselines, 36 independent runs.
- Writing Quality: ⭐⭐⭐⭐ MIL problem formalization is clear; design choices are well motivated.
- Value: ⭐⭐⭐⭐ Offers direct practical value to resource-constrained medical AI research groups; strong engineering significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gender-adversarial_attention_multi.md)
- [\[CVPR 2026\] LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)
- [\[CVPR 2026\] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[CVPR 2026\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segment.md)

</div>

<!-- RELATED:END -->
