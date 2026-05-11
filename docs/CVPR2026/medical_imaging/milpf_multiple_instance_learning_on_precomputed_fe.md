---
title: >-
  [Paper Note] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification
description: >-
  [CVPR 2026][Medical Imaging][multiple instance learning] Combining frozen general-purpose foundation encoders (DINOv2 ViT-Giant / MedSigLIP) with a lightweight MIL aggregation head of only ~40k parameters…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "multiple instance learning"
  - "mammography"
  - "precomputed features"
  - "frozen foundation models"
  - "weakly supervised classification"
date: 2026-05-08
content_hash: 35a725bdbc9af0ed
---

# MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**Conference**: CVPR 2026
**arXiv**: [2603.09374](https://arxiv.org/abs/2603.09374)
**Code**: Available (open-sourced)
**Area**: Medical Imaging
**Keywords**: multiple instance learning, mammography, precomputed features, frozen foundation models, weakly supervised classification

## TL;DR

Combining frozen general-purpose foundation encoders (DINOv2 ViT-Giant / MedSigLIP) with a lightweight MIL aggregation head of only ~40k parameters, MIL-PF achieves state-of-the-art performance on large-scale mammography classification benchmarks such as EMBED (AUC 0.916, Spec@Sens=0.9 of 0.762) via a dual-stream aggregation strategy (global mean pooling + local Perceiver cross-attention), training in 5–7 minutes with 35–458× fewer trainable parameters than baselines.

## Background & Motivation

**Background**: Breast cancer is the most common malignancy and the leading cause of cancer-related death in women; mammography is the preferred screening modality. Existing methods either fine-tune large backbones end-to-end (14–23M parameters, computationally expensive) or rely on domain-specific pretrained encoders (e.g., MammoCLIP) with limited generalization.

**Limitations of Prior Work**: Mammography presents three unique challenges: (1) extremely high resolution (up to 4708×5844 pixels), making end-to-end fine-tuning of large models impractical; (2) annotations are available only at the breast level (BI-RADS grading), without pixel-level labels, yielding a weakly supervised setting; and (3) a single examination contains multiple views (CC, MLO), requiring joint reasoning across views.

**Key Challenge**: Can powerful general-purpose foundation models (e.g., DINOv2) zero-shot generalize to out-of-distribution mammography data? If so, precomputed features could substantially reduce experimental costs—challenging the prevailing assumption that medical imaging requires domain-specific pretraining.

**Goal**: To design a lightweight classification framework that models both global tissue structure and sparse local lesion signals using features from frozen foundation models, without fine-tuning large visual encoders.

**Key Insight**: The authors observe that DINOv2 ViT-Giant and MedSigLIP generalize surprisingly well to mammography in a zero-shot setting, substantially outperforming the domain-specific MammoCLIP (AUC 0.897 vs. 0.870), validating the feasibility of the "frozen general-purpose encoder + lightweight task head" paradigm.

**Core Idea**: Precomputed features from frozen general-purpose foundation encoders, combined with a dual-stream MIL aggregation head (global mean pooling + local Perceiver attention), achieve SOTA performance with ~40k parameters and 5 minutes of training.

## Method

### Overall Architecture

MIL-PF operates in two stages: (1) **Feature Precomputation**—a frozen encoder $\mathcal{F}$ (DINOv2 ViT-Giant or MedSigLIP) extracts global features (whole-image encoding $\mathcal{G}_i$) and local features (tile-level encoding $\mathcal{T}_i$) for each mammogram, producing an embedding dataset $\mathcal{E} = \{(\mathcal{G}_i, \mathcal{T}_i, y_i)\}$; (2) **MIL Head Training**—a ~40k-parameter aggregation head is trained on these embeddings, comprising a global-stream aggregator $\mathcal{A}_\psi^G$, a local-stream Perceiver aggregator $\mathcal{A}_\omega^T$, and a final classification layer $h_\theta$. A bag is defined as all view images of the same breast within the same examination.

### Key Designs

1. **Dual-Stream Embedding Dataset Construction**

    - **Function**: Decomposes each mammographic examination into two complementary signal sources—global tissue context and local lesion candidates.
    - **Mechanism**: The global stream encodes each full image as $\mathcal{G}_i = \{\mathcal{F}(I_i^{(n)})\}_n$ to capture overall tissue density. The local stream tiles each image into a non-overlapping grid, discards pure-background tiles, encodes each tissue-containing tile individually, and merges all tiles across views: $\mathcal{T}_i = \bigcup_n \bigcup_k \{\mathcal{F}(C_i^{(n)(k)})\}$. Tile size is chosen to be large enough to encompass the expected ROI while not exceeding the encoder's maximum supported resolution (448/518 pixels).
    - **Design Motivation**: The global stream provides macroscopic tissue density information; the local stream captures fine-grained signals from sparse lesions. The two streams are complementary.

2. **Perceiver-Style Local Attention Aggregator**

    - **Function**: Extracts the most relevant lesion information from a large set of local tile embeddings into a single summary vector.
    - **Mechanism**: A single trainable latent vector $z$ serves as the query; all tile embeddings are projected into keys and values; the aggregated representation is computed as $\text{softmax}(zK^T)V$. A single latent query is found to be sufficient; adding more yields no benefit.
    - **Design Motivation**: Mean pooling dilutes the signal with background tiles; max pooling captures only the single most salient tile. Cross-attention learns which tiles are task-relevant and is more parameter-efficient than self-attention, as it does not need to model inter-tile dependencies.

3. **Late-Fusion Classification Head**

    - **Function**: Combines the global and local streams into a final prediction.
    - **Mechanism**: The aggregated vectors from each stream are concatenated and passed through a classification layer: $\hat{y}_i = h_\theta(\text{concat}(\mathcal{A}_\psi^G(\mathcal{G}_i), \mathcal{A}_\omega^T(\mathcal{T}_i)))$. Each aggregator contains a 2-layer MLP (embedding_dim→16→8, ReLU).
    - **Design Motivation**: Late fusion preserves modularity and interpretability; more complex early fusion offers no meaningful benefit for this task.

### Loss & Training

- Binary Cross-Entropy loss.
- The entire embedding dataset fits in a single batch on one A100 40 GB GPU; each training run takes 5–7 minutes with ~2M FLOPs per breast forward pass.
- Each experiment is repeated for 36 independent runs; the model with the highest validation AUC is selected, leveraging the low training cost to mitigate variance.
- Data splits are 70/10/20, stratified by BI-RADS label with no patient leakage.
- Non-overlapping tiles are used for classification; 75% overlapping tiles are used for attention map visualization.

## Key Experimental Results

### Main Results — EMBED + VinDr BI-RADS Classification

| Method | Trainable Params | Level | EMBED AUC↑ | EMBED Spec@Sens=0.9↑ | VinDr AUC↑ |
|--------|-----------------|-------|-----------|---------------------|-----------|
| GMIC [Shen] | 14.11M | Image | 0.816 | 0.380 | 0.899 |
| SIL IL GMIC [Pathak] | 22.49M | Image | 0.875 | 0.566 | 0.911 |
| FPN-AbMIL [Mourão] | 1.76M | Image | 0.802 | 0.367 | 0.920 |
| FPN-AbMIL (mean) | 1.76M | Breast | 0.835 | 0.403 | 0.911 |
| **MIL-PF (DINOv2 attn)** | **0.05M** | Breast | **0.916** | **0.762** | 0.894 |
| **MIL-PF (MedSigLIP attn)** | **0.04M** | Breast | 0.914 | 0.746 | **0.911** |

### Results on Additional Datasets

| Dataset | MIL-PF (DINOv2 attn) AUC | MIL-PF (MedSigLIP attn) AUC | Best Baseline AUC |
|---------|--------------------------|----------------------------|------------------|
| VinDr Calcification | 0.967 | 0.967 | 0.954 (FPN-AbMIL) |
| VinDr Mass | 0.800 | 0.814 | 0.808 (FPN-AbMIL mean) |
| RSNA Cancer | 0.923 | 0.923 | 0.914 (FPN-AbMIL mean) |

### Ablation Study — Encoder Choice and Aggregation Strategy

| Encoder | Resolution | AUC (EMBED) | Spec@Sens=0.9 |
|---------|-----------|-------------|---------------|
| DINOv2 ViT-Giant | 518×518 | 0.897 | 0.655 |
| MedSigLIP | 448×448 | 0.897 | 0.691 |
| MammoCLIP | 1520×912 | 0.870 | 0.558 |
| BiomedCLIP | 224×224 | 0.872 | 0.606 |
| DINOv3 ViT-Huge+ | 512×512 | 0.831 | 0.497 |

| Aggregation Strategy | DINOv2 AUC | DINOv2 Spec@Sens=0.9 |
|---------------------|-----------|---------------------|
| Global max + Local max | 0.905 | 0.703 |
| Global max + Local attn | **0.916** | **0.762** |

### Key Findings

- General-purpose foundation models (DINOv2/MedSigLIP) substantially outperform the domain-specific MammoCLIP in zero-shot generalization to mammography (AUC 0.897 vs. 0.870), challenging the assumption that medical imaging requires domain-specific pretraining.
- The attention aggregator demonstrates a particularly pronounced advantage on Spec@Sens=0.9 (0.762 vs. 0.703), which is the clinically more critical metric.
- DINOv3 unexpectedly underperforms (AUC 0.831), demonstrating that the latest model version is not always superior.
- Small lesion detection is constrained by tile size (448–518 pixels), with small-lesion mAP values as low as 0.1–1.2.

## Highlights & Insights

- The core insight—"frozen general-purpose foundation models applied directly to mammography yield surprisingly strong results"—challenges the assumed necessity of domain-specific pretraining and carries broad implications for the medical imaging community. It validates the robust out-of-distribution generalization of large-scale general-purpose learned representations.
- With only 40k parameters and 5–7 minutes of training, the approach substantially lowers the barrier to entry for research, particularly benefiting resource-constrained teams; precomputed features enable rapid experimental iteration and encoder comparison.
- The Perceiver-style single-query cross-attention is an elegant design choice for ROI-sparse scenarios—more parameter-efficient than self-attention and better at capturing sparse signals than mean/max pooling.
- The hierarchical MIL problem formulation is broadly transferable: the nested structure, complementary dual-stream design, and weak-label framework can be directly adapted to other high-resolution weakly supervised settings such as pathology and radiology.

## Limitations & Future Work

- On the smaller VinDr dataset, BI-RADS classification underperforms end-to-end fine-tuning (AUC 0.894 vs. 0.911), suggesting that the frozen encoder approach is less advantageous on small datasets.
- Small lesion detection is constrained by tile size, yielding extremely low mAP values (0.1–1.2); multi-scale tiling strategies could address this.
- Patient longitudinal history and bilateral symmetry information are not utilized—the authors identify these as important future directions.
- Inter-run variance is substantial (Spec@Sens=0.9 variance up to 11%), requiring 36 training runs with a selection strategy, which adds operational complexity.
- BI-RADS labels are inherently noisy with limited inter-radiologist agreement, placing an upper bound on the quality of training supervision.

## Related Work & Insights

- **vs. FPN-AbMIL/SetTrans (Mourão et al.)**: Requires 1.76–5.38M parameters for end-to-end training. MIL-PF comprehensively outperforms these methods on the largest benchmark, EMBED (AUC 0.916 vs. 0.835). The key advantages are training efficiency (35× fewer parameters) and breast-level modeling that better aligns with clinical workflows.
- **vs. GMIC/SIL IL GMIC (Shen/Pathak et al.)**: Requires 14–23M parameters. SIL IL GMIC achieves stronger performance on the smaller VinDr dataset (0.911), but is surpassed by MIL-PF on the largest and most diverse EMBED benchmark, indicating that the precomputed feature approach is more robust at scale.
- **vs. MammoCLIP**: Domain-specific pretraining underperforms general-purpose DINOv2/MedSigLIP on out-of-distribution datasets, suggesting that large-scale general-purpose pretraining yields greater cross-domain robustness.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐: The method combines existing components; the core contribution is an empirical finding rather than a methodological innovation.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Three datasets, multi-encoder comparisons, 36 independent runs, comprehensive ablations, and detection-based interpretability analysis.
- **Writing Quality** ⭐⭐⭐⭐: Problem formulation is clear and experimental design is rigorous.
- **Value** ⭐⭐⭐⭐: Significant practical value for the medical imaging community—demonstrates a low-resource, high-efficiency research pathway.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [\[CVPR 2026\] LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)
- [\[CVPR 2026\] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)

</div>

<!-- RELATED:END -->
