---
title: >-
  [Paper Note] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis
description: >-
  [CVPR2026][Medical Imaging][Computational Pathology] This paper proposes CARE, a slide-level pathology foundation model that employs an Adaptive Region Generator (ARG) to partition WSIs into morphologically coherent irregular regions (analogous to word-level tokens in NLP), combined with two-stage pretraining via cross-modal alignment with RNA/protein expression profiles. Using approximately 1/10 the data of mainstream models, CARE achieves state-of-the-art average performance across 33 downstream tasks.
tags:
  - CVPR2026
  - Medical Imaging
  - Computational Pathology
  - Whole Slide Image Analysis
  - Foundation Model
  - Adaptive Region Modeling
  - Cross-modal Alignment
  - RNA/Protein Guidance
date: 2026-05-08
content_hash: db49d9541920c7e0
---

# CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis

**Conference**: CVPR2026  
**arXiv**: [2602.21637](https://arxiv.org/abs/2602.21637)  
**Code**: [zdipath/CARE](https://github.com/zdipath/CARE)  
**Area**: Medical Imaging  
**Keywords**: Computational Pathology, Whole Slide Image Analysis, Foundation Model, Adaptive Region Modeling, Cross-modal Alignment, RNA/Protein Guidance

## TL;DR

This paper proposes CARE, a slide-level pathology foundation model that employs an Adaptive Region Generator (ARG) to partition WSIs into morphologically coherent irregular regions (analogous to word-level tokens in NLP), combined with two-stage pretraining via cross-modal alignment with RNA/protein expression profiles. Using approximately 1/10 the data of mainstream models, CARE achieves state-of-the-art average performance across 33 downstream tasks.

## Background & Motivation

**Patch dependency in existing pathology foundation models**: Current slide-level foundation models adopt natural image backbones and treat WSIs as collections of fixed-size patches, lacking the capacity to model tissue morphological heterogeneity and spatial structure.

**Absence of explicit ROI concepts**: Clinical pathologists first localize diagnostically critical ROIs when reading slides, yet typical foundation models either apply global attention over all patches or impose fixed regular partitions, failing to emulate this clinical workflow.

**Difficulties in direct patch-to-WSI aggregation**: Aggregating directly from thousands of patches to WSI-level features introduces extremely long-range interactions, making it difficult for aggregators to learn effectively; a hierarchical patch→region→WSI scheme better reflects tissue organization.

**Coarse-grained existing region partitions**: Fixed-grid patch chunks (character-level tokens) are semantically myopic, while fixed-size region chunks (uniform segmentation) frequently cause semantic misalignment, neither aligning with true tissue boundaries.

**Disconnect between molecular information and morphology**: Region construction in existing models is not guided by biological signals, offering no guarantee that learned regions correspond to underlying molecular patterns (gene expression, protein abundance).

**Low data efficiency**: Mainstream pathology foundation models typically require hundreds of thousands of WSIs for pretraining, incurring prohibitive data acquisition and computational costs, motivating the need for more efficient pretraining paradigms.

## Method

### Overall Architecture

CARE (Cross-modal Adaptive Region Encoder) comprises three core modules:

- **Adaptive Region Generator (ARG)**: Repartitions WSIs from fixed-grid sub-regions into morphologically consistent, irregular adaptive regions.
- **Adaptive Region Self-Attention (ARSA)**: Performs self-attention within each adaptive region to extract region-level features.
- **Semantic and Prior Fusion (SPF)**: Fuses coverage priors with semantic attention weights to aggregate region features into a slide-level embedding.

Pretraining follows a two-stage strategy: Stage I applies iBOT-style self-supervised pretraining (34,277 WSIs); Stage II performs cross-modal contrastive training (WSI–RNA alignment followed by WSI–protein alignment).

### Key Designs

**1. Sub-region representation**: $M$ non-overlapping $k \times k$ square sub-regions are defined on the patch grid. Each sub-region obtains a CLS-aggregated feature $g_i^{\text{CLS}}$ and a query-aggregated feature $g_i^Q$ (via cross-attention with a learnable query) through intra-region self-attention. A soft containment matrix $C$ based on anchor distances quantifies patch–sub-region spatial relationships.

**2. Adaptive Region Generator (ARG)**: For each patch, the top-3 candidate sub-regions with the highest soft containment scores are selected. A four-way cosine similarity (original/augmented patch features × CLS/query sub-region features) is computed, normalized via softmax, and averaged to obtain a semantic affinity score $\rho_{ji}$. The final selection score $w_{ji} = \rho_{ji} \cdot C_{ji}$ combines semantic affinity with spatial proximity, and each patch is assigned to the sub-region with the highest score, forming irregular adaptive regions.

**3. SPF aggregation**: The slide-level embedding is obtained by a weighted fusion of the coverage prior $\alpha_i$ (fraction of patches in a region) and the gated attention semantic weight $\beta_i$:
$$z_{\text{WSI}} = \sum_i \left(\lambda_{\text{SPF}} \alpha_i + (1 - \lambda_{\text{SPF}}) \beta_i\right) g_i^{\text{AR}}$$
The region with the highest SPF weight is selected as the ROI.

**4. Cross-modal alignment**: The RNA branch selects 3,999 genes guided by the Hallmark gene sets and encodes them with a scGPT-initialized Transformer; the protein branch takes the 10 most abundant proteins per sample and uses an ESM-2-initialized embedding table. Both stages apply a CLIP-style symmetric InfoNCE loss for alignment.

### Loss & Training

- **Main loss $\mathcal{L}_{\text{main}}$**: Stage I uses iBOT masked patch prediction and multi-view consistency; Stage II uses InfoNCE contrastive loss.
- **Region Structuring Loss $\mathcal{L}_{\text{RSL}}$**: Prevents degenerate solutions in which patches always select their nearest sub-region by pulling the global mean expected rank $\bar{E}$ toward a target value $E^\star$:
$$\mathcal{L}_{\text{RSL}} = (\bar{E} - E^\star)^2$$
- **Total loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda_{\text{RSL}} \mathcal{L}_{\text{RSL}}$

## Key Experimental Results

### Main Results

Average performance across 33 downstream tasks (morphological classification, molecular prediction, survival analysis):

| Task Category | Evaluation | CHIEF | GigaPath | TITAN | **CARE** |
|--------------|------------|-------|----------|-------|----------|
| Morphological Classification | LR | 75.6 | 78.3 | 85.4 | **85.7** |
| Morphological Classification | FT | 81.4 | 84.9 | 87.7 | 87.0 |
| Molecular Prediction | LR | 69.1 | 66.6 | 67.8 | **69.5** |
| Molecular Prediction | kNN | 65.2 | 62.0 | 66.9 | **68.0** |
| Survival Analysis | linear | 42.1 | 49.8 | 47.2 | **58.0** |

Among 33 LR benchmarks, CARE achieves state-of-the-art on 7 tasks and ranks second on 15 tasks; across AUC/F1/C-index metrics, it achieves state-of-the-art on 12 tasks and ranks second on 9.

### ROI Feature Analysis

| Method | CCRCC-BAP1 AUC | HNSCC-CASP8 AUC | Cross-LUNG AUC |
|--------|---------------|-----------------|----------------|
| WSI Features | 65.8 | 58.7 | 77.0 |
| ROI Features | **69.1** | **61.5** | **81.2** |

ROI features outperform global WSI features on tasks that rely on local signals, consistent with clinical knowledge.

### Ablation Study

- **Necessity of ARG**: Removing adaptive regions (replacing with fixed sub-regions) decreases ACC from 72.5→71.6 on EBRAINS-fine and from 64.8→61.4 on BCNB-HER2.
- **Hyperparameters**: The optimal configuration is $k=8$, $\lambda_{\text{RSL}}=0.1$, $E^\star=0.5$, $\lambda_{\text{SPF}}=0.5$.
- **Pretraining stages**: Performance improves progressively from iBOT→RNA→protein alignment; scaling self-supervised data yields substantial initial performance gains.

### Key Findings

- CARE achieves top average performance using only ~34K WSIs (approximately 1/10 of mainstream models), demonstrating the data efficiency of adaptive region modeling.
- Molecular-guided pretraining confers particularly pronounced advantages on molecular prediction tasks without degrading morphological classification performance.
- Attention map visualizations show that CARE's attended regions align with nuclear atypia and mitosis-dense areas as annotated by pathology experts.
- On RCC subtype classification, CARE surpasses the second-best model by 3.8 percentage points in balanced accuracy and 0.7 points in F1.
- On survival analysis, CARE's C-index of 58.0 substantially outperforms all baselines (second-best PRISM: 55.7), highlighting the value of molecular guidance for prognostic modeling.
- RNA alignment contributes larger gains than protein alignment during cross-modal pretraining, while proteins provide more specific complementary signals.

## Highlights & Insights

- **Adaptive regions ≈ word-level tokenization**: The proposed adaptive region scheme draws an analogy to NLP tokenization, yielding semantically more coherent partitions that better reflect tissue organization than fixed-grid patch decompositions.
- **Molecular-guided region construction**: RNA/protein expression profiles are used not only for feature alignment but also to inversely optimize region boundaries, endowing learned regions with biological meaning.
- **Exceptional data efficiency**: Pretraining on ~34K WSIs surpasses models such as GigaPath and CHIEF trained on hundreds of thousands of WSIs.
- **Automatic ROI identification**: The adaptive region with the highest SPF weight naturally emerges as the ROI, enabling ROI-level analysis without additional supervision.

## Limitations & Future Work

- ROI features do not universally outperform WSI features across all tasks, suggesting that the adaptive region's definition of "most important region" may not align with the requirements of certain tasks.
- Pretraining data (TCGA + GTEx) is predominantly from Western cohorts; generalizability to other ethnicities, rare cancer types, and non-H&E stained tissue remains to be validated.
- The DBSCAN-based sub-WSI splitting strategy introduces additional hyperparameters (≤360 patches), and its behavior on fragmented tissue is not thoroughly discussed.
- The protein branch uses only the top-10 most abundant proteins, potentially underutilizing available information; the protein dataset (8,225 pairs) is also relatively small.
- No comparison is made with recent vision–language pathology models (e.g., report- or dialogue-based PRISM) on generative tasks.
- The patch encoder is fixed as CONCH v1.5; combinations with other patch encoders (e.g., UNI, Virchow) are not explored.

## Related Work & Insights

- **Patch-level foundation models**: UNI, CONCH, PLIP, BiomedCLIP — provide high-quality patch encodings but do not support slide-level inference.
- **Slide-level foundation models**: CHIEF (anatomy-aware attention), GigaPath (LongNet for ultra-long sequences), TITAN (coordinate-aware feature grids), FEATHER (lightweight ABMIL), PRISM (report supervision), TANGLE (transcriptome alignment).
- **Hierarchical methods**: HIPT introduces region-level Transformers but applies fixed partitions; CARE's adaptive regions represent a significant improvement.
- **MIL aggregators**: ABMIL, DSMIL, DTFD-MIL focus on inter-patch interactions but all operate over fixed patch sets.
- **Cross-modal pathology**: MUSK (vision–language tile encoder) and TANGLE (transcriptome alignment) are the most closely related cross-modal works; CARE further leverages molecular signals to guide region construction rather than performing feature alignment alone.

## Rating

- Novelty: ⭐⭐⭐⭐ — The Adaptive Region Generator and molecular-guided region construction represent meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 33 downstream tasks, multiple evaluation settings, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure; the NLP tokenization analogy is intuitive.
- Value: ⭐⭐⭐⭐ — Offers a more data-efficient and clinically aligned regional modeling paradigm for pathology foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[CVPR 2026\] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [\[CVPR 2026\] Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis](sparse_task_vector_mixup_with_hypernetworks_for_efficient_knowledge_transfer_in_.md)
- [\[CVPR 2026\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)

</div>

<!-- RELATED:END -->
