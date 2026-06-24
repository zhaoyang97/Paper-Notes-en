---
title: >-
  [Paper Note] Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra
description: >-
  [AAAI 2026][Image Generation][Mass spectrometry] This paper proposes GLMR, a two-stage framework (contrastive pre-retrieval + generative language model re-ranking) that transforms cross-modal retrieval into unimodal retrieval by generating molecular structures aligned with input mass spectra, achieving over 40% improvement in Recall@1 on MassSpecGym.
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Mass spectrometry"
  - "molecule retrieval"
  - "cross-modal alignment"
  - "generative retrieval"
  - "contrastive learning"
date: 2026-05-08
content_hash: 0b8a1bd4fcadb5a3
---

# Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra

**Conference**: AAAI 2026
**arXiv**: [2511.06259](https://arxiv.org/abs/2511.06259)  
**Code**: None  
**Area**: Cross-modal Retrieval / Generative Language Models
**Keywords**: Mass spectrometry, molecule retrieval, cross-modal alignment, generative retrieval, contrastive learning

## TL;DR

This paper proposes GLMR, a two-stage framework (contrastive pre-retrieval + generative language model re-ranking) that transforms cross-modal retrieval into unimodal retrieval by generating molecular structures aligned with input mass spectra, achieving over 40% improvement in Recall@1 on MassSpecGym.

## Background & Motivation

Tandem mass spectrometry (MS/MS) is a core tool for molecular structure identification, and retrieving matched molecular structures from spectra is a fundamental step in metabolomics, drug development, and related fields. Existing methods face two major bottlenecks:

**Spectral library matching**: Traditional methods compare experimental spectra against reference spectra of known compounds, but are limited by library coverage and cannot handle out-of-library compounds.

**Cross-modal representation learning**: Recent deep learning methods (e.g., MIST, JESTR) encode mass spectra and molecular structures into a shared latent space for retrieval. However, mass spectra describe physical fragmentation behavior while molecular structures describe chemical connectivity — these are fundamentally different modalities, leading to severe **modality misalignment**. The current SOTA method JESTR achieves less than 20% top-1 accuracy on MassSpecGym.

Core insight: Rather than enforcing hard alignment between two modalities, a generative model can produce a molecule aligned with the input spectrum, converting cross-modal retrieval into unimodal molecule-to-molecule retrieval.

## Method

### Overall Architecture

GLMR adopts a two-stage retrieval strategy:

- **Pre-Retrieval**: Contrastive learning trains spectrum and molecule encoders to retrieve top-K candidate molecules from a library as contextual priors.
- **Generative Retrieval**: Candidate molecules and input spectrum features are fused to guide a generative language model in producing a refined molecular structure, which is then used to re-rank candidates by molecular similarity.

### Key Designs

#### 1. Cross-modal Contrastive Learning (Pre-Retrieval)

**Molecule encoder**: ChemFormer (a BART variant pre-trained on the ZINC database) takes SMILES sequences as input and uses the hidden state of the [CLS] token as the global molecular embedding $\mathbf{E}^m$.

**Spectrum encoder**: A Transformer with multi-head attention. Unlike the binning strategies of JESTR/CMSSP, each spectrum is represented as a sequence of (m/z, intensity) tuples with intensity normalized to $(0,1]$, and a fixed-size representation $\mathbf{E}^s$ is obtained via average pooling.

**Training objective**: Bidirectional Info-NCE loss following the CLIP framework. For the mol2ms direction, $N$ negative samples are constructed by applying random intensity perturbations to spectral peaks; for ms2mol, $M$ negative samples are drawn from other molecules in the same batch. Temperature $\tau$ controls the sharpness of the similarity distribution. At retrieval time, candidates are ranked by cosine similarity and the top-K are passed to the next stage.

#### 2. Context-Aware Generative Retrieval

**Cross-Fusion module**: A cross-attention mechanism fuses spectrum features $\mathbf{H}^s$ and top-K candidate molecule features $\mathbf{H}^m_K$, using the spectrum as Query and candidate molecules as Key/Value, enabling selective focus on the most informative candidates. During training, both encoders are **frozen**; only the fusion module and decoder are updated.

**ChemFormer Decoder**: Autoregressively generates SMILES strings, trained to maximize the conditional likelihood.

**Re-ranking mechanism**: The generated molecule is encoded as $\mathbf{E}^m_+$, and its cosine similarity to each candidate molecule embedding is computed for re-ranking, fully converting cross-modal retrieval into unimodal comparison.

### Loss & Training

- **Stage 1**: Contrastive learning for 300 epochs; molecule encoder (ChemFormer pre-trained weights) is frozen; only the spectrum encoder is updated.
- **Stage 2**: Generative training for 30 epochs; both encoders are frozen; only the Cross-Fusion module and ChemFormer decoder are updated.
- Pre-retrieval candidate count $K=40$ (ablation studies show diminishing returns beyond $K>40$).

## Key Experimental Results

### Main Results

**Table 1: Retrieval Performance on MassSpecGym**

| Library Type | Method | Recall@1 | Recall@5 | MRR | MCES@1↓ |
|---|---|---|---|---|---|
| Weight-based | JESTR | 17.62 | 40.36 | 29.12 | 15.82 |
| Weight-based | MIST | 18.46 | 40.01 | 29.30 | 15.37 |
| Weight-based | **GLMR** | **64.17** | **72.96** | **67.82** | **11.14** |
| Formula-based | JESTR | 11.77 | 33.26 | 22.83 | 11.73 |
| Formula-based | **GLMR** | **68.48** | **78.09** | **72.47** | **5.05** |

GLMR surpasses JESTR by approximately **46 percentage points** in Recall@1 on the weight-based library.

**Table 2: Zero-shot Transfer on MassRET-20k**

| Library Type | Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| Weight-based | JESTR | 16.49 | 38.45 | 27.45 |
| Weight-based | **GLMR** | **54.04** | **64.35** | **58.84** |
| Formula-based | JESTR | 7.44 | 23.31 | 16.28 |
| Formula-based | **GLMR** | **51.14** | **60.06** | **55.57** |

### Ablation Study

**Contribution of Each Stage (Weight-based)**

| Configuration | Recall@1 | MRR |
|---|---|---|
| Pre-retrieval only | 20.34 | 32.19 |
| Generative retrieval only | 41.50 | 49.71 |
| **Full GLMR** | **64.17** | **67.82** |

Generative retrieval alone already outperforms pre-retrieval; combining both stages yields the best performance.

### Key Findings

1. **Modality gap visualization**: KDE analysis shows the modality gap distribution shifts significantly leftward after generative retrieval, confirming that the generated molecules effectively bridge the spectrum–molecule modality gap.
2. **Generation quality**: The generative model ranks second in MCES (after DiffMS only), demonstrating chemical structural validity.
3. **Sensitivity to K**: Performance saturates for $K > 40$.

## Highlights & Insights

- **Paradigm shift**: Cross-modal retrieval is reformulated as unimodal retrieval via generative modeling, a strategy generalizable to other cross-modal scenarios.
- **Complementary two-stage design**: Pre-retrieval provides contextual priors; generative retrieval leverages these priors for further refinement.
- **New benchmark MassRET-20k**: Covers 12 ionization adduct types with complete collision energy metadata, better reflecting real-world conditions.

## Limitations & Future Work

1. The generative stage requires encoding, fusing, and decoding $K=40$ candidates, resulting in relatively high inference cost.
2. Generated SMILES may be chemically invalid, as no explicit chemical constraints are enforced.
3. Evaluation is limited to positive ion mode; generalizability to negative ion mode remains unverified.

## Related Work & Insights

- Compared to CLIP-style methods (MIST, JESTR), GLMR's innovation lies in going beyond alignment by incorporating generation to compensate for the limits of direct alignment.
- The generative retrieval paradigm is transferable to other retrieval tasks with large modality gaps (e.g., text → protein).

## Rating

- **Novelty**: ★★★★☆ — Generative retrieval paradigm is novel in the mass spectrometry domain
- **Technical Depth**: ★★★★☆ — Two-stage design is well-motivated; Cross-Fusion is effective
- **Experiments**: ★★★★★ — Substantial gains (40%+), with a new benchmark and comprehensive ablations
- **Writing Quality**: ★★★★☆ — Motivation is clear; figures and tables are professional
- **Value**: ★★★☆☆ — High inference cost and lack of chemical constraints are practical concerns

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] IRGen: Generative Modeling for Image Retrieval](../../ECCV2024/image_generation/irgen_generative_modeling_for_image_retrieval.md)
- [\[CVPR 2026\] UnicEdit-10M: A Dataset and Benchmark Breaking the Scale-Quality Barrier via Unified Verification for Reasoning-Enriched Edits](../../CVPR2026/image_generation/unicedit-10m_a_dataset_and_benchmark_breaking_the_scale-quality_barrier_via_unif.md)
- [\[CVPR 2026\] IncreFA: Breaking the Static Wall of Generative Model Attribution](../../CVPR2026/image_generation/increfa_breaking_the_static_wall_of_generative_model_attribution.md)
- [\[AAAI 2026\] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective](enhancing_multimodal_misinformation_detection_by_replaying_the_whole_story_from_.md)
- [\[AAAI 2026\] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)

</div>

<!-- RELATED:END -->
