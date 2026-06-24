---
title: >-
  [Paper Note] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing
description: >-
  [ICML 2025][Computational Biology][De Novo Peptide Sequencing] LIPNovo proposes a new paradigm of performing latent imputation before peptide prediction to address missing fragmentation information in mass spectrometry. By utilizing learnable peak queries and bipartite matching to impute latent representations of theoretical peaks, it significantly outperforms state-of-the-art models like CasaNovo across three benchmarks (improving amino acid precision by 5.6%-20%).
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "De Novo Peptide Sequencing"
  - "Missing Fragmentation"
  - "Latent Imputation"
  - "Bipartite Matching"
  - "Mass Spectrometry"
date: 2026-05-08
content_hash: d9d4e11ae2e20a9c
---

# Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing

**Conference**: ICML 2025  
**arXiv**: [2505.17524](https://arxiv.org/abs/2505.17524)  
**Code**: [https://github.com/usr922/LIPNovo](https://github.com/usr922/LIPNovo)  
**Area**: Bioinformatics / Proteomics  
**Keywords**: De Novo Peptide Sequencing, Missing Fragmentation, Latent Imputation, Bipartite Matching, Mass Spectrometry

## TL;DR
LIPNovo proposes a new paradigm of performing latent imputation before peptide prediction to address missing fragmentation information in mass spectrometry. By utilizing learnable peak queries and bipartite matching to impute latent representations of theoretical peaks, it significantly outperforms state-of-the-art models like CasaNovo across three benchmarks (improving amino acid precision by 5.6%-20%).

## Background & Motivation
**Background**: De novo peptide sequencing utilizes encoder-decoder architectures to encode mass spectrometry data into latent representations, followed by autoregressive prediction of amino acid sequences.

**Limitations of Prior Work**: Mass spectrometry suffers from severe fragmentation information loss due to insufficient fragmentation efficiency and instrument limitations, resulting in insufficient information for sequence prediction.

**Key Challenge**: Existing methods predict sequences directly from incomplete spectra, leading to more pronounced performance degradation as the ratio of missing fragments increases.

**Key Insight**: Imputing missing theoretical fragment spectra in the latent space before prediction, akin to the set prediction framework of DETR.

**Core Idea**: Utilizing theoretical spectra generated from known peptide sequences as supervision during training, while the imputation module operates without theoretical spectra during inference.

## Method

### Overall Architecture
Observed Spectrum → Peak Encoder → Latent Representation $z$ → Imputation Module (Transformer Decoder + Learnable Queries) → Imputed Theoretical Peak Latent Representation → Concatenated to Original $z$ → Peptide Decoder → Amino Acid Sequence.

### Key Designs

1. **Latent Space Imputation**:

    - Compute the theoretical spectrum $x'$ (the m/z values of all b- and y-ions) and encode it using the same encoder to obtain $z'$.
    - The imputation module $\Phi_\theta$ takes the observed spectrum latent representation $z$ as input to generate a fixed size of $M$ predictions.
    - Training objective: Align the predictions with the theoretical spectrum latent representation $z'$.

2. **Bipartite Matching**:

    - Inspired by DETR, the Hungarian algorithm is used to find the optimal matching $\hat{\sigma}$.
    - The matching cost combines MSE distance and confidence probability: $\mathcal{L}_{\text{Match}} = \|...\|^2 + 1 - p_{\sigma(j)}$.
    - Imputation loss combines matching MSE and binary cross-entropy (CE).

3. **Discarding Theoretical Spectrum Branch during Inference**:

    - The theoretical spectrum is required during training using ground truth labels, whereas only the predictions from the imputation module are used during inference.
    - Highly confident imputed results are concatenated into the original spectrum representation.

### Loss & Training
Total Loss = Imputation Loss $\mathcal{L}_{\text{Imputation}}$ + Peptide Prediction Cross-Entropy.

## Key Experimental Results

### Main Results

| Dataset | Metric | LIPNovo | CasaNovo | Gain |
|--------|------|---------|----------|------|
| Nine-species | AA Prec | 0.848 | 0.792 | +5.6% |
| Seven-species | AA Prec | 0.652 | 0.452 | +20.0% |
| HC-PT | AA Prec | 0.699 | 0.587 | +11.2% |

### Ablation Study

| Configuration | Nine-species AA Prec | Description |
|------|---------------------|------|
| Full LIPNovo | 0.848 | Full model |
| w/o Imputation | 0.792 | Degenerates to CasaNovo |
| w/o High-confidence filtering | 0.831 | Directly concatenate all predictions |
| w/o Bipartite matching | 0.815 | Use sequential matching |

### Key Findings
- The higher the ratio of missing fragments, the more pronounced the advantage of LIPNovo becomes.
- Imputation quality is positively correlated with peptide sequencing performance, validating the effectiveness of the imputation.
- Improvements are observed even under low missingness rates, indicating that imputation provides complementary information.

## Highlights & Insights
- The "imputation before prediction" paradigm is elegant and effective, with potential transferability to other missing data scenarios.
- Leverages the set prediction framework of DETR to handle variable-length target problems.
- Clever design where the theoretical spectrum is required as supervision during training but not during inference.

## Limitations & Future Work
- The theoretical spectrum calculation assumes a charge of +1 and uniform intensity, which may lack precision.
- The fixed prediction set size $M$ might lead to waste or insufficiency.
- Handling of post-translational modifications (PTMs) needs to be enhanced.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The imputation paradigm is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks + ablation study.
- Writing Quality: ⭐⭐⭐⭐ High clarity in the presentation of the pipeline.
- Value: ⭐⭐⭐⭐⭐ Significant driving force for proteomics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[ICLR 2026\] A New Paradigm for Genome-wide DNA Methylation Prediction Without Methylation Input](../../ICLR2026/computational_biology/a_new_paradigm_for_genome-wide_dna_methylation_prediction_without_methylation_in.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)

</div>

<!-- RELATED:END -->
