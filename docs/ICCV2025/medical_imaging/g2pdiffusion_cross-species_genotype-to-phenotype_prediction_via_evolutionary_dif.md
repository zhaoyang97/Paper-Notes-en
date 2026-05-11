---
title: >-
  [Paper Note] G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion
description: >-
  [ICCV 2025][Medical Imaging][genotype-to-phenotype] This paper proposes G2PDiffusion, the first diffusion model-based cross-species genotype-to-phenotype prediction framework…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "genotype-to-phenotype"
  - "diffusion model"
  - "multiple sequence alignment"
  - "cross-species"
  - "evolutionary biology"
date: 2026-05-08
content_hash: beeccc00ee4d6a94
---

# G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion

**Conference**: ICCV 2025
**arXiv**: [2502.04684](https://arxiv.org/abs/2502.04684)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: genotype-to-phenotype, diffusion model, multiple sequence alignment, cross-species, evolutionary biology

## TL;DR

This paper proposes G2PDiffusion, the first diffusion model-based cross-species genotype-to-phenotype prediction framework, which generates morphological images conditioned on evolutionary signals (multiple sequence alignments, MSA, and environmental context) to predict species appearance from DNA sequences.

## Background & Motivation

Understanding how genes interact with environmental factors to determine phenotype is a central challenge in genetic engineering, with significant implications for crop breeding, conservation biology, and personalized medicine. Existing genotype-to-phenotype prediction models face two major bottlenecks:

**Single-species limitation**: Traditional approaches such as GWAS and QTL mapping rely on species-specific annotated data and cannot generalize across species.

**High annotation cost**: Phenotypic traits reside in high-dimensional spaces and require specialized equipment and extensive manual annotation, with costs amplified further in cross-species studies.

The authors' core insight is to **use images as proxies for morphological phenotypes**, reformulating genotype-to-phenotype prediction as a conditional image generation task. By learning from millions of cross-species DNA–image pairs, the framework enables efficient and scalable cross-species prediction. This perspective overcomes the limitations of conventional approaches that model phenotype as numerical regression or classification outputs.

## Method

### Overall Architecture

G2PDiffusion comprises three core components: (1) an MSA retrieval engine that retrieves homologous sequences from external databases to identify evolutionarily conserved and variable regions; (2) an environment-aware MSA conditioner that integrates genetic information with environmental factors (latitude and longitude) to learn genotype–environment (GxE) interaction representations; and (3) a dynamic phenotype alignment module that enhances genotype–phenotype consistency during the diffusion denoising process. The framework generates 256×256 morphological images based on a standard conditional diffusion model.

### Key Designs

1. **Multiple Sequence Alignments Retrieval Engine**: MMseqs2 is employed to rapidly retrieve the top-$m$ homologous sequences most evolutionarily similar to the query DNA sequence from an external sequence database: $\mathcal{D}(G_q, m) = \text{top}_m(\{G_i\}, \text{MMseqs}(\cdot, G_q))$. The retrieved sequences are column-wise compared to generate an evolutionary conservation vector $V = [v_1, ..., v_l]$, where $v_i \in \{0,1\}$ indicates whether each position is conserved. DNA sequences are tokenized using $k$-mer tokenization ($k=3$). **Design Motivation**: MSA reveals evolutionary constraints and functionally conserved regions, helping distinguish which genetic variants genuinely influence phenotype.

2. **Environment-Aware MSA Conditioner**: A dual-attention mechanism is designed to process the MSA matrix. Evolutionarily-aware row attention modulates intra-sequence attention using gating weights $\mathbf{w}_v$ derived from the conservation vector: $H^{row}_{i,:} = \text{Softmax}\left(\frac{QK^\top \odot \mathbf{w}_v}{\sqrt{d}}\right)V$. Environment-aware column attention maps latitude and longitude to spherical coordinates $(x,y,z) = (\cos\beta\cos\lambda, \cos\beta\sin\lambda, \sin\beta)$ and passes them through an MLP to produce environmental weights $\mathbf{w}_e$ that modulate cross-sequence attention. Row and column attention outputs are merged via LayerNorm: $H^{MSA} = \text{LayerNorm}(H^{row} + H^{col})$. **Design Motivation**: Row attention captures evolutionary constraints while column attention injects the influence of environmental selection pressure.

3. **Dynamic Alignment Sampling**: An alignment model $g_\phi(X_t, t)$ is trained to align the noisy image embeddings at each diffusion timestep to DNA embeddings, using a contrastive learning loss: $\mathcal{L}_{align} = -\log\frac{\exp[g_\phi(X_t,t) \cdot C^+]}{\sum_{j=1}^{B}\exp[g_\phi(X_t,t) \cdot C_j]}$. During sampling, alignment signals are injected into the denoising process via gradient guidance. **Design Motivation**: Compared to directly applying CLIP loss for guidance, dynamic alignment better accommodates the noise characteristics at different timesteps of the diffusion process.

### Loss & Training

- The diffusion model is trained with the standard denoising objective: $L_{DM} = \mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(X_t, t, C)\|_2^2]$
- The alignment model is trained with the contrastive loss $\mathcal{L}_{align}$
- Training is conducted on 8× A100 GPUs using the Adam optimizer, with a learning rate of 1e-5, batch size of 128, 100k training steps, and cosine annealing
- During sampling, guidance strength $w$ controls the magnitude of alignment guidance

## Key Experimental Results

### Main Results

**CLIBDScore and Success Rate (BIOSCAN-5M Seen Set)**:

| Method | CLIBDScore Top-1 | Top-5 | Success Rate Top-1 | Top-5 | Top-100 |
|--------|-----------------|-------|--------------------|-------|---------|
| Random | 0.005 | - | 4.4% | - | - |
| DF-GAN | 0.054 | 0.154 | 5.6% | 18.7% | 52.6% |
| Stable Diffusion | 0.100 | 0.219 | 11.5% | 36.6% | 74.8% |
| ControlNet | 0.107 | 0.228 | 12.4% | 39.1% | 77.0% |
| **G2PDiffusion** | **0.182** | **0.302** | **31.7%** | **65.8%** | **94.0%** |

**PES (Phenotype Embedding Similarity)**:

| Method | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|--------|-------|-------|--------|--------|---------|
| DF-GAN | 0.021 | 0.134 | 0.167 | 0.276 | 0.301 |
| Stable Diffusion | 0.062 | 0.207 | 0.240 | 0.349 | 0.389 |
| ControlNet | 0.061 | 0.212 | 0.254 | 0.359 | 0.403 |
| **G2PDiffusion** | **0.152** | **0.291** | **0.346** | **0.478** | **0.511** |

### Ablation Study

**Effect of the Environment-Aware MSA Conditioner and Dynamic Alignment**:

| Configuration | CLIBDScore Top-1/5 | Success Rate Top-1/5 | PES Top-1/5 |
|---------------|--------------------|----------------------|-------------|
| Baseline (DNABERT) | 0.100/0.219 | 11.5%/26.6% | 0.062/0.187 |
| + Conditioner | 0.125/0.235 | 16.7%/28.2% | 0.098/0.254 |
| + Alignment | 0.167/0.289 | 27.1%/51.2% | 0.137/0.268 |
| **+ Both (Full)** | **0.182/0.302** | **31.7%/65.8%** | **0.152/0.291** |

**Effect of MSA Retrieval Count $m$**:

| $m$ | CLIBDScore Top-1 | Success Rate Top-5 | PES Top-1 |
|-----|-----------------|-------------------|-----------|
| 0 (no MSA) | 0.178 | 53.1% | 0.151 |
| 1 | **0.193** | **65.8%** | 0.143 |
| 2 (default) | 0.182 | 65.8% | **0.152** |
| 3 | 0.166 | 58.9% | 0.128 |

### Key Findings

- G2PDiffusion substantially outperforms all baselines across all metrics; Top-5 success rate reaches 65.8% vs. 39.1% for ControlNet.
- The environment-aware conditioner and dynamic alignment each contribute meaningfully, with their combination yielding the best performance (success rate improves from 11.5% to 31.7%).
- MSA retrieval with $m=1$ or $m=2$ performs best; retrieving too many sequences introduces noisy alignments that degrade performance.
- **Generalization to unseen species**: Top-100 success rate reaches 80.3% on unseen species, demonstrating robust cross-species generalization.
- The generative model can produce images of the same genotype from multiple viewpoints, indicating an implicit understanding of 3D structure.

## Highlights & Insights

1. **Paradigm innovation through problem reformulation**: Redefining genotype-to-phenotype prediction from numerical regression to conditional image generation simultaneously addresses cross-species generalization and high-dimensional phenotype modeling.
2. **Elegant use of evolutionary signals**: The dual row/column attention mechanism separately models intra-sequence evolutionary constraints and cross-sequence environmental selection pressure, yielding a formulation that is both mathematically elegant and biologically grounded.
3. **Comprehensive evaluation framework**: Three complementary metrics—CLIBDScore, success rate, and PES—are proposed to assess performance from the perspectives of DNA–image alignment, statistical thresholding, and phenotype embedding space similarity.
4. **Spherical coordinate environmental encoding**: Mapping latitude and longitude to spherical coordinates rather than planar coordinates correctly reflects the geometric properties of the Earth's surface.

## Limitations & Future Work

1. Validation is limited to an insect specimen dataset (BIOSCAN-5M) and has not been extended to other taxonomic groups such as plants or mammals.
2. Environmental factors are represented solely by latitude and longitude, lacking richer ecological variables such as temperature, humidity, and elevation.
3. DNA barcodes typically cover only a subset of genes (e.g., COI) and may fail to capture all genetic variation influencing phenotype.
4. Top-1 success rate of 31.7% remains insufficient for practical deployment, requiring multiple sampling iterations to obtain optimal outputs.
5. Quantitative evaluation of fine-grained phenotypic features in generated images (e.g., wing venation, setae) is lacking.
6. Interpretability analyses linking phenotypic variation to specific genetic loci represent a promising direction for future work.

## Related Work & Insights

- **BIOSCAN-5M**: The largest multimodal insect specimen dataset, providing paired DNA–image–taxonomy–geography data.
- **CLIBD**: A CLIP-style contrastive learning model for DNA–image alignment, whose embeddings are used to construct the CLIBDScore evaluation metric in this work.
- **Stable Diffusion / ControlNet**: Representative conditional image generation methods serving as diffusion-based baselines.
- **AlphaFold**: A paradigm-shifting approach to protein structure prediction; this work pursues an analogous breakthrough at the genotype–phenotype level.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first diffusion model for genotype-to-phenotype prediction; both the problem formulation and methodological design are pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A comprehensive evaluation framework, thorough ablation studies, and unseen-species generalization tests are included.
- **Writing Quality**: ⭐⭐⭐⭐ — Biological motivation is clearly articulated, though the density of LaTeX equations in the methods section somewhat impairs readability.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for AI-assisted genomic analysis; while practical application remains distant, the long-term prospects are substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](../../CVPR2026/medical_imaging/forecasting_epileptic_seizures_from_contactless_ca.md)
- [\[NeurIPS 2025\] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](../../NeurIPS2025/medical_imaging/fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[AAAI 2026\] Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction](../../AAAI2026/medical_imaging/cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)
- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/medical_imaging/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)

</div>

<!-- RELATED:END -->
