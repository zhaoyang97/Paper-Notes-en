---
title: >-
  [Paper Note] SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model
description: >-
  [ICML2025][Computational Biology][DNA Foundation Models] Proposes SPACE (Species-Profile Adaptive Collaborative Experts), demonstrating that **supervised genomic profile prediction** learns more effective DNA representations than unsupervised sequence pre-training, and achieves state-of-the-art (SOTA) performance on 11 out of 18 downstream NT tasks using a species-aware MoE encoder and a dual-gated decoder.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "DNA Foundation Models"
  - "Genomic Profile Prediction"
  - "Mixture of Experts"
  - "Cross-species Modeling"
  - "Supervised Pre-training"
date: 2026-05-08
content_hash: 814f258dd0c1987d
---

# SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model

**Conference**: ICML2025  
**arXiv**: [2506.01833](https://arxiv.org/abs/2506.01833)  
**Code**: [ZhuJiwei111/SPACE](https://github.com/ZhuJiwei111/SPACE)  
**Area**: Genomics / DNA Foundation Models  
**Keywords**: DNA Foundation Models, Genomic Profile Prediction, Mixture of Experts, Cross-species Modeling, Supervised Pre-training

## TL;DR

Proposes SPACE (Species-Profile Adaptive Collaborative Experts), demonstrating that **supervised genomic profile prediction** learns more effective DNA representations than unsupervised sequence pre-training, and achieves state-of-the-art (SOTA) performance on 11 out of 18 downstream NT tasks using a species-aware MoE encoder and a dual-gated decoder.

## Background & Motivation

- **Dilemma of DNA Foundation Models**: Existing DNA foundation models (e.g., DNABERT, HyenaDNA, Nucleotide Transformer) copy the NLP paradigm (MLM/NTP) for unsupervised pre-training. However, unlike natural language, the DNA sequence itself lacks intrinsic semantics—its function is regulated by **genomic profiles** such as chromatin accessibility, epigenetic modifications, and transcription factor binding.
- **Value of Supervised Signals**: Genomic Profile Prediction Models (GPPMs) like Enformer learn DNA representations by predicting experimentally measurable genomic profiles, which naturally contain richer biological contextual information. However, there has been a lack of systematic studies to exploit their potential for representation learning.
- **Limitations of Prior Work**:
    - Shared encoders fail to capture species-specific features (e.g., differences in regulatory mechanisms between humans and mice).
    - Independent prediction heads ignore functional dependencies among different genomic profiles (e.g., highly expressed regions are typically accompanied by high chromatin accessibility).

## Method

### Overall Architecture

SPACE adopts a three-stage design:

1. **CNN Local Context Aggregation**: Following Enformer, a 1D-CNN module is used to compress the raw nucleotide sequences, generating hidden states $h_m \in \mathbb{R}^{L \times d_h}$ at a 128bp resolution.
2. **Species-aware Transformer Encoder**: Cross-species modeling based on sparse MoE.
3. **Profile-grouped Enhancement Decoder**: Cross-profile relationship modeling based on dual-gated MoE.

### Species-aware Encoder

**Species-Specific Embeddings**: Introduce a learnable embedding $e_m \in \mathbb{R}^{1 \times d_h}$ for each species $S_m$. This embedding is concatenated with the sequence hidden states and then fed into the Transformer layers, explicitly guiding the model to distinguish among species.

**Cross-species MoE**: Each MoE layer comprises $N$ shared experts $\{E_1, \dots, E_N\}$ and $M$ species-specific gating networks $\{G_1, \dots, G_M\}$. For the hidden state of species $S_m$, the output is:

$$\hat{h}_m = \text{MHAttention}([e_m, h_m])$$

$$y_m = \sum_{k=1}^{N} G_m(\hat{h}_m)_k \cdot E_k(\hat{h}_m)$$

The gating weights are computed via TopK sparse routing with noise injection:

$$G_m(\hat{h}_m) = \text{Softmax}(\text{TopK}(g(\hat{h}_m) + \mathcal{R}_{\text{noise}}))$$

**Mutual Information Loss**: Maximize the mutual information between the species identity $S$ and the expert selection $E$ to encourage experts to learn species-specific patterns:

$$\mathcal{L}_{\text{MI}} = -MI(S; E) = -H(S) - H(E) + H(S, E)$$

### Profile-grouped Enhancement Decoder

Designed based on two biological principles: (P1) evolutionary conservation—homologous profiles across species share regulatory mechanisms; (P2) functional interdependence—different profiles are regulated by shared mechanisms.

**Profile Classification**: Classify the initial predictions $o_{\text{base}}$ into $Q$ categories based on experimental types (DNase/ATAC-seq, TF ChIP-seq, Histone ChIP-seq, CAGE).

**Dual-Gated Expert Weighted Aggregation**:

- **Group-level Gating** (First Layer): Dynamically weights $R$ expert groups based on species embeddings and sequence context to capture evolutionary conservation patterns.

$$\hat{G}^q = \text{Softmax}(G^q_{\text{species}}(e) + G^q_{\text{sequence}}(\text{Pool}(y)))$$

- **Expert-level Gating** (Second Layer): Selects specific experts within each group based on the prediction patterns to capture functional dependencies.

$$o^q_{\text{enhanced}} = \sum_{r=1}^{R} \hat{G}_r^q \cdot \left(\sum_{k=1}^{K} G_r^q(o^q)_k \cdot E_k(o^q)\right)$$

The final prediction is computed via a residual connection: $o_{\text{final}} = o_{\text{base}} + o_{\text{enhanced}}^T$.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Poisson}} - \alpha \sum_{d=1}^{D} MI(S; E_d)$$

where the Poisson negative log-likelihood serves as the main loss, and $\alpha=0.01$ controls the strength of the mutual information regularization.

## Key Experimental Results

### NT Downstream Tasks (18 Datasets)

| Model | H3K4me3 | H3K9ac | Enhancers | Enhancers(types) | Donors | Acceptors |
|------|---------|--------|-----------|------------------|--------|-----------|
| DNABERT-2 | 0.646 | 0.564 | 0.517 | 0.476 | 0.837 | 0.855 |
| NT-Multi (2.5B) | 0.618 | 0.527 | 0.527 | 0.484 | 0.958 | 0.964 |
| Enformer | 0.635 | 0.593 | 0.614 | 0.573 | 0.749 | 0.739 |
| **SPACE** | **0.661** | **0.635** | **0.631** | **0.583** | **0.942** | **0.902** |

SPACE achieves **SOTA on 11 out of 18 tasks**, while having significantly fewer parameters than NT-Multispecies (2.5B).

### GUE Cross-species Verification (Yeast + Virus)

| Task | Enformer | SPACE | Gain |
|------|----------|-------|------|
| H3 | 70.65 | 79.53 | +8.88 |
| H3K14ac | 37.87 | 54.12 | +16.25 |
| H3K4me3 | 22.19 | 49.47 | **+27.28** |
| H4ac | 32.90 | 53.09 | +20.19 |
| Covid | 61.33 | 70.26 | +8.93 |

SPACE achieves substantial improvements over Enformer (up to +27.28) on yeast and viral genomes that are **phylogenetically extremely distant** from the training species.

### Genomic Benchmarks

| Task | Enformer | SPACE |
|------|----------|-------|
| Mouse Enhancers | 0.835 | **0.905** |
| Drosophila Enhancers | 0.613 | **0.721** |
| Human Enhancer Ensembl | 0.844 | **0.919** |
| Human Regulatory | 0.903 | **0.944** |

## Highlights & Insights

1. **Challenging the Paradigm**: Systematically demonstrates the crucial conclusion that "supervised genomic profile prediction > unsupervised sequence pre-training", offering a new paradigm for the DNA foundation model field.
2. **Exquisite Application of MoE**: The encoder MoE is utilized for cross-species knowledge routing, and the decoder dual-gated MoE is used for cross-profile relationship modeling; both MoE designs have clear biological motivations.
3. **Cross-species Generalization**: Explicitly guides expert-species alignment through mutual information loss, demonstrating outstanding generalization capabilities on phylogenetically distant species (yeast/virus).
4. **Comprehensive Evaluation**: Covers three major evaluation benchmarks—NT downstream tasks, GUE cross-species benchmarks, and Genomic Benchmarks—providing highly convincing results.

## Limitations & Future Work

1. **Limited Training Species**: Pre-trained only on humans and mice, without including more species (e.g., plants, insects); thus, the scalability of the species MoE has not yet been verified.
2. **Computational Overhead**: The MoE architecture increases model complexity, and the paper does not thoroughly analyze the inference efficiency and parameter size comparison.
3. **Long Sequence Modeling**: The input length follows Enformer's ~196K bp window, and modeling of longer-range regulatory elements has not been explored.
4. **Dependency of Profile Classification on Priors**: The grouping operator $\Phi$ in the decoder is based on manually defined experimental types, which may limit the discovery of unrevealed profile relationships.
5. **Narrow Downstream Tasks**: Mainly focuses on classification tasks, lacking in-depth evaluation on regression tasks such as gene expression prediction and variant effect prediction.

## Related Work & Insights

- **Enformer** (Avsec et al., 2021): Direct baseline of SPACE, sharing the CNN front-end and Poisson loss.
- **Nucleotide Transformer** (Dalla-Torre et al., 2024): A representative 2.5B large model of the unsupervised route, which SPACE outperforms with far fewer parameters.
- **DNABERT-2** (Zhou et al., 2024): A representative of MLM pre-training with improved tokenization.
- **Mod-Squad** (Chen et al., 2023): The inspiration for the mutual information loss, used for expert specialization in multi-task learning.

## Rating
- Novelty: ⭐⭐⭐⭐ — Systematic demonstration of supervised pre-training outperforming unsupervised methods + dual-layer application of MoE in genomics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Complete validation across three major benchmarks, multiple species, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and rigorous methodological derivation.
- Value: ⭐⭐⭐⭐ — Provides important references for paradigm selection in the DNA foundation model community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](../../NeurIPS2025/computational_biology/janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[ICML 2025\] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models](ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](../../NeurIPS2025/computational_biology/iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)

</div>

<!-- RELATED:END -->
