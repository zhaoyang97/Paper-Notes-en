---
title: >-
  [Paper Note] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models
description: >-
  [ICML2025][Computational Biology][Latent Diffusion Models] This work proposes LDMol, which constructs a structure-aware latent space through SMILES-enumeration contrastive learning. A conditional latent diffusion model is then trained on this space to achieve text-to-molecule generation, outperforming autoregressive (AR) models on text-based data generation tasks for the first time.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Latent Diffusion Models"
  - "Text-to-Molecule Generation"
  - "Contrastive Learning"
  - "SMILES"
  - "Structure-Aware Representation"
date: 2026-05-08
content_hash: c078f17acd4d5a1e
---

# LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models

**Conference**: ICML2025  
**arXiv**: [2405.17829](https://arxiv.org/abs/2405.17829)  
**Code**: [jinhojsk515/LDMol](https://github.com/jinhojsk515/LDMol)  
**Area**: Molecule Generation / Drug Discovery  
**Keywords**: Latent Diffusion Models, Text-to-Molecule Generation, Contrastive Learning, SMILES, Structure-Aware Representation

## TL;DR

This work proposes LDMol, which constructs a structure-aware latent space through SMILES-enumeration contrastive learning. A conditional latent diffusion model is then trained on this space to achieve text-to-molecule generation, outperforming autoregressive (AR) models on text-based data generation tasks for the first time.

## Background & Motivation

- **Difficulties of diffusion models in molecular generation**: Molecular data is inherently discrete (atoms/bond types, SMILES tokens). Training diffusion models directly on raw molecular representations makes it difficult to handle complex conditional inputs (such as natural language).
- **Limitations of prior molecular diffusion models**: Models like TGM-DLM perform continuous Gaussian diffusion directly on SMILES token sequences, which introduces unreasonable numerical noise, leading to poor generation quality and weak conditional-following capability.
- **The latent space design is key**: Although simple autoencoders (e.g., $\beta$-VAE) can provide a continuous and reconstructible latent space, their features do not guarantee representation of molecular structural characteristics. In $\beta$-VAE, the feature distance between different SMILES enumerations of the same molecule is not significantly different from that of random molecule pairs.
- **Key Insight**: A carefully designed, chemical structure-rich latent space can substantially improve the performance of diffusion models in molecular generation.

## Method

LDMol consists of three stages: (1) contrastive pre-training of the SMILES encoder, (2) training the compression layer and decoder, and (3) training the text-conditional latent diffusion model.

### Stage 1: Structure-Aware SMILES Encoder (Contrastive Learning)

Positive sample pairs are constructed using **SMILES enumeration** (where a single molecule yields multiple equivalent SMILES representations based on different node traversal orders), while SMILES of different molecules serve as negative pairs.

The key argument for contrastive learning: the mutual information between SMILES enumeration pairs is minimal (only sharing the molecular structural identity). This forces the encoder to understand the complete connectivity of the molecular graph to recognize all enumerated variants, thereby encoding **distinct structural features**.

Symmetric InfoNCE loss:

$$\mathcal{L}_{enc}(M, M') = \mathcal{L}_{con}(M, M') + \mathcal{L}_{con}(M', M)$$

where:

$$\mathcal{L}_{con}(M, M') = -\sum_{k=1}^{N} \log \frac{\exp(v_k \cdot v_k' / \tau)}{\sum_{i=1}^{N} \exp(v_k \cdot v_i' / \tau)}$$

$v_k$ and $v_k'$ are the vectors after linear projection and normalization of the [SOS] tokens outputted by the encoder for the positive sample pair $m_k, m_k'$, respectively. The training data consists of 10 million molecules from PubChem. A **hard negative** strategy is also introduced: stereoisomers are treated as difficult negative samples to enhance the encoder's sensitivity to stereochemical information.

### Stage 2: Compression Layer + Autoregressive Decoder

- **Linear compression layer** $f(\cdot)$: Compresses the encoder output from $[L \times d_{enc}]$ to $[L \times d_z]$ to reduce the curse of dimensionality while retaining structure-aware features. It is deliberately kept simple (only a linear layer) to avoid deviating from the pre-trained feature space.
- **Autoregressive Transformer Decoder**: Reconstructs SMILES from the compressed latent vector $f(\mathcal{E}(m))$ via cross-attention, using the standard next-token prediction loss:

$$\mathcal{L}_{dec} = -\sum_{i=1}^{n} \log p(t_n | t_{0:n-1}, f(\mathcal{E}(m)))$$

During training, the encoder parameters are frozen, and only the compression layer and decoder are trained. The reconstruction accuracy is approximately **98%**.

### Stage 3: Text-Conditional Latent Diffusion Model

- **Diffusion Target Domain**: The compressed latent space $[L \times d_z]$.
- **Architecture**: A **DiT-base** (Transformer-based diffusion) model is adopted instead of UNet, as the spatial inductive bias of UNet is unsuitable for SMILES latent spaces. Text conditions are injected via cross-attention.
- **Text Encoder**: The encoder component of MolT5-large.
- **Conditional Diffusion Training Loss**:

$$\theta^* = \arg\min_\theta \mathbb{E}_{x_0, c, t, \epsilon} \| \epsilon - \epsilon_\theta(x_t, t, c) \|_2^2$$

- **Sampling**: DDIM with 100 steps + classifier-free guidance (the conditioning text is replaced with an empty string with a 3% probability during training, and $\omega=2.5$ during inference).
- **Training Data**: PubchemSTM + ChEBI-20 + PCdes containing about 320,000 pairs in total, which is much smaller than the million-level dataset size of baselines like MolT5/bioT5.

## Key Experimental Results

### Text-to-Molecule Generation (ChEBI-20 Test Set)

| Model | Type | Validity↑ | BLEU↑ | Exact Match↑ | FCD↓ | Morgan FTS↑ |
|------|------|-----------|-------|--------------|------|-------------|
| bioT5+ | AR | 1.000 | 0.872 | 0.522 | 0.35 | 0.779 |
| bioT5 | AR | 1.000 | 0.867 | 0.413 | 0.43 | 0.734 |
| MolT5-large | AR | 0.905 | 0.854 | 0.311 | 1.20 | 0.684 |
| TGM-DLM | DM | 0.871 | 0.826 | 0.242 | 0.77 | 0.688 |
| **LDMol** | **DM** | **0.941** | **0.926** | **0.530** | **0.20** | **0.931** |

LDMol substantially surpasses AR and DM baselines on almost all metrics. The FCD drops from 0.35 in bioT5+ to 0.20, and the Morgan FTS increases from 0.779 to 0.931.

### Molecule-to-Text Retrieval (64-way Accuracy)

| Model | PCdes-Paragraph | MoMu-Paragraph |
|------|-----------|----------|
| MolCA | 86.4% | 73.4% |
| **LDMol (n=25)** | **90.3%** | **87.1%** |

### Ablation Study

| Model Variant | Reconstruction Accuracy | Validity | Match | FCD |
|---------|-----------|----------|-------|-----|
| Without Contrastive Learning | 1.000 | 0.019 | 0.000 | 58.60 |
| β-VAE (β=0.001) | 0.999 | 0.847 | 0.492 | 0.34 |
| **LDMol (Full)** | 0.983 | **0.941** | **0.530** | **0.20** |

Without contrastive pre-training, the diffusion model fails to learn the latent distribution completely (Validity is only 0.019); $\beta$-VAE works but underperforms the structure-aware LDMol significantly.

## Highlights & Insights

1. **First time a diffusion model surpasses autoregressive models in generating text-like data**: SMILES is essentially a text format. LDMol's outperformance of AR models on text-based data generation carries paradigmatic significance.
2. **Clever design of using SMILES enumeration as a contrastive learning augmentation**: Naturally construct positive sample pairs by utilizing the traversal invariance of molecular graphs, which theoretically minimizes mutual information without losing information.
3. **Versatility of the diffusion model**: The same trained LDMol can be leveraged for molecule-text retrieval (using noise prediction error as matching alignment) and text-guided molecular editing (inspired by DDS) without additional training.
4. **Data efficient**: Achieves better performance than AR baselines with only 320k training pairs, which is significantly fewer than the millions of data pairs used by the baselines.
5. **Methodological inspiration for latent space design**: A well-designed latent space should not only be reconstructible but also encode semantic/structural information, which serves as a general guide for all latent diffusion models.

## Limitations & Future Work

1. **Insufficient conditioning-following capability for complex biological properties**: The generation accuracy under descriptions of complex biological activities still needs enhancement.
2. **Validity has not reached 100%**: Although 0.941 is high, bioT5+ achieves 1.0, indicating robustness challenges for the decoder when processing latent vectors generated by the diffusion model.
3. **Inference speed**: DDIM 100-step sampling + autoregressive decoder generation incurs higher inference overhead than autoregressive models that only require direct decoding.
4. **Dependence on SMILES representation**: No exploration of whether other molecular representations, such as molecular graphs, could benefit further from this framework.
5. **Fixed text encoder**: Uses a frozen MolT5-large, leaving the potential of combining with stronger LLMs unexplored.

## Related Work & Insights

- **TGM-DLM** (Gong et al., 2024): Direct training of diffusion models on SMILES tokens resulted in poor performance, verifying the difficulty of raw discrete spaces.
- **Stable Diffusion / LDM** (Rombach et al., 2022): The paradigm origin for latent diffusion models in the image domain.
- **DiT** (Peebles & Xie, 2023): Transformer-based diffusion model architecture.
- **DDS** (Hertz et al., 2023): Text-guided image editing method, adapted to molecular editing.
- **Inspirations for Natural Language Diffusion Models**: LDMol's success implies that, with better latent space design, diffusion models might catch up with AR models in general text generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Clever combination of SMILES-enumeration contrastive learning and latent diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task evaluation and comprehensive ablation studies, though lacking comparison with a broader range of molecular generation methods.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive diagrams, and coherent logic.
- Value: ⭐⭐⭐⭐ — Inspiring for both molecular generation and text-based diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[ICML 2025\] SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model](space_your_genomic_profile_predictor_is_a_powerful_dna_foundation_model.md)
- [\[ICML 2025\] MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning](mf-lal_drug_compound_generation_using_multi-fidelity_latent_space_active_learnin.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](../../NeurIPS2025/computational_biology/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)

</div>

<!-- RELATED:END -->
