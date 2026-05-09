---
title: >-
  [Paper Note] Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers
description: >-
  [ICLR2026][Image Generation][3D genome] This paper proposes DiffBacChrom — a conditional diffusion Transformer (CrossDiT) that generates ensembles of 3D genome conformations for *E. coli* from Hi-C contact maps. The method employs a ResNet VAE to maintain bin-aligned latent encodings, a Transformer encoder with cross-attention for Hi-C conditioning, and flow-matching training. The generated ensembles exhibit high agreement with input Hi-C data in terms of distance-decay curves $P(s)$ and SCC metrics, while preserving conformational diversity.
tags:
  - ICLR2026
  - Image Generation
  - 3D genome
  - Hi-C
  - diffusion transformer
  - CrossDiT
  - latent diffusion
  - flow matching
  - E. coli
date: 2026-05-08
content_hash: 9b8900fd934d54ff
---

# Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers

**Conference**: ICLR2026  
**arXiv**: [2603.07472](https://arxiv.org/abs/2603.07472)  
**Code**: To be confirmed  
**Area**: Image Generation  
**Keywords**: 3D genome, Hi-C, diffusion transformer, CrossDiT, latent diffusion, flow matching, E. coli

## TL;DR
This paper proposes DiffBacChrom — a conditional diffusion Transformer (CrossDiT) that generates ensembles of 3D genome conformations for *E. coli* from Hi-C contact maps. The method employs a ResNet VAE to maintain bin-aligned latent encodings, a Transformer encoder with cross-attention for Hi-C conditioning, and flow-matching training. The generated ensembles exhibit high agreement with input Hi-C data in terms of distance-decay curves $P(s)$ and SCC metrics, while preserving conformational diversity.

## Background & Motivation

**State of the Field**: Hi-C technology provides population-averaged contact frequencies between genomic loci, yet Hi-C matrices are indirect measurements — a single contact map may correspond to many distinct 3D conformational ensembles.

**Limitations of Prior Work**: Deterministic methods such as FLAMINGO (optimization-based) and CHROMFORMER (Transformer regression) output a single consensus conformation, ignoring the inherent heterogeneity of chromatin organization.

**Root Cause**: Ensemble methods based on polymer simulations (e.g., Li & Schlick 2024) are computationally expensive and difficult to scale to large datasets.

**Paper Goals**: Frame genome reconstruction as a conditional generative problem — given Hi-C, sample a distribution of physically plausible 3D conformations. Once trained, the model enables efficient ensemble generation for new inputs via amortized inference.

**Starting Point**: Bacterial chromosomes have well-established physical and biological constraints suitable for validation; prokaryotes also exhibit diverse cross-species organizational patterns, making them appropriate for evaluating model transferability.

**Core Idea**: Employ the CrossDiT architecture to perform unidirectional conditional generation from Hi-C to 3D structure, where cross-attention ensures Hi-C acts as an "external field" constraining the structure without being updated by it.

## Method

### Overall Architecture
(1) Coarse-grained molecular dynamics simulations generate training data (synthetic Hi-C paired with conformational ensembles); (2) a ResNet VAE encodes 3D coordinate sequences into a latent space while preserving sequence length to maintain bin alignment; (3) a CrossDiT diffusion model generates conformations in latent space conditioned on Hi-C; (4) the VAE decoder reconstructs 3D structures.

### Key Designs

1. **ResNet VAE (per-bin alignment)**:

    - 1D ResNet18 architecture encoding a $928 \times 16$-dimensional input (each Hi-C bin corresponds to 2 beads × 2 chromosomes × (xyz + mask))
    - Sequence length is not compressed — the latent vector length equals the Hi-C matrix dimension, ensuring per-bin alignment
    - A replication mask is introduced to handle branched structures arising from DNA replication; the loss is $\mathcal{L} = \mathcal{L}_{coord} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{KL}\mathcal{L}_{KL}$
    - $\mathcal{L}_{coord}$ computes MSE only at mask-activated positions, avoiding bias from varying bead counts across replication stages

2. **CrossDiT Conditioning**:

    - Hi-C encoder: a Transformer processes the Hi-C matrix along the row dimension (columns serve as features), producing per-bin conditional embeddings $z_c$
    - Global condition: $c = t + \tilde{z}_c$ (timestep embedding + globally average-pooled $z_c$), injected into each DiT block via AdaLN-Zero
    - Cross-attention: $x$ serves as Q, $z_c$ as K/V — the structure queries information from the condition, while the condition is not updated by the structure, yielding a physically interpretable design

3. **Flow-Matching Training**:

    - Adopts the rectified flow framework in place of DDPM for more direct and stable optimization
    - Inference uses 50 sampling steps with CFG scale $= 1.0$ (without amplifying the conditional signal, preserving diversity)
    - Latent space normalization scale $= 1.335$ ensures calibrated noise scheduling

### Loss & Training
VAE: $\mathcal{L} = \mathcal{L}_{coord} + 1.0 \cdot \mathcal{L}_{mask} + 5 \times 10^{-3} \cdot \mathcal{L}_{KL}$; DiT: flow-matching velocity MSE loss.

## Key Experimental Results

### Model Configurations and Generation Quality

| Model | Depth | Hidden Dim | Attention Heads | Parameters | SCC (mean±std) | dRMSD |
|------|------|--------|----------|--------|----------------|-------|
| CrossDiT-S | 12 | 384 | 6 | 45M | 0.824±0.022 | 0.666 |
| CrossDiT-L | 24 | 1024 | 16 | 634M | **0.962±0.008** | **0.700** |
| Gaussian perturbation baseline | — | — | — | — | — | 0.072 |

### Key Findings

| Metric | Description | CrossDiT-L | CrossDiT-S |
|------|------|------------|------------|
| SCC | Hi-C stratum-adjusted correlation coefficient | 0.951–0.975 | 0.787–0.865 |
| $P(s)$ distance decay | Contact frequency vs. genomic distance | Highly consistent | Broadly consistent |
| dRMSD / bond length | Conformational diversity | 1.94× | 1.84× |
| Baseline dRMSD / bond length | Single-conformation perturbation | 0.20× | — |

- CrossDiT-L achieves a mean SCC of 0.962, indicating that the Hi-C reconstructed from generated ensembles closely matches the input across all distance strata.
- The dRMSD of generated ensembles is approximately 1.9× the bond length, far exceeding the single-conformation perturbation baseline (0.2×), demonstrating that the model does not collapse to a single structure.

## Highlights & Insights
- **Reframing the problem**: shifting from "reconstruct a single optimal structure" to "sample a conformational ensemble distribution" more faithfully reflects the nature of Hi-C data as a population-averaged measurement.
- **Physical interpretation of cross-attention**: Hi-C serves as an "external field constraint" → unidirectional conditional injection → the structure queries the condition but does not update it in return; the architectural design aligns with physical intuition.
- **Replication mask for branched structures**: binary masks elegantly unify the handling of chromosomes at different replication stages, enabling the model to learn replication-aware representations.

## Limitations & Future Work
- Training data are entirely derived from coarse-grained MD simulations rather than experimental measurements; the biological fidelity of generated structures depends on simulation quality.
- The current setup assigns 2 beads per Hi-C bin with a sequence length of 928; scaling to eukaryotes (with much longer sequences) requires addressing the computational bottleneck of long-sequence modeling.
- No quantitative comparison is made against sequence-conditioned methods such as ChromoGen on the same organism.
- The VAE architecture is relatively simple (1D ResNet18) and may lack sufficient expressiveness for more complex nested replication branch structures.

## Related Work & Insights
- **vs. FLAMINGO (Wang et al., 2022)**: an optimization method outputting a single conformation; the proposed method outputs a distribution, more faithfully reflecting cell population heterogeneity.
- **vs. CHROMFORMER (Valeyre et al., 2022)**: a Transformer regressing a single structure; the proposed method uses diffusion to generate diverse ensembles.
- **vs. ChromoGen (Schuette et al., 2025)**: ChromoGen conditions on DNA sequence and chromatin accessibility; the proposed method conditions directly on Hi-C to generate explicit 3D coordinates.
- **Insights**: the unidirectional conditioning pattern of CrossDiT is transferable to other "physical constraint → structure generation" scenarios (e.g., density map → protein conformation).

## Rating
- Novelty: ⭐⭐⭐⭐ First application of diffusion Transformers to whole-genome ensemble generation in bacteria; the problem formulation is original.
- Experimental Thoroughness: ⭐⭐⭐ Metric design is sound (SCC + $P(s)$ + dRMSD), but validation is limited to synthetic data with no quantitative comparison against competing methods.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and physical intuitions are well articulated.
- Value: ⭐⭐⭐ Proof-of-concept stage; validation on real experimental data and additional species is needed to establish broader impact.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers](../../AAAI2026/image_generation/relactrl_relevance-guided_efficient_control_for_diffusion_transformers.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[ICLR 2026\] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](../../CVPR2026/image_generation/bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)

<!-- RELATED:END -->
