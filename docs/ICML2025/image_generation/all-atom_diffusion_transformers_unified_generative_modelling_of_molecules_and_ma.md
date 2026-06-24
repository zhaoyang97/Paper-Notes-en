---
title: >-
  [Paper Note] All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials
description: >-
  [ICML2025][Image Generation][latent diffusion] This work proposes the All-atom Diffusion Transformer (ADiT), a two-stage framework that maps molecules and crystals into a unified latent space via a VAE, and then utilizes a Diffusion Transformer to generate new samples within this latent space. It is the first to achieve simultaneous generation of periodic materials (crystals) and non-periodic molecular systems using a **single model**. ADiT achieves SOTA performance on MP20…
tags:
  - "ICML2025"
  - "Image Generation"
  - "latent diffusion"
  - "diffusion transformer"
  - "molecular generation"
  - "crystal generation"
  - "unified model"
  - "scaling law"
date: 2026-05-08
content_hash: 5f33b9f276259a82
---

# All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials

**Conference**: ICML2025  
**arXiv**: [2503.03965](https://arxiv.org/abs/2503.03965)  
**Authors**: Chaitanya K. Joshi, Xiang Fu, Yi-Lun Liao, Vahe Gharakhanyan, Benjamin Kurt Miller, Anuroop Sriram, Zachary W. Ulissi (Meta FAIR, Cambridge, MIT)  
**Code**: [facebookresearch/all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer)  
**Area**: Image Generation  
**Keywords**: latent diffusion, diffusion transformer, molecular generation, crystal generation, unified model, scaling law

## TL;DR

This work proposes the All-atom Diffusion Transformer (ADiT), a two-stage framework that maps molecules and crystals into a unified latent space via a VAE, and then utilizes a Diffusion Transformer to generate new samples within this latent space. It is the first to achieve simultaneous generation of periodic materials (crystals) and non-periodic molecular systems using a **single model**. ADiT achieves SOTA performance on MP20, QM9, and GEOM-DRUGS, while being an order of magnitude faster than equivariant diffusion models.

## Background & Motivation

Generative modeling of 3D atomic systems is a core problem in molecular design and materials discovery. Current diffusion models are highly specialized for different types of atomic systems:

- **Small molecule generation** (e.g., EDM): Establishes separate diffusion processes for atomic types (discrete) and 3D coordinates (continuous), leading to unrealistic intermediate states.
- **Biomolecule generation** (e.g., FrameDiff): Treats atomic groups as rigid bodies, introducing additional rotational manifolds.
- **Crystal/Material generation** (e.g., DiffCSP, FlowMM): Requires handling periodicity, diffusing over a joint manifold of atomic types, fractional coordinates, and lattice parameters.

While these methods share the same underlying physics, their model designs are completely different, preventing knowledge sharing. **Core Problem**: Can a unified diffusion model be built to simultaneously generate periodic materials and non-periodic molecules?

Key bottlenecks of existing methods:

**Low efficiency of equivariant networks**: Equivariant networks such as EGNN and GVP act as denoisers but suffer from high computational overhead, making them difficult to scale.

**Complex multi-manifold diffusion**: Heterogeneous data (such as categories, coordinates, and lattice angles) require separate diffusion processes defined on a product manifold.

**Domain isolation**: Molecular and material models are trained separately, preventing cross-domain transfer learning.

## Method

### Overall Architecture

ADiT adopts a two-stage **Latent Diffusion** architecture:

1. **Stage 1 — VAE Autoencoder**: Learns to map the all-atom representation of molecules and crystals into a shared latent space.
2. **Stage 2 — Diffusion Transformer**: Trains a DiT in the latent space to generate new latent vectors, which are then decoded by the VAE into valid molecules or crystals.

The core idea is to use an autoencoder to absorb the complexity of heterogeneous data types (categorical + continuous), making the generation process in the latent space unified and scalable.

### Key Designs

#### Key Design 1: Unified All-atom Representation

All atomic systems (molecules and crystals) are unified as a collection of atoms in 3D space:

- **Atomic types** $\mathbf{A} = \{a_i\}_{i=1}^N \in \mathbb{Z}^{1 \times N}$: Discrete categorical attributes.
- **3D coordinates** $\mathbf{X} = \{x_i\}_{i=1}^N \in \mathbb{R}^{3 \times N}$: Continuous spatial attributes.
- **Lattice matrix** $\mathbf{L} \in \mathbb{R}^{3 \times 3}$ (crystals only): Defines the periodic unit cell.

For molecules, $\mathbf{L}$ is set to a zero matrix or a special token, thereby processing both periodic and non-periodic systems under the same representation framework.

#### Key Design 2: VAE Autoencoder

Both the VAE encoder and decoder utilize standard Transformers (non-equivariant networks). The key designs include:

- **Encoder**: Encodes the all-atom representation (atom type one-hot + coordinates + lattice information) into a set of fixed-dimension latent vectors $\mathbf{Z} \in \mathbb{R}^{M \times d}$.
- **Decoder**: Reconstructs atomic types (classification loss), 3D coordinates (regression loss), and lattice parameters from the latent vector $\mathbf{Z}$.
- **Data Augmentation**: Applies random rotation and translation to molecules, enabling the model to implicitly learn SE(3) invariance and avoiding the computational overhead associated with explicit equivariant constraints.
- **Training objective**: Reconstruction loss + KL divergence regularization, jointly training on both molecular and crystal data.

Abandoning equivariant networks is a key decision of ADiT—replacing architectural constraints with data augmentation greatly simplifies the model and improves training/inference speed.

#### Key Design 3: Diffusion Transformer (DiT)

A standard Diffusion Transformer is trained in the VAE latent space:

- **Forward process**: Adds Gaussian noise to the latent vector $\mathbf{Z}$.
- **Denoising network**: Standard Transformer + AdaLN (Adaptive Layer Normalization) conditioned on timesteps.
- **Classifier-free guidance (CFG)**: Uses the system type label (molecule vs. crystal) as a condition, randomly discarding the labels during training for classifier-free guidance.
- **Inference**: Samples latent vectors -> VAE decoding -> post-processing to obtain valid molecules/crystals.

Operating in the latent space, DiT avoids the complexity of direct diffusion on the product manifold of atomic types and coordinates.

### Loss & Training

- **Joint training**: Jointly trains the same model on mixed QM9 molecular data and MP20 material data.
- **Two-stage training**: Trains the VAE to convergence first, then freezes the VAE to train the DiT.
- **Data augmentation**: Substitutes equivariant architectural constraints with random SE(3) transformations.
- **Model scaling**: Scales from 30M to 500M parameters, observing a predictable scaling law.

## Key Experimental Results

### Datasets

| Dataset | Type | Samples | Atom Count Range | Characteristics |
|---|---|---|---|---|
| QM9 | Small molecules | ~130K | ≤9 heavy atoms | Standard molecule generation benchmark |
| MP20 | Crystal materials | ~45K | ≤20 atoms/unit cell | Materials Project subset |
| GEOM-DRUGS | Drug molecules | ~300K conformations | Hundreds of atoms | Large molecule 3D conformation generation |

### Table 1: MP20 Crystal Generation Results

| Method | Type | Match Rate (%) | RMSD | S.U.N. Rate (%) |
|---|---|---|---|---|
| CDVAE | Equivariant Diffusion | 45.4 | 0.356 | 3.5 |
| DiffCSP | Equivariant Diffusion | 51.1 | 0.252 | 4.0 |
| FlowMM | Equivariant Flow Matching | 65.3 | 0.195 | 4.8 |
| **ADiT (MP20-only)** | Latent Diffusion | 62.8 | 0.201 | 5.2 |
| **ADiT (Joint)** | Latent Diffusion | **67.1** | **0.188** | **6.0** |

- The S.U.N. (Stable, Unique, Novel) rate of the jointly trained ADiT reaches **6.0%**, representing a **25%** gain over the best baseline FlowMM.
- Joint training outperforms single-dataset training, demonstrating the efficacy of molecule-material transfer learning.

### Table 2: QM9 Molecule Generation Results

| Method | Type | Atom Stability (%) | Mol Stability (%) | Validity (%) | Uniqueness (%) |
|---|---|---|---|---|---|
| EDM | Equivariant Diffusion | 98.7 | 82.0 | 91.9 | 90.7 |
| GeoLDM | Latent Diffusion | 98.9 | 89.4 | 93.8 | 92.7 |
| EQGAT-diff | Equivariant Diffusion | 98.2 | 71.6 | 86.7 | 96.1 |
| **ADiT (QM9-only)** | Latent Diffusion | 99.0 | 90.1 | 95.2 | 91.8 |
| **ADiT (Joint)** | Latent Diffusion | **99.2** | **91.3** | **96.1** | 92.4 |

- ADiT surpasses specialized models in both atom stability and molecule stability.
- Joint training again outperforms single-dataset training, with Validity improving from 95.2% to **96.1%**.

### Efficiency Comparison

| Method | 10K Sample Inference Time | Speedup |
|---|---|---|
| Equivariant diffusion baseline | ~2.5 hours (V100) | 1x |
| **ADiT** | **< 20 minutes (V100)** | **~7-8x** |

The inference speed of ADiT is approximately **7-8x** faster than equivariant diffusion models, owing to standard Transformers avoiding the computational overhead of equivariant calculations.

### Scaling Law

When scaling ADiT from ~30M to ~500M parameters under a constant data size:
- Generation quality (across various metrics) scales predictably with the number of model parameters.
- This indicates that further scaling will continue to yield benefits, pointing towards foundation models for molecular and material generation.

## Highlights & Insights

- **First unified model**: The first diffusion model capable of simultaneously generating periodic crystals and non-periodic molecules, breaking domain isolation.
- **Efficacy of transfer learning**: Joint training outperforms individual training in both molecular and material domains, proving the feasibility of cross-system knowledge transfer.
- **Paradigm shift from equivariance to data augmentation**: Standard Transformers paired with data augmentation replace equivariant networks to significantly boost efficiency without performance loss, offering key insights for the scientific machine learning community.
- **Scalability**: The DiT architecture inherently supports parameter scaling, showing a clear scaling law that lays the foundation for chemical generation foundation models.
- **Practical efficiency**: Generating 10K samples takes under 20 minutes on a single V100, which is an order of magnitude faster than equivariant baselines, demonstrating high practical utility.

## Limitations & Future Work

- **VAE information bottleneck**: In the two-stage approach, the reconstruction quality of the VAE determines the upper bound of generation. Latent space compression might lose fine-grained structural information.
- **Data augmentation vs. equivariance**: Symmetries implicitly learned through data augmentation are less strict than explicit equivariant constraints, which may be insufficient in data-sparse scenarios.
- **Validation on small-scale systems only**: QM9 (≤9 heavy atoms) and MP20 (≤20 atoms) are limited in scale. The generalization to large-scale systems such as proteins remains unverified.
- **High cost of DFT evaluation**: S.U.N. metrics require DFT calculations to verify stability, which is computationally expensive for large-scale evaluations.
- **Unexplored conditional generation**: Only unconditional generation is demonstrated. Property-conditioned generation for practical applications (e.g., target bandgap, solubility) has not yet been verified.

## Related Work & Insights

- **Molecular diffusion models**: EDM (Hoogeboom et al., 2022) diffuses over the joint manifold of atomic types + coordinates; GeoLDM (Xu et al., 2023) introduces molecular latent diffusion; EQGAT-diff (Le et al., 2024) utilizes equivariant graph attention. ADiT avoids complex multi-manifold diffusion through a unified latent space.
- **Crystal generation**: CDVAE (Xie et al., 2022) pioneered crystal VAE + diffusion; DiffCSP (Jiao et al., 2023) and FlowMM (Miller et al., 2024) improved generation over crystal manifolds. ADiT achieves unified molecule-crystal generation for the first time.
- **Diffusion Transformer**: DiT (Peebles & Xie, 2023) replaces U-Net with a Transformer in image generation; Latent Diffusion (Rombach et al., 2022) introduces the two-stage paradigm. ADiT successfully transfers these concepts to 3D atomic systems.
- **Transformers in scientific machine learning (SciML)**: Recent works (e.g., Liao & Smidt, 2023; Duval et al., 2023) have begun to explore the potential of standard Transformers in atomic system modeling. ADiT validates the feasibility of this direction for generative tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to unify molecule and crystal generation. Although the latent diffusion concept is borrowed from the image domain, transferring it to atomic systems represents a methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets + DFT validation + scaling law analysis + efficiency comparison, presenting a comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous method description, and high-quality figures/tables; a solid output from Meta FAIR.
- Value: ⭐⭐⭐⭐ — Points out the direction for chemical generation foundation models; the unified framework and scalability analysis possess long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: All Current Generative Fidelity and Diversity Metrics are Flawed](position_all_current_generative_fidelity_and_diversity_metrics_are_flawed.md)
- [\[ICLR 2026\] Generalised Flow Maps for Few-Step Generative Modelling on Riemannian Manifolds](../../ICLR2026/image_generation/generalised_flow_maps_for_few-step_generative_modelling_on_riemannian_manifolds.md)
- [\[ICLR 2026\] Scaling Laws for Diffusion Transformers](../../ICLR2026/image_generation/scaling_laws_for_diffusion_transformers.md)
- [\[ICLR 2026\] Diffusion Transformers with Representation Autoencoders](../../ICLR2026/image_generation/diffusion_transformers_with_representation_autoencoders.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](../../NeurIPS2025/image_generation/scaling_diffusion_transformers_efficiently_via_μp.md)

</div>

<!-- RELATED:END -->
