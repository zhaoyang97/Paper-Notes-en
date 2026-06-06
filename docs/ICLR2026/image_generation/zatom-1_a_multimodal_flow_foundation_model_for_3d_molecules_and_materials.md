---
title: >-
  [Paper Note] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials
description: >-
  [ICLR 2026][Image Generation][Foundation model] Zatom-1 is the first end-to-end fully open-source foundation model that unifies generative modeling and property prediction for 3D molecules and materials via multimodal fl…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Foundation model"
  - "flow matching"
  - "3D molecules"
  - "3D materials"
  - "multimodal generation"
  - "property prediction"
date: 2026-05-08
content_hash: 0d1af8861d89b9bc
---

# Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials

**Conference**: ICLR 2026
**arXiv**: [2602.22251](https://arxiv.org/abs/2602.22251)  
**Code**: Open-source (fully open-source)  
**Area**: Image Generation / Scientific Machine Learning
**Keywords**: Foundation model, flow matching, 3D molecules, 3D materials, multimodal generation, property prediction

## TL;DR
Zatom-1 is the first end-to-end fully open-source foundation model that unifies generative modeling and property prediction for 3D molecules and materials via multimodal flow matching. Using a standard Transformer architecture, it directly models discrete atom types and continuous 3D geometry in Euclidean space, achieving positive transfer learning across chemical domains.

## Background & Motivation

**Background**: AI-driven chemical modeling has achieved major breakthroughs (e.g., AlphaFold), yet existing methods are typically optimized for a single domain (molecules or materials) and a single task (generation or prediction), limiting representation sharing and transfer learning.

**Limitations of Prior Work**: (1) Generative models for molecules and materials are trained separately, failing to leverage complementary cross-domain information. (2) Sparse graph neural network architectures and hand-crafted generative priors limit scalability and inference speed. (3) Generation and prediction tasks rely on separate models, precluding shared representations.

**Key Challenge**: 3D chemical systems involve multiple modalities—discrete (atom types) and continuous (3D coordinates, lattice parameters). How can these modalities be jointly modeled in a unified framework? How can generative pretraining provide effective initialization for downstream prediction tasks?

**Goal**: To build a unified foundation model capable of both generative modeling and representation learning for 3D molecules and materials, enabling positive cross-domain transfer.

**Key Insight**: Generative modeling is treated as an ideal pretraining task for chemical representation learning, employing a standard Transformer architecture with multimodal flow matching directly in ambient all-atom space $\mathbb{R}^3$.

**Core Idea**: Jointly model discrete atom types and continuous 3D geometry within a unified Transformer via multimodal flow matching; the pretrained model is then fine-tuned for multi-task property prediction.

## Method

### Overall Architecture
Zatom-1 adopts a two-stage training pipeline: (1) multimodal flow pretraining for 3D molecule and material generation; and (2) multi-task fine-tuning for energy, force, and property prediction. The core architecture is the Trunk-based Flow Transformer (TFT), consisting of a standard Transformer encoder coupled with a cross-attention decoder.

### Key Designs

1. **Unified Five-Modality Representation**:

    - Function: Represents both molecules and materials under a unified five-modality scheme.
    - Mechanism: Atom types $\bm{A} \in \mathbb{Z}^{1 \times N}$ (discrete), 3D coordinates $\bm{X} \in \mathbb{R}^{3 \times N}$ (continuous), fractional coordinates $\bm{F} \in [0,1)^{3 \times N}$, lattice lengths $\bm{L}_{\text{len}} \in \mathbb{R}^{3 \times 1}$, and lattice angles $\bm{L}_{\text{ang}} \in \mathbb{R}^{3 \times 1}$. Lattice-related inputs are masked for molecules; 3D coordinates are masked for materials.
    - Design Motivation: A unified input format enables a single model to handle both periodic materials and non-periodic molecules simultaneously.

2. **Multimodal Flow Matching Training**:

    - Function: Jointly trains generation across both discrete and continuous modalities.
    - Mechanism: Continuous modalities use Euclidean CFM: $\bm{X}_t = t \cdot \bm{X} + (1-t) \cdot \epsilon$, $L_{\text{metric}}(\bm{X}) = \mathbb{E}_{\epsilon,t}\left[\frac{1}{N}\|\bm{X}' - \bm{X}\|_2^2\right]$; discrete modalities use Discrete CFM: $\bm{A}_t \sim \text{Cat}(t \cdot \delta(\bm{A}) + (1-t) \cdot \delta(\frac{1}{\#\text{atom types}}))$, $L_{\text{discrete}}(\bm{A}) = \mathbb{E}_t\left[-\sum_i a_i \log(a_i')\right]$.
    - Design Motivation: The endpoint formulation performs better in scientific applications and eliminates the need for a pretrained autoencoder (as in latent diffusion), substantially reducing training and inference cost.

3. **Trunk-based Flow Transformer (TFT)**:

    - Function: A unified encoder–decoder architecture.
    - Mechanism: An $L$-layer Transformer encoder extracts shared representations $\bm{Z}$, followed by residual cross-attention decoders that separately predict denoised outputs for each modality. Representations from encoder layer $K$ are used for downstream prediction tasks (properties, energy, forces), while layer $L$ representations are used for generation.
    - Design Motivation: A standard Transformer architecture (QK Normalization, Flash Attention, SwiGLU FFN) yields predictable performance scaling with parameter count.

4. **Sampling Strategy (SDE Sampling with Controllable Stochasticity)**:

    - Function: Introduces controllable stochasticity on top of ODE-based sampling.
    - Mechanism: SDE sampling is applied to continuous modalities: $\mathbf{z}_t \leftarrow (\mathbf{v}_t + \mathbf{s}_t + d\mathbf{W}_t)\Delta t$, where $d\mathbf{W}_t \leftarrow \sqrt{2\gamma_g g(t)}\mathcal{N}(0,I)$; categorical sampling is used for discrete modalities.
    - Design Motivation: Stochasticity improves generation diversity and sample quality.

### Loss & Training
The total loss is $L_{\text{total}} = L_{\text{metric}}(\bm{X}) + L_{\text{metric}}(\bm{F}) + L_{\text{metric}}(\bm{L}_{\text{len}}) + L_{\text{metric}}(\bm{L}_{\text{ang}}) + \lambda_{\text{discrete}} \cdot L_{\text{discrete}}(\bm{A})$, with $\lambda_{\text{discrete}} = 0.1$. Time is sampled as $t \sim \text{Beta}(1.8, 1)$, with loss scaling $\beta(t) = \min\{100, \frac{1}{(1-t)^2}\}$. Random rotations and translations are applied to input data for augmentation during training. The model contains 300M parameters and generates 10,000 samples in under 4 minutes on a single A100 GPU.

## Key Experimental Results

### Main Results (Material Generation — MP20 Dataset)

| Method | Match Rate ↑ | RMSD ↓ | Inference Time |
|--------|-------------|--------|----------------|
| DiffCSP | — | — | Slow |
| FlowMM (500M) | Baseline | Baseline | ~50 min / 10K |
| **Zatom-1 (300M)** | **Competitive** | **Competitive** | **<4 min / 10K** |

### Molecule Generation (QM9 + GEOM-Drugs)

| Method | QM9 Stability ↑ | GEOM-Drugs Validity ↑ |
|--------|-----------------|----------------------|
| EDM | Baseline | Baseline |
| **Zatom-1** | **SOTA** | **SOTA** |

### Property Prediction (QM9 Multi-task)

Zatom-1 achieves state-of-the-art performance on QM9 multi-task property prediction, demonstrating that generative pretraining provides effective representation initialization for prediction tasks.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Molecule-only pretraining → molecular property prediction | Baseline |
| Joint molecule + material pretraining → molecular property prediction | **Better** (positive cross-domain transfer) |
| No pretraining → property prediction | Substantially worse |

### Key Findings
- **Positive Cross-Domain Transfer**: Incorporating material data during pretraining improves molecular property prediction accuracy—the first observation of positive transfer in a chemical foundation model.
- **12.5× Inference Speedup**: Compared to the 500M-parameter latent diffusion baseline (FlowMM), Zatom-1 (300M) achieves a 12.5× acceleration.
- **3× Training Savings**: By eliminating the need for a pretrained autoencoder, GPU training hours are reduced by 3× relative to latent diffusion approaches.
- **Predictable Scaling**: Both generative and predictive performance improve predictably as model size scales from 50M to 300M parameters.

## Highlights & Insights
- **Paradigm Unification**: The first model to unify molecule/material generation and prediction within a single framework, demonstrating the feasibility of "generation as pretraining" in chemistry. This paradigm is extensible to proteins, crystals, and other scientific domains.
- **Ambient-Space Design**: Direct modeling in ambient space avoids the autoencoder required by latent diffusion, substantially simplifying the pipeline and accelerating inference. This stands in interesting contrast to the "latent over pixel" trend in computer vision.
- **Standardized Architecture**: The use of standard Transformers rather than domain-specific equivariant GNNs suggests that large-scale data augmentation combined with standard architectures can substitute for hand-crafted equivariance.
- **O(3)-Equivariant Variant Platom-1**: An equivariant Platonic Transformer variant is also explored, demonstrating architectural flexibility.

## Limitations & Future Work
- The current model supports systems of at most ~200 atoms (limited by token length), precluding large biomolecules such as proteins.
- Material generation is primarily evaluated on the MP20 dataset; generalization to extreme conditions (e.g., high temperature and pressure) remains unknown.
- The effect of the pretraining data mixture ratio (molecules vs. materials) on downstream performance warrants further investigation.
- Integration with large language models to enable text-conditioned molecular design is a promising direction for future work.

## Related Work & Insights
- **vs. FlowMM**: FlowMM also applies flow matching to material generation but uses a sparse GNN and does not support property prediction. Zatom-1 achieves a 13× inference speedup through its standard Transformer and unified framework.
- **vs. EDM/GeoLDM**: These molecular generation methods do not support materials and lack predictive capability. Zatom-1 surpasses them on QM9.
- **vs. AlphaFold3**: AlphaFold3 inspires the idea of "learning scientific data with standard Transformers"; Zatom-1 extends this paradigm to chemical generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First foundation model to unify molecule/material generation and prediction; positive cross-domain transfer is a significant finding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-task evaluation with scaling experiments and ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly presented, though some implementation details are deferred to the appendix.
- Value: ⭐⭐⭐⭐⭐ Substantially advances foundation model research for scientific AI; open-source contribution is highly impactful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoD: A Diffusion Foundation Model for Image Compression](../../CVPR2026/image_generation/cod_a_diffusion_foundation_model_for_image_compression.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](../../ICML2026/image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
