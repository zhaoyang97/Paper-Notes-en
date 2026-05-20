---
title: >-
  [Paper Note] Skipping the Zeros in Diffusion Models for Sparse Data Generation
description: >-
  [ICML 2026][Scientific Computing][Sparse Diffusion] SED transforms diffusion models from "full dense denoising across all dimensions" to "diffusion only on nonzero dimensions + autoregressive decoding of dimension-value…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Sparse Diffusion"
  - "Latent Diffusion"
  - "Autoregressive Decoding"
  - "Single-cell Sequencing"
  - "Calorimeter Images"
date: 2026-05-08
content_hash: cbecd1b1f9984191
---

# Skipping the Zeros in Diffusion Models for Sparse Data Generation

**Conference**: ICML 2026  
**arXiv**: [2605.01817](https://arxiv.org/abs/2605.01817)  
**Code**: <https://github.com/PhilSid/sparsity-exploiting-diffusion>  
**Area**: Diffusion Models / Sparse Data Generation / Scientific Computing / Generative Modeling  
**Keywords**: Sparse Diffusion, Latent Diffusion, Autoregressive Decoding, Single-cell Sequencing, Calorimeter Images

## TL;DR
SED transforms diffusion models from "full dense denoising across all dimensions" to "diffusion only on nonzero dimensions + autoregressive decoding of dimension-value pairs," reducing computation from linear in total dimensions to nearly constant in the number of nonzeros, while strictly preserving the semantic meaning of explicit zeros in scientific data.

## Background & Motivation

**Background**: Diffusion models (DM) have achieved SOTA on dense, continuous data such as images, audio, and text, with DDPM/LDM as de facto standards. However, many scientific datasets are inherently sparse: particle physics calorimeters (~95% zeros), single-cell RNA sequencing (scRNA, 90-98% zeros), recommender systems, sparse images, etc.—most coordinates are truly **zero** (no signal), not "close to zero."

**Limitations of Prior Work**: (1) **Sparsity is lost**—feeding sparse data to DDPM/LDM produces spurious nonzero values at all zero coordinates, destroying the sparse pattern (see Figure 1, MNIST demo). This is disastrous for zeros with clear physical meaning, such as biological dropout or "no energy deposition" in physics. (2) **Wasted computation**—rate-distortion analysis (Figure 3) shows diffusion models allocate nearly zero bitrate to zero dimensions, yet the denoising network still runs forward passes on all dimensions—"information capacity is concentrated in informative dimensions, but computation is not." (3) Existing patchwork solutions are flawed: thresholding outputs (DDPM-T/LDM-T) preserves sparsity but sacrifices detail; domain-specific models (SARM) rely on hand-crafted priors for zero locations, with poor generalization; discrete DMs cannot handle continuous values; recent Ostheimer 2025's sparse-aware DM doubles the number of dimensions by adding a binary indicator per dimension.

**Key Challenge**: Information (rate) is sparse, but the **computation and parameterization** of dense DMs is dense, leading to a fundamental mismatch. One must either retain dense architectures and lose sparse semantics, or abandon scalable Transformers for hand-coded sparse priors—no solution achieves both.

**Goal**: (1) Enable DMs to perform diffusion on compact representations that **only process nonzero dimensions**, making computation scale with **signal density** rather than ambient dimension; (2) **Strictly preserve zero patterns**, ensuring output zeros match real data; (3) Avoid hand-crafted priors, enabling use across multiple scientific domains (physics/biology/vision).

**Key Insight**: Each sparse sample is represented as a pair of "(set of dimension indices, set of corresponding nonzero values)"; a Transformer encoder pools this variable-length set into a **fixed-length** dense latent variable $\mathbf{z}$; diffusion operates in this dense latent space (mature and stable), and decoding is autoregressive, generating "next dimension-value pair" until [EOS].

**Core Idea**: Break the implicit assumption that "diffusion should span all dimensions"—diffusion remains dense and stable in **latent space**, but **input space representation** and **decoding** skip zeros, aligning computation with signal.

## Method

### Overall Architecture
Two-stage LDM-style training: (1) **SAVAE (Sparsity-Aware VAE)**—the nonzero extractor NZE converts $\mathbf{x}^{(i)} \in \mathbb{R}^s$ into $(\mathbf{d}^{(i)}, \mathbf{v}^{(i)})$ (length $l_i \ll s$); Transformer encoder $q_\phi$ processes the variable-length "dimension-value" token sequence, outputs a fixed-length $\mathbf{z}$ via mean pooling; the autoregressive decoder $p_\theta = p_{\theta_1}(\mathbf{d}) p_{\theta_2}(\mathbf{v})$ sequentially predicts the next dimension (multinomial) and its value (Gaussian). (2) **Latent diffusion**—after SAVAE is trained and frozen, standard DM (DDPM/DDIM) is trained in $\mathbf{z}$ space, denoted SEDP/SEDI. At generation, sample from $\mathcal{N}(0,I)$, denoise to obtain $\mathbf{z}_0$, then use $p_\theta$ to autoregressively decode the dimension-value sequence and fill back the sparse vector.

### Key Designs

1. **Sparse-to-Dense Latent Encoding (SAVAE)**:

    - **Function**: Compress high-dimensional sparse data into a fixed-size dense latent representation, enabling stable diffusion model training in low-dimensional dense space.
    - **Mechanism**: Use **NZE** to extract the set of nonzero indices $\mathbf{d}^{(i)} = \{j | \mathbf{x}^{(i)}_j \neq 0\}$ and corresponding values $\mathbf{v}^{(i)}$, with length $l_i = \|\mathbf{x}^{(i)}\|_0 \ll s$. Introduce **Dimension Encoding (DE)**—formally similar to positional encoding $\text{DE}_{(dim, 2i)} = \sin(dim / k^{2i/d_{model}})$ ($k=20000$), but encoding **dimension indices** instead of sequence positions. Values are embedded via linear projection; DE and value embeddings are summed as input to the Transformer encoder. The encoder output is mean pooled to obtain fixed-length $\mathbf{z}$ (the authors also tried adding a [CLS] token, but mean pooling was more stable). Reparameterization sampling $\mathbf{z} \sim q_\phi$ is used.
    - **Design Motivation**: Transformer input sequence length scales with **number of nonzeros $l_i$** rather than ambient dimension $s$, so computation is linear in sparsity—not dimension—underpinning SED's computational efficiency. The resulting $\mathbf{z}$ is a dense, low-dimensional vector, compatible with standard DDPM/DDIM backends.

2. **Autoregressive Sparse Decoding (dimension-value pairs)**:

    - **Function**: Decode latent variable $\mathbf{z}$ back to sparse space, determining **which** dimensions are nonzero (variable length) and their **values**.
    - **Mechanism**: Decoder $p_\theta(\mathbf{d}, \mathbf{v} | \mathbf{z})$ is split into two heads: $p_{\theta_1}$ outputs a multinomial over remaining dimensions to predict the next nonzero index, $p_{\theta_2}$ outputs a Gaussian for the corresponding value. Both heads are **jointly trained**, decoding in canonical ascending order of indices, stopping at [EOS]. Importantly, **teacher forcing with parallel evaluation** is used during training, so efficiency is not bottlenecked by autoregression; only sampling is sequential.
    - **Design Motivation**: The number of nonzeros $l_i$ varies per sample (e.g., a cell may have few or many active genes), so fixed-length decoding is infeasible; autoregression is structurally necessary. Canonical ascending order removes permutation ambiguity.

3. **Sparse-Aware Latent Diffusion SED + Self-Conditioned Training**:

    - **Function**: Perform diffusion in the dense, low-dimensional latent space provided by SAVAE, with loss focused only on informative dimensions.
    - **Mechanism**: After freezing SAVAE, train diffusion with $\mathcal{L}_{\text{SED}}(\theta) = \mathbb{E}\|\mathbf{z}_0 - f_\theta(\mathbf{z}_t, t, \tilde{\mathbf{z}}_0)\|^2$, where $\mathbf{z}_t = \sqrt{\gamma(t)}\mathbf{z}_0 + \sqrt{1-\gamma(t)}\boldsymbol{\epsilon}$, and $\tilde{\mathbf{z}}_0$ is the previous estimate for self-conditioning (Chen 2023). The backbone is an MLP-based time-conditioned U-Net (no convolutions, as $\mathbf{z}$ lacks grid structure). Both DDPM and DDIM samplers are supported at generation, yielding SEDP/SEDI.
    - **Design Motivation**: Diffusion itself remains mature and stable, with all "sparse customization" isolated in SAVAE; this modular decoupling allows independent replacement or upgrading of components. Diffusion in dense $\mathbf{z}$ is much more stable than in high-dimensional sparse space.

### Loss & Training

SAVAE uses a $\beta$-VAE objective: $\mathcal{L}_{\text{SAVAE}} = -\log p_\theta(\mathbf{d}, \mathbf{v}|\mathbf{z}) + \beta \cdot D_{\text{KL}}(q_\phi \| p)$, with $\beta = 10^{-6}$ as light regularization; the negative log-likelihood decomposes into dimension (multinomial) and value (Gaussian) parts. Two-stage training: train SAVAE to convergence, then freeze and train diffusion. $\gamma(t)$ decreases monotonically from 1 to 0, parameterized by log-SNR.

## Key Experimental Results

### Main Results

Across three domains and six datasets: **Physics**—muon signal/background calorimeter images ($32 \times 32$, ~95% zeros); **Biology**—Tabula Muris (98% zeros) and Human Lung PF (96% zeros) scRNA; **Vision**—MNIST (81% zeros), Fashion-MNIST (50% zeros). Metrics: physics uses Wasserstein distance $W_P$ for $P_T$ and invariant mass; scRNA uses SCC and MMD; vision uses sparsity histogram matching.

| Task | Model (Params) | Metric | Value | Note |
|------|---------------|--------|-------|------|
| Muon Signal | DDPM (37M) | $W_P (P_T)$↓ | 220.32 | dense fails completely |
| Muon Signal | DDPM-T (37M) | $W_P (P_T)$↓ | 24.22 | thresholding mitigates |
| Muon Signal | SARM (25M, domain) | $W_P (P_T)$↓ | 28.01 | uses spiral prior |
| Muon Signal | **SEDP (15M)** | $W_P (P_T)$↓ | **16.31** | fewest params, best result |
| Tabula Muris | DDPM (5M) | SCC↑ / MMD↓ | 0.50 / 3.60 | dense fails |
| Tabula Muris | scDiffusion (5M, domain) | SCC↑ / MMD↓ | 0.71 / 1.53 | needs cell corpus pretraining |
| Tabula Muris | **SEDP (4M)** | SCC↑ / MMD↓ | **0.74 / 0.55** | no domain pretraining needed |
| Human Lung PF | **SEDP (4M)** | SCC↑ / MMD↓ | **0.82 / 0.54** | outperforms scDiffusion |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| SED full (SAVAE + latent diffusion + AR decoding) | Best | — |
| LDM (no sparse awareness) | LDM SCC=0.87 but MMD=5.82 | correct shape but large distribution distance |
| LDM-T (thresholded) | SCC drops to 0.26 | thresholding destroys LDM detail |
| DDPM/DDIM original | Complete failure | neither preserves sparsity nor achieves low distance |
| SARM (physics domain prior) | Weaker than SED | hand-crafted spiral prior lacks generality |
| Sampling time (95% sparsity) | SED 24ms vs DDPM 453ms | 19× speedup at high sparsity |
| Dimension order accuracy (Fashion-MNIST) | 100% | long sequences but simple, no errors |
| Dimension order accuracy (Muon BG) | 87.9% | highest error rate in complex structure |

### Key Findings

- **Computation nearly constant with dimension**: For scRNA data with 1000 active genes, adding extra zero genes up to 27k dimensions, DDPM/LDM scale linearly, SED remains nearly flat (Figure 2/9).
- **Higher sparsity yields greater speedup**: Muon (95%) achieves nearly 20× speedup, MNIST (81%) 7×, Fashion-MNIST (50%) almost no speedup—SED's advantage increases strictly with sparsity.
- **On scRNA tasks, SED outperforms dense baselines, thresholded variants, and domain-specific scDiffusion**—the latter also requires expensive cell corpus pretraining.
- Autoregressive misordering does **not** systematically worsen for long sequences (Fashion-MNIST, 100% correct); highest error rates are for complex structures (Muon BG 87.9%)—indicating difficulty is due to data complexity, not sequence length.

## Highlights & Insights

- **"Only informative dimensions need computation" is a principle overlooked in dense DM era**: The authors use rate-distortion analysis to show DDPM allocates nearly zero bitrate to zero dimensions yet spends full computation on them—a compelling diagnostic.
- **Dimension Encoding repurposes positional encoding**: Swapping "sequence position" for "feature index" is an elegant engineering adaptation, enabling Transformers to directly process sparse (index, value) pairs.
- **Two-stage decoupling**: SAVAE addresses "how to represent sparsity," diffusion handles "how to generate dense latent variables"—clean separation, each module independently replaceable or upgradable.
- This approach is transferable to graph generation (sparse edges), 3D point clouds (spatial sparsity), KV cache compression in sparse attention, sparse activation MoE, etc.

## Limitations & Future Work

- **Relies on autoregressive decoding**—sampling must be sequential, so very long nonzero sequences still incur latency; the authors explicitly call for non-autoregressive alternatives.
- **Dimension ordering errors** can produce unrealistic samples (Figure 7, MNIST demo), especially in complex sparse patterns (e.g., particle physics, 12% samples affected); but authors verify such errors do not affect overall generation quality in physics experiments.
- **Advantage disappears at low sparsity**: On Fashion-MNIST (50% zeros), SED's sampling time is nearly the same as DDPM, and LDM is actually fastest—SED is specialized for high-sparsity.
- On scRNA, SED is slightly weaker than some LDM configurations in SCC (but better in MMD)—indicating a subtle trade-off between preserving sparsity and matching overall distribution.
- Lacks finer comparison with sparse Transformer works (e.g., XTrimoGene).

## Related Work & Insights

- **vs DDPM/LDM (dense baselines)**: They denoise across all dimensions; SED only processes nonzeros, achieving linear speedup with sparsity and strictly preserving zero patterns.
- **vs DDPM-T / LDM-T (post-hoc thresholding)**: Thresholding is a hack—preserves sparsity but destroys boundary details; SED is a structural solution.
- **vs SARM (Lu 2021, domain-specific)**: SARM hard-codes physics zero location priors via spiral sampling, with poor generalization; SED requires no domain prior and performs better.
- **vs Discrete DM (Austin 2021)**: Discrete DMs can generate zeros exactly but only in discrete state spaces, unable to handle continuous sparse values; SED models both "where signal exists + signal value."
- **vs scDiffusion (Luo 2024)**: scDiffusion requires large-scale cell corpus pretraining for its autoencoder; SED is end-to-end and outperforms it without domain pretraining.
- **vs Sparse Transformer (XTrimoGene, scGPT)**: Those are for representation learning, using MSE loss only on masked genes; SED is generative, needing to predict dimension indices from scratch.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The implicit assumption that "diffusion should run across sparse dimensions" is broken; the autoregressive dimension-value decoding perspective is truly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three distinct sparse domains (physics/biology/vision), with comprehensive comparison across dense/thresholded/domain-specific baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic chain from rate-distortion diagnosis → method motivation → experimental validation is very clear, with intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Provides an immediately usable, computation- and fidelity-efficient solution for scientific computing (high sparsity); open-source code is available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](../../CVPR2026/image_generation/memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[CVPR 2026\] Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](../../CVPR2026/image_generation/interpretable_and_steerable_concept_bottleneck_sparse_autoencoders.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](../../ICCV2025/image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
