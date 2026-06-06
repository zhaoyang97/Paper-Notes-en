---
title: >-
  [Paper Note] Skipping the Zeros in Diffusion Models for Sparse Data Generation
description: >-
  [ICML 2026][Image Generation][Sparse Diffusion] SED transforms diffusion models from "full dense denoising across all dimensions" to "diffusion only on non-zero dimensions + autoregressive decoding of dimension-value pai…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Sparse Diffusion"
  - "Latent Diffusion"
  - "Autoregressive Decoding"
  - "Single-cell Sequencing"
  - "Calorimeter Images"
date: 2026-05-08
content_hash: 52487140799391ed
---

# Skipping the Zeros in Diffusion Models for Sparse Data Generation

**Conference**: ICML 2026  
**arXiv**: [2605.01817](https://arxiv.org/abs/2605.01817)  
**Code**: <https://github.com/PhilSid/sparsity-exploiting-diffusion>  
**Area**: Diffusion Models / Sparse Data Generation / Scientific Computing / Generative Modeling  
**Keywords**: Sparse Diffusion, Latent Diffusion, Autoregressive Decoding, Single-cell Sequencing, Calorimeter Images

## TL;DR
SED transforms diffusion models from "full dense denoising across all dimensions" to "diffusion only on non-zero dimensions + autoregressive decoding of dimension-value pairs," making computational complexity almost constant relative to the number of non-zeros instead of growing linearly with dimension, while strictly preserving the semantic information of "explicit zeros" in scientific data.

## Background & Motivation

**Background**: Diffusion Models (DM) have achieved SOTA on dense continuous data such as images, audio, and text, with DDPM/LDM serving as the fact standards. However, many scientific datasets are inherently sparse: particle physics calorimeters (~95% zeros), single-cell RNA sequencing (scRNA) (90-98% zeros), recommender systems, and sparse images—where most coordinates are **explicit zeros** (absence of signal) rather than just "values close to zero."

**Limitations of Prior Work**: (1) **Sparsity is flattened**—feeding sparse data into DDPM/LDM causes the output to produce spurious non-zero values at zero coordinates, destroying sparse patterns (see MNIST demonstration in Figure 1). This is catastrophic for zeros with specific physical meanings, such as biological dropout or "no energy deposition" in physics. (2) **Computational waste**—Rate-distortion analysis (Figure 3) shows that while DMs allocate almost zero bit rate to zero dimensions, the denoising network still performs forward passes across all dimensions: "Information capacity is concentrated in informative dimensions, but computation is not." (3) Existing workarounds are flawed: thresholding outputs (DDPM-T/LDM-T) preserves sparsity but sacrifices detail; domain-specific models (SARM) rely on manual spiral-sampling priors for zero positions, limiting generalization; discrete DMs cannot handle continuous values; and recent sparse-aware DMs (Ostheimer 2025) double the dimensions by adding binary indicators.

**Key Challenge**: Information (rate) is sparse, but the **computation and parameterization** of dense DMs are dense, creating an inherent mismatch. There is currently no solution that simultaneously preserves dense architecture scalability and sparse semantic information without relying on manual priors.

**Goal**: (1) Enable DMs to perform diffusion on compact representations containing **only non-zero dimensions**, so computation scales with **signal density** rather than ambient dimension; (2) **Strictly preserve zero patterns**, ensuring output zero positions match the ground truth; (3) Remain domain-agnostic across physics, biology, and vision.

**Key Insight**: Each sparse sample is represented as a "(set of dimension indices, set of corresponding non-zero values)" pair. A Transformer encoder pools this variable-length set into a **fixed-length** dense latent variable $\mathbf{z}$. Diffusion is performed in this dense latent space, while decoding involves autoregressively generating "the next dimension-value pair" until an [EOS] token is reached.

**Core Idea**: Break the implicit assumption that "diffusion must occur across all dimensions." Maintain dense stability in the **latent space** for diffusion, but ensure the **input representation** and **decoding** skip zeros, allowing computational power to follow the signal.

## Method

### Overall Architecture
A two-stage LDM-style training approach: (1) **SAVAE (Sparsity-Aware VAE)**—The Non-Zero Extractor (NZE) converts $\mathbf{x}^{(i)} \in \mathbb{R}^s$ into $(\mathbf{d}^{(i)}, \mathbf{v}^{(i)})$ (of length $l_i \ll s$). A Transformer encoder $q_\phi$ processes variable-length sequences of "dimension-value" tokens, and mean pooling produces a fixed-length $\mathbf{z}$. An autoregressive decoder $p_\theta = p_{\theta_1}(\mathbf{d}) p_{\theta_2}(\mathbf{v})$ sequentially predicts the next dimension (multinomial distribution) and value (Gaussian). (2) **Latent Space Diffusion**—Once SAVAE is trained and frozen, a standard DM (DDPM/DDIM) is trained on the $\mathbf{z}$ space, denoted as SEDP/SEDI. At generation, $\mathbf{z}_0$ is sampled and then decoded into a dimension-value sequence to fill the sparse vector.

### Key Designs

1. **Sparse-to-Dense Latent Encoding (SAVAE)**:
    - **Function**: Compresses high-dimensional sparse data into a fixed-size dense latent representation, allowing the diffusion model to train stably in a low-dimensional dense space.
    - **Mechanism**: NZE extracts the set of non-zero indices $\mathbf{d}^{(i)} = \{j | \mathbf{x}^{(i)}_j \neq 0\}$ and values $\mathbf{v}^{(i)}$, where length $l_i = \|\mathbf{x}^{(i)}\|_0 \ll s$. **Dimension Encoding (DE)** is introduced—similar to positional encoding, $\text{DE}_{(dim, 2i)} = \sin(dim / k^{2i/d_{model}})$ ($k=20000$), but it encodes **dimension indices** rather than sequence positions. Values are embedded via linear projection and added to the DE. Mean pooling of the encoder output yields $\mathbf{z}$ (mean pooling was chosen over [CLS] tokens for higher stability).
    - **Design Motivation**: Transformer input sequence length depends on the **number of non-zeros $l_i$** rather than ambient dimension $s$, which is the source of SED's computational efficiency. The resulting $\mathbf{z}$ is a dense vector compatible with standard DDPM/DDIM.

2. **Autoregressive Sparse Decoding (dim-value pairs)**:
    - **Function**: Decodes $\mathbf{z}$ back to the sparse space, determining both **which** dimensions are non-zero (variable length) and their **values**.
    - **Mechanism**: The decoder $p_\theta(\mathbf{d}, \mathbf{v} | \mathbf{z})$ is decomposed into two heads: $p_{\theta_1}$ outputs a multinomial distribution over remaining dimensions to predict the next non-zero index, and $p_{\theta_2}$ outputs a Gaussian distribution at that position for the value. The heads are **jointly trained**. Decoding follows a canonical ascending order of indices until [EOS].
    - **Design Motivation**: The number of non-zeros $l_i$ varies by sample (e.g., active genes in different cells); autoregression addresses this structural requirement. Sorting by index eliminates permutation ambiguity. In training, **teacher forcing** allows for parallel evaluation.

3. **Sparsity-Aware Latent Diffusion SED + Self-Conditioning Training**:
    - **Function**: Performs diffusion in the dense low-dimensional latent space provided by SAVAE, focusing computational loss only on informative dimensions.
    - **Mechanism**: After freezing SAVAE, training proceeds as $\mathcal{L}_{\text{SED}}(\theta) = \mathbb{E}\|\mathbf{z}_0 - f_\theta(\mathbf{z}_t, t, \tilde{\mathbf{z}}_0)\|^2$, where $\mathbf{z}_t = \sqrt{\gamma(t)}\mathbf{z}_0 + \sqrt{1-\gamma(t)}\boldsymbol{\epsilon}$ and $\tilde{\mathbf{z}}_0$ is the previous estimate from self-conditioning. The backbone is an MLP-based time-conditioned U-Net.
    - **Design Motivation**: Keeps diffusion stable while isolating sparse customizations to SAVAE. This decoupling makes modules replaceable.

### Loss & Training
SAVAE uses a $\beta$-VAE formulation: $\mathcal{L}_{\text{SAVAE}} = -\log p_\theta(\mathbf{d}, \mathbf{v}|\mathbf{z}) + \beta \cdot D_{\text{KL}}(q_\phi \| p)$, with $\beta = 10^{-6}$. The negative log-likelihood is split into dimension (multinomial) and value (Gaussian) components. Training is two-stage: SAVAE reaches convergence, then the diffusion model is trained on the frozen latents. $\gamma(t)$ is parameterized via log-SNR.

## Key Experimental Results

### Main Results
Evaluation across three domains and six datasets: **Physics**—muon calorimeter images ($32 \times 32$, ~95% zero); **Biology**—Tabula Muris (98% zero) and Human Lung PF (96% zero) scRNA; **Vision**—MNIST (81% zero), Fashion-MNIST (50% zero).

| Task | Model (Params) | Metric | Value | Notes |
|------|------------|------|------|------|
| Muon Signal | DDPM (37M) | $W_P (P_T)$↓ | 220.32 | Dense model failure |
| Muon Signal | DDPM-T (37M) | $W_P (P_T)$↓ | 24.22 | Thresholding mitigates |
| Muon Signal | SARM (25M, domain) | $W_P (P_T)$↓ | 28.01 | Uses spiral prior |
| Muon Signal | **SEDP (15M)** | $W_P (P_T)$↓ | **16.31** | Fewest params, best performance |
| Tabula Muris | DDPM (5M) | SCC↑ / MMD↓ | 0.50 / 3.60 | Dense model failure |
| Tabula Muris | scDiffusion (5M, domain) | SCC↑ / MMD↓ | 0.71 / 1.53 | Requires cell corpus pretraining |
| Tabula Muris | **SEDP (4M)** | SCC↑ / MMD↓ | **0.74 / 0.55** | No domain pretraining required |
| Human Lung PF | **SEDP (4M)** | SCC↑ / MMD↓ | **0.82 / 0.54** | Outperforms scDiffusion |

### Ablation Study

| Config | Key Metric | Description |
|------|---------|------|
| SED Full | Optimal | — |
| LDM (no sparsity support) | MMD=5.82 | Good shape, poor distribution distance |
| LDM-T (thresholded) | SCC drops to 0.26 | Thresholding destroys LDM details |
| Original DDPM/DDIM | Failure | Poor sparsity and large distance |
| Sampling Time (95% sparse) | 24ms (SED) vs 453ms (DDPM) | 19× speedup at high sparsity |
| Index Acc (Muon BG) | 87.9% | Failure rate highest in complex structures |

### Key Findings
- **Computation is almost constant relative to dimension**: For scRNA data, adding zero genes up to 27k dimensions causes linear growth for DDPM/LDM, but SED remains almost flat (Figure 2/9).
- **Higher sparsity leads to greater speedup**: Muon (95%) shows ~20× acceleration, MNIST (81%) 7×, while Fashion-MNIST (50%) shows almost no speedup—SED’s advantage scales with sparsity.
- **SED outperforms domain-specific baselines on scRNA** without requiring expensive cell corpus pretraining.
- Autoregressive ordering errors **do not** systematically worsen on long sequences (100% accuracy on Fashion-MNIST); errors are driven by data complexity rather than sequence length.

## Highlights & Insights
- **"Information dimensions are the only ones needing computation"**: The rate-distortion analysis proving that DDPM wastes full computation on zero-bit dimensions is a powerful diagnostic.
- **Dimension Encoding (DE)**: Adapting positional encoding concepts to feature indices is a concise engineering improvement that allows Transformers to ingest sparse (index, value) pairs directly.
- **Two-stage Decoupling**: SAVAE handles "how to represent sparsity," while diffusion handles "how to generate latents"—allowing for modularity.
- This approach is transferable to graph generation (edge sparsity), 3D point clouds (spatial sparsity), and KV cache compression.

## Limitations & Future Work
- **Dependency on Autoregressive Decoding**: Sampling must be serial, creating latency overhead for very long non-zero sequences; non-autoregressive alternatives are needed.
- **Dimension Ordering Errors**: Can produce unrealistic samples (Figure 7 MNIST), particularly in complex patterns where ~12% of Muon samples are affected.
- **Diminishing Returns at Low Sparsity**: For Fashion-MNIST (50% zero), SED's sampling time is comparable to DDPM, while LDM is faster.
- **Sparsity vs. Fidelity Trade-off**: SED is occasionally slightly weaker than specific LDM configurations in SCC on scRNA, despite having better MMD.

## Related Work & Insights
- **vs DDPM/LDM (dense baselines)**: SED runs only on non-zeros, providing efficiency and strict zero-pattern preservation.
- **vs DDPM-T / LDM-T (post-hoc thresholding)**: Thresholding is a hack that preserves sparsity but ruins boundary details.
- **vs SARM (Lu 2021)**: SARM relies on hard-coded spiral priors; SED is domain-agnostic and performs better.
- **vs Discrete DM (Austin 2021)**: Discrete DMs cannot handle continuous values; SED models both signal location and continuous magnitude.
- **vs scDiffusion (Luo 2024)**: SED surpasses it without needing large-scale cell corpus pretraining.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICML 2026\] Unified Masked Diffusion Models with Diverse Generation Orders](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)
- [\[ICML 2026\] When Preference Labels Fall Short: Aligning Diffusion Models from Real Data](when_preference_labels_fall_short_aligning_diffusion_models_from_real_data.md)

</div>

<!-- RELATED:END -->
