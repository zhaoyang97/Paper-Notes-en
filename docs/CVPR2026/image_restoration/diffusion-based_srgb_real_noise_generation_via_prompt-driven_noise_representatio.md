---
title: >-
  [Paper Note] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning
description: >-
  [CVPR 2026][Image Restoration][sRGB noise generation] PNG introduces learnable Global/Local Prompt components to automatically extract noise characteristics from real noise (replacing metadata such as ISO and camera model). A Prompt AutoEncoder encodes noise into a latent space, and a Prompt DiT (based on a consistency model) generates latent codes in a single step, enabling realistic sRGB noise synthesis without any metadata. The downstream DnCNN denoiser trained on PNG-synthesized data trails real-data training by only 0.08 dB on SIDD.
tags:
  - CVPR 2026
  - Image Restoration
  - sRGB noise generation
  - prompt learning
  - consistency model
  - denoising
  - metadata-free
date: 2026-05-08
content_hash: fbb468f358fa8f16
---

# PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning

**Conference**: CVPR 2026
**arXiv**: [2603.04870](https://arxiv.org/abs/2603.04870)
**Code**: None
**Area**: Image Denoising / Noise Generation
**Keywords**: sRGB noise generation, prompt learning, consistency model, denoising, metadata-free

## TL;DR

PNG introduces learnable Global/Local Prompt components to automatically extract noise characteristics from real noise (replacing metadata such as ISO and camera model). A Prompt AutoEncoder encodes noise into a latent space, and a Prompt DiT (based on a consistency model) generates latent codes in a single step, enabling realistic sRGB noise synthesis without any metadata. The downstream DnCNN denoiser trained on PNG-synthesized data trails real-data training by only 0.08 dB on SIDD.

## Background & Motivation

**State of the Field**: sRGB-domain denoising is a core problem in low-level vision. Mainstream supervised methods rely on large collections of noisy-clean image pairs, but acquiring such paired real data is extremely costly (requiring multi-frame averaging or specialized hardware), limiting practical applicability. As a result, **noise synthesis** methods have emerged, using generative models to synthesize realistic noisy images for data augmentation.

**Limitations of Prior Work**: Existing noise generation methods (NoiseFlow, Flow-sRGB, NeCA-W, etc.) depend on camera metadata (e.g., ISO value, sensor model, shutter speed) as conditioning inputs at both training and inference time. In practice, however: (a) publicly available sRGB images are typically post-processed, with EXIF tags stripped; (b) metadata formats in scientific imaging and similar domains are inconsistent or absent; (c) metadata semantics are inconsistent across devices. These factors limit the generalizability of existing approaches.

**Root Cause**: Metadata is essentially a compact description of the noise distribution (ISO → gain → noise intensity; camera model → ISP pipeline → spatial noise correlations). The key question is whether such a description can be learned directly from the noisy image itself, without relying on external metadata.

**Paper Goals**: (a) Eliminate dependence on metadata at both training and inference stages; (b) learn high-dimensional feature representations that substitute for metadata from limited noise samples; (c) generate sufficiently realistic noise such that downstream denoisers approach or surpass the performance of models trained on real data.

**Starting Point**: Drawing inspiration from prompt learning in NLP and vision, the paper employs learnable prompt components as implicit encoders of noise characteristics, automatically extracting prompt features from the statistical properties of input noise (channel mean/variance reflecting ISO; local correlation coefficients reflecting sensor characteristics) as a substitute for explicit metadata.

**Core Idea**: Replace metadata with learned Global Prompts (capturing ISO-related global noise statistics) and Local Prompts (capturing ISP-pipeline-induced local spatial correlations), which drive a diffusion-based noise generation framework built on a consistency model.

## Method

### Overall Architecture

PNG consists of two core components trained in two stages:

**Stage 1: Prompt AutoEncoder (PAE)**. The input is real noise $\mathbf{n}_{Real} = \mathcal{I}_{Noisy} - \mathcal{I}_{Clean}$. The Prompt Encoder $\mathcal{E}$ encodes the noise into a latent code $\mathbf{z}$, while simultaneously generating prompt features $\mathbf{F}_{Global}$ and $\mathbf{F}_{Local}$ via the Global Prompt Block and Local Prompt Block. The Decoder $\mathcal{D}$ reconstructs the noisy image $\hat{\mathcal{I}}_{Noisy}$ conditioned on $\mathbf{z}$ and the clean image $\mathcal{I}_{Clean}$ (learning signal-dependent characteristics). The PAE is trained with an $\mathcal{L}_1$ reconstruction loss and an $\mathcal{L}_2$ latent regularization term.

**Stage 2: Prompt DiT (P-DiT)**. A consistency model (CM) is trained in the latent space of the PAE. Conditioned on prompt features and the clean image, P-DiT maps random noise $\mathbf{z}_T$ to a latent code $\hat{\mathbf{z}}_0$ in a single step, which is then decoded by the PAE Decoder into a noisy image.

**Inference**: Given a small number of real noise samples $\mathbf{n}_{Real}$, the Prompt Encoder extracts prompt features → P-DiT generates a new latent code in one step → Decoder + clean image → synthesized noisy image.

### Key Designs

1. **Global Prompt Block (GPB)**:

    - **Function**: Captures global noise statistics related to ISO/gain.
    - **Mechanism**: Defines learnable parameters $\mathbf{P}_{Global}^{\ell} \in \mathbb{R}^{\frac{H}{2^\ell} \times \frac{W}{2^\ell} \times C_{Global}^\ell}$. The channel mean $\mu$ and standard deviation $\Sigma$ are computed from the input feature $\mathbf{F}_{In}^\ell$ (reflecting global noise intensity). A modulation coefficient is produced via $1 \times 1$ convolution and softmax: $\mathbf{w}_{Global}^\ell = \text{Softmax}(\text{Conv}_{1 \times 1}[\mu(\mathbf{F}_{In}^\ell), \Sigma(\mathbf{F}_{In}^\ell)])$, which is then element-wise multiplied with the prompt component and passed through a $3 \times 3$ convolution: $\mathbf{F}_{Global}^\ell = \text{Conv}_{3 \times 3}(\mathbf{w}_{Global}^\ell \odot \mathbf{P}_{Global}^\ell)$.
    - **Design Motivation**: ISO directly determines sensor gain, and high ISO amplifies noise. Channel mean and standard deviation naturally reflect this global noise level, enabling implicit encoding without requiring the ISO value.

2. **Local Prompt Block (LPB)**:

    - **Function**: Captures local spatial noise correlations introduced by the camera ISP pipeline (noise patterns specific to a given camera model).
    - **Mechanism**: For each pixel location, a $\rho \times \rho$ patch is extracted and the Pearson correlation coefficient between neighboring pixels and the center pixel is computed, yielding a correlation map $\mathbf{F}_\rho \in \mathbb{R}^{H \times W \times \rho^2}$. Row-wise and column-wise means are computed separately (capturing directional noise patterns caused by ISP nonlinearities). These are processed through CoMB ($1 \times 1$ conv → bilinear upsampling → $3 \times 3$ conv) and softmax to produce $\mathbf{w}_{Local} = \text{Softmax}(\text{CoMB}([\text{Avg}_{row}(\mathbf{F}_\rho), \text{Avg}_{col}(\mathbf{F}_\rho)]))$, which modulates the prompt component: $\mathbf{F}_{Local} = \text{Conv}_{3 \times 3}(\mathbf{w}_{Local} \odot \mathbf{P}_{Local})$.
    - **Design Motivation**: Real sRGB noise is not i.i.d. — operations in the ISP pipeline such as demosaicing, nonlinear tone mapping, and spatially adaptive processing introduce spatially correlated noise. These local correlation patterns are key features distinguishing different camera models and cannot be captured by global statistics alone.

3. **Prompt DiT (P-DiT)**:

    - **Function**: Trains a consistency model in the PAE latent space to generate noise-characteristic-embedded latent codes in a single step.
    - **Mechanism**: Based on the DiT-S architecture ($B=8$ blocks, patch size=1 to preserve fine-grained information). Conditioning inputs include timestep embeddings, the clean image, and prompt features. The clean image, $\mathbf{F}_{Local}$, and $\mathbf{F}_{Global}$ are pixel-downsampled, processed through $3 \times 3$ convolutions, and concatenated to form $\mathbf{F}_{Cond}$, which is globally average-pooled and added to the timestep embedding. Each P-DiT block additionally incorporates Prompt Attention, injecting Q/K/V generated from the condition features into the attention layers to capture spatial information from the prompt features.
    - **Design Motivation**: The single-step generation capability of consistency models substantially improves inference speed (57 images/sec at $256 \times 256$). Prompt Attention enables the model to fully exploit spatial conditioning information; compared to injecting only global conditions via AdaLN, this reduces KLD from 0.0287 to 0.0261 in ablations.

### Loss & Training

**PAE Training**: $\mathcal{L}_1$ loss (noisy image reconstruction) + $\mathcal{L}_2$ latent regularization. Adam optimizer with learning rate cosine-annealed from $10^{-4}$ to $10^{-6}$; 400k iterations; patch size $256 \times 256$; batch size 64.

**P-DiT Training**: Consistency training loss (pseudo-Huber loss), $d(\mathbf{x},\mathbf{y}) = \sqrt{\|\mathbf{x}-\mathbf{y}\|_2^2 + c^2} - c$. RAdam optimizer with fixed learning rate $2 \times 10^{-4}$; 250k iterations; $256 \times 256$ patches encoded to $32 \times 32$ latents; batch size 512. EMA (decay 0.9999) is used for training stability. The discretization curriculum increases from $s_0=10$ to $s_1=160$ with lognormal noise sampling.

## Key Experimental Results

### Main Results: Noise Quality on SIDD Validation Set (KLD↓ / AKLD↓)

| Camera | C2N KLD | NeCA-W KLD | NAFlow KLD | **PNG KLD** |
|--------|---------|-----------|------------|------------|
| G4 | 0.1660 | 0.0242 | 0.0254 | **0.0174** |
| GP | 0.1315 | 0.0432 | 0.0352 | **0.0143** |
| IP | 0.0581 | 0.0410 | 0.0339 | **0.0291** |
| N6 | 0.3524 | 0.0206 | 0.0309 | **0.0167** |
| S6 | 0.4517 | 0.0302 | 0.0272 | **0.0193** |
| **Avg** | 0.2129 | 0.0342 | 0.0305 | **0.0194** |

PNG achieves the best KLD and AKLD across all five smartphones.

### Downstream Denoising Performance (DnCNN on SIDD Benchmark)

| Training Data | PSNR (dB) | SSIM |
|---------------|-----------|------|
| C2N (synthetic) | 33.76 | 0.901 |
| NeCA-W (synthetic) | 34.74 | 0.912 |
| NAFlow (synthetic) | 37.22 | 0.935 |
| **PNG (synthetic)** | **37.55** | **0.937** |
| Real (real data) | 37.63 | 0.936 |

The denoiser trained on PNG-synthesized data lags the real-data oracle by only 0.08 dB in PSNR, while even surpassing it in SSIM by 0.001.

### Cross-Domain Generalization (Mixed Training: 50% Real + 50% Synthetic)

| Method | PolyU PSNR | Nam PSNR | SIDD Val PSNR | SIDD+ PSNR | Avg |
|--------|-----------|---------|--------------|-----------|-----|
| Real (100%) | 36.34 | 35.35 | 37.72 | 35.68 | 36.27 |
| NAFlow-Mixed | 37.29 | 37.47 | 37.66 | 36.27 | 37.17 |
| **PNG-Mixed** | **37.98** | **38.09** | **37.96** | **36.57** | **37.65** |

In mixed training, PNG outperforms real-data-only training on all four datasets, with an average PSNR gain of +1.38 dB.

### Ablation Study

| GPB | LPB | KLD↓ | AKLD↓ | Note |
|-----|-----|------|-------|------|
| ✗ | ✗ | 0.6182 | 0.4387 | No prompt; fails |
| ✓ | ✗ | 0.0287 | 0.1112 | Global only; large improvement |
| ✓ | ✓ | **0.0261** | **0.1108** | Global + Local; best |

In a metadata classification experiment, prompt features achieve 94.47% accuracy for camera sensor classification (baseline: 75.80%); for the joint ISO + sensor 16-class task, Top-1 accuracy is 75.48% and Top-3 is 98.64%, confirming that the prompts encode metadata-equivalent information.

### Key Findings

- GPB contributes the most — adding GPB reduces KLD from 0.6182 to 0.0287. LPB further reduces it to 0.0261; the individual gain is smaller but indispensable for modeling local correlations.
- Prompt Attention in P-DiT is important — jointly injecting conditions into both timestep embeddings and attention layers reduces KLD from 0.0287 to 0.0261 compared to using timestep embeddings alone.
- Inference speed: PNG achieves 57 images/sec at $256 \times 256$, 4.4× faster than NAFlow (13 images/sec); at $512 \times 512$, 21 images/sec vs. 8 images/sec.
- Scaling up the volume of synthetic data consistently improves generalization (×1 → ×4: average PSNR on external datasets improves from 36.74 to 37.23).

## Highlights & Insights

- **The prompt-as-metadata paradigm is elegant**: Rather than simply discarding metadata conditioning, the paper proposes a structured approach — channel statistics → global prompt, local Pearson correlations → local prompt — to extract equivalent information directly from the noise. The classification experiment provides interpretability, confirming that the prompts do encode sensor and ISO information.
- **Two-stage training + single-step CM generation**: The PAE learns a compact latent representation and prompt extraction in image space, while P-DiT performs single-step generation in latent space via a consistency model, achieving a favorable balance between generation quality and inference efficiency. This "learn a compact space first, then generate" paradigm is broadly applicable to other conditional generation tasks.
- **Cross-domain capability is a genuine highlight**: Trained on SIDD (smartphones), the model can directly generate corresponding noise using noise samples from PolyU/Nam (DSLRs), a scenario where metadata-dependent methods fail entirely.

## Limitations & Future Work

- A small number of real noise samples is still required (for prompt feature extraction); the approach is not fully zero-shot.
- The PAE contains approximately 44M parameters, and two-stage training (400k + 250k iterations) remains time-consuming.
- All experiments use DnCNN as the downstream denoiser; performance with stronger denoisers (e.g., Restormer, NAFNet) is not validated.
- Computing Pearson correlation coefficients for Local Prompts requires real noise patches, which may be limiting in fully unpaired settings.
- RAW-domain noise generation is not explored; RAW noise is more regular and metadata is more clearly defined, representing a potentially valuable direction.

## Related Work & Insights

- **vs. NAFlow**: NAFlow uses normalizing flows and does not require metadata at inference, but still requires it during training. PNG eliminates metadata dependence at both stages, achieving an average KLD improvement of 0.0111, a denoising PSNR gain of 0.33 dB, and 4.4× faster inference.
- **vs. NeCA-W**: NeCA-W trains a separate model for each camera model (5 × 40.5M parameters), whereas PNG handles all devices with a single unified model (44M parameters total) and is substantially superior in cross-domain settings.
- **vs. C2N/Flow-sRGB**: These earlier methods generate noise of far lower quality than PNG, with a large gap in downstream denoising performance (33–34 dB vs. 37.55 dB).
- This paper demonstrates the strong potential of prompt learning in low-level vision tasks. The paradigm of using prompts as implicit condition encoders is worth exploring in other degradation modeling tasks (e.g., blur, compression artifacts).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The prompt-as-metadata framework is creative; the design of GPB/LPB for automatically extracting features from noise statistics is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on SIDD/PolyU/Nam/SIDD+/MAI2021; dual evaluation of noise quality and denoising performance; paired and unpaired settings; comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Method descriptions are clear, figures are intuitive, experimental analysis is thorough. Supplementary material is very complete.
- **Value**: ⭐⭐⭐⭐ — Eliminating metadata dependence addresses a genuine practical need; strong cross-domain generalization makes the approach highly valuable for data augmentation scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)
- [\[CVPR 2026\] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](../../ACL2026/image_restoration/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)

<!-- RELATED:END -->
