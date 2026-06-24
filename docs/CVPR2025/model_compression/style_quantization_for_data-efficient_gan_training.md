---
title: >-
  [Paper Note] Style Quantization for Data-Efficient GAN Training
description: >-
  [CVPR 2025][Model Compression][Few-shot GAN Training] SQ-GAN enhances the effectiveness of discriminator consistency regularization under limited data by discretely quantizing the intermediate style space of StyleGAN into a learnable codebook, compressing the sparse continuous latent space into a compact and structured discrete proxy space. By utilizing CLIP embeddings and optimal transport distance to initialize the codebook, external semantic knowledge is injected into the…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Few-shot GAN Training"
  - "Style Space Quantization"
  - "Consistency Regularization"
  - "Codebook Learning"
  - "Optimal Transport"
date: 2026-05-08
content_hash: 8c10eb3402c924e4
---

# Style Quantization for Data-Efficient GAN Training

**Conference**: CVPR 2025  
**arXiv**: [2503.24282](https://arxiv.org/abs/2503.24282)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Few-shot GAN Training, Style Space Quantization, Consistency Regularization, Codebook Learning, Optimal Transport

## TL;DR

SQ-GAN enhances the effectiveness of discriminator consistency regularization under limited data by discretely quantizing the intermediate style space of StyleGAN into a learnable codebook, compressing the sparse continuous latent space into a compact and structured discrete proxy space. By utilizing CLIP embeddings and optimal transport distance to initialize the codebook, external semantic knowledge is injected into the codebook, significantly improving the generation quality of few-shot GANs.

## Background & Motivation

**Background**: Training GANs under limited data scenarios (hundreds to thousands of images) makes the discriminator highly susceptible to overfitting, which leads to a decline in generation quality. Existing solutions include data augmentation (DiffAug, ADA), model regularization (LeCam, DigGAN, CR), and introducing external knowledge using pretrained models (KD-DLGAN). Among these, consistency regularization (CR) enhances robustness by forcing the discriminator to yield consistent scores for images generated from $z$ and $z+\epsilon$.

**Limitations of Prior Work**: CR faces a fundamental challenge under limited data: the generator's exploration of the input latent space $\mathcal{Z}$ is insufficient, causing neighboring latent variables $z$ and $z+\epsilon$ in the latent space to potentially map to images with drastically different levels of realism. Forcing the discriminator to output consistent scores for these vastly different images actually impairs its discriminative capability.

**Key Challenge**: There is a fundamental conflict between limited data and the sufficient exploration of continuous latent spaces. Continuous latent spaces are too large and sparse to be fully covered by limited samples, causing sampled pairs to fall into "barren regions", making the generation quality uncontrollable.

**Goal**: To construct a more compact, structured discrete proxy space to replace the original continuous latent space, ensuring that neighboring latent variables map more reliably to images of similar quality under limited data, thereby truly unleashing the benefits of consistency regularization.

**Key Insight**: Inspired by vector quantization (VQ-VAE, VQ-GAN), but instead of performing quantization in the data space, quantization is performed on the intermediate latent space of StyleGAN (style space $\mathcal{W}$)—which is more disentangled than the original $\mathcal{Z}$, with each dimension controlling different attributes.

**Core Idea**: Segment-wise quantize the intermediate latent variable $w$ onto a learnable codebook to form a discrete proxy space $\mathcal{W}^q$, where consistency regularization is performed; meanwhile, inject training data semantic knowledge into the codebook initialization using CLIP and optimal transport.

## Method

### Overall Architecture

SQ-GAN introduces three key components to the StyleGAN2 framework: (1) Style Quantization—segmenting $w$ and quantizing it to a codebook, then driving the synthesis network with the quantized $w^q$; (2) Quantized Consistency Regularization—performing CR on the quantized proxy space; (3) Knowledge-Enhanced Codebook Initialization (CBI)—pre-aligning the codebook codes with the semantics of the training data prior to training using CLIP features and optimal transport distance.

### Key Designs

1. **Style Quantization**:

    - **Function**: Compresses the continuous and sparse style space $\mathcal{W}$ into a compact, structured discrete proxy space $\mathcal{W}^q$.
    - **Mechanism**: Subdivides the intermediate latent variable $w \in \mathbb{R}^{d_w}$ into $s$ sub-vectors $\{\hat{w}_i\}_{i=1}^s$, where each $\hat{w}_i \in \mathbb{R}^{d_w/s}$ (assuming $d_w/s=4$). Each sub-vector is quantized to its nearest neighbor in the learnable codebook $\mathcal{C} \in \mathbb{R}^{k \times (d_w/s)}$: $\hat{w}_i^q = \arg\min_{c_j \in \mathcal{C}} \|\hat{w}_i - c_j\|$. The quantized sub-vectors are concatenated as $w^q = [\hat{w}_1^q, ..., \hat{w}_s^q]$ and fed into the synthesis network. The proxy space is the Cartesian product of the $s$ codebooks, $\mathcal{W}^q = \mathcal{C}^1 \times ... \times \mathcal{C}^s$.
    - **Design Motivation**: The original continuous $\mathcal{W}$ cannot be sufficiently covered under limited data, whereas the discretized $\mathcal{W}^q$ is much smaller and each code is more semantic. Neighboring quantized codes are more likely to map to images of consistent quality, fundamentally solving the effectiveness issue of CR. Moreover, performing quantization in the style space (rather than the original $z$ space) ensures disentanglement—where each sub-vector controls different attributes.

2. **Quantized Consistency Regularization (Quantized CR) + Uniformity Constraint**:

    - **Function**: Enhances discriminator robustness on the quantized proxy space while preventing codebook collapse.
    - **Mechanism**: Maps a perturbed $z+\epsilon$ to $w' = f_\mathcal{W}(z+\epsilon)$ and quantizes it to obtain $w'^q$. The CR loss is defined as $\mathcal{L}_{qcr} = \mathbb{E}[\|f_D(g(w^q)) - f_D(g(w'^q))\|^2]$. Even if the perturbed $w'$ differs from $w$, they may fall onto the same codebook code after quantization (yielding identical outputs), or onto neighboring but equally high-quality codes. To prevent codebook collapse, the codebook is projected onto a unit hypersphere and the RBF kernel uniformity loss is minimized: $\mathcal{L}_{uf} = \log \mathbb{E}[\exp(-t\|\bar{c}_i - \bar{c}_j\|^2)]$.
    - **Design Motivation**: Discretization naturally provides "quantization robustness"—small perturbations do not necessarily alter the quantization result, making the CR constraint more reasonable. The uniformity constraint ensures that the codebook codes are uniformly distributed on the hypersphere, preventing degeneration to a small subset of codes.

3. **Knowledge-Enhanced Codebook Initialization (CBI)**:

    - **Function**: Leverages semantic knowledge from a pretrained foundation model to provide a meaningful initial state for the codebook.
    - **Mechanism**: Extracts training image features $F = \{f_i\}$ using the CLIP image encoder, and processes the quantized discrete codes using the CLIP text encoder (mapped to token embeddings via an MLP) to obtain features $T = \{t_i\}$. The optimal transport distance based on the Sinkhorn algorithm is calculated and minimized between the two feature sets: $\mathcal{L}_{ot} = \mathbb{E}[d(T, F) \cdot \gamma^*]$, where $\gamma^*$ is the optimal transport plan. The overall initialization optimizes $\mathcal{L}_{sq} + \mathcal{L}_{uf} + \mathcal{L}_{ot}$.
    - **Design Motivation**: Learning a semantically rich codebook from scratch with limited data is challenging. As a foundation model pretrained on large-scale data, the CLIP feature space carries rich visual-semantic priors. Aligning the codebook with the CLIP feature space via optimal transport essentially pre-builds a "semantic vocabulary" for the codebook, significantly accelerating subsequent training convergence.

### Loss & Training

- Generator loss: $\mathcal{L}(g, f_\mathcal{W}, \mathcal{C}, P) = \mathcal{L}_{adv}(g) + \lambda_{sq}(\mathcal{L}_{sq} + \mathcal{L}_{uf})$
- Discriminator loss: $\mathcal{L}(f_D) = \mathcal{L}_{adv}(f_D) + \lambda_{qcr} \mathcal{L}_{qcr}$
- $\lambda_{sq} = 0.01$, $\lambda_{qcr} = 0.01$, perturbation strength $\sigma = 0.1$
- Quantization employs a straight-through gradient estimator to handle non-differentiable operations.
- During the CBI stage, the codebook is pretrained using $\mathcal{L}_{sq} + \mathcal{L}_{uf} + \mathcal{L}_{ot}$ before formal GAN training.
- Resolution of 256×256 with StyleGAN2 architecture.

## Key Experimental Results

### Main Results

| Dataset | Metric | SQ-GAN+CBI | CR | StyleGAN2 | Gain (vs CR) |
|--------|------|------|----------|------|------|
| Oxford-Dog | FID↓ | **35.01** | 48.73 | 64.26 | -13.72 |
| Oxford-Dog | IS↑ | **12.44** | 10.47 | 9.69 | +1.97 |
| FFHQ-2.5K | FID↓ | **22.04** | 41.43 | 48.11 | -19.39 |
| FFHQ-2.5K | IS↑ | **4.20** | 4.06 | 3.50 | +0.14 |
| MetFaces (1.2K) | FID↓ | **35.44** | 48.89 | 53.21 | -13.45 |
| BreCaHAD (1.75K) | FID↓ | **42.42** | 80.72 | 97.06 | -38.30 |

The performance is further boosted when integrated with ADA augmentation: MetFaces FID drops to 24.77 (vs CR+ADA 29.91), and BreCaHAD FID drops to 22.61 (vs CR+ADA 22.69).

### Ablation Study

| Configuration | FID↓ (Oxford-Dog) | Description |
|------|---------|------|
| SQ-GAN + CBI | 35.01 | Full model |
| SQ-GAN (w/o CBI) | 36.30 | CBI provides an additional 1.3 FID improvement |
| CR only (w/o quantization) | 48.73 | Quantization brings a ~12 FID improvement |
| StyleGAN2 baseline | 64.26 | Baseline |
| SQ-GAN w/o uniformity constraint | ~40 | Degenerates due to codebook collapse |

### Key Findings

- **Quantization is the most critical contribution**: Introducing quantization alone (without CBI) reduces FID from 48.73 to 36.30, contributing approximately 85% of the total improvement.
- CBI delivers extra gains across all datasets, acting more prominently on extremely small datasets (such as MetFaces 1.2K).
- SQ-GAN is orthogonal to and stackable with ADA augmentation, showing that quantization and augmentation address distinct challenges.
- The FID curve is smoother during training—training dynamics are more stable after quantization, mitigating the discriminator overfitting trend.
- The improvement is most significant on extremely small datasets (MetFaces 1.2K, BreCaHAD 1.75K), suggesting that quantization provides higher value as data size decreases.

## Highlights & Insights

- **Analyzing the failure of CR from the perspective of latent space coverage** is a profound insight—CR papers usually focus on regularization forms, while this work points out that the premise of CR's effectiveness is that adjacent latent variables map to images of "consistent quality", which does not hold true under limited data. Quantization fundamentally restores this premise.
- **Performing quantization in style space rather than z space** is a crucial design choice—the $\mathcal{W}$ space is more decoupled than $\mathcal{Z}$, and each segment naturally corresponds to different attributes, making the quantized combinations still meaningful.
- **Comparing the codebook to a "semantic vocabulary"** is an elegant conceptualization: codebook codes = words, quantized combinations = sentences describing images, CBI = pre-building the vocabulary using CLIP.

## Limitations & Future Work

- The proposed method is verified only on StyleGAN2 at 256×256 resolution, without scaling to larger models (e.g., StyleGAN3) or higher resolutions.
- The codebook size $k$ and the number of segments $s$ are hyperparameters that need tuning, which may require different configurations for different datasets.
- CBI relies on the quality of the CLIP model, and its effectiveness might be limited in specialized domains where CLIP exhibits poor coverage (such as the medical imaging dataset BreCaHAD).
- Future directions: exploring adaptive codebook size adjusting; applying the quantization concept to the training of diffusion models with limited data; combining the codebook to achieve controllable attribute manipulation.

## Related Work & Insights

- **vs CR (Zhao et al.)**: CR directly exerts consistency constraints in continuous space, which is effective when data is abundant but fails under limited data. SQ-GAN restores the prerequisite for CR via quantization, which can be viewed as "the correct way to use CR".
- **vs VQ-GAN / VQ-VAE**: VQ-GAN performs vector quantization in the data space for image tokenization, while SQ-GAN performs quantization in the latent space to enhance training regularization—differing in both objective and location.
- **vs KD-DLGAN**: KD-DLGAN distills knowledge from pretrained models into discriminator features, while SQ-GAN's CBI injects knowledge into the codebook initialization. The two approaches for introducing external knowledge are complementary.
- **vs ADA**: ADA mitigates overfitting via adaptive data augmentation, while SQ-GAN improves regularization through latent space compression. The two are orthogonal and stackable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to apply style space quantization to enhance GAN training on limited data, with a novel concept and theoretical support.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 4 datasets, compared with various SOTA methods, with complete ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ In-depth motivation analysis, with logical derivation of the quantization scheme based on the failure causes of CR.
- Value: ⭐⭐⭐⭐ Restructures a new paradigm for training GANs on limited data; the quantization concept is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](../../ECCV2024/model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](../../ACL2025/model_compression/efficientqat.md)

</div>

<!-- RELATED:END -->
