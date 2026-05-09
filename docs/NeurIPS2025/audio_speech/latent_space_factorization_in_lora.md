---
title: >-
  [Paper Note] Latent Space Factorization in LoRA
description: >-
  [NeurIPS 2025][Audio & Speech][LoRA] This paper proposes FVAE-LoRA, which incorporates a VAE with dual latent spaces into the LoRA framework. Through a novel ELBO objective, it explicitly factorizes task-relevant features ($\mathbf{z}_1$) from residual information ($\mathbf{z}_2$), consistently outperforming standard LoRA across text, image, and audio tasks.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - LoRA
  - VAE
  - latent space factorization
  - parameter-efficient fine-tuning
  - spurious correlation robustness
date: 2026-05-08
content_hash: e58aec3487bd0ab7
---

# Latent Space Factorization in LoRA

**Conference**: NeurIPS 2025
**arXiv**: [2510.19640](https://arxiv.org/abs/2510.19640)
**Code**: [GitHub](https://github.com/idiap/FVAE-LoRA)
**Area**: Audio/Speech (Parameter-Efficient Fine-Tuning)
**Keywords**: LoRA, VAE, latent space factorization, parameter-efficient fine-tuning, spurious correlation robustness

## TL;DR

This paper proposes FVAE-LoRA, which incorporates a VAE with dual latent spaces into the LoRA framework. Through a novel ELBO objective, it explicitly factorizes task-relevant features ($\mathbf{z}_1$) from residual information ($\mathbf{z}_2$), consistently outperforming standard LoRA across text, image, and audio tasks.

## Background & Motivation

**State of the Field**: LoRA has become the dominant parameter-efficient fine-tuning (PEFT) method, enabling efficient adaptation via low-rank matrices $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ ($r \ll \min(d,k)$). Numerous variants (AdaLoRA, DoRA, PiSSA, rsLoRA, etc.) have extended LoRA from structural, optimization, and compression perspectives.

**Limitations of Prior Work**: The update mechanism of standard LoRA lacks explicit control — the low-rank subspace $\text{Im}(\mathbf{A})$ learned via gradient descent does not necessarily contain only task-relevant information, and may retain irrelevant or even harmful features from pretraining.

**Root Cause**: While low-rank constraints provide parameter efficiency, *what information is encoded* in the low-rank update is equally critical — yet none of the existing variants provide semantic control over the update content.

**Paper Goals**: To design a mechanism that explicitly separates task-critical information ($\mathbf{z}_1$) from residual information ($\mathbf{z}_2$), such that the LoRA update is driven solely by task-relevant features.

**Starting Point**: Embedding a VAE into LoRA's parameterization — replacing the $\mathbf{A}$ matrix with a VAE equipped with dual latent spaces, and guiding information separation through a factorized ELBO.

**Core Idea**: A VAE with dual latent spaces is injected into the $\mathbf{A}$ matrix of LoRA. Downstream tasks utilize only the task-relevant $\mathbf{z}_1$, enabling semantic-level information filtering.

## Method

### Overall Architecture

For each target linear layer:
- **During training**: Input $\mathbf{x}$ is fed into both the FVAE (to compute reconstruction loss) and the $\mathbf{B}$ matrix (multiplied by $\mathbf{z}_1$), yielding output $\mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{z}_1$
- **During inference**: Only encoder $q_{\phi_1}$ is used (mean or sample); $\mathbf{z}_2$-related components are discarded

### Key Designs

#### 1. Dual Latent Space VAE (FVAE)
Let $p(\mathbf{z}_1, \mathbf{z}_2) = p_1(\mathbf{z}_1) p_2(\mathbf{z}_2)$ and $q_\phi(\mathbf{z}_1, \mathbf{z}_2 | \mathbf{x}) = q_{\phi_1}(\mathbf{z}_1 | \mathbf{x}) q_{\phi_2}(\mathbf{z}_2 | \mathbf{x})$.

Standard dual-latent ELBO:
$$\mathcal{L}^{\text{VAE2LAT}} = \mathbb{E}_{\mathbf{z}_1, \mathbf{z}_2}[\log p_\theta(\mathbf{x} | \mathbf{z}_1, \mathbf{z}_2)] - D_{\text{KL}}(q_{\phi_1} \| p_1) - D_{\text{KL}}(q_{\phi_2} \| p_2)$$

#### 2. Factorized ELBO Objective (Core Innovation)
A repulsion regularization term is introduced to prevent $q_{\phi_2}$ from encoding information in the same region as $p_1$:

$$\mathcal{L}^{\text{FVAE}} = \alpha \underset{\mathbf{z}_1, \mathbf{z}_2}{\mathbb{E}}[\log p_\theta(\mathbf{x} | \mathbf{z}_1, \mathbf{z}_2)] - \beta D_{\text{KL}}(q_{\phi_1} \| p_1) + \delta \underbrace{\mathbb{E}_{\mathbf{z}_2, \mathbf{z}_1} \log \frac{p_2(\mathbf{z}_2)}{p_1(\mathbf{z}_1)}}_{\Gamma}$$

#### 3. Mechanism of the $\Gamma$ Regulator

$\Gamma$ decomposes into a **mismatch term** $\Lambda$ and a **divergence term** $\Delta$:
- Mismatch term $\Lambda = D_{\text{KL}}(q_{\phi_2} \| p_1) - D_{\text{KL}}(q_{\phi_2} \| p_2)$ → encourages $q_{\phi_2}$ to move away from $p_1$
- An upper bound on the divergence term $\Delta$ is proportional to the 2-Wasserstein distance → maximizing $\Delta$ increases the Wasserstein repulsion distance between the two encoders

Prior settings: $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$, $p_2 = \mathcal{N}(\mathbf{1.5}, \mathbf{I})$, providing an initial separation signal via distinct centers.

### Loss & Training

$$\min_{\phi, \theta} \mathcal{L}_{\text{downstream}} - \boldsymbol{\lambda} \sum_{l \in \text{layer}} \mathcal{L}^{\text{FVAE}}_{\theta, \phi}(\mathbf{x}_l)$$

- LoRA rank $r=16$ (also the dimensionality of $\mathbf{z}_1$)
- Only query and key matrices are adapted
- $\alpha, \beta, \delta$ are hyperparameters controlling reconstruction, regularization, and factorization strength, respectively

## Key Experimental Results

### Main Results — Image Classification (ViT-B/16)

| Method | Params | DTD | EuroSAT | GTSRB | RESISC45 | SUN397 | SVHN | Avg |
|--------|--------|-----|---------|-------|----------|--------|------|-----|
| Full FT | - | 78.12 | 98.30 | 98.85 | 94.35 | 69.34 | 97.34 | 89.38 |
| LoRA | 0.72% | 74.65 | 97.28 | 96.95 | 90.11 | 71.11 | 94.22 | 87.39 |
| DoRA | 0.75% | 75.74 | 97.28 | 97.27 | 91.72 | 71.53 | 96.41 | 88.32 |
| **FVAE-LoRA** | **0.73%** | **78.19** | **97.78** | **97.98** | **93.57** | **73.14** | **96.55** | **89.53** |

FVAE-LoRA surpasses full fine-tuning with a comparable parameter budget (89.53 vs. 89.38)!

### Text Tasks — Commonsense Reasoning (Llama-3-8B)

| Method | PIQA | SIQA | ARC-c | ARC-e | OBQA | HellaSwag | WinoGrande | Avg |
|--------|------|------|-------|-------|------|-----------|------------|-----|
| LoRA | 80.74 | 75.59 | 67.58 | 82.11 | 75.20 | 85.73 | 77.82 | 77.82 |
| HiRA | 88.63 | 80.40 | 81.66 | 93.56 | 87.20 | 94.48 | 85.87 | 87.40 |
| **FVAE-LoRA** | **88.96** | **81.58** | 81.06 | 92.72 | 86.20 | **95.30** | **88.95** | **87.82** |

### Audio Tasks — Speech Recognition (Wav2Vec2-Large)

| Method | TIMIT PER↓ |
|--------|-----------|
| Full FT | 7.48 |
| LoRA | 9.38 |
| **FVAE-LoRA** | **8.09** |

### Ablation Study — Robustness to Spurious Correlations

| Method | ANIMALS WG | WATERBIRDS WG | CELEBA WG | Gap | WG-AVG |
|--------|-----------|--------------|----------|-----|--------|
| LoRA | 54.79 | 75.49 | 40.00 | 34.8 | |
| **FVAE-LoRA** | **62.0** | **75.85** | **43.33** | **31.71** | |

FVAE-LoRA consistently achieves higher worst-group accuracy, confirming that $\mathbf{z}_1$ captures causally relevant task features.

### Ablation — Necessity of the Factorization Objective

| Method | Avg Accuracy |
|--------|-------------|
| VAE2LAT (no factorization) | 86.43 |
| $\beta$-VAE2LAT | 87.29 |
| **FVAE-LoRA** | **89.53** |

### Key Findings

1. FVAE-LoRA **consistently** outperforms standard LoRA across image, text, and audio modalities
2. On 6 image datasets, FVAE-LoRA marginally surpasses full fine-tuning with only 0.73% of parameters
3. Spurious correlation experiments confirm that $\mathbf{z}_1$ learns causally relevant features
4. The $\Gamma$ repulsion regularizer is critical for performance (ablation: +3.1 pp)

## Highlights & Insights

- **Principled Innovation**: The first work to apply VAE latent space factorization to PEFT, enabling semantic-level control over LoRA update content
- **Theoretical Rigor**: An upper-bound proof on the Wasserstein repulsion distance provides theoretical grounding for the $\Gamma$ regulator
- **Cross-Modal Generality**: Demonstrated effectiveness on text (Llama-3-8B), image (ViT-B/16), and audio (Wav2Vec2)
- **Robustness Gains**: The spurious correlation experiments offer compelling validation — improvements are observed not only in average accuracy but also in worst-group performance

## Limitations & Future Work

1. Additional hyperparameters ($\alpha, \beta, \delta, \lambda$, prior centers) increase tuning complexity
2. Training requires running the full FVAE (two encoders + one decoder), incurring greater computational cost than standard LoRA
3. Inference requires only $q_{\phi_1}$, but more parameters must be stored
4. The prior center $\mu_2 = 1.5$ is chosen empirically, and sensitivity analysis remains limited

## Related Work & Insights

- **LoRA** (Hu et al., 2022): The foundation of this work
- **$\beta$-VAE** (Higgins et al., 2017): A seminal method for disentangled latent representations
- **DoRA** (Liu et al., 2024): A LoRA variant based on weight decomposition
- **HiRA** (Huang et al., 2025): Hadamard high-rank adaptation
- **Insight**: The next frontier in PEFT lies not only in *how* to fine-tune, but in *what* to fine-tune — semantic-level information filtering is a critical direction

## Rating

⭐⭐⭐⭐⭐ (4.5/5)
The theoretical formulation is elegant, the experiments are comprehensive, cross-modal validation is thorough, and the spurious correlation experiments are convincing. The main drawbacks are increased training overhead and a relatively large number of hyperparameters.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)

<!-- RELATED:END -->
