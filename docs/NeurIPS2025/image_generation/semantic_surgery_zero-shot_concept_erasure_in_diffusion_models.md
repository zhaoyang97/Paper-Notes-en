---
title: >-
  [Paper Note] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Concept Erasure] This paper proposes Semantic Surgery, a training-free zero-shot inference-time concept erasure framework that calibrates text embeddings via vector subtraction prior to the diffusion process, incorporates Co-Occurrence Encoding for multi-concept erasure, and employs a visual feedback loop to address latent concept persistence (LCP). The method comprehensively outperforms state-of-the-art approaches across object, NSFW, style, and celebrity erasure tasks.
tags:
  - NeurIPS 2025
  - Image Generation
  - Concept Erasure
  - Diffusion Models
  - Text Embedding Manipulation
  - Zero-Shot
  - Inference-Time Method
  - Safe Generation
date: 2026-05-08
content_hash: f7e341eb8310f6d4
---

# Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.22851](https://arxiv.org/abs/2510.22851)
**Authors**: Lexiang Xiong, Chengyu Liu, Jingwen Ye, Yan Liu, Yuecong Xu (NUS, Sichuan University)
**Code**: [GitHub](https://github.com/Lexiang-Xiong/Semantic-Surgery)
**Area**: Image Generation
**Keywords**: Concept Erasure, Diffusion Models, Text Embedding Manipulation, Zero-Shot, Inference-Time Method, Safe Generation

## TL;DR

This paper proposes Semantic Surgery, a training-free zero-shot inference-time concept erasure framework that calibrates text embeddings via vector subtraction prior to the diffusion process, incorporates Co-Occurrence Encoding for multi-concept erasure, and employs a visual feedback loop to address latent concept persistence (LCP). The method comprehensively outperforms state-of-the-art approaches across object, NSFW, style, and celebrity erasure tasks.

## Background & Motivation

### State of the Field

Text-to-image diffusion models (e.g., Stable Diffusion) pose risks of generating harmful or copyright-infringing content (e.g., explicit material, copyrighted artistic styles), necessitating concept erasure techniques.Existing methods fall into two broad categories: parameter modification methods and inference-time methods.

### Limitations of Prior Work

- **Parameter modification methods** (ESD, UCE, MACE, Receler, etc.): Fine-tune or edit model weights to "forget" concepts, but suffer from catastrophic forgetting that degrades general generation capability; establish static defenses with poor robustness to concept variants (e.g., paraphrased prompts); and exhibit cumulative interference in multi-concept erasure settings.
- **Existing inference-time methods** (SLD, SAFREE, etc.): Operate at the token level or at intermediate diffusion stages, but self-attention mechanisms spread target concept semantics across the entire token sequence, rendering local token-level interventions insufficient; furthermore, they fail to address concept "resurrection" caused by U-Net priors.
- **Key Challenge**: How to simultaneously achieve erasure completeness and locality under a zero-shot/inference-time strategy while maintaining robustness to prompt variants.

### Core Idea

Leveraging the linear structure of language embedding spaces (analogous to word2vec-style analogical relations), the method performs "semantic surgery" on global text embeddings prior to the diffusion process—dynamically estimating the presence intensity of target concepts and executing calibrated vector subtraction to neutralize their influence. This approach is inspired by "activation engineering" in large language models.

## Method

### Overall Architecture

Semantic Surgery consists of three core modules: (A) **Semantic Analysis** — Semantic Biopsy detects concept presence and Co-Occurrence Encoding constructs a unified erasure direction; (B) **Core Surgery** — calibrated vector subtraction produces purified embeddings; (C) **Visual Feedback Loop** — detects and mitigates latent concept persistence.

### 1. Semantic Modeling and Single-Concept Erasure

The method builds on the linear analogy property of the CLIP embedding space (e.g., $\phi(\text{king}) - \phi(\text{man}) \approx \phi(\text{queen}) - \phi(\text{woman})$). A neutral reference embedding is defined as $e_n = \phi("")$; for a target concept $c$, the erasure direction is:

$$\Delta e_{\text{erase}} = e_{\text{erase}} - e_n$$

A binary presence indicator $\rho \in \{0,1\}$ is introduced, and the post-surgery embedding is:

$$e'_{\text{input}} = e_{\text{input}} - \rho \cdot \Delta e_{\text{erase}}$$

When $\rho=1$ (concept present), the subtraction projects the embedding into the concept-free subspace; when $\rho=0$, the embedding remains unchanged.

### 2. Co-Occurrence Encoding (Multi-Concept Erasure)

Naïve multi-concept erasure (subtracting each concept vector sequentially, $\sum_i \rho_{c_i} \Delta e_{c_i}$) over-eliminates semantically overlapping features. For instance, simultaneously erasing "seagull" and "sparrow" would excessively suppress shared bird-related attributes. The proposed solution leverages CLIP's contextual modeling capability by concatenating all active concepts into a composite prompt, allowing the CLIP encoder to resolve semantic overlap through phrase-level interaction automatically:

$$\Delta e_{\text{co}} = \phi(p_{\text{co}}) - e_n, \quad p_{\text{co}} = \bigoplus_{i=1}^{n} p_{c_i}$$

The resulting joint surgery operation is $\hat{e}'_{\text{input}} = e_{\text{input}} - \hat{\rho}_{\text{joint}} \cdot \Delta e_{\text{co}}$.

### 3. Semantic Biopsy (Concept Detection)

The core challenge is estimating concept presence intensity $\rho$ solely from input embeddings at inference time. Via the projection decomposition in Theorem 1, concept presence intensity is proportional to the magnitude of the embedding's projection onto the erasure direction. Defining cosine similarity $\alpha_c = \cos(e_{\text{input}}, \Delta e_{\text{erase}})$, empirical analysis reveals that the distributions of $\alpha_c$ for concept-present and concept-absent cases are statistically separable (Assumption 3.1): there exists a threshold $\beta$ such that the two distributions are separated with high probability outside the interval $[\beta-\epsilon, \beta+\epsilon]$. Based on this, a sigmoid calibrator estimates:

$$\hat{\rho}(\alpha_c) = \sigma_{\text{sigmoid}}\left(\frac{\alpha_c - \beta}{\gamma}\right)$$

where $\gamma$ controls classification sharpness. Theoretical guarantees ensure that with high probability, the error between $\hat{\rho}$ and the true binary label does not exceed $\delta_{\text{err}}$.

### 4. Visual Feedback Loop (LCP Mitigation)

This module addresses **Latent Concept Persistence (LCP)**—even after concept semantics are removed from text embeddings, the visual priors of the U-Net may regenerate target content through implicit associations with other concepts (e.g., "road" in the prompt implicitly suggests "trees" via U-Net priors). The mitigation procedure is as follows:

1. Generate an image using the initially surgically modified embedding $\hat{e}'_s$.
2. Apply a visual detector $\mathcal{D}$ to check whether the target concept is present in the generated image.
3. If concept "resurrection" is detected ($\hat{\rho}^{(k)}_{\text{im}} \geq \tau_{\text{vis}}$), add the visually detected concept to the erasure set.
4. Construct an augmented erasure direction $\Delta e^*_{\text{co}}$ and re-execute surgery with a stronger $\hat{\rho}^*_{\text{joint}}$.

Theorem 3 theoretically proves that the secondary surgery effectively reduces LCP risk, and the additional inference overhead is acceptable in practice for safety-critical evaluation.

## Key Experimental Results

### Experiment 1: Object Erasure (CIFAR-10, 10 Classes)

Evaluated using the OWL-ViT independent detector. $\text{Acc}_E$: erasure rate on simple prompts (↓ better); $\text{Acc}_R$: erasure rate on paraphrased prompts (↓ better); $\text{Acc}_L$: retention rate for non-target concepts (↑ better); H: harmonic mean (↑ better).

| Method | $\text{Acc}_E$↓ | $\text{Acc}_R$↓ | $\text{Acc}_L$↑ | H↑ |
|------|:---:|:---:|:---:|:---:|
| SD v1.4 | 99.10 | 87.20 | 87.33 | - |
| ESD-x | 22.20 | 63.20 | 85.49 | 56.03 |
| ESD-u | 12.50 | 39.40 | 81.87 | 73.72 |
| AC | 3.30 | 47.60 | 85.53 | 70.00 |
| UCE | 2.30 | 28.20 | 85.50 | 81.90 |
| Receler | 2.50 | 10.00 | 81.58 | 88.74 |
| MACE | 0.40 | 13.80 | 79.09 | 87.13 |
| **Ours** | **1.50** | **2.00** | **85.56** | **93.58** |

The H-score reaches 93.58 (+4.84 vs. Receler). The robustness metric $\text{Acc}_R$ is only 2.00—one-fifth of Receler and one-seventh of MACE—reflecting the natural resilience of global embedding manipulation to prompt variants. Optimal locality (85.56) is also maintained.

### Experiment 2: Explicit Content Removal + Style Erasure + Adversarial Robustness

**NSFW Removal (I2P, 4,703 prompts)**: Erasing four concepts — "nude/naked/sexual/erotic."

| Method | Type | Total Detections↓ | FID↓ | CLIP↑ |
|------|------|:---:|:---:|:---:|
| SD v1.4 | Original | 751 | 14.04 | 31.34 |
| ESD-u | Param. Mod. | 55 | 15.1 | 30.21 |
| MACE | Param. Mod. | 123 | 13.42 | 29.41 |
| SAFREE | Inference-time | 82 | - | - |
| **Ours** | **Inference-time** | **1** | **12.2** | **30.75** |

NSFW instances are reduced from 751 to just 1, representing a >98% reduction over SAFREE (82); FID (12.2) even surpasses that of the original model.

**Artistic Style Erasure (100 Artists)**: $H_a = \text{CLIP}_s - \text{CLIP}_e$, higher is better.

| Method | CLIPe↓ | CLIPs↑ | $H_a$↑ | FID-30K↓ | CLIP-30K↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| UCE | 21.35 | 26.32 | 4.97 | 77.72 | 19.17 |
| ESD-u | 19.66 | 19.55 | -0.11 | 17.07 | 27.76 |
| MACE | 22.59 | 28.58 | 5.99 | 12.71 | 29.51 |
| **Ours** | **20.75** | **28.84** | **8.09** | **14.04** | **31.34** |

$H_a$ = 8.09 (+2.1 vs. MACE); FID/CLIP scores are identical to the original SD v1.4, indicating zero degradation in general generation quality.

**Adversarial Robustness**:

| Attack Type | Method | ASR↓ |
|---------|------|:---:|
| Black-box (RAB, 380 prompts) | SLD | 78.68% |
| Black-box | SAFREE | 55.80% |
| Black-box | MACE | 3.95% |
| **Black-box** | **Ours** | **1.05%** |
| **White-box (UnlearnDiffAtk)** | **Ours** | **0.0%** |

Black-box ASR is only 1.05% ($p=0.0089$ vs. MACE); white-box ASR is 0%. Semantic Biopsy additionally functions as a threat detection system.

**Multi-Concept Celebrity Erasure (100 Celebrities)**: $H_c$ reaches 0.965, significantly outperforming MACE (0.892), UCE (0.554), and Receler (0.441), while FID/CLIP remain consistent with the original model.

## Highlights & Insights

- **Training-Free Global Embedding Manipulation**: Unlike all parameter modification methods, Semantic Surgery directly operates on text embeddings prior to diffusion, leaving the original model completely intact, while surpassing retraining-based SOTA methods across multiple tasks.
- **Co-Occurrence Encoding**: Leverages CLIP's contextual modeling capability to resolve semantic overlap in multi-concept erasure, avoiding the over-suppression inherent in naïve vector subtraction, with superiority demonstrated both theoretically and empirically.
- **LCP Visual Feedback Mechanism**: For the first time identifies and addresses the latent concept persistence problem, mitigating U-Net-prior-induced concept "resurrection" through a visual detection and secondary surgery feedback loop.
- **Natural Adversarial Robustness**: Semantic Biopsy detects concept presence based on global semantic similarity, providing natural immunity to prompt paraphrasing and adversarial attacks (white-box ASR = 0%), and can additionally serve as a threat detection system.
- **Zero Degradation in General Quality**: FID/CLIP scores in style and celebrity erasure tasks are fully consistent with the original model, fundamentally resolving the quality degradation problem that plagues parameter modification methods.

## Limitations & Future Work

- **Reliance on Linearity of CLIP Embedding Space**: The theoretical foundation assumes linear analogical relations in the embedding space, which may fail for highly nonlinear or complex semantic relationships.
- **Visual Feedback Increases Inference Overhead**: The LCP loop requires an additional round of image generation and visual detection; although restricted to safety-critical tasks, this introduces additional latency.
- **Threshold Parameters Require Task-Specific Tuning**: The decision threshold $\beta$ still requires empirical setting per task; $\gamma=0.02$ and $\tau=0.5$ are fixed across tasks but may not be universally optimal.
- **Validation Limited to SD v1.4**: Applicability to more recent architectures such as SDXL, SD3, and Flux has not been demonstrated.
- **Single-Concept Simple-Prompt Erasure Slightly Inferior to MACE** ($\text{Acc}_E$: 1.50 vs. 0.40), though robustness is substantially superior.
- **Concept Embedding Quality Depends on the Text Encoder**: If CLIP encodes certain concepts poorly, the effectiveness of vector subtraction is inherently limited.

## Related Work & Insights

- **ESD (ICCV'23)**: Fine-tunes the U-Net; ESD-u/ESD-x exhibit poor erasure–retention trade-offs (H = 73.72/56.03 vs. 93.58) and weak robustness.
- **UCE (WACV'24)**: Unified concept editing; general quality degrades severely in multi-concept scenarios (style erasure FID = 77.72).
- **MACE (NeurIPS'24)**: The strongest parameter modification method for multi-concept erasure, but robustness is far inferior to the proposed method ($\text{Acc}_R$: 13.80 vs. 2.00), with quality degradation across multiple tasks.
- **Receler (ECCV'24)**: A lightweight parameter modification method with strong object erasure performance but lower locality ($\text{Acc}_L$: 81.58 vs. 85.56).
- **SLD (ICCV'23)**: A safety-guidance-based inference-time method with insufficient erasure precision (I2P: 149 vs. 1) and extremely poor adversarial robustness (ASR = 78.68%).
- **SAFREE (NeurIPS'24)**: A token-level embedding projection inference-time method that outperforms SLD but falls far short of the proposed method (I2P: 82 vs. 1), as local token manipulation cannot prevent self-attention from propagating concept semantics.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to propose zero-shot concept erasure at the global text embedding level; Co-Occurrence Encoding and the LCP feedback mechanism are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers five evaluation dimensions (object, NSFW, style, celebrity, adversarial attacks) with comprehensive comparison against both parameter modification and inference-time methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formalization is rigorous and the integration of theory and practice is strong, though the dense notation increases reading burden.
- **Value**: ⭐⭐⭐⭐⭐ — Achieves comprehensive SOTA across all major tasks without retraining; highly practical and immediately deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](../../ICCV2025/image_generation/anyportal_zero-shot_consistent_video_background_replacement.md)

</div>

<!-- RELATED:END -->
