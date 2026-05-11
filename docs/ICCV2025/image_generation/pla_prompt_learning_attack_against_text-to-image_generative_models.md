---
title: >-
  [Paper Note] PLA: Prompt Learning Attack against Text-to-Image Generative Models
description: >-
  [ICCV 2025][Image Generation][Adversarial Attack] This paper proposes PLA (Prompt Learning Attack), a gradient-driven adversarial attack framework targeting black-box T2I models. By leveraging sensitive knowledge encodin…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Adversarial Attack"
  - "T2I Safety"
  - "Black-box Attack"
  - "Prompt Learning"
  - "NSFW Content Detection"
date: 2026-05-08
content_hash: 1df208ba1533d738
---

# PLA: Prompt Learning Attack against Text-to-Image Generative Models

**Conference**: ICCV 2025
**arXiv**: [2508.03696](https://arxiv.org/abs/2508.03696)
**Code**: None
**Area**: Diffusion Models / AI Security
**Keywords**: Adversarial Attack, T2I Safety, Black-box Attack, Prompt Learning, NSFW Content Detection

## TL;DR

This paper proposes PLA (Prompt Learning Attack), a gradient-driven adversarial attack framework targeting black-box T2I models. By leveraging sensitive knowledge encoding and multimodal similarity losses, PLA learns adversarial prompts that bypass both prompt filters and post-hoc safety checkers, achieving an average ASR-4 exceeding 90%, substantially outperforming existing methods.

## Background & Motivation

**Background**: T2I models (e.g., Stable Diffusion, DALL·E 3) have been widely adopted for artistic creation and content generation, yet they face the risk of misuse for generating NSFW (Not-Safe-For-Work) content. To mitigate this, developers deploy two categories of safety mechanisms: prompt filters (blocking inputs via sensitive keyword lists) and post-hoc safety checkers (detecting inappropriate content in generated images).

**Limitations of Prior Work**: Existing black-box attack methods (e.g., SneakyPrompt) predominantly rely on word-substitution strategies, seeking replacement tokens within a constrained search space to evade prompt filters. The limited search space, however, leads to suboptimal attack success rates. Gradient-driven optimization offers stronger capability, but cannot be directly applied in black-box settings where internal model parameters are inaccessible.

**Key Challenge**: Black-box T2I models not only conceal their internal architectures and parameters, but their safety mechanisms also interrupt the forward pass upon detecting NSFW content, returning a blank (black) image. This renders conventional gradient estimation methods based on model outputs ineffective, as black images yield zero gradients.

**Goal**: (a) How to enable effective gradient-driven adversarial prompt learning under a black-box setting? (b) How to address the gradient vanishing problem caused by safety mechanisms returning black images?

**Key Insight**: The paper exploits the sensitive information embedded in target prompts as semantic guidance, uses an auxiliary model without safety mechanisms to generate target images, and constructs a differentiable training objective via multimodal (text–image and image–image) similarity.

**Core Idea**: PLA retains the semantic intent of target prompts through sensitive knowledge encoding, and combines multimodal CLIP-based losses with an improved zeroth-order gradient optimization strategy to train a prompt encoder that generates adversarial prompts capable of bypassing dual safety mechanisms in black-box settings.

## Method

### Overall Architecture

PLA consists of three core components: (1) **Sensitive Knowledge Guided Encoding (SKE)**, which encodes target prompts into learnable embeddings containing sensitive semantics; (2) an **attack pipeline** that uses a pre-trained language model to generate adversarial prompts and attempts to bypass the safety mechanisms of black-box T2I models; and (3) a **Multimodal Loss**, which guides gradient optimization via text–image and image–image similarity.

Given a target prompt $p_{tar}$ containing sensitive words, PLA outputs an adversarial prompt $p_{adv}$ that contains no sensitive words yet induces the generation of NSFW images semantically consistent with $p_{tar}$.

### Key Designs

1. **Sensitive Knowledge Encoding (SKE) Module**:

    - *Function*: Extracts sensitive semantic information from the target prompt to produce a sensitive embedding $e_{sen}$.
    - *Mechanism*: A pre-trained text encoder $\mathcal{T}_\theta$ encodes $p_{tar}$ into a text embedding $e_{tar} \in \mathbb{R}^d$, which is then projected via a two-layer mapping (low-dimensional projection $W_l \in \mathbb{R}^{d \times d_l}$ followed by high-dimensional projection $W_h \in \mathbb{R}^{d_l \times d_s}$) to yield $e_{sen} \in \mathbb{R}^{M \times d_s}$.
    - *Design Motivation*: High-dimensional text features preserve the sensitive semantic intent of the target prompt, enabling the adversarial prompt to implicitly carry sensitive information without triggering keyword-based filters.

2. **Prompt Encoder**:

    - *Function*: Fuses the sensitive embedding into the encoding process of a random prompt to produce a learnable embedding $e_{pe}$.
    - *Mechanism*: Given a random prompt $p_{ran}$, the sensitive embedding is injected at layer $l$ of the encoder: $\hat{e}_l = e_l + \omega \cdot e_{sen}$, where $\omega$ controls the degree of fusion. The resulting $e_{pe}$ is concatenated with $p_{tar}$ and fed into a PLM (e.g., BERT or T5) to generate the adversarial prompt: $p_{adv} = \mathcal{PLM}([e_{pe}; p_{tar}])$.
    - *Design Motivation*: Intermediate-layer injection, rather than simple concatenation, enables deep fusion of sensitive information with random text features, enhancing the stealthiness of the adversarial prompt.

3. **Auxiliary Model for Target Image Generation**:

    - *Function*: Uses an auxiliary T2I model without safety mechanisms (e.g., SDv1.4) to generate a target image $I_{tar} = \mathcal{M}_s(p_{tar})$.
    - *Mechanism*: Since the black-box model's safety mechanisms return black images, direct target image acquisition is impossible. The auxiliary model provides image-level supervision signals.
    - *Design Motivation*: Resolves the lack of target image references under the black-box setting, supplying image–image contrastive signals for the multimodal loss.

### Loss & Training

The **multimodal loss** $\mathcal{L}_{MS}$ comprises two components:

- **Text–image similarity loss**: $\mathcal{L}_a = 1 - \cos(\mathcal{T}_{en}(p_{tar}), \mathcal{V}_{en}(I_{gen}))$, measuring the semantic consistency between the target prompt and the generated image using CLIP's text and image encoders.
- **Image–image similarity loss**: $\mathcal{L}_b = 1 - \cos(\mathcal{V}_{en}(I_{tar}), \mathcal{V}_{en}(I_{gen}))$, measuring the consistency between the auxiliary model's target image and the black-box model's generated image.

**Gradient Optimization**: As gradients cannot be directly backpropagated in the black-box setting, an improved zeroth-order optimization (ZOO) is employed. Traditional ZOO estimates gradients via finite differences:

$$g_1(\varsigma) = \frac{\mathcal{L}_{MS}(\varsigma + c \cdot \Delta) - \mathcal{L}_{MS}(\varsigma - c \cdot \Delta)}{2c \cdot \Delta}$$

However, when both perturbations produce black images, the gradient is zero. The proposed improvement introduces historical gradient momentum:

$$g_2(\varsigma) = \beta \hat{g}_2 + (1 - \beta) \eta \cdot g_1(\varsigma + \hat{g}_2)$$

so that updates continue along the historical direction when the current gradient vanishes. A **restart** strategy is also proposed: when a black image is encountered at the very first step, Gaussian noise replaces the black image in gradient computation.

## Key Experimental Results

### Main Results

Evaluation is conducted on the I2P dataset using 100 nudity prompts and 30 violence prompts, attacking three black-box models (SDv1.5, SDXLv1.0, SLD) in combination with three post-hoc safety checkers (SC, Q16, MHSC).

| Model | Method | AVG ASR-4 (Nudity) | AVG ASR-1 (Nudity) | AVG ASR-4 (Violence) | AVG ASR-1 (Violence) |
|------|------|---------------------|---------------------|----------------------|----------------------|
| SDv1.5 | MMA-Diffusion | 77.76 | 58.38 | 78.26 | 61.04 |
| SDv1.5 | **PLA-BERT** | **91.45** | **68.69** | **88.62** | **69.51** |
| SDXLv1.0 | MMA-Diffusion | 73.30 | 45.24 | 75.53 | 50.61 |
| SDXLv1.0 | **PLA-BERT** | **90.57** | **71.43** | **86.95** | **66.61** |
| SLD | MMA-Diffusion | 76.48 | 53.00 | 76.45 | 56.95 |
| SLD | **PLA-BERT** | **90.82** | **69.30** | **89.20** | **72.03** |

On online services (Stability.ai and DALL·E 3), PLA-T5 achieves ASR-4 of 69.70% and 51.98% respectively on violence prompts, substantially surpassing all compared methods.

### Ablation Study

| Configuration | ASR-4 (Violence) | ASR-1 (Violence) | ASR-4 (Nudity) | ASR-1 (Nudity) |
|------|-------------------|-------------------|-----------------|-----------------|
| $\mathcal{L}_a + \mathcal{L}_b$ (Full) | 93.34 | 79.62 | 93.41 | 75.60 |
| w/o $\mathcal{L}_a$ | 81.02 | 54.57 | 82.99 | 51.07 |
| w/o $\mathcal{L}_b$ | 79.34 | 47.88 | 74.66 | 44.87 |
| $G_{PLA}$ (full gradient) | 91.69 | 70.23 | 95.37 | 76.20 |
| $G_{ZOO}$ (standard ZOO) | 52.89 | 46.73 | 58.44 | 41.27 |
| $G_{RE}$ (w/o restart) | 70.12 | 58.24 | 78.33 | 53.90 |

### Key Findings

- The image–image similarity loss $\mathcal{L}_b$ contributes more than the text–image loss $\mathcal{L}_a$; its removal causes a more severe drop in ASR, indicating that target images carry richer sensitive information.
- The improved gradient optimization $G_{PLA}$ significantly outperforms standard ZOO $G_{ZOO}$ (approximately 35% gap in ASR-4), validating the effectiveness of historical gradient momentum for addressing gradient vanishing caused by black images.
- The restart strategy is critical for handling gradient vanishing at the first step; its removal reduces ASR-4 by approximately 17–20 percentage points.
- PLA-BERT and PLA-T5 exhibit complementary strengths across different datasets, suggesting that different PLMs have distinct "preferences" for different types of sensitive content.

## Highlights & Insights

- **Elegant multimodal loss design**: Without access to black-box model parameters, the framework constructs effective gradient signals through an auxiliary model combined with CLIP similarity. This paradigm of "bridging black-box access via an auxiliary model" is transferable to other black-box optimization tasks.
- **Practical solution to gradient vanishing**: The gradient vanishing induced by safety mechanisms returning black images is a unique challenge in black-box attacks. The proposed historical gradient momentum and Gaussian noise restart strategy address this effectively, and the underlying idea is generalizable to other zeroth-order optimization scenarios involving vanishing gradients.
- **Systematic safety evaluation**: The evaluation covers three black-box models, three safety checkers, and two online services, providing a comprehensive perspective on T2I safety assessment.

## Limitations & Future Work

- The paper focuses on attack without offering substantive suggestions for improving defenses; it does not discuss how to design more robust safety mechanisms informed by PLA's attack patterns.
- The auxiliary model (SDv1.4) and the black-box models may share similar architectural biases; transferability to T2I models with fundamentally different architectures (e.g., autoregressive models) is not validated.
- Evaluation is limited to nudity and violence; the effectiveness on other sensitive categories (e.g., hate speech, self-harm) is not explored.
- The readability and naturalness of PLM-generated adversarial prompts are not quantitatively assessed.

## Related Work & Insights

- **vs. MMA-Diffusion**: MMA-Diffusion is a white-box attack requiring access to internal model parameters; PLA substantially surpasses its white-box performance under a black-box setting, demonstrating that multimodal similarity losses provide sufficient optimization signal.
- **vs. SneakyPrompt**: SneakyPrompt employs reinforcement learning for word-substitution search, constrained by a limited discrete search space; PLA bypasses this bottleneck through continuous optimization of prompt encoder parameters.
- This work reveals the vulnerability of current T2I safety mechanisms, offering important reference value for red-teaming and defensive security research.

## Rating

- Novelty: ⭐⭐⭐⭐ The black-box gradient attack design is creative, though the core components (CLIP similarity + zeroth-order optimization) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, safety checkers, and online services, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Significant contribution to T2I safety research, though potential misuse risks warrant attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)
- [\[NeurIPS 2025\] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)
- [\[ICCV 2025\] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification](towards_robust_defense_against_customization_via_protective_perturbation_resista.md)
- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)

</div>

<!-- RELATED:END -->
