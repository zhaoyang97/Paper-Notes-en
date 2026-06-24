---
title: >-
  [Paper Note] Latent Guard: A Safety Framework for Text-to-Image Generation
description: >-
  [ECCV 2024][Image Generation][text-to-image safety] This paper proposes the Latent Guard framework, which learns a latent space on top of the text encoder of T2I models. Through contrastive learning, it maps blacklist concepts and input prompts containing these concepts to nearby locations, achieving highly efficient unsafe prompt detection (ID Explicit AUC 0.985) and allowing flexible updates of the blacklist during test time without retraining.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "text-to-image safety"
  - "blacklist"
  - "contrastive learning"
  - "latent space"
  - "adversarial robustness"
date: 2026-05-08
content_hash: 7c258692a8eda67c
---

# Latent Guard: A Safety Framework for Text-to-Image Generation

**Conference**: ECCV 2024  
**arXiv**: [2404.08031](https://arxiv.org/abs/2404.08031)  
**Code**: [Project Page](https://latentguard.github.io/)  
**Area**: T2I Safety / Content Moderation  
**Keywords**: text-to-image safety, blacklist, contrastive learning, latent space, adversarial robustness

## TL;DR

This paper proposes the Latent Guard framework, which learns a latent space on top of the text encoder of T2I models. Through contrastive learning, it maps blacklist concepts and input prompts containing these concepts to nearby locations, achieving highly efficient unsafe prompt detection (ID Explicit AUC 0.985) and allowing flexible updates of the blacklist during test time without retraining.

## Background & Motivation

**Background**: T2I models (e.g., DALL·E 3, Stable Diffusion) can generate high-quality images but can also be misused to generate unsafe content such as deepfakes, violence, and discrimination. Existing safety measures include text blacklists, LLM-based moderation, and NSFW image classifiers.

**Limitations of Prior Work**: (1) **Text blacklists** are easily bypassed by synonymous paraphrasing or adversarial attacks; (2) **LLM moderation** is computationally expensive and susceptible to adversarial jailbreaks; (3) **Image classifiers** require generating images first before classification, wasting computational resources; (4) **Concept erasing** methods require expensive fine-tuning and cannot update blacklists flexibly.

**Key Challenge**: There is a need for a safety framework that can detect paraphrases/adversarial attacks, support flexible blacklist updates, and remain computationally efficient.

**Goal**: Efficiently detect the presence of blacklisted concepts within the latent space of T2I text encoders, achieving flexible and robust safety guardrails.

**Key Insight**: Instead of performing binary safe/unsafe classification, the task is framed as detecting whether a prompt contains specific blacklisted concepts by measuring the distance between concepts and prompts in the latent space.

**Core Idea**: Train an Embedding Mapping Layer via contrastive learning to align the embeddings of prompts containing a blacklisted concept with the embedding of the concept itself in the latent space, enabling detection through simple cosine distance computation during inference.

## Method

### Overall Architecture

Define blacklist $\mathcal{C}$ → Generate unsafe/safe prompt pairs using an LLM → Extract features using a CLIP text encoder → Map to the latent space via the Embedding Mapping Layer (cross-attention + MLP) → Train with contrastive learning → Check safety during inference by calculating the cosine distance between concepts and the prompt.

### Key Designs

1. **LLM-driven Training Data Generation**

    - Starting from a blacklisted concept $c$ (e.g., "murder"), use an LLM to generate an unsafe prompt $u_c$ (e.g., "a man gets murdered").
    - The LLM replaces the concept in the unsafe prompt with a harmless word to generate a safe prompt $s_c$ (e.g., "a man gets kissed").
    - Triplet: {concept $c$, unsafe prompt $u_c$, safe prompt $s_c$}.
    - **Design Motivation**: Safe and unsafe prompts share similar structures but differ in concepts, forcing the model to precisely locate concept-related tokens.

2. **Embedding Mapping Layer**

    - The concept embedding $z_c$ serves as the query, and the prompt embedding $z_p$ serves as the key/value for multi-head cross-attention.
    - $h_p = \text{MLP}_p({}^1h_p \| \cdots \| {}^I h_p)$, concept embedding $h_c = \text{MLP}_c(z_c)$.
    - The attention matrix $A \in \mathbb{R}^{C \times P}$ automatically learns which prompt tokens relate to the concept.
    - **Design Motivation**: Cross-attention automatically weights tokens in the prompt that are related to the concept (e.g., "murdered") while filtering out irrelevant tokens (e.g., "a man").

3. **Contrastive Learning Training Strategy**

    - Anchor $a$: Concept embedding $h_c^b$.
    - Positive sample $p$: Corresponding unsafe prompt embedding $h_{u_c}^b$.
    - Negative sample $n$: Other unsafe prompts in the batch $h_{u_c}^{\bar{b}}$ + corresponding safe prompt $h_{s_c}^b$ + other safe prompts $h_{s_c}^{\bar{b}}$.
    - $\mathcal{L}_{\text{cont}} = \sum_b \mathcal{L}_{\text{supcon}}(h_c^b, h_{u_c}^b, h_{u_c}^{\bar{b}} \| h_{s_c}^b \| h_{s_c}^{\bar{b}})$.
    - **Design Motivation**: Using safe prompts as negative samples helps the model distinguish concept presence/absence, and other in-batch concepts prevent confusion.

### Loss & Training

Only the Embedding Mapping Layer is trained, while the CLIP encoder is frozen. Trained using AdamW, lr=1e-3, weight decay=1e-2, batch size=64, converging within 1000 iterations. Training takes only about **30 minutes** on a single 3090 GPU. Concept embeddings can be pre-computed and cached during inference.

## Key Experimental Results

### Main Results

Classification accuracy on the CoPro dataset (723 concepts, 226K prompts):

| Method | ID Explicit↑ | ID Synonym↑ | ID Adversarial↑ | OOD Explicit↑ | OOD Synonym↑ | OOD Adversarial↑ |
|------|-------------|-------------|-----------------|---------------|-------------|-----------------|
| Text Blacklist | 0.805 | 0.549 | 0.587 | 0.895 | 0.482 | 0.494 |
| CLIPScore | 0.628 | 0.557 | 0.504 | 0.672 | 0.572 | 0.533 |
| BERTScore | 0.632 | 0.549 | 0.509 | 0.739 | 0.594 | 0.512 |
| LLM | 0.747 | 0.764 | **0.867** | 0.746 | 0.757 | **0.862** |
| **Latent Guard** | **0.868** | **0.828** | 0.829 | **0.867** | **0.824** | 0.819 |

AUC comparison (threshold-based methods):

| Method | ID Explicit↑ | ID Synonym↑ | ID Adversarial↑ | OOD Explicit↑ | OOD Synonym↑ | OOD Adversarial↑ |
|------|-------------|-------------|-----------------|---------------|-------------|-----------------|
| CLIPScore | 0.697 | 0.587 | 0.504 | 0.733 | 0.596 | 0.560 |
| BERTScore | 0.783 | 0.591 | 0.481 | 0.832 | 0.622 | 0.556 |
| **Latent Guard** | **0.985** | **0.914** | **0.908** | **0.944** | **0.913** | **0.915** |

### Ablation Study

| Design Choice | Impact on Performance |
|---------|---------|
| No safe prompt negative samples | Accuracy drops significantly, making it hard to distinguish concept presence/absence |
| No cross-attention | Degenerates to simple projection, accuracy decreases |
| Embedding dimension $d$ variation | Moderate dimension is optimal |

### Key Findings

- The ID Explicit AUC reaches **0.985**, vastly outperforming CLIPScore (0.697) and BERTScore (0.783).
- Maintains an accuracy of 0.828 in synonym scenarios (vs. 0.549 for Text Blacklist), proving the robustness of latent space representation.
- Achieves an AUC of 0.908 in adversarial attack scenarios, significantly superior to CLIP/BERT (~0.5), demonstrating defensive capability against encoder-level attacks.
- High generalization, with OOD (out-of-distribution, unseen during training) concept accuracy close to ID (in-distribution) performance (0.867 vs. 0.868).
- Highly efficient: Training requires only 30 minutes (on a single 3090), and inference does not require image generation.

## Highlights & Insights

- Framing the task as **concept detection rather than safety classification** allows the blacklist to be flexibly updated during testing without retraining.
- The design of contrastive learning combined with safe prompts as negative samples precisely decouples concepts from sentence context.
- High inference efficiency with 30-minute training and pre-computable concept embeddings, making it highly suitable for industrial deployment.
- Demonstrates strong robustness against encoder-level adversarial attacks (e.g., SneakyPrompt).

## Limitations & Future Work

- In adversarial scenarios, accuracy (0.829) is lower than the LLM-based method (0.867), since LLMs possess stronger semantic understanding capabilities.
- Dependent on the CLIP text encoder; switching to other T2I encoders requires retraining.
- The concept blacklist must be manually defined or generated with LLM assistance, limiting its coverage.
- Only handles text-side safety and cannot detect unsafe content generated through image post-processing.

## Related Work & Insights

- **vs Text Blacklist**: Blacklists only perform string matching, whereas Latent Guard detects within the latent space, making it robust to paraphrasing.
- **vs LLM Moderation**: LLMs are computationally expensive and cannot be flexibly updated, whereas Latent Guard is lightweight and allows dynamic blacklist adjustments.
- **vs Safe Latent Diffusion**: SLD manipulates the diffusion process but still requires generation, whereas Latent Guard blocks at the text stage directly.
- Insight: The application of contrastive learning in T2I safety moderation deserves further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of concept detection, latent space representation, and contrastive learning is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Self-constructed CoPro dataset, 6 test scenarios, 4 baselines, and dual evaluation metrics (AUC and accuracy).
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and well-motivated methodology.
- Value: ⭐⭐⭐⭐ Directly applicable for real-world T2I safety deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[ICML 2026\] ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models](../../ICML2026/image_generation/forceforget_reinforcement_concept_removal_for_enhancing_safety_in_text-to-image_.md)
- [\[NeurIPS 2025\] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)
- [\[ICLR 2026\] Safety-Guided Flow (SGF): A Unified Framework for Negative Guidance in Safe Generation](../../ICLR2026/image_generation/safety-guided_flow_sgf_a_unified_framework_for_negative_guidance_in_safe_generat.md)

</div>

<!-- RELATED:END -->
