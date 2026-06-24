---
title: >-
  [Paper Note] LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation
description: >-
  [NEURIPS2025][Dialogue Systems][LLM safety] This paper proposes LatentGuard, a three-stage framework that combines behavior-level alignment fine-tuning, structured VAE-supervised latent space modeling, and latent-space dimensional manipulation to achieve interpretable and controllable regulation of LLM refusal behavior — robustly defending against adversarial attacks while preserving responsiveness to benign queries.
tags:
  - "NEURIPS2025"
  - "Dialogue Systems"
  - "LLM safety"
  - "Latent Space Steering"
  - "VAE"
  - "Refusal Alignment"
  - "Jailbreak Defense"
date: 2026-05-08
content_hash: 417fa294bd82a5dd
---

# LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation

**Conference**: NEURIPS2025
**arXiv**: [2509.19839](https://arxiv.org/abs/2509.19839)  
**Code**: Not released  
**Area**: Robotics
**Keywords**: LLM safety, Latent Space Steering, VAE, Refusal Alignment, Jailbreak Defense

## TL;DR

This paper proposes LatentGuard, a three-stage framework that combines behavior-level alignment fine-tuning, structured VAE-supervised latent space modeling, and latent-space dimensional manipulation to achieve interpretable and controllable regulation of LLM refusal behavior — robustly defending against adversarial attacks while preserving responsiveness to benign queries.

## Background & Motivation

LLM safety alignment faces two core tensions:

1. **Safety vs. Usability**: Existing alignment methods (SFT, RLHF, Constitutional AI) focus on behavioral-level training and are prone to "over-refusal" — incorrectly classifying benign queries as harmful and refusing to respond, severely degrading user experience.
2. **Robustness vs. Controllability**: Unsupervised latent space methods such as Sparse Autoencoders (SAE) can identify interpretable feature directions but suffer from critical limitations: (a) unsupervised feature discovery fails to capture task-relevant safety semantics; (b) sparsity constraints limit expressiveness for complex adversarial patterns; (c) post-hoc interpretation requires extensive analysis and is difficult to apply for real-time control. Wu et al. (2025) have shown that simple baselines outperform SAE on steering and concept detection.

Reasoning-enhanced fine-tuning approaches improve refusal transparency but often cause over-refusal of benign queries due to "hallucination risk," and lack fine-grained safety-utility calibration.

## Core Problem

How to establish a **supervised, interpretable, and manipulable** safety control mechanism within LLM representation space, enabling the model to:

- Precisely identify and refuse diverse adversarial attacks (including adaptive attacks, DRA, PAP, and other advanced attacks)
- Simultaneously eliminate false refusals of benign queries
- Provide interpretable grounds for refusal decisions

## Method

LatentGuard adopts a three-stage pipeline:

### Stage 1: Reasoning-Enhanced SFT

Parameter-efficient fine-tuning of Qwen3-8B using LoRA:

- **Adversarial data**: SorryBench augmented with multiple attack techniques (Adaptive, DRA, PAP, etc.), covering a broad spectrum of attack strategies
- **Benign data**: 10k_prompts_ranked dataset with high-quality instruction-following samples
- **Response generation**: Gemini 2.5 Pro is used to generate refusal/acceptance responses with step-by-step reasoning for each prompt
- **Objective**: Establish behavioral safety priors so the model learns transparent refusal with reasoning

Training loss is standard cross-entropy: $\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log P(y_i | x_i; \theta)$

### Stage 2: Structured VAE Latent Space Supervision

MLP residual activations are extracted from the intermediate Transformer layer (layer 24) of the fine-tuned model to train a structured VAE:

**Latent space design** — the latent representation $z \in \mathbb{R}^{C+R}$ is disentangled into two components:

- **Semantic dimensions** $z_c \in \mathbb{R}^{52}$: learned via multi-label supervision, encoding 30 prompt categories (violence, terrorism, political sensitivity, etc.) + 21 attack strategies + 1 benign flag
- **Residual dimensions** $z_r \in \mathbb{R}^{2000}$: capture contextual information to ensure reconstruction quality

**Multi-objective training loss**:

$$\mathcal{L}_{\text{VAE}} = \alpha \cdot \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{BCE}} + \gamma \cdot \mathcal{L}_{\text{KL}}$$

where $\mathcal{L}_{\text{recon}}$ ensures representational fidelity, $\mathcal{L}_{\text{BCE}}$ aligns semantic dimensions with multi-label supervision, and $\mathcal{L}_{\text{KL}}$ regularizes the latent distribution (with linear warm-up over 10k steps to prevent posterior collapse). Hyperparameters: $\alpha=1.0, \beta=0.2, \gamma=0.2$.

### Stage 3: Latent Space Manipulation for Behavior Control

At inference time, targeted interventions are applied to the semantic dimensions:

- **Safety-enhanced mode** (refusing attacks): amplify attack-relevant dimensions $z'_{c,\text{attack}} = 2.0 \cdot \alpha$, suppress the benign flag $z'_{c,\text{benign}} = -2.0 \cdot \alpha$
- **Normal-preservation mode** (accepting benign queries): amplify the benign flag, suppress attack features

The manipulated latent representation is reconstructed back to the hidden state via the Decoder, replacing the original activations before being passed to subsequent Transformer layers, thereby enabling sequence-level behavior steering.

## Key Experimental Results

Validated on Qwen3-8B and Mistral-7B; evaluation metrics include refusal rate, safety score (judged by Claude), and fluency score.

**Qwen3-8B core results** (after SFT → after VAE intervention):

| Scenario | Refusal Rate | Safety Score | Fluency |
|----------|-------------|-------------|---------|
| Benign queries | 41.4% → **0.0%** | 0.95 → 1.0 | 0.79 → 0.97 |
| AdvBench | 98.4% → **100%** | 0.98 → 1.0 | 0.79 → 0.83 |
| + Adaptive attack | 94.4% → **97.7%** | 1.0 → 1.0 | 0.85 → 0.87 |
| + PAP attack | 79.0% → **92.2%** | 0.97 → 0.98 | 0.85 → 0.94 |
| + DRA attack | 91.4% → **99.2%** | 0.95 → 0.99 | 0.76 → 0.76 |

**Key Findings**:

- **Over-refusal elimination**: False refusal rate on benign queries drops from 41.4% to 0%, while fluency improves from 0.79 to 0.97
- **Substantial gains against advanced attacks**: PAP attack refusal rate improves from 79% to 92.2%; DRA from 91.4% to 99.2%
- **Optimal intervention layers are layers 13–23**, with moderate intervention strength $\alpha=2.5$ yielding the best results
- **Cross-architecture generalization**: Consistent effectiveness demonstrated on Mistral-7B

## Highlights & Insights

1. **A new approach to the safety-utility trade-off**: Through a supervised disentangled latent space, the method simultaneously achieves stronger attack refusal and elimination of benign over-refusal — the two objectives are no longer in conflict.
2. **Structured VAE is better suited than SAE for safety tasks**: Supervised semantic dimensions directly encode safety-relevant concepts, avoiding the semantic alignment issues inherent to unsupervised feature discovery.
3. **Fine-grained controllability**: The continuous latent space supports smooth interpolation; the intervention strength $\alpha$ allows tuning of the safety-utility trade-off curve.
4. **Cross-architecture generalization**: Effective across both Qwen3-8B and Mistral-7B model families.

## Limitations & Future Work

1. **Upstream classifier dependency**: Attack type and method labels originate from a commercial firewall product; classification errors propagate into the latent space.
2. **MLP activations only**: Extending the approach to other components such as attention mechanisms requires new supervision designs.
3. **Limited model scale**: Validation is confined to 7B–8B models; generalization to smaller or quantized models remains unknown.
4. **Real-time deployment efficiency**: The additional overhead of VAE encode–manipulate–decode in real-time settings has not been verified.
5. **Adaptability to evolving attacks**: The current label taxonomy is fixed at 30 categories + 21 attack types; generalization to unseen attack types is unclear.

## Related Work & Insights

| Method | Control Level | Supervision | Interpretability | Over-Refusal |
|--------|--------------|-------------|-----------------|-------------|
| SFT/RLHF | Behavioral | Human preference | Low | Severe |
| SAE steering | Representation | Unsupervised | Medium (post-hoc) | Unresolved |
| Reasoning-enhanced SFT | Behavioral | Reasoning templates | Medium | Present |
| **LatentGuard** | **Behavioral + Representation** | **Multi-label supervision** | **High (explicit dimensional semantics)** | **Eliminated** |

The core advantage of LatentGuard lies in unifying behavioral alignment with representational control through a supervised structured latent space that bridges the two.

**Insights and Connections**:

- **Supervised vs. unsupervised latent spaces**: This work demonstrates that, for safety tasks, supervised disentanglement is more reliable than unsupervised methods such as SAE — a principle transferable to other scenarios requiring precise control (e.g., bias mitigation, style control).
- **A general paradigm for latent space intervention**: The three-stage design of "fine-tuning → latent space modeling → inference-time intervention" is transferable to other alignment tasks.
- **Scalability of safety labels**: The current approach relies on a fixed label taxonomy; future work could explore open-set detection or adaptive label expansion.

## Rating

- Novelty: ⭐⭐⭐⭐ — Supervised structured VAE for LLM safety control is a novel combination
- Experimental Thoroughness: ⭐⭐⭐ — Two-model validation is adequate, but larger models and broader attack types are lacking
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed three-stage pipeline description
- Value: ⭐⭐⭐⭐ — Practical approach to the safety-utility balance; over-refusal elimination is particularly compelling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](../../ACL2025/dialogue/uniconv_retrieval_response_gen.md)
- [\[ACL 2026\] LOCKET: Robust Feature-Locking Technique for Language Models](../../ACL2026/dialogue/locket_robust_feature-locking_technique_for_language_models.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](../../ICLR2026/dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2025\] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](../../ACL2025/dialogue/enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)
- [\[ECCV 2024\] BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](../../ECCV2024/dialogue/bi-mdrg_bridging_image_history_in_multimodal_dialogue_response_generation.md)

</div>

<!-- RELATED:END -->
