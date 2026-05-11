---
title: >-
  [Paper Note] Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering
description: >-
  [ICCV 2025][AI Safety][continual learning] This paper proposes QUAD—a continual VQA method that stores only past task questions (without images). Through question replay and attention consistency distillation, QUAD achieves privacy preservation while outperforming methods that store full image–question–answer triplets.
tags:
  - ICCV 2025
  - AI Safety
  - continual learning
  - visual question answering
  - question replay
  - attention distillation
  - privacy preservation
date: 2026-05-08
content_hash: c77aa5ed4b0d1beb
---

# Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering

**Conference**: ICCV 2025
**arXiv**: [2502.04469](https://arxiv.org/abs/2502.04469)
**Code**: [https://github.com/IemProg/QUAD](https://github.com/IemProg/QUAD)
**Area**: AI Safety (Continual Learning)
**Keywords**: continual learning, visual question answering, question replay, attention distillation, privacy preservation

## TL;DR

This paper proposes QUAD—a continual VQA method that stores only past task questions (without images). Through question replay and attention consistency distillation, QUAD achieves privacy preservation while outperforming methods that store full image–question–answer triplets.

## Background & Motivation

Continual learning for VQA presents unique challenges: a model must **simultaneously** maintain stability (retaining prior knowledge) and plasticity (learning new tasks) across both visual and linguistic modalities, while also possessing compositional generalization ability (transferring learned skills to novel objects).

Core tension in existing approaches:
- **Memory replay methods** (ER, VQACL, etc.): store image–question–answer triplets, reducing forgetting but raising serious **privacy concerns** (images may contain identities, license plates, and other sensitive data) and **storage overhead** (up to 5,000 samples per task).
- **Memory-free methods** (EWC, MAS, etc.): store no data, preserving privacy at the cost of poor performance.
- **Key question**: **Is image storage truly necessary? Is storing questions alone sufficient?**

The paper identifies a critical phenomenon—the **Out-of-Answer-Set Problem**: during sequential fine-tuning, a model overfits to the answer space of the current task and begins answering prior-task questions with current-task answers (e.g., after learning a color task, responding "Red" instead of "Two" to a counting question). This closely mirrors the **recency bias** observed in class-incremental learning. Replaying stored questions naturally mitigates this problem.

## Method

### Overall Architecture

QUAD (QUestion-only replay with Attention Distillation) comprises three components: (1) a **question memory buffer** storing only the text of past-task questions; (2) a **question replay mechanism** pairing stored questions with current-task images and using the previous model to generate pseudo-labels; and (3) **attention consistency distillation** enforcing cross-task consistency in attention patterns.

The overall loss function is:

$$\mathcal{L}_{\text{VQACL}} = (1-\lambda)\mathcal{L}_{\text{Plasticity}} + \lambda\mathcal{L}_{\text{Stability}}$$

where $\mathcal{L}_{\text{Stability}} = \mathcal{L}_{\text{QR}} + \mathcal{L}_{\text{ACD}}$ and $\lambda=0.5$.

### Key Designs

1. **Question-only Replay ($\mathcal{L}_{\text{QR}}$)**: For each image $x^t$ from the current task, questions $q^m$ are sampled from the memory buffer and paired with $x^t$. The previous model $\phi^{t-1}$ generates soft pseudo-labels (without argmax), and the current model $\phi^t$ is trained to match these outputs:

$$\mathcal{L}_{\text{QR}} = \mathbb{E}_{x^t \sim \mathcal{T}^t} \mathbb{E}_{q^m \sim \mathcal{M}} \mathcal{L}_{\text{CE}}[\phi^t(x^t, q^m), \phi^{t-1}(x^t, q^m)]$$

   Although $(x^t, q^m)$ pairs may not be semantically aligned, this formulation forces the model to retain answer formats and distributions across diverse question types, thereby preventing the out-of-answer-set problem.

   **Question selection strategy**: Rather than random pairing, questions are preferentially selected based on object category relevance to the current visual sub-task. For example, when the current task involves counting cars, historical questions such as "What's the color of the car?" are prioritized to ensure semantic coherence.

2. **Attention Consistency Distillation ($\mathcal{L}_{\text{ACD}}$)**: Question replay ensures output-level consistency but does not constrain internal representations—self-attention patterns gradually drift during sequential fine-tuning. To address this, cross-task consistency constraints are applied to the attention distributions of all attention heads:

$$\mathcal{L}_{\text{ACD}} = \mathbb{E}_{x^t,q^m} \mathbb{E}_{k \sim \mathcal{K}_\phi} \mathcal{L}_{\text{CE}}[A_k^t(x^t,q^m), A_k^{t-1}(x^t,q^m)]$$

   **Key distinction from an L1 baseline**: ACD computes cross-entropy over softmax-normalized attention distributions, yielding gradients $\partial\mathcal{L}/\partial A^t = -A^{t-1}/A^t + 1$, which impose stronger constraints on highly attended regions (large $A^{t-1}$) while allowing flexibility in low-attention regions. By contrast, L1 operates on raw query–key products, uniformly penalizing all deviations and thus being overly rigid.

3. **Multimodal attention preservation**: The VQA backbone encodes image and text tokens in a unified Transformer sequence, where self-attention naturally captures intra-modal (text–text, image–image) and cross-modal (text–image) dependencies. ACD simultaneously stabilizes all three relationship types, ensuring that visual–linguistic associations are not disrupted by new task learning.

### Loss & Training

- VQA backbone: T5 (12-layer encoder + 12-layer decoder, 12 attention heads)
- VQAv2: 5,000 questions stored per task; NExT-QA: 500 questions per task
- 3 training epochs per task, batch size 80, Adam optimizer with lr=1e-4
- No visual data or question prototypes are stored (unlike the VQACL baseline which requires prototypes)

## Key Experimental Results

### Main Results

| Method | Memory Type | VQAv2 AP↑ | VQAv2 Forget↓ | VQAv2 Novel AP↑ | NExT-QA AP↑ | NExT-QA Forget↓ |
|------|----------|-----------|---------------|-----------------|-------------|-----------------|
| Vanilla | None | 14.92 | 30.80 | 11.79 | 12.68 | 25.94 |
| EWC | None | 15.77 | 30.62 | 12.83 | 13.01 | 24.06 |
| ER | Questions + Images | 36.99 | 5.99 | 33.78 | 30.55 | 4.91 |
| VQACL | Questions + Images | 37.46 | 6.96 | 35.40 | 30.86 | 4.12 |
| **QUAD** | **Questions only** | **39.25** | **4.91** | **40.00** | **31.70** | **2.91** |

*QUAD surpasses all methods that store both images and questions, using questions alone.*

### Ablation Study

| $\mathcal{L}_{\text{QR}}$ | $\mathcal{L}_{\text{ACD}}$ | VQAv2 AP↑ | VQAv2 Forget↓ | NExT-QA AP↑ | NExT-QA Forget↓ |
|:-:|:-:|-----------|---------------|-------------|-----------------|
| ✓ | ✗ | 30.72 | 13.74 | 29.04 | 4.58 |
| ✗ | ✓ | 13.34 | 32.08 | 13.24 | 24.56 |
| **✓** | **✓** | **39.25** | **4.91** | **31.70** | **2.91** |

*Both components are indispensable: QR alone is effective but insufficient; ACD alone constrains internal representations but cannot prevent the out-of-answer-set problem.*

| Distillation Method | VQAv2 AP↑ | VQAv2 Forget↓ | NExT-QA AP↑ | NExT-QA Forget↓ |
|----------|-----------|---------------|-------------|-----------------|
| QR + Attn-dist (L1) | 34.56 | 7.91 | 30.14 | 5.78 |
| QR + Asym-Attn | 38.15 | 5.57 | 31.18 | 4.13 |
| **QUAD (CE)** | **39.25** | **4.91** | **31.70** | **2.91** |

*Cross-entropy outperforms both L1 and asymmetric ReLU+L1 variants.*

### Key Findings

- Object-matched question selection significantly outperforms random pairing, with the gap widening as memory size increases.
- QUAD remains effective on BLIP-2 (AP=50.27 vs. VQACL=49.80).
- Entropy-difference analysis shows that QUAD exhibits the smallest attention distribution drift across task transitions (a reduction of 83.5%).
- Question-only storage reduces memory complexity from $O(N \cdot (I+L_q+L_a))$ to $O(N \cdot L_q)$.

## Highlights & Insights

- **Counter-intuitive finding**: Storing no images outperforms storing images—suggesting that the core bottleneck in continual VQA is answer space shift rather than visual feature forgetting.
- **Privacy-friendly**: Visual data storage is entirely eliminated, satisfying privacy regulations such as GDPR.
- The diagnosis and visualization of the out-of-answer-set problem (via confusion matrices) is intuitive and compelling.
- The design of performing attention distillation over normalized probability distributions is elegant.

## Limitations & Future Work

- For tasks heavily reliant on visual spatial reasoning (e.g., object type recognition, spatial relationship judgment), question replay alone may be insufficient.
- Current evaluation is limited to VQAv2 and NExT-QA; open-domain VQA settings remain unvalidated.
- The method does not defend against advanced privacy threats such as model inversion attacks.
- A hybrid approach—selectively retaining a small number of critical images alongside question replay—could further improve performance.

## Related Work & Insights

- An interesting contrast emerges with experience replay in unimodal continual learning: the multimodal setting allows exploitation of cross-modal information redundancy.
- Attention consistency distillation is generalizable to other multimodal continual learning scenarios (e.g., video understanding, image captioning).
- The VQACL-QR setting—intermediate between memory-free and full-memory methods—holds independent research value.

## Rating

- Novelty: ⭐⭐⭐⭐ (question-only replay combined with ACD is novel; the VQACL-QR setting is a meaningful contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (two datasets, multiple baselines, thorough ablations, attention drift analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear motivation, outstanding confusion matrix visualization)
- Value: ⭐⭐⭐⭐ (practical significance of combining privacy preservation with performance improvement)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] Staining and Locking Computer Vision Models without Retraining](staining_and_locking_computer_vision_models_without_retraining.md)

</div>

<!-- RELATED:END -->
