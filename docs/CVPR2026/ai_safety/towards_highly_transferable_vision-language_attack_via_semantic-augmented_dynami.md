---
title: >-
  [Paper Note] Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction
description: >-
  [CVPR 2026][AI Safety][Adversarial Attack] This paper proposes SADCA (Semantic-Augmented Dynamic Contrastive Attack), which iteratively disrupts cross-modal semantic consistency between adversarial images and texts via a dynamic contrastive interaction mechanism and a semantic augmentation module. SADCA significantly improves adversarial transferability against vision-language pre-training (VLP) models, surpassing existing SOTA methods in both cross-model and cross-task attack settings.
tags:
  - CVPR 2026
  - AI Safety
  - Adversarial Attack
  - Vision-Language Models
  - Adversarial Transferability
  - Contrastive Learning
  - Semantic Augmentation
date: 2026-05-08
content_hash: 8df94e8265fc8091
---

# Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction

**Conference**: CVPR 2026
**arXiv**: [2603.04839](https://arxiv.org/abs/2603.04839)
**Code**: [GitHub](https://github.com/LiYuanBoJNU/SADCA)
**Area**: AI Security
**Keywords**: Adversarial Attack, Vision-Language Models, Adversarial Transferability, Contrastive Learning, Semantic Augmentation

## TL;DR

This paper proposes SADCA (Semantic-Augmented Dynamic Contrastive Attack), which iteratively disrupts cross-modal semantic consistency between adversarial images and texts via a dynamic contrastive interaction mechanism and a semantic augmentation module. SADCA significantly improves adversarial transferability against vision-language pre-training (VLP) models, surpassing existing SOTA methods in both cross-model and cross-task attack settings.

## Background & Motivation

**Security Concerns in VLP Models**: Vision-language pre-training models such as CLIP, ALBEF, and TCL achieve strong performance on tasks including image-text retrieval (ITR), image captioning (IC), and visual grounding (VG) through large-scale image-text joint training. However, their adversarial robustness remains a significant concern. Investigating adversarial attacks is essential for evaluating and improving the security of VLP models.

**Limitations of Prior Work — Static Interaction**: Existing VLP attack methods (e.g., SGA, SA-AET) rely on static cross-modal interaction — performing only one or two interactions on the original image-text pairs. Adversarial examples deviate from the semantic center along a fixed direction, lacking the ability to explore diverse attack directions in the semantic space, which leads to poor cross-model transferability.

**Limitations of Prior Work — Neglect of Negative Samples**: Existing methods utilize only positive image-text pairs, overlooking the role of negative samples in shaping semantic decision boundaries. With only a "push" force (away from positive samples) and no "pull" force (toward negative samples), adversarial examples are insufficiently separated from benign samples in the embedding space.

**Limitations of Prior Work — Insufficient Input Diversity**: Input transformations are an effective strategy for improving transferability in conventional image attacks (e.g., SIA, BSR), yet existing VLP attack methods largely neglect this aspect, considering only limited scale invariance, resulting in insufficient semantic diversity.

## Method

### Overall Architecture

SADCA comprises two core components:
1. **Dynamic Contrastive Interaction**: Iteratively updates adversarial images and texts, continuously disrupting cross-modal semantic consistency using contrastive losses over positive and negative samples at each iteration.
2. **Semantic Augmentation Module**: Enriches semantic diversity during the attack via local semantic image augmentation and mixed semantic text augmentation.

The attack pipeline first obtains a positive image $v_p$ via semantic alignment, then performs $I$ rounds of dynamic interaction. In each round, the adversarial text is updated first, followed by $J$ PGD iterations to update the adversarial image, with contrastive losses over positive and negative samples and semantic augmentation applied at each step.

### Key Designs

1. **Semantic-Center Positive Sample Alignment**

    - **Function**: Aligns the benign image with multiple textual descriptions in the semantic space to obtain a positive image $v_p$ closer to the semantic center.
    - **Design Motivation**: The feature embedding of the original image contains abundant redundant information irrelevant to the text. Directly using biased image-text pairs as positive samples impedes the adversarial example from moving away from the correct semantic center.
    - **Mechanism**: $v_p = \arg\max_{v_p \in B[v,\epsilon_v]} \sum_{m=1}^{M} Cos(v, t_m)$, where $T = \{t_1, ..., t_M\}$ denotes multiple textual descriptions paired with the benign image. Negative sample sets $V_n, T_n$ are constructed by randomly sampling $K$ mismatched samples from the dataset.
    - **Novelty**: Whereas SGA directly uses the original image-text pair, SADCA performs semantic alignment first to obtain a more accurate positive anchor.

2. **Dynamic Contrastive Interaction Mechanism**

    - **Function**: Alternately updates adversarial texts and adversarial images at each iteration of the adversarial optimization, continuously exploiting the current adversarial state for cross-modal semantic disruption.
    - **Design Motivation**: Static interaction can only shift the adversarial example along a fixed direction. Dynamic interaction allows gradient directions to be recalibrated at each round based on the latest semantic state, enabling exploration of a broader attack direction space.
    - **Mechanism**:
        - Adversarial image loss (with dynamic term): $\mathcal{L}_v = \sum_m Cos(v'_i, t_{pm}) - \lambda \sum_k Cos(v'_i, t_{nk}) + \sum_m Cos(v'_i, t'_{im}) - \lambda \sum_k Cos(v'_i, t_{nk})$
        - Adversarial text loss (with dynamic term): $\mathcal{L}_t = Cos(v_p, t'_i) - \lambda \sum_k Cos(v_{nk}, t'_i) + Cos(v'_i, t'_i) - \lambda \sum_k Cos(v_{nk}, t'_i)$
        - In each round, the text $t'_{i+1}$ is updated first, then the image $v'_{i+1}$ is updated via $J$ PGD steps using the updated text.
    - **Key Formula**: Gradient update $g_{i(j+1)} = \mu \cdot g_{ij} + \nabla\mathcal{L}_v / \|\nabla\mathcal{L}_v\|$, $v'_{i(j+1)} = clip(v'_{ij} + \alpha \cdot sign(g_{i(j+1)}))$
    - **Novelty**: Compared to SGA's single interaction and SA-AET's two static interactions, SADCA performs $I=5$ rounds of dynamic interaction, each with a distinct semantic state.

3. **Semantic Augmentation Module**

    - **Function**: Enriches the diversity of semantic representations during the attack through augmentation along both image and text modalities.
    - **Design Motivation**: Conventional input transformations (e.g., SIA) operate only at the perceptual level, neglecting the cross-modal semantic alignment mechanism of VLP models. Richer semantic gradients reduce overfitting to a single semantic viewpoint.
    - **Mechanism**:
        - **Local Semantic Image Augmentation**: Random cropping (ratio $r_s \sim U(0.4, 0.8)$) + random augmentation (rotation/brightness/flipping) → $V'_{sa} = \{A_s(Resize(Crop(v'; r_s)))\}_{s=1}^S$
        - **Mixed Semantic Text Augmentation**: Randomly select and concatenate two texts from the adversarial text set → $T'_{sa} = \{Concat(t'_i, t'_j) | i \neq j\}_{s=1}^S$
    - **Novelty**: SGA employs only limited scale invariance; SADCA augments both image and text modalities simultaneously, with local cropping focusing on fine-grained semantics rather than global transformations.

### Loss & Training

- Total image attack loss: $\mathcal{L}_v = \mathcal{L}(V'_{sa}, T_p, T_n) + \mathcal{L}(V'_{sa}, T'_{sa}, T_n)$ (contrastive loss over positive/negative samples after semantic augmentation + contrastive loss with dynamic adversarial texts)
- Total text attack loss: $\mathcal{L}_t = \mathcal{L}(t'_m, v'_i, V_n) + \mathcal{L}(t'_m, v_p, V_n)$
- Key hyperparameters: step size $\alpha = 2/255$, momentum $\mu = 1.0$, dynamic interaction rounds $I = 5$, image attack iterations $J = 10$, number of negative samples $K = 20$, negative sample weight $\lambda = 0.2$, augmentation count $S = 10$
- Image perturbation constraint: $\ell_\infty$ norm, $\epsilon_v = 8/255$; text perturbation: BERT-Attack, $\epsilon_t = 1$

## Key Experimental Results

### Main Results (Cross-Model Transferability — Flickr30K ITR Task)

| Source→Target | Metric | SADCA | SA-AET(LI)+SIA | Gain |
|---------------|--------|-------|----------------|------|
| ALBEF→CLIPViT | TR R@1 ASR | **81.10** | 75.71 | +5.39 |
| ALBEF→CLIPCNN | IR R@1 ASR | **86.11** | 80.41 | +5.70 |
| TCL→CLIPViT | TR R@1 ASR | **78.28** | 77.04 | +1.24 |
| TCL→CLIPCNN | IR R@1 ASR | **88.71** | 84.05 | +4.66 |
| CLIPViT→ALBEF | TR R@1 ASR | **87.07** | 79.04 | +8.03 |
| CLIPViT→TCL | IR R@1 ASR | **87.98** | 82.57 | +5.41 |
| CLIPCNN→CLIPViT | TR R@1 ASR | **49.43** | 38.69 | +10.74 |

SADCA significantly outperforms the SOTA (SA-AET(LI)+SIA) across all four source models in cross-model transfer, with an average ASR improvement of approximately 5–10%.

### Cross-Task Transferability (ALBEF → Other Tasks)

| Task | Metric | SADCA | SA-AET | Gain |
|------|--------|-------|--------|------|
| VG (Val) | Acc ↓ | **46.78** | 47.44 | −0.66 |
| IC (B@4) | ↓ | **17.4** | 21.0 | −3.6 |
| IC (CIDEr) | ↓ | **50.3** | 65.7 | −15.4 |
| IC (SPICE) | ↓ | **10.7** | 13.6 | −2.9 |

### Attack Against LVLMs (ALBEF → Large Vision-Language Models)

| Target Model | Clean | SADCA | SA-AET(LI)+SIA |
|--------------|-------|-------|----------------|
| LLaVA-1.5-7B | 3.46 | **40.34** | 35.20 |
| Qwen3-VL-8B | 14.4 | **86.34** | 80.14 |
| GPT-5 | 23.88 | **78.61** | 68.08 |
| GPT-4o-mini | 15.00 | **79.12** | 62.48 |
| Gemini-2.0 | 6.96 | **52.06** | 41.56 |

### Key Findings

- **Dynamic interaction is the core driver**: Moving from SGA's single interaction to SADCA's 5-round dynamic interaction yields substantial ASR improvements (e.g., ALBEF→CLIPCNN TR increases from 39.59% to 85.44%).
- **Input transformations are also effective for VLP attacks**: Integrating SIA into SGA/SA-AET yields significant performance gains, validating the importance of input diversity in VLP attacks.
- **Contribution of negative samples**: Introducing negative contrastive samples enables more thorough semantic-space deviation; the full SADCA outperforms its variant without negative samples by approximately 3–5% ASR.
- **Effectiveness against closed-source commercial models**: SADCA achieves an attack success rate of 78.61% against GPT-5, highlighting the pervasive adversarial security risks in VLP models.

## Highlights & Insights

1. **Dynamic vs. Static Interaction**: Static methods apply a one-time push in a fixed direction, whereas the dynamic approach continuously adjusts the push direction — after each interaction round, the semantic state of both adversarial texts and images changes, causing gradient directions to update accordingly and naturally enabling exploration of a broader attack direction space.
2. **Effective Application of Positive-Negative Contrastive Learning**: The contrastive learning paradigm is applied to adversarial attacks — positive samples provide a "push" force while negative samples provide a "pull" force, jointly driving adversarial examples across semantic decision boundaries.
3. **Dual-Modality Design of Semantic Augmentation**: Local cropping focuses on fine-grained semantic regions, while text concatenation generates richer semantic representations; the two strategies jointly reduce overfitting to a single semantic viewpoint.
4. **Natural Extension to LVLM Attacks**: Although primarily designed for VLP models, SADCA demonstrates strong attack capability against LVLMs including GPT-5.

## Limitations & Future Work

1. Dynamic interaction introduces additional computational overhead (total iterations $I \times J = 50$ steps), which is approximately 5× slower than SGA's 10 steps.
2. Random selection of negative samples may be suboptimal; negative mining strategies based on semantic distance or adversarial difficulty could further improve performance.
3. Text perturbation relies on the word substitution strategy of BERT-Attack, which offers limited preservation of semantic coherence for long texts.
4. Only the $L_\infty$ norm constraint is evaluated; performance under other constraints such as $L_2$ remains unexplored.
5. The effect of defense strategies (e.g., adversarial training, input denoising) on suppressing SADCA is not discussed.

## Related Work & Insights

- **SGA (ICLR 2023)**: The first VLP attack method to propose expanding image-text pair diversity using multiple textual descriptions, but limited to a single static interaction.
- **SA-AET**: Introduces contrastive feature space optimization to improve adversarial trajectories, but remains constrained to two static interactions.
- **SIA (CVPR 2024)**: Structure-Invariant Attack — a general input transformation strategy; SADCA validates its effectiveness in VLP attacks and further proposes semantic-level augmentation.
- **Insights**: The contrastive learning framework can be more broadly applied to other adversarial attack settings (e.g., 3D vision, speech models); the dynamic interaction paradigm is also applicable to the defense side in adversarial training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dynamic contrastive interaction mechanism is novel; the integration of positive and negative samples in VLP attacks is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 VLP models, 2 task categories, multiple LVLMs (including GPT-5), cross-model and cross-task evaluations, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; comparison figures with prior work are intuitive; algorithm pseudocode is complete.
- **Value**: ⭐⭐⭐⭐ — Exposes the cross-modal attack vulnerability of VLP models and provides a stronger attack baseline for adversarial robustness research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [\[AAAI 2026\] Transferable Hypergraph Attack via Injecting Nodes into Pivotal Hyperedges](../../AAAI2026/ai_safety/transferable_hypergraph_attack_via_injecting_nodes_into_pivotal_hyperedges.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](../../ICCV2025/ai_safety/semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)

<!-- RELATED:END -->
