---
title: >-
  [Paper Note] DisenQ: Disentangling Q-Former for Activity-Biometrics
description: >-
  [ICCV 2025][Multimodal VLM][Activity-biometrics recognition] This paper proposes DisenQ (Disentangling Q-Former), which leverages structured language guidance to disentangle video features into three independent spaces—b…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Activity-biometrics recognition"
  - "Q-Former"
  - "feature disentanglement"
  - "multimodal learning"
  - "video person re-identification"
date: 2026-05-08
content_hash: a13717f8d6fd1502
---

# DisenQ: Disentangling Q-Former for Activity-Biometrics

**Conference**: ICCV 2025
**arXiv**: [2507.07262](https://arxiv.org/abs/2507.07262)
**Code**: None (Project Page available)
**Area**: Multimodal VLM
**Keywords**: Activity-biometrics recognition, Q-Former, feature disentanglement, multimodal learning, video person re-identification

## TL;DR

This paper proposes DisenQ (Disentangling Q-Former), which leverages structured language guidance to disentangle video features into three independent spaces—biometric, motion, and non-biometric—achieving state-of-the-art activity-aware person recognition without requiring additional visual modalities.

## Background & Motivation

**Activity-Biometrics Recognition** is an emerging task that identifies individuals while they perform daily activities (beyond walking/standing), posing greater challenges than conventional person re-identification:

**Entanglement of identity cues with motion and appearance**: Different activities such as walking and jumping introduce substantial motion variations that intermingle with identity features, hindering recognition.

**Reliance on additional visual modalities in existing methods**: Methods such as ABNet require silhouette maps, whose extraction accuracy is limited by environmental conditions.

**Limitations of CLIP-based approaches**: Global image-text alignment lacks identity-specific feature separation and cannot maintain temporal consistency.

Core motivation: **Can language supervision replace additional visual modalities?** Specifically, can structured textual descriptions guide feature disentanglement such that biometric features remain invariant to appearance and motion changes?

## Method

### Overall Architecture

Input RGB video → ViT visual encoder extracts per-frame features → temporal attention pooling yields video feature $F$ → DisenQ employs three groups of learnable queries to extract biometric features, motion features, and non-biometric features, respectively → identity classification head performs recognition. During training, a frozen VLM generates structured textual descriptions as guidance; no text is required at inference.

### Key Designs

1. **DisenQ (Disentangling Querying Transformer)**:

    - Built upon the BLIP-2 Q-Former architecture, introducing **three independent groups of learnable queries**: $z_b$ (biometric), $z_m$ (motion), and $z_{\hat{b}}$ (non-biometric).
    - The three query groups share self-attention and cross-attention layers but do not interact with each other, maintaining independence.
    - Each query group performs cross-attention with its corresponding visual and textual features:
    $Q_b = Wz_b, \quad K_b = W[F, T_b], \quad V_b = W[F, T_b]$
    - Motion query $z_m$ and non-biometric query $z_{\hat{b}}$ follow the same design, using $T_m$ and $T_{\hat{b}}$ respectively.
    - Design motivation: shared layers reduce parameters (ablations show that three fully independent Q-Formers triple the parameter count while yielding only +0.23% R@1), while independent queries ensure feature separation.

2. **Structured Text Generation and Encoding**:

    - A frozen LLaVA 1.5 7B generates three categories of descriptions from keyframes:
        - **Biometric description $P_b$**: body shape, posture, and salient physical characteristics; generated once per identity and subsequently updated via a running average.
        - **Motion description $P_m$**: action labels and movement patterns.
        - **Non-biometric description $P_{\hat{b}}$**: clothing, accessories, etc.
    - Descriptions are encoded by a frozen BERT into $T_b, T_m, T_{\hat{b}}$.
    - Neither the VLM nor text is required at inference—DisenQ learns the disentanglement pattern during training.

3. **Adaptive Identity Similarity Computation**:

    - Query embeddings are mean-pooled to obtain $F_b$, $F_m$, and $F_{\hat{b}}$; only $F_b$ and $F_m$ are used for final recognition.
    - A lightweight MLP dynamically computes weights $\alpha_1, \alpha_2$ for biometric and motion features:
    $Sim(A,B) = \alpha_1 Sim_b(A,B) + \alpha_2 Sim_m(A,B)$
    - Dynamic weighting allows the model to leverage motion cues when they are informative and rely on biometric features otherwise.

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_{ID} + \lambda_2 \mathcal{L}_{Tri} + \lambda_3 \mathcal{L}_{Orth} + \lambda_4 \mathcal{L}_{Act}$$

- $\mathcal{L}_{ID}$: cross-entropy classification loss on $F_b$, $\lambda_1=0.01$
- $\mathcal{L}_{Tri}$: triplet loss (margin $m=0.3$) to encourage intra-identity feature compactness
- $\mathcal{L}_{Orth} = \|F_b^T F_{\hat{b}}\|$: orthogonality constraint enforcing independence between biometric and non-biometric features
- $\mathcal{L}_{Act}$: cross-entropy for action classification on $F_m$, ensuring motion features retain movement information
- Training: EVA-CLIP ViT-G/14 visual encoder; DisenQ initialized from InstructBLIP weights; AdamW, lr=1e-4, 60 epochs, batch size=32 (8 identities × 4 clips)

## Key Experimental Results

### Main Results (Activity-Biometrics Recognition Benchmarks)

| Method | NTU Same R@1 | NTU Cross R@1 | PKU Same R@1 | PKU Cross R@1 | Charades Same R@1 | Charades Cross R@1 |
|------|------|------|------|------|------|------|
| ABNet (CVPR24) | 78.8 | 77.0 | 86.8 | 81.4 | 45.8 | 44.8 |
| CLIP-ReID (AAAI23) | 77.1 | 75.2 | 82.3 | 81.2 | 44.2 | 42.1 |
| Instruct-ReID (CVPR24) | 78.2 | 75.9 | 84.3 | 81.7 | 44.8 | 40.1 |
| **DisenQ (Ours)** | **82.2** | **80.9** | **89.2** | **84.1** | **49.9** | **48.4** |

Average R@1 improvements across the three datasets are 3.7%, 2.4%, and 3.9%, respectively. On the conventional MEVID benchmark, DisenQ also achieves 60.7% R@1 (SOTA).

### Ablation Study

| Configuration | NTU R@1 | NTU mAP | Charades R@1 | Charades mAP |
|------|---------|---------|-------------|-------------|
| Vision encoder only | 73.2 | 36.2 | 40.1 | 29.2 |
| + Text encoder | 77.7 | 40.6 | 46.5 | 31.8 |
| + DisenQ (full) | **82.2** | **43.8** | **49.9** | **34.8** |

Ablation on disentanglement types:

| Disentanglement | NTU R@1 | Charades R@1 |
|----------|---------|-------------|
| No disentanglement | 74.2 | 42.3 |
| $F_b$ + $F_{\hat{b}}$ (biometric + non-biometric) | 76.6 | 44.7 |
| $F_b$ + $F_m$ (biometric + motion) | 79.2 | 48.2 |
| $F_b$ + $F_{\hat{b}}$ + $F_m$ (all) | **82.2** | **49.9** |

When used alone, non-biometric features achieve only 3.8% R@1, confirming that identity information is successfully stripped from this space.

## Key Findings

- **DisenQ contributes the largest gain**: the improvement from adding the text encoder (+4.5%) and adding DisenQ (+4.5%) is most pronounced.
- **Three-way disentanglement offers complementary benefits**: biometric + motion (79.2%) outperforms biometric + non-biometric (76.6%), indicating that motion cues are more critical for identity recognition.
- **Non-biometric features are effectively "purged"**: with only 3.8% R@1 when used alone, they carry almost no identity information.
- **Insensitivity to VLM choice**: differences among LLaVA, InstructBLIP, and GPT-4V are below 0.2%, demonstrating robustness to text quality.
- **Adaptive weighting outperforms fixed weighting**: for low-motion activities (e.g., gestures), fixed weights cause motion features to introduce noise, while adaptive weights suppress this effect.
- Replacing ground-truth clothing descriptions with random ones leads to a 9.2% R@1 drop, validating the necessity of accurate textual guidance.

## Highlights & Insights

- **First application of Q-Former to feature disentanglement**: the paper elegantly extends BLIP-2's single query set into multiple independent query groups, each capturing a distinct information dimension.
- **No VLM or text required at inference**: language guidance is used exclusively during training; the queries internalize disentanglement patterns, incurring zero additional inference overhead.
- **Strong cross-setting generalization**: achieves SOTA or competitive performance on both activity-biometrics and conventional person re-identification benchmarks.
- **Running average update for biometric descriptions**: elegantly handles inconsistencies in descriptions of the same identity across different videos.

## Limitations & Future Work

- The approach depends on VLM-generated text quality; although ablations demonstrate insensitivity to VLM choice, quality may degrade in more extreme scenarios (severe occlusion, low resolution).
- ViT-G/14 as the visual encoder carries a large parameter count (1.8B), making it impractical in resource-constrained settings.
- Unsupervised or semi-supervised settings are not explored; the method requires ground-truth identity and action labels.
- Action labels are assumed to be provided as ground truth, necessitating an action recognition module in real-world deployment.

## Related Work & Insights

- Key distinction from ABNet: ABNet relies on additional visual modalities such as silhouette maps, whereas DisenQ replaces them with language supervision.
- The disentanglement paradigm of Q-Former can generalize to other tasks requiring multi-dimensional feature separation (e.g., separating content from emotion in sentiment analysis).
- The adaptive weighting mechanism is broadly applicable to any retrieval task that requires dynamic fusion of heterogeneous features.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Q-Former disentanglement is a genuine contribution; replacing additional visual modalities with language supervision is a well-motivated idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, detailed ablations, feature space visualizations, and analysis of VLM/encoder choices.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, complete method description, and high-quality figures.
- **Value**: ⭐⭐⭐⭐ Provides an effective solution for activity-aware person recognition with promising applications in surveillance and intelligent environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ACL 2026\] From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](../../ACL2026/multimodal_vlm/from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)
- [\[ICCV 2025\] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning](openvision_a_fully-open_cost-effective_family_of_advanced_vision_encoders_for_mu.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] Generalizable Object Re-Identification via Visual In-Context Prompting](generalizable_object_re-identification_via_visual_in-context_prompting.md)

</div>

<!-- RELATED:END -->
