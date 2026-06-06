---
title: >-
  [Paper Note] CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID
description: >-
  [ACL 2026][Human Understanding][Federated Domain Generalization] Addressing the "semantic-style conflict" in Federated Domain Generalization for Person Re-identification (FedDG-ReID)…
tags:
  - "ACL 2026"
  - "Human Understanding"
  - "Federated Domain Generalization"
  - "Person Re-identification"
  - "CLIP Semantic Anchoring"
  - "Style Diversification"
  - "Camera Invariance"
date: 2026-05-08
content_hash: b134e04751b8943b
---

# CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID

**Conference**: ACL 2026  
**arXiv**: [2604.26363](https://arxiv.org/abs/2604.26363)  
**Code**: https://github.com/NanYiyuzurn/ACL-LGPS-2026  
**Area**: Federated Learning / Domain Generalization / Person ReID  
**Keywords**: Federated Domain Generalization, Person Re-identification, CLIP Semantic Anchoring, Style Diversification, Camera Invariance

## TL;DR
Addressing the "semantic-style conflict" in Federated Domain Generalization for Person Re-identification (FedDG-ReID), CO-EVO proposes CSA (Camera-invariant Semantic Anchoring) to learn frozen identity-level text prototypes as "centers of gravity" and GSD (Global Style Diversification) using a lightweight GCSB (Global Camera-Style Bank) to synthesize realistic cross-domain perturbations. Their coupled optimization achieves a ViT mAP improvement of 14 points (34.1→48.1) over the Prev. SOTA on Market-1501/MSMT17/CUHK03 leave-one-out benchmarks.

## Background & Motivation

**Background**: Person ReID faces severe cross-camera domain shifts in real-world deployments. DG-ReID attempts to generalize from multi-source training to unseen target domains. Privacy requirements have driven the FedDG-ReID paradigm, where multiple clients hold source domain data without sharing raw images, collaboratively training a retrieval model capable of generalizing to unseen target domains. Prev. SOTA methods like DACS and SSCU use Style Transformation Models (STM) for local diversification.

**Limitations of Prior Work**: (1) **Shortcut Learning**: Due to the lack of global semantic reference during local optimization, models take "shortcuts" by utilizing background textures and camera color imprints to distinguish identities. While performance is stable locally, it fails completely across federated domains. (2) **Difficult VLM Deployment**: VLMs like CLIP have strong semantic priors, but ReID labels are anonymous IDs without natural language names. Furthermore, transmitting LLMs in a federated setting incurs massive communication costs. (3) **Style Perturbation Issues**: STMs require training auxiliary generative networks, which are computationally expensive, unstable under federated settings, and often produce repetitive artifacts or overexposure.

**Key Challenge**: **Semantic-style conflict**. Purely visual supervision leads to domain-specific shortcuts, while aggressive style augmentation can destroy identity-sensitive cues without grounding anchors. Existing methods either have grounding without diversification or diversification without grounding, failing to achieve both simultaneously.

**Goal**: Within a strict federated setting (no raw data sharing, controllable communication overhead), **simultaneously achieve** (a) stable global semantic references to prevent shortcut learning, and (b) high-quality cross-domain style diversification for visual boundary exploration, ensuring the two components reinforce each other through coupled loops.

**Key Insight**: This work introduces "language-guided semantic supervision" to FedDG-ReID for the first time, utilizing CLIP to learn learnable prompt tokens for each identity as text anchors. Simultaneously, it shares camera styles using statistics (channel mean/var) instead of raw data, bypassing the training burden of STMs.

**Core Idea**: The learned identity text prototypes from CSA serve as frozen "centers of gravity" $T_k$, while GSD transforms images into diverse stylized views $x'$ via channel re-normalization. This forces the encoder to map both the original view $x$ and the perturbed view $x'$ to the same anchor $t_{y_i}$. While styles vary, anchors remain fixed, forcing representations to focus on anatomical identity features.

## Method

### Overall Architecture
CO-EVO is a three-phase federated framework:

- **Phase I (Semantic Anchoring)**: Each client learns $L$ learnable tokens locally for each identity to form the prompt "a photo of a [X1]...[XL] person". By freezing the CLIP encoder and minimizing bidirectional contrastive and cross-camera consistency losses, purified identity text prototypes $T_k = \{t_y\}_{y \in Y_k}$ are obtained and stored in a **frozen cache** locally. Simultaneously, each client uploads channel statistics $(\mu_{k,c}, \sigma^2_{k,c})$ to the server.
- **Phase II/III (Coupled Federated Loop)**: The server aggregates these to form the GCSB $\mathcal{B}$ and broadcasts the global image encoder $\theta^0$. In each round $R = 1 \dots 60$, clients use GSD to sample $(\mu_s, \sigma_s^2)$ from $\mathcal{B}$ for re-normalization to generate $x'$. For both $x$ and $x'$, $L_{id} + L_{tri} + \lambda L_{align}$ is calculated ($L_{align}$ aligns visual features to frozen anchors $t_{y_i}$). After $E=1$ local epoch, model parameters are uploaded for FedAvg aggregation and subsequent broadcasting.
- **Output**: Global image encoder $\theta^R$, used for ReID retrieval in unseen target domains during deployment.

### Key Designs

1. **Camera-Invariant Semantic Anchoring (CSA)**:
    - **Function**: Uses CLIP-based learnable prompts to learn a "pure, cross-camera robust" text prototype for each identity, serving as a fixed anchor to prevent shortcut learning during local training.
    - **Mechanism**: $L$ learnable tokens are introduced for each identity $y$ within the template "a photo of a [$X_1^y$]...[$X_L^y$] person". With CLIP encoders frozen, only tokens are optimized. A bidirectional contrastive loss aligns visual $v_i$ and text $t_{y_i}$: $L_{i2t}(i) = -\log\frac{\exp(s(v_i, t_{y_i})/\tau)}{\sum_a \exp(s(v_i, t_a)/\tau)}$ and its symmetric $L_{t2i}(y)$. A key innovation is the **Cross-Camera Consistency** regularization $L_{c3} = \sum_y \sum_{i,j \in P(y), c_i \ne c_j} \|s(v_i, t_y) - s(v_j, t_y)\|^2$, which minimizes the difference in visual-text similarity for the same identity across different cameras. Total loss: $L_{CSA} = L_{i2t} + L_{t2i} + \lambda_{c3} L_{c3}$ ($\lambda_{c3} = 0.1$). After Phase I, prototypes $T_k$ are **frozen and cached**, remaining unchanged during the federated loop.
    - **Design Motivation**: (1) Traditional CLIP-ReID lacks cross-camera constraints, causing prompts to internalize camera imprints. $L_{c3}$ enforces consistent similarity for different cameras of the same person. (2) **Static caching** is used instead of dynamic updates—ablation shows dynamic prompts drop mAP by 3.3% as they are re-polluted by local camera noise. (3) Token length $L=4$ is optimal; $L=1$ is too generic, while $L \ge 8$ introduces camera noise.

2. **Global Style Diversification (GSD) + Global Camera-Style Bank (GCSB)**:
    - **Function**: Allows the encoder to see diverse cross-client and cross-camera visual distributions without sharing raw images or training generators.
    - **Mechanism**: Channel-wise first/second-order statistics represent domain styles. Channel mean/var primarily capture illumination, color tone, and texture while preserving semantic structure. At the end of Phase I, client $k$ extracts $(\mu_{k,c}, \sigma^2_{k,c}) = \mathrm{Stat}(\{x_i^k | c_i^k = c\})$ for each camera $c$ and uploads them. The server aggregates these into GCSB $\mathcal{B} = \cup_k \cup_c \{(\mu_{k,c}, \sigma^2_{k,c})\}$ and broadcasts it. During local training, $(\mu_s, \sigma_s^2)$ are sampled from $\mathcal{B}$ for template-based re-normalization: $\hat{x} = \frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \epsilon}}$, $x' = \hat{x} \odot \sqrt{\sigma_s^2} + \mu_s$. The resulting view is based on **real camera distributions**. K-means pseudo-groups can substitute camera IDs if unavailable. GCSB construction takes 4s/client with zero trainable parameters.
    - **Design Motivation**: (1) STM-based methods are unstable and produce artifacts; GSD is lightweight and realistic. (2) **Global** banks outperform local ones (Global 42.4% vs Local 40.2% mAP on Market) because local styles have already been seen. (3) Sharing low-dimensional statistics protects privacy and is robust to metadata noise.

3. **Coupled Optimization Loop with Frozen Anchors**:
    - **Function**: Couples CSA's frozen anchors with GSD's dynamic styles in every training step to learn "style-variant, identity-invariant" features.
    - **Mechanism**: Each mini-batch includes $x$ (original) and $x' \sim GSD(\mathcal{B})$ (perturbed). For each view, classification loss $L_{id}$, triplet loss $L_{tri}$, and semantic alignment loss $L_{align}(i; \tilde{x}) = -\log\frac{\exp(s(v_i(\tilde{x}), t_{y_i})/\tau)}{\sum_{y\in Y_k} \exp(s(v_i(\tilde{x}), t_y)/\tau)}$ are calculated. **Key Design**: Both $x$ and $x'$ are aligned to the **same** frozen anchor $t_{y_i}$. Total objective: $L_{loc} = \sum_{\tilde{x}\in\{x,x'\}}(L_{id} + L_{tri} + \lambda L_{align})$.
    - **Design Motivation**: "Co-evolution" is defined as the **evolution of input distributions via GSD and encoder parameters under the pull of fixed anchors**. Cosine distance analysis shows CO-EVO reduces same-identity distance from 0.68 to 0.35 (-48.5%) and increases different-identity distance from 0.42 to 0.78 (+85.7%) compared to SSCU. CSA+GSD synergy provides a +17 mAP gain (25.4→42.4).

### Loss & Training
Phase I: $L_{CSA} = L_{i2t} + L_{t2i} + \lambda_{c3} L_{c3}$ (120 rounds, $\lambda_{c3}=0.1, L=4, \tau=0.07$, CLIP ViT-B/16 frozen, only tokens learned). Phase II/III: $L_{loc} = \sum_{\tilde{x}}(L_{id} + L_{tri} + \lambda L_{align})$ (60 rounds, $E=1$, batch 64, SGD lr=1e-3, FedAvg aggregation).

## Key Experimental Results

### Main Results Protocol I — Leave-One-Domain-Out (mAP / Rank-1 %)

| Method | MS+C2+C3→M | M+C2+C3→MS | MS+C2+M→C3 | Average mAP/R1 |
|--------|------------|------------|------------|-----------------|
| FedPav | 25.4 / 49.4 | 5.2 / 15.5 | 22.5 / 24.3 | 17.7 / 29.7 |
| MixStyle | 31.2 / 53.5 | 5.5 / 16.0 | 28.6 / 31.5 | 21.8 / 33.6 |
| SNR | 32.7 / 59.4 | 5.1 / 15.3 | 28.5 / 30.0 | 22.1 / 34.9 |
| DACS (RN50) | 36.3 / 61.2 | 10.4 / 27.5 | 30.7 / 34.1 | 25.8 / 40.9 |
| SSCU (RN50) | 39.5 / 66.4 | 11.9 / 32.3 | 32.8 / 34.1 | 28.1 / 44.3 |
| **Ours (RN50)** | **42.4** / **71.2** | **12.9** / **33.7** | **34.9** / **37.1** | **30.1** / **47.3** |
| DACS (ViT) | 45.4 / 70.7 | 20.3 / 44.2 | 36.6 / 42.1 | 34.1 / 52.3 |
| **Ours (ViT)** | **60.7** / **80.2** | **32.2** / **60.3** | **51.3** / **52.7** | **48.1** / **64.4** |

Across the ViT backbone, average mAP increases by **+14.0** (34.1→48.1) and R1 increases by **+12.1** (52.3→64.4) over DACS-ViT.

### Ablation Study

| Configuration | MS+C2+C3→M mAP/R1 | M+C2+C3→MS mAP/R1 | MS+C2+M→C3 mAP/R1 | Note |
|------|-------------------|-------------------|-------------------|------|
| Baseline (no CSA, no GSD) | 25.4 / 49.4 | 5.2 / 15.5 | 22.5 / 24.3 | FedPav |
| + CSA only | 38.7 / 66.7 | 9.1 / 30.7 | 33.8 / 35.2 | +13.3 mAP avg |
| + GSD only | 39.8 / 67.6 | 10.8 / 30.3 | 34.1 / 35.9 | +14.4 mAP avg |
| **+ CSA + GSD (full)** | **42.4** / **71.2** | **12.9** / **33.7** | **37.1** / **38.9** | **+17.0 mAP avg, non-additive synergy** |
| CSA Dynamic (vs Static) | 39.1 / 67.8 | 10.4 / 30.1 | 34.2 / 35.6 | -3.3 mAP for dynamic |
| GSD Local-only (vs Global) | 40.2 / 68.5 | 10.5 / 31.4 | 35.1 / 36.3 | -2.2 mAP for local |

### Key Findings
- **Non-additive Synergy**: CSA (+13.3) and GSD (+14.4) independently improve performance, but combined they yield +17.0 mAP, proving they resolve different aspects (grounding vs. diversification) of the semantic-style conflict.
- **Static > Dynamic Anchoring**: Frozen prototypes outperform dynamic updates by 3.3 mAP, validating the "frozen anchor" philosophy.
- **Global GCSB Advantage**: Shared real statistics create effective unseen domain proxies, whereas local styles are redundant and random noise is unrealistic.
- **Robustness**: Performance drops are mild with 30% camera ID noise or K-means clustering, outperforming SSCU's clean baseline.
- **Decision Boundary Restoration**: Same-identity cosine distance drops by 48.5%, while different-identity distance rises by 85.7%.
- **Backbone Scalability**: Gains are even larger on the ViT backbone, suggesting the method's potential has not yet peaked.

## Highlights & Insights
- **Semantic-Style Conflict Paradox**: Formalizes the tension between domain-invariant features and style augmentation in FedDG, resolved via the "frozen anchor + dynamic style" mechanism.
- **CLIP Prompt-based Anchoring for Label-free Domains**: Solves the issue where ID-based tasks lack natural language labels by using learnable prompts and cross-camera consistency to create purified text anchors.
- **Privacy-Preserving Style Sharing**: Uses $2C$-dimensional vectors (mean/var) instead of images or models to share domain knowledge, offering an elegant engineering solution for federated settings.
- **Frozen Anchor Philosophy**: Demonstrates that maintaining static anchors is more stable than dynamic updates in federated representation learning.

## Limitations & Future Work
- **Ours** acknowledges that CSA anchors are derived only from source identities, limiting effectiveness on extreme occlusion or heavily out-of-distribution samples.
- GCSB only captures photometric changes via channel statistics, lacking modeling for geometric or background structural variations.
- High dependency on metadata (camera ID or groupable attributes) and a lack of formal privacy guarantees like Differential Privacy.
- Future work involves introducing spatial-aware augmentation and optimizing communication costs for large backbones.

## Related Work & Insights
- **vs MixStyle/CrossStyle**: These focus on single-machine style perturbation, whereas CO-EVO scales this to a federated global bank with semantic anchors.
- **vs DACS/SSCU**: Replaces heavy STM generators with lightweight statistics, providing stability and grounding when paired with CSA.
- **vs CLIP-ReID/TF-CLIP**: First to introduce CLIP to FedDG-ReID with cross-camera consistency constraints.
- **vs DiPrompT**: While DiPrompT uses prompts for disentanglement, Ours employs a coupled anchor+style mechanism better suited for identity-strict tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Diagnosis of "semantic-style conflict" and the frozen anchor coupled training paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive protocols, multiple backbones, comprehensive ablation, and robustness tests.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts and intuitive analogies; drawback: CV-heavy task in an NLP-centric venue (ACL).
- Value: ⭐⭐⭐⭐ Significant mAP gains and generalizable GCSB design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](../../ICCV2025/human_understanding/semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[CVPR 2026\] Talking Together: Synthesizing Co-Located 3D Conversations from Audio](../../CVPR2026/human_understanding/talking_together_synthesizing_co-located_3d_conversations_from_audio.md)
- [\[AAAI 2026\] Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion](../../AAAI2026/human_understanding/streaming_generation_of_co-speech_gestures_via_accelerated_rolling_diffusion.md)
- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](../../ICCV2025/human_understanding/gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[ICLR 2026\] Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](../../ICLR2026/human_understanding/inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)

</div>

<!-- RELATED:END -->
