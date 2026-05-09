---
title: >-
  [Paper Note] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning
description: >-
  [CVPR 2026][Segmentation][Visual in-context learning] This paper identifies a critical yet overlooked problem in visual in-context learning (VICL): existing prompt retrieval methods ignore label information, leading to label inconsistency. The proposed LaPR framework addresses this through joint image-label representation and a mixture-of-experts (MoE) mechanism, achieving label-aware prompt retrieval that consistently outperforms state-of-the-art methods on foreground segmentation, object detection, and image colorization tasks.
tags:
  - CVPR 2026
  - Segmentation
  - Visual in-context learning
  - prompt retrieval
  - label consistency
  - mixture of experts
  - contrastive learning
date: 2026-05-08
content_hash: f708f967594a1bff
---

# Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning

**Conference**: CVPR 2026
**arXiv**: [2604.03657](https://arxiv.org/abs/2604.03657)
**Code**: [https://github.com/luotc-why/CVPR26-LaPR](https://github.com/luotc-why/CVPR26-LaPR)
**Area**: Multimodal VLM / Segmentation
**Keywords**: Visual in-context learning, prompt retrieval, label consistency, mixture of experts, contrastive learning

## TL;DR

This paper identifies a critical yet overlooked problem in visual in-context learning (VICL): existing prompt retrieval methods ignore label information, leading to label inconsistency. The proposed LaPR framework addresses this through joint image-label representation and a mixture-of-experts (MoE) mechanism, achieving label-aware prompt retrieval that consistently outperforms state-of-the-art methods on foreground segmentation, object detection, and image colorization tasks.

## Background & Motivation

**Background**: Visual in-context learning (VICL) enables visual foundation models to handle diverse tasks via demonstrative prompts (image-label pairs). The representative MAE-VQGAN model formulates VICL as pixel inpainting using a 2×2 grid: top-left for the prompt image, top-right for the prompt label, bottom-left for the query image, and bottom-right to be predicted. Prompt selection significantly affects VICL performance, and prior works (SupPR, Partial2Global, RH-Partial2Global) have focused on image-similarity-based retrieval or re-ranking.

**Limitations of Prior Work**: Existing prompt retrieval methods focus exclusively on image similarity while ignoring label information. This leads to a typical failure case: retrieved prompt images may be visually similar to the query yet contain inconsistent labels. For instance, when the query depicts a cat, the retrieved prompt may contain a cat but with a label annotating a flower, causing erroneous VICL inference.

**Key Challenge**: Through empirical analysis, the authors demonstrate that among visually similar prompts, query-prompt label consistency is positively correlated with VICL performance, establishing labels as a critical yet neglected signal in prompt selection. The key challenge is that query labels are unknown at inference time, precluding direct label comparison.

**Goal**: (1) How to explicitly incorporate label information into prompt representations? (2) How to infer and match label semantics when query labels are unavailable?

**Key Insight**: Labels are treated as an integral part of prompt representation—joint image-label encoding constructs label-aware embeddings for prompts; on the query side, an MoE mechanism enables different experts to capture distinct label patterns (e.g., elongated shapes, pointed contours), with a router adaptively inferring the implicit label of each query.

**Core Idea**: Labels are incorporated as auxiliary signals in prompt retrieval; an MoE mechanism estimates implicit label patterns for unlabeled queries at inference time, enabling label-consistent prompt selection.

## Method

### Overall Architecture

Given a prompt database $\mathcal{B}=\{(I_i^p, L_i^p)\}$ (image-label pairs) and a query image $x_q$ (with unknown label $y_q$), the goal is to select the optimal prompt $c_q^\star$ that maximizes VICL inference quality. LaPR comprises three core components: (1) a frozen ViT that extracts image and label features, which are fused into a joint prompt embedding; (2) $K$ experts on both the query and prompt sides that generate pattern-specific representations, with a query-side router producing mixture weights; and (3) an alternating optimization scheme—an expert step with performance-guided contrastive loss, and a router step with label-guided contrastive loss.

### Key Designs

1. **Joint Image-Label Prompt Representation**:

    - **Function**: Explicitly injects label information into prompt embeddings.
    - **Mechanism**: A frozen feature extractor $f$ (e.g., CLIP ViT) separately encodes the prompt image and label to obtain $z_i^I = f(I_i^p)$ and $z_i^L = f(L_i^p)$, which are fused via element-wise addition: $z_i = z_i^I + z_i^L$. Pattern-specific prompt representations are then generated by expert $\bar{E}_k$: $p_{i,k} = \bar{E}_k(z_i)$.
    - **Design Motivation**: Conventional retrieval relies solely on $z_i^I$, discarding label signals. Encoding labels as auxiliary inputs endows prompt embeddings with dual information—both visual appearance and annotation semantics.

2. **Query-Side Mixture of Experts with Adaptive Routing**:

    - **Function**: Infers implicit label patterns when query labels are unavailable.
    - **Mechanism**: $K$ experts $E_k$ map query features $u_q = f(x_q)$ to pattern-specific representations $q_k = E_k(u_q)$. A router $R$ outputs a probability distribution $\pi_q = R(u_q) \in \Delta^K$ representing the query's mixture weights across patterns. The final query embedding is $\tilde{u}_q = \sum_k \pi_{q,k} q_k$, and the prompt embedding is $\tilde{p}_{i|q} = \sum_k \pi_{q,k} p_{i,k}$—the router weights are applied symmetrically to both sides, enabling retrieval to adapt to the query's implicit label.
    - **Design Motivation**: Different queries require attention to different aspects of prompts (e.g., queries depicting cats may rely more on animal-shape experts, while flower queries favor texture experts). The router learns to map queries to appropriate pattern combinations, indirectly estimating query labels.

3. **Alternating Optimization Strategy**:

    - **Function**: Decouples expert and router training to avoid instabilities in joint optimization.
    - **Mechanism**: Each mini-batch involves two update steps: (1) **Expert step**—the router is frozen, and experts are trained with a performance-guided contrastive loss $\mathcal{L}_{PG}$, where positive/negative samples are selected based on actual VICL inference scores (top-5 best and top-5 worst prompts from the candidate pool); (2) **Router step**—experts are frozen, and the router is trained with a label-guided contrastive loss $\mathcal{L}_{LG}$, with positive/negative samples based on prompt-query label matching scores, supplemented by a load-balancing loss $\mathcal{L}_{LB} = \mathrm{KL}(\bar{\pi} \| r)$ to prevent expert collapse.
    - **Design Motivation**: Experts must learn strong pattern-specific representations driven by VICL performance, while the router must learn accurate pattern assignment driven by label matching. Joint training causes these two objectives to interfere; alternating optimization yields greater stability and effectiveness.

### Loss & Training

Expert step loss: $\mathcal{L}_{PG}$ is an InfoNCE contrastive loss where positive samples are the highest-scoring prompts by VICL evaluation. Router step loss: $\mathcal{L}_R = \mathcal{L}_{LG} + \mathcal{L}_{LB}$, where $\mathcal{L}_{LG}$ is a label-matching-guided contrastive loss and $\mathcal{L}_{LB}$ is a KL-divergence load-balancing term. Training uses the SGD optimizer with a learning rate of 0.005, batch size of 64, 200 epochs on a single A100 GPU, with $K=10$ experts.

## Key Experimental Results

### Main Results

| Task | Metric | RH-Partial2Global | LaPR (Ours) | Gain |
|------|--------|-------------------|-------------|------|
| Foreground Segmentation (avg. mIoU) | mIoU↑ | 39.02 | **41.36** | +6.0% |
| Object Detection | mIoU↑ | 30.94 | **32.01** | +3.5% |
| Image Colorization | MSE↓ | 0.56 | 0.60 | -7.1% |
| Segmentation + Voting | mIoU↑ | 43.08 | 42.27 | -1.9% |
| Detection + Voting | mIoU↑ | 33.28 | **34.64** | +4.1% |

Note: LaPR, as a retrieval method, achieves comprehensive superiority in the non-voting setting; under the voting setting, it remains leading on detection. The marginal MSE gap on colorization is small.

### Ablation Study

| Configuration | Seg. mIoU | Det. mIoU | Notes |
|---------------|-----------|-----------|-------|
| Full LaPR | 41.36 | 32.01 | Baseline |
| w/o Router (uniform distribution) | 38.20 | 29.69 | Adaptive routing is critical |
| w/o Prompt Label | 39.19 | 30.94 | Label injection is effective |
| CLIP→DINOv2 features | 41.39 | 32.06 | Generalizes well across encoders |
| Single-stage joint training | 39.86 | 31.21 | Alternating optimization is superior |
| w/o $\mathcal{L}_{PG}$ | 35.05 | 27.30 | Performance-guided loss is the core supervision |
| w/o $\mathcal{L}_{LG}$ | 39.67 | 30.14 | Label-guided loss provides important auxiliary signal |

### Key Findings

- Cross-fold transferability significantly outperforms baselines: LaPR achieves an average cross-fold mIoU of 39.87 vs. 33.42 for SupPR, a 19.3% improvement.
- Analysis of MoE expert activation patterns shows that semantically similar categories (e.g., dog and horse) share similar expert activation distributions.
- Visualizations confirm that prompts retrieved by LaPR exhibit greater label consistency compared to those retrieved by SupPR.

## Highlights & Insights

- This work is the first to systematically demonstrate the importance of labels in VICL prompt retrieval, addressing a widely overlooked research gap.
- The MoE + router design elegantly resolves the core challenge of unknown query labels—replacing hard decisions with soft mixture weighting.
- The alternating optimization strategy is well-motivated, with experts and the router serving distinct roles, yielding more stable training.
- The method generalizes well across feature extractors (consistent performance with CLIP and DINOv2).

## Limitations & Future Work

- The number of experts $K=10$ is a fixed hyperparameter; the optimal value may vary across tasks.
- Image-label fusion relies on simple addition $z_i = z_i^I + z_i^L$; more sophisticated fusion strategies (e.g., cross-attention) may yield further improvements.
- Gains on the colorization task are limited, possibly because the label (colorized image) is already visually similar to the input image.
- Candidate pool construction still depends on top-50 retrieval using pretrained features, potentially missing prompts with high label compatibility but lower image similarity.
- Training requires running MAE-VQGAN to obtain VICL scores for each candidate, incurring substantial data preparation overhead.

## Related Work & Insights

- SupPR first introduced contrastive learning into VICL retrieval optimization; LaPR extends this by incorporating the label dimension.
- MoE mechanisms have been applied in NLP-based ICL (e.g., MOICL, MoD); this work is the first to adopt MoE for prompt retrieval in visual ICL.
- The finding on label consistency may inspire other ICL settings (e.g., few-shot retrieval in NLP) to reconsider the role of labels.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically address label roles in VICL retrieval; the problem identification is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three tasks, cross-fold transfer, cross-encoder generalization, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, systematic experimental design, and intuitive figures.
- **Value**: ⭐⭐⭐⭐ — Provides a new perspective and framework for VICL prompt selection, with potential generalization to other ICL scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)
- [\[CVPR 2026\] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](erecu_pseudolabel_evolution_unsupervised_camouflage.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](../../ICCV2025/segmentation/hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](insid3_training-free_in-context_segmentation_with_dinov3.md)

<!-- RELATED:END -->
