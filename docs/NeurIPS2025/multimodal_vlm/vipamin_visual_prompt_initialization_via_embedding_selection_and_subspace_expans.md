---
title: >-
  [Paper Note] VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion
description: >-
  [NeurIPS 2025][Multimodal VLM][Visual Prompt Tuning] This paper proposes VIPAMIN—a zero-extra-parameter visual prompt initialization strategy comprising two modules: attention-guided semantic Matching and orthogonal subs…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Visual Prompt Tuning"
  - "Parameter-Efficient Fine-Tuning"
  - "Self-Supervised Learning"
  - "Vision Transformer"
  - "Subspace Expansion"
date: 2026-05-08
content_hash: a5d06f2e8c28a796
---

# VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion

**Conference**: NeurIPS 2025
**arXiv**: [2510.16446](https://arxiv.org/abs/2510.16446)
**Authors**: Jaekyun Park, Hye Won Chung (KAIST)
**Code**: [iamjaekyun/vipamin](https://github.com/iamjaekyun/vipamin)
**Area**: Multimodal VLM
**Keywords**: Visual Prompt Tuning, Parameter-Efficient Fine-Tuning, Self-Supervised Learning, Vision Transformer, Subspace Expansion

## TL;DR

This paper proposes VIPAMIN—a zero-extra-parameter visual prompt initialization strategy comprising two modules: attention-guided semantic Matching and orthogonal subspace injection (Orthogonalizing). It addresses two failure modes of self-supervised VPT—prompt attention uniformization and subspace collapse—requiring only a single forward pass, and achieves state-of-the-art performance across 24 visual tasks.

## Background & Motivation

### State of the Field
Full fine-tuning of large-scale Vision Transformers (ViTs) is computationally expensive. Visual Prompt Tuning (VPT) offers a lightweight adaptation paradigm by prepending a small number of learnable tokens to a frozen backbone. However, VPT performs poorly on self-supervised pretrained models (MoCo-v3, MAE), particularly degrading on distribution-shifted tasks and few-shot scenarios.

### Limitations of Prior Work
- **VPT**: Randomly initialized prompts underperform full fine-tuning by an average of 34.6% on Structured tasks when using self-supervised backbones.
- **GatedPT**: Introduces additional learnable gating modules, increasing training overhead.
- **SPT**: Initializes prompts via K-means clustering; clustering on CUB-200-2011 takes approximately 27 days, making it computationally unacceptable.
- **iVPT/VFPT/DA-VPT**: Introduce architectural modifications such as attention reinforcement modules, Fourier transforms, and metric learning, increasing design complexity.

### Root Cause
The authors empirically identify two failure modes of VPT on self-supervised models:

**Attention Uniformization**: The attention entropy of prompts over all input tokens approaches the maximum value $\ln(N_e)$, preventing focus on semantically relevant regions.

**Subspace Collapse**: The row space of the prompt's value projection $\mathbf{P}_0\mathbf{W}_V$ is entirely covered by $\text{SA}(\mathbf{X}_0)$ (projection energy approaches 1), leaving no room to inject new representational directions.

Both problems are particularly severe on tasks with large distribution shifts (e.g., dSprites/loc) and in few-shot settings.

## Method

### Overall Architecture
VIPAMIN consists of two complementary modules. Initialization requires only a single forward pass and two lightweight matrix operations, introducing no additional learnable parameters.

### Module 1: Semantic Matching (Matching Module)
**Goal**: Ensure each prompt focuses on semantically consistent local regions from the initialization stage.

1. Sample $B$ images from the downstream training set and extract embeddings $\mathbf{E}_0 \in \mathbb{R}^{N_e \times d}$ via the frozen ViT (batch mean pooling).
2. For each Xavier-randomly initialized prompt $\mathbf{p}_i$, project it and the embeddings into the Key space $\mathbf{W}_K$ of the first transformer layer.
3. Compute cosine similarity and select the top-$k$ most similar token indices.
4. Initialize the prompt as the mean of the matched tokens: $\mathbf{p}_i^{\text{avg}} \leftarrow \frac{1}{k}\sum_{j=1}^{k}(\mathbf{E}_0)_{\alpha_j}$

**Key Insight**: In ViT-B/16, each token covers only 0.5% of the image area; semantically related tokens naturally cluster in Key space, so top-$k$ selection yields semantically coherent tokens. The hyperparameter $k$ controls attention locality—smaller $k$ yields more concentrated attention.

### Module 2: Orthogonal Subspace Injection (Orthogonalizing Module)
**Goal**: Enable prompts to express directions beyond the pretraining subspace, preventing subspace collapse.

1. Apply SVD to $\text{SA}(\mathbf{E}_0)$ to obtain the row-space basis $\mathbf{V}$.
2. Project the random prompt $\mathbf{p}_i$ through $\mathbf{W}_V$, remove its component in $\mathbf{V}$, and map back to the original space via the pseudoinverse of $\mathbf{W}_V$: $\mathbf{p}_i^{\text{orth}} \leftarrow (\mathbf{I} - \mathbf{V}\mathbf{V}^\top)(\mathbf{p}_i \mathbf{W}_V)(\mathbf{W}_V)^{\dagger}$
3. The final prompt is a weighted combination of the matched and orthogonal components: $\mathbf{p}_i^{\text{VIPAMIN}} \leftarrow (1-\lambda)\mathbf{p}_i^{\text{avg}} + \lambda \mathbf{p}_i^{\text{orth}}$

The hyperparameter $\lambda \in [0,1]$ controls the degree of orthogonalization. Tasks with larger distribution shifts require larger $\lambda$ (more novel directions), while similar tasks use smaller $\lambda$.

### Extension to VPT-Deep
In VIPAMIN-Deep, both the Matching and Orthogonalizing operations are applied separately to the input $\mathbf{X}_l$ of each layer, with a fixed prompt length of 20.

## Key Experimental Results

### Experiment 1: VTAB-1k Benchmark (19 Visual Classification Tasks)

| Method | Natural | Specialized | Structured | Mean |
|--------|---------|-------------|------------|------|
| **MoCo-v3 backbone** | | | | |
| Full Fine-tuning | 71.95 | 84.72 | 51.98 | 66.23 |
| VPT | 67.34 | 82.26 | 37.55 | 57.94 |
| GateVPT | 74.84 | 83.38 | 49.10 | 65.80 |
| SPT | 74.47 | 83.93 | 55.16 | 68.33 |
| **VIPAMIN** | **76.75** | **84.14** | **56.68** | **69.86** |
| **MAE backbone** | | | | |
| Full Fine-tuning | 59.31 | 79.68 | 53.82 | 61.28 |
| VPT | 39.96 | 69.65 | 27.50 | 40.96 |
| SPT | 62.53 | 80.90 | 53.46 | 62.58 |
| **VIPAMIN** | **62.60** | **79.96** | **57.47** | **64.09** |

- On MoCo-v3, VIPAMIN improves over VPT by **+19.13%** on Structured tasks and exceeds full fine-tuning by **+4.8%** on Natural tasks.
- On MAE, VIPAMIN is the first prompt method to surpass full fine-tuning across all VTAB categories.
- Compared to SPT, VIPAMIN improves mean accuracy by +1.5% on both MoCo-v3 and MAE, without requiring K-means clustering.

### Experiment 2: Few-Shot FGVC Classification (5 Fine-Grained Datasets)

| Method | k=1 Mean | k=2 Mean | k=4 Mean | k=8 Mean |
|--------|---------|---------|---------|---------|
| VPT | 18.1 | 27.6 | 31.7 | 41.7 |
| SPT/rand | 23.7 | 36.5 | 51.7 | 65.5 |
| **VIPAMIN** | **25.8** | **38.2** | **52.4** | **66.4** |

- At k=1, VIPAMIN outperforms VPT by **+7.7%**; at k=8, by **+24.7%**.
- VIPAMIN surpasses SPT/rand by 1–2% across all shot settings, without requiring full-dataset clustering.

### Ablation Study

| Matching | Orth | Natural | Specialized | Structured |
|----------|------|---------|-------------|------------|
| SPT baseline | — | 74.47 | 83.93 | 55.16 |
| Yes | — | 76.50 | 82.85 | 56.51 |
| Yes | Yes | **76.75** | **84.14** | **56.68** |

The Matching module primarily improves Natural tasks (+2.03%), while the Orthogonalizing module contributes notably to Specialized tasks (+1.29%); the two modules are complementary.

## Highlights & Insights

- **Zero-overhead design**: Only initialization weights are modified; no additional parameters, computational latency, or memory overhead are introduced, enabling seamless integration into existing VPT pipelines.
- **Theory-driven methodology**: Two quantitative diagnostics—attention entropy and projection energy—precisely identify VPT failure modes, and the proposed modules directly address each.
- **Minimal computational cost**: Only a single forward pass plus two matrix operations (top-$k$ selection + SVD orthogonalization) are required, negligible compared to SPT's K-means clustering (27 days).
- **Strong scalability**: Performance remains stable across ViT-B/L/H; VIPAMIN is the only method that consistently benefits from increasing prompt length.
- **First to surpass full fine-tuning on MAE**: VIPAMIN exceeds full fine-tuning on all VTAB categories under the MAE backbone.

## Limitations & Future Work

- **Classification tasks only**: Effectiveness on dense prediction tasks such as detection and segmentation has not been validated.
- **Task-dependent hyperparameter tuning**: Although a qualitative relationship between distribution shift and $k$/$\lambda$ is provided, no automatic selection mechanism exists.
- **ViT-only evaluation**: The method has not been validated on CNN or hybrid architectures.
- **Limited gains on Specialized tasks**: On MoCo-v3's Specialized group, VIPAMIN surpasses SPT by only 0.21%, and underperforms SPT on MAE.
- **Theoretical analysis limited to single-layer SA**: Multi-layer propagation is not theoretically analyzed, and how prompts in VPT-Shallow maintain semantic selectivity across subsequent layers is not thoroughly discussed.

## Related Work & Insights

- **VPT (Jia et al. 2022)**: Random Xavier initialization; VIPAMIN improves mean accuracy by +11.9% on MoCo-v3.
- **SPT (Wang et al. 2024)**: K-means clustering initialization with comparable performance but prohibitive cost (27 days vs. seconds); VIPAMIN surpasses it at a fraction of the cost.
- **GatedPT**: Learnable gating promotes inter-block interaction but falls short of VIPAMIN on both Natural and Structured tasks.
- **E2VPT**: Requires architectural modifications (e.g., token pruning); VIPAMIN-Deep achieves competitive performance without any architectural changes.
- **iVPT/VFPT/DA-VPT**: Introduce attention reinforcement, Fourier modulation, and metric learning respectively, increasing training complexity; VIPAMIN achieves comparable or superior performance at zero overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ — Designing an initialization strategy from two quantitative failure-mode metrics (attention entropy and subspace collapse) yields a clear motivation and an elegant solution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 19 VTAB tasks, 5 FGVC few-shot datasets, deep extension, multiple backbones, ablations, and Grad-CAM analysis; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative from failure-mode analysis to method design is exceptionally coherent; mathematical derivations are rigorous and clear.
- Value: ⭐⭐⭐⭐ — Highly practical (zero-overhead, plug-and-play), though limited to classification tasks and ViT architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[ICLR 2026\] Visual Prompt-Agnostic Evolution](../../ICLR2026/multimodal_vlm/visual_prompt-agnostic_evolution.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](../../ICCV2025/multimodal_vlm/pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](../../ICCV2025/multimodal_vlm/cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](../../ICCV2025/multimodal_vlm/attention_to_the_burstiness_in_visual_prompt_tuning.md)

</div>

<!-- RELATED:END -->
