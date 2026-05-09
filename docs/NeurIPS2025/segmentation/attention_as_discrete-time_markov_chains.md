---
title: >-
  [Paper Note] Attention (as Discrete-Time Markov) Chains
description: >-
  [NeurIPS 2025][Segmentation][Attention Markov Chains] This work reinterprets the softmax-normalized attention matrix as the transition probability matrix of a Discrete-Time Markov Chain (DTMC), and proposes Multi-Bounce Attention and TokenRank (stationary distribution, analogous to PageRank) to capture indirect attention paths and global token importance. The approach achieves 94.29% mAP on ImageNet segmentation and enhances image generation quality in Self-Attention Guidance.
tags:
  - NeurIPS 2025
  - Segmentation
  - Attention Markov Chains
  - TokenRank
  - PageRank
  - Multi-Bounce Attention
  - Image Segmentation
date: 2026-05-08
content_hash: d06e723810398b7e
---

# Attention (as Discrete-Time Markov) Chains

**Conference**: NeurIPS 2025
**arXiv**: [2507.17657](https://arxiv.org/abs/2507.17657)
**Code**: [https://yoterel.github.io/attention_chains/](https://yoterel.github.io/attention_chains/)
**Area**: Attention Analysis / Visualization
**Keywords**: Attention Markov Chains, TokenRank, PageRank, Multi-Bounce Attention, Image Segmentation

## TL;DR
This work reinterprets the softmax-normalized attention matrix as the transition probability matrix of a Discrete-Time Markov Chain (DTMC), and proposes Multi-Bounce Attention and TokenRank (stationary distribution, analogous to PageRank) to capture indirect attention paths and global token importance. The approach achieves 94.29% mAP on ImageNet segmentation and enhances image generation quality in Self-Attention Guidance.

## Background & Motivation

**Background**: Attention analysis relies on direct operations — row selection (which tokens a token attends to), column selection (which tokens attend to a given token), and summation (global aggregation). These capture only first-order/direct attention effects.

**Limitations of Prior Work**: Analogous to PageRank's insight into web hyperlinks — direct link counting is inferior to propagation-based importance estimation. Attention also exhibits indirect influence paths: token A attends to B, and B attends to C, yet direct operations cannot reveal the indirect relationship A→C.

**Key Challenge**: Row and column operations on the attention matrix are first-order and neglect higher-order indirect paths. A mathematical framework is needed to unify direct and indirect attention effects.

**Goal**: To provide a Markov chain-based framework for attention analysis that captures multi-order indirect dependencies and global token importance.

**Key Insight**: The softmax-normalized attention matrix naturally satisfies the definition of a Markov chain transition probability matrix (non-negative entries, rows summing to one), enabling direct reuse of Markov chain and PageRank theoretical tools.

**Core Idea**: Attention matrix = DTMC transition matrix → Multi-hop propagation = $k$-th matrix power → Stationary vector = TokenRank (analogous to PageRank) → $\lambda_2$-weighted mixture of multiple heads.

## Method

### Overall Architecture
Attention matrix $A$ (softmax output) → **Multi-Bounce**: $\mathbf{v}_{i,n+1}^T = \mathbf{v}_{i,n}^T A$ ($n=1$ reduces to standard row selection) → **TokenRank**: compute the stationary vector $\pi$ of $A$ satisfying $\pi^T A = \pi^T$ → **$\lambda_2$ weighting**: weight the contribution of each attention head by the magnitude of the second-largest eigenvalue → Apply to segmentation / generation / token masking.

Core insight: The softmax-normalized attention matrix naturally satisfies the Markov chain transition probability definition (non-negative, rows sum to one), allowing all theoretical tools from PageRank to be directly transferred.

### Key Designs

1. **Multi-Bounce Attention**:

    - Function: Captures indirect attention paths via matrix power propagation.
    - Mechanism: For token $i$, initialize a one-hot vector $\mathbf{v}_{i,0} = \mathbf{e}_i$ and iterate $\mathbf{v}_{i,n+1}^T = \mathbf{v}_{i,n}^T A$. $n=1$ corresponds to standard row selection (direct attention); $n=2$ incorporates second-order indirect paths; $n \to \infty$ converges to the stationary distribution.
    - Design Motivation: In image segmentation, a pixel may not directly attend to the center of its object, but can be indirectly associated through intermediate pixels — Multi-Bounce captures such indirect relationships.

2. **TokenRank (Stationary Distribution)**:

    - Function: Computes the global importance of each token (analogous to PageRank).
    - Mechanism: The stationary vector $\pi$ is obtained via power iteration or the PageRank correction ($P' = \alpha P + (1-\alpha)\frac{1}{n}\mathbf{e}\mathbf{e}^T$, ensuring ergodicity and aperiodicity). A higher $\pi(i)$ indicates that token $i$ is more "important" in attention propagation.
    - Design Motivation: To assess global token influence rather than local relationships — "Which tokens are most central in the overall attention graph?"

3. **$\lambda_2$-Weighted Multi-Head Aggregation**:

    - Function: Weights the contribution of each attention head by the second-largest eigenvalue $\lambda_2$.
    - Mechanism: A larger $\lambda_2$ indicates slower convergence of the Markov chain (more metastable states), corresponding to attention heads with richer multi-scale structure. These heads are assigned greater weight via $\lambda_2$.
    - Design Motivation: Not all attention heads are equally important for downstream tasks — $\lambda_2$ provides an unsupervised estimate of head importance.

### Loss & Training
- A purely analytical method; no training is required.
- TokenRank is computed via 10–20 power iterations.

## Key Experimental Results

### Main Results

| Task | Method | Accuracy | mIoU | mAP |
|------|--------|----------|------|-----|
| ImageNet Segmentation | Concept Attention | 83.07% | 71.04% | — |
| | **Ours (FLUX DiT)** | **84.12%** | 70.20% | **94.29%** |
| SAG Image Generation | SD1.5 base | IS 16.32 | — | — |
| | **TokenRank** | IS **18.37** | — | — |
| DiffSeg | Uniform sampling | 72.50 mACC | 43.60 mIoU | — |
| | **TokenRank grid** | **84.97** mACC | **44.87** mIoU | — |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| $\lambda_2$ weighting vs. uniform | Statistically significant improvement (test E.1) |
| $n=1$ (standard) vs. $n=2$ (two-hop) | $n=2$ is optimal for segmentation |
| Structured features (DINOv2) vs. unstructured (ViT) | DINOv2 + TokenRank yields greater gains |
| Token Masking | Removing TokenRank tokens → AUC 0.26–0.64 (accuracy drops faster than baseline 0.27–0.79), confirming that TokenRank identifies genuinely important tokens |

### Key Findings
- Multi-Bounce attention ($n=2$) outperforms direct attention ($n=1$) on segmentation tasks — indirect paths do carry useful information.
- TokenRank significantly improves generation quality over random seed sampling in SAG — using "important tokens" as guidance is more effective.
- Structured attention (DINOv2 + registers) benefits more from TokenRank — suggesting that Markov chain analysis requires a certain degree of structural regularity in the attention.
- $\lambda_2$ weighting yields modest but statistically significant improvements — automated head selection is valuable.

## Highlights & Insights
- **The PageRank-to-TokenRank analogy is highly natural**: The row-stochastic property of the softmax-normalized attention matrix exactly satisfies the Markov chain definition, allowing all theoretical tools from PageRank to be directly transferred.
- **Multi-Bounce attention reveals "attention of attention"**: A token's true influence encompasses not only what it directly attends to, but also what it reaches indirectly through other tokens — this carries deep implications for understanding information flow in Transformers.
- **A unified framework spanning multiple applications**: Segmentation (Multi-Bounce), generation guidance (TokenRank + SAG), and token importance analysis (masking) — one theoretical framework, multiple applications.

## Limitations & Future Work
- Applicable only to square matrices (self-attention / mixed attention); cross-attention introduces unreachable states, requiring theoretical extension.
- Computing $\lambda_2$ is expensive for large matrices, though Multi-Bounce and TokenRank are efficient (10–20 iterations).
- Limited gains on unstructured attention (e.g., standard ViT without registers), suggesting that Markov chain analysis requires some structural prior in the attention.
- Automatic selection of the optimal number of hops for Multi-Bounce is unexplored; $n=2$ is currently set manually.
- Cross-layer Markov chain properties of attention are not analyzed; the current work operates within a single layer.
- The stationary vector may not be unique (periodic chains or disconnected attention graphs) — PageRank correction is required to guarantee uniqueness.

## Related Work & Insights
- **vs. Attention Rollout**: Rollout performs cross-layer attention propagation by accumulating along layer depth but does not conduct Markov chain analysis.
- **vs. GradCAM**: GradCAM requires gradients, whereas TokenRank is entirely gradient-free.
- **vs. Concept Attention**: Concept Attention performs attention analysis for specific concepts; TokenRank provides a general global importance assessment.
- **Transferability**: TokenRank can be applied to any Transformer — not limited to vision; text and audio Transformers are equally applicable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Markov chain perspective is entirely novel; the PageRank→TokenRank transfer is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three applications (segmentation + generation + masking) with ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and analogies are intuitive.
- Value: ⭐⭐⭐⭐ Provides a new mathematical framework for attention analysis with deep implications for understanding information flow in Transformers.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models](mechanistic_interpretability_of_rnns_emulating_hidden_markov_models.md)
- [\[NeurIPS 2025\] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition](interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)
- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](../../ICCV2025/segmentation/online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](../../ICCV2025/segmentation/topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)

<!-- RELATED:END -->
