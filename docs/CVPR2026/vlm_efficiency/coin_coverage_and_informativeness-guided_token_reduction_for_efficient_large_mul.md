---
title: >-
  [Paper Note] CoIn: Coverage and Informativeness-Guided Token Reduction for Efficient Large Multimodal Models
description: >-
  [CVPR 2026][Model Compression][Paper Note] This paper reformulates visual token reduction in large multimodal models (LMMs) as an "optimal subset selection" problem. It uses **informativeness** (visual saliency + cross-modal alignment) to score each token and **coverage** (log-det volume) to ensure the selected subset spans the feature space. A compact subset i
tags:
  - CVPR 2026
  - Model Compression
date: 2026-05-08
content_hash: b4e3cd58fccc97aa
---
# CoIn: Coverage and Informativeness-Guided Token Reduction for Efficient Large Multimodal Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.html)  
**Code**: Not yet released  
**Area**: Model Compression / Multimodal VLM  
**Keywords**: Visual token reduction, training-free inference acceleration, subset selection, submodular optimization, cross-modal alignment

## TL;DR
This paper reformulates visual token reduction in large multimodal models (LMMs) as an "optimal subset selection" problem. It uses **informativeness** (visual saliency + cross-modal alignment) to score each token and **coverage** (log-det volume) to ensure the selected subset spans the feature space. A compact subset is then selected end-to-end via greedy submodular optimization—requiring no training, being independent of attention mechanisms, and compatible with FlashAttention/KV cache. On LLaVA-NeXT-7B, pruning 94.4% of visual tokens retains 86.7% performance with a 6.5× prefill speedup.

## Background & Motivation
**Background**: LMMs encode images into a long sequence of visual tokens, which are concatenated with text tokens for LLM processing. However, the number of visual tokens explodes with resolution—LLaVA-1.5 uses 576 tokens per image, LLaVA-NeXT approaches 3K, and video models are even more extreme. Since LLM inference latency and VRAM scale quadratically with sequence length, these visual tokens become the primary bottleneck for deployment on mobile devices or interactive assistants. Training-free pruning of redundant visual tokens is the most practical acceleration method.

**Limitations of Prior Work**: Existing token reduction methods have inherent flaws. **Importance-based methods** (PDrop, SparseVLM, etc.) rely on attention weights or `[CLS]` similarity for scoring to keep top tokens—but high-attention regions are often highly correlated, leading to redundant, low-density information. Furthermore, relying on attention introduces bias and conflicts with efficient implementations like FlashAttention that do not expose the attention matrix. `[CLS]` scoring also ties the method to specific vision encoders. They also ignore text queries by focusing only on unimodal information. **Diversity-based methods** (DivPrune, DART, etc.) reduce redundancy via clustering or punishing pairwise similarity, but they treat all tokens equally, losing token saliency and cross-modal relevance, leading to the loss of critical information.

**Key Challenge**: Importance and diversity are **complementary but currently handled separately**. Recent hybrid methods (VisionZip, PruMerge, CDPruner) simply apply them in a **sequential pipeline**—scoring by importance followed by merging/clustering—which fails to address the inherent defects of each metric and does not achieve a global optimum.

**Goal**: Reformulate token reduction as a unified **optimal subset selection** problem where the selected subset ensures that "every token is important" and "the whole set spans the scene."

**Core Idea**: Jointly drive selection through two complementary criteria: **informativeness** (visual saliency + cross-modal alignment, avoiding attention and `[CLS]`) and **coverage** (volume criterion via log-det, measuring global representativeness by the volume spanned by the selected subset). These are coupled into a single objective solved via greedy submodular optimization.

## Method

### Overall Architecture
CoIn is a **single-stage, training-free** token selector inserted after the vision encoder/projector and before the LLM. Given an image, the vision encoder and projector produce $N$ visual tokens $F_V \in \mathbb{R}^{N\times d}$. The text query is encoded by the tokenizer and averaged to obtain a text representation $\bar F_T \in \mathbb{R}^{d}$. CoIn's task is to select a subset $S$ of size $K \ll N$ from the $N$ tokens such that the LLM output on $[S;T]$ approximates the output on the full set $[V;T]$.

It computes scores through two paths and merges them for solving: (1) **Informativeness Estimation** calculates a scalar score $s_{\text{info}}$ for each token, fusing visual saliency (feature norm) and cross-modal alignment (cosine similarity with text); (2) **Coverage Selection** uses the log-det volume to measure the global representativeness of a subset in the feature space. Finally, "sum of informativeness" and "coverage" are coupled into a unified objective and solved in one pass via **greedy submodular optimization** to select $K$ tokens.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Visual tokens F_V (N×d)<br/>+ Text mean F̄_T"] --> B["Informativeness Estimation<br/>Saliency ‖F_V‖_p + Alignment cos"]
    A --> C["Coverage·Volume Subset Selection<br/>Vol(S)=log det(FₛᵀFₛ+λI)"]
    B --> D["Couple Info and Coverage<br/>(1−α)Σs_info + α·log det"]
    C --> D
    D -->|Greedy Submodular Opt O(NK)| E["Compact Subset S (K tokens)"]
    E --> F["LLM Inference"]
```

### Key Designs

**1. Informativeness Estimation: Scoring tokens via "Visual Saliency + Cross-modal Alignment" without attention**

To address the bias and implementation conflicts of attention-based methods, CoIn uses two orthogonal cues. The first is **visual saliency** $s_{\text{vis}}$: approximated by the $p$-norm (default $p=2$) of the token embedding $s_{\text{vis}}=\lVert F_V\rVert_p$. High activation strength indicates a prominent token in the visual scene, independent of text. The second is **cross-modal alignment** $s_{\text{align}}$: calculated as the cosine similarity between each visual token and the mean text representation:

$$s_{\text{align}}=\frac{F_V \bar F_T^{\top}}{\lVert F_V\rVert\,\lVert \bar F_T^{\top}\rVert},$$

This injects query semantics (e.g., "shoes") into the scoring. Since the scales differ, both are min-max normalized to $[0,1]$ before convex combination: $s_{\text{info}}=\beta\, s_{\text{vis}} + (1-\beta)\, s_{\text{align}}$, where $\beta$ balances perceptual saliency and semantic relevance. This approach is attention-efficient, bias-free, and model-agnostic.

**2. Coverage·Volume Subset Selection: Ensuring tokens span the scene via log-det volume**

Selecting tokens based only on informativeness leads to spatial redundancy. Traditional diversity methods rely on Max-Min Diversity (punishing pairwise similarity), which is suboptimal for large-scale selection. CoIn adopts the **Volume-based Subset Selection (VSS)** criterion: measuring global representativeness by the geometric volume spanned by selected tokens in the feature space. For a subset $S$ of size $K$, the volume is defined as:

$$\text{Vol}(S)=\log\det\!\big(F_S^{\top}F_S + \lambda I\big),$$

where $F_S$ is the feature matrix of the selected tokens and $\lambda I$ is a small ridge term for stability. Maximizing this log-det term favors subsets that span a large volume, effectively preserving the global structure and covering various directions of the original set, naturally discouraging redundancy.

**3. CoIn Objective + Greedy Submodular Optimization: Unified optimization**

CoIn couples informativeness and coverage into a single joint objective. Both terms are min-max normalized to $[0,1]$. The objective is:

$$S^{*}=\arg\max_{S\subseteq V,\,|S|=K}\Big[(1-\alpha)\sum_{i\in S}s_{\text{info},i} + \alpha\,\log\det\!\big(F_S^{\top}F_S\big)\Big],$$

where $\alpha$ adjusts the weight between informativeness and diversity. Although solving this is combinatorially difficult, the objective is **monotone submodular**, guaranteeing that a simple greedy process yields a near-optimal solution. Implementation-wise, the determinant is approximated via QR decomposition with **incremental Gram–Schmidt** to avoid full recomputation, reducing complexity to $O(NK)$.

## Key Experimental Results

### Main Results
Evaluated on LLaVA-1.5-13B, LLaVA-NeXT-7B, Qwen2.5-VL-7B, and LLaVA-OneVision-7B (video) across 9 image benchmarks and 3 video benchmarks. "Avg." denotes performance relative to the original model.

Comparison on LLaVA-NeXT-7B (High resolution, up to 2880 tokens) with different budgets:

| Retained Tokens | Reduction Rate | Ours (CoIn) | Second-best Baseline | Gain |
|-----------------|----------------|-------------|----------------------|------|
| 640 | ↓77.8% | **94.0%** | VisionZip 92.7% | +1.3% |
| 320 | ↓88.9% | **90.0%** | VisionZip 88.6% | +1.4% |
| 160 | ↓94.4% | **86.7%** | VisionZip 83.5% | +3.2% |

On LLaVA-1.5-13B with extreme 94.4% reduction (32 tokens), CoIn retains 91.0% (vs. DivPrune 88.6%). On video tasks, LLaVA-OneVision-7B achieves 99.6% performance (nearly lossless) at 75% reduction.

### Efficiency Analysis
LLaVA-NeXT-7B + POPE, single A100 80GB, reducing 2880 tokens to 320:

| Method | Memory (GB) | Prefill | Speedup | Decoding | Speedup | F1 |
|--------|-------------|---------|---------|----------|---------|-----|
| Original | 16.8 | 233ms | 1.0× | 27ms | 1.0× | 86.5 |
| PDrop | 15.5 | 54ms | 4.3× | 23ms | 1.2× | 60.2 |
| VisionZip | 14.9 | 36ms | 6.5× | 21ms | 1.3× | 80.1 |
| DivPrune | 13.9 | 36ms | 6.5× | 21ms | 1.3× | 83.4 |
| **CoIn** | 14.1 | 36ms | **6.5×** | 21ms | **1.3×** | **85.4** |

### Ablation Study
Info vs. Coverage (LLaVA-1.5-7B, 64 tokens):

| Configuration | POPE | GQA | MME | Note |
|---------------|------|-----|-----|------|
| Original | 85.9 | 62.0 | 1508 | Full tokens |
| Info-only | 73.7 | 51.2 | 1288 | Redundant tokens lead to highest drop |
| Cov-only | 74.4 | 53.0 | 1307 | Diverse but misses info |
| **Combination** | **86.2** | **57.8** | 1378 | Full CoIn; POPE even exceeds original |

### Key Findings
- **Both criteria are essential and complementary**: Info-only and Cov-only perform significantly worse than the combination. Notably, CoIn (86.2) outperforms the original model (85.9) on POPE, suggesting a compact subset can improve grounding by removing noise.
- **Saliency vs. Alignment division of labor**: Perceptual tasks (VizWiz) favor both; tasks requiring strong grounding (RealWorldQA) rely more on cross-modal alignment, but the combination is always best.
- **Superiorness in extreme pruning**: At 160 tokens (94.4% reduction), CoIn leads VisionZip by 3.2%, much higher than the 1.3% lead at 640 tokens, showing global coverage modeling is critical at high compression levels.

## Highlights & Insights
- **Log-det volume as coverage is a shift from "pairwise similarity" to "collective subspace"**: While traditional diversity focuses on pairs, CoIn maximizes the volume spanned by the subset for global representativeness. Submodularity makes this both mathematically elegant and computationally cheap ($O(NK)$).
- **Bypassing attention is the most practical selling point**: By using norms for saliency and cosine similarity for relevance, CoIn is fully compatible with FlashAttention/KV cache and remains model-agnostic.
- **Compact subsets can outperform full tokens**: The result on POPE suggests that removing redundant or distracting tokens can "purify" grounding. Pruning is not just about cost—it can be a denoising process.

## Limitations & Future Work
- The specific incremental update formulas for the greedy step and the submodularity proof are in the supplementary material; the main text only provides the $O(NK)$ complexity.
- While performance is robust, the optimal $\alpha, \beta$ vary slightly by task. A "universal" set of parameters across all models/tasks was not explicitly provided.
- The token budget is pre-defined. The paper does not explore **adaptive** budgets based on image complexity or query difficulty.
- Volume criteria require computing determinants; the impact of ridge term $\lambda$ and numerical stability for extremely large $d$ or $K$ requires further analysis.

## Related Work & Insights
- **vs. Importance-based (PDrop / SparseVLM)**: These rely on attention/`[CLS]` and suffer from unimodal bias and incompatibility with FlashAttention. CoIn is model-agnostic and attention-independent.
- **vs. Diversity-based (DivPrune / DART)**: These maximize pairwise dissimilarity while ignoring saliency. CoIn uses log-det volume for global coverage without losing key information.
- **vs. Hybrid (VisionZip / PruMerge)**: These use sequential two-stage processing. CoIn treats it as a single joint submodular objective for global consistency.
- **vs. Video-specific (DyCoke / FrameFusion)**: CoIn serves as a general-purpose selector and outperforms specialized methods on video benchmarks (LLaVA-OneVision).

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulating token reduction as an informativeness+coverage submodular selection with log-det volume is clear and self-consistent.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 backbones × multiple reduction rates × 12 benchmarks (image/video) + efficiency/ablation/hyperparameter studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and diagrams are intuitive, though offloading greedy algorithm details to the supplement affects self-containment.
- Value: ⭐⭐⭐⭐⭐ Training-free, FlashAttention-compatible, 6.5× prefill speedup with virtually no loss; high industrial deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] SCoRe: Salience-Coverage Reduction for Vision Token Pruning in Vision-Language Models](score_salience-coverage_reduction_for_vision_token_pruning_in_vision-language_mo.md)
- [\[CVPR 2026\] Rethinking Token Reduction for Large Vision-Language Models](rethinking_token_reduction_for_large_vision-language_models.md)
- [\[CVPR 2026\] Merge3D: Efficient 3D Multimodal LLMs via Joint 2D-3D Token Merging](merge3d_efficient_3d_multimodal_llms_via_joint_2d-3d_token_merging.md)
- [\[CVPR 2026\] AdaSVD: Singular Value Decomposition with Adaptive Mechanisms for Large Multimodal Models](adasvd_singular_value_decomposition_with_adaptive_mechanisms_for_large_multimoda.md)
- [\[CVPR 2026\] IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models](if-prune_information-flow_guided_token_pruning_for_efficient_vision-language_mod.md)

</div>

<!-- RELATED:END -->
