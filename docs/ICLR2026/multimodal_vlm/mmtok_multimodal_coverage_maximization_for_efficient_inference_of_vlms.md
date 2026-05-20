---
title: >-
  [Paper Note] MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
description: >-
  [ICLR 2026][Multimodal VLM][vision token selection] This paper proposes MMTok, a multimodal visual token selection framework formulated as a Maximum Coverage Problem. By jointly leveraging text-visual and visual-visual c…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "vision token selection"
  - "coverage maximization"
  - "submodular optimization"
  - "VLM efficiency"
  - "token pruning"
date: 2026-05-08
content_hash: a4dd214e5ee0c0e0
---

# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs

**Conference**: ICLR 2026
**arXiv**: [2508.18264](https://arxiv.org/abs/2508.18264)  
**Code**: None  
**Area**: Multimodal / VLM
**Keywords**: vision token selection, coverage maximization, submodular optimization, VLM efficiency, token pruning

## TL;DR

This paper proposes MMTok, a multimodal visual token selection framework formulated as a Maximum Coverage Problem. By jointly leveraging text-visual and visual-visual coverage signals, MMTok selects the most informative subset of visual tokens in a training-free manner, significantly outperforming unimodal baselines and even surpassing methods that require fine-tuning.

## Background & Motivation

Vision-Language Models (VLMs) convert images into visual tokens and concatenate them with text tokens as input to an LLM. However, the number of visual tokens far exceeds that of text tokens—for instance, LLaVA-NeXT can generate up to 2,880 visual tokens per image, whereas a query such as "Describe the image" contains fewer than 10 text tokens. Since the computational complexity of self-attention in LLMs scales quadratically with the total number of tokens, the large volume of visual tokens severely limits inference efficiency.

The core limitation of existing visual token selection methods is their reliance on **unimodal information**:

- **Vision-only methods** (VisionZip, FastV): rank tokens using attention signals internal to the visual encoder (e.g., [CLS] token attention), ignoring semantic guidance from the text query.
- **Text-only methods** (SparseVLM): score tokens using text-to-visual attention, but ignore global image information.

A key observation is that **the same image requires different visual tokens for different text queries** (e.g., "What animal is this?" vs. "What is the background color?"), while **the same text instruction can apply to different images** (e.g., captioning tasks). Unimodal approaches are therefore inherently suboptimal, and both visual and textual information must be jointly utilized.

## Method

### Overall Architecture

MMTok formalizes visual token selection as a **Maximum Coverage Problem**, jointly covering textual semantics and global visual information via submodular function optimization:

1. Compute the text-visual similarity matrix $M^{tv}$ and the visual-visual similarity matrix $M^{vv}$.
2. Jointly optimize a multimodal coverage objective.
3. Solve efficiently using a greedy algorithm with a guaranteed $(1-1/e)$ approximation ratio.

### Key Designs

1. **Coverage Function Definition**:
$$f(\mathcal{S}; M) = \frac{1}{m} \sum_{i=1}^{m} \max M_{i,\mathcal{S}}$$
For each target token, this computes the maximum similarity to the selected subset, averaged over all target tokens. This function is proven to be a **submodular function** (Proposition 1), guaranteeing that the greedy algorithm achieves at least $(1-1/e) \approx 63.2\%$ of the optimal solution.

2. **Text-Visual Coverage (T-V Coverage)**:
    - Similarity matrix: $M_{i,j}^{tv} = \mathbf{t}_i^\top \mathbf{v}_j$, using projected visual tokens (aligned with text).
    - Objective: select visual tokens most semantically relevant to the text query.
    - Limitation: text queries may be uninformative (e.g., "Please describe the image"), providing insufficient semantic guidance.

3. **Visual-Visual Coverage (V-V Coverage)**:
    - Similarity matrix: $M_{i,j}^{vv} = \mathbf{v}_i^{\prime\top} \mathbf{v}_j'$, using pre-projection visual features (capturing purely visual similarity).
    - Objective: select a subset of visual tokens representative of the entire image.
    - Complementary to T-V coverage.

4. **Multimodal Coverage Fusion**:
    - Softmax calibration: $M_{i,j}^{tv'} = \frac{\exp(M_{i,j}^{tv}/\tau_t)}{\sum_j \exp(M_{i,j}^{tv}/\tau_t)}$
    - Joint objective: $f(\mathcal{S}; M^{tv'}, M^{vv'}) = f(\mathcal{S}; M^{tv'}) + \alpha \cdot f(\mathcal{S}; M^{vv'})$
    - **Corollary 1**: The sum of two submodular functions remains submodular, preserving the greedy algorithm's validity.
    - Default hyperparameters: $\tau_t=0.02$, $\tau_v=0.2$, $\alpha=0.5$.

5. **Optional Agent-Enhanced Text**:
    - A lightweight VLM (SmolVLM2-256M) generates a preliminary response.
    - The response tokens are appended to the original text tokens to enhance T-V coverage guidance.
    - Applicable when text queries carry insufficient information.

### Algorithmic Complexity

The greedy algorithm (Algorithm 1/2) involves only simple matrix operations (addition, multiplication, max). At each step, it selects the token with the largest marginal gain from the remaining candidates, iterating until $k$ tokens are selected. Empirical results show its runtime is comparable to simple methods such as VisionZip.

## Key Experimental Results

### Main Results (LLaVA-1.5-7B, 576 original tokens)

| Method | 192 tokens (retention) | 128 tokens (retention) | 64 tokens (retention) |
|---|---|---|---|
| FastV | 89.6% | 84.4% | 75.6% |
| SparseVLM | 95.5% | 92.9% | 86.9% |
| VisionZip | 97.9% | 96.8% | 93.2% |
| DivPrune | 98.0% | 97.0% | 94.8% |
| VisionZip🔥 (fine-tuned) | 98.4% | 97.7% | 95.0% |
| **MMTok** | **98.7%** | **97.9%** | **96.5%** |

### Cross-Model Generalization

| Model | Setting | VisionZip | DivPrune | MMTok |
|---|---|---|---|---|
| LLaVA-1.5-13B | 64 tokens | 93.7% | 95.4% | **96.3%** |
| LLaVA-NeXT-7B | Up 160 | 90.4% | 92.4% | **95.1%** |
| LLaVA-NeXT-13B | Up 160 | 91.4% | 92.1% | **95.1%** |
| Qwen-2.5-VL-7B | 20% | 94.2% | 91.5% | **94.6%** |

### Extreme Compression (LLaVA-1.5-7B, High-IC Datasets)

| # Tokens | VisionZip | DivPrune | MMTok |
|---|---|---|---|
| 16 | 78.3% | 86.2% | **88.3%** |
| 8 | 63.2% | 76.3% | **82.9%** |
| 4 | 58.8% | 66.3% | **76.7%** |
| 2 | 57.8% | 63.5% | **70.0%** |

With only 4 tokens, MMTok retains 87.7% of the original performance on POPE.

### Ablation Study

| Configuration | 64 tokens (retention) | Note |
|---|---|---|
| T-V only (no softmax) | 93.7% | Text-guided only |
| V-V only (no softmax) | 94.7% | Visual self-coverage only |
| T-V (softmax calibration) | 93.8% | Calibration does not hurt |
| V-V (softmax calibration) | 95.7% | Calibration yields marginal gain |
| **MMTok (T-V + V-V)** | **96.6%** | Multimodal complementarity is significant |

### Inference Efficiency (LLaVA-NeXT-13B, H100 GPU)

| Method | Total Inference Time | POPE Time | GPU Utilization | Runtime Memory | Avg. Performance |
|---|---|---|---|---|---|
| Full (2880) | 15204s | 1705s | 86.7% | 4.59GB | 100% |
| VisionZip (160) | 7551s | 866s | 52.4% | 1.92GB | 89.6% |
| DivPrune (160) | 8186s | 1060s | 50.9% | 1.23GB | 90.5% |
| **MMTok (160)** | **7768s** | **913s** | 58.0% | 1.78GB | **93.7%** |

MMTok achieves a 1.87× speedup while retaining 98.7% performance on POPE.

### Key Findings

- **Multimodal complementarity**: Combining T-V and V-V coverage outperforms any unimodal approach by 2–3%.
- **Image Contribution (IC) metric**: The paper introduces this metric to highlight that some datasets yield high performance even with zero visual tokens (e.g., SQA 82%, MMMU 92%), indicating that evaluation should focus on high-IC datasets.
- **Hyperparameter robustness**: Performance is insensitive to the choice of $\tau_t$, $\tau_v$, and $\alpha$; fixed default values are sufficient.
- **Training-free advantage**: MMTok surpasses fine-tuned methods such as VisionZip🔥 without any training.
- **Extreme compression potential**: At 4 tokens, MMTok outperforms VisionZip by 18%, demonstrating that coverage criteria yield greater advantages under extreme compression.

## Highlights & Insights

- **Formalizing token selection as a classical combinatorial optimization problem**: The theoretical framework of submodular functions combined with greedy algorithms is both elegant and practical.
- **Differentiated use of pre- and post-projection features**: Post-projection features serve cross-modal alignment (T-V), while pre-projection features capture purely visual similarity (V-V), reflecting a deep understanding of VLM architecture.
- **IC metric as a reflection on evaluation practice**: The paper identifies that datasets such as SQA and MMMU are unsuitable for evaluating visual token selection quality.
- **Agent augmentation**: Using the preliminary responses from a lightweight VLM as auxiliary signals is a novel idea, though its effectiveness varies by task type.

## Limitations & Future Work

- Token selection currently occurs only before LLM input; dynamic token pruning during LLM inference remains unexplored.
- The agent-augmented approach performs poorly on multiple-choice QA tasks, as responses such as "A" provide no meaningful guidance for token selection.
- Gains on Qwen-2.5-VL (which already incorporates token merging layers) are relatively modest, suggesting limited incremental value for already-optimized models.
- Despite its theoretical guarantees, the greedy algorithm may not represent the optimal optimization strategy.
- Extension to multi-frame scenarios such as video understanding has not been explored.

## Related Work & Insights

This paper introduces **submodular function optimization** (a classical combinatorial optimization theory) into the VLM acceleration setting. Compared to methods such as VisionZip ([CLS] attention ranking), FastV (intra-layer attention pruning), SparseVLM (text-to-vision attention), and DivPrune (diversity criteria), the coverage criterion is distinctive in that it **simultaneously optimizes relevance and coverage**. The use of SmolVLM2-256M as an agent also hints at an intriguing direction for small models assisting large model inference.

## Rating

- Novelty: ⭐⭐⭐⭐ (Coverage criterion + multimodal fusion is novel, though token pruning itself is not new)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 VLMs × 9 datasets × multiple compression ratios + extreme compression + efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear methodology, complete theoretical guarantees, thorough experiments)
- Value: ⭐⭐⭐⭐ (Highly practical; training-free, parameter-robust, and easy to deploy)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multimodal Classification via Total Correlation Maximization](multimodal_classification_via_total_correlation_maximization.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)

</div>

<!-- RELATED:END -->
