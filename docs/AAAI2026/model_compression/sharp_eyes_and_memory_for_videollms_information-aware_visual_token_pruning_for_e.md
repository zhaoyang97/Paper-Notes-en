---
title: >-
  [Paper Note] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning
description: >-
  [AAAI 2026][Model Compression][Visual Token Pruning] SharpV proposes a two-stage training-free visual token pruning framework. In the Pre-LLM stage…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Visual Token Pruning"
  - "VideoLLM"
  - "KV Cache"
  - "Adaptive Pruning"
  - "Flash Attention"
date: 2026-05-08
content_hash: 7c96a4b69fc43783
---

# Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.08003](https://arxiv.org/abs/2511.08003)
**Code**: N/A
**Area**: Model Compression
**Keywords**: Visual Token Pruning, VideoLLM, KV Cache, Adaptive Pruning, Flash Attention

## TL;DR
SharpV proposes a two-stage training-free visual token pruning framework. In the Pre-LLM stage, it adaptively adjusts the pruning ratio per frame based on spatiotemporal information; in the Intra-LLM stage, it prunes the KV Cache based on a visual information degradation hypothesis. SharpV is the first method to achieve full compatibility with Flash Attention, retaining approximately 12% of tokens while matching or surpassing dense model performance across multiple video understanding benchmarks.

## Background & Motivation

### State of the Field
Video Large Language Models (VideoLLMs) have demonstrated strong capabilities in video understanding and reasoning, yet the abundant spatiotemporal information in long videos results in excessive input tokens to the LLM, causing quadratic computational complexity and KV Cache bloat.

### Limitations of Prior Work

**Fixed pruning ratio**: Existing methods (FastV, VTW, DyCoke, FrameFusion, etc.) apply a uniform fixed pruning rate, which is inherently a dataset-specific local optimum. They lack adaptive adjustment based on video information content, leading to insufficient generalization and robustness.

**Computational overhead**: Some methods rely on computationally intensive clustering algorithms (PruneVid) or complex planning techniques (LLaVA-Scissor, DivPrune), where the overhead introduced by pruning itself may offset the savings.

**Flash Attention incompatibility**: Most Intra-LLM-stage methods rely on exposed attention scores for pruning. Since Flash Attention hides attention scores to reduce complexity from $O(n^2 \cdot d)$ to $O(n \cdot d)$, these methods cannot be used in conjunction with it.

### Root Cause
How can information-aware adaptive pruning be achieved at low computational cost while remaining fully compatible with hardware acceleration techniques such as Flash Attention?

### Starting Point
From an information-theoretic perspective, rather than relying on attention scores, SharpV measures information degradation via the similarity between visual features and their original representations, enabling a two-stage hierarchical pruning strategy.

## Method

### Overall Architecture
SharpV is a two-stage plug-and-play framework:
- **Visual SharpV (Pre-LLM stage)**: Selects important visual tokens based on spatiotemporal importance scores, and adaptively determines the per-frame pruning ratio using L2 norm and a dissimilarity module.
- **Memory SharpV (Intra-LLM stage)**: Dynamically discards KV Cache entries by evaluating the degree of layer-wise visual information degradation.

### Key Designs

#### 1. **Dissimilarity Computation Module**
- **Mechanism**: Using $1 - \cos$ directly in high-dimensional space is problematic (cosine similarity of random vectors tends toward zero), so the Euclidean distance between unit vectors is used as the dissimilarity measure.
- **Key formula**: $\text{Dissim}(\mathbf{v}_1, \mathbf{v}_2) = \|\hat{\mathbf{v}}_1 - \hat{\mathbf{v}}_2\|_2 = \sqrt{2 - 2\cos(\theta)}$
- **Design Motivation**: This metric amplifies subtle directional differences between nearly aligned vectors, making it more suitable for token discrimination in high-dimensional space.

#### 2. **Spatiotemporal Token Importance Estimation**
- **Spatial importance $\mathcal{S}$**: Dissimilarity between each token and the frame-level mean representation, measuring spatial uniqueness.
    - $\mathcal{S} = \text{Dissim}(F_t, \overline{F_t})$
- **Temporal importance $\mathcal{T}$**: Dissimilarity between tokens at corresponding positions in adjacent frames, measuring motion change.
    - $\mathcal{T} = \text{Dissim}(\mathbf{F}_t, \mathbf{F}_{t-1})$
- **Combined score**: $\mathcal{I} = \mathcal{T} + w \cdot \mathcal{S}$, where $w$ controls the contribution of spatial information.
- **Complexity**: The entire estimation process is $O(n \cdot d)$, far below clustering-based $O(n^2)$ methods.

#### 3. **Information-Aware Adaptive Threshold Pruning**
- **Mechanism**: The L2 norm of temporal importance is used to quantify inter-frame visual variation, automatically determining the retention ratio per frame.
- **Threshold calculation**:
    - For frame $t$ ($t \geq 2$): $\text{threshold}_t = \frac{\|\mathcal{T}_t\|_2}{2\sqrt{f}}$
    - For frame 1: $\text{threshold}_1 = \frac{\|\mathcal{S}_1\|_2}{2\sqrt{f}}$
- **Effect**: High-motion sequences automatically retain more tokens; static frames are pruned aggressively.

#### 4. **Visual Information Degradation Hypothesis and Degradation-Aware Pruning (Memory SharpV)**
- **Key observation**: Visual tokens maintain high cosine similarity to original features in shallow layers, then drop sharply in deeper layers before stabilizing (below 0.2), analogous to the human memory curve; system and instruction tokens rapidly converge to near-zero similarity.
- **Theoretical interpretation**: Consistent with the information bottleneck principle—the network discards irrelevant details and retains task-relevant features through successive transformations.
- **Pruning strategy**: When the cosine similarity between visual tokens at layer $l$ and original visual features falls below threshold $M$, the KV Cache at that layer is discarded.
    - $\text{Discard}(l) = \text{True}, \text{if } \cos(\mathbf{V}_l, \mathbf{V}) < M$
- **Advantage**: Completely independent of attention scores, achieving full compatibility with Flash Attention.

### Loss & Training
- No training required; plug-and-play.
- Hyperparameters: $M=0.2$, $w=1$, manual mode $K=1.6$.

## Key Experimental Results

### Main Results

| Model/Method | Token Budget | MVBench | VideoMME(wo/mc) | NextQA | ActNet-QA | Avg. | TTFT Speedup |
|---|---|---|---|---|---|---|---|
| LLaVA-OV-7B (Dense) | 100% | 57.7 | 58.7/79.1 | 51.9 | 2.86 | 61.9 | 1.00× |
| + FastV | 30% | 56.3 | 56.4/76.5 | 50.7 | 2.80 | 60.0 | 1.30× |
| + DyCoke | 19% | 57.7 | 59.3/78.4 | 52.1 | 2.88 | 61.9 | 1.36× |
| + SharpV (Adaptive) | 12% | **58.2** | **60.0/78.8** | 51.9 | 2.86 | **62.2** | **1.64×** |
| LLaVA-OV-0.5B (Dense) | 100% | 46.6 | 45.9/57.5 | 47.9 | 2.66 | 49.5 | 1.00× |
| + SharpV (Adaptive) | 12% | 46.6 | 46.9/57.7 | 47.8 | 2.65 | 49.8 | 1.65× |

### Ablation Study

| Configuration | MVBench | VideoMME | Notes |
|---|---|---|---|
| Visual SharpV | 48.0/43.6 (PLLaVA) | 52.1/42.2 | Full spatiotemporal scoring |
| V-Random† (same ratio as SharpV) | 46.3/42.0 | 50.3/40.4 | Random selection validates scoring effectiveness |
| V-Random* (random ratio) | 45.0/41.3 | 49.4/40.2 | Removing adaptive ratio causes significant drop |
| LLaVA-OV-7B Visual SharpV | 58.2/59.5 | 70.8/57.4 | **Surpasses Dense (57.7/58.7)** |

### Key Findings
1. SharpV with approximately 12% token retention **occasionally surpasses dense models** by 1–2% on multiple benchmarks, demonstrating that appropriate pruning can reduce video noise.
2. Effectiveness is maintained on the 0.5B small model, confirming scalability.
3. In Memory SharpV, approximately 54% of visual token layers exhibit similarity below 0.1, validating the visual information degradation hypothesis.
4. Performance is optimal and stable when $w=1$ and $M \leq 0.2$.

## Highlights & Insights
1. **First two-stage pruning framework fully compatible with Flash Attention**: By not relying on attention scores, it provides a novel information-theoretic paradigm for Intra-LLM pruning.
2. **Adaptive pruning occasionally surpasses dense models**: This challenges the assumption that "more tokens are always better," demonstrating that appropriate denoising is beneficial.
3. **Visual information degradation hypothesis**: Reveals a regularity of cross-modal information flow in VideoLLMs—LLMs primarily process visual information in shallow layers.
4. **Minimal complexity**: Both spatiotemporal scoring and similarity computation are $O(n \cdot d)$, introducing virtually no additional overhead.

## Limitations & Future Work
1. Information loss may occur for subtle visual details (e.g., micro-expressions); finer-grained token pruning strategies could be explored.
2. The visual degradation phenomenon lacks a rigorous theoretical framework; the transformation mechanism of deep-layer tokens warrants further investigation.
3. Although hyperparameters ($w$, $M$) are robust, no automatic search mechanism is provided.
4. Effectiveness on longer videos (>100 frames) has not been verified.

## Related Work & Insights
- The high-complexity methods such as DivPrune and LLaVA-Scissor highlight an important lesson: the computational overhead of pruning methods themselves cannot be ignored.
- The information bottleneck theory holds broad potential for application in multimodal LLMs.
- Flash Attention compatibility is a necessary requirement for future efficiency methods in VLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ — The visual degradation hypothesis and adaptive pruning surpassing dense models are notable highlights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple models, multiple benchmarks, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-motivated arguments.
- Value: ⭐⭐⭐⭐ — Flash Attention compatibility offers strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[AAAI 2026\] Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)

</div>

<!-- RELATED:END -->
