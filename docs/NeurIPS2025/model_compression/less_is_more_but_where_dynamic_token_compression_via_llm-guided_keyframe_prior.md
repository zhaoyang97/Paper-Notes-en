---
title: >-
  [Paper Note] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior
description: >-
  [NeurIPS 2025][Model Compression][video-understanding] This paper proposes DyToK, a training-free dynamic video token compression method that leverages query-conditioned keyframe priors inherent in the deep attention layers of VLLMs to adaptively allocate token budgets across frames, achieving plug-and-play optimal efficiency–accuracy trade-offs.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "video-understanding"
  - "token-compression"
  - "vllm"
  - "efficiency"
  - "keyframe-selection"
date: 2026-05-08
content_hash: 325ca3e6798bff09
---

# Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior

**Conference**: NeurIPS 2025
**arXiv**: [2512.06866](https://arxiv.org/abs/2512.06866)  
**Code**: None  
**Area**: Model Compression
**Keywords**: video-understanding, token-compression, vllm, efficiency, keyframe-selection

## TL;DR

This paper proposes DyToK, a training-free dynamic video token compression method that leverages query-conditioned keyframe priors inherent in the deep attention layers of VLLMs to adaptively allocate token budgets across frames, achieving plug-and-play optimal efficiency–accuracy trade-offs.

## Background & Motivation

1. **Computational bottleneck for long videos**: The computational complexity of Video LLMs scales quadratically with visual token sequence length, making long-video inference highly inefficient and limiting practical deployment.
2. **Uniform compression is suboptimal**: Existing encoder-feature methods (VisionZip, LLaMA-VID) apply a uniform compression ratio across all frames, ignoring temporal importance differences—keyframes are over-pruned while redundant frames retain excessive tokens.
3. **Instability of LLM attention-based pruning**: Methods such as FastV rely on attention maps from specific intermediate layers; shallow-layer noise causes unstable pruning quality, while deep-layer guidance forfeits efficiency advantages.
4. **Binary keyframe selection**: Existing keyframe selection methods (e.g., AKS) treat frame retention as a binary decision; information potentially useful in discarded frames is irreversibly lost, and redundancy in retained frames is left unaddressed.
5. **Additional computational overhead**: Prior keyframe selection approaches rely on pretrained VLMs or auxiliary modules to perform selection before feature encoding, introducing extra computational cost that undermines efficiency gains.
6. **Implicit keyframe priors in VLM attention**: Experiments reveal that VLLM attention layers naturally assign higher attention weights to query-relevant keyframes, and even when the model produces incorrect answers, attention still correctly localizes keyframes—yet this prior has not been fully exploited.

## Method

### 3.1 Overall Architecture

DyToK is a two-stage training-free framework:
1. **Temporal importance estimation**: Frame-level importance scores are obtained from deep-layer attention of a lightweight auxiliary model.
2. **Dynamic frame-level compression**: Token budgets are allocated proportionally to importance scores, after which any compatible token pruning method is invoked to execute the compression.

### 3.2 Temporal Importance Estimation

Cross-modal attention scores are computed between the last text query token and the visual tokens of each frame:

$$w_f = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \text{Softmax}\left(\frac{\mathbf{Q}_l \mathbf{K}_l^\top}{\sqrt{D}}\right)$$

**Key finding**: Deep-layer attention provides significantly better keyframe priors (layer 20 is optimal; see Ablation Study).

**Lightweight auxiliary model**: Using the deep-layer attention of the primary model to guide shallow layers requires recomputation, doubling the computational cost. The proposed solution uses a 0.5B model from the same architecture family as the auxiliary, achieving comparable keyframe identification accuracy at only 1/14 of the computational cost.

### 3.3 Dynamic Frame-Level Compression

The token budget allocation algorithm proceeds as follows:
1. **Initial allocation**: $a_f = \lfloor \hat{w}_f \times T_{\text{total}} \rfloor$
2. **Remainder allocation**: The fractional remainder $r_f$ for each frame is computed, and remaining tokens are allocated in descending order of remainder.
3. **Upper-bound truncation**: Tokens exceeding the per-frame ceiling $T_{\max}$ are redistributed to frames below their budget in order of importance rank.
4. **Modular compression**: The actual pruning is executed via $\text{Compression}(x_f, a_f)$, which can be replaced in a plug-and-play manner with any compatible method such as VisionZip, FastV, or DyCoke.

### 3.4 Design Highlights

- Breaks the binary selection paradigm: frames are no longer "kept or discarded" but instead assigned "more or fewer" tokens.
- Training-free: entirely exploits intrinsic attention mechanisms of pretrained models.
- Plug-and-play: compatible with both encoder-feature methods and LLM attention-based methods.

## Key Experimental Results

### Table 1: Enhancement over Encoder-Feature Methods (LLaVA-OneVision, 32 frames)

| Method | Retained Tokens | Compression | VideoMME | LongVideoBench | MLVU | Avg. | Relative% |
|--------|----------------|-------------|----------|---------------|------|------|-----------|
| Vanilla | 6272 | — | 58.5 | 56.6 | 47.1 | 54.1 | 100 |
| VisionZip | 3136 | ↓50% | 57.2 | 53.8 | 43.7 | 51.6 | 95.4 |
| VisionZip†+DyToK | 3136 | ↓50% | 59.1 | 56.4 | 46.2 | **53.9** | **99.6** |
| VisionZip | 448 | ↓90% | 44.5 | 41.4 | 29.8 | 38.6 | 71.3 |
| VisionZip†+DyToK | 448 | ↓90% | 53.2 | 50.4 | 42.8 | **48.8** | **90.2** |

**Key Findings**: DyToK yields a 4.2% improvement at 50% compression and an 18.9% improvement at the extreme 90% compression ratio; gains increase as the compression ratio rises.

### Table 2: Enhancement over LLM Attention-Based Methods

| Method | Retained Tokens | Compression | VideoMME | LongVideoBench | MLVU | Avg. | Relative% |
|--------|----------------|-------------|----------|---------------|------|------|-----------|
| FastV | 4704 | ↓25% | 57.6 | 57.1 | 46.5 | 53.7 | 99.3 |
| FastV+DyToK | 4704 | ↓25% | 58.4 | 56.8 | 46.8 | **54.0** | **99.8** |
| FastV | 896 | ↓85% | 51.1 | 51.2 | 38.3 | 46.9 | 86.7 |
| FastV+DyToK | 896 | ↓85% | 54.8 | 52.6 | 43.2 | **50.2** | **92.8** |

**Key Findings**: Applied to LLM attention-based methods, DyToK improves FastV by 6.1% at 85% compression, demonstrating cross-paradigm compatibility.

### Ablation Study

- **Attention layer position**: Layer 20 (out of 24) provides the best keyframe prior; shallow layers (0–8) perform poorly, validating the hypothesis that deep-layer attention encodes high-level semantics.
- **Auxiliary model size**: The 0.5B model achieves nearly identical keyframe prior quality compared to the 7B model (performance gap ≤1.5%) while incurring only 1/14 of the computational cost.

## Highlights & Insights

- Discovers and validates query-conditioned keyframe priors inherent in VLLM deep attention: even when the model produces incorrect answers, attention correctly localizes keyframes.
- Paradigm shift from binary frame selection to continuous token budget allocation, enabling finer-grained balance between efficiency and information retention.
- Retains 90.2% of the uncompressed model's accuracy at 90% extreme compression, with up to 4.3× speedup.
- Zero-training, plug-and-play design compatible with both major categories of compression methods, with minimal barriers to practical adoption.

## Limitations & Future Work

- An additional lightweight auxiliary model (0.5B) is still required to extract keyframe priors, adding system complexity without fully eliminating extra model overhead.
- Validation is limited to a small number of architectures (LLaVA-OneVision, Qwen2.5-VL); generalizability to other video LLMs (e.g., VideoChat, Video-LLaMA) remains unknown.
- The quality of keyframe priors depends on architectural homogeneity between the auxiliary and primary models; cross-family transferability is unexplored.
- Evaluation is primarily conducted on multiple-choice benchmarks (VideoMME, LongVideoBench, MLVU); performance on open-ended generation tasks is not assessed.
- For videos with minimal visual variation (e.g., surveillance footage), the benefit of importance-based per-frame token allocation may be limited.

## Related Work & Insights

### vs. FastV (ECCV 2024)
FastV dynamically prunes visual tokens during LLM inference based on attention maps from specific intermediate layers, but its effectiveness is highly sensitive to layer selection—shallow-layer attention noise leads to unstable pruning. DyToK does not modify the inference process itself; instead, it pre-allocates frame-level token budgets via deep-layer attention of an auxiliary model before inference, making it orthogonal and complementary to FastV. DyToK improves FastV by 6.1% at 85% compression.

### vs. VisionZip (CVPR 2025)
VisionZip statically selects tokens based on inter-patch correlations in encoder features but applies a uniform compression ratio across all frames, ignoring temporal dynamics. Furthermore, the original VisionZip discards spatial positional information, making it incompatible with mainstream 2D pooling approaches. DyToK introduces dynamic per-frame compression ratios, and when combined with the improved VisionZip†, achieves an 18.9% gain at 90% compression.

### vs. Keyframe Selection Methods (AKS et al.)
Methods such as AKS employ pretrained VLMs to perform binary frame selection before feature encoding, incurring additional computational overhead and irreversibly discarding potentially useful information from dropped frames. DyToK replaces the binary decision with continuous token budget allocation, retaining a small number of tokens even for "unimportant" frames to avoid irreversible information loss.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Paradigm shift from binary frame selection to continuous token budget allocation.
- ⭐⭐⭐⭐ **Technical Quality**: Keyframe prior discovery has theoretical depth; ablation study is thorough.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Comprehensive evaluation across three benchmarks × two method categories × multiple compression ratios.
- ⭐⭐⭐⭐ **Reproducibility**: Training-free; code is open-sourced and easy to integrate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](../../ACL2025/model_compression/llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)
- [\[NeurIPS 2025\] KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference](kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)
- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[ACL 2026\] Quantize What Counts: More for Keys, Less for Values](../../ACL2026/model_compression/quantize_what_counts_more_for_keys_less_for_values.md)

</div>

<!-- RELATED:END -->
