---
title: >-
  [Paper Note] MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention
description: >-
  [ICML 2025][Multimodal Efficiency][long-context VLM] This paper proposes MMInference, which accelerates the prefill stage of long-context VLMs by up to $8.3\times$ in a $1\text{M}$ token scenario without modifying model weights or fine-tuning, while maintaining task accuracy. This is achieved via "modality-aware permutation sparse attention + head-level offline pattern search + online dynamic indexing + customized GPU kernels."
tags:
  - "ICML 2025"
  - "Multimodal Efficiency"
  - "long-context VLM"
  - "prefill acceleration"
  - "dynamic sparse attention"
  - "modality boundary"
  - "Grid Pattern"
date: 2026-05-08
content_hash: e6c3cda5c696fd0f
---

# MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention

**Conference**: ICML 2025  
**arXiv**: [2504.16083](https://arxiv.org/abs/2504.16083)  
**Code**: [https://aka.ms/MMInference](https://aka.ms/MMInference)  
**Area**: multimodal_vlm (long-context VLM inference acceleration)  
**Keywords**: long-context VLM, prefill acceleration, dynamic sparse attention, modality boundary, Grid Pattern

## TL;DR
This paper proposes MMInference, which accelerates the prefill stage of long-context VLMs by up to $8.3\times$ in a $1\text{M}$ token scenario without modifying model weights or fine-tuning, while maintaining task accuracy. This is achieved via "modality-aware permutation sparse attention + head-level offline pattern search + online dynamic indexing + customized GPU kernels."

## Background & Motivation

### Background
Long-context VLMs (video-and-text, image-and-text) are becoming core capabilities of practical systems, as many tasks naturally require long-horizon and cross-modality joint modeling.
However, in real deployments, Time-to-First-Token (TTFT) is dominated by the prefill stage, where the complexity of attention calculation is quadratic.
When the context scales to $128\text{k}$ or even $1\text{M}$ tokens, the prefill computational time reaches minutes, which significantly degrades system usability.

### Limitations of Prior Work
Various sparse attention and dynamic indexing methods have been proposed for text-based LLMs (e.g., MInference).
However, directly transferring these methods to VLMs introduces two critical issues:
- The attention sparse patterns of VLMs are not only "sparse" but also constrained by visual spatial-temporal structures, displaying unique grid patterns (Grid Pattern).
- Multi-modal inputs possess modality boundaries, leading to significant distributional differences between cross-modal and intra-modal attentions. Directly applying text-specific sparse patterns will hurt recall and accuracy.

### Key Challenge
There exists a trade-off between speed (requiring sparsification) and accuracy (requiring preserving key attention).
In VLMs, key attention is both "dynamic" (query-dependent) and "modality-aware" (with distinct variations across modalities).
Thus, "fixed sparse templates" are insufficient; dynamic and modality-aware sparse structures must be constructed.

### Key Insight
Based on observations of attention map structures, this work proposes:
- Utilizing the grid patterns brought by visual locality.
- Explicitly handling two types of modality boundaries: Q-Boundary and 2D-Boundary.
- Applying permutations to transform discontinuous sparse blocks into more continuous, highly-sequential access patterns that can be efficiently accelerated by GPU kernels.

## Method

## Overall Architecture
The overall workflow of MMInference consists of four steps:
1. **Offline Phase**: Search for the optimal combination of sparse patterns (Grid / A-shape / Vertical-Slash / Boundary-related) for each attention head.
2. **Online Phase**: Dynamically estimate sparse indexes based on the current input (instead of reusing top-k patterns from other requests).
3. **Permutation Phase**: Perform row, column, or modality permutations on Q/K/V according to the head and modality boundary types to obtain orderly sparse memory access.
4. **Computation Phase**: Execute hardware-friendly sparse computation inside custom GPU kernels with dynamic loading/writing-back to avoid explicit tensor transposition overhead.

This design represents a classic "algorithm-system co-design":
- **Algorithmically**, it guarantees that the sparse patterns closely approximate the original attention.
- **Systematically**, it ensures that the sparse patterns map efficiently onto the kernels, achieving actual end-to-end speedups rather than just theoretical FLOPs reduction.

### Key Observation A: VLM Attention Is Dynamically Sparse
The paper provides key statistics:
- In a $128\text{k} \times 128\text{k}$ VLM attention matrix, preserving only the top $5.78\%$ of weights yields $95\%$ attention recall.
- In comparison, text LLMs require about $1.79\%$ under the same target recall.
Explanation: VLMs are sparse but denser than pure text LLMs, indicating that cross-modal interactions introduce additional valid connections.

Another crucial point:
- The "degree" of sparsity can generalize, but the "position" of sparsity is highly dynamic.
- Reusing top-k indexes from one request to another leads to a severe drop in recall.
Therefore, online dynamic indexing is indispensable.

### Key Observation B: Spatial Structure of Grid Heads
The authors discover that some heads exhibit structured grid patterns:
- Horizontal and vertical lines are equally spaced and often approximately symmetrical.
- This stems from the locality of visual tokens in spatial and temporal dimensions.

These patterns differ significantly from general text attention. If generic sparse templates are used, crucial positions can be missed. MMInference's strategy:
- Estimate the stride and phase of the Grid online.
- Aggregate grid-related positions via row/column permutations.
- Perform sparse attention computations.

### Key Observation C: Modality Boundaries
The study categorizes boundary patterns into four types:
- No-Boundary
- K-Boundary
- Q-Boundary
- 2D-Boundary

The most challenging are Q-Boundary and 2D-Boundary, as they sever the continuous, extrapolative sparse structures within the same modality. The core approach features "modality permutation":
- **Q-Boundary**: Group and permute the query dimension to aggregate queries from the same modality.
- **2D-Boundary**: Process both query and key dimensions to address cross-modality fragmentation, aiming to restore continuous, modelable sparse blocks.

### Key Designs

### Module 1: Grid Head (Algorithm 1 in the paper)
- **Function**: Capture grid attention dynamically for vision-dominated heads.
- **Mechanism**:
	- Construct an approximate attention map using a small number of queries (e.g., `last_q = 64`).
	- Search for the optimal stride/phase within candidate stride sets.
	- Perform permutation and block-sparse computation based on the stride/phase.
- **Design Motivation**:
	- Grid patterns correspond to visual locality, rather than random sparsity.
	- Approximating with a few queries keeps indexing overhead within an acceptable limit.

### Module 2: Hybrid Modality Sparse Attention
- **Function**: Handle discontinuous sparsity caused by modality boundaries in a unified manner.
- **Mechanism**:
	- Use global intra-modality sparse patterns for No/K-Boundary.
	- Use row permutations for Q-Boundary to aggregate queries of the same modality, followed by intra-modality sparse approximation.
	- Handle two-dimensional boundaries for 2D-Boundary to avoid cross-modality fragmentation from breaking index continuity.
- **Design Motivation**:
	- Under multi-modal inputs, boundaries are a primary source of error.
	- Without boundary awareness, dynamic sparsity would disconnect related regions and connect unrelated ones.

### Module 3: Modality-Aware Sparse Attention Search
- **Function**: Select patterns for each head offline, while performing lightweight dynamic indexing online.
- **Mechanism**:
	- Search offline for combinations of inter-modality and intra-modality sparse patterns.
	- Build indexes online based on inputs to resolve mismatches caused by cross-request reuse.
- **Design Motivation**:
	- Purely online searching is too slow.
	- Purely offline fixed templates offer unstable accuracy.
	- The hybrid strategy balances throughput and accuracy.

### Loss & Training
This work is an inference-stage acceleration method; it does not require changing backbone parameters or fine-tuning, which is critical for production deployment:
- No modifications to model weights result in a low integration cost.
- It can serve as an "inference-layer enhancement plug-in" for existing long-context VLMs.

## Key Experimental Results

Note: The following experimental details are retrieved strictly from the local cache. The current cache truncates at Algorithm 2 and does not contain the fully detailed experimental sections or all tables. Therefore, only the verified metrics explicitly appearing in the cached text are documented here, without generating hypothetical numbers.

### Main Results

| Dimension | Conclusion / Value | Remarks |
|---|---:|---|
| Target Acceleration Phase | Prefill (long-context multimodal inputs) | Focuses on TTFT bottlenecks |
| Max Acceleration (1M tokens) | $8.3\times$ vs. FlashAttention-2 | Explicitly stated in the cache text |
| Relative to MInference | $1.7\times$ | Under the same $1\text{M}$ tokens scenario |
| Model Coverage | LongVila, Llava-Video, VideoChat-Flash, Qwen2.5-VL | Covers 4 long-context VLMs |
| Task Coverage | Video QA, Captioning, Vision-NIAH, Mixed-Modality-NIAH | Diverse video/multimodal tasks |
| Accuracy Description | Achieves acceleration while maintaining accuracy | Qualitative description in the cache |

### Ablation Study

| Analysis Item | Cached Evidence | Implications for Design |
|---|---|---|
| Sparse Recall Requirement | VLMs under $128\text{k}$ context can reach $95\%$ recall with top $5.78\%$ weights | Proves feasibility of sparsification, but requires dynamic indexing |
| Difference from Text LLMs | Text LLMs require around $1.79\%$ (for the same $95\%$ recall) | VLMs are "denser"; borrowing text-based sparse configurations directly is unsuitable |
| Head Heterogeneity | $52.3\%$ of heads can recover major attention with $<2\%$ weights | Head-level pattern allocation is needed, rather than uniform templates |
| Dynamicity Test | Reusing top-k indexes across different queries drops recall significantly | Online index estimation is necessary |
| Boundary Structure | Explicitly identifies Q-Boundary and 2D-Boundary | Modality-aware permutation is an essential component |

### Key Findings
- **Finding 1**: Long-context VLM attention exhibits significant redundancy but is highly structured; structured-aware sparsity is required rather than simple top-k.
- **Finding 2**: Spatial-temporal locality of visual tokens forms grid heads. The Grid pattern serves as an important additional prior for VLMs compared to text models.
- **Finding 3**: Modality boundaries are major reasons for the failure of sparse extrapolation. Permutation can reconstruct discontinuous patterns into highly-efficient continuous blocks.
- **Finding 4**: At the $1\text{M}$ token level, system-level implementations (kernel-level dynamic loading/writing-back) are as crucial as algorithmic improvements.

## Detailed Commentary on Method and Experiments

### Why This Work Has High Engineering Value
It does not just report a sparse pattern but strings together "observation-pattern-indexing-kernel" into a complete pipeline. This and similar works are more deployable for online systems because they address three common failure points:
- What to do when sparse patterns are unstable?
- What to do when modality boundaries cause accuracy loss?
- What to do when sparse FLOPs decrease but end-to-end latency does not?

### Potential Sources of Error (Inferred from Cache)
- If modality boundary estimation is inaccurate, the permutation of Q/2D Boundary may introduce erroneous aggregation.
- If online estimation of Grid stride/phase mismatches, critical attention recall may be lost.
- Under extremely mixed inputs, boundaries and grids may interfere with each other, increasing the complexity of pattern selection.

### Relationship with MInference
It can be understood as a "multimodal- and boundary-aware enhanced version" of dynamic sparse inference:
- MInference excels at the dynamic sparse framework and kernel implementation.
- MMInference systematically incorporates VLM-specific structures (Grid + boundary).

## Highlights & Insights
- **Highlight 1**: Proposes permutation sparse processing of Grid Heads, converting visual structure priors directly into computational speedup.
- **Highlight 2**: Categorizes modality boundaries (No/K/Q/2D) and designs corresponding processing paths, offering strong interpretability and extensibility.
- **Highlight 3**: The trade-off between offline head-level search + online lightweight estimation is practical, evading high purely-online overheads and purely-offline mismatches.
- **Highlight 4**: Emphasizes dynamic load/write-back inside kernels rather than explicit transposition, demonstrating strong system-level optimization awareness.
- **Highlight 5**: Evaluates multiple SOTA long-context VLMs and multi-task benchmarks, showing it is not a single-model-specific trick.

## Limitations & Future Work

### Current Limitations Evident in the Cache
- **Limitation 1**: The cache file is truncated in the middle of the method section, lacking the complete experimental chapter and preventing the verification of exact scores across all benchmarks.
- **Limitation 2**: Lack of a complete ablation table makes it difficult to quantify the independent contribution of each sub-module (Grid/Q-Boundary/2D-Boundary).

### Potential Methodological Limitations (based on the disclosed structure)
- Modality boundaries rely on the formatting of inputs. If upstream tokenizers/packing strategies change, re-adaptation may be required.
- Offline head-level search might incur repetitive overheads when migrating across model versions.
- The hardware-friendliness of sparse patterns is correlated with GPU architectures; performance gains may vary across different hardwares.

### Future Directions
- **Direction 1**: Learnable boundary detectors to replace rule-based or static classifications, improving robustness under complex mixed inputs.
- **Direction 2**: Turn pattern search into online self-adaptation (low-frequency updates) to reduce re-calibration costs after model upgrades.
- **Direction 3**: Combine with KV cache compression and hierarchical routing to explore joint optimization of prefill and decode.

## Related Work & Insights
- **vs. StreamingLLM / Sliding Window Methods**: The latter focuses on temporal text locality; MMInference explicitly utilizes visual 2D/spatial-temporal structures and modality boundaries.
- **vs. MInference**: Both leverage dynamic sparsity and kernel optimization, but MMInference emphasizes "multimodal structure priors + permutation-based restructuring."
- **My Insights**:
	- When accelerating VLM inference, one should focus on whether the "sparse pattern is hardware-friendly" rather than just the "sparsity rate."
	- The core of multimodal tasks is division of labor by heads and modality relations, rather than applying a uniform template.

### Replication & Deployment Suggestions
- If the target is production TTFT, prioritize testing benefits in $128\text{k}+$ long-context scenarios.
- Visualize attention first to confirm whether the target model contains distinct Grid/Q/2D boundary heads before committing to kernel engineering.
- Record the following simultaneously during evaluation:
	- prefill latency
	- end-to-end TTFT
	- changes in task accuracy
	- stability under different modality ratios

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ (4/5)  
  *Reason*: Systematically incorporating multimodal boundaries and grid structures into dynamic sparse inference targets a highly focused problem and solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ (4/5)  
  *Reason*: Cached descriptions show wide coverage of models and tasks, but the missing complete numerical tables in the local cache prevent a higher score.
- **Writing Quality**: ⭐⭐⭐⭐☆ (4/5)  
  *Reason*: Clear logic flow from observations to methods, with precise diagrammatic logic.
- **Value**: ⭐⭐⭐⭐⭐ (5/5)  
  *Reason*: Directly accelerates prefill without changing model weights, showing high engineering practicality through a concrete system implementation path.

## Remarks
This note is drafted strictly based on the locally cached file without external web search. If a complete version of the cache is acquired afterwards, the main results and ablation numerical tables can be supplemented, and qualitative-only items can be updated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](../../ACL2025/vlm_efficiency/madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [\[ACL 2026\] APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention](../../ACL2026/vlm_efficiency/apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro.md)
- [\[ICML 2025\] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)
- [\[NeurIPS 2025\] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](../../NeurIPS2025/vlm_efficiency/vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)
- [\[ACL 2026\] MACS: Modality-Aware Capacity Scaling for Efficient Multimodal MoE Inference](../../ACL2026/vlm_efficiency/macs_modality-aware_capacity_scaling_for_efficient_multimodal_moe_inference.md)

</div>

<!-- RELATED:END -->
