---
title: >-
  [Paper Note] Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Large Language Models] The paper shifts the VRAM bottleneck of Multimodal Large Language Models (MLLMs) from "long-context caching during the decoding phase" forward to the "peak vis…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "KV Cache"
  - "prefill compression"
  - "VRAM optimization"
  - "visual tokens"
date: 2026-05-08
content_hash: 125003be85dcde66
---

# Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines

**Conference**: ACL 2026  
**arXiv**: [2604.16734](https://arxiv.org/abs/2604.16734)  
**Code**: Not disclosed  
**Area**: Multimodal VLM / Inference Efficiency / KV Cache Compression  
**Keywords**: Multimodal Large Language Models, KV Cache, prefill compression, VRAM optimization, visual tokens

## TL;DR
The paper shifts the VRAM bottleneck of Multimodal Large Language Models (MLLMs) from "long-context caching during the decoding phase" forward to the "peak visual token caching during the prefill phase." It proposes a structure-aware KV-cache framework that performs computation and compression simultaneously during prefill. This maintains peak VRAM within a fixed cache budget while preserving image and video understanding capabilities as much as possible.

## Background & Motivation
**Background**: Modern MLLMs typically use visual encoders to extract image or video features, which are then fed into the LLM backbone via projection layers. This allows text tokens and a large number of visual tokens to enter self-attention together. High-resolution images, multi-tile inputs, and long video frame sequences cause visual token counts to expand rapidly, requiring the model to process extremely long multimodal prefixes before decoding.

**Limitations of Prior Work**: KV cache was originally designed to reduce redundant computation in autoregressive decoding, but cache size grows linearly with the number of tokens, layers, and heads. In MLLMs, peak VRAM usage often occurs not when generating long answers, but during the prefill phase: the model must first construct the KV cache for the complete visual context before generation begins. Many existing KV compression methods perform eviction or merging only after the full cache has already been built, thus failing to avoid VRAM spikes and OOM (Out-of-Memory) during prefill.

**Key Challenge**: Multimodal inference needs to retain sufficiently fine-grained visual information, especially for high-resolution localization and long-video temporal cues. However, if full encoding precedes compression, the VRAM peak has already occurred. Simply lowering input resolution or deleting visual tokens directly results in the loss of task-essential details.

**Goal**: To design an inference framework that maintains a fixed KV-cache budget during the prefill process, enabling the model to handle larger-scale visual inputs. Additionally, the paper compares query-aware and query-agnostic online eviction strategies and analyzes the relationship between VRAM, latency, and accuracy.

**Key Insight**: The authors observe that visual tokens differ from pure text tokens; images possess spatial continuity, and videos possess temporal redundancy. These structures imply that compression granularity should not be entirely random or rely solely on global attention statistics but should align with visual patches, tiles, or groups of frames.

**Core Idea**: Instead of waiting for the entire multimodal prefix to be encoded before compressing, the visual context is processed in blocks during prefill. Each time a block is processed, the KV cache is compressed back to a fixed budget, changing "process first, compress later" to "compress as you prefill."

## Method
The method in this paper does not train a new model but modifies the inference execution path of MLLMs. The key is to break the traditional one-time prefill into block-wise prefill: the input sequence is divided into continuous blocks, and the model calculates KV block by block. Every time new KV blocks are appended, if the cache exceeds the budget, tokens are immediately removed according to an eviction strategy. Consequently, the complete visual context never resides in VRAM as a full cache; peak VRAM is controlled by the budget rather than the raw number of visual tokens.

### Overall Architecture
Given a multimodal input sequence $S$, block size $b$, and cache budget $M$, the framework first divides $S$ into several continuous blocks. When processing the $i$-th block, the model calculates the key/value for that block and adds them to the existing cache $C$. If $|C| > M$, the system calculates the excess amount $k_{excess}=|C|-M$ and calls the eviction strategy to compress the cache back within the budget. This process continues until prefill is complete, resulting in a budget-controlled compressed KV cache for subsequent decoding.

The paper considers two types of online eviction. In single-turn QA, where the text query is already visible, a query-aware SnapKV-style strategy can be used to estimate the importance of visual tokens using the similarity between a proxy query in the prompt and the cached keys. In potential multi-turn scenarios where subsequent queries are unknown, a query-agnostic KeyDiff-style strategy is used to retain tokens that deviate more from the mean key representation to maintain visual diversity.

### Key Designs
1. **Fixed-budget block-wise prefill**:

    - **Function**: Limits the peak VRAM of the KV cache during the prefill phase to near budget $M$, avoiding transient OOM caused by full visual contexts.
    - **Mechanism**: The input is split into continuous blocks; the model appends KV and immediately performs eviction after processing each block. Unlike post-prefill compression, it never needs to store the full cache, so peak VRAM grows with the budget rather than the original token count.
    - **Design Motivation**: Long visual inputs in MLLMs create VRAM peaks before decoding; optimizing only the generation phase is insufficient. Online compression acts directly where the peak occurs.

2. **Visual structure-aligned compression granularity**:

    - **Function**: Ensures cache compression respects the structure of image tiles, spatial grids, and video frame groups to reduce the risk of erroneously deleting critical visual information.
    - **Mechanism**: Block boundaries are aligned as much as possible with the natural structure of the visual input. The paper's analysis shows Qwen2.5-VL-7B performs best with a block size of 784, which corresponds exactly to its $28 \times 28$ visual tokenization.
    - **Design Motivation**: Redundancy in visual tokens is not unstructured noise but comes from repetitions in adjacent areas and frames. If compression granularity disrupts this structure, it harms localization and fine-grained understanding.

3. **Query-aware and query-agnostic dual-path eviction**:

    - **Function**: Covers both single-turn and potential multi-turn task scenarios.
    - **Mechanism**: The query-aware path scores cached keys using a proxy query from the text prompt to retain tokens related to the current question. The query-agnostic path calculates the difference between keys and the mean representation to retain representative or rare visual tokens.
    - **Design Motivation**: Single-turn tasks prioritize task relevance, while multi-turn tasks require reusable visual caches. These strategies correspond to the assumptions that "the input is most useful for the current question" versus "may be useful for future questions."

### Loss & Training
This paper does not introduce new training objectives or require model fine-tuning; it is an inference-time KV-cache management method. Main hyperparameters include KV budget, block size, and eviction strategy. The default block size is 256, but the authors found that structure-aligned block sizes significantly improve performance. To reduce latency from pure block-wise execution, the paper uses a hybrid strategy: it attempts to process the part that fits within the budget via a single bulk forward pass, entering block-wise execution only when the budget is exceeded.

## Key Experimental Results

### Main Results
The main experiments cover fine-grained image localization and long video understanding tasks, including ImageNeedleInHaystack, V*, MLVU, and Video-MME long setting. The following results are for 8B/7B models, where Average is higher-is-better and Delta indicates the average decline relative to full cache.

| Model | Method / KV Budget | ImageNeedle | V* | MLVU | Video-MME(L) | Average | Delta ↓ |
|------|---------------|-------------|----|------|--------------|---------|---------|
| InternVL3.5-8B | Full Cache | 80.31 | 84.35 | 51.28 | 53.89 | 67.46 | - |
| InternVL3.5-8B | SnapKV 1024 | 80.00 | 82.61 | 51.00 | 53.11 | 66.68 | 0.78 |
| InternVL3.5-8B | KeyDiff 1024 | 74.06 | 74.35 | 50.40 | 52.22 | 62.76 | 4.70 |
| Qwen2.5-VL-7B | Full Cache | 83.70 | 79.58 | 48.80 | 50.00 | 65.52 | - |
| Qwen2.5-VL-7B | SnapKV 4096 | 85.00 | 78.53 | 44.82 | 48.77 | 64.28 | 1.24 |
| Qwen2.5-VL-7B | KeyDiff 4096 | 81.56 | 79.58 | 47.41 | 49.33 | 64.47 | 1.05 |

Model scale experiments show the method works on larger models, though performance drops more noticeably under a 1024 budget.

| Model | Method / KV Budget | ImageNeedle | V* | MLVU | Video-MME(L) | Average | Delta ↓ |
|------|---------------|-------------|----|------|--------------|---------|---------|
| InternVL3.5-14B | Full Cache | 84.06 | 83.76 | 49.67 | 57.89 | 68.95 | - |
| InternVL3.5-14B | SnapKV 1024 | 66.25 | 82.72 | 47.61 | 56.45 | 63.26 | 7.73 |
| InternVL3.5-14B | KeyDiff 1024 | 75.94 | 64.92 | 46.41 | 52.78 | 60.01 | 8.84 |
| Qwen2.5-VL-32B | Full Cache | 96.56 | 83.77 | 48.40 | 55.22 | 70.99 | - |
| Qwen2.5-VL-32B | SnapKV 1024 | 66.25 | 82.72 | 47.61 | 56.45 | 63.26 | 7.73 |
| Qwen2.5-VL-32B | KeyDiff 1024 | 67.19 | 72.25 | 45.82 | 54.56 | 59.96 | 11.03 |

### Ablation Study
Prefill strategy analysis shows that the gains from this method are not simply from reduced resolution, nor is any dynamic budget effective.

| Analysis Item | Setting | ImageNeedle | Note |
|--------|------|-------------|------|
| Forward under budget | Block Forward 1024 | 80.94 | Entire prefill performed in blocks; stable VRAM but higher latency |
| Forward under budget | Bulk Forward 1024 | 80.31 | Default hybrid strategy; similar accuracy with lower latency |
| Static vs. Dynamic | Static 1024 | 80.31 | Fixed budget is more reliable during prefill |
| Static vs. Dynamic | Dynamic 1024 | 74.68 | Dynamic layer-wise budget drops by 5.63 points |
| Input res. vs. Compression | Compression 1024 | 80.31 | Keep high-res input, compress only KV cache |
| Input res. vs. Compression | Reduction 1024 | 9.38 | Direct resolution reduction severely breaks visual localization |

Block size ablation proves "structural alignment" is a key factor affecting compression robustness.

| Block size / KV budget | ImageNeedle | Global Peak (GB) | Avg. Peak (GB) | Observation |
|----------------------|-------------|------------------|----------------|------|
| 256 / 2048 | 72.19 | 17.80 | 17.12 | Small default block; lowest VRAM but weak accuracy |
| 512 / 2048 | 75.31 | 18.00 | 17.19 | Not fully aligned with visual grid |
| 784 / 2048 | 80.63 | 18.21 | 17.26 | Matches $28 \times 28$ tokenization; best performance |
| 1024 / 2048 | 79.38 | 18.38 | 17.37 | Slightly higher VRAM yields better accuracy |

### Key Findings
- The prefill phase is the critical VRAM peak for MLLMs with many visual tokens; post-prefill compression does not solve this.
- On InternVL3.5-8B, a SnapKV 1024 budget resulted in only a 0.78 average drop, indicating that ~90% KV cache compression can still preserve primary visual abilities.
- Query-aware eviction is stronger when the query is visible; query-agnostic KeyDiff provides a reusable solution for multi-turn or unknown query scenarios.
- Directly reducing input resolution causes ImageNeedle to drop from 80.31 to 9.38, suggesting that keeping the original visual input and only compressing the cache is safer than input-level reduction.
- This method transforms unrunnable OOM cases into adjustable memory-latency trade-offs, though increased latency is a necessary system cost.

## Highlights & Insights
- The paper accurately identifies the problem: VRAM pressure in MLLMs comes not just from "generating long outputs" but from "loading massive visual tokens before generation starts." This perspective extends KV-cache compression from decoding optimization to prefill optimization.
- The method does not retrain models and has low engineering overhead. Any inference system supporting block-wise prefill and online eviction can integrate existing MLLMs.
- The analysis of structural alignment is enlightening. Visual token compression should respect structures like patch grids, tiles, and frame sequences rather than just numerical importance.
- The comparison with input downsampling is persuasive. The paper shows that "seeing less of the image" and "seeing the whole image but compressing the cache" are not equivalent; the latter is better for fine-grained visual tasks.

## Limitations & Future Work
- Block-wise prefill increases time-to-first-token (TTFT), especially with smaller budgets; the system requires a trade-off between runnability and latency.
- Query-agnostic eviction only preserves representation diversity and does not know what a user will actually ask, making it weaker than query-aware strategies in task relevance.
- Compression effectiveness depends on the alignment of block boundaries with visual tokenization; different MLLM visual encoding methods may require separate tuning.
- This method only compresses at inference time and does not adapt the model to prefill-stage compression during training; future work could train models to be more robust under compressed cache conditions.
- VRAM curves in the paper are mainly shown as figures without accompanying precise textual values, requiring further verification via the original figures or code when reproducing experiments.

## Related Work & Insights
- **vs SnapKV / H2O etc. post-prefill KV eviction**: These methods compress the decoding phase cache but usually construct the full context first. This paper moves compression into the prefill phase, directly reducing peak VRAM.
- **vs KeyDiff**: KeyDiff is a query-agnostic diversity-preserving strategy. This paper places it into an online prefill framework, allowing continuous budget control during visual context construction.
- **vs Input-level token pruning / resolution reduction**: Input pruning reduces the visual information entering the model, which can easily lose details. This paper retains high-resolution input and only applies budget control at the KV cache layer.
- **vs MEDA / FlowMM**: These methods emphasize hierarchical or cross-modal structures of multimodal caches. The insight here is that system execution timing is equally important; compression occurring after the full cache versus during prefill leads to entirely different peak VRAM.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly moving KV-cache compression to the MLLM prefill stage provides a clear problem definition and system perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers images, videos, various model sizes, and multiple ablations, though VRAM curve numerical readability is limited.
- Writing Quality: ⭐⭐⭐⭐ Concise structure, direct motivation, and clear explanation of system trade-offs.
- Value: ⭐⭐⭐⭐ Practical for deploying high-resolution and long-video MLLMs, especially in VRAM-constrained inference scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](../../CVPR2026/multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[ACL 2026\] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)

</div>

<!-- RELATED:END -->
