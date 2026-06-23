---
title: >-
  [Paper Note] Enhancing Multi-Image Understanding through Delimiter Token Scaling
description: >-
  [ICLR 2026][Multimodal VLM][Attention] By scaling the hidden states of image delimiter tokens in vision-language models, the ability to isolate information between images is enhanced. This achieves performance gains across multi-image (Mantis/MuirBench/MIRB/QBench2) and multi-document/multi-table (TQABench/MultiNews/WCEP-10) benchmarks without any additiona
tags:
  - ICLR 2026
  - Multimodal VLM
  - Attention
date: 2026-05-08
content_hash: 993fe8aa7306eb23
---
# Enhancing Multi-Image Understanding through Delimiter Token Scaling

**Conference**: ICLR 2026  
**arXiv**: [2602.01984](https://arxiv.org/abs/2602.01984)  
**Code**: [GitHub](https://github.com/MYMY-young/DelimScaling)  
**Area**: Multimodal / VLM  
**Keywords**: Multi-image understanding, Large Vision-Language Models, Delimiter tokens, Cross-image information leakage, Attention mechanism

## TL;DR
By scaling the hidden states of image delimiter tokens in vision-language models, the ability to isolate information between images is enhanced. This achieves performance gains across multi-image (Mantis/MuirBench/MIRB/QBench2) and multi-document/multi-table (TQABench/MultiNews/WCEP-10) benchmarks without any additional training or inference costs.

## Background & Motivation
Large Vision-Language Models (LVLMs, such as LLaVA and InternVL) have achieved excellent performance on single-image tasks, but their performance drops significantly when handling multi-image inputs. A core reason is **cross-image information leakage**—the model struggles to distinguish information from different images, leading to "misattribution" during reasoning.

**Limitations of Prior Work**: Existing LVLMs already use delimiter tokens to mark the start and end of each image (e.g., `<image_start>` and `<image_end>`), but these delimiters fail to effectively block cross-image leakage. In self-attention computations, visual tokens from different images still interact with each other, causing image-specific information to be "diluted."

**Key Challenge**: The presence of delimiter tokens provides image boundary information, but the magnitude of their hidden states is insufficient to form an effective "information barrier" in attention calculations.

**Key Insight**: An extremely concise approach—directly amplifying the hidden states of delimiter tokens (by multiplying them by a scaling factor) to enhance their "isolation" effect in the attention mechanism. This operation is applied directly during inference without retraining the model.

## Method

### Overall Architecture
The input is a multimodal sequence: visual tokens of multiple images are interspersed with delimiter tokens marking the start and end of each image (e.g., `<|vision_start|>`/`<|vision_end|>` in Qwen2.5-VL), followed by a text prompt. LVLMs naturally include these delimiters. The authors first conducted a diagnosis: removing delimiters or replacing them with other special tokens caused the clear "triangular block" boundaries in the attention map to disappear, dropping multi-image task performance by approximately 10 percentage points. This indicates that while they do delineate image boundaries, considerable cross-image attention remains in certain regions, meaning **cross-image information leakage is not fully blocked**.

The revision involves only one step: during the forward pass, the hidden states of delimiter tokens are multiplied by a scaling factor $\lambda > 1$, while all other weights, modules, and processes remain unchanged. This simultaneously accomplishes two seemingly contradictory goals: forcing softmax to focus more attention on the delimiters (weakening mutual attention between cross-image tokens) while preserving the aggregation effect where tokens from the same image share an image tag, as the delimiter's value vector is also amplified. The intervention occurs only during inference and is training-free.

### Key Designs

**1. Diagnosis: Delimiters are useful but do not truly isolate multiple images**

**Key Challenge**: The root cause of performance drops in multi-image LVLMs is cross-image information leakage. The authors decomposed the behavior of delimiter tokens in attention and found they possess two properties: (i) tokens from image $i$ focus on the $i$-th delimiter, forming a one-to-one vertical stripe rather than being globally attended like sink tokens; (ii) this attended delimiter acts as an "image tag," contributing a shared additive term $p_{d_i} v_{d_i}$ to the attention output for all tokens in the same image, thereby strengthening intra-image interaction. The problem is that both properties are weak: delimiters do not receive strong enough attention, and cross-image tokens still show significant mutual attention, meaning the boundary marking is "effective but incomplete."

**2. Scaling Delimiter Hidden States: A scalar multiplication suppressing cross-image while preserving intra-image**

**Mechanism**: Within each Transformer layer, the hidden state $h_t^{(l)}$ of delimiter tokens is replaced with $\lambda \cdot h_t^{(l)}$ ($\lambda > 1$, while non-delimiter tokens remain unchanged). $D$ is the set of delimiter indices:

$$h_t^{(l)*}=\begin{cases}\lambda \cdot h_t^{(l)} & t \in D\\ h_t^{(l)} & t \notin D\end{cases}$$

The key lies in it **simultaneously adjusting the two aforementioned properties without conflict**. On one hand, the larger norm causes delimiters to draw more attention like sink tokens; due to softmax normalization, attention assigned to other image tokens is suppressed, thus inhibiting cross-image interaction. On the other hand, the scaling also amplifies the delimiter's value vector $v_d$. The contribution of the shared additive term $p_{d_i} v_{d_i}$ within the image is amplified (experimentally about 15–30 times compared to corresponding terms in adjacent images), offsetting the softmax suppression. Consequently, intra-image interaction is reinforced rather than sacrificed. A single scalar multiplication accomplishes both tasks, which is the core of why this simple method is effective.

**3. Function: Training-free, zero additional cost**

The method adds no parameters or modules and requires no retraining or fine-tuning. It involves only one scalar multiplication at delimiter positions during the forward pass. Computational overhead is negligible, and inference speed and memory usage remain largely unchanged. It is architecture-agnostic and can be applied to various LVLMs. This also explains why the same technique works for text delimiters distinguishing documents or tables—it addresses a general "weak boundary signal" problem in attention mechanisms rather than a vision-specific issue.

### Loss & Training
No training is required. The method is a pure inference-time intervention. The only hyperparameters to set are the scaling factor $\lambda$ ($>1$) and the range of layers to apply the scaling.

## Key Experimental Results

### Main Results
The paper evaluated the method across several multi-image understanding benchmarks:

| Dataset | Task Type | Result |
|--------|---------|------|
| Mantis | Multi-image Reasoning | Gain |
| MuirBench | Multi-image Benchmark | Gain |
| MIRB | Multi-image Reasoning Benchmark | Gain |
| QBench2 | Image Quality Comparison | Gain |

Additionally, the method's effectiveness was verified on text-only tasks requiring differentiation between entities:

| Dataset | Task Type | Result |
|--------|---------|------|
| TQABench | Multi-table Understanding | Gain |
| MultiNews | Multi-doc Summarization | Gain |
| WCEP-10 | Multi-doc Event Understanding | Gain |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Scaling Factor $\lambda$ | Optimization interval exists; too small is ineffective, too large may disrupt distributions. |
| Application Layer Range | Middle layers are most effective; effects in early and final layers may be weaker. |
| Delimiter Type | Both start and end delimiters are effective; scaling both contributes to performance. |

### Key Findings
- Existing delimiter tokens in LVLMs fail to function effectively as boundary markers at the hidden state level.
- A simple scaling operation significantly enhances their function, suggesting the problem lies in insufficient learning of delimiters during training rather than architectural design.
- The method is effective not only for visual delimiters but also for text delimiters (distinguishing multiple documents/tables), showing a universal mechanism.
- The method is agnostic to model architecture.

## Highlights & Insights
- **Minimalist approach, significant effect**: Improving multi-image understanding via mere hidden state scaling is impressive in its simplicity.
- **Zero cost**: A true "free lunch"—no training, no extra parameters, and negligible inference overhead.
- **Universal mechanism**: Extends from visual delimiters to text delimiters, indicating a general issue in attention mechanisms rather than vision-specific limitations.
- **Diagnostic insight**: Analysis of why delimiters fail (insufficient hidden state norm to influence attention distribution) provides valuable understanding of LVLM internal workings.
- **Highly practical**: Directly applicable to any existing LVLM for immediate deployment.

## Limitations & Future Work
- The scaling factor $\lambda$ requires manual tuning; optimal values may vary by model and task.
- As an inference-time intervention, considering delimiter learning during training might yield better results.
- Scaling strategies might need further adjustment for extremely long multi-image sequences (e.g., video frames).
- Comparison or combination with other attention intervention methods (e.g., attention masks, positional encoding modifications) was not explored.
- Not tested on the latest closed-source ultra-large scale LVLMs like GPT-4V.

## Related Work & Insights
- Unlike visual token compression methods (e.g., TrimTokenator-LC, VisionTrim) focusing on efficiency, this work focuses on information isolation quality in multi-image scenarios.
- It provides a training-free complement to specialized training methods for multi-image understanding.
- **Insight**: The "signal strength" of special tokens in attention mechanisms is an overlooked design dimension; future training may need to explicitly encourage delimiters to learn stronger boundary representations.
- The "hidden state scaling" intervention might apply to other scenarios requiring information isolation, such as distinguishing turns in multi-turn dialogue or retrieved documents in RAG.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[ICLR 2026\] Patch-as-Decodable-Token: Towards Unified Multi-Modal Vision Tasks in MLLMs](patch-as-decodable-token_towards_unified_multi-modal_vision_tasks_in_mllms.md)
- [\[ICLR 2026\] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM](omnivinci_enhancing_architecture_and_data_for_omni-modal_understanding_llm.md)
- [\[ICLR 2026\] MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence](mmsi-bench_a_benchmark_for_multi-image_spatial_intelligence.md)
- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)

</div>

<!-- RELATED:END -->
