---
title: >-
  [Paper Note] ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism
description: >-
  [NeurIPS 2025][Multimodal VLM][MLLM inference serving] This paper proposes the Elastic Multimodal Parallelism (EMP) paradigm and the ElasticMM system…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "MLLM inference serving"
  - "elastic parallelism"
  - "modality-aware scheduling"
  - "inference disaggregation"
  - "resource allocation"
date: 2026-05-08
content_hash: 183315e9931bf030
---

# ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism

**Conference**: NeurIPS 2025
**arXiv**: [2507.10069](https://arxiv.org/abs/2507.10069)
**Code**: Unavailable (coming soon)
**Area**: Multimodal VLM
**Keywords**: MLLM inference serving, elastic parallelism, modality-aware scheduling, inference disaggregation, resource allocation

## TL;DR

This paper proposes the Elastic Multimodal Parallelism (EMP) paradigm and the ElasticMM system, which disaggregates different stages of multimodal inference into independent instances via modality-aware load balancing and elastic partition scheduling, achieving up to 4.2× TTFT reduction and 3.2–4.5× throughput improvement over vLLM.

## Background & Motivation

MLLM inference is substantially more complex than text-only LLM inference: (1) additional components (visual encoders, cross-attention layers) increase architectural complexity; (2) multimodal data, once encoded and concatenated with text prompts, significantly increases context length (the paper shows that multimodal requests have far longer average prompt lengths than text-only requests).

Existing inference systems (vLLM, SGLang, DeepSpeed) adopt tightly coupled architectures, suffering from two layers of problems:

- **Serving-layer coupling**: text and multimodal requests are handled uniformly, despite drastically different resource demands.
- **Infrastructure-layer coupling**: preprocessors, visual encoders, and LLM backends are co-located on the same hardware, sharing compute and memory resources.

This leads to: sharp TTFT increases under heavy multimodal load, inefficient mixed batching for encoder-decoder architectures (e.g., LLaMA3.2-Vision), and inability of static resource allocation to handle bursty traffic.

## Method

### Overall Architecture

ElasticMM adopts a two-level hierarchical scheduling framework:

1. **Modality level**: instances are grouped by the modality of the model they serve (text-only vs. multimodal), with modality-aware load balancing for dynamic resource allocation.
2. **Stage level**: the inference pipeline is further decomposed into three independent stages—encoding, prefill, and decode—with elastic partition scheduling enabling independent parallelism adjustment at each stage.

Both levels provide two core capabilities: **disaggregation** and **elasticity**.

### Key Designs

#### Modality-Aware Load Balancing

Two complementary mechanisms are combined—proactive and reactive:

**Proactive mechanism**: exploits the predictability of long-term load patterns (low at night, high during the day) to pre-assign idle elastic instances to each modality group. The objective is to maximize the minimum burst tolerance across all groups:

$$bt(i) = \frac{N_i^{\text{peak}}}{N_i^{\text{avg}}}$$

A greedy strategy iteratively assigns each instance to the group with the currently lowest burst tolerance.

**Reactive scaling**: handles unpredictable short-term bursts (e.g., sudden surges of image requests). The system evaluates the benefit-cost trade-off between intra-group parallelism adjustment and inter-group reactive scaling, selecting the optimal strategy. When an instance is preempted, its workload is migrated to other instances at the same stage.

#### Elastic Partition Scheduling

Addresses three sub-problems for intra-group request scheduling and parallelism adjustment:

**Request dispatching**: an FCFS policy selects a prefill request set $R_p$ from the queue, subject to GPU memory and compute throughput constraints. Continuing to add requests when the system transitions from memory-bound to compute-bound degrades performance.

**Stage allocation**: assigns an elastic instance set $E_p$ to $R_p$, prioritizing idle instances and preempting decode-stage instances when necessary. A gain-cost model evaluates the benefit of preemption:

$$\text{Gain} = \sum_{r \in R_p} \frac{T(R_p, E_p) - T(R_p, E_p \cup e_{\max})}{r.\text{input\_len}}$$
$$\text{Cost} = \sum_{r \in B_d} \frac{M(e_{\max}) + w \cdot L(B_d, E_d - e_{\max})}{r.\text{output\_len}}$$

A tunable penalty factor $w$ controls the aggressiveness of preemption. Data parallelism (DP) is preferred within stages, as elastic scaling only requires KV cache migration, avoiding expensive weight transfers.

**Elastic auto-scaling**: the decode stage is monitored to trigger scaling. Given the poor scalability of the decode stage, it is first reduced to minimum parallelism; when resources are insufficient, preemption candidates are selected from prefill instances within the group or from inter-group instances, using a similar gain-cost model.

#### Unified Multimodal Prefix Caching

To address request redundancy in real-world scenarios (e.g., identical system prompts, repeated images), a unified caching scheme is constructed:

- Cache pool 1: tokens produced by encoding multimodal inputs.
- Cache pool 2: prefix tokens of unified sequences (multimodal + text tokens).

Hash matching is used to skip redundant encoding; prefix-tree lookup finds the longest matching prefix to skip redundant prefill. Both pools use LRU eviction.

#### Non-Blocking Encoding

Image preprocessing and encoding are isolated to independent processes/instances for asynchronous execution, breaking the blocking dependency between encoding and prefill, thereby reducing TTFT and improving overall throughput.

### Loss & Training

ElasticMM is an inference system and does not involve training. It is built on top of vLLM and is compatible with both decoder-only (e.g., Qwen2.5-VL) and encoder-decoder (e.g., LLaMA3.2-Vision) architectures.

## Key Experimental Results

### Main Results

Experiments are conducted on 8 NVIDIA A800 80GB GPUs using LLaMA3.2-Vision-11B and Qwen2.5-VL-7B, evaluated on the ShareGPT-4o and VisualWebInstruct datasets.

**Table 1: TTFT Reduction (relative to vLLM)**

| Dataset | Qwen2.5-VL (DecOnly) | LLaMA3.2-Vision (EncDec) |
|--------|---------------------|-------------------------|
| ShareGPT-4o | 4.2× | 3.5× |
| VisualWebInstruct | 3.7× | 2.9× |

**Table 2: Maximum Throughput Improvement (under SLO constraints, relative to vLLM)**

| Dataset | Qwen2.5-VL | LLaMA3.2-Vision |
|--------|-----------|----------------|
| ShareGPT-4o | 4.5× | 3.2× |
| VisualWebInstruct | ~3.5× | ~2.8× |

ElasticMM also achieves a 2.3× throughput advantage over DistServe (static disaggregation baseline).

### Ablation Study

**EMP effectiveness**: three static resource allocation strategies (text-first, equal split, multimodal-first) all underperform EMP dynamic scheduling. ElasticMM achieves 1.8× and 2.3× throughput gains over the best static strategy on Qwen2.5-VL and LLaMA3.2-Vision, respectively.

**Inference optimization effectiveness**:
- EMP alone → limited improvement in TTFT.
- + Unified multimodal prefix caching → significantly reduces latency from redundant computation and data transfer.
- + Non-blocking encoding → further eliminates encoding-stage blocking on subsequent stages, achieving additional latency reduction.

Both optimizations deliver consistent performance gains across the majority of requests.

### Key Findings

1. Decoder-only models (Qwen2.5-VL) benefit more than encoder-decoder models, as their prefill computation is heavier and conflicts with encoding more severely.
2. ElasticMM's advantage is more pronounced on vision-intensive datasets (higher-resolution images in ShareGPT-4o).
3. Static resource allocation is suboptimal regardless of which modality it favors; elastic scheduling is the only viable solution.
4. The decode stage has poor scalability and should be reduced to minimum parallelism before scaling up on demand.

## Highlights & Insights

- The two-level disaggregation-plus-elasticity design is elegant: the modality level isolates text and multimodal requests, while the stage level separates encoding/prefill/decode, with dynamic resource adjustment at each level.
- The gain-cost model provides a quantitative framework for preemption decisions, avoiding heuristic rules.
- The unified prefix cache integrates text and multimodal caching, achieving efficient matching via hashing and prefix trees.
- Non-blocking encoding, though conceptually straightforward, yields significant practical gains (encoding latency typically exceeds prefill latency by 5×).

## Limitations & Future Work

- Validation is currently limited to a single node (8 GPUs); communication latency and parallelism strategy search space in multi-node distributed settings remain open problems.
- The gain-cost model for elastic scaling relies on offline profiling to determine thresholds, requiring re-calibration for different hardware/models.
- Performance on very large models (e.g., 72B) has not been evaluated.
- The penalty factor $w$ in reactive scaling requires manual tuning.
- No direct comparison with recent multimodal inference systems such as ModServe.

## Related Work & Insights

- DistServe/Splitwise propose prefill-decode disaggregation but use static allocation; ElasticMM extends this with elastic scheduling.
- LoongServe introduces elastic sequence parallelism; ElasticMM extends the elasticity concept to the multimodal dimension.
- Operator-level optimizations such as FlashAttention/Flash-Decoding are orthogonal to ElasticMM and can be combined.
- The two-tier architecture of modality-group isolation and stage separation offers a valuable reference for future MLLM serving system design.

## Rating

- Novelty: ⭐⭐⭐⭐ — The two-level elastic paradigm represents a systematic architectural innovation for MLLM inference.
- Technical Depth: ⭐⭐⭐⭐ — The gain-cost model and elastic scheduling algorithm design are rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple models, datasets, and ablations, though limited to a single node.
- Practical Value: ⭐⭐⭐⭐⭐ — A 4.2× TTFT reduction is highly significant for online serving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[ICCV 2025\] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](../../ICCV2025/multimodal_vlm/physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)

</div>

<!-- RELATED:END -->
