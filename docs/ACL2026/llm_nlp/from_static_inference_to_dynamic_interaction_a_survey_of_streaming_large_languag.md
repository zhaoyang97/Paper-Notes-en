---
title: >-
  [Paper Note] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Streaming LLM] This paper presents the first systematic survey of Streaming Large Language Models (Streaming LLMs)…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Streaming LLM"
  - "real-time interaction"
  - "incremental encoding"
  - "full-duplex"
  - "speculative decoding"
date: 2026-05-08
content_hash: 5157e1c4614c155f
---

# From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models

**Conference**: ACL 2026
**arXiv**: [2603.04592](https://arxiv.org/abs/2603.04592)  
**Code**: [GitHub](https://github.com/EIT-NLP/Awesome-Streaming-LLMs)  
**Area**: LLM Systems / Streaming Inference
**Keywords**: Streaming LLM, real-time interaction, incremental encoding, full-duplex, speculative decoding

## TL;DR

This paper presents the first systematic survey of Streaming Large Language Models (Streaming LLMs), proposing a unified definition grounded in data flow and interaction concurrency. It organizes existing approaches into a three-level progressive taxonomy — Output-streaming, Sequential-streaming, and Concurrent-streaming — and covers methodologies and applications across text, speech, and video streaming scenarios.

## Background & Motivation

**Background**: Standard LLMs operate under a static inference paradigm that reads the entire input at once — encoding the complete input into a KV cache before performing autoregressive decoding. While effective on benchmarks, this paradigm fundamentally limits applicability in dynamic scenarios such as real-time translation, streaming video understanding, and interactive tool agents.

**Limitations of Prior Work**: (1) Real-world information arrives incrementally, accumulates over time, and may be unbounded in length — none of which the static paradigm can handle. (2) The definition of "Streaming LLM" in the literature is inconsistent: autoregressive decoding, incremental/chunked encoding, and full-duplex interaction are conflated under the same label. (3) Large-scale pretraining data that supports real-time interaction, partial-input supervision, and fine-grained temporal alignment is largely absent.

**Key Challenge**: There is a fundamental mismatch between the offline, full-context design of LLMs and the online, incremental, interactive data streams of the real world — models must dynamically decide when to respond, when to wait for more input, and when to stop.

**Goal**: To propose a unified definition and systematic taxonomy of Streaming LLMs, resolve terminological ambiguity, and provide a structured research roadmap for this emerging field.

**Key Insight**: Streaming LLMs are defined along two dimensions — data flow and interaction concurrency — rather than by modality or architecture.

**Core Idea**: Streaming LLMs form a three-level progressive hierarchy: (1) Output-streaming: static input + streaming output (standard autoregressive); (2) Sequential-streaming: streaming input + streaming output (generate after incremental encoding); (3) Concurrent-streaming: simultaneous streaming input and output (full-duplex interaction). Each level introduces new challenges on top of the previous one.

## Method

### Overall Architecture

The survey is organized around the three-level taxonomy. Output-streaming addresses streaming generation mechanisms and efficient decoding; Sequential-streaming focuses on incremental encoding and context management; Concurrent-streaming additionally introduces architectural adaptation and interaction strategies. For each level, methods are discussed across text, speech, and video modalities.

### Key Designs

1. **Output-Streaming LLMs**:

    - **Function**: Achieve efficient streaming output given static input.
    - **Mechanism**: Three classes of generation mechanisms — (a) Token-level: standard autoregressive (GPT, LlamaGen); (b) Block-level: semi-autoregressive (SoT generates multiple tokens in parallel) and block diffusion (SSD-LM); (c) Refinement-based: multi-scale (VAR, coarse-to-fine) and global diffusion (LLaDA, iterative denoising). Efficient streaming generation techniques include speculative decoding (small draft model + large model verification), layer skipping (AdaInfer dynamically skips redundant layers), and memory optimization (StreamingLLM's attention sink mechanism retaining KV cache of initial tokens).
    - **Design Motivation**: Token-by-token autoregressive generation is the primary throughput bottleneck in streaming applications; quality must be maintained while improving efficiency.

2. **Sequential-Streaming LLMs**:

    - **Function**: Process incrementally arriving input streams and generate output at appropriate moments.
    - **Mechanism**: Two core challenges — (a) Incremental encoding: classifies continuously arriving inputs into "atomic encoding" (for inputs with natural discrete units, e.g., subwords, ViT patches) and "fragmented encoding" (requiring segmentation, e.g., fixed-interval splitting or semantically driven segmentation via CTC/DiSeg); (b) Context management: KV cache compression (token eviction such as H2O, quantization such as KVQuant, merging such as CaM), retrieval-augmented memory (MemWalker for hierarchical navigation of long contexts), and state space models (Mamba's linear complexity suits streaming input).
    - **Design Motivation**: Streaming input means the encoder cannot observe the complete context, requiring decisions under incomplete information; the linear growth of KV cache with sequence length renders long-context memory infeasible.

3. **Concurrent-Streaming LLMs**:

    - **Function**: Enable full-duplex interaction with simultaneous input and output streams.
    - **Mechanism**: Two core challenges — (a) Architectural adaptation: single-channel architectures interleave input and output as a unified sequence (e.g., SpeechGPT, MiniOmni), while dual-channel architectures use separate channels for input and output (e.g., Moshi's inner/outer stream dual-Transformer); (b) Interaction strategies: explicit strategies based on special tokens (using `<wait>/<speak>` tokens to control timing), implicit heuristic strategies (triggering generation when output probability exceeds a threshold), and learned strategies (training optimal interaction timing via RL).
    - **Design Motivation**: Full-duplex interaction mirrors the natural mode of human conversation — listening while speaking — which the conventional turn-based LLM paradigm cannot support.

### Loss & Training

As a survey paper, no specific loss functions are proposed. The authors systematically review key challenges in streaming training: the scarcity of streaming pretraining data, the design of partial-input supervision, and the high annotation cost of temporal alignment labels.

## Key Experimental Results

### Main Results

As a survey, no original experiments are conducted; however, key technical comparisons are compiled:

**Comparison of Three Streaming Paradigms**

| Paradigm | Input Mode | Output Mode | Core Challenge | Representative Methods |
|---|---|---|---|---|
| Output-streaming | Static | Streaming | Efficient generation | Speculative Decoding, SoT |
| Sequential-streaming | Streaming | Streaming (delayed) | Incremental encoding + context management | StreamingLLM, H2O, Mamba |
| Concurrent-streaming | Streaming | Streaming (simultaneous) | Full-duplex architecture + interaction strategy | Moshi, MiniOmni, OmniChat |

**Comparison of KV Cache Compression Methods**

| Category | Representative Work | Core Mechanism | Pros & Cons |
|---|---|---|---|
| Token eviction | H2O, FastGen | Evict unimportant tokens based on attention scores | Simple and efficient but may discard critical information |
| Quantization | KVQuant, KIVI | Reduce KV cache numerical precision | High compression ratio but potential accuracy loss |
| Merging | CaM, D2O | Merge KV representations of similar tokens | Better information retention but slightly higher computational overhead |

### Key Findings

- The three-level taxonomy clearly demonstrates the progressive nature of technical challenges — Concurrent-streaming builds upon the prior two levels and additionally introduces architectural adaptation and interaction strategies.
- Speech streaming is currently the most mature direction (Moshi, SpeechGPT-Gen), followed by text streaming, with video streaming the least developed.
- KV cache management is a common bottleneck across all streaming scenarios; the attention sink finding in StreamingLLM (initial tokens are disproportionately important for attention scores) has been pivotal to progress in this area.
- The largest open problem in full-duplex interaction is learning when to respond — threshold-based methods are simple but brittle, while RL-based methods are promising but difficult to train.

## Highlights & Insights

- The three-level progressive taxonomy is conceptually clean — unifying disparate terminology from a data-flow perspective and providing a structured roadmap for subsequent research.
- The comparison between single-channel and dual-channel architectures reveals the core design trade-off in full-duplex interaction: single-channel models are simpler but suffer from input-output interference, while dual-channel models enable parallelism at the cost of doubled parameters.
- The systematic organization of "when to respond" strategies (explicit tokens, implicit thresholds, learned policies) covers the full spectrum from simple to complex approaches.

## Limitations & Future Work

- Evaluation standards for Streaming LLMs remain unsettled — there is no consensus on how to jointly measure latency, output quality, and interactivity.
- The scarcity of streaming pretraining data is a fundamental bottleneck, particularly for full-duplex interaction scenarios.
- A unified architecture for multimodal concurrent streaming (simultaneously processing speech, video, and text) remains an open problem.
- Safety considerations are insufficiently addressed — once generated, streaming outputs cannot be retracted, making error costs higher than in static inference.

## Related Work & Insights

- **vs. Xiao et al. (2023) StreamingLLM**: StreamingLLM is a specific method (attention sink + sliding window); the present survey classifies it as one KV cache management approach under Sequential-streaming.
- **vs. Surveys on Speculative Decoding**: Speculative decoding is an efficient generation technique within Output-streaming; this survey situates it within the broader Streaming LLM framework.
- **vs. Surveys on Spoken Dialogue Models**: Spoken dialogue model surveys primarily address Concurrent-streaming; this survey additionally covers text and video streaming scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-level progressive taxonomy and unified definition represent a significant conceptual contribution.
- Experimental Thoroughness: ⭐⭐ — No original experiments, as expected of a survey paper.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear taxonomy, excellent figure design, and comprehensive coverage.
- Value: ⭐⭐⭐⭐⭐ — The first systematic survey on Streaming LLMs, providing a much-needed structured framework for a rapidly evolving field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)

</div>

<!-- RELATED:END -->
