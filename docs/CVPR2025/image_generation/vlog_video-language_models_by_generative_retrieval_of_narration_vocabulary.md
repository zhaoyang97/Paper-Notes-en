---
title: >-
  [Paper Note] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary
description: >-
  [CVPR 2025][Image Generation][Video understanding] Proposes VLog, which defines video narration as vocabulary units, achieving efficient video understanding that is 10-20 times faster than generative VideoLLMs through a generative retrieval architecture (GPT-2 reasoning + SigLIP retrieval).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video understanding"
  - "generative retrieval"
  - "narration vocabulary"
  - "video-language models"
  - "efficient inference"
date: 2026-05-08
content_hash: c115dda41bff9ca1
---

# VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary

**Conference**: CVPR 2025  
**arXiv**: [2503.09402](https://arxiv.org/abs/2503.09402)  
**Code**: [GitHub](https://github.com/showlab/VLog)  
**Area**: Image Generation  
**Keywords**: Video understanding, generative retrieval, narration vocabulary, video-language models, efficient inference

## TL;DR

Proposes VLog, which defines video narration as vocabulary units, achieving efficient video understanding that is 10-20 times faster than generative VideoLLMs through a generative retrieval architecture (GPT-2 reasoning + SigLIP retrieval).

## Background & Motivation

Existing VideoLLMs inherit the subword vocabularies of LLMs (e.g., LLaMA-3's 128K vocabulary, which contains a large number of visually meaningless subwords like 'happ') and the token-by-token autoregressive decoding paradigm. This results in slow inference speeds, making real-time processing of video streams difficult.

Practical applications (such as AR glasses assistants) require concise and contextually relevant real-time responses rather than exhaustive descriptions. When humans recall daily activities, they naturally organize their experiences into a series of narrative events (e.g., "turning off the alarm", "washing dishes"), forming an action "vocabulary".

**Core Problem**: How to construct a vocabulary with narration as the minimal unit to replace subword vocabularies while retaining the reasoning capabilities of LLMs? Retrieval models (e.g., CLIP) support flexible vocabulary updates but lack reasoning capabilities, whereas generative models exhibit strong reasoning but suffer from slow decoding. How can the advantages of both be combined?

## Method

### Overall Architecture

VLog is built upon the lightweight GPT-2-medium and SigLIP. The core innovations include: (1) a generative retrieval architecture—introducing a retrieval token at the end of the GPT-2 sequence, merging visual and query information before performing dot-product retrieval with vocabulary embeddings; (2) Narrative Pair Encoding (NPE) to construct a hierarchical vocabulary, supporting two-level retrieval of prefix + suffix; and (3) an agent workflow based on LMM+LLM to achieve automatic vocabulary expansion.

### Key Design 1: Generative Retrieval Architecture

**Function**: Combines the reasoning capabilities of generative models with the efficiency and flexibility of retrieval models.

**Mechanism**: A retrieval token $\mathbf{t}$ is appended to the end of the GPT-2 language model's input sequence, which attends to the preceding visual and query inputs via self-attention. After passing through GPT-2, the output embedding $\tilde{\mathbf{t}}$ encodes visual and query information and is used for dot-product retrieval with vocabulary embeddings: $\Pr(\mathcal{X} = \tilde{o_i} | \mathcal{V}, \mathcal{Q}) = \tilde{\mathbf{t}}^T \tilde{\mathbf{o}_i}$. The vocabulary embeddings are pre-computed and cached by SigLIP without passing through GPT-2, forming an asymmetric structure that reduces computational overhead.

**Design Motivation**: Pure retrieval models (e.g., SigLIP) lack reasoning capabilities and cannot answer causal queries such as "What is the next action?". Pure generative models are too slow due to token-by-token decoding. Bridging the two via a retrieval token preserves GPT-2's causal reasoning capabilities while achieving single-step retrieval at the narration level.

### Key Design 2: Narrative Pair Encoding (NPE) and Hierarchical Indexing

**Function**: Constructs a structured vocabulary from large-scale narrative data and enables efficient retrieval.

**Mechanism**: Similar to the tokenization concept of BPE, narration is decomposed into a prefix set (core actions, e.g., "cutting potatoes") and a suffix set (modifying details, e.g., "with the left hand"). During retrieval, the scene level (e.g., "kitchen") is first used to narrow down the prefix search space, followed by matching the suffix. This forms a three-level hierarchy: Scene $\rightarrow$ Prefix Narration Subset $\rightarrow$ Suffix.

**Design Motivation**: Brute-force searching a million-scale vocabulary is infeasible. Human activities are naturally associated with scenes ("cutting potatoes" in the kitchen), and hierarchical indexing compresses the search space by several orders of magnitude. Separating prefixes and suffixes makes the vocabulary more compact and expressive.

### Key Design 3: Automatic Vocabulary Expansion

**Function**: Handles unseen new events during inference.

**Mechanism**: When the similarity between the retrieval token and the best-matching vocabulary entry falls below a threshold of 0.4, it is determined as an OOV (Out-Of-Vocabulary) event. An agent workflow is then triggered: (1) LLaVA-OV-0.5B is used to generate a visual scene description; (2) Qwen2.5-0.5B reasons about possible events based on the scene description and parses them into new vocabulary entries. This constitutes a "Generative-Augmented Retrieval" paradigm.

**Design Motivation**: No matter how large the initial vocabulary is, it cannot cover all novel scenes. The advantage of retrieval models is that vocabulary embeddings are independent of model weights (directly encoded by SigLIP), meaning that adding new words does not require retraining.

### Loss & Training

Standard contrastive learning loss: $\mathcal{L} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \log \frac{\exp(\tilde{\mathbf{t}}_i^T \tilde{\mathbf{o}_i}/\tau)}{\sum_{j \in \mathcal{B}} \exp(\tilde{\mathbf{t}}_i^T \tilde{\mathbf{o}_j}/\tau)}$, with temperature $\tau=0.05$.

## Key Experimental Results

### Main Results (Retrieval Performance on Vidcab-Eval)

| Method | CIDEr(Naive) | R@1(Naive) | CIDEr(Causal) | R@1(Causal) | Decoding Time (s) |
|------|-------------|-----------|--------------|------------|-----------|
| Generative GPT2 | 64.8 | 7.9 | 53.7 | 3.1 | 0.362 |
| Retrieval (FT) | 95.8 | 11.8 | 48.9 | 2.1 | 0.016 |
| **VLog** | **96.9** | **12.4** | **87.3** | **5.0** | **0.018** |

### COIN Benchmark (Action Perception)

| Method | Model Size | Step Acc | Task Acc | Next Acc |
|------|---------|----------|----------|----------|
| VideoLLM-online | 7B | 59.8 | 92.1 | 48.1 |
| GPT2 (Generative) | 355M | 44.6 | 82.4 | 32.1 |
| **VLog** | 355M | 56.1 | 93.0 | 46.0 |
| **VLog+Ego4D Pre-training** | 355M | **57.4** | **94.4** | **48.4** |

### Key Findings

1. **Causal Retrieval Far Exceeds Competitors**: Under the Causal setting (which requires reasoning about "before/after" relationships), VLog achieves a CIDEr of 87.3, far exceeding the retrieval model's 48.9 and the generative model's 53.7. This proves that generative retrieval effectively integrates reasoning and retrieval.
2. **20x Speedup**: VLog's decoding time is 0.018s vs. the generative model's 0.362s, which is close to the speed of pure retrieval models.
3. **Lightweight Models Comparable to Large Models**: With only 355M parameters, VLog achieves performance on COIN comparable to the 7B VideoLLM-online.
4. **Transferable Vocabulary**: The Ego4D pre-trained vocabulary successfully transfers to the COIN dataset, improving all metrics.

## Highlights & Insights

- **Paradigm Innovation**: The "narration as vocabulary" concept transforms video understanding from token-by-token generation to narration-level retrieval, fundamentally solving the speed bottleneck.
- **Elegant Architecture**: The retrieval token serves as a bridge between generation and retrieval. The design is simple, and the asymmetric structure avoids redundant computation of vocabulary embeddings.
- **Generative-Augmented Retrieval**: A new paradigm converse to RAG, utilizing a generative model to expand the retrieval vocabulary.

## Limitations & Future Work

- **Closed-Vocabulary Assumption**: It still relies on a predefined vocabulary, which limits its ability to describe complex, open-world scenes.
- **Ego4D Bias**: The vocabulary mainly originates from egocentric (first-person) videos; its applicability to exocentric (third-person) scenes has not been fully verified.
- **Limited Suffix Expressiveness**: The suffix set cannot capture all fine-grained differences, such as specific quantities, colors, or other attributes.
- Future work can explore vocabulary continual learning, multimodal vocabularies, and integration with large LLMs.

## Related Work & Insights

- **CLIP Retrieval**: SigLIP provides flexible open-vocabulary embedding capabilities but lacks reasoning. VLog complements this reasoning capability through the retrieval token.
- **BPE Tokenization**: NPE extends the subword decomposition concept of BPE to the narration level, inspiring how to construct domain-specific vocabularies.
- **Insights**: The "retrieval token" design can be generalized to other multimodal tasks that require the integration of reasoning and retrieval.

## Rating

⭐⭐⭐⭐ — Novel problem definition (narrative vocabulary replacing subwords), elegant generative retrieval architecture design, and a significant 20x speedup. The performance of the lightweight model comparable to large models is highly impressive. However, the closed-vocabulary assumption limits its applicability to open-world scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)
- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[ECCV 2024\] IRGen: Generative Modeling for Image Retrieval](../../ECCV2024/image_generation/irgen_generative_modeling_for_image_retrieval.md)
- [\[CVPR 2025\] Font-Agent: Enhancing Font Understanding with Large Language Models](font-agent_enhancing_font_understanding_with_large_language_models.md)

</div>

<!-- RELATED:END -->
