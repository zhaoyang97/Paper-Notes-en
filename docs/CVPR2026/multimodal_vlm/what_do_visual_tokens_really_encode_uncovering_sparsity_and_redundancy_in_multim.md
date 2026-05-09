---
title: >-
  [Paper Note] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][visual token analysis] This paper proposes EmbedLens, a probing tool for systematically analyzing the internal structure of visual tokens in MLLMs. It reveals that visual tokens fall into three categories—sink, dead, and alive (approximately 40% are uninformative)—that alive tokens already encode rich semantics before entering the LLM (a "pre-linguistic" property), and that intra-LLM visual computation is redundant for most tasks, such that direct mid-layer injection suffices.
tags:
  - CVPR 2026
  - Multimodal VLM
  - visual token analysis
  - semantic sparsity
  - MLLM interpretability
  - attention sink
  - token redundancy
date: 2026-05-08
content_hash: b523de97ee1751d1
---

# What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.00510](https://arxiv.org/abs/2603.00510)
**Code**: [https://github.com/EIT-NLP/EmbedLens](https://github.com/EIT-NLP/EmbedLens)
**Area**: Multimodal VLM
**Keywords**: visual token analysis, semantic sparsity, MLLM interpretability, attention sink, token redundancy

## TL;DR
This paper proposes EmbedLens, a probing tool for systematically analyzing the internal structure of visual tokens in MLLMs. It reveals that visual tokens fall into three categories—sink, dead, and alive (approximately 40% are uninformative)—that alive tokens already encode rich semantics before entering the LLM (a "pre-linguistic" property), and that intra-LLM visual computation is redundant for most tasks, such that direct mid-layer injection suffices.

## Background & Motivation
**Background**: MLLMs map patch embeddings from visual encoders such as CLIP into the LLM embedding space via projection layers, achieving notable progress on vision-language tasks. However, the structured organization and processing of visual tokens inside the LLM remain poorly understood.

**Limitations of Prior Work**: Contrastive pretraining encourages global image-text alignment, whereas LLMs process inputs as sequences of local patch-level tokens. This mismatch creates a critical gap: how is globally aligned semantic information distributed across local tokens? Do all patches carry meaningful semantics?

**Key Challenge**: Existing analysis methods (e.g., LogitLens using the unembedding matrix) cannot distinguish whether semantics are intrinsic to the visual encoder/projection layer or injected by the language backbone.

**Goal**: (a) The semantic organization of visual tokens at the input layer; (b) how much information alive tokens carry before entering the LLM; (c) the necessity and optimal depth of intra-LLM visual computation.

**Key Insight**: EmbedLens is proposed to probe token semantics directly in the input embedding space, avoiding confounds introduced by subsequent layer transformations.

**Core Idea**: Visual tokens exhibit a tripartite sink/dead/alive structure; alive tokens are already "pre-linguistic," and intra-LLM visual computation is largely redundant.

## Method

### Overall Architecture
A two-level analytical framework: (1) macro-level—similarity-based clustering reveals the structured organization of tokens; (2) micro-level—EmbedLens probes the fine-grained semantic attributes of individual tokens and clusters. Based on these findings, three token categories are identified and analyzed in depth with respect to their behavior and function.

### Key Designs

1. **EmbedLens Probing Tool**:

    - Function: Directly probes the semantic content of visual tokens in the input embedding space.
    - Mechanism: Computes cosine similarity between a target representation $\mathbf{h}$ and the embeddings $\mathbf{e}_i$ of all vocabulary tokens, returning the Top-k results: $\text{EmbedLens}(\mathbf{h}) = \text{TopK}_{i \in \mathcal{V}} \frac{\mathbf{h}^\top \mathbf{e}_i}{\|\mathbf{h}\|_2 \|\mathbf{e}_i\|_2}$
    - Design Motivation: Unlike LogitLens, which operates via the unembedding matrix, EmbedLens works in the input embedding space and can thus distinguish whether semantics originate from the projection layer or are injected by the LLM. Experiments confirm higher matching accuracy at shallow and intermediate layers.

2. **Discovery of Three Token Categories**:

    - **Sink Tokens (~10%)**: Embeddings are nearly identical across images (cosine similarity > 0.99) and carry no image-specific semantics. ViT sinks originate from CLIP (high L2 norm); LLM sinks align with ⟨bos⟩ after the layer-2 MLP. Pruning them does not degrade performance.
    - **Dead Tokens (~30%)**: The largest cluster, exhibiting high cross-image similarity; EmbedLens retrieves fragmented, semantically void subwords. Pruning them yields performance improvements. Their representations remain nearly unchanged throughout the network, and they receive less than 8% of cross-modal attention.
    - **Alive Tokens (~60%)**: Cluster near text-semantic centroids and serve as the sole carriers of image-specific semantics.

3. **"Pre-linguistic" Property of Alive Tokens**:

    - Core Finding: Individual alive tokens can simultaneously encode multiple semantic attributes (object identity, color, shape, OCR).
    - Validation: A patch-level compression benchmark is constructed by rendering individual objects or characters within exactly one visual patch, then testing whether the VLM can decode multiple semantic traces from a single patch.
    - Key Insight: Among correctly identified objects, the majority of color and counting questions are also answered correctly, demonstrating that a single patch indeed encodes multiple semantic dimensions.

4. **Redundancy of Intra-LLM Visual Computation**:

    - Overall Finding: For most standard tasks, completely bypassing visual-only FFN and self-attention layers has negligible or even positive impact on performance.
    - Shallow-Layer Processing Is Unnecessary: The vector norms of alive tokens naturally align with intermediate LLM layers; the projector intentionally amplifies visual norms to bypass redundant shallow-layer processing.
    - Experimental Validation: Scaling visual embeddings down to 0.01× reintroduces text-like transformations but degrades performance, indicating that high norm is an intentional design choice.

## Key Experimental Results

### Main Results — Impact of Sink Token Pruning

| Method | General | OCR | CV Centric | Hallu. | Avg. |
|--------|---------|-----|------------|--------|------|
| LLaVA-v1.5 7B | 58.7 | 36.9 | 54.0 | 61.1 | 52.7 |
| −Sink(LLM) | 58.6 | 37.0 | 54.4 | 61.3 | 52.7 |
| −Sink(ViT) | 58.7 | 37.2 | 53.8 | 61.1 | 52.8 |
| −All Sinks | 58.6 | 37.2 | 54.1 | 61.3 | 52.8 |

### Ablation Study — Dead Token Pruning

| Method | General | OCR | CV Centric | Hallu. | Avg. |
|--------|---------|-----|------------|--------|------|
| Original | 58.7 | 36.9 | 54.0 | 61.1 | 52.7 |
| −Dead | 58.7 | 37.1 | **57.7** | 61.2 | **53.7** |
| −Same # Remaining | 57.2 | 34.3 | 48.5 | 60.2 | 50.1 |

### Key Findings
- Removing approximately 40% of sink and dead tokens yields performance improvement rather than degradation (52.7 → 53.7), confirming input-level semantic sparsity.
- Randomly removing the same number of alive tokens causes a notable performance drop (50.1), confirming that alive tokens are the sole information carriers.
- Dead token representations remain nearly unchanged throughout the network (cross-layer cosine similarity approaching 1) and receive minimal cross-modal attention.
- The layer-2 MLP in the LLM acts as a "sink aligner," projecting LLM sink tokens into a direction nearly identical to ⟨bos⟩.
- The L2 norms of visual tokens substantially exceed those of text tokens, suppressing effective transformation at shallow layers—an intentional design of the projector.

## Highlights & Insights
- **Tripartite Classification (sink/dead/alive)**: The first work to empirically delineate the functional roles of visual tokens, providing theoretical grounding for token pruning—directly discarding 40% of uninformative tokens suffices.
- **EmbedLens Probe**: A model-agnostic practical tool validated across multiple model families including LLaVA, Qwen-VL, and InternVL, demonstrating higher accuracy than LogitLens at shallow layers.
- **Architectural Implications of the Pre-linguistic Finding**: Since alive tokens already encode sufficient semantics, intra-LLM visual computation is redundant, suggesting that (a) visual FFN/self-attention can be bypassed, and (b) visual tokens can be injected at intermediate rather than first layers.
- **Fine-grained Color Bias Finding**: Model color predictions frequently rely on background statistics rather than the target object itself, exposing a systematic bias in VLMs.

## Limitations & Future Work
- The analysis is primarily based on the LLaVA-1.5 architecture; although consistency is verified on InternVL and Qwen-VL, broader coverage of visual encoders (e.g., SigLIP) is lacking.
- The tripartite categorization relies on clustering, and the proportions may vary across images.
- The patch compression benchmark used for "pre-linguistic" validation is highly controlled (70 images), and its generalizability to natural images remains to be verified.
- No concrete efficient architecture is proposed to exploit these findings; the contribution is primarily mechanistic insight.
- The optimal position for mid-layer injection may vary across tasks and models.

## Related Work & Insights
- **vs. FastV/VLM-Pruner and other pruning methods**: These methods prune based on attention or redundancy heuristics, whereas this paper reveals from an information-theoretic perspective that approximately 40% of tokens are inherently uninformative, providing more fundamental theoretical grounding.
- **vs. LogitLens**: LogitLens decodes at the output side via the unembedding matrix and cannot distinguish the source of semantics; EmbedLens operates at the input side, yielding a cleaner signal.
- **Connection to Other Papers in This Batch**: Complements "When Token Pruning is Worse than Random"—this paper explains from a representational structure perspective why deep-layer pruning is equivalent to random pruning (information has already dissipated), while that paper validates the same observation from an information quantification perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal the tripartite structure of visual tokens and the pre-linguistic property, with deep insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-model validation, though the patch benchmark is small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical progression, advancing systematically from macro to micro.
- Value: ⭐⭐⭐⭐⭐ Provides critical mechanistic understanding for the design of efficient MLLM architectures.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[CVPR 2026\] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)
- [\[CVPR 2026\] Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)

<!-- RELATED:END -->
