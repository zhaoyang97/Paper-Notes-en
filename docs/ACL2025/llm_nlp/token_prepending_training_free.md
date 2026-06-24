---
title: >-
  [Paper Note] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs
description: >-
  [ACL 2025][LLM (Other)][Sentence Embedding] Proposes the Token Prepending (TP) technique, which prepends the sentence embedding decoded from each layer to the beginning of the sentence. This allows early tokens under causal attention to perceive the complete sentence information, significantly improving the quality of LLM sentence embeddings without requiring any training.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Sentence Embedding"
  - "Token Prepending"
  - "Causal Attention"
  - "training-free"
  - "LLM"
date: 2026-05-08
content_hash: e033c88de4d2163a
---

# Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs

**Conference**: ACL 2025  
**arXiv**: [2412.11556](https://arxiv.org/abs/2412.11556)  
**Code**: [https://github.com/fuyuchenIfyw/token_prepending.git](https://github.com/fuyuchenIfyw/token_prepending.git)  
**Area**: LLM NLP / Sentence Representation Learning  
**Keywords**: Sentence Embedding, Token Prepending, Causal Attention, training-free, LLM

## TL;DR

Proposes the Token Prepending (TP) technique, which prepends the sentence embedding decoded from each layer to the beginning of the sentence. This allows early tokens under causal attention to perceive the complete sentence information, significantly improving the quality of LLM sentence embeddings without requiring any training.

## Background & Motivation

**Background**:
   - Sentence embeddings are widely applied in fields such as information retrieval, recommendation systems, sentiment analysis, and document clustering.
   - In recent years, LLMs have demonstrated powerful semantic understanding capabilities, prompting researchers to extract sentence embeddings directly from LLMs without additional fine-tuning.
   - Existing methods such as PromptEOL, MetaEOL, and Pretended CoT guide the LLMs to encode sentence information into the embedding of the last token through prompt engineering.

**Limitations of Prior Work**:
   - Current mainstream LLMs utilize a decoder-only architecture with causal attention, which prevents early tokens in a sentence from attending to subsequent tokens.
   - Even though the final token can attend to all preceding tokens, the representations of early tokens themselves are biased (lacking backward dependency information). This bias propagates sequentially to the final decoded token.
   - Existing approaches like Echo Embeddings address this by repeating the input, which significantly increases sequence length and inference costs.

**Key Challenge**:
   - The causal attention of decoder-only LLMs restricts bidirectional information flow, while training-free sentence embedding methods must still rely on the causal attention framework to operate.
   - How to enable causal attention mechanisms to achieve effects similar to bidirectional attention without altering model parameters or introducing any training?

**Goal**:
   - To resolve the encoding bias in preceding tokens caused by causal attention in decoder-only LLMs, without introducing extra training or substantial inference overhead.

**Key Insight**:
   - By intervening in the layer-to-layer information transmission within the LLM, the decoder-decoded sentence embedding token from each layer is prepended to the beginning of the sentence input for the next layer.

**Core Idea**:
   - Prepending the representation of the last token from the previous layer (which contains complete sentence information) to the start of the sentence in the next layer, enabling all tokens under causal attention to perceive the global sentence semantics.

## Method

### Overall Architecture

The TP method modifies the layer-to-layer propagation process of LLMs and consists of three main components:
- **Initial Token Prepending**: At the input layer, a custom placeholder token `<PST>` is prepended to the beginning of the sentence.
- **Intermediate Token Prepending**: In the first few layers, the representation of `<PST>` is replaced by the representation of the last token (SET) from the preceding layer.
- **Early Exit Strategy**: The sentence embedding is extracted from an intermediate layer rather than the final layer.

### Key Designs

1. **Initial Token Prepending**:

    - **Function**: Inserts a randomly initialized `<PST>` token at the beginning of the sentence text after the embedding layer output.
    - **Mechanism**: The `<PST>` token serves as a placeholder for sentence embedding information, allowing subsequent tokens to attend to it via the causal attention mechanism.
    - **Design Motivation**: Since no sentence embedding is available at the first layer, a randomly initialized placeholder is used.

2. **Intermediate Token Prepending**:

    - **Function**: Between the 2nd layer and the $k$-th layer, the representation at the `<PST>` position is replaced at each layer with the hidden state of the last token (i.e., the SET representation) from the previous layer.
    - **Mechanism**: The $f(h^{l-1})$ function copies the representation of the $n$-th position (the last token) to the $i^*$ position (the PST position), enabling all sentence tokens to perceive the embedding containing complete sentence information through causal attention.
    - **Design Motivation**: The TP operation is executed only in the first $k$ layers, as experiments show that performing it across all layers degrades performance; after the first few layers of TP, the sentence tokens have already perceived sufficient global information.

3. **Early Exit Strategy**:

    - **Function**: Extracts the sentence embedding from an intermediate layer instead of the final layer.
    - **Mechanism**: The last few layers of LLMs are primarily optimized for next-token generation, which contains less general semantic information.
    - **Design Motivation**: Intermediate layers retain richer semantic representations, making them more suitable as sentence embeddings.

### Loss & Training

- **Entirely training-free**: TP does not introduce any new learnable parameters and only modifies layer-to-layer information propagation during inference.
- It adds only a single token to the original sequence, making the inference overhead almost negligible (approximately $1.04\times$ time overhead).

## Key Experimental Results

### Main Results

Results on 7 STS tasks using LLaMA2-7B:

| Method | STS Avg | Inference Time |
|------|---------|----------|
| PromptEOL | 70.03 | $1.00\times$ |
| **PromptEOL + TP** | **77.19 (↑7.16)** | $1.04\times$ |
| MetaEOL | 75.96 | $8.17\times$ |
| **MetaEOL + TP** | **77.91 (↑1.95)** | $8.29\times$ |
| Pretended CoT | 76.86 | $1.18\times$ |
| **Pretended CoT + TP** | **78.02 (↑1.16)** | $1.22\times$ |

- TP achieves the most significant improvement on PromptEOL, boosting performance by 9.01 points on STS-B.
- It also consistently improves the already strong Pretended CoT baseline.
- The inference time overhead is trivial; PromptEOL + TP is only $1.04\times$.

### Key Findings

1. **TP operates best in early layers**: Experiments indicate that performing TP only in the early layers outperforms executing it across all layers.
2. **Early exit strategy is effective**: Extracting embeddings from intermediate layers yields better results than doing so from the final layer.
3. **Cross-model generalization**: TP is highly effective across multiple LLMs, including LLaMA2-7B and Mistral-7B.
4. **Plug-and-play**: TP can be seamlessly integrated with various existing prompting methods.
5. **Comparison with Echo Embeddings**: Echo requires repeating the entire input sequence ($1.67\times$ overhead), whereas TP adds only a single token ($1.04\times$ overhead) and achieves superior performance.

## Highlights & Insights

- **Elegant design concept**: By making minor modifications to layer-to-layer information transmission, it cleverly achieves an effect similar to bidirectional information flow within a causal attention framework.
- **Truly training-free**: It requires no training data or fine-tuning, preserving the general capabilities of the LLM.
- **Almost zero cost**: Compared to Echo's sequence repetition approach, TP only adds one token, resulting in negligible inference overhead.
- **Highly versatile**: As a plug-and-play technique, it can be layered on top of various prompting methods.

## Limitations & Future Work

1. Currently validated only on 7B-scale models; performance on larger-scale models remains to be verified.
2. Random initialization of the `<PST>` token may not be optimal; superior initialization strategies deserve further exploration.
3. The optimal number of layers $k$ for the TP operation needs to be tuned for different models.
4. Evaluation has been mainly conducted on STS tasks; its performance on other downstream tasks (e.g., retrieval, clustering) requires further verification.
5. The selection of the early exit layer also needs adjustment based on individual models, lacking an automated selection mechanism.

## Related Work & Insights

- **PromptEOL** (Jiang et al., 2023): The first work to extract sentence embeddings from LLMs using prompting, serving as the baseline for this study.
- **Echo Embeddings** (Springer et al., 2024): Computes backward dependency by repeating the input; similar in concept but introduces high overhead.
- **MetaEOL** (Lei et al., 2024): Uses ChatGPT-4 to design meta-task prompts, considering sentence representations from multiple perspectives.
- Insights for future sentence embedding research: Representation quality can be significantly enhanced through clever inference-time intervention without modifying model parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of layer-wise prepending is novel, but essentially it functions as an inference-time trick.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Conducted across multiple STS benchmarks, prompting methods, and models, though evaluations on broader downstream tasks are lacking.
- Writing Quality: ⭐⭐⭐⭐ — The method is clearly described, and the diagrams are intuitive.
- Value: ⭐⭐⭐⭐ — Both training-free and plug-and-play, offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)

</div>

<!-- RELATED:END -->
