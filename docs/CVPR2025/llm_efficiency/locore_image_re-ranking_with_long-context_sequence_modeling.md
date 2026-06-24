---
title: >-
  [Paper Note] LOCORE: Image Re-ranking with Long-Context Sequence Modeling
description: >-
  [CVPR 2025][LLM Efficiency][Image Re-ranking] This paper proposes LoCoRe (Long-Context Re-ranker), achieving list-wise image re-ranking based on local descriptors for the first time. By leveraging the Longformer long-context sequence model to process the local descriptors of both the query image and the entire candidate list simultaneously, LoCoRe significantly improves re-ranking performance by capturing transitive relations among candidate images.
tags:
  - "CVPR 2025"
  - "LLM Efficiency"
  - "Image Re-ranking"
  - "Long-Context Sequence Modeling"
  - "Local Descriptors"
  - "Longformer"
  - "List-wise Learning"
date: 2026-05-08
content_hash: 81758a6c64410943
---

# LOCORE: Image Re-ranking with Long-Context Sequence Modeling

**Conference**: CVPR 2025  
**arXiv**: [2503.21772](https://arxiv.org/abs/2503.21772)  
**Code**: [GitHub](https://github.com/MrZilinXiao/LongContextReranker)  
**Area**: LLM Efficiency  
**Keywords**: Image Re-ranking, Long-Context Sequence Modeling, Local Descriptors, Longformer, List-wise Learning

## TL;DR

This paper proposes LoCoRe (Long-Context Re-ranker), achieving list-wise image re-ranking based on local descriptors for the first time. By leveraging the Longformer long-context sequence model to process the local descriptors of both the query image and the entire candidate list simultaneously, LoCoRe significantly improves re-ranking performance by capturing transitive relations among candidate images.

## Background & Motivation

**Background**: Image retrieval typically follows a two-stage paradigm: first, retrieving a candidate list quickly using global descriptors, followed by a second-stage fine-grained re-ranking. The re-ranking stage commonly utilizes local feature descriptors for pairwise similarity estimation.

**Limitations of Prior Work**:
- **Pairwise re-ranking** (e.g., RRT, CVNet, AMES) only compares the query with a single candidate image at a time, failing to exploit the relationships among candidate images.
- **List-wise re-ranking** (e.g., SSR Rerank), although capable of considering candidate relationships, relies solely on global descriptors and lacks the fine-grained information provided by local features.
- Pairwise re-rankers require $K$ forward passes to process $K$ candidates, incurring large computational overhead.

**Key Challenge**: Although local descriptors provide fine-grained matching capabilities, each image contains multiple descriptors. Processing the local descriptors of all candidate images simultaneously poses a significant challenge due to the massive sequence length.

**Goal**: How to achieve list-wise re-ranking to exploit transitive relationships among candidate images without sacrificing the fine-grained advantages of local descriptors.

**Key Insight**: Drawing inspiration from sequence labeling and extractive question-answering paradigms in NLP, image re-ranking is formulated as a token-level classification problem over long sequences.

**Core Idea**: Concatenate the local descriptors of the query and all candidate images into an ultra-long sequence, model the contextual dependencies using Longformer, and achieve list-wise re-ranking via token-level classification.

## Method

### Overall Architecture

LoCoRe concatenates the local descriptors of the query image and $K$ candidate images into a single long sequence, which is then fed into a pre-trained Longformer model. The model performs binary classification (positive or negative image) for each token. During inference, token scores belonging to the same image are aggregated to serve as the similarity score for that image.

### Key Designs

1. **Local Descriptor Serialization and Separator Tokens**:
    - **Function**: Organizing local descriptors of multiple images into a processable long sequence.
    - **Mechanism**: The sequence is formatted as $\text{[query, SEP, gallery\_1, SEP, ..., gallery\_K, SEP]}$, where each image contributes $L$ local descriptors and $\text{SEP}$ is a learnable separator token. The total sequence length is $M = (L+1)(K+1)$, which equals $5,050$ tokens by default when $L=50$ and $K=100$.
    - **Design Motivation**: Separator tokens both demarcate image boundaries and serve as anchors for global attention.

2. **Query Global Attention Mechanism**:
    - **Function**: Ensuring long-range dependency modeling on top of Longformer's sliding window attention.
    - **Mechanism**: All tokens of the query image and all $\text{SEP}$ tokens are designated as global attention tokens (which symmetrically attend to all tokens), while the remaining tokens only participate in local window attention. This guarantees linear computational complexity without losing global information.
    - **Design Motivation**: Removing the global query attention causes the R@1 to plunge from 82.4% to 60.7%, proving its indispensability.

3. **Gallery Shuffle Training and Token-level Classification**:
    - **Function**: Preventing position bias shortcuts and enabling end-to-end training.
    - **Mechanism**: Global retrieval typically ranks positive samples at the front. Directly utilizing this order would allow the model to learn the shortcut of "position = label". Therefore, the order of candidates is randomly shuffled during training. All $(L+1) \times K$ tokens are trained using the BCE loss, and the token scores of the same image are aggregated during inference.
    - **Design Motivation**: Without shuffling during training, the model completely degenerates (the mAP becomes identical to that of global retrieval).

### Loss & Training

- **Loss Function**: Binary Cross-Entropy Loss (BCELoss) computed over all gallery tokens.
- **Inference Aggregation**: $\text{SEP}$ token score, average token score, or the first token score (yielding comparable performance).
- **Sliding Window Strategy**: During inference, if the number of candidates $N > K$, the model slides from the end of the list backward with a window size $K$ and step size $S$, averaging the scores in overlapping areas.
- **Model Initialization**: LoCoRe-small is initialized from the first 6 layers of longformer-base-4096, while the base version is initialized from all 12 layers. The positional embeddings are linearly interpolated to extend from 4,096 to 5,120.
- **Training Configuration**: AdamW optimizer, learning rate of 5e-5, 8 $\times$ A100 GPUs, global batch size of 128.

## Key Experimental Results

### Main Results

**Landmark Retrieval (ROxf, RPar)**:

| Setting | Method | ROxf+1M Hard | RPar+1M Hard |
|------|------|-------------|-------------|
| RN50-DELG | CVNet Reranker | +13.4 mAP | +13.8 mAP |
| RN50-DELG | LoCoRe-base | **+17.8 mAP** | **+13.8 mAP** |

**Metric Learning Benchmarks**:

| Dataset | Metric | Global | RRT | LoCoRe-base |
|--------|------|--------|-----|-------------|
| CUB-200 | R@1 | 68.9 | 68.7 | **78.3** |
| CUB-200 | mAP@R | 49.8 | 55.6 | **64.8** |
| SOP | R@1 | 80.8 | 81.9 | **83.8** |
| SOP | mAP@R | 65.1 | 67.2 | **71.0** |
| In-Shop | R@1 | 86.3 | 88.3 | **89.4** |

### Ablation Study

| Ablation Item | R@1 (SOP) | mAP@R (SOP) |
|--------|-----------|-------------|
| Global Retrieval Baseline | 80.8 | 65.1 |
| LoCoRe-tiny | 82.4 | 68.0 |
| w/o Gallery Shuffle Training | 80.7 | 65.1 |
| w/o Global Query Attention | 60.7 | 53.0 |
| LoCoRe-base | **83.8** | **71.0** |

### Key Findings

- **Gallery shuffling is crucial**: Without shuffling, the model degenerates into a repeater of the global retrieval results.
- **Model scaling brings gains**: Performance continuously improves from tiny to small to base.
- **Transitive relationships are effective**: Qualitative analysis indicates that the model indeed utilizes shared local features among candidate images.
- **Significant latency advantage**: LoCoRe-small takes 24.7ms vs. RRT's 74.4ms (re-ranking 100 images).
- **Recurrent models are unsuitable**: Mamba and RWKV perform worse than Transformer.

## Highlights & Insights

1. **Paradigm Innovation**: Feasibility of list-wise re-ranking using local descriptors is demonstrated for the first time, establishing a new re-ranking paradigm.
2. **Transitive Relationship Modeling**: Models transitive relationships among candidates through long context—shared local features between two positive samples can mutually reinforce confidence.
3. **NLP-inspired**: Cleverly reformulates image re-ranking as a token-level sequence labeling task, akin to NER/QA.
4. **Efficiency Advantage**: Processes 100 candidate images in a single forward pass, whereas pairwise methods require 100 passes.
5. **Importance of Training Tricks**: The simple technique of random gallery shuffling is pivotal to the success of the methodology.

## Limitations & Future Work

1. **Context Window Constraint**: The maximum context length of Longformer limits the number of candidates that can be processed in a single pass (default is 100 images).
2. **Suboptimal Performance of Recurrent Models**: Architectures like Mamba and RWKV fail to capture list-wise re-ranking dependencies effectively.
3. **Future Directions**: Exploring decoder-only large models (for longer context) and context parallelization (e.g., RingAttention).
4. **Cross-modal Extensions**: Extending the approach to document retrieval, video re-ranking, etc.

## Related Work & Insights

- **Longformer**: The core backbone for linear-complexity long-sequence modeling.
- **RRT / CVNet / AMES**: Representative pairwise re-ranking methods.
- **Sequence Labeling Tasks (NER, QA)**: Sources of inspiration for the token-level classification design in NLP.
- **Insights for Future Research**: The potential of list-wise learning signals in ranking/recommendation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ACL 2025\] Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](../../ACL2025/llm_efficiency/sliding_windows_full_ranking.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](../../ACL2026/llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[ICLR 2026\] MoM: Linear Sequence Modeling with Mixture-of-Memories](../../ICLR2026/llm_efficiency/mom_linear_sequence_modeling_with_mixture-of-memories.md)
- [\[ICML 2025\] Curse of High Dimensionality Issue in Transformer for Long-context Modeling](../../ICML2025/llm_efficiency/curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)

</div>

<!-- RELATED:END -->
