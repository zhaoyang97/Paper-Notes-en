---
title: >-
  [Paper Note] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs
description: >-
  [NeurIPS 2025][LLM Efficiency][sparse attention] This paper proposes Dynamic Hierarchical Sparse Attention (DHSA), a hierarchical framework that replaces dense attention with sparse attention via adaptive chunk segmentation, chunk-level similarity prediction, and upsampling to token level — without retraining the base model. On Gemma2/3, DHSA achieves accuracy on par with dense attention while reducing prefill latency by 20–60% and peak memory by 35%.
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "sparse attention"
  - "dynamic chunking"
  - "hierarchical sparsity prediction"
  - "on-device LLM"
  - "long context"
  - "chunk representation"
  - "boundary detection"
date: 2026-05-08
content_hash: 81220f94dc19c784
---

# Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.24606](https://arxiv.org/abs/2510.24606)  
**Code**: [GitHub](https://github.com/xiongsiheng/DHSA)  
**Area**: LLM Efficiency / Sparse Attention / On-Device Deployment
**Keywords**: sparse attention, dynamic chunking, hierarchical sparsity prediction, on-device LLM, long context, chunk representation, boundary detection

## TL;DR

This paper proposes Dynamic Hierarchical Sparse Attention (DHSA), a hierarchical framework that replaces dense attention with sparse attention via adaptive chunk segmentation, chunk-level similarity prediction, and upsampling to token level — without retraining the base model. On Gemma2/3, DHSA achieves accuracy on par with dense attention while reducing prefill latency by 20–60% and peak memory by 35%.

## Background & Motivation

**Background**: Long-context modeling is a core requirement for LLMs, but the $O(L^2)$ complexity of attention makes processing long sequences on edge devices highly challenging. Sparse attention is the dominant optimization direction.

**Limitations of Prior Work**:
   - **Static sparse methods** (e.g., Longformer's sliding window, BigBird's global tokens) use fixed sparsity patterns and cannot adapt to varying attention distributions across inputs.
   - **Existing dynamic methods** (MInference, LM-Infinite, H2O, Scissorhands) rely on predefined templates or heuristic KV cache eviction rules, lacking generality and discarding contextually important tokens.

**Key Challenge**: Efficiency demands reducing attention computation, while accuracy demands preserving critical token interactions. Directly predicting an $L \times L$ token-level sparse mask is itself $O(L^2)$, offering no complexity reduction.

**Goal**: Design a plug-in module that requires no retraining, dynamically predicts attention sparsity patterns, and applies to both the prefill and decode stages.

**Key Insight**: Hierarchical prediction — first perform coarse-grained similarity estimation at the chunk level ($N_c \times N_c$, $N_c \ll L$), then upsample to the token level for fine-grained selection.

**Core Idea**: Chunk-level similarity can be computed at very low cost and effectively proxies token-level importance. Combined with adaptive chunk boundary prediction and length normalization, this enables data-driven dynamic sparse attention.

## Method

### Overall Architecture

DHSA is embedded as a plug-in module in each Transformer layer. Given the token embeddings of the current layer, it outputs a sparse mask $\mathbf{M} \in \{0,1\}^{L \times L}$. The core pipeline is: dynamic chunk segmentation → chunk-level similarity computation → upsampling to token level → TopK selection of retained token pairs.

### Key Designs

1. **Hierarchical Sparsity Prediction**

    - **Function**: The sequence is divided into $N_c$ non-overlapping chunks. A chunk-level similarity matrix $\mathbf{S}_c \in \mathbb{R}^{N_c \times N_c}$ is computed and upsampled to a token-level similarity matrix $\mathbf{S}_t \in \mathbb{R}^{L \times L}$, from which TopK selection (budget $N_b$) is applied per query token.
    - **Mechanism**: Since $N_c \ll L$, chunk-level computation is extremely cheap ($O(N_c^2)$ vs. $O(L^2)$); all token pairs within the same chunk pair share a single importance score.
    - **Design Motivation**: Directly predicting an $L \times L$ mask has the same cost as dense attention. Hierarchical prediction reduces complexity to $O(N_c^2 + L \cdot N_b)$.

2. **Dynamic Boundary Detection**

    - **Function**: A lightweight neural network predicts whether each token position is a chunk boundary. An encoder aggregates key vectors from left and right windows via MHA; features are fused by concatenating $[\mathbf{k}_{\text{left}}, \mathbf{k}_{\text{right}}, |\mathbf{k}_{\text{left}} - \mathbf{k}_{\text{right}}|, \mathbf{k}_{\text{left}} \odot \mathbf{k}_{\text{right}}, \text{sim}(\mathbf{k}_{\text{left}}, \mathbf{k}_{\text{right}})]$, and an MLP outputs a binary classification probability.
    - **Mechanism**: Positions with large content shifts should serve as chunk boundaries (semantic segmentation); left-right window discrepancy is used for detection.
    - **Design Motivation**: Fixed-size chunks are too rigid and cannot adapt to the varying semantic paragraph structure within documents. Adaptive segmentation improves intra-chunk semantic coherence, making chunk-level similarity a more accurate proxy for token-level importance.

3. **Robust Chunk Representation**

    - **Function**: Token embeddings within a chunk are average-pooled and then scaled by $\sqrt{|\mathbf{C}|}$ for length normalization.
    - **Mechanism**: $\mathbf{q}_c = \sqrt{|\mathbf{C}|} \cdot \bar{\mathbf{q}}$, $\mathbf{k}_c = \sqrt{|\mathbf{C}|} \cdot \bar{\mathbf{k}}$. Chunk-level similarity is then $\mathbf{S}_c = \mathbf{Q}_c \mathbf{K}_c^{\top}$.
    - **Design Motivation**:
        - Averaging after padding dilutes representation quality with zero values.
        - Average vectors of chunks with different lengths have different norms, introducing bias in similarity scores. The $\sqrt{|\mathbf{C}|}$ normalization eliminates the effect of chunk length on dot products.

4. **Prefill and Decode Stage Adaptation**

    - **Function**: During prefill, all boundaries are predicted at once and the complete $\mathbf{S}_c$ is computed; during decode, boundaries are incrementally extended and only the newly added row is computed.
    - **Mechanism**: At decode time, previously generated tokens form one additional chunk and the current token forms a singleton chunk, requiring only the last row of chunk similarities to be computed.
    - **Design Motivation**: Avoids redundant recomputation of chunk similarities for previously cached chunks during decoding.

### Loss & Training

- The boundary detector is trained with binary cross-entropy loss, with ground-truth semantic boundary positions as positive samples.
- DHSA does not require retraining the base model; only the lightweight boundary predictor needs to be trained.
- Cross-layer boundary sharing is supported to further reduce overhead at a slight cost to accuracy.

## Key Experimental Results

### Main Results — LongBench (Gemma2-2b-it, budget=2k)

| Method | NrtvQA | Qasper | Mf-en | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum | MultiNews | TriviaQA | SAMSum |
|--------|--------|--------|-------|----------|----------|---------|-----------|-------|-----------|----------|--------|
| Dense | 22.37 | 35.32 | 37.32 | 41.63 | 32.05 | 19.05 | 27.08 | 21.08 | 25.48 | 87.00 | 41.26 |
| Block Sparse | 16.74 | 26.15 | 32.83 | 35.74 | 31.93 | 14.44 | 26.20 | 19.54 | 25.30 | 86.12 | 40.38 |
| **DHSA** | **20.69** | **30.20** | **34.98** | **38.78** | **31.96** | **15.90** | **26.75** | **20.74** | **25.38** | **87.03** | **41.46** |

### Ablation Study — Latency and Memory (NarrativeQA, Gemma2)

| Attention Impl. | Method | Accuracy (%) | Latency (s) | Peak Memory (GB) |
|-----------------|--------|--------------|-------------|------------------|
| eager | Dense | 21.15 | 1.65 | 10.72 |
| eager | Block Sparse | 17.04 | 1.00 | 9.08 |
| eager | **DHSA** | **20.12** | 1.19 | **6.91** |
| torch.sdpa | Dense | 22.37 | 1.10 | 6.33 |
| torch.sdpa | Block Sparse | 16.74 | 0.88 | 9.88 |
| torch.sdpa | **DHSA** | 19.37 | **0.91** | 6.99 |

### Key Findings

1. **Accuracy Retention**: Across 11 LongBench subtasks, DHSA outperforms block sparse on 10 of them and even surpasses dense attention on TriviaQA and SAMSum. On the Needle-in-a-Haystack benchmark, DHSA (1k budget) matches dense attention exactly.
2. **Significant Memory Advantage**: In eager mode, DHSA's peak memory is only 6.91 GB — a 35.5% reduction over dense (10.72 GB) and 24% lower than block sparse (9.08 GB).
3. **Competitive Latency**: In torch.sdpa mode, DHSA achieves 0.91s latency, only 3% slower than block sparse (0.88s), while outperforming it by 2.6 percentage points in accuracy.
4. **Long-Context Scalability**: At sequence lengths of 16k and 32k, dense eager runs out of memory, whereas DHSA operates normally with latency approximately 40–60% of that of sdpa dense.
5. **Boundary Sharing Trade-off**: Cross-layer boundary sharing (DHSA+bs) further reduces overhead but slightly degrades accuracy on some tasks (e.g., Mf-en drops from 34.98 to 31.20).

## Highlights & Insights

- **Hierarchical prediction is the core innovation**: It circumvents the paradox that "predicting an $L^2$ sparse mask is itself $O(L^2)$." Chunk-level prediction compresses the search space by a factor of $(L/N_c)^2$, which is key to achieving actual speedups.
- **Fully data-driven**: DHSA does not rely on predefined attention pattern templates (e.g., A-shape, vertical-slash) and instead learns input-adaptive sparsity patterns automatically, yielding better generalization across tasks.
- **The $\sqrt{|\mathbf{C}|}$ normalization is simple yet critical**: It resolves the representation bias introduced by variable-length chunks. Naive mean pooling renders similarity scores of long and short chunks incomparable.
- **Plug-in design**: The method modifies no base model weights and requires no retraining, making it highly practical for already-deployed on-device models.

## Limitations & Future Work

1. **Latency is not absolutely optimal**: In eager mode, DHSA takes 1.19s versus 1.00s for block sparse — a 19% overhead primarily attributable to boundary prediction and chunk representation computation. Further operator-level optimization is needed for on-device deployment.
2. **No context length extension**: The authors note that the Gemma series lacks reliable context extension implementations, limiting further validation at longer contexts.
3. **Boundary detector training**: Annotated semantic boundary data is required for training the detector, raising the deployment barrier. Unsupervised boundary learning would be preferable.
4. **Hyperparameter sensitivity**: The chunk budget $N_b$ and chunk size still require manual tuning; adaptive learning of these hyperparameters is an important future direction.
5. **Evaluation limited to small models**: Experiments are confined to Gemma2-2b and Gemma3-1b; performance on larger models remains to be validated.

## Related Work & Insights

- **Longformer / BigBird**: Classic static sparse attention methods using fixed sliding windows and global tokens, lacking input adaptability.
- **MInference**: Dynamic sparsity but relying on predefined patterns (A-shape, vertical-slash); the present work is fully data-driven.
- **H2O / Scissorhands**: Dynamic methods based on KV cache eviction with heuristic criteria.
- **PyramidKV**: Pyramid-style KV cache compression; complementary in spirit.
- **Block Sparse Attention (Han Lab)**: MIT Han Lab's block sparse implementation, serving as the primary baseline.
- **Insights**: The hierarchical prediction paradigm can be extended to other attention variants (e.g., cross-attention, multi-query attention); adaptive segmentation may also benefit document chunking in RAG pipelines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of dynamic hierarchical prediction, adaptive boundary detection, and length normalization is novel, with each component having a clear design motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Needle-in-a-Haystack, multi-task LongBench, latency/memory analysis, and comparisons across different attention implementations provide comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Method motivation and pipeline description are clear, with good intuitive explanation of hierarchical prediction.
- **Value**: ⭐⭐⭐⭐ — A practical solution for on-device long-context LLM deployment; the plug-in design lowers the engineering barrier significantly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ICLR 2026\] Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs](../../ICLR2026/llm_efficiency/tactic_adaptive_sparse_attention_with_clustering_and_distribution_fitting_for_lo.md)
- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](../../ICML2025/llm_efficiency/long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)

</div>

<!-- RELATED:END -->
