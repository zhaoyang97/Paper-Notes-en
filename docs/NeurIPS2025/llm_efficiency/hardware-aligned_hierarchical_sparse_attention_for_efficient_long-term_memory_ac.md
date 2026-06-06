---
title: >-
  [Paper Note] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access
description: >-
  [NEURIPS2025][LLM Efficiency][sparse attention] This paper proposes Hierarchical Sparse Attention (HSA) and the RAMba architecture, which enable Mamba to perform efficient long-range random access through a two-stage tok…
tags:
  - "NEURIPS2025"
  - "LLM Efficiency"
  - "sparse attention"
  - "RNN"
  - "Mamba"
  - "long context"
  - "length generalization"
  - "chunk selection"
  - "hardware-aligned kernel"
date: 2026-05-08
content_hash: 917719aff4a8fe04
---

# Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access

**Conference**: NEURIPS2025
**arXiv**: [2504.16795](https://arxiv.org/abs/2504.16795)  
**Code**: [ant-research/long-context-modeling](https://github.com/ant-research/long-context-modeling)  
**Area**: LLM Efficiency
**Keywords**: sparse attention, RNN, Mamba, long context, length generalization, chunk selection, hardware-aligned kernel

## TL;DR
This paper proposes Hierarchical Sparse Attention (HSA) and the RAMba architecture, which enable Mamba to perform efficient long-range random access through a two-stage token-to-chunk relevance learning mechanism and hardware-aligned kernel design. Pretrained on only 4K context, RAMba achieves 100% accuracy on 64M passkey retrieval.

## Background & Motivation
- RNNs (e.g., Mamba) enjoy linear complexity but suffer from an information bottleneck in their fixed-dimensional hidden states, preventing random access to historical context.
- Full attention in Transformers incurs quadratic training and inference costs with respect to sequence length and generalizes poorly beyond the training length.
- Existing sparse attention methods (NSA, MoBA) adopt chunk selection strategies but learn chunk importance via token-level gradients (chunk-unaware), resulting in inaccurate chunk selection.
- Directly grafting attention onto RNNs undermines their efficiency advantage, creating a trilemma among efficiency, random access, and length generalization.
- KV cache grows linearly with sequence length at inference time, imposing significant memory overhead that limits practical long-context deployment.
- Existing methods (Landmark Attention, DCA) suffer sharp perplexity increases when extrapolating beyond 32× the training length; GCA can extrapolate to 16M but retrieves only once per 64 tokens, limiting flexibility.

## Method

### Overall Architecture (RAMba)
- Built upon Mamba-2, consisting of upper and lower decoders (each with $L/2$ layers) and a shared chunk selection layer in between.
- The lower decoder output is segmented into chunks of size $S$, encoded into chunk memories via a bidirectional Transformer encoder.
- The upper decoder alternates between HSA layers and $G$ Mamba layers; HSA layers perform sparse attention over selected chunks.

### Hierarchical Sparse Attention (HSA)
1. **Chunk Selection**: Each token computes a query from the lower decoder output and performs dot-product scoring against mean-pooled chunk keys, selecting top-$K$ chunks independently per query group.
2. **Stage 1 (token-level attention)**: Standard attention is applied within each selected chunk independently to obtain chunk-level representations $O_{t,k}$.
3. **Stage 2 (chunk-level attention)**: Chunk representations are aggregated via stick-breaking weights (position-encoding-free, based solely on sigmoid of relevance scores).
4. **End-to-end learning**: During backpropagation, chunk weights are adjusted according to each chunk's contribution to next-token prediction, enabling chunk-aware learning.

### Hardware-Aligned Kernel
- Implemented in Triton: each GPU thread handles the query of one token and the KV pairs of its corresponding $K$ chunks.
- Softmax-off-by-one is used to allow tokens within the current chunk to ignore retrieved tokens.
- Backpropagation proceeds in two phases: first accumulating gradients for $Q$ and $w$, then for $K$ and $V$.
- In the forward pass, each thread initializes $O'=0$ and loops over $K$ chunks: loads chunk $K/V$ into SRAM, computes softmax attention, and accumulates with chunk weight $w$.
- Backward Phase 1 (Algorithm 2): each thread $t$ iterates over $K$ chunks to compute $\nabla Q$ and $\nabla w$.
- Backward Phase 2 (Algorithm 3): each thread $i$ iterates over all tokens that selected the chunk to accumulate $\nabla K$ and $\nabla V$.
- The overall design avoids the large memory overhead in naïve implementations caused by each token referencing a different set of $K$ chunks.

### Training and Inference Optimization
- **Memory Reset**: During training, the preceding RNN hidden state is randomly replaced with zero or a random segment-final state to break shortcuts and improve length generalization.
- **Truncated BPTT**: The initial state for each sequence is initialized from the final state of the previous sequence; combined with memory reset, this provides diverse training signals.
- **KV Cache Offloading**: At inference time, token-level KV cache is offloaded to CPU; the GPU retains only the compact chunk-level representation $K^{slc}$ (dimensionality $\lfloor L/S \rfloor \times d$) for chunk selection, loading only selected chunks' KV per step.
- **Shared KV Cache**: All HSA layers share a single KV cache derived from an intermediate layer, requiring only one chunk selection and one CPU–GPU transfer per step, significantly reducing communication overhead.
- **Theoretically Unlimited Memory**: $K^{slc}$ can be further offloaded to a FAISS database for constant GPU memory usage, though in practice its memory footprint is negligible.

## Key Experimental Results

### Long-range Language Modeling (370M model, 4K pretraining)

| Model (370M) | PG19 PPL (4K) | PG19 PPL (64K) | ArXiv PPL (64K) | Code PPL (64K) |
|---|---|---|---|---|
| Transformer (full attn) | 18.61 | >10⁴ | >10⁴ | 2865.51 |
| Mamba-2 | 17.92 | 17.30 | 3.86 | 3.05 |
| Mamba + NSA (w/ m.r.) | 17.87 | 17.31 | 3.87 | 3.05 |
| Mamba + NSA (w/o m.r.) | 17.74 | 17.62 | 4.35 | 3.28 |
| **RAMba (w/ m.r.)** | **17.82** | **17.01** | **3.65** | **3.07** |
| RAMba (w/o m.r.) | 17.63 | 17.11 | 3.87 | 3.21 |

### Downstream Tasks and Retrieval

| Task | RAMba (w/ m.r.) | Mamba-2 | Transformer |
|---|---|---|---|
| Passkey Retrieval 64M | **100%** | ~0% | ~0% |
| RULER S-N 64K | 85.07 | 10.45 | 0.00 |
| RULER MQ-N 64K | 55.22 | 0.00 | 0.00 |
| RULER VT 64K | 55.22 | 8.96 | 0.00 |
| LongBench Overall | 25.7 | 22.4 | 24.8 |
| Downstream AVG (SFT) | **33.64** | 31.03 | 32.82 |
| SQuaD EM/F1 | 48.24/59.17 | 41.33/52.03 | 45.50/56.13 |
| HotpotQA EM/F1 | 22.30/30.53 | 18.70/26.20 | 21.90/29.49 |

## Highlights & Insights
- RAMba is the first Mamba-based model to achieve 100% accuracy on 64M passkey retrieval while pretrained on only 4K context.
- HSA forward pass is 3× faster than NSA and 5–25× faster than full attention at 16K+ context lengths.
- Inference memory is near-constant due to KV cache offloading, enabling theoretically unlimited context.
- The chunk-aware two-stage attention mechanism maintains accurate chunk selection even at context lengths 10,000× beyond the training length.
- The Memory Reset training strategy is simple yet effective and applies to both NSA and HSA.
- At the 2.7B scale, RAMba retains substantial advantages (large margins over Mamba-2 on RULER 64K multi-task benchmarks).
- Stick-breaking weights replace positional encodings, making them naturally suited for length extrapolation.
- Ablation studies are thorough: covering the importance of the chunk encoder, the effect of memory reset, and softmax vs. stick-breaking comparisons.

## Limitations & Future Work
- Experiments are primarily conducted at the 370M scale; 2.7B results are only partially reported, and larger-scale (7B/13B) validation is lacking.
- Performance on harder RULER retrieval tasks (MQ-N, VT) degrades significantly beyond 256K tokens, leaving precise chunk selection at extreme lengths an open problem.
- The bidirectional Transformer encoder introduces an additional 5.4% parameter overhead, and chunk encoding increases prefill time.
- CPU–GPU memory swapping, while acceptable, may become a bottleneck in large-scale deployment (each step transfers $g \times d_h \times K \times S$ parameters).
- Length generalization is sensitive to prompt/task format (passkey 64M vs. S-N only 4M), and robustness remains to be improved.
- The chunk size $S=64$ is fixed; adaptive chunk sizing has not been explored.
- Memory Reset enhances extrapolation but slightly sacrifices in-domain performance (marginally higher 4K PPL).
- Pretraining is conducted solely on the Pile dataset; validation on large-scale diverse corpora is absent.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Two-stage hierarchical attention + chunk-aware learning + stick-breaking weights; conceptually novel and cognitively inspired)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive multi-task and multi-length evaluation with rich ablations, though large-scale model validation is limited)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, detailed algorithmic pseudocode, thorough kernel design explanations, and precise analysis of distinctions from NSA/MoBA)
- **Value**: ⭐⭐⭐⭐⭐ (Addresses the core bottleneck of random access in RNN long-context modeling; 64M extrapolation capability is a breakthrough; open-source and reproducible)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](../../ICML2026/llm_efficiency/stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)
- [\[ACL 2026\] Threshold Differential Attention: Sink-free, Ultra-sparse, and Non-dispersive Long-context Attention](../../ACL2026/llm_efficiency/threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l.md)

</div>

<!-- RELATED:END -->
