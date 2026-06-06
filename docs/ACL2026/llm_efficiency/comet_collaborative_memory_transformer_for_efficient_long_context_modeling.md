---
title: >-
  [Paper Note] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling
description: >-
  [ACL 2026][LLM Efficiency][Long context] CoMeT introduces a "global memory + FIFO temporary memory" dual-memory plugin for existing LLMs. By processing inputs in chunks…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Long context"
  - "recurrent Transformer"
  - "dual memory system"
  - "gated update"
  - "pipeline parallelism"
date: 2026-05-08
content_hash: 5c3d9a37f7ea6973
---

# CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling

**Conference**: ACL 2026  
**arXiv**: [2602.01766](https://arxiv.org/abs/2602.01766)  
**Code**: https://github.com/LivingFutureLab/Comet (Yes)  
**Area**: LLM Efficiency / Long Context / Memory Mechanisms  
**Keywords**: Long context, recurrent Transformer, dual memory system, gated update, pipeline parallelism

## TL;DR
CoMeT introduces a "global memory + FIFO temporary memory" dual-memory plugin for existing LLMs. By processing inputs in chunks, it achieves constant memory and linear time complexity. After fine-tuning on only 32k context, it achieves accurate password retrieval at any position within 1M tokens. It also proposes layer-level pipeline parallelism, enabling the fine-tuning of 128k context on 16×80GB GPUs.

## Background & Motivation
**Background**: Standard Transformers are practically unusable for million-token scales due to KV cache growing linearly with context length and the quadratic complexity of attention. Dominant solutions include context compression (LLMLingua / ICAE / Activation Beacon) and finite-state memory models (Transformer-XL / RMT / HMT).

**Limitations of Prior Work**: (1) Compression methods are bounded by information theory, where the compressed length still scales linearly with the original length, only improving constant factors while keeping asymptotic complexity unchanged; (2) Finite-state memory models achieve $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space but generally lack explicit gating to protect critical information and treat all history uniformly, losing fine-grained signals from recent context.

**Key Challenge**: There is an inherent conflict between the "stable retention of compressed information" in long-term memory and the "high-fidelity capture of details" in recent context under a fixed capacity. Excessive compression loses recent details, while retaining too much recent data flushes out critical long-term information.

**Goal**: (1) Design a dual-memory mechanism that simultaneously protects long-term critical information and preserves recent details; (2) Ensure the mechanism is plug-and-play for pretrained LLMs without requiring retraining from scratch; (3) Resolve the pipeline bubble issue of naive context parallelism during ultra-long context training.

**Key Insight**: The authors observe that gated updates in LSTM/GRU allow for selective "overwriting vs. retention," while FIFO queues naturally maintain temporal continuity. Combining these two can respectively address long-term forgetting and recent detail loss.

**Core Idea**: Use "global memory with gated updates" for long-term memory and "temporary memory managed by a FIFO queue" for recent high-fidelity information. Both are prepended as soft prompts to the hidden states of the current chunk.

## Method

### Overall Architecture
The input is split into chunks and processed sequentially. For the $\tau$-th chunk at the $i$-th layer, the model prepends the global memory $\mathbf{G}^i_\tau$ and temporary memory $\mathbf{T}^i_\tau$ to the current hidden states $\mathbf{H}^i_\tau$. Compression tokens $\mathbf{C}^i_\tau$ are interspersed to extract fine-grained local information, and $m$ readout tokens $\mathbf{R}^i_\tau$ are appended to summarize the key content of the current chunk. The computation for a Transformer layer is defined as $\mathbf{H}^{i+1}_\tau, \mathbf{C}^{i+1}_\tau, \mathbf{R}^{i+1}_\tau = \mathrm{TL}(\mathbf{G}^i_\tau, \mathbf{T}^i_\tau, \mathbf{H}^i_\tau, \mathbf{C}^i_\tau, \mathbf{R}^i_\tau)$. All tokens interact via causal self-attention.

### Key Designs

1.  **Global Memory + Gated Update**:
    - **Function**: Distills and protects long-range critical information using a fixed-size persistent state $\mathbf{S}^i_\tau$ to prevent overwriting by new information.
    - **Mechanism**: A Residual Low-Rank Adapter (RLA) transforms states into memory: $\mathrm{RLA}(\mathbf{X}) = \mathbf{X} + \mathbf{W}_{\text{up}}(\mathbf{W}_{\text{down}} \mathbf{X})$, where $\mathbf{W}_{\text{down}} \in \mathbb{R}^{r \times d}$ has rank $r=8$. This ensures the CoMeT module only adds 3.95M parameters (0.098% of Qwen3-4B). State updates use gating: candidate state $\tilde{\mathbf{S}}^i_{\tau+1} = \mathrm{RMSNorm}(\mathbf{R}^{i+1}_\tau)$, gate $\mathbf{g} = \sigma(\mathbf{W}_g([\mathbf{S}^i_\tau; \tilde{\mathbf{S}}^i_{\tau+1}]))$, and final state $\mathbf{S}^i_{\tau+1} = \mathbf{g} \odot \mathbf{S}^i_\tau + (\mathbf{1} - \mathbf{g}) \odot \tilde{\mathbf{S}}^i_{\tau+1}$.
    - **Design Motivation**: Experiments showed that excessive state-to-memory parameters hurt performance; thus, a LoRA-style low-rank residual is used for parameter efficiency and training stability. LSTM-like gating selectively absorbs new info while providing a shortcut path for gradients across chunks, avoiding the "uniform overwriting" problem seen in RMT/HMT.

2.  **Temporary Memory + FIFO Queue**:
    - **Function**: Manages high-fidelity compressed representations of recent chunks via a fixed-capacity FIFO queue, providing a "rolling high-resolution view" of the recent context.
    - **Mechanism**: Compressed tokens $\mathbf{C}^{i+1}_\tau$ from new chunks pass through RMSNorm and the same RLA module before entering the queue; the oldest entry is automatically removed when the queue is full. This "First-In-First-Out" approach naturally maintains temporal continuity.
    - **Design Motivation**: Long-term memory tends to compress specific details (e.g., numbers from 3 chunks ago) into averaged semantics; the FIFO queue compensates for this. From an optimization perspective, FIFO provides direct gradient backpropagation paths for recent chunks, enhancing training stability.

3.  **Layer-Level Pipeline Parallelism**:
    - **Function**: Resolves the significant pipeline bubbles in naive context parallelism where worker $j+1$ must wait for worker $j$ to complete a full forward pass.
    - **Mechanism**: Communication granularity is reduced from the chunk level to the layer level. As soon as worker $j$ finishes layer $i$, it sends the memory state to worker $j+1$, who starts layer $i$ immediately while worker $j$ continues with layer $i+1$. This transforms "serial chain waiting" into an "interleaved pipeline," maximizing GPU concurrency.
    - **Design Motivation**: Naive context parallelism suffers from low resource utilization. CoMeT’s memory transfer is naturally suited for fine-grained pipelining since memory states per layer are independent and fixed in size. This strategy yields a $2.7\times$ speedup, enabling 128k context fine-tuning of Qwen3-4B on 16×80GB GPUs.

### Loss & Training
The backbone used is Qwen3-4B-Instruct-2507. All efficient methods are compared fairly with a memory budget of approximately 3k tokens, fine-tuned for 3 epochs on 32k context. For CoMeT, total memory $ms=2560$ (combined G and T) and rank $r=8$.

## Key Experimental Results

### Main Results (Scrolls benchmark, ~3k memory budget)

| Method | GovRep R-1 | SumScr R-1 | Qspr F1 | Nrtv F1 | QALT F1 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Full Attn (FT) | 61.0 | 32.5 | 40.3 | 22.1 | 64.2 | **42.23** |
| LongLLMLingua (3072 tok) | 38.0 | 28.2 | 35.7 | 19.2 | 65.9 | 37.36 |
| ActivationBeacon | 52.3 | 28.0 | 33.5 | 23.2 | 56.8 | 30.71 |
| Transformer-XL (ws=5120) | 51.2 | 30.7 | 35.5 | 4.5 | 33.6 | 31.83 |
| SWA (ws=5120) | 55.3 | 30.7 | 39.1 | 16.1 | 54.8 | 38.24 |
| HMT (ms=3072) | 47.3 | 29.0 | 16.8 | 11.3 | 53.5 | 30.31 |
| **CoMeT (ms=2560)** | **62.5** | **33.4** | 35.5 | 22.6 | 56.0 | **40.10** |

CoMeT achieves the highest average score among efficient methods and **outperforms** the fine-tuned Full Attention baseline on long-form summarization tasks (GovRep / SumScr), reaching or exceeding full attention performance with only 1/4 effective memory.

In password retrieval tasks: trained on only 32k, CoMeT accurately retrieves passwords at any position within **1M tokens**, achieving $21\times$ inference speedup and $10\times$ memory savings compared to Full Attention at that length. In real-world applications: CoMeT reaches 78.7 accuracy in User Behavior QA with 4k memory, exceeding industrial xRAG baseline by +2.7 and 4k Truncation by +27.4. It also scores 20.27 on Terminal-Bench agent tasks, close to the 21.33 of 128k Full Attention.

### Ablation Study (Short Context Compatibility)

| Dataset | Avg Length | Full Attn EM/F1 | CoMeT EM/F1 |
| :--- | :--- | :--- | :--- |
| 2WikiMQA | 1033 | 75.4 / 80.8 | **75.5 / 81.0** |
| HotpotQA | 1443 | 65.0 / 78.9 | **65.9 / 80.0** |

CoMeT performs comparably to or slightly better than Full Attention on short sequences, proving the plugin does not degrade short-context capabilities.

### Key Findings
- **Summarization tasks can exceed Full Attention**: The authors hypothesize that chunking forces the model to generate high-quality local summaries, acting as an implicit "outline then synthesize" training signal beneficial for global understanding.
- **Dual Mechanism is Essential**: Removing gating (becoming RMT-style uniform overwriting) significantly degrades long-range tasks. Removing FIFO (leaving only global memory) degrades tasks requiring recent details (e.g., QASPER).
- **Layer-Level Pipeline Parallelism vs. Naive Context Parallel**: Achieves a **$2.7\times$ speedup** that scales with the number of GPUs, making "128k training on 16×80GB" feasible.
- During inference, CoMeT shows linear time and constant memory growth, contrasting sharply with the quadratic time and linear memory of Full Attention, empirically validating the improvement in asymptotic complexity.

## Highlights & Insights
- **Dual-Track Memory (Long-term Gated + Recent FIFO)**: An elegant decomposition of the "single memory fits all" approach of RMT/HMT. Long-range critical information and recent details are fundamentally different and benefit from distinct handling. This approach is transferable to other sequence tasks like video understanding or RL trajectory modeling.
- **RLA with rank=8 for 0.098% Parameter Increase**: Using a LoRA-style residual low-rank adapter to transform states prevents the destruction of pretrained knowledge while maintaining stability. Using LoRA as an architectural component rather than just a fine-tuning technique is an insightful design.
- **Layer-Level Pipeline Parallelism**: The bottleneck for long context training is often GPU memory rather than compute. By interleaving layer-level P2P communication, the authors nearly eliminate pipeline bubbles, providing a finer granularity than Megatron-style sequence parallelism.
- **32k Training to 1M Testing Extrapolation**: The $30\times$ extrapolation achieved on NIAH is more robust than RoPE-scaling methods because CoMeT’s fixed memory capacity naturally eliminates positional encoding extrapolation pressure.

## Limitations & Future Work
- The authors acknowledge that while memory remains constant at 1M+ contexts, the capacity is also fixed. Detailed recall tasks exceeding this capacity may suffer (as seen in QALT / QMSum performance falling behind Full Attention).
- Observation: The dual-memory system involves many hyperparameters (global/temporary ratio, FIFO capacity, number of compression and readout tokens), and a systematic sensitivity analysis is lacking; optimal configurations likely vary by task.
- Training was limited to 32k; 1M extrapolation was verified primarily on structured NIAH tasks. The quality of free-form generation at 1M length has not been fully evaluated.
- The FIFO temporary memory uses a fixed token count rather than fixed information content, which might lack flexibility for varying information densities. Adaptive queuing based on entropy or importance could be explored.

## Related Work & Insights
- **vs. Transformer-XL / RMT (NeurIPS 2022)**: Both are recurrent Transformers. However, XL only caches hidden states, and RMT updates memory tokens in-place. CoMeT explicitly splits memory into two tracks, outperforming Transformer-XL significantly on Scrolls (40.10 vs. 31.83).
- **vs. HMT (NAACL 2025)**: HMT uses hierarchical memory but lacks explicit gating and FIFO protections. CoMeT outperforms HMT even with a smaller memory budget ($ms=2560$ vs. $ms=3072$).
- **vs. Activation Beacon / ICAE**: These compression methods suffer from linear scalability in compressed length. CoMeT achieves true $\mathcal{O}(1)$ space; while compression methods change the constant factor, CoMeT changes the asymptotic behavior.
- **vs. Mamba / RWKV**: These linear RNNs require pretraining from scratch. CoMeT acts as a plugin for mature models like Qwen, reusing the existing LLM ecosystem with minimal parameter overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-memory mechanism combines existing ideas (gating + FIFO), but the layer-level pipeline parallelism and 1M extrapolation are impressive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across academic benchmarks, real applications (User Behavior QA, Terminal-Bench), and efficiency analysis against 7+ baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture diagrams and formal notations, though the explanation of layer-level pipeline parallelism is slightly brief (requiring the appendix for full clarity).
- Value: ⭐⭐⭐⭐⭐ Provides a practical efficient long-context solution for academia and a deployable plugin for industry with low training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ACL 2026\] Threshold Differential Attention: Sink-free, Ultra-sparse, and Non-dispersive Long-context Attention](threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)

</div>

<!-- RELATED:END -->
