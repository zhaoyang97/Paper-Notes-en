---
title: >-
  [Paper Note] FocusLLM: Precise Understanding of Long Context by Dynamic Condensing
description: >-
  [ACL 2025][LLM Efficiency][Long Context Understanding] This paper proposes FocusLLM, a framework that dynamically condenses key information from long text by partitioning it into chunks and injecting dynamic prompts into each chunk. These chunks are condensed into candidate tokens using a trainable mechanism, which are then aggregated into the local context via a parallel decoding mechanism to generate the next token. Only requiring an 8K training length and a 0.5B token trai…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long Context Understanding"
  - "Context Compression"
  - "Parallel Decoding"
  - "Dynamic Condensing"
  - "Lossless Information"
date: 2026-05-08
content_hash: 7eb7763c80763dc0
---

# FocusLLM: Precise Understanding of Long Context by Dynamic Condensing

**Conference**: ACL 2025  
**arXiv**: [2408.11745](https://arxiv.org/abs/2408.11745)  
**Code**: [https://github.com/leezythu/focusllm](https://github.com/leezythu/focusllm)  
**Area**: LLM Efficiency  
**Keywords**: Long Context Understanding, Context Compression, Parallel Decoding, Dynamic Condensing, Lossless Information

## TL;DR
This paper proposes FocusLLM, a framework that dynamically condenses key information from long text by partitioning it into chunks and injecting dynamic prompts into each chunk. These chunks are condensed into candidate tokens using a trainable mechanism, which are then aggregated into the local context via a parallel decoding mechanism to generate the next token. Only requiring an 8K training length and a 0.5B token training budget, it successfully extends LLaMA-2 to a 400K context, outperforming all baselines on LongBench and ∞-Bench.

## Background & Motivation
- **Background**: Long context processing in LLMs remains a core challenge. Existing methods fall into three categories: (1) Position embedding modification (PI, NTK, YaRN), which suffers from extrapolation limits; (2) Continual training on long sequences (LongChat, LongAlpaca), which is computationally expensive; (3) Context compression (StreamingLLM, Activation Beacon), which reduces overhead by discarding or compressing tokens.
- **Limitations of Prior Work**:
    - Position embedding methods suffer from perplexity (PPL) explosion on extremely long sequences.
    - Compression methods suffer from **information loss**: in Passkey Retrieval tasks, their accuracy drops sharply as context length grows.
    - The root cause is that token importance **dynamically changes** during decoding—tokens that are unimportant at the current step may become critical in future steps.
    - Existing compression methods make one-time decisions to keep/discard tokens, which cannot adapt to this dynamic nature.
- **Key Challenge**: Low computational overhead vs. Lossless information vs. Efficient training.
- **Key Insight**: Do not discard any tokens. Instead, allow the model to "extract" the most relevant information from each chunk at each decoding step via dynamic prompts.
- **Core Idea**: During each decoding step, splice the local context fragment (dynamic prompt) to the end of each chunk. Generate candidate tokens via newly added trainable parameters, and generate the final token after fusing the candidate tokens of all chunks.

## Method

### Overall Architecture
Given a long text $\{x_1,...,x_S\}$:
1. **Chunking**: Segment the first $m$ tokens into $k$ chunks $C_1,...,C_k$, and treat the remaining tokens as local context.
2. **Dynamic Condensing**: Append a dynamic prompt (a fragment of the local context) to each $C_i$, and generate candidate tokens through a modified decoder.
3. **Parallel Decoding**: Process the $k$ chunks in parallel, and concatenate the Key-Value (KV) states of the $k$ candidate tokens into the local context.
4. **Frozen Decoder Generation**: Use the original (frozen) decoder to generate the next token on the concatenated representations.

### Key Designs
1. **Dynamic Prompt Injection**:
    - Append the last 512 tokens of the local context to each chunk (at inference time).
    - Motivation: Allow the model to decide what information to extract from each chunk based on the **current decoding context**.
    - After generating a new token at each step, append this token to the dynamic prompt (the oldest token can be discarded to maintain a fixed length).
    - Key: The dynamic prompt **evolves dynamically** during the decoding process, ensuring different information is extracted at each step—thus avoiding information loss.

2. **Candidate Token Generation**:
    - Candidate token = **trainable hidden states** corresponding to the last local token $x_S$ in each chunk.
    - Introduce new linear projection parameters $\{W_Q^c, W_K^c, W_V^c, W_O^c\}_l$ for each layer $l$.
    - The query of the candidate token is generated using the new parameters, while the key/value it attends to contains the original tokens + the candidate token itself.
    - The original model parameters are completely frozen; only the newly added parameters are trained.
    - The candidate token is essentially a **conditional compressed representation** of the chunk content—conditioned on the dynamic prompt.

3. **Parallel Decoding**:
    - Candidate token generation for the $k$ chunks is **mutually independent** and can be forward-propagated in parallel.
    - The K/V cache of all candidate tokens is concatenated layer-by-layer to the local context KV cache.
    - The frozen original decoder then generates the next token based on this concatenated representation.
    - Computational complexity is reduced from $O(L^2)$ to $O((L/n)^2)$.

4. **Joint Training with Dual Losses**:
    - **Continuation Loss**: The local context naturally continues the memory tokens; the model is trained to generate the subsequent tokens.
    - **Reconstruction Loss**: Randomly select $L$ consecutive memory tokens as the local context and train the model to reconstruct them.
    - Both losses are indispensable: using only Continuation Loss drops Passkey Retrieval accuracy from 99% to 1.69%; using only Reconstruction Loss degrades generation capability.

### Loss & Training
- Autoregressive loss: Loss is calculated only on the tokens of the local context.
- Training data: RedPajama, sequence length of 3K-8K.
- Training budget: Only 0.5B tokens (vs. 7B tokens for LongLlama).
- Trainable parameters: Approximately 1/3 (only the newly added attention projection parameters).
- Base model: LLaMA-2-7B (chat/base).

## Key Experimental Results

### Main Results

| Dataset | Metric | FocusLLM | Activation Beacon | InfLLM | StreamingLLM |
|--------|------|------|----------|------|------|
| ∞-Bench Avg | Acc | **44.03** | - | 43.05 | 15.64 |
| ∞-Bench Retrieve.Number | Acc | **83.56** | - | 81.69 | 4.41 |
| ∞-Bench Retrieve.PassKey | Acc | **95.76** | - | 99.15 | 4.92 |
| ∞-Bench Retrieve.KV | Acc | **12.40** | - | 0.60 | 0.00 |
| LongBench Avg (Vicuna) | Score | **36.17** | - | 33.24 | 31.92 |
| LongBench HotpotQA | F1 | **40.65** | - | 22.53 | 22.17 |
| LongBench Avg (LLaMA) | Score | **39.01** | 38.54 | - | - |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Continuation Loss Only | PassKey=1.69% | Loss of information recovery capability; Passkey task almost completely fails. |
| Reconstruction Loss Only | PassKey=91.19% | Degradation of new token generation capability; performance on NarrativeQA drops. |
| Local context 1K vs 2K | Slight decrease | Candidate tokens cannot fully replace the information in the local context. |
| Chunk size (256-2048) | Stable PPL | Chunk size has little effect on PPL, indicating that larger chunk sizes can be used with longer context models. |

### Key Findings
- Passkey Retrieval achieves ~99% accuracy up to 400K context, outperforming all baselines (including YaRN-128K's 92.71%).
- StreamingLLM and Activation Beacon score near 0 on retrieval-focused tasks in ∞-Bench—information loss is fatal.
- FocusLLM's perplexity (PPL) in language modeling does not significantly deteriorate as context length increases (at 100K context on PG19, PPL is 10.59 vs. 8.68 of Activation Beacon).
- Visualization Analysis: In the Passkey task, the model only allocates high attention to the candidate tokens of the chunk containing the answer. In QA tasks, multiple candidate tokens are attended to, indicating that the model learns to aggregate information across multiple chunks.
- Memory and Speed: Memory consumption grows more gently compared to the Standard method, achieving higher efficiency than CEPE and LongLlama.

## Highlights & Insights
- The core idea of "dynamic condensing" aligns with the nature of the attention mechanism: allowing the model to autonomously select information based on current demands.
- Extremely high training efficiency: training on a length of only 8K generalizes to 400K, thanks to the natural extrapolation property of chunk processing.
- Comparison with RefreshKV: RefreshKV refreshes a small KV cache during decoding, while FocusLLM dynamically extracts information at each step—similar philosophies but different implementation paths.
- The dual-loss design reveals two essential capabilities for long context understanding: text continuation capability + information recovery capability.

## Limitations & Future Work
- Constrained by hardware, the model was only tested up to 400K; the actual upper limit remains unknown.
- The training data size is relatively small (0.5B tokens); scaling up the training data should further improve performance.
- The local context size is fixed, and adaptive adjustment has not been explored.
- Each decoding step requires $k$ forward passes (though parallelizable), increasing computational overhead.
- **Promising Directions**: (1) Utilize larger chunk sizes on models with longer inherent contexts (e.g., LLaMA-2-32K); (2) Combine with KV quantization to further reduce overhead; (3) Explore adaptive selection of the number of candidate tokens (some chunks might not require candidate tokens).

## Related Work & Insights
- Activation Beacon compresses context through beacon tokens, but one-time compression causes information loss.
- CEPE uses a small encoder to chunk-encode long documents and fuses them into the decoder via cross-attention, but memory does not extrapolate.
- AutoCompressor recursively compresses context, but perplexity (PPL) explodes on long sequences.
- InfLLM stores processed contexts in a memory unit and retrieves them using attention scores, which is conceptually similar but FocusLLM is more end-to-end.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of dynamic prompting, candidate tokens, and parallel decoding is novel, although the general framework of chunking and compression is not.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across Language Modeling, LongBench, ∞-Bench, and Passkey Retrieval, with thorough ablation analysis and intuitive visualizations.
- Writing Quality: ⭐⭐⭐⭐ The architecture diagram is clear, and the method description is systematic, though the abundance of equations may increase the reading burden.
- Value: ⭐⭐⭐⭐⭐ High training efficiency, strong performance, and scalability up to 400K provide extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)

</div>

<!-- RELATED:END -->
