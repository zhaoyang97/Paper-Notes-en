---
title: >-
  [Paper Note] Proxy Compression for Language Modeling
description: >-
  [ICML 2026][LLM Efficiency][byte-level LM] The authors propose "proxy compression"—where 90% of the training data is fed as short sequences produced by a tokenizer or neural compressor, and 10% is fed as raw UTF-8 bytes…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "byte-level LM"
  - "tokenizer-free inference"
  - "mixed-representation training"
  - "arithmetic coding"
  - "neural compressor"
date: 2026-05-08
content_hash: 6bfccaf7f2314822
---

# Proxy Compression for Language Modeling

**Conference**: ICML 2026  
**arXiv**: [2602.04289](https://arxiv.org/abs/2602.04289)  
**Code**: https://github.com/LZhengisme/proxy-compression (Available)  
**Area**: LLM Efficiency / Byte-level Modeling / Tokenization Alternatives  
**Keywords**: byte-level LM, tokenizer-free inference, mixed-representation training, arithmetic coding, neural compressor

## TL;DR
The authors propose "proxy compression"—where 90% of the training data is fed as short sequences produced by a tokenizer or neural compressor, and 10% is fed as raw UTF-8 bytes, complemented by sentinel tokens and brief in-context translation warm-up. During inference, all compressors are discarded, and the model operates solely on raw bytes. This approach significantly outperforms pure byte-level models under fixed compute budgets and matches or exceeds tokenizer baselines at scale.

## Background & Motivation
**Background**: Modern LMs are almost entirely built upon "external fixed tokenizers." Methods like BPE or SentencePiece compress UTF-8 bytes into tokens to keep training sequence lengths manageable; arithmetic coding combined with small byte LMs follows a similar compression logic. While tokenizers maximize training efficiency, tokens are permanently hard-coded into the model interface.

**Limitations of Prior Work**: Hard-linking tokenizers leads to numerous well-documented side effects: prompt-boundary issues, retokenization drift, glitch tokens (e.g., "SolidGoldMagikarp"), bias against low-resource languages, and poor adversarial robustness. Fundamentally, these models learn statistics of the token space rather than becoming true end-to-end byte-level modelers. While pure byte-level training solves these issues, it drastically increases sequence length, reducing the amount of data processed under the same compute budget and resulting in worse convergence than tokenizer models.

**Key Challenge**: Training efficiency (short sequences) $\leftrightarrow$ Inference flexibility (byte-level interface) $\leftrightarrow$ Robustness. Existing solutions can only satisfy two of these at once. Tokenizer models prioritize the first two, while pure byte-level models prioritize the latter two. No solution currently achieves all three.

**Goal**: Retain the efficiency advantages of "compressed short sequences" during training while allowing the inference side to run entirely on raw UTF-8 without architectural modifications (no change to embeddings, tokenizers, or attention), with benefits scaling as the model size increases.

**Key Insight**: Treat external compressors as "training proxies" rather than permanent interfaces. During training, the same model simultaneously learns both representations and automatically establishes internal mappings. During inference, the compressor is discarded, leaving only the byte-level interface. The key observation is that large models have the capacity to embed this cross-representation alignment within their weights.

**Core Idea**: Use a shared vocabulary, add $\langle\text{comp}\rangle/\langle\text{raw}\rangle$ sentinels, perform mixed-representation next-token prediction, and conduct in-context translation pairing warm-up for the first 10k steps. Inference is pure byte-level.

## Method

### Overall Architecture
Training Pipeline: (1) For each sample $x_{\text{raw}}$, replace it with $x_{\text{comp}}=f(x_{\text{raw}})$ with Bernoulli probability $r$ (default 0.9), otherwise keep it raw. (2) Wrap each segment with sentinels as $[\langle\text{raw}\rangle x_{\text{raw}}\langle/\text{raw}\rangle]$ or $[\langle\text{comp}\rangle x_{\text{comp}}\langle/\text{comp}\rangle]$. (3) During the warm-up phase (first 10k steps), enable in-context pairing by concatenating two views of the same sample in one context, linearly increasing $r$ from 0.4 to 0.9. (4) Disable pairing after warm-up and fix $r=0.9$. (5) Inference uses only raw bytes, discarding all compressors. Shared Vocabulary: Indices 0-63 for sentinels, 64-319 for UTF-8 bytes, and the rest for compression symbols (the tokenizer uses the OpenCoder 96,640 vocab; the neural compressor uses 16-bit packs for 65,536 symbols; gzip uses 256 bytes).

### Key Designs

1.  **Tokenizer-based proxy (§2.2)**:
    - **Function**: Uses an existing tokenizer to offline compress raw byte streams into token index sequences as $x_{\text{comp}}$, implementing the simplest instantiation of "training with tokenizer efficiency, inferring without the tokenizer."
    - **Mechanism**: Directly calls OpenCoder BPE, achieving an average compression ratio of approximately $2.9\times$. Tokens are input as vocabulary indices; the only difference from a standard tokenizer model is their presence within $\langle\text{comp}\rangle$ tags and the 10% probability of using raw bytes. Experiments showed that directly using ID representations performed better than re-encoding token IDs into fixed-length byte sequences.
    - **Design Motivation**: Tokenizer outputs are highly stable (Levenshtein distance remains nearly unchanged under 10% character deletion), making it easiest for the LM to learn the "comp $\leftrightarrow$ raw" mapping. It also allows for full offline preprocessing without additional training costs.

2.  **Neural proxy + Entropy segmentation parallelism (§2.3)**:
    - **Function**: Uses a 40M byte-level LM + arithmetic coding to perform optimal entropy coding on byte streams, resulting in a "fuzzy" compressed stream with a compression ratio of $\sim 2.6\times$.
    - **Mechanism**: A small byte LM provides $p(\cdot|\text{ctx})$ for each position, followed by arithmetic coding using equal-information windows. To avoid the speed bottleneck of sequential byte-by-byte encoding, "entropy segmentation" is introduced—per-byte entropy is calculated, high-entropy positions are used as boundaries, and each segment is compressed independently in parallel. Every 16 bits are packed into a symbol. Note that while this mapping is a deterministic injection from raw to comp, the reverse is not strictly injective ("fuzzy"); however, 90%+ of colliding raw chunks share an LCP $\geq 0.8$, differing only in low-entropy tails like whitespace or newlines.
    - **Design Motivation**: While tokenizers are products of manual BPE, neural compressors are theoretically superior. Entropy segmentation is the key to engineering feasibility, as parallelization is necessary to process a 3.3 TB corpus. Structural fuzziness actually helps the model abstract away formatting noise, improving robustness.

3.  **In-context translation pairing + sentinels + high $r$ warm-up (§2.1, 2.5)**:
    - **Function**: Establishes comp $\leftrightarrow$ raw alignment within the weights without making the model dependent on seeing comp during inference.
    - **Mechanism**: (a) Explicitly informs the model of the segment's representation type using $\langle\text{raw}\rangle/\langle\text{comp}\rangle$ sentinels, allowing next-token prediction to be conditioned on the type. (b) Concatenates $[\langle\text{raw}\rangle x_{\text{raw}}\langle/\text{raw}\rangle\langle\text{comp}\rangle x_{\text{comp}}\langle/\text{comp}\rangle]$ (random order) in the same context during warm-up to force the model to see both views. (c) Disables pairing immediately after warm-up to prevent dependency on having a preceding comp segment. $r$ increases from 0.4 to 0.9 to prevent a lack of raw samples early on, which would hinder representation alignment.
    - **Design Motivation**: Without pairing, oracle-translation pass@1 only reaches 30-46%; always-on pairing reaches 95%+ but makes the model reliant on it, slightly decreasing downstream raw-byte pass@1. Warm-up-only ensures both early alignment and independence, serving as the empirical optimal trade-off (validated in Table 3).

### Loss & Training
The only loss is standard next-token cross-entropy (CE), treating raw and comp segments equally. The architecture utilizes EvaByte (efficient byte-level multi-byte prediction). Training consists of 50K steps with a batch size of 2M symbols across five sizes: 0.5B, 1.5B, 4B, 7B, and 14B.

## Key Experimental Results

### Main Results
Pass@1 results on HumanEval-Plus / MBPP-Plus under a fixed 100B symbol training budget (compute roughly matched):

| Task | Model | 0.5B | 1.5B | 4B | 7B | 14B |
|------|------|------|------|----|----|-----|
| HumanEval-Plus | Tokenizer | 17.7 | 18.3 | 28.0 | 28.7 | 29.3 |
|  | Byte-level | 15.9 | 18.3 | 22.0 | 23.8 | 24.4 |
|  | Proxy (Neural) | 13.4 | 18.3 | 22.6 | 26.8 | 29.9 |
|  | Proxy (Tokenizer) | 12.2 | 20.7 | 24.4 | 26.2 | **30.5** |
| MBPP-Plus | Tokenizer | 29.4 | 41.0 | 46.3 | 45.2 | 48.1 |
|  | Byte-level | 25.9 | 33.6 | 41.8 | 41.3 | 42.1 |
|  | Proxy (Neural) | 22.0 | 29.6 | 41.8 | 41.8 | **49.2** |
|  | Proxy (Tokenizer) | 25.4 | 38.4 | 44.4 | 45.5 | **49.5** |

Ours (Proxy) surpasses pure byte-level models at $\geq 1.5\text{B}$ and outperforms the tokenizer baseline at 14B, showing that transfer performance scales with model size.

### Ablation Study

| Configuration | HumanEval-Plus pass@1 (1.5B) | Notes |
|------|-------------------------------|------|
| Always-on pairing | 17.0 | Oracle-translation 96%, but ordinary performance is lower |
| Warmup-only (Default) | **20.7** | Maintains alignment without inducing dependency |
| No pairs | 17.0 | No explicit cross-rep signal |
| Gzip proxy (any ratio) | < Byte-level | Unstable stream, unable to transfer |
| Tokenizer / Neural proxy | Sig. > Byte-level | Stable + Structured |

### Key Findings
- **Positive correlation between proxy gain and model size**: Gains are weak or negative at 0.5B but outperform both byte-level and tokenizer baselines at 14B.
- **Compressor stability is key to transfer**: The tokenizer has the lowest Levenshtein distance, followed by the neural compressor, with gzip being the highest. The first two achieve successful transfer, while gzip fails completely.
- **Inherited robustness of byte-level models**: On ReCode perturbations (function rewrite/format/syntax/docstring), the 7B proxy model achieves a Robust Pass@1 of 19.1 (neural) vs. 14.9 for the tokenizer baseline and 18.7 for the byte baseline, with almost no degradation on format/docstring tasks.
- **In-context vs. Weight internalization**: Always-on pairing raises oracle-translation to 95%+ but slightly lowers ordinary pass@1. This suggests that "translation via context" and "internalization in weights" are distinct paths, with the latter determining downstream performance on pure bytes.

## Highlights & Insights
- The idea that "the external compressor is only a training proxy and is discarded during inference" is an elegant decoupling strategy. This approach could be generalized to any structure involving an "external encoder + internal modeling," such as VAEs in latent diffusion or codecs in audio.
- Using sentinel tokens to explicitly mark different representations allows a single model to perform next-token prediction across multiple representations. This is far simpler than designing multi-branch or multi-decoder architectures and serves as a methodological reference for future multi-modal training.
- "Structured fuzziness" is a counter-intuitive finding: the non-injectivity of the neural compressor acts as a regularizer, smoothing out formatting noise and resulting in robustness that even exceeds that of lossless tokenizers.

## Limitations & Future Work
- Experiments were primarily conducted on code corpora (RefineCode); natural language was only verified at the 1.5B scale. Whether transfer in multilingual or low-resource settings still scales with model size remains to be explored.
- The total vocabulary size (byte + tokenizer + sentinel) slightly increases embedding table memory overhead; the neural proxy requires an additional 40M byte LM to be trained and maintained.
- Inference speed comparisons are not provided: although tokenizer-free, raw byte sequences are longer ($\sim 2.9\times$). The degree to which EvaByte's multi-byte prediction offsets this requires more detailed quantification.
- The total number of bytes seen by the model during training is actually less than that of a pure byte-level model. While the fixed FLOPs comparison is fair, performance in scenarios where both data and FLOPs are limited remains an open question.

## Related Work & Insights
- **vs. Lester 2024 (neural arithmetic coding LM)**: They use the neural compressed stream as the final representation for both training and inference; this work uses it only as a training proxy, reverting to raw bytes for inference.
- **vs. EvaByte / ByT5 / MegaByte**: These are extensions of pure byte-level directions; this work shares byte-level inference but offers order-of-magnitude improvements in training efficiency.
- **vs. Tokenizer + occasional raw mix**: While superficially similar to token-byte mixed training, the combination of sentinels, warm-up pairing, and high $r$ values are critical differences; omitting any of these leads to significant performance drops.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perspective of "compressors as training proxies" is fresh and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 scales, 3 types of proxies, and multiple robust/in-context probes.
- **Writing Quality**: ⭐⭐⭐⭐ Clear storyline with intuitive scaling curves.
- **Value**: ⭐⭐⭐⭐ Provides a feasible path for the long-suppressed direction of "byte-level inference."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](../../ACL2026/llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](../../ACL2026/llm_efficiency/comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)
- [\[ICML 2026\] ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](../../ICLR2026/llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)

</div>

<!-- RELATED:END -->
