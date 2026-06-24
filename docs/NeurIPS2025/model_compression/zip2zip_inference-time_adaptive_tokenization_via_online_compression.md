---
title: >-
  [Paper Note] zip2zip: Inference-Time Adaptive Tokenization via Online Compression
description: >-
  [NeurIPS 2025][Model Compression][Adaptive Tokenization] This paper proposes zip2zip, which deeply integrates the classical LZW online lossless compression algorithm into the LLM inference pipeline. During decoding, frequently co-occurring tokens are continuously merged into reusable "hypertokens" to dynamically expand the vocabulary. Combined with a dynamic embedding layer and training on compressed-space language modeling, zip2zip enables existing LLMs to acquire inference-…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Adaptive Tokenization"
  - "LZW Compression"
  - "Hypertoken"
  - "Inference Acceleration"
  - "Vocabulary Expansion"
  - "Dynamic Embedding"
date: 2026-05-08
content_hash: 1897cac10abcd9b5
---

# zip2zip: Inference-Time Adaptive Tokenization via Online Compression

**Conference**: NeurIPS 2025
**arXiv**: [2506.01084](https://arxiv.org/abs/2506.01084)  
**Code**: [https://github.com/epfl-dlab/zip2zip](https://github.com/epfl-dlab/zip2zip)  
**Area**: Model Compression
**Keywords**: Adaptive Tokenization, LZW Compression, Hypertoken, Inference Acceleration, Vocabulary Expansion, Dynamic Embedding

## TL;DR

This paper proposes zip2zip, which deeply integrates the classical LZW online lossless compression algorithm into the LLM inference pipeline. During decoding, frequently co-occurring tokens are continuously merged into reusable "hypertokens" to dynamically expand the vocabulary. Combined with a dynamic embedding layer and training on compressed-space language modeling, zip2zip enables existing LLMs to acquire inference-time adaptive tokenization capability with only 10 GPU-hours of LoRA fine-tuning, achieving 15–40% reduction in input/output sequence length and up to 40% reduction in end-to-end decoding latency, with negligible downstream task performance degradation.

## Background & Motivation

The tokenizer is a critical component affecting both model performance and inference cost in large language models (LLMs). Nearly all mainstream LLMs employ static tokenizers based on Byte Pair Encoding (BPE), whose vocabularies are optimized once on large-scale general web corpora and then fixed. This "one-size-fits-all" static vocabulary performs reasonably well on general text, but tokenization efficiency degrades significantly for domain-specific or language-specific inputs. In fields such as biomedical text, programming code, and mathematical expressions, static tokenizers often fragment specialized terminology into excessive tokens, resulting in token sequence lengths 2–3× longer than general text, directly increasing inference computation cost, latency, and user fees.

More compact tokenization offers three core advantages: (1) a larger effective context window—more content can fit within the same context length limit; (2) lower computational and economic cost—attention complexity scales directly with sequence length; (3) shorter response times—autoregressive decoding latency is proportional to the number of output tokens.

Two prior approaches address the inefficiency of static tokenization but both have limitations. The first is **domain-adaptive tokenizers**, which train extended vocabularies for specific target domains or languages. While effective, this requires maintaining separate tokenizers for each domain and does not scale. The second is **enlarging the general vocabulary**—commercial LLMs have expanded vocabulary sizes from 32K to 128K (Llama-3) and even 200K (Phi-4), but research shows that simply enlarging the vocabulary yields diminishing returns, and overly large vocabularies may even harm model performance.

These limitations point to a core need: an **inference-time adaptive** tokenization mechanism that dynamically adjusts the vocabulary based on the characteristics of the current input text, without retraining the model or maintaining multiple tokenizer variants. Realizing this goal faces architectural challenges—the embedding layer and language modeling head of a Transformer model are static matrices tied to a fixed vocabulary size. How to dynamically expand the vocabulary at inference time and enable the model to correctly handle new tokens is the central problem this paper addresses. Inspired by the LZW algorithm used in ZIP compression, the paper proposes the zip2zip framework, achieving truly inference-time adaptive tokenization.

## Method

### Overall Architecture

The name zip2zip pays homage to seq2seq, where the two "zip"s represent input-side compression (first zip) and output-side compression (second zip), simultaneously reducing the token count of both input and output. The overall framework comprises three core components working in concert to achieve dynamic tokenization at inference time.

The system workflow proceeds as follows: input text is first converted to a base token sequence via a standard BPE tokenizer, then an online LZW compression module merges frequently co-occurring tokens into hypertokens to form a more compact representation. Since hypertokens are not in the original vocabulary, their embedding vectors are dynamically computed by a hyper-encoder at inference time. The mixed sequence of embedded base tokens and hypertokens is fed into standard Transformer layers to produce contextualized hidden states. At the output layer, hypertoken unembedding vectors are dynamically appended to the original language modeling head matrix, enabling the model to perform joint softmax sampling over the union of the base vocabulary and hyper vocabulary. The sampled result may be either a base token or a hypertoken; after generation, LZW decoding restores all hypertokens to their constituent base token sequences.

A key design constraint is **vocabulary consistency updates**: the hypertoken sets in the embedding layer and unembedding layer must remain synchronized; otherwise two types of errors arise—either a hypertoken is generated that cannot be decoded, or decoding is attempted for a nonexistent hypertoken. The system also employs a **hyper-embedding cache**: since hypertoken embeddings are context-independent (depending only on their constituent base tokens), they can be cached and reused across decoding steps analogously to the KV cache. Only the embeddings of newly created hypertokens need to be computed at each step, maintaining a constant per-step computational overhead.

### Key Designs

**LZW Tokenizer: Online Incremental Vocabulary Construction.** zip2zip adopts the Lempel-Ziv-Welch (LZW) compression algorithm as the core engine for dynamic vocabulary expansion. LZW is a dictionary-based lossless compression method that incrementally builds a codebook to encode recurring sequence patterns as shorter symbols. In zip2zip, the codebook is initialized with the base vocabulary, then continuously scans the token stream during inference, merging frequently co-occurring token pairs into new hypertokens that are added to the codebook. The maximum merge length $M$ limits the maximum number of base tokens a single hypertoken can contain (default $M=3$), controlling the growth rate of the dynamic vocabulary.

LZW offers three key advantages over other compression methods (such as BPE itself) that make it particularly suited for zip2zip: (1) **Online nature**—a hypertoken created at step $t$ can be immediately reused at step $t+1$, whereas BPE requires access to the complete sequence and can only operate offline; (2) **Self-containedness**—the compressed token sequence can be independently decoded to recover the original input without transmitting a separate codebook, maintaining compatibility with existing LLM libraries and interfaces; (3) **Unambiguity**—when both base tokens and hypertokens are applicable, the LZW algorithm provides a consistent, deterministic selection rule. Additionally, hypertokens support recursive merging—existing hypertokens can be further merged with other tokens to form nested super-tokens, enabling multi-level compression.

**Dynamic Embedding Layer (Hyper-Encoder): Mapping from Base Embeddings to Hyper Embeddings.** Hypertokens are absent from the original model's embedding matrix, so a method is needed to compute their embedding vectors at runtime. zip2zip introduces a hyper-encoder $f_\phi: \mathcal{V}^M \to \mathbb{R}^d$, a two-layer Transformer encoder that takes the embeddings of the $M$ base tokens constituting a hypertoken as input and outputs the corresponding hypertoken embedding $h = f_\phi(y_{1:M}) \in \mathbb{R}^d$. For hypertokens containing fewer than $M$ base tokens, the input sequence is padded to length $M$. Mathematically, the hyper-encoder is essentially a nonlinear dimensionality reducer mapping the $M \times d$-dimensional space (concatenated base token embeddings) to the $d$-dimensional space. For the output-side unembedding layer, if the model uses tied embeddings (sharing the embedding and unembedding matrices), the same hyper-encoder output can be directly reused; otherwise, a separate hyper-encoder must be trained to produce unembedding vectors.

### Loss & Training

zip2zip's training employs a joint optimization objective consisting of two components:

**Primary Loss: Language Modeling Loss on Compressed Sequences.** zip2zip trains on LZW-compressed token sequences rather than the original base token sequences. Given a corpus distribution $\mathcal{D}$ and online compression algorithm $\mathcal{C}$ (i.e., LZW), the training objective minimizes the causal language modeling loss on compressed sequences:

$$\min_{\theta, \phi} \mathbb{E}_{y \sim \mathcal{D}} \left[ -\log \pi_{\theta, \phi}(\mathcal{C}(y)) \right]$$

where $\theta$ denotes base model parameters and $\phi$ denotes hyper-encoder parameters. Training data preprocessing requires only a single pass: the corpus is tokenized with the standard tokenizer, then LZW compression is applied independently at the document level to avoid learning irrelevant cross-document patterns. Training remains fully parallelizable—using standard causal masking, the model can predict the next token (whether base token or hypertoken) at each position in parallel. To avoid the need for sequential codebook updates at inference time, a fixed codebook is precomputed for the complete input sequence during training, ensuring efficiency and compatibility with standard training pipelines.

**Auxiliary Loss: Autoencoding Reconstruction Loss.** To ensure that hypertoken embeddings preserve the semantic information of their constituent base tokens, zip2zip introduces an autoencoding auxiliary loss. Specifically, a decoder $f_\psi: \mathbb{R}^d \to \mathcal{V}^M$ is introduced and trained to reconstruct the original base tokens from the hypertoken embedding. The complete joint optimization objective is:

$$\min_{\theta, \phi, \psi} \mathbb{E}_{y \sim \mathcal{D}} \left[ -\log \pi_{\theta, \phi}(\mathcal{C}(y)) \right] + \lambda \mathbb{E}_{y_{1:M}} \left[ \Delta(y_{1:M}, f_\psi(f_\phi(y_{1:M}))) \right]$$

where $\Delta$ is the reconstruction loss function (cross-entropy loss) and $\lambda = 0.1$ controls the trade-off between the two loss components. The reconstruction loss can be understood as an autoencoding constraint—hypertokens serve as compressed latent representations, and the reconstruction objective encourages them to preserve semantic content, making the compression process as lossless as possible. Experiments observe that the reconstruction loss converges to near zero during training, indicating that the model can nearly perfectly recover the original base token sequence corresponding to each hypertoken, validating the high information-preserving nature of the learned compression.

**Parameter-Efficient Training Strategy.** To reduce training cost, zip2zip applies LoRA for parameter-efficient fine-tuning of the base model while training the newly introduced hyper-embedding and hyper-unembedding modules. The entire training process requires approximately 10 H100 GPU-hours for Phi-3-4B and approximately 40 H100 GPU-hours for Phi-3-14B, using only 100 million training tokens. This means that any research laboratory with modest GPU resources can upgrade an existing LLM to a zip2zip model with adaptive tokenization capability.

**Theoretical Guarantee: Entropy Invariance.** The paper proves an important theoretical result—for any lossless mapping $g$, there always exists a corresponding transport model distribution in the compressed space whose cross-entropy is identical to that in the original space. Formally, if $g$ is a bijective lossless compression, then $H(P_Z) = H(P_X)$ and $H(P_Z, p_\gamma) = H(P_X, p_\theta)$. This implies that lossless compression does not fundamentally compromise model performance—the optimal model can achieve the same likelihood in the compressed representation as in the original space. In practice, due to optimization challenges (gradient descent may converge more slowly or settle in suboptimal regions), the trained model may not perfectly approximate the theoretically optimal transport model.

## Key Experimental Results

### Main Results: Token Efficiency Improvement

The paper evaluates Token efficiency (bytes per token, higher is better) of 4 mainstream LLM tokenizers with and without zip2zip across 5 representative domains (code, math, chat, multilingual, web):

| Tokenizer | Domain | Base Efficiency | +zip2zip | Gain |
|-----------|--------|----------------|---------|------|
| Llama-3-128K | Code | 4.1 | 6.3 | +54% |
| Llama-3-128K | Math | 2.7 | 4.0 | +48% |
| Llama-3-128K | Chat | 5.1 | 6.4 | +25% |
| Llama-3-128K | Multilingual | 3.8 | 4.7 | +24% |
| Llama-3-128K | Web | 4.6 | 5.4 | +17% |
| Qwen-2-150K | Code | 4.0 | 6.2 | +55% |
| Qwen-2-150K | Math | 2.3 | 3.7 | +61% |
| Phi-4-200K | Code | 4.1 | 6.3 | +54% |
| Phi-4-200K | Math | 2.7 | 4.1 | +52% |
| Gemma-3-256K | Code | 3.3 | 5.6 | +70% |
| Gemma-3-256K | Math | 2.3 | 3.7 | +61% |

zip2zip consistently and significantly improves token efficiency across all tokenizers and domains. Gains are most pronounced in structured domains (code and math), with Gemma-3 achieving up to 70% improvement on code. Notably, tokenizers with larger vocabularies do not always achieve better token efficiency—Gemma-3 (256K vocabulary) has a lower base code efficiency of 3.3 compared to Llama-3 (128K vocabulary) at 4.1, further demonstrating that simply enlarging the vocabulary does not effectively address domain adaptation.

### Ablation Study: Downstream Task Performance and Perplexity

| Model | Method | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WG | GSM8K |
|-------|--------|-------|-------|-----------|------|------|-----|-------|
| Phi-3.5-4B | Baseline | 0.60 | 0.83 | 0.66 | 0.46 | 0.79 | 0.75 | 0.82 |
| Phi-3.5-4B | Continual Pretraining | 0.60 | 0.82 | 0.63 | 0.47 | 0.82 | 0.75 | 0.40 |
| Phi-3.5-4B | zip2zip | 0.57 | 0.83 | 0.61 | 0.46 | 0.82 | 0.75 | 0.15 |
| Phi-3-14B | Baseline | 0.62 | 0.80 | 0.70 | 0.51 | 0.83 | 0.76 | 0.84 |
| Phi-3-14B | Continual Pretraining | 0.62 | 0.88 | 0.66 | 0.52 | 0.87 | 0.80 | 0.52 |
| Phi-3-14B | zip2zip | 0.62 | 0.86 | 0.68 | 0.51 | 0.85 | 0.79 | 0.25 |

Byte-level perplexity (lower is better):

| Model | Method | Wikitext | The Pile | mC4 | dC4 |
|-------|--------|---------|---------|-----|-----|
| Phi-3.5-4B | Baseline | 1.58 | 1.79 | 1.88 | 1.74 |
| Phi-3.5-4B | zip2zip | 1.69 | 1.95 | 2.00 | 1.82 |
| Phi-3-14B | Baseline | 1.43 | 1.72 | 1.82 | 1.67 |
| Phi-3-14B | zip2zip | 1.56 | 1.90 | 1.96 | 1.75 |

### Inference Efficiency (Phi-3.5-4B on H100 GPU)

| Setting | Method | Prefill (tok/s) | Decode (tok/s) | Prefill Gain | Decode Gain |
|---------|--------|----------------|----------------|-------------|-------------|
| 256+256 | Baseline | 700.9 | 56.2 | - | - |
| 256+256 | zip2zip | 936.6 | 61.4 | +33.6% | +9.3% |
| 512+256 | Baseline | 1347.2 | 54.4 | - | - |
| 512+256 | zip2zip | 2722.1 | 79.8 | +102.6% | +46.6% |
| 1024+256 | Baseline | 2689.4 | 52.8 | - | - |
| 1024+256 | zip2zip | 4326.7 | 61.5 | +60.9% | +16.6% |
| 2048+256 | Baseline | 4993.2 | 53.1 | - | - |
| 2048+256 | zip2zip | 9258.1 | 61.9 | +85.4% | +16.5% |

### Key Findings

1. **Token efficiency gains are most pronounced in structured text**—bytes/token improvements reach 48–70% in code and math domains, as these domains contain large numbers of repetitive fixed patterns (e.g., function names, keywords, formula symbols). Chat and web text yield smaller gains (15–28%) due to greater content diversity and fewer repetitive patterns.

2. **Downstream task performance is broadly robust**—performance loss on most NLP benchmarks (ARC, PIQA, WinoGrande, etc.) is minimal, typically within 1–3 percentage points. However, GSM8K (mathematical computation) exhibits significant degradation (Phi-3.5-4B drops from 0.82 to 0.15), likely because numerical computation intrinsically relies on token-level precise operations, and adaptive tokenization may alter the token segmentation of digits, thereby disrupting arithmetic capability.

3. **Perplexity increases moderately but remains controlled**—byte-level perplexity increases by approximately 5–10%, consistent with theoretical predictions: while the entropy invariance theorem guarantees that the theoretically optimal performance is unchanged, practical optimization does not perfectly approximate the transport model.

4. **Inference throughput improves substantially, especially on datacenter-grade GPUs**—on H100, prefill throughput improves by up to 102.6% (512+256 setting) and decode throughput by up to 46.6%. Improvements on consumer hardware (Apple M1) are more moderate but remain positive.

5. **Hypertokens carry semantic meaning**—observed hypertokens largely correspond to domain-specific semantic units, such as "torch," "Attention," and "MultiHead" in code; "mRNA" and "cellular" in biomedical text; and "Eiffel" and "Gustave" in French, indicating that the LZW merging strategy captures meaningful linguistic patterns in practice.

6. **Only approximately 25% of hypertokens are reused during generation**, suggesting substantial room for optimizing codebook management strategies (e.g., pruning unused entries, pre-population).

## Highlights & Insights

- **The online nature of LZW is the key innovation**: Unlike offline methods such as BPE that require global statistics, LZW allows a hypertoken created at step $t$ to be immediately available at step $t+1$, naturally matching the autoregressive generation process. This "create and use immediately" property enables dynamic vocabulary expansion to be seamlessly embedded into the standard LLM decoding loop without modifying the inference architecture.

- **The entropy invariance theorem provides theoretical grounding for practicality**: Proving that lossless compression does not fundamentally harm model performance establishes a solid theoretical foundation for the entire methodology—performance losses stem from optimization insufficiency rather than representational capacity limitations, implying that performance can be further recovered as training budget increases.

- **The extremely low training cost is a standout practical advantage**: A fine-tuning cost of 10 GPU-hours and 100M tokens is even lower than many domain-adaptive fine-tuning overheads, making zip2zip a genuinely plug-and-play inference acceleration technique. Combined with an open-source high-performance Rust LZW tokenizer and a HuggingFace Transformers-compatible drop-in model architecture, the method offers high engineering reproducibility and deployability.

- **The dual-compression design philosophy for both input and output**: The double "zip" in the name reflects the core insight—unlike prompt compression methods that can only compress the input, zip2zip simultaneously reduces the number of output tokens. Under low-batch-size workloads, output tokens are typically the primary bottleneck for inference latency, so reducing output token count yields far greater benefit than compressing only the input.

- **The analogy between hyper-embedding caching and KV cache is elegant**: The context-independence of hypertoken embeddings allows them to be incrementally updated analogously to the KV cache, computing only one new token embedding per step. This maintains $O(1)$ per-step additional overhead, avoiding computational explosion as the vocabulary grows dynamically.

## Limitations & Future Work

- **The maximum merge length $M=3$ limits the compression rate ceiling**: The current design allows a single hypertoken to encompass at most 3 base tokens, limiting compression potential in extreme cases. While hypertokens support recursive merging, the depth of recursive merging is limited in practice; larger $M$ may yield higher compression rates but would also require a more powerful hyper-encoder.

- **Severe performance degradation on mathematical computation tasks**: The catastrophic drop on GSM8K from 0.82 to 0.15 indicates that adaptive tokenization may fundamentally disrupt the model's numerical reasoning capability. Merging of digit tokens may alter the arithmetic operation patterns learned by the model, posing a significant limitation for applications requiring precise numerical computation.

- **Moderate perplexity increase reflects an optimization gap**: Although theory proves that lossless compression should not harm performance, practical training fails to fully recover the theoretically optimal model, suggesting that the optimization landscape in the compressed representation space may be more complex, requiring longer training or more refined optimization strategies.

- **Low codebook efficiency**: Only approximately 25% of created hypertokens are actually reused, wasting a large proportion of codebook entries. The current method lacks codebook pruning or selective retention strategies, limiting memory efficiency.

- **Limited gains on unstructured text**: In dialogue and web domains with high text diversity and fewer repetitive patterns, token efficiency improvements are only 15–20%, substantially lower than the 48–70% gains in code and math, indicating that the LZW dictionary-matching mechanism has limited utility when repetitive patterns are scarce.

- **Limited model coverage**: Validation is currently performed only on the Phi-3 series (4B and 14B); results on other model architectures such as Llama and Qwen, as well as larger-scale models, have not yet been verified.

## Related Work & Insights

**Domain-Adaptive Tokenizers.** Works by Zhao et al. (2024), Kim et al. (2024), Liu et al. (2023, 2024a), and others adapt tokenizers by adding new tokens for Chinese, Korean, or specific domains (mental health, legal) to the Llama tokenizer. However, these methods still produce fixed vocabularies that do not adapt dynamically at inference time.

**Input Compression Methods.** Prompt compression approaches such as Gist tokens (Mu et al., 2023), LLMLingua (Jiang et al., 2023), and In-context Autoencoders (Ge et al., 2024) reduce context length through lossy compression, enabling higher compression ratios, but can only compress input tokens and cannot compress output tokens. Lester et al. (2024) propose training LLMs on arithmetic-coded compressed text.

**Dynamic Embedding Transformers.** Several works explore dynamic embedding schemes to reduce Transformer computation costs: Hourglass (Nawrot et al., 2022) uses an hourglass-shaped token merging-and-expanding architecture; MegaByte (Yu et al., 2023) processes byte sequences with a multi-scale Transformer; BLT (Pagnoni et al., 2025) performs dynamic chunking at the byte level. Conceptually closest to zip2zip is Dynamic Vocab (Liu et al., 2024b), which also dynamically expands the vocabulary during generation, but differs significantly from zip2zip in vocabulary construction algorithm and training procedure. ZeTT (Minixhofer et al., 2024) and Dynamic Tokenization (Feher et al., 2025) also explore dynamic tokenization directions but adopt different architectures and chunking strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Deeply integrates the classical LZW compression algorithm with LLM inference, achieving for the first time truly online inference-time adaptive tokenization—conceptually very elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multi-model, multi-domain, multi-dimensional evaluation (token efficiency, perplexity, downstream tasks, inference throughput), though the severe degradation on GSM8K warrants deeper analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Concepts are clearly articulated, figures are intuitive and accessible, theory and experiments are tightly integrated, and the visualization of the LZW workflow is particularly outstanding.
- **Value**: ⭐⭐⭐⭐⭐ Extremely high practical value—existing models can be upgraded in 10 GPU-hours, the complete toolchain is open-sourced, and the method constitutes a plug-and-play inference acceleration solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Hyper-Scaling with KV Cache Compression](inference-time_hyper-scaling_with_kv_cache_compression.md)
- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)

</div>

<!-- RELATED:END -->
