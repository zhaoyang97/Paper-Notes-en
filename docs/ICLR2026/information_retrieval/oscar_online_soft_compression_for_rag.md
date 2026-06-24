---
title: >-
  [Paper Note] OSCAR: Online Soft Compression for RAG
description: >-
  [ICLR 2026][Information Retrieval & RAG][RAG] OSCAR utilizes a lightweight compressor to perform online, query-aware compression of each retrieved document into a few embedding tokens. This achieves 2–5× end-to-end inference acceleration on 1B–24B LLMs with negligible performance degradation.
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Soft Compression"
  - "Query-aware Compression"
  - "Sequence-level Distillation"
  - "Reranking"
date: 2026-05-08
content_hash: 5e41696956368e3c
---

# OSCAR: Online Soft Compression for RAG

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ideKAUWvFE](https://openreview.net/forum?id=ideKAUWvFE)  
**Code**: https://github.com/naver/pisco (Model weights: naver/oscar)  
**Area**: Information Retrieval / RAG Context Compression  
**Keywords**: RAG, Soft Compression, Query-aware Compression, Sequence-level Distillation, Reranking

## TL;DR
OSCAR utilizes a lightweight compressor to perform online, query-aware compression of each retrieved document into a few embedding tokens. This achieves 2–5× end-to-end inference acceleration on 1B–24B LLMs with negligible performance degradation.

## Background & Motivation
**Background**: RAG enhances LLMs by prepending retrieved documents to the prompt. However, long context significantly increases inference costs. Current approaches for compressing retrieved content follow two main paths: **hard compression**, which performs summarization or pruning at the text level (e.g., Provence, RECOMP), and **soft compression**, which maps text into continuous vectors or KV caches (e.g., COCOM, PISCO, xRAG).

**Limitations of Prior Work**: Hard compression is online and query-aware, allowing for simultaneous retrieval and pruning or reranking. However, because the output remains text, the compression rate is limited ($\approx \times 2$), offering minimal savings. Soft compression achieves high compression rates ($\approx \times 16$), but often at the cost of significant performance drops. More importantly, **existing soft compression methods are entirely offline**: they require an LLM of the same size as the generator for the compression forward pass, meaning the entire corpus must be pre-compressed and stored. This prevents online compression for open-web or large-scale dynamic corpora.

**Key Challenge**: It is difficult to simultaneously achieve online processing, query-awareness, and high compression rates. Soft compression is typically offline because it relies on the compressor and generator being isomorphic and iso-sized to align hidden spaces. If the compressor is as large as the generator, the overhead of online compression consumes all the time saved during generation, nullifying any speedup. Furthermore, existing soft compression methods **ignore the query during compression**, attempting to force all document information into a fixed set of vectors.

**Goal**: Develop an online soft compressor that is sufficiently fast and query-aware, retaining the high compression rates of soft methods while regaining the online and query-aware advantages of hard methods.

**Key Insight**: The authors make two observations: first, introducing the query during compression allows the model to retain only information relevant to the current question, enabling more concise content representation within a fixed number of embeddings. Second, the compressor **does not need to be isomorphic or iso-sized to the generator**. As long as hidden states can be aligned to the generator's input space, a much smaller backbone (a small model or the first few layers of the generator) can be used to accelerate compression.

**Core Idea**: Use a lightweight compressor to map (query, document) pairs online to $l$ memory embeddings. Sequence-level distillation is then used to ensure the entire compression pipeline approximates the output of an uncompressed RAG system—achieving "query-aware online soft compression."

## Method

### Overall Architecture
OSCAR's inference pipeline is straightforward: Retrieval → Compression → Generation. After retrieving top-$k$ documents, a **compressor LLM** maps each (query + document) pair into a few embedding tokens (defaulting to 8 tokens for a 128-token document, a $\times 16$ compression rate). These compressed vectors, along with the original query, are fed into the **generator LLM** to produce the answer. Since each document is reduced from hundreds of tokens to a few embeddings, attention computation during generation is significantly accelerated.

For training, **sequence-level distillation** is employed. A standard uncompressed RAG pipeline (Teacher = Mistral-7B) is run first, and its answers are saved as labels. The OSCAR "compressor + generator" pipeline is then trained end-to-end to fit these teacher outputs, with gradients propagated back to both models. This pipeline requires no human-annotated answers. Additionally, the compressor can learn a `[RR]` reranking token, merging compression and reranking into a single forward pass, making compression nearly "free" in RAG pipelines that already include reranking.

```mermaid
flowchart TD
    A["Query + Top-k Retrieved Documents"] --> B["Lightweight Compressor Backbone<br/>Small model or generator first N layers"]
    B --> C["Query-aware Online Compression<br/>(q, d) → l [MEM] embeddings"]
    C -->|Shared forward yields [RR] token| D["Reranking<br/>Predict relevance score r_i"]
    C --> E["Generator LLM<br/>Reads query + compressed vectors"]
    E -.Sequence-level distillation.-> F["Teacher Mistral-7B<br/>Uncompressed RAG Output"]
```

### Key Designs

**1. Query-aware memory-token compression: Fitting query-relevant info into fixed vectors**

This is the core distinction between OSCAR and prior soft compression methods. Previous methods (COCOM, PISCO, xRAG) ignore the query during document compression, forcing the entire document into fixed vectors. OSCAR feeds the query $q$, the $i$-th document $d_i$, and a set of learnable memory tokens $[\text{MEM}_j]_{j=1\ldots l}$ together into the compressor $C$. It uses the last-layer hidden states at these memory token positions as the compressed representation $c_i = (c_i^1,\ldots,c_i^l) = C(q, d_i)$. These `[MEM]` tokens act like BERT's `[CLS]`—task-specific tokens that signal the model to "store information here." Because the query is known during compression, the fixed $l$ vectors can encode only the content relevant to the question, maintaining accuracy at high compression rates. Ablations show that removing query dependency (query-independent) leads to a 4-point drop at $\times 16$ and a 6-point drop at $\times 128$, proving the query optimizes the representation. Needle-in-a-haystack analysis also confirms that compressed embeddings have the highest cosine similarity with text near the "needle."

**2. Lightweight Compressor Backbone: Moving soft compression from offline to online**

The primary reason soft compression has remained offline is the requirement for the compressor and generator to be isomorphic and iso-sized. While this facilitates hidden space alignment, running such a large forward pass for compression offsets the speed gained in generation. OSCAR proposes two lightweight backbones:
- **OSCAR-N-Layers**: Uses the first $N$ layers (without the head) of the pre-trained backbone as the compressor. Being derived from the generator, it **requires no additional pre-training** for hidden space alignment. Efficiency is controlled by $N$, typically 1/4–1/3 of the total layers (experiments used $N=5, 8, 10$).
- **OSCAR-llama**: Uses a smaller independent LLM (default: Llama-3.2-1B) as the compressor. A two-layer MLP with ReLU is added to the last layer to map representations to the generator's embedding space. Since the compressor and generator are heterogeneous, this mapping requires pre-training on auto-encoding or text continuation tasks before QA fine-tuning.
Both approaches keep compression costs much lower than generation costs, enabling overall online acceleration. OSCAR-llama is typically faster and more performant in experiments.

**3. Sequence-level Distillation: Approximating uncompressed RAG without labels**

OSCAR aims to replicate the behavior of the uncompressed pipeline. It does not rely on ground-truth answers but uses outputs $a_1,\ldots,a_r$ generated by a teacher LLM (Mistral-7B) under standard uncompressed RAG. The training objective jointly optimizes compressor $C$ and generator $G$:

$$\mathcal{L}(C,G) = -\sum_{i=1}^{r} \log G(a_i \mid q, c_1,\ldots,c_k, a_{<i}),\quad c_i = C(q,d_i)$$

Where $k$ is the number of documents. Gradients flow to both the generator and compressor. In practice, retrieval results and teacher answers only need to be pre-calculated once, after which OSCAR training reduces to standard supervised fine-tuning (SFT), requiring only 1–5 GPU-days for 1B–24B backbones. Ablations found that joint training is mandatory; freezing the generator results in a 6-point drop.

**4. Integrated Reranking: Nearly-free compression in RAG pipelines**

Borrowing insights from Provence, query-aware online compression is functionally similar to cross-encoder reranking, as both involve contextual encoding of a document relative to a query. Since robust RAG pipelines already include a reranking step, OSCAR produces compressed representations and reranking scores in a **single forward pass**. A `[RR]` token is added to the compressor prompt, and its hidden state is mapped to a relevance score $r_i$ via an MLP. A pointwise distillation loss is added to Equation (1): $\lambda \sum_{i=1}^{k}(r_i - r_i')^2$, where $r_i'$ is the reference reranker score (default $\lambda=0.05$). Experiments show this simple distillation is effective: "standalone" (ground truth reranking) vs. "e2e" (OSCAR's own reranking) performance is nearly identical.

### Loss & Training
- **Primary Loss**: Sequence-level distillation NLL (Eq. 1). Teacher: Mistral-7B. Gradients are backpropagated to both $C$ and $G$.
- **Reranking Branch (Optional)**: Pointwise distillation MSE, weight $\lambda=0.05$.
- **Training Config**: Generator uses LoRA, compressor uses full fine-tuning (which outperforms LoRA). Trained with $k=5$ documents, generalizes to 10–50 documents at inference. OSCAR-llama requires pre-training on auto-encoding and text continuation.

## Key Experimental Results

### Main Results
Comparison of compression methods on Mistral-7B (Average is the mean accuracy across 6 QA datasets, Total is compression+generation T-FLOPs, percentages denote E2E acceleration):

| Method | Average Accuracy | Total T-FLOPs | E2E Acceleration |
|------|------|------|------|
| No compression | 0.68 | 20.33 | 1× |
| RECOMP (Hard) | 0.67 | 8.13 | 2.5× |
| Provence (Hard) | 0.68 | 9.43 | 2.2× |
| PISCO (Offline Soft) | 0.65 | 3.49 | 5.8× (Offline) |
| OSCAR-llama | 0.68 | 6.15 | 3.3× |
| OSCAR-8-Layers | 0.68 | 8.36 | 2.4× |

OSCAR becomes more efficient as the model size increases: On Mistral-24B, OSCAR-llama reduces FLOPs from 64.29 to 13.37 (**4.8×**) while the Average accuracy actually increases to 0.69. On Llama-1B, OSCAR-5-Layers provides 2.1× acceleration and improves Average accuracy from 0.55 to 0.60. GPT-4o pairwise evaluations show OSCAR-llama remains competitive with uncompressed, Provence, and PISCO results while being faster.

### Ablation Study
Based on Mistral-7B, reporting mean LLM-eval scores across 5 QA tasks ($\Delta$ relative to OSCAR-llama $\times 16$ at 0.77):

| Config | Avg | $\Delta$ | Description |
|------|------|------|------|
| OSCAR-llama $\times 16$ | 0.77 | — | Full model |
| query-independent $\times 16$ | 0.73 | -0.04 | Compressed without query |
| No compressor pre-training | 0.70 | -0.07 | OSCAR-llama without alignment |
| Frozen generator | 0.71 | -0.06 | Generator does not participate in training |
| OSCAR-llama $\times 128$ | 0.75 | -0.02 | Pushed to $\times 128$ compression |
| query-independent $\times 128$ | 0.71 | -0.06 | Query dependency more critical at higher rates |
| Modern-bert-large compressor | 0.76 | -0.01 | Low-latency alternative compressor |

### Key Findings
- **Query dependency is central**: Removing it causes a 4-point drop at $\times 16$ and a 6-point drop at $\times 128$; higher compression rates necessitate query awareness.
- **Joint training + Pre-training are essential**: Freezing the generator drops performance by 6 points; skipping compressor pre-training for OSCAR-llama drops it by 7 points (as heterogeneous compressors must align hidden spaces).
- **Compression can scale to $\times 128$** with only a 2-point drop, demonstrating the potential of soft compression. Llama-1B is the best compressor choice, while Modern-bert-large is a good low-latency alternative.
- **Robustness**: When using weak retrieval (BM25 without reranking), OSCAR's drop relative to Mistral-7B is consistent with the uncompressed version, proving it can handle noise. With 50 documents ($\approx 7k$ context tokens), it is 5× faster than uncompressed RAG.

## Highlights & Insights
- **Query-aware compression is a paradigm shift**: It transforms soft compression from "offline library pre-compression" to "online on-demand compression," allowing it to handle open-web and dynamic corpora.
- **Decoupling compressor/generator isomorphism**: By using either the first $N$ layers or a smaller LLM, OSCAR lowers the latency barrier that previously made online soft compression impractical.
- **Unified Forward Pass**: Merging compression and reranking is a clever engineering insight. Since strong RAG pipelines require reranking anyway, the cost of compression is essentially absorbed into the reranker, making it "free."
- **Teacher Distillation**: The sequence-level distillation paradigm (fitting uncompressed output without labels) makes migrating to new backbones cost-effective (1-5 GPU-days).

## Limitations & Future Work
- **Backbone-specific**: OSCAR must be retrained for each generator LLM, unlike hard compression which is LLM-agnostic.
- **Pre-training requirements**: OSCAR-llama requires auto-encoding and text-continuation pre-training for heterogeneous alignment, which is more complex than the N-Layers approach.
- **Task focus**: Evaluation is centered on QA tasks; performance on long-form generation or complex multi-hop reasoning remains to be verified.
- **Orthogonality**: While potentially complementary to KV-cache compression methods, this was not explored in the current paper.

## Related Work & Insights
- **vs. Provence / RECOMP (Hard)**: These are LLM-agnostic and online but limited by text-level compression ($\approx \times 2$). OSCAR compresses at the embedding layer, achieving higher rates and faster inference at comparable accuracy, though it requires backbone-specific training.
- **vs. PISCO / COCOM (Offline Soft)**: These are query-independent and require compressors of the same size as the generator. OSCAR introduces query dependency and lightweight compressors to enable online processing with less accuracy loss.
- **vs. FiD-light / DODO**: FiD-light use encoder-decoders but with lower compression rates. DODO focuses on long context and is weaker for multi-document RAG. OSCAR simultaneously solves for high rate, online speed, and query relevance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First online soft compression RAG method; effectively balances query awareness and lightweight heterogeneous compressors.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four backbones (1B–24B), 6 QA sets, and includes FLOPs, LLM-eval, GPT-4o comparisons, and multidimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and methodology; high information density in tables/figures.
- Value: ⭐⭐⭐⭐⭐ 2–5× acceleration with minimal performance drop; integrated free reranking makes it highly practical for production RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document RAG](smartchunk_retrieval_query-aware_chunk_compression_with_planning_for_efficient_d.md)
- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](../../ICML2026/information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](../../ACL2026/information_retrieval/codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)

</div>

<!-- RELATED:END -->
