---
title: >-
  [Paper Note] Very Efficient Listwise Multimodal Reranking for Long Documents
description: >-
  [ICML 2026][Information Retrieval & RAG][listwise reranking] ZipRerank simultaneously eliminates the two primary bottlenecks of VLM listwise reranking—"excessive visual token sequence length" and "autoregressive decoding…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "listwise reranking"
  - "VLM"
  - "visual token pruning"
  - "single-step decoding"
  - "teacher distillation"
date: 2026-05-08
content_hash: bbf2d56e59971420
---

# Very Efficient Listwise Multimodal Reranking for Long Documents

**Conference**: ICML 2026  
**arXiv**: [2605.11864](https://arxiv.org/abs/2605.11864)  
**Code**: <https://github.com/dukesun99/ZipRerank>  
**Area**: Information Retrieval / Multimodal RAG / Efficient Inference  
**Keywords**: listwise reranking, VLM, visual token pruning, single-step decoding, teacher distillation

## TL;DR
ZipRerank simultaneously eliminates the two primary bottlenecks of VLM listwise reranking—"excessive visual token sequence length" and "autoregressive decoding for token-by-token output"—by utilizing query-aware token pruning and single-logit ranking. This approach reduces LLM inference latency by an order of magnitude on MMDocIR while matching or exceeding the current SOTA, MM-R5.

## Background & Motivation
**Background**: Vision-centric long document retrieval (M-RAG, Document VQA) typically adopts a "retrieval + reranking" two-stage architecture. The first stage (DSE, ColPali, etc.) performs large-scale similarity retrieval, while the second-stage reranker further refines the top-$k$ candidates. Listwise rerankers process all candidates simultaneously and are theoretically more efficient than pointwise methods; the latest MM-R5 achieved SOTA status on MMDocIR by introducing CoT reasoning.

**Limitations of Prior Work**: The practicality of methods like MM-R5 is hindered by two latency bottlenecks: (i) Prefill length explosion: each candidate page is a high-resolution image contributing hundreds to thousands of visual tokens, with sequences easily exceeding 10,000 for $k=10$ candidates; (ii) Autoregressive decoding: the output length of CoT + ranking grows linearly with $k$, and KV cache cannot mitigate the sequential dependency. Consequently, MM-R5 takes 3.82s for a single reranking, making it difficult to deploy in real-time M-RAG.

**Key Challenge**: The tension between "Accuracy (requiring long context + reasoning chains)" and "Latency (requiring sequence compression + single-step output)." Removing visual tokens risks losing critical evidence, while removing CoT risks losing ranking capability.

**Goal**: (i) Decompose latency into prefill and decode components for separate optimization; (ii) Design a single-step listwise scoring mechanism to eliminate autoregression; (iii) Introduce query-aware visual pruning to mitigate prefill explosion; (iv) Enable robust listwise behavior even under weak supervision (where VQA provides only one positive example).

**Key Insight**: The authors noted an interesting two-stage strategy: one can first learn "general listwise behavior" on large-scale text-only reranking data, then perform soft supervision using a strong VLM teacher on multimodal data, decoupling "learning to rank" from "learning multimodality." Simultaneously, they observed that generating a full ranking sequence is unnecessary at inference; the first-step logits of candidate identifiers (A, B, C...) are sufficient for ordering.

**Core Idea**: A two-stage training approach ("text-based listwise pretraining + VLM teacher soft-supervised multimodal fine-tuning") combined with an inference strategy of "query-aware visual pruning + single-logit scoring," compressing listwise multimodal reranking into a single forward pass.

## Method
ZipRerank employs a synergistic design for both training and inference optimization.

### Overall Architecture
Input: text query $\bm{q}$ + image set of $k$ retrieved candidate pages $\bm{I}=(I_1,\dots,I_k)$. Output: reranked list $\hat{\bm{I}}=(I_{\pi(1)},\dots,I_{\pi(k)})$. The model is a VLM where each candidate is assigned a unique single-token identifier (A, B, ...). The prompt template instructs the model to "output a string of these identifiers in descending order of relevance."

Training consists of two stages: Stage 1 performs supervised pretraining on text reranking corpora (passages rendered as page-like images) to establish general listwise capabilities. Stage 2 uses soft labels from a GPT-5 teacher on VQA-style multimodal data to adapt to fine-grained alignment in document images. During inference, candidates are fed into ZipRerank. A "query prefix" pass first obtains query hidden states for visual token pruning, followed by a single forward pass over the remaining tokens to extract first-step identifier logits for ranking. In the latency model $F(n,u)\approx L(c_{\text{att}}dn^2+c_{\text{ffn}}d^2n)+uLdn\cdot c_{\text{dec}}$, the former grows quadratically with $n$ (visual tokens) and the latter linearly with $u$ (generation length). ZipRerank reduces both $n$ (via pruning) and $u$ (setting it to 1).

### Key Designs

1.  **Two-Stage Training + Soft-Ranking Loss**:
    *   **Function**: Learns robust listwise capabilities under weak multimodal supervision, decoupling "general ranking" from "visual alignment."
    *   **Mechanism**: Stage 1 uses RankNet $\mathcal{L}_{\text{ranknet}}=\sum_{r_i<r_j}w_{i,j}\log(1+\exp(s_j-s_i))$ ($w_{i,j}=1/(r_i+r_j)$) + language modeling loss $\mathcal{L}_{\text{LM}}$ for supervised pretraining. Stage 2 leverages a GPT-5 teacher for full candidate rankings; considering potential teacher noise, a soft-ranking loss $\mathcal{L}_{\text{softrank}}=-\sum_i q_i\log p_i$ is introduced, where $q_{\pi(k)}=\gamma^k/\sum_{\ell=0}^{m-1}\gamma^\ell$ is a geometric decay target distribution based on Rank-Biased Precision (RBP). $\gamma\in(0,1)$ controls tolerance for lower-ranked items. Both stages directly supervise first-step identifier logits, aligning with the inference protocol.
    *   **Design Motivation**: Multimodal document retrieval data typically contains only one positive sample (VQA annotation). Direct hard listwise supervision can overfit to teacher noise. The geometric decay distribution anchors the true positive while assigning descending weights to plausible candidates, consistent with the RBP "top-to-bottom browsing" assumption, ensuring robustness.

2.  **Query-Image Early Interaction (Visual Token Pruning)**:
    *   **Function**: Filters out 50-80% of irrelevant patches using query semantics before feeding tokens into the LLM, reducing prefill complexity from $O(n^2)$ to $O((\rho n)^2)$.
    *   **Mechanism**: The LLM first processes the "prompt prefix until the first image token" to extract $N_q$ query hidden states $\bm{H}_q\in\mathbb{R}^{N_q\times D}$. For each candidate image $i$ with pre-computed visual tokens $\bm{V}_i$, max-sim importance is calculated as $a_{i,j}=\max_{1\le t\le N_q}\cos(\bm{h}_t,\bm{v}_{i,j})$. The top-$\mathrm{round}(\rho N_i)$ tokens are kept according to the ratio $\rho$, yielding pruned $\tilde{\bm{V}}_i$. Pruning preserves original RoPE positional encodings and reuses prefix computations via KV cache, making the hidden state extraction step nearly zero-overhead. Theoretical analysis (Appendix) shows that when the culled tokens represent a small tail mass $\varepsilon$ in original attention, the output change is bounded by $O(\varepsilon)$.
    *   **Design Motivation**: Many patches in long document pages are blank, decorative, or irrelevant. Processing these in attention consumes $O(n^2)$ compute for minimal gain. Max-sim is more effective than mean-pooling for preserving sparse critical patches correlated with specific query tokens (e.g., specific numbers or key phrases).

3.  **Single-Logit Listwise Scoring**:
    *   **Function**: Compresses the autoregressive decoding of the rank sequence into "one forward pass + argsort," eliminating decoding latency.
    *   **Mechanism**: During training, the model treats the first token of the target ranking as the critical score point, and RankNet/Soft-Ranking directly supervise the logits $s_i$ of each candidate identifier token $t_i$. At inference, one forward pass yields the first-step logit vector $\bm{z}\in\mathbb{R}^{|\mathcal{V}|}$, and ordering is determined by $\pi=\mathrm{argsort}_{\downarrow}(z_{t_1},\dots,z_{t_k})$.
    *   **Design Motivation**: Autoregressive rerankers must re-attend to long contexts at each step, and KV caches do not solve the linear step dependency. Listwise ranking is essentially about relative preferences; a single logit can carry information about a candidate's relative position among all candidates, which is sufficient when aligned with training objectives. This approach, proven effective for text by Gangi Reddy 2024 et al., is extended here to VLM listwise reranking.

### Loss & Training
Stage 1: $\mathcal{L}_{\text{stage1}}=\mathcal{L}_{\text{LM}}+\lambda_1\mathcal{L}_{\text{ranknet}}$; Stage 2: $\mathcal{L}_{\text{stage2}}=\mathcal{L}_{\text{LM}}+\lambda_2\mathcal{L}_{\text{softrank}}$. At inference, $u=1$ and $n$ is controlled by $\rho$, allowing for a smooth trade-off between accuracy and latency.

## Key Experimental Results

### Main Results
Evaluated on 9 domains of page-level retrieval in MMDocIR, using DSE-wiki-ss as the first-stage retriever to rerank top candidates. Results include Recall@1/3/5 and LLM latency (seconds per query):

| Method | Macro R@3 | Micro R@3 | Latency (s) |
| :--- | :---: | :---: | :---: |
| DSE-wiki-ss (Retriever) | 69.5 | 70.2 | – |
| UniME (Listwise) | 70.9 | 71.4 | 0.24 |
| LamRA (Listwise) | 77.6 | 77.8 | 0.53 |
| MM-R5 (CoT) | 79.1 | 79.0 | 3.82 |
| GPT-5-mini | 88.0 | 88.3 | 23.38 |
| **ZipRerank** | **84.8** | **84.5** | **0.36** |
| ZipRerank-50% (Aggressive Pruning) | 83.4 | 83.4 | 0.30 |

ZipRerank outperforms MM-R5 across all metrics while reducing latency from 3.82s to 0.36s (approx. $10.6\times$ speedup). It also narrows the R@3 gap with GPT-5-mini from ~9 points to ~3.2 points with $65\times$ lower latency.

### Ablation Study

| Configuration | Key Metrics | Note |
| :--- | :--- | :--- |
| Full ZipRerank | Optimal | 2-stage training + soft-ranking + pruning + single-logit |
| w/o Stage 1 Text Pretraining | Significant drop | Lacks general listwise ability; difficult to learn from weak multimodal supervision |
| w/o Soft-Ranking (using hard label)| Drop | Teacher noise propagates, causing overfitting to sub-optimal ranks |
| w/o Query-Aware Pruning (full tokens)| Latency up ~3-5× | Only marginal gain in accuracy; pruned tokens are redundant |
| w/o Single-Token Decoding (autoregressive)| Latency doubled | Little change in accuracy; single-logit scoring is highly efficient |
| Varying keep ratio $\rho\in\{0.3,0.5,0.7,1.0\}$| Smooth trade-off | $\rho=0.5$ is Pareto optimal on MMDocIR |

### Key Findings
- Single-logit scoring has minimal impact on accuracy but offers massive latency benefits, indicating that listwise ranking information can be condensed into first-step logits. This aligns with the intuition that reasoning chains provide optimization directions, while the final rank is a 1-token decision.
- Query-aware token pruning is a major efficiency driver; at $\rho=0.5$, prefill complexity is reduced by nearly three-quarters with almost no performance loss. Visible drops only occur at $\rho=0.3$, suggesting 50-70% of visual tokens in MMDocIR pages are redundant.
- Soft-ranking loss gains are most evident in small datasets or noisy teacher domains, proving that listwise supervision is better suited as soft targets for knowledge distillation.
- ZipRerank consistently outperforms MM-R5 by ~5 points, except in the News domain where candidates are highly similar (different pages on the same topic), suggesting a need for domain-adaptive pruning thresholds.

## Highlights & Insights
- Using the latency decomposition formula $F(n,u)\approx L(c_{\text{att}}dn^2+c_{\text{ffn}}d^2n)+uLdn\cdot c_{\text{dec}}$ in Section 3 as motivation provides a rare, clear mathematical explanation for slowness. The design then targets each term: $n$ via pruning and $u$ via single-logit output, turning the accuracy-latency trade-off into two tunable hyperparameters ($\rho$ and target length).
- The "text-first, multimodal-second" two-stage training combined with "teacher soft labels + geometric decay" provides an elegant paradigm for learning ranking behavior under multimodal data scarcity.
- Max-sim visual pruning can be interpreted as a "tight upper bound for smooth attention pooling." The author provides an $O(\varepsilon)$ bound for attention output perturbation, a rare piece of theoretical sanity checking in efficient VLM literature.

## Limitations & Future Work
- Single-logit scoring is limited by token vocabulary: when $k$ is large (candidates > alphabet size), the identifier set must be expanded, potentially conflicting with the tokenizer. $k>26$ was not discussed.
- Query-aware pruning risks losing vital information in "detail-dense" domains (e.g., academic figures, dense PDFs). Verification focused on general MMDocIR without domain-adaptive $\rho$ strategies.
- Stage 2 depends on GPT-5 teacher soft labels, presenting barriers in distillation cost and teacher availability. The performance "ceiling" is locked by the teacher's ranking ability.
- Lack of end-to-end full-stack comparisons with recent hybrid retrievers (e.g., multi-vector ColPali + token reduction); only reranking latency was measured.
- Evaluated only on MMDocIR; not verified for massive documents (hundreds of pages) or cross-document M-RAG scenarios.

## Related Work & Insights
- **vs MM-R5 (Xu et al. 2025)**: MM-R5 uses CoT for SOTA results, but dual bottlenecks (autoregression + context length) lead to 3.82s latency. ZipRerank proves that reasoning benefits can be distilled into first-step logits via soft supervision, removing the need for runtime CoT.
- **vs FIRST / RankZephyr (Gangi Reddy 2024)**: Single-logit scoring was verified for pure text listwise ranking; ZipRerank extends this to multimodality and adds query-aware pruning.
- **vs Light-ColPali / token reduction**: These optimize first-stage multi-vector retrieval. ZipRerank is complementary, squashing second-stage reranking latency.
- **Insights**: (i) The "latency decomposition $\to$ targeted reduction" methodology is transferable to any VLM inference optimization; (ii) "Teacher soft labels + geometric decay" is a universal template for distilling ranking under weak supervision; (iii) Query-aware pruning could be a standard pre-processing step for any query-conditioned VLM task.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While single-logit scoring and max-sim pruning are not entirely new, their integration into multimodal listwise reranking combined with two-stage distillation constitutes a clear systemic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 9 domains in MMDocIR, 3 Recall metrics, multiple baselines, keep ratio sweeps, and training ablation. Methods are well-accounted for, though limited to a single benchmark.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is driven by a latency decomposition formula and followed by theoretical sanity checks (attention perturbation bounds). Very cohesive structure.
- **Value**: ⭐⭐⭐⭐⭐ Reduces latency by an order of magnitude for MMDocIR/M-RAG without sacrificial accuracy. Directly applicable to production RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking](../../NeurIPS2025/information_retrieval/acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)
- [\[ICML 2026\] ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)
- [\[ICML 2026\] LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](lazyattention_efficient_retrieval-augmented_generation_with_deferred_positional_.md)
- [\[ICML 2026\] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs](pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->
