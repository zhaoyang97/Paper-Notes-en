---
title: >-
  [Paper Note] Very Efficient Listwise Multimodal Reranking for Long Documents
description: >-
  [ICML 2026][Information Retrieval & RAG][listwise reranking] ZipRerank simultaneously eliminates the two main bottlenecks of VLM-based listwise reranking—"overly long visual token sequences" and "autoregressive decoding…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "listwise reranking"
  - "VLM"
  - "visual token pruning"
  - "single-step decoding"
  - "teacher distillation"
date: 2026-05-08
content_hash: 87a58e9b2c72f923
---

# Very Efficient Listwise Multimodal Reranking for Long Documents

**Conference**: ICML 2026  
**arXiv**: [2605.11864](https://arxiv.org/abs/2605.11864)  
**Code**: <https://github.com/dukesun99/ZipRerank>  
**Area**: Information Retrieval / Multimodal RAG / Efficient Inference  
**Keywords**: listwise reranking, VLM, visual token pruning, single-step decoding, teacher distillation

## TL;DR
ZipRerank simultaneously eliminates the two main bottlenecks of VLM-based listwise reranking—"overly long visual token sequences" and "autoregressive decoding with per-token ranking output"—by employing query-aware token pruning and single-logit sorting. On MMDocIR, it reduces LLM inference latency by an order of magnitude while matching or surpassing the current SOTA MM-R5.

## Background & Motivation
**Background**: Vision-centric long-document retrieval (M-RAG, document VQA) typically adopts a two-stage "retrieval + reranking" architecture. The first stage (e.g., DSE, ColPali) performs large-scale similarity search; the second-stage reranker further refines the top-$k$ candidate pages. Listwise rerankers process all candidates at once and are theoretically more efficient than pointwise approaches. The latest MM-R5 achieves SOTA on MMDocIR by introducing CoT reasoning.

**Limitations of Prior Work**: Methods like MM-R5 are hindered by two latency bottlenecks—(i) Exploding prefill length: each candidate page is a high-resolution image, contributing hundreds to thousands of visual tokens per page; with $k=10$ candidates, input sequences easily exceed ten thousand tokens. (ii) Autoregressive decoding: CoT + ranking output length grows linearly with $k$, and even KV cache cannot mitigate the sequential dependency. As a result, MM-R5 requires 3.82s per reranking, making real-time M-RAG deployment impractical.

**Key Challenge**: The tension between "accuracy (requiring long context + reasoning chains)" and "latency (must shorten sequence + produce results in a single step)." Pruning visual tokens risks losing key evidence; removing CoT may harm ranking ability.

**Goal**: (i) Use latency decomposition to split latency into prefill and decode components and optimize each; (ii) design a single-step listwise scoring mechanism to eliminate autoregression; (iii) introduce query-aware visual pruning to address prefill explosion; (iv) enable robust listwise behavior even under weak supervision (VQA with only one positive example).

**Key Insight**: The authors observe a promising two-stage strategy—first learn "general listwise behavior" on large-scale pure-text reranking data, then use a strong VLM teacher for soft supervision in the multimodal setting, decoupling "learning to rank" from "learning multimodality." They also note that at inference, generating the full ranking sequence is unnecessary; only the first-step logit for candidate identifiers (A, B, C, ...) suffices for ordering.

**Core Idea**: Training uses "two stages (pure-text listwise pretraining + VLM teacher soft-supervised multimodal finetuning)"; inference uses "query-aware visual pruning + single-logit scoring," compressing listwise multimodal reranking into a single forward pass.

## Method

ZipRerank coordinates optimizations on both the training and inference sides.

### Overall Architecture
Input: text query $\bm{q}$ + $k$ candidate page images $\bm{I}=(I_1,\dots,I_k)$ from the first-stage retriever. Output: reranked list $\hat{\bm{I}}=(I_{\pi(1)},\dots,I_{\pi(k)})$. The model is a VLM, assigning each candidate a unique single-token identifier (A, B, ...), and the prompt template instructs the model to "output a string of these identifiers in descending order."

Training is two-stage: Stage 1 performs supervised pretraining on pure-text reranking corpora (passages rendered as page-like images) to establish general listwise ability; Stage 2 uses GPT-5 as a teacher to generate listwise soft labels on VQA-style multimodal data, adapting to fine-grained document-image alignment. At inference, the retrieved candidates are fed into ZipRerank: first, a "query prefix" is run to obtain query hidden states for visual token pruning; then, the remaining tokens are processed in a single forward pass to extract the first-step identifier logit for ordering. The latency model $F(n,u)\approx L(c_{\text{att}}dn^2+c_{\text{ffn}}d^2n)+uLdn\cdot c_{\text{dec}}$ shows the first term grows quadratically with $n$ (number of visual tokens), and the second grows linearly with $u$ (generation length); ZipRerank reduces both $n$ (via pruning) and $u$ (set to 1).

### Key Designs

1. **Two-Stage Training + Soft Ranking Loss**:

    - **Function**: Learns robust listwise ability under weak multimodal supervision, decoupling "general ranking" from "visual alignment."
    - **Mechanism**: Stage 1 uses RankNet $\mathcal{L}_{\text{ranknet}}=\sum_{r_i<r_j}w_{i,j}\log(1+\exp(s_j-s_i))$ ($w_{i,j}=1/(r_i+r_j)$) plus language modeling loss $\mathcal{L}_{\text{LM}}$ for fully supervised pretraining; Stage 2 uses GPT-5 as teacher to generate full candidate rankings, but since the teacher may be noisy, introduces a soft-ranking loss $\mathcal{L}_{\text{softrank}}=-\sum_i q_i\log p_i$, where $q_{\pi(k)}=\gamma^k/\sum_{\ell=0}^{m-1}\gamma^\ell$ is a geometrically decaying target distribution based on Rank-Biased Precision, with $\gamma\in(0,1)$ controlling tolerance for lower ranks. Both stages directly supervise the first-step identifier logit, aligning with the inference scoring protocol.
    - **Design Motivation**: Multimodal document retrieval data naturally has only one positive example (VQA annotation); hard listwise supervision would overfit teacher noise. The geometric decay target anchors the true positive while assigning gradually decreasing weights to plausible candidates, consistent with the RBP "users browse from top down" assumption and robust to noise.

2. **Query-Image Early Interaction (Visual Token Pruning)**:

    - **Function**: Screens out 50–80% of irrelevant patches using query semantics before feeding image tokens, reducing prefill complexity from $O(n^2)$ to $O((\rho n)^2)$.
    - **Mechanism**: First, the LLM runs the "prompt prefix up to the first image token" and extracts $N_q$ query hidden states $\bm{H}_q\in\mathbb{R}^{N_q\times D}$; for each candidate image $i$ with precomputed visual tokens $\bm{V}_i$, compute max-sim importance $a_{i,j}=\max_{1\le t\le N_q}\cos(\bm{h}_t,\bm{v}_{i,j})$; select the top-$\mathrm{round}(\rho N_i)$ by keep ratio $\rho$, yielding pruned $\tilde{\bm{V}}_i$. Original RoPE positional encoding is preserved after pruning, and the prefix computation is reused via KV cache, making "extracting query hidden states" nearly cost-free. Theoretical analysis (appendix) shows that when the pruned tokens account for a tail mass $\varepsilon$ in the original attention, the attention output perturbation is bounded by $O(\varepsilon)$.
    - **Design Motivation**: Long document pages contain many patches that are blank, decorative, or unrelated to the query (e.g., charts), which consume $O(n^2)$ compute in attention but contribute little information. Max-sim, compared to mean-pooling, better preserves the few critical patches highly relevant to at least one query token (e.g., specific numbers or key phrases in the main figure).

3. **Single-Logit Listwise Scoring**:

    - **Function**: Compresses the entire ranking sequence's autoregressive decoding into "one forward pass + argsort," eliminating decode latency.
    - **Mechanism**: During training, the model is supervised to predict the first token of the target ranking as the key scoring step, with RankNet/Softrank directly supervising the logit $s_i$ for each candidate identifier token $t_i$; at inference, a single model pass yields the first-step logit vector $\bm{z}\in\mathbb{R}^{|\mathcal{V}|}$, and ordering is determined by $\pi=\mathrm{argsort}_{\downarrow}(z_{t_1},\dots,z_{t_k})$.
    - **Design Motivation**: Autoregressive rerankers must re-attend to long contexts at each step, and KV cache cannot avoid the linear growth in steps; listwise ranking is fundamentally about relative preference, and a single logit can encode "this candidate's relative position among all candidates." With aligned training objectives, this suffices. Gangi Reddy 2024 et al. have validated this for text; this work extends it to VLM listwise reranking.

### Loss & Training

Stage 1: $\mathcal{L}_{\text{stage1}}=\mathcal{L}_{\text{LM}}+\lambda_1\mathcal{L}_{\text{ranknet}}$;  
Stage 2: $\mathcal{L}_{\text{stage2}}=\mathcal{L}_{\text{LM}}+\lambda_2\mathcal{L}_{\text{softrank}}$.  
At inference, $u=1$ and $n$ is controlled by $\rho$, allowing smooth accuracy/latency trade-off by tuning $\rho$.

## Key Experimental Results

### Main Results

On MMDocIR's 9-domain page-level retrieval tasks, using DSE-wiki-ss as the first-stage retriever, reranking top candidates, and reporting Recall@1/3/5 and LLM latency (seconds/query):

| Method | Macro R@3 | Micro R@3 | Latency (s) |
|--------|-----------|-----------|-------------|
| DSE-wiki-ss (retriever) | 69.5 | 70.2 | – |
| UniME (Listwise) | 70.9 | 71.4 | 0.24 |
| LamRA (Listwise) | 77.6 | 77.8 | 0.53 |
| MM-R5 (CoT) | 79.1 | 79.0 | 3.82 |
| GPT-5-mini | 88.0 | 88.3 | 23.38 |
| **ZipRerank** | **84.8** | **84.5** | **0.36** |
| ZipRerank-50% (more aggressive pruning) | 83.4 | 83.4 | 0.30 |

ZipRerank outperforms MM-R5 across the board while reducing latency from 3.82s to 0.36s (about $10.6\times$ speedup), and narrows the R@3 gap with GPT-5-mini from ~9 points to ~3.2 points while being $65\times$ faster.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full ZipRerank | Best | Two-stage training + soft ranking + token pruning + single-logit scoring |
| w/o Stage 1 text pretraining | Significant drop | Lacks general listwise ability; hard to learn ranking from weak multimodal supervision |
| w/o Soft-Ranking (hard label) | Drop | Teacher noise directly propagates, overfitting suboptimal rankings |
| w/o Query-Aware Pruning (all tokens) | Latency up ~3-5× | Accuracy only slightly increases, showing most pruned tokens are redundant |
| w/o Single-Token Decoding (autoregressive ranking) | Latency doubles | Accuracy changes little; single-logit scoring is highly cost-effective |
| Different keep ratio $\rho\in\{0.3,0.5,0.7,1.0\}$ | Smooth trade-off | $\rho=0.5$ is Pareto optimal on MMDocIR |

### Key Findings
- Single-logit scoring has negligible impact on accuracy but yields huge latency gains, indicating that effective listwise ranking information can indeed be concentrated in the first-step logit; this aligns with the intuition that "reasoning chains provide optimization direction, but the final ranking is essentially a one-token decision."
- Query-aware token pruning is another major time-saving lever: $\rho=0.5$ barely affects accuracy while reducing prefill complexity by nearly three-quarters; only at 30% does accuracy visibly drop, implying that 50–70% of visual tokens in MMDocIR page images are redundant.
- The robustness gain from soft ranking loss is most evident in small-data/noisy-teacher domains, showing that "listwise need not be hard rank; soft supervision is better suited for knowledge distillation."
- ZipRerank outperforms MM-R5 by about 5 points in 8 out of 9 domains, but is weaker in News, possibly due to high candidate similarity (same topic, different pages), suggesting future work on domain-adaptive pruning thresholds.

## Highlights & Insights
- The latency decomposition $F(n,u)\approx L(c_{\text{att}}dn^2+c_{\text{ffn}}d^2n)+uLdn\cdot c_{\text{dec}}$ is presented as motivation in Section 3, making this a rare work that clearly explains "why it's slow" in a single equation. Subsequent designs target each term: $n$ via pruning, $u$ via single-logit, turning the "accuracy-latency" trade-off into two tunable hyperparameters ($\rho$ and target decode length).
- The "text-first, then multimodal" two-stage training plus "teacher soft labels + geometric decay targets" is an elegant paradigm for learning ranking under "weak multimodal supervision + strong text resources." The RBP-inspired geometric decay target is robust to teacher noise and transferable to any distillation scenario where the teacher provides only coarse-grained rankings.
- Max-sim visual pruning can be interpreted as a "tight upper bound for smooth attention pooling"; the authors provide an $O(\varepsilon)$ attention output perturbation bound, a rare example of "theory-backed engineering optimization" in efficient VLM literature.

## Limitations & Future Work
- Single-logit scoring is inherently limited by the token vocabulary: when $k$ is large (candidates > alphabet), the identifier set must be extended, which may conflict with tokenizer segmentation; the paper does not discuss $k>26$.
- Query-aware pruning may pose higher risks in "visually dense" domains (e.g., academic figures, encrypted PDFs); the authors mainly validate on general MMDocIR and lack domain-adaptive $\rho$ strategies.
- Stage 2 relies on GPT-5 teacher for listwise soft labels, making distillation cost and teacher availability engineering bottlenecks; when the teacher's ranking ability is limited, the soft target's "ceiling" is fixed.
- No end-to-end comparison with the latest hybrid retrievers (e.g., multi-vector ColPali + token reduction); only reranking latency is measured.
- Evaluated only on MMDocIR; not validated on truly long documents (hundreds of pages) or cross-document (M-RAG with merged candidates) scenarios.

## Related Work & Insights
- **vs MM-R5 (Xu et al. 2025)**: MM-R5 achieves SOTA with CoT reasoning but suffers from dual bottlenecks of autoregression and long context, resulting in 3.82s latency; ZipRerank eliminates both via single-logit and token pruning, showing that "the benefits of reasoning chains can be distilled into the first-step logit via soft supervision," without runtime CoT.
- **vs FIRST / RankZephyr (Gangi Reddy 2024)**: Single-logit scoring is validated for pure-text listwise reranking; ZipRerank extends this to multimodal and addresses the unique challenge of visual token pruning.
- **vs Light-ColPali / token reduction**: These works optimize the first-stage retriever's multi-vector search; ZipRerank complements by compressing the second-stage reranking, and both can be combined for a more efficient full-stack system.
- **Insights**: (i) The "latency decomposition → targeted mitigation" methodology is broadly transferable to any VLM inference optimization; (ii) "Teacher soft labels + geometric decay targets" is a general template for distilling ranking behavior under weak supervision; (iii) "Query-aware visual token pruning" can be a standard preprocessing step for all query-conditioned VLM tasks (VQA, grounded captioning).

## Rating
- Novelty: ⭐⭐⭐⭐ Single-logit scoring and max-sim visual pruning are not new individually, but their first combination for multimodal listwise reranking, together with two-stage teacher distillation, constitutes a clear systematic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ MMDocIR 9 domains × 3 Recall metrics × multiple baselines + token keep ratio sweep + training stage ablations; comprehensive and well-controlled, though limited to a single benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is driven by latency decomposition equations, with each component addressed in training/inference sections and theoretical sanity checks (attention perturbation bounds); structure is tight.
- Value: ⭐⭐⭐⭐⭐ Reduces latency by an order of magnitude in MMDocIR/M-RAG scenarios without sacrificing accuracy, directly deployable in production RAG systems; open-source code further lowers the barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking](../../NeurIPS2025/information_retrieval/acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](../../AAAI2026/information_retrieval/mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](../../ACL2026/information_retrieval/why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)

</div>

<!-- RELATED:END -->
