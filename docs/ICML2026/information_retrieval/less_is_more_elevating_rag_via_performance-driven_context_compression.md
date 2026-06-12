---
title: >-
  [Paper Note] Less Is More: Elevating RAG via Performance-Driven Context Compression
description: >-
  [ICML 2026][Information Retrieval & RAG][RAG] CORE-RAG utilizes Group Relative Policy Optimization (GRPO) reinforcement learning with a "performance-as-reward" mechanism to train a 1.5B small compressor. It compresses re…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Context Compression"
  - "GRPO"
  - "Knowledge Distillation"
  - "Performance-Driven"
date: 2026-05-08
content_hash: a11e0cfde9573489
---

# Less Is More: Elevating RAG via Performance-Driven Context Compression

**Conference**: ICML 2026  
**arXiv**: [2508.19282](https://arxiv.org/abs/2508.19282)  
**Code**: https://github.com/ziqiangcui/CORE-RAG-ICML26 (Available)  
**Area**: Information Retrieval / RAG / Context Compression  
**Keywords**: RAG, Context Compression, GRPO, Knowledge Distillation, Performance-Driven

## TL;DR
CORE-RAG utilizes Group Relative Policy Optimization (GRPO) reinforcement learning with a "performance-as-reward" mechanism to train a 1.5B small compressor. It compresses retrieved top-k documents into summaries at ~3% of their original length. Instead of performance degradation, it achieves an average 3.3 EM improvement over full-context RAG across four QA benchmarks.

## Background & Motivation

**Background**: RAG enhances factual QA performance of LLMs by 10+ EM by prepending top-k retrieved documents to the query. However, token counts increase linearly—5 documents require ~700 tokens, and 10 documents require ~1400 tokens—leading to high encoding costs and latency.

**Limitations of Prior Work**: Existing compression methods (RECOMP / NoiseFilter-IB / LongLLMLingua / QGC, etc.) almost inevitably lead to performance drops, typically 2-6 EM lower than the full-context baseline. This is because their training objectives rely on **proxy heuristics**: maximizing mutual information between source and summary, imitating teacher outputs, BM25 lexical overlap, or information entropy pruning. There is no causal chain between these objectives and whether the "downstream LLM answers correctly."

**Key Challenge**: Compression tasks lack ground-truth labels—it is unknown what kind of summary is "optimal" for a downstream LLM to answer a specific question. All surrogate losses are speculative, leading to performance drops when those guesses are inaccurate. Meanwhile, some compression models (e.g., NoiseFilter-IB) have parameter counts approaching those of the downstream LLM itself, causing the saved computation to be consumed by the compressor.

**Goal**: (i) Align the compression objective strictly with downstream task performance to achieve "lossless or superior" results; (ii) Ensure the compressor is significantly smaller than the downstream LLM to truly save computational resources.

**Key Insight**: Since gold summaries do not exist, the **accuracy of the downstream LLM itself is treated as the reward**. The compressor is trained as a policy within an RL framework. The downstream LLM remains frozen throughout (black box), while only the lightweight compressor is trained, making it naturally compatible with API-based model scenarios.

**Core Idea**: Compression is treated as a decision-making process. GRPO is used to optimize the compressor using downstream EM/F1 as rewards. A distillation phase using DeepSeek-V3 combined with data filtering rules provides a warm-start, followed by RL refinement.

## Method

### Overall Architecture
Input: question $q$ + retrieved $k$ documents $D$ (obtained via existing retrievers like DPR / BM25 / Contriever).
Compressor: A small language model $\pi_\theta:(q,D)\mapsto s$ (the study uses Qwen2.5-1.5B-Instruct).
Downstream LLM: A black-box large model $M:(s,q)\mapsto \hat{y}$ (the study uses Qwen2.5-14B-Instruct, with transfer to Llama-3.1-8B for generalization verification), frozen during training and inference.

Training consists of two stages: **Stage 1: Distillation Warm-Start** (enabling the small model to generate "reasonable" summaries and learn "when to output empty strings"), and **Stage 2: Performance-Driven RL via GRPO** (refinement using downstream accuracy as the reward). During inference, the process involves only two steps: $s = \pi_\theta(q,D)$ → $\hat{y}=M(s,q)$, with the compressor acting as a plug-in.

### Key Designs

1. **Performance-Driven RL with GRPO**:

    - **Function**: Trains the compressor using the downstream LLM's accuracy directly as the reward, bypassing the ill-posed question of "what constitutes a good summary."
    - **Mechanism**: The compressor $\pi_\theta$ serves as the policy. For each input $x=(q,D)$, $G$ summaries $\{s_i\}$ are sampled. Each summary is prepended to $q$ and fed into the frozen downstream LLM $M$ to obtain an answer $\hat{y}_i = M(s_i,q)$. The reward $r = r_{\text{EM}} + \alpha \cdot r_{\text{F1}}$ is computed, where $r_{\text{EM}}=\mathbb{I}(\hat{y}=y)$ provides a sparse hard signal and $r_{\text{F1}}$ provides a dense partial signal, with $\alpha\in(0,1]$ as a weight. GRPO (proposed by DeepSeek-Math) is used for optimization, normalizing advantages $A_i=(r_i-\text{mean})/\text{std}$ across the group of $G$ rollouts without a critic model, saving memory. A KL term $\beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\theta_{\text{ref}}})$ is added to prevent deviation from the warm-start.
    - **Design Motivation**: Previous RL-for-RAG methods (e.g., TACO using token-level keep/drop) still used proxy rewards and were restricted to selection rather than rewriting. Here, the reward directly equals downstream performance, and the generative compressor can synthesize content. The frozen black-box LLM makes the method compatible with commercial APIs.

2. **Distillation Warm-Start with Performance-Gated Filtering**:

    - **Function**: Provides a robust initial policy for the 1.5B compressor to prevent RL cold-start collapse and teaches the model critical defensive behaviors, such as outputting an empty string when a summary might be detrimental.
    - **Mechanism**: DeepSeek-V3 (671B) acts as a teacher generating a summary $\hat{s}$ for each $(q,D)$ in the training set. The downstream LLM then scores two conditions: the summary-aided score $p_{\text{summary}}$ and the bare query score $p_{\text{original}}$. Training samples are constructed based on these scores: (a) If $p_{\text{summary}}>p_{\text{original}}$, the sample is retained with the teacher summary $\hat{s}$ as the target; (b) If $p_{\text{original}}=1$ and $p_{\text{summary}}<p_{\text{original}}$ (the summary caused a correct answer to become incorrect), the target is set to an empty string to teach "refusal when necessary"; all other samples are discarded. Standard SFT is performed on the filtered set $\mathcal{X}_f$ with loss $\mathcal{L}_d = -\frac{1}{|\mathcal{X}_f|}\sum \sum_t \log P_{\pi_\theta}(\hat{s}_t\mid q,D,\hat{s}_{<t})$.
    - **Design Motivation**: Teacher summaries are not guaranteed to be optimal. Bidirectional data filtering aligns distillation data with task performance, embedding the groundwork for RL rewards during the SFT phase. Ablations show an average drop of ~3 EM without distillation, verifying it as critical for RL success.

3. **Asymmetric Compressor/Generator Architecture**:

    - **Function**: Ensures the computational savings of compression are realized by keeping the compressor lightweight.
    - **Mechanism**: $\pi_\theta$ is intentionally designed to be much smaller than $M$ (1.5B vs. 14B, roughly 1:10). Only the compressor's gradients are backpropagated; $M$ is frozen as a reward function. During inference, the compressor generates a ~30-50 token summary from 5-10 documents, reducing the LLM's input sequence from ~700-1400 tokens to ~50 tokens (compression ratio ~3-6%).
    - **Design Motivation**: Many prior works (e.g., NoiseFilter-IB) used compressors of similar scale to the downstream LLM, where "FLOPs saved via compression were consumed by the compressor itself." Unlike search-augmented RL paths (e.g., ReSearch / R1-Searcher) that require training the large generator, CORE's training and inference costs are significantly lower.

### Loss & Training
- **Stage 1 (Distillation)**: Standard token-level cross-entropy, objective $\mathcal{L}_d$ (Eq. 1).
- **Stage 2 (GRPO)**: Objective $\mathcal{J}(\theta)$ (Eq. 2), including clipping $\epsilon$ and KL coefficient $\beta$, with group size $G$ rollouts.
- **Composite reward**: $r = r_{\text{EM}} + \alpha r_{\text{F1}}$, with $\alpha\in(0,1]$ analyzed in Figure 4.
- **Inference**: Uses **greedy decoding** for the compressor to ensure deterministic and reproducible summaries.

## Key Experimental Results

### Main Results
Testing on 4 QA benchmarks (NQ / TriviaQA / HotpotQA / 2WikiMultihopQA) with downstream LLM = Qwen2.5-14B-Instruct. All trainable baselines were initialized with Qwen2.5-1.5B-Instruct and trained on 5 documents.

| Dataset | Metric | Full Context Top-5 | CORE(1.5B) | Gain over Full Context | Token Compression Ratio |
|--------|------|----------------|-------------|---------------|----------------|
| NQ | EM | 38.03 | **41.02** | +2.99 | 712→46 (~6.5%) |
| TriviaQA | EM | 64.10 | **65.63** | +1.53 | 715→32 (~4.5%) |
| HotpotQA | EM | 32.99 | **33.67** | +0.68 | 737→36 (~4.9%) |
| 2WikiMultihopQA | EM | 29.64 | **36.72** | +7.08 | 766→49 (~6.4%) |

Compared to strong baselines (trained on 5 docs): RECOMP-Abs/Ext, NoiseFilter-IB, LongLLMLingua, and QGC suffered 2-6 EM drops across almost all datasets. DeepSeek-V3 (671B) summaries also failed to beat full context on NQ/2Wiki. CORE (1.5B) outperformed all compression baselines and the DeepSeek-V3 teacher by 4-5 EM, and even surpassed the full-context top-10 baseline (top-10 EM 38.67 vs. CORE-top5 41.02).

**Length Generalization**: A compressor trained on top-5 documents was applied zero-shot to top-10 documents. CORE achieved 41.88 EM on NQ using only 52 tokens (compression ratio ~3.6%), maintaining a 3.2-point lead over the top-10 full-context baseline.

### Ablation Study
| Configuration | NQ EM | TQA EM | HotpotQA EM | 2Wiki EM | Description |
|------|-------|--------|-------------|----------|------|
| CORE (full) | **41.02** | **65.63** | **33.67** | **36.72** | Distillation + GRPO RL |
| w/o distillation | 36.37 | 65.23 | 32.01 | 31.40 | Direct RL cold start; ~5 EM drop on NQ/2Wiki |
| w/o RL | 34.18 | 60.31 | 28.96 | 30.25 | Distillation only; 3-6 EM drop across datasets |

### Key Findings
- **RL is the primary driver**: Removing RL resulted in larger drops than removing distillation (NQ 41.02→34.18 vs. 41.02→36.37), confirming that teacher summaries are not task-optimal and performance-driven signals are irreplaceable.
- **Distillation is non-trivial**: RL still functions without distillation but is significantly weaker (e.g., 36.72→31.40 on 2Wiki), suggesting that 1.5B models struggle to converge to optimal policies through GRPO alone; warm-starting determines the RL ceiling.
- **Cross-length and cross-LLM generalization**: Compressors trained on top-5 documents remain superior on top-10, indicating they learn task-relevant trade-offs rather than fixed length limits. Transfer to Llama-3.1-8B also maintained the advantage.
- **Small compressors are sufficient**: 1B-3B Llama/Qwen backbones all benefited (Figure 5); 3B was not significantly stronger than 1.5B, suggesting the bottleneck is the training signal rather than capacity.

## Highlights & Insights
- **Simplicity of "Downstream Accuracy as Reward"**: By discarding surrogate losses (mutual information, BM25, teacher imitation) and directly optimizing for the user's goal, this "reward-as-task" approach can be applied beyond compression to reranking, query rewriting, and prompt engineering.
- **"Summary Can Be Empty" is a Critical Freedom**: Explicitly modeling "when to remain silent" during the distillation phase acts as a safeguard. This behavior is rarely explicitly modeled in other compression research.
- **Black-box LLM + Lightweight Plug-in Training**: Since the downstream LLM requires no gradients, the method is easily applied to commercial APIs like GPT-4 or Claude, contrasting with search-reasoning methods that require training the generator.
- **Paradigm Value of GRPO in Generative Compression**: Previously popular in mathematical reasoning, this work proves GRPO is applicable to generative compression tasks as long as rewards can be defined from downstream tasks. Group-wise comparisons naturally fit the "which summary makes the LLM answer correctly" scenario.

## Limitations & Future Work
- **Lack of hard rewards for open generation**: For tasks like dialogue or long-form generation where EM/F1 is unsuitable, LLM-as-Judge or ROUGE/BARTScore with hallucination penalties would be required. This was discussed (Section 3.4) but not empirically tested.
- **Dependence on ground-truth answers**: The method is essentially supervised RL. The cost of cold-starting on domains without labeled answers (e.g., private knowledge bases) might be significant, potentially requiring LLM-as-Judge for pseudo-labels.
- **Coupling with downstream LLMs**: Rewards are calculated by a specific $M$. While the Qwen-trained compressor transferred well to Llama-3.1-8B, the effectiveness of more radical transfers (e.g., different languages or domains) remains unknown.
- **Future Directions**: (i) Introducing process rewards (token or sentence-level) to improve credit assignment; (ii) End-to-end joint training of the compressor and retriever; (iii) Upgrading rewards to calibrated LLM-as-Judge for open generation; (iv) Introducing compression ratio as an auxiliary reward to explicitly control the token-quality trade-off.

## Related Work & Insights
- **vs RECOMP (Xu et al., 2024)**: RECOMP uses token-level cross-entropy to imitate teacher summaries. CORE replaces this with performance rewards, achieving significantly higher results (CORE 41.02 vs. RECOMP-Abs 34.18 on NQ).
- **vs NoiseFilter-IB (Zhu et al., 2024)**: NoiseFilter-IB uses Information Bottleneck to maximize mutual information, which remains a proxy objective; additionally, its compressor scale is similar to the downstream LLM. CORE uses performance alignment and a ~10x smaller compressor.
- **vs LongLLMLingua / QGC (Jiang/Cao et al., 2024)**: These use query-guided token-level pruning (hard pruning). CORE is generative (rewriting) and guided by downstream rewards rather than entropy.
- **vs TACO (Shandilya et al., 2025)**: TACO uses RL for binary keep/drop decisions but is not specialized for RAG. CORE's generative approach with pure performance rewards avoids the performance drops seen in TACO.
- **vs ReSearch / R1-Searcher / WebThinker / DeepResearcher**: This family integrates search+reasoning into the LLM via RL, requiring white-box models and high inference costs. CORE treats the LLM as a black box and trains only a lightweight plug-in, reducing costs by an order of magnitude.

## Rating
- **Novelty**: ⭐⭐⭐⭐ "Downstream performance as reward" is powerful and the "empty string" supervision is clever; however, GRPO is an existing technique, making this a precise application to a new scenario rather than an entirely new algorithm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive coverage across 4 QA benchmarks, multiple retrievers, cross-LLM transfer (Qwen2.5-14B / Llama-3.1-8B), cross-compressor backbones, and length generalization.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression of motivations and clear descriptions; the only minor drawback is the lack of empirical tests for open generation mentioned in the discussion.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the "compression-induced drop" bottleneck in RAG, achieving performance gains at 3-6% length. Highly valuable for industrial RAG systems using API-based LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](../../ACL2026/information_retrieval/more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)
- [\[ICML 2026\] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)
- [\[ICML 2026\] Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](../../ACL2026/information_retrieval/more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](../../ACL2025/information_retrieval/exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[AAAI 2026\] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](../../AAAI2026/information_retrieval/does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)

</div>

<!-- RELATED:END -->
