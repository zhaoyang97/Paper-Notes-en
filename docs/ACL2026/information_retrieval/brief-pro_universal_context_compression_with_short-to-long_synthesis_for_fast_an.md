---
title: >-
  [Paper Note] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning
description: >-
  [ACL 2026][Information Retrieval & RAG][Long-context compression] To address the issues of slow inference and information drowning in RAG with 10k+ word contexts…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Long-context compression"
  - "RAG"
  - "Multi-hop QA"
  - "Synthetic data"
  - "User-controllable"
date: 2026-05-08
content_hash: 8aab5c6f426824f0
---

# BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning

**Conference**: ACL 2026  
**arXiv**: [2510.13799](https://arxiv.org/abs/2510.13799)  
**Code**: https://github.com/JasonForJoy/BRIEF (Available)  
**Area**: Information Retrieval / RAG / Context Compression  
**Keywords**: Long-context compression, RAG, Multi-hop QA, Synthetic data, User-controllable

## TL;DR
To address the issues of slow inference and information drowning in RAG with 10k+ word contexts, the authors synthesize multi-hop long-context training data using "short-context seed data $\rightarrow$ Wikipedia expansion $\rightarrow$ head-tail iterative pruning." They fine-tune a 3B Llama-3.2 extractive summarizer, BRIEF-Pro, which outperforms LongLLMLingua (9$\times$ compression) at a 32$\times$ compression rate across four multi-hop QA datasets. It also supports direct control of summary length via sentence count instructions.

## Background & Motivation

**Background**: RAG has become the mainstream paradigm for mitigating LLM hallucinations and supplementing new knowledge. However, as retrieved documents increase and contexts extend to the 10k-word level, inference latency and the "cognitive burden" on reader models rise sharply. Multi-hop QA is particularly problematic because reasoning across multiple evidence segments causes relevant signals to be diluted by a large number of distractors, leading to a severe "lost in the middle" phenomenon.

**Limitations of Prior Work**: Existing context compression work follows two main paths. The first is soft prompting (GIST / AutoCompressor), which must be tied to a specific reader and is non-transferable. The second is text summarization (RECOMP, BRIEF, CompAct, LongLLMLingua); however, their training data mostly comes from short contexts (<1k words), which leads to failures in capturing critical dependencies when generalizing to 10k+ word inputs. Furthermore, compression budgets are often fixed, preventing users from adjusting summary granularity. While LongLLMLingua supports long inputs, it relies on perplexity for token-level pruning, and the cost of repeated encoding by its 7B backbone results in FLOPs that can even exceed those of the reader.

**Key Challenge**: The most direct method to train a compressor capable of handling 10k+ word contexts is to collect 10k+ word training data, but such data is scarce and expensive to annotate. Training only on short data fails when the model encounters long inputs—creating a gap between "training data availability" and the "target input length."

**Goal**: (1) Synthesize long-context training samples from inexpensive short-context seed data; (2) Provide user-controllable compression granularity (sentence count); (3) Achieve higher QA accuracy, higher compression rates, and lower FLOPs simultaneously on multi-hop QA.

**Key Insight**: The authors noted that oracle paragraphs in HotpotQA / MuSiQue originally come from specific Wikipedia pages. One can manually construct long contexts by expanding "by position" around the original paragraphs. Furthermore, since oracle paragraphs themselves contain redundancy, they can be tightened using a "remove tokens without drops in performance" criterion. This increases the input length while compressing the target summary, creating greater contrast.

**Core Idea**: Use "short-to-long synthesis" to create data—expanding short oracle/distractor documents to ~6k words based on Wikipedia positions, then generating compact summaries through head-tail iterative pruning. Finally, perform length-controllable SFT using sentence instructions to train a lightweight 3B compressor.

## Method

### Overall Architecture
The system consists of a 3B compressor $\mathcal{C}$ (Llama-3.2-3B-Instruct) and a frozen reader $\mathcal{M}$ (8B/70B/GPT-4.1-nano). Flow: The retriever returns long context $\mathbf{D}$ for query $\mathbf{x} \rightarrow$ Compressor $\mathcal{C}$ outputs summary $\mathbf{s}$ under an optional instruction $\mathbf{i}$ ("Summarize ... in K sentences, K=[P] k [\P]") $\rightarrow$ Reader generates the answer using $\mathbf{x}+\mathbf{s}$. The training data $\mathcal{D}_{comp}$ is produced entirely by the synthesis pipeline, using a standard next-token loss: $\max_{\mathcal{C}} \mathbb{E}\log p_{\mathcal{C}}(\mathbf{s}|\mathbf{x},\mathbf{D},\mathbf{i})$.

### Key Designs

1.  **Short-to-Long Context Expansion**:
    - **Function**: Expands seed documents from HotpotQA/MuSiQue/LongAlign (<1k words) into long inputs of ~6k words.
    - **Mechanism**: For each oracle or distractor document, the source page is first retrieved from the Wikipedia corpus used by Izacard et al. Using the original paragraph as an anchor, several sentences are expanded before and after it. Expansion ratios are sampled from a normal distribution ($\mu=20$) to ensure diverse lengths. **Crucially**, both oracle and distractor documents are expanded—expanding only oracles results in unnaturally "clean" long contexts (Table 4 shows this leads to a 2.77~3.77 drop in average QA score).
    - **Design Motivation**: Uses existing Wikipedia to provide semantically coherent long text, avoiding the discontinuity of concatenated pseudo-long contexts while retaining distractors to simulate "signal-noise mixture" in real RAG results.

2.  **Head-Tail Iterative Pruning**:
    - **Function**: Removes sentences from oracle paragraphs that are "unnecessary for answering," resulting in dense, continuous, and useful target summaries.
    - **Mechanism**: Defines "helpfulness" by comparing the log-likelihood $\log p(\mathbf{y}|\cdot)$ of the correct answer $\mathbf{y}$ before and after removing a sentence $\mathbf{p}_{ij}$. If likelihood increases after deletion, the sentence is deemed "useless." **Implementation**: Assuming key information is usually central, iterative detection is performed only on the head and tail sentences of each oracle document: sentences are deleted from the start until a useful one is found; the same applies to the tail. This yields target summaries as continuous segments from the middle of the document. The final training data averages 6.0k word inputs $\rightarrow$ 0.2k word summaries (~30$\times$ compression).
    - **Design Motivation**: Direct LLM "summarization" is prone to abstractive hallucinations. LM-likelihood pruning provides an unsupervised usefulness criterion that avoids dependency on manual labels. Pruning only the head and tail ensures the summary is a continuous span in the original text, stabilizing both readability and training targets.

3.  **User-controllable Compression Instruction**:
    - **Function**: Allows users to directly specify "how many sentences to compress into" during inference, enabling flexible trade-offs between compression rate and information retention.
    - **Mechanism**: Append "Summarize the documents relevant to the question in K sentences, where K = [P] k [\P]" to the compressor's input. During training, $k$ is set to the actual number of sentences in the target summary, helping the model learn the exact mapping. Inference supports High/Medium/Low (5/10/20 sentences) and Auto modes. An Auto$_{L7C}$ variant initialized with Llama2-7B-Chat is used for fair comparison with LongLLMLingua.
    - **Design Motivation**: Previous compressors (RECOMP, CompAct, BRIEF) had fixed compression rates. Modeling "length control" as a natural language condition is more user-friendly and allows one model to cover multiple compression levels.

### Loss & Training
LoRA fine-tuning of Llama-3.2-3B-Instruct using AdamW, batch 64, 3 epochs, running for approximately 2 days on 2$\times$ A100-80GB. The training set contains 45.2k samples with an average context of 6.0k words (std dev 3.5k) and summaries averaging 0.2k words.

## Key Experimental Results

### Main Results
Evaluated on 4 multi-hop QA datasets (MuSiQue / HotpotQA / 2WikiMultiHopQA / LongSeal, contexts 4.9k–14.8k words) using 3 readers (Llama-3.1-8B/70B, GPT-4.1-nano). Metrics include EM/F1 and compression rate (Rate).

| Reader | Method | Average QA (EM+F1)/2 | Rate |
|--------|------|----|----|
| Llama-3.1-8B | Non-compression | 32.09 | 1× |
| Llama-3.1-8B | LongLLMLingua | 32.02 | 9× |
| Llama-3.1-8B | GPT-4.1-nano as Compressor | 36.87 | 110× |
| Llama-3.1-8B | **BRIEF-Pro-Auto** | **38.79** | **32×** |
| Llama-3.1-8B | **BRIEF-Pro-Low** | **40.06** | **25×** |
| Llama-3.1-70B | Non-compression | 44.98 | 1× |
| Llama-3.1-70B | LongLLMLingua | 40.91 | 9× |
| Llama-3.1-70B | **BRIEF-Pro-Auto** | **45.58** | **32×** |
| Llama-3.1-70B | **BRIEF-Pro-Low** | **46.49** | **25×** |
| GPT-4.1-nano | Non-compression | 33.53 | 1× |
| GPT-4.1-nano | LongLLMLingua | 33.03 | 9× |
| GPT-4.1-nano | **BRIEF-Pro-Auto** | **40.80** | **32×** |

On the 70B reader, BRIEF-Pro-Auto is 4.67 points higher than LongLLMLingua, with a 3.5$\times$ higher compression rate, while total FLOPs are only 23% of the latter.

### Ablation Study

| Configuration | Avg Input Length | 8B Avg QA | 70B Avg QA | GPT-4.1-nano Avg QA |
|------|------|------|------|------|
| Oracle++ & Distractor++ (Full) | 6.0k | **38.79** | **45.58** | **40.80** |
| Oracle+ & Distractor+ (Less Expansion) | 3.6k | 36.02 | 41.74 | 39.11 |
| Oracle+++ only (Oracle expansion only) | 3.6k | 33.76 | 41.68 | 37.03 |

| Compression Mode | Expected Sentences | Actual Avg Sentences |
|------|------|------|
| High | 5 | 6.2 |
| Medium | 10 | 10.4 |
| Low | 20 | 18.0 |

### Key Findings
- **Long input + long-distance noise is key**: Compressors trained only on oracle expansions drop 3~5 points. Distractors provide the "signal-noise mixture" necessary for learning robust compression.
- **Compression outperforms non-compression**: On 70B and GPT-4.1-nano, 32$\times$ compression is 0.6~7.3 points higher than feeding raw text, proving long contexts hinder multi-hop integration.
- **Good controllability of sentence instructions**: Error for High/Medium modes is only 0.4~1.2 sentences. Low (20 sentences) generates slightly shorter results (18) but remains close to the instruction.
- **Significant computational gains**: For the 70B reader, total TFLOPs drop to 8% of non-compression and 24% of LongLLMLingua; end-to-end latency drops to 7% of LongLLMLingua.

## Highlights & Insights
- The approach of "synthesizing long data from short data" is practical. Since HotpotQA/MuSiQue oracles have Wikipedia sources, "position-controlled context expansion" using structured corpora is nearly zero-cost and more natural than forced document concatenation.
- Head-tail iterative pruning bypasses the high cost of "LM likelihood for the whole segment": it requires only $O(\text{Head}+\text{Tail})$ likelihood evaluations, reducing complexity from $O(\text{Length})$ to nearly constant. It ensures the summary is a continuous span, making the model prefer "extraction" over "fabrication."
- Treating "compression length" as a natural language instruction rather than a hyperparameter or special token provides a "text interface." This allows readers to dynamically determine summary length based on context, offering better transferability than fixed-budget schemes.
- The fact that 32$\times$ compression outperforms non-compression on the 70B reader is strong evidence for "lost in the middle"—proving that an LLM's ability to "read" a long document does not equate to its ability to "utilize" it effectively.

## Limitations & Future Work
- The training data caps at ~10k words; quality may not be maintained for extremely long contexts (e.g., full papers, multi-document dialogues) of 20k+ words.
- Evaluation is limited to multi-hop QA. Tasks requiring high "completeness" like few-shot ICL, code completion, or long dialogue memory were not tested. For code, extractive summarization might break syntax or variable dependencies.
- Head-tail pruning assumes "key information is centered." It might miss critical sentences in domains where answers appear at the beginning or end (e.g., news leads, technical document conclusions).
- Future Work: (1) Use hierarchical compression for 20k+ inputs; (2) Train a "compression length selector" to adaptively determine sentence counts in Auto mode; (3) Add code-aware supervision for structured scenarios.

## Related Work & Insights
- **vs RECOMP / BRIEF**: All use text summarization for compression, but RECOMP distills GPT-3.5 and BRIEF uses T5 + chunking, with limited input lengths (<1k or requiring chunking). This work uses short-to-long synthesis + 3B Llama to process 6k word inputs directly.
- **vs LongLLMLingua**: LongLLMLingua uses a 7B causal LM for perplexity-based token compression, which is computationally heavy. This work's 3B abstractive compressor generates a summary in a single forward pass, using only 23% of the FLOPs while achieving 3.5$\times$ higher compression.
- **vs CoLoR (Seo et al. 2025)**: Both use synthetic data to train compressors, but CoLoR differs in its pipeline, supervision signals, and target lengths. This work emphasizes "short-to-long + user-controllable" dimensions.
- **Insight**: The technique of "expanding original data by position using structured external corpora" can be transferred to other long-context SFT data synthesis (e.g., long-context instruction-tuning, long-context RM data), providing more natural results than needle-in-haystack synthesis.

## Rating
- Novelty: ⭐⭐⭐⭐ Short-to-long synthesis + controllable instructions is a simple yet effective combination; head-tail pruning is a clever engineering shortcut.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 readers $\times$ 4 datasets + non-Wikipedia cross-domain + FLOPs/Latency + instruction accuracy + expansion ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear storyline, though synthesis pipeline details are spread across sections 3.3.1~3.3.3.
- Value: ⭐⭐⭐⭐⭐ 3B compressor + 32$\times$ compression + outperforming non-compression is highly practical for industrial RAG; code and data pipeline are reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs](../../ICML2026/information_retrieval/pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)
- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](../../ICML2026/information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[AAAI 2026\] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](../../AAAI2026/information_retrieval/opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)

</div>

<!-- RELATED:END -->
