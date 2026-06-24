---
title: >-
  [Paper Note] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering
description: >-
  [ACL 2025][Graph Learning][Knowledge Graph Question Answering] This paper systematically diagnoses the root cause of why Small Language Models (SLMs, 0.5B–8B) fail under the Think-on-Graph (ToG) knowledge graph question answering framework, revealing that the **exploration phase rather than the reasoning phase is the performance bottleneck**. It demonstrates that replacing SLM-based KG traversal with a zero-shot, plug-and-play, lightweight sentence retrieval module (SentenceB…
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "Small Language Models"
  - "Think-on-Graph"
  - "Sentence Retrieval"
  - "Exploration Module"
date: 2026-05-08
content_hash: 40a2e378b1a99068
---

# The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering

**Conference**: ACL 2025  
**arXiv**: [2509.07399](https://arxiv.org/abs/2509.07399)  
**Code**: [yijie-cheng/SLM-ToG](https://github.com/yijie-cheng/SLM-ToG/)  
**Authors**: Yi-Jie Cheng, Oscar Chew, Yun-Nung Chen (ASUS / National Taiwan University)
**Area**: Graph Learning / KGQA  
**Keywords**: Knowledge Graph Question Answering, Small Language Models, Think-on-Graph, Sentence Retrieval, Exploration Module

## TL;DR

This paper systematically diagnoses the root cause of why Small Language Models (SLMs, 0.5B–8B) fail under the Think-on-Graph (ToG) knowledge graph question answering framework, revealing that the **exploration phase rather than the reasoning phase is the performance bottleneck**. It demonstrates that replacing SLM-based KG traversal with a zero-shot, plug-and-play, lightweight sentence retrieval module (SentenceBERT/GTR, only ~110M parameters) consistently and significantly improves Exact Match (EM) scores on both the CWQ and WebQSP benchmarks.

## Background & Motivation

**Background**: The LLM×KG paradigm (eg, Think-on-Graph) allows language models to act as agents that dynamically interact with knowledge graphs to retrieve external knowledge, which is widely considered an effective path to alleviate LLM hallucination and enhance explainability. The ToG framework has shown excellent performance on LLMs such as GPT-4 (with EM increasing from 0.457 to 0.540 on CWQ, and from 0.710 to 0.813 on WebQSP).

**Limitations of Prior Work**: Existing LLM×KG approaches almost entirely rely on ultra-large-scale proprietary models (e.g., GPT-4) or require task-specific training/fine-tuning, rendering them unavailable in low-resource environments. There is a lack of systematic research on whether these methods remain effective when end-users only have access to small language models with 0.5B–8B parameters.

**Key Challenge**: Directly applying ToG to SLMs leads to degraded performance—the average EM of SLMs using ToG is only 0.217 on CWQ, which is even lower than the 0.219 of a simple CoT baseline. The interaction paradigm designed by ToG for LLMs completely fails on small models, yet the specific phase of failure (initialization/exploration/reasoning) and its underlying causes remain unclear.

**Goal**: To precisely diagnose which phase of the ToG framework fails when using SLMs, and propose a training-free, plug-and-play solution that enables SLMs to effectively benefit from knowledge graphs.

**Key Insight**: Decoupling the "exploration" and "reasoning" capabilities through controlled experiments—substituting only the exploration phase with GPT-4.1 while keeping the reasoning phase handled by SLMs—quantitatively confirms that exploration is the sole bottleneck for SLMs. Subsequently, lightweight models well-established in the sentence retrieval domain are utilized to specifically handle the sub-task of KG exploration.

**Core Idea**: Small models are not incapable of reasoning; rather, they struggle to navigate ("find the path") on knowledge graphs. A specialized, lightweight retrieval model can be used to navigate for them.

## Method

### Overall Architecture

This work is built upon the Think-on-Graph (ToG) framework, retaining its three-stage process but modifying the key exploration phase. The original ToG workflow is as follows:

1. **Initialization**: The SLM extracts topic entities from the input question, locates these entities in the KG, and forms the initial reasoning paths.
2. **Exploration**: It iteratively expands the reasoning paths using a beam search strategy, selecting and pruning candidates from neighboring relations and entities at each step. **The core modification of this work lies in this step**—replacing the SLM's ranking/pruning with a lightweight retrieval module.
3. **Reasoning**: After collecting sufficient evidence, the SLM generates the final answer based on the maintained reasoning paths.

The modified framework decouples "exploration" from "reasoning": exploration is handled by specialized retrieval modules, while reasoning is conducted by the SLM, with each performing its respective duty. The entire process requires no training or fine-tuning, and the retrieval modules are plugged in in a zero-shot manner. Furthermore, the number of SLM calls is reduced from $2ND+D+1$ to $D+1$ (where $D$ is the number of iterations and $N$ is the number of reasoning paths), significantly saving inference costs.

### Key Designs

#### Key Design 1: Precise Diagnosis of the Exploration Bottleneck

This work's most critical contribution lies in confirming that the exploration phase is the performance bottleneck for SLMs through carefully designed controlled experiments:

- **GPT-4.1-Assisted Exploration Experiment**: GPT-4.1 is tasked solely with KG exploration, and the resulting knowledge triplets are provided to the SLMs for reasoning. The results show that SLMs achieve substantial performance improvements once they receive high-quality context (average gains of +0.159 on CWQ and +0.238 on WebQSP across SLMs), proving that the reasoning capability of SLMs is inherently sufficient and the bottleneck lies in the exploration phase.
- **Typical Failure Case**: For the question "government type of the country where Northern District is located", autonomous exploration by the SLM only retrieved two triplets such as `(Northern District, country, Israel)`, missing the critical `(Israel, form_of_government, Parliamentary system)`, and thus failed to answer. In contrast, the same SLM provided correct answers when given the complete triplets explored by GPT-4.1.
- **Cross-Entropy Alignment Metric**: Using GPT-4.1's exploration output as pseudo-labels, the cross-entropy between the SLM's exploration output and the pseudo-labels consistently decreases as model scale increases, quantitatively supporting the claim that "exploration quality is the limiting factor."
- **Constrained Decoding to Exclude Formatting Noise**: Forcing all models to output in JSON format eliminates the artificially low performance of SLMs caused by non-standard formatting. The results prove that the conclusion of the exploration bottleneck still holds even after formatting factors are controlled.

#### Key Design 2: Lightweight Sentence Retrieval to Replace SLM Exploration

Having confirmed the bottleneck, this work replaces SLM-based KG exploration with sentence retrieval models. Given the question $q$ and a candidate set $P_{cand}$ at each step, the retrieval module computes relevance scores and selects the top-$k$:

$$P_q = \text{Top}_k(\text{score}(p, q)), \quad \forall p \in P_{cand}$$

Three retrieval schemes are investigated, each with distinct features:

- **BM25**: A classic sparse ranking function based on term frequency and inverse document frequency. It matches the tokenized question with TF-IDF of candidate passages. It does not require a GPU but suffers from an inability to capture semantic information through pure lexical matching.
- **SentenceBERT** (~110M parameters): A BERT-based Siamese network trained via contrastive learning to generate semantically rich sentence embeddings. The similarity score is computed as the dot product of the encoded vectors: $\text{score}(p,q) = \langle \mathcal{T}(p), \mathcal{T}(q) \rangle$.
- **GTR** (~110M parameters): A large-scale dual-encoder based on T5, pre-trained on diverse retrieval tasks, exhibiting stronger generalization capabilities.

SentenceBERT and GTR have only around 110M parameters, which is far smaller than the smallest SLM (0.5B), introducing minimal computational overhead while requiring no fine-tuning and working zero-shot.

#### Key Design 3: Asymmetrical Effects of LLMs vs. SLMs

An important discovery is that the exact same technique yields opposite effects on LLMs versus SLMs. Sun et al. (2024) reported that using sentence retrieval to replace GPT-4's own exploration leads to a significant decrease in EM (e.g., from 0.575 to 0.505 on CWQ). This paper explains the underlying reason: LLMs already possess strong KG exploration capabilities, making external retrieval modules a downgrade; whereas for SLMs, whose exploration path is a weakness, the retrieval module acts as an upgrade. This finding serves as a reminder to the community that conclusions drawn from large models cannot be simply transferred to small models.

## Key Experimental Results

### Main Results

#### Table 1: Performance Comparison of ToG on LLMs vs. SLMs

| Model | Method | CWQ (EM) | WebQSP (EM) |
|------|------|----------|-------------|
| GPT-4.1 | CoT | 0.457 | 0.710 |
| GPT-4.1 | ToG | **0.540** (+0.083) | **0.813** (+0.103) |
| Qwen2-0.5b | CoT | 0.129 | 0.272 |
| Qwen2-0.5b | ToG | 0.081 (-0.048) | 0.210 (-0.062) |
| Gemma2-2b | CoT | 0.127 | 0.373 |
| Gemma2-2b | ToG | 0.140 (+0.013) | 0.382 (+0.009) |
| Phi-3-mini-3.8b | CoT | 0.273 | 0.522 |
| Phi-3-mini-3.8b | ToG | 0.270 (-0.003) | 0.520 (-0.002) |
| Qwen2-7b | CoT | 0.275 | 0.544 |
| Qwen2-7b | ToG | 0.300 (+0.025) | 0.637 (+0.093) |
| Llama-3-8b | CoT | 0.291 | 0.603 |
| Llama-3-8b | ToG | 0.296 (+0.005) | 0.569 (-0.034) |
| **SLM Average** | CoT | 0.219 | 0.456 |
| **SLM Average** | ToG | 0.217 (-0.002) | 0.464 (+0.008) |

GPT-4.1 gains a significant improvement from ToG, whereas the SLM average shows almost no change or even a slight drop, indicating that ToG is largely ineffective for SLMs. Under ToG, Qwen2-0.5b's CWQ EM actually plunged from 0.129 to 0.081.

#### Table 2: Effect of Replacing Exploration with Lightweight Retrieval Modules

| Model | ToG Original | +BM25 | +SentenceBERT | +GTR |
|------|----------|-------|---------------|------|
| Qwen2-0.5b (CWQ) | 0.081 | 0.106 | 0.061 | **0.123** |
| Qwen2-0.5b (WebQSP) | 0.210 | 0.236 | 0.174 | **0.262** |
| Gemma2-2b (CWQ) | 0.140 | 0.127 | 0.136 | **0.152** |
| Gemma2-2b (WebQSP) | 0.382 | 0.236 | **0.520** | 0.511 |
| Phi-3-mini (CWQ) | 0.270 | 0.254 | 0.284 | **0.291** |
| Phi-3-mini (WebQSP) | 0.520 | 0.416 | 0.577 | **0.605** |
| Qwen2-7b (CWQ) | 0.300 | 0.251 | 0.318 | **0.331** |
| Qwen2-7b (WebQSP) | 0.637 | 0.513 | 0.665 | **0.671** |
| Llama-3-8b (CWQ) | 0.296 | 0.274 | 0.313 | **0.325** |
| Llama-3-8b (WebQSP) | 0.569 | 0.456 | 0.640 | **0.642** |

GTR consistently yields the best results for all SLMs on CWQ; SentenceBERT and GTR demonstrate more substantial improvements on WebQSP (e.g., Gemma2-2b: 0.382→0.520). The sparse token frequency matching of BM25 exhibits unstable performance in KG relation selection, proving detrimental in certain scenarios.

#### Table 3: Performance Upper Bound with GPT-4.1-Assisted Exploration

| Model | CoT Baseline | GPT-4.1-Assisted ToG | Gain |
|------|----------|-------------------|------|
| Qwen2-0.5b (CWQ/WebQSP) | 0.129 / 0.272 | 0.301 / 0.578 | +0.172 / +0.306 |
| Gemma2-2b (CWQ/WebQSP) | 0.127 / 0.373 | 0.306 / 0.672 | +0.179 / +0.299 |
| Phi-3-mini (CWQ/WebQSP) | 0.273 / 0.522 | 0.421 / 0.736 | +0.148 / +0.214 |
| Qwen2-7b (CWQ/WebQSP) | 0.275 / 0.544 | 0.409 / 0.746 | +0.134 / +0.202 |
| Llama-3-8b (CWQ/WebQSP) | 0.291 / 0.603 | 0.451 / 0.772 | +0.160 / +0.169 |

Even the 0.5B Qwen2 model significantly outperforms the CoT baseline when provided with high-quality exploration context, demonstrating that reasoning capability is not the performance bottleneck.

## Highlights & Insights

- **Diagnostic value outweighs method innovation**: The core contribution of this paper is not proposing a new architecture, but rather pinpointing the root cause of "SLM failure in KGQA" through rigorous experimental design. This diagnostic style of research helps the community avoid investing in wrong directions (such as trying to improve the reasoning capability of SLMs).
- **Engineering wisdom of "decoupling + specialization"**: Assigning different stages of complex tasks to the most suitable components—using only a 110M retrieval model can significantly benefit an 8B model, which is much more practical than end-to-end training or scaling up.
- **Conclusions from LLMs and SLMs cannot be simply transferred**: The exact same technique (replacing exploration with sentence retrieval) degrades performance on GPT-4.1 (CWQ: 0.575→0.505) but brings consistent performance gains on SLMs—a profound and counter-intuitive finding.
- **Zero-shot, zero training, plug-and-play**: All retrieval schemes require no task-specific training, which is highly friendly to actual deployment scenarios.

## Limitations & Future Work

- **No variance from single runs**: All results are based on single runs. Restricted by computational resources, the standard deviations of multiple runs could not be reported, leaving statistical significance to be confirmed.
- **Single Knowledge Graph**: The method was validated only on Freebase; its generalizability to other mainstream KGs such as Wikidata and YAGO remains unknown.
- **Under-explored retrieval models**: Only BM25, SentenceBERT, and GTR were tested. More advanced retrieval models (e.g., ColBERT, BGE series) or reranking strategies might yield better results.
- **Unanalyzed bottlenecks in other phases**: This work focuses specifically on the exploration phase, whereas whether the initialization phase (topic entity extraction) also presents a bottleneck for SLMs was not analyzed in parallel.
- **Lack of comparison with training-based approaches**: The proposed method was not directly compared in terms of EM with SLM-KG approaches that require training (e.g., Reasoning on Graphs, G-Retriever).

## Related Work & Insights

- **Think-on-Graph (Sun et al., 2024)**: This is the core baseline framework of this work, which proposes the paradigm of using LLMs as agents to dynamically traverse KGs. This paper reveals the failure modes and root causes of this framework on SLMs.
- **Reasoning on Graphs (Luo et al., 2024)** / **G-Retriever (He et al., 2024)**: These approaches propose additional reasoning/exploration modules to improve LLM-KG integration but require task-specific training, making them inapplicable to zero-shot settings.
- **LightProf (Ao et al., 2025)**: Also focuses on reducing the inference overhead of LLM-KG integration, proposing a lightweight reasoning framework.
- **CuriousLLM (Yang et al., 2025)**: Enhances multi-document QA using LLM-augmented KG reasoning, representing another direction of leveraging KGs for downstream NLP tasks.
- **SentenceBERT (Reimers & Gurevych, 2019)** / **GTR (Ni et al., 2022)**: Established models in the sentence retrieval field. This paper demonstrates their cross-domain generalizability in the novel scenario of KG exploration.
- **Insight**: Under resource-constrained scenarios, modular design—letting models of different scales perform their respective duties—is more practical and easier to deploy than blindly scaling up.

## Rating

- **Novelty**: ⭐⭐⭐ The method itself lacks technical innovation (directly employing existing retrieval modules), but the diagnostic perspective is unique, revealing critical issues overlooked by the community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Involving 5 SLMs + 1 LLM, 2 datasets, 3 retrieval modules, controlled variables through constrained decoding, cross-entropy alignment analysis, and case studies, the experimental design is highly rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ Three research questions are logically progressive, effectively combining qualitative case studies and quantitative analysis.
- **Value**: ⭐⭐⭐⭐ Practical and direct guidance for the SLM+KG application community, illustrating the correct approach for small models to leverage knowledge graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)

</div>

<!-- RELATED:END -->
