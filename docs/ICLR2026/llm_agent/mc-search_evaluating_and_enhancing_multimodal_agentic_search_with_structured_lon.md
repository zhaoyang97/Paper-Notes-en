---
title: >-
  [Paper Note] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains
description: >-
  [ICLR 2026][LLM Agent][Multimodal RAG] This paper proposes MC-Search, the first benchmark for agentic multimodal RAG, comprising 3…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Multimodal RAG"
  - "Agentic Search"
  - "Multi-hop Reasoning"
  - "Process-level Evaluation"
  - "Retrieval-Augmented Reasoning"
date: 2026-05-08
content_hash: 8cb4c108d08f9300
---

# MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains

**Conference**: ICLR 2026
**arXiv**: [2603.00873](https://arxiv.org/abs/2603.00873)  
**Code**: [https://mc-search-project.github.io](https://mc-search-project.github.io)  
**Area**: LLM Agent
**Keywords**: Multimodal RAG, Agentic Search, Multi-hop Reasoning, Process-level Evaluation, Retrieval-Augmented Reasoning

## TL;DR
This paper proposes MC-Search, the first benchmark for agentic multimodal RAG, comprising 3,333 high-quality samples (averaging 3.7 hops) across 5 reasoning topology types. The benchmark employs HAVE verification to ensure the necessity of each reasoning step, and introduces the Search-Align process-supervised fine-tuning framework, which substantially improves retrieval planning in open-source models (Qwen2.5-VL-7B F1 +13.7).

## Background & Motivation
**Background**: Multimodal large language models (MLLMs) are evolving from fixed retrieve-then-generate paradigms toward more complex agentic multimodal retrieval-augmented generation (MM-RAG), requiring models to iteratively decompose queries, adaptively retrieve across modalities, and integrate multimodal evidence.

**Limitations of Prior Work**: Existing MM-RAG benchmarks exhibit three critical limitations: (a) most adopt simple QA formats that compress multimodal evidence into text-only pipelines (e.g., MRAG); (b) evaluation is restricted to shallow 1–2-hop retrieval without long reasoning chains (e.g., Dyn-VQA); (c) step-level annotations and explicit reasoning topologies are absent, precluding analysis of modality roles during reasoning.

**Key Challenge**: Real-world queries are typically ambiguous and complex, demanding multi-step, cross-modal, knowledge-intensive reasoning. Yet no suitable benchmark exists to evaluate whether MLLMs can perform long-chain, structured multimodal search reasoning.

**Goal**: (a) Construct the first multimodal agentic RAG benchmark supporting long reasoning chains (≥4 hops); (b) provide step-level annotations and diverse reasoning topologies; (c) design process-level evaluation metrics; (d) leverage verified reasoning chains to improve open-source models.

**Key Insight**: The authors build multimodal knowledge clusters from Wikipedia, define 5 representative reasoning topology structures (serial/parallel, image-initiated/text-initiated/multi-image fork, etc.), and apply HAVE filtering to ensure each reasoning step is both necessary and non-redundant.

**Core Idea**: Long-chain multi-hop reasoning + 5 reasoning topologies + HAVE verification + process-level metrics + Search-Align fine-tuning = comprehensive evaluation and improvement of agentic MM-RAG.

## Method

### Overall Architecture
MC-Search comprises two major components: (1) **Benchmark Construction**—a multimodal knowledge base is built from Wikipedia to generate multi-hop QA pairs covering 5 reasoning topologies, which are then filtered via HAVE and quality validation to yield 3,333 high-quality samples; (2) **Evaluation and Training**—a unified agentic MM-RAG pipeline and process-level metrics are designed for fair evaluation, and Search-Align is used to fine-tune open-source models on verified reasoning chains.

### Key Designs

1. **5 Search-Augmented Reasoning Topologies**:

    - **Function**: Defines 5 representative multi-hop reasoning graph structures. Each reasoning chain is formalized as $\mathcal{G}(Q,A) = \{(q_t, m_t, r_t, a_t)\}_{t=1}^{T}$, where $q_t$ is the sub-question, $m_t$ is the retrieval modality, $r_t$ is the evidence, and $a_t$ is the intermediate answer.
    - **5 Structures**: (i) Image-Initiated Chain (image retrieval followed by text retrieval); (ii) Text-Initiated Chain (text retrieval followed by image verification); (iii) Parallel Image-Text Fork (simultaneous image and text retrieval without cross-step dependencies); (iv) Multi-Images Fork (multi-image visual comparison with textual support); (v) Text-Only Chain (pure-text baseline).
    - **Design Motivation**: To capture serial/parallel reasoning patterns and diverse modality combinations found in real-world scenarios, enabling more comprehensive evaluation.

2. **HAVE (Hop-wise Attribution and Verification of Evidence)**:

    - **Function**: Filters hallucinated and redundant steps from reasoning chains.
    - **Mechanism**: For each step, a contextual utility score is computed as $\text{Util}(t) = \text{F1}(\mathcal{C}) - \text{F1}(\mathcal{C} \setminus r_t)$, measuring the drop in answer accuracy upon removing that step's evidence. A navigational role is also assessed: $\text{Nav}(t)=1$ if the intermediate answer entity of that step appears in downstream sub-questions. Steps with utility below a threshold and $\text{Nav}=0$ are deemed redundant.
    - **Design Motivation**: LLM-generated long reasoning chains frequently contain fabricated steps (plausible but unsupported by evidence) or superfluous steps (non-contributing to the final answer). HAVE's dual verification (direct utility + navigational role) ensures every retained step is indispensable.

3. **Process-Level Evaluation Metrics**:

    - **Function**: Goes beyond answer accuracy to assess reasoning process quality.
    - **Mechanism**: (i) **Hit per Step (HPS)**—the proportion of gold reasoning steps successfully covered by the predicted graph; (ii) **Rollout Deviation (RD)**—the step-count difference between predicted and gold chains, $\text{RD} = ||\hat{\mathcal{G}}| - |\mathcal{G}||$, reflecting over- or under-retrieval; (iii) **LLM-as-a-Judge (LJ)**—scoring along four dimensions: answer accuracy, reasoning coherence, entity coverage, and step alignment.
    - **Design Motivation**: Evaluating only the final answer cannot diagnose failures in retrieval planning or modality selection.

4. **Agentic MM-RAG Pipeline**:

    - **Function**: A unified iterative search-reasoning pipeline enabling fair evaluation across models.
    - **Mechanism**: Each iteration consists of: (a) generating a sub-query and retrieval action (text search / image search / image-to-image search); (b) retrieving the top-1 evidence from the multimodal knowledge base; (c) generating a sub-answer and determining whether to continue searching. Modalities and evidence are logged throughout for chain-level evaluation.
    - **Design Motivation**: Existing work employs heterogeneous pipelines, precluding fair comparison.

5. **Search-Align Process-Supervised Fine-Tuning**:

    - **Function**: Fine-tunes open-source MLLMs using HAVE-verified reasoning chains via SFT.
    - **Mechanism**: Reasoning graphs are converted into dialogue format (assistant generates sub-questions and reasoning; user executes retrieval and returns results). Gemini-2.5-Flash is used to generate reasoning thoughts for each step connecting adjacent hops. Supervised fine-tuning is then performed on these dialogue-style traces.
    - **Design Motivation**: Conventional SFT supervises only the final answer, whereas Search-Align provides step-level supervision signals, teaching the model how to plan retrieval, select modalities, and integrate evidence across hops.

### Loss & Training
Search-Align applies standard next-token prediction loss over dialogue-style reasoning traces. Training data consists of 3,333 HAVE-verified reasoning chains.

## Key Experimental Results

### Main Results (Image-Initiated Chain topology)

| Model | F1(↑) | ΔF1(↑) | LJ(↑) | HPS(↑) | RD(↓) | Golden F1 |
|------|-------|--------|--------|---------|-------|-----------|
| GPT-4o-Mini | 36.49 | 34.18 | 2.63 | 27.51 | 1.46 | 68.29 |
| Gemini-2.5-Flash | 44.10 | 37.38 | 3.01 | 31.46 | 2.91 | 72.39 |
| Gemini-2.5-Pro | **47.61** | **42.76** | **3.18** | 25.90 | 1.05 | 69.83 |
| Claude-3.7-Sonnet | 37.80 | 33.09 | 2.60 | 27.31 | 1.18 | 72.62 |
| InternVL3.5-8B | 39.11 | 29.49 | 2.27 | 22.59 | 1.58 | - |
| + Search-Align | 42.27 | 32.65 | 2.53 | **32.49** | **0.94** | 63.86 |
| Qwen2.5-VL-7B | 26.30 | 8.65 | 1.34 | 16.51 | 4.04 | - |
| + Search-Align | 45.70 | 28.05 | 2.23 | **33.59** | **0.70** | 60.95 |

### Ablation Study (Modality Coverage Analysis)

| Query Type | Modality | Gemini-2.5-Pro Coverage | InternVL-3.5-8B Coverage |
|---------|------|---------------------|----------------------|
| With-image queries | Image | 87.35% | 63.84% |
| With-image queries | Text | 78.61% | 82.67% |
| Without-image queries | Image | **29.50%** | **0.66%** |
| Without-image queries | Text | 83.55% | 89.78% |

### Key Findings
- **Search-Align yields substantial gains**: Qwen2.5-VL-7B achieves an average F1 improvement of +13.7, HPS improvement of +16.0, and RD reduction of 3.1 after fine-tuning, nearly matching Gemini-2.5-Pro.
- **Parallel Image-Text Fork is the hardest topology**: All models achieve the lowest F1 and HPS on this topology, as it requires simultaneously covering both text and image branches.
- **Severe modality bias**: When queries contain no explicit image cues, InternVL's image retrieval coverage drops sharply from 63.84% to 0.66%, indicating a strong default preference for text retrieval.
- **Performance degrades with chain length**: All models exhibit sharp performance drops on 4–5-hop reasoning chains, primarily due to compounding retrieval errors and unstable planning.
- **Moderate over-retrieval is beneficial**: Retrieving 1–2 extra steps (ΔStep=1~2) generally improves accuracy, but over-retrieval of ≥4 steps introduces noise and causes performance to collapse.
- **Retrieval planning is the primary bottleneck**: Error analysis shows that Retrieval-Failure (84.7%), Hallucinated Entity (75.8%), and Step-Omission (74.3%) are the most frequent error types.

## Highlights & Insights
- **The 5-topology design is highly systematic**: Rather than arbitrarily composing multi-hop questions, the authors define a complete combinatorial space of serial/parallel × image/text modalities grounded in real MM-RAG requirements, providing a clear analytical framework for future research.
- **HAVE filtering is elegantly designed**: Necessity is verified by measuring accuracy drop upon step removal, while navigational steps are identified by checking whether intermediate answer entities appear in downstream sub-questions. This dual criterion avoids both under-filtering and over-pruning.
- **Process-level metrics fill a critical gap**: HPS and RD precisely localize whether a model suffers from under-retrieval or over-retrieval, making them practically useful for debugging agentic RAG systems.
- **The modality bias finding is thought-provoking**: Near-zero image retrieval in the absence of explicit image cues indicates that models are far from capable of proactively selecting modalities based on the information needs of the query.

## Limitations & Future Work
- The knowledge base is derived from Wikipedia, limiting domain coverage (scientific and mathematical domains are not included).
- Data generation relies on Gemini-2.5-Flash, introducing model-specific biases.
- Evaluation covers only 6 MLLMs, excluding stronger reasoning models (e.g., GPT-5 series, Gemini-2.5-Pro with thinking).
- Search-Align employs only SFT; reinforcement learning approaches such as RL or DPO remain unexplored.
- The top-1 retrieval constraint may be overly strict, as practical systems typically retrieve multiple results.

## Related Work & Insights
- **vs. MMSearch**: MMSearch is limited to single-hop retrieval and focuses on mixed image-text results from search engines. MC-Search targets long-chain multi-hop reasoning with an emphasis on reasoning structure and process evaluation.
- **vs. WebQA**: WebQA contains at most 2 hops and lacks step-level annotations. MC-Search averages 3.7 hops and provides complete reasoning graph annotations.
- **vs. Agentic RAG systems (e.g., ReAct-style)**: These systems are predominantly designed for text-only scenarios. MC-Search extends agentic RAG to the multimodal setting and, for the first time, systematically evaluates modality planning capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first long-chain multimodal agentic RAG benchmark; the combination of 5 reasoning topologies, HAVE verification, and process-level metrics is highly systematic.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 MLLMs with multi-dimensional analysis (chain length, over-retrieval, modality bias, error types), though broader model coverage would strengthen conclusions.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with complete formalization and rich figures and tables; high information density requires careful reading in places.
- **Value**: ⭐⭐⭐⭐⭐ Provides much-needed evaluation infrastructure and training methodology for the multimodal agentic search community; the effectiveness of Search-Align further validates the training value of the curated data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)
- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](../../CVPR2026/llm_agent/haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](../../CVPR2026/llm_agent/worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](../../ACL2026/llm_agent/structmem_structured_memory_for_long-horizon_behavior_in_llms.md)

</div>

<!-- RELATED:END -->
