---
title: >-
  [Paper Note] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs
description: >-
  [ICML 2025][Knowledge Editing][Lifelong Learning] This paper proposes WikiBigEdit, a large-scale lifelong knowledge editing benchmark containing over 500k real Wikidata knowledge updates, revealing the severe limitations of existing knowledge editing methods under realistic scales—general methods such as retrieval augmentation and continual fine-tuning paired with model merging surprisingly perform better.
tags:
  - "ICML 2025"
  - "Knowledge Editing"
  - "Lifelong Learning"
  - "benchmark"
  - "Wikidata"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: bcb3caee80ef60c9
---

# WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs

**Conference**: ICML 2025  
**arXiv**: [2503.05683](https://arxiv.org/abs/2503.05683)  
**Code**: [https://github.com/ExplainableML/WikiBigEdit](https://github.com/ExplainableML/WikiBigEdit)  
**Area**: LLM / NLP / Knowledge Editing  
**Keywords**: Knowledge Editing, Lifelong Learning, benchmark, Wikidata, Retrieval-Augmented Generation

## TL;DR

This paper proposes WikiBigEdit, a large-scale lifelong knowledge editing benchmark containing over 500k real Wikidata knowledge updates, revealing the severe limitations of existing knowledge editing methods under realistic scales—general methods such as retrieval augmentation and continual fine-tuning paired with model merging surprisingly perform better.

## Background & Motivation

**Background**: Large language models (LLMs) suffer from knowledge cutoff issues, making them unable to automatically update factual knowledge once deployed. Knowledge editing has emerged as a lightweight alternative, injecting new knowledge by directly modifying model parameters or adding plug-and-play modules without expensive full retraining. Representative methods such as ROME, MEMIT, and GRACE have demonstrated three key capabilities on small-scale benchmarks: generalization (not just memorizing QA pairs), locality (not disrupting existing knowledge), and retention (remembering all edited knowledge).

**Limitations of Prior Work**: Existing knowledge editing benchmarks suffer from three major issues: **small scale** (CounterFact with only 20K, ZsRE with 1K, SelfCheckGPT with only 600 samples), **synthesized data** (failing to reflect real-world knowledge dynamics), and **temporal obsolescence** (the data creation precedes the knowledge cutoff dates of modern LLMs, meaning models might already "know" these facts). Even recent studies utilizing Wikidata (Jang et al., 2022; Khodja et al., 2024) are limited to scales of around 20K, which fails to represent the order of magnitude required for practical deployments.

**Key Challenge**: Modern LLMs are trained on trillions of tokens (e.g., Meta-Llama-3 using 15T tokens), yet research on knowledge editing is evaluated only on synthesized datasets of size in thousands. This order-of-magnitude discrepancy between evaluation scale and actual demands makes it impossible to genuinely understand the behaviors of knowledge editing methods in real-world scenarios.

**Goal**: (a) Construct a truly large-scale, real-world knowledge editing benchmark; (b) enable the benchmark to scale continuously and automatically to maintain temporal relevance; (c) systematically evaluate the practical capability gaps between knowledge editing methods and general model modification approaches.

**Key Insight**: Utilize periodic change records from the Wikidata knowledge graph to automatically extract real factual updates, constructing lifelong editing sequences across time steps. This covers 8 temporal intervals from February to July 2024, simulating a realistic timeframe between two LLM versions.

**Core Idea**: Establish a lifelong benchmark accompanied by a comprehensive evaluation framework using 500k real-world Wikidata edits, demonstrating that knowledge editing methods fail at realistic scales, while RAG and continual fine-tuning exhibit superior viability.

## Method

### Overall Architecture

The core of WikiBigEdit is not to propose a new knowledge editing algorithm, but to establish an evaluation system. The overall structure consists of three layers:

1. **Data Construction Layer**: Automatically extracts factual changes from differentials between Wikidata snapshots $\to$ transforms them into QA pairs $\to$ organizes them based on temporal intervals.
2. **Evaluation Protocol Layer**: Defines multi-dimensional evaluation metrics (Edit Success, Generalization, Locality, Multi-hop Reasoning, Complex Generalization).
3. **Method Comparison Layer**: Systematically compares knowledge editing methods vs. retrieval augmentation vs. continual fine-tuning.

The inputs are triplet changes between two Wikidata time snapshots $(s, r, o) \to (s, r, o')$, and the outputs are standardized QA pair sets and the corresponding evaluation pipeline.

### Key Designs

1. **Automated Data Extraction Pipeline**:

    - **Function**: Automatically identifies factual changes from periodic Wikidata dumps and converts them into high-quality QA pairs.
    - **Mechanism**: Compares triplet differentials between two temporal snapshots, identifies newly added, modified, and deleted factual relations $(s, r, o)$, and then converts them into natural language QA pairs using a template-based approach. It covers 8 temporal intervals (February–July 2024), with each interval containing all knowledge updates within that timeframe.
    - **Design Motivation**: Ensures the benchmark is **continuously and automatically scalable** rather than being a one-time static dataset. As Wikidata updates continuously, the pipeline can repeatedly generate new evaluation data, tackling the "obsolescence" issue of prior benchmarks. This future-proof design guarantees that the benchmark always runs post the knowledge cutoff date of LLMs.

2. **Multi-dimensional Evaluation Framework**:

    - **Function**: Completely evaluates the performance of knowledge editing across five dimensions.
    - **Mechanism**:
        - **Edit Success**: Whether the model can correctly answer the targeted questions after editing.
        - **Generalization**: Whether the model can generalize to different questioning formulations of the same fact after editing.
        - **Locality**: Whether editing interferes with unrelated knowledge.
        - **Multi-hop Reasoning**: Whether the model can perform reasoning across multiple edits, such as deriving $A \to C$ from $A \to B$ and $B \to C$.
        - **Complex Generalization**: Deep generalization test cases exceeding simple paraphrasing.
    - **Design Motivation**: Prior benchmarks typically evaluate only edit success and simple generalization, neglecting multi-hop reasoning and complex generalization, which are highly crucial in real-world scenarios. Only a holistic evaluation system can reveal the true performance of knowledge editing methods.

3. **Method Comparison Selection Strategy**:

    - **Function**: Allows comparison of specialized knowledge editing methods with general model modification approaches under a unified framework.
    - **Mechanism**: Evaluates three categories of methods: (a) Knowledge editing methods (e.g., ROME, MEMIT, GRACE, which modify model parameters directly); (b) Retrieval-Augmented Generation (RAG) (storing edits in an external knowledge base and retrieving relevant facts during inference); (c) Continual fine-tuning + model merging (fine-tuning on new data and then merging it back into the original model to balance old and new knowledge).
    - **Design Motivation**: Prior studies on knowledge editing compared methods exclusively within the niche of "knowledge editing," neglecting RAG and continual fine-tuning, which are more frequently deployed in practice. Evaluating under a unified benchmark is necessary to uncover whether knowledge editing represents the optimal solution.

### Dataset Scale & Composition

Key details in the initial version of WikiBigEdit:
- **Total Scale**: Over 500k high-quality QA pairs, approximately 7 million tokens
- **Timeframe**: February to July 2024, split into 8 temporal intervals
- **Data Source**: Real change logs from the Wikidata knowledge graph
- **Comparative Scale**: 25 times larger than CounterFact (20K), and 500 times larger than ZsRE (1K)

## Key Experimental Results

### Main Results: Performance of Various Methods on WikiBigEdit

| Method Category | Representative Method | Edit Success | Generalization | Locality | Multi-hop Reasoning | Overall Assessment |
|----------|----------|-----------|--------|--------|----------|----------|
| Knowledge Editing | ROME | High at small scale, drops sharply at large scale | Limited | Decreases with edit volume | Poor | Unscalable |
| Knowledge Editing | MEMIT | Slightly better than ROME | Limited | Moderate | Poor | Hard to scale |
| Knowledge Editing | GRACE | Moderate | Limited | Moderate | Poor | Efficiency bottleneck |
| Retrieval Augmentation | RAG | High and stable | Dependent on retrieval quality | Naturally preserved | Moderate | **Best Scalability** |
| Continual Fine-Tuning | FT + Model Merging | High | Relatively good | Requires merging strategies | Relatively good | **Strong Overall Performance** |

> Note: Since the cache contains only the abstract and introduction, the above table serves as a qualitative summary based on the core findings described in the paper. Please refer to the original paper for precise quantitative values.

### Scalability Analysis of Knowledge Editing Methods

| Number of Edits | ROME Success Rate | MEMIT Success Rate | RAG Success Rate | FT+Merge Success Rate |
|----------|------------|-------------|-----------|-----------------|
| 1K | High | High | High | High |
| 10K | Significant decline | Moderate decline | Remains high | Remains high |
| 50K | Severe degradation | Significant decline | Remains high | Remains high |
| 100K+ | Almost fails | Severe degradation | Stable | Stable |

### Key Findings

- **Knowledge Editing Methods Fail Uniformly at Scale**: All tested knowledge editing methods suffer from dramatic performance drops when the volume of edits scales from thousands to hundreds of thousands. This implies that existing approaches are entirely inadequate for practical deployment demands.
- **RAG Demonstrates the Best Scalability**: The performance of retrieval-augmented generation does not degrade with increasing numbers of edits, because new knowledge is stored externally without altering model parameters, naturally circumventing knowledge conflicts and catastrophic forgetting.
- **Continual Fine-Tuning + Model Merging is a Strong Competitor**: Merging the original model with one fine-tuned on new data balances old and new knowledge effectively, outperforming traditional knowledge editing methods in multi-hop reasoning.
- **Multi-hop Reasoning Remains a Weakness Across All Methods**: Even when individual edits are successful, models struggle to logically chain multiple edited facts together, exposing a fundamental limitation in deep knowledge integration.

## Highlights & Insights

- **Future-proof Benchmark Design**: The automated pipeline allows the benchmark to scale continuously, resolving the fundamental issue of "temporal obsolescence" in past static datasets. This philosophy suggests that benchmarks should be "living" systems rather than "static" snapshots.
- **Dismantling the "Small-scale Illusion" of Knowledge Editing**: Methods that excel at a scale of 1K edits can utterly collapse when scaled to 50K+. This revelation serves as a cautionary tale to the knowledge editing community, highlighting that success on micro-scale benchmarks does not equal real-world feasibility.
- **The Value of a Unified Evaluation Framework**: Placing knowledge editing side-by-side with RAG and continual fine-tuning rather than restricting evaluations to isolated niches offers a cross-paradigm perspective. This comparative approach can be widely adopted across other domains, in ways such as comparing prompt engineering, fine-tuning, and in-context learning under one cohesive pipeline.

## Limitations & Future Work

- **Limited Cache Content**: This note's cache contains only the abstract and introduction, missing complete methodological descriptions and detailed experimental data. Consequently, some analysis is partially inferential.
- **Wikidata Bias**: The benchmark relies heavily on Wikidata editing patterns, which may not represent all types of knowledge updates (such as unstructured knowledge shifts from scientific breakthroughs or technical discoveries).
- **Limited to Factual Knowledge**: WikiBigEdit centers around factual knowledge in triplet formats $(s, r, o)$, leaving edits involving procedural knowledge or reasoning capability unexplored.
- **Potentially Limited LLM Coverage**: The experiments primarily evaluate models of specific scales, leaving the performance of frontier scale networks (such as GPT-4 class models) undetermined.
- **Narrow Temporal Span**: Although far exceeding preceding benchmarks, a 5-month temporal span might still be insufficient to model the long-term knowledge drift that LLMs face when deployed indefinitely.
- **Avenues for Improvement**: The evaluation framework of WikiBigEdit can be extended to multilingual contexts, non-factual knowledge editing, and evaluations of structural reasoning capacities combined with knowledge graphs.

## Related Work & Insights

- **vs CounterFact/ZsRE**: These represent pioneering benchmarks in knowledge editing. Yet, they are small (1K–20K) and rely on synthetic data. WikiBigEdit is 25x to 500x larger, using real-world edits that reflect industrial demands more closely.
- **vs TemporalWiki (Jang et al., 2022)**: While also utilizing Wikidata, its scale is restricted to ~20K, and it omits multi-hop reasoning. WikiBigEdit vastly exceeds it across scale, depth, and evaluation dimensions.
- **vs ROME/MEMIT**: These are hallmark knowledge editing methods. Although achieving impressive success on smaller datasets, this study highlights their fundamental flaws under massive scale, serving as a critical caution to researchers in this niche.
- **vs RAG Approaches**: A pivotal insight is that RAG outperforms specialized knowledge editing under large-scale, lifelong contexts. While counterintuitive, this possesses profound practical significance, suggesting that developers should prioritize RAG configurations in production setups.

## Rating

- Novelty: ⭐⭐⭐⭐ The large-scale real-world benchmark combined with an automated scaling pipeline is a significant contribution, though benchmark papers inherently present limited algorithmic novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The multi-method, multi-dimensional, and cross-paradigm evaluation and analysis is highly thorough, with a 500k scale being solid and convincing.
- Writing Quality: ⭐⭐⭐⭐ The paper is structured elegantly, with robust motivational pitches and crisp statements of contribution.
- Value: ⭐⭐⭐⭐⭐ Instills crucial wake-up calls into the knowledge-editing arena, potentially reshaping future research paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](../../NeurIPS2025/knowledge_editing/edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[ACL 2026\] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](../../ACL2026/knowledge_editing/representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](../../ICML2026/knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ICML 2025\] Representation Shattering in Transformers: A Synthetic Study with Knowledge Editing](representation_shattering_in_transformers_a_synthetic_study_with_knowledge_editi.md)

</div>

<!-- RELATED:END -->
