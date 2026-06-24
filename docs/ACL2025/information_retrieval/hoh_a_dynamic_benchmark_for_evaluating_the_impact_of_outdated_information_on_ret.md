---
title: >-
  [Paper Note] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes HoH, the first large-scale dynamic benchmark specifically designed to evaluate the impact of outdated information on RAG systems, containing 96,124 QA pairs and 219,463 documents, revealing the severe hazards of outdated information on RAG performance and safety.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Outdated Information"
  - "Dynamic Benchmark"
  - "Time-Sensitive QA"
  - "Knowledge Updates"
date: 2026-05-08
content_hash: 8b7bc1b9907d91ae
---

# HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2503.04800](https://arxiv.org/abs/2503.04800)  
**Code**: [GitHub](https://github.com/0russwest0/HoH)  
**Area**: NLP / RAG  
**Keywords**: RAG, Outdated Information, Dynamic Benchmark, Time-Sensitive QA, Knowledge Updates

## TL;DR

This paper proposes HoH, the first large-scale dynamic benchmark specifically designed to evaluate the impact of outdated information on RAG systems, containing 96,124 QA pairs and 219,463 documents, revealing the severe hazards of outdated information on RAG performance and safety.

## Background & Motivation

Retrieval-Augmented Generation (RAG) is widely applied to address knowledge obsolescence in LLMs by retrieving the latest information from external knowledge bases to assist generation. However, current research mainly focuses on how to obtain the newly updated information, while ignoring a critical challenge: **the ubiquitous presence of outdated information within knowledge bases**.

To illustrate, when querying "Who is the current President of the United States?", a RAG system might simultaneously retrieve both current and outdated information (such as documents about the previous president), causing model confusion or even leading to incorrect responses. This is especially prevalent in search engine scenarios where historical content is cached and redistributed.

Existing dynamic QA benchmarks suffer from several limitations: (1) relying heavily on human annotation, leading to limited scale; (2) lacking dedicated labeling for outdated information; and (3) being unable to maintain continuous updates. The introduction of HoH aims to fill these gaps, providing the first systematic study on the hazards of outdated information for RAG.

## Method

### Overall Architecture

The HoH benchmark is composed of two core components:
- **HoH-QA**: A dynamic QA dataset that tracks factual changes over time.
- **HoH-SearchEngine**: A simulated search engine mimicking real-world search scenarios, maintaining both current and historical documents.

### Key Designs

#### 1. Factual Change Extraction

**Function**: To extract factual changes from different Wikipedia snapshots over time.

**Mechanism**: A two-stage method is adopted: first, the Myers Diff algorithm is used to detect modifications at the sentence level, then specific differences are identified at the character level, and finally a semantic filtering model filters out non-factual changes.

**Design Motivation**: Prior methods (e.g., EvolvingQA, GrowOVER) only extract changes at the sentence level, yet many sentence modifications are not factual changes (such as grammatical corrections or wording adjustments). Introducing character-level diff combined with heuristic filtering and semantic model screening significantly improves extraction quality.

The semantic filtering model is fine-tuned based on Qwen2.5-0.5B, trained on 2,000 human-annotated sentence pairs, achieving 96.8% accuracy and 95.1% F1.

#### 2. QA Generation and Automatic Update

**Function**: To automatically generate time-sensitive QA pairs from extracted factual changes and maintain the continuous evolution of the dataset.

**Mechanism**: For newly discovered factual changes, LLMs are used to generate questions containing temporal dimensions, while simultaneously producing current and outdated answers. For subsequent changes of existing facts, LLMs are leveraged to update the answers of existing questions, forming an answer evolution chain.

**Design Motivation**: Real-world knowledge is continually shifting—the same fact can undergo multiple updates. Preserving the complete historical record of answers allows for a more accurate evaluation of RAG performance when handling multi-version information.

#### 3. HoH-SearchEngine

**Function**: To construct a simulated search engine based on Elasticsearch, maintaining both current and historical versions of documents.

**Mechanism**: A Gaussian decay function is introduced on top of the default BM25 ranking to apply temporal penalties to outdated information, simulating the temporal preferences of real-world search engines.

**Design Motivation**: Outdated information is inevitable in real search engines, where old documents are cached, cited, or forwarded. Pure QA evaluation fails to capture this complexity, whereas a simulated search engine reflects the actual challenges faced by RAG in real-world scenarios more realistically.

### Evaluation Framework

Documents/passages are classified into three categories:
- **Relevant (R)**: Relevant passages containing the correct answers.
- **Outdated (O)**: Passages that were once relevant but are now outdated.
- **Distracting (D)**: Semantic-resembling distractor passages that do not contain the correct answer.

Scoring system: Perfect (+1) / Missing (0) / Harmful (-1), penalizing erroneous information rather than omission.

## Key Experimental Results

### Main Results: RAG End-to-End Performance

| Model | Total Score | R only | R+O | O only | None |
|------|------|--------|-----|--------|------|
| Llama-70B | 51.7% | High | Significant Drop | Dangerous | Low but Safe |
| Llama-8B | 40.0% | Medium | Severe Drop | Highly Dangerous | Low |
| Qwen-7B | 29.9% | Medium | Critically Dropped | Worse than Random | Low |

### Retrieval Module Analysis

| Retrieval Method | R@5 (R) | R@5 (O) | Description |
|----------|---------|---------|------|
| No Temporal Decay | 0.8707 | 0.8837 | Outdated information has a higher hit rate |
| Gaussian Decay | 0.7023 | 0.4950 | R drops 17% to achieve O reduction |
| BGE-M3 | 0.7312 | 0.5507 | Still has a 55% probability of retrieving outdated info |

### Generation Module Analysis (Key Findings)

| Condition | Llama-70B Score | Llama-8B Score | Qwen-7B Score |
|------|--------------|-------------|-------------|
| R×1, D×0 | 89.12 | 85.70 | 76.31 |
| R×1, D×6 | 87.98 | 80.94 | 70.24 |
| R×1, O×1 (Score↓) | 72.95 | 50.93 | 33.93 |
| R×1, O×1, D×5 (Date↓) | 75.06 | 32.46 | **-2.77** |

### Key Findings

1. **Outdated information is more dangerous than no information**: When only outdated information is retrieved, models generate overconfident but incorrect responses; whereas when no relevant information is retrieved, models instead exhibit appropriate uncertainty.
2. **The hazard of a single outdated information chunk far exceeds six distractors**: Introducing just one outdated passage causes the score to drop by over 24%, whereas six distractor passages only cause a less than 2% decrease.
3. **Ranking strategies have an enormous impact**: Qwen-7B scores up to 61.54% or as low as -2.77% depending on the specific ranking, showing a gap of over 60 percentage points.
4. **Temporal awareness is strongly correlated with RAG performance**: Models possessing both Current Awareness and Outdated Awareness perform the best (93.24 vs 24.70).

## Highlights & Insights

- **First to quantify the hazards of outdated information**: It does not merely "decrease accuracy", but rather leads models to generate harmful misinformation.
- **"Identifying obsolescence $\neq$ avoiding obsolescence"**: Even if models can recognize that information is outdated, they can still be misled by it (the harmful rate in Table 6 for A_O only remains higher than A_C only), indicating a need for dedicated alignment training.
- **Diff algorithm + LLM data construction workflow**: Balances efficiency and quality, proving more reliable than pure LLM-based pipelines.
- **Significant scale advantage**: 96K QA pairs + 219K documents, vastly surpassing RealtimeQA (2.3K) and FreshQA (600).

## Limitations & Future Work

1. Wikipedia snapshots have a fixed update frequency (monthly updates), failing to reflect real-time dynamics in fast-changing domains (e.g., the stock market).
2. QA pairs are primarily generated based on single articles, which makes it difficult to evaluate complex reasoning requiring multi-source information synthesis.
3. The sources of outdated information in the simulated search engine are single-track (historical versions of the same article), whereas sources of outdated information are much more diverse in reality.
4. Remedies to improve the actual robustness of RAG systems against outdated information have not been explored (this work focuses primarily on diagnosing the problem, rather than providing a solution).

## Related Work & Insights

- **Time-Sensitive QA**: Works like RealtimeQA and FreshQA heavily rely on human efforts, whereas HoH achieves automation and continuous updating.
- **CLARK-News**: The only prior work that logs historical changes of answers simultaneously, but it has a small scale and relies on human labor.
- **RAG Evaluation**: Existing RAG benchmarks (e.g., CRAG) focus primarily on retrieval quality, whereas HoH is the first to focus on the negative impacts of outdated information.
- **Insights**: Future RAG systems should incorporate temporal-awareness mechanisms into both the retrieval and generation phases, rather than relying solely on simple chronological sorting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — For pioneering the systematic study on the impact of outdated information on RAG, filling a critical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive experimental design, progressing step-by-step from retrieval to generation to temporal awareness with in-depth analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive examples, and a highly valuable comparison in Table 1.
- **Value**: ⭐⭐⭐⭐⭐ — Provides much-needed benchmarks and tools to the community, offering findings with practical guiding significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation](../../ICCV2025/information_retrieval/ocr_hinders_rag_evaluating_the_cascading_impact_of_ocr_on_retrieval-augmented_ge.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)

</div>

<!-- RELATED:END -->
