---
title: >-
  [Paper Note] Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs
description: >-
  [ACL 2025][LLM Evaluation][Patent Generation] This work constructs the Pap2Pat benchmark containing 1.8k patent-paper pairs, proposes COPGen, an outline-guided chunked patent description generation method, and designs NLI-based factuality, coverage, and style evaluation metrics to systematically assess the capabilities and limitations of modern LLMs in generating ultra-long patent documents.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Patent Generation"
  - "Long-Document Generation"
  - "Outline Guidance"
  - "Patent-Paper Pairs"
  - "Chunked Generation"
date: 2026-05-08
content_hash: bbe73a4b7c51dacd
---

# Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs

**Conference**: ACL 2025  
**arXiv**: [2410.07009](https://arxiv.org/abs/2410.07009)  
**Code**: [Yes](https://github.com/boschresearch/Pap2Pat)  
**Area**: LLM Evaluation  
**Keywords**: Patent Generation, Long-Document Generation, Outline Guidance, Patent-Paper Pairs, Chunked Generation

## TL;DR

This work constructs the Pap2Pat benchmark containing 1.8k patent-paper pairs, proposes COPGen, an outline-guided chunked patent description generation method, and designs NLI-based factuality, coverage, and style evaluation metrics to systematically assess the capabilities and limitations of modern LLMs in generating ultra-long patent documents.

## Background & Motivation

Patent application is a lengthy and expensive process requiring deep technical knowledge and patent law expertise. Although NLP has been applied to tasks such as patent retrieval and patent landscaping, the automated drafting of patent documents still relies heavily on manual effort. Within a patent, the **description section on average accounts for over 90% of the length** (covering sections like technical field, background, summary, and detailed description). Automated support for generating this section would yield the greatest productivity gains, but also faces the most significant challenges:

- **Extreme document length**: The patent descriptions in Pap2Pat average 18k tokens and can exceed 180k tokens, which is far beyond the generation limits of current LLMs.
- **Limitations of prior work**: Most patent generation research only focuses on titles, abstracts, and claims, ignoring the description section which constitutes the majority of the content. Additionally, existing task setups are unrealistic, and open benchmarks are lacking.
- **Data acquisition difficulties**: Invention reports (IR) typically submitted by inventors are confidential. However, in research labs, pre-publication papers are often used as invention reports, yielding "patent-paper pairs", which makes the construction of open datasets feasible.
- **Lack of evaluation standards**: Patent texts are long and highly technical. Traditional text similarity metrics (such as ROUGE) underperform on long documents, and there is a lack of dedicated evaluation methods for factuality, coverage, and linguistic style.

## Method

### Overall Architecture

The workflow consists of three major components: (1) building the Pap2Pat benchmark dataset, (2) the COPGen chunked outline-guided generation method, and (3) designing patent generation evaluation metrics.

### Key Designs

1. **Pap2Pat Dataset Construction**:

    - Starting from 6.7 million USPTO patent applications, SemOpenAlex was used to query papers with overlapping authors and relevant publication/filing dates.
    - Through multi-level filtering including title/abstract term overlap, candidate uniqueness, and open licenses, the initial candidate pool of 930k was filtered down to 1.8k high-quality patent-paper pairs.
    - Human verification accuracy: Out of 60 randomly sampled pairs, 55/60 (91.7%) were exact matches.
    - LLama-3 70B was used to automatically generate outlines of three granularities (short/medium/long, averaging 37/74/150 bullet points) from raw patents, with outline points averaging 5.4 words.
    - Data split: train 1000, val 242, test 500, plus an adversarial/non-leakage test set (nc-test) of 71 pairs (2024 patents to prevent data leakage).

2. **COPGen: Chunked Outline-Guided Patent Generation**:

    - **Core Idea**: Segment the long outline into multiple "chunks", generate a patent text segment for each chunk independently, and concatenate them in the end.
    - **Token Allocation & Chunking**: By default, each chunk is allocated 2k instruction tokens + 3k paper context tokens + 2k patent output tokens. The number of outline points included in each chunk is determined by the average characters per outline point.
    - **Paper Context Selection**: BM25 is used to retrieve the most relevant paragraphs from the paper using the current chunk's outline points as queries. The paper abstract and all section headings are always included, with the remaining budget filled with retrieved paragraphs ordered by relevance.
    - **Length Control Mechanism**: Based on the findings of Bai et al. that LLM generation length is only controllable when keeping it short, the overall output length is precisely controlled by reducing each chunk's output token allocation (and thereby increasing the total number of chunks). After calibration, a setting of 400 tokens/chunk is used to match the average length of the reference patents.
    - **Global Context**: The generation of each chunk also incorporates the outlines of previous chunks to maintain global document structure awareness.

3. **Evaluation Metric Design**:

    - **Factuality**: Based on the NLI-based SCALE metric, calculating the entailment score of the reference patent (and paper) on the generated text (Ref $\rightarrow$ Gen, Ref+Pap $\rightarrow$ Gen).
    - **Coverage**: Calculated in reverse, using the generated text as the premise and the reference patent as the hypothesis (Gen $\rightarrow$ Ref).
    - **Linguistic Style**: Measured using n-gram profile similarity (1-4 gram) + StyloMetrix linguistic features (196 rule-based detections) to quantify similarity to patent style.
    - **Repetition**: Measured via sliding window Repetition Rate (RR) to distinguish between reasonable legal language repetition and non-sensical looping generation.

### Loss & Training

- COPGen itself does not involve training; it is an inference-time framework.
- The experiments also evaluated supervised fine-tuning (SFT) of Llama-3 8B on the Pap2Pat training set, exploring the effects of SFT on style and factuality.

## Key Experimental Results

### Main Results: Performance of Different Models and Methods

| Method | Tokens | Coverage (Gen $\rightarrow$ Ref) | Factuality (Ref $\rightarrow$ Gen) | Style |
|------|--------|-------------------|---------------------|-------|
| Reference Patent (Upper Bound) | 18.1k (100%) | 88.6 | 88.5 | 100 |
| Original Paper | 8.1k (45%) | 44.8 | 46.5 | 47.2 |
| Qwen2-72B (Single Call) | 2.8k (16%) | 40.3 | 65.8 | 39.6 |
| COPGen + Llama-3 8B | 9.6k (53%) | 40.3 | 60.8 | 43.2 |
| COPGen + Llama-3 70B | 6.1k (34%) | 42.7 | 64.5 | 49.5 |
| COPGen + Qwen2-72B | 8.1k (45%) | 44.1 | 62.5 | 47.5 |
| COPGen + Qwen2-72B (Calibrated Length) | **18.1k (100%)** | **46.8** | 59.7 | 47.8 |
| COPGen + Llama-3 8B SFT | 27.5k (152%) | 42.0 | 49.3 | **59.4** |

### Ablation Study: Impact of Input Components

| Variant | Coverage | Factuality (Ref+Pap $\rightarrow$ Gen) |
|------|----------|-------------------------|
| COPGen + Qwen2-72B (Full) | 44.1 | 67.9 |
| Without Paper Input | 38.9 | 66.6 |
| Without Outline Input | 34.9 | 75.3 |

### Key Findings

1. **COPGen significantly improves coverage**: For the same model (Qwen2-72B), coverage increases from 40.3 with a single call to 44.1 with chunked generation, indicating that the chunking strategy effectively mitigates the difficulties of long-text generation in LLMs.
2. **Effective length control**: By calibrating token allocation, the output length can be precisely matched to the reference patent length (18.1k), and the coverage is further improved to 46.8.
3. **Fine-tuning as a double-edged sword**: SFT boosts the style score from 43.2 to 59.4 (making it sound more like a patent), but factuality plunges from 60.8 to 49.3 due to increased hallucination. The repetition rate also increases significantly.
4. **Paper is a key information source**: Removing paper input leads to a 5.2-point drop in coverage; removing the outline leads to an even larger drop (9.2 points), while factuality increases (the model becomes more "conservative" but with less content).
5. **Only 8.3% 4-gram overlap**: Although patents and papers describe the same invention, their linguistic styles differ drastically, making the task far from simple paraphrasing.

## Highlights & Insights

- **High methodological value in dataset construction**: Effectively utilizing the under-exploited phenomenon in academia/industry that "papers are often used as invention reports" as a data resource. The construction method is generalizable to other domains.
- **Elegant and effective chunking strategy**: Without the need for specialized training, COPGen achieves controllable ultra-long document generation through engineered context management and length control, supporting parallel generation of individual chunks.
- **Meticulous evaluation metric design**: Decoupling factuality, coverage, and style into independent metrics provides far more insights than a single ROUGE score. The BM25 pre-screening + sampling strategy of SCALE also makes it scalable to ultra-long documents.
- **Insight on SFT-induced hallucinations**: This provides an important caveat—fine-tuning LLMs in specialized domains may capture stylistic features at the expense of factual accuracy.

## Limitations & Future Work

- The dataset size is constrained by paper open access licenses (only 1.8k pairs); expanding it requires broader open access in academic publishing.
- Generating chunks independently in COPGen might lead to a lack of coherence across blocks. Future work could introduce inter-chunk dependency modeling.
- Evaluation still relies heavily on automatic metrics. Professional human expert evaluation is highly costly, and the human case study in this paper only serves as a small-scale validation.
- Only open-source LLMs were evaluated, excluding closed-source models like GPT-4 (primarily for reproducibility reasons).
- Outlines were automatically generated from reference patents rather than created by actual patent attorneys, resulting in a gap from real-world applications.

## Related Work & Insights

This work is closely related to three research directions: long-document generation (Sun et al. 2022; Wang et al. 2024d), outline-guided generation (Shao et al. 2024), and patent NLP (Christofidellis et al. 2022; Wang et al. 2024c). The key insight is that the "divide-and-conquer" strategy for long-document generation (planning an outline first, then generating chunk by chunk) might be the optimal practice under the architectural constraints of current LLMs. This paradigm can be generalized to other long-form text generation tasks such as academic papers and technical reports.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dataset construction methodology is novel (PPP matching pipeline). Although the chunk-wise generation framework is simple, it is highly targeted. The evaluation metric design is creative.  
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive and multi-dimensional evaluation, including multi-model comparison, ablation study, length control analysis, SFT analysis, and human evaluation.  
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Exceptionally clear structure; the flow from motivation to method and evaluation is logical; excellent illustration/table design.  
- **Value**: ⭐⭐⭐⭐ — Fills the benchmark gap in automated patent description generation. Open-sourcing the data and code directly benefits legal AI and the broader long-document generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)
- [\[ACL 2025\] Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics](calibration_confidence_text_gen.md)
- [\[ACL 2026\] Illusions of the Gold Standard: A Large-scale Analysis of Human Evaluation Protocols for Long-form Text Generation](../../ACL2026/llm_evaluation/illusions_of_the_gold_standard_a_large-scale_analysis_of_human_evaluation_protoc.md)
- [\[ICLR 2026\] ExpertLongBench: Benchmarking Language Models on Expert-Level Long-Form Generation Tasks with Structured Checklists](../../ICLR2026/llm_evaluation/expertlongbench_benchmarking_language_models_on_expert-level_long-form_generatio.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)

</div>

<!-- RELATED:END -->
