---
title: >-
  [Paper Note] HALoGEN: Fantastic LLM Hallucinations and Where to Find Them
description: >-
  [ACL 2025 (Outstanding Paper)][Hallucination Detection][Hallucination Benchmark] This paper proposes HALoGEN, a large-scale hallucination evaluation framework containing 10,923 prompts across 9 domains (including programming, scientific citation, translation, etc.), equipped with an atomic-level automated verifier. It systematically evaluates hallucinations on approximately 150,000 generation samples from 14 LLMs, discovering that even the best models can have hallucination r…
tags:
  - "ACL 2025 (Outstanding Paper)"
  - "Hallucination Detection"
  - "Hallucination Benchmark"
  - "Atomic Fact Verification"
  - "Error Taxonomy"
  - "Automated Verifier"
  - "Multi-Domain Evaluation"
date: 2026-05-08
content_hash: 35f67e1d7ebf8e37
---

# HALoGEN: Fantastic LLM Hallucinations and Where to Find Them

**Conference**: ACL 2025 (Outstanding Paper)  
**Code**: [Yes](https://github.com/google/halogen)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Benchmark, Atomic Fact Verification, Error Taxonomy, Automated Verifier, Multi-Domain Evaluation

## TL;DR

This paper proposes HALoGEN, a large-scale hallucination evaluation framework containing 10,923 prompts across 9 domains (including programming, scientific citation, translation, etc.), equipped with an atomic-level automated verifier. It systematically evaluates hallucinations on approximately 150,000 generation samples from 14 LLMs, discovering that even the best models can have hallucination rates of up to 86% in atomic facts within certain domains, and introduces a taxonomy of Type A/B/C errors.

## Background & Motivation

**Background**: Although LLMs can generate fluent and high-quality text, they frequently generate hallucinations that are inconsistent with world knowledge or input contexts. This problem severely limits the reliability of LLMs in high-risk areas such as medicine, law, and journalism.

**Limitations of Prior Work**: Existing hallucination evaluation methods face three major issues: (a) reliance on manual case-by-case verification, which is highly costly and time-consuming; (b) most only cover a narrow range of domains (e.g., summarization), lacking a systematic cross-domain evaluation; (c) lack of taxonomic analysis regarding the root causes of hallucinations.

**Key Challenge**: Hallucination is prevalent, yet standardized and scalable evaluation protocols are lacking—manual evaluation is unscalable, whereas simple automated metrics (e.g., BLEU/ROUGE) cannot capture factual errors.

**Goal**: To build a large-scale, multi-domain, automated hallucination evaluation framework and provide a meaningful hallucination taxonomy to facilitate understanding of the root causes.

**Key Insight**: Decompose LLM outputs into the smallest verifiable atomic units (atomic facts), and automatically verify each against high-quality knowledge sources across 9 different application domains.

**Core Idea**: Large-scale atomic-level hallucination detection combined with a three-category error taxonomy to reveal the ubiquity and deep-rooted causes of LLM hallucinations.

## Method

### Overall Architecture

HALoGEN consists of two core components: (1) A dataset of 10,923 generative prompts covering 9 domains—including programming (code generation), scientific citation (literature retrieval), summarization, question answering, entity attributes, biography generation, geographical facts, medical terminology, and multilingual translation. (2) High-precision automated verifiers tailored for each domain, which decompose LLM generations into atomic units and verify them individually against trusted knowledge sources.

### Key Designs

1. **Atomic Decomposition & Verification**: Rather than evaluating global generation quality, outputs are deconstructed into the smallest verifiable factual units—such as a specific number, an entity relation, or a code execution output. Each atomic fact is verified independently against authorized knowledge sources (e.g., Wikipedia, PubMed, code execution results). This approach is more granular than overall scoring and can identify generations that are "partially correct but contain critical errors."
2. **Domain-Specific Verification Strategies for 9 Domains**: Each domain employs a domain-specific verification pipeline—programming tasks are verified via code execution, scientific citations are cross-checked with literature databases, summarization is aligned with source documents, and geographic facts are verified through knowledge base queries. The verifiers are calibrated via manual sampling to ensure high precision.
3. **Error Taxonomy (Three Error Types)**:

    - Type A (Faulty Recall): Correct information is present in the training data, but the model fails to recall/generate it correctly—reflecting limitations in model capability.
    - Type B (Faulty Knowledge): The training data itself contains incorrect or outdated information—reflecting data quality issues.
    - Type C (Fabrication): The model generates content without any factual basis—reflecting uncontrolled "creativity" of the model.
4. **Large-Scale Systematic Evaluation**: Approximately 150,000 responses from 14 LLMs (including GPT-4, Claude, LLaMA, Gemini, etc.) were generated and comprehensively evaluated across 9 domains.

### Loss & Training

As an evaluation framework, this work does not involve model training. The automated verifiers are constructed based on rules and existing tools (literature search APIs, code executors, knowledge base queries, etc.) and do not require training additional neural network verifiers.

## Key Experimental Results

### Main Results

Atomic-level hallucination rates of 14 LLMs across different domains:

| Evaluation Dimension | Findings |
|---------|------|
| Worst Domain for Best Model | Even the strongest model exhibits hallucination rates of up to 86% in atomic facts within certain domains. |
| Scientific Citation | Citation generation is one of the worst-performing domains for hallucinations, featuring a massive fabrication of non-existent papers. |
| Programming Tasks | Code functional correctness can be directly verified via execution, leading to relatively manageable hallucination rates. |
| Biography Generation | Factual errors about public figures are common—with frequent mistakes in dates, positions, achievements, etc. |
| Model Comparison | Larger/newer models generally exhibit lower overall hallucination rates, but no single model performs optimally across all domains. |

### Ablation Study

Evaluation of verifier precision:

| Evaluation Dimension | Results |
|---------|------|
| Automated Verification vs. Human Evaluation | Automated verifiers show high alignment with human evaluation across multiple domains (high precision). |
| Atomic Decomposition Quality | The granularity of decomposition directly impacts final hallucination statistics—too coarse leads to missed detections, and too fine leads to overcounting. |
| Knowledge Source Coverage | The completeness and timeliness of knowledge sources directly affect verification accuracy. |

### Key Findings

- **Hallucination is Ubiquitous**: All 14 models exhibit varying degrees of hallucination across all 9 domains; no model is completely "hallucination-free".
- **Huge Domain Discrepancy**: The hallucination rate of a single model can range from below 10% to above 80% depending on the domain.
- **Type C (Fabrication) is Most Severe in Scientific Citation**: Models generate plausible-looking yet completely fabricated paper titles, authors, and journals.
- **Model Scale and Hallucination**: Larger models suffer less from overall hallucinations, but the improvement scale varies significantly across different domains.
- **Atomic-level Evaluation is More Sensitive**: Traditional holistic scoring can easily miss factual errors subtly hidden within fluent text.

## Highlights & Insights

- **Unprecedented Scale**: 10,923 prompts × 14 models ≈ 150K generations, constituting the largest systematic study of hallucinations to date.
- **High Utility of the Taxonomy**: Distinguishing "model capability issues" (Type A), "data quality issues" (Type B), and "pure fabrication" (Type C) provides actionable insights for targeted improvements.
- **Atomic Verification** uncovers hidden errors missed by holistic scoring—a fluent biography might be 80% correct but contain entirely wrong key dates.
- **Outstanding Paper Quality**: Offers infrastructure-level contributions to the field of hallucination research.
- **Cross-Domain Analysis Reveals Domain Specificity**: Hallucination levels cannot be simply summarized by a single, monolithic metric.

## Limitations & Future Work

- Although highly precise, automated verifiers are not 100% accurate; defining "hallucinations" in subjective domains (such as creative writing or opinion expression) remains difficult.
- Knowledge sources themselves can be outdated or incomplete, leading to false positives where correct facts are labeled as hallucinations.
- The boundaries of the Type A/B/C taxonomy can be blurry in practice—some errors are hard to attribute strictly to either model capability or data quality.
- The current version primarily covers English, leaving multilingual hallucination evaluation to be expanded in future work.
- The consistency of the atomic decomposition process may vary across different domains.
- It lacks evaluations on the effectiveness of hallucination-mitigation methods (such as RAG and alignment tuning).

## Related Work & Insights

- Compared to FActScore (atomic fact scoring in biography generation), HALoGEN generalizes atomic-level evaluation to 9 distinct domains.
- Complementary to TruthfulQA (multiple-choice hallucination QA)—HALoGEN evaluates open-ended generation rather than multiple-choice tasks.
- Directly inspires RAG research—revealing the domain distribution of hallucinations can guide the selection of retrieved knowledge sources.
- The three-category error taxonomy guides model alignment training—different types of hallucinations necessitate distinct mitigation strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ Atomic decomposition + the three-category taxonomy represent novel contributions, though substantial work already exists in the domain of hallucination evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Unprecedented scale—14 models × 9 domains × 150K generations, offering exceptionally comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined classification, recognized as an ACL Outstanding Paper.
- Value: ⭐⭐⭐⭐⭐ Provides foundational evaluation tools and benchmark references for hallucination research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hallucination Begins Where Saliency Drops](../../ICLR2026/hallucination/hallucination_begins_where_saliency_drops.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](../../CVPR2026/hallucination/tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](hallulens_llm_hallucination_benchmark.md)
- [\[ACL 2025\] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge](correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)
- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](../../NeurIPS2025/hallucination/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)

</div>

<!-- RELATED:END -->
