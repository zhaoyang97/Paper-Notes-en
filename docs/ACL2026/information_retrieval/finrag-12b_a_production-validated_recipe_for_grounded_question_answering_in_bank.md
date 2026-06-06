---
title: >-
  [Paper Note] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking
description: >-
  [ACL 2026][Information Retrieval & RAG][Financial QA] Based on Gemma 3 12B-IT, the Kasisto team used an efficient 143M-token data recipe (LLM-as-Judge filtering + citation labeling + 22% unanswerable samples + two-stage…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Financial QA"
  - "Citation Grounding"
  - "Calibrated Refusal"
  - "Curriculum Learning"
  - "W4A16 Quantization"
date: 2026-05-08
content_hash: e130afc6fa974ccb
---

# FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking

**Conference**: ACL 2026  
**arXiv**: [2605.05482](https://arxiv.org/abs/2605.05482)  
**Code**: Not open-sourced (Stage 1 reproducible via RAG-v1)  
**Area**: RAG / Finance / Model Compression  
**Keywords**: Financial QA, Citation Grounding, Calibrated Refusal, Curriculum Learning, W4A16 Quantization

## TL;DR
Based on Gemma 3 12B-IT, the Kasisto team used an efficient 143M-token data recipe (LLM-as-Judge filtering + citation labeling + 22% unanswerable samples + two-stage curriculum) to train FinRAG-12B. Compressed to 8.4 GB via W4A16 quantization for single-card deployment, it exceeds GPT-4.1 in both answer quality (JudgeLM 6.21) and citation quality (73.1). Its refusal rate (12%) is balanced between the base model's unsafe 4.3% and GPT-4.1's over-refusal of 20.2%. Following deployment across 40+ financial institutions, query resolution significantly improved by +7.1pp ($p<0.001$), with latency and costs reduced by 3–5× and 20–50×, respectively, compared to GPT-4.1.

## Background & Motivation

**Background**: The primary obstacle for adopting LLMs in banking is compliance—answers must be verifiable, traceable, and free of hallucinations. RAG is required to handle daily fluctuations in interest rates, account balances, and policies. While the BloombergGPT (50B/346B tokens) route proved scale works, it is closed-source and lacks data on latency and cost. FinGPT uses LoRA but focuses on classification rather than generative RAG.

**Limitations of Prior Work**: (a) General instruction-tuned LLMs are often sycophantic—answering even when retrieval context is insufficient (Sharma 2024); base Gemma 3 12B says "I don't know" (IDK) for only 4.3% of queries, mostly hallucinating. (b) GPT-4.1 is extremely conservative—20.2% over-refusal, rejecting answerable queries. (c) RAG models suffer from "lost in the middle" position bias (Liu 2024). (d) Mixed training on all data causes catastrophic performance (JudgeLM 3.28, IDK jumps to 46.5%), while training on only external or internal data has limitations. (e) Deployment costs—GPT-4.1 at $0.02–$0.05 per query is unaffordable for 40 institutions with 10K daily queries.

**Key Challenge**: There is a conflict between five dimensions: grounded answer quality, refusal calibration, citation quality, latency, and cost. Direct SFT easily overfits one dimension at the expense of others, yet banking scenarios require all five. Additionally, compliance restrictions prevent the direct use of large amounts of real user data (containing PII) for training.

**Goal**: (i) A data-efficient (≤200M tokens) training recipe to simultaneously optimize answer quality, citations, and calibrated refusal; (ii) A service solution deployable on a single card with costs ≤$0.005/query; (iii) An end-to-end production methodology from data curation to quantized serving; (iv) Validation in real production environments (40+ institutions).

**Key Insight**: The authors found that "data quality > data scale" (as proven by Phi-3 and LIMA) + "curriculum learning to resolve multi-objective conflicts" + "controlled negative sample ratios to calibrate refusal" can be combined into a single recipe.

**Core Idea**: Use LLM-as-Judge filtering + multi-stage synthesis + curriculum learning to optimize answer/citation/refusal on 143M tokens. Refusal is calibrated by sweeping for the optimal negative sample ratio (22% unanswerable). Finally, W4A16 quantization preserves >99% of citation quality.

## Method

### Overall Architecture
The end-to-end recipe consists of five components:
(1) **Data Pipeline**: Merges RAG-v1 (43,581 samples, filtered by JudgeLM < 5) + Synthetic SEC QA (16,773 samples) + CommonCrawl financial subset (20,499 samples) + internal refusal calibration (17,795 samples), totaling 98,648 samples / 143M tokens.
(2) **5-Step Synthetic SEC QA**: Chunking (400–600 tokens) → 4-level difficulty generation → Style-conditional rewriting → Grounded answer + citations → Injection of 3–7 distractor passages.
(3) **Position Bias Mitigation**: The position of gold evidence among distractors is sampled via a hybrid "right-trapezoid harmonic decay distribution": $P(X=x)=\frac{1}{N-K_{\min}+1}\sum_{K=\max(x,K_{\min})}^{N}\frac{1}{K}$.
(4) **Two-Stage Curriculum**: Stage 1 uses RAG-v1+SEC synthetic data (lr=$1\times10^{-6}$ cosine) to learn citation norms. Stage 2 uses CC+internal data (lr=$5\times10^{-6}$ linear) to calibrate refusal and real-world style.
(5) **Quantized Deployment**: LoRA ($r=64, \alpha=256$) on all attention+MLP layers; SmoothQuant W4A16 compresses 24GB to 8.4GB, achieving TTFT 0.14s / TTC 0.57s.

### Key Designs

1. **5-Step Synthetic SEC QA Pipeline (vs. Single-Shot Generation)**:
    - **Function**: Generates synthetic training data from 10-K/10-Q SEC reports that is both "grounded" and "aligned with real query distributions."
    - **Mechanism**: Chunking → 4 difficulty levels (easy/medium/hard/expert, with higher weights for easy/medium) → Few-shot style rewriting (fragmented/how-do-I/what-is style, log-normal length, adjustable formality; e.g., changing "What is the minimum credit score required for mortgage approval?" to "min credit score for mortgage") → Grounded answer with citation → Injection of 3–7 topically-similar distractors with random gold evidence placement. Compared to single-shot, question-type JS divergence dropped from 0.434 to 0.041 (10× improvement), and length dropped from 19.55 words to 8.85 (ground truth: 9.91).
    - **Design Motivation**: Single-shot LLM generation is often verbose and long, differing significantly from "fragmented, colloquial" user queries. Direct training would overfit to the synthetic distribution. The authors prioritize "distribution alignment" over pure volume.

2. **22% Unanswerable Samples + Curriculum to Calibrate Refusal**:
    - **Function**: Enables the model to proactively state "I don't know" when evidence is insufficient, without over-refusing.
    - **Mechanism**: The negative sample ratio was swept from 10%–30% in 2pp increments. 22% was found to be the "sweet spot"—ratios above 26% led to sharp recall drops (over-refusal), while lower ratios were dominated by sycophancy (high false positives). Mixed training on all data caused IDK to spike to 46.5% with only 39% being true negatives. The two-stage curriculum (Stage 1: external citation data; Stage 2: internal refusal calibration) adjusted IDK to 13.2% with a TN accuracy of 56%.
    - **Design Motivation**: Refusal calibration cannot be solved by single-objective SFT as it inherently conflicts with "answer quality." Separating these into curriculum stages addresses this directly; a ratio sweep identifies the Pareto optimum.

3. **W4A16 Quantization Preserving >99% Citation Quality**:
    - **Function**: Compresses the 24GB model to 8.4GB (2.86×) to enable deployment on a single GPU (RTX 6000 Ada), reducing cost to ~$0.001 per query.
    - **Mechanism**: SmoothQuant W4A16 (4-bit weights + 16-bit activations). Experiments show citation quality remains virtually unchanged (>99% retention), with TTFT 0.14s / TTC 0.57s—TTC is 3.2× faster than the prior production model (Mistral-7B-based FinRAG-v3).
    - **Design Motivation**: Financial RAG requires real-time response and manageable hardware costs for multi-institution deployment. W4A16 is the most aggressive quantization that preserves generation quality. The authors notably validated that grounded generation does not degrade under low-bit quantization.

### Loss & Training
LoRA ($r=64, \alpha=256$, dropout 0.05) applied to all attention and MLP layers. 8-bit AdamW with lr $2\times10^{-5}$, per-device batch size 4 + grad accum 4 (effective 16), max seq length 16,384, early stopping with patience 5. Total training: 1,400 steps / ~360 GPU-hours / 8× RTX A6000 / Cost ~ $1,800.

## Key Experimental Results

### Main Results: 258 Banking QA Test Set (3 Financial Institutions)

| Model | JudgeLM (1–10) | Citation Q (0–100) | QA F1 | IDK% |
|------|---------------|--------------------|-------|------|
| Gemma 3 12B (base) | 5.70 | 80.2 | 0.964 | 4.3 (under-refuse) |
| GPT-4.1 (API) | 5.72 | 70.8 | 0.900 | 20.2 (over-refuse) |
| **FinRAG-12B** | **6.21** | **73.1** | 0.936 | **12.0 (calibrated)** |

Public Benchmark FinanceBench (150 SEC questions):

| Model | FinanceBench F1 |
|------|-----------------|
| Gemma 3 12B (base) | 0.249 |
| GPT-4.1 | 0.238 |
| **FinRAG-12B** | **0.284** (Citation Rate 97.3%) |

### Ablation Study: Curriculum / Data Strategy (258 Banking QA)

| Strategy | JudgeLM | QA F1 | Cit. Q | IDK% | TN% (Refusal Accuracy) |
|------|---------|-------|--------|------|---------------------|
| External only (RAG-v1+SEC synthesis) | 5.72 | 0.972 | 76.1 | 0.4 | 0 (Catastrophically low) |
| Internal only (CC+internal) | 5.62 | 0.913 | 69.2 | 17.4 | 53 |
| Combined (Full mixing) | 3.28 | 0.706 | 51.2 | 46.5 | 39 (Collapse) |
| **Curriculum (staged)** | **5.91** | 0.938 | **74.7** | 13.2 | **56** |

### Key Findings
- **Mixed Training ≠ More Data is Better**: Mixing all data caused JudgeLM to drop from 5.91 to 3.28 and IDK to spike to 46.5%. Multi-objective conflicts must be isolated via curriculum learning.
- **The 4.3% IDK of the Base Model is Dangerous**: The low refusal rate means it attempts to answer 95.7% of unanswerable queries, which is unacceptable in regulatory contexts.
- **The 20.2% IDK of GPT-4.1 is Wasteful**: Over 20% of queries are rejected, hurting user experience and business value. FinRAG-12B's 12% is the calibrated sweet spot.
- **Production Metrics +7.1pp Resolution Rate**: Over 3,297 real queries in 7 months, resolution rose from 77.4% to 84.5% ($\chi^2=24.4, p<0.001$). User satisfaction rose +3.4pp (not significant, $p=0.26$). Attribution shows gains came from "more queries being resolved" rather than "better individual answers"—revealing the true value driver for production RAG.
- **W4A16 No Dropped Points**: 4-bit quantization preserves >99% citation quality, proving grounded generation's robustness.
- **5-Step Synthesis vs. Single-Shot**: Improved question-type JS divergence by 10× and aligned length (8.85 words vs 9.91 ground truth). Distribution alignment is more critical than raw generation quality.

## Highlights & Insights
- **The "data-efficient + curriculum" combo is practical**: Training a grounded QA model that beats GPT-4.1 on 143M tokens proves "recipe > scale" in vertical domains. Stage 1 (open-source data) and Stage 2 (private data) design addresses both reproducibility and data privacy.
- **Negative sample ratio sweeping is a transferable hack**: Refusal calibration is a Pareto optimization problem. A simple grid search found 22%—a method applicable to any SFT pipeline.
- **Trapezoid harmonic decay mitigates lost-in-the-middle**: Using a hybrid distribution to randomize gold evidence placement forces the model to stop relying on positional heuristics.
- **Satisfaction Attribution Analysis**: The authors decomposed gains to "distribution shift" rather than individual quality—an honest causal attribution rare in industry papers.
- **W4A16 + LoRA + Gemma 3 is a rational recipe for 2025 financial LLM deployment**: This can be directly applied to legal, medical, or customer service domains.

## Limitations & Future Work
- Small evaluation set: 258 banking QAs from 3 institutions, biased toward retail banking. Corporate, investment banking, and insurance are not covered.
- Internal data (Stage 2) is not open-source due to compliance, limiting full academic reproduction.
- Refusal evaluation only identifies explicit "I don't know" variants, failing to model hedged or partially uncertain expressions (e.g., "It might be..."), possibly underestimating abstention.
- The +3.4pp satisfaction gain was not significant ($p=0.26$), potentially requiring more data or time to confirm.
- Synthetic data depends on teacher models (like GPTs), risking "model collapse" over the long term.
- While covering 47 institutions, they are all within US retail banking; global regulatory systems (EU, APAC) are unverified.

## Related Work & Insights
- **vs. BloombergGPT (Wu 2023)**: 50B / 346B tokens / closed-source / unknown costs. FinRAG-12B (12B / 143M tokens) proves "data quality" is more reasonable in vertical domains than "brute-force scaling."
- **vs. FinGPT (Yang 2023)**: Both use LoRA, but FinGPT is limited to classification whereas FinRAG-12B handles generative RAG with citations.
- **vs. Phi-3 (Abdin 2024) / LIMA (Zhou 2023)**: Shares the "small but high quality" philosophy but applies it to grounded RAG and refusal.
- **vs. RARR (Gao 2023) / Self-RAG (Asai 2024)**: These optimize grounding at inference-time; FinRAG-12B makes the model inherently grounded at training-time, reducing latency.
- **Insight**: (a) Any vertical LLM deployment should sweep negative ratios to calibrate refusal; (b) Prioritize curriculum over mixing for conflicting objectives; (c) Production models should report business metrics like "resolution rate" rather than just benchmark F1; (d) W4A16 is the standard for 12B-class single-card deployment.

## Rating
- Novelty: ⭐⭐⭐ Individual techniques are not new; however, the "pipeline + curriculum + 22% negative + W4A16 + production validation" as a complete financial recipe is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ Combination of internal/public benchmarks, detailed ablation, 7 months of production data, and attribution analysis. Test set (258) is somewhat small.
- Writing Quality: ⭐⭐⭐⭐ Exceptionally clear for an industry paper; steps are reproducible, and causal analysis is deep.
- Value: ⭐⭐⭐⭐⭐ Production-validated across 40+ institutions; provides an end-to-end recipe for grounded vertical LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](../../AAAI2026/information_retrieval/mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)

</div>

<!-- RELATED:END -->
