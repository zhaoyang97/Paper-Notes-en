---
title: >-
  [Paper Note] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation
description: >-
  [ACL 2026][Information Retrieval & RAG][PageRank] AuthorityBench constructs the first LLM "authority perception" benchmark using 10K web domains (PageRank ground truth) + 22K entities (Wikipedia cross-lingual sitelink ground truth) + 120 RAG questions. The study finds that ListJudge / PairJudge + PointScore yields the most accurate outputs, adding web text can degrade
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - PageRank
  - LLM-as-a-Judge
date: 2026-05-08
content_hash: 6daa103eaeff4eb0
---
# AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2603.25092](https://arxiv.org/abs/2603.25092)  
**Code**: [Trustworthy-Information-Access/AuthorityBench](https://github.com/Trustworthy-Information-Access/AuthorityBench)  
**Area**: Information Retrieval / RAG / Trustworthy AI  
**Keywords**: Authority Perception, RAG Filtering, PageRank, Entity Popularity, LLM-as-a-Judge

## TL;DR
AuthorityBench constructs the first LLM "authority perception" benchmark using 10K web domains (PageRank ground truth) + 22K entities (Wikipedia cross-lingual sitelink ground truth) + 120 RAG questions. The study finds that ListJudge / PairJudge + PointScore yields the most accurate outputs, adding web text can degrade performance, and utilizing authority signals for RAG filtering improves answer accuracy by up to 14 percentage points.

## Background & Motivation

**Background**: RAG has become a standard approach for reducing LLM hallucinations, but generation quality depends heavily on the reliability of retrieved sources. Current LLM-as-a-Judge research focuses on "relevance" and "utility," with minimal exploration into whether a source is "authoritative."

**Limitations of Prior Work**: Low-authority sources (e.g., personal blogs, marketing content) may provide medical advice contradicting established institutions like the Mayo Clinic. If RAG systems cannot distinguish between levels of authority and select answers based solely on linguistic fluency, they risk propagating misinformation. Existing methods either rely on manual credibility annotations or external scoring, lacking a systematic evaluation of "whether LLMs can perceive authority themselves."

**Key Challenge**: Authority is a "content-independent" source attribute (an anonymous blog should not be trusted even if it is beautifully written), yet LLMs are typically trained to "believe what they read." Whether LLMs can recognize authority based solely on URLs or entity names without content guidance is a critical prerequisite for RAG reliability.

**Goal**: (1) Provide a quantifiable definition of authority; (2) Evaluate the authority perception capabilities of mainstream LLMs under various prompt paradigms; (3) Verify whether "authority filtering" can improve the correctness of real-world RAG answers.

**Key Insight**: Authority is decomposed into two widely recognized proxies: Source Authority (using Web Graph PageRank) and Entity Authority (using the number of Wikipedia cross-lingual sitelinks). These proxies are objective, scalable, and content-independent, making them ideal tools for verifying "prior" perceptions in LLMs.

**Core Idea**: Construct three datasets (DomainAuth / EntityAuth / RAGAuth) + three LLM-as-a-Judge paradigms (Point / Pair / List) + two output formats (direct ranking vs. absolute scoring) to systematically investigate "can LLMs perceive authority, how should it be queried effectively, and is it useful for RAG."

## Method

### Overall Architecture

This is a benchmark and evaluation paper. The "method" consists of a three-stage pipeline: "data construction + evaluation protocol + RAG implementation experiments." The starting point is decomposing abstract "authority" into two objective proxies—PageRank for sources and sitelink counts for entities. Three datasets are created: DomainAuth (10K domains with Google Toolbar PageRank 0-9 ground truth), EntityAuth (22K entities spanning Basketball / Movies / Songs with sitelinks mapped to 0-9), and RAGAuth (120 yes/no questions with 10 mixed-authority documents, focusing on medical and current events topics prone to misinformation). Evaluation involves Listwise ranking (Spearman $\rho$ / Kendall $\tau$) + Pairwise preferences (paired accuracy on 5K easy and 5K hard pairs) + RAG answer accuracy, tested on Qwen3-8B/14B/32B, Llama-3-8B, and Llama-3.1-8B.

### Key Designs

**1. Three Judge Paradigms × Two Output Formats: Exhaustive Querying to Elicit Authority Perception**

The core problem is how to query LLMs to extract their authority priors. The authors decompose query methods into two orthogonal dimensions. Paradigms include PointJudge (absolute score for single input), PairJudge (pairwise comparison), and ListJudge (scoring a list). Output formats include direct ranking (PairRank / ListRank) or ranking based on absolute scores (PointScore, using BubbleSort or AverageScore). For fine-grained tasks, PairJudge uses an anchor-based approximation. AverageScore proved more robust than BubbleSort against non-transitive contradictions.

**2. Comparison with and without Web Text (Ctx vs w/o Ctx): Testing if "Authority Equals Text Quality"**

Each judge runs in two versions: one providing only the domain/entity name, and another including the web text snippet. This comparison addresses whether authority can be proxied by linguistic style. Results favor the former: List/Pair settings almost always drop in performance when text is added (e.g., Qwen3-8B PointScore drops from 71.35 to 63.91), except on hard pairs where text improves accuracy—suggesting text is a compensatory signal when structural signals are ambiguous rather than a default gain.

**3. Authority-Aware RAG Filtering Pipeline: Applying Authority Signals to End-to-End RAG**

The authors use the strongest protocol (ListJudge + PointScore) to score 10 documents and select top-$k$ ($k \in \{1, 3, 5\}$) based on three criteria: (a) Relevance Filter (query-relatedness), (b) Utility Filter (scoring based on pseudo-answers), and (c) Authority Filter (URL-only, without reading document content). Using only the URL for the Authority Filter ensures that any gain comes purely from "authority priors" rather than document content.

### Loss & Training

This is an evaluation-only paper. Three metric sets are used: Listwise (Spearman $\rho$ / Kendall $\tau$), Pairwise (paired-preference accuracy), and RAG-end (answer accuracy).

## Key Experimental Results

### Main Results: DomainAuth (Fine-grained 10-level, Spearman $\rho$ %)

| Model | Ctx | PointJudge | List+ListRank | List+PointScore | Pair+PairRank | Pair+PointScore |
|------|------|------|------|------|------|------|
| Qwen3-32B | w/o | 73.72 | 73.63 | 74.41 | 72.10 | **75.28** |
| Qwen3-32B | w/ | 73.57 | 55.85 | 63.10 | 66.32 | 69.93 |
| Qwen3-14B | w/o | 71.97 | 72.02 | 73.09 | 70.21 | **73.43** |
| Llama-3-8B | w/o | 63.87 | 57.53 | 66.08 | 61.05 | 64.83 |
| Qwen3-8B | w/o | 41.97 | 54.01 | 67.11 | 15.18 | **71.35** |

EntityAuth (Basketball, w/o text, Spearman $\rho$ %): Llama-3-8B PointScore reached **88.90**, and Qwen3-32B reached 85.94—generally 10+ points higher than DomainAuth, suggesting entity authority is easier for LLMs to perceive than web authority.

### RAG Answer Accuracy (RAGAuth, 120 Questions, %)

| Model | k | Relevance | Utility | **Authority** | w/o Filter |
|------|------|------|------|------|------|
| Qwen3-14B | 1 | 51.67 | 60.00 | **76.67** | 58.33 |
| Qwen3-14B | 3 | 45.00 | 66.67 | **75.00** | 58.33 |
| Qwen3-32B | 1 | 63.33 | 65.00 | **70.00** | 55.00 |
| Llama-3-8B | 3 | 41.67 | 52.50 | **64.17** | 50.83 |
| Llama-3.1-8B | 3 | 55.00 | 48.33 | **71.76** | 57.50 |

### Key Findings
- **ListJudge / PairJudge + PointScore is strongest**: Allowing the model to "see context" before providing an "absolute score" performs best. Isolated PointJudge signals are the weakest due to the lack of calibration across entries.
- **ListRank is inferior to PointScore**: Forcing a complete ranking leads to confusion during near-ties, whereas scores are smoother and more stable.
- **Model scale monotonically improves performance**: Spearman $\rho$ rises from Qwen3 8B to 32B, indicating authority perception depends on "world knowledge."
- **Adding web text usually degrades performance** (dropping 10-20 points in List/Pair settings), but **aids hard pairs** (increasing accuracy by 30+ points). This proves text acts as compensation rather than a default gain, validating the "Authority $\neq$ Style" hypothesis.
- **RAG experiments are pivotal**: Authority Filter outperforms Relevance/Utility, achieving 76.67% on Qwen3-14B (an 18.3 point gain over w/o Filter). This shows many RAG errors stem from "relevant but low-authority" sources.
- **Scoring Biases**: Models assign higher scores to `.gov` / `.edu` (+2.5 to +3.3 bias) and also overestimate social media (+1 to +2.4). LLMs have internalized TLDs as authority signals but remain overly sensitive to branding, showing potential for being misled.

## Highlights & Insights
- **First to isolate "Authority Perception" as an independent dimension of LLM capability**, which was previously buried under terms like "credibility" or "trustworthiness."
- **Empirical proof that "Authority $\neq$ Text Quality"** is a vital insight—the performance drop when adding text suggests LLMs can be misled by linguistic fluency, a vulnerability in RAG hallucination prevention.
- **ListJudge + PointScore offers high engineering efficiency** (scoring all documents in one forward pass) and serves as a directly reusable practice for RAG pipelines.
- **Bias analysis (overestimating .gov/.edu and social media)** provides clear targets for future calibration.

## Limitations & Future Work
- **Ceiling of PageRank as ground truth**: Niche but authoritative expert blogs may have low PR; small government sites may be undervalued due to few backlinks. Dynamic, topic-related authority measures are needed.
- **Small scale of RAGAuth (120 questions)**: Sufficient for demonstrating value but insufficient for cross-topic statistical analysis; expansion to 1K+ is planned.
- **Limited to 5 open-source LLMs**: Closed-source flagships (GPT-5, Claude 4) were not covered. Authority perception likely correlates strongly with RLHF data.
- **The nuance of "authoritative errors"** (e.g., a Reuters retraction) was not explored.

## Related Work & Insights
- **vs. Relevance Rankers (RankGPT / Setwise)**: They perform topical matching; this work targets source priors. Both are orthogonal and can be combined.
- **vs. Utility Rankers (Zhang 2024b)**: Utility looks at "can an answer be generated," while authority looks at "should it be trusted." RAGAuth shows they are complementary.
- **vs. Credibility-aware RAG (Pan 2024 / Deng 2025)**: Those rely on pre-labels or cross-consistency, while this work tests "intrinsic knowledge" and shows gains without extra training.
- **vs. Fake News Detection**: That field focuses on content veracity; this work isolates source authority as an independent prior, serving as a deployable intermediate signal in RAG pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ Authority as an independent dimension for systematic evaluation is a first in LLM-as-a-Judge literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 datasets, 5 models, 5 judge types, and end-to-end RAG; RAGAuth scale and lack of closed-source models are slight drawbacks.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 motivation is impactful; distinctions between ListRank and PointScore are clear; bias analysis in the appendix is detailed.
- Value: ⭐⭐⭐⭐ Provides a practical "cookbook" for RAG teams: filtering with ListJudge+PointScore can yield 10-20 point accuracy gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[ACL 2026\] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)
- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](../../NeurIPS2025/information_retrieval/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)

</div>

<!-- RELATED:END -->
