---
title: >-
  [Paper Note] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation
description: >-
  [ACL 2026][Information Retrieval & RAG][Authority Perception] AuthorityBench constructs the first benchmark for LLM "authority perception" using 10K web domains (PageRank ground truth)…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Authority Perception"
  - "RAG Filtering"
  - "PageRank"
  - "Entity Popularity"
  - "LLM-as-a-Judge"
date: 2026-05-08
content_hash: 0298a50c67c63a33
---

# AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2603.25092](https://arxiv.org/abs/2603.25092)  
**Code**: [Trustworthy-Information-Access/AuthorityBench](https://github.com/Trustworthy-Information-Access/AuthorityBench)  
**Area**: Information Retrieval / RAG / Trustworthy AI  
**Keywords**: Authority Perception, RAG Filtering, PageRank, Entity Popularity, LLM-as-a-Judge

## TL;DR
AuthorityBench constructs the first benchmark for LLM "authority perception" using 10K web domains (PageRank ground truth), 22K entities (Wikipedia cross-lingual sitelink ground truth), and 120 RAG questions. The study finds that ListJudge / PairJudge + PointScore paradigms are the most accurate. Interestingly, adding webpage text often degrades performance, and utilizing authority signals for RAG filtering can improve answer accuracy by up to 14 percentage points.

## Background & Motivation

**Background**: RAG has become a standard for reducing LLM hallucinations, but generation quality depends heavily on the reliability of retrieved sources. Current LLM-as-a-Judge research primarily focuses on "relevance" and "utility," with minimal exploration into "source authority."

**Limitations of Prior Work**: Low-authority sources (e.g., personal blogs, marketing content) may provide medical advice that contradicts reputable sources like the Mayo Clinic. If RAG systems cannot discern authority and select answers based solely on linguistic fluency, they risk disseminating misinformation. Existing methods either rely on manual credibility annotations or external scoring, lacking a systemic evaluation of whether LLMs can perceive authority inherently.

**Key Challenge**: Authority is a "content-independent" source attribute (an anonymous blog should not be trusted even if the writing is professional), yet LLMs are typically trained to "trust what they read." Whether an LLM can identify authority based on URLs or entity names without content guidance is a critical prerequisite for RAG reliability.

**Goal**: (1) Provide a quantifiable definition of authority; (2) Evaluate the authority perception capabilities of mainstream LLMs under various prompt paradigms; (3) Verify whether "authority filtering" can improve answer accuracy in real-world RAG scenarios.

**Key Insight**: Authority is decomposed into two recognized proxies: source authority (Google PageRank) and entity authority (Wikipedia cross-lingual sitelink counts). These proxies are objective, scalable, and content-independent, making them ideal tools for verifying the "prior" perception of LLMs.

**Core Idea**: The authors construct three datasets (DomainAuth / EntityAuth / RAGAuth), test three LLM-as-a-Judge paradigms (Point / Pair / List), and evaluate two output formats (direct ranking vs. absolute scoring) to systematically investigate if LLMs perceive authority, the most effective way to query them, and the utility for RAG.

## Method
> This is a benchmark and evaluation paper. The following sections detail the data construction, evaluation protocols, and RAG experiments.

### Overall Architecture
- **Data**: DomainAuth (10K domains + PageRank 0-9 ground truth, including 10-level fine-grained and 5-level coarse-grained labels); EntityAuth (22K entities across Basketball / Movies / Songs, with log-binned Wikipedia sitelink counts mapped to 0-9); RAGAuth (120 yes/no questions × 10 mixed-authority documents, specifically targeting topics prone to misinformation, such as healthcare and current events).
- **Evaluation**: Listwise (ranking 10 items, measured by Spearman $\rho$ / Kendall $\tau$) + Pairwise (5K easy pairs + 5K hard pairs, measured by paired-preference accuracy) + RAG answer accuracy.
- **Models**: Qwen3-8B/14B/32B, Llama-3-8B, Llama-3.1-8B; $temperature=0$, Qwen3 "thinking" mode disabled.

### Key Designs

1.  **Three Judge Paradigms × Two Output Formats**:
    - **Function**: Exhausting prompt variations to identify the setting that best triggers LLM authority perception.
    - **Mechanism**: **PointJudge** scores items in isolation; **PairJudge** performs pairwise comparisons with outputs as PairRank (winner selection) or PointScore (absolute scores followed by BubbleSort / AverageScore); **ListJudge** processes the entire list with outputs as ListRank (reordering) or PointScore (scoring then sorting). For fine-grained tasks, Pairs use anchor-based approximation; for coarse-grained, all pairs are used.
    - **Design Motivation**: By comparing these settings, the study isolates whether "comparison" is better than "absolute valuation" and whether "ranking outputs" are more stable than "score outputs." AverageScore was found to be more robust to non-transitive judgments.

2.  **Comparison with/without Webpage Text (Ctx vs. w/o Ctx)**:
    - **Function**: Testing if authority is perceived as synonymous with text quality.
    - **Mechanism**: Each judge is run using "domain/entity name only" versus "domain/entity name + webpage snippet."
    - **Design Motivation**: If text improves judgment, authority is proxied by linguistic style; if it degrades judgment, authority is an independent signal. Results showed that for List/Pair settings, text almost always lowered scores (e.g., Qwen3-8B PointScore dropped from 71.35 to 63.91). However, text improved performance on hard pairs, acting as a compensation when structural signals were ambiguous.

3.  **Authority-Aware RAG Filtering Pipeline**:
    - **Function**: Integrating authority perception into a real RAG pipeline to measure gain in answer accuracy.
    - **Mechanism**: Using the ListJudge + PointScore protocol to score 10 documents, which are then filtered to top-$k$ using three criteria: (a) Relevance Filter (query relevance), (b) Utility Filter (utility scoring based on a generated pseudo-answer), and (c) Authority Filter (source URL only, ignoring content).
    - **Design Motivation**: Using the same judge protocol ensures fairness. Restricting the Authority Filter to URLs demonstrates that "authority signals" are distinct and can be utilized without document text.

### Loss & Training
As an evaluation paper, no training is involved. Metrics include Spearman / Kendall (listwise), paired accuracy (pairwise), and answer accuracy (RAG).

## Key Experimental Results

### Main Results: DomainAuth (Fine-grained 10-level, Spearman $\rho$ %)

| Model | Ctx | PointJudge | List+ListRank | List+PointScore | Pair+PairRank | Pair+PointScore |
|------|------|------|------|------|------|------|
| Qwen3-32B | w/o | 73.72 | 73.63 | 74.41 | 72.10 | **75.28** |
| Qwen3-32B | w/ | 73.57 | 55.85 | 63.10 | 66.32 | 69.93 |
| Qwen3-14B | w/o | 71.97 | 72.02 | 73.09 | 70.21 | **73.43** |
| Llama-3-8B | w/o | 63.87 | 57.53 | 66.08 | 61.05 | 64.83 |
| Qwen3-8B | w/o | 41.97 | 54.01 | 67.11 | 15.18 | **71.35** |

For EntityAuth (Basketball, w/o text), Spearman $\rho$ reached **88.90** for Llama-3-8B PointScore and 85.94 for Qwen3-32B, consistently 10+ points higher than DomainAuth, indicating entity authority is easier for LLMs to perceive than web authority.

### RAG Answer Accuracy (RAGAuth, 120 questions, %)

| Model | k | Relevance | Utility | **Authority** | w/o Filter |
|------|------|------|------|------|------|
| Qwen3-14B | 1 | 51.67 | 60.00 | **76.67** | 58.33 |
| Qwen3-14B | 3 | 45.00 | 66.67 | **75.00** | 58.33 |
| Qwen3-32B | 1 | 63.33 | 65.00 | **70.00** | 55.00 |
| Llama-3-8B | 3 | 41.67 | 52.50 | **64.17** | 50.83 |
| Llama-3.1-8B | 3 | 55.00 | 48.33 | **71.76** | 57.50 |

### Key Findings
- **ListJudge / PairJudge + PointScore are strongest**: Allowing the model to see context before assigning absolute scores yields the best result. Isolated PointJudge is weakest as models lack a calibrated baseline across items.
- **ListRank is inferior to PointScore**: Forcing a complete ranking causes confusion during near-ties, whereas scores are smoother and more stable.
- **Model scale monotonic improvement**: Spearman scores increase with Qwen3 size (8B → 14B → 32B), indicating authority perception depends on internal world knowledge.
- **Adding webpage text is often counterproductive**: Scores dropped 10-20 points in List/Pair settings but improved on hard pairs. This supports the hypothesis that "authority $\neq$ style" and shows LLMs can be misled by linguistic fluency.
- **Authority filtering is effective for RAG**: The Authority Filter consistently outperforms Relevance and Utility filters, achieving a **Gain** of up to 18.3 points on Qwen3-14B compared to no filtering, showing that many RAG errors stem from relevant but low-authority sources.
- **Scoring Biases**: Models exhibit high positive bias for `.gov` and `.edu` domains (+2.5 to +3.3) and overestimate social media (+1 to +2.4). LLMs internalize Top-Level Domains (TLDs) as authority signals but are overly sensitive to brand effects beyond structural authority.

## Highlights & Insights
- **Isolated Authority Perception**: This study is the first to isolate "authority perception" as a distinct LLM capability, operationalizing what was previously buried under broader terms like credibility.
- **Authority vs. Text Quality**: The finding that text can degrade performance suggests that LLMs can be misled by "fluency," highlightng a vulnerability in RAG anti-hallucination efforts.
- **Engineering Cookbook**: The ListJudge + PointScore protocol represents a highly efficient engineering implementation (scoring all documents in one forward pass) for RAG filtering.
- **Bias Identification**: The analysis of TLD and social media biases provides clear targets for future calibration research.

## Limitations & Future Work
- **Proxy Ceiling**: PageRank may undervalue niche expert blogs or small government sites with few backlinks. Dynamic, topic-specific authority measures are needed.
- **RAGAuth Scale**: 120 questions are sufficient to demonstrate value but insufficient for broad statistical analysis across themes.
- **Model Coverage**: Only 5 open-source LLMs were tested. Closed-source flagships (e.g., GPT-5, Claude 4) might show different results due to RLHF data differences.
- **Authoritative Errors**: The relationship between high authority and occasional errors (e.g., news retractions) remains unexplored.

## Related Work & Insights
- **vs. Relevance Rankers**: While others focus on topical matching, Ours focuses on source priors; these are orthogonal and can be combined.
- **vs. Utility Rankers**: Utility assesses if an answer *can* be generated; authority assesses if it *should* be trusted. RAGAuth shows they are complementary, with authority being more critical in medical/current events.
- **vs. Credibility-aware RAG**: Unlike methods relying on pre-labels, this work tests inherent LLM knowledge and demonstrates benefits without additional training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Authority as an independent dimension is a first in LLM-as-a-Judge literature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 3 datasets, 5 models, 5 paradigms, and end-to-end RAG; limited only by RAGAuth scale and lack of closed-source models.
- **Writing Quality**: ⭐⭐⭐⭐ Logical explanations for ListRank vs. PointScore and detailed bias analysis.
- **Value**: ⭐⭐⭐⭐ Provides a practical RAG cookbook: using ListJudge+PointScore for authority filtering can yield 10-20 points in accuracy gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](../../NeurIPS2025/information_retrieval/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)

</div>

<!-- RELATED:END -->
