---
title: >-
  [Paper Note] ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests
description: >-
  [NeurIPS 2025][LLM Safety][Recommender Systems] This paper proposes ORBIT, a unified benchmark for recommender systems comprising standardized evaluation on 5 public datasets and a privacy-safe hidden test set…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Recommender Systems"
  - "Benchmark"
  - "Privacy-Preserving Dataset"
  - "Web Recommendation"
  - "LLM-based Recommendation"
date: 2026-05-08
content_hash: 43cccc44844c3384
---

# ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests

**Conference**: NeurIPS 2025
**arXiv**: [2510.26095](https://arxiv.org/abs/2510.26095)  
**Code**: [Official Website](https://www.open-reco-bench.ai)  
**Area**: AI Safety
**Keywords**: Recommender Systems, Benchmark, Privacy-Preserving Dataset, Web Recommendation, LLM-based Recommendation

## TL;DR

This paper proposes ORBIT, a unified benchmark for recommender systems comprising standardized evaluation on 5 public datasets and a privacy-safe hidden test set, ClueWeb-Reco, constructed from real users' browsing histories. The benchmark systematically evaluates 12 recommendation models and introduces the LLM-QueryGen baseline, revealing the limitations of existing approaches in large-scale, real-world recommendation scenarios.

## Background & Motivation

### Limitations of Prior Work

**Background**: Recommender systems are among the most impactful AI applications today, yet their research evaluation is hindered by two core problems:

**Dataset Issues**: Existing public datasets (e.g., Amazon Reviews, Yelp) primarily capture review/rating behaviors rather than more authentic browsing behaviors—review actions account for only 1–2% of user interactions, introducing severe popularity bias. Some newer datasets (e.g., PixelRec, Tenrec) were collected without user consent, raising privacy and ethical concerns.

**Evaluation Inconsistencies**: Different studies vary substantially in data splitting strategies, inference-time candidate pool configurations (full ranking vs. sampled candidates), and evaluation metrics, making fair comparison infeasible. Multiple prior studies have demonstrated that such inconsistencies alter method rankings and impede field progress.

The authors argue for a unified benchmark that simultaneously addresses data authenticity and evaluation consistency, while ensuring that data collection satisfies informed consent and privacy protection requirements. This directly motivates ORBIT—a unified recommendation benchmark featuring standardized public evaluation and a hidden test set derived from real browsing behavior.

## Method

### Overall Architecture

ORBIT consists of two core components: (1) standardized reproducible evaluation on 5 public datasets, and (2) the ClueWeb-Reco hidden test set, constructed from real U.S. user browsing data to assess the generalization capability of recommendation models.

### Key Designs

1. **Standardized Public Evaluation**: Five datasets are selected—MovieLens-1M and four Amazon Reviews categories (Beauty/Toys/Sports/Books)—with a unified leave-one-out splitting strategy. For a user interaction sequence of length $n$, the first $n-2$ items are used for training, the $(n-1)$-th item serves as the validation target, and the $n$-th item as the test target. All models are evaluated via full-corpus ranking, reporting Recall@K and NDCG@K ($K \in \{1,10,50,100\}$), eliminating evaluation bias introduced by sampled candidate pools.

2. **ClueWeb-Reco Dataset Construction**: A two-stage pipeline is employed to balance authenticity and privacy. In the first stage, user browsing histories are collected via Amazon Mechanical Turk and Prolific.co (1,747 users consented, IRB-approved), and after online and offline quality control, 1,024 sessions comprising 12,282 records are retained. In the second stage, semantic soft matching is performed—the MiniCPM-Embedding-Light encoder maps collected webpages against ClueWeb22-B EN (87 million public webpages), with DiskANN approximate nearest neighbor search identifying the most semantically similar public pages to replace original URLs. Key design choices include: removing exact matches (11.07% of URLs have exact matches, replaced by their Top-2 alternatives), enforcing one-to-one mapping (identical URLs map to the same page; distinct URLs map to distinct pages), and ensuring the final dataset consists entirely of synthetic sequences.

3. **LLM-QueryGen Baseline**: The recommendation problem is reformulated as a retrieval task—an LLM is prompted to generate a query from the titles of a user's browsed webpages, which is then used to retrieve the most relevant pages from ClueWeb22 via ANN indexing as recommendations. This design leverages the LLM's semantic understanding and zero-shot generalization capabilities without requiring training on recommendation data.

### Loss & Training

- The 12 models evaluated on public datasets follow their respective original training strategies, with a maximum history length fixed at 50 (10 for HLLM).
- ClueWeb-Reco adopts zero-shot evaluation, testing model generalization across 87 million candidates.
- Quality control is applied at two levels: online removal of fraudulent and malformed data, and offline removal of inappropriate or uninformative content, yielding an overall retention rate of approximately 30%.

## Key Experimental Results

### Main Results (Public Datasets, Recall@10 / NDCG@10)

| Model | ML-1M (NDCG) | Beauty (NDCG) | Books (NDCG) | Avg. NDCG |
|-------|-------------|--------------|-------------|-----------|
| GRU4Rec | 0.1438 | 0.0065 | 0.0473 | 0.0447 |
| SASRec | 0.0967 | 0.0630 | 0.0384 | 0.0464 |
| BERT4Rec | 0.1820 | 0.0254 | 0.0325 | 0.0532 |
| HSTU | 0.1838 | 0.0343 | 0.0375 | 0.0589 |
| TASTE | 0.1505 | 0.0122 | 0.0386 | 0.0490 |
| **HLLM** | **0.1880** | 0.0027 | **0.0663** | **0.0641** |

### ClueWeb-Reco Hidden Test (Zero-Shot)

| Model | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 |
|-------|-----------|---------|------------|----------|
| TASTE | 0.0020 | 0.0015 | 0.0039 | 0.0019 |
| HLLM | 0.0088 | 0.0041 | 0.0176 | 0.0059 |
| GPT-3.5-Turbo-QG | 0.0088 | 0.0058 | 0.0254 | 0.0089 |
| GPT-4.1-QG | 0.0107 | 0.0050 | 0.0254 | 0.0077 |
| **DeepSeek-V3-QG** | **0.0127** | **0.0082** | **0.0371** | **0.0129** |

### Key Findings

- **Content-based models outperform ID-based models**: This advantage is particularly pronounced on high-sparsity datasets (Amazon Books), as content-based models can leverage item metadata to build more accurate user profiles.
- **HLLM achieves overall best performance**: Using LLMs to construct item and user representations yields significant gains, though performance degrades on small datasets (Beauty) due to insufficient training data.
- **ClueWeb-Reco exposes real-world difficulty**: All models suffer a dramatic performance drop against 87 million candidates; traditional collaborative filtering models nearly fail, while LLM-QueryGen demonstrates substantially better zero-shot generalization.
- **Soft matching quality**: Human annotations show a positive correlation between retrieval scores and relevance, with Top-1 matches receiving the highest relevance ratings, and 88% of human annotations passing validity checks.

## Highlights & Insights

- **Privacy-preserving dataset construction paradigm**: The soft matching approach completely eliminates PII while preserving user behavioral patterns—a more thorough approach than direct anonymization, generalizable to other privacy-sensitive domains.
- **Hidden tests prevent data leakage**: Access to ClueWeb22 content requires signing a license agreement with CMU, effectively preventing overfitting to the test set.
- **Reframing recommendation as generative retrieval**: LLM-QueryGen redefines the recommendation problem as a "query generation → document retrieval" pipeline, opening a new avenue for integrating LLMs into recommender systems.

## Limitations & Future Work

- ClueWeb-Reco is limited in scale (only 1,024 sessions) and restricted to U.S. adult users.
- Soft matching incurs approximately 20% information loss due to low-relevance mappings, which may hinder fine-grained user interest modeling.
- Public datasets cover only movies and e-commerce, excluding news, social media, music, and other domains.
- Although promising, the LLM-QueryGen baseline incurs high inference costs, posing challenges for real-world deployment.
- Evaluation metric values are universally low under extreme sparsity (99.999%+), potentially limiting the discriminability between methods.

## Related Work & Insights

- Privacy handling strategies from MSMARCO and TREC directly inspired the soft matching approach adopted in this work.
- Evaluation toolkits such as RecBole and BARS have made pioneering contributions to standardization but lack hidden tests grounded in real user behavior.
- This work reveals an important trend: as recommendation candidate pools scale to tens of millions, traditional collaborative filtering paradigms may need to transition toward semantic understanding-based paradigms.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of constructing a privacy-safe recommendation dataset via soft matching is novel, though the contribution on the public dataset side is primarily engineering-oriented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation across 12 models, 5 datasets, and a hidden test set is highly comprehensive; human annotations validate soft matching quality.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed descriptions of the data collection and privacy-protection pipeline; case studies are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed standardized benchmark and realistic evaluation scenario for the recommender systems community; LLM-QueryGen opens a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)
- [\[NeurIPS 2025\] Stop DDoS Attacking the Research Community with AI-Generated Survey Papers](stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)
- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](../../ACL2026/llm_safety/membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[NeurIPS 2025\] Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery](unlearning_as_ablation_toward_a_falsifiable_benchmark_for_generative_scientific_.md)

</div>

<!-- RELATED:END -->
