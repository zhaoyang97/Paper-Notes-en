---
title: >-
  [Paper Note] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives
description: >-
  [AAAI 2026][Medical Imaging][dense retrieval] This paper proposes a hard negative mining method that constructs a multi-hop semantic graph from PubMed citation chains and performs random walks thereon. Using only 20k tra…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "dense retrieval"
  - "hard negative mining"
  - "citation graph"
  - "biomedical IR"
  - "PubMed"
date: 2026-05-08
content_hash: ad5f1124b9ff9ccb
---

# BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives

**Conference**: AAAI 2026
**arXiv**: [2511.08029](https://arxiv.org/abs/2511.08029)  
**Code**: [bisect-group/BiCA](https://github.com/bisect-group/BiCA)  
**Area**: Medical Imaging
**Keywords**: dense retrieval, hard negative mining, citation graph, biomedical IR, PubMed

## TL;DR

This paper proposes a hard negative mining method that constructs a multi-hop semantic graph from PubMed citation chains and performs random walks thereon. Using only 20k training samples and minimal fine-tuning steps, 33M/110M small models surpass retrieval baselines with billions of parameters on BEIR and LoTTE.

## Background & Motivation

**Explosive growth of biomedical literature**: PubMed indexing continues to expand, making it difficult for traditional keyword-based retrieval to precisely locate relevant documents in highly specialized, terminology-dense literature.

**Difficulty of hard negative mining**: The biomedical domain contains numerous semantically similar papers, making it challenging for traditional hard negative sampling methods—based on cross-encoders or static embedding cosine distances—to effectively distinguish positive from negative samples.

**Citation relations as natural signals**: Cited papers share contextual relevance with source documents without being duplicates, making them naturally suited as hard negatives. However, no prior work has systematically exploited citation structure for negative sampling.

**High deployment cost of large models**: Large-parameter retrieval models such as GTR-xxl (4.8B) offer strong performance but incur high inference latency and deployment costs, making them unsuitable for real-time scenarios and resource-constrained environments.

**Zero-shot generalization requirements**: Biomedical retrieval frequently lacks annotated data, requiring models to generalize well to both in-domain and out-of-domain tasks under zero-shot conditions.

**Low data efficiency**: Most retrieval models rely on large-scale annotated or pseudo-annotated data; efficiently adapting to a domain with minimal high-quality data remains an open challenge.

## Method

### Overall Architecture

BiCA adopts a four-stage pipeline: (1) synthetic query generation from positive document abstracts using a T5 model; (2) construction of a 2-hop citation neighborhood via the PubMed API; (3) hard negative mining through multi-path random walks on the semantic graph; and (4) fine-tuning of GTE models using (query, positive, hard negatives) triplets.

### Key Design 1: 2-Hop Citation Neighborhood Construction

- Starting from 20,000 PubMed seed documents, the NCBI E-utilities API is accessed via pubmed-parser to retrieve 1-hop citations (directly cited papers) and 2-hop citations (citations of citations) for each article.
- 80 parallel processes are used to accelerate API calls, ultimately constructing for each seed document a complete neighborhood structure comprising its abstract, the set of 1-hop abstracts, and the set of 2-hop abstracts.
- Only records with successfully retrieved abstracts are retained, ensuring data quality for subsequent mining.

### Key Design 2: Diversified Semantic Graph Walks (Core Innovation)

- **Dense semantic graph construction**: PubMedBERT encodes all abstracts within the 1-hop and 2-hop neighborhoods into high-dimensional vectors, and a complete pairwise cosine similarity matrix is computed.
- **Multi-origin walks**: Independent walk paths are initiated from the 3 most query-similar 1-hop documents ($N_{paths}=3$), each with a path length of 3 steps ($L_{path}=3$).
- **Stochastic sampling strategy**: Rather than greedily selecting the most similar node at each step, the method samples from the top-5 unvisited neighbors with similarity-weighted probability, increasing negative sample diversity.
- **Global deduplication set**: All paths share a single visited set, ensuring that different paths explore distinct documents.
- **Random negative augmentation**: One additional uniformly sampled unvisited document is appended to improve training robustness. On average, 6.5 hard negatives are generated per query.

### Key Design 3: Query Synthesis

The Doc2Query (all-t5-base-v1) model is used to generate synthetic queries from positive document abstracts, simulating realistic user search behavior and eliminating dependence on manually annotated queries.

## Loss & Training

- **Loss function**: Multiple Negative Ranking Loss (MNR), formulated as $\mathcal{L}_{MNR} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{d}_+)}{\exp(\mathbf{q} \cdot \mathbf{d}_+) + \sum_{i=1}^{K} \exp(\mathbf{q} \cdot \mathbf{d}_i^-)}$.
- **Base models**: GTE-small (33M, 384-dim) and GTE-Base (110M, 768-dim), both BERT-architecture models trained with multi-stage contrastive learning.
- **Training setup**: Fine-tuned for only 20 steps on a single V100 GPU using approximately 20,000 training instances (totaling ~150,000 documents), demonstrating exceptionally high data efficiency.

## Key Experimental Results

### Main Results 1: Zero-Shot Evaluation on 14 BEIR Tasks (nDCG@10)

| Model | Params | COVID | NFC | SciFact | SciDocs | ArguAna | FEVER | HotpotQA | Avg |
|-------|--------|-------|-----|---------|---------|---------|-------|----------|-----|
| GTR-xxl | 4.8B | 0.500 | 0.342 | 0.662 | 0.161 | 0.540 | 0.740 | 0.599 | 0.486 |
| ColBERTv2 | 110M | 0.738 | 0.338 | 0.693 | 0.154 | 0.463 | 0.785 | 0.667 | 0.490 |
| DRAGON+ | 110M | 0.759 | 0.339 | 0.679 | 0.159 | 0.469 | 0.781 | 0.662 | 0.491 |
| **BiCA-small** | **33M** | 0.661 | 0.347 | 0.727 | 0.214 | 0.555 | **0.815** | 0.637 | **0.501** |
| **BiCA-Base** | **110M** | 0.684 | **0.378** | **0.762** | **0.231** | **0.571** | **0.815** | 0.657 | **0.518** |

- BiCA-Base achieves the highest 14-task average nDCG@10 of 0.518 with 110M parameters, surpassing GTR-xxl (0.486) at 4.8B.
- BiCA-small (33M) ranks second with an average of 0.501, outperforming most 110M-level baselines, yielding a parameter efficiency ratio of 145×.

### Main Results 2: LoTTE Long-Tail Topic Retrieval (Success@5)

| Model | Search-Writing | Search-Lifestyle | Forum-Writing | Forum-Lifestyle |
|-------|---------------|-----------------|--------------|----------------|
| ColBERTv2 | 80.1 | 84.7 | 76.3 | 76.9 |
| BiCA-small | 79.8 | 86.8 | 78.1 | 82.2 |
| **BiCA-Base** | **81.6** | **87.7** | **80.8** | **84.0** |

- BiCA-Base achieves the best Success@5 across all 4 LoTTE subdomains and both query types.
- BiCA-small consistently ranks second across all subdomains.

### Latency Analysis

- BiCA-small achieves a total latency of only 994ms at batch size 2000, approximately half that of ColBERTv2 (1844ms), making it suitable for real-time deployment.

### Ablation Study

- Walk parameter ablation: $N_{paths}=3, L_{path}=3$ achieves the highest average nDCG@10 (0.2739) across 5 datasets with the best stability.
- Data scale ablation: Performance increases monotonically from 1k to 20k training instances (e.g., SciFact improves from 0.262 to 0.493).
- Architecture generalization: Fine-tuning on DistilBERT and E5-base-v2 also yields average improvements of +1.56 and +0.84, respectively.

## Highlights & Insights

- **Citation graph as a hard negative source** is highly creative: it cleverly exploits the structural signals of academic citation relationships to generate semantically proximate yet non-redundant negatives, outperforming traditional cosine-distance or cross-encoder sampling.
- **Extreme data efficiency**: Surpassing billion-parameter models with only 20k samples and 20 fine-tuning steps demonstrates the training value of high-quality hard negatives.
- **Small models, large impact**: BiCA-small (33M) excels in both performance and latency, validating the practical viability of lightweight retrieval models for real-world deployment.
- **Comprehensive evaluation**: Covering 14 BEIR tasks, 8 LoTTE subtasks, latency analysis, multi-dimensional ablations, and cross-architecture validation, the experimental design is thorough.

## Limitations & Future Work

- Citation neighborhood construction depends on the PubMed API and is not directly applicable to domains not indexed by PubMed (e.g., computer vision, social sciences).
- Synthetic queries generated by Doc2Query may exhibit distribution shift relative to real user queries.
- Only BERT-family small models are evaluated; the effect of fine-tuning larger language models (e.g., LLaMA-series embeddings) remains unexplored.
- The coverage of 2-hop neighborhoods is limited by citation counts; papers with few citations may yield insufficiently rich negative sample pools.
- Performance on non-English biomedical literature is not discussed.

## Related Work & Insights

- **MedCPT**: Leverages PubMed user click logs for contrastive pre-training and serves as an important biomedical retrieval baseline. BiCA relies on citation structure rather than user behavior data.
- **SL-HyDE**: Employs LLMs to generate hypothetical documents for zero-shot retrieval—a novel idea but computationally expensive. BiCA is more lightweight.
- **DRAGON+**: A retrieval model combining multiple pre-training objectives; BiCA surpasses it on most tasks.
- **ColBERTv2**: A late-interaction retrieval model and a strong baseline in both efficiency and effectiveness; BiCA-Base substantially outperforms it on average.
- **LinkBERT / DRAGON**: Enhance language models using knowledge graphs, sharing conceptual similarity with BiCA's use of citation graphs but differing in application.

## Rating

- Novelty: ⭐⭐⭐⭐ — Hard negative mining via citation graph semantic walks is a novel and intuitively clear contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ — 14 BEIR tasks + LoTTE + latency + multi-dimensional ablations provide comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, informative pipeline diagrams, and well-formatted algorithm pseudocode
- Value: ⭐⭐⭐⭐ — Practically informative for biomedical retrieval and data-efficient domain adaptation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](../../ACL2026/medical_imaging/efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](../../ACL2026/medical_imaging/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)

</div>

<!-- RELATED:END -->
