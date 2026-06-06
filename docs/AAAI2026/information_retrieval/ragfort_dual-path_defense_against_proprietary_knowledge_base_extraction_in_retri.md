---
title: >-
  [Paper Note] RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation
description: >-
  [AAAI 2026][Information Retrieval & RAG][RAG Security] This paper proposes RAGFort, the first systematic dual-path framework for defending against RAG knowledge base extraction attacks. It combines contrastive reindexing…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "RAG Security"
  - "Knowledge Base Protection"
  - "Contrastive Reindexing"
  - "Cascade Generation"
  - "Dual-Path Defense"
date: 2026-05-08
content_hash: 59e7ea9415a51ba4
---

# RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation

**Conference**: AAAI 2026
**arXiv**: [2511.10128](https://arxiv.org/abs/2511.10128)  
**Code**: [https://github.com/happywinder/RAGFort](https://github.com/happywinder/RAGFort)  
**Area**: Information Retrieval
**Keywords**: RAG Security, Knowledge Base Protection, Contrastive Reindexing, Cascade Generation, Dual-Path Defense

## TL;DR

This paper proposes RAGFort, the first systematic dual-path framework for defending against RAG knowledge base extraction attacks. It combines contrastive reindexing (inter-class) to isolate topic boundaries with constrained cascade generation (intra-class) to suppress sensitive content output. RAGFort reduces the knowledge recovery rate to 0.51× that of an unprotected system while preserving answer quality.

## Background & Motivation

### State of the Field

RAG systems are increasingly deployed in high-value domains (healthcare, finance), where knowledge bases represent core intellectual property. Attackers can systematically reconstruct a knowledge base through black-box queries:

**Intra-class extraction**: Iteratively refining queries within a specific topic to progressively extract detailed information on that topic.

**Inter-class extraction**: Recursively generating queries targeting semantically related topics by leveraging already-extracted knowledge, thereby expanding coverage across topics.

For example, the RAG-Thief attack employs an automated agent that starts from an initial query, stores extracted chunks, analyzes them to generate increasingly targeted queries, and ultimately reconstructs nearly the entire knowledge base by both deepening within topics and expanding to new ones.

### Limitations of Prior Work

Existing methods defend only one attack path:
- **Intra-class defenses** (e.g., paraphrasing/summarization): Reduce leakage within individual chunks but cannot prevent cross-topic aggregation.
- **Inter-class defenses** (e.g., retrieval distance thresholding): Limit topic drift but permit deep extraction within a single topic.

More critically, these two classes of defenses are **inherently conflicting**: inter-class protection requires the system to retrieve only a small number of highly relevant chunks, whereas intra-class protection (e.g., summarization) requires aggregating information from multiple chunks to produce diverse outputs.

### Key Experimental Findings

The authors design masking experiments to validate the complementarity of the two paths:
- With only intra-class or only inter-class protection, the surrogate knowledge base retains substantial usable information.
- Joint protection significantly degrades the utility of the surrogate knowledge base.
- As the protection ratio increases, joint protection substantially outperforms either single-path approach.

**Conclusion**: Effective defense must simultaneously protect both paths.

## Method

### Overall Architecture

RAGFort comprises two cooperative modules:
1. **Inter-class protection**: Structure-aware Contrastive Reindexing — reorganizes the retriever index to enhance semantic separation between topics.
2. **Intra-class protection**: Constrained Cascade Generation — a cascade framework combining a lightweight draft model with a powerful verifier model to suppress sensitive token generation.

### Key Designs

#### 1. Inter-class Protection: Contrastive Reindexing

**Core Idea**: Unsupervised clustering is used to discover the topic structure of the knowledge base, followed by supervised contrastive learning to train an encoder that better separates topics in the embedding space, making it harder for attackers to retrieve content from one topic via queries targeting another.

Three stages:

**Stage 1: HDBSCAN Pseudo-Label Generation**
- A pretrained encoder (e.g., Sentence-BERT) obtains a dense representation $E(k_i)$ for each chunk.
- HDBSCAN clustering partitions chunks into $C$ clusters $\{\mathcal{C}_1, \ldots, \mathcal{C}_C\}$, automatically determining the number of clusters and handling outliers.
- Each chunk is assigned a pseudo-label $y_i \in \{1, \ldots, C\}$.

**Stage 2: SupCon Contrastive Learning**
- A new encoder is trained using the supervised contrastive loss:
$$\mathcal{L}_{\text{sup}} = \sum_{i=1}^{B} -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a)/\tau)}$$
- Same-cluster samples are pulled together while different-cluster samples are pushed apart, reinforcing topic boundaries.

**Stage 3: Index Replacement**
- All chunks are re-encoded using the trained structure-aware encoder $f_{\text{sup}}(\cdot)$.
- A new index is constructed for retrieval, while the generator still operates on the original chunk content.
- Query encoders and decoders remain unchanged to preserve answer quality.

**Design Motivation**: Attackers rely on semantic similarity for inter-class expansion; enhanced inter-class separation makes it difficult for queries to "drift" from one topic to an adjacent one and retrieve relevant content.

#### 2. Intra-class Protection: Constrained Cascade Generation

**Core Idea**: A lightweight draft model and a powerful reference model are combined, with a near-optimal rejection rule designed to suppress the generation probability of sensitive content.

**Cascade Framework**:
- The draft model first proposes $\gamma$ candidate tokens with corresponding probabilities $q_t$.
- The verifier model evaluates each token and decides whether to accept it according to the rejection rule.
- If rejected, the verifier generates a substitute token.

**Optimization Objective**: Design a rejection rule $r(x_{<t}, z)$ that minimizes the expected loss under verifier supervision while constraining the rejection budget $B$ and the sensitive content generation probability threshold $C$:

$$\min_r \mathbb{E}[(1-r) \cdot \ell(y, q_t) + r \cdot \ell(y, p_t)]$$
$$\text{s.t. } r \cdot D_{\text{TV}}(p_t, q_t) \leq B, \quad (1-r) \cdot \frac{q_t(z)}{p_t(z)} \leq C$$

**Approximate Rejection Rule** (derived via Lemma 1):
$$\hat{r}(x_{<t}, z) = 1 \iff \max_y q_t(y) < \max_y p_t(y) - \alpha \cdot D_{\text{TV}}(p_t, q_t) + \eta \cdot \frac{q_t(z)}{p_t(z)}$$

Rejection is triggered when the draft model's confidence is substantially lower than that of the verifier.

**Theoretical Guarantee** (Lemma 2): The regret bound between the approximate rule and the optimal rule is $\max_y |\mathbb{P}_t(y) - q_t(y)| + \max_y |\mathbb{P}_t(y) - p_t(y)|$, which is small when both models closely approximate the true distribution.

### Loss & Training

- The SupCon training for inter-class protection uses the standard contrastive loss with temperature parameter $\tau$.
- Intra-class protection is an inference-time method requiring no additional training.
- Draft/Verifier pairings: Qwen-14B uses Qwen-7B as the draft; Gemma-3-27B uses Gemma-3-4B; DeepSeek-R1-8B serves as both roles simultaneously.

## Key Experimental Results

### Main Results

Three datasets: HealthCareMagic (medical QA), Enron Email (corporate email), Math QA (mathematical reasoning); three generative models; two attacks (Worm-Attack, RAG-Thief).

| Defense Method | Relative Average CRR (↓) |
|----------------|--------------------------|
| No Protection | 1× |
| Re-ranking (inter-class) | 0.91× |
| Summarization (intra-class) | 0.87× |
| **RAGFort (dual-path)** | **0.51×** |

Detailed results (HealthCareMagic, Qwen-14B):

| Defense | Worm CRR (%) | RAG-Thief CRR (%) |
|---------|-------------|-------------------|
| No Protection | 17.60 | 57.16 |
| RAGFort | **8.84** | **27.96** |
| Re-ranking | 16.28 | 56.44 |
| Summarization | 15.44 | 47.24 |

### Ablation Study

| Configuration | Relative Average CRR | Note |
|---------------|----------------------|------|
| Full RAGFort | 0.51× | Joint dual-path optimum |
| RAGFort_InterOnly | 0.75× | Inter-class protection alone insufficient |
| RAGFort_IntraOnly | 0.83× | Intra-class protection alone weaker |
| No Protection | 1.00× | Baseline |

Impact on system performance:

| Model / Dataset | ACC Before | ACC After | ACC Drop |
|-----------------|-----------|-----------|----------|
| Qwen-14B / HealthCare | 68.16% | 66.57% | -1.59% |
| DeepSeek-R1-8B / HealthCare | 61.36% | 61.12% | -0.24% |
| Gemma-3-27B / HealthCare | 69.64% | 68.88% | -0.76% |
| Qwen-14B / Enron | 99.10% | 98.85% | -0.25% |
| Qwen-14B / MathQA | 85.31% | 82.81% | -2.50% |

### Key Findings

1. **Dual-path significantly outperforms single-path**: RAGFort (0.51×) vs. Re-ranking (0.91×) vs. Summarization (0.87×), achieving a 49% reduction in recovery rate.
2. **Inter-class and intra-class protections are complementary**: Each alone reduces CRR to only 0.75×/0.83×, while their combination achieves 0.51×, validating the dual-path complementarity hypothesis.
3. **Negligible ACC degradation**: No more than 2.5 percentage points across all scenarios, demonstrating that the defense does not impair user experience.
4. **FLOPs reduced rather than increased**: Cascade generation replaces part of the large model's inference with a smaller model, actually lowering computational cost (e.g., Qwen-14B/HealthCare decreases from 38T to 19.4T FLOPs).
5. **Draft model capability gap affects ACC**: DeepSeek (using itself as draft) exhibits the smallest ACC drop (0.24%), while Qwen (using 7B as draft) shows a slightly larger drop (1.59%).

## Highlights & Insights

1. **Systematic thinking**: This is the first work to formalize RAG knowledge protection as a dual-path problem rather than an ad hoc stack of defensive measures.
2. **Theory meets practice**: The rejection rule for cascade generation is derived with rigorous theoretical justification and a bounded regret guarantee, rather than being a simple engineering heuristic.
3. **No LLM modification required**: Neither module requires retraining the generative model, making deployment straightforward.
4. **Comprehensive attack-defense evaluation**: Two mainstream attack methods are tested across three domains and three models.
5. **Efficiency-aware design**: Cascade generation reduces FLOPs while simultaneously providing security protection.

## Limitations & Future Work

1. Only black-box attacks are evaluated; robustness under white-box or gray-box scenarios remains unknown.
2. HDBSCAN clustering quality may affect inter-class protection, implying implicit assumptions about knowledge base structure.
3. Parameter sensitivity analysis mentioned in the paper is not presented; tuning would be required in practical deployments.
4. Adaptive attacks in which the adversary is aware of the defense mechanism are not considered.
5. Evaluation is limited to chunk-level recovery rate; fine-grained information leakage is not assessed.
6. Although categorized under "human body understanding," the work is substantively an AI security/NLP contribution.

## Related Work & Insights

- **RAG-Thief** (Jiang 2024) serves as the strongest attack baseline, demonstrating the power of automated iterative extraction.
- **Worm-Attack** (Cohen 2024) simulates a more extreme self-replicating propagation attack.
- **Speculative Decoding** inspires the cascade generation design; however, the original method targets inference acceleration, whereas this work repurposes it as a security tool.
- **SupCon** (supervised contrastive learning) is applied here in a natural way — using document cluster labels as supervision signals to enhance inter-class separation in the retrieval space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First systematic dual-path RAG defense framework)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple datasets, models, and attacks; lacks parameter sensitivity and adaptive attack evaluation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous theoretical derivation)
- Value: ⭐⭐⭐⭐⭐ (Addresses practical security challenges in commercial RAG deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](../../ACL2026/information_retrieval/quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ICML 2026\] Predictive Prefetching for Retrieval-Augmented Generation](../../ICML2026/information_retrieval/predictive_prefetching_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
