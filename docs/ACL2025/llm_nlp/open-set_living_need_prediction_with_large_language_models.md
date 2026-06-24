---
title: >-
  [Paper Note] Open-Set Living Need Prediction with Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Open-Set Classification] This work proposes the PIGEON system, which reformulates user need prediction on life service platforms from a closed-set classification problem into an open-set generation problem. It leverages GNN-based behavior embeddings to retrieve historical records to assist LLM prediction, refines predictions guided by Maslow's hierarchy of needs, and fine-tunes a text embedding model to retrieve services from flexible needs…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Open-Set Classification"
  - "Living Need Prediction"
  - "LLM"
  - "Graph Neural Networks"
  - "Life Service Recommendation"
date: 2026-05-08
content_hash: a9b9f53a7f2ee090
---

# Open-Set Living Need Prediction with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.02713](https://arxiv.org/abs/2506.02713)  
**Code**: [https://github.com/tsinghua-fib-lab/PIGEON](https://github.com/tsinghua-fib-lab/PIGEON)  
**Area**: LLM/NLP, Recommender Systems  
**Keywords**: Open-Set Classification, Living Need Prediction, LLM, Graph Neural Networks, Life Service Recommendation

## TL;DR

This work proposes the PIGEON system, which reformulates user need prediction on life service platforms from a closed-set classification problem into an open-set generation problem. It leverages GNN-based behavior embeddings to retrieve historical records to assist LLM prediction, refines predictions guided by Maslow's hierarchy of needs, and fine-tunes a text embedding model to retrieve services from flexible needs, achieving an average improvement of 19.37% on real Meituan data.

## Background & Motivation

**Background**: On life service platforms such as Meituan, users' consumption behaviors are driven by their latent living needs (e.g., "wanting fast food" drives takeout orders). Accurately predicting user needs is critical for personalized recommendations.

**Limitations of Prior Work**: Traditional methods treat need prediction as a closed-set classification problem, selecting from a predefined and limited set of need categories. However, this paradigm suffers from three fundamental flaws: (1) user needs are highly diverse and cannot be exhaustively categorized; (2) needs can be vague (e.g., "looking for a place to relax" rather than a specific venue type); (3) needs can be composite (e.g., "wanting a filling yet healthy meal").

**Key Challenge**: Fixed category sets cannot capture the diversity, vagueness, and compositeness of real-world user needs.

**Goal**: To achieve open-set living need prediction on life service platforms, generating free-text need descriptions unconstrained by predefined categories and applying them to downstream service retrieval.

**Key Insight**: Leveraging the generative capabilities and common-sense knowledge of LLMs to overcome the limitations of closed-set methods, while integrating user behavior data and psychological theories to enhance the personalization and plausibility of predictions.

**Core Idea**: Use GNN to encode behavioral preferences to assist LLMs in personalized open-set need generation, and then use Maslow's hierarchy of needs to align predictions with the human needs framework.

## Method

### Overall Architecture

The PIGEON system consists of four core modules: (1) a GNN-based behavior record retrieval module to help the LLM comprehend individual user preferences; (2) an LLM need prediction module that generates flexible need descriptions based on retrieved historical records and spatio-temporal contexts; (3) a Maslow's hierarchy-guided refinement module that aligns initial predictions with a structured human needs framework; and (4) a fine-tuned text embedding-based retrieval module that maps flexible need descriptions to actual life services on the platform.

### Key Designs

1. **GNN Behavior Embedding and Record Retrieval**:

    - **Function**: Learning behavioral embeddings of users, spatio-temporal contexts, and needs to retrieve the most relevant historical consumption records for LLM input.
    - **Mechanism**: Constructing a heterogeneous graph with three types of nodes (user $u$, spatio-temporal context $c$, need $n$) and two types of weighted edges (user-need, context-need). A LightGCN-like graph propagation is used to learn embeddings. The propagation formula is $\mathbf{e}_i^{(k+1)} = \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{D_i \cdot D_j} \mathbf{e}_j^{(k)}$, and the final embeddings are obtained by multi-layer combination and L2 normalization. The training uses the BPR loss: $\mathcal{L}_{\text{BPR}} = -\sum \ln \sigma(\mathbf{e}_uc^\top \mathbf{e}_n - \mathbf{e}_uc^\top \mathbf{e}_{n'})$.
    - **Design Motivation**: Traditional text embedding retrieval only captures general textual similarities, whereas behavior embeddings directly encode user preferences and spatio-temporal effects, allowing for more precise retrieval of relevant records. Combining personal records with similar users' records handles scenarios where users have unexplored needs or are in new contexts.

2. **Maslow's Hierarchy of Needs Guided Refinement**:

    - **Function**: Utilizing Maslow's hierarchy of needs to construct a structured needs framework that guides the LLM to refine initial predictions into descriptions that better align with real human needs.
    - **Mechanism**: Prompting the LLM first to generate a three-layer need framework based on the platform's service list and Maslow's theory; then constraining initial predictions using this framework to align them with a reasonable scope of needs (around 20 words). For massive service lists that exceed token limits, classification or batching is applied first.
    - **Design Motivation**: Free generation by LLMs can sometimes deviate from realistic needs—becoming overly specific actions (e.g., "eating KFC burgers") or generating unsupported needs (e.g., "writing an email"). A structured framework effectively corrects this.

3. **Open-Set Need-Adaptive Service Retrieval**:

    - **Function**: Mapping flexible need descriptions to specific life services on the platform to achieve high-efficiency retrieval.
    - **Mechanism**: Fine-tuning a text embedding model to use need descriptions as queries to retrieve relevant services. Training data construction: first generating flexible need descriptions for historical records, and then refining them with the LLM based on closed-set ground-truth needs. The model is trained using a triplet loss: $\mathcal{L} = \sum \max(0, \text{sim}(q, s') - \text{sim}(q, s) + \alpha)$.
    - **Design Motivation**: Closed-set methods associate needs and services using manual rules, but the infinite potential needs generated by an LLM cannot be mapped manually, necessitating machine learning to automatically understand need-service relationships.

### Loss & Training

- **GNN Encoder**: Trained with BPR ranking loss to optimize user/context/need embeddings.
- **Retrieval Model**: Fine-tuned `bge-base-v1.5` text embedding model using triplet loss.
- **Domain Adaptation**: Collecting prompt-prediction pairs from the large model as instruction tuning data, followed by end-to-end full-parameter fine-tuning of a smaller LLM to achieve the two-step framework output in a single-step inference.
- The LLM backbone defaults to GPT-4o mini (temperature=0), costing approximately $1.05 per 10,000 predictions.

## Key Experimental Results

### Main Results

| Method | Category | Beijing R@10 | Beijing R@20 | Shanghai R@10 | Shanghai R@20 |
|------|------|--------|--------|--------|--------|
| EulerNet | Closed-set CTR | 0.0350 | 0.0687 | 0.0634 | 0.0962 |
| DisenHCN | Closed-set Graph | 0.0924 | 0.1335 | 0.0727 | 0.1234 |
| Zero-shot CoT | Open-set | 0.0345 | 0.0574 | 0.0236 | 0.0471 |
| ReLLa | Open-set | 0.0588 | 0.1041 | 0.0535 | 0.0961 |
| LLMSREC-Syn | Open-set | 0.0804 | 0.1271 | 0.0632 | 0.1003 |
| **PIGEON** | **Open-set** | **0.1041** | **0.1662** | **0.1050** | **0.1490** |

PIGEON significantly outperforms the strongest baselines across all metrics in both cities ($p < 0.05$), with improvements of 15.18% in Beijing and 23.55% in Shanghai, respectively.

### Ablation Study

| Configuration | Beijing R@10 | Beijing R@20 | Description |
|------|--------|--------|------|
| PIGEON Full | 0.1041 | 0.1662 | — |
| W/o all history | 0.0277 | 0.0574 | Largest drop, common sense alone is far from enough |
| W/o personal history | 0.0784 | 0.1392 | Personal records are crucial for personalization |
| W/o similar users' history | 0.0781 | 0.1351 | Contribution comparable to personal records |
| W/o need refinement | 0.0932 | 0.1493 | Maslow framework refinement is effective |
| W/o retrieval fine-tuning | 0.0710 | 0.1149 | Fine-tuning is critical for flexible need mapping |
| Using closed-set query only | 0.0750 | 0.1203 | Flexible need descriptions are superior as queries |

### Key Findings

- Removing all history records leads to a performance drop to the Zero-shot CoT level, demonstrating that LLM common-sense alone is insufficient for personalized need prediction.
- Human evaluation (by 116 domain experts) reveals that PIGEON significantly outperforms LLMSREC-Syn in specificity (3.900 vs. 3.580, $p < 0.001$) and information density (3.737 vs. 3.413, $p < 0.001$).
- Case studies show that PIGEON can handle vague needs (e.g., "unwinding and recharging after work") and composite needs (e.g., "lunch + family gathering"), whereas closed-set methods can only predict a single coarse-grained category.

## Highlights & Insights

- Redefining need prediction from a closed-set to an open-set task is a valuable formulation innovation, fundamentally addressing the diversity, vagueness, and compositeness of user needs.
- GNN behavior embeddings elegantly compensate for the LLM's lack of individual user knowledge while outperforming text embedding-based retrieval, demonstrating the paradigm benefit of combining domain knowledge with LLMs.
- Incorporating Maslow's hierarchy of needs as a refinement framework is a creative interdisciplinary design, helping LLM-generated need descriptions better align with the hierarchical structure of human daily needs.
- The domain adaptation design using instruction-tuned smaller models opens up the possibility for online deployment.

## Limitations & Future Work

- Evaluation relies on indirect metrics from downstream retrieval tasks, lacking direct metrics to evaluate the quality of open-set need descriptions.
- The dataset is derived solely from two cities in Meituan (Beijing and Shanghai); generalizability remains to be verified on other platforms and scenarios.
- The construction of Maslow's framework relies on automated generation by LLMs, making consistency and quality hard to guarantee.
- Temporal dynamics and long-term evolutionary trends in user needs are not yet considered.

## Related Work & Insights

- **vs. Closed-set Methods (DisenHCN)**: PIGEON's open-set prediction captures vague and composite needs that are undiscoverable by closed-set methods, completely outperforming them in downstream retrieval.
- **vs. LLM Retrieval-Augmented Methods (ReLLa/LLMSREC-Syn)**: PIGEON's GNN behavior embedding is more effective than text embedding retrieval, demonstrating that behavioral signals reflect user preferences better than mere textual similarity.
- **vs. Zero-shot LLM (Zero-shot CoT)**: Demonstrates that bare LLM common-sense is insufficient for personalized need prediction, highlighting the necessity of historical behavioral data.

## Rating

- Novelty: ⭐⭐⭐⭐ Open-set need prediction is a valuable task redefinition, and the combined design of GNN + LLM + Maslow's theory is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative results (two cities) + ablations (7 configurations) + human evaluation (116 experts) + case studies, providing comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, detailed methodology, and a smooth narrative.
- Value: ⭐⭐⭐ Direct practical value for life service recommendations, though the application scenario remains somewhat niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)
- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)
- [\[ICLR 2026\] Beyond the Known: An Unknown-Aware Large Language Model for Open-Set Text Classification](../../ICLR2026/llm_nlp/beyond_the_known_an_unknown-aware_large_language_model_for_open-set_text_classif.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)

</div>

<!-- RELATED:END -->
