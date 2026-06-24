---
title: >-
  [Paper Note] CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems
description: >-
  [ACL 2025][Recommender Systems][LLM-based Recommender Systems] The CoVE framework is proposed to expand the LLM vocabulary by assigning a unique token ID and embedding to each item, which converts sequential recommendation into a next-token prediction task. Compared to existing methods, CoVE improves recommendation accuracy by up to 62% and achieves an approximate 100x speedup in inference, while addressing memory constraints in large-scale scenarios via hashed embedding comp…
tags:
  - "ACL 2025"
  - "Recommender Systems"
  - "LLM-based Recommender Systems"
  - "Vocabulary Expansion"
  - "Embedding Compression"
  - "Sequential Recommendation"
  - "Hashed Compression"
date: 2026-05-08
content_hash: dd7d62c137a91bd0
---

# CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems

**Conference**: ACL 2025  
**arXiv**: [2506.19993](https://arxiv.org/abs/2506.19993)  
**Code**: [GitHub](https://github.com/HaochenZhang717/CoVE-official-Repo)  
**Area**: Recommender Systems  
**Keywords**: LLM-based Recommender Systems, Vocabulary Expansion, Embedding Compression, Sequential Recommendation, Hashed Compression

## TL;DR

The CoVE framework is proposed to expand the LLM vocabulary by assigning a unique token ID and embedding to each item, which converts sequential recommendation into a next-token prediction task. Compared to existing methods, CoVE improves recommendation accuracy by up to 62% and achieves an approximate 100x speedup in inference, while addressing memory constraints in large-scale scenarios via hashed embedding compression.

## Background & Motivation

**Background**: Large Language Models (LLMs) are increasingly applied in recommender systems, primarily through two paradigms: (a) utilizing LLMs to provide embedding initialization for non-LLM recommender models; (b) fine-tuning LLMs to directly generate target item titles, which are then mapped to real items through embedding retrieval (e.g., BIGRec).

**Limitations of Prior Work**: 
   - Paradigm (a) only exploits the embedding capability of LLMs without leveraging their content comprehension capabilities.
   - Paradigm (b), namely the finetune-and-retrieval framework, suffers from three key issues: LLMs must accurately predict multi-token item titles (difficult), generated titles may not exist in the item space (hallucination problem), and text generation inference is slow.

**Key Challenge**: LLMs possess powerful next-token prediction capabilities, but existing recommendation frameworks fail to exploit this capability directly, instead requiring LLMs to perform the more challenging task of multi-token title generation.

**Goal**: To design a framework that allows LLMs to directly utilize next-token prediction for recommendation while addressing the memory efficiency issues of embedding tables in large-scale item spaces.

**Key Insight**: Drawing inspiration from vocabulary expansion techniques in domain adaptation, unique tokens are assigned to each item to transform recommendation into a single-token prediction task.

**Core Idea**: Expand the LLM's vocabulary so that each item corresponds to a unique token, directly recommend using next-token prediction logits, and solve the embedding table memory bottleneck with hash compression.

## Method

### Overall Architecture

The core workflow of CoVE consists of:
1. **Vocabulary Expansion**: Add a unique token (e.g., `<|205|>`) for each item in the item space $\mathcal{I}$ to the LLM's tokenizer.
2. **Embedding Table Expansion**: Map each item token to an independent, trainable embedding vector.
3. **Fine-Tuning**: Simultaneously train the item embedding table, LoRA adapter, and lm_head to align the LLM with the recommendation task.
4. **Inference**: Given a user's historical interaction sequence, extract the scores of the dimensions corresponding to item IDs in the logits for ranking and recommendation, entirely avoiding text generation.

### Key Designs

#### 1. Fine-tuning Task Design
- **Function**: Model the recommendation task as standard next-token prediction.
- **Mechanism**: Training samples contain task instructions, user history (task input, containing item IDs and titles), and the target item (task output). During training, the next-token prediction loss is minimized; during inference, only the scores corresponding to the last $|\mathcal{I}|$ dimensions of the logits output by the lm_head are needed.
- **Design Motivation**: Simplify multi-token title generation into single-token ID prediction, eliminating hallucinations and significantly accelerating inference.

#### 2. Hashed Embedding Compression
- **Function**: Compress the item embedding table from $|\mathcal{I}|$ to $|\mathcal{S}|$ (where $|\mathcal{S}| \ll |\mathcal{I}|$).
- **Mechanism**: Define $k$ universal hash functions $h_1, \ldots, h_k$, each mapping items to a shared embedding space. The embedding of item $i$ is obtained by averaging its hash-mapped shared embeddings:

$$\mathbf{e}_i = \frac{1}{k} \sum_{j=1}^{k} \mathbf{e}_{h_j(i)}$$

The hash functions leverage simple arithmetic operations: $h(i) = ((ai + b) \bmod p) \bmod |\mathcal{S}|$

- **Design Motivation**: In large-scale scenarios (e.g., the Amazon dataset containing 48.19 million items), directly storing the embedding table requires approximately 96GB of GPU memory. Hashed compression makes training feasible.

### Loss & Training

- **Loss Function**: Standard next-token prediction loss (cross-entropy).
- **Training Configuration**: 
    - Beauty/Toys/Sports datasets: LLaMA-3.2-3B, learning rate $10^{-4}$, batch size 32, LoRA rank 8, alpha 16, up to 10 epochs.
    - Video Games dataset: LLaMA-2-7B + 4-bit QLoRA.
- **Trainable Parameters**: Item embedding table, LoRA adapter, lm_head.

## Key Experimental Results

### Main Results

On three Amazon datasets (Beauty/Toys/Sports) with a compression ratio of 2, CoVE vs. the best baseline (TIGER):

| Dataset | Metric | TIGER | CoVE | Gain |
|--------|------|-------|------|------|
| Beauty | NG@5 | 0.0321 | **0.0498** | +55% |
| Beauty | HR@10 | 0.0648 | **0.1009** | +56% |
| Toys | NG@5 | 0.0371 | **0.0509** | +37% |
| Toys | HR@5 | 0.0521 | **0.0719** | +38% |
| Sports | NG@5 | 0.0204 | **0.0296** | +45% |
| Sports | HR@10 | 0.0400 | **0.0624** | +56% |

CoVE vs. BIGRec (finetune-and-retrieval) on the Video Games dataset:

| Metric | BIGRec | CoVE | Gain |
|------|--------|------|------|
| NG@5 | 0.0189 | **0.0221** | +17% |
| HR@10 | 0.0329 | **0.0437** | +33% |
| HR@20 | 0.0457 | **0.0621** | +36% |

Inference speed: CoVE runs at 6.5 samples/s compared to BIGRec's 0.066 samples/s, achieving an **approximate 100x speedup**.

### Ablation Study

Importance of item titles and embedding table training (Beauty dataset):

| Setting | NG@5 | HR@5 |
|------|------|------|
| Trainable Embeddings Only (No Titles) | 0.045 | 0.0622 |
| Title Information Only (Frozen Embeddings) | 0.0057 | 0.0094 |
| CoVE (Both Combined) | **0.0498** | **0.0714** |

Robustness of embedding compression: Under a 16x compression ratio, CoVE still outperforms the SOTA baseline (TIGER) on HR@5 and NG@5, with the sole exception of HR@10 on the Toys dataset.

### Key Findings

1. CoVE consistently outperforms all baselines across four datasets, with improvements of 30%-62% in NG and HR metrics.
2. The fine-tuned LLM successfully learns the mapping between item IDs and titles, which is crucial for high-quality recommendation.
3. Freezing the embedding table causes a drastic decline in performance, indicating that learning high-quality item embeddings is critical.
4. The robustness of embedding compression varies across datasets; Sports and Toys remain stable under 8x compression, while Beauty is more sensitive.

## Highlights & Insights

- **Elegant Problem Transformation**: Converts recommendation from "generating item titles" to "predicting item ID tokens", resolving hallucination, speed, and accuracy issues simultaneously.
- **Balance between Theory and Practice**: Hashed embedding compression makes the framework viable for large-scale industrial scenarios (reducing GPU memory overhead from 96GB in a 48M-item setup).
- **Thorough Experiments**: Evaluated across 4 datasets, with 12+ baselines compared, multiple ablation studies, inference speed analyses, and case studies, maintaining highly solid evidence.
- **Insightful Case Study**: Shows that the fine-tuned LLM can automatically output correct ID-title correspondences during generation, proving that CoVE indeed enables the LLM to learn item semantics.

## Limitations & Future Work

1. Embedding compression was only explored using hash methods; more advanced compression techniques (quantization, low-rank approximation) warrant future investigation.
2. Experiments were restricted to Amazon e-commerce datasets; validation on other domains (news, video, music) is lacking.
3. The cold-start problem (how to rapidly obtain high-quality embeddings for new items) remains undiscoused.
4. Sensitivity to compression ratios varies by dataset, and adaptive compression strategies are currently absent.

## Related Work & Insights

- **BIGRec** (Bao et al., 2023): A representative of the finetune-and-retrieval framework and the primary comparison target for CoVE.
- **TIGER** (Rajput et al., 2023): A SOTA method among non-LLM baselines, which CoVE significantly outperforms.
- **ALPT** (Li et al., 2023b): Adaptive low-precision training, which can potentially be integrated into CoVE's embedding compression in the future.
- Vocabulary expansion for domain adaptation (Cui et al., 2023; Liu et al., 2024a): The inspiration source for CoVE, extending vocabulary expansion from language adaptation to recommendation scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying vocabulary expansion to recommender systems introduces a novel perspective with an elegant problem transformation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 datasets, 12+ baselines, multi-dimensional ablation studies, and inference speed analysis yield an exceptionally solid evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Features a clear structure, well-articulated motivation, and well-designed figures/tables.
- **Value**: ⭐⭐⭐⭐ — High practical value for industry deployment given the 100x inference acceleration and substantial accuracy gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[ICLR 2026\] Continual Low-Rank Adapters for LLM-based Generative Recommender Systems](../../ICLR2026/recommender/continual_low-rank_adapters_for_llm-based_generative_recommender_systems.md)
- [\[ICLR 2026\] Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](../../ICLR2026/recommender/rank-grpo_training_llm-based_conversational_recommender_systems_with_reinforceme.md)
- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](../../AAAI2026/recommender/rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)
- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)

</div>

<!-- RELATED:END -->
