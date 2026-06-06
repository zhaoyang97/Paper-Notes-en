---
title: >-
  [Paper Note] Continual Model Routing in Evolving Model Hubs
description: >-
  [ICML 2026][Model Compression][continual learning] When the number of available experts in a model hub grows from hundreds to thousands with continuous additions and retirements…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "continual learning"
  - "pre-inference routing"
  - "model hub"
  - "contrastive embedding"
  - "anchoring"
  - "experience replay"
date: 2026-05-08
content_hash: 4b93c5803cb59423
---

# Continual Model Routing in Evolving Model Hubs

**Conference**: ICML 2026  
**arXiv**: [2605.28577](https://arxiv.org/abs/2605.28577)  
**Code**: Available (Noted in paper, repository link to be updated)  
**Area**: Continual Learning / Model Routing / Embedding Retrieval  
**Keywords**: continual learning, pre-inference routing, model hub, contrastive embedding, anchoring, experience replay  

## TL;DR
When the number of available experts in a model hub grows from hundreds to thousands with continuous additions and retirements, traditional "train-once routers" or "pure model card retrieval" become inadequate. The authors formalize this problem as "continual classification with an expanding label space," construct CMRBench—a benchmark spanning 4 periods with over 2000 candidate models—and propose CARvE. CARvE utilizes contrastive embedding scoring, checkpoint anchoring to prevent drift, and structured negative replay to maintain discriminative power. It achieves a 5% improvement in D-Acc over standard LoRA replay, with only half the forgetting.

## Background & Motivation

**Background**: Model hubs like Hugging Face already host millions of pre-trained models. In deploying MoE or tool-augmented systems, the core problem has shifted from "can we train a model" to "which model should be executed." This pre-inference routing must be completed under strict latency and cost constraints without executing multiple candidate models. Representative methods include Gorilla (using RAT/RAG to retrieve model cards), HuggingGPT (using an LLM controller to select models via metadata), and various BM25 or dense retrievers that score model cards directly.

**Limitations of Prior Work**: (1) Once candidate scales reach thousands, static retrieval methods (based on model card similarity) significantly degrade—BGE-M3 achieves only 13.6% M-Acc with 2000+ models. (2) Model hubs are inherently non-stationary: new models arrive, old ones are deprecated, and new versions of the same series are frequently released. Training a router as a one-time classifier leads to rapid collapse when new models arrive. (3) Joint training using all prior data violates the deployment constraints of continual learning (old data may not be retainable, and computational budgets are limited).

**Key Challenge**: Routing must simultaneously satisfy three conflicting requirements: stable scoring in an open label space of 1000+ categories, incremental adaptation to new models, and zero overhead for executing candidates. Existing methods often sacrifice two of these to solve one.

**Goal**: Formalize pre-inference model routing as a "continual classification" problem where the label space grows over time; design a new benchmark for fair evaluation; and provide a specific router capable of handling scale, drift, and forgetting.

**Key Insight**: The authors observe three facts: (1) Routing is essentially a discriminative task (query → model-ID) that can be handled via contrastive embeddings without involving generation. (2) In continual learning, "parameter/output anchoring" and replay buffers can suppress catastrophic forgetting. (3) Large-vocabulary softmax is expensive, but using a fixed-size candidate set reduces per-instance scoring to $O(Kd)$. Combining these avoids the bottlenecks of SFT/RAT approaches like Gorilla.

**Core Idea**: Model IDs are learned as continuously appendable contrastive embedding vectors. When new experiences arrive, checkpoint anchoring locks the geometry of old model embeddings and projection matrices, combined with structured hard/semantic/far negative sampling and domain-weighted coreset replay to maintain the discriminative surface.

## Method

### Overall Architecture
Formalization of Continual Routing: A stream of experiences $\{E_t\}_{t=1}^T$, where each $E_t$ provides triplets $(q_i, m_i, d_i)$. The candidate pool expands cumulatively as $\mathcal{M}_{\leq t} = \bigcup_{k \leq t} \mathcal{M}_k$. The core of CARvE is an embedding scorer: a frozen backbone LLM (default LLaMA2-7B + LoRA) extracts hidden states $h(q) \in \mathbb{R}^D$ for a query, passed through a learnable projection $W$ to obtain $z(q) = h(q)W / \lVert h(q)W \rVert_2$. Each model ID maintains a learnable normalized embedding $e(m) = v(m)/\lVert v(m) \rVert_2$. Scoring is performed over a candidate set $\mathcal{C}(q)$ as $s(q,m) = z(q)^\top e(m) / \tau$, with the prediction $\hat m = \arg\max_{m \in \mathcal{C}(q)} s(q,m)$. As each experience arrives, the registry $\mathcal{R}$ appends new model ID rows, and the projection $W$, model embedding table $\{v(m)\}$, and LoRA adapters are updated using the snapshot from the end of the previous experience as an anchor.

### Key Designs

1.  **Checkpoint-based Asymmetric Anchoring**:
    - **Function**: Stabilizes the geometry of old model ID embeddings and query projections during experience transitions without constraining the embedding learning of newly added model IDs.
    - **Mechanism**: At the start of experience $t$, a snapshot of the parameters $\{v_{t-1}(m)\}_{m \in \mathcal{M}_{\leq t-1}}$ and $\Theta_{t-1}$ from the end of the previous experience is saved. During training on $E_t$, two anchoring terms are added to the main contrastive loss: embedding cosine drift $\mathcal{L}_{\mathrm{emb}} = \frac{1}{|\mathcal{M}_{\leq t-1}|}\sum_m (1 - \cos(v_t(m), v_{t-1}(m)))$ and projection MSE drift $\mathcal{L}_{\mathrm{proj}} = \frac{1}{|\Theta_t|}\sum_\theta \frac{1}{|\theta|}\lVert \theta - \theta_{t-1}\rVert_2^2$. Embeddings for new IDs are excluded from $\mathcal{L}_{\mathrm{emb}}$.
    - **Design Motivation**: Routing is performed based on embedding similarity rather than fixed classification heads, requiring the locking of geometry rather than decision boundaries. Simultaneously, new models must find their positions in the embedding space, necessitating "asymmetric" anchoring—locking the old while leaving the new free.

2.  **Fixed-size Candidate Training + Structured Negative Sampling**:
    - **Function**: Avoids training on a full softmax over 1000+ classes while maintaining discriminative power across domains and families.
    - **Mechanism**: For each $(q, m^+)$, a candidate set $\mathcal{C}(q)$ of size $K$ is constructed, always including the positive sample $m^+$ and three types of negative samples: high-scoring hard confusers (periodically mined), semantic negatives from the same or related domains, and far negatives from different domains. The loss is $\mathcal{L}_{\mathrm{route}} = -\log \frac{\exp(s(q,m^+))}{\sum_{m \in \mathcal{C}(q)} \exp(s(q,m))}$.
    - **Design Motivation**: (1) Reduces per-instance scoring cost to $O(Kd)$ instead of $O(|\mathcal{M}_{\leq t}| d)$; during deployment, this can be further reduced to $O(\log |M|)$ using FAISS. (2) Hard negatives support fine-grained discrimination, semantic negatives support intra-family differentiation (e.g., yolov8m/n/s), and far negatives maintain the macro-structure across domains.

3.  **Domain-Model Coreset Replay + Random Initialization**:
    - **Function**: Selects the most informative old samples under long-tailed hub distributions and avoids bias from the linguistic geometry of model card text.
    - **Mechanism**: Given a replay budget $B$, quotas are allocated to domains based on frequency (with minimum and maximum caps). Within each domain, the number of samples per model ID is limited, and farthest-point sampling (FPS) in a fixed embedding space is used to select the most diverse samples. Model embeddings are initialized as random vectors rather than warm-started with model card encodings.
    - **Design Motivation**: Random replay wastes budget on common domains in heavy-tailed distributions; FPS ensures coverage while reducing redundancy. Regarding random initialization, the authors found that 4 card-based initialization schemes performed 3-5pp lower in D-Acc and doubled forgetting compared to random initialization, as card embeddings encode descriptive language similarity rather than routing discriminability.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\mathrm{route}} + \lambda_{\mathrm{emb}} \mathcal{L}_{\mathrm{emb}} + \lambda_{\mathrm{proj}} \mathcal{L}_{\mathrm{proj}}$. The backbone LLM remains frozen throughout, with only the LoRA adapters, projection $W$, and model embedding table being updated. When new experiences arrive, new embedding rows are appended by ID without reindexing old ones. Anchoring is applied only to the router parameters and does not constrain the LoRA.

## Key Experimental Results

### Main Results
CMRBench consists of 4 sequential experiences covering APIBench (852 models), ToolMMBench (481), HuggingBench E3 (520), and HuggingBench E4 (547), totaling ~34k samples. Metrics include Model-ID accuracy (M), Model-family accuracy (F), and Domain accuracy (D), each paired with a forgetting (FGT) measure. The table shows averages across the 4 experiences (LLaMA2-7B backbone):

| Method | M-Acc ↑ | F-Acc ↑ | D-Acc ↑ | D-FGT ↓ |
| :--- | :--- | :--- | :--- | :--- |
| BGE-M3 retrieval | 13.6 | 16.2 | 44.0 | 3.3 |
| Gorilla RAG | 6.7 | 10.4 | 43.0 | 0.1 |
| HuggingGPT (Qwen3-32B) | – | – | 51.7 | – |
| Sequential Finetuning | 28.0 | 34.8 | 64.3 | 37.2 |
| TIES merging | 7.6 | 10.9 | 28.6 | 32.6 |
| LwF | 28.8 | 35.9 | 56.4 | 39.5 |
| EWC | 31.3 | 38.4 | 66.2 | 31.4 |
| Random Replay 10% | 39.1 | 47.3 | 75.9 | 13.1 |
| Random Replay 20% | 41.3 | 49.8 | 78.1 | 7.8 |
| **CARvE 10% replay** | **~46.4** | – | **80.7** | **5.9** |
| **CARvE 20% replay** | **46.4** | – | **82.9** | **3.0** |
| LoRA Joint Training | – | – | 79.3 | – |

Key Observations: (1) Pure retrieval routing is completely outperformed by SFT at hub scales (BGE-M3 achieved only 13.6% M-Acc). (2) In the continual learning setting, given the same 10% replay budget, CARvE achieves 4.8pp higher D-Acc than standard LoRA replay with only 5.9% vs 13.1% forgetting. (3) CARvE even surpasses the joint training upper bound (79.3 → 80.7), suggesting that anchoring and structured negative sampling provide beneficial regularization.

### Ablation Study

| Configuration | Key Effect | Description |
| :--- | :--- | :--- |
| Full CARvE | D-Acc 80.7 / D-FGT 5.9 | Baseline |
| Card-based initialization | D-Acc −3 ~ −5pp, FGT roughly doubled | Geometric conflict; 4 variants consistently worse |
| CARvE + EWC | On par with CARvE | Fisher regularization is not the source of CARvE’s gains |
| Backbone: Qwen2.5-7B | D-Acc 81.5 | Stable conclusions across similar backbone sizes |
| Backbone: Qwen3-4B | D-Acc and FGT degraded | Small model representation quality is a bottleneck |
| Exp1 evaluation after Exp3 | Standard replay 60.8 vs CARvE 74.5 | Anchoring effectively prevents drift |
| Exp3 evaluation after Exp4 | Standard replay 54 vs CARvE 69.7 | Confirms inflow of new models is the primary pressure |

### Key Findings
- Domain-level accuracy benefits the most (+5pp), followed by model family, while model-ID level is the most challenging. This aligns with the idea that macro-structures in the embedding space are easier to stabilize, whereas fine-grained differentiation tests negative sample quality.
- Standard replay shows a significant collapse after the introduction of HuggingBench (Exp3-4), whereas CARvE remains stable, validating anchoring as the key to resisting hub expansion.
- Initializing model embeddings with model cards is worse—a counter-intuitive but repeatedly verified conclusion: routing requires "discriminative geometry" rather than "semantic similarity geometry."

## Highlights & Insights
- **Problem Reframing**: For the first time, pre-inference model routing is explicitly treated as "continual classification with a label space expanding over time." This allows the legitimate introduction of the continual learning toolkit (replay/anchor/coreset) into routing.
- **Asymmetric Anchoring**: Conventional continual learning usually locks all parameters; CARvE locks only the embeddings and projections of old IDs while leaving new IDs free. This "semi-frozen" approach can be directly transferred to any continual task with expandable embedding tables (retrieval, recommendation, open-vocab classification).
- **Counter-intuitive Experiment**: Random initialization of model embeddings > model card initialization. This indicates that "semantic similarity $\neq$ routing discriminability," serving as a valuable negative lesson for those attempting warm-starts with text encoders.

## Limitations & Future Work
- The candidate set size $K$ and hard negative mining frequency use fixed values without adaptive schemes; $K$ may be insufficient for extremely large model families.
- The 4 evaluated experiences are sequentially concatenated in time, but real hubs involve simultaneous "additions" and "retirements/replacements." This paper does not explicitly handle indices compression or cleanup for deprecated models.
- The router learns a direct query→model mapping without considering engineering constraints like cost, latency, or licenses; industrial deployment would require an additional multi-objective reranking layer.
- Tests were limited to 7B-class backbones. Larger ones (70B+) would improve query embedding quality but might undermine the cost-saving premise of "running only the router."

## Related Work & Insights
- **vs Gorilla / RAT**: Gorilla uses RAG/RAT for LLM-based model-ID generation. At hub scale, retrieval noise makes it worse than zero-shot SFT. CARvE abandons model card text for a pure embedding approach, avoiding failure modes where retrieval provides misleading context.
- **vs HuggingGPT-style Controllers**: Using a large LLM as a router (Qwen3-32B) yields 51.7% D-Acc, but inference costs are much higher than embedding scoring. CARvE’s 80.7% shows that a small backbone with embedding contrast can outperform large LLM controllers at a fraction of the cost.
- **vs Traditional Continual Learning Baselines**: LwF/EWC focus on regularization of classification layers, but routers lack a fixed classification head. CARvE succeeds by applying anchoring to embeddings and projections, serving as a model for translating continual learning to "open label spaces."
- **vs Classic MoE Routers**: MoE gating networks are end-to-end differentiable routers with fixed, symmetric candidates. This work addresses the non-stationary, heterogeneous scenario of "external hubs with 1000+ candidates growing over time," providing the first systematic solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Effective Model Pruning: Measure the Redundancy of Model Components](effective_model_pruning_measure_the_redundancy_of_model_components.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)

</div>

<!-- RELATED:END -->
