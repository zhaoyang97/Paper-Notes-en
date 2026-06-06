---
title: >-
  [Paper Note] GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation
description: >-
  [ICML 2026][Recommender Systems][Multi-behavior Recommendation] GCIB utilizes a dual strategy of "Graph Information Bottleneck + Cross-behavior Contrastive Learning." It first prunes edges in the auxiliary behavior graph…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Multi-behavior Recommendation"
  - "Graph Information Bottleneck"
  - "Contrastive Learning"
  - "HSIC"
  - "Denoising"
date: 2026-05-08
content_hash: d1a0e9a48e93c22d
---

# GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation

**Conference**: ICML 2026  
**arXiv**: [2605.25690](https://arxiv.org/abs/2605.25690)  
**Code**: https://github.com/akajinchen/GCIB  
**Area**: Recommendation Systems / Information Retrieval  
**Keywords**: Multi-behavior Recommendation, Graph Information Bottleneck, Contrastive Learning, HSIC, Denoising

## TL;DR
GCIB utilizes a dual strategy of "Graph Information Bottleneck + Cross-behavior Contrastive Learning." It first prunes edges in the auxiliary behavior graphs that are irrelevant to the target task at the structural level (maximizing mutual information with the target behavior and minimizing mutual information with the original auxiliary graph using an HSIC surrogate). It then aligns the denoised auxiliary representations with sparse target representations via InfoNCE at the feature level, achieving a 7%–40% relative improvement in HR@10 / NDCG@10 over the best baselines across four multi-behavior recommendation benchmarks.

## Background & Motivation

**Background**: Multi-behavior recommendation mitigates data sparsity issues in target behaviors (e.g., purchasing) by incorporating auxiliary behaviors such as "click," "add-to-cart," and "favorite." Current mainstream approaches utilize GNNs to build bipartite graphs for each behavior and fuse multi-behavior representations through attention or concatenation.

**Limitations of Prior Work**: The authors conducted a controlled experiment on Tmall (Figure 1) using a LightGCN backbone. The results showed that using only the auxiliary graph yielded the lowest HR@10, followed by using only the target graph. While mixing all behaviors performed best, the gain over a single graph was limited. This highlights two persistent issues: auxiliary graphs contain many edges irrelevant or harmful to the target task, and the target behavior itself is too sparse to provide sufficient supervisory signals for robust representation learning.

**Key Challenge**: Existing IB-based recommendation methods perform "denoising" in the representation space by compressing fused embeddings. However, these methods are equivalent to "denoising after noise propagation"—once noise is aggregated into user/item embeddings during the message-passing phase, it cannot be completely removed. In other words, **graph cleaning at the structural level must occur before GNN message passing**, not after.

**Goal**: To learn, in an end-to-end manner without relying on any "ground truth" noise labels: (a) a denoised auxiliary graph $\mathcal{G}_k'$ tailored for the target task, and (b) a set of user/item representations that are robust to noise and aligned with the target task.

**Key Insight**: Directly apply the Graph Information Bottleneck principle to the edge level. Learn a Bernoulli edge mask for the original auxiliary graph $\mathcal{G}_k$ such that the denoised graph $\mathcal{G}_k'$ is "sufficient for target signals $\mathcal{R}$" while being "compressive of the original $\mathcal{G}_k$," i.e., $\max\ I(\mathcal{R}; \mathcal{G}_k') - \beta I(\mathcal{G}_k'; \mathcal{G}_k)$. Since mutual information terms lack explicit forms, the authors bypass them using BPR equivalence and HSIC surrogates.

**Core Idea**: Use **edge-level IB** to prune auxiliary graphs and **cross-behavior InfoNCE** to treat denoised auxiliary representations as "semantic supplements" for target representations, achieving dual denoising at both structural and feature levels.

## Method

### Overall Architecture
The input consists of a set of user-item interaction matrices $\{\mathcal{R}^{(k)}\}$ for $\mathcal{K}$ behaviors. The GCIB pipeline is divided into four stages:

1. **Global Encoding**: All behavior edges are merged into a heterogeneous bipartite graph $\mathcal{G}_{global}$, using LightGCN to learn shared initial embeddings $\mathbf{E}_{global}$.
2. **Structural Denoising (GIB)**: Guided by target behavior representations $\mathbf{E}_{target}$, differentiable retention probabilities $w_{ab}$ are assigned to auxiliary edges. Denoised graphs $\mathcal{G}_k'$ are obtained via Bernoulli sampling. HSIC is then used to reduce the statistical dependence between the node representations of $\mathcal{G}_k'$ and the original $\mathcal{G}_k$.
3. **Feature Alignment (GCL)**: LightGCN is run on $\mathcal{G}_{target}$ and each $\mathcal{G}_k'$ to obtain target views $\mathbf{z}^{tgt}$ and auxiliary views $\mathbf{z}^{aux}$. InfoNCE is used to align representations of the same user/item across views while pushing negative samples apart.
4. **Prediction**: The final recommendation score is computed via the inner product of the weighted average of $\mathbf{z}^{tgt}$ and $\mathbf{z}^{aux}$.

The entire network is optimized simultaneously using $\mathcal{L} = \mathcal{L}_{BPR} + \beta \mathcal{L}_{IB} + \lambda \mathcal{L}_{CL} + \gamma \|\Theta\|_2$.

### Key Designs

1. **Target-Guided Edge-Level IB Denoising**:
    - **Function**: Prunes target-irrelevant edges from the auxiliary graph $\mathcal{G}_k$ before message passing to obtain $\mathcal{G}_k'$.
    - **Mechanism**: Denoising is modeled as an edge-dropping problem. The probability $w_{ab}$ of retaining an auxiliary edge $e_{<u_a,i_b>}$ is defined as $f([\mathbf{e}_a;\mathbf{e}_b])$, where $\mathbf{e}_a, \mathbf{e}_b$ are embeddings learned from the target graph and $f$ is a single-layer MLP. Retention is thus determined by target preferences, treating target preferences as labels $Y$ in the IB framework. Bernoulli sampling is made differentiable using Concrete relaxation: $\mathrm{sigmoid}((\log(\delta/(1-\delta))+w_{ab})/t)$. The $\max I(\mathcal{R};\mathcal{G}_k')$ term is replaced by the target BPR loss (since BPR optimization is equivalent to maximizing the log-likelihood of target behaviors).
    - **Design Motivation**: Previous IB methods compress representations rather than graphs; denoising after GNN aggregation is often too late. Moving IB to the edge level filters noisy edges before message passing, ensuring cleaner embeddings at the source.

2. **HSIC Surrogate for Graph Compression**:
    - **Function**: Implements the IB compression term $\min I(\mathcal{G}_k'; \mathcal{G}_k)$, making the denoised graph and original graph "statistically independent" in the node representation space.
    - **Mechanism**: Since graphs are non-Euclidean, mutual information is difficult to estimate directly. The authors introduce HSIC (Hilbert-Schmidt Independence Criterion) as a surrogate. For mini-batch node representations $\mathbf{E}'^{\mathbf{B}}_k$ and $\mathbf{E}^{\mathbf{B}}_k$, the RBF kernel estimates $\hat{HSIC}(X,Y) = (n-1)^{-2}\mathrm{Tr}(K_X H K_Y H)$. The loss is $\mathcal{L}_{IB} = \frac{1}{|\mathcal{K}|}\sum_k \hat{HSIC}(\mathbf{E}'^{\mathbf{B}}_k, \mathbf{E}^{\mathbf{B}}_k)$. This estimate is fully differentiable and does not require prior assumptions about $p(\mathcal{G}_k'|\mathcal{G}_k)$.
    - **Design Motivation**: Variational bounds for MI require assuming a specific conditional distribution, which is hard for discrete graph structures. HSIC is model-free and turns "compression" into "independence regularization," making it more stable for engineering.

3. **Cross-Behavior InfoNCE Semantic Alignment**:
    - **Function**: Establishes a semantic bridge between the denoised auxiliary graph and the target graph, supplementing sparse target representations with rich semantics from auxiliary behaviors.
    - **Mechanism**: LightGCN on $\mathcal{G}_k'$ yields auxiliary views $\mathbf{z}^{aux_k}_u$, which are averaged across behaviors to get $\mathbf{z}^{aux}_u$. LightGCN on the target graph yields $\mathbf{z}^{tgt}_u$. InfoNCE $\mathcal{L}^u_{CL} = -\log\frac{\exp(s(\mathbf{z}^{tgt}_u,\mathbf{z}^{aux}_u)/\tau)}{\sum_{u'} \exp(s(\mathbf{z}^{tgt}_u,\mathbf{z}^{aux}_{u'})/\tau)}$ pulls the two views of the same user closer while pushing other users in the batch away. Item-side alignment is processed similarly.
    - **Design Motivation**: BPR signals are insufficient under sparse target behaviors. Direct fusion of auxiliary and target embeddings leads to noise contamination. By cleaning the auxiliary graph with GIB and then applying "soft alignment" via contrastive learning, the model gains oversight without introducing noise—alignment occurs after denoising, so "click" and "buy" are not forced to share identical semantics.

### Loss & Training
The total loss is the sum of four terms: target behavior BPR loss $\mathcal{L}_{BPR}$ (the "sufficiency" term), HSIC loss $\mathcal{L}_{IB}$ (the "compression" term), cross-behavior contrastive loss $\mathcal{L}_{CL}$, and $L_2$ regularization $\gamma\|\Theta\|_2$. Weights $\beta$ and $\lambda$ control the components. The model is trained end-to-end without a pre-training phase.

## Key Experimental Results

### Main Results
Experiments were conducted on four datasets: Tmall, Taobao, Yelp, and ML-10M. GCIB was compared against 13 baselines (including MF-BPR, LightGCN, R-GCN, NMTR, MBGCN, S-MBRec, CRGCN, MB-CGCN, PKEF, BCIPM, NSED, MBLFE) using HR@10/20 and NDCG@10/20 with leave-one-out evaluation.

| Dataset | Metric | GCIB | Best Baseline | Gain |
|--------|------|------|----------------|---------|
| Tmall | HR@10 / NDCG@10 | 0.1617 / 0.0944 | 0.1502 / 0.0831 (NSED/BCIPM) | +7.66% / +13.60% |
| Taobao | HR@10 / NDCG@10 | 0.1815 / 0.1199 | 0.1577 / 0.1004 (MBLFE/NSED) | +15.09% / +19.42% |
| Yelp | HR@10 / NDCG@10 | 0.0746 / 0.0358 | 0.0531 / 0.0261 (MBLFE) | +40.49% / +37.16% |
| ML-10M | HR@10 / NDCG@10 | 0.0916 / 0.0429 | 0.0810 / 0.0392 (BCIPM) | +13.09% / +9.44% |

Improvements on Yelp (the sparsest dataset) were the most significant, confirming that GCIB effectively addresses the "sparse target behavior + noisy auxiliary behavior" problem.

### Ablation Study
| Configuration | Tmall HR@10 | Taobao HR@10 | Description |
|------|-------------|--------------|------|
| GCIB (Full) | 0.1617 | 0.1815 | Full model |
| − Global | 0.1101 | 0.1666 | W/o global heterogeneous encoding |
| − IB | 0.1089 | 0.1724 | W/o structural GIB denoising |
| − InfoNCE | 0.1523 | 0.1661 | W/o cross-behavior contrastive alignment |
| − Both | 0.0356 | 0.0352 | W/o both IB and alignment |

### Key Findings
- Removing both IB and InfoNCE caused Tmall HR@10 to drop by 78%, indicating that structural denoising and feature alignment are both indispensable.
- GCIB achieved the largest relative gain on Yelp (+40% HR@10), validating the efficacy of denoising and contrastive supplementation in sparse scenarios.
- Removing global encoding (−Global) impacted Tmall more than Taobao, suggesting that a good starting point from global structural encoding is more critical as interaction complexity increases.

## Highlights & Insights
- **Moving IB to the edge level** rather than the representation level is the cleanest insight: instead of "contaminate then clean," GCIB "filters before contamination," which is more thorough from an information flow perspective.
- **Using BPR for $I(\mathcal{R};\mathcal{G}_k')$ and HSIC for $I(\mathcal{G}_k';\mathcal{G}_k)$** provides a practical engineering solution for the difficulty of estimating mutual information on graph structures. This can be transferred to any scenario involving graph pruning via IB.
- **Alignment before fusion, fusion after denoising**: The sequence of GIB $\rightarrow$ InfoNCE $\rightarrow$ summation ensures contrastive learning aligns meaningful semantics rather than noise. This design is applicable to any multi-view or multi-modal recommendation system.

## Limitations & Future Work
- Hyperparameters such as the IB coefficient $\beta$, contrastive weight $\lambda$, and temperature $\tau$ are sensitive to datasets; no automated tuning scheme was provided.
- Edge Bernoulli masks are scored based on target representations. For cold-start users/items with zero target interactions, this mechanism may fail—cold start was not discussed.
- HSIC estimation relies on mini-batch Monte Carlo sampling; small batch sizes might lead to high variance in independence estimation.
- Future work could include making edge masks user-aware to handle heterogeneous user interests or incorporating temporal information to extend IB to sequential multi-behavior recommendation.

## Related Work & Insights
- **vs BCIPM / NSED**: These are also IB-based, but they compress at the representation layer. GCIB compresses at the graph structure layer, explaining its consistent superiority by intervening earlier in the information flow.
- **vs CRGCN / MB-CGCN (Cascading Models)**: Cascading methods assume an ordered propagation (e.g., Click $\rightarrow$ Cart $\rightarrow$ Buy) and are sensitive to negative transfer. GCIB uses soft alignment via contrastive learning, avoiding hard assumptions about behavioral order.
- **vs S-MBRec / PKEF (Fusion Models)**: These use attention or MoE for behavior fusion without explicit denoising. GCIB explicitly separates denoising and alignment, making modules more interpretable and effective.

## Rating
- Novelty: ⭐⭐⭐⭐ Moving IB from representations to edges combined with HSIC is solid, though GIB + contrastive learning has some precedents in graph classification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, 13 baselines, and four metrics with full ablation studies, though lacking verification on large-scale industrial data.
- Writing Quality: ⭐⭐⭐⭐ The motivation derivation (Figure 1 experiments) is very clear, and formulas align well with diagrams.
- Value: ⭐⭐⭐⭐ An average 20%+ improvement in sparse target scenarios makes this a practical solution for industrial auxiliary behavior denoising.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)
- [\[AAAI 2026\] Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary](../../AAAI2026/recommender/behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)

</div>

<!-- RELATED:END -->
