---
title: >-
  [Paper Note] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion
description: >-
  [NeurIPS 2025][Graph Learning][Knowledge Graph Reasoning] DuetGraph proposes a dual-pathway (message passing + global attention) parallel fusion model and a coarse-to-fine reasoning optimization strategy. By separating r…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Knowledge Graph Reasoning"
  - "Dual-Pathway Fusion"
  - "Coarse-to-Fine Optimization"
  - "Over-Smoothing"
  - "Graph Neural Networks"
date: 2026-05-08
content_hash: 80c1656d19af6807
---

# DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion

**Conference**: NeurIPS 2025
**arXiv**: [2507.11229](https://arxiv.org/abs/2507.11229)  
**Code**: [https://github.com/USTC-DataDarknessLab/DuetGraph](https://github.com/USTC-DataDarknessLab/DuetGraph)  
**Area**: Graph Learning
**Keywords**: Knowledge Graph Reasoning, Dual-Pathway Fusion, Coarse-to-Fine Optimization, Over-Smoothing, Graph Neural Networks

## TL;DR
DuetGraph proposes a dual-pathway (message passing + global attention) parallel fusion model and a coarse-to-fine reasoning optimization strategy. By separating rather than stacking local/global information processing, it mitigates score over-smoothing in KG reasoning, achieving SOTA on both inductive and transductive tasks with up to 8.7% MRR improvement and 1.8× training speedup.

## Background & Motivation

**Background**: Knowledge graph (KG) reasoning aims to complete missing entities or relations. Mainstream approaches include message-passing networks (capturing local neighborhoods) and Transformers (capturing global dependencies). Recent leading methods (e.g., KnowFormer) stack both within a single stage to exploit local and global information simultaneously.

**Limitations of Prior Work**: The single-pathway stacking design causes **score over-smoothing**—scores of correct and incorrect answers converge (as shown in Figure 1, a large fraction of incorrect entities receive scores close to the correct answer across most baselines), degrading reasoning quality.

**Key Challenge**:
   - **Challenge 1**: Stacking message-passing layers and attention layers individually already exacerbates over-smoothing; concatenating both compounds these effects further.
   - **Challenge 2**: Single-stage models have limited discriminative capacity, as one-shot reasoning cannot effectively distinguish among candidate entities.

**Goal**: Simultaneously mitigate both sources of over-smoothing—information mixing (Challenge 1) and insufficient single-stage discrimination (Challenge 2).

**Key Insight**: **Decouple** local/global information processing into two independent pathways rather than stacking them; decompose inference into two stages: coarse-grained filtering followed by fine-grained discrimination.

**Core Idea**: Replace stacking with separation via dual-pathway fusion, combined with coarse-to-fine two-stage reasoning, to address score over-smoothing in KG reasoning from both architectural and inference-strategy perspectives.

## Method

### Overall Architecture
The input is a knowledge graph $\mathcal{G}=\{\mathcal{V},\mathcal{E},\mathcal{R}\}$ and a query $(h,r,?)$. The framework consists of two major components:
- **Training phase (Steps ①–④)**: Dual-pathway fusion model—a GNN encoder produces entity/relation representations; a global attention pathway and a local message-passing pathway then compute their respective weight matrices independently; an MLP adaptively fuses the two outputs.
- **Inference phase (Steps ⑤–⑧)**: Coarse-to-fine optimization—a coarse model generates an initial entity score table, which is partitioned into high- and low-score sub-tables by Top-$k$; a fine model updates scores in both sub-tables; the final answer is selected based on threshold $\Delta$.

### Key Designs

1. **Dual-Pathway Global-Local Fusion Model**:

    - **Function**: Separates local message passing and global attention into two independent pathways, each producing its own representations before adaptive fusion.
    - **Mechanism**: The global pathway applies a simple global attention mechanism to obtain $Z_{\text{global}}$; the local pathway applies a query-aware message-passing network to obtain $Z_{\text{local}}$; the final representation is $Z = \alpha \cdot Z_{\text{local}} + (1-\alpha) \cdot Z_{\text{global}}$, where $\alpha$ is a learnable parameter.
    - **Design Motivation**: Compared to single-pathway cascading, parallel dual-pathway processing avoids mutual interference between message-passing and attention layers. Theorem 1 proves that when $\alpha < \frac{\sigma_{\max}(\mathcal{M}_D)+\sigma_{\max}(\mathcal{M}_O)}{1+\sigma_{\max}(\mathcal{M}_O)}$, the upper bound on score difference in the dual-pathway model decays more slowly, i.e., it is more resistant to over-smoothing.
    - **Complexity Advantage**: Parallel execution yields complexity $\mathcal{O}(\max(L_m(|\mathcal{E}|d+|\mathcal{V}|d^2), L_t|\mathcal{V}|d^2))$, which is superior to the single-pathway complexity of $\mathcal{O}(L_m|\mathcal{E}|d+(L_m+L_t)|\mathcal{V}|d^2)$.

2. **Coarse-to-Fine Reasoning Optimization**:

    - **Function**: Decomposes single-stage inference into coarse-grained ranking followed by fine-grained discrimination.
    - **Mechanism**:
        - **Coarse stage**: An existing model (e.g., HousE, RED-GNN) generates an initial score table $\mathcal{T}$, partitioned by Top-$k$ into a high-score sub-table $\mathcal{T}^{\text{high}}$ and a low-score sub-table $\mathcal{T}^{\text{low}}$.
        - **Fine stage**: The dual-pathway model updates scores in both sub-tables. The top-scoring entities from each sub-table, $(e_h, s_{e_h})$ and $(e_l, s_{e_l})$, are compared; if $\gamma = s_{e_l} - s_{e_h} > \Delta$, the top entity from the low-score sub-table is selected as the final answer; otherwise the top entity from the high-score sub-table is selected.
    - **Design Motivation**: Theorem 2 proves that the expected score gap between the top entities of the two sub-tables satisfies $\mathbb{E}[|s_{e_h}-s_{e_l}|] > |(\frac{1}{N_h^2+1}-\frac{1}{N_l^2+1})\cdot\sigma|$; since $N_l/N_h > 1000$, this lower bound exceeds $0.1\sigma$, far larger than the $0.02\sigma$ of baseline methods. Theorem 3 further proves $P > P'$, i.e., coarse-to-fine optimization increases the probability of correct prediction.

3. **Loss Function**:

    - For each training triple $(h,r,t)$: $\mathcal{L} = -\log(\sigma(t|h,r)) - \sum_{t'}\log(1-\sigma(t'|h,r))$
    - Negative samples are drawn uniformly from remaining entities after masking correct answers; the Adam optimizer is used.

## Key Experimental Results

### Main Results (Inductive Reasoning)

| Dataset | Version | Metric | DuetGraph | KnowFormer (Prev. SOTA) | Gain |
|--------|------|------|-----------|----------------------|------|
| FB15k-237 | v1 | MRR | **0.507** | 0.466 | +8.8% |
| FB15k-237 | v2 | MRR | **0.549** | 0.532 | +3.2% |
| WN18RR | v3 | MRR | **0.501** | 0.467 | +7.3% |
| NELL-995 | v1 | MRR | **0.850** | 0.827 | +2.8% |
| NELL-995 | v2 | MRR | **0.543** | 0.465 | +16.8% |
| NELL-995 | v4 | MRR | **0.464** | 0.378 | +22.8% |

### Main Results (Transductive Reasoning)

| Dataset | Metric | Ours | AdaProp / HousE | Gain |
|--------|------|-----------|-----------------|------|
| FB15k-237 | MRR | **0.419** | 0.417 | +0.5% |
| WN18RR | MRR | **0.573** | 0.562 | +2.0% |
| NELL-995 | MRR | **0.560** | 0.554 | +1.1% |
| YAGO3-10 | MRR | **0.580** | 0.573 | +1.2% |

### Ablation Study

| Configuration | FB15k-237 v1 MRR | NELL-995 v1 MRR | Note |
|------|-------------------|-----------------|------|
| Full DuetGraph | **0.507** | **0.850** | Complete model |
| w/o coarse-to-fine | ~0.48 | ~0.83 | Removing coarse-to-fine degrades reasoning quality |
| w/o dual-pathway (single) | ~0.47 | ~0.82 | Degrades to single-pathway stacking, worsening over-smoothing |
| Different coarse model choices | varying | varying | HousE/RED-GNN etc. are all viable coarse models |

### Key Findings
- Both **dual-pathway fusion** and **coarse-to-fine optimization** independently contribute to performance gains, with their combination yielding the best results.
- DuetGraph achieves the most significant improvements on NELL-995 (up to 22.8% MRR gain), indicating greater advantage in settings with higher relational diversity.
- Training efficiency improves by 1.8×, owing to parallel dual-pathway computation.
- The hyperparameter $k$ (Top-$k$ split point) exhibits reasonable robustness, though values that are too small risk excluding correct answers, while values that are too large yield insufficiently discriminative sub-tables.

## Highlights & Insights
- The **"separation over stacking"** philosophy is elegant: existing methods implicitly assume that layering heterogeneous information improves performance, while this paper identifies such stacking as the root cause of over-smoothing. Separating the two pathways preserves the discriminative power of each information source independently—a principle transferable to other multi-source information fusion tasks.
- The **two-stage coarse-to-fine inference** essentially introduces a contrastive step: rather than directly selecting the globally highest-scoring entity, the model compares top candidates across two subsets, amplifying the score gap between correct and incorrect answers. Analogous cascaded refinement strategies could be applied to ranking tasks such as recommender systems.
- The theoretical analysis is rigorous, forming a complete proof chain from Lemmas 1–3 to Theorems 1–3, covering the over-smoothing upper bound, dual-pathway advantage, and coarse-to-fine validity.

## Limitations & Future Work
- The choice of coarse model currently relies on manual specification; if the coarse model is of poor quality (i.e., initial rankings are disordered), the Top-$k$ split may assign the correct answer to the low-score sub-table, degrading final performance.
- The method focuses solely on tail entity completion $(h,r,?)$; although other tasks can be converted to this form, direct multi-task joint training may be more efficient.
- The fusion parameter $\alpha$ is a global scalar with limited adaptability across different query types; a query-dependent $\alpha$ could potentially yield further improvements.
- Gains on transductive tasks are relatively modest (<2%), suggesting that score over-smoothing is a more severe issue in inductive settings.

## Related Work & Insights
- **vs. KnowFormer**: KnowFormer is the current SOTA hybrid method that stacks message passing and Transformer within a single stage. DuetGraph's dual-pathway design directly targets this single-pathway scheme, achieving comprehensive improvements across 12 inductive benchmark splits.
- **vs. RED-GNN / A\*Net**: These are purely message-passing methods; DuetGraph supplements long-range dependency modeling through the additional global pathway.
- **vs. TransE / RotatE**: Triple-based methods disregard graph topology; by integrating both local and global structural information, DuetGraph substantially outperforms them on transductive tasks as well.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual-pathway separated fusion and coarse-to-fine reasoning is novel, although the individual components (parallel pathways, cascaded filtering) have precedents in other domains.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 12 inductive subsets, 4 transductive datasets, triple classification, ablation studies, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with complete theoretical analysis; notation is dense and some proofs require careful cross-referencing.
- Value: ⭐⭐⭐⭐ Score over-smoothing is a genuine bottleneck in KG reasoning; the proposed solution is practical and theoretically grounded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](../../CVPR2026/graph_learning/graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[NeurIPS 2025\] GnnXemplar: Exemplars to Explanations -- Natural Language Rules for Global GNN Interpretability](gnnxemplar_exemplars_to_explanations_--_natural_language_rules_for_global_gnn_in.md)
- [\[NeurIPS 2025\] SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation](spot-trip_dual-preference_driven_out-of-town_trip_recommendation.md)

</div>

<!-- RELATED:END -->
