---
title: >-
  [Paper Note] A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning
description: >-
  [ACL 2025][Graph Learning][Temporal Knowledge Graph Reasoning] This paper proposes the Deep Generative Adaptive Replay (DGAR) method, which utilizes a pre-trained diffusion model to generate historical entity distribution representations, mitigates distribution conflicts by enhancing shared features between the historical and current distributions, and designs a layer-wise adaptive replay mechanism to integrate historical and current knowledge…
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Temporal Knowledge Graph Reasoning"
  - "Continual Learning"
  - "Catastrophic Forgetting"
  - "Diffusion Models"
  - "Adaptive Replay"
date: 2026-05-08
content_hash: 39720665f7d29c28
---

# A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning

**Conference**: ACL 2025  
**arXiv**: [2506.04083](https://arxiv.org/abs/2506.04083)  
**Code**: None  
**Area**: Graph Learning / Knowledge Graphs  
**Keywords**: Temporal Knowledge Graph Reasoning, Continual Learning, Catastrophic Forgetting, Diffusion Models, Adaptive Replay

## TL;DR

This paper proposes the Deep Generative Adaptive Replay (DGAR) method, which utilizes a pre-trained diffusion model to generate historical entity distribution representations, mitigates distribution conflicts by enhancing shared features between the historical and current distributions, and designs a layer-wise adaptive replay mechanism to integrate historical and current knowledge, significantly alleviating the catastrophic forgetting problem in continual learning scenarios for temporal knowledge graph reasoning.

## Background & Motivation

**Background**: Temporal Knowledge Graphs (TKGs) extend traditional knowledge graphs by associating timestamps with triples, providing dynamic time-sensitive knowledge for downstream applications such as event prediction, financial forecasting, and large language model reasoning. Temporal Knowledge Graph Reasoning (TKGR) utilizes historical knowledge to infer missing temporal facts. In real-world scenarios, TKGs are continuously updated—new entities, relations, and facts constantly emerge. Early methods required full retraining when new data arrived, which is computationally expensive. Continual Learning (CL) reduces costs by fine-tuning models only on new data, but faces the problem of catastrophic forgetting.

**Limitations of Prior Work**: Current CL-based TKGR methods have two key limitations: (1) they typically reorganize and replay individual historical facts in a one-sided manner (e.g., based on frequency or clustering), neglecting the historical context information necessary for a proper semantic understanding of historical facts; (2) they preserve historical knowledge by simply replaying historical facts, but ignore possible conflicts between historical and new data distributions—over time, the same entity associates with different neighbors, leading to differences in its semantic representation distribution at different times.

**Key Challenge**: There is a fundamental tension between historical knowledge retention and new knowledge learning. Simply replaying historical data not only fails to guarantee semantic integrity (due to lack of context) but can also backfire due to distribution conflicts—forcing the model to fit highly disparate historical and current distributions simultaneously leads to representation confusion.

**Goal**: (1) Design a replay strategy that retains complete historical context semantics; (2) address the conflict between historical and current distributions.

**Key Insight**: Rather than replaying discrete historical facts, it is better to generate distribution representations of historical entities. Utilizing the powerful generative capabilities of diffusion models, historical context prompts (instead of single facts) are used to generate entity distribution representations that balance historical semantics and current compatibility.

**Core Idea**: Use diffusion models to generate historical entity distributions, enhance the common features of historical and current distributions through the gradient guidance of the current TKGR model during the generation process, and inject historical distributions into current representations via layer-wise deep adaptive replay.

## Method

### Overall Architecture

DGAR consists of three core components: Historical Context Prompt Building (HCP Building), Diffusion-enhanced Historical Distribution Generation (Diff-HDG), and Deep Adaptive Replay (DAR). When a new query for the $t$-th task arrives, HCP is first constructed based on the query entity; then, based on HCP, a pre-trained diffusion model is used to generate historical entity distribution representations guided by the current TKGR model; finally, DAR is used to inject the historical distributions into the current distribution representations to support reasoning.

### Key Designs

1. **Historical Context Prompt Building (HCP Building)**:

    - **Function**: Construct replay data sampling units that preserve complete historical context semantics for each affected entity.
    - **Mechanism**: For a query entity $e_q$, collect the set of all triples involving it across historical timestamps as an HCP unit $Prompt_{replay}^i = \{(s, r, e_q) | (s, r, e_q) \in G_i\}$. To control computational costs, randomly select HCPs from $k$ different time slices as the final replay data. Unlike traditional ER methods that only replay isolated facts, HCP takes the complete context structure as the sampling unit, ensuring the integrity of historical semantics.
    - **Design Motivation**: Isolated historical facts lack semantic completeness without context. HCP preserves all associations of an entity at a specific point in time, enabling more accurate historical distribution generation. Randomly selecting different time slices rather than the most recent ones provides more global historical information.

2. **Diffusion-enhanced Historical Distribution Generation (Diff-HDG)**:

    - **Function**: Generate historical entity distribution representations compatible with the current distribution based on HCP and a pre-trained diffusion model.
    - **Mechanism**: Use neighboring entities and relations in the HCP as generation conditions, $X_n = \text{Condition}(S_0, R_0, Z)$, where $Z \sim \mathcal{N}(0, I)$. In each denoising step, use the gradient of the current TKGR model $f_{\theta_t}$ to enhance the shared features of the historical and current distributions: $X_{n-1} = X_{n-1} + \gamma \frac{\partial \sigma}{\partial X_{n-1}}$, where $\sigma$ is the softmax function used to evaluate the scores of historical facts under the current model. Finally, the generation results of multiple time slices and multiple neighbors are aggregated via mean pooling to obtain the historical distribution representation $H_e^{replay}$ of the entity.
    - **Design Motivation**: Pre-trained diffusion models may generate historical distributions that conflict with the current distribution. Through gradient guidance from the current model, common features of the two distributions are enhanced while different features are weakened during generation, fundamentally mitigating the distribution conflict problem.

3. **Deep Adaptive Replay (DAR)**:

    - **Function**: Inject generated historical distribution representations into current entity representations, executed layer-by-layer in each GNN layer.
    - **Mechanism**: In each graph neural network layer of the base TKGR model (e.g., RE-GCN), perform distribution fusion for entities participating in replay $e \in V_{replay}$: $H_e^l = \alpha H_e^{replay} + (1-\alpha) H_e^{current,l}$, where $\alpha \in [0,1]$ adaptively balances the weights of historical and new knowledge. For entities not requiring replay, maintain their original current distribution.
    - **Design Motivation**: Simple one-off injection (such as direct addition) can degrade performance, while overly complex fusion increases the learning burden. Layer-wise deep injection allows historical information to be iteratively integrated with current features as layers progress, while incorporating temporal evolution features. The adaptive parameter $\alpha$ avoids over-suppression of historical or current information caused by fixed weights.

### Loss & Training

The total loss is $\mathcal{L}_t = \mathcal{L}_{t,c} + \mu \mathcal{L}_{t,r}$, where $\mathcal{L}_{t,c}$ is the multi-class classification loss based on current facts (entity prediction), $\mathcal{L}_{t,r}$ is the regularization loss of historical facts in HCP (calculated similarly to $\mathcal{L}_{t,c}$), and $\mu$ is typically set to 1. The diffusion model is trained in a continual learning manner, inheriting parameters from the previous step and updating on the current data at each time step.

## Key Experimental Results

### Main Results

| Dataset | Metric | DGAR | TIE | ER | IncDE | Gain (vs Best) |
|--------|------|------|-----|-----|-------|-------------|
| ICE14 | Current MRR | 58.59 | 53.74 | 48.75 | 45.03 | +9.0% |
| ICE14 | Average MRR | 50.12 | 41.07 | 42.14 | 36.57 | +18.9% |
| ICE18 | Current MRR | 36.53 | 34.45 | 30.39 | 31.83 | +6.0% |
| ICE18 | Average MRR | 33.00 | 28.73 | 27.20 | 25.52 | +14.9% |
| ICE05-15 | Current MRR | 66.01 | 60.77 | 52.50 | 46.33 | +8.6% |
| ICE05-15 | Average MRR | 54.33 | 42.56 | 45.55 | 40.56 | +19.3% |
| GDELT | Current MRR | 23.25 | 15.56 | 15.42 | 15.14 | +49.4% |
| GDELT | Average MRR | 28.30 | 16.40 | 16.21 | 15.49 | +72.6% |

### Ablation Study

| Configuration | ICE14 Avg MRR | ICE18 Avg MRR | ICE05-15 Avg MRR | GDELT Avg MRR |
|------|--------------|--------------|-----------------|--------------|
| DGAR (Full) | 50.12 | 33.00 | 54.33 | 28.30 |
| w/o HP (No HCP, use ER instead) | 46.74 | 25.89 | 45.71 | 17.18 |
| w/o GR (No DAR+Diff-HDG) | 39.15 | 27.62 | 38.71 | 16.67 |
| w/o AR (Direct addition instead of DAR) | 45.55 | 29.79 | 51.93 | 26.98 |
| w/o Guider (No guided enhancement) | 49.32 | 32.25 | 52.53 | 27.99 |
| w/o $\mathcal{L}_r$ (No historical regularization) | 44.43 | 30.62 | 51.98 | 25.95 |

### Key Findings

- DGAR consistently outperforms all baselines across all datasets. The improvement in Average MRR on the GDELT dataset is the most impressive (+72.6%), indicating that generative replay is highly effective in scenarios with extremely high update frequencies (GDELT updates every 15 minutes).
- In the ablation study, w/o GR (removing both DAR and Diff-HDG) caused the largest drop in performance, showing that the distribution conflict resolution mechanism is the most critical contribution.
- The contribution of HCP is also significant—replacing it with standard ER resulted in an average drop of 3-11 MRR points across datasets, validating the importance of contextual integrity for accurate historical distribution generation.
- The contribution of the gradient guider varies across datasets—it is smaller on GDELT (due to small shifts between adjacent time steps) and larger on datasets with longer time spans, such as ICE14.
- In terms of inference time, DGAR (k=35) takes about 5.42 seconds per task on ICE14, which is much lower than the 553.48 seconds required for full retraining, indicating that the computational overhead is within an acceptable range.

## Highlights & Insights

- Using diffusion models to "generate" historical distributions rather than "store" historical data represents a qualitative upgrade to the experience replay paradigm in continual learning. While traditional methods store and replay raw data points, DGAR generates more generalizable distribution representations, which not only saves storage but also yields better representations.
- Gradient-guided generation is the most elegant design in this paper—introducing the gradient of the task model during diffusion denoising to shape the generated results allows the historical and current distributions to "coexist harmoniously." This idea can be transferred to any generative scenario requiring distribution alignment.
- The layer-wise DAR injection mechanism is more elegant than one-off fusion, allowing historical information to step-by-step integrate into the current representation at different levels of abstraction.

## Limitations & Future Work

- For newly emerging entities and relations, only Xavier initialization is currently used without dedicated modeling strategies, which may limit the model's ability to learn new knowledge.
- Introducing the diffusion model as an extra component increases model complexity and training parameter size.
- The sample size $k$ in HCP needs to be tuned for different datasets, and adaptively determining the optimal $k$ remains an open problem.
- Future work could explore extending DGAR to rule-based TKGR methods or knowledge graph completion (KGC) tasks.

## Related Work & Insights

- **vs TIE**: TIE uses strict regularization to prevent forgetting, but this restricts the model's freedom to learn new knowledge, leading to poor performance on historical tasks. DGAR replaces regularization constraints with generative replay, balancing both old and new knowledge.
- **vs ER (Experience Replay)**: Standard ER only replays isolated historical facts. DGAR preserves contextual integrity via HCP and addresses distribution conflicts via Diff-HDG, substantially outperforming ER across all datasets.
- **vs IncDE**: IncDE and LKGE preserve historical knowledge solely through embedding constraints, whereas DGAR's distribution-level generation and deep fusion offer stronger representation capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing diffusion models to continual learning experience replay is a completely new paradigm, and the gradient-guided distribution generation is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, extensive ablations, verification using multiple base models, efficiency analysis, and hyperparameter sensitivity analyses are all thoroughly executed.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the experimental analysis is detailed.
- Value: ⭐⭐⭐⭐ Provides a powerful new baseline for TKGR continual learning, and the distribution conflict resolution concept is widely applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation](../../NeurIPS2025/graph_learning/elastic_weight_consolidation_for_knowledge_graph_continual_learning_an_empirical.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ACL 2025\] Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning](disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re.md)
- [\[ICLR 2026\] Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning](../../ICLR2026/graph_learning/knowledge_reasoning_language_model_unifying_knowledge_and_language_for_inductive.md)

</div>

<!-- RELATED:END -->
