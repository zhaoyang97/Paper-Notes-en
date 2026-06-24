---
title: >-
  [Paper Note] Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation
description: >-
  [NeurIPS 2025 (NORA Workshop)][Graph Learning][Continual Learning] This paper systematically evaluates Elastic Weight Consolidation (EWC) for continual learning of TransE knowledge graph embeddings on FB15k-237, finding that EWC reduces catastrophic forgetting from 12.62% to 6.85% (a 45.7% reduction), and reveals that task partitioning strategy (relation-based vs. random) has a substantial impact on forgetting metrics (a difference of 9.8 percentage points).
tags:
  - "NeurIPS 2025 (NORA Workshop)"
  - "Graph Learning"
  - "Continual Learning"
  - "Knowledge Graphs"
  - "Elastic Weight Consolidation"
  - "Catastrophic Forgetting"
  - "Link Prediction"
date: 2026-05-08
content_hash: a2eb849d4de4151b
---

# Elastic Weight Consolidation for Knowledge Graph Continual Learning: An Empirical Evaluation

**Conference**: NeurIPS 2025 (NORA Workshop)  
**arXiv**: [2512.01890](https://arxiv.org/abs/2512.01890)  
**Code**: None  
**Area**: Graph Learning  
**Keywords**: Continual Learning, Knowledge Graphs, Elastic Weight Consolidation, Catastrophic Forgetting, Link Prediction

## TL;DR
This paper systematically evaluates Elastic Weight Consolidation (EWC) for continual learning of TransE knowledge graph embeddings on FB15k-237, finding that EWC reduces catastrophic forgetting from 12.62% to 6.85% (a 45.7% reduction), and reveals that task partitioning strategy (relation-based vs. random) has a substantial impact on forgetting metrics (a difference of 9.8 percentage points).

## Background & Motivation

**Background**: Knowledge graphs evolve continuously and require ongoing updates. Neural embedding models such as TransE learn vector representations for link prediction, but suffer from catastrophic forgetting during sequential task learning—performance on previous tasks degrades sharply.

**Limitations of Prior Work**: Continual learning methods have been extensively studied in image classification and NLP, yet empirical evaluation on KG link prediction remains insufficient. In particular, the effectiveness of the classical regularization approach EWC on KG embeddings lacks systematic validation.

**Key Challenge**: KG embeddings have a structured parameter space in which specific dimensions encode semantic attributes. It is unclear whether simple parameter protection suffices to preserve semantic structure. Moreover, the effect of task partitioning strategy on the measurement of catastrophic forgetting is not well understood.

**Goal**: (a) Can EWC effectively mitigate catastrophic forgetting in KG embeddings? (b) How does the task construction strategy affect forgetting metrics?

**Key Insight**: Using TransE on FB15k-237 as the testbed, the paper designs rigorous multi-seed experiments comparing EWC, naive sequential training, and experience replay, under both relation-based and random task partitioning strategies.

**Core Idea**: Through systematic empirical analysis, the paper demonstrates that EWC's Fisher Information Matrix regularization can effectively protect KG embedding parameters, while revealing that task partitioning strategy is an overlooked yet highly influential factor in continual learning evaluation.

## Method

### Overall Architecture
The KG $\mathcal{G}=(\mathcal{E},\mathcal{R},\mathcal{T})$ is divided into $T$ sequential tasks $\mathcal{G}_1,\dots,\mathcal{G}_T$. TransE embeddings are trained task-by-task, and the degree of forgetting on each task is evaluated. Three types of methods are assessed: naive sequential training, EWC regularization, and experience replay.

### Key Designs

1. **TransE Embedding Training**:

    - Function: Learns entity/relation vectors such that $\mathbf{h}+\mathbf{r}\approx\mathbf{t}$.
    - Mechanism: Minimizes the margin-based loss $\mathcal{L}=\sum_{(h,r,t)\in\mathcal{T}}\sum_{(h',r,t')\in\mathcal{T}'}\max(0, \gamma+d(\mathbf{h}+\mathbf{r},\mathbf{t})-d(\mathbf{h'}+\mathbf{r},\mathbf{t'}))$, with $\gamma=1.0$, L2 distance, and 50-dimensional embeddings.

2. **Elastic Weight Consolidation (EWC)**:

    - Function: Protects parameters important to previous tasks when learning new ones.
    - Mechanism: Adds a Fisher Information Matrix regularization term to the loss for task $i$: $\mathcal{L}^i_{\text{EWC}}=\mathcal{L}^i+\frac{\lambda}{2}\sum_k F_k(\theta_k-\theta^*_{k,i-1})^2$, where $F_k=\mathbb{E}_{(h,r,t)\sim\mathcal{G}_{i-1}}[(\frac{\partial\log p(y|x;\theta)}{\partial\theta_k})^2]$ is approximated using mini-batches (size 256) over all data from the previous task.
    - Design Motivation: The Fisher Information Matrix identifies parameters critical to encoding prior tasks, and a quadratic penalty prevents these parameters from drifting substantially, thereby preserving the semantic structure of the embedding space.

3. **Task Partitioning Strategies**:

    - **Relation-based Partitioning**: Relations are sorted by frequency and assigned to 4 tasks via round-robin, with approximately 59 relation types per task. All triples of a given relation reside in the same task, with no overlap of relations across tasks, inducing a large distributional shift.
    - **Random Partitioning**: The 272,115 training triples are randomly and evenly divided into 4 splits (approximately 68,000 each), with relations overlapping across tasks. Distributional shift is minimal.

### Forgetting Metrics
- Forgetting on task $j$ after learning task $i$: $F^j_i = M^j_j - M^j_i$ ($i>j$)
- Average forgetting: $\bar{F}=\frac{1}{T-1}\sum_{j=1}^{T-1}F^j_T$
- Evaluation metric: MRR (filtered ranking)

## Key Experimental Results

### Main Results (Relation-based Partitioning)

| Method | Forgetting (%) | Final MRR |
|------|-----------|-----------|
| Naive Sequential Training | 12.62 ± 0.35 | 0.206 ± 0.006 |
| EWC (λ=0.1) | 10.44 ± 0.26 | 0.229 ± 0.005 |
| EWC (λ=1.0) | 7.51 ± 0.44 | 0.250 ± 0.006 |
| **EWC (λ=10)** | **6.85 ± 0.33** | **0.242 ± 0.004** |
| EWC + Wave Replay | 9.91 ± 0.20 | 0.234 ± 0.005 |
| Random Replay | 13.78 ± 0.44 | 0.196 ± 0.006 |
| Wave Replay | 12.54 ± 0.14 | 0.216 ± 0.007 |

### Effect of Task Partitioning Strategy

| Partitioning Strategy | Naive Forgetting (%) | EWC Forgetting (%) | Difference |
|----------|---------------|-------------|------|
| Relation-based | 12.62 ± 0.35 | 6.85 ± 0.33 | — |
| Random | 2.81 ± 0.34 | 5.08 ± 0.22 | — |
| Difference | **9.81 pp** | 1.77 pp | Relation-based partitioning is substantially harder |

### EWC Hyperparameter Sensitivity

| λ | Relation-based Forgetting (%) | Random Forgetting (%) | Relation-based MRR |
|---|-----------------|-----------------|-------------|
| 0.1 | 10.44 | 2.88 | 0.229 |
| 1.0 | 7.51 | 3.88 | 0.250 |
| 10.0 | **6.85** | 5.08 | 0.242 |

### Key Findings
- **EWC is significantly effective**: Under relation-based partitioning, forgetting is reduced from 12.62% to 6.85%, a decrease of 45.7%. MRR also improves from 0.206 to 0.242.
- **Replay methods perform worse**: Random Replay exhibits a forgetting rate of 13.78%, exceeding that of naive sequential training, indicating that simple replay is less effective than parameter protection.
- **Task partitioning strategy has a large impact**: Relation-based vs. random partitioning differs by 9.8 percentage points under naive training, as relation-based partitioning induces greater distributional shift across tasks.
- **Optimal λ depends on task construction**: Relation-based partitioning requires strong regularization (λ=10), whereas random partitioning benefits from weaker regularization (λ=0.1). Overly strong regularization under random partitioning restricts necessary parameter adaptation.
- EWC substantially narrows the forgetting gap between the two partitioning strategies (from 9.81 pp to 1.77 pp), suggesting that effective continual learning methods can generalize across different task construction schemes.

## Highlights & Insights
- **The revelation of the task partitioning effect** is the most valuable finding of this paper: the same method can exhibit forgetting rate differences of up to 9.8 percentage points depending on the partitioning scheme. This underscores the need to explicitly report task construction methodology in continual learning evaluations, as results are otherwise incomparable—a finding transferable to all continual learning experimental designs.
- **The superiority of EWC over experience replay** is consistent with synaptic consolidation theory in neuroscience: for structured knowledge representations (KG embeddings), protecting parameters is more effective than replaying samples, since specific dimensions in KG embeddings encode specific semantic properties.
- The experimental design is rigorous, employing 5 random seeds with mean and standard deviation reporting, and is reproducible on consumer-grade GPUs.

## Limitations & Future Work
- **Single model and dataset**: Only TransE on FB15k-237 is evaluated; generalizability to more expressive embeddings such as RotatE and ComplEx, and to datasets such as WN18RR and YAGO, remains uncertain.
- **Only 4 tasks**: The dynamics of longer task sequences (10+) are unexplored, and forgetting may grow nonlinearly with the number of tasks.
- **Workshop paper scope**: The depth is limited; no comparison with dedicated KG continual learning benchmarks such as PS-CKGE is provided.
- The round-robin strategy for relation-based partitioning is a specific design choice; entity-based or domain-based partitioning may exhibit different patterns.
- Combining EWC with more advanced replay strategies (e.g., GEM, A-GEM) or architectural approaches (e.g., Progressive Networks) warrants future investigation.

## Related Work & Insights
- **vs. PS-CKGE (Zhao2025)**: PS-CKGE focuses on the effect of pattern shifts on forgetting, whereas this paper centers on empirical evaluation of classical EWC. The two are complementary: PS-CKGE provides a more comprehensive benchmark, while this paper offers a deeper analysis of regularization.
- **vs. Daruna2021**: That work evaluates KG continual learning in robotic manipulation tasks across multiple architectures (TransE/DistMult/ComplEx). This paper has a narrower but more in-depth focus (task partitioning analysis).
- **vs. Online EWC / SI**: This paper employs standard EWC without comparing online EWC or Synaptic Intelligence variants, which represent promising directions for extension.

## Rating
- Novelty: ⭐⭐⭐⭐ No methodological contribution (direct application of EWC), but the task partitioning effect finding has independent value
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-seed experimental design is rigorous, but limited to a single model and dataset
- Writing Quality: ⭐⭐⭐⭐⭐ Meets empirical paper writing standards with cautious conclusion statements
- Value: ⭐⭐⭐⭐ Appropriate as a workshop paper; the insight on task partitioning effects has reference value for the field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](../../ACL2025/graph_learning/a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)
- [\[ICML 2025\] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](../../ICML2025/graph_learning/from_rag_to_memory_non-parametric_continual_learning_for_large_language_models.md)
- [\[ICLR 2026\] G-Merging: Graph Models Merging for Parameter-Efficient Multi-Task Knowledge Consolidation](../../ICLR2026/graph_learning/g-merging_graph_models_merging_for_parameter-efficient_multi-task_knowledge_cons.md)
- [\[NeurIPS 2025\] MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)

</div>

<!-- RELATED:END -->
