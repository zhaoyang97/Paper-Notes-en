---
title: >-
  [Paper Note] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning
description: >-
  [NeurIPS 2025][Graph Learning][Knowledge Graphs] This paper proposes MoEMeta, a framework that employs a Mixture-of-Experts model to learn globally shared relational prototypes for cross-task generalization…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Knowledge Graphs"
  - "Few-Shot Relational Learning"
  - "Meta-Learning"
  - "Mixture of Experts"
  - "Task Adaptation"
date: 2026-05-08
content_hash: 73312d89cc20e283
---

# MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.23013](https://arxiv.org/abs/2510.23013)
**Code**: [GitHub](https://github.com/alexhw15/MoEMeta)
**Area**: Graph Learning
**Keywords**: Knowledge Graphs, Few-Shot Relational Learning, Meta-Learning, Mixture of Experts, Task Adaptation

## TL;DR

This paper proposes MoEMeta, a framework that employs a Mixture-of-Experts model to learn globally shared relational prototypes for cross-task generalization, combined with a task-customized projection adaptation mechanism to capture local context, achieving state-of-the-art performance on three KG benchmarks.

## Background & Motivation

Few-shot relational learning on knowledge graphs (FSRL) aims to reason about novel relations given only a handful of training triples. Existing MAML-based methods suffer from two critical limitations:

**Neglect of cross-task shared patterns**: These methods assume meta-training tasks are i.i.d. and learn meta-knowledge in isolation per task. However, relations in KGs naturally form semantic clusters—for example, *FatherOfPerson* and *BrotherOf* share a "family bond" theme, while *ColorOf* belongs to "physical attributes." Ignoring such cross-task commonalities impedes generalization.

**Lack of flexibility in global parameters**: Using a single global initialization with gradient-based adaptation fails to accommodate the diverse interaction patterns in KGs (1-1, 1-N, N-1, N-N). For instance, Elon Musk manifests entirely different aspects under *CeoOf* versus *FatherOfPerson*, and a shared initialization struggles to capture such divergent local contexts.

**Core challenge**: How to disentangle globally shared knowledge from task-specific context while enabling effective generalization and rapid adaptation?

## Method

### Overall Architecture

MoEMeta consists of three core components: (1) an attentive neighbor aggregation module that enriches entity representations; (2) a MoE meta-knowledge learner that dynamically selects experts to generate relational meta-representations; and (3) a task-customized projection mechanism for local adaptation. Global parameters $\bm{\Phi}$ are optimized in the outer loop, while local parameters $\bm{\eta}$ are independently optimized per task in the inner loop.

### Key Designs

1. **Attentive Neighbor Aggregation**: Enhances target entity representations by incorporating relational and entity information from first-order neighbors.

   For each neighbor tuple $(r_i, e'_i)$ of entity $e$, the method first concatenates and transforms:
   $\mathbf{c}'_i = \text{ReLU}(\mathbf{W} \mathbf{c}_i)$

   A sigmoid gate selects informative neighbors:
   $g_i = \sigma(\bm{\beta}^T \mathbf{c}'_i)$

   The weighted aggregation is added to the entity's own embedding:
   $\mathbf{e} = \text{Aggregate}(\{g_i \cdot \mathbf{c}'_i\}_{i=1}^n) + \mathbf{e}$

   **Design Motivation**: Entity semantics in KGs are highly context-dependent; neighbor aggregation provides richer entity representations.

2. **MoE Meta-Knowledge Learning**: Learns composable relational prototypes via a globally shared sparse MoE.

   For each support triple, the gating network computes relevance scores over experts:
   $s_{i,j} = \text{softmax}(\text{Gate}(\mathbf{h}_i, \mathbf{t}_i; \bm{\theta}_g))$

   The Top-$N$ experts are selected and combined to produce the relational representation:
   $\mathbf{r}_i = \frac{1}{N} \sum_{j=1}^{M} g_{i,j} f_j(\mathbf{h}_i, \mathbf{t}_i; \bm{\theta}_j)$

   The final relational meta-representation is the mean over all support triples:
   $\mathbf{R}_{\mathcal{T}_r} = \frac{1}{K} \sum_{i=1}^{K} \mathbf{r}_i$

   **Design Motivation**: Different relations can be represented as different combinations of shared "relational building blocks" (experts). Sparse activation encourages expert specialization, and semantically similar relations naturally activate similar expert subsets.

3. **Task-Customized Local Adaptation**: Projects embeddings into task-specific subspaces via learnable projection vectors.

   Each task maintains three projection vectors $\mathbf{p}_h, \mathbf{p}_r, \mathbf{p}_t$, modulated by the relational meta-representation:
   $\mathbf{h}'_i = \mathbf{h}_i + (\mathbf{p}_h^\top \mathbf{R}_{\mathcal{T}_r}) \cdot \mathbf{R}_{\mathcal{T}_r}$
   $\mathbf{R}'_{\mathcal{T}_r} = \mathbf{R}_{\mathcal{T}_r} + (\mathbf{p}_r^\top \mathbf{R}_{\mathcal{T}_r}) \cdot \mathbf{R}_{\mathcal{T}_r}$
   $\mathbf{t}'_i = \mathbf{t}_i + (\mathbf{p}_t^\top \mathbf{R}_{\mathcal{T}_r}) \cdot \mathbf{R}_{\mathcal{T}_r}$

   **Design Motivation**: Inspired by TransD, different relations impose different constraints. Projection vectors achieve task-specific shifts in embedding space with minimal parameter overhead, avoiding overfitting.

### Loss & Training

The scoring function uses $\ell_2$ distance: $\text{score}(h_i, t_i) = \|\mathbf{h}'_i + \mathbf{R}'_{\mathcal{T}_r} - \mathbf{t}'_i\|_2$

**Inner loop** (adaptation on the support set): A margin-based loss updates $\bm{\eta}$ and $\mathbf{R}_{\mathcal{T}_r}$:
$$\mathcal{L}(\mathcal{S}_r) = \sum_{(h_i,r,t_i) \in \mathcal{S}_r} \max\{0, \text{score}(h_i, t_i) + \gamma - \text{score}(h_i, t'_i)\}$$

**Outer loop** (meta-update on the query set): Query loss is back-propagated to update global parameters $\bm{\Phi}$.

## Key Experimental Results

### Main Results (Nell-One)

| Method | 1-shot MRR | 1-shot Hits@1 | 5-shot MRR | 5-shot Hits@1 |
|--------|-----------|--------------|-----------|--------------|
| MetaR-P | 0.164 | 0.093 | 0.209 | 0.141 |
| GANA | 0.236 | 0.173 | 0.245 | 0.166 |
| HiRe | 0.288 | 0.184 | 0.306 | 0.207 |
| **MoEMeta** | **0.322** | **0.228** | **0.339** | **0.236** |
| Gain | +11.8% | +23.9% | +10.8% | +14.0% |

### FB15K-One (3-shot)

| Method | MRR | Hits@1 | Hits@10 |
|--------|-----|--------|---------|
| RelAdapter | 0.405 | 0.297 | 0.575 |
| **MoEMeta** | **0.423** | **0.302** | **0.651** |
| Gain | +4.4% | +0.3% | +13.2% |

### Ablation Study (Nell-One)

| Configuration | 1-shot MRR | 5-shot MRR | Note |
|---------------|-----------|-----------|------|
| MoEMeta | 0.322 | 0.339 | Full model |
| w/o N.A | 0.311 | 0.328 | Remove neighbor aggregation |
| w/o MoE | 0.291 (↓9.6%) | 0.293 (↓13.6%) | Replace MoE with MLP |
| w/o L.A | 0.301 | 0.315 | Remove local adaptation |

### Key Findings

1. The MoE module is the most critical component—its removal causes MRR to drop by up to 13.6%, underscoring the importance of cross-task shared pattern learning.
2. Gating value visualizations reveal that semantically similar relations (e.g., family-type relations) activate similar expert subsets, while dissimilar relations activate distinct experts.
3. Local adaptation yields the greatest improvements for N-1 and N-N relation types, with MRR gains of 8.1% and 8.6%, respectively.
4. Hyperparameter analysis indicates that 32 experts with Top-5 selection is optimal.

## Highlights & Insights

1. **Global–local decoupling**: The MoE component handles "what is generalizable," while the projection handles "how to adapt"—the two mechanisms are complementary.
2. **Relational prototype composition**: Treating relations as combinations of fundamental patterns rather than independently learned entities better reflects the semantic structure of KGs.
3. **Lightweight adaptation**: Only three projection vectors (scalar-scale parameters) suffice for effective task adaptation, mitigating few-shot overfitting.

## Limitations & Future Work

- Only tail entity prediction $(h, r, ?)$ is addressed; head entity prediction is not considered.
- Reliance on TransE initialization may limit coverage of certain complex relational patterns.
- Expert networks are relatively simple (two-layer MLPs); more expressive experts may yield further gains.
- Transfer learning across datasets remains unexplored.

## Related Work & Insights

- **Meta-learning FSRL**: MetaR, HiRe, GANA, RelAdapter
- **Metric-learning FSRL**: GMatching, FAAN, NP-FKGC
- **MoE architectures**: Sparse Gated MoE, Switch Transformer
- **KG embeddings**: TransE, TransD, DistMult, ComplEx

## Rating

- Novelty: ⭐⭐⭐⭐ — Employing MoE as a meta-learner for relational prototype learning is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, 16 baselines, ablations, visualizations, and relation-type analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated and method design is well-justified.
- Value: ⭐⭐⭐⭐ — Strong practical utility for few-shot KG reasoning; the framework is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[NeurIPS 2025\] Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](learning_repetition-invariant_representations_for_polymer_informatics.md)

</div>

<!-- RELATED:END -->
