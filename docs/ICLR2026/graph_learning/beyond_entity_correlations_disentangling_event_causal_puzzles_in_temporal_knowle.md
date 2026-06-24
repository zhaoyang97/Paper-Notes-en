---
title: >-
  [Paper Note] Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs
description: >-
  [ICLR 2026][Graph Learning][Temporal Knowledge Graph] This paper proposes HEDRA, the first representation learning framework for heterogeneous causal disentanglement at the **event level** in Temporal Knowledge Graphs (TKGs). By using three modules—counterfactual detection, instrumental variable guidance, and evolutionary orthogonality—it sequentially strips away non-causal and pseudo-causal relations while separating dynamic and static causality…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Temporal Knowledge Graph"
  - "Event Prediction"
  - "Causal Disentanglement"
  - "Instrumental Variable"
  - "Structural Causal Model"
date: 2026-05-08
content_hash: 42a36265044c1950
---

# Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RdoXks7VmJ](https://openreview.net/forum?id=RdoXks7VmJ)  
**Code**: To be confirmed  
**Area**: Temporal Knowledge Graph / Graph Representation Learning / Causal Disentanglement  
**Keywords**: Temporal Knowledge Graph, Event Prediction, Causal Disentanglement, Instrumental Variable, Structural Causal Model  

## TL;DR
This paper proposes HEDRA, the first representation learning framework for heterogeneous causal disentanglement at the **event level** in Temporal Knowledge Graphs (TKGs). By using three modules—counterfactual detection, instrumental variable guidance, and evolutionary orthogonality—it sequentially strips away non-causal and pseudo-causal relations while separating dynamic and static causality, achieving SOTA on five real-world datasets.

## Background & Motivation
**Background**: TKGs consist of event quadruplets $(s, r, o, t)$. Event prediction tasks aim to predict future relations between entities based on historical event sequences. Prevailing methods (RE-GCN, TiRGN, DECRL, etc.) generally use GCN+RNN to model **entity- or relation-level correlations**, or use derived structures like hypergraphs and evolutionary clusters to capture high-order correlations.

**Limitations of Prior Work**: TKG datasets (e.g., ICEWS international political event database) are inherently constructed from events, which naturally contain **heterogeneous causal relationships**. Focusing solely on entity/relation-level correlations fails to characterize the true causal drivers between events, which is insufficient for event prediction.

**Key Challenge**: The authors decompose event-level causality into four categories: **Static Causality** (time-invariant causal dependencies, e.g., IAEA safety assessments providing an institutional framework for Japan's nuclear wastewater discharge), **Dynamic Causality** (causality evolving with timestamps, e.g., China announcing a ban on Japanese seafood imports on the day of discharge), **Non-Causality** (irrelevant to prediction, e.g., a BRICS summit), and **Pseudo-Causality** (misleading spurious correlations, e.g., Typhoon "Lan" landing in Japan disrupting traffic; over-focusing on this leads the model to misattribute export changes to the typhoon rather than policy). The challenge lies in the lack of explicit supervisory signals in existing TKGs to distinguish these four categories, making their identification and estimation from observed data non-trivial.

**Goal**: Establish an event-level TKG Structural Causal Model (SCM) as a theoretical framework and design a practical disentanglement mechanism to retain dynamic/static causality while eliminating non/pseudo-causality, thereby learning more discriminative representations.

**Core Idea**:
- **Theoretical Foundation**: Propose an event-level TKG-SCM, using backdoor adjustment and $do$-calculus to formally define the four types of causality, identifying $N$ and $P$ as confounders, while $D$ and $S$ serve as mutual confounders.
- **Sequential Disentanglement**: Use three modules to block backdoor paths—a counterfactual detector to strip non-causality, IV guidance to strip pseudo-causality, and evolutionary orthogonality to separate dynamic/static causality.

## Method

### Overall Architecture
At each timestamp, HEDRA first uses a relation-aware GCN to update entity/relation representations and construct event representations. Subsequently, three modules are cascaded: the **Counterfactual Detection Module** (CDM) generates non-causal masks and contrastive loss via event importance and distribution differences; the **IV-guided Disentanglement Module** (IVDM) constructs instrumental variable scores to partition edges into true/pseudo-causality with multi-view subgraph propagation and robust loss; the **Evolutionary Orthogonality Module** (EOM) projects true causal representations into static/dynamic components using evolutionary loss to maintain temporal dependence for dynamic components and temporal independence for static ones. Finally, static/dynamic causal event graphs are constructed, refined via event GCNs, and processed by a ConvTransE decoder for relation prediction.

```mermaid
flowchart LR
    A[Quadruplets s,r,o,t] --> B[Relation-aware GCN<br/>Event Representation Construction]
    B --> C[CDM<br/>Counterfactual Detector<br/>Strip Non-causality M^NC]
    C --> D[IVDM<br/>IV-guided<br/>Strip Pseudo-causality M^P]
    D --> E[EOM<br/>Evolutionary Orthogonality<br/>Separate Dynamic/Static]
    E --> F[Event GCN Refinement<br/>ConvTransE Decoding]
    F --> G[Event Prediction p r̂|s,o]
```

### Key Designs

**1. Counterfactual Detector: Jointly identifies non-causality using importance and distribution differences.** To avoid quadratic complexity of full event pairs, kNN is used to construct a candidate graph $C$ in the representation space. For each candidate edge $i \to j$, the model calculates attention-style **event importance** $A_{ij}$ (higher importance indicates higher causal dependency) and maps each event to a diagonal Gaussian posterior $q_i = \mathcal{N}(\mu_i, \mathrm{diag}(\sigma_i^2))$, using KL divergence to measure **distribution difference** $D_{ij}$ (larger difference suggests lower causality). These fuse into a soft non-causal mask $M^{NC} = 1 - \sigma\big((\alpha_{attn} \cdot \mathrm{logit}(A + \varepsilon) - \beta_{KL} \cdot D) \odot C\big)$, where $\alpha_{attn} + \beta_{KL}$ is fixed at 0.5. A contrastive loss $L_{con}$ pulls pairs with low non-causal weights closer and pushes those with high weights away.

**2. IV-guided Disentanglement: Separates true and pseudo-causality using instrumental variables and multi-view robustness.** After stripping non-causality, remaining edges still confound true and pseudo-causality. Borrowing from instrumental variable (IV) theory, an IV encoder $f_{IV}$ outputs an IV score $\Pi_{ij} = f_{IV}(h^t_{event,i}, h^t_{event,j}, \mathrm{logit}(A_{ij}+\varepsilon), -D_{ij})$. Since $\Pi_{ij}$ does not enter the final scoring function, it forms a neural analogy of the "exclusion restriction," approximately satisfying the IV independence assumption. Based on the gated matrix $\tilde\Pi = M^C \odot \Pi$, a threshold $\alpha$ selects the top edges as the true causal mask $M^P$. To handle imperfect IV estimation, three subgraphs (true, pseudo, full) are used for heterogeneous convolution, constrained by a robust loss $L_{rob} = \lambda_{align}L_{align} + \lambda_{sep}L_{sep}$—aligning the full view with the true causal view while pushing the pseudo view away.

**3. Evolutionary Orthogonality: Separates static and dynamic causality via orthogonal projection.** True causal representations are projected into static components $h^S$ and raw dynamic components via two MLP encoders. The raw dynamic component is orthogonalized against the static component via Gram-Schmidt to obtain a pure dynamic component $h^{D,t}_{event,i} = h^{raw,D,t}_{event,i} - \frac{\langle h^{raw,D},h^{S}\rangle}{\|h^{S}\|_2^2+\varepsilon}h^{S}$. An evolutionary loss $L_{evo} = \lambda_{dyn}L_{dyn} + \lambda_{stat}L_{stat}$ is applied: the dynamic term uses GRU to maintain temporal dependence, while the static term uses historical means to maintain temporal independence.

**4. Backdoor Adjustment Support.** The process corresponds to blocking backdoor paths in the TKG-SCM: estimating $C \to Y$ requires adjusting for confounders $N, P$, and estimating $D \to Y$ requires adjusting for $S$, formalized as:
$$P(Y|do(D)) = \sum_S P(Y|do(D),S)P(S|do(D))$$
The three modules serve as practical implementations of $do$-calculus adjustments on TKGs.

## Key Experimental Results

### Main Results (ICEWS14 / ICEWS18, MRR & Hits)

| Method | ICEWS14 MRR | Hits@1 | Hits@3 | Hits@10 | ICEWS18 MRR | Hits@1 |
|---|---|---|---|---|---|---|
| TiRGN (IJCAI 2022) | 41.28 | 29.52 | 46.69 | 70.66 | 42.26 | 30.19 |
| DHyper (TOIS 2024) | 41.71 | 29.37 | 45.69 | 69.32 | 42.84 | 29.96 |
| DECRL (NeurIPS 2024) | 42.90 | 30.49 | 47.06 | 70.01 | 43.36 | 30.64 |
| **HEDRA (Ours)** | **47.86** | **35.28** | **53.32** | **75.65** | **46.77** | **33.66** |
| Gain | +11.56% | +15.71% | +13.30% | +2.16% | +7.86% | +9.86% |

On average, HEDRA outperforms the runner-up across five datasets by 5.70% in MRR and 7.51% in Hits@1.

### Ablation Study (ICEWS14)

| Variant | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| HEDRA-w/o-CDM | 47.11 | 34.25 | 52.12 | 75.04 |
| HEDRA-w/o-IVDM | 46.47 | 33.77 | 51.65 | 74.75 |
| HEDRA-w/o-EOM | 46.24 | 33.49 | 51.79 | 74.10 |
| **HEDRA (Full)** | **47.86** | **35.28** | **53.32** | **75.65** |

### Key Findings
- **IVDM and EOM contribute the most**: Removing these modules causes the most significant performance drops, indicating that stripping pseudo-causality and separating dynamic/static causality are the primary drivers of gain.
- **Robustness to Hyperparameters**: The model is insensitive to history window lengths but sensitive to the number of neighbors in the candidate graph. Coefficients like $\alpha_{attn}$ and $\lambda_{align}$ can be fixed at 0.5 without per-dataset tuning.
- **Case Study**: In samples like Obama-Xi or HK Police-Protesters, HEDRA's top-5 predictions hit more correct relations than DECRL, showing that causal disentanglement improves high-ranking accuracy.

## Highlights & Insights
- **First TKG work to push causal disentanglement to the "event-level"**: It moves beyond the inertia of entity/relation correlations, treating heterogeneous causality between events as a first-class citizen.
- **Elegant alignment between theory and implementation**: The framework starts with TKG-SCM and backdoor adjustment and maps them to three sequential modules, rather than being a "stack of modules with a post-hoc story."
- **Innovative use of Instrumental Variables**: Using an IV score without including it in the final scoring function to approximate exclusion restrictions is a valuable migration of econometric ideas to unsupervised causal disentanglement.

## Limitations & Future Work
- **Lack of explicit causal supervision**: Disentanglement relies on unsupervised signals and assumptions (IV independence). The threshold $\alpha$ for partitioning true/pseudo-causality is a design choice that is difficult to verify directly.
- **Computational Overhead**: The kNN candidate graph, multi-view heterogeneous convolutions, and cascaded modules increase runtime. Its scalability on massive graphs like GDELT warrants further investigation.
- **Interpretability Gaps**: While case studies are provided, there is a lack of quantitative explanation or diagnostic visualization regarding exactly which pseudo-causal edges are stripped.

## Related Work & Insights
- **TKG Representation Learning**: RE-GCN, TiRGN represent the GCN+RNN mainlime; EvoExplore and DECRL build high-order correlations. This paper identifies that they remain at the entity/relation correlation level.
- **Graph Causal Learning**: Static graph causality focuses on explainable subgraphs (GNNExplainer); dynamic work explores spatio-temporal causality. This paper fills the gap in pseudo-causality and event-level frameworks.
- **Insight**: The combination of causal disentanglement, instrumental variables, and orthogonal projection could be generalized to other scenarios like time-series de-biasing in recommendation systems or influence attribution in dynamic social networks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐
- **Experimental Thoroughness**: ⭐⭐⭐⭐
- **Writing Quality**: ⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inductive Reasoning for Temporal Knowledge Graphs with Emerging Entities](inductive_reasoning_for_temporal_knowledge_graphs_with_emerging_entities.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](revisting_node_affinity_prediction_in_temporal_graphs.md)
- [\[ICLR 2026\] TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs](tgm_a_modular_and_efficient_library_for_machine_learning_on_temporal_graphs.md)
- [\[ICLR 2026\] UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction](urbangraph_physics-informed_spatio-temporal_dynamic_heterogeneous_graphs_for_urb.md)

</div>

<!-- RELATED:END -->
