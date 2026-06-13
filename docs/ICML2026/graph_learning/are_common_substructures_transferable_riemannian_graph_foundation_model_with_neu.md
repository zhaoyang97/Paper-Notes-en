---
title: >-
  [Paper Note] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles
description: >-
  [ICML 2026][Graph Learning][Graph Foundation Models] This paper redefines "transferable common substructures" in graph pre-training as behavioral invariance within the representation space. By constructing Gauge with neu…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Foundation Models"
  - "Riemannian Geometry"
  - "Neural Vector Bundles"
  - "Structural Transfer"
  - "Dirichlet Energy"
date: 2026-05-08
content_hash: 348f13c421537ae7
---

# Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles

**Conference**: ICML 2026  
**arXiv**: [2606.03270](https://arxiv.org/abs/2606.03270)  
**Code**: https://github.com/RiemannGraph/GAUGE  
**Area**: Graph Learning / Graph Foundation Models  
**Keywords**: Graph Foundation Models, Riemannian Geometry, Neural Vector Bundles, Structural Transfer, Dirichlet Energy

## TL;DR
This paper redefines "transferable common substructures" in graph pre-training as behavioral invariance within the representation space. By constructing Gauge with neural vector bundles, gated geometric flattening, and Dirichlet loss, the model achieves enhanced structural generalization for cross-domain few-shot transfer, zero-shot link prediction, and graph isomorphism tasks.

## Background & Motivation
**Background**: Graph Foundation Models (GFMs) aim to learn reusable structural patterns through pre-training, similar to language and vision foundation models, for transfer to new graphs and tasks. Current approaches generally fall into two categories: those leveraging LLMs for graphs with textual attributes, and those searching for discrete common substructures like motifs, trees, graphons, or structural vocabularies in pure structural graphs.

**Limitations of Prior Work**: Frequent occurrence of a discrete substructure does not equate to "functional transferability." The same local motif may play entirely different structural roles in different neighborhoods; simply matching shapes risks misidentifying context-dependent patterns as general knowledge. Furthermore, existing Riemannian graph learning often assumes extrinsic geometric priors (e.g., hyperbolic, spherical, or product spaces), making it difficult to characterize the intrinsic geometry induced by the graph model's representations themselves.

**Key Challenge**: What graph transfer truly needs to reuse is structural behavior rather than isolated shapes. If the representation mechanism of a certain substructure requires almost no adjustment in a new graph, it should manifest as a behavioral invariance. However, this invariance is hidden within the model's representation space and local neighborhood interactions, making it undetectable via discrete frequency counts.

**Goal**: The authors aim to answer the question, "Are common substructures truly transferable?" and provide a trainable model. This involves establishing a theoretical link between behavioral invariance and intrinsic geometric flatness, while designing a graph neural architecture capable of learning this geometry during pre-training to improve cross-graph transfer and structural identification.

**Key Insight**: The paper adopts vector bundles from Riemannian geometry as the representation language. Intuitively, the graph structure is placed on an abstract base manifold with a local fiber space attached to each node. If the parallel transport between adjacent fibers is close to an identity transformation, the local structural behavior is more stable and likely to transfer.

**Core Idea**: Use neural vector bundles to explicitly learn local coordinates and inter-fiber transport of graph representations, identifying geometrically flat and behaviorally invariant substructures by minimizing Dirichlet energy.

## Method
The proposed method consists of two layers: a theoretical and representation framework that formalizes local structural behavior as geometric objects on vector bundles, and the Gauge architecture that transforms these objects into a pre-trainable network and loss function.

### Overall Architecture
Given a graph $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$, Gauge first generates node global representations $\mathbf{z}_i$ using a graph encoder and then learns a set of local coordinates $\mathbf{Q}^{(i)}$ for each node. These local coordinates act as a basis for the node's fiber space, describing how the structure near that node unfolds in the representation space.

In each layer, the model constructs local fiber bases from the attention residuals of nodes and their neighborhoods, then calculates the pseudo-parallel transport $\mathbf{P}^{(i,j)}=(\mathbf{Q}^{(i)})^\top\mathbf{Q}^{(j)}$ between adjacent fibers. If $\mathbf{P}^{(i,j)}$ is close to an identity matrix, the local coordinate geometries of the two neighborhoods are compatible. Gauge more strongly aggregates the coordinates of these neighbors, thereby flattening homogeneous local geometry layer by layer.

During pre-training, Gauge does not rely on external task labels. Instead, it uses a Dirichlet loss to require that a node's local representation be predictable from its neighbors' local representations. Connected regions with low loss are identified as approximately behaviorally invariant regions—what the authors define as transferable structures. For downstream adaptation, the pre-trained backbone is frozen, and only the input encoding and task adaptation layers are fine-tuned to minimize disruption to learned structural mechanisms.

### Key Designs
1. **Neural Vector Bundle Representation**:
    - **Function**: Extends each node in the graph from a single embedding point to a "global representation + local fiber coordinates" to characterize the intrinsic geometry induced by the model.
    - **Core Idea**: For each node, the model learns an orthonormal basis $\mathbf{Q}^{(i)}$, whose transpose acts as a local trivialization map, projecting the global representation into the local coordinates of the node's fiber. Pseudo-parallel transport is established between neighbors via $\mathbf{P}^{(i,j)}=(\mathbf{Q}^{(i)})^\top\mathbf{Q}^{(j)}$.
    - **Design Motivation**: Traditional representations assign only one vector per node, failing to distinguish between "similar representation values" and "consistent local structural behavior." Vector bundles allow the model to compare coordinate systems of different neighborhoods, transforming transferability into a geometric compatibility problem.

2. **Gated Flattening Mechanism**:
    - **Function**: Selects geometrically compatible neighbors within network layers for local coordinate aggregation, making transferable regions progressively flatter.
    - **Core Idea**: The model calculates edge gate weights based on $\mathrm{Tr}(\mathbf{I}_r-\mathbf{P}^{(i,j)})$. The closer the parallel transport is to an identity transform, the more neighbor $j$'s local coordinates are absorbed by node $i$. QR decomposition is used after updates to maintain orthonormality of the local bases.
    - **Design Motivation**: Standard message passing does not distinguish whether neighbors share the same local geometry, often aggregating non-transferable contextual noise. Gated flattening restricts aggregation to geometrically consistent neighborhoods, aligning with the theoretical intuition that transferable structures should possess flat intrinsic geometry.

3. **Dirichlet Loss and Invariant Substructure Recovery**:
    - **Function**: Learns behaviorally invariant structures during unsupervised pre-training and provides a measure for transfer overhead.
    - **Core Idea**: Dirichlet energy measures local fiber differences such as $\|\mathbf{x}_i-\mathbf{P}^{(i,j)}\mathbf{x}_j\|^2$. The paper reformulates this as a predictive loss: a node's initial local representation should be predictable by the mean of its neighbors' final local representations. Connected components with low error are decoded as $\tau$-invariant substructures.
    - **Design Motivation**: If a structural behavior remains consistent across neighborhoods, its local representation should be stable and predictable; conversely, high prediction error indicates the structure is more context-dependent and requires more adjustment during transfer.

### Loss & Training
The pre-training objective for Gauge is the Dirichlet loss. For each graph, the loss is approximated as $\mathcal{L}(\mathcal{G})=\sum_i\|(\mathbf{Q}^{(i)})^\top\hat{\mathbf{z}}^{(0)}_i-\frac{1}{|\mathcal{N}_i|}\sum_{j\in\mathcal{N}_i}(\mathbf{Q}^{(j)})^\top\mathbf{z}^L_j\|_2^2$, where the initial representation uses a stop-gradient to prevent the model from trivializing the objective through scaling.

Downstream adaptation employs $\mathcal{L}_{ft}=\mathcal{L}_{task}+\lambda\mathcal{L}(\mathcal{G})$. The paper provides parameter stability results: for invariant structures identified during pre-training, fine-tuning gradients are constrained by both the task gradient and the structural prediction error, making low-error structures more resistant to being destroyed during adaptation.

## Key Experimental Results

### Main Results
Gauge was evaluated on cross-domain few-shot node classification, zero-shot link prediction, and graph isomorphism classification. Pre-training graphs were sourced from academic, social, and e-commerce domains, while target graphs included PubMed, FacebookPagePage, Roman-empire, and Photo. Most results report the mean and standard deviation of 10 independent runs.

| Task / Dataset | Metric | Gauge | Strongest Baseline | Key Conclusion |
|----------------|--------|-------|--------------------|----------------|
| PubMed 1-shot | Accuracy | 61.26±5.43 | RiemannGFM 56.82±9.00 | Gauge leads by approx. 4.44 points in biomedical graph transfer |
| PubMed 5-shot | Accuracy | 71.63±3.89 | GraphAny 70.19±2.20 | Maintains top performance with few labels |
| Facebook 1-shot| Accuracy | 51.61±9.53 | GRACE 49.79±7.82 | Leading performance in social graph transfer |
| Photo 5-shot | Accuracy | 81.33±2.93 | GraphAny 78.45±0.84 | Significant transfer advantage in e-commerce graphs |
| Roman 5-shot | Accuracy | 26.43±1.47 | GraphAny 26.72±1.14 | Close to best but not first on heterophilic graphs |

### Ablation Study
Rather than simple module removal, the paper validates whether "intrinsic geometry + Dirichlet invariance" genuinely provides structural capability through zero-shot link prediction, graph isomorphism, and visualization.

| Analysis | Key Metric | Description |
|----------|------------|-------------|
| Zero-shot Link Prediction (PubMed) | AUC 64.03 / AP 61.40 | Outperforms GFMs like RAGraph, GFT, and RiemannGFM without target graph fine-tuning |
| Zero-shot Link Prediction (Facebook)| AUC 93.88 / AP 90.73 | Significantly exceeds most baselines, suggesting pre-trained structural patterns are directly transferable |
| Zero-shot Link Prediction (Roman) | AUC 66.22 / AP 67.21 | Achieves best zero-shot performance even on heterophilic graphs |
| Graph Isomorphism (CSL) | ACC 92.56±4.37 | Significantly enhanced structural identification compared to SAMGPT (64.17±15.24) |
| Graph Isomorphism (ZINC12K) | MAE 0.1570±0.005 | Lower than GIN's 0.1630±0.004, indicating geometric modeling aids molecular graph regression |

### Key Findings
- Cross-domain transfer is not solely a product of pre-training scale. While multiple GFM baselines use a pre-train/fine-tune paradigm, Gauge is stronger across most target graphs and few-shot settings, indicating intrinsic geometry provides an independent contribution to structural transfer.
- Zero-shot link prediction serves as the most direct test: the model achieves best or near-best results on PubMed, Facebook, Roman, and Photo without any parameter updates, supporting the argument that behaviorally invariant structures can be reused.
- Graph isomorphism experiments shift the task from attribute prediction to pure structural identification. Gauge outperforms GCN, GraphSAGE, GIN, GAT, and SAMGPT on CSL and MUTAG, demonstrating that vector bundles not only serve transferability but also enhance representation of structural equivalence.
- In visualizations, Gauge recovers invariant substructures matching true topologies (e.g., binary trees, grids, paths, stars), providing intuitive evidence for the abstract Dirichlet energy interpretation.

## Highlights & Insights
- The major highlight is the shift of "substructure transferability" from a discrete matching problem to whether "functional behavior remains invariant within local geometry." This perspective avoids the ambiguity of heuristic metrics like motif frequency and explains why identical shapes may not be reusable across different contexts.
- The neural vector bundle modeling is ingenious: it does not require a predefined hyperbolic or spherical space but learns local coordinates from the node representations of a standard graph model. This retains the interpretability of geometric tools while reducing the issue of mismatched extrinsic geometric priors.
- The Dirichlet loss serves as both a pre-training objective and a transfer metric, resulting in a unified design. Low-loss regions not only assist in training but can be decoded into specific invariant substructures to explain which structural mechanisms the model has learned.
- This philosophy is applicable to other structured data. For instance, reusable patterns in molecules, program graphs, or knowledge graphs often depend on contextual functionality rather than isolated shapes; using local coordinate consistency to measure transfer cost may be more robust than manual patterns.

## Limitations & Future Work
- While the theoretical framework is strong, the practical system still depends on hyperparameters like local coordinate dimensions, gating temperature, and Dirichlet weights; sensitivity across different graph scales, noise levels, and heterophily requires more systematic reporting.
- The experiments cover various graph domains, but the target tasks are primarily node classification, link prediction, and graph isomorphism. The computational cost and stability of neural vector bundles for dynamic graphs, heterogeneous graphs, knowledge graph reasoning, or large-scale industrial graph retrieval remain to be verified.
- While the geometric interpretation of Gauge is attractive, visualization cases remain largely qualitative. Future work could map identified invariant substructures to actual downstream error types, domain knowledge, or manual structural labels to enhance the testability of its interpretability.
- The paper uses a frozen backbone with a small adaptation layer for transfer, which is suitable for verifying structural stability. However, real-world applications often require heavier fine-tuning; how to maintain invariant structures under strong adaptation is a worthwhile future direction.

## Related Work & Insights
- **vs. Motif / Graphon / Structural Vocabulary methods**: These methods look for common patterns in discrete structural space, whereas Gauge defines commonality as behavioral invariance in the representation space. The former is more intuitive and enumerable, while the latter better handles context-dependent structural functions.
- **vs. RiemannGFM**: RiemannGFM also uses Riemannian geometry but relies on predefined extrinsic spaces. Gauge learns the intrinsic geometry induced by model representations, making it less constrained by artificial geometric priors.
- **vs. Self-supervised GNNs (e.g., GraphMAE, GRACE)**: These models obtain general representations through reconstruction or contrastive learning. Gauge can be seen as extending "positive pair alignment" to Dirichlet smoothing of geometrically compatible neighborhoods, more directly targeting structural transfer.
- **vs. LLM-based GFMs**: LLM-based methods are powerful for text-attributed graphs but limited for pure structural graphs. Gauge's value lies in starting entirely from graph structure and node representations, filling a theoretical and architectural gap for text-free graph foundation models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Defining graph structural transfer through vector bundles and intrinsic geometry is highly distinctive in both problem reformulation and model design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers cross-domain few-shot, zero-shot link prediction, graph isomorphism, and visualization, though module-level ablation and large-scale cost analysis could be more detailed.
- Writing Quality: ⭐⭐⭐⭐ Complete theoretical narrative with clear geometric motivation; some formulas and notation are heavy, requiring some background in Riemannian geometry.
- Value: ⭐⭐⭐⭐⭐ Provides a trainable, interpretable, and effective solution to a core GFM problem, particularly as an inspiration for future research on structural transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)
- [\[ICML 2026\] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)

</div>

<!-- RELATED:END -->
