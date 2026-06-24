---
title: >-
  [Paper Note] A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks
description: >-
  [ICML 2025][Graph Learning][Graph Neural Networks] Cognac is proposed as the first effective corrective unlearning method for GNNs. By alternating between Contrastive Unlearning on Graph Neighborhoods (CoGN) and AsCent DesCent DeCoupled (AC⚡DC), it restores performance close to the oracle (trained on fully clean data) while identifying only 5% of the manipulated entities, achieving 8× higher efficiency than retraining from scratch.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Corrective Unlearning"
  - "Adversarial Manipulation"
  - "Message Passing"
  - "Graph Unlearning"
date: 2026-05-08
content_hash: a18515d64f08cce7
---

# A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2412.00789](https://arxiv.org/abs/2412.00789)  
**Code**: [https://github.com/corrective-unlearning-for-gnns](https://github.com/corrective-unlearning-for-gnns)  
**Area**: Graph Learning  
**Keywords**: Graph Neural Networks, Corrective Unlearning, Adversarial Manipulation, Message Passing, Graph Unlearning

## TL;DR
Cognac is proposed as the first effective corrective unlearning method for GNNs. By alternating between Contrastive Unlearning on Graph Neighborhoods (CoGN) and AsCent DesCent DeCoupled (AC⚡DC), it restores performance close to the oracle (trained on fully clean data) while identifying only 5% of the manipulated entities, achieving 8× higher efficiency than retraining from scratch.

## Background & Motivation

**Background**: GNNs are widely applied in domains such as recommendation systems and drug discovery, and are beginning to scale to large-scale graph foundation models. However, verifying the integrity of every sample in a massive training set is prohibitively expensive.

**Limitations of Prior Work**:
   - Graph data violates the i.i.d. assumption—message passing allows adversarial manipulation to propagate through neighborhoods, affecting a wider range of nodes.
   - Existing graph unlearning methods (GraphEraser, GUIDE, etc.) focus on privacy-compliant deletion and are not suitable for corrective unlearning—even with full knowledge of the manipulated set, they fail to effectively unlearn the manipulative influence.
   - "Retraining from scratch" is not the gold standard for corrective unlearning—when unidentified manipulated data remains in the training set, retraining reinforces the manipulation effect.

**Key Challenge**: While powerful, message passing is a double-edged sword—manipulating a small fraction of nodes can cause large-scale prediction biases; unlearning must simultaneously correct the manipulated nodes themselves and their affected neighbors.

**Goal**: Remove the adverse effects of data manipulation from a trained GNN, requiring only a small subset of the manipulated set to be identified.

**Key Insight**: A two-step alternation—(1) push representation of affected neighbors away from manipulated nodes using contrastive loss; (2) apply gradient ascent on the manipulated set and gradient descent on the remaining data.

**Core Idea**: Utilize the graph structure—the neighbors of manipulated nodes are a critical "infected zone"; "sanitize" the neighborhood via contrastive learning, and then correct the magnitude using decoupled gradients.

## Method

### Overall Architecture
Cognac alternately executes two components:
1. **CoGN（Contrastive Unlearning on Graph Neighborhoods）**:
    - Identifies affected neighbors of the forgot set.
    - Pushes neighbors' representations away from the forgot set and pulls them closer to other normal neighbors using contrastive loss.
2. **AC⚡DC（AsCent DesCent DeCoupled）**:
    - Performs gradient ascent on the forgot set labels ("forgetting" incorrect labels).
    - Performs gradient descent on the remaining set labels (maintaining normal performance).
    - Employs independent optimizers for the two processes (decoupled to avoid interference).

### Key Designs

1. **Contrastive Unlearning on Graph Neighborhoods (CoGN)**:

    - **Function**: Corrects the "infection" of neighbors caused by manipulated nodes.
    - **Mechanism**:
        - Find the $k$-hop neighbors $\mathcal{N}(S_f)$ of the forgot set $S_f$.
        - Contrastive loss: $\mathcal{L}_{\text{CoGN}} = -\log \frac{\exp(\text{sim}(h_v, h_{v^+})/\tau)}{\exp(\text{sim}(h_v, h_{v^+})/\tau) + \sum_{u \in S_f} \exp(\text{sim}(h_v, h_u)/\tau)}$
        - Positive sample $v^+$: normal nodes in the neighborhood not belonging to the forgot set; negative samples: nodes in the forgot set.
    - **Design Motivation**: Message passing propagates manipulation through neighborhoods $\rightarrow$ neighborhood representations must be explicitly corrected.
    - **Novelty**: Even when the forgot set is only 5% of the manipulated set, neighborhood search can "discover" more unidentified affected nodes.

2. **AsCent DesCent DeCoupled (AC⚡DC)**:

    - **Function**: "Forgets" on the forgot set + "maintains" on the remaining set.
    - **Mechanism**:
        - Gradient ascent: $\theta \leftarrow \theta + \eta_1 \nabla_\theta \mathcal{L}(S_f)$ (increases loss on the forgot set $\rightarrow$ forgetting).
        - Gradient descent: $\theta \leftarrow \theta - \eta_2 \nabla_\theta \mathcal{L}(S_r)$ (decreases loss on the remaining set $\rightarrow$ maintaining).
        - Key: The two processes use independent optimizer instances (independent momentum/adaptive learning rates).
    - **Design Motivation**: Coupled optimization (e.g., SCRUB) causes unlearning and maintaining to interfere with each other $\rightarrow$ decoupling stabilizes both.
    - Adaptation of classic i.i.d. unlearning methods to graphs—AC⚡DC provides foundational unlearning, and CoGN provides graph-specific neighborhood correction.

3. **Utilization of Representative Subsets**:

    - **Function**: Unlearns all manipulation influence using only a 5% identified subset of manipulated entities.
    - **Mechanism**: The identified subset $S_f \subset S_m$ shares manipulation patterns $\rightarrow$ CoGN naturally discovers more affected nodes via neighborhood expansion $\rightarrow$ the unlearning signal of AC⚡DC on $S_f$ generalizes to the entire manipulated set.
    - **Design Motivation**: It is impossible to identify 100% of the manipulated data in practice—the method must be effective under partial identification.

### Loss & Training
- Overall optimization: Alternately executes CoGN and AC⚡DC.
- CoGN uses contrastive loss + temperature parameter $\tau$.
- AC⚡DC uses standard cross-entropy (with opposite ascent/descent directions).
- Two independent Adam optimizers are used.
- Early stopping is based on validation set accuracy.

## Key Experimental Results

### Main Results
Cora/Citeseer/PubMed + ogbn-arxiv (edge manipulation + node label flipping):

| Method | Affected Acc. ↑ | Unaffected Acc. | Known Ratio |
|------|---------------|-----------|---------|
| No Unlearning (Manipulated Model) | 55.2% | 82.1% | - |
| Retraining (w/o Forgot Set) | 58.7% | 81.5% | 100% |
| GraphEraser | 56.3% | 79.8% | 100% |
| SCRUB | 61.2% | 80.3% | 100% |
| **Cognac (5% Identified)** | **72.8%** | **81.2%** | **5%** |
| **Cognac (100% Identified)** | **78.5%** | **81.8%** | **100%** |
| Oracle (Trained on Clean Data) | 82.3% | 82.5% | - |

### Ablation Study

| Configuration | Affected Acc. | Description |
|------|-----------|------|
| AC⚡DC Only (No neighborhood correction) | 64.5% | i.i.d. methods do not handle propagation |
| CoGN Only (No gradient unlearning) | 67.3% | Neighborhood corrected but source not unlearned |
| **Cognac (Alternating both)** | **72.8%** | Optimal combination |
| Coupled Optimizer (Non-decoupled) | 68.1% | Unlearning and maintaining interfere with each other |
| **Decoupled Optimizer** | **72.8%** | Independent optimization is more stable |
| 1% Identified | 65.2% | Too small to be representative |
| 5% Identified | 72.8% | Sufficiently representative |
| 20% Identified | 76.1% | More information yields better results |

### Key Findings
- Existing graph unlearning methods (GraphEraser, GUIDE) completely fail—they cannot effectively unlearn even with 100% knowledge of the manipulated set.
- Retraining from scratch is not the gold standard—(58.7% vs. Oracle 82.3%) because unidentified manipulated data remains in the training set.
- Cognac achieves 72.8% with only 5% identified—close to the 78.5% of 100% identification, proving the method's robustness to partial identification.
- The contribution of CoGN's neighborhood correction (+8.3%) is almost equal to AC⚡DC's label unlearning (+7.8%)—both are equally important.
- The decoupled optimizer improves performance by +4.7% over the coupled optimizer—independence is crucial for stability.
- It achieves an 8× speedup over retraining on the large ogbn-arxiv graph—scalable to large-scale scenarios.

## Highlights & Insights
- **"Retraining from scratch is not the gold standard"** subverts traditional assumptions in the unlearning field—especially in corrective unlearning.
- Contrastive unlearning (CoGN) exploits the graph structure to scale "from points to areas"—5% known $\rightarrow$ neighborhood discovery $\rightarrow$ generalization to the entire manipulated set.
- Decoupling optimizers is a simple yet crucial engineering contribution—preventing positive (maintain) and negative (forget) gradients from cancelling each other out.
- The double entendre in the paper's title is excellent—Cognac is used to "forget bad memories", and its initials represent Contrastive + GNN.
- It holds significant practical value for graph data security and trustworthy AI.

## Limitations & Future Work
- Assumes knowledge of the manipulation type (edges or nodes)—pre-diagnosis is required when the type is unknown.
- CoGN's neighborhood search can be computationally expensive in dense graphs.
- Evaluated only on node classification tasks—link prediction and graph classification remain to be explored.
- Evaluated on label flipping and edge injection—more covert manipulations (e.g., feature perturbations) have not been tested.
- Scenarios with sequential manipulation are not discussed.

## Related Work & Insights
- **vs GraphEraser**: Focuses on privacy deletion (removing data), rather than corrective unlearning (correcting influence).
- **vs SCRUB**: An i.i.d. unlearning method that does not handle the propagation of message passing on graphs.
- **vs Goel et al. (corrective unlearning)**: Proposed corrective unlearning in the i.i.d. setting; Cognac extends this concept to graphs.
- **Insight**: The propagation characteristics of message passing imply that graph unlearning requires "neighborhood-level" rather than "node-level" handling—providing insights for all post-processing interventions in graph machine learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first effective corrective unlearning method for GNNs
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple graphs + multiple manipulations + multiple baselines + varying identification ratios
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem statement, intuitive method, engaging title
- Value: ⭐⭐⭐⭐⭐ Provides a significant push for graph data security and trustworthy GNNs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)
- [\[ICML 2025\] Unifews: You Need Fewer Operations for Efficient Graph Neural Networks](unifews_you_need_fewer_operations_for_efficient_graph_neural_networks.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)

</div>

<!-- RELATED:END -->
