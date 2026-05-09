---
title: >-
  [Paper Note] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification
description: >-
  [NeurIPS 2025][Graph Learning][Few-shot node classification] PKD is a framework that jointly leverages LLMs and multiple GNN teachers for few-shot node classification on text-attributed graphs. A GNN-preference node selector (GNS) uses KL divergence-based uncertainty to identify nodes requiring LLM annotation, while a node-preference GNN selector (NGS) employs RL to match each node with its optimal GNN teacher. PKD achieves consistent state-of-the-art performance across 9 datasets (e.g., Cornell 87% vs. baselines 59–82%).
tags:
  - NeurIPS 2025
  - Graph Learning
  - Few-shot node classification
  - LLM-GNN collaboration
  - knowledge distillation
  - RL-based teacher selection
  - text-attributed graphs
date: 2026-05-08
content_hash: e52a43dbe833d91c
---

# PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification

**Conference**: NeurIPS 2025
**arXiv**: [2510.10116](https://arxiv.org/abs/2510.10116)
**Code**: [https://github.com/GEEX-Weixing/PKD](https://github.com/GEEX-Weixing/PKD)
**Area**: Graph Learning / Few-shot Learning
**Keywords**: Few-shot node classification, LLM-GNN collaboration, knowledge distillation, RL-based teacher selection, text-attributed graphs

## TL;DR
PKD is a framework that jointly leverages LLMs and multiple GNN teachers for few-shot node classification on text-attributed graphs. A GNN-preference node selector (GNS) uses KL divergence-based uncertainty to identify nodes requiring LLM annotation, while a node-preference GNN selector (NGS) employs RL to match each node with its optimal GNN teacher. PKD achieves consistent state-of-the-art performance across 9 datasets (e.g., Cornell 87% vs. baselines 59–82%).

## Background & Motivation

**State of the Field**: Few-shot node classification on text-attributed graphs (TAGs) requires combining LLMs' language understanding with GNNs' graph-structural modeling. Existing methods either use LLMs to generate labels for GNN training or align the two via adapter modules.

**Limitations of Prior Work**: (a) The embedding spaces of decoder-only LLMs and encoder-only GNNs are substantially misaligned, making direct alignment difficult; (b) the heterogeneous local topology across nodes makes a single GNN architecture suboptimal for all nodes; (c) LLM inference is expensive and should not be applied uniformly to all nodes.

**Root Cause**: World knowledge from LLMs and structural awareness from GNNs are complementary yet computationally asymmetric — the key challenge is how to allocate them intelligently.

**Paper Goals**: (a) Determine which nodes warrant LLM annotation; (b) identify the optimal GNN teacher for each node.

**Starting Point**: Bidirectional preference-driven collaboration — GNNs inform the LLM of nodes they are uncertain about, while the LLM guides each node's choice of message-passing mechanism.

**Core Idea**: GNN uncertainty → select nodes for LLM annotation + RL-tuned LLM → select the optimal GNN teacher per node = bidirectional preference-driven LLM-GNN collaboration.

## Method

### Overall Architecture
**GNS module**: $B$ GNN teachers independently produce predictions → KL divergence-based consensus quantifies uncertainty $\delta_K(v)$ → high-uncertainty nodes and their KNN neighbors are sent to the LLM for annotation. **NGS module**: A fine-tuned LLM serves as a PPO-based RL agent → state = node semantic/structural/prediction features → action = GNN teacher selection → reward = classification accuracy + distillation loss. **KD**: The selected teacher supervises the student GNN.

### Key Designs

1. **GNN-Preference Node Selector (GNS)**:

    - Function: Select nodes requiring LLM annotation.
    - Mechanism: $\delta_K(v) = \sum_{i<j} [D_{KL}(f_{T_i}(v) \| f_{T_j}(v)) + D_{KL}(f_{T_j}(v) \| f_{T_i}(v))]$ — the sum of pairwise KL divergences among $B$ GNN teachers quantifies prediction uncertainty. High uncertainty implies disagreement among teachers, signaling the need for LLM assistance. DNS: KNN retrieval expands the neighborhood context.
    - Design Motivation: Costly LLM calls should be reserved for nodes that GNNs cannot handle reliably.

2. **Node-Preference GNN Selector (NGS)**:

    - Function: Select the optimal GNN teacher for each node.
    - Mechanism: The LLM is fine-tuned as a PPO agent. State = node prompt (semantic + structural + prediction attributes). Action = teacher selection probability distribution $\pi_T$. Reward $R = \eta(\mathcal{L}_{DL}' - \mathcal{L}_{CE}) + (1-\eta) A_{cc}$ — jointly optimizing distillation loss improvement and classification accuracy.
    - Design Motivation: Nodes with different topologies suit different message-passing patterns — dense regions benefit from multi-hop GCN, while sparse regions are better handled by attention-based GAT.

3. **GTA Prompt Fine-tuning**:

    - Function: Enable the LLM to understand graph structure.
    - Mechanism: The LLM is fine-tuned on four graph topology tasks (connectivity, degree, cycle detection, text generation) to acquire graph-structural awareness.
    - Design Motivation: Vanilla LLMs lack graph comprehension; GTA prompts inject the necessary structural knowledge.

### Loss & Training
- $\mathcal{L}_{KD} = \alpha \cdot \text{soft KD} + \beta \cdot \text{hard label CE} + \gamma \cdot \text{entropy reg}$
- PPO optimizes the RL agent.
- Compatible with multiple LLM backbones including Llama-3.1, Qwen2.5, and Mixtral.

## Key Experimental Results

### Main Results (5 labeled nodes/class)

| Dataset | PKD | Runner-up | Gain |
|--------|-----|------|------|
| Cornell | **87.0%** | 82.0% (IceBerg) | +5.0% |
| Cora | **91.14%** | 79.5% (PopT) | +11.6% |
| Washington | **83.74%** | 79.76% (IceBerg) | +3.98% |
| Texas | **86.31%** | 84.85% (FairGKD) | +1.46% |

PKD achieves consistently best or second-best performance across all 9 datasets.

### Ablation Study

| Component | Effect of Removal |
|------|----------|
| GTA prompts | Degraded graph comprehension |
| DNS module | Lower neighbor selection quality |
| K-uncertainty | Random node selection performs poorly |
| Full model | **Best** |

### Key Findings
- The framework generalizes across different LLM backbones (Llama/Qwen/Mixtral), demonstrating backbone-agnostic applicability.
- K-uncertainty-based selection outperforms random selection, validating the value of intelligent node allocation.
- RL-based teacher selection outperforms fixed single-teacher assignment, confirming that topological diversity necessitates varied processing strategies.

## Highlights & Insights
- The **bidirectional preference-driven** design elegantly exploits the complementary strengths of LLMs and GNNs — GNNs identify their own uncertainty, while LLMs determine how to resolve it.
- **RL-based teacher selection** is more adaptive than static assignment, accommodating diverse node topologies.

## Limitations & Future Work
- Training $B$ GNN teachers alongside LLM fine-tuning incurs substantial computational cost.
- The framework is limited to node-level classification and has not been extended to edge- or graph-level tasks.
- RL training may exhibit instability.

## Related Work & Insights
- **vs. GraphLLM**: GraphLLM replaces GNNs with LLMs entirely; PKD promotes bidirectional collaboration between the two.
- **vs. TAPE**: TAPE uses LLMs to generate textual features for GNNs; PKD engages in deeper, bidirectional interaction.

## Rating
- Novelty: ⭐⭐⭐⭐ Bidirectional preference-driven design + RL-based teacher selection
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets + 3 LLM backbones + ablation study
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear and well-structured
- Value: ⭐⭐⭐⭐ A practical solution for few-shot graph learning
- The two-level preference-driven design — LLM selects nodes, RL selects GNNs — reflects the key insight that different nodes suit different GNN architectures.
- PKD surpasses prior state-of-the-art on few-shot node classification and maintains its advantage as the number of labeled samples increases.
- The core contribution lies in the conceptual simplicity and empirical effectiveness of its design.
- Experimental results comprehensively validate the central hypotheses.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation](spot-trip_dual-preference_driven_out-of-town_trip_recommendation.md)
- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](../../AAAI2026/graph_learning/rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)

<!-- RELATED:END -->
