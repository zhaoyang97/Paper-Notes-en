---
title: >-
  [Paper Note] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design
description: >-
  [ICML 2026][Graph Learning][Neural Architecture Search] This paper proposes the M-DESIGN framework, which models neural network design as a retrieval-augmented iterative modification process. By constructing a Modificati…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Neural Architecture Search"
  - "Model Retrieval"
  - "Knowledge Graph"
  - "Graph Neural Networks"
  - "Bayesian Optimization"
date: 2026-05-08
content_hash: c2ce519243495e7e
---

# Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design

**Conference**: ICML 2026  
**arXiv**: [2507.15336](https://arxiv.org/abs/2507.15336)  
**Code**: https://github.com/jilwang84/M-DESIGN  
**Area**: Graph Learning / AutoML  
**Keywords**: Neural Architecture Search, Model Retrieval, Knowledge Graph, Graph Neural Networks, Bayesian Optimization  

## TL;DR

This paper proposes the M-DESIGN framework, which models neural network design as a retrieval-augmented iterative modification process. By constructing a Modification-Gain Graph to encode fine-grained architecture editing effects and utilizing Bayesian dynamic task similarity for online calibration of transfer signals, it achieves design-space optimality in 26 out of 33 GNN tasks.

## Background & Motivation

**Background**: Mainstream methods for designing high-performance neural networks fall into two categories: Neural Architecture Search (NAS), which finds optimal structures through exhaustive trial and error, and Model Retrieval, which selects a starting model from a pre-trained model bank. The former is computationally expensive, while the latter struggles to reach peak performance.

**Limitations of Prior Work**: NAS methods search from scratch for every new task without reusing historical experience, leading to a severe cold-start problem. Retrieval methods focus only on selecting a reasonable starting model, leaving subsequent architecture adaptation dependent on ad-hoc trial-and-error adjustments. Crucially, both types of methods ignore the specific performance impact records of fine-grained architecture modifications (e.g., replacing the message-passing mechanism of a GNN).

**Key Challenge**: There is a fundamental trade-off between search efficiency and optimality in existing methods. NAS can find good architectures at a high cost, while retrieval is efficient but yields sub-optimal results. The root causes are: (1) The transfer effect of architecture modifications evolves along the modification trajectory, making static task similarity ineffective; (2) Evidence from direct retrieval is unreliable when the distribution of a new task deviates significantly from the knowledge base.

**Goal**: To transform model design from a one-off retrieval into a knowledge-driven iterative modification process, quickly finding near-optimal architectures under a limited evaluation budget.

**Key Insight**: The authors observe that if the performance impact of each fine-grained architecture modification (e.g., changing activation functions or aggregation methods) is explicitly recorded as "edit-effect evidence," this evidence can be organized into a graph structure to support relational reasoning across tasks.

**Core Idea**: Construct a Modification-Gain Graph to encode historical design experience and weave cross-task evidence through Bayesian online-updated dynamic task similarity, transforming model design into an adaptive retrieval-augmented modification process.

## Method

### Overall Architecture

The input to M-DESIGN is a new target task $D^u$ and a Model Knowledge Base (MKB) containing historical records of $N$ benchmark tasks. The output is the optimal architecture $\theta^*$ found within a fixed evaluation budget $T$. The process consists of three phases: (1) Offline construction of the Modification-Gain Graph, encoding performance differences between architecture variants on benchmark tasks as directed weighted edges; (2) Online iterative modification, where candidate modifications are retrieved from the knowledge base each step, expected gains are estimated via cross-task knowledge weaving, and the optimal modification is executed to observe real effects; (3) Bayesian update, utilizing observed real gains to calibrate task similarity beliefs online, gradually aligning transfer signals with the target task.

### Key Designs

1.  **Modification-Gain Graph and Knowledge Weaving**:

    - **Function**: Structures historical design experience into composable graph representations to support fine-grained gain estimation across tasks.
    - **Mechanism**: For each benchmark task $D^i$, a graph $G_\Delta^i = (V, E^i, \omega^i)$ is constructed, where nodes $V$ are architecture configurations and edge weights for $(θ, θ')$ are performance gains $\omega^i(e) = \mathcal{P}(θ', D^i) - \mathcal{P}(θ, D^i)$. When selecting the optimal modification, evidence is aggregated via weighted task similarity: $\Delta\theta_t^* = \arg\max_{\Delta\theta} \sum_i \mathcal{S}_t(D^u, D^i) \cdot \widetilde{\Delta P}_t^i(\Delta\theta)$, where $\mathcal{S}_t$ is the dynamic task similarity.
    - **Design Motivation**: Unlike traditional methods that only store complete model performance records, edge-represented relative gains possess natural reversibility and linkability, allowing the composition of local edit patterns across tasks to approximate optimal paths.

2.  **Bayesian Dynamic Task Similarity**:

    - **Function**: Calibrates task similarity online during the iterative modification process to avoid negative transfer caused by static similarity failure.
    - **Mechanism**: Task similarity $\mathcal{S}_t^{u,i}$ is treated as a Bayesian posterior, updated via Bayesian rules using real gains observed at each step: $\mathcal{S}_t^{u,i} \propto \mathcal{N}(\Delta P_t^u; \gamma_{i,t} \Delta P_t^i, \sigma^2) \cdot \mathcal{S}_{t-1}^{u,i}$. The likelihood function is based on a Gaussian assumption of gain consistency, with parameters estimated from recent observations using a sliding window (size 30-40).
    - **Design Motivation**: Experimental validation shows (Kendall rank correlation of 0.34 vs. 0.08 for static methods) that real modification consistency varies drastically across modification trajectories, necessitating dynamic updates to track the evolving transfer landscape.

3.  **Predictive Task Planner (OOD Adaptation)**:

    - **Function**: Generates synthetic gain evidence to fill retrieval gaps when relevant evidence is missing from the knowledge base or when the target task significantly deviates from the distribution.
    - **Mechanism**: An EdgeConv GNN-based regression model $f_{\psi_i}$ is trained for each benchmark task, taking the current architecture and candidate modified architecture as input to predict the gain $\widehat{\Delta P}_t^i(\Delta\theta) = f_{\psi_i}(\theta_t, \theta_t + \Delta\theta)$. The system switches to planner predictions when posterior similarity falls below a threshold $\delta$, and synthetic evidence is aligned with the target distribution via online replay buffer fine-tuning.
    - **Design Motivation**: On OOD datasets like Cornell, direct retrieval correlation is extremely low ($R^2=0.03$). Enabling the planner improves this to $R^2=0.11$, preventing error accumulation from misleading evidence.

## Key Experimental Results

### Main Results

Evaluated on 67,760 GNN models across 22 datasets and 33 task-data pairs, with a maximum evaluation budget of 100:

| Dataset | Design Space Optimum | AutoTransfer | DesiGNN | M-DESIGN | Reached Optimum? |
|--------|-------------|-------------|---------|----------|----------|
| Actor | 34.89 | 33.97 | 34.43 | **34.89** | ✓ |
| Computers | 89.59 | 87.72 | 88.40 | **89.22** | — |
| Photo | 94.75 | 94.62 | 94.60 | **94.75** | ✓ |
| CiteSeer | 74.59 | 73.89 | 74.54 | **74.59** | ✓ |
| CS | 95.33 | 95.16 | 95.03 | **95.33** | ✓ |
| Cora | 88.50 | 88.50 | 88.34 | **88.50** | ✓ |
| Cornell | 77.48 | 76.58 | 75.50 | **77.48** | ✓ |
| DBLP | 84.29 | 83.59 | 84.29 | **84.29** | ✓ |
| PubMed | 89.08 | 89.08 | 89.08 | **89.08** | ✓ |
| Texas | 84.68 | 78.38 | 81.80 | 83.79 | — |
| Wisconsin | 91.33 | 88.67 | 90.66 | **91.33** | ✓ |

M-DESIGN reached the **Design Space Optimum** in **26 out of 33** task-data pairs, outperforming all baselines.

### Ablation Study

| Variant | Avg. Accuracy Drop | Kendall Rank Corr. | Note |
|------|-------------|---------------|------|
| M-DESIGN (Full) | — | 0.34 | Dynamic Similarity + Sliding Window + OOD Adaptation |
| w/o Sliding Window | Slight decrease | 0.27 | Early unreliable evidence not down-weighted |
| w/o Dynamic Update | Largest decrease | 0.08 | Static similarity fails to track local consistency |
| w/o OOD Adaptation | Large drop on OOD | 0.31 | Significant degradation on Computers/Cornell/Texas |

Knowledge base scale ablation: Maintaining only 25% of benchmark tasks resulted in an average accuracy of 81.50, compared to 82.11 with 100%, indicating graceful performance degradation.

### Search Efficiency Comparison

| Method | Evals to reach target (Cornell) | Evals to reach target (Wisconsin) |
|------|-------------------------|---------------------------|
| Random | ∞ | 79 |
| RL | 92.7 | 91.2 |
| EA | ∞ | 96.9 |
| DesiGNN | ∞ | 62.6 |
| **M-DESIGN** | **22** | **5** |

The per-step overhead of M-DESIGN for MKB operations is <0.31s (<0.44s with OOD adaptation), which is negligible compared to the ~30s required for a single model evaluation.

## Highlights & Insights

1.  **Paradigm Shift in Knowledge Representation**: Shifts from storing static model performance records to encoding fine-grained modification gains, making historical experience composable and reason-able rather than just searchable.
2.  **Dynamic Transfer Calibration**: Bayesian online updates are the primary driver of performance gains (ablation shows Kendall correlation dropping from 0.34 to 0.08 without them), solving the fundamental failure of static task similarity during iterative modifications.
3.  **Empirical Support for Theoretical Assumptions**: Linear gain transfer and Gaussian distribution assumptions were validated on high-similarity task pairs (Cora-DBLP $R^2=0.87$), providing a reliable likelihood model for Bayesian updates.
4.  **Cross-domain Transfer Potential**: Performed excellently on tabular data (e.g., Protein, Slice, Naval datasets), ranking within the top 0.05%-0.47% of the design space.

## Limitations & Future Work

1.  The current instantiation only covers the GNN design space (3,080 candidate architectures). Extending to larger spaces like CNNs/Transformers requires solving scalability issues in knowledge base construction.
2.  Offline MKB construction requires pre-training a large number of models (67,760), leading to high initial costs.
3.  OOD adaptation shows limited improvement under extreme distribution shifts (Cornell $R^2$ only improved from 0.03 to 0.11); multi-hop reasoning capabilities need enhancement.
4.  The Bayesian update assumes Gaussian-distributed gains, which may not hold in highly non-convex design spaces.

## Related Work & Insights

1.  **DesiGNN** (Wang et al., 2026): Retrieval-augmented GNN design, but relies on static task similarity and lacks online calibration.
2.  **AutoTransfer** (Cao et al., 2023): Embedding-based model transfer, also suffering from static similarity issues.
3.  **NAS-Bench-Graph** (Qin et al., 2022): A GNN architecture search benchmark utilizing rank correlation to measure task similarity.

## Rating
- Novelty: 9/10 — Reframing model design as retrieval-augmented iterative optimization on a Modification-Gain Graph; Bayesian dynamic task similarity is a highly original design.
- Experimental Thoroughness: 9/10 — 33 task-data pairs + 10 baselines + detailed ablation + theoretical validation + cross-domain experiments.
- Writing Quality: 8/10 — Formal definitions are clear and rigorous, though notation density is high.
- Value: 8/10 — Provides a new paradigm for AutoML, though feasibility for larger design spaces remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[AAAI 2026\] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](../../AAAI2026/graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](../../ICLR2026/graph_learning/beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)

</div>

<!-- RELATED:END -->
