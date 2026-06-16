---
title: >-
  [Paper Note] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design
description: >-
  [ICML 2026][Graph Learning][Paper Note] The M-DESIGN framework is proposed, modeling neural network design as a retrieval-augmented iterative modification process. By constructing a Modification-Gain Graph to encode fine-grained architecture editing effects and utilizing Bayesian dynamic task similarity for online calibration of transfer signals, the method
tags:
  - ICML 2026
  - Graph Learning
date: 2026-05-08
content_hash: bd0519974659c8e6
---
# Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design

**Conference**: ICML 2026  
**arXiv**: [2507.15336](https://arxiv.org/abs/2507.15336)  
**Code**: https://github.com/jilwang84/M-DESIGN  
**Area**: Graph Learning / AutoML  
**Keywords**: Neural Architecture Search, Model Retrieval, Knowledge Graph, Graph Neural Networks, Bayesian Optimization  

## TL;DR

The M-DESIGN framework is proposed, modeling neural network design as a retrieval-augmented iterative modification process. By constructing a Modification-Gain Graph to encode fine-grained architecture editing effects and utilizing Bayesian dynamic task similarity for online calibration of transfer signals, the method achieves design-space optimality in 26 out of 33 GNN tasks.

## Background & Motivation

**Background**: Prevailing methods for designing high-performance neural networks are divided into two categories: Neural Architecture Search (NAS), which seeks optimal structures through exhaustive trial and error, and Model Retrieval, which selects starting models from pre-trained model zoos. The former is computationally expensive, while the latter struggles to reach optimal performance.

**Limitations of Prior Work**: NAS methods search from scratch for every new task, failing to reuse historical experience and suffering from severe cold-start issues. Although retrieval methods are efficient, they focus only on selecting a reasonable starting model, leaving subsequent architectural adaptation to ad-hoc trial-and-error adjustments. Crucially, both categories neglect the systematic recording of the specific impact of fine-grained architectural modifications (e.g., replacing message-passing mechanisms in GNNs) on performance.

**Key Challenge**: Existing methods face a fundamental trade-off between search efficiency and optimality. NAS identifies superior architectures at high costs, while retrieval is efficient but suboptimal. The root causes are: (1) the transferability of architectural modifications evolves along the modification trajectory, rendering static task similarity ineffective; (2) when the distribution of a new task deviates significantly from the knowledge base, direct retrieval evidence becomes unreliable.

**Goal**: To transform model design from a one-time retrieval into a knowledge-driven iterative modification process, quickly identifying near-optimal architectures within a limited evaluation budget.

**Key Insight**: It is observed that if the performance impact of each fine-grained architectural modification (e.g., changing activation functions or aggregation methods) is explicitly recorded as "edit-effect evidence," these evidences can be organized into a graph structure to support relational reasoning across tasks.

**Core Idea**: Construct a Modification-Gain Graph to encode historical design experience and weave cross-task evidence via Bayesian-updated dynamic task similarity, transforming model design into an adaptive retrieval-augmented modification process.

## Method

### Overall Architecture

The input to M-DESIGN is a new target task $D^u$ and a Model Knowledge Base (MKB) containing historical records of $N$ benchmark tasks. The output is the optimal architecture $\theta^*$ found within a fixed evaluation budget $T$. The process consists of three phases: (1) Offline construction of the Modification-Gain Graph, encoding performance differences between architectural variants on each benchmark task as directed weighted edges; (2) Online iterative modification, where candidate modifications are retrieved from the knowledge base at each step, expected gains are estimated via cross-task knowledge weaving, and the optimal modification is executed to observe actual effects; (3) Bayesian update, utilizing observed actual gains to calibrate task similarity beliefs online, progressively aligning transfer signals with the target task.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Target Task $D^u$ + Model Knowledge Base MKB ($N$ benchmark tasks)"] --> B["Modification-Gain Graph and Knowledge Weaving<br/>Offline: Encode performance differences as weighted edges; Aggregate candidate modifications via similarity"]
    B --> C["Execute modification $\Delta\theta^*$ with optimal expected gain<br/>Observe actual gain on target task"]
    C --> D["Bayesian Dynamic Task Similarity<br/>Calibrate similarity beliefs online using actual gains"]
    D -->|Posterior similarity < threshold $\delta$ (OOD)| E["Predictive Task Planner<br/>EdgeConv regressor synthesizes gains to fill retrieval gaps"]
    E --> F{"Evaluation budget $T$ reached?"}
    D -->|Similarity sufficient| F
    F -->|No| B
    F -->|Yes| G["Output near-optimal architecture $\theta^*$"]
```

### Key Designs

**1. Modification-Gain Graph & Knowledge Weaving: Compiling History into a Composable Graph**

Traditional retrieval methods only store "how a complete model scored on a specific task," which allows for lookup but not reasoning. M-DESIGN records "relative gains" instead: for each benchmark task $D^i$, a graph $G_\Delta^i = (V, E^i, \omega^i)$ is constructed where nodes $V$ represent architecture configurations and edge weights $\omega^i(e) = \mathcal{P}(\theta', D^i) - \mathcal{P}(\theta, D^i)$ directly denote the performance difference from a modification. Relative gains are naturally reversible and concatenable, allowing local edit patterns across tasks to be combined to approximate optimal paths. When selecting the next modification for the target task, gain evidence from benchmark tasks is aggregated via similarity weighting: $\Delta\theta_t^* = \arg\max_{\Delta\theta} \sum_i \mathcal{S}_t(D^u, D^i) \cdot \widetilde{\Delta P}_t^i(\Delta\theta)$, where $\mathcal{S}_t(D^u, D^i)$ is the dynamic task similarity calibrated online.

**2. Bayesian Dynamic Task Similarity: Correcting Transfer Signals During Modification**

The problem with static similarity is that the guidance value of the same modification can shift dramatically at different stages of the modification trajectory (empirical results show Kendall rank correlation rising from 0.08 for static to 0.34 for dynamic similarity); mismatch leads to negative transfer. M-DESIGN treats similarity $\mathcal{S}_t^{u,i}$ as a Bayesian posterior. After executing a modification and observing the actual gain, the belief is updated via Bayesian rules: $\mathcal{S}_t^{u,i} \propto \mathcal{N}(\Delta P_t^u; \gamma_{i,t} \Delta P_t^i, \sigma^2) \cdot \mathcal{S}_{t-1}^{u,i}$. The likelihood assumes the target task's real gain $\Delta P_t^u$ aligns with the benchmark gain $\Delta P_t^i$ under a scaling factor $\gamma_{i,t}$ with Gaussian noise. Parameters are estimated from recent sliding window observations (size 30-40), allowing similarity to align with target task performance.

**3. Predictive Task Planner: Synthesizing Gains for Insufficient Knowledge Base Evidence**

When a target task deviates significantly from the knowledge base distribution (OOD) and lacks reliable evidence, the relevance of direct retrieval can collapse to $R^2=0.03$ (e.g., Cornell). To prevent error accumulation from misleading evidence, an additional EdgeConv GNN-based gain regressor $f_{\psi_i}$ is trained for each benchmark task. It predicts gains: $\widehat{\Delta P}_t^i(\Delta\theta) = f_{\psi_i}(\theta_t, \theta_t + \Delta\theta)$. If the posterior similarity falls below threshold $\delta$, the system switches to the planner's predictions to fill retrieval gaps. Online replay buffer fine-tuning is used to pull synthesized evidence toward the target distribution, increasing correlation for Cornell from $R^2=0.03$ to $R^2=0.11$.

## Key Experimental Results

### Main Results

Evaluated on 67,760 GNN models across 22 datasets (33 task-data pairs) with a maximum evaluation budget of 100:

| Dataset | Space Optimal | AutoTransfer | DesiGNN | M-DESIGN | Optimal Reached? |
|---------|---------------|--------------|---------|----------|-----------------|
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

M-DESIGN reached the design space optimum in **26 out of 33** task-data pairs, outperforming all baselines.

### Ablation Study

| Variant | Avg. Accuracy Drop | Kendall Rank Corr. | Note |
|---------|--------------------|--------------------|------|
| M-DESIGN (Full) | — | 0.34 | Dynamic Similarity + Sliding Window + OOD Adaptation |
| w/o Sliding Window | Slight drop | 0.27 | Early unreliable evidence not downweighted |
| w/o Dynamic Update | Largest drop | 0.08 | Static similarity fails to track local consistency |
| w/o OOD Adaptation| Significant drop on OOD | 0.31 | Computers/Cornell/Texas performance degraded |

Knowledge base scale ablation: Retaining only 25% of benchmark tasks yields an average accuracy of 81.50, compared to 82.11 with 100%, showing graceful degradation.

### Search Efficiency Comparison

| Method | Evals to Target (Cornell) | Evals to Target (Wisconsin) |
|--------|---------------------------|-----------------------------|
| Random | ∞ | 79 |
| RL | 92.7 | 91.2 |
| EA | ∞ | 96.9 |
| DesiGNN | ∞ | 62.6 |
| **M-DESIGN** | **22** | **5** |

Per-step MKB operation overhead for M-DESIGN is <0.31s (<0.44s with OOD adaptation), significantly less than the ~30s required for a single model evaluation.

## Highlights & Insights

1.  **Paradigm Shift in Knowledge Representation**: Moving from storing complete model records to encoding fine-grained modification gains makes historical experience composable and reason-able, rather than just searchable.
2.  **Dynamic Transfer Calibration**: Bayesian online updates are the primary source of performance gains (ablation shows removal drops Kendall correlation from 0.34 to 0.08), solving the fundamental failure of static task similarity during iterative modification.
3.  **Empirical Support for Theoretical Assumptions**: Linear gain transfer and Gaussian distribution assumptions were validated on high-similarity task pairs (e.g., Cora-DBLP with $R^2=0.87$), providing a reliable likelihood model for Bayesian updates.
4.  **Cross-domain Transfer Potential**: Performance was also excellent on tabular data (4 datasets including Protein/Slice/Naval), with rankings within the top 0.05%-0.47% of the design space.

## Limitations & Future Work

1.  The current instantiation only covers the GNN design space (3,080 candidates); scaling to larger spaces like CNNs/Transformers requires addressing knowledge base construction scalability.
2.  Offline MKB construction requires training a large number of models (67,760), incurring high initial costs.
3.  OOD adaptation improvement is limited under extreme distribution shifts (Cornell's $R^2$ only improved from 0.03 to 0.11), suggesting a need for enhanced multi-hop reasoning.
4.  Bayesian updates assume gains follow a Gaussian distribution, which may not hold in highly non-convex design spaces.

## Related Work & Insights

1.  **DesiGNN** (Wang et al., 2026): Retrieval-augmented GNN design, but uses static similarity and lacks online calibration.
2.  **AutoTransfer** (Cao et al., 2023): Embedding-based model transfer, suffering from similar static similarity issues.
3.  **NAS-Bench-Graph** (Qin et al., 2022): A GNN architecture search benchmark using rank correlation to measure task similarity.

## Rating
- Novelty: 9/10 — Reformulates model design as retrieval-augmented optimization on a Modification-Gain Graph; Bayesian dynamic task similarity is an original design.
- Experimental Thoroughness: 9/10 — 33 task-data pairs + 10 baselines + detailed ablation + theoretical validation + cross-domain experiments.
- Writing Quality: 8/10 — Formal definitions are clear and rigorous, though symbolic density is high.
- Value: 8/10 — Provides a new paradigm for AutoML, though feasibility in larger design spaces remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)

</div>

<!-- RELATED:END -->
