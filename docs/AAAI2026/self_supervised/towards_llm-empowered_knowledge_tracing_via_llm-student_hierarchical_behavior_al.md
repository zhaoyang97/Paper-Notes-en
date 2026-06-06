---
title: >-
  [Paper Note] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space
description: >-
  [AAAI 2026][Self-Supervised Learning][Knowledge Tracing] This paper proposes L-HAKT, a framework that for the first time integrates LLM dual-agent design with hyperbolic geometry for knowledge tracing. A Teacher Agent pa…
tags:
  - "AAAI 2026"
  - "Self-Supervised Learning"
  - "Knowledge Tracing"
  - "LLM Dual-Agent"
  - "Hyperbolic Space"
  - "Hierarchical Knowledge Graph"
  - "Contrastive Alignment"
  - "Learning Behavior Simulation"
date: 2026-05-08
content_hash: 00f7f59b775331f6
---

# Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space

**Conference**: AAAI 2026
**arXiv**: [2602.22879](https://arxiv.org/abs/2602.22879)  
**Area**: Educational Intelligence / Knowledge Tracing
**Keywords**: Knowledge Tracing, LLM Dual-Agent, Hyperbolic Space, Hierarchical Knowledge Graph, Contrastive Alignment, Learning Behavior Simulation

## TL;DR

This paper proposes L-HAKT, a framework that for the first time integrates LLM dual-agent design with hyperbolic geometry for knowledge tracing. A Teacher Agent parses exercise semantics and constructs a hierarchical knowledge graph, while a Student Agent simulates individual learning behaviors to generate synthetic interaction data. Hyperbolic contrastive learning is employed to calibrate the distributional gap between synthetic and real data. L-HAKT achieves an AUC of up to 80.29% across four educational datasets, with an AUC improvement of 13.03% over the GKT baseline on EdNet.

## Background & Motivation

**Background**: Knowledge Tracing (KT) is a core technique in educational intelligence, dynamically inferring students' knowledge states by analyzing historical interaction data. Existing approaches fall into two paradigms: sequential modeling (DKT, SAKT, SAINT, etc., based on RNN/Transformer) and graph-based modeling (GKT, SKT, etc., based on GNN).

**Limitations of Prior Work**:
(1) **Lack of hierarchical concept representation** — conventional methods operate in Euclidean space, where flat geometry cannot capture the tree-structured hierarchy of knowledge systems (e.g., basic definitions → derivations → synthesis);
(2) **Insufficient utilization of exercise semantics** — existing methods rely on simple IDs or shallow textual features, failing to exploit the topological relationships among knowledge concepts latent in exercise text;
(3) **Neglect of individual cognitive bias** — the population-level distribution in training data distorts individual difficulty perception (e.g., a model trained on low-ability populations may incorrectly label medium-difficulty exercises as highly difficult).

**Key Challenge**: The hierarchical dependencies among knowledge concepts exhibit a naturally tree-structured topology, yet the exponential volume growth in Euclidean space prevents efficient representation of such structures (as shown in Figure 1, the hyperbolic metric of the student–exercise–concept relational graph approaches 0, indicating strong tree-like properties).

**Key Insight**: The paper leverages LLM dual agents to extract hierarchical knowledge structures from exercise semantics and simulate learning behaviors, explicitly models hierarchical dependencies in hyperbolic space, and applies contrastive learning to align synthetic and real data for cognitive bias calibration.

## Method

### Overall Architecture

L-HAKT consists of three components:
(1) **LLM Dual-Agent Behavior Augmentation**: the Teacher Agent constructs a hierarchical knowledge graph and quantifies exercise difficulty; the Student Agent simulates individual learning behaviors to generate synthetic interaction data.
(2) **Hyperbolic Encoding and Alignment**: a relation-aware hyperbolic graph neural network encodes hierarchical structures; hyperbolic contrastive learning aligns synthetic and real data distributions.
(3) **Hyperbolic Knowledge State Tracing**: GRU-based sequential propagation with alternating computation between the tangent space and the hyperbolic manifold.

### Key Designs

1. **Teacher–Student LLM Dual-Agent**

    - **Teacher Agent**: processes exercise images via a VLM (e.g., Qwen-2.5VL) and produces three types of outputs — (a) hierarchical knowledge concept identification: concepts are categorized into 4 levels ($L \in \{1,2,3,4\}$, from basic definitions to complex reasoning); (b) structured knowledge graph construction: parent–child dependency relations among concepts are established to form a tree-structured pedagogical hierarchy; (c) exercise difficulty quantification: an objective difficulty score is computed based on the level combination of associated knowledge concepts.
    - **Student Agent**: simulates learning behaviors based on historical interaction sequences via two specialized modules: (a) a cognitive engagement module — dynamically estimates learning engagement based on exercise difficulty, concept mastery, and time interval, $\Gamma_j = \sigma(W_q \cdot [X_{q_i}; X_{c_{ij}}; t] + b_q)$; (b) a hierarchical forgetting module — models differentiated forgetting across concept difficulty levels, $F_j = \exp(-\lambda \cdot L_{avg} \cdot t_j)$, with higher-level knowledge decaying faster.
    - **Knowledge State Update**: $\mathbf{h}_j^s = \text{LSTM}([X_{q_j}; \sum_{c} w_c X_c] \oplus (\Gamma_j \odot F_j \odot h_{j-1}^s))$

2. **Relation-Aware Hyperbolic Graph Neural Network**

    - **Function**: encodes hierarchical concept dependencies in hyperbolic space, where foundational concepts reside in low-curvature central regions and higher-order concepts are distributed in high-curvature peripheral regions.
    - **Mechanism**: constructs a heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ containing exercise–concept edges and hierarchical dependency edges $\mathcal{E}_{hie} = \{(c_i, c_j) | L_{c_i} < L_{c_j}\}$. Separate curvatures $\kappa_{real}$ and $\kappa_{syn}$ are defined for real and synthetic data, respectively. Euclidean embeddings are projected onto the hyperbolic manifold via exponential maps, and attention-based aggregation is performed in the tangent space before mapping back.
    - **Update Formula**: $\mathbf{h}_i^{(L+1)} = \exp_0^\kappa(\sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(L)} \cdot \mathbf{W}^{(L)} \log_0^\kappa(\mathbf{h}_j^{\mathbb{H}_\kappa^{(L)}})))$
    - **Design Motivation**: the exponentially growing volume of hyperbolic space is inherently suited to representing tree-structured hierarchies — achieving low distortion and high efficiency.

3. **Hyperbolic Contrastive Alignment Mechanism**

    - **Function**: reduces the distributional gap between synthetic and real student interaction data on key features such as exercise difficulty and forgetting patterns.
    - **Mechanism**: in hyperbolic space, exercise–concept pairs shared between real and synthetic embedding spaces serve as positive samples, while other entities serve as negatives. A contrastive loss $\mathcal{L}_{con}$ attracts positive pairs and repels negative pairs.
    - **Design Motivation**: although LLM-generated synthetic data supplements reasoning path information, it suffers from distributional shift relative to real data; contrastive learning effectively calibrates this discrepancy.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{KT} + \alpha \mathcal{L}_{con}$, where $\mathcal{L}_{KT}$ is the binary cross-entropy prediction loss and $\alpha$ controls the weight of the contrastive loss. Datasets are split 80%/20% for training and testing.

## Key Experimental Results

### Main Results — AUC/ACC Comparison (16 Baselines)

| Model | ASSIST09 AUC | ASSIST12 AUC | EdNet AUC | Eedi AUC |
|-------|-------------|-------------|----------|---------|
| DKT | 75.97 | 72.90 | 70.10 | 76.01 |
| AKT | 78.23 | 78.21 | 76.78 | 78.84 |
| GIKT | 77.33 | 76.32 | 76.02 | 79.68 |
| MIKT | 79.38 | 78.65 | 77.10 | 79.59 |
| **L-HAKT** | **80.22** | **80.27** | **78.23** | **80.29** |

### Ablation Study — Contributions of Hyperbolic Space and Contrastive Learning

| Configuration | ASSIST09 AUC | Δ | EdNet AUC | Δ |
|--------------|-------------|--|----------|--|
| GKT (baseline) | 76.32 | - | 69.21 | - |
| L-HVKT (w/o hyperbolic) | 77.55 | +1.61% | 76.54 | +10.59% |
| L-HVKT (w/o contrastive) | 76.98 | +0.86% | 75.51 | +9.10% |
| **L-HAKT (full)** | **80.22** | **+5.11%** | **78.23** | **+13.03%** |

### Key Findings

- L-HAKT achieves state-of-the-art performance on all four datasets, with a maximum AUC improvement of 13.03% over GKT on EdNet.
- Both hyperbolic space and contrastive learning contribute independently: removing the hyperbolic constraint reduces ASSIST09 AUC by 2.67%, while removing contrastive learning reduces it by 3.24%.
- Strong performance on Eedi (which contains image-based exercises) validates the effectiveness of the VLM-based exercise image parsing strategy.
- Compared to AKT (a pure Transformer attention-based method), L-HAKT achieves approximately 2% AUC improvement across all datasets.

## Highlights & Insights

1. **Well-motivated dual-agent design**: the Teacher Agent provides objective knowledge structure while the Student Agent simulates subjective learning behavior, yielding a complementary collaboration.
2. **Hierarchical forgetting modeling aligns with cognitive theory**: higher-level knowledge decays faster while foundational knowledge is retained longer, consistent with educational psychology.
3. **Theoretically grounded choice of hyperbolic space**: the tree-structured hypothesis is empirically verified by computing the hyperbolic metric of real data (H-all approaching 0).
4. **Effective deployment of synthetic data via contrastive learning**: distributional shift arising from direct use of LLM-generated synthetic data is successfully mitigated.

## Limitations & Future Work

1. The inference overhead of LLM agents is substantial, making large-scale deployment costly.
2. Knowledge concepts are categorized into only 4 levels, which may lack sufficient granularity.
3. Numerical stability of hyperbolic space operations (exponential/logarithmic maps) may degrade at high curvatures.
4. Evaluation is limited to the knowledge tracing task; extension to downstream applications such as exercise recommendation and learning path planning has not been explored.
5. The Student Agent's cognitive engagement and forgetting models are relatively simplified; more refined cognitive science models could be incorporated.

## Related Work & Insights

- The Teacher–Student dual-agent collaboration paradigm can be generalized to other educational intelligence scenarios, such as automatic question generation and personalized tutoring.
- The success of hyperbolic space in hierarchical structure modeling can inspire further exploration in knowledge graph completion and recommender systems.
- The strategy of synthetic data combined with contrastive alignment offers a new paradigm for addressing data scarcity in educational settings.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: first to combine LLM dual-agent design with hyperbolic geometry for knowledge tracing.
- **Experimental Thoroughness** ⭐⭐⭐⭐: four datasets, sixteen baselines, and detailed ablation studies.
- **Writing Quality** ⭐⭐⭐: some notation definitions are insufficiently rigorous and formula typesetting has room for improvement.
- **Value** ⭐⭐⭐⭐: provides a theoretically motivated new framework for knowledge tracing in educational intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[ICLR 2026\] PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[ICML 2026\] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](../../ICML2026/self_supervised/the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)

</div>

<!-- RELATED:END -->
