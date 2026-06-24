---
title: >-
  [Paper Note] Graph Generative Pre-trained Transformer (G2PT)
description: >-
  [ICML2025][Computational Biology][Graph Generation] Proposes G2PT, which encodes a graph as a sequence of node and edge tokens, generates graphs using a GPT-style autoregressive Transformer via next-token prediction, and achieves goal-directed molecular generation through Rejection Fine-Tuning (RFT) and PPO reinforcement learning, achieving SOTA performance on both general graph and molecular datasets.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Graph Generation"
  - "Autoregressive Transformer"
  - "Edge Sequence Representation"
  - "Molecular Generation"
  - "Pre-training & Fine-tuning"
date: 2026-05-08
content_hash: 0b4a8727e94e3056
---

# Graph Generative Pre-trained Transformer (G2PT)

**Conference**: ICML2025  
**arXiv**: [2501.01073](https://arxiv.org/abs/2501.01073)  
**Code**: [tufts-ml/G2PT](https://github.com/tufts-ml/G2PT)  
**Area**: Graph Generation / Molecular Design  
**Keywords**: Graph Generation, Autoregressive Transformer, Edge Sequence Representation, Molecular Generation, Pre-training & Fine-tuning

## TL;DR

Proposes G2PT, which encodes a graph as a sequence of node and edge tokens, generates graphs using a GPT-style autoregressive Transformer via next-token prediction, and achieves goal-directed molecular generation through Rejection Fine-Tuning (RFT) and PPO reinforcement learning, achieving SOTA performance on both general graph and molecular datasets.

## Background & Motivation

- **Core Problem**: Most graph generative models represent graphs using adjacency matrices with $O(n^2)$ computational complexity, which is inefficient for sparse graphs. Discrete diffusion models sample matrix elements independently at each step, suffering from compounding decoding errors.
- **Limitations of Prior Work**:
    - Diffusion models (DiGress, EDGE) require multi-step denoising and permutation-invariant GNN architectures, limiting architectural choices.
    - Early sequential models (GraphRNN, DeepGMG) based on RNN/LSTM still flatten the adjacency matrix by row, resulting in sequence lengths of $O(n^2)$.
    - Edge-list methods (DeepGMG, BiGG) rely on GNNs to learn node representations, which has limited expressive power.
- **Design Motivation**: Inspired by the tremendous success of LLMs, this work designs an efficient edge-list-based token sequence representation to make general Transformer architectures directly applicable to graph generation.

## Method

### 1. Token Sequence Representation of Graphs

Encodes a graph $G = (V, E)$ into a one-dimensional sequence that lists nodes first, followed by edges:

$$[\underbrace{v_1^c, v_1^{id}, \ldots, v_n^c, v_n^{id}}_{n \times 2},\ a_\Delta,\ \underbrace{v_{src}^{id}, v_{dest}^{id}, e_1^c, \ldots, v_{src}^{id}, v_{dest}^{id}, e_m^c}_{m \times 3}]$$

- Node token: $(v^c, v^{id})$, representing type and index
- Edge token: $(v_{src}^{id}, v_{dest}^{id}, e^c)$, representing source node, destination node, and edge type
- $a_\Delta$: A special token separating the node segment and the edge segment
- Sequence length is $O(2n + 3m)$, which is far superior to the $O(n^2)$ of adjacency matrices for sparse graphs

### 2. Edge Ordering Strategy

Adopts a **Degree-Based Edge Removal** strategy in reverse order:
- Removes the edge of the current lowest-degree node at each step. Its reverse order yields an edge sequence that "constructs the dense core first, then expands to the periphery".
- Compared to BFS/DFS ordering, it is more suitable for generative learning on sparse graphs.

### 3. Autoregressive Training

Uses a standard Transformer decoder. The vocabulary includes node IDs $\{1,\ldots,n_{max}\}$, node types, edge types, and special tokens [SOG]/[EOG]/$a_\Delta$.

The pre-training loss is the negative log-likelihood:

$$\mathcal{L}_{pt}(\mathbf{s};\theta) = -\log p_\theta(\mathbf{s}) = \sum_{l=1}^{L} -\log p_\theta(s_l | \mathbf{s}_{<l})$$

**Theoretical Guarantee**: Maximizing sequence likelihood is equivalent to maximizing the lower bound of the graph likelihood. This lower bound can be tightened through data augmentation (multiple sequences for the same graph).

### 4. Goal-Directed Fine-Tuning

**Rejection Fine-Tuning (RFT)**:
- Samples from the pre-trained model and retains samples satisfying $|z^* - \zeta(G)| < \omega$ to construct the fine-tuning set.
- Self-Bootstrap (SBS): Progressively tightens the threshold $\omega_1 > \omega_2 > \ldots > \omega_k$ to approach the target distribution through multi-round iterations.

**Reinforcement Learning (PPO)**:
- KL-regularized objective: $\phi^* = \arg\max_\phi \mathbb{E}_{p_\phi}[r_{z^*}(\mathbf{s}) - \rho_1 \mathrm{KL}(p_\phi \| p_\theta)]$
- Rewards are assigned only at the [EOG] token, with zero rewards at other positions.
- The critic model is initialized with the same architecture. The overall loss is: $\mathcal{L}_{ppo} = \mathcal{L}_{pg\text{-}clip} + \rho_2 \mathcal{L}_{critic} + \rho_3 \mathcal{L}_{pt}$

### 5. Graph Property Prediction Fine-Tuning

Takes the last-layer Transformer activation $\mathbf{h}$ of the last token as the graph representation, which is followed by an MLP for classification:

$$p(y|\mathbf{s}) = \text{Softmax}(\text{Dropout}(\text{Linear}(\mathbf{h})))$$

Unfreezing the parameters of the latter half of the Transformer for fine-tuning yields significantly better results than freezing all parameters.

## Key Experimental Results

### Model Specifications

| Specification | Layers | Attention Heads | $d_{model}$ | Parameters |
|------|------|---------|------------|--------|
| Small | 6 | 6 | 384 | ~10M |
| Base | 12 | 12 | 768 | ~85M |
| Large | 24 | 16 | 1024 | ~300M |

### General Graph Generation (Table 2, V.U.N. ↑)

| Model | Planar | Tree | Lobster | SBM |
|------|--------|------|---------|-----|
| DiGress | 77.5 | 90 | - | 60 |
| HSpectre | 95 | 100 | - | 45 |
| DeFoG | 99.5 | 96.5 | - | 90 |
| **G2PT-base** | **100** | 99 | **100** | **100** |

### Molecular Generation (Table 4)

| Model | MOSES Validity↑ | MOSES FCD↓ | GuacaMol Validity↑ | GuacaMol FCD↑ |
|------|-----------------|-----------|--------------------|--------------| 
| DiGress | 85.7 | 1.19 | 85.2 | 68.0 |
| DeFoG | 92.8 | 1.95 | 99.0 | 73.8 |
| **G2PT-large** | **97.2** | **1.02** | **95.3** | **92.7** |

### QM9 (FCD↓)

| Model | FCD↓ |
|------|------|
| DisCo | 0.25 |
| Cometh | 0.11 |
| **G2PT (all sizes)** | **0.06** |

### Molecular Property Prediction (MoleculeNet, ROC-AUC, Table 5)

| Model | Average |
|------|------|
| GraphMAE | 73.3 |
| **G2PT-base** | **73.3** |
| G2PT-base (No pre-training) | 64.9 |

Pre-training brings an average improvement of +8.4%, validating the effectiveness of the graph representations learned by G2PT.

## Highlights & Insights

1. **Representation Innovation**: Replacing adjacency matrices with edge-list token sequences reduces sequence length from $O(n^2)$ to $O(n+m)$. On the Planar dataset, the number of tokens is reduced from 2018 to 737, while achieving even better performance.
2. **Unified Pre-training & Fine-tuning Paradigm**: Similar to NLP, it follows a "pre-training + task fine-tuning" pipeline, where both RFT and PPO can effectively guide goal-directed molecular generation.
3. **RFT+SBS vs PPO**: RFT combined with multi-round bootstrapping far outperforms PPO on challenging tasks like GSK3β. PPO is overly constrained by KL regularization, making it difficult to cross distributional barriers.
4. **Scaling Laws**: Scaling model parameters from 1M to 1.5B demonstrates clear scaling trends. Data augmentation (multiple sequences per graph) also shows significant efficacy.
5. **No Permutation Invariance Needed**: Dispensing with the permutation invariance assumption of graphs and compensating via data augmentation yields better practical results than diffusion methods that rely on permutation-invariant GNNs.

## Limitations & Future Work

1. **Sensitivity to Ordering**: Different graph domains may require different edge ordering strategies; a universal ordering scheme is currently lacking.
2. **Non-utilization of 3D Information**: Uses only 2D topology, without incorporating 3D stereoconformation or chirality information of molecules.
3. **Insufficient Scaffold Diversity**: The scaffold similarity on MOSES is relatively low (only 2.9 for G2PT-large), indicating that the model tends to memorize training set scaffolds.
4. **Scalability to Large Graphs**: Although more efficient than adjacency matrices, the $O(L^2)$ attention complexity of Transformers remains a bottleneck on extremely large graphs.
5. **Limited Effectiveness of PPO**: On high-difficulty target property tasks (GSK3β), PPO fails to break through distribution barriers. RFT+SBS is more reliable but requires multi-round sampling.

## Related Work & Insights

- **DiGress / EDGE**: Discrete diffusion baselines, denoising full adjacency matrices. They are permutation-invariant but feature limited architectural choices.
- **BiGG**: Bipartite graph generation, directly modeling edge lists but relying on GNNs for node representation.
- **GraphMAE**: Self-supervised pre-trained graph representation. G2PT's property prediction is on par with it while using only 2D information.
- **Inspiration**: Bringing the GPT paradigm to structured data (graphs/molecules) is a promising direction; tokenized representation design is a key bottleneck.

## Rating

- Novelty: ⭐⭐⭐⭐ (Edge sequence representation + GPT-style graph generation paradigm, clear and effective idea)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 generation datasets + 8 prediction datasets + scaling analysis + goal-directed generation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, comprehensive theoretical derivation)
- Value: ⭐⭐⭐⭐ (Unifies the pre-training & fine-tuning paradigm for graph generation, highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis](eccdnamamba_a_pre-trained_model_for_ultra-long_eccdna_sequence_analysis.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[ICML 2026\] Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](../../ICML2026/computational_biology/routing_by_reaching_composition_of_pre-trained_gflownets_for_multi-objective_gen.md)
- [\[ICLR 2026\] A Joint Diffusion Model with Pre-Trained Priors for RNA Sequence-Structure Co-Design](../../ICLR2026/computational_biology/a_joint_diffusion_model_with_pre-trained_priors_for_rna_sequence-structure_co-de.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/computational_biology/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)

</div>

<!-- RELATED:END -->
