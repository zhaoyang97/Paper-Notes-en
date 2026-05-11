---
title: >-
  [Paper Note] Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs
description: >-
  [NeurIPS 2025][Graph Learning][GNN depth dilemma] Grounded in PAC-Bayes generalization theory, this paper proves that varying GNN depth induces generalization preference drift across node subgroups with different homophi…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "GNN depth dilemma"
  - "PAC-Bayes bound"
  - "decoupled mixture of experts"
  - "test-time gating"
  - "homophily subgroups"
date: 2026-05-08
content_hash: 652cc1ba2f9fecff
---

# Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs

**Conference**: NeurIPS 2025
**arXiv**: [2409.06998](https://arxiv.org/abs/2409.06998)
**Code**: [https://github.com/Hydrapse/moscat](https://github.com/Hydrapse/moscat)
**Area**: Graph Neural Networks / Generalization Theory
**Keywords**: GNN depth dilemma, PAC-Bayes bound, decoupled mixture of experts, test-time gating, homophily subgroups

## TL;DR

Grounded in PAC-Bayes generalization theory, this paper proves that varying GNN depth induces generalization preference drift across node subgroups with different homophily levels. It proposes Moscat—a post-processing attention-gating model that adaptively fuses independently trained GNN experts of different depths at test time on a per-node basis—achieving significant improvements across diverse GNN architectures and datasets.

## Background & Motivation

**Background**: GNNs perform well on homophilic graphs (where connected nodes are similar) but struggle on heterophilic graphs (where connected nodes are dissimilar). Increasing GNN depth expands the receptive field (scope), enabling the model to seek homophily from higher-order neighbors; however, deeper GNNs consistently suffer from performance degradation in practice.

**Limitations of Prior Work**: The GNN depth dilemma has been attributed to three issues—over-smoothing (loss of expressivity), optimization degradation (gradient problems), and generalization gap (overfitting). Existing solutions (skip connections, regularization, parameter reduction, etc.) tend to alleviate one problem while exacerbating another. Existing Graph MoE methods employ "soft scoping," jointly training experts of different depths through gating; however, gradient backpropagation exposes shallow experts to deep-layer noisy information, leading to overfitting.

**Key Challenge**: Prior discussions adopt a global perspective ("is deeper better or shallower better?"), overlooking the diversity of local structural contexts across nodes. In practice, deeper GNNs perform better on certain node subgroups and worse on others—the question is not "how deep" but "how deep for which nodes."

**Goal**: To improve the overall generalization performance of deep GNNs while preserving their expressivity.

**Key Insight**: The authors derive new subgroup generalization bounds via PAC-Bayes analysis, theoretically establishing that depth variation causes generalization preference drift. They validate this empirically (the Jaccard overlap between correctly predicted node sets across GNNs of different depths is very low), and subsequently design a decoupled expert-gating paradigm.

**Core Idea**: GNNs of different depths are trained independently (hard scoping), and a lightweight attention-based gating model learns node-adaptive expert fusion weights on a holdout set.

## Method

### Overall Architecture

Three steps:

1. **Expert Training**: Train $L_{\max}+1$ GNN models independently (depth ranging from 0 to $L_{\max}$, where depth 0 is an MLP), using identical architectures and hyperparameters, differing only in depth.

2. **Gating Training**: Collect logits from all experts on the training set and a holdout set, apply scope-aware logit augmentation, and train an attention gating model on the holdout set.

3. **Test-Time Fusion**: The gating model computes per-node weights for each expert; the final prediction is obtained by weighted aggregation.

### Key Designs

1. **Decoupled Hard-Scope Expert-Gating Paradigm**:

    - Function: Ensures that the GNN at each depth learns only from information within its corresponding receptive field, without gradient contamination from other levels.
    - Mechanism: Unlike Graph MoE's joint training (soft scoping), Moscat first trains all experts independently, then trains the gating model separately. The gating model is trained on a holdout set (disjoint from the expert training set), so as to accurately measure each expert's generalization ability rather than its training fit.
    - Design Motivation: Under soft scoping, gradients propagate back from deeper experts to shallower ones through the gating mechanism, causing shallow experts to encode deep-layer information and thus overfit. Hard scoping physically isolates the training processes, eliminating this issue at its root.

2. **Heterophily-Biased Sample Filtering**:

    - Function: Removes noisy samples from the gating training data.
    - Mechanism: Experts tend to overfit on heterophilic nodes (nodes with diverse neighbor labels)—predicting correctly during training but generalizing poorly. Such samples mislead the gating model into assigning high weights to incorrect experts. A hyperparameter $\gamma \in [0,1]$ is introduced to randomly filter out heterophilic nodes from the expert training set. Optionally, nodes on which all experts make incorrect predictions (likely noisy data or shared blind spots across all architectures) are also removed.
    - Design Motivation: The quality of gating training directly determines the quality of fusion—data fed to the gating model must faithfully reflect each expert's generalization patterns, rather than its overfitting behavior during training.

3. **Scope-Aware Logit Augmentation**:

    - Function: Provides richer signals to the gating model for identifying each expert's generalization patterns.
    - Mechanism: Two types of augmentation are used: (a) Label embeddings—expert logits are used as pseudo-labels to compute pseudo-label distributions over neighbors of each order, $\xi_{\text{label}}^{(L)} = [\tilde{A}^1 Z^{(L)} \| \cdots \| \tilde{A}^{L_{\max}} Z^{(L)}]$, approximating homophily ratios at each hop; (b) Structural encodings—distances between aggregated features at each layer and either the original features or fully smoothed features, $\bar{\epsilon}_v^{(L)}, \tilde{\epsilon}_v^{(L)}$, are computed alongside PageRank centrality to detect the degree of over-smoothing.
    - Design Motivation: Theorem 3.3 shows that generalization preference correlates with node homophily, but labels are unavailable at test time. Label embeddings serve as a label-free proxy. Structural encodings help the gating model identify which experts are experiencing over-smoothing at which nodes.

### Gating Model Architecture

The augmented logit $\zeta^{(L)} \in \mathbb{R}^{F_{\text{aug}}}$ for each expert (with $F_{\text{aug}} = C + L_{\max} C + 3$) is mapped to a hidden representation $H_L$ via **expert-specific transformation weights** $W_L, a_L$, followed by sigmoid attention and cross-expert softmax to compute node-adaptive weights $g_L$. The final prediction is produced by an MLP classifier applied to the weighted mixture of hidden representations. Expert-specific transformation weights are critical—different experts exhibit different overfitting/over-smoothing patterns, necessitating dedicated feature extraction.

### Loss & Training

Both the gating model and experts use the same cross-entropy loss. Expert hyperparameters are inherited directly from the baseline; only gating hyperparameters are tuned. The holdout set can be 90% of the validation set, with the remaining 10% reserved for gating validation.

## Key Experimental Results

### Main Results (Multiple GNN Architectures × Multiple Datasets)

| GNN Architecture | Avg. Gain over 8 Datasets (Moscat*) | Avg. Gain over 8 Datasets (Moscat) |
|------------------|-------------------------------------|-------------------------------------|
| SGC | +6.05% | +6.53% |
| GCN | +4.25% | +4.96% |
| GAT | +6.29% | +6.90% |
| GCNII | +3.59% | +4.05% |
| ACMGCN | +11.97% | +12.28% |

### Comparison with State-of-the-Art

Selecting the best GNN architecture paired with Moscat on each of 8 datasets, the method is compared against heterophilic GNNs (H2GCN, GPRGNN, FSGNN), Graph Transformers (GraphGPS, SGFormer, Polynormer), and Graph MoE methods: Moscat achieves the best performance on 6 out of 8 datasets.

### Ablation Study

| Configuration | Effect | Remarks |
|---------------|--------|---------|
| Soft scoping (joint training) vs. hard scoping (independent training) | Hard scoping is superior | Shallow experts overfit to deep-layer noise under soft scoping |
| w/o label embeddings | Performance drops | Homophily information is critical for gating decisions |
| w/o structural encodings | Performance drops | Over-smoothing detection helps gating avoid failing experts |
| w/o heterophily-biased filtering | Performance drops | Training noise distorts the gating model's generalization learning |
| Multi-depth ensemble vs. same-depth multi-seed ensemble | Multi-depth is far superior | Complementarity from depth diversity far exceeds that from random variation |

### Key Findings

- The oracle ensemble accuracy (selecting the best depth per node) far exceeds the best single-depth model, demonstrating large complementary potential.
- Gains are largest on heterophilic graphs (amazon-ratings, Penn94) and smaller on homophilic graphs (PubMed)—precisely consistent with the predictions of Theorem 3.3.
- MLP (0-layer GNN) as a "zero-hop expert" contributes non-negligibly and performs best for certain nodes.
- Every GNN architecture benefits from Moscat, confirming the generality of the approach.

## Highlights & Insights

- **Theory-Driven Method Design**: The PAC-Bayes subgroup generalization bound (Theorem 3.3) not only explains observed phenomena but directly guides method design. The term $\Gamma_{L-1} = \mathbb{E}[(p_o - q_o)^{L-1}]$ in the bound clearly shows how increasing depth shifts the generalization boundary for different subgroups—the optimal depth differs when $p_S > p_m$ versus $p_S < p_m$. This theory–empirics–method logical chain is exemplary.
- **Elegance of the Post-Processing Paradigm**: Moscat is a purely post-processing method—it does not modify any expert's training process, does not increase training complexity, and any existing GNN model can directly benefit. This zero-invasiveness design substantially lowers the adoption barrier.
- **Incisive Critique of Graph MoE**: The paper reveals the fundamental flaw of soft scoping—gradient leakage causes corruption of shallow experts—and resolves it elegantly via hard scoping and decoupled training. This insight is not limited to GNNs but carries reference value for any MoE system.

## Limitations & Future Work

- Training $L_{\max}+1$ independent expert models is required (with the default $L_{\max}=6$, this amounts to 7 models), making total training cost approximately 7× that of a single model. Although the gating model itself is lightweight, the expert training cost is non-negligible.
- A holdout set must be reserved for gating training, further partitioning already scarce labeled data—potentially disadvantageous in settings with very few annotations.
- The theoretical analysis is based on the Contextual Stochastic Block Model (CSBM), which entails considerable simplifying assumptions about real-world graphs. Although experiments confirm the conclusions, a theory-practice gap remains.
- The selection of maximum depth $L_{\max}$ lacks automated guidance and must be specified manually.
- Gains on large-scale homophilic graphs are limited; the core advantages of the method are concentrated on heterophilic or mixed-homophily graphs.

## Related Work & Insights

- **vs. GraphGPS/SGFormer (Graph Transformers)**: Graph Transformers bypass the depth problem by replacing local aggregation with global attention, but suffer from OOM issues on large graphs. Moscat retains the efficiency advantages of GNNs.
- **vs. Graph MoE (GraphMoE, MoWST, DA-MoE)**: All employ soft scoping with joint training—gating and experts are trained together, with mutual gradient influence. Moscat's decoupled paradigm constitutes a fundamental distinction.
- **vs. Skip-Connection Methods (GCNII, Jumping Knowledge)**: These methods mix information across different layers via residual connections during the forward pass. Moscat mixes at post-processing (inference) time—without altering the learning objectives of individual layers.
- **Insights**: The "independent training + post-processing gating" paradigm has strong generality and can be transferred to NLP (LLMs with different context window sizes), CV (CNNs with different receptive fields), and any domain involving a scale-quality trade-off.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The PAC-Bayes subgroup bound combined with the decoupled gating paradigm opens an entirely new perspective for understanding and improving the GNN depth dilemma.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five GNN architectures × 8 datasets × oracle analysis × extensive ablations—statistically convincing.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from theoretical motivation → empirical validation → method design is exceptionally fluent; Figures 1 and 2 precisely convey the core insights.
- Value: ⭐⭐⭐⭐⭐ Provides a fundamentally new understanding of the GNN depth dilemma; the method is highly general and plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[NeurIPS 2025\] What Expressivity Theory Misses: Message Passing Complexity for GNNs](what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)
- [\[NeurIPS 2025\] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](../../AAAI2026/graph_learning/self-adaptive_graph_mixture_of_models.md)

</div>

<!-- RELATED:END -->
