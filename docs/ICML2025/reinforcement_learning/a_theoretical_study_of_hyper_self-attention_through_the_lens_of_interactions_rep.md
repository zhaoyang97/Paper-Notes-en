---
title: >-
  [Paper Note] A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization
description: >-
  [ICML 2025][Reinforcement Learning][self-attention] From the unified perspective of "interacting entities", this paper proves that a single-layer linear self-attention can efficiently represent, learn, and generalize pairwise interaction functions with $\Theta(|\mathcal{S}|^2)$ parameters (whereas fully connected networks require $\Omega(L^2|\mathcal{S}|^2)$). Based on this theoretical insight, two new modules, HyperFeatureAttention (feature-level interaction coupling) and Hy…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "self-attention"
  - "mutual interaction"
  - "representation theory"
  - "HyperAttention"
  - "length generalization"
date: 2026-05-08
content_hash: c1f7362f7052c2ff
---

# A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization

**Conference**: ICML 2025  
**Authors**: Muhammed Ustaomeroglu, Guannan Qu (CMU)  
**arXiv**: [2506.06179](https://arxiv.org/abs/2506.06179)  
**Code**: None  
**Area**: Theoretical Analysis, Self-Attention  
**Keywords**: self-attention, mutual interaction, representation theory, HyperAttention, length generalization  

## TL;DR

From the unified perspective of "interacting entities", this paper proves that a single-layer linear self-attention can efficiently represent, learn, and generalize pairwise interaction functions with $\Theta(|\mathcal{S}|^2)$ parameters (whereas fully connected networks require $\Omega(L^2|\mathcal{S}|^2)$). Based on this theoretical insight, two new modules, HyperFeatureAttention (feature-level interaction coupling) and HyperAttention (higher-order multi-entity interactions), are proposed, which reduce perplexity in language modeling.

## Background & Motivation

**Background**: Self-attention is the core component of Transformers and has been widely applied in NLP, CV, protein structure prediction, and reinforcement learning. However, its theoretical understanding remains preliminary, and existing analyses are mostly confined to specific tasks (e.g., in-context learning, image classification).

**Limitations of Prior Work**: (1) Existing theoretical analyses target isolated problems and lack a unified cross-domain perspective; (2) most rigorous theories ignore test-time generalization, especially out-of-distribution (OOD, length generalization) scenarios; (3) existing theories can only explain a subset of pre-configured parameters, leaving many learned parameters appearing counter-intuitive; (4) restrictive assumptions are imposed on model parameters.

**Key Challenge**: The success of self-attention across various domains implies some unified fundamental capability, yet existing theories are fragmented and fail to reveal this unity.

**Key Insight**: Each token is viewed as an "interacting entity" (e.g., agents in MARL, alleles in DNA, patches/pixels in images). The dependencies between tokens can be modeled uniformly as "pairwise interaction functions." This abstraction naturally holds across multiple domains.

**Core Idea**: Self-attention is essentially an efficient learner of pairwise interactions — using orthogonal embeddings to encode interactions between entities into the attention score matrix, and designing stronger attention variants based on this theoretical foundation.

## Method

### Overall Architecture

The paper is divided into two main parts:
1. **Theoretical Analysis** (Sec 3-4): Proves the representational sufficiency, parameter efficiency, training convergence, and generalization capability of linear self-attention for pairwise interaction functions.
2. **New Module Design** (Sec 5-6): Proposes HyperFeatureAttention and HyperAttention based on theoretical insights.

### Key Designs

1. **Unified Modeling of Pairwise Interaction Functions**:

    - Function: Unifies multi-domain problems into a single mathematical framework.
    - Mechanism: Given a domain $\mathcal{S}$ (vocabulary) and a sequence $\mathcal{X}$ of length $L$, the aggregated influence of all other entities on the $i$-th entity is defined as $\mathbf{y}_{\mathcal{X}(i)} = \sum_{j \in [L]} f(\mathcal{X}(i), \mathcal{X}(j)) \mathbf{w}_{\mathcal{X}(j)}$, where $f:\mathcal{S}\times\mathcal{S}\to\mathbb{R}$ measures the interaction strength, and $\mathbf{w}$ denotes how the influence is expressed. Multi-agent collisions (value function depending on relative positions), genotype-phenotype mapping (allele activation dependencies), and time-series forecasting (lag dependency) naturally fit this formulation.
    - Design Motivation: To find a function class that is sufficiently simple yet highly general, making theoretical analysis feasible and conclusions transferable.

2. **Representation and Efficiency Theorems (Theorem 3.1 & 3.2)**:

    - Function: Proves the representational capacity and parameter efficiency of self-attention for pairwise interactions.
    - Mechanism: When the embedding dimension $d=|\mathcal{S}|$ (domain size), orthogonal embeddings permit the construction of $\mathbf{C}$ and $\mathbf{W}_V$ such that $\text{SA}_{\text{lin}}(\mathbf{X}) = (\mathbf{X}\mathbf{C}\mathbf{X}^\top)\mathbf{X}\mathbf{W}_V$ exactly represents any pairwise interaction function. Conversely, $d \geq |\mathcal{S}|$ is a necessary condition. In contrast, fully connected networks require $\Omega(L^2 \cdot |\mathcal{S}|^2)$ parameters — the extra $L^2$ factor arises because they lack an inherent weight-sharing mechanism.
    - Design Motivation: To answer the fundamental question of "why use self-attention instead of MLP" — the parameter efficiency of self-attention stems from its inductive bias of sharing interaction patterns across different token positions.

3. **Convergence and Generalization Theorems (Theorem 4.4, 4.6, 4.8)**:

    - Function: Proves the global convergence and zero-error generalization of gradient flow.
    - Mechanism: (1) **Convergence**: Under the initialization $\mathbf{C}(0)=\mathbf{0}$ and $\langle\mathbf{x}(\alpha), \mathbf{w}(0)\rangle \geq b > 0$, gradient flow converges to zero training error on the MSE loss. The key step is to prove that the minimum eigenvalue of $\mathbf{M}^\top\mathbf{M}$ has a positive lower bound — this is guaranteed by the **training data diversity assumption** (where $\mathbf{S}_{\mathcal{B}_\mu}$ has full column rank), and holds with probability $1 - e^{-\gamma B}$ under mild covariance positive-definiteness conditions. (2) **Generalization**: Zero training error + data diversity $\to$ zero generalization error over the entire population distribution. (3) **Length Generalization**: Under a stronger "universal realizability" assumption, a model trained on sequence length $L^*$ generalizes to any length $L$. The key insight is the invariant $\mathcal{T}_{\mu,k}(\mathbf{C}, \mathbf{W}_V) = \sum_\nu (\mathbf{x}^\top(\mu)\mathbf{C}\mathbf{x}(\nu))(\mathbf{x}^\top(\nu)\mathbf{W}_{:,k})$ given by Corollary 4.9 — all functionally equivalent parameters map to the same matrix under this transformation.
    - Design Motivation: To bridge the gap where existing theories only focus on representation but ignore learnability and generalization. Length generalization is particularly vital — self-attention naturally handles variable-length inputs, but when can it theoretically generalize to lengths unseen during training?

4. **HyperFeatureAttention (HFA)**:

    - Function: Captures the coupling of interactions across different feature dimensions.
    - Mechanism: When an entity is composed of multiple features (e.g., Agent = Position × Policy × Type), standard SA requires an embedding dimension of $d = \prod_\phi |\mathcal{S}_\phi|$ (exponential), because it treats all feature combinations as independent domain elements. HFA factorizes the attention score into the Hadamard product of multiple feature-level attention matrices: $\text{HFA}_{\text{lin}}(\mathbf{X}) = (\prod_{\odot a} \mathbf{X}\mathbf{C}^{(a)}\mathbf{X}^\top)(\prod_{\odot a} \mathbf{X}\mathbf{W}_V^{(a)})$, requiring only $d = \sum_\phi |\mathcal{S}_\phi|$ (linear).
    - Design Motivation: In non-homogeneous multi-agent environments (e.g., a "predator-prey" game, where actions depend on the coupling of type × position × policy), the parameter requirement of standard SA scales exponentially, whereas HFA recovers the same expressive power using the product of $O(M)$ independent attention heads.

5. **HyperAttention (HA)**:

    - Function: Captures three-way or multi-way higher-order interactions.
    - Mechanism: Extends attention from a 2D matrix $A_{ij}$ to a higher-order tensor $A_{i,j_1,...,j_{n-1}}$ — the output of the $i$-th token depends on the joint effect of all $n-1$ other tokens. Through parameter sharing and low-rank decomposition, the computation is reduced from $O(L^n)$ to $O(L \cdot R^2)$.
    - Design Motivation: Standard SA can only learn pairwise interactions, but real-world scenarios exhibit higher-order dependencies such as skip-trigrams — for instance, "keep...in $\to$ mind" requires simultaneously considering the relationship among three tokens. HA can allocate independent weights for each specific trigram, bypassing the skip-trigram bug.

### Loss & Training

Theoretical analysis employs MSE loss + gradient flow. Language modeling in experiments uses standard next-token prediction (cross-entropy).

## Key Experimental Results

### Theoretical Verification Experiments (Colliding Agents Environment)

| Configuration | Training MSE | Test Error (In-distribution) | Test Error (OOD Length) |
|---|---|---|---|
| One-hot embedding | $\to 0$ (converges) | $\Theta(10^{-7})$ | $\Theta(10^{-7})$ |
| Sinusoidal embedding | $\to 0$ (converges) | $\Theta(10^{-7})$ | $\Theta(10^{-7})$ |
| Learned params vs. Theoretical design params (after $\mathcal{T}$ transformation) | MSE $\sim O(10^{-4})$, functionally equivalent | — | — |

### Main Results

#### Language Modeling Experiments (OpenWebText, GPT3-small Setting)

| Model | Order | Validation Perplexity |
|---|---|---|
| Self-Attention (SA) | — | 62.28 |
| HyperFeatureAttention (HFA) | 4 | 60.22 |
| HyperAttention (HA) | 3 | 51.26 |
| HA (no weight sharing) | 3 | 48.50 |

In the 3-layer, 1024-window experiments, the HFA hybrid (2 SA + 2 HFA heads) further decreased the perplexity to 27.75 (compared to SA's 28.70).

### Key Findings

- Although the learned parameters superficially differ from the theoretical design parameters, they become completely identical after applying the transformation $\mathcal{T}_{\mu,k}$ from Corollary 4.9 — which validates the core prediction of the generalization theory.
- HFA shares the same $\Theta(L^2)$ computational complexity and parameter count as SA, but is able to capture richer feature-coupling interactions.
- HA significantly outperforms SA and HFA in language modeling, suggesting that language modeling indeed benefits from higher-order interactions that go beyond pairwise dependencies.

## Highlights & Insights

- The unified perspective that "self-attention is a pairwise interaction learner" is highly explanatory, connecting seemingly unrelated tasks such as MARL value functions, gene expression, time series, and computer vision.
- The invariant transformation in Corollary 4.9 serves as an exquisite theoretical tool — it explains why parameters learned across different random seeds appear different yet remain functionally equivalent.
- The design philosophy of HFA (using Hadamard products to achieve feature-level attention coupling) is simple and highly efficient, incurring negligible computational overhead (<0.1%), functioning as a practical enhancement for multi-head attention.

## Limitations & Future Work

- All theoretical results are based on **linear** self-attention; although the authors argue it preserves critical optimization dynamics, a gap still exists compared to the softmax SA used in practice.
- The assumption $d = |\mathcal{S}|$ is impractical in large vocabulary scenarios (e.g., 50K tokens); although Theorem B.2 provides an approximate version, the error bound depends on $\sigma_{d+1}(\mathbf{F})$.
- The language modeling experiments are relatively small in scale (GPT3-small configuration), and the efficacy of HFA/HA has not yet been validated on models with >1B parameters.
- There is a lack of empirical validation for the universal realizability assumption — do interaction functions in practical tasks truly admit exact representations by linear SA?

## Related Work & Insights

- **vs. Ahn et al. (2023, 2024)**: They proved that linear SA can implement (preconditioned) gradient descent. This work provides a complementary theoretical perspective from the viewpoint of interaction learning.
- **vs. Sanford et al. (2023)**: They characterized the representational limits of SA using tensor products. The HyperAttention proposed in this paper extends this foundation to higher-order tensors.
- **vs. Jelassi et al. (2022)**: They showed that ViTs can learn spatial structures. The framework in this work is more general — the colliding agent environment employs similar spatial interactions but applies to a wider range of domains.

## Rating

| Dimension | Rating | Reason |
|---|---|---|
| Novelty | ⭐⭐⭐⭐⭐ | Unified perspective of interacting entities + new HFA/HA module designs, highly original |
| Technical Depth | ⭐⭐⭐⭐⭐ | Quadruple theoretical guarantees (representation, convergence, generalization, and length generalization) with rigorous and complete proofs |
| Experimental Thoroughness | ⭐⭐⭐ | Thorough theoretical validation but limited in practical scale, lacking large-scale models and multi-domain experiments |
| Writing Quality | ⭐⭐⭐⭐ | Multiple concrete examples aid the understanding of abstract theories, though the page length is relatively long |
| Practicality | ⭐⭐⭐⭐ | HFA improves perplexity with zero extra computation, making it well worth large-scale validation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Diving into Self-Evolving Training for Multimodal Reasoning](diving_into_self-evolving_training_for_multimodal_reasoning.md)
- [\[ICML 2026\] Probing RLVR Training Instability through the Lens of Objective-Level Hacking](../../ICML2026/reinforcement_learning/probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)
- [\[ICML 2025\] Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning](sliding_puzzles_gym_a_scalable_benchmark_for_state_representation_in_visual_rein.md)
- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[ICML 2025\] Principal-Agent Bandit Games with Self-Interested and Exploratory Learning Agents](principal-agent_bandit_games_with_self-interested_and_exploratory_learning_agent.md)

</div>

<!-- RELATED:END -->
