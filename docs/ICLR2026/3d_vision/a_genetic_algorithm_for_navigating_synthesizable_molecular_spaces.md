---
title: >-
  [Paper Note] A Genetic Algorithm for Navigating Synthesizable Molecular Spaces
description: >-
  [ICLR 2026][3D Vision][genetic algorithm] This paper proposes SynGA, a genetic algorithm that operates directly on synthesis routes (synthesis trees)…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "genetic algorithm"
  - "synthesizable molecular design"
  - "synthesis routes"
  - "Bayesian optimization"
  - "building block filtering"
date: 2026-05-08
content_hash: 4be109bda86127c1
---

# A Genetic Algorithm for Navigating Synthesizable Molecular Spaces

**Conference**: ICLR 2026
**arXiv**: [2509.20719](https://arxiv.org/abs/2509.20719)  
**Code**: [https://github.com/alstonlo/synga](https://github.com/alstonlo/synga)  
**Area**: Molecular Design / Optimization
**Keywords**: genetic algorithm, synthesizable molecular design, synthesis routes, Bayesian optimization, building block filtering

## TL;DR
This paper proposes SynGA, a genetic algorithm that operates directly on synthesis routes (synthesis trees), constraining the search strictly within synthesizable molecular space via custom crossover and mutation operators. Combined with ML-guided building block filtering, SynGA achieves state-of-the-art performance on synthesizable analog search and property optimization.

## Background & Motivation

**Background**: Generative models for molecular design (VAE, RL, GFlowNet, LLM) have advanced rapidly, yet classical genetic algorithms remain highly competitive due to their simplicity, sample efficiency, and exploration capability.

**Limitations of Prior Work**: Most molecular generative models do not account for synthesizability, potentially proposing unstable or unsynthesizable designs—a critical barrier in practical applications. Post-hoc retrosynthesis models can partially mitigate this but incur substantial computational overhead (several minutes per evaluation). Existing synthesis-constrained GA methods require pre-training an ML projection model to map molecules back into synthesizable space, imposing upfront training costs and limiting performance to projection quality.

**Key Challenge**: Efficient search must be conducted over a vast discrete combinatorial molecular space while ensuring that every generated molecule possesses a feasible synthesis route.

**Goal**: Design a lightweight genetic algorithm that guarantees synthesizability by construction, serving as both a standalone baseline and a modular component.

**Key Insight**: Rather than generate-then-validate, the method defines genetic operators directly on synthesis trees, so the search space is confined to synthesizable molecules throughout.

**Core Idea**: By defining crossover/mutation operators on synthesis trees, the GA search is naturally restricted to synthesizable space; ML-guided building block filtering further accelerates convergence toward promising regions.

## Method

### Overall Architecture
The inputs are a building block set $\mathcal{B}$ (~200k purchasable molecules) and a reaction template set $\mathcal{R}$ (91 templates); the output is the optimal synthesizable molecule under a given objective along with its synthesis route. The core pipeline: GA iteratively evolves a population in the synthesis tree space $\mathcal{T}$, where each tree's leaf nodes are building blocks and internal nodes are reactions.

### Key Designs

1. **Genetic Operators on Synthesis Trees**:

    - **Function**: Define crossover and mutation operators that act directly on synthesis trees.
    - **Mechanism**: *Crossover*—enumerate subtrees from two parent trees, identify compatible subtree pairs connectable via a bimolecular reaction, and join them at a new root node using a randomly selected compatible reaction. *Mutation*—five operations: Grow (extend via a new reaction), Shrink (select a subtree), Rerun (resample products while preserving structure), Change Internal (replace a reaction), Change Leaf (replace a building block).
    - **Design Motivation**: Since every node in a synthesis tree satisfies the constraint that leaf nodes are building blocks and internal nodes are valid reactions, any offspring produced by genetic operators is automatically a valid synthesis route, requiring no post-hoc verification or projection.

2. **ML-Guided Building Block Filtering (Analog Search)**:

    - **Function**: Train a lightweight MLP classifier to predict which building blocks are likely relevant for synthesizing analogs of a query molecule.
    - **Mechanism**: Learn $\pi_\theta: \mathcal{M} \times \mathcal{B} \to (0,1)$ to filter the 200k building blocks down to a relevant subset. An $\varepsilon$-filtering strategy is applied: with probability $1-\varepsilon$, sample from the filtered set; with probability $\varepsilon$, sample from the full set.
    - **Design Motivation**: The 200k building block search space is immense; ML filtering effectively focuses the search on relevant regions without sacrificing completeness.

3. **Block-Additive Model + Bayesian Optimization (SynGBO)**:

    - **Function**: Use a Neural Additive Model (NAM) to score individual building blocks for filtering, embedded within a Bayesian optimization outer loop.
    - **Mechanism**: The NAM models $\rho_\theta(\mathcal{B}_M) = (\alpha + (1-\alpha)|\mathcal{B}_M|^{-1})\sum_{B \in \mathcal{B}_M} s_\theta(B)$, trained with a ranking loss rather than a regression loss. The outer loop uses a GP surrogate to guide SynGA toward maximizing the acquisition function.
    - **Design Motivation**: In property optimization, the target molecule is unknown, precluding classification-based filtering. However, the interpretability of NAMs allows direct scoring and ranking of building blocks for filtering purposes.

### Loss & Training
The analog search filter is trained with binary cross-entropy; the NAM for property optimization is trained with a pairwise ranking loss; the GP surrogate follows standard Gaussian process training. The GA employs an elitist selection strategy with a population size of 500 and an offspring size of 5.

## Key Experimental Results

### Main Results

**Synthesizable Analog Search (ChEMBL, 100 molecules):**

| Method | Morgan Sim↑ | Scaffold Sim↑ | Gobbi Sim↑ | Subset Sim↑ |
|--------|------------|--------------|-----------|-------------|
| SynGA (no filter) | 0.313 | 0.372 | 0.536 | 0.397 |
| SynGA + MLP | 0.380 | 0.452 | 0.607 | 0.465 |
| SynGA + MLP+Mine | **0.393** | **0.465** | **0.617** | **0.475** |
| SynNet | 0.325 | 0.383 | 0.549 | 0.427 |
| Pasithea | 0.278 | 0.310 | 0.491 | 0.361 |

**Property Optimization (PMO Benchmark, GuacaMol subset):**

| Method | Top-10 AUC↑ | Synthesizability |
|--------|-----------|-----------------|
| SynGBO | SOTA | Guaranteed |
| Graph GA | High, no synthesis guarantee | Not guaranteed |
| REINVENT | Relatively high | Not guaranteed |

### Ablation Study

| Configuration | Morgan Sim↑ | Notes |
|---------------|------------|-------|
| SynGA no filter | 0.313 | Baseline |
| + Sim heuristic filter | 0.336 | Simple heuristic yields modest gains |
| + MLP filter | 0.380 | ML filtering provides significant improvement |
| + MLP + Hard Negative Mining | 0.393 | Hard negative mining further boosts precision |

### Key Findings
- SynGA serves as a strong baseline without any ML components; adding building block filtering achieves SOTA.
- The Rerun mutation is a distinctive design—it fixes the synthesis tree structure while resampling products, efficiently exploring variants within the same scaffold.
- Filtering 200k building blocks can reduce the effective search space by orders of magnitude, which is critical for performance gains.
- SynGBO performs strongly in property optimization; the ranking loss for NAM training is better suited for filtering purposes than regression loss.

## Highlights & Insights
- **Synthesizability by Construction**: Rather than following a generate-then-validate paradigm, the search space itself is the synthesizable space—a fundamental methodological advantage. Every molecule generated during search comes with an associated synthesis route.
- **Modular Design Philosophy**: SynGA is a lightweight, ML-free core component that can be readily embedded into larger workflows (e.g., Bayesian optimization), exemplifying good engineering design: a simple core with optional enhancements.
- **Elegance of the Rerun Mutation**: By exploring chemical space while preserving the synthetic scaffold, this operator exploits the inherent ambiguity in synthesis routes—the same route can yield different molecules.

## Limitations & Future Work
- The method relies on a predefined reaction template library (91 templates), potentially missing synthesis routes beyond the template set.
- The building block set is fixed to the Enamine commercial catalog, limiting coverage of chemical space.
- The additive assumption of NAMs constrains modeling capacity for complex nonlinear properties.
- Synthesis routes are limited to at most 5 steps, which may preclude complex molecules requiring longer synthetic sequences.

## Related Work & Insights
- **vs. Graph GA**: Traditional molecular graph GAs offer strong search capability but provide no synthesizability guarantee; SynGA automatically ensures synthesizability by operating on synthesis trees, achieving comparable performance.
- **vs. SynNet/Pasithea**: These ML-based projection methods require training additional projection models; SynGA's core is ML-free, making it simpler and more reliable.
- **vs. RetroGNN (Gao et al., 2024)**: Also performs synthesis-aware GA but uses a trained projection network for constraints; SynGA searches directly within synthesizable space, avoiding projection errors.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Defining GA operators on synthesis trees is not entirely novel, but the implementation is refined; the ML/GA integration for building block filtering is a noteworthy contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers analog search, property optimization, and 2D/3D objectives—comprehensive coverage of molecular design tasks with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Method descriptions are precise, formal definitions are rigorous, and code is open-sourced.
- **Value**: ⭐⭐⭐⭐ Provides a practical synthesizable search tool for molecular design; the open-source code is directly usable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MARCO: Navigating the Unseen Space of Semantic Correspondence](../../CVPR2026/3d_vision/marco_semantic_correspondence.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[ICLR 2026\] Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](joint_shadow_generation_and_relighting_via_light-geometry_interaction_maps.md)
- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[ICLR 2026\] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering](radiogs_radiometric_gaussian_surfels.md)

</div>

<!-- RELATED:END -->
