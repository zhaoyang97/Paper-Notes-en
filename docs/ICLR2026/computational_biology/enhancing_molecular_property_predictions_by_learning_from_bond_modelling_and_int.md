---
title: >-
  [Paper Note] Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions
description: >-
  [ICLR 2026][Computational Biology][molecular representation] DeMol is a dual-graph enhanced multi-scale interaction framework that introduces parallel atom-centric and bond-centric channels along with Double-Helix Blocks…
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "molecular representation"
  - "dual-graph"
  - "bond modeling"
  - "GNN"
  - "property prediction"
date: 2026-05-08
content_hash: 19243b4495a9e022
---

# Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions

**Conference**: ICLR 2026
**arXiv**: [2603.00568](https://arxiv.org/abs/2603.00568)  
**Code**: Available  
**Area**: Self-Supervised Learning / Molecular Representation Learning / Graph Neural Networks
**Keywords**: molecular representation, dual-graph, bond modeling, GNN, property prediction

## TL;DR
DeMol is a dual-graph enhanced multi-scale interaction framework that introduces parallel atom-centric and bond-centric channels along with Double-Helix Blocks to explicitly model atom–atom, atom–bond, and bond–bond interactions, achieving state-of-the-art performance on PCQM4Mv2, OC20, QM9, and related benchmarks.

## Background & Motivation

**Background**: Mainstream molecular representation learning methods are based on GNNs, modeling molecules as graphs (atoms as nodes, bonds as edges). Recent approaches further leverage 3D geometric information (distances, angles) to enhance predictions.

**Limitations of Prior Work**: Existing methods are "atom-centric," treating chemical bonds merely as pairwise interactions between atoms. However, bonds themselves carry rich information (bond order, bond length, hybridization state), and non-additive interactions exist between bonds (e.g., delocalized π-electron systems in benzene rings, or the cisplatin/transplatin configurational differences that directly determine pharmacological activity).

**Key Challenge**: Single-graph models cannot simultaneously encode atomic topological relationships and bond geometric relationships (dihedral angles, bond angles), limiting prediction accuracy.

**Goal**: Explicitly model chemical bond information and inter-bond interactions by constructing a dual-channel atom–bond fusion framework.

**Key Insight**: An information-theoretic analysis demonstrates that the bond-centric graph (line graph) contains additional structural information not present in the original graph (Proposition 1), and dual-graph representations strictly preserve more mutual information (Proposition 2).

**Core Idea**: Encode molecules in parallel using a dual-graph (atom graph + bond graph), fuse information across both channels at multiple scales via Double-Helix Blocks, and enforce geometric consistency through covalent radius regularization.

## Method

### Overall Architecture
The input molecule is represented as two graphs: an atom-centric graph $\mathcal{G}$ (atoms as nodes, bonds as edges) and a bond-centric graph $\mathcal{L}(\mathcal{G})$ (bonds as nodes, with edges connecting bonds that share an atom). Each graph is processed through its own Transformer encoder, with cross-channel information exchange performed by Double-Helix Blocks at intermediate layers. The two representations are ultimately fused to predict molecular properties.

### Key Designs

1. **Atom-Centric Channel**:

    - Function: Encodes atom-level features and inter-atomic relationships.
    - Mechanism: Uses structural encodings (Gaussian basis kernels over 3D distances + 2D shortest-path distances) as attention biases; updates atom embeddings via Transformer self-attention.
    - Design Motivation: Captures both spatial and topological information while remaining compatible with existing methods.

2. **Bond-Centric Channel**:

    - Function: Explicitly encodes bond-level features and inter-bond relationships.
    - Mechanism: Bonds become nodes, connected when they share an atom. Innovatively introduces **torsion encoding** $\Phi_b^{tors}$, encoding bond angles $\theta_{ijk}$ and dihedral angles $\varphi_{ijkl}$ as attention biases via Gaussian basis kernels.
    - Design Motivation: The bond-centric graph is the natural domain for representing geometric relationships (Proposition 3).

3. **Double-Helix Blocks**:

    - Function: Bidirectionally exchanges information between atom and bond channels.
    - Mechanism: Employs bidirectional cross-attention modules in which atom embeddings query bond embeddings and vice versa, performing multi-scale fusion.
    - Design Motivation: Proposition 2 shows that predictive capacity arises from effective fusion of the two representations.

4. **Bond Prediction Regularization**:

    - Function: Enforces geometric consistency via covalent radius constraints.
    - Mechanism: Predicts whether atom pairs form bonds, ensuring that learned structural representations are consistent with chemically valid structures.

### Loss & Training
Task-specific primary loss (e.g., MAE regression loss) + covalent-radius bond prediction regularization term + structure-aware masking for improved efficiency.

## Key Experimental Results

### Main Results: PCQM4Mv2

| Model | Parameters | MAE↓ |
|-------|-----------|------|
| GPS++ | 44.3M | 0.0778 |
| Transformer-M | 69M | 0.0772 |
| Unimol+ | 77M | 0.0693 |
| TGT-At | 203M | 0.0671 |
| **DeMol** | **186M** | **0.0603** |

### OC20 IS2RE Validation Set

| Model | Avg. Energy MAE (eV)↓ | Avg. EwT (%)↑ |
|-------|-----------------------|---------------|
| Unimol+ | 0.4088 | 8.61 |
| TGT-At | 0.4030 | 8.82 |
| **DeMol** | **0.3879** | **9.23** |

### Ablation Study

| Configuration | Description |
|--------------|-------------|
| Atom channel only | Performance drops; bond information absent |
| Without Double-Helix | Performance drops; no inter-channel fusion |
| Without torsion encoding | Performance drops; bond angles/dihedral angles absent |
| Full DeMol | 0.0603; best performance with complete model |

### Key Findings
- Outperforms the previous SOTA (TGT-At) by 10.1% using a single model rather than an ensemble.
- Demonstrates stable performance in out-of-distribution (OOD) settings on OC20 IS2RE, indicating strong generalization.
- Torsion encoding in the bond-centric channel contributes substantially to overall performance.

## Highlights & Insights
- The information-theoretic analysis provides rigorous theoretical grounding for the dual-graph design, with four Propositions arguing from complementary perspectives.
- The cisplatin/transplatin example is highly intuitive—identical atomic composition, different bond configurations, and vastly different pharmacological effects.
- Double-Helix Blocks are transferable to other multi-view fusion scenarios.

## Limitations & Future Work
- The model has a large parameter count (186M), resulting in high training cost.
- Validation is primarily conducted on small molecules and materials; applicability to macromolecules remains to be explored.
- The information-theoretic analysis establishes necessary conditions but does not guarantee that the network fully exploits the available information.

## Related Work & Insights
- **vs. Transformer-M**: Integrates 3D distance encoding only on the atom graph; DeMol additionally introduces a bond-centric channel.
- **vs. ALIGNN**: Uses a line graph for message passing but lacks atom–bond cross-attention.
- **vs. GemNet**: Encodes dihedral angles but still operates in the atom space.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual-graph architecture, information-theoretic motivation, and Double-Helix fusion is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four mainstream benchmarks with comprehensive SOTA results.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical analysis supported by intuitive examples.
- Value: ⭐⭐⭐⭐ Significant contribution to molecular representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Molecular Chirality via Chiral Determinant Kernels](learning_molecular_chirality_via_chiral_determinant_kernels.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/computational_biology/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[ICLR 2026\] A Genetic Algorithm for Navigating Synthesizable Molecular Spaces](a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)
- [\[ICML 2026\] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](../../ICML2026/computational_biology/learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](../../NeurIPS2025/computational_biology/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)

</div>

<!-- RELATED:END -->
