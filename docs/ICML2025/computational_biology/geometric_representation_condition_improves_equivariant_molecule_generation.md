---
title: >-
  [Paper Note] Geometric Representation Condition Improves Equivariant Molecule Generation
description: >-
  [ICML 2025 (Spotlight)][Computational Biology][Molecule Generation] GeoRCG proposes a two-stage molecule generation framework: first generating a low-dimensional geometric representation (informative representation), and then generating the complete molecule conditioned on this representation. This achieves an average improvement of 50% on conditional molecule generation tasks, while reducing the number of diffusion steps from 1000 to 100.
tags:
  - "ICML 2025 (Spotlight)"
  - "Computational Biology"
  - "Molecule Generation"
  - "Equivariant Diffusion"
  - "Geometric Representation Condition"
  - "Two-Stage Generation"
  - "Conditional Molecule Generation"
date: 2026-05-08
content_hash: 69ee117a68e13f2a
---

# Geometric Representation Condition Improves Equivariant Molecule Generation

**Conference**: ICML 2025 (Spotlight)  
**arXiv**: [2410.03655](https://arxiv.org/abs/2410.03655)  
**Code**: [https://github.com/GraphPKU/GeoRCG](https://github.com/GraphPKU/GeoRCG)  
**Area**: Computational Biology  
**Keywords**: Molecule Generation, Equivariant Diffusion, Geometric Representation Condition, Two-Stage Generation, Conditional Molecule Generation

## TL;DR
GeoRCG proposes a two-stage molecule generation framework: first generating a low-dimensional geometric representation (informative representation), and then generating the complete molecule conditioned on this representation. This achieves an average improvement of 50% on conditional molecule generation tasks, while reducing the number of diffusion steps from 1000 to 100.

## Background & Motivation

**Background**: 3D molecule generation is a critical step in drug design. Equivariant diffusion models (such as EDM, GeoLDM) have become mainstream methods in this field, capable of generating SE(3)-equivariant 3D molecular structures.

**Limitations of Prior Work**: Existing models perform reasonably well in unconditional generation but show significant limitations in conditional generation (generating molecules with specific properties). This is because conditional signals (such as the HOMO-LUMO gap) struggle to effectively guide the generation process of high-dimensional 3D structures.

**Key Challenge**: Directly conditioning the 3D molecular diffusion process on scalar properties yields weak guidance, as the conditional signal is too simple (a single numerical value vs. the 3D coordinates of dozens to hundreds of atoms).

**Goal**: To effectively inject conditional information into equivariant molecule generation, specifically achieving substantial performance enhancements in conditional molecule generation tasks.

**Key Insight**: Introduce an information-rich intermediate "geometric representation" as a bridge—first generate the representation, and then generate the molecule conditioned on this representation.

**Core Idea**: Use semantic-rich geometric representations, which are easier to generate, to provide goal-oriented guidance for the challenging task of 3D molecule generation.

## Method

### Overall Architecture

- **First Stage**: Generate the geometric representation $\mathbf{r}$ in a low-dimensional space (which encodes the key structural and property information of the molecule).
- **Second Stage**: Conditioned on $\mathbf{r}$, generate the complete 3D molecule $(\mathbf{x}, \mathbf{h})$ using an equivariant diffusion model.
- Both stages can be trained and inferred independently.

### Key Designs

1. **Geometric Representation Design**:

    - Extract the geometric representation of the molecule from a pre-trained equivariant GNN.
    - The representation encodes atom types, spatial arrangement, local geometric features, etc.
    - It has a lower dimension than raw 3D coordinates but carries critical semantic information.
    - **Design Motivation**: Conditional signals need to be both informative and easy to generate—signals that are too simple (scalar properties) fail to guide effectively, while those that are too complex (complete molecules) defeat the purpose of simplification.

2. **Two-Stage Decoupled Generation**:

    - The first stage independently trains a diffusion model (which can be conditional or unconditional) to generate $\mathbf{r}$.
    - The second stage trains a conditional diffusion: $p_\theta(\mathbf{x}, \mathbf{h} | \mathbf{r})$.
    - Theoretical guarantee: The joint distribution of the two stages can approximate the true molecular distribution.
    - **Design Motivation**: Decompose a complex generation problem into two simpler sub-problems.

3. **Equivariant Guarantee**:

    - The geometric representation $\mathbf{r}$ itself satisfies SE(3)-equivariance.
    - The conditional diffusion process maintains equivariance.
    - The combined overall generation process remains equivariant.
    - **Design Motivation**: Physical molecules must respect rotational and translational symmetries.

### Loss & Training

- Representation extraction: Latent layer outputs of a pre-trained equivariant GNN.
- First stage: Standard diffusion loss on $\mathbf{r}$.
- Second stage: Conditional equivariant diffusion loss $\|\epsilon - \epsilon_\theta(\mathbf{x}_t, \mathbf{h}_t, t, \mathbf{r})\|^2$.
- Base generator options include EDM or SemlaFlow.

## Key Experimental Results

### Main Results

| Dataset | Task | GeoRCG | Baseline Methods | Gain |
|--------|------|--------|---------|------|
| QM9 | Unconditional Generation | Significantly Improved | EDM/GeoLDM | Obvious quality improvement |
| GEOM-DRUG | Unconditional Generation | Significantly Improved | EDM | Better |
| QM9 | Conditional Generation (α) | SOTA | EDM/cond | **~50% average improvement** |
| QM9 | Conditional Generation (gap) | SOTA | EDM/cond | **~50% average improvement** |
| QM9 | Conditional Generation (μ) | SOTA | EDM/cond | **~50% average improvement** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| No geometric representation condition | Baseline | Standard single-stage generation |
| Random noise as condition | Close to baseline | Demonstrates the necessity of informative representations |
| Scalar property condition | Slight improvement | Information is too sparse |
| Geometric representation condition (1000 steps) | Optimal | Full configuration |
| Geometric representation condition (**100 steps**) | **Close to 1000 steps** | 10x step reduction |

### Key Findings

- **Huge improvement in conditional generation**: An average performance improvement of 50%, accomplished through a better conditioning strategy.
- **Drastic step reduction**: Guided by the geometric representation, 100 steps can achieve quality close to that of 1000 steps.
- **General framework**: Both EDM and SemlaFlow base generators benefit from GeoRCG.
- The semantic richness of the representation is key—random representations are ineffective.

## Highlights & Insights

1. **Theoretical Backing**: Proves that the two-stage decoupled generation can approximate the true distribution.
2. **Simple yet Effective**: The conceptual framework is simple, yet it yields remarkable performance (50% improvement).
3. **Dual Benefits**: Simultaneously improves generation quality and reduces sampling steps.
4. **Spotlight Paper**: Accepted as an ICML 2025 Spotlight, indicating strong community recognition.

## Limitations & Future Work

1. Requires a pre-trained GNN to extract representations, introducing an extra dependency.
2. Representation design may require domain knowledge to select the optimal GNN and layer.
3. The overall inference time of both stages needs evaluation (although the second stage reduces steps).
4. Scalability to larger molecules (such as proteins) remains unverified.

## Related Work & Insights

- EDM (Equivariant Diffusion Model) and GeoLDM are the primary baselines.
- The idea of classifier-guided diffusion is generalized here to "representation-guided diffusion".
- Insight: Intermediate representation conditioning could be a general strategy to enhance all conditional diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of two-stage representation conditioning is clean and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ QM9 + GEOM-DRUG, multiple property conditions, step ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear, with a good balance of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ 50% improvement + Spotlight = High-impact work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/computational_biology/unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[ICLR 2026\] GeomMotif: A Benchmark for Arbitrary Geometric Preservation in Protein Generation](../../ICLR2026/computational_biology/geommotif_a_benchmark_for_arbitrary_geometric_preservation_in_protein_generation.md)

</div>

<!-- RELATED:END -->
