---
title: >-
  [Paper Note] BeeRNA: Tertiary Structure-Based RNA Inverse Folding Using Artificial Bee Colony
description: >-
  [AAAI2026][Computational Biology][RNA inverse folding] This paper proposes BeeRNA, which applies the Artificial Bee Colony (ABC) optimization algorithm to the RNA tertiary structure inverse folding problem. Through a two…
tags:
  - "AAAI2026"
  - "Computational Biology"
  - "RNA inverse folding"
  - "Artificial Bee Colony"
  - "tertiary structure"
  - "bio-inspired optimization"
  - "RhoFold"
date: 2026-05-08
content_hash: ff2adc497d41aef5
---

# BeeRNA: Tertiary Structure-Based RNA Inverse Folding Using Artificial Bee Colony

**Conference**: AAAI2026
**arXiv**: [2511.21781](https://arxiv.org/abs/2511.21781)  
**Code**: To be released  
**Area**: Optimization
**Keywords**: RNA inverse folding, Artificial Bee Colony, tertiary structure, bio-inspired optimization, RhoFold

## TL;DR

This paper proposes BeeRNA, which applies the Artificial Bee Colony (ABC) optimization algorithm to the RNA tertiary structure inverse folding problem. Through a two-stage fitness evaluation combining base-pair distance pre-screening and RMSD scoring, BeeRNA outperforms deep learning methods gRNAde and RiboDiffusion on short-to-medium-length RNAs (<100 nt).

## Background & Motivation

- RNA inverse folding aims to design nucleotide sequences that fold into a specified target structure, with important applications in synthetic biology, aptamer therapeutics, riboswitches, and related domains.
- Most existing methods focus on **secondary structure** inverse folding (e.g., ViennaRNA, NUPACK); tertiary structure inverse folding remains an insufficiently addressed challenge in computational biology.
- Deep learning approaches (gRNAde, RiboDiffusion, RISoTTo) offer fast inference but rely on large-scale training data and perform poorly on **short RNAs (<50 nt)**—precisely the length range occupied by functional RNAs such as miRNAs, aptamers, and ribozymes.
- The ABC algorithm has demonstrated the ability to traverse complex energy landscapes in protein inverse folding, yet has not previously been applied to RNA tertiary structure inverse folding.

## Core Problem

Given a target RNA tertiary structure $T_{\text{3D}}$ (3D atomic coordinates from a PDB file), find the nucleotide sequence $S^* = \arg\min_S \text{RMSD}(F(S), T_{\text{3D}})$, where $F(S)$ denotes the structure predicted by RhoFold for sequence $S$. Additional constraints include thermodynamic stability (minimum free energy) and GC content between 40% and 60%.

## Method

### Overall Architecture

BeeRNA integrates ABC swarm optimization with RhoFold structure prediction, employing a two-stage fitness evaluation strategy:

1. **Stage 1 (Rapid Screening)**: ViennaRNA is used to compute the secondary structure of each candidate sequence, which is then compared against the target secondary structure via base-pair distance (BPD). Sequences with BPD > 0 are immediately marked as infeasible (fitness = ∞).
2. **Stage 2 (Precise Evaluation)**: Only sequences with BPD = 0 invoke RhoFold for tertiary structure prediction; the RMSD against the target is computed as the fitness value.

### Three Phases of the ABC Algorithm

**Initialization**: A population of 40 RNA sequences (N = 40) is generated. Base-pair constraints are extracted from the target secondary structure; paired positions are assigned complementary bases (G-C or C-G), while unpaired positions are randomly assigned nucleotides, with GC content maintained between 40% and 60%.

**Employed Bee Phase**: Each sequence generates a neighborhood solution via an adaptive mutation rate defined as:

$$\text{mutation\_rate} = \max\left(0.1,\ 0.095 \cdot e^{-\frac{\text{best\_RMSD}}{5n}}\right)$$

Mutation operations include random nucleotide substitution, swap mutation within adjacent positions (within 3 positions, 20% probability), and $\{A,U\}/\{G,C\}$ exchange when GC content exceeds 50%. If the neighborhood solution achieves a lower RMSD, it replaces the original sequence; otherwise, the trial counter is incremented.

**Onlooker Bee Phase**: Sequences are selected probabilistically according to softmax selection probabilities $p_i = e^{-r_i/\tau} / \sum_j e^{-r_j/\tau}$ for further exploration, where the temperature parameter $\tau = 5.0 \cdot (1 + t/T)$ increases with iteration count, promoting exploration early and exploitation later.

**Scout Bee Phase**: Sequences that fail to improve over 5 consecutive iterations are randomly reinitialized to prevent entrapment in local optima.

### Evaluation Metrics

- **RMSD**: The primary metric, computed after optimal superimposition via US-align over backbone phosphorus atoms (P), sugar carbon atoms (C4'), and base nitrogen atoms (N1/N9).
- **GDT-TS**: An auxiliary metric measuring the proportion of residues within distance thresholds of 1/2/4/8 Å.

## Key Experimental Results

### RNASolo Dataset (Short RNA, 3–30 nt)

| Metric | BeeRNA | gRNAde |
|--------|--------|--------|
| RMSD (Å) | **2.50** | 9.33 |
| GDT-TS (%) | **26.91** | 18.97 |

### RFAM Dataset (25–200 nt ncRNA)

| Metric | BeeRNA | gRNAde |
|--------|--------|--------|
| RMSD (Å) | **14.98** | 16.24 |
| GDT-TS (%) | **11.56** | 9.77 |

### 14 Benchmark RNA Structures

| Metric | BeeRNA | gRNAde | RiboDiffusion |
|--------|--------|--------|---------------|
| RMSD (Å) | 12.02 | 14.63 | **10.31** |
| GDT-TS (%) | 15.92 | 10.16 | **22.69** |

- BeeRNA achieves the lowest RMSD on **10 out of 14** structures, with particular advantages on short RNAs (e.g., 1F27: 2.21 Å vs. gRNAde's 14.94 Å).
- RiboDiffusion performs better on long RNAs (>100 nt), though its training data may overlap with the test set.
- Runtime efficiency: approximately 3 minutes for <50 nt and 7–10 minutes for 50–100 nt (64-core CPU).

### Key Findings

The paper presents a compelling case study: for RNA 2OUE (61 nt), a single-nucleotide mutation (sequence recovery rate 98.4%) causes RMSD to spike to 19.34 Å, demonstrating that **high sequence similarity does not guarantee structural correctness** and underscoring the superiority of structure-based evaluation over sequence recovery rate.

## Highlights & Insights

- **Training-free**: Requires no large-scale dataset pretraining; plug-and-play, equally applicable to novel RNA families.
- **Elegant two-stage screening design**: The lightweight BPD filter eliminates large numbers of infeasible sequences before invoking the expensive RhoFold, substantially reducing computational cost.
- **Adaptive mutation mechanism**: An adaptive mutation rate incorporating simulated annealing principles balances exploration and exploitation.
- **Structure-guided evaluation**: The paper makes a compelling case for using RMSD/GDT-TS rather than sequence recovery rate as the evaluation criterion.
- **Integration of biological constraints**: GC content, Watson-Crick base pairing, and other constraints are directly embedded into the optimization pipeline.

## Limitations & Future Work

- **Poor scalability to long RNAs**: For sequences >100 nt, the search space grows exponentially and RMSD increases substantially (e.g., 2R8S at 159 nt reaches 26 Å).
- **Dependence on RhoFold accuracy**: Prediction errors inherent to RhoFold propagate into BeeRNA's optimization results.
- **Strict BPD = 0 prerequisite**: When the target structure contains wobble pairs or non-canonical base pairs, BPD cannot be reduced to zero, necessitating a fixed 20 Å penalty that impairs convergence.
- **Sequential CPU inference**: Each iteration requires multiple RhoFold calls; GPU acceleration could substantially improve speed.
- **Fixed population and iteration parameters**: The 40×40 configuration may not be sufficiently flexible for RNAs of varying lengths.

## Related Work & Insights

| Method | Type | Training Required | Short RNA | Long RNA | Structure Evaluation |
|--------|------|-------------------|-----------|----------|----------------------|
| ViennaRNA | Deterministic | None | Secondary structure only | Secondary structure only | None |
| gRNAde | Deep learning GNN | Large-scale pretraining | Weak | Moderate | RMSD |
| RiboDiffusion | Diffusion model | Large-scale pretraining | Weak (<50 nt difficult) | Strong | RMSD |
| RISoTTo | Geometric Transformer | Large-scale pretraining | Not extensively tested | Strong | Sequence recovery |
| **BeeRNA** | Bio-inspired metaheuristic | **None** | **Strong** | Weak | RMSD + GDT-TS |

BeeRNA fills the gap of "training-free tertiary structure inverse folding," complementing deep learning methods: BeeRNA for short RNAs, gRNAde/RiboDiffusion for long RNAs.

The successful transfer of the ABC algorithm from protein inverse folding to RNA inverse folding suggests that other bio-inspired algorithms (ant colony optimization, particle swarm optimization) may also be worth exploring. The two-stage screening paradigm (cheap pre-filter + expensive precise evaluation) represents a general search acceleration strategy transferable to other structural design problems. Future work could explore using BeeRNA as a post-processing optimizer for deep learning methods, or initializing the BeeRNA population with deep learning predictions. As more accurate RNA structure prediction tools such as AlphaFold3 become available, BeeRNA can seamlessly substitute RhoFold for improved results.

## Rating

- Novelty: ⭐⭐⭐ (First application of ABC to RNA tertiary structure inverse folding, though the methodology itself is relatively conventional)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets and multi-metric comparisons, but ablation studies are absent)
- Writing Quality: ⭐⭐⭐⭐ (Clear and complete, with well-argued motivation)
- Value: ⭐⭐⭐ (Practically valuable for short RNA design, but limitations for long RNAs are pronounced)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](../../ICML2026/computational_biology/protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[ICLR 2026\] Protein Structure Tokenization via Geometric Byte Pair Encoding](../../ICLR2026/computational_biology/protein_structure_tokenization_via_geometric_byte_pair_encoding.md)
- [\[ICLR 2026\] AntigenLM: Structure-Aware DNA Language Modeling for Influenza](../../ICLR2026/computational_biology/antigenlm_structure-aware_dna_language_modeling_for_influenza.md)
- [\[AAAI 2026\] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening](s2drug_bridging_protein_sequence_and_3d_structure_in_contrastive_representation_.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](../../ICML2026/computational_biology/evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)

</div>

<!-- RELATED:END -->
