---
title: >-
  [Paper Note] S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening
description: >-
  [AAAI 2026][Medical Imaging][Virtual screening] This paper proposes S2Drug, a two-stage contrastive learning framework. Stage 1 performs large-scale protein sequence–ligand contrastive pre-training on ChemBL with a bilateral data sampling strategy to reduce noise and redundancy. Stage 2 fine-tunes on PDBBind by fusing sequence and 3D structural information via a residue-level gating module and incorporating a binding site prediction auxiliary task. S2Drug substantially outperforms existing methods on the DUD-E and LIT-PCBA virtual screening benchmarks.
tags:
  - AAAI 2026
  - Medical Imaging
  - Virtual screening
  - protein–ligand interaction
  - contrastive learning
  - protein sequence
  - 3D structure
  - binding site prediction
  - drug discovery
date: 2026-05-08
content_hash: cd36305f4008ea89
---

# S2Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening

**Conference**: AAAI 2026
**arXiv**: [2511.07006](https://arxiv.org/abs/2511.07006)
**Code**: Available (provided in appendix)
**Area**: Medical Imaging / Drug Discovery / Virtual Screening
**Keywords**: Virtual screening, protein–ligand interaction, contrastive learning, protein sequence, 3D structure, binding site prediction, drug discovery

## TL;DR
This paper proposes S2Drug, a two-stage contrastive learning framework. Stage 1 performs large-scale protein sequence–ligand contrastive pre-training on ChemBL with a bilateral data sampling strategy to reduce noise and redundancy. Stage 2 fine-tunes on PDBBind by fusing sequence and 3D structural information via a residue-level gating module and incorporating a binding site prediction auxiliary task. S2Drug substantially outperforms existing methods on the DUD-E and LIT-PCBA virtual screening benchmarks.

## Background & Motivation

**Background**: Virtual screening (VS) is a core step in drug discovery, aiming to identify small molecules that bind to a target protein pocket from large compound libraries. Existing methods fall into two categories: molecular docking (e.g., AutoDock Vina — accurate but slow) and deep learning approaches (e.g., DrugCLIP/DrugHash — using contrastive learning to align protein–ligand representations).

**A Critical Blind Spot — Protein Sequence is Neglected**:
- Nearly all mainstream methods rely solely on 3D structural information.
- Single-conformation, atom-level structural models are sensitive to input perturbations and struggle with pocket conformational flexibility.
- Obtaining protein 3D structures (X-ray, Cryo-EM) is costly and time-consuming, limiting the scale of training data.
- By contrast, protein sequence data is widely available, and "sequence determines structure, structure determines function" is a foundational principle of protein biology.

**Challenges of Using Sequences Directly**:
- Large-scale protein–ligand datasets (e.g., ChemBL, 745K entries) suffer from severe **redundancy and noise**.
- Protein side: homologous redundancy, repeated functional isoforms.
- Ligand side: variability in affinity measurements, non-specific frequent hitter compounds.
- Using sequences alone while discarding structural context loses critical spatial interaction information.

**Key Insight**: A two-stage learning paradigm — sequence pre-training addresses data scale and generalization, while structure-fusion fine-tuning addresses spatial precision.

## Method

### Overall Architecture: Two-Stage Contrastive Learning

**Stage 1: Sequence Pre-training** (ChemBL, large-scale) → **Stage 2: Sequence–Structure Fusion Fine-tuning** (PDBBind, small-scale, high-quality)

### Stage 1: Sequence Model Pre-training

#### Bilateral Data Sampling Strategy

A high-quality subset is curated from ChemBL's 745K entries:

**Protein-side Redundancy Reduction**:
1. Homology-aware downweighting: MMseqs2 clusters sequences at 40% identity threshold; sampling probability for proteins in large clusters is reduced: $\Pr(P_n) = |C_m^{hom}|^{-\alpha}$, $\alpha=0.5$
2. Functional deduplication: Based on UniProt/GO annotations, only the representative protein with the highest ligand diversity is retained per functional group.

**Ligand-side Noise Mitigation**:
1. Affinity variance filtering: Protein–ligand pairs with standard deviation $\sigma_n > 1.0$ across experimental conditions are removed.
2. Frequent hitter removal: Ligands binding to more than 20 proteins are treated as non-specific compounds and removed; PAINS reactive substructures are also filtered.

#### Representation Learning

- **Sequence encoder**: ESM2-650M; inputs amino acid sequences and produces protein embeddings via mean pooling.
- **Ligand structure encoder**: Uni-Mol; inputs 3D conformations (atom coordinates + types) and produces ligand embeddings via mean pooling.
- Two MLP heads project representations into a shared space; trained with symmetric InfoNCE contrastive loss.

### Stage 2: Sequence–Structure Fusion Fine-tuning

#### Residue-Level Gated Fusion Module

For each pocket residue $r_i$, representations $x_{n,i}^s$ and $x_{n,i}^g$ are obtained from the sequence encoder and structure encoder, respectively, and fused via adaptive gating:

$$\beta_{n,i} = \sigma(W_\beta^\top [W_s x_{n,i}^s; W_g x_{n,i}^g] + b_\beta)$$
$$x_{n,i}^f = \beta_{n,i} \cdot W_s x_{n,i}^s + (1-\beta_{n,i}) \cdot W_g x_{n,i}^g$$

The gating weights $\beta$ are learned, enabling the model to dynamically select the more informative modality for each residue. The fused representations are passed through two Transformer layers followed by mean pooling to yield the final pocket representation.

#### Binding Site Prediction Auxiliary Task

**Core Idea**: The binding pocket consists of residues that are scattered along the primary sequence but spatially clustered to form a binding cavity in 3D space. Predicting binding sites encourages the model to understand protein 3D folding, particularly in the pocket region.

- $K$ ligand probes are sampled and their attention-based correlation with each residue is computed.
- Only sequence representations $x_{n,i}^s$ are used (to avoid information leakage); trained with BCE loss.

#### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fc}} + \lambda \cdot \mathcal{L}_{\text{bsp}}$$

where $\mathcal{L}_{\text{fc}}$ is the contrastive loss on fused representations and $\mathcal{L}_{\text{bsp}}$ is the binding site prediction loss.

## Key Experimental Results

### Main Results: Virtual Screening Performance

**DUD-E dataset** (large-scale, with decoys):

| Method | AUROC | BEDROC | EF 0.5% | EF 1% | EF 5% |
|--------|-------|--------|---------|-------|-------|
| Glide-SP | 76.70 | 40.70 | 19.39 | 16.18 | 7.23 |
| DrugCLIP | 79.45 | 47.82 | 37.86 | 30.76 | 10.10 |
| DrugHash | 83.73 | 57.16 | 43.03 | 37.18 | 12.07 |
| **S2Drug** | **92.46** | **79.25** | **58.37** | **43.06** | **18.82** |

S2Drug surpasses DrugHash by 8.73 AUROC points and DrugCLIP by 13.01 points.

**LIT-PCBA dataset** (more realistic screening scenario):

| Method | AUROC | BEDROC | EF 0.5% | EF 1% |
|--------|-------|--------|---------|-------|
| DrugCLIP | 56.36 | 6.78 | 7.77 | 5.66 |
| DrugHash | 54.58 | 7.14 | 9.65 | 6.14 |
| **S2Drug** | **58.23** | **8.69** | **11.44** | **7.38** |

### Homology Exclusion Experiments (Generalization Evaluation)

Training–test overlap is removed at varying sequence identity thresholds (90%/60%/30%/HMM):
- S2Drug substantially outperforms DrugCLIP at all thresholds.
- Even at 90% and 60% thresholds, S2Drug exceeds DrugHash/DrugCLIP under the no-exclusion setting.
- This demonstrates that bilateral data sampling effectively reduces overfitting and dependence on homologous patterns.

### Ablation Study

| Variant | DUD-E AUROC | LIT-PCBA AUROC |
|---------|-------------|----------------|
| − BDS (w/o bilateral data sampling) | 88.73 | 56.12 |
| − SSF (w/o sequence–structure fusion) | 87.92 | 55.03 |
| − BSP (w/o binding site prediction) | 89.58 | 56.47 |
| **S2Drug** | **92.46** | **58.23** |

- Removing sequence–structure fusion (SSF) has the largest impact: AUROC drops by 4.54/3.20.
- Bilateral data sampling (BDS) also contributes substantially: 3.73/2.11.
- The binding site prediction auxiliary task (BSP) provides approximately 2.88/1.76 improvement.

### Binding Site Prediction

S2Drug also demonstrates competitive binding site prediction performance on three benchmarks — HOLO4K, COACH420, and ASD — validating the effectiveness of the auxiliary task.

## Highlights & Insights

1. **Operationalizing the "sequence → structure → function" principle**: The first systematic integration of protein sequences into contrastive representation learning for virtual screening.
2. **Elegant bilateral data sampling strategy**: Homology downweighting + functional deduplication + affinity variance filtering + frequent hitter removal collectively address ChemBL data quality issues from four angles.
3. **Residue-level gated fusion**: Allows each residue to dynamically weight sequence vs. structural information, offering greater flexibility than simple concatenation or scalar weighting.
4. **Biologically motivated auxiliary task design**: Binding site residues are spatially clustered but linearly dispersed; predicting them guides the model toward understanding 3D protein folding.
5. **Substantial lead under strict homology exclusion** confirms that the model has learned underlying protein–ligand interaction principles rather than memorizing homologous patterns.

## Limitations & Future Work

1. The two-stage training pipeline is relatively complex; Stage 1 pre-training on ChemBL requires 8×A6000 GPUs.
2. The Stage 2 fine-tuning dataset PDBBind is limited in scale (~19K entries), potentially constraining the learning capacity of the fusion module.
3. The binding site prediction auxiliary task requires ground-truth binding site annotations, making it inapplicable to unannotated proteins.
4. The absolute AUROC on LIT-PCBA remains low (58.23), indicating that realistic screening scenarios remain highly challenging.
5. Computational cost is not discussed (inference latency of ESM2-650M + Uni-Mol for large-scale screening).

## Related Work & Insights

- **Virtual screening**: Molecular docking (Glide-SP/Vina), regression (DeepDTA/Planet), classification (OnionNet-2), retrieval (DrugCLIP/DrugHash).
- **Protein representation learning**: ESM2 (sequence), UniMol (structure); recent trends toward sequence–structure co-modeling (SaProt, ESMFold).
- **Contrastive learning in drug discovery**: DrugCLIP pioneered the retrieval paradigm; DrugHash introduced hashing for acceleration.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — The combination of two-stage learning, bilateral data sampling, and residue-level gated fusion is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — DUD-E + LIT-PCBA + homology exclusion + binding site prediction; comprehensive and rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear; mathematical notation is well-defined.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to drug discovery; code is open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [\[NeurIPS 2025\] AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation](../../NeurIPS2025/medical_imaging/aanet_virtual_screening_under_structural_uncertainty_via_alignment_and_aggregati.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[ICLR 2026\] Protein Structure Tokenization via Geometric Byte Pair Encoding](../../ICLR2026/medical_imaging/protein_structure_tokenization_via_geometric_byte_pair_encoding.md)
- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)

<!-- RELATED:END -->
