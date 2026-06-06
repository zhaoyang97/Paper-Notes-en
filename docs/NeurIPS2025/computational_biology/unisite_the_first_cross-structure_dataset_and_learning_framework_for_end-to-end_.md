---
title: >-
  [Paper Note] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection
description: >-
  [NeurIPS 2025][Computational Biology][protein binding site detection] This work introduces UniSite-DS, the first UniProt (unique protein)-centric ligand binding site dataset, and UniSite…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "protein binding site detection"
  - "end-to-end set prediction"
  - "UniProt-centric dataset"
  - "Hungarian matching"
  - "IoU evaluation metric"
date: 2026-05-08
content_hash: d501078bbf94ab08
---

# UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection

**Conference**: NeurIPS 2025
**arXiv**: [2506.03237](https://arxiv.org/abs/2506.03237)  
**Code**: [GitHub](https://github.com/quanlin-wu/unisite)  
**Area**: Medical Imaging / Structural Biology
**Keywords**: protein binding site detection, end-to-end set prediction, UniProt-centric dataset, Hungarian matching, IoU evaluation metric

## TL;DR

This work introduces UniSite-DS, the first UniProt (unique protein)-centric ligand binding site dataset, and UniSite, the first end-to-end binding site detection framework. UniSite directly predicts multiple potentially overlapping binding sites via set prediction loss and bijective matching, and further proposes IoU-based AP as a more accurate evaluation metric.

## Background & Motivation

Protein–ligand binding site detection is a fundamental step in structure-based drug design. However, existing methods and datasets face three critical issues:

**PDB-centric statistical bias**: Existing datasets are organized around individual protein–ligand complex structures, ignoring the fact that the same protein may exhibit multiple distinct binding sites across different complexes. For example, a single UniProt ID (Q8WS26) has only one binding site recorded in PDBbind2020, whereas UniSite-DS integrates 17 unique binding sites from 13 representative PDB entries. This data construction approach introduces severe statistical bias.

**Fragmented workflow**: Most methods follow a pipeline of semantic segmentation to produce binary masks, followed by clustering into discrete binding sites. This fragmented pipeline is highly dependent on post-processing steps (e.g., clustering algorithms), which limits end-to-end optimization and makes it difficult to handle overlapping binding sites.

**Limitations of evaluation metrics**: The conventional DCC (distance between predicted and true centers) and DCA metrics suffer from two fundamental flaws: (a) they completely ignore key structural attributes such as shape, size, and residue composition of binding sites; and (b) they lack proper matching between predictions and ground truth, which can lead to duplicate counting. Quantitative analysis shows that approximately 20% of proteins are affected by duplicate counting during evaluation.

## Method

### Overall Architecture

UniSite formulates binding site detection as a set prediction task: given a protein sequence $S$ of length $L$, the goal is to identify a set of binding sites $\{m_i^{gt}\}_{i=1}^{N_{gt}}$, where each binding site is represented by a binary mask $m_i^{gt} \in \{0,1\}^L$. The overall architecture consists of an encoder, a Transformer decoder, and a segmentation module, enabling direct prediction of $N$ potentially overlapping binding sites in a single forward pass.

### Key Designs

1. **UniSite-DS Dataset Construction**: The AHoJ tool is used to systematically search all protein–ligand interactions in the PDB. After rigorous filtering (resolution < 2.5 Å, crystallographic method, number of ligand atoms ≥ 5, etc.), all binding sites of the same protein are mapped to its sequence via UniProt unique identifiers. NMS (IoM threshold 0.7, IoU threshold 0.5) is applied to remove redundancy. The final dataset contains 11,510 valid UniProt IDs, of which 3,670 have multiple binding sites—representing 4.81× more multi-site data than existing datasets.

2. **Set Prediction Loss with Bijective Matching**: The model produces a fixed set of $N$ predictions $z = \{(p_i, m_i)\}_{i=1}^N$ in a single forward pass. The Hungarian algorithm is used to compute the optimal assignment between predictions and ground truth:

$$\hat{\sigma} = \arg\min_\sigma \sum_i^N \mathcal{L}_{\text{match}}(z_i^{gt}, z_{\sigma(i)})$$

The matching cost jointly considers classification probability and mask similarity: $\mathcal{L}_{\text{match}} = -\mathbf{1}_{\{c_i^{gt}\neq\emptyset\}} \log p_{\sigma(i)}(c_i^{gt}) + \mathbf{1}_{\{c_i^{gt}\neq\emptyset\}} \mathcal{L}_{\text{mask}}$. The design motivation is drawn from the DETR object detection paradigm; bijective matching avoids duplicate predictions and enables true end-to-end detection.

3. **Dual-Path Encoder (Sequence + Structure)**:

    - **Sequence Encoder**: Input features include learnable embeddings for 21 amino acid types, sinusoidal positional encodings, and pretrained ESM-2 embeddings. A 3-layer MLP generates initial features, followed by Transformer encoder layers to capture inter-residue interactions.
    - **Structure Encoder (optional)**: GearNet-Edge (an E(3)-invariant GNN) is used, representing the protein structure as a residue-level relational graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$. Node features are updated via relational graph convolution: $h_i^{(l)} = h_i^{(l-1)} + \text{ReLU}(\text{BN}(\sum_{r} W_r \sum_{j \in \mathcal{N}_r(i)} h_j^{(l-1)}))$.

4. **Segmentation Module**: $N$ site queries are projected via MLP into mask embeddings $\mathcal{E}_{\text{mask}} \in \mathbb{R}^{N \times d_{\text{model}}}$, which are dot-producted with residue-level features $\mathcal{F}$ and passed through a sigmoid to obtain mask predictions: $m_i[j] = \text{sigmoid}(\mathcal{E}_{\text{mask}}[i,:] \cdot \mathcal{F}[j,:]^T)$.

### Loss & Training

The total loss combines classification and mask losses: $\mathcal{L}_{\text{mask\&cls}} = \lambda_{\text{cls}} \sum_i^N -\log p_{\hat{\sigma}(i)}(c_i^{gt}) + \mathbf{1}_{\{c_i^{gt}\neq\emptyset\}} \mathcal{L}_{\text{mask}}$, where the mask loss is $\mathcal{L}_{\text{mask}} = 5.0 \cdot \mathcal{L}_{\text{bce}} + 5.0 \cdot \mathcal{L}_{\text{dice}}$ and the classification loss weight is $\lambda_{\text{cls}} = 2.0$. The classification loss for the no-binding class is down-weighted by a factor of 10 to mitigate class imbalance. Auxiliary supervision is applied at each decoder layer using shared-weight segmentation modules. The AdamW optimizer is used with a learning rate of $10^{-4}$ and weight decay of 0.05.

The IoU-based AP metric ranks predictions by confidence, matches each ground-truth site to the highest-scoring prediction with IoU exceeding a threshold (one-to-one constraint), and computes the area under the interpolated precision-recall curve.

## Key Experimental Results

### Main Results (UniSite-DS)

| Method | Type | AP₀.₃ ↑ | AP₀.₅ ↑ |
|--------|------|---------|---------|
| Fpocket | Geometric | 0.1836 | 0.1017 |
| P2Rank | Machine Learning | 0.5056 | 0.2157 |
| DeepPocket | CNN | 0.4273 | 0.2334 |
| GrASP | GNN | 0.4469 | 0.2848 |
| VN-EGNN | GNN | 0.1621 | 0.0705 |
| **UniSite-1D** | **Ours** | **0.5121** | **0.3033** |
| **UniSite-3D** | **Ours** | **0.5603** | **0.3835** |

### Ablation Study

| Configuration | AP₀.₃ | AP₀.₅ | Note |
|---------------|-------|-------|------|
| Sequence similarity < 0.9 | 0.5603 | 0.3835 | Default setting |
| Sequence similarity < 0.7 | 0.5579 | 0.3734 | Good generalization |
| Sequence similarity < 0.5 | 0.4677 | 0.2801 | Performance drops on distant proteins |
| 16 queries | 0.5515 | 0.3795 | Minimal impact from query count |
| 32 queries (default) | 0.5603 | 0.3835 | — |
| 64 queries | 0.5615 | 0.3867 | Marginal improvement |

### Key Findings

- UniSite-1D, using only sequence information, outperforms all baseline methods, demonstrating the feasibility of structure-free binding site detection.
- VN-EGNN performs well on conventional DCC/DCA metrics but extremely poorly under AP (it predicts only centers, discarding structural information).
- IoU-based AP is consistent with traditional metric rankings but provides stronger discriminability: on HOLO4K-sc, DeepPocket and GrASP differ by <0.01 in DCA but by >0.10 in AP₀.₃.
- Approximately 20% of proteins suffer from duplicate counting under DCC/DCA evaluation.

## Highlights & Insights

- **Data perspective innovation**: The shift from PDB-centric to UniProt-centric data construction reveals a fundamental statistical bias in prior datasets. This paradigm is generalizable to the construction of other biomolecular datasets.
- **Transfer of the DETR paradigm**: The work successfully transfers the mature set prediction framework from object detection to protein binding site detection, demonstrating the effectiveness of general-purpose architectures in biological tasks.
- **Rethinking evaluation metrics**: A rigorous analysis exposes the fundamental flaws of DCC/DCA; the introduction of IoU-based AP has lasting implications for the field.

## Limitations & Future Work

- Dataset construction still requires manual inspection to remove erroneous entries; automated curation methods could be developed in the future.
- The model design does not employ specialized feature engineering; incorporating protein surface features may further improve performance.
- The current approach considers binding sites only at the amino acid residue level; fine-grained atomic-level prediction remains to be explored.

## Related Work & Insights

- This work extends the DETR → MaskFormer set prediction paradigm to the biological domain.
- The use of ESM-2 pretrained embeddings highlights the value of protein language models for downstream tasks.
- The proposed framework has direct applicability to virtual screening and de novo molecular design in drug discovery.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first UniProt-centric dataset and end-to-end detection framework; all three contributions (dataset, method, evaluation metric) are pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation on both the proposed dataset and established benchmarks, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, and the analysis of the three identified issues is logically rigorous.
- Value: ⭐⭐⭐⭐⭐ — The dataset, method, and evaluation metric have long-term impact on structural biology and drug discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[ICML 2026\] SwitchCraft: A Programmatic Framework for Designing State-Switching Proteins](../../ICML2026/computational_biology/switchcraft_a_programmatic_framework_for_designing_state-switching_proteins.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [\[ICML 2026\] Site4Drug: Predicting Drug-Binding Target Sites with an AI Agent](../../ICML2026/computational_biology/site4drug_predicting_drug-binding_target_sites_with_an_ai_agent.md)

</div>

<!-- RELATED:END -->
