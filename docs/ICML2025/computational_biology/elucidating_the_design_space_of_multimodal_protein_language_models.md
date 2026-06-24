---
title: >-
  [Paper Note] Elucidating the Design Space of Multimodal Protein Language Models
description: >-
  [ICML2025 Spotlight][Computational Biology][Multimodal Protein Language Models] This work systematically explores the design space of token-based multimodal protein language models (PLMs). Through innovations across four dimensions—bit-wise discrete modeling, geometry-aware architectures, representation alignment, and multimer data expansion—it reduces the folding RMSD of a 650M parameter model from 5.52 to 2.36, surpassing a 3B baseline model and approaching the level of spe…
tags:
  - "ICML2025 Spotlight"
  - "Computational Biology"
  - "Multimodal Protein Language Models"
  - "Discrete Structure Tokenizer"
  - "Geometry-Aware Attention"
  - "Representation Alignment"
  - "Bit-wise Classification"
  - "Flow Matching"
  - "Multimer Data"
date: 2026-05-08
content_hash: 7e9f3c5c909d858a
---

# Elucidating the Design Space of Multimodal Protein Language Models

**Conference**: ICML2025 Spotlight  
**arXiv**: [2504.11454](https://arxiv.org/abs/2504.11454)  
**Authors**: Cheng-Yen Hsieh, Xinyou Wang, Daiheng Zhang, Dongyu Xue, Fei Ye, Shujian Huang, Zaixiang Zheng, Quanquan Gu (ByteDance Research & UCLA & NJU)  
**Code**: [bytedance.github.io/dplm/dplm-2.1](https://bytedance.github.io/dplm/dplm-2.1)  
**Area**: Computational Biology  
**Keywords**: Multimodal Protein Language Models, Discrete Structure Tokenizer, Geometry-Aware Attention, Representation Alignment, Bit-wise Classification, Flow Matching, Multimer Data  

## TL;DR

This work systematically explores the design space of token-based multimodal protein language models (PLMs). Through innovations across four dimensions—bit-wise discrete modeling, geometry-aware architectures, representation alignment, and multimer data expansion—it reduces the folding RMSD of a 650M parameter model from 5.52 to 2.36, surpassing a 3B baseline model and approaching the level of specialized folding models.

## Background & Motivation

### Background
Proteins are the fundamental molecular machines of life, with their amino acid sequences determining their 3D structures and biological functions. Traditional methods treat sequence modeling (such as ESM) and structure prediction (such as AlphaFold) as independent tasks, failing to capture the interactions between the two modalities. Recently, multimodal protein language models (such as ESM3, DPLM-2) have achieved simultaneous modeling of sequence and structure within a unified language modeling framework by tokenizing 3D structures into discrete tokens.

### Limitations of Prior Work
- **Information Loss in Structure Tokenization**: Discretizing continuous 3D coordinates into tokens inevitably loses fine-grained geometric relationships and structural details.
- **Inaccurate Structure Token Prediction**: When predicting structure tokens, language models rely on index-based supervision labels, which ignore the correlations between semantically similar tokens, hindering learning.
- **Lack of Geometric Inductive Bias**: Standard Transformer architectures lack the ability to model high-order spatial relationships among residues in protein structures.
- **Training Data Limitations**: Existing multimodal PLMs are typically trained only on single-chain proteins, lacking the rich structural interaction information offered by multimer data.
- **Tokenizer Reconstruction Accuracy is Not the Root Bottleneck**: The authors reveal that the bottleneck of structure token prediction lies in the prediction capability of the language model itself rather than the decoding side.

### Core Motivation
Centered around the DPLM-2 framework, this work systematically reviews and expands the design space of multimodal PLMs—tackling four dimensions: generative modeling, architectural design, representation learning, and data strategy—to enable token-based multimodal PLMs to achieve robust structural modeling performance.

## Method

### Base Framework: DPLM-2 Review
DPLM-2 is a multimodal protein language model based on a discrete diffusion framework. For a protein $\text{prot} = (r_1, r_2, \ldots, r_L)$, each residue $r_i = (s_i, x_i)$ consists of an amino acid type $s_i \in \{1, \ldots, 20\}$ and backbone atom coordinates $x_i \in \mathbb{R}^{N_{\text{atoms}} \times 3}$. Structures are encoded into a sequence of discrete tokens via a VQ-VAE, and are jointly modeled alongside amino acid sequences within the discrete diffusion framework.

### Design Dimension 1: Improving Generative Modeling

**Bit-wise Discrete Modeling (Bit-wise Classification)**: Traditional methods treat structure tokens as independent categorical indices under cross-entropy supervision, ignoring relationships between semantically similar tokens. This paper proposes converting the integer index of each structure token into its binary representation (a bit sequence) and performing binary classification on each bit independently. This naturally introduces a similarity structure between tokens—tokens with minor binary representational differences are physically closer in the bit space, thereby providing finer-grained supervisory signals.

**Hybrid Data-space Modeling**: This combines the strengths of discrete diffusion and continuous Flow Matching. While discrete diffusion predicts structure tokens, an auxiliary continuous Flow Matching branch is introduced to directly model the continuous embedding space of tokens. This hybrid strategy compensates for the continuous information lost during pure discrete modeling, enabling the model to capture subtle variations more precisely when generating structure tokens.

### Design Dimension 2: Geometry-Aware Architecture

**Geometry-Aware Attention**: This integrates spatial distance information between residues into the Transformer's attention mechanism. Drawing inspiration from the pairwise representation concept in AlphaFold, a bias term based on residue pair distances or an additional pair embedding is introduced to make attention weights sensitive to the spatial proximity of residues in 3D space. This architectural improvement injects geometric inductive bias into the language model, addressing the inherent limitations of standard Transformers in modeling protein structures.

### Design Dimension 3: Representation Alignment

**Representation Alignment (Structure Representation Alignment)**: Alignment constraints are introduced at the hidden representation level of the language model. The residue representations learned by the PLM are aligned with the structural representations generated by a structural encoder (such as a VQ-VAE encoder or a dedicated structural encoder) during training. Without altering the model's input-output format, this method helps the language model's internal representations better encode spatial structural information, effectively enhancing the diversity of structural generation.

### Design Dimension 4: Data Expansion - Multimer Training

**Multimer Data**: Existing multimodal PLMs are typically trained only on single-chain proteins (monomers). This work presents the first systematic exploration of the impact of multi-chain protein (multimer) data on model capabilities. Multimer structures contain rich inter-chain interaction patterns, such as interface contacts, symmetry, and co-evolutionary signals. Experiments reveal a deep correlation between multimer and monomer modeling: using multimer data not only improves multi-chain protein modeling but also reciprocally enhances single-chain folding performance.

### Loss & Training
- The improvements are implemented incrementally on top of DPLM-2 (650M parameters), maintaining the discrete diffusion training framework.
- Improvements across the four dimensions are added progressively to ensure the contribution of each can be independently measured.
- Pre-training utilizes large-scale sequence databases combined with PDB structural data, with multimer data serving as an additional training set.
- The final DPLM-2.1 model is constructed by combining all the proposed designs.

## Key Experimental Results

### Protein Folding Performance Comparison

| Model | Parameters | PDB Test Set RMSD (Å) ↓ | Notes |
|------|--------|-----------------|------|
| DPLM-2 (baseline) | 650M | 5.52 | Original multimodal PLM baseline |
| DPLM-2 + Bit-wise Classification | 650M | ~4.0 | Finer-grained supervision |
| DPLM-2 + All Designs (DPLM-2.1) | 650M | **2.36** | Ours |
| ESM3 | 3B | >2.36 | 3B parameter baseline surpassed by 650M |
| Specialized Folding Models | - | ~2.3 | Close to specialized model performance |

DPLM-2.1 significantly reduces the folding RMSD on the PDB test set from 5.52 to 2.36, representing a 57% reduction, and surpasses the 3B-parameter baseline model using only 650M parameters.

### Ablation Study of Design Dimensions

| Design Dimension | Main Improvement Direction | Contribution to Folding RMSD | Contribution to Gen. Diversity |
|---------|------------|-------------|--------------|
| Bit-wise Discrete Modeling | Finer supervision signals | Significant decrease | Moderate improvement |
| Flow Matching Hybrid Modeling | Continuous info compensation | Moderate decrease | Significant improvement |
| Geometry-Aware Attention | Spatial inductive bias | Moderate decrease | Significant improvement |
| Representation Alignment | Struct. info internalization | Moderate decrease | Significant improvement |
| Multimer Data | Rich interacting data | Noticeable decrease | Moderate improvement |
| **All Combinations** | **Synergistic stacking** | **5.52→2.36** | **Huge improvement** |

Ablation studies demonstrate the orthogonal and complementary nature of the improvements in each dimension. The combined effect significantly exceeds the sum of individual improvements.

### Structure Generation Diversity
Improvements in the design space (especially geometry-aware architecture and representation alignment) significantly enhance the diversity of unconditional protein structure generation. The generated structures cover a wider range of fold types in topological space, resolving the mode collapse problem of the original DPLM-2.

## Highlights & Insights

- **Systematic Design Space Exploration**: Rather than proposing a single trick, this paper comprehensively reviews four orthogonal dimensions (generative modeling, architecture, representation learning, and data). Each dimension has clear motivations and independent ablation validations, offering high methodological value.
- **Elegant Design of Bit-wise Classification**: Mapping token indices to binary representations and classifying bit-by-bit introduces token-level structural similarity at negligible implementation costs. This idea is generalizable to other VQ-based systems.
- **Small Models Defeating Large Models**: The 650M parameter model outperforms the 3B baseline in folding RMSD, demonstrating that meticulous optimization of the design space can be more effective than brute-force scaling.
- **Bi-directional Gain of Multimer Data**: The discovery that multimer training not only benefits multi-chain modeling but also reciprocally enhances single-chain folding reveals the deep value of data diversity in protein structure modeling.
- **In-depth Analysis of Bottleneck Diagnosis**: Clarifies that improving tokenizer reconstruction accuracy cannot solve the prediction issue; the real bottleneck lies in the language model's token prediction capacity. This insight guided all subsequent design directions.

## Limitations & Future Work

- **Backbone Atoms Only**: Currently models only protein backbone atoms (N, Cα, C, O) without side-chain atom conformations, limiting its application in tasks requiring all-atom precision, such as drug design.
- **Reliance on VQ-VAE Tokenizer**: Structural information is still discretized via VQ-VAE, making tokenizer-specific information loss an architectural performance ceiling. Future work could explore fully continuous alternatives.
- **Evaluation Focused Mainly on Folding**: Primary experiments concentrate on folding and unconditional generation tasks, lacking comprehensive evaluations on downstream applications like motif scaffolding or protein-ligand docking.
- **Unclear Computational Overhead**: The additional training and inference computational overhead introduced by geometry-aware attention and multi-branch training is not fully detailed.
- **Sensitivity to Multimer Data Ratio**: The effect of the proportion of multimer data and the optimal mixing ratio lacks systematic discussion.
- **Gap with End-to-End Structure Prediction Methods**: While folding RMSD is close to specialized models, whether the generative framework of multimodal PLMs can ultimately equal discriminative folding models in precision remains to be verified.

## Related Work & Insights

- **DPLM / DPLM-2 (Wang et al., 2024a/b)**: Built directly upon DPLM-2. DPLM pioneered the application of discrete diffusion to protein sequence generation, and DPLM-2 extended it to multimodality.
- **ESM3 (Hayes et al., 2024)**: EvolutionaryScale's 3B multimodal protein model, which also utilizes a structural tokenization scheme and serves as the major baseline for comparison.
- **AlphaFold2 (Jumper et al., 2021)**: A milestone in protein structure prediction. Its pairwise representation and geometric inductive bias designs inspired the geometry-aware attention module in this paper.
- **ESM-2 / ESMFold (Lin et al., 2022/2023)**: Proved that pure language model pre-training can support high-quality structure prediction, though remaining a single-modality solution.
- **Discrete Diffusion (Austin et al., 2021)**: Theoretical discrete diffusion frameworks like D3PM, which provide the mathematical foundation for the DPLM series.
- **VQ-VAE Structure Tokenizer**: The core component that encodes protein 3D structures into discrete codebooks, whose reconstruction accuracy and codebook size directly impact downstream PLM performance.

The core insight of this paper is that when addressing bottlenecks in multimodal generative models, one should systematically diagnose and improve the system across four dimensions: supervision signals, architectural inductive bias, representation learning, and data, rather than focusing solely on a single technical detail.

## Rating

- Novelty: ⭐⭐⭐⭐ — The methodology of systematic design space exploration is highly valuable, and specific components like bit-wise classification are novel, though the individual technologies are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablation studies are comprehensive and folding performance gains are significant, but more downstream task evaluations are needed.
- Writing Quality: ⭐⭐⭐⭐⭐ — In-depth and clear problem analysis, with a complete logical chain from bottleneck diagnosis to solution design.
- Value: ⭐⭐⭐⭐ — Provides a systematic improvement blueprint for multimodal protein language models, and the result of the 650M model outperforming the 3B model has strong practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[ICML 2025\] Steering Protein Language Models](steering_protein_language_models.md)
- [\[CVPR 2025\] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](../../CVPR2025/computational_biology/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)

</div>

<!-- RELATED:END -->
