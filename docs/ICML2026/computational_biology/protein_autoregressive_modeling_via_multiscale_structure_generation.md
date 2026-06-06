---
title: >-
  [Paper Note] Protein Autoregressive Modeling via Multiscale Structure Generation
description: >-
  [ICML 2026][Computational Biology][Protein backbone generation] PAR adapts the "next-scale prediction" concept from the image domain (VAR) to protein $C\alpha$ backbone generation. By replacing single-scale diffusion mod…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Protein backbone generation"
  - "multiscale autoregressive"
  - "next-scale prediction"
  - "flow matching"
  - "exposure bias"
date: 2026-05-08
content_hash: 81a35393d0d69319
---

# Protein Autoregressive Modeling via Multiscale Structure Generation

**Conference**: ICML 2026  
**arXiv**: [2602.04883](https://arxiv.org/abs/2602.04883)  
**Code**: https://par-protein.github.io (Project Page)  
**Area**: Scientific Computing / Protein Structure Generation / Autoregressive Generative Models  
**Keywords**: Protein backbone generation, multiscale autoregressive, next-scale prediction, flow matching, exposure bias  

## TL;DR
PAR adapts the "next-scale prediction" concept from the image domain (VAR) to protein $C\alpha$ backbone generation. By replacing single-scale diffusion models with multiscale downsampling, an autoregressive (AR) Transformer, and a flow-based decoder—complemented by noisy context learning and scheduled sampling to mitigate exposure bias—it achieves a state-of-the-art FPSD of 161.0 in unconditional generation, enables zero-shot point-prompt generation and motif scaffolding, and provides a 2.5$\times$ sampling speedup.

## Background & Motivation

**Background**: Protein backbone generation is largely dominated by diffusion/flow matching models. These generally fall into two categories: those predicting per-residue SE(3) frames (FrameDiff, RFDiffusion, Genie2) and those directly modeling $C\alpha$ coordinates (Proteina, etc.). Both classes are **single-scale**, meaning they denoise the full structure of length $L$ in one go.

**Limitations of Prior Work**: Autoregressive (AR) models have demonstrated strong scaling and zero-shot capabilities in LLMs and image generation, yet they struggle with protein structures. Two specific obstacles exist: (i) continuous 3D atomic coordinates require discretization (e.g., VQ-VAE), which loses fine-grained structural details and hurts designability; (ii) protein residues exhibit strong bidirectional dependencies—residues far apart in sequence may form hydrogen bonds or hydrophobic contacts in space—which directly conflicts with the **unidirectional next-token** assumption of standard AR. Previous token-based AR approaches like ESM3 or Gaujac et al. show FPSD metrics an order of magnitude worse than diffusion baselines.

**Key Challenge**: To use AR models for proteins, one must bypass the "precision loss from discretization" and the "unidirectional order disrupting bidirectional dependencies." Both issues stem from the standard implementation of the AR paradigm.

**Goal**: Design an AR framework that (1) models continuous $C\alpha$ coordinates instead of discrete tokens, (2) expands across scales rather than residue-by-residue to preserve intra-scale bidirectional dependencies, and (3) leverages the zero-shot and scaling advantages of AR models.

**Key Insight**: Proteins naturally possess a **hierarchical structure**—from coarse-grained tertiary topology and secondary structure arrangements to atomic coordinates. This multiscale granularity is isomorphic to the "next-scale prediction" proposed by VAR in image generation: shifting the AR expansion dimension from "spatial position" to "scale," while maintaining full bidirectional attention within each scale.

**Core Idea**: Hierarchically downsample the protein along the sequence dimension to obtain $\{\mathbf{x}^1, \dots, \mathbf{x}^n\}$ (e.g., scales of 64→128→256). Use an AR Transformer for next-scale prediction to generate conditional embeddings for each scale, then use a flow-matching decoder to generate continuous $C\alpha$ coordinates conditioned on these embeddings—AR "carves the silhouette," while the flow "carves the details."

## Method

### Overall Architecture

Input: Protein backbones $\mathbf{x} \in \mathbb{R}^{L\times 3}$ from the training set (modeling $C\alpha$ atoms).  
Output: A complete backbone structure generated autoregressively from coarse to fine.

The framework is modeled as:  
$$p_\theta(\mathbf{x}) = \prod_{i=1}^n p_\theta(\mathbf{x}^i \mid \mathbf{z}^i = \mathcal{T}_\theta(X^{<i}))$$

The pipeline consists of three stages:
1. **Multiscale Downsampling (qdecompose)**: Interpolate $\mathbf{x}$ along the sequence dimension to obtain $n$ coarse-to-fine scales $\{\mathbf{x}^1, \dots, \mathbf{x}^n = \mathbf{x}\}$, serving as training contexts and targets. Downsampling is **deterministic and parameter-free**.
2. **AR Transformer $\mathcal{T}_\theta$ for Inter-scale Conditional Prediction**: Upsample previous scales $i-1$ to $\texttt{size}(i)$, concatenate them, and feed them into a non-equivariant Transformer to output a conditional embedding $\mathbf{z}^i$ for the current scale.
3. **Flow-based Backbone Decoder $\mathbf{v}_\theta$ for Intra-scale Continuous Generation**: Conditioned on $\mathbf{z}^i$, use flow matching to directly generate $C\alpha$ coordinates in $\mathbb{R}^{\texttt{size}(i)\times 3}$. During inference, start from $\texttt{bos}$ and iterate $n$ times (accelerated by KV caching).

### Key Designs

1. **Multiscale Downsampling + Next-scale Prediction**:
    - **Function**: Represents protein structure as $\{\mathbf{x}^1, \dots, \mathbf{x}^n\}$, shifting the AR expansion from "residues" to "scales."
    - **Mechanism**: Each scale $i$ uses $\text{Down}(\mathbf{x}, \texttt{size}(i))$ to get $\texttt{size}(i)$ 3D centroids. Scale configurations $\mathcal{S}$ can be fixed by length (e.g., $\{64, 128, 256\}$) or ratio ($\{L/4, L/2, L\}$). Positional encodings $p^i = \text{linspace}(1, L, \texttt{size}(i))$ allow coarse scales to capture global layout and fine scales to focus on local details.
    - **Design Motivation**: Maintaining full bidirectional attention within each scale (using a non-equivariant Transformer) avoids the conflict between AR's unidirectional assumption and residue dependencies. Reducing the AR depth to $n=3$ instead of $L$ makes the chain shorter and less prone to error accumulation.

2. **AR Transformer Condition + Flow-based Decoder for Continuous $C\alpha$**:
    - **Function**: Allows the AR model to output conditional embeddings instead of discrete tokens.
    - **Mechanism**: $\mathbf{z}^i = \mathcal{T}_\theta([\texttt{bos}, \text{Up}(\mathbf{x}^1, \texttt{size}(2)), \dots, \text{Up}(\mathbf{x}^{i-1}, \texttt{size}(i))])$. For flow matching, training involves interpolating $\mathbf{x}^i_{t^i} = t^i \mathbf{x}^i + (1-t^i)\boldsymbol{\epsilon}^i$, optimizing $\mathcal{L}(\theta) = \mathbb{E}[\frac{1}{n}\sum_i \frac{1}{\texttt{size}(i)} \|\mathbf{v}_\theta(\mathbf{x}^i_{t^i}, t^i, \mathbf{z}^i) - (\mathbf{x}^i - \boldsymbol{\epsilon}^i)\|^2]$. $\mathbf{z}^i$ is injected via adaptive layer norm. Inference follows the ODE $d\mathbf{x}_t = \mathbf{v}_\theta dt$ or an SDE with a score term.
    - **Design Motivation**: Completely bypasses detail loss from VQ discretization. When $n=1$, PAR defaults to standard flow matching, ensuring backward compatibility with techniques like self-conditioning.

3. **Noisy Context Learning (NCL) + Scheduled Sampling (SS) to Mitigate Exposure Bias**:
    - **Function**: Addresses the train-inference mismatch where training uses ground-truth contexts but inference uses model predictions.
    - **Mechanism**: NCL adds noise to the context of previous scales during training: $\mathbf{x}^i_{\text{ncl}} = w^i_{\text{ncl}} \mathbf{x}^i + (1-w^i_{\text{ncl}}) \boldsymbol{\epsilon}^i_{\text{ncl}}$, with weights sampled from $[0,1]$. SS replaces the ground-truth context with the flow decoder's prediction $\mathbf{x}^i_{\text{pred}}$ at scale $i$ with a 0.5 probability.
    - **Design Motivation**: Preliminary studies showed that pure teacher forcing causes designability to collapse (sc-RMSD 2.20). NCL improves sc-RMSD to 1.58 and PDB-FPSD from 99.66 to 89.70; adding SS further reaches 1.48.

### Loss & Training
A single flow matching objective (Eq. 5) is used, with the AR Transformer and flow decoder **trained jointly end-to-end**. Training involves two stages: pre-training on the AFDB representative dataset (200K steps) followed by fine-tuning on a 21K designable PDB subset (5K steps) to yield $\text{PAR}_{\text{pdb}}$. The default uses 3 scales $\{64, 128, 256\}$ with model sizes up to 400M parameters.

## Key Experimental Results

### Main Results: Unconditional Backbone Generation

| Method | Params | Designability ↑ | sc-RMSD ↓ | FPSD vs PDB ↓ | FPSD vs AFDB ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FrameDiff | 17M | 65.4% | - | 194.2 | 258.1 |
| RFDiffusion | 60M | 94.4% | - | 253.7 | 252.4 |
| ESM3 (token AR) | 1.4B | 22.0% | - | 933.9 | 855.4 |
| Genie2 | 16M | 95.2% | - | 350.0 | 313.8 |
| Proteina | 400M | 92.6% | 1.09 | 271.3 | 272.6 |
| **PAR** | 400M | **96.0%** | **1.01** | 313.9 | 296.4 |
| **PAR$_{\text{pdb}}$** | 400M | **96.6%** | 1.04 | **161.0** | 228.4 |

PAR outperforms SOTA single-scale baselines in designability and sc-RMSD. After fine-tuning, PDB-FPSD reaches 161.0, which is 36% lower than RFDiffusion (253.7), proving that multiscale AR better captures the true protein distribution.

### Ablation Study: Mitigating Exposure Bias (60M, 100K steps)

| Training Strategy | sc-RMSD ↓ | FPSD vs PDB ↓ | FPSD vs AFDB ↓ |
| :--- | :--- | :--- | :--- |
| Teacher Forcing | 2.20 | 99.66 | 37.64 |
| + NCL | 1.58 | 89.70 | 23.69 |
| + NCL + SS | **1.48** | 90.66 | 24.59 |

NCL alone reduces sc-RMSD from 2.20 to 1.58, making it the most critical engineering component for PAR's success.

### Ablation Study: Scale Configuration (60M)

| Scale Configuration | Designability ↑ | FPSD vs AFDB ↓ |
| :--- | :--- | :--- |
| {64, 256} | 83.0% | 274.32 |
| **{64, 128, 256}** | **85.0%** | **267.35** |
| {64, 128, 192, 256} | 77.8% | 282.69 |

Three scales represent the "sweet spot." Performance drops at 4-5 scales due to intensified exposure bias from error accumulation.

### Key Findings
- **Sampling Efficiency**: By restricting SDE to the first scale (to establish global topology) and using ODE for subsequent scales (only 2 steps), PAR achieves a **2.5$\times$ speedup** over Proteina (400 steps) for length 200, while maintaining 94-98% designability.
- **Zero-shot Point Prompt Generation**: Injecting 16 3D points as prompts at the first scale allows PAR to generate structures following that layout without fine-tuning.
- **Zero-shot Motif Scaffolding**: Teacher-forcing motif coordinates at each scale enables scaffold generation without specialized training or masking.
- **Scaling Laws**: FPSD continues to drop and designability improves from 60M to 400M parameters.
- **Attention Visualization**: Each scale primarily attends to the **previous scale**, but maintains non-zero attention to earlier scales, indicating true cross-scale information integration.

## Highlights & Insights
- **Clean Paradigm Shift**: Effectively maps the VAR paradigm to proteins by shifting AR from tokens to scales to solve discretization and dependency issues.
- **Elegant AR + Flow Hybrid**: Inherits the maturity of flow matching while enjoying the multiscale context.
- **Impact of NCL**: Adding noise to context is a simple but powerful trick that significantly improves performance.
- **Coarse-to-Fine SDE/ODE**: Utilizing SDE only at the coarse stage to "anchor" the distribution allows for efficient ODE refinement later.

## Limitations & Future Work
- Currently only models the **$C\alpha$ backbone**, lacking sidechains and full-atom details.
- Scale configurations rely on heuristics; performance gains plateau or decrease beyond 3 scales.
- Lack of **wet-lab validation** to confirm if generated proteins are truly foldable in a biological setting.

## Related Work & Insights
- **vs. Proteina**: Proteina is a special case of PAR where $n=1$. PAR adds scale context for better distribution fitting and zero-shot capabilities.
- **vs. ESM3**: ESM3's 22% designability highlights the failure of discretization-based AR for structure, which PAR solves by remaining continuous.
- **vs. VAR**: While sharing the "next-scale prediction" logic, PAR replaces VQ-VAE with a flow decoder to handle the continuous nature of 3D coordinates.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First successful adaptation of hierarchical scale-based AR to protein structures.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations and scaling analysis, though missing wet-lab data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and rigorous formulation.
- **Value**: ⭐⭐⭐⭐ Offers a new architectural path for protein design with practical speed and zero-shot benefits.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)
- [\[ICML 2026\] LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation](lineageflow_flow_matching_for_high-fidelity_family-aware_protein_sequence_genera.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](../../NeurIPS2025/computational_biology/multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)

</div>

<!-- RELATED:END -->
