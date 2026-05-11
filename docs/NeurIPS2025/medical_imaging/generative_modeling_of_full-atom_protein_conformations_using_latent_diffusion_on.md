---
title: >-
  [Paper Note] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings
description: >-
  [NeurIPS 2025][Medical Imaging][protein conformation generation] This work proposes **LD-FPG**, a framework that encodes full-atom MD trajectories into a low-dimensional latent space via Chebyshev graph neural networks a…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "protein conformation generation"
  - "latent diffusion"
  - "graph neural networks"
  - "full-atom modeling"
  - "GPCR"
date: 2026-05-08
content_hash: e26a9e392036b162
---

# Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings

**Conference**: NeurIPS 2025
**arXiv**: [2506.17064](https://arxiv.org/abs/2506.17064)
**Code**: Available (open-source)
**Area**: Medical Imaging
**Keywords**: protein conformation generation, latent diffusion, graph neural networks, full-atom modeling, GPCR
**arXiv**: [2506.17064](https://arxiv.org/abs/2506.17064)
**Code**: Unavailable
**Area**: Medical Imaging

## TL;DR

This work proposes **LD-FPG**, a framework that encodes full-atom MD trajectories into a low-dimensional latent space via Chebyshev graph neural networks and applies DDPM in that space to generate novel conformational ensembles. To the authors' knowledge, this is the first approach to generate protein conformations that includes all heavy atoms of the side chains.

## Background & Motivation

**Background**: Protein function depends on dynamic transitions among distinct conformational states. Methods such as AlphaFold2 primarily predict a single static conformation and cannot capture functionally relevant conformational diversity.

**Limitations of Prior Work**: Existing generative models either produce backbone-only representations (without side chains), yield coarse-grained outputs, or are limited to de novo design rather than conformation sampling for a specific protein. Side-chain rearrangements, which often govern molecular recognition and catalytic mechanisms, are thus largely neglected.

**Key Challenge**: There is a fundamental gap between the need to generate full-atom conformational ensembles (including every side-chain heavy atom) and the capabilities of existing methods—particularly for GPCRs, which exhibit complex dynamics in membrane environments.

**Goal**: To learn and generate high-fidelity full-atom conformational ensembles of specific proteins (e.g., the dopamine D2 receptor) from existing molecular dynamics (MD) simulation data.

**Key Insight**: Rather than simulating new MD trajectories, the method learns a latent representation of MD data by modeling conformations as deformations relative to a reference structure.

**Core Idea**: A four-stage pipeline—ChebNet encoding, pooling-based compression, DDPM sampling, and conditional decoding—generates full-atom conformations within a compact latent space.

## Method

### Overall Architecture (Figure 1)

1. **ChebNet Encoding**: Full-atom coordinates of each MD frame are encoded into per-atom latent embeddings $Z^{(t)} \in \mathbb{R}^{N \times d_z}$.
2. **Pooling Compression**: The high-dimensional $Z^{(t)}$ is pooled into a compact latent vector $\mathbf{h}_0$ (approximately 60–1100 dimensions).
3. **DDPM Generation**: A DDPM is trained in the pooled latent space to generate $\mathbf{h}_0^{\text{gen}}$.
4. **Conditional Decoding**: The latent representation of a reference structure $Z_{\text{ref}}$ is used as a condition to decode $\mathbf{h}_0^{\text{gen}}$ back to full-atom coordinates.

### Key Design 1: ChebNet Multi-Hop Encoding

- **Function**: Maps Kabsch-aligned atomic coordinates of each frame to latent embeddings.
- **Mechanism**: Employs four layers of Chebyshev graph convolution ($K=4$ polynomial order):
$$H^{(l+1)} = \sigma\left(\sum_{k=0}^{K-1} \Theta_k^{(l)} T_k(\tilde{L}) H^{(l)}\right)$$
  A $k$-NN graph ($k=4$) is constructed, with BatchNorm applied after each layer and $L_2$ normalization at the output.
- **Design Motivation**: Spectral graph convolution captures multi-hop inter-atomic relationships and encodes local geometry without relying on global attention.
- **Conditioning Mechanism**: A frozen pretrained encoder generates the reference structure embedding $C = Z_{\text{ref}}$, which proves more effective than conditioning on raw coordinates directly.

### Key Design 2: Three Pooling Strategies

| Strategy | Description | $d_z$ | Latent Dimension |
|---|---|---|---|
| **Blind pooling** | Global adaptive average pooling over all $N$ atoms | 16 | ~100 |
| **Sequential pooling** | Backbone decoded first; side chains decoded conditioned on backbone | 8 | ~100 |
| **Residue pooling** | Pooling performed per residue, each described independently | 4 | $N_{\text{res}} \times d_p \approx 1100$ |

- **Design Motivation**: The high-dimensional $Z^{(t)}$ (up to 35K dimensions for D2R) cannot be fed directly into a DDPM and must be compressed. Values of $d_p > 200$–$300$ impede diffusion training, while $d_p < 50$ degrades reconstruction quality.

### Key Design 3: Diffusion and Loss Functions

- **DDPM Loss**: $\mathcal{L}_{\text{diffusion}}(\theta) = \mathbb{E}_{t,\mathbf{h}_0,\epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{h}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2]$
- **Decoder Loss**: Blind and Residue strategies use $\mathcal{L}_{\text{coord}}$ (coordinate MSE); Sequential uses $\mathcal{L}_{\text{BB}}$ and $\mathcal{L}_{\text{SC}}$ in separate stages.
- **Optional Dihedral Fine-Tuning**: $\mathcal{L}_{\text{Dec}} = w_{\text{base}}\mathcal{L}_{\text{coord}} + \lambda_{\text{mse}}\mathcal{L}_{\text{mse\_dih}} + \lambda_{\text{div}}\mathcal{L}_{\text{div\_dih}}$

## Key Experimental Results

### Decoder Reconstruction Performance (Table 1)

| Decoder | lDDT$_{\text{All}}$ ↑ | lDDT$_{\text{BB}}$ ↑ | $\sum$JSD$_{\text{bb}}$ ↓ | $\sum$JSD$_{\text{sc}}$ ↓ | MSE$_{\text{sc}}$ ↓ |
|---|---|---|---|---|---|
| Blind (dz=16) | 0.714 | 0.792 | 0.0032 | 0.0290 | 0.3934 |
| Sequential (dz=8) | **0.718** | **0.800** | **0.0026** | 0.0192 | 0.5130 |
| Residue (dz=4) | 0.704 | 0.777 | 0.0078 | **0.0125** | **0.2257** |
| Ground Truth (MD) Ref | 0.698 | 0.779 | - | - | - |

### Diffusion Generation Performance (Table 2)

| Model | lDDT$_{\text{All}}$ ↑ | $\sum$JSD$_{\text{bb}}$ ↓ | $\sum$JSD$_{\text{sc}}$ ↓ | Avg. Clashes ↓ |
|---|---|---|---|---|
| Blind pooling | **0.719** | 0.006582 | 0.04185 | 1350.5 |
| Sequential pooling | 0.712 | **0.0029** | 0.02895 | 1220.5 |
| Residue pooling | 0.688 | 0.0117 | **0.0224** | **1145.6** |
| MD reference | ~0.698 | - | - | ~1023 |

### Ablation Study

- **ChebNet Encoding Fidelity**: At $d_z=16$, reconstruction achieves MSE$_{\text{bb}}=0.0008$ and JSD~$0.00016$, establishing an upper bound on fidelity.
- **Dihedral Fine-Tuning**: Yields only marginal JSD improvements for the Blind strategy while slightly reducing lDDT.
- **Comparison with BioEmu**: The general-purpose MD model BioEmu produces A100 distributions (mean = −17.19) that deviate substantially from the D2R-MD reference (mean ≈ −47.5).

### Key Findings

- **Each pooling strategy has distinct strengths**: Blind pooling achieves the best global fidelity; Sequential pooling excels at backbone geometry; Residue pooling performs best on side-chain rotamers and clash count.
- **Residue pooling**, despite slightly weaker global backbone metrics, provides the most complete coverage of the A100 conformational landscape (after multi-epoch sampling), attributable to its larger effective latent space (~1.1K dimensions).
- Clash counts in generated structures (1145–1350) remain higher than the MD reference (~1023), representing the primary current limitation.

## Highlights & Insights

1. **First full-atom conformation generation framework**: To the authors' knowledge, LD-FPG is the first latent diffusion framework designed specifically for full-atom protein conformational ensemble generation.
2. **Deformation-based modeling**: Reformulating generation as learning deformations relative to a reference structure substantially simplifies the generative task.
3. **Design intuition behind residue pooling**: Pooling per residue grants each amino acid an independent deformation descriptor, naturally aligning with the fundamental chemical unit of proteins.
4. **Compelling comparison with BioEmu**: The analysis demonstrates that general-purpose models substantially undersample functionally relevant states of specific membrane proteins, underscoring the necessity of system-specific approaches.

## Limitations & Future Work

1. **Steric clashes**: Generated structures exhibit significantly more atomic clashes than the MD reference; incorporation of lightweight energy proxies or physical constraints is needed.
2. **Multi-epoch sampling dependence of residue pooling**: Complete conformational diversity requires aggregating samples across multiple DDPM training epochs.
3. **Single-system validation**: Evaluation is limited to D2R (one GPCR); generalization to other protein systems requires further investigation.
4. **Lack of equivariance**: ChebNet does not inherently guarantee SE(3) equivariance, though Kabsch alignment partially mitigates this limitation.
5. **Information loss from pooling compression**: Aggressive compression from ~35K to ~100 dimensions inevitably discards fine-grained detail; larger $d_p$ values require more training data.

## Related Work & Insights

- **Distinction from AlphaFlow/ESMFlow**: The latter approaches sample conformations by perturbing static predictions, whereas LD-FPG directly learns system-specific conformational distributions from MD data.
- **Complementarity with LatentDiff**: LatentDiff generates novel protein backbone folds, while LD-FPG generates conformational ensembles of known proteins.
- **Implications for drug discovery**: GPCRs are the targets of approximately 50% of approved drugs; accurate modeling of conformational landscapes can accelerate allosteric site discovery and drug design.

## Rating

⭐⭐⭐⭐ (4/5)

This work addresses an important gap in full-atom protein conformation generation. The framework is well-motivated, and the systematic comparison of three pooling strategies provides valuable design guidance. Primary shortcomings include the steric clash issue, single-system validation, and the multi-epoch sampling requirement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning](generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)

</div>

<!-- RELATED:END -->
