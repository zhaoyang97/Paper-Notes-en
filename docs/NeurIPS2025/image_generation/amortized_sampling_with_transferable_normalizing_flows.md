---
title: >-
  [Paper Note] Amortized Sampling with Transferable Normalizing Flows
description: >-
  [NeurIPS 2025][Image Generation][normalizing flow] This work proposes Prose — a 285M-parameter all-atom transferable normalizing flow based on the TarFlow architecture, trained on 21,700 short-peptide MD trajectories (totaling 4.3 ms of simulation time). Prose enables zero-shot uncorrelated proposal sampling for arbitrary short-peptide systems, outperforms MD baselines under equal energy evaluation budgets, and generates samples 4,000× faster than the prior transferable Boltzmann generator (TBG).
tags:
  - NeurIPS 2025
  - Image Generation
  - normalizing flow
  - Boltzmann generator
  - transferable sampler
  - peptide
  - importance sampling
date: 2026-05-08
content_hash: e483e31ff70d38eb
---

# Amortized Sampling with Transferable Normalizing Flows

**Conference**: NeurIPS 2025
**arXiv**: [2508.18175](https://arxiv.org/abs/2508.18175)
**Code**: [GitHub](https://github.com/transferable-samplers/transferable-samplers) | [Model Weights](https://huggingface.co/transferable-samplers/model-weights) | [Dataset](https://huggingface.co/datasets/transferable-samplers/many-peptides-md)
**Area**: Molecular Generation / Normalizing Flows / Statistical Sampling
**Keywords**: normalizing flow, Boltzmann generator, transferable sampler, peptide, importance sampling

## TL;DR
This work proposes Prose — a 285M-parameter all-atom transferable normalizing flow based on the TarFlow architecture, trained on 21,700 short-peptide MD trajectories (totaling 4.3 ms of simulation time). Prose enables zero-shot uncorrelated proposal sampling for arbitrary short-peptide systems, outperforms MD baselines under equal energy evaluation budgets, and generates samples 4,000× faster than the prior transferable Boltzmann generator (TBG).

## Background & Motivation

**Background**: Sampling molecular conformations from the Boltzmann distribution is a central problem in computational chemistry, with applications in protein folding and drug design. Traditional approaches (MD, MCMC) are Markovian — the computational cost for each system must be paid from scratch and cannot be amortized. They also produce highly correlated samples, making efficient exploration of multimodal distributions difficult.

**Limitations of Prior Work**: (a) MD requires femtosecond time steps, yielding highly correlated samples that necessitate long simulations to cover metastable states; (b) deep learning samplers (Boltzmann generators, BGs) are effective for single systems but **are largely non-transferable to new systems**; (c) the only prior transferable BG (TBG) is based on continuous normalizing flows, making density evaluation extremely slow (only 3×10⁴ samples in 4 GPU-days), and successful transfer was demonstrated only on dipeptides.

**Key Challenge**: There is a need for a sampling method that simultaneously generates samples efficiently, evaluates likelihoods precisely (for importance sampling correction), and transfers across systems.

**Goal**
   - Can a sampler be trained that transfers across varying amino acid compositions, sequence lengths, and temperatures?
   - Can it achieve better sample efficiency than MD?

**Key Insight**: Large-scale autoregressive normalizing flows (TarFlow architecture) + a large-scale short-peptide MD dataset + chemistry-aware sequence permutations, enabling a scalable transferable Boltzmann generator.

**Core Idea**: A 285M-parameter Transformer normalizing flow is trained on 21,700 short-peptide systems, enabling zero-shot cross-system proposal sampling corrected via self-normalized importance sampling (SNIS).

## Method

### Overall Architecture
Prose is an all-atom autoregressive normalizing flow: Gaussian noise $z$ is mapped to molecular conformations $x = f_\theta^{-1}(z)$ through an invertible transformation, while the exact likelihood $q_\theta(x)$ is computed via the Jacobian determinant of the transformation. Training uses maximum likelihood on MD trajectories; at inference, SNIS reweighting corrects model errors. Cross-system transfer is achieved by conditioning on atom type, residue type, residue position, and sequence length.

### Key Designs

1. **TarFlow Architecture with Variable-Length Sequence Support**

    - Function: Supports parallel training and inference across peptide sequences of varying lengths.
    - Mechanism: TarFlow parameterizes a sequence of autoregressive affine transformations using Transformer blocks. Prose extends TarFlow by: (a) applying masking to padding tokens for variable-length sequences, preventing padding from affecting computations and log-determinants; (b) replacing fixed learned embeddings with sinusoidal positional encodings to support length extrapolation; (c) normalizing the log-likelihood per dimension $\frac{1}{d(s)} \log q_\theta(x)$ to handle dimensional discrepancies across systems.
    - Design Motivation: Transformers naturally handle variable-length sequences, but normalizing flows require fixed dimensionality — masking elegantly resolves this conflict.

2. **Chemistry-Aware Permutations**

    - Function: Defines autoregressive ordering more suited to peptide modeling.
    - Mechanism: Standard TarFlow uses only identity and reversal permutations. Prose introduces a "backbone-first" permutation: all backbone atoms $[N_i, C_{\alpha,i}, C_i, O_i]$ across residues are processed before side chains. This allows the model to attend to the complete backbone structure via causal attention when generating side chains — local updates are informed by global structure.
    - Design Motivation: Unlike images on a regular grid, molecules exhibit a hierarchical backbone–side-chain structure. A "skeleton before details" ordering aligns with molecular physics, as side-chain conformations are strongly dependent on backbone conformation.

3. **Adaptive System Conditioning**

    - Function: Informs the model of the peptide system currently being generated.
    - Mechanism: Conditioning features $h[i] = [A_i, R_i, P_i, L]$ (atom type, residue type, residue position, sequence length) are injected into the Transformer via adaptive LayerNorm, adaptive scaling, and SwiGLU transition blocks (inspired by AlphaFold3).
    - Design Motivation: More expressive than simple additive conditioning, enabling the model to capture complex physicochemical properties of different amino acids.

4. **Self-Improvement Fine-tuning**

    - Function: Fine-tunes the model on unseen systems without any training data.
    - Mechanism: Samples are drawn from $q_\theta(x|s)$, resampled using importance weights $w_i = p(x_i)/q_\theta(x_i)$, and the resampled "pseudo-real data" is used for maximum likelihood fine-tuning. No real MD trajectories are required — the process is fully self-bootstrapped.
    - Design Motivation: While zero-shot performance is already strong, fine-tuning yields further gains. SNIS resampling produces samples closer to the target distribution, serving as training data at no additional simulation cost.

### Loss & Training
- Training: Maximum likelihood $\max_\theta \mathbb{E}_s \frac{1}{d(s)} \mathbb{E}_{x \sim p(x|s)} \log q_\theta(x)$
- Inference: SNIS (self-normalized importance sampling) correction
- Dataset: ManyPeptidesMD — 21,700 sequences × 200 ns = 4.3 ms total simulation time
- Model: 285M parameters, TarFlow + Transformer

## Key Experimental Results

### Main Results: Zero-Shot Performance vs. MD (30 Unseen Tetrapeptide Systems)

| Method | Energy W₂↓ | Dihedral W₂↓ | TICA W₂↓ |
|--------|-----------|-------------|---------|
| MD (1μs) | Baseline | Baseline | Baseline |
| **Prose + SNIS** | **Better than MD** | **Better than MD** | **Significantly better than MD** |

Prose comprehensively outperforms 1μs MD under equal energy evaluation budgets, with particularly large advantages on macroscopic structural metrics (TICA).

### Speed Comparison

| Method | Proposal Speed |
|--------|--------------|
| TBG (continuous normalizing flow) | 4 GPU-days / 3×10⁴ samples |
| **Prose** | **4,000× faster** |

### Cross-Length Transfer

| Length | Seen During Training | Zero-Shot Performance |
|--------|--------------------|-----------------------|
| 2–8 residues | ✓ | Excellent |
| >8 residues | ✗ | To be verified |

### Key Findings
- **Zero-shot outperforms MD**: On macroscopic structural metrics, Prose surpasses computationally expensive MD without ever observing the target system — because MD may become trapped in a single metastable state within a limited time budget, whereas Prose generates uncorrelated samples.
- **SNIS is sufficient**: Complex advanced sampling algorithms such as SMC are unnecessary; simple importance sampling reweighting suffices to produce high-quality distributional estimates, indicating that Prose's proposal distribution is of high quality.
- **Self-improvement fine-tuning is effective**: Iterative self-improvement on unseen systems yields continuous performance gains, approaching the quality of models trained on real MD data.
- **Temperature transfer**: Simply scaling the prior temperature $\beta \log q_z(z)$ yields reasonable proposals at different temperatures — although theoretically imprecise for non-volume-preserving flows, it proves effective in practice.

## Highlights & Insights
- The **paradigm shift of "amortized sampling"** is highly valuable: whereas traditional MD requires full simulation for each system from scratch, Prose "prepays" most computation during training, making inference nearly cost-free across new systems.
- **Chemistry-aware permutations** represent a practically useful design choice for molecular generation: exploiting the hierarchical structure of molecules (backbone → side chains) to define autoregressive ordering aligns with physical intuition and yields significant performance improvements.
- **Full open-sourcing** of code, model weights, and dataset substantially facilitates community research.
- The **4,000× speedup** over the only prior transferable BG renders the method genuinely practical for real-world research.

## Limitations & Future Work
- **Limited to short peptides (≤8 residues)**: Proteins typically contain 100–300 residues; the current method remains far from protein-scale applicability.
- **Implicit solvent**: Training uses implicit solvent models; explicit solvent effects are not accounted for.
- **Restricted to the Amber14 force field**: Performance on other force fields remains unvalidated.
- **Temperature transfer is approximate**: Simple prior temperature scaling is only an approximation for non-volume-preserving flows.
- **Long-sequence extrapolation**: Although variable-length sequences are supported, performance may degrade for systems beyond the training length range.

## Related Work & Insights
- **vs. TBG (Klein & Noe)**: TBG employs continuous normalizing flows with extremely slow density evaluation; Prose uses autoregressive normalizing flows (TarFlow), achieving 4,000× faster density evaluation and successful transfer to longer sequences.
- **vs. BG (Noé et al.)**: Standard BGs are trained on single systems; Prose achieves cross-system transfer through conditioning and large-scale training.
- **vs. AlphaFold3 / AITHYRA**: These methods target structure prediction, while Prose targets equilibrium sampling — the two are complementary rather than competing.
- **Transferability insight**: The paradigm of large-scale cross-system training combined with system conditioning is transferable to other scientific sampling tasks (crystal structures, small-molecule conformations, etc.).

## Rating
- Novelty: ⭐⭐⭐⭐ First BG to successfully transfer across peptide lengths, though the core architecture (TarFlow) is not original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale dataset, multiple baselines, diverse metrics, ablations, and self-improvement analysis — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and rigorous; background, method, and experiments are very well organized.
- Value: ⭐⭐⭐⭐⭐ Significant implications for computational chemistry and drug design; fully open-sourced resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FALCON: Few-step Accurate Likelihoods for Continuous Flows](falcon_few-step_accurate_likelihoods_for_continuous_flows.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[NeurIPS 2025\] Multimodal Generative Flows for LHC Jets](multimodal_generative_flows_for_lhc_jets.md)
- [\[NeurIPS 2025\] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models](transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)

</div>

<!-- RELATED:END -->
