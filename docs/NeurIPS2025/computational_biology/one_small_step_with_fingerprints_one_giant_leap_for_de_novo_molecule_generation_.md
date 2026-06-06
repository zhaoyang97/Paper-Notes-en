---
title: >-
  [Paper Note] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra
description: >-
  [NeurIPS 2025 (AI4Mat Workshop)][Computational Biology][Mass Spectrometry] By employing MIST as a spectra-to-fingerprint encoder and MolForge as a fingerprint-to-structure decoder…
tags:
  - "NeurIPS 2025 (AI4Mat Workshop)"
  - "Computational Biology"
  - "Mass Spectrometry"
  - "Molecular Fingerprints"
  - "De Novo Molecule Generation"
  - "MolForge"
  - "MIST"
date: 2026-05-08
content_hash: 43df39708d4980c5
---

# One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra

**Conference**: NeurIPS 2025 (AI4Mat Workshop)  
**arXiv**: [2508.04180](https://arxiv.org/abs/2508.04180)  
**Code**: Available (using open-source MIST + MolForge)  
**Area**: Computational Chemistry / Molecule Generation  
**Keywords**: Mass Spectrometry, Molecular Fingerprints, De Novo Molecule Generation, MolForge, MIST

## TL;DR

By employing MIST as a spectra-to-fingerprint encoder and MolForge as a fingerprint-to-structure decoder, combined with a prior-adjusted thresholding strategy, this work achieves a tenfold performance improvement on the MassSpecGym benchmark for de novo molecular structure generation from mass spectra (top-1 accuracy from 2.3% to 31%).

## Background & Motivation

Mass spectrometry (MS) is a foundational technique in analytical chemistry for characterizing small molecule structures, yet elucidating the structure of unknown compounds from spectra remains a major bottleneck, requiring extensive domain expertise and manual effort. Automated de novo molecule generation aims to directly infer molecular structures from mass spectra.

Existing two-stage pipeline approaches (spectra → molecular fingerprint → molecular structure) suffer from a critical issue: per-bit uncertainty in the encoder compounds into cascading errors in the decoder, leading to poor overall performance. The previous state-of-the-art method DiffMS employs a diffusion model as the decoder but achieves only 2.3% top-1 accuracy.

The core motivation of this paper is that selecting the appropriate decoder can effectively mitigate the weaknesses of the encoder. MolForge operates exclusively on the on-bits (bits with value 1) of the fingerprint, focusing on present substructures and thereby reducing noise introduced by false-positive bits.

## Method

### Overall Architecture

A classical two-stage pipeline is adopted:
1. **Encoding stage**: A pretrained MIST model encodes mass spectra into 4096-bit molecular fingerprints (as probability values).
2. **Thresholding**: Fingerprint probabilities are binarized via a threshold.
3. **Decoding stage**: MolForge, a Transformer-based autoregressive model, decodes on-bit indices of the fingerprint into SMILES strings.

### Key Designs

**MIST Encoder**: The pretrained MIST model from the DiffMS paper, trained on MassSpecGym, is adopted. Each spectral peak is annotated with a chemical formula, then encoded via a chemical formula Transformer, ultimately producing a predicted fingerprint.

**MolForge Decoder**:
- An autoregressive Transformer model that takes on-bit indices of the fingerprint as input and generates SMILES token sequences as output.
- Encoder–decoder architecture: the encoder processes on-bit index mappings; the decoder generates SMILES tokens.
- Beam search is employed to generate top-$k$ candidate molecules.

**Thresholding Strategy**:
- Fixed threshold $t=0.5$: bits with probability $\geq 0.5$ are set to 1.
- Prior-adjusted threshold $t=0.172$: computed as the proportion of on-bits in the training set (1.09%), selecting the threshold such that the on-bit ratio after binarization on the test set matches the training set prior, with no data leakage.

### Loss & Training

- MIST uses pretrained weights provided by DiffMS, requiring no additional training.
- MolForge is trained on a combined dataset of approximately 2.8 million compounds (DSSTox + HMDB + COCONUT + MOSES) using Morgan fingerprints (radius 2, 4096 bits).
- Training hyperparameters: learning rate $5\times10^{-4}$, batch size 128, 6 epochs, approximately 3 days on an Nvidia A40 GPU.
- Inference beam size = 10.

## Key Experimental Results

### Main Results: De Novo Molecule Generation Performance

| Model | Top-1 Acc ↑ | Top-1 MCES ↓ | Top-1 Tanimoto ↑ | Top-10 Acc ↑ | Top-10 MCES ↓ | Top-10 Tanimoto ↑ |
|------|------------|--------------|------------------|-------------|---------------|-------------------|
| MADGEN | 1.31% | 27.47 | 0.20 | 1.54% | 16.84 | 0.26 |
| DiffMS | 2.30% | 18.45 | 0.28 | 4.25% | 14.73 | 0.39 |
| MIST+MolForge ($t=0.5$) | 28.27% | 14.72 | 0.64 | 36.11% | 10.69 | 0.70 |
| MIST+MolForge ($t=0.172$)* | **30.97%** | **12.38** | **0.68** | **40.04%** | **8.63** | **0.74** |

### Ablation Study: MolForge Decoding Performance

| Training Dataset | Input Fingerprint | Top-1 Acc ↑ | Top-1 Tanimoto ↑ | Top-10 Acc ↑ | Top-10 Tanimoto ↑ |
|-----------|---------|------------|------------------|-------------|-------------------|
| Combined DiffMS (~3M) | MIST predicted | 30.97% | 0.68 | 40.04% | 0.74 |
| Combined DiffMS (~3M) | Ground truth | 46.00% | 0.89 | 59.28% | 0.93 |
| MassSpecGym (~17k) | Ground truth | 0.00% | 0.18 | 0.00% | 0.19 |

### Key Findings

1. **Decoder selection is critical**: Replacing the decoder with MolForge alone yields approximately a tenfold accuracy improvement (2.3% → 31%).
2. **Data scale effects are significant**: MolForge completely fails on 17k samples (0% accuracy) but achieves 31% top-1 accuracy when trained on 3M samples, consistent with data scaling laws.
3. **Thresholding strategy is effective**: The prior-adjusted threshold (0.172) outperforms the fixed threshold (0.5) by approximately 3 percentage points.
4. **The encoder is the bottleneck**: Top-10 accuracy reaches 59% with ground-truth fingerprints, indicating that fingerprint prediction quality is the primary limiting factor.
5. **MolForge's error-correction capability**: In 36% of test cases, MolForge generates structures closer to the ground truth than those implied by the input fingerprint.
6. **Peak annotation matters**: Removing chemical formula annotations from spectral peaks reduces Tanimoto similarity from 0.731 to 0.627.

## Highlights & Insights

- **Simple yet effective approach**: Rather than designing a novel model, this work achieves an order-of-magnitude improvement by judiciously combining existing models (MIST + MolForge), exemplifying the principle that "choosing the right tool matters more than building a new one."
- **On-bit index input design**: MolForge accepts only on-bit indices rather than the full fingerprint vector, making it inherently well-suited for handling sparse, noisy predicted fingerprints.
- **Prior-adjusted threshold with no data leakage**: Training set statistics are cleverly leveraged to calibrate the threshold without accessing test set labels.
- **Clear directions for future improvement**: Fingerprint prediction (the encoder) is identified as the primary bottleneck, providing actionable guidance for subsequent research.

## Limitations & Future Work

- The encoder (MIST) constitutes the bottleneck of the entire pipeline; fingerprint prediction quality directly limits final performance.
- The method relies on chemical formula annotations of spectral peaks; the quality of the automated annotation procedure affects results.
- The prior-adjusted threshold assumes similar on-bit distributions between training and test sets, which may fail under severe distribution shift.
- Evaluation is conducted solely on the MassSpecGym dataset; generalizability remains to be verified.
- MolForge training requires approximately 3 days, and scalability to larger datasets has not been explored.

## Related Work & Insights

- **DiffMS**: Previous SOTA, using a diffusion model to decode fingerprints into molecular graphs, but with limited performance (2.3%).
- **CSI:FingerID**: Generates fingerprints using fragmentation trees, but uses a different fingerprint type from Morgan fingerprints, with poor scalability.
- **MSNovelist**: Decodes fingerprints using RNN+LSTM; MolForge's Transformer architecture offers superior data scalability.
- **Insight**: In complex pipelines, bottleneck analysis and component-level optimization are often more effective than end-to-end redesign.

## Rating

- **Novelty**: ⭐⭐⭐ — No architectural innovation per se, but the combination strategy and thresholding approach demonstrate genuine insight.
- **Effectiveness**: ⭐⭐⭐⭐⭐ — Tenfold improvement; results are highly convincing.
- **Reproducibility**: ⭐⭐⭐⭐⭐ — All components are open-source; experimental details are complete.
- **Impact**: ⭐⭐⭐⭐ — Establishes a strong baseline for MS-based molecule generation and clearly delineates future directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICLR 2026\] CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints](../../ICLR2026/computational_biology/cryonetrefine_a_one-step_diffusion_model_for_rapid_refinement_of_structural_mode.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)

</div>

<!-- RELATED:END -->
