---
title: >-
  [Paper Note] Protein Fold Classification at Scale: Benchmarking and Pretraining
description: >-
  [ICML 2026][Computational Biology][Protein Fold Classification] The authors construct TEDBench, an unprecedented non-redundant protein fold classification benchmark (~490k entries…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Protein Fold Classification"
  - "Large-scale Benchmark"
  - "Masked Autoencoder"
  - "SE(3) Invariant"
  - "Self-supervised Pretraining"
date: 2026-05-08
content_hash: 452f85c605c90bea
---

# Protein Fold Classification at Scale: Benchmarking and Pretraining

**Conference**: ICML 2026  
**arXiv**: [2605.18552](https://arxiv.org/abs/2605.18552)  
**Code**: https://github.com/BorgwardtLab/TEDBench  
**Area**: Scientific Computing / Protein Structure Representation Learning  
**Keywords**: Protein Fold Classification, Large-scale Benchmark, Masked Autoencoder, SE(3) Invariant, Self-supervised Pretraining

## TL;DR
The authors construct TEDBench, an unprecedented non-redundant protein fold classification benchmark (~490k entries, 965 classes) based on AlphaFold structures clustered via TED and Foldseek. They propose MiAE, an SE(3)-invariant Masked Autoencoder using an extreme 90% mask rate and an asymmetric architecture (heavy encoder/light decoder). With only 100M parameters, it outperforms larger models like ESM2-650M and SaProt-650M in linear probing and fine-tuning.

## Background & Motivation

**Background**: Structural classification systems like CATH/SCOP organize protein domains into hierarchical labels: class → architecture → topology → homology. Traditionally, fold recognition relies on structural alignment (e.g., DALI, Foldseek) for nearest-neighbor transfer. Recently, geometric deep learning (E3NN, MACE, GotenNet) and protein representation learning (ESM2, ProteinMPNN, SaProt) treat fold recognition as a supervised classification or representation learning task.

**Limitations of Prior Work**: Supervised benchmarks for protein fold classification have long been limited to approximately 15k proteins (e.g., SCOPe, PDB-fold), suffering from high redundancy and label noise. Current methods either rely on massive sequence models (ESM2-15B) or face performance bottlenecks with structure-only models. In other words, the protein domain has yet to reach its "ImageNet moment."

**Key Challenge**: While the AlphaFold Database contains hundreds of millions of predicted structures, it lacks a large-scale, non-redundant, and reliably labeled standard classification task to drive architectural iterations. Furthermore, mainstream geometric GNNs do not scale well to hundreds of thousands of samples, while sequence-based models ignore explicit 3D structures.

**Goal**: (1) Construct a fold classification benchmark an order of magnitude larger with controlled redundancy; (2) Provide a scale-friendly, structure-only self-supervised baseline to demonstrate the sufficiency of structural representations.

**Key Insight**: MAE (He et al. 2022) in CV uses a 75% mask rate and asymmetric encoder-decoder to learn transferable visual representations. Protein backbones similarly possess local redundancy akin to a "tertiary alphabet," allowing for more aggressive masking. By applying the MAE paradigm to SE(3)-invariant frame representations, one can efficiently encode sparse visible frames and use a lightweight decoder to reconstruct coordinates from latent and mask tokens.

**Core Idea**: Represent each residue as an SE(3) local frame, utilizing a 90% mask rate with an asymmetric MAE (heavy encoder/light decoder) to learn structural representations. Simultaneously, construct TEDBench—a non-redundant dataset of 460k entries and 965 classes—based on TED + Foldseek clustering and pLDDT filtering.

## Method

The work consists of two parts: benchmark construction (TEDBench) and representation learning (MiAE).

### Overall Architecture

**TEDBench Data Flow**: AlphaFold Database (200M+) → Foldseek clustering yields ~2.27M representative proteins → Use TED (Lau et al. 2024) to decompose proteins into domains and match them to CATH topologies → Filter for mean pLDDT > 80 → Assign the CATH T-level of the largest domain as a single label → Merge T-classes with fewer than 10 samples into "A.x" format → Result: 462,175 proteins / 965 classes, with an 8:1:1 stratified split. An additional 27,638 CATH v4.4 experimental structures serve as an external test set.

**MiAE Pipeline**: Input protein backbone → Extract SE(3) frames (position + orientation) for each residue → Randomly and uniformly mask 90% of frames → Heavy encoder processes only the 10% visible frames to output latents → Re-insert mask tokens to full length → Lightweight decoder reconstructs all backbone coordinates from latents + mask tokens + RoPE → Supervised by ESM3's compound loss. After pretraining, the decoder is discarded, and the encoder is connected to a linear head or fine-tuned for downstream fold classification.

### Key Designs

1.  **Frame-based SE(3)-invariant Representation**:

    - **Function**: Encodes each residue as $\mathbf{T}_i = [\mathbf{R}_i, \mathbf{t}_i] \in \mathrm{SE}(3)$, where $\mathbf{t}_i$ is the $C_\alpha$ global coordinate and $\mathbf{R}_i$ is an orthogonal basis constructed from backbone atoms $(N, C_\alpha, C)$.
    - **Mechanism**: All attention is performed within local coordinate systems (referencing ESM3's geometric self-attention). Specifically, a global point $p$ is mapped to the local system of frame $i$ via $p_{\text{local}} = \mathbf{R}_i^\top (p - \mathbf{t}_i)$, ensuring natural invariance to rigid-body translation and rotation. This is significantly cheaper than high-order tensor products in E3NN/MACE. Unlike ESM3, this work does not restrict attention to k-nearest neighbors but performs global attention over visible frames—efficient because only 10% of residues remain after masking.
    - **Design Motivation**: Fold categories are defined by CATH based on 3D structure, necessitating structure-aware models. Frame representation preserves geometric information while avoiding the complexity of equivariant high-order tensors, allowing scaling to 339M parameters.

2.  **90% Extreme Masking + Asymmetric Architecture**:

    - **Function**: During training, 10% of frames are sampled as the visible set; the remaining 90% are **completely removed** (not replaced by mask tokens) before entering the encoder. The heavy encoder (up to 24 layers / 339M) performs geometric attention + Transformer only on the visible set. The decoder (2 layers, 512 width) reconstructs coordinates after mask tokens are inserted back into the sequence.
    - **Mechanism**: High mask rates break the shortcut of "neighbor interpolation reconstruction," forcing the model to learn global geometry rather than local smoothness. The asymmetric design drastically reduces computational costs. Ablations show that at 90% masking, reconstruction RMSD rises sharply (whereas at 70%, RMSD is only 0.57), and linear probing F1 is maximized. Changing to 0% masking (pure AE) drops linear probing F1 from 58.5 to 45.7 (test) / 23.9 (external). For the decoder, a width of 512 is optimal; for depth, mean pooling favors deeper decoders (1/2/4 layers yield F1 55→58→59), while CLS pooling is negatively affected by deeper decoders.
    - **Design Motivation**: Local redundancy in protein backbones (e.g., $\alpha$-helices, $\beta$-sheets) makes low-mask training trivial. Under high masking, the encoder must perform "long-range geometric reasoning" to produce latents effective for fold classification. Offloading heavy computation to the 10% visible set is key to scaling.

3.  **ESM3 Compound Reconstruction Loss + Optional Sequence Channel**:

    - **Function**: The training objective is $\mathcal{L}_{\text{ESM3}}$, comprising 5 terms: geometric distance, geometric orientation (primary supervision) + binned distance/orientation classification (auxiliary stability) + inverse folding token prediction (encouraging latents to retain sequence-related info). The loss applies to **all** backbone atoms, not just masked ones. Optionally, amino acid sequences are masked using the same pattern, and AA embeddings of unmasked residues are added to the visible frame representations.
    - **Mechanism**: The inverse folding loss ensures latents "guess" the AA identity, compressing functional/evolutionary information into the representation. Ablations show removing this loss drops linear probing F1 from 58.5 to 52.5. Adding the sequence channel pushes linear probing F1 to 62.1 and fine-tuning F1 to 74.6 (exceeding SaProt-650M's 73.5).
    - **Design Motivation**: Pure geometric reconstruction may learn "geometrically perfect but biologically vacuous" representations. Weak supervision from the sequence side aligns representations with the "geometry + evolution" mixed definition of CATH labels.

### Loss & Training

- Optimizer: AdamW with cosine learning rate; layer-wise lr decay for fine-tuning.
- Pretraining Data: 749,679 unlabeled structures from Foldseek clustering with pLDDT > 80 (no overlap with TEDBench supervised set).
- Model Sizes: MiAE-S (29M / 6 layers), MiAE-B (102M / 12 layers), MiAE-L (339M / 24 layers).
- Metrics: Due to the long-tail nature of the 965 classes, macro-F1 is the primary metric (accuracy also reported); external test set uses CATH v4.4 40% non-redundant experimental structures.

## Key Experimental Results

### Main Results

| Protocol | Model | Params | test F1 | external F1 | Remarks |
|----------|-------|--------|---------|-------------|---------|
| Supervised-from-scratch | GotenNet | 1.9M | 64.02 | 65.44 | Strongest equivariant baseline |
| Supervised-from-scratch | E3NN | 1.9M | 57.63 | 42.40 | Significant drop on external test |
| Supervised-from-scratch | MACE | 1.5M | 50.58 | 44.73 | — |
| Supervised-from-scratch | **MiAE-B** | 102M | **71.60** | **75.02** | +7.6 / +9.6 over GotenNet |
| Pretrain + Fine-tune | ESM2-650M | 650M | 66.19 | 72.29 | Sequence-only LLM |
| Pretrain + Fine-tune | SaProt-650M | 650M | 73.48 | 76.78 | Mixed Seq+Str SOTA |
| Pretrain + Fine-tune | **MiAE-B+seq** | 102M | **74.56** | **77.34** | Outperforms SaProt-650M with 6.4× fewer params |
| Linear Probing | ESM2-15B | 15B | 70.85 | 76.27 | Strongest but 44× more params |
| Linear Probing | MiAE-L | 339M | 63.50 | 70.44 | Strongest structure model ≤650M |

### Ablation Study (MiAE-B default, Linear Probing F1, test/external)

| Configuration | test F1 | external F1 | Note |
|---------------|---------|-------------|------|
| Default (90% mask + invf + seq + dec 2L×512) | 62.14 | 68.88 | — |
| 0% Mask (Pure AE) | 45.70 | 23.90 | Lacks sparse challenge, drops 16.4 / 45.0 |
| Remove invf loss | 52.55 | — | Seq-level supervision is vital, drops 6.0 |
| Without AA sequence | 58.52 | 66.18 | Seq channel contributes ~3.6 / 2.7 |
| Decoder width 256 / 768 | 35.50 / 27.83 | — | Significant degradation away from 512 |
| Decoder depth 1L / 2L / 4L (mean pool) | 46.61 / 58.52 / 59.65 | — | Deeper is better (under mean pool) |
| Model S / B / L | 49.43 / 58.52 / 63.50 | — | Clean scaling in linear probing |

### Key Findings

- **Higher masking rate is superior, opposite to CV's MAE trend**: While MAE is optimal at 75% for images and 15% for BERT, proteins require **90%** for optimality. This is due to extreme local redundancy; at 70% masking, reconstruction RMSD is only 0.57, allowing the model to rely on local interpolation. 90% forces "global geometric reasoning."
- **Structure > Sequence+Structure > Sequence**: Structure-only MiAE significantly outperforms sequence-only ESM2 at similar parameter budgets and matches SaProt-650M with 1/6 parameters. For CATH topology (geometrically defined), structural signals are sufficient; sequence is "nice-to-have."
- **MiAE benefits more from fine-tuning than SaProt/ESM2**: MiAE-B+seq jumps 12.5/8.5 points from linear probing to fine-tuning, whereas ESM2 jumps only 4/2 and SaProt 7/6. This suggests MiAE's pretraining objectives are better aligned with downstream fold classification.
- **Scaling is clean in linear probing but saturates in fine-tuning**: Linear probing improves by 14 F1 points from S to L, but fine-tuning B to L is almost flat. This suggests 102M is sufficient for the downstream task; future progress lies in larger pretraining data rather than model size.
- **External test set performance is higher**: All models perform ~10 points better on experimental structures (CATH v4.4) than on AFDB predicted structures. The authors attribute this to lower diversity in experimental structures and cleaner manual CATH labels.

## Highlights & Insights

- **Path to "Protein's ImageNet Moment"**: Combining TED + Foldseek clustering + pLDDT filtering into a reproducible pipeline creates a 490k-scale, non-redundant, single-label fold classification benchmark. This infrastructure is arguably more important than the method itself.
- **Optimal mask rate reflects modal redundancy**: The values (Image 75%, Text 15%, Protein backbone 90%) reflect the "compressibility" of each modality. This provides a diagnostic metric for MAE in new modalities: measure reconstruction RMSDs against mask rates to find the "steep rise" point.
- **Asymmetric design + SE(3)-invariant frames as a practical solution**: By avoiding high-order equivariant tensor products and using "local system attention," the model achieves scalability, offering a more engineering-friendly alternative to equivariant GNNs.
- **Transferable Tricks**: (a) High masking to break local shortcuts can be applied to other redundant geometric modalities like point clouds; (b) Synchronous auxiliary supervision via sequence/text channels can be reused in multi-modal bio-tasks.

## Limitations & Future Work

- The authors acknowledge: (1) TEDBench only performs protein-level fold recognition (largest domain label), missing smaller domains; future work should involve domain-level segmentation. (2) MiAE was only validated on fold classification. (3) Pretraining large models remains expensive.
- Observations: (a) The single-label setup discards CATH's hierarchical structure (class→architecture→topology→homology); incorporating hierarchical loss would be more comprehensive. (b) The domain gap between AFDB and experimental structures hasn't been fully stress-tested. (c) Linear probing for MiAE-L still lags behind ESM2-15B, suggesting structure-only models struggle in "weak-label-correlation regions." (d) Decoder width sensitivity (F1 35→58→28) suggests geometric MAE has brittle hyperparameters.

## Related Work & Insights

- **vs ESM2 / SaProt**: ESM2 uses sequence only; SaProt adds discrete structural tokens to sequences. MiAE prioritizes continuous 3D structure. This work proves structural signals are significantly stronger for structure-defined tasks (outperforming SaProt-650M with 6.4× fewer params).
- **vs ProteinMPNN / MIF**: Both use inverse folding but with smaller models (1.6M-3.4M). MiAE integrates inverse folding as one of five losses within an MAE framework, benefiting from MAE scaling while retaining sequence signals.
- **vs CV's MAE**: Architecturally similar, but MiAE's loss applies to **all** atoms (to ensure SE(3) invariance) rather than just masked regions, and the mask rate is pushed to 90%.
- **vs Equivariant GNNs**: Standard equivariant GNNs saturate at 1.5M-1.9M. MiAE bypasses the cost of high-order tensor products, scaling to 339M. It suggests that "scaling up Transformers + geometric priors" is more cost-effective than "strict high-order equivariance + small models" for protein tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic transfer of MAE to protein geometry and construction of a large-scale benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three training protocols, multiple baselines, six major ablations, external test sets, and visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and high information density.
- Value: ⭐⭐⭐⭐⭐ TEDBench is likely to become a standard benchmark for protein fold classification, and MiAE provides a scale-friendly baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Investigating Data Pruning for Pretraining Biological Foundation Models at Scale](../../AAAI2026/computational_biology/investigating_data_pruning_for_pretraining_biological_foundation_models_at_scale.md)
- [\[ICML 2026\] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)
- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](../../ICLR2026/computational_biology/heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](../../NeurIPS2025/computational_biology/understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)

</div>

<!-- RELATED:END -->
