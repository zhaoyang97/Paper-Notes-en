---
title: >-
  [Paper Note] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation
description: >-
  [NeurIPS 2025][Image Restoration][Mass spectrometry] This paper proposes MS-BART, which maps molecular fingerprints and molecular structures (SELFIES) into a shared token space via a unified vocabulary…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "Mass spectrometry"
  - "molecular fingerprints"
  - "pretrain-finetune-align"
  - "BART"
  - "structure elucidation"
date: 2026-05-08
content_hash: cdfc64d3e39f506c
---

# MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation

**Conference**: NeurIPS 2025
**arXiv**: [2510.20615](https://arxiv.org/abs/2510.20615)  
**Code**: [Available](https://github.com/OpenDFM/MS-BART)  
**Area**: Scientific AI / Molecular Structure Elucidation
**Keywords**: Mass spectrometry, molecular fingerprints, pretrain-finetune-align, BART, structure elucidation

## TL;DR

This paper proposes MS-BART, which maps molecular fingerprints and molecular structures (SELFIES) into a shared token space via a unified vocabulary, performs multi-task pretraining on 4 million fingerprint–molecule pairs, and subsequently applies experimental spectra fine-tuning and chemical feedback alignment to enable efficient generation of molecular structures from mass spectra.

## Background & Motivation

Inferring molecular structures from mass spectrometry (MS) data is a central task in analytical chemistry, with broad applications in drug discovery, environmental biochemistry, and materials science. Existing approaches face two major bottlenecks:

**Scarcity of annotated spectral data**: Acquiring high-quality experimental mass spectra paired with molecular structures is costly, limiting the training of data-driven models.

**Complexity and variability of MS signals**: Spectra of the same molecule can differ substantially under different collision energies, adduct types, or instrument settings (as illustrated in Figure 1), and fluctuations occur even under identical conditions. This makes direct pretraining on raw spectra extremely challenging.

**Key Insight**: Molecular fingerprints are abstract encodings of mass spectral information that indicate the presence of chemical substructures. Unlike raw spectra, fingerprints are invariant to experimental conditions and can be reliably computed from molecular structures via RDKit, providing a pathway to large-scale pretraining data.

Existing methods (e.g., Spec2Mol, MSNovelist, DiffMS) typically treat spectra and molecular structures as independent modalities, which tends to produce **molecular hallucinations**—generating chemically valid but structurally incorrect molecules.

## Method

### Overall Architecture

MS-BART follows the **pretrain → fine-tune → align** paradigm from NLP:
1. **Pretraining**: Multi-task learning on 4 million computed fingerprint–molecule pairs
2. **Fine-tuning**: Adaptation to real distributions using experimental MS data
3. **Alignment**: Chemical feedback-guided training to reduce molecular hallucinations

### Key Designs

1. **Unified Vocabulary**

   **Function**: Maps mass spectra (represented as fingerprints) and molecular structures into a shared token space.

   **Mechanism**:
   - **Fingerprint tokens**: Each activated bit $FP_i = 1$ in a 4096-bit Morgan fingerprint is converted to a token `<fp{i:04d}>` (e.g., `<fp0123>`)
   - **Molecular tokens**: Molecular structures are encoded using 185 SELFIES tokens, guaranteeing chemical validity
   - A special separator token `<fps_sep>` connects the two modalities

   **Design Motivation**: A unified vocabulary enables the model to jointly learn spectral and molecular representations within the same sequence space, facilitating genuine cross-modal learning.

2. **Multi-Task Pretraining**

   **Function**: Learns representations of fingerprints and molecules through four self-supervised / cross-modal tasks.

   **Mechanism**:
   - **SELFIES denoising**: Randomly masks 30% of SELFIES tokens and recovers them
   - **Fingerprint-to-molecule translation**: Generates complete SELFIES from fingerprint tokens
   - **Hybrid denoising**: Concatenates fingerprints and masked SELFIES in varying orders to predict complete SELFIES

   All tasks employ a unified cross-entropy loss: $\mathcal{L}_{ce} = -\sum_i \log P(y_i | y_{<i}, X; \theta)$

   **Design Motivation**: Denoising alone is poorly aligned with structure elucidation; the translation task provides the core cross-modal capability; the hybrid task combines the advantages of both.

3. **Chemical Feedback Alignment (Contrastive Alignment)**

   **Function**: Uses a ranking loss to guide the model to assign higher probabilities to structurally more similar molecules.

   **Mechanism**: Given a fingerprint, $n$ candidate molecules are generated and scored by Tanimoto similarity as preference signals. A contrastive ranking loss is introduced:

   $\mathcal{L}_{rank}(C) = \sum_i \sum_{j>i} \max(0, P_\theta(S_j) - P_\theta(S_i) + \gamma_{ij})$

   where $\gamma_{ij} = (j-i) \times \gamma$ is the margin. The total loss is $\mathcal{L} = \mathcal{L}_{ce} + \alpha \mathcal{L}_{rank}$.

   **Design Motivation**: Models trained with cross-entropy alone may assign similar probabilities to all chemically valid—but not necessarily correct—molecules. The ranking loss compels the model to distinguish molecules that are "closer to correct."

### Loss & Training

- Backbone: BART-Base, all parameters initialized from scratch
- Pretraining: Maximum sequence length 512, 4 million unannotated molecules
- Fine-tuning / Alignment: Input and output token length 256
- Encoder is frozen during the alignment stage; only the decoder is updated
- The MIST model converts experimental spectra to predicted fingerprints, with threshold $\epsilon = 0.2$ (NPLIB1) / $0.11$ (MassSpecGym)

## Key Experimental Results

### Main Results

Comparison with multiple baselines on two public benchmarks: NPLIB1 and MassSpecGym.

| Dataset | Method | Top-1 Acc | Top-1 MCES↓ | Top-1 Tanimoto↑ | Top-10 Acc | Top-10 MCES↓ | Top-10 Tanimoto↑ |
|--------|------|-----------|-------------|-----------------|------------|--------------|------------------|
| NPLIB1 | DiffMS | 8.34% | 11.95 | 0.35 | 15.44% | 9.23 | 0.47 |
| NPLIB1 | **MS-Bart** | 7.45% | **9.66** | **0.44** | 10.99% | **8.31** | **0.51** |
| NPLIB1 | MS-Bart (Gold FP) | 73.50% | 2.14 | 0.90 | 79.12% | 1.60 | 0.94 |
| MassSpecGym | DiffMS | **2.30%** | 18.45 | **0.28** | **4.25%** | 14.73 | **0.39** |
| MassSpecGym | **MS-Bart** | 1.07% | **16.47** | 0.23 | 1.11% | **15.12** | 0.28 |

MS-BART achieves comprehensive improvements on similarity metrics on NPLIB1: MCES improves by 19.16% and Tanimoto by 25.71%.

### Ablation Study

| Pretraining Strategy | Top-1 Acc | Top-1 MCES↓ | Top-1 Tanimoto↑ |
|-----------|-----------|-------------|-----------------|
| No pretraining | 1.71% | 12.93 | 0.27 |
| Denoising only (Sd) | 0.37% | 14.41 | 0.24 |
| Translation only (Trans) | 6.23% | 9.37 | 0.42 |
| Hybrid denoising (Hybrid) | 5.13% | 9.96 | 0.41 |
| **Full MS-Bart** | **7.45%** | **9.66** | **0.44** |

### Key Findings

- **Pretraining is critical**: Performance without pretraining is substantially lower than the full model, yet still surpasses baselines that directly encode raw spectra
- **Multi-task complementarity**: Denoising alone degrades performance (misaligned with structure elucidation); the translation task contributes most; combining both yields the best results
- **Chemical feedback is effective**: Across Pretrain → Pretrain-FT → Pretrain-FT-Rank, Top-1 accuracy improves from 0% → 1.07% → final value, with Tanimoto similarity consistently increasing
- **Gold Fingerprint experiments** reveal substantial headroom: accuracy reaches 73.5% when ground-truth fingerprints are used as input, indicating that the bottleneck lies in MIST's fingerprint prediction quality
- Inference speed is an order of magnitude faster than DiffMS (a diffusion-based model)

## Highlights & Insights

- The **unified vocabulary design** is elegant: it integrates the discretized spectral representation (fingerprint tokens) and the string-based molecular representation (SELFIES tokens) into a single sequence space
- The use of **fingerprints as a bridge** is a clever design choice that avoids the complexity of directly modeling spectra across varying experimental conditions
- The paper naturally transfers the mature NLP paradigm of "pretrain → fine-tune → align" to the molecular domain
- The Gold Fingerprint experiment clearly identifies the direction for future improvement: enhancing the spectra-to-fingerprint prediction model

## Limitations & Future Work

- The current bottleneck lies in MIST's fingerprint prediction quality; MS-BART's molecular generation capability is already strong
- Failure to filter [M+Na]+ test data on MassSpecGym leads to some performance degradation
- Pretraining data excludes molecules with MCES distance < 2 from the test set (stricter than DiffMS), resulting in lower exact-match metrics
- The SELFIES vocabulary of only 185 tokens may limit the expressiveness of complex molecules

## Related Work & Insights

- The pretraining → fine-tuning paradigm is analogous to that used in NMR (Yao et al. 2023), but MS-BART introduces fingerprint abstraction to address the unique challenges of mass spectrometry
- DiffMS employs implicit fingerprint representations combined with discrete diffusion models—a different but complementary approach
- The CSI:FingerID → MSNovelist pipeline shares conceptual similarities with the MIST → unified model approach of MS-BART, but MS-BART acquires stronger molecular generation priors through large-scale pretraining

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-stage design of unified vocabulary + multi-task pretraining + chemical feedback alignment is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, extensive ablations, and Gold FP analysis are convincing
- Writing Quality: ⭐⭐⭐⭐ — Clear structure; the transfer of the NLP paradigm to chemistry is well articulated
- Value: ⭐⭐⭐⭐ — Provides a scalable pretraining paradigm for mass spectrometry-based structure elucidation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](../../ICLR2026/image_restoration/soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [\[ICCV 2025\] UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control](../../ICCV2025/image_restoration/uniphys_unified_planner_and_controller_with_diffusion_for_flexible_physics-based.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](../../CVPR2026/image_restoration/rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)

</div>

<!-- RELATED:END -->
