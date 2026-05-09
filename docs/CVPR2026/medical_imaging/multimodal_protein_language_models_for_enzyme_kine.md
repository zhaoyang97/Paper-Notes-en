---
title: >-
  [Paper Note] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation
description: >-
  [CVPR 2026][Medical Imaging][enzyme kinetic parameter prediction] This paper proposes ERBA (Enzyme-Reaction Bridging Adapter), which reformulates enzyme kinetic parameter prediction as a staged conditioning problem aligned with catalytic mechanisms — first injecting substrate information via MRCA to capture molecular recognition, then fusing active-site 3D geometry via G-MoE to model conformational adaptation, and applying ESDA for distribution alignment to preserve PLM priors — achieving state-of-the-art performance across three kinetic metrics.
tags:
  - CVPR 2026
  - Medical Imaging
  - enzyme kinetic parameter prediction
  - protein language model
  - cross-attention
  - mixture of experts
  - distribution alignment
date: 2026-05-08
content_hash: 0bd66008db68f87b
---

# Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation

**Conference**: CVPR 2026
**arXiv**: [2603.12845](https://arxiv.org/abs/2603.12845)
**Code**: None
**Area**: Medical Imaging / Protein Language Models / Bioinformatics
**Keywords**: enzyme kinetic parameter prediction, protein language model, cross-attention, mixture of experts, distribution alignment

## TL;DR
This paper proposes ERBA (Enzyme-Reaction Bridging Adapter), which reformulates enzyme kinetic parameter prediction as a staged conditioning problem aligned with catalytic mechanisms — first injecting substrate information via MRCA to capture molecular recognition, then fusing active-site 3D geometry via G-MoE to model conformational adaptation, and applying ESDA for distribution alignment to preserve PLM priors — achieving state-of-the-art performance across three kinetic metrics.

## Background & Motivation
Accurate prediction of enzyme kinetic parameters (e.g., turnover number $k_{cat}$, Michaelis constant $K_m$, inhibition constant $K_i$) is critical for high-throughput protein design and synthetic biology, enabling candidate enzyme screening prior to wet-lab experiments. Existing methods such as DLKcat, UniKP, CataPro, and CatPred incorporate protein language model (PLM) features and substrate SMILES encodings, but share two fundamental limitations:

1. **Shallow fusion**: Most methods simply concatenate enzyme and substrate representations before regression, treating catalysis as a static compatibility problem and ignoring its staged nature (substrate recognition followed by conformational adaptation).
2. **Passive use of PLMs**: PLMs serve merely as frozen feature extractors or lightly fine-tuned backbones, without being explicitly conditioned on specific substrates or pocket geometry. Furthermore, directly injecting 3D structural information risks disrupting the biochemical semantics learned during PLM pre-training.

## Core Problem
How to systematically inject substrate chemical information and active-site 3D structural information when fine-tuning PLMs for enzyme kinetic prediction, while preserving the biochemical priors acquired through large-scale self-supervised pre-training? The core challenge lies in determining the order, mechanism, and stability of fusion.

## Method

### Overall Architecture
ERBA's pipeline emulates the two stages of real catalytic mechanisms:
- **Input**: enzyme amino acid sequence $\mathbf{S}_e$, substrate SMILES $\mathbf{S}_m$, active-site 3D structure $\mathbf{S}_g$
- **Stage 1 — Molecular Recognition**: substrate semantics are injected into the PLM enzyme representation via MRCA
- **Stage 2 — Conformational Adaptation**: active-site geometric information is fused via G-MoE, routing to pocket-specialized experts
- **Distribution Alignment**: ESDA constrains representations at each stage to remain consistent with the PLM manifold
- **Output**: kinetic parameters predicted in $\log_{10}$ space using a heteroscedastic Gaussian NLL objective

The overall formulation is $\hat{\mathbf{y}} = \mathcal{G}^{(2)}(\mathcal{M}^{(1)}(\mathbf{S}_e, \mathbf{S}_m), \mathbf{S}_g)$.

### Key Designs

1. **MRCA (Molecular Recognition Cross-Attention)**: The PLM's shallow layers produce residue embeddings $\mathbf{H}_e \in \mathbb{R}^{L_e \times D}$; the substrate is encoded by an MPNN into $\mathbf{H}_m \in \mathbb{R}^{L_m \times D}$. MRCA injects substrate semantics into the enzyme representation via a single-layer cross-attention mechanism, using enzyme tokens as queries and substrate tokens as keys/values. This emulates the process by which the enzyme first recognizes and localizes the substrate. Compared to simple concatenation followed by self-attention, MRCA improves $R^2$ on $K_i$ from 0.47 to 0.61.

2. **G-MoE (Geometry-aware Mixture-of-Experts)**: A geometry-aware mixture-of-experts layer is designed in which the routing mechanism jointly considers the pocket-region representation from MRCA output and 3D geometric descriptors encoded by an E-GNN. Top-$k$ sparse gating selects 2 out of 4 experts. Each expert performs pocket-local low-rank adaptation (analogous to LoRA), with geometric information modulating channel activations via bias terms. This routes enzyme–substrate pairs of different conformational types to different experts, emulating the induced-fit effect in catalysis. Compared to standard MoE, $R^2$ improves from 0.50 to 0.61.

3. **ESDA (Enzyme-Substrate Distribution Alignment)**: Three levels of representations are defined — original sequence representation $\mathcal{Z}^{(0)}$, substrate-conditioned $\mathcal{Z}^{(1)}$, and geometry-conditioned $\mathcal{Z}^{(2)}$. MMD (Maximum Mean Discrepancy) with an RBF kernel constrains $\mathcal{Z}^{(1)}$ and $\mathcal{Z}^{(2)}$ to remain close in distribution to $\mathcal{Z}^{(0)}$, preventing "feature flooding" and PLM semantic forgetting during multimodal information injection. Compared to a standard L2 loss, $R^2$ improves from 0.48 to 0.61.

### Loss & Training
- **Primary task loss**: heteroscedastic Gaussian NLL in $\log_{10}$ space (jointly predicting mean and variance), suited for the positive-value constraints and multiplicative noise of kinetic parameters
- **G-MoE balance loss** $\mathcal{L}_\text{G-MoE}$: encourages uniform expert utilization to prevent expert collapse
- **ESDA alignment loss** $\mathcal{L}_\text{ESDA}$: MMD-based distribution alignment
- Total loss: $\mathcal{L} = \mathcal{L}_\text{task} + 0.01 \cdot \mathcal{L}_\text{G-MoE} + 0.1 \cdot \mathcal{L}_\text{ESDA}$
- Parameter-efficient fine-tuning of the PLM's top layers via LoRA (rank=8, scaling=16)

## Key Experimental Results

| Dataset / Metric | Metric | ERBA (Ours) | CatPred (Prev. SOTA) | Gain |
|-----------------|--------|-------------|----------------------|------|
| $k_{cat}$ | R² | 0.54 | 0.40 | +0.14 |
| $k_{cat}$ | PCC | 0.74 | 0.67 | +0.07 |
| $k_{cat}$ | RMSE | 1.13 | 1.30 | -0.17 |
| $K_m$ | R² | 0.61 | 0.49 | +0.12 |
| $K_m$ | PCC | 0.79 | 0.65 | +0.14 |
| $K_m$ | RMSE | 0.70 | 0.93 | -0.23 |
| $K_i$ | R² | 0.61 | 0.45 | +0.16 |
| $K_i$ | PCC | 0.78 | 0.60 | +0.18 |
| OOD $k_{cat}$ | R² | 0.50 | 0.25 (CatPred) | +0.25 |
| OOD $K_m$ | R² | 0.55 | 0.30 (CatPred) | +0.25 |

**Backbone scaling**: Applying ERBA to ESM2 (8M→3B), ProtT5-3B, and Ankh3 (1.8B/5.7B) consistently yields improvements, validating the method's generality.

### Ablation Study
- **Fusion order matters**: the substrate-first-then-structure order ($\mathbf{S}_e \to \mathbf{S}_m \to \mathbf{S}_g$) substantially outperforms the reverse, improving $R^2$ by 37.8%
- **MRCA vs. concatenation + self-attention**: MRCA improves $R^2$ by 29.8%, demonstrating the superiority of explicit cross-attention over shallow fusion
- **G-MoE vs. standard MoE**: geometry-aware routing improves $R^2$ by 22% over standard routing, confirming the effectiveness of routing based on pocket morphology
- **ESDA vs. L2**: distribution alignment improves $R^2$ by 27.1%, validating the importance of maintaining PLM manifold consistency
- **Module-wise accumulation**: PLM baseline (0.49) → +MRCA (0.51) → +G-MoE (0.54) → +ESDA (0.61), with each module contributing independently

## Highlights & Insights
- **Mechanism-aligned problem formulation**: mapping the two-stage enzymatic "recognition → adaptation" process directly onto model design (MRCA → G-MoE) is both biologically grounded and architecturally elegant
- **ESDA's distribution alignment perspective**: using MMD in RKHS to prevent semantic drift during multimodal fine-tuning is a general and transferable idea applicable to any multimodal fine-tuning scenario
- **Geometry-guided routing in G-MoE**: routing based on 3D pocket structure, allowing experts specialized for different active-site morphologies, aligns well with the biological intuition of enzyme catalysis
- **Substantial OOD generalization**: $R^2$ nearly doubles on the EITLEM test set (0.25 → 0.50), demonstrating that conditioned PLMs generalize far more robustly than passively used PLMs

## Limitations & Future Work
- The current work focuses exclusively on wild-type enzymes, with no systematic evaluation of mutants (though the OOD set includes some)
- 3D structures rely on AlphaFold2/ESMFold predictions, which may deviate from experimental structures
- Environmental variables such as cofactors, pH, and temperature are not modeled
- Performance on EC-6 (ligase) remains insufficient ($K_m$ $R^2$ of only 0.34), potentially requiring domain-specific data augmentation
- Inference efficiency with ESM2-3B as the backbone warrants further optimization

## Related Work & Insights
- **vs. CatPred (Nat.C 2025)**: CatPred also incorporates 3D structures but employs shallow fusion; ERBA's staged conditioning substantially surpasses it, with the gap widening further under OOD evaluation
- **vs. EITLEM (Chem.Catal 2024)**: EITLEM uses ESM-1v with residue-level attention for mutant modeling but lacks explicit substrate conditioning and structural fusion
- **vs. UniKP/CataPro**: these methods perform multi-endpoint prediction with ProtT5 and SMILES but are fundamentally based on shallow concatenation
- The ESDA distribution alignment strategy is transferable to multimodal medical image analysis — preserving pre-trained semantics when fine-tuning visual foundation models
- The staged modeling paradigm of "recognition then adaptation" is broadly applicable to any task involving two-stage decision processes
- Geometry-aware, domain-sensitive routing in MoE (routing based on structural properties of the input) constitutes a general and reusable design pattern

## Rating
- **Novelty**: ⭐⭐⭐⭐ The mechanism-aligned staged conditioning has solid biological grounding and ESDA is an elegant contribution, though individual components are not entirely novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three endpoints × multiple PLM backbones × OOD evaluation × six EC-class analyses × comprehensive ablations — coverage is exceptional
- **Writing Quality**: ⭐⭐⭐⭐ The mathematical framework is complete and biological context is well-explained, though the paper is somewhat lengthy
- **Value**: ⭐⭐⭐⭐ Practically valuable for computational biology; the distribution alignment idea transfers readily to other multimodal tasks

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)

</div>

<!-- RELATED:END -->
