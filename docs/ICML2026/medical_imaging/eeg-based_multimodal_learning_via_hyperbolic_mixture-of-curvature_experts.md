---
title: >-
  [Paper Note] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts
description: >-
  [ICML 2026][Medical Imaging][EEG] EEG-MoCE assigns each modality in EEG-based multimodal learning (emotion/sleep/cognition) a Lorentz manifold expert with **learnable curvature**…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "EEG"
  - "Mixture-of-Curvature"
  - "Lorentz Manifold"
  - "Cross-Subject Generalization"
  - "δ-hyperbolicity"
date: 2026-05-08
content_hash: 3cb86950050de1ce
---

# EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts

**Conference**: ICML 2026  
**arXiv**: [2604.12579](https://arxiv.org/abs/2604.12579)  
**Code**: Mentioned in the paper as "Code will be released", not yet public  
**Area**: Medical Imaging / Brain-Computer Interface / Multimodal Learning / Hyperbolic Geometry  
**Keywords**: EEG, Mixture-of-Curvature, Lorentz Manifold, Cross-Subject Generalization, δ-hyperbolicity

## TL;DR
EEG-MoCE assigns each modality in EEG-based multimodal learning (emotion/sleep/cognition) a Lorentz manifold expert with **learnable curvature**, then uses curvature-aware attention for cross-modal fusion, where "higher curvature → richer hierarchy → higher fusion weight". On EAV/ISRUC/Cognitive datasets, cross-subject accuracy improves by +14.14%, +3.34%, and +7.98%, respectively.

## Background & Motivation

**Background**: Isolated EEG signals are highly susceptible to electrophysiological noise and subject variability. Thus, recent work increasingly combines EEG with other modalities such as video (facial expression), audio, EMG/EOG/NIRS for multimodal learning, enhancing robustness in emotion recognition, sleep staging, and cognitive load assessment. The mainstream approach remains Euclidean architectures (CNN+Transformer+Cross-modal attention).

**Limitations of Prior Work**: (1) Both EEG and brain-related modalities are known to have **hierarchical organization** (e.g., emotion processing from subcortical to limbic to neocortex; frequency bands are also hierarchical); (2) Euclidean embeddings, due to linear/quadratic growth of distance/volume, cannot accommodate exponentially expanding hierarchies; (3) Existing hyperbolic EEG work (HEEGNet) uses **fixed curvature** and only for unimodal EEG, while in multimodal settings, the "hierarchy strength" varies greatly across modalities but is treated uniformly.

**Key Challenge**: The inherent hierarchical complexity differs across modalities (quantified by δ-hyperbolicity: EEG δ_rel≈0.10, audio ≈0.22, video ≈0.28). Using the same curvature or Euclidean space for all is suboptimal. To make "adaptive curvature" effective during fusion, a mechanism is needed to inform the fusion layer "which modality is more reliable at this moment".

**Goal**: (i) Assign each modality its own Lorentz manifold with learnable curvature; (ii) Explicitly leverage learned curvature during fusion, giving higher weight to modalities with richer hierarchical information.

**Key Insight**: The authors' crucial observation—theoretically, the larger the absolute curvature $|K|$, the deeper the hierarchy that can be embedded with less distortion at fixed dimension (Sala et al., 2018). Thus, if a modality's $|K|$ is large after end-to-end training, it indicates it encodes more hierarchical information, and this $|K|$ can be used as a fusion weight.

**Core Idea**: Mixture-of-Curvature experts + curvature-aware cross-modal attention (making $|K|$ determine both single-modality geometry and fusion weight).

## Method

### Overall Architecture
The model $h_\Theta=g_\psi\circ F_\omega\circ(\bigoplus_{m\in\mathcal{M}}E_\phi^{(m)}\circ e_\theta^{(m)})$ consists of four components:

- **Euclidean encoder** $e_\theta^{(m)}$: Each modality uses its own backbone (EEG uses EEGNet, EMG/EOG use EEGNet variants, video uses lightweight CNN+Temporal Transformer, audio uses 1D CNN on mel-spectrogram + Temporal Transformer), outputting $\mathbf{x}^{(m)}\in\mathbb{R}^d$.
- **Hyperbolic expert** $E_\phi^{(m)}$: Projects $\mathbf{x}^{(m)}$ onto the modality-specific Lorentz manifold $\mathcal{L}_{K^{(m)}}^d$ (learnable curvature $K^{(m)}<0$), followed by Lorentz BN (with moments alignment for cross-subject), Lorentz activation/pooling, outputting $\mathbf{z}^{(m)}\in\mathcal{L}_{K^{(m)}}^d$.
- **Curvature-oriented fusion** $F_\omega$: Projects all $\mathbf{z}^{(m)}$ onto a shared fusion manifold $\mathcal{L}_{K_f}^d$ ($K_f$ = mean curvature across modalities), stacks multiple curvature-guided cross-attention layers, and aggregates via weighted Fréchet mean.
- **Hyperbolic classifier** $g_\psi$: Lorentz multinomial logistic regression (HMLR), using geodesic hyperplanes as decision boundaries.

### Key Designs

1. **Mixture-of-Curvature Experts (one learnable-curvature Lorentz manifold per modality)**:

    - **Function**: Allows each modality to have its own geometric space, avoiding under-representation of highly hierarchical modalities or over-parameterization of low-hierarchy ones due to shared curvature.
    - **Mechanism**: Each modality $m$ has a learnable curvature $K^{(m)}<0$. Euclidean features $\mathbf{x}^{(m)}$ are mapped via exponential map $\mathbf{h}^{(m)}=\exp_\mathbf{o}^{K^{(m)}}(\mathbf{x}^{(m)})$ to the modality's Lorentz hyperboloid ($\mathbf{o}=[\sqrt{-1/K^{(m)}},\mathbf{0}]^\top$ is the origin). Subsequent operations (BN, activation, attention) are performed on the Lorentz manifold.
    - **Design Motivation**: The authors use δ-hyperbolicity to quantify each modality's hierarchy strength, finding significant differences (EEG≈0.10, audio≈0.22, video≈0.28, NIRS≈0.30; see Table 1). Fixed curvature inevitably leads to over- or under-fitting for some modalities; learnable curvature allows the model to **automatically converge to suitable $|K|$ for each modality**—experiments show learned EEG $|K|=2.34 >$ Vision 2.29 > Audio 1.91, perfectly anti-correlated with δ_rel.

2. **Curvature-guided cross-modal attention (using curvature to modulate temperature and prior bias)**:

    - **Function**: Ensures that "hierarchically rich modalities" are both more selective in cross-modal attention and more likely to be selected by other modalities.
    - **Mechanism**: All modalities are projected onto the shared fusion manifold ($\mathbf{z}_f^{(m)}=\exp_\mathbf{o}^{K_f}(\sqrt{K^{(m)}/K_f}\cdot\log_\mathbf{o}^{K^{(m)}}(\mathbf{z}^{(m)}))$), preserving hyperbolic geometry. Attention uses negative squared geodesic distance as similarity (replacing dot product), with two curvature-coupled mechanisms: (i) Temperature $\tau^{(m)}=\tau_0/\sqrt{|K^{(m)}|}$—higher $|K|$ yields lower temperature and sharper attention; (ii) Adds prior bias $\lambda\cdot\phi(K^{(j)})$ ($\phi(K)=\log(|K|+\epsilon)$), making queries favor keys with larger $|K|$: $\tilde{\alpha}_{m\to j}\propto\exp(-d_{\mathcal{L}}^2(\mathbf{q}^{(m)},\mathbf{k}^{(j)})/\tau^{(m)}+\lambda\cdot\phi(K^{(j)}))$. Aggregation uses weighted Fréchet mean (the "weighted average" in hyperbolic space).
    - **Design Motivation**: Curvature is treated as a **learnable indicator of modality information content**. Temperature adjustment enables strong modalities to perform more precise cross-modal queries; prior bias ensures weak modalities can still be aggregated by strong ones. Coupling both, attention's hierarchical preference is upgraded from "feature similarity" to "geometric complexity + similarity". Table 2 shows EAV EEG attention contribution 36% > Vision 33.6% > Audio 30.5%, matching the $|K|$ ranking.

3. **Full-stack hyperbolic processing + cross-subject normalization**:

    - **Function**: All computations from encoder to classifier are performed on the Lorentz manifold, avoiding hierarchical distortion in Euclidean space; hyperbolic BN with moments alignment addresses cross-subject distribution shift.
    - **Mechanism**: (i) Lorentz fully-connected $f_\mathcal{L}(\mathbf{p})=(\sqrt{\|\tilde{\mathbf{p}}_s\|^2-1/K},\tilde{\mathbf{p}}_s)$, $\tilde{\mathbf{p}}_s=\psi(\mathbf{Wp}+\mathbf{b})$, ensuring outputs remain on the manifold; (ii) Lorentz BN adopts moments alignment from HEEGNet, aligning feature statistics across subjects to a shared center; (iii) Classification uses HMLR, defining geodesic hyperplanes as class boundaries.
    - **Design Motivation**: The authors emphasize "compositional design"—Euclidean encoders are suitable for learning local time-frequency features, while hyperbolic components model hierarchy and cross-modal fusion, with exp map bridging the two. The Lorentz model uses the hyperboloid rather than the Poincaré ball for more stable gradient optimization.

### Loss & Training

- Classification loss plus auxiliary terms (hyperparameters in appendix), 100 epochs; Euclidean parameters optimized with Adam, hyperbolic parameters with Riemannian Adam; lr=1e-3, early stopping patience=20.
- Trained on 4×RTX 4090; cross-subject evaluation uses leave-one-group-out or 10-fold leave-groups-out (grouped by subject ID).

## Key Experimental Results

### Main Results

Three EEG multimodal benchmarks (balanced accuracy %):

| Dataset | Task / Modalities | Prev. SOTA | EEG-MoCE | Gain |
|---------|------------------|------------|----------|------|
| EAV (n=42) | Emotion Recognition / EEG+Audio+Video | HEEGNet 61.74 | **75.88** | **+14.14** |
| ISRUC (n=10) | Sleep Staging / EEG+EMG+EOG | XSleepFusion 75.19 | **78.53** | +3.34 |
| Cognitive (n=26) | N-back Working Memory / EEG+EOG+NIRS | EF-Net 54.41 | **62.39** | +7.98 |

### Ablation Study

Architecture ablation on EAV (Table 7):

| Encoder | Fusion | Acc (%) | F1 (%) | Notes |
|---------|--------|---------|--------|-------|
| Euclidean | Euclidean | 60.33 | 57.24 | All-Euclidean baseline |
| Euclidean | Hyperbolic | 61.48 | 58.79 | Only fusion hyperbolic +1.15 |
| Hyperbolic | Euclidean | 74.17 | 73.41 | Only encoder hyperbolic +13.84 |
| Hyperbolic | Hyperbolic (Full) | **75.88** | **75.47** | Full hyperbolic +1.71 |

Hyperbolic component ablation (Figure 4):

| Configuration | Acc Gain |
|---------------|----------|
| Fixed K=-2 | baseline |
| + Learnable K | +2.14% |
| + COMF (curvature prior bias) | +1.38% |
| All enabled (learnable K + COMF) | best |

Modality contribution analysis (Table 2, EAV, model with learnable K only, no COMF):

| Modality | δ_rel | Learned $|K|$ | Attention Contribution |
|----------|-------|----------|------------------------|
| EEG | 0.160 | 2.34 | 36.0% |
| Video | 0.278 | 2.29 | 33.6% |
| Audio | 0.293 | 1.91 | 30.5% |

Single-modality ablation (Table 8):

| Modality | Acc | Notes |
|----------|-----|-------|
| Video only | 53.75 | |
| Audio only | 60.52 | |
| EEG only | 62.74 | Consistent with largest $|K|$, smallest δ_rel |
| All modalities | 75.88 | Multimodal vs best unimodal +13.14 |

### Key Findings
- **Most gains come from hyperbolic encoder** (+13.84), while hyperbolic fusion adds only +1.71—indicating the main bottleneck is Euclidean space's inability to represent EEG frequency/semantic hierarchies.
- **Strong correlation among $|K|$, δ_rel, and attention contribution** (Table 2 + 4.2): The hypothesis that curvature serves as an "indicator of hierarchical information content" is quantitatively validated, which is the most convincing part of the paper.
- **Learnable curvature outperforms fixed curvature by 2.14%, COMF adds another 1.38%**: Both mechanisms are independently effective and complementary.
- **Learned curvature prior weight λ automatically increases from 0.30 to 0.33–0.53 during training** (Table 3)—indicating the model increasingly relies on curvature information for attention weighting, not a hard-coded rule.
- **EAV emotion recognition jumps from 61.74→75.88 (+14 points)**, a rare leap in EEG multimodal work, showing hyperbolic geometry is especially beneficial for tasks with strong hierarchy like "subjective emotion".

## Highlights & Insights
- **"Geometric parameter as both expressive capacity and modality weight" is an elegant dual use**: The authors use the single parameter K to (i) determine the embedding space, (ii) control temperature sharpness, and (iii) set fusion bias, making "modality importance" a learnable geometric quantity rather than an extra attention head. This design philosophy can be transferred to any scenario with significant modality information disparity (e.g., medical imaging + text, perception + control).
- **Methodological contribution of using δ-hyperbolicity for modality selection/data profiling**: Turning a purely geometric metric into an engineering tool for multimodal system design, indicating "whether this modality deserves hyperbolic modeling".
- **First systematic extension of mixture-of-curvature to EEG multimodal**, paired with cross-subject evaluation (the most challenging setting), with stable results (low standard deviation over 5 seeds).
- **Using weighted Fréchet mean for fusion instead of Euclidean weighted sum** is a subtle but important detail—preserving manifold semantics throughout.

## Limitations & Future Work
- **Relies on HEEGNet's moments alignment for cross-subject normalization**, without contributing new domain adaptation mechanisms.
- **All three datasets have relatively few subjects (n=10/26/42)**; scalability to large-scale EEG datasets (e.g., SEED-IV, HMS-HBAC with hundreds of subjects) remains untested.
- **Training cost of hyperbolic operations**: Riemannian optimizer and Lorentz attention are 1.5–3× slower than standard Euclidean, but the paper lacks detailed comparison.
- **Sensitivity to initial curvature values**: All initial values are set near K=-2; it is unclear whether extreme initializations can still converge to the correct order.
- **No comparison with more general geometric schemes such as mixture-of-Gaussian-curvature or Riemannian symmetric spaces**.
- **Future directions**: (i) Use δ-hyperbolicity as a self-supervised loss to explicitly regularize learned geometry; (ii) Extend curvature to per-token/per-channel; (iii) Combine hyperbolic attention with Mamba/linear attention to address long-sequence EEG.

## Related Work & Insights
- **vs HEEGNet (Li et al., 2026)**: Unimodal EEG + fixed curvature; this work extends to multimodal + learnable curvature, adding curvature-guided fusion. HEEGNet was previous SOTA on EAV (61.74), surpassed by EEG-MoCE by 14.14 points.
- **vs Hyper-MML (Kang et al., 2025)**: Also hyperbolic multimodal, but uses fixed shared curvature; EEG-MoCE uses per-modality learnable curvature, outperforming by 15.12 points.
- **vs MMML / CTMWA / LMF**: Euclidean multimodal fusion baselines, all outperformed by EEG-MoCE, indicating that geometric choice is the fundamental bottleneck in EEG multimodal learning.
- **vs Mettes et al. (2024) hyperbolic facial expression work**: This work extends that approach from unimodal to EEG-dominated multimodal fusion.
- **vs Gu et al. (2019) mixed-curvature graph**: This work generalizes mixed-curvature from graph embedding to multimodal learning, with the key innovation of coupling curvature into attention.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Curvature = geometry + modality weight" dual use is a truly original design philosophy; δ-hyperbolicity as a modality profiling tool is a methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + multiple tasks, comprehensive ablation (architecture + components + unimodal + multi-seed), quantitative validation of $|K|$ and attention correlation; lacks training cost comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous geometric notation, clear motivation, and beautifully presented quantitative relationships in Table 1-2.
- Value: ⭐⭐⭐⭐ Raising EEG multimodal from 60% to 75% is a clinical deployment-level leap; also provides a template for all domains with large modality hierarchy differences.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](../../ICLR2026/medical_imaging/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[ICML 2026\] Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)

</div>

<!-- RELATED:END -->
