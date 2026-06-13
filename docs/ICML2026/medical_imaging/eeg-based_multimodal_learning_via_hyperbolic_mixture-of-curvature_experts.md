---
title: >-
  [Paper Note] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts
description: >-
  [ICML 2026][Medical Imaging][EEG] EEG-MoCE assigns a Lorentz manifold expert with a **learnable curvature** to each modality in EEG-based multimodal learning (emotion/sleep/cognition). It then utilizes "curvature-aware a…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "EEG"
  - "Mixture-of-Curvature"
  - "Lorentz Manifold"
  - "Cross-subject Generalization"
  - "$\\delta$-hyperbolicity"
date: 2026-05-08
content_hash: 6cb741a5964588d8
---

# EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts

**Conference**: ICML 2026  
**arXiv**: [2604.12579](https://arxiv.org/abs/2604.12579)  
**Code**: The paper mentions "Code will be released", currently not public  
**Area**: Medical Imaging / Brain-Computer Interface / Multimodal Learning / Hyperbolic Geometry  
**Keywords**: EEG, Mixture-of-Curvature, Lorentz Manifold, Cross-subject Generalization, $\delta$-hyperbolicity

## TL;DR
EEG-MoCE assigns a Lorentz manifold expert with a **learnable curvature** to each modality in EEG-based multimodal learning (emotion/sleep/cognition). It then utilizes "curvature-aware attention"—where "larger curvature $\rightarrow$ richer hierarchical structure $\rightarrow$ higher weight in fusion"—to perform cross-modal integration. The approach achieves cross-subject accuracy gains of +14.14%, +3.34%, and +7.98% on EAV, ISRUC, and Cognitive datasets, respectively.

## Background & Motivation

**Background**: Isolated EEG signals are heavily susceptible to electrophysiological noise and subject variability. Consequently, an increasing number of works combine EEG with video (facial expressions), audio, and EMG/EOG/NIRS for multimodal learning to enhance the robustness of tasks like emotion recognition, sleep staging, and cognitive load assessment. Current mainstream approaches rely on Euclidean architectures (CNN + Transformer + Cross-modal attention).

**Limitations of Prior Work**: (1) Neuroscience confirms that EEG and brain-related modalities exhibit **hierarchical organization** (e.g., emotion processing from subcortical to limbic to neocortex; hierarchical frequency bands). (2) Euclidean embeddings fail to accommodate exponentially expanding hierarchical structures due to linear/quadratic volume growth. (3) Existing hyperbolic EEG work (e.g., HEEGNet) utilizes **fixed curvature** and focuses only on single-modality EEG, treating all modalities uniformly despite massive differences in "hierarchical intensity."

**Key Challenge**: The hierarchical complexity of different modalities naturally varies (quantified in the paper via $\delta$-hyperbolicity: EEG $\delta_{rel} \approx 0.10$, audio $\approx 0.22$, video $\approx 0.28$). Representing them in the same curvature or the same Euclidean space is inaccurate. To let "adaptive curvature" function during multimodal fusion, a mechanism is needed to inform the fusion layer which modality is more reliable at any given moment.

**Goal**: (i) Assign each modality its own Lorentz manifold with a learnable curvature; (ii) explicitly utilize the learned curvatures during fusion for weighting, ensuring modalities with more hierarchical information receive higher weights.

**Key Insight**: A critical observation is that theoretically, a larger absolute curvature $|K|$ allows for embedding deeper hierarchies into fixed dimensions with lower distortion (Sala et al., 2018). Therefore, if a modality's $|K|$ is learned end-to-end to be large, it indicates "richer hierarchical information," and this $|K|$ can inversely serve as a fusion weight.

**Core Idea**: Mixture-of-Curvature experts + curvature-aware cross-modal attention (where $|K|$ determines both single-modality geometry and fusion weights).

## Method

### Overall Architecture
The model $h_\Theta=g_\psi\circ F_\omega\circ(\bigoplus_{m\in\mathcal{M}}E_\phi^{(m)}\circ e_\theta^{(m)})$ consists of four components:

- **Euclidean encoder** $e_\theta^{(m)}$: Specific backbone for each modality (EEG uses EEGNet; EMG/EOG use EEGNet variants; video uses lightweight CNN + Temporal Transformer; audio uses 1D CNN on mel-spectrogram + Temporal Transformer), outputting $\mathbf{x}^{(m)}\in\mathbb{R}^d$.
- **Hyperbolic expert** $E_\phi^{(m)}$: Projects $\mathbf{x}^{(m)}$ onto the modality's specific Lorentz manifold $\mathcal{L}_{K^{(m)}}^d$ (with learnable curvature $K^{(m)}<0$). This is followed by Lorentz BN (with moments alignment for cross-subject adaptation), Lorentz activation/pooling, outputting $\mathbf{z}^{(m)}\in\mathcal{L}_{K^{(m)}}^d$.
- **Curvature-oriented fusion** $F_\omega$: Projects all $\mathbf{z}^{(m)}$ into a shared fusion manifold $\mathcal{L}_{K_f}^d$ (where $K_f$ is the mean of modality curvatures), stacks multiple curvature-guided cross-attention layers, and aggregates via weighted Fréchet mean.
- **Hyperbolic classifier** $g_\psi$: Lorentz multinomial logistic regression (HMLR), using geodesic hyperplanes as decision boundaries.

### Key Designs

1. **Mixture-of-Curvature Experts (Learnable Lorentz Manifold per Modality)**:
    - **Function**: Assigns each modality a suitable geometric space, preventing "under-expression of high-hierarchy modalities" or "over-parameterization of low-hierarchy modalities" caused by shared curvature.
    - **Mechanism**: Each modality $m$ has a learnable curvature $K^{(m)}<0$. Euclidean features $\mathbf{x}^{(m)}$ are projected to the modality's Lorentz hyperboloid via the exponential map $\mathbf{h}^{(m)}=\exp_\mathbf{o}^{K^{(m)}}(\mathbf{x}^{(m)})$ (where $\mathbf{o}=[\sqrt{-1/K^{(m)}},\mathbf{0}]^\top$ is the origin). Subsequent operations (BN, activation, attention) occur on the Lorentz manifold.
    - **Design Motivation**: The authors quantify hierarchical intensity using $\delta$-hyperbolicity and find significant differences (Table 1: EEG $\approx 0.10$, audio $\approx 0.22$, video $\approx 0.28$, NIRS $\approx 0.30$). Fixed curvature is inevitably too loose or too tight for some modalities. Learnable curvature allows the model to **automatically converge to an appropriate $|K|$ for each modality**—experiments show learned values of EEG $|K|=2.34 >$ Vision $2.29 >$ Audio $1.91$, perfectly inversely correlated with $\delta_{rel}$.

2. **Curvature-guided cross-modal attention (Curvature as Temperature and Prior Bias)**:
    - **Function**: Ensures "hierarchy-rich modalities" are more selective in picking information and are prioritized by other modalities in cross-modal attention.
    - **Mechanism**: Modalities are first projected to the shared fusion manifold ($\mathbf{z}_f^{(m)}=\exp_\mathbf{o}^{K_f}(\sqrt{K^{(m)}/K_f}\cdot\log_\mathbf{o}^{K^{(m)}}(\mathbf{z}^{(m)}))$) to preserve hyperbolic geometry. Attention uses negative squared geodesic distance for similarity. Two curvature coupling mechanisms are added: (i) Temperature $\tau^{(m)}=\tau_0/\sqrt{|K^{(m)}|}$, where larger $|K|$ results in lower temperature and sharper attention; (ii) Prior bias $\lambda\cdot\phi(K^{(j)})$ (where $\phi(K)=\log(|K|+\epsilon)$), steering queries to favor keys with large $|K|$: $\tilde{\alpha}_{m\to j}\propto\exp(-d_{\mathcal{L}}^2(\mathbf{q}^{(m)},\mathbf{k}^{(j)})/\tau^{(m)}+\lambda\cdot\phi(K^{(j)}))$. Aggregation is performed via weighted Fréchet mean.
    - **Design Motivation**: Curvature is treated not just as a geometric parameter, but as a **learnable indicator of a modality's information content**. Temperature adjustment enables precise cross-modal queries for strong modalities, while prior bias ensures weak modalities still aggregate from strong ones. Table 2 shows EEG attention contribution at 36% > Vision 33.6% > Audio 30.5%, aligning with the $|K|$ ranking.

3. **Hyperbolic Full-stack Processing + Cross-subject Normalization**:
    - **Function**: Performs computations on the Lorentz manifold from encoder to classifier to avoid hierarchical distortion. Uses hyperbolic BN with moments alignment for cross-subject distribution shifts.
    - **Mechanism**: (i) Lorentz fully-connected $f_\mathcal{L}(\mathbf{p})=(\sqrt{\|\tilde{\mathbf{p}}_s\|^2-1/K},\tilde{\mathbf{p}}_s)$ where $\tilde{\mathbf{p}}_s=\psi(\mathbf{Wp}+\mathbf{b})$; (ii) Lorentz BN derived from HEEGNet for alignment; (iii) HMLR classification via geodesic hyperplanes.
    - **Design Motivation**: A "compositional design" is emphasized—Euclidean encoders learn spatio-temporal local features, while hyperbolic components handle hierarchical modeling and fusion. The Lorentz model is chosen over the Poincaré ball due to better numerical stability during gradient optimization.

### Loss & Training
- Classification loss + auxiliary terms, 100 epochs. Adam for Euclidean parameters, Riemannian Adam for hyperbolic parameters; lr=1e-3, early stopping patience=20.
- Trained on 4×RTX 4090. Cross-subject evaluation via leave-one-group-out or 10-fold leave-groups-out.

## Key Experimental Results

### Main Results

Three EEG multimodal benchmarks (balanced accuracy %):

| Dataset | Task / Modality | Prev. SOTA | EEG-MoCE | Gain |
|--------|------------|---------|----------|------|
| EAV (n=42) | Emotion / EEG+Audio+Video | HEEGNet 61.74 | **75.88** | **+14.14** |
| ISRUC (n=10) | Sleep Staging / EEG+EMG+EOG | XSleepFusion 75.19 | **78.53** | +3.34 |
| Cognitive (n=26) | Working Memory / EEG+EOG+NIRS | EF-Net 54.41 | **62.39** | +7.98 |

### Ablation Study

Architecture ablation on EAV (Table 7):

| Encoder | Fusion | Acc (%) | F1 (%) | Description |
|---------|--------|---------|--------|------|
| Euclidean | Euclidean | 60.33 | 57.24 | All-Euclidean baseline |
| Euclidean | Hyperbolic | 61.48 | 58.79 | Hyperbolic fusion only (+1.15) |
| Hyperbolic | Euclidean | 74.17 | 73.41 | Hyperbolic encoder only (+13.84) |
| Hyperbolic | Hyperbolic (Full) | **75.88** | **75.47** | All-hyperbolic (+1.71) |

Hyperbolic component ablation (Figure 4):

| Configuration | Acc Gain |
|------|---------|
| Fixed K=-2 | baseline |
| + Learnable K | +2.14% |
| + COMF (prior bias) | +1.38% |
| Complete (Learnable K + COMF) | Best |

### Key Findings
- **Most gains stem from the hyperbolic encoder** (+13.84%), while hyperbolic fusion adds +1.71%. This suggests the primary bottleneck is the Euclidean space's inability to represent EEG frequency/semantic hierarchies.
- **Strong correlation exists between $|K|$, $\delta_{rel}$, and attention contribution** (Table 2 & 4.2). The hypothesis of curvature as a "hierarchical information indicator" is quantitatively validated.
- **Learnable curvature outperforms fixed curvature by 2.14%**, and COMF adds another 1.38%, showing complementarity.
- **Emotion recognition on EAV jumped from 61.74 to 75.88**, a rare 14-point leap in the field, indicating that tasks with the strongest hierarchies (like subjective emotion) benefit most from hyperbolic geometry.

## Highlights & Insights
- **Dual utilization of geometry as representation power and modality weight**: The parameter $K$ determines (i) the embedding space, (ii) attention sharpness (temperature), and (iii) fusion bias. Modality importance becomes a learnable geometric quantity rather than an arbitrary attention head.
- **Methodological contribution via $\delta$-hyperbolicity**: Converts a pure geometric metric into an engineering tool for modality profiling to determine if a modality warrants hyperbolic treatment.
- **First systematic extension of mixture-of-curvature to EEG multimodal learning**, uniquely paired with rigorous cross-subject evaluation.
- **Weighted Fréchet mean** for fusion ensures manifold semantics are preserved better than Euclidean weighted sums.

## Limitations & Future Work
- **Reliance on HEEGNet's moments alignment** for cross-subject normalization; no new domain adaptation mechanism proposed.
- **Small sample sizes** across the three datasets (n=10/26/42); scalability to large-scale EEG data (hundreds of subjects) remains unverified.
- **Computational cost**: Riemannian optimizers and Lorentz attention are 1.5-3x slower than Euclidean counterparts.
- **Sensitivity to initial curvature**: All initializations were near $K=-2$; convergence from extreme initial values was not explored.
- **Future directions**: Integrating $\delta$-hyperbolicity as a self-supervised loss to regularize geometry; extending curvature to per-token/per-channel; combining hyperbolic attention with Mamba for long-sequence EEG.

## Related Work & Insights
- **vs HEEGNet (Li et al., 2026)**: HEEGNet uses single-modality EEG and fixed curvature. This work extends to multimodal and learnable curvature with a new fusion mechanism, outperforming HEEGNet by 14.14 points on EAV.
- **vs Hyper-MML (Kang et al., 2025)**: Hyper-MML uses fixed shared curvature; EEG-MoCE's per-modality learnable curvature leads by 15.12 points.
- **vs Euclidean baselines (MMML / CTMWA / LMF)**: Broadly outperformed, suggesting that incorrect geometric assumptions are a fundamental hurdle in EEG multimodal learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "curvature = geometry + modality weight" design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid multi-task/multi-dataset validation and ablations; lacks detailed training efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous notation and clear motivation.
- **Value**: ⭐⭐⭐⭐ A significant leap (60% to 75%) toward clinical feasibility for EEG-based multimodal systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SEMC: Structure-Enhanced Mixture-of-Experts Contrastive Learning for Ultrasound Standard Plane Recognition](../../AAAI2026/medical_imaging/semc_structure-enhanced_mixture-of-experts_contrastive_learning_for_ultrasound_s.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICML 2026\] Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](../../NeurIPS2025/medical_imaging/mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)

</div>

<!-- RELATED:END -->
