---
title: >-
  [Paper Note] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions
description: >-
  [ICML 2026][Medical Imaging][Lung cancer screening] This paper introduces S(H)NAP—a generative interventional framework based on 3D diffusion bridges for "removal + insertion"—which reverses the decision process of Sybil…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Lung cancer screening"
  - "Sybil"
  - "counterfactual explanation"
  - "diffusion bridge"
  - "Shapley interaction"
  - "interventional auditing"
date: 2026-05-08
content_hash: 5fba5a69c3fe737c
---

# Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions

**Conference**: ICML 2026  
**arXiv**: [2602.02560](https://arxiv.org/abs/2602.02560)  
**Code**: Not yet released  
**Area**: Medical Imaging / Explainable AI / Causal Attribution  
**Keywords**: Lung cancer screening, Sybil, counterfactual explanation, diffusion bridge, Shapley interaction, interventional auditing

## TL;DR
This paper introduces S(H)NAP—a generative interventional framework based on 3D diffusion bridges for "removal + insertion"—which reverses the decision process of Sybil, a state-of-the-art lung cancer risk prediction model, into an LMPI (linear + second-order interaction model) comprising "nodule main effects + pairwise interactions + background." For the first time, it causally (rather than correlationally) audits Sybil’s reliance on in-hospital artifacts such as ECG electrodes and clothing metal fasteners, and reveals a severe "radial insensitivity" failure mode for peripheral lung nodules.

## Background & Motivation

**Background**: Lung cancer remains the leading cause of cancer mortality worldwide, with LDCT screening as the mainstream approach. Sybil (Mikhael 2023), a deep model predicting 6-year risk from a single CT, has undergone observational clinical validation in multi-center datasets like NLST. Trust in Sybil is currently based almost entirely on **purely observational** metrics such as AUC and subgroup calibration.

**Limitations of Prior Work**: Observational metrics only indicate "how well the model performs on data," but cannot answer "why it performs well" or "when it will fail." In high-risk medical deployments, this is a critical blind spot—the model may rely on artifacts like ECG electrodes or scanning beds, or systematically underestimate nodules in certain anatomical locations, yet the AUC provides no indication.

**Key Challenge**: Traditional attribution methods (SHAP/IG/Grad-CAM) either remain at the pixel level, violating the data manifold, or are correlational rather than causal. Visual counterfactual explanations (VCE) stand at the top of Pearl’s causal ladder but can only show "what was changed," not decompose "the contribution of each specific change," and thus cannot answer clinical questions like "which nodule drove the risk."

**Goal**: To construct a **generative interventional attribution** method that remains on the LDCT data manifold and precisely decomposes each lung nodule’s main and pairwise interaction effects, while also probing the model’s sensitivity bias to any spatial location.

**Key Insight**: The authors treat clinical consensus as a structural prior—"lung nodules are the primary imaging biomarkers for lung cancer risk prediction"—and propose **Hypothesis 1**: Sybil’s decisions can be well-approximated by an LMPI, consisting of a background term $\mu_\mathbf{x}$, nodule main effects, and pairwise nodule interactions. Once this hypothesis holds, "counterfactuals" become equivalent to "switching certain nodules on or off," which aligns naturally with the controllable repair of diffusion bridges.

**Core Idea**: System-Embedded Diffusion Bridges (SDB) are used to perform high-fidelity "nodule removal" and "nodule insertion" interventions on 3D CT subvolumes, generating every possible nodule coalition as input to Sybil. n-Shapley Values (n=2) are then regressed to obtain LMPI coefficients, resulting in the first **causal-level** auditing framework for Sybil.

## Method

### Overall Architecture
S(H)NAP = SHNAP (explanatory attribution) + SNAP (sensitivity probe). Both routes share the same underlying SDB intervention engine. SHNAP follows the "removal path": given a real CT, all $2^N$ subsets of $N$ lung nodules are generated (retaining or removing each nodule, with removals replaced by healthy tissue), each sample is fed to Sybil to obtain the risk logit, and n-SV regresses the logit to main effects $\phi_i$ and interactions $\phi_{ij}$. SNAP follows the "insertion path": a known nodule is inserted at arbitrary locations in the CT, and the change in predicted logit $\psi_\mathbf{c}=f(y_0\mid\mathbf{x}_{\mathbf{c}\leftarrow\mathbf{r}})-f(y_0\mid\mathbf{x})$ is recorded, producing a high-resolution "spatial sensitivity" heatmap.

### Key Designs

1. **Diffusion Bridge-based Nodule Removal/Insertion (SDB-driven In-distribution Intervention)**:

    - **Function**: Replaces a target nodule with healthy lung tissue or generates a realistic nodule at any location, while keeping surrounding anatomy unchanged.
    - **Mechanism**: SDB generalizes the endpoint of the diffusion process from pure noise to a linear measurement $\mathbf{x}'=\mathbf{A}\mathbf{x}+\Sigma^{1/2}\varepsilon$. When $\mathbf{A}$ is a binary mask and $\Sigma=0$, it degenerates to specialized inpainting; reverse sampling updates only within the mask, ensuring strict invariance outside. Theoretically, by Verdú 2009’s "mismatched estimation" theorem, the score model $\mathbf{s}_\xi$ will, after sufficient diffusion time, map any "copy-paste" or "cut-out" input to be indistinguishable from the training distribution. For removal, the prior acts as a healthy tissue generator (since nodules occupy <0.1% of lung volume); for insertion, a heterologous nodule is pasted into the mask, forward diffused to time $\tau$ (set to 0.3 in experiments), then reverse denoised to blend with the new background.
    - **Design Motivation**: Traditional counterfactuals either use GANs to flip labels in one shot (losing locality) or crudely fill with zeros/means (off-manifold, destabilizing SHAP). SDB packages "local repair + manifold fidelity" into a mathematically rigorous operation, and in double-blind expert studies, radiologists could not statistically distinguish real tissue from SDB removals (point estimate 0.57), demonstrating that the intervention is "seamless" in clinical terms.

2. **n-Shapley Regression for LMPI Coefficients (SHNAP)**:

    - **Function**: Decomposes Sybil’s logit response over $2^N$ nodule coalitions into baseline + each nodule’s main effect + each pairwise interaction.
    - **Mechanism**: Constructs a dataset $D=\{(S,v_\mathbf{x}(S))\}_{\mathbf{x}_S\in\mathcal{X}}$, where $v_\mathbf{x}(S)=f(y_0\mid \mathbf{x}_S)$, and uses SHAP-IQ to regress $\phi_\emptyset,\phi_i,\phi_{ij}$ on the n=2 truncated n-Shapley formula. Fit quality is measured by $R^2=1-\sum(v-\hat v_{\text{nSV}})^2/\sum(v-\bar v)^2$; typical $N$ is single-digit (each patient usually has few nodules), so $2^N$ evaluations are computationally feasible in clinical settings.
    - **Design Motivation**: n-SV is the unique least-squares projection of LMPI, naturally inheriting SHAP’s local accuracy/consistency axioms, providing the first interpretable, error-barred numerical answer to "how much risk does each nodule contribute." Empirically, median $R^2\approx 1$, confirming Hypothesis 1.

3. **Insertion-based Spatial Sensitivity Probe (SNAP) + gSHNAP**:

    - **Function**: SNAP inserts the same known nodule at thousands of locations in a single CT to map spatial sensitivity; gSHNAP replaces "nodule indicator" with "arbitrary ROI indicator," auditing any non-nodule region Sybil attends to.
    - **Mechanism**: SNAP uses log-odds difference $\psi_\mathbf{c}$ for pointwise attribution; in 240 patient-nodule pairs × ≈900 insertion points, two-way ANOVA reveals significant lobe main effects ($p<0.001$) but non-significant patient×lobe interaction, proving lobar bias is a **global** model property; further, linear regression on distance-to-pleura quantifies "radial decay." gSHNAP binarizes the attention map to obtain ROI sets, using the same SDB-removal process to audit each attention region.
    - **Design Motivation**: Removal-based SHNAP only explains "existing nodules," unable to discover "what the model relies on in nodule-free regions"—yet real failure modes often stem from in-hospital artifacts, scanning frames, ECG electrodes, etc. SNAP/gSHNAP extend auditing from "existing features" to "arbitrary space/regions," reaching the counterfactual space and uncovering shortcuts invisible to traditional observational studies.

### Loss & Training

SDB follows a discrete variant of the Schrödinger Bridge, with 1000 steps, $64^3$ cubes, and procedurally generated metaball training masks; the backbone learns a healthy tissue prior on NLST 28K training scans. Sybil itself remains frozen; the entire audit is model-agnostic, relying only on input-output pairs, so the same pipeline can be directly applied to closed-source commercial models like Optellum. Each removal/insertion inference uses 100 NFE.

## Key Experimental Results

### Main Results

S(H)NAP systematically audits Sybil on three datasets.

| Dataset | Size | Key Findings | Clinical Implications |
|---------|------|--------------|----------------------|
| NLST | 28K train / 6K test | Radiologists’ accuracy in distinguishing real vs SDB-removed healthy tissue: 0.57, statistically indistinguishable | SDB intervention is in-distribution |
| LUNA25 | 4,069 scans | LMPI main effects alone achieve $R^2\approx 1$ | Hypothesis 1 holds; Sybil is indeed LMPI |
| iLDCT | 243 OOD scans | In severe cases, Sybil focuses more on nodules, but artifact reliance also increases | Failure modes are coupled with case severity |

### Ablation Study

| Configuration | Main Observation | Interpretation |
|---------------|------------------|---------------|
| SHNAP main effects (first-order) | $R^2\approx 1$ for most samples | Sybil’s decisions are largely explained by independent nodule terms |
| + Second-order interactions | Outliers almost entirely eliminated | Some complex cases exhibit nodule interaction effects |
| Naive perturbation (zero-fill) instead of SDB | Attribution variance is large and unstable | OOD inputs degrade SHAP to adversarial noise |
| gSHNAP on random lung ROIs | Importance distribution centered at 0 | Influential regions are sparse; Sybil does not "react to any perturbation" |

### Key Findings

- **Radial decay of nodule effect**: Distance-to-pleura significantly predicts SNAP attribution (positive coefficient, $p<0.001$); adding nodule identity interaction increases $R^2$ from 0.071 to 0.455—malignant nodules near the pleura are heavily suppressed, benign ones are not, possibly due to zero-padding in 3D convs. This matches the clinical blind spot where adenocarcinoma, the most common lung cancer subtype, often occurs peripherally.
- **Lobar bias**: Post-hoc Tukey HSD shows upper lobe attributions are significantly higher than middle/lower lobes ($p\le 0.009$), consistent with PanCan/Mayo clinical priors; Sybil also correctly ignores left-right differences.
- **Dangerous artifact reliance**: gSHNAP finds that in negative cases, 50% of predicted risk comes from two symmetric ECG electrodes outside the chest wall—misreading "ECG monitoring" as "high risk," akin to the classic "hospital tag" shortcut.
- **"Correct for the wrong reason"**: In some malignant cases, Sybil classifies the true nodule as "negative evidence," with background features and nodule interactions offsetting each other to yield a correct high-risk prediction—a double failure invisible to AUC.

## Highlights & Insights

- Elevates the standard for "trusting a deep medical model" from observational metrics to the counterfactual layer at the top of Pearl’s causal ladder; the entire process is model-agnostic and transferable to any CT risk predictor.
- Uses clinical priors (nodules as main biomarkers) to compress the usually intractable $2^d$ Shapley problem to $2^N$ (with $N$ typically small), making LMPI a computable and rigorous "white-box approximation."
- Constructs high-resolution spatial sensitivity maps via hundreds of thousands of SNAP insertions, enabling visualization and statistical testing of "where the model is blind or hypersensitive" for the first time; this design is directly transferable to any lesion-driven task such as breast or skin cancer.

## Limitations & Future Work

- Relies partly on synthetic data; despite expert blind review, there remains a risk of generative artifacts. Ideally, provably robust counterfactuals are needed (the authors mention Zaher 2026 in this direction).
- The LMPI assumption fails for rare, very large, or morphologically unusual nodules (SDB reconstruction degrades), requiring larger voxel training or multi-scale SDB.
- SNAP currently inserts only single nodules at a time; emergent interactions among multiple nodules are not yet characterized. SDB is trained only on LDCT; cross-modal auditing for PET/MRI would require retraining the prior.

## Related Work & Insights

- **vs Classic SHAP / IG**: Those methods use black pixels or mean images as baselines, violating the data manifold and destabilizing attribution; SHNAP replaces the baseline with SDB-generated "healthy lung," ensuring Shapley values are truly in-distribution.
- **vs Visual Counterfactuals (DiME, Jeanneret series)**: VCE only provides "flipped images," without interpreting each structure’s contribution; SHNAP builds LMPI + n-SV regression atop VCE, upgrading "counterfactual images" to "causal attribution coefficients."
- **vs Mind-the-Pad (Alsallakh 2021)**: That work structurally showed that 3D conv padding causes boundary activation decay; S(H)NAP empirically demonstrates on clinical data that this evolves into "systematic underreporting of peripheral lung cancer" in Sybil, linking architectural flaws to clinical consequences.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to combine generative diffusion bridges + Shapley interaction models for clinical high-risk model auditing
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, double-blind expert study, ANOVA, Tukey HSD—all covered, statistically rigorous
- Writing Quality: ⭐⭐⭐⭐ Strong integration of theory and empirical findings, but SDB section may be challenging for readers without diffusion background
- Value: ⭐⭐⭐⭐⭐ Directly reveals deployment risks of Sybil; methodology is transferable to pre-deployment auditing of other medical AI systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[ICML 2026\] SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)
- [\[ICML 2026\] OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)
- [\[ICML 2026\] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)

</div>

<!-- RELATED:END -->
