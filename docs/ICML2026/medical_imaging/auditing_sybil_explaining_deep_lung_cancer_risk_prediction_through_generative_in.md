---
title: >-
  [Paper Note] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions
description: >-
  [ICML 2026][Medical Imaging][Lung Cancer Screening] This paper proposes S(H)NAP—a generative interventional framework based on 3D diffusion bridges using "removal + insertion." It decomposes the decisions of Sybil…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Lung Cancer Screening"
  - "Sybil"
  - "Counterfactual Explanations"
  - "Diffusion Bridge"
  - "Shapley Interaction"
  - "Interventional Auditing"
date: 2026-05-08
content_hash: e77f77bb0bfed16a
---

# Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions

**Conference**: ICML 2026  
**arXiv**: [2602.02560](https://arxiv.org/abs/2602.02560)  
**Code**: Not yet released  
**Area**: Medical Imaging / Explainable AI / Causal Attribution  
**Keywords**: Lung Cancer Screening, Sybil, Counterfactual Explanations, Diffusion Bridge, Shapley Interaction, Interventional Auditing

## TL;DR
This paper proposes S(H)NAP—a generative interventional framework based on 3D diffusion bridges using "removal + insertion." It decomposes the decisions of Sybil, a state-of-the-art lung cancer risk prediction model, into an LMPI (Linear + Second-order Interaction Model) consisting of "nodule main effects + pairwise interactions + background." For the first time, it audits the model's reliance on in-hospital artifacts (e.g., ECG electrodes, metallic clothing buttons) and identifies a severe "radial insensitivity" failure mode regarding peripheral lung nodules through causal rather than correlative methods.

## Background & Motivation

**Background**: Lung cancer remains the leading cause of cancer-related mortality globally, with LDCT screening being the primary tool. Sybil (Mikhael 2023), a deep learning model predicting 6-year risk from a single CT scan, has undergone observational clinical validation across multiple centers like NLST. Currently, "trust" in Sybil relies almost entirely on **purely observational** metrics such as AUC and subgroup calibration.

**Limitations of Prior Work**: Observational metrics only indicate "how well the model performs on data" but fail to explain "why it performs well" or "when it will fail." In high-stakes medical deployment, this is a fatal blind spot—a model might rely on artifacts like ECG electrodes or scan beds, or systematically underestimate nodules in specific anatomical locations, yet these issues remain hidden in AUC figures.

**Key Challenge**: Traditional attribution methods (SHAP/IG/Grad-CAM) either operate at the pixel level violating the data manifold or capture correlations rather than causation. While Visual Counterfactual Explanations (VCE) reside at the top of Pearl’s causal ladder, they only show "what changed" without decomposing "how much each specific change contributed." They cannot answer clinical questions like "which specific nodule drove the risk?"

**Goal**: Construct a **generative interventional attribution** that resides on the LDCT data manifold, precisely decomposes main and pairwise interaction effects of each lung nodule, and detects spatial sensitivity biases across arbitrary locations.

**Key Insight**: The authors adopt clinical consensus as a structural prior—"lung nodules are the primary imaging biomarkers for lung cancer risk prediction." Based on this, they propose **Hypothesis 1**: Sybil’s decisions can be well-approximated by an LMPI, comprising a background term $\mu_\mathbf{x}$ + nodule main effects + pairwise interactions. Once this hypothesis holds, "counterfactuals" become equivalent to "toggling specific nodules," which aligns naturally with controllable inpainting via diffusion bridges.

**Core Idea**: Use System-Embedded Diffusion Bridges (SDB) for high-fidelity "nodule removal" and "nodule insertion" interventions on 3D CT sub-volumes. By generating all possible nodule coalitions as inputs for Sybil and using n-Shapley Values ($n=2$) to regress LMPI coefficients, the authors establish the first **causal-level** auditing framework for Sybil.

## Method

### Overall Architecture
S(H)NAP = SHNAP (interpretative attribution) + SNAP (sensitivity probe). Both pathways share the underlying SDB intervention engine. SHNAP follows the "removal path": given a real CT, it generates all $2^N$ subsets of $N$ lung nodules (preserving selected nodules and replacing others with healthy tissue). Each generated sample is fed to Sybil to obtain risk logits, which are regressed using n-SV into main effects $\phi_i$ and interactions $\phi_{ij}$. SNAP follows the "insertion path": it inserts a nodule of known properties into arbitrary CT locations and records the change in predicted logit $\psi_\mathbf{c}=f(y_0\mid\mathbf{x}_{\mathbf{c}\leftarrow\mathbf{r}})-f(y_0\mid\mathbf{x})$, generating a high-resolution "spatial sensitivity" heatmap.

### Key Designs

1. **SDB-driven In-distribution Intervention**:
    - **Function**: Replaces target nodules with "healthy lung tissue" or generates "real nodules at arbitrary locations" while maintaining the surrounding anatomical structure.
    - **Mechanism**: SDB generalizes the diffusion process endpoint from pure noise to a linear measurement $\mathbf{x}'=\mathbf{A}\mathbf{x}+\Sigma^{1/2}\varepsilon$. When $\mathbf{A}$ is a binary mask and $\Sigma=0$, it reduces to specialized inpainting. Reverse sampling updates only within the mask, ensuring the exterior remains strictly unchanged. Theoretically, via the "mismatched estimation" theorem (Verdú 2009), the score model $\mathbf{s}_\xi$ pulls any "copy-paste" or "void" input into the training distribution after sufficient diffusion time. During removal, the prior acts as a healthy tissue generator (since nodules occupy $<0.1\%$ of lung volume). During insertion, a source nodule is pasted into the mask, forward diffused to time $\tau$ (0.3 in experiments), and reverse denoised to blend with the new background.
    - **Design Motivation**: Traditional counterfactuals either flip labels globally using GANs (losing locality) or use crude zero/mean filling (drifting off-manifold, making SHAP unstable). SDB packages "local repair + manifold fidelity" into a mathematically rigorous operation. In a double-blind study, radiologists could not distinguish real tissue from SDB-removed regions better than random guessing (point estimate 0.57), proving the interventions are clinically "seamless."

2. **n-Shapley Regression for LMPI Coefficients (SHNAP)**:
    - **Function**: Decomposes Sybil’s logit responses across $2^N$ nodule coalitions into a baseline + individual nodule main effects + pairwise interactions.
    - **Mechanism**: Construct a dataset $D=\{(S,v_\mathbf{x}(S))\}_{\mathbf{x}_S\in\mathcal{X}}$ where $v_\mathbf{x}(S)=f(y_0\mid \mathbf{x}_S)$, then use SHAP-IQ on the n-Shapley formula truncated at $n=2$ to regress $\phi_\emptyset, \phi_i, \phi_{ij}$. Fitting quality is measured by $R^2=1-\sum(v-\hat v_{\text{nSV}})^2/\sum(v-\bar v)^2$. Since $N$ is typically small in clinical cases (single digits), the $2^N$ evaluations are computationally feasible.
    - **Design Motivation**: n-SV is the unique least-squares projection of LMPI, naturally inheriting SHAP’s local accuracy/consistency axioms. This provides the first interpretable numerical answer with error bars for "how much risk each nodule contributes." Empirical results show median $R^2 \approx 1$, confirming Hypothesis 1.

3. **Insertion-based Spatial Sensitivity Probes (SNAP) + gSHNAP**:
    - **Function**: SNAP inserts the same known nodule into thousands of locations in a single CT to map spatial sensitivity; gSHNAP replaces "nodule indicators" with "arbitrary ROI indicators" to audit any non-nodule region focused on by Sybil's attention.
    - **Mechanism**: SNAP uses the log-odds difference $\psi_\mathbf{c}$ for point attribution. A two-way ANOVA on 240 patient-nodule combinations across $\approx 900$ insertion points revealed a significant lobe main effect ($p<0.001$) while the patient-lobe interaction was non-significant, proving lobar bias is a **global** model characteristic. Linear regression on distance-to-pleura quantifies "radial decay." gSHNAP binarizes attention maps to obtain ROI sets and audits them via the SDB-removal pipeline.
    - **Design Motivation**: Removal-based SHNAP only explains "existing nodules" and cannot detect "what the model relies on where there are no nodules"—a real failure mode involving hospital artifacts. SNAP/gSHNAP extends auditing to the counterfactual space, uncovering shortcuts invisible to traditional observational studies.

### Loss & Training
SDB utilizes a discrete variant of the Schrödinger Bridge, with 1,000 steps and 64³ cubes. Training masks are procedurally generated using metaballs. The backbone learns a healthy tissue prior on 28K scans from the NLST dataset. Sybil itself remains frozen; the entire auditing suite is model-agnostic, meaning the pipeline can be applied directly to closed-source commercial models like Optellum. Removal/insertion inference takes 100 NFE.

## Key Experimental Results

### Main Results
S(H)NAP systematically audited Sybil across three datasets.

| Dataset | Scale | Key Findings | Clinical Implication |
| :--- | :--- | :--- | :--- |
| NLST | 28K Train / 6K Test | Radiologists' acc=0.57 in distinguishing real vs. SDB-removed tissue (statistically indistinguishable) | SDB intervention is in-distribution |
| LUNA25 | 4,069 Scans | Main effects alone reach $R^2 \approx 1$ | Hypothesis 1 holds; Sybil is effectively an LMPI |
| iLDCT | 243 OOD Scans | Sybil focuses more on nodules in severe cases, but artifact dependency becomes more apparent | Failure modes couple with sample severity |

### Ablation Study

| Configuration | Primary Observation | Interpretation |
| :--- | :--- | :--- |
| SHNAP Main Effect (1st order) | $R^2 \approx 1$ for most samples | Sybil decisions are largely explained by independent nodule terms |
| + 2nd Order Interaction | Outliers almost completely eliminated | Complex interactions exist in a minority of cases |
| Naive Perturbation (zero-fill) | High attribution variance, unstable | OOD inputs cause SHAP to degrade into adversarial noise |
| gSHNAP on Random Lung ROIs | Importance distribution concentrated at 0 | Influential regions are sparse; Sybil does not react to "any" perturbation |

### Key Findings
- **Nodule Radial Decay**: Predicting SNAP attribution using distance-to-pleura yielded a significant positive coefficient ($p<0.001$). Adding nodule identity interaction increased $R^2$ from 0.071 to 0.455. Malignant nodules are "suppressed" more heavily as they approach the pleura, likely due to 3D convolution zero-padding. This coincides with a clinical blind spot for adenocarcinoma, which frequently occurs in the periphery.
- **Lobar Bias**: Post-hoc Tukey HSD showed upper lobe attribution was significantly higher than middle/lower lobes ($p \le 0.009$), consistent with PanCan/Mayo clinical priors. Sybil correctly ignored left-right differences.
- **Dangerous Artifact Dependency**: gSHNAP found that 50% of the predicted risk in certain negative cases came from two symmetric ECG electrodes outside the chest wall, misinterpreting "cardiac monitoring" as "high risk," similar to the classic "hospital tag" shortcut.
- **"Right for the Wrong Reasons"**: In some malignant cases, Sybil categorized real nodules as "negative evidence," yet achieved correct high-risk predictions through background features and nodule interaction terms—a double failure hidden from AUC.

## Highlights & Insights
- Elevates the standard for "trusting a deep medical model" from observational metrics to the counterfactual level of Pearl’s causal ladder. The model-agnostic workflow can be applied to any CT-based risk predictor.
- Leverages clinical priors to compress the intractable $2^d$ Shapley problem into $2^N$ (where $N$ is small), making LMPI a computationally rigorous "white-box approximation."
- Uses hundreds of thousands of SNAP insertions to construct high-resolution spatial sensitivity maps, visualizing exactly where the model is "blind" or "hypersensitive." This design is transferable to any lesion-driven task (e.g., breast or skin cancer).

## Limitations & Future Work
- Dependency on synthetic data: despite expert blinding, risks of generative artifacts remain. Ideally, provably robust counterfactuals are needed.
- LMPI assumptions may fail for rare, massive, or morphologically atypical nodules (SDB reconstruction degrades), requiring larger-voxel training or multi-scale SDB.
- SNAP insertions focus on single nodules; emergent interactions between multiple inserted nodules are not yet characterized. SDB was trained on LDCT; cross-modal auditing (PET/MRI) requires retraining the prior.

## Related Work & Insights
- **vs. Classic SHAP / IG**: These methods use black pixels or mean images as baselines, violating the data manifold and causing unstable attribution. SHNAP uses SDB-generated "healthy lungs," placing Shapley values firmly in-distribution.
- **vs. Visual Counterfactuals (DiME, Jeanneret et al.)**: VCE only provides "flipped images" without quantifying specific structural contributions. SHNAP overlays LMPI + n-SV regression to upgrade "counterfactual maps" into "causal attribution coefficients."
- **vs. Mind-the-Pad (Alsallakh 2021)**: That work structurally identified how zero-padding in 3D convs causes boundary activation decay. S(H)NAP provides clinical evidence of this effect in Sybil, linking architectural flaws to a "systemic underreporting of peripheral lung cancer."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First combination of generative diffusion bridges and Shapley interaction models for clinical high-stakes auditing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, double-blind expert studies, ANOVA, and Tukey HSD; statistically rigorous.
- Writing Quality: ⭐⭐⭐⭐ Strong integration of theory and empirical findings, though the SDB section may be dense for readers without a diffusion background.
- Value: ⭐⭐⭐⭐⭐ Directly reveals Sybil deployment risks; the methodology is highly transferable for pre-deployment auditing of medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_imaging/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[CVPR 2026\] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](../../CVPR2026/medical_imaging/association_of_radiologic_ppfe_change_with_mortali.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[AAAI 2026\] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](../../AAAI2026/medical_imaging/grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)

</div>

<!-- RELATED:END -->
