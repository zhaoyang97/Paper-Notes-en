---
title: >-
  [Paper Note] Factored Classifier-Free Guidance
description: >-
  [ICML 2026][Image Generation][Classifier-Free Guidance] This paper identifies an "attribute amplification" failure mode of CFG in counterfactual generation with diffusion models—using a single global $\omega$ amplifies n…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Classifier-Free Guidance"
  - "Counterfactual Generation"
  - "Causal Intervention"
  - "Attribute Amplification"
  - "DDIM"
date: 2026-05-08
content_hash: c74f1eb6a587e550
---

# Factored Classifier-Free Guidance

**Conference**: ICML 2026  
**arXiv**: [2506.14399](https://arxiv.org/abs/2506.14399)  
**Code**: No public link  
**Area**: Diffusion Models / Counterfactual Generation / Medical Imaging  
**Keywords**: Classifier-Free Guidance, Counterfactual Generation, Causal Intervention, Attribute Amplification, DDIM

## TL;DR
This paper identifies an "attribute amplification" failure mode of CFG in counterfactual generation with diffusion models—using a single global $\omega$ amplifies not only the target attribute but also unintended ones. The authors propose FCFG: grouping attributes by causal graph and assigning independent guidance weights to each group, which significantly reduces non-target attribute drift and improves counterfactual reversibility on CelebA-HQ / EMBED / MIMIC-CXR.

## Background & Motivation
**Background**: Diffusion models have become the de facto standard for conditional generation. The standard pipeline for counterfactual generation is a three-stage process: DDIM inversion (abduction) → do-intervention (action) → reverse DDIM with CFG guidance (prediction). Classifier-Free Guidance interpolates between conditional and unconditional scores via $\epsilon_\text{CFG}=(1-\omega)\epsilon_\theta(\varnothing)+\omega\epsilon_\theta(\mathbf{c})$, widely used as a knob to make generated images more strongly reflect target attributes.

**Limitations of Prior Work**: CFG's $\omega$ is a global scalar applied to the entire condition vector $\mathbf{c}$. In counterfactual scenarios, $\mathbf{c}$ typically encodes multiple attributes (e.g., gender, age, smile), but users often wish to intervene on only one, yet are forced to scale all attributes by the same $\omega$. As a result, do(Male=no) also amplifies Smiling, and do(Young=no) changes identity and expression together; such off-target changes violate the invariance axiom of the causal graph, termed attribute amplification.

**Key Challenge**: There is a fundamental tension between "intervention effectiveness" (strongly changing the target attribute) and "stability of non-target attributes"—as long as guidance is a scalar, the two are inevitably coupled. Xia et al. (2024) attributed this to predictor-finetuning during training, but this paper points out that the guidance mechanism itself is the root cause.

**Goal**: Without modifying training or model architecture, break the coupling between attributes at inference by assigning each semantic/causal group an independent guidance strength.

**Key Insight**: If attribute groups are conditionally independent given $\mathbf{x}_t$, $p(\mathbf{pa}\mid\mathbf{x}_t)=\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)$, then the proxy posterior naturally factorizes as $p^\omega(\mathbf{x}_t\mid\mathbf{pa})\propto p(\mathbf{x}_t)\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)^{\omega_m}$—each group has its own $\omega_m$, with CFG as the $M=1$ special case.

**Core Idea**: Rewrite CFG's score update using "attribute-blocked embeddings + group-wise $\omega_m$ assignment," replacing global amplification with fine-grained, group-specific amplification; no model or training changes, only inference-time modification.

## Method

### Overall Architecture
FCFG consists of three parts: (i) Each semantic attribute $\mathbf{pa}=(pa_1,\dots,pa_K)$ is embedded via independent MLPs and concatenated to form an "attribute-split" structure $\mathbf{c}=\text{concat}(\mathcal{E}_1(pa_1),\dots,\mathcal{E}_K(pa_K))$, so each attribute occupies a separate block in the embedding; (ii) At inference, attributes are divided into $M$ groups (e.g., "affected" + "invariant" in counterfactuals), and for each group, a masked embedding $\underaccent{\rule{4.09723pt}{0.4pt}}{\mathbf{c}}^{(m)}$ is constructed by zeroing out non-group blocks; (iii) The CFG score difference is extended to an $M$-term weighted sum, with a separate $\omega_m$ for each group. The full process is embedded in the DDIM counterfactual abduction-action-prediction pipeline—only the prediction step replaces $\epsilon_\text{CFG}$ with $\epsilon_\text{FCFG}$.

### Key Designs

1. **Attribute-Split Embedding**:

    - **Function**: Ensures each attribute occupies a unique segment in $\mathbf{c}$, facilitating group-wise null-token masking at inference.
    - **Mechanism**: Each $pa_i$ is embedded via an independent MLP $\mathcal{E}_i:\mathbb{R}^{d_i}\to\mathbb{R}^d$, and $\mathbf{c}\in\mathbb{R}^{Kd}$ is formed by concatenating all blocks; to mask the $i$-th attribute, multiply the corresponding block by indicator $\delta_i^{(m)}\in\{0,1\}$. All $\mathcal{E}_i$ are jointly trained end-to-end with the denoising network, not as independent feature extractors.
    - **Design Motivation**: Conventional designs mix multiple attributes into a dense vector, entangling semantics in embedding space; attribute-split naturally decouples them, forming the basis for subsequent group-wise guidance.

2. **Group-wise Factored Score**:

    - **Function**: Allows different attribute groups to have independent guidance strengths, breaking CFG's global coupling.
    - **Mechanism**: Assuming group-wise conditional independence $p(\mathbf{pa}\mid\mathbf{x}_t)=\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)$, the proxy posterior is $p^\omega(\mathbf{x}_t\mid\mathbf{pa})\propto p(\mathbf{x}_t)\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)^{\omega_m}$. The corresponding score is $\epsilon_\text{FCFG}=\epsilon_\theta(\varnothing)+\sum_m \omega_m(\epsilon_\theta(\underaccent{\rule{4.09723pt}{0.4pt}}{\mathbf{c}}^{(m)})-\epsilon_\theta(\varnothing))$. $M=1$ reduces to standard CFG; $M=K$ gives each attribute an independent weight.
    - **Design Motivation**: The core mathematical observation is that a global $\omega$ is equivalent to assuming all attributes are conditionally independent and equally weighted; relaxing "equal weights" yields group-wise FCFG, which aligns with the causal graph and only requires inference-time changes.

3. **Causally Guided Affected/Invariant Dual Grouping**:

    - **Function**: Instantiates abstract attribute groups—according to the user's causal graph, assign intervened attributes and their descendants to the "affected" group, others to "invariant," controlled by $\omega_\text{aff}$ and $\omega_\text{inv}$.
    - **Mechanism**: In typical counterfactual do$(A)$, set $\omega_\text{aff}$ high (e.g., 2.5) to enhance target attribute change, and $\omega_\text{inv}$ near 1 (no amplification) to keep non-target attributes stable; this two-group split preserves CFG's effectiveness while eliminating pull on invariant attributes.
    - **Design Motivation**: Directly corresponds to the counterfactual axiom that non-intervened attributes should remain stable, making the axiom-based metric (Δ on invariant) nearly zero without sacrificing Δ on target; the framework also naturally supports finer granularity (e.g., $M=K$ for per-attribute control).

### Loss & Training
The training objective is the standard conditional diffusion loss $\mathbb{E}\|\epsilon-\epsilon_\theta(\mathbf{x}_t,t,\mathbf{c})\|^2$, with classic classifier-free dropout (randomly replacing the entire $\mathbf{c}$ with $\varnothing$), introducing no new losses; FCFG only modifies score computation at inference. The authors acknowledge a slight train-test mismatch (training sees only all-null, inference sees partial-null), but observe no stability issues in experiments. FCFG can also be combined with improved guidance methods like CFG++ and APG by embedding the grouping idea into their score formulas.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | CFG | FCFG | Notes |
|---------|------|--------|-----|------|-------|
| CelebA-HQ 64×64 | do(Smiling) | Δ target ↑ / Δ off-target ↓ | High target but also high off-target | Target close, off-target nearly 0 | Key off-target suppression |
| CelebA-HQ | do(Smiling) inverse reconstruction MAE/LPIPS | Lower is better | Rises sharply with $\omega$ | Significantly lower at same $\omega$ | Better identity preservation |
| EMBED 192×192 (mammography) | do(circle) | Δ density (off-target) | Increases significantly | Nearly 0 | Avoids false feature amplification in medicine |
| MIMIC-CXR | do(finding) | Δ race/sex (off-target) | Significant drift | Strongly suppressed | Important for clinical fairness |
| MIMIC-CXR | do(finding) Δ target AUC | +18.8 | +18.8 (FCFG) vs CFG +X | Off-target only +0.6 | Off-target reduced by an order of magnitude at same target effectiveness |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| $M=1$ (degenerate CFG) | Attribute amplification occurs | Verifies FCFG is a strict generalization |
| Two groups affected/invariant ($M=2$) | Main experiment setting, best effectiveness/off-target trade-off | Default configuration |
| Multi-attribute independent ($M=K$) | Supports do(Smiling, Male, Young) multi-intervention, each attribute with independent $\omega_s,\omega_m,\omega_y$ | When all attributes are intervened, $M=2$ degenerates to global CFG, $M=K$ is required |
| FCFG + CFG++ / FCFG + APG | Stacked on advanced guidance | Also improves off-target amplification, framework compatible |
| Comparison to SA-DCG / HVAE / HVAE-soft | CelebA-HQ do(Smiling) target +13.1 / off-target -1.5 vs SA-DCG +12.9 / +3.0 | Slightly better target, off-target in opposite direction (less drift) |

### Key Findings
- **Root Cause of Attribute Amplification**: Through controlled experiments (CelebA-HQ with three independent attributes), the authors show amplification is not due to dataset artifacts or causal graph mismatch, but the guidance mechanism itself—shifting the blame from "data/model" to "inference algorithm."
- **FID Improvement**: Intuitively, multi-component scores might be less stable, but FCFG significantly outperforms global CFG in FID on CelebA-HQ, indicating that reducing off-target drift helps stay on the data manifold.
- **Counterfactual Reversibility**: After do(A) followed by do$(A^{-1})$, CFG accumulates off-target drift, worsening MAE/LPIPS, while FCFG nearly maintains initial values, serving as a good new metric for counterfactual soundness.
- **Extreme Multi-Attribute Cases**: When all attributes are intervened simultaneously, $M=2$ grouping fails (no invariant group), and only per-attribute FCFG ($M=K$) is viable; the authors discuss this corner case.

## Highlights & Insights
- Directly decomposing "CFG's global $\omega$" into a vector $\omega_m$ grouped by causal graph is an intuitive yet previously overlooked extension; the score formula is cleanly derived from the proxy posterior.
- The attribute-split embedding is a lightweight training design that enables arbitrary inference-time grouping, effectively "preparing a mask interface" for future use—valuable for any conditional diffusion framework.
- Defines a dual-dimensional evaluation for counterfactual generation: "intervention effectiveness vs reversibility," which aligns better with causal axioms than FID alone; this evaluation can also be applied to video editing, 3D consistency, and other conditional generation scenarios.
- Compatible with advanced guidance variants like CFG++ and APG, showing this is an orthogonal dimension to score improvements—future conditional sampling advances can consider "factorization first, then improvement."

## Limitations & Future Work
- Relies on pre-specified causal graphs or semantic groupings; FCFG itself does not solve causal discovery. If attribute relationships are unknown or dynamic, incorrect grouping may worsen the problem.
- $\omega_m$ still requires manual tuning; future work could adaptively select $\omega$ based on input conditions or timestep, enabling timestep-aware FCFG.
- Train-test mismatch is mild but present: training only sees all-null, inference sees group-masked; with large $M$ or strong $\omega$, stability issues may arise.
- When all attributes are intervened, two-group splitting degenerates to global CFG, requiring finer $M=K$ granularity; this corner case exposes the fragility of grouping.
- Maximum experimental resolution is 192×192; effectiveness on high-resolution latent diffusion / SDXL / video diffusion remains to be validated.

## Related Work & Insights
- **vs Standard CFG (Ho & Salimans 2022)**: This work is a strict generalization, fully equivalent when $M=1$; upgrades $\omega$ to a vector $\omega_m$ via conditional independence.
- **vs CFG++ (Chung 2025) / APG (Sadat 2025)**: These improve score shape or manifold constraints for fidelity, but still use global $\omega$; FCFG is orthogonal and can be combined.
- **vs Compositional Diffusion (Liu 2022) / Shen 2024**: Those methods use spatial masks or multiple conditional models for local control; FCFG requires only one model plus semantic grouping.
- **vs HVAE / HVAE-soft (Ribeiro 2023; Xia 2024)**: They address attribute amplification via predictor-finetuning during training; FCFG shifts the solution to inference, leaving training unchanged and lighter-weight.
- **vs SA-DCG (Rasal 2025)**: Uses diffusion autoencoder + identity preservation optimization, which is heavier; FCFG achieves lower off-target and better FID at the same target effectiveness.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet incisive idea, a natural but overlooked extension of the CFG formula
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers CelebA-HQ/EMBED/MIMIC-CXR datasets + multi-angle comparison with HVAE/SA-DCG/CFG++/APG, but lacks high-resolution latent diffusion validation
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivation, failure mode quantified by Δ metrics, intuitive visual comparisons
- Value: ⭐⭐⭐⭐ Plug-and-play, directly valuable for medical counterfactual reasoning and fairness evaluation, extremely low adoption cost for the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](../../ICCV2025/image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)

</div>

<!-- RELATED:END -->
