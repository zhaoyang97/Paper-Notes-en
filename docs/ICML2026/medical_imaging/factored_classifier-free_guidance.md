---
title: >-
  [Paper Note] Factored Classifier-Free Guidance
description: >-
  [ICML 2026][Medical Imaging][Classifier-Free Guidance] This paper identifies the "attribute amplification" failure mode of CFG in diffusion-based counterfactual generation—where a single global $\omega$ amplifies attribu…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Classifier-Free Guidance"
  - "Counterfactual Generation"
  - "Causal Intervention"
  - "Attribute Amplification"
  - "DDIM"
date: 2026-05-08
content_hash: 7f51975d71f8a2fe
---

# Factored Classifier-Free Guidance

**Conference**: ICML 2026  
**arXiv**: [2506.14399](https://arxiv.org/abs/2506.14399)  
**Code**: No public link  
**Area**: Diffusion Models / Counterfactual Generation / Medical Imaging  
**Keywords**: Classifier-Free Guidance, Counterfactual Generation, Causal Intervention, Attribute Amplification, DDIM

## TL;DR
This paper identifies the "attribute amplification" failure mode of CFG in diffusion-based counterfactual generation—where a single global $\omega$ amplifies attributes that should remain unchanged. It proposes FCFG: grouping attributes according to a causal graph and assigning independent guidance weights to each group, significantly reducing off-target attribute drift and improving counterfactual reversibility on CelebA-HQ, EMBED, and MIMIC-CXR.

## Background & Motivation
**Background**: Diffusion models have become the de facto standard for conditional generation. The standard workflow for counterfactual generation is a three-stage process: DDIM inversion (abduction) $\rightarrow$ do-intervention (action) $\rightarrow$ reverse DDIM guided by CFG (prediction). Classifier-Free Guidance interpolates between conditional and unconditional scores via $\epsilon_\text{CFG}=(1-\omega)\epsilon_\theta(\varnothing)+\omega\epsilon_\theta(\mathbf{c})$, serving as a popular knob to ensure the generated image strongly reflects target attributes.

**Limitations of Prior Work**: The weight $\omega$ in CFG is a global scalar acting on the entire condition vector $\mathbf{c}$. In counterfactual scenarios, $\mathbf{c}$ typically encodes multiple attributes (e.g., gender, age, smile). If a user intervenes on only one, they are forced to multiply all attributes by the same $\omega$. Consequently, $do(\text{Male=no})$ might amplify "Smiling," and $do(\text{Young=no})$ might alter identity and expression. This off-target modification violates the invariance axioms of causal graphs, a phenomenon termed attribute amplification.

**Key Challenge**: There is a fundamental tension between "intervention effectiveness (strongly changing target attributes)" and "maintaining the stability of non-target attributes." As long as the guidance is a scalar, these two are inherently coupled. While Xia et al. (2024) attributed this to predictor-finetuning during training, this work argues the guidance mechanism itself is the culprit.

**Goal**: Break the coupling between attributes during inference only, providing independent guidance strengths for each semantic/causal group without modifying the training process or model architecture.

**Key Insight**: If attribute groups are conditionally independent given $\mathbf{x}_t$, i.e., $p(\mathbf{pa}\mid\mathbf{x}_t)=\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)$, then the proxy posterior naturally factorizes into $p^\omega(\mathbf{x}_t\mid\mathbf{pa})\propto p(\mathbf{x}_t)\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)^{\omega_m}$. Here, each group has its own $\omega_m$, making CFG a special case where $M=1$.

**Core Idea**: Rewrite the CFG score update using "attribute-split embeddings" combined with group-assigned $\omega_m$, shifting from global amplification to group-wise fine-grained amplification. This is effective purely at inference time.

## Method

### Overall Architecture
FCFG consists of three components: (i) an "attribute-split" structure where semantic attributes $\mathbf{pa}=(pa_1,\dots,pa_K)$ are embedded via independent MLPs and concatenated as $\mathbf{c}=\text{concat}(\mathcal{E}_1(pa_1),\dots,\mathcal{E}_K(pa_K))$, ensuring each attribute occupies an independent block; (ii) a grouping mechanism at inference time where attributes are divided into $M$ groups (e.g., "affected" vs. "invariant" in counterfactual tasks), and masked embeddings $\underaccent{\rule{4.09723pt}{0.4pt}}{\mathbf{c}}^{(m)}$ are constructed by zeroing out non-group blocks; (iii) an extension of the CFG score difference into a weighted sum of $M$ terms, one $\omega_m$ per group. The full pipeline is embedded into the abduction-action-prediction steps of DDIM counterfactual inference, replacing $\epsilon_\text{CFG}$ with $\epsilon_\text{FCFG}$ only during the prediction stage.

### Key Designs

1.  **Attribute-Split Embedding**:
    - **Function**: Assigns each attribute a dedicated dimension segment in $\mathbf{c}$, facilitating the null-tokenization of specific attribute groups as needed.
    - **Mechanism**: Each $pa_i$ is embedded through an independent MLP $\mathcal{E}_i:\mathbb{R}^{d_i}\to\mathbb{R}^d$, resulting in $\mathbf{c}\in\mathbb{R}^{Kd}$ as a concatenation of blocks. Masking the $i$-th attribute involves multiplying the corresponding block by an indicator $\delta_i^{(m)}\in\{0,1\}$. All $\mathcal{E}_i$ are trained end-to-end with the denoising network rather than being pre-trained features.
    - **Design Motivation**: Conventional designs often mix attributes into a dense vector, leading to semantic entanglement. Attribute-split naturally decouples them, providing the architectural foundation for group-wise guidance.

2.  **Group-wise Factored Score**:
    - **Function**: Provides independent guidance intensities for different attribute groups, breaking the global coupling of CFG.
    - **Mechanism**: Assuming conditional independence between groups $p(\mathbf{pa}\mid\mathbf{x}_t)=\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)$, the derived proxy posterior is $p^\omega(\mathbf{x}_t\mid\mathbf{pa})\propto p(\mathbf{x}_t)\prod_m p(\mathbf{pa}^{(m)}\mid\mathbf{x}_t)^{\omega_m}$. The corresponding score is $\epsilon_\text{FCFG}=\epsilon_\theta(\varnothing)+\sum_m \omega_m(\epsilon_\theta(\underaccent{\rule{4.09723pt}{0.4pt}}{\mathbf{c}}^{(m)})-\epsilon_\theta(\varnothing))$. Standard CFG is recovered when $M=1$, while $M=K$ allows independent weights for every attribute.
    - **Design Motivation**: Global $\omega$ implicitly assumes all attributes are conditionally independent and share the same weight. Relaxing the "same weight" constraint leads to FCFG, which is theoretically closer to the causal graph while requiring only inference-time changes.

3.  **Causal-Oriented Affected/Invariant Grouping**:
    - **Function**: Ground the abstract attribute grouping—dividing attributes into "affected" (the intervened attribute and its descendants) and "invariant" (all others) based on a user-defined causal graph.
    - **Mechanism**: In a typical $do(A)$ scenario, $\omega_\text{aff}$ is set high (e.g., 2.5) to reinforce the target change, while $\omega_\text{inv}$ is kept near 1 (no amplification) to stabilize non-target attributes. This dual-grouping maintains effectiveness while eliminating drift.
    - **Design Motivation**: Directly corresponds to the counterfactual axiom that "attributes outside the intervention influence should remain stable," reducing off-target metrics near zero without sacrificing target effectiveness.

### Loss & Training
The training objective follows the standard conditional diffusion loss $\mathbb{E}\|\epsilon-\epsilon_\theta(\mathbf{x}_t,t,\mathbf{c})\|^2$ with classic classifier-free dropout (randomly replacing the entire $\mathbf{c}$ with $\varnothing$). No new loss terms are introduced. FCFG modifies only the score calculation during inference. While this introduces a slight train-test mismatch (training sees all null tokens, inference sees partial masks), no stability issues were observed. FCFG can also be combined with advanced guidance like CFG++ or APG.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | CFG | FCFG | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CelebA-HQ 64x64 | $do(\text{Smiling})$ | $\Delta$ target $\uparrow$ / $\Delta$ off-target $\downarrow$ | High target, high off-target | High target, off-target $\approx 0$ | Key off-target suppression |
| CelebA-HQ | $do(\text{Smiling})$ | Reversibility MAE/LPIPS $\downarrow$ | Sharp increase with $\omega$ | Significantly lower at same $\omega$ | Improved identity preservation |
| EMBED 192x192 (Breast) | $do(\text{circle})$ | $\Delta$ density (off-target) | Significant increase | Near 0 | Avoids false feature amplification |
| MIMIC-CXR | $do(\text{finding})$ | $\Delta$ race/sex (off-target) | Obvious drift | Substantial suppression | High clinical fairness value |
| MIMIC-CXR | $do(\text{finding})$ | $\Delta$ target AUC | +18.8 | +18.8 (Ours) vs CFG +X | Off-target drift reduced by 10x |

### Ablation Study

| Configuration | Effect | Note |
| :--- | :--- | :--- |
| $M=1$ (Vanilla CFG) | Attribute amplification observed | Confirms FCFG is a strict generalization |
| Two-group affected/invariant ($M=2$) | Best effectiveness/off-target trade-off | Default configuration |
| Per-attribute independence ($M=K$) | Supports $do(\text{S,M,Y})$ multi-interventions | Necessary when all attributes are intervened |
| FCFG + CFG++ / FCFG + APG | Layered on advanced guidance | Also improves off-target amplification |
| vs. SA-DCG / HVAE / HVAE-soft | Target +13.1 / off-target -1.5 (CelebA-HQ) | Superior to SA-DCG (+12.9 / +3.0) |

### Key Findings
- **Root of Attribute Amplification**: Controlled experiments on CelebA-HQ show that amplification is not due to dataset artifacts or causal graph mismatch but the guidance mechanism itself, shifting the responsibility from data/models to the inference algorithm.
- **FID Benefits**: Although multi-group scores might seem unstable, FCFG yields significantly better FID on CelebA-HQ than global CFG, suggesting that reducing off-target drift helps latent trajectories stay on the data manifold.
- **Counterfactual Reversibility**: Performing $do(A)$ then $do(A^{-1})$ leads to deteriorating MAE/LPIPS for CFG due to residual off-target drift. FCFG maintains initial fidelity, serving as a robust metric for counterfactual soundness.
- **Multi-Attribute Extremes**: When all attributes are intervened simultaneously, $M=2$ grouping collapses to global CFG. The only solution is $M=K$ per-attribute FCFG.

## Highlights & Insights
- Factorizing the "global $\omega$" into a "vector $\omega_m$ based on causal groups" is an intuitive yet previously overlooked extension. The derivation from the proxy posterior to the score formula is mathematically clean.
- The attribute-split embedding is a lightweight design that enables flexible grouping at inference time, effectively "pre-installing" a mask interface for any conditional diffusion framework.
- Evaluates counterfactual generation through the lens of "intervention effectiveness vs. reversibility," which is more aligned with causal axioms than FID alone. This framework is applicable to video editing and 3D consistency.
- Compatibility with CFG++ / APG suggests that factorization is a dimension orthogonal to score refinement—future conditional sampling improvements can likely benefit from "factorizing first."

## Limitations & Future Work
- Relies on pre-specified causal graphs or semantic groupings; FCFG does not solve causal discovery. Incorrect grouping might exacerbate problems.
- $\omega_m$ still requires manual tuning. Future work could explore adaptive $\omega$ selection based on input conditions or timesteps (timestep-aware FCFG).
- A mild train-test mismatch exists: training uses all-null tokens, while inference uses group-wise masks. Stability at very large $M$ or extreme $\omega$ is a potential concern.
- The $M=2$ grouping collapses under simultaneous intervention of all attributes, revealing a vulnerability in the heuristic grouping approach.
- Maximum experimental resolution is 192x192; validity on high-resolution latent diffusion (SDXL) or video diffusion remains to be verified.

## Related Work & Insights
- **vs. Standard CFG (Ho & Salimans 2022)**: FCFG is a strict generalization, equivalent when $M=1$.
- **vs. CFG++ (Chung 2025) / APG (Sadat 2025)**: These refine score shapes or manifold constraints for fidelity but retain a global $\omega$; FCFG is orthogonal and complementary.
- **vs. Compositional Diffusion (Liu 2022)**: Those methods rely on spatial masks or multiple conditional models for localized control; FCFG uses a single model and semantic grouping.
- **vs. HVAE / HVAE-soft (Ribeiro 2023; Xia 2024)**: These fix attribute amplification during training via predictor-finetuning. FCFG addresses it at the inference end, keeping training unchanged.
- **vs. SA-DCG (Rasal 2025)**: Uses diffusion autoencoders with heavy identity preservation optimization. FCFG achieves lower off-target drift and better FID with less complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple idea that addresses a critical flaw; a natural extension of the CFG formula.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various datasets and compares against HVAE/SA-DCG/CFG++/APG, though lacks high-res latent diffusion validation.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivation, failure modes are well-quantified via $\Delta$ metrics.
- Value: ⭐⭐⭐⭐ Plug-and-play applicability with direct value for medical counterfactual reasoning and fairness assessments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](../../CVPR2026/medical_imaging/prototypebased_knowledge_guidance_for_finegrained.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](../../ICCV2025/medical_imaging/multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[AAAI 2026\] qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs](../../AAAI2026/medical_imaging/qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)

</div>

<!-- RELATED:END -->
