---
title: >-
  [Paper Note] The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench
description: >-
  [ICML 2026][Remote Sensing][Scientific Alignment] The authors point out that Vision Foundational Models (VFMs) "seem" proficient at predicting from satellite images but collapse along physical axes in extreme physical re…
tags:
  - "ICML 2026"
  - "Remote Sensing"
  - "Scientific Alignment"
  - "Structural Isomorphism"
  - "Vision Foundational Models"
  - "Tropical Cyclone"
  - "Probing Evaluation"
date: 2026-05-08
content_hash: 801c018a56bcce57
---

# The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench

**Conference**: ICML 2026  
**arXiv**: [2605.24782](https://arxiv.org/abs/2405.24782)  
**Code**: https://github.com/CausalLearningAI/tc-bench  
**Area**: Remote Sensing / Evaluation of Vision Foundational Models / Representation Learning Diagnostics  
**Keywords**: Scientific Alignment, Structural Isomorphism, Vision Foundational Models, Tropical Cyclone, Probing Evaluation

## TL;DR
The authors point out that Vision Foundational Models (VFMs) "seem" proficient at predicting from satellite images but collapse along physical axes in extreme physical regimes. Consequently, this work formalizes the concept of "Scientific Alignment" via "Structural Isomorphism" and releases TC-Bench, a global tropical cyclone benchmark. Through a three-layered suite of linear probes (Static, Dynamic, and Constrained), the authors systematically reveal representation collapse in frozen backbones like DINO, CLIP, SigLIP, and MAE during intense cyclone regimes ($P_c < 980$ hPa).

## Background & Motivation
**Background**: Migrating general-purpose VFMs to scientific scenarios (meteorology, ecology, medicine) is a growing trend. The industry default metrics are in-distribution accuracy and cross-domain out-of-distribution (OOD) accuracy (e.g., changes in geographic regions or observation agencies). If a VFM maintains predictive power on OOD data, it is interpreted as having learned "invariant physical structures." Benchmarks like Digital Typhoon generally follow this "average case + OOD" evaluation paradigm.

**Limitations of Prior Work**: Tropical cyclone samples naturally cluster around moderate intensities ($P_c$ near 1000 hPa). Average errors mask the high-risk, intense cyclone regimes. Inter-agency OOD primarily perturbs visual appearance (banding conventions, reporting habits) but does not perturb physical axes. Thus, "stable OOD performance" might stem from "stable visual features" rather than "stable physical coordinates." In other words, existing evaluations fail to distinguish between perceptual robustness and physical utility.

**Key Challenge**: When visual signals saturate (e.g., eye-wall morphologies of intense cyclones look highly similar), visual variance $\to 0$, while physical variables (minimum central pressure $P_c$, maximum sustained wind speed $V_m$) can still vary significantly. This constitutes a paradox where "Perception $\approx$ Invariant / Physics $\neq$ Invariant," which the authors name the **Perception–Physics Paradox**.

**Goal**: To decompose this issue into two parts: (i) find a minimal falsifiable definition for "Scientific Alignment"; (ii) provide a testable diagnostic protocol and a fair benchmark to identify when, where, and why VFMs collapse.

**Key Insight**: Borrowing from Causal Representation Learning (CRL) but weakening the requirements—not demanding coordinate-wise disentanglement, but requiring a unique linear mapping from latent space to physical space, termed **structural isomorphism**. This weak condition corresponds to "recoverability via linear probes," making it engineering-testable.

**Core Idea**: Use the condition that "latent representations can be mapped back to physical state space by the same linear decoder across all regimes with uniformly bounded residuals" as a necessary condition for scientific alignment. This condition is hierarchicalized into three linear probes—Static Fidelity, Dynamic Consistency, and Manifold Constraints—to measure failures of VFMs in intense regimes.

## Method

### Overall Architecture
The pipeline consists of four components: (1) Formal definition of the "physical system $\mathcal{S}$" and the "representation $\mathbf{z}=g(\mathbf{x})$"; (2) Definition of structural isomorphism $\mathbf{z}=\mathbf{A}\mathbf{y}+\epsilon_{\mu}(\mathbf{y})$ (linear mapping $\mathbf{A}$ + bounded residual), proving that it implies three population-level error bounds (Static, Dynamic, Constraint); (3) Instantiation of each error bound as uniform residuals on a "constrained proxy $h\in\mathcal{H}$" ($V(g)=\inf_h \sup_{\mu} \mathbb{E}[\psi(h(\mathbf{z}),\mathbf{r})]$)—the Structural Alignment Probes $\mathcal{Q}$; (4) Running linear probes and geometric diagnostics on frozen backbones in TC-Bench to pinpoint failure modes. Input consists of $224 \times 224$ infrared satellite images, and output consists of regime-stratified probe residual curves with confidence intervals.

### Key Designs

1.  **Structural Isomorphism as the Weakest Falsifiable Condition for Scientific Alignment**:
    *   **Function**: Tightens the ambiguous question of "whether model representations can serve as physical states" into a verifiable geometric statement.
    *   **Mechanism**: For any regime $\mu\in\mathcal{M}$, there exists an injective linear mapping $\mathbf{A}\in\mathbb{R}^{d\times m}$ such that $\mathbf{z}=\mathbf{A}\mathbf{y}+\epsilon_{\mu}(\mathbf{y})$, where the residual and its Jacobian are uniformly bounded: $\sup_\mu \mathbb{E}[\|\epsilon_\mu\|]\le\bar{\epsilon}$ and $\sup_\mu \mathbb{E}[\|J_{\epsilon_\mu}\|]\le\bar{\delta}$. Proposition 2.1 derives three uniform error bounds: **Static Fidelity** $\sup_\mu \mathbb{E}\|\mathbf{y}-L\mathbf{z}\|\le\|L\|\bar{\epsilon}$, **Dynamic Consistency** $\sup_\mu \mathbb{E}\|\dot{\mathbf{y}}-L\dot{\mathbf{z}}\|\le\|L\|\bar{\delta}K$, and **Manifold Constraint** $\sup_\mu \mathbb{E}\|\mathcal{P}(L\mathbf{z})\|\le\Lambda_{\mathcal{P}}\|L\|\bar{\epsilon}$, where $L$ is the same left-inverse decoder and $K$ is the upper bound of the physical vector field. Theorem 2.1 further shows that this alignment automatically yields an $n$-step intervention replay error $\epsilon_{\text{int}}(n)\le\epsilon_{\text{stat}}+\epsilon_{\text{dyn}}(t_n-t^*)$, linking representation geometry to interventional causal consistency.
    *   **Design Motivation**: CRL usually requires coordinate-wise identifiability, which is a strong assumption and almost impossible to verify on large-scale observational data. Structural isomorphism allows for distributed representations while ensuring uniqueness and interventional consistency, serving as the weakest necessary condition for "deployment-ready measurement."

2.  **Structural Alignment Probes $\mathcal{Q}=(\mathcal{Z},\mathcal{R},\mathcal{H},\psi)$**:
    *   **Function**: Translates abstract error bounds into concrete tests specifying the head, data, and metrics, while restricting proxy functions to the linear family to prevent "cheating" via decoder expressivity.
    *   **Mechanism**: $\mathcal{Q}_{\text{stat}}$ uses $\xi_{\text{stat}}=\|h(\mathbf{z})-P_c\|/\sigma(P_c)$ (normalized to mean-baseline=1) to test physical state recoverability; $\mathcal{Q}_{\text{dyn}}$ uses a 3-hour finite difference $\xi_{\text{dyn}}=\|L\Delta\mathbf{z}_t-\Delta\mathbf{y}_t\|$ to test time-derivative consistency; $\mathcal{Q}_{\text{con}}$ uses monotonic constraints between low vs. high latitude bands of $\Delta V_m$ (derived from gradient wind balance where $f\propto\sin\phi$) to test physical coupling. All probes are trained on **regime-balanced** subsets with the same linear head and stratified by $P_c=980$ hPa into Moderate/Intense categories.
    *   **Design Motivation**: Forcing linear heads decouples the "information presence" from "head expressivity." If a linear probe fails, it indicates physical variables are not explicitly encoded in a linear subspace. Regime-balanced slicing excludes sample imbalance as a shortcut explanation.

3.  **TC-Bench + Geometric Diagnostics of Failure Modes**:
    *   **Function**: Provides the first reproducible, versioned, global tropical cyclone benchmark covering all major basins (IBTrACS v4r01 + GridSat-B1 IR, 1980–2024, 3-hour steps, $224\times 224$ patches, 2601 cleaned trajectories), along with geometric analysis of latent spaces for strong backbones like DINOv3.
    *   **Mechanism**: Within $P_c$ bins where $N\ge 500$, three metrics are calculated: (a) Relationship between PCA PC1 and $P_c$; (b) Effective dimension $d_{\text{eff}}=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$; (c) Mean pairwise distance of centralized features (feature spread). Simultaneous drops in these metrics for $P_c < 980$ hPa identify "latent space collapse along the physical axis" as the root cause. A pixel-level supervised baseline trained from scratch proves that $P_c$ signals in intense regimes are physically recoverable, attributing failure to VFM representation geometry rather than task difficulty.
    *   **Design Motivation**: Prevents the counter-argument "Did the probe fail because the task was inherently hard or because the space collapsed?" It also provides a unified data foundation for future research in tropical cyclone representation learning.

### Loss & Training
Backbones are not trained during the evaluation phase. The CLS token of each VFM (DINOv2/v3, CLIP, SigLIP/2, MAE, and in Appendix: VideoMAE, V-JEPA2, X-CLIP) is used as the representation $\mathbf{z}$. Only a linear head $h$ (least squares) is trained downstream, using trajectory-level splits to prevent spatiotemporal leakage. Appendix E includes four sets of ablations using MLP/Transformer probes, spatial mean pooling, pixel baselines, and video backbones to ensure the phenomenon is independent of the probe family or aggregation method.

## Key Experimental Results

### Main Results: Probing Performance in Moderate vs. Intense Regimes

| Probe | Moderate ($P_c \ge 980$ hPa) | Intense ($P_c < 980$ hPa) | Catastrophic ($P_c < 920$ hPa) | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{Q}_{\text{stat}}$ ($\xi_{\text{stat}}$ median) | Consistently low, low variance (well below 1.0) | Median and variance rise synchronously; catastrophic samples increase | — | Consistent across 6 VFM families; not model-specific |
| $\mathcal{Q}_{\text{dyn}}$ ($\xi_{\text{dyn}}$) | Stable | Monotonic increase | Sharp peaks | Representation time-derivative deviates from physical derivative |
| $\mathcal{Q}_{\text{con}}$ ($\psi_{\text{con}}$) | $\approx 20\%$ | $\approx 55\%$ | — | Increased violation of low/high latitude wind speed ranking |

### Ablation Study: DINOv3-base Latent Space Geometry

| Configuration / Bin | PC1 vs. $P_c$ Slope | Effective Dimension $d_{\text{eff}}$ | Feature Spread | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Moderate bin ($P_c \ge 980$) | Clearly monotonic | Baseline | Baseline | Physical changes resolved along main components |
| Intense bin ($P_c < 980$) | Significant compression | $\downarrow$ approx. 60% | Synchronous drop | Multiple latent directions collapse into a few primary ones |
| Pixel-supervised baseline | — | — | — | **Lower** error than frozen VFMs; proves signal is recoverable |
| Video Backbones (App. E.4) | Similar collapse | — | — | Failure is not exclusive to static-image pre-training |

### Key Findings
*   Failure across all three probes occurs simultaneously across 6 backbone families (DINOv2/3, CLIP, SigLIP/2, MAE), indicating a structural issue rather than a bias of a specific self-supervision loss.
*   The moderate-to-intense gap persists even with non-linear MLP/Transformer probes (App. E.2), proving the issue is not the expressivity of the linear head.
*   Pixel-supervised models achieve lower errors in the $P_c < 980$ hPa regime $\to$ the signal exists in the data; the problem lies in the representation geometry. This decouples "scientific alignment failure" from "task difficulty."
*   Expanding OOD evaluation from "agency changes" to "intensity changes" reveals that VFMs previously thought to be OOD-robust collapse entirely—this paper acts as a counter-example report stating "OOD $\not\approx$ Physical Invariance."

## Highlights & Insights
*   **A Named Paradox**: The "Perception–Physics Paradox" label is concise; this framework can be reused in other remote sensing scenarios (wildfire thermal saturation, medical imaging saturation, fluid turbulence scale saturation).
*   **Formalization Converging to a Testable Weak Condition**: Previous CRL concepts of "alignment" were too strong (identifiability), making them hard to test empirically. This work lowers alignment to "unique linear reparameterization," matching the industry's preference for linear probes and bridging theory with practice.
*   **Diagnostics $\to$ Interventional Causality**: Theorem 2.1 shows that static and dynamic alignment bounds automatically derive an $n$-step intervention replay error bound. This bridges representation geometry and world model interventional consistency, serving as a yardstick for "whether a world model can be used for do-calculus."
*   **Transferable Trick**: The triad of regime-balanced data, forced linear heads, and pixel baselines can be applied to any scientific ML subfield where "OOD evaluation seems OK but physics may collapse."

## Limitations & Future Work
*   The authors acknowledge providing only necessary conditions, not sufficient ones. The scope covers perception-based world models but excludes emulator-based ones (e.g., GenCast, FourCastNet).
*   Counterfactual reasoning is explicitly excluded; the framework only guarantees intervention-level causal consistency—a pragmatic choice for scientific scenarios, though the boundary remains significant.
*   Physical variables are limited to $P_c$ and $V_m$. Differences in wind speed reporting across agencies led $V_m$ to be demoted to auxiliary analysis. Introducing finer physical quantities (radial wind profiles, temperature fields) might expose more failure modes.
*   TC-Bench uses 2D infrared patches, lacking multi-channel (visible, microwave) or 3D structural data. Investigating if multi-modal/multi-view fusion improves probe performance is an obvious follow-up.

## Related Work & Insights
*   **vs. Digital Typhoon (kitamoto2023digital)**: They focus on single-agency, Tokyo-centric, leaderboard-oriented metrics. This work focuses on global multi-agency data, regime-stratified diagnostics, and identifies blind spots in the Digital Typhoon paradigm.
*   **vs. Classical CRL (scholkopf2021toward; von2024nonparametric)**: Classical CRL requires multi-environment interventions and element-wise identifiability (strong assumptions). This work uses structural isomorphism to provide a weaker alignment definition verifiable on observational data, suited for the VFM era.
*   **vs. OOD Generalization Literature**: Previous work assumed "Good OOD accuracy $\to$ Learned invariant structures." This work provides a clear counter-example where high inter-agency OOD robustness coexists with collapse on the physical axis.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Formalizing "Scientific Alignment" into linear-testable conditions and naming the Perception–Physics Paradox is a major contribution to VFM-for-science evaluation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 backbone families $\times$ 3 probes $\times$ regime-stratified analysis + 4 sets of ablations (non-linear probes, pixel baselines, spatial pooling, video backbones) provides comprehensive coverage of potential counter-arguments.
*   Writing Quality: ⭐⭐⭐⭐ Concepts are clear; the theorem-probe-experiment mapping is well-organized. The physical background in the appendix is excellent, while the main text is slightly theoretical.
*   Value: ⭐⭐⭐⭐⭐ TC-Bench and the probing framework are directly applicable to other remote sensing/scientific ML scenarios. It is one of the few works providing concepts, theorems, data, and code simultaneously.

## Related Papers

- [\[CVPR 2025\] DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery](../../CVPR2025/remote_sensing/disciple_learning_interpretable_programs_for_scientific_visual_discovery.md)
- [\[ICML 2025\] Causal Foundation Models: Disentangling Physics from Instrument Properties](../../ICML2025/remote_sensing/causal_foundation_models_disentangling_physics_from_instrument_properties.md)
- [\[ICCV 2025\] Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening](../../ICCV2025/remote_sensing/pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](../../NeurIPS2025/remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening](../../ICCV2025/remote_sensing/pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](../../NeurIPS2025/remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[ICML 2026\] Localized, High-resolution Geographic Representations with Slepian Functions](localized_high-resolution_geographic_representations_with_slepian_functions.md)
- [\[ICML 2026\] Any2Any: Unified Arbitrary Modality Translation for Remote Sensing](any2any_unified_arbitrary_modality_translation_for_remote_sensing.md)

</div>

<!-- RELATED:END -->
