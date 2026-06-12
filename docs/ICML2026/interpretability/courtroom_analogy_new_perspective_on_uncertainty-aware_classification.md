---
title: >-
  [Paper Note] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification
description: >-
  [ICML 2026][Interpretability][Uncertainty Quantification] This paper proposes a "courtroom analogy" perspective, modeling the second-order uncertainty of classification as a structured mixture of $K$ class advocate Diric…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Uncertainty Quantification"
  - "Evidential Deep Learning"
  - "Dirichlet Mixture"
  - "Single-forward UQ"
date: 2026-05-08
content_hash: 37e6d79c9f9785aa
---

# Courtroom Analogy: New Perspective on Uncertainty-Aware Classification

**Conference**: ICML 2026  
**arXiv**: [2605.25616](https://arxiv.org/abs/2605.25616)  
**Code**: TBD  
**Area**: interpretability  
**Keywords**: Uncertainty Quantification, Evidential Deep Learning, Dirichlet Mixture, Interpretability, Single-forward UQ

## TL;DR
This paper proposes a "courtroom analogy" perspective, modeling the second-order uncertainty of classification as a structured mixture of $K$ class advocate Dirichlet opinions under input-dependent weights. It is instantiated as the MoDEX network (comprising three lightweight heads: shared evidence $\bm{\alpha}$, class-specific advocacy strength $\tau_k$, and credibility $\bm{\omega}$). MoDEX consistently outperforms a series of baselines like EDL and $\mathcal{F}$-EDL across benchmarks including CIFAR/SVHN/TIN/CIFAR-10-C/CIFAR-10-LT with a single forward pass, providing semantically clear uncertainty decomposition.

## Background & Motivation
**Background**: Single-forward second-order UQ methods, represented by EDL (Sensoy et al., 2018), model classification uncertainty as a distribution $q \in \mathcal{Q}$ over the category probability vector, usually choosing the Dirichlet family. These methods provide closed-form predictive mean/variance and interpret concentration parameters as evidence. Subsequent works like $\mathcal{I}$-EDL, R-EDL, Re-EDL, and $\mathcal{F}$-EDL follow this path.

**Limitations of Prior Work**: The mainstream optimization direction in this line of research is "increasing expressivity"—either by switching to more flexible distribution families or by relaxing the original EDL assumptions. While $\mathcal{Q}$ becomes increasingly capable of fitting complex uncertainty patterns, **the mechanism of how uncertainty is "formed" and "aggregated" remains a black box**. After obtaining the Dirichlet $\bm{\alpha}$, it can only be interpreted as "total evidence," offering almost no semantic explanation as to "why the model hesitates" or "where the hesitation comes from" for a specific image.

**Key Challenge**: There is no bridge between expressivity and structural interpretability. Simply stacking the capacity of $\mathcal{Q}$ does not inform users about the internal structure of uncertainty (is it due to lack of evidence or conflicting interpretations between different classes?), which is precisely the most valuable part of UQ in high-risk scenarios.

**Goal**: Design a framework that retains the excellent properties of single-forward passes, closed-form moments, and Dirichlet-based optimization, while explicitly encoding the **formation mechanism** of uncertainty into the structure of $\mathcal{Q}$.

**Key Insight**: The authors start from an intuitive analogy—viewing classification as a courtroom debate. Each class corresponds to an advocate; all advocates observe the same case evidence $\mathbf{x}$ but form different probability beliefs based on different focuses. The final judgment is the result of aggregating these beliefs weighted by "credibility." This metaphor naturally distinguishes three sources of uncertainty: (i) insufficient evidence, (ii) inconsistent interpretations of the same evidence by advocates, and (iii) which advocate is more credible.

**Core Idea**: Use a "shared base evidence + class-specific advocacy increment" approach to structurally decompose $K$ Dirichlet opinions, then mix them into $\mathcal{Q}$ using input-related credibility weights. this results in a second-order distribution with $\mathcal{O}(K)$ parameters and courtroom semantics for each parameter, which happens to be equivalent to the Extended Flexible Dirichlet (EFD) proposed by Ongaro et al.

## Method

### Overall Architecture
The input $\mathbf{x}_i$ passes through a feature extractor $f_{\bm{\psi}}$ to obtain representation $\mathbf{z}_i$. Three lightweight prediction heads then output courtroom parameters in parallel: shared evidence $\bm{\alpha}(\mathbf{x}_i)\in\mathbb{R}_{>0}^K$, credibility weights $\bm{\omega}(\mathbf{x}_i)\in\Delta^{K-1}$, and class-specific advocacy strength $\bm{\tau}(\mathbf{x}_i)\in\mathbb{R}_{>0}^K$. These parameters jointly define an EFD distribution $p(\bm{\pi}_i\mid\mathbf{x}_i)=\sum_k \omega_k(\mathbf{x}_i)\,\mathrm{Dir}(\bm{\pi}_i\mid\bm{\alpha}(\mathbf{x}_i)+\tau_k(\mathbf{x}_i)\mathbf{e}_k)$. During prediction, the first moment is obtained from the EFD in closed form to get $\hat{p}(y^\star=k\mid\mathbf{x}^\star)$, and the argmax yields the label. Simultaneously, aleatoric and epistemic uncertainties are output using the first-order entropy and the second-order trace-covariance, respectively. The entire process is strictly single-forward without requiring sampling or multiple models.

### Key Designs

1. **Courtroom generative process**:
    - **Function**: Explicitly models "classification uncertainty" as an input-dependent mixture of $K$ Dirichlet advocate opinions $p(\bm{\pi}\mid\mathbf{x})=\sum_k \omega_k(\mathbf{x})\mathrm{Dir}(\bm{\pi}\mid\bm{\alpha}_k(\mathbf{x}))$, where each component corresponds to a class advocate's belief about the true probability vector.
    - **Mechanism**: Introduces a latent variable $L\sim\mathrm{Cat}(\bm{\omega}(\mathbf{x}))$ to select an advocate. Given $L=k$, $\bm{\pi}\sim\mathrm{Dir}(\bm{\alpha}_k(\mathbf{x}))$, and the label is generated by $y\sim\mathrm{Cat}(\bm{\pi})$. Marginalizing out $L$ yields the structured second-order distribution mentioned above.
    - **Design Motivation**: Binds three heterogeneous sources—"insufficient evidence," "disagreement between advocates," and "advocate credibility"—to independent mechanisms: the internal variance of Dirichlet, the difference between components, and the mixture weight $\bm{\omega}$. This transforms $\mathcal{Q}$ from a "bucket for uncertainty" into a structured distribution that explains the source of uncertainty.

2. **Structured decomposition $\bm{\alpha}_k(\mathbf{x})=\bm{\alpha}(\mathbf{x})+\tau_k(\mathbf{x})\mathbf{e}_k$**:
    - **Function**: Decomposes each advocate's concentration into "shared base evidence $\bm{\alpha}(\mathbf{x})$ + advocacy strength $\tau_k(\mathbf{x})\mathbf{e}_k$ added only to their own class," allowing the $K$ Dirichlet components to share a backbone while protruding in their respective dimensions.
    - **Mechanism**: Naively providing an independent $K$-dimensional concentration for each component would require $\mathcal{O}(K^2)$ parameters. This decomposition reduces the parameter count to $\mathcal{O}(K)$ and makes the distribution equivalent to the EFD family (Ongaro et al., 2020), allowing the use of closed-form moment formulas for single-forward prediction and uncertainty calculation.
    - **Design Motivation**: Explicitly decouples "objective facts" from "subjective advocacy" via inductive bias—$\bm{\alpha}$ carries evidence seen by all, while $\tau_k$ carries the extra push by advocate $k$ for their own class. This ensures each learned parameter has clear courtroom semantics and provides the theoretical foundation for splitting EU into inter-expert and intra-expert components.

3. **Three-head network + Compound loss + Dual uncertainty metrics**:
    - **Function**: Predicts $(\bm{\alpha},\bm{\omega},\bm{\tau})$ using three logit heads (concentration / gating / advocacy) with exp/softmax activations. The training loss is the sum of MSE on the predictive mean, Brier regularization for $\bm{\omega}$, and KL regularization supervising $\sigma^{\text{SM}}(\bm{\tau})$ with label-smoothed one-hot vectors. Aleatoric uncertainty (AU) and epistemic uncertainty (EU) are quantified at test time via predictive entropy and covariance trace, respectively.
    - **Mechanism**: Spectral normalization is added to $f_{\bm{\psi}}$ and the concentration head to stabilize UQ. KL regularization provides soft supervision for $\bm{\tau}$ ("the correct class should advocate harder"), while Brier regularization prevents gating from collapsing into one-hot. EU can be provably decomposed into $\mathrm{EU}_{\text{inter}}=\sum_k\omega_k\|\bm{\mu}^{(k)}-\bar{\bm{\mu}}\|_2^2$ (inter-expert disagreement) and $\mathrm{EU}_{\text{intra}}=\sum_k\omega_k\sum_j\mathrm{Var}_{\bm{\pi}\sim\mathrm{Dir}(\bm{\alpha}_k)}[\pi_j]$ (intra-expert evidence deficiency).
    - **Design Motivation**: The MSE+regularization combination inherits mature training practices from the EDL series while avoiding instability from direct EFD likelihood maximization. The decomposability of EU translates "semantics" into specific readable numbers, identifying whether a high EU is driven by disagreement or lack of evidence.

### Loss & Training
$$\mathcal{L}=\|\mathbf{y}-\mathbb{E}_{\bm{\pi}\sim\mathrm{EFD}}[\bm{\pi}]\|_2^2+\|\mathbf{y}-\bm{\omega}\|_2^2+D_{\mathrm{KL}}(\sigma^{\text{SM}}(\bm{\tau})\,\|\,\tilde{\mathbf{y}})$$
The first term is MSE for predictive mean alignment, the second is Brier regularization for $\bm{\omega}$ calibration, and the third is label-smoothed KL for $\bm{\tau}$ soft supervision. Label smoothing $\epsilon \in [0,1]$ controls the hardness of $\tilde{\mathbf{y}}$. Spectral normalization is used to enhance UQ robustness during end-to-end training.

## Key Experimental Results

### Main Results
Evaluation tasks include: ID test accuracy, misclassification detection (Miscl. AUPR, aleatoric), OOD detection (AUPR, epistemic), CIFAR-10-C distribution shift detection, and CIFAR-10-LT long-tail robustness. Baselines include Dropout, EDL, $\mathcal{I}$-EDL, R-EDL, DAEDL, Re-EDL, and $\mathcal{F}$-EDL.

| Dataset | Metric | $\mathcal{F}$-EDL (Prev. SOTA) | MoDEX (Ours) | Gain |
|--------|------|-----------------------------|-------|------|
| CIFAR-10 ID | Test Acc | 91.19 | **92.46** | +1.27 |
| CIFAR-10 | Miscl. AUPR (aleatoric) | 99.10 | **99.18** | +0.08 |
| CIFAR-10 → SVHN / C-100 | OOD AUPR | 91.20 / 88.37 | **91.58 / 89.28** | +0.38 / +0.91 |
| CIFAR-100 ID | Test Acc | 69.40 | **75.91** | +6.51 |
| CIFAR-100 | Miscl. AUPR | 94.01 | **96.17** | +2.16 |
| CIFAR-100 → SVHN / TIN | OOD AUPR | 75.35 / 80.58 | **77.90 / 81.76** | +2.55 / +1.18 |
| CIFAR-10-C ($\mathcal{C}{=}5$) | Shift AUPR | 78.52 | **80.63** | +2.11 |
| CIFAR-10-LT ($\rho{=}0.01$) | Test Acc | 63.73 | **71.53** | +7.80 |
| CIFAR-10-LT | OOD SVHN / C-100 | 62.56 / 70.18 | **72.05 / 76.52** | +9.49 / +6.34 |

### Ablation Study
| Configuration / Property | Behavior | Description |
|-------------|------|------|
| Full MoDEX | Best across all | Shared $\bm{\alpha}$ + class-specific $\tau_k$ + input-dependent $\bm{\omega}$ |
| $\tau_k\equiv\tau$ (Single advocacy strength) | Degenerates to $\mathcal{F}$-EDL (Thm 5.1) | Loses structural differences between advocates |
| $\tau=1$ and $\bm{\omega}=\bm{\alpha}/\|\bm{\alpha}\|_1$ | Degenerates to EDL (Thm 5.1) | Reverts to original evidential baseline |
| EU Decomposition (Prop 5.4) | $\mathrm{EU}=\mathrm{EU}_{\text{inter}}+\mathrm{EU}_{\text{intra}}$ | Distinguishes "advocate disagreement" vs "insufficient evidence" |
| Equivalent Representation (Thm 5.3) | Weighted ensemble of $K$ EDL experts | Two inference perspectives for the same model |

### Key Findings
- **Performance gains increase with class count and long-tail distribution**: Significant improvement of 6.5 points on CIFAR-100 and up to +7.8 accuracy and +9.5 OOD AUPR in long-tail settings. This suggests that the decoupling of $\bm{\alpha}$ vs $\tau_k$ is not just "interpretability candy" but a substantive mechanism that allows minority class advocates to express themselves when head classes are dominant.
- **Structure > Expressivity**: Compared to $\mathcal{F}$-EDL (a more flexible single distribution family), MoDEX's similarly single-forward mixture structure is stronger in almost all UQ tasks, validating the author's argument that structural inductive bias is key.
- **Monotonic improvement with distribution shift severity**: MoDEX maintains its lead as shift severity moves from $\mathcal{C}=1$ to $\mathcal{C}=5$ (increasing AUPR gap from +0.56 to +2.11), indicating that epistemic metrics are sensitive to shifts and well-calibrated.
- **Visualizing inter/intra EU decomposition**: The authors show that on clean ID data, EU is primarily intra-expert (lack of evidence), while for OOD or ambiguous inputs, inter-expert (advocate disagreement) weight increases significantly, providing human-readable explanations for uncertainty.

## Highlights & Insights
- A Reframing from "increasing expressivity" to "adding structural semantics"—the authors leverage a structured decomposition of the Dirichlet family to achieve EFD equivalence and EU decomposability, providing a unified perspective that covers EDL and $\mathcal{F}$-EDL.
- The "courtroom" analogy is not just marketing: $\bm{\alpha}$, $\tau_k$, and $\bm{\omega}$ correspond to case evidence, advocacy strategy, and judicial conviction. With each experiment explained by the analogy, interpretability is built-in by design.
- Dual equivalence representations (ensemble of EDL experts vs base-EDL + softmax mixture) inspire a general pattern: rewriting "ensemble models" as mixtures of "main branches + correction branches," which has potential for knowledge distillation and MoE LLM reliability modeling.
- The EU decomposition $\mathrm{EU}_{\text{inter}}+\mathrm{EU}_{\text{intra}}$ provides independent semantic answers to why the model hesitates. This can be used as a signal for active learning: high intra-EU suggests adding more data, while high inter-EU suggests adding better labels or re-evaluating annotations.

## Limitations & Future Work
- The paper only validates on medium-scale vision datasets (CIFAR, SVHN, TIN). Whether the structural bias remains superior for ImageNet-scale, NLP text classification, or multi-label scenarios remains to be tested.
- The weights of the three loss terms were set empirically; a systematic sensitivity analysis is missing. The impact of label smoothing $\epsilon$ on long-tail results seems significant but is not detailed.
- Computational cost: While single-forward, MoDEX uses two additional $K$-dimensional heads and EFD moment calculations. The overhead of spectral normalization and inference latency for long sequences needs more detailed benchmarking.
- "Advocates" are currently limited to $K$ (one per class). A natural extension would be hierarchical courtrooms (super-class selection then fine-grained) or multiple advocates per class for fine-grained/hierarchical labels.
- Lack of a controllable interface for translating the inter/intra-EU semantics into actionable advice for end-users like doctors or legal experts.

## Related Work & Insights
- **vs EDL (Sensoy 2018) / $\mathcal{I}$-EDL / R-EDL / Re-EDL**: These methods focus on single Dirichlets (adding Fisher information, relaxing assumptions). MoDEX subsumes them as special cases (Thm 5.1) and adds the interpretable EU decomposition.
- **vs $\mathcal{F}$-EDL (Yoon & Kim 2026)**: $\mathcal{F}$-EDL seeks a "more flexible distribution," whereas MoDEX uses structural inductive bias. MoDEX outperforms it in most tasks, quantifying the difference via the $\tau_k\equiv\tau$ ablation.
- **vs Bayesian/Deep Ensembles (Lakshminarayanan 2017)**: Ensembles rely on multiple forward passes. MoDEX "internalizes" a $K$-expert ensemble within a single pass via the mixture-of-experts perspective, offering efficiency and better interpretability.
- **vs Deterministic / Distance-aware UQ (DUQ, SNGP)**: These map uncertainty to feature-space distance but lack second-order distributions over probability vectors. MoDEX retains second-order semantics while adopting stability techniques like spectral normalization.
- **vs Subjective Logic / Dempster-Shafer**: This work represents a clean modernization of traditional formal logic ideas (opinions + credibility aggregation) within end-to-end learnable neural networks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Original framework (courtroom analogy + structured decomposition), though fundamentally an extension of the Dirichlet family.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers ID/OOD/Shift/Long-tail tasks with consistent leads; lacks ImageNet scale and loss weight ablation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The analogy is consistent throughout. Theorem-proposition-case analysis is rigorous, and every parameter is well-explained.
- **Value**: ⭐⭐⭐⭐ Interpretable UQ is essential for high-risk deployment. The EU decomposition and degradation analysis provide direct inspiration for follow-up work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)
- [\[ICML 2026\] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)
- [\[ICCV 2025\] "Principal Components" Enable A New Language of Images](../../ICCV2025/interpretability/principal_components_enable_a_new_language_of_images.md)
- [\[ICML 2026\] OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization](omnisapiens_a_foundation_model_for_social_behavior_processing_via_heterogeneity-.md)
- [\[ACL 2026\] Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective](../../ACL2026/interpretability/investigating_more_explainable_and_partition-free_compositionality_estimation_fo.md)

</div>

<!-- RELATED:END -->
