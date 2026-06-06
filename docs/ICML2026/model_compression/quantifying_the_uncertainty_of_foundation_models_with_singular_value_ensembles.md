---
title: >-
  [Paper Note] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles
description: >-
  [ICML 2026][Model Compression][Uncertainty Quantification] Singular Value Ensemble (SVE) expresses "ensemble diversity" purely through different re-weightings of SVD singular values. By freezing the pre-trained left and…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Uncertainty Quantification"
  - "Implicit Ensemble"
  - "Singular Value Fine-tuning"
  - "Parameter-Efficient"
  - "Calibration"
date: 2026-05-08
content_hash: 2bad1034a7d3094d
---

# Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles

**Conference**: ICML 2026  
**arXiv**: [2601.22068](https://arxiv.org/abs/2601.22068)  
**Code**: https://github.com/moturkoglu/Singular-Value-Ensemble  
**Area**: AI Safety / Uncertainty Quantification  
**Keywords**: Uncertainty Quantification, Implicit Ensemble, Singular Value Fine-tuning, Parameter-Efficient, Calibration  

## TL;DR
Singular Value Ensemble (SVE) expresses "ensemble diversity" purely through different re-weightings of SVD singular values. By freezing the pre-trained left and right singular vectors (serving as a shared "knowledge basis") and training an independent set of singular values for each ensemble member, SVE achieves calibration quality close to Deep Ensemble with a parameter overhead of $\lesssim1\%$, bringing uncertainty quantification (UQ) to PEFT-friendly, resource-constrained scenarios.

## Background & Motivation

**Background**: Deep models are increasingly deployed in high-risk scenarios (medical diagnosis, autonomous driving, agricultural decision-making). However, models trained with maximum likelihood are generally overconfident and "do not know what they do not know." Deep Ensemble remains the gold standard for quantifying epistemic uncertainty—averaging $M$ independently trained models—consistently outperforming single-model approximations like MC-Dropout in calibration and OOD detection.

**Limitations of Prior Work**: The training and memory costs of Deep Ensemble grow linearly by $M$ times, which is almost unaffordable for foundation models with billions of parameters (even $M=4$ is prohibitive). Implicit ensembles (BatchEnsemble's rank-1 perturbations, MIMO, FiLM-Ensemble, LoRA-Ensemble) attempt to share the backbone and add only a few parameters per member, but they still require "learning a new set of directions useful for the final prediction from scratch"—these new parameters fail to inherit pre-trained semantic priors. Meanwhile, single-model Bayesian methods (Laplace-LoRA, BLoB, C-LoRA, SNGP) have fewer parameters but require either complex posterior fitting or perform poorly on transformers.

**Key Challenge**: Modern PEFT paradigms (LoRA, Adapter, Prompt-Tuning) have minimized "fine-tuning" costs, whereas "UQ" remains computationally expensive. A structural gap exists between the two.

**Goal**: To provide an implicit ensemble method with $<1\%$ parameter overhead that reuses the "knowledge basis" of pre-trained foundation models while generating diverse predictive distributions among members to measure epistemic uncertainty.

**Key Insight**: The authors leverage an emerging consensus in interpretability and PEFT—"knowledge is organized along linear subspaces in the weight space." SVF (Sun et al., 2022) found that freezing the singular vectors of pre-trained weights and fine-tuning only the singular values is sufficient for various downstream adaptations, as singular values act as "re-weighting factors for pre-trained representations."

**Core Idea**: Since different singular value re-weightings yield functionally distinct models, having each member learn only a set of singular values while sharing pre-trained singular vectors naturally forms an implicit ensemble that preserves pre-trained priors.

## Method

### Overall Architecture
For all linear layers $\mathbf{W}\in\mathbb{R}^{m\times n}$ targeted for ensembling in a pre-trained foundation model, perform SVD: $\mathbf{W}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$. Freeze $\mathbf{U}$ and $\mathbf{V}$ as the "knowledge basis" shared across all members. For each ensemble member $m$, maintain an independent set of trainable singular values $\boldsymbol{\Sigma}^{(m)}$ and a bias $\mathbf{b}^{(m)}$. During the forward pass, the weight for the $m$-th member is $\mathbf{W}^{(m)}=\mathbf{U}\boldsymbol{\Sigma}^{(m)}\mathbf{V}^{\top}$, producing output $\mathbf{y}^{(m)}=\mathbf{W}^{(m)}\mathbf{x}+\mathbf{b}^{(m)}$. All members are trained jointly using the mean cross-entropy loss $\mathcal{L}=\frac{1}{M}\sum_m \mathcal{L}_{\text{CE}}(f^{(m)}(\mathbf{x}),\mathbf{t})$. Each member has an independent classification head. At inference, predictions from $M$ members are averaged following the standard Deep Ensemble approach.

### Key Designs

1. **Shared Singular Vector Basis + Member-Private Singular Values**:
    - **Function**: Compresses the dimension of "ensemble diversity" from "full weight copies" to "different intensity combinations of pre-trained subspaces."
    - **Mechanism**: After decomposing each target weight $\mathbf{W}$ into $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$, the left and right singular vectors are regarded as "semantic directions" based on interpretability research of transformer weights, while singular values represent the relative importance of these directions. Freezing the directions and re-scaling them anisotropically per member allows each member to pull a different "weight profile" from the same set of explanatory directions. Each member only adds $\min(m,n)$ scalars (plus optional bias terms), with parameter overhead $\approx (M-1)\cdot 5/(4d)$. For LLaMA-2-7B with $d\!=\!4096$ and $M\!=\!16$, this remains $\approx 0.2\%$.
    - **Design Motivation**: Avoids the "cold start" problem of learning new parameters from scratch (a pain point for LoRA-Ensemble) while using the choice of "which directions to amplify or suppress" as the sole degree of freedom for diversity. This preserves pre-trained priors while generating functional differences, representing a key insight in extending the SVF paradigm from single-model adaptation to ensemble UQ.

2. **Multiplicative Initialization + Diversity Driven by Joint Training**:
    - **Function**: Ensures that $M$ members learn truly different solutions without introducing extra regularization.
    - **Mechanism**: Small perturbations are applied using $\boldsymbol{\Sigma}^{(m)}=\boldsymbol{\Sigma}\odot(1+\boldsymbol{\epsilon}^{(m)}), \boldsymbol{\epsilon}^{(m)}\!\sim\!\mathcal{N}(\mathbf{0},\sigma_{\text{init}}^2\mathbf{I})$ (default $\sigma_{\text{init}}=0.01$, corresponding to $\sim 1\%$ relative offset), while biases use additive perturbations $\mathbf{b}^{(m)}=\mathbf{b}+\boldsymbol{\eta}^{(m)}$. During joint training, randomness from different mini-batches amplifies this symmetry breaking, causing members to converge to different singular value combinations under the same basis. $\boldsymbol{\Sigma}^{(m)}$ is constrained to be non-negative via `clamp(min=0)` during forward passes.
    - **Design Motivation**: Multiplicative perturbations adapt noise magnitude relative to singular value size, preserving the relative ordering (important directions remain important) while ensuring perturbations are meaningful. This is an implicit ensemble version of Deep Ensemble's "different seeds converge to different modes," replacing expensive "independent initialization" with "shared basis + minimal perturbation."

3. **Backbone Quality Scaling Law for SVE Gains**:
    - **Function**: Establishes the application boundaries of the method using an empirical law—the stronger the backbone, the more SVE wins.
    - **Mechanism**: The authors tested three backbones (Random initialization, DINOv1, DINOv2) for the same ViT-S architecture on CIFAR-100 to observe SVE's accuracy gain relative to other ensembles. The gain is monotonically increasing: SVE performs worst under random initialization (no "meaningful basis" to share), surpasses mainstream implicit ensembles on DINOv1, and even exceeds Deep Ensemble on DINOv2.
    - **Design Motivation**: Formulates "when to use SVE" as a falsifiable proposition. The method assumes that "pre-trained singular vectors already define semantic directions." The stronger the pre-training, the more valid this assumption, and the greater the SVE dividend. This also explains why SVE provides the best calibration on large models like BERT or LLaMA.

### Loss & Training
All members are trained synchronously with mean cross-entropy as the sole loss. Each member has an independent classification head (additional $M\cdot d\cdot C$ parameters, negligible relative to the total). $M=4$ for vision tasks, $M=8$ for SST-2, and $M=16$ for ARC-Easy. $\sigma_{\text{init}}$ works well between 0.001 and 0.1, with 0.01 as default. SVE can be applied to a subset of linear layers (e.g., attention projections only) to further reduce parameters.

## Key Experimental Results

### Main Results

| Dataset / Backbone | Method | Acc ↑ | ECE ↓ | NLL ↓ | Brier ↓ |
|--------------------|--------|-------|-------|-------|---------|
| Flowers102 / DINO ViT-S | Single | 86.3 | 3.9 | 0.56 | 0.20 |
| Flowers102 / DINO ViT-S | Deep Ensemble (M=4) | 91.5 | 0.9 | 0.33 | 0.12 |
| Flowers102 / DINO ViT-S | LoRA-Ensemble (M=4) | 94.6 | 1.1 | 0.21 | 0.08 |
| Flowers102 / DINO ViT-S | **SV-Ensemble (M=4)** | **95.4** | 1.0 | **0.18** | **0.07** |
| Oxford Pets / DINOv2 ViT-S | Deep Ensemble (M=4) | 89.2 | 13.3 | 0.43 | 0.17 |
| Oxford Pets / DINOv2 ViT-S | LoRA-Ensemble (M=4) | 86.1 | 9.0 | 0.49 | 0.22 |
| Oxford Pets / DINOv2 ViT-S | **SV-Ensemble (M=4)** | **90.1** | **2.2** | **0.30** | **0.15** |
| SST-2 / BERT-base | Deep Ensemble (M=8) | **93.2** | 4.7 | 0.23 | — |
| SST-2 / BERT-base | LoRA-Ensemble (M=8) | 92.7 | 3.8 | 0.21 | — |
| SST-2 / BERT-base | **SV-Ensemble (M=8)** | 92.0 | **2.8** | **0.21** | — |
| ARC-Easy / LLaMA-2-7B | Deep Ensemble (M=3) | 85.8 | 9.9 | 0.83 | — |
| ARC-Easy / LLaMA-2-7B | LoRA-Ensemble (M=5) | 86.0 | 9.0 | 0.92 | — |
| ARC-Easy / LLaMA-2-7B | Bayes-LoRA (LA) | 85.1 | 5.4 | 0.49 | — |

### Ablation Study

| Configuration | Key Conclusion | Source |
|---------------|----------------|--------|
| Backbone Quality (Random / DINOv1 / DINOv2) | SVE relative gain increases with representation quality; exceeds Deep Ensemble on DINOv2 | Fig. 2 |
| Freeze $\mathbf{U},\mathbf{V}$ vs. Tune $\boldsymbol{\Sigma}$ (Single w/ SVF) | Single-model SVF consistently outperforms Single (e.g., Flowers102 86.3→91.8) | Table 1 |
| Number of members $M$ ($M\!=\!4/8/16$) | Calibration improves with larger $M$, while SVE parameter increment per $M$ is $\sim$ per mille | Appendix B |
| $\sigma_{\text{init}}$ in $0.001\sim 0.1$ | Performance is robust; no fine-tuning required | Appendix C |
| Partial layer SVE (Attention projection only) | Further reduces parameters with slight accuracy loss | Appendix D |

### Key Findings
- Calibration is SVE's primary advantage: On ARC-Easy, SVE's ECE is significantly lower than all explicit/implicit ensemble baselines, approaching Bayesian methods that jointly learn mean/covariance (e.g., BLoB) without requiring complex posterior fitting.
- Low-resource "few-shot + short training" scenarios like Oxford Pets reveal the fragility of competing methods: BatchEnsemble ECE reached 48.7%, Deep Ensemble 13.3%, and LoRA-Ensemble 9.0%, while SVE remained at 2.2%. This indicates that sharing the pre-trained basis provides a regularization effect in low-resource settings, where diversity stems from singular value reshaping rather than overfitting new parameters.
- Single-member SVF (where $M=1$) generally outperforms standard fine-tuning, suggesting that "learning only singular values" is a competitive PEFT method in itself; SVE seamlessly integrates the additional dimension of "ensembling" on top of this.
- The method's "weakness" lies in weak backbones: SVE performed worst on randomly initialized ViTs, supporting the author's hypothesis—without a meaningful singular vector basis to share, the re-weighting mechanism loses its physical meaning. This serves as an application boundary test.

## Highlights & Insights
- Redefining "ensemble diversity" within a semantic subspace: Instead of letting members learn new directions from scratch, they are allowed to create different intensity combinations of the same "knowledge directions"—a remarkably frugal diversity paradigm. This could be extended to: "Multi-tasking"—one set of singular values per task; "Personalization"—one set per user; "Continual Learning"—recording task knowledge through singular value deltas.
- Inverse validation of hypothesis via backbone strength: Rather than just claiming "superior performance," the authors designed a scaling law experiment (Random→DINOv1→DINOv2) to demonstrate when the method fails versus when it succeeds. This is an excellent research methodology, more persuasive than simply topping leaderboards.
- The parameter cost formula $(M-1)\cdot 5/(4d)$ provides a clear engineering budget: For LLaMA-2-7B ($d=4096$), even $M=16$ only costs $\approx 0.2\%$, meaning UQ can be treat as a "standard feature" in any PEFT pipeline without architectural compromises.

## Limitations & Future Work
- The paper does not verify if "true predictive diversity among members" saturates with $M$: Since singular vectors are locked, members can only adjust intensities along the same directions, potentially hitting an expressivity ceiling. The discussion on this boundary is limited.
- LLM experiments are limited to LLaMA-2-7B and evaluated only on the ARC-Easy single task; SVE's UQ behavior on newer models (LLaMA-3, Qwen) and generative tasks (open-ended QA, code generation) is not covered. Whether token-level calibration improves similarly remains an open question.
- The non-negativity constraint `clamp(min=0)` for $\boldsymbol{\Sigma}^{(m)}$ creates non-differentiable boundaries during training, theoretically causing occasional gradient clipping; smooth constraints like softplus might be more stable, but no comparison was provided.
- Inference still requires explicit reconstruction of $\mathbf{W}^{(m)}=\mathbf{U}\boldsymbol{\Sigma}^{(m)}\mathbf{V}^{\top}$ for each member; for LLaMA layers with large $D$, this implies reconstruction overhead $M$ times. While parameter-efficient, wall-clock inference costs still grow with $M$, necessitating engineering optimization.

## Related Work & Insights
- **vs. Deep Ensemble**: The gold standard and strongest baseline, but with $M\times$ parameters and computation. SVE squeezes $M$ independent models into the same set of singular vector bases, achieving $\lesssim1\%$ parameters while matching or exceeding calibration.
- **vs. LoRA-Ensemble**: Another implicit ensemble. LoRA-Ensemble adds $\mathcal{O}((m+n)r)$ new parameters per member to learn "low-rank updates." SVE learns only $\min(m,n)$ singular values without introducing any "new directions," saving more parameters and relying on pre-trained priors.
- **vs. SVF / SVFit**: SVF uses "tuning only singular values" for single-model adaptation. SVE applies this primitive to ensembles, asserting that singular vectors are the shared language of backbone knowledge.
- **vs. BatchEnsemble**: BatchEnsemble uses rank-1 scaling vectors to multiply shared weights. SVE performs a similar "direction-wise scaling" in the SVD domain, but the scaling targets are semantic singular directions rather than arbitrary bases.
- **vs. Bayes-LoRA / BLoB / C-LoRA**: These methods quantify uncertainty through posterior approximation—theoretically rigorous but complex to implement. SVE approximates the posterior indirectly through an ensemble paradigm, offering simpler implementation with comparable ECE.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hankel Singular Value Regularization for Highly Compressible State Space Models](../../NeurIPS2025/model_compression/hankel_singular_value_regularization_for_highly_compressible_state_space_models.md)
- [\[ICML 2026\] End-to-End Compression for Tabular Foundation Models](end-to-end_compression_for_tabular_foundation_models.md)
- [\[ICML 2026\] BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models](bioarc_discovering_optimal_neural_architectures_for_biological_foundation_models.md)
- [\[ICML 2026\] PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)

</div>

<!-- RELATED:END -->
