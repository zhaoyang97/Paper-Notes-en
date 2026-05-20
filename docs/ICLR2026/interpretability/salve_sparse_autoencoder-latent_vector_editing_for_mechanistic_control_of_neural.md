---
title: >-
  [Paper Note] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks
description: >-
  [ICLR2026][Interpretability][Mechanistic Interpretability] This paper proposes SALVE, a three-stage "Discover–Verify–Control" framework: (1) an L1-regularized sparse autoencoder (SAE) is trained to discover interpretable…
tags:
  - "ICLR2026"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "Sparse Autoencoder"
  - "Model Editing"
  - "Feature Visualization"
  - "Weight-Space Intervention"
date: 2026-05-08
content_hash: 190583c5c1c8b287
---

# SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks

**Conference**: ICLR2026
**arXiv**: [2512.15938](https://arxiv.org/abs/2512.15938)  
**Code**: To be confirmed  
**Area**: Interpretability
**Keywords**: Mechanistic Interpretability, Sparse Autoencoder, Model Editing, Feature Visualization, Weight-Space Intervention

## TL;DR
This paper proposes SALVE, a three-stage "Discover–Verify–Control" framework: (1) an L1-regularized sparse autoencoder (SAE) is trained to discover interpretable feature bases within a model; (2) Grad-FAM visualization is employed to verify the semantic meaning of discovered features; (3) the SAE decoder matrix guides permanent weight-space editing. The framework is validated on ResNet-18 and ViT-B/16, demonstrating precise, persistent, and low-side-effect control ranging from class suppression to cross-class feature modulation.

## Background & Motivation
- Mechanistic interpretability has advanced substantially in recent years, with SAEs becoming a mainstream tool for discovering internal features of neural networks (e.g., Anthropic's landmark "Mapping the Mind" work).
- However, most existing work remains at the stage of "discovering and visualizing features"—understanding what the model "thinks," but lacking the ability to precisely modify its behavior.
- Model editing methods (e.g., ROME, MEMIT) can modify models but lack interpretability support, making edits opaque.
- The core vision of SALVE: **interpretability → editing**—first use SAEs to understand what features the model has learned, then precisely edit those features.
- Inference-time interventions (activation steering) are transient; SALVE pursues permanent weight modifications.

## Method

### Overall Architecture
A three-stage "Discover–Verify–Control" pipeline: (1) train an L1-regularized linear SAE at the target layer (ResNet average pooling layer / ViT [CLS] token) to obtain a sparse, interpretable feature basis; (2) use activation maximization and Grad-FAM to verify feature semantics; (3) leverage the SAE decoder matrix $D$ to guide permanent weight-space editing.

### Key Designs

1. **Sparse Feature Discovery (SAE Stage)**:

    - A linear autoencoder is trained at the target layer: $Z = \text{Encoder}(x)$, with L1 regularization to encourage sparsity.
    - Class-conditional mean latent activations $\mu_k = \frac{1}{|C_k|}\sum_{n \in C_k} Z_n$ are computed, and dominant features per class are identified by ranking $|\mu_k|$.
    - Sparsity ensures each feature is independently operable—infrequently activated features approach zero in the class mean.

2. **Grad-FAM Feature Visualization**:

    - Analogous to Grad-CAM but operating on the SAE latent space rather than CNN feature maps.
    - Generates spatial heatmaps per SAE feature, showing which regions of the input image the feature "attends to."
    - Complementary to activation maximization: the latter reveals the abstract concept of a feature, while Grad-FAM localizes it within specific images.

3. **Permanent Weight-Space Editing**:

    - Editing formula: $w_{ij}' = w_{ij} \cdot \max(0, 1 \pm \alpha \cdot |c_j|)$, where $c_j = D[j, l]$ is the contribution of feature $l$ to activation dimension $j$.
    - $\alpha$ controls intervention strength; $\pm$ controls enhancement/suppression direction.
    - Design Motivation: multiplicative editing preserves the sign structure of learned classifier weights; the effect is conditioned on sample activation patterns rather than global coverage.

4. **Critical Suppression Threshold $\alpha_{crit}$**:

    - Linear approximation: $\alpha_{crit}^{(n)} \approx \frac{z_i^{(n)}}{R_i(\mathbf{x}^{(n)})}$, where $R_i$ quantifies suppression sensitivity along the feature direction.
    - Physical interpretation: the minimum suppression strength required to reduce the target class logit to zero.
    - Low $\alpha_{crit}$ indicates high dependency (fragile representation); high $\alpha_{crit}$ indicates the class is supported by multiple redundant features.

### Loss & Training
SAE training: reconstruction loss + L1 regularization. Weight editing is a post-processing step and involves no additional training.

## Key Experimental Results

### Main Results (ResNet-18 on Imagenette, ViT-B/16 validation)

| Operation | Target Class Accuracy | Non-Target Class Accuracy | Notes |
|---|---|---|---|
| Original model | ~95% | ~95% | Baseline |
| Suppress "Church" feature | ~0% | ~95% | Precisely disables target class, zero spillover |
| Enhance "Golf ball" feature | ~95% | ~95% | Enhancement does not affect other classes |
| Suppress cross-class "Tower" feature | Petrol Pump↓, Church unchanged | Minor variation | Reveals feature sharing and entanglement |

### Ablation Study

| Analysis | Result | Notes |
|---|---|---|
| $\alpha_{crit}$ distribution (analytical vs. numerical vs. empirical) | All three consistent | Analytical estimate provides a lower bound; numerical computation is exact |
| Cross-class "Tower" feature editing | Petrol Pump has high dependency; Church has high redundancy | Differential dependency on the same feature reveals representational structure |
| SAE initialization robustness | Consistent results across 10 random initializations | Editing effects are independent of the specific SAE basis |
| ViT-B/16 validation | Similar suppression curves and editing precision | Generalizes across CNN and Transformer architectures |
| CIFAR-100 extension | Effective but with more cross-class sharing | Reveals limitations of a simple L1 SAE under high class diversity |

### Key Findings
- Permanent weight editing achieves target-class zeroing effects comparable to inference-time activation steering and ROME.
- $\alpha_{crit}$ successfully identifies "fragile" classes—those relying on a single dominant feature with little redundancy are more susceptible to suppression.
- Cross-class feature editing reveals hidden feature entanglement: suppressing/enhancing the "Tower" feature produces an inverse effect on the "Chain Saw" class, suggesting a learned spurious negative correlation.

## Highlights & Insights
- SALVE is the first work to systematically bridge SAE-based interpretability discoveries to permanent weight editing, closing the critical gap between "understanding" and "control."
- Grad-FAM is a useful tool for SAE feature visualization—more intuitive than directly inspecting activation distributions and complementary to Grad-CAM.
- The $\alpha_{crit}$ threshold concept is elegant—a single scalar quantifies "how important a feature is to a class," enabling robustness diagnostics and adversarial fragility prediction.
- The comparison between permanent weight editing and inference-time intervention is clearly articulated—persistent modifications are more valuable in compliance-sensitive settings.
- Cross-class feature editing reveals the internal feature entanglement structure of the model—the inverse relationship between the "Tower" and "Chain Saw" features is discoverable only through SALVE.

## Limitations & Future Work
- Validation is limited to image classification tasks (Imagenette, CIFAR-100); applicability to LLMs is a more important and challenging direction, as their internal representations are higher-dimensional and more entangled.
- The quality of SAE training directly determines the quality of downstream editing—insufficiently disentangled features lead to side effects (already partially observed in CIFAR-100 experiments).
- Weight-space back-projection may be distorted by accumulated nonlinearities in deep networks—the current implementation edits only the final layer.
- Direct quantitative comparisons with model editing methods (ROME, MEMIT) are insufficient—only qualitative comparisons on class suppression are provided.
- The quality of sparse bases when scaling to larger models (e.g., ViT-L, ResNet-101) and datasets requires validation—more advanced SAE variants such as Gated/Top-k SAE may be necessary.

## Related Work & Insights
- **vs. ROME/MEMIT**: ROME performs single-sample factual correction (rank-1 weight update), while SALVE performs feature-driven global behavior modulation—the objectives differ but the approaches are complementary.
- **vs. Activation Steering**: Steering is a transient inference-time intervention (requiring an offset vector injected at each forward pass), whereas SALVE performs permanent weight editing with zero inference-time overhead.
- **vs. Anthropic's Dictionary Learning**: Anthropic's SAE research focuses on "discovering and understanding" features; SALVE elevates this to a "control" tool—advancing from interpretability to editability.
- **Insight**: The paradigm of SAE + weight editing has the potential to become a practical tool for AI safety—first understanding what the model "thinks," then precisely correcting unwanted behavior.

## Rating
- Novelty: ⭐⭐⭐⭐ The bridging concept from understanding to control is novel; the distinction between permanent weight editing and inference-time steering is clearly positioned.
- Experimental Thoroughness: ⭐⭐⭐ Limited to image classification (Imagenette + CIFAR-100); experiments on LLMs and large-scale models are absent.
- Writing Quality: ⭐⭐⭐⭐ The "Discover–Verify–Control" pipeline is well-motivated; ablation and qualitative analyses are comprehensive.
- Value: ⭐⭐⭐⭐ The work holds long-term value for AI safety; the SAE + weight editing paradigm warrants attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICLR 2026\] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study](initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)
- [\[NeurIPS 2025\] Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](../../NeurIPS2025/interpretability/out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)

</div>

<!-- RELATED:END -->
