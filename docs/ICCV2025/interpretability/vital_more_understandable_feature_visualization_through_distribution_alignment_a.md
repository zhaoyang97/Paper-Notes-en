---
title: >-
  [Paper Note] VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow
description: >-
  [ICCV 2025][Interpretability][Feature Visualization] This paper proposes VITAL, a feature visualization method that reframes the problem as aligning intermediate feature distributions with those of real images (rather th…
tags:
  - "ICCV 2025"
  - "Interpretability"
  - "Feature Visualization"
  - "Explainable AI"
  - "Distribution Matching"
  - "Layer-wise Relevance Propagation"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: fe83cada4379154d
---

# VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow

**Conference**: ICCV 2025
**arXiv**: [2503.22399](https://arxiv.org/abs/2503.22399)  
**Code**: [GitHub](https://github.com/VITAL-Framework)  
**Area**: Interpretability
**Keywords**: Feature Visualization, Explainable AI, Distribution Matching, Layer-wise Relevance Propagation, Mechanistic Interpretability

## TL;DR

This paper proposes VITAL, a feature visualization method that reframes the problem as aligning intermediate feature distributions with those of real images (rather than conventional activation maximization), and incorporates relevance scores to filter irrelevant features, producing neuron visualizations that are more interpretable to humans.

## Background & Motivation

Feature Visualization (FV) is an important tool for understanding what information neural networks learn internally, by generating images that strongly activate specific neurons to explain network behavior. In safety-critical domains such as healthcare, understanding network decision processes is especially important.

However, existing FV methods suffer from severe **interpretability issues**:

**Repetitive Patterns**: Activation maximization tends to repeat the same pattern throughout an image to repeatedly stimulate the target neuron, resulting in "kaleidoscope-like" visual effects.

**Artifacts**: Generated images contain unnatural colors, textures, and other artifacts.

**Irrelevant Features**: Visualizations incorporate background features unrelated to the target neuron (e.g., grass appearing in the visualization of a beak detector).

These problems are particularly pronounced on modern large-scale architectures (ResNet-50, ViT, etc.), severely limiting the practical utility of FV as an interpretability tool.

The core insight is that **activation maximization itself is the root cause of these problems** — it encourages any pattern that increases activation values (including unnatural and repetitive ones) without regard for whether the generated image is realistic. If instead the intermediate feature distributions of generated images are required to match those of real images, repetitive patterns (which do not appear in real images) and artifacts (which deviate from the real data manifold) are naturally suppressed.

## Method

### Overall Architecture

Given a target neuron $f_i^{(l)}$, VITAL no longer seeks an image that maximizes its activation. Instead, it seeks an image $x^*$ whose feature distributions at various network layers are aligned with those of reference real images $x' \in \mathcal{X}_{ref}$:

$$\forall l' < l, x' \in \mathcal{X}_{ref}: \text{dist}(A^{(l')}(x^*)) \approx \text{dist}(A^{(l')}(x'))$$

where $A^{(l')}(x) = f^{(l')}(x) \in \mathbb{R}^{C_l \times D}$ denotes the activation at layer $l'$ and $D$ is the flattened spatial dimension.

### Key Designs

1. **Sort-Matching Distribution Alignment**: The key challenge is how to efficiently match two empirical distributions while supporting backpropagation. VITAL employs a sort-matching approach: sorting indices $\pi$ and $\pi'$ are computed to sort the feature vectors of the generated image and the reference image respectively, and a reverse sorting index $\bar{\pi}$ is used to align the two sorted distributions, with the MSE loss computed as:

$$\text{MSE}(z, z^r) = \frac{1}{|z|} \sum_{i=1}^{|z|} (z_i - z_i^r)^2$$

Since $z$ is a function of $x^*$, gradients can be backpropagated through this loss to optimize $x^*$. For multiple reference images, their sorted feature vectors are averaged to form a representative prototype. The elegance of this approach lies in the fact that while sorting is non-differentiable, the sorting indices form a discrete fixed mapping, and the continuous feature values — the quantities that actually participate in gradient computation — remain differentiable.

2. **Relevance Score Integration (LRP Weighting)**: Distribution matching alone still introduces irrelevant features (e.g., grassy backgrounds in bird images). VITAL incorporates Layer-wise Relevance Propagation (LRP) to assess the **relevance** of each intermediate feature to the target neuron, and uses relevance-weighted activations for distribution matching:

$$A^{(l')}(x) \odot R_n^{(l')}(x)$$

where $R_n$ is the LRP relevance score for target neuron $n$ and $\odot$ denotes the Hadamard product. This ensures that only features genuinely relevant to the target neuron participate in distribution matching, effectively eliminating background features that are "co-activated but unrelated."

3. **Reference Image Selection**: For class neurons, random training images from the corresponding class are used directly. For intermediate neurons, the Top-$k$ highest-activating image patches (drawn from distinct images) are selected, cropped, and resized to form the reference set $\mathcal{X}_{ref}$.

4. **Transparency Map and Auxiliary Regularization**: A transparency map is generated via gradient accumulation to display only those regions attended to by the network during optimization. Auxiliary regularization is also applied:

$$\mathcal{L}_{\text{VITAL}}(x^*, \mathcal{X}_{ref}) = \mathcal{L}_{\text{SM}}(x^*, \mathcal{X}_{ref}) + \alpha_{\text{TV}} \mathcal{L}_{\text{TV}}(x^*) + \alpha_{\ell_2} \ell_2(x^*)$$

### Loss & Training

- Primary loss: Sort-matching distribution alignment loss (accumulated across multiple layers)
- Auxiliary losses: Total Variation (TV) regularization + $\ell_2$ norm regularization
- Layer selection: Ablation experiments confirm that aligning only the first and last block outputs of ResNet-50 is sufficient to produce high-quality images
- Runtime: Approximately 40 seconds per image (including reference distribution computation), comparable to MACO (23–28 s) and DeepInversion (1–3 min)

## Key Experimental Results

### Main Results

Quantitative evaluation on ImageNet pretrained models (classification accuracy, FID, CLIP zero-shot prediction):

| Method | Architecture | Acc.↑ | FID↓ | CLIP Top1↑ | CLIP Top5↑ |
|--------|-------------|-------|------|-----------|-----------|
| MACO | ResNet50 | 29.43 | 360.74 | 12.87 | 29.73 |
| DeepInv | ResNet50 | 100.00 | 35.76 | 29.90 | 55.20 |
| **VITAL** | **ResNet50** | **99.90** | **58.79** | **66.62** | **92.56** |
| MACO | ViT-L-16 | 44.33 | 946.96 | 3.93 | 10.57 |
| **VITAL** | **ViT-L-16** | **99.80** | **126.29** | **68.17** | **92.80** |
| MACO | ConvNeXt | 66.07 | 62.55 | 7.20 | 19.77 |
| **VITAL** | **ConvNeXt** | **99.97** | **3.92** | **63.53** | **90.30** |

### Ablation Study

Human user study (58 participants, three-part evaluation):

| Evaluation Task | MACO | Fourier | DeepInv | VITAL | Description |
|----------------|------|---------|---------|-------|-------------|
| (a) Class visualization + class name (1–5) | ~2.0 median | ~1.5 | ~3.0 | **~4.0** | Rate visualization quality given class name |
| (b) Intermediate neurons (1–5) | ~2.5 | ~2.0 | N/A | **~4.0** | Rate FV–reference image correspondence |
| (c) Free annotation (similarity) | ~0.35 median | ~0.30 | ~0.40 | **~0.60** | Annotate FV content without prompts |

### Key Findings

- VITAL approaches the CLIP zero-shot accuracy of real images (66.62% vs. 69.11% on ResNet-50), substantially outperforming all baselines
- On ConvNeXt-base, VITAL achieves an FID of 3.92, an order of magnitude lower than MACO's 62.55
- In the human user study, VITAL produces an order of magnitude more "high-score" (4–5) visualizations than competing methods
- t-SNE embedding analysis shows that VITAL-generated images cluster at the center of real image clusters, whereas other methods either cluster far from real data or drift from the cluster centers
- VITAL demonstrates markedly superior generalization to ViT architectures compared to all baselines

## Highlights & Insights

- **Paradigm shift**: Transitioning from "maximizing activations" to "aligning distributions" represents a methodological innovation in the field of feature visualization
- **LRP-weighted distribution matching** precisely addresses the "co-activation ≠ relevance" problem, constituting both a theoretical and practical contribution
- The method is entirely architecture-agnostic and scales seamlessly to both CNNs and ViTs
- Small circuit visualizations demonstrate VITAL's potential in mechanistic interpretability — complementing the "what" dimension to the "where" of circuit analysis

## Limitations & Future Work

- Generated images do not yet achieve photorealistic quality, resembling "Monet-style paintings" more than photographs
- Visualizing complex spatial arrangements remains challenging
- Visualization of intermediate neurons is slower (2–3 minutes) due to the additional LRP backward pass
- Reference image selection strategies may introduce bias; robustness warrants further investigation
- The lack of standardized benchmarks for interpretability evaluation means the design and scale of human studies remains an open challenge

## Related Work & Insights

- Within the Mechanistic Interpretability (MI) framework, VITAL complements circuit analysis by addressing the "what" dimension
- DeepInversion also leverages feature statistics but targets batch normalization statistics; VITAL's direct matching of empirical distributions is more flexible
- MACO operates in the frequency domain while VITAL operates in the feature domain — the two approaches are orthogonal
- The proposed framework could inspire extensions to visualization analysis of multimodal models such as CLIP

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Paradigm shift from activation maximization to distribution alignment; elegant integration of LRP
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five architectures, multiple quantitative metrics, two human user studies
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, rigorous method exposition, rich illustrations
- **Value**: ⭐⭐⭐⭐⭐ Significant impact on the explainable AI community; opens a new direction for feature visualization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models](../../ACL2026/interpretability/follow_the_flow_on_information_flow_across_textual_tokens_in_text-to-image_model.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[AAAI 2026\] Unsupervised Feature Selection Through Group Discovery](../../AAAI2026/interpretability/unsupervised_feature_selection_through_group_discovery.md)
- [\[ICCV 2025\] ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)

</div>

<!-- RELATED:END -->
