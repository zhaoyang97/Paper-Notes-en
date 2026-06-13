---
title: >-
  [Paper Note] Looking in the Mirror: A Faithful Counterfactual Explanation Method for Interpreting Deep Image Classification Models
description: >-
  [ICCV 2025][Image Generation][counterfactual explanation] This paper treats a classifier's decision boundary as a "mirror" and generates counterfactual explanations (CFEs) by reflecting feature representations to the oth…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "counterfactual explanation"
  - "decision boundary"
  - "faithful explanation"
  - "classifier interpretability"
  - "animated transition"
date: 2026-05-08
content_hash: 999a058d679e6e8e
---

# Looking in the Mirror: A Faithful Counterfactual Explanation Method for Interpreting Deep Image Classification Models

**Conference**: ICCV 2025
**arXiv**: [2509.16822](https://arxiv.org/abs/2509.16822)  
**Code**: [https://github.com/AIML-MED/Mirror-CFE](https://github.com/AIML-MED/Mirror-CFE)  
**Area**: Explainable AI / Counterfactual Explanation
**Keywords**: counterfactual explanation, decision boundary, faithful explanation, classifier interpretability, animated transition

## TL;DR

This paper treats a classifier's decision boundary as a "mirror" and generates counterfactual explanations (CFEs) by reflecting feature representations to the other side of the mirror. A triangulation loss is designed to preserve distance relationships between the latent space and image space, yielding faithful, controllable, and animatable counterfactual explanations.

## Background & Motivation

Counterfactual Explanation (CFE) aims to answer the question "how should the input change for the model to make a different decision," which is particularly important in high-stakes scenarios such as medical imaging.

**Three key problems with existing CFE methods**:

**Unfaithfulness**: Generative-model-based methods (e.g., StyleGAN2, Diffusion) use additional encoders and generators to produce realistic images, but the CFE generation process does not correspond to the classifier's actual decision boundary, causing the explanation to be decoupled from what the classifier has learned.

**Lack of continuity**: Existing methods generate only a single decision-flipping sample and cannot show how the change occurs progressively.

**Tendency toward adversarial examples**: When optimizing the proximity constraint, generators easily learn a shortcut of adding imperceptible noise to flip the decision.

**Core motivation**: A faithful CFE should operate directly in the classifier's own feature space and exploit the decision boundary learned by the classifier to generate explanations, rather than relying on external generative models.

## Method

### Overall Architecture

Mirror-CFE consists of two stages: (1) defining the position of the CFE point in the classifier's latent space $\mathcal{Z}$; and (2) training a mapping function $G: \mathcal{Z} \to \mathcal{I}$ that projects latent-space points into image space.

### Key Designs

1. **Mirror Reflection Mechanism (CFE definition in latent space)**:

    - Given classification weight vectors $\mathbf{W}_s, \mathbf{W}_t$ for source class $s$ and target class $t$, the decision boundary ("mirror") is parameterized by $\mathbf{W}_m = \mathbf{W}_t - \mathbf{W}_s$.
    - Position function: $P(\mathbf{z}_s, \mathbf{W}_m, \mathbf{b}_m, k) = \mathbf{z}_s - 2k(\mathbf{W}_m^\top \mathbf{z}_s + \mathbf{b}_m)\hat{\mathbf{W}}_m$
    - $k=0.5$: reaches the mirror (projection point $\mathbf{z}_p$, equal probability for both classes)
    - $k=1.0$: full reflection (reflection point $\mathbf{z}_r$, confidence flipped)
    - $k=0.5+\epsilon$: CFE point (just across the boundary)
    - By continuously varying $k \in [0,1]$, an animated gradual transition from source class to target class can be generated.

2. **Triangulation Loss**:

    - The conventional proximity loss $\mathcal{L}_{prox} = |\mathbf{x}_s - \mathbf{x}_{cf}|$ tends to produce adversarial examples.
    - Core idea: **preserve the distance ratio relationships of the latent space in image space**:
    $$\frac{|\mathbf{x}_k - \mathbf{x}_t|}{|\mathbf{x}_s - \mathbf{x}_k|} \approx \frac{\|\mathbf{z}_k - \mathbf{z}_t\|}{\|\mathbf{z}_s - \mathbf{z}_k\|} = \beta$$
    - Implemented via upper- and lower-bound relaxation, analogous to triangulation that determines an unknown position from known base-station distances.
    - Distance relationships for semifactual explanations (SFE, $k < 0.5$) are handled symmetrically.

3. **Skip Connection Controller (SSC)**:

    - The classifier $F$ is used as the encoder (frozen to ensure faithfulness), and the decoder $G$ is trained.
    - For high-resolution datasets, U-Net-style skip connections are introduced to transfer high-frequency information.
    - **SPE module**: blends source image features with KFE encodings to control style information.
    - **CSP module**: computes a spatial prior mask $\mathbf{M}_k^i$ using CAM to restrict changes to discriminative regions. The mask size grows as $k$ increases.

### Loss & Training

The total loss comprises:

- **Classification loss** $\mathcal{L}_{cls}$: KL divergence ensuring generated KFE images maintain the expected classification probability distribution.
- **Adversarial loss** $\mathcal{L}_{adv}$: promotes realism of generated images.
- **Reconstruction loss** $\mathcal{L}_{rec} = \mathbb{E}[|\mathbf{x} - G(F(\mathbf{x}))|]$: guarantees encode–decode cycle consistency.
- **Feature reconstruction loss** $\mathcal{L}_{fea} = \mathbb{E}[\|\mathbf{z}_k - F(G(\mathbf{z}_k))\|]$: consistency at KFE points.
- **Triangulation loss** $\mathcal{L}_{tri}$: prevents adversarial example generation.

## Key Experimental Results

### Main Results

| Dataset | Method | L1↓ | LPIPS↓ | FID↓ | D.Val↑ | Val.↑ | %Fail |
|---------|--------|-----|--------|------|--------|-------|-------|
| MNIST | PGD | 0.42 | 0.31 | 15.62 | 0.74 | 1.0 | 0.0 |
| MNIST | C3LT | 0.17 | 0.25 | 8.95 | 0.79 | 1.0 | 0.0 |
| MNIST | **Mirror-CFE (1st)** | **0.16** | **0.17** | **3.25** | **0.99** | **1.0** | **0.0** |
| F-MNIST | PGD | 0.34 | 0.28 | 10.12 | 0.88 | 1.0 | 0.0 |
| F-MNIST | C3LT | 0.32 | 0.30 | 11.55 | 0.87 | 1.0 | 0.0 |
| F-MNIST | **Mirror-CFE (1st)** | **0.12** | **0.10** | **2.80** | **0.99** | **0.99** | **0.0** |
| B-MNIST | C3LT | 0.16 | 30.14 | 96.03 | 0.99 | 0.99 | 0.0 |
| B-MNIST | **Mirror-CFE (1st)** | **0.05** | **11.81** | **86.02** | **0.99** | **0.99** | **0.0** |

Mirror-CFE achieves optimal or near-optimal proximity (L1/LPIPS) and realism (FID) simultaneously across all datasets, with a near-perfect denoised validity score (D.Val), indicating that the generated examples are not adversarial.

### Ablation Study

| Property | Mirror-CFE Advantage | Limitation of Prior Work |
|----------|----------------------|--------------------------|
| Faithfulness | Directly uses the classifier's feature space and decision boundary | External encoders introduce bias |
| Adversarial example detection | D.Val ≈ 0.99 (effective after denoising) | CEM/REVISE D.Val as low as 0.03–0.16 |
| Controllability | Continuously adjusting $k$ controls the magnitude of change | Only a single CFE is generated |
| Animated transition | Frame-by-frame interpolation from $k=0$ to $k=1$ | This capability is absent |

### Key Findings

- Mirror-CFE's 1st CFE (just past the decision boundary) reveals the "most critical change" as perceived by the classifier, while the reflection point at $k=1$ demonstrates a complete class transition.
- On CelebA-HQ, CFEs generated by C3LT exhibit correct mouth shape but lose subject identity; Mirror-CFE preserves identity while precisely modifying the mouth region via the skip connection controller.
- The convergence failure rate is 0% ($\%Fail = 0$), whereas CEM and REVISE exhibit failure rates of 16.72% and 9.77%, respectively.

## Highlights & Insights

- The **"mirror" metaphor** is intuitively clear: decision boundary = mirror, CFE = reflection, reducing a complex explanation problem to a geometric operation.
- The **triangulation loss** elegantly resolves the adversarial example problem in CFE by enforcing distance ratio constraints between the latent and image spaces.
- The **SSC module** addresses the blurriness issue in high-resolution generation while maintaining faithfulness (the classifier is not fine-tuned).

## Limitations & Future Work

- The current method is validated only on linear decision boundaries in fully connected classification layers; extension to nonlinear decision boundaries (e.g., nonlinear classification heads in deep feature spaces) is required.
- Performance on high-resolution datasets (e.g., CelebA-HQ 224×224) relies on the SSC module design; larger resolutions may demand more refined architectures.
- Only pairwise CFE between two classes is supported; multi-class scenarios require constructing separate mirrors for each class pair.

## Related Work & Insights

- The fundamental distinction from GAN/Diffusion-based CFE methods lies in **faithfulness**: no external generative model is used; instead, the method directly manipulates the feature space learned by the classifier.
- The approach of designing CFEs from the geometric structure of the decision boundary can inspire other tasks requiring faithful explanations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The mirror reflection mechanism and triangulation loss are distinctive
- **Technical Depth**: ⭐⭐⭐⭐ — Loss function design is rigorous with clear geometric derivations
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, six metrics, and multi-method comparisons
- **Value**: ⭐⭐⭐⭐ — Animated CFEs have direct practical value for medical image interpretation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](../../NeurIPS2025/image_generation/leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)
- [\[ICCV 2025\] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models](revelio_interpreting_and_leveraging_semantic_information_in_diffusion_models.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[NeurIPS 2025\] V-CECE: Visual Counterfactual Explanations via Conceptual Edits](../../NeurIPS2025/image_generation/v-cece_visual_counterfactual_explanations_via_conceptual_edits.md)
- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](../../NeurIPS2025/image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)

</div>

<!-- RELATED:END -->
