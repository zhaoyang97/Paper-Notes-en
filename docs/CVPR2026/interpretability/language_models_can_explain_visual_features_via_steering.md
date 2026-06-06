---
title: >-
  [Paper Note] Language Models Can Explain Visual Features via Steering
description: >-
  [CVPR 2026][Interpretability][Sparse Autoencoders] This paper proposes a method for scalable automatic explanation of visual features by causally intervening (steering) on SAE features in VLM visual encoders. By injectin…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Visual Feature Explanation"
  - "Causal Intervention"
  - "VLM"
  - "Automated Interpretability"
date: 2026-05-08
content_hash: 5b564e4b0b5640ef
---

# Language Models Can Explain Visual Features via Steering

**Conference**: CVPR 2026
**arXiv**: [2603.22593](https://arxiv.org/abs/2603.22593)  
**Code**: [GitHub](https://github.com/HPAI-BSC/vision-interp)  
**Area**: Interpretability
**Keywords**: Sparse Autoencoders, Visual Feature Explanation, Causal Intervention, VLM, Automated Interpretability

## TL;DR

This paper proposes a method for scalable automatic explanation of visual features by causally intervening (steering) on SAE features in VLM visual encoders. By injecting feature vectors into a blank image's forward pass and prompting the language model to describe what it "sees," the approach eliminates the need for an evaluation image set. A hybrid method, Steering-informed Top-k, is further proposed and achieves state-of-the-art performance.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Sparse autoencoders (SAEs) have emerged as a powerful tool for discovering interpretable features in vision models. However, automatically explaining thousands of discovered features at scale remains an open problem.

Limitations of existing methods (Top-k approaches):
1. **Correlation-based rather than causal**: Selecting highest-activating images and identifying common patterns is fundamentally a correlational analysis.
2. **Dependence on evaluation image sets**: Retrieving top-activating images requires large-scale datasets, introducing dataset bias.
3. **High computational cost**: Ranking activations requires a full forward pass over the entire evaluation set.

Core insight of this paper: VLMs connect a visual encoder to a pretrained language model. If a **causal intervention** is applied to the visual encoder — injecting a specific SAE feature vector into a blank image's forward pass — the language model should be able to express what visual concept it "perceives."

## Method

### Overall Architecture

1. Train a TopK SAE ($d_{\text{SAE}} = 8192$) to decode visual encoder features on ImageNet.
2. For each SAE feature, inject the corresponding feature vector during the forward pass of a blank image.
3. Prompt the language model to explain what it "sees."
4. Optionally: apply a hybrid approach that combines Top-k images with causal intervention.

### Key Designs

1. **Steering-based Explanation (pure causal intervention)**:
    - **Function**: Generates natural language explanations of SAE features without requiring an image set.
    - **Mechanism**: A blank (all-white) image $\tilde{I}$ is fed into the VLM. At layer $l$ of the visual encoder, the SAE decoder weight vector $W_{dec}[i,:] \times \alpha$ is added to the residual stream at all token positions. The language model then generates an explanation conditioned on the intervened visual representation, formalized as: $e_i \sim m_{exp}(e | P, \tilde{I}, \text{do}(m_{sub}^l(\tilde{I}) \leftarrow m_{sub}^l(\tilde{I}) + \alpha W_{dec}[i,:]))$
    - **Design Motivation**: A blank image provides no meaningful visual signal, so the language model's output is entirely driven by the causal intervention, yielding a purely causal explanation. Only a single forward pass is required, making the approach highly efficient.

2. **Steering-informed Top-k (hybrid method)**:
    - **Function**: Combines the advantages of causal intervention and input images.
    - **Mechanism**: The same SAE feature causal intervention is applied to the visual encoder while also conditioning on Top-k activating images. This integrates correlational evidence (Top-k images) with causal evidence (feature injection) to guide the explainer toward more precise descriptions.
    - **Design Motivation**: Pure steering performs better on low-level features, while Top-k is stronger for high-level semantic features. The hybrid method achieves optimal performance across all four complementary metrics without incurring additional computational cost.

3. **Evaluation Metric Suite**:
    - **Function**: Quantifies explanation quality from multiple complementary perspectives.
    - Four complementary metrics are adopted:
        - **Activation IoU**: Overlap between images highly activated by the explanation text and those highly activated by the SAE feature.
        - **Detection Score**: Whether the described concept can be detected by a VLM in the activating images.
        - **CLIP Similarity**: CLIP embedding distance between the explanation and the top-activating images.
        - **Monosemanticity**: Whether the feature corresponds to a single coherent concept.

### Loss & Training

The SAE is trained with a standard TopK objective on ImageNet. The intervention strength $\alpha$ is selected on a validation set of 500 features. The explanation generation procedure itself involves no training — it operates purely at inference time via intervention.

## Key Experimental Results

### Main Results — Explanation Quality Comparison (Gemma 3 Visual Encoder)

| Method | Activation IoU↑ | Detection↑ | CLIP↑ | Monosemanticity↑ |
|--------|----------------|-----------|-------|-----------------|
| Top-k (original images) | baseline | baseline | baseline | baseline |
| Top-k (Mask) | slightly better | slightly better | slightly lower | comparable |
| Top-k (Heatmap) | comparable | comparable | comparable | comparable |
| Steering (pure intervention) | below Top-k | below Top-k | below Top-k | comparable |
| **Steering-informed Top-k** | **best** | **best** | **best** | **best** |

### Ablation Study — Effect of Language Model Scale

| LM Scale | Explanation Quality Trend |
|----------|--------------------------|
| Small | baseline level |
| Medium | significant improvement |
| Large | continued improvement |

Explanation quality improves consistently with language model scale, with no sign of saturation.

### Key Findings

- Pure steering outperforms Top-k on low-level features (texture, color, edges), as causal intervention more effectively captures primitive visual concepts.
- Top-k is stronger on high-level semantic features (object categories), as concrete images provide useful reference context.
- The hybrid method (Steering-informed Top-k) achieves state-of-the-art across all metrics with no additional computational overhead.
- Language model scale is a key factor for explanation quality — larger LMs more effectively "verbalize" visual concepts.
- Findings are consistent across two distinct VLMs: Gemma 3 and InternVL3.

## Highlights & Insights

- A paradigm shift from **correlation** to **causation**: Steering directly intervenes on internal model representations, providing a more causally grounded basis than the correlational analysis of Top-k methods.
- Highly efficient: explaining a single feature requires only one forward pass, with no need to iterate over an evaluation set.
- The scaling effect of language models suggests that stronger future LMs will further advance automated interpretability.
- The hybrid method is elegantly designed: causal signals are injected within the image context provided by Top-k, allowing the two sources of evidence to complement each other.

## Limitations & Future Work

- Pure steering underperforms Top-k on high-level semantic features due to the lack of contextual information in blank images.
- The intervention strength $\alpha$ is sensitive and requires tuning on a validation set.
- Validation is limited to VLM architectures; the approach cannot be directly applied to pure vision models without a language model component.
- The SAE dictionary size is fixed at 8192; the scalability to larger dictionaries remains unknown.
- Evaluation relies primarily on automated metrics, with no human evaluation.

## Related Work & Insights

- **vs. standard Top-k methods**: Top-k is correlation-based, requires an evaluation set, and is computationally intensive; Steering is causal, requires no image set, and needs only a single forward pass.
- **vs. PatchScopes/SELFIE**: These methods perform self-explanation within language models; this paper is the first to extend the paradigm to visual encoders.
- **vs. CB-SAE (same venue)**: CB-SAE focuses on controllability and interpretability metrics of SAEs, whereas this paper focuses on generating natural language explanations for SAE features.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Causal intervention for visual feature explanation is a genuinely novel paradigm; the method is concise and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-metric, multi-VLM, and scaling analysis are provided, though human evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and the method is intuitive, though some LaTeX rendering issues are present.
- **Value**: ⭐⭐⭐⭐ — Makes a significant contribution to automated interpretability research for vision models, with strong scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [\[ACL 2026\] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](../../ACL2026/interpretability/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)

</div>

<!-- RELATED:END -->
