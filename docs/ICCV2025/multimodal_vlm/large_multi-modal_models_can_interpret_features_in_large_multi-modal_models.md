---
title: >-
  [Paper Note] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models
description: >-
  [Multimodal VLM] This paper proposes the first automated feature interpretation framework for Large Multimodal Models (LMMs). It employs Sparse Autoencoders (SAEs) to decompose LMM internal representations into monosemantic features, leverages larger LMMs to automatically interpret these features, and demonstrates that feature steering can correct model hallucinations.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: 35a04151c143eef9
---

# Large Multi-modal Models Can Interpret Features in Large Multi-modal Models

## Basic Information

- **Conference**: ICCV 2025
- **arXiv**: 2411.14982
- **Code**: [GitHub](https://github.com/EvolvingLMMs-Lab/multimodal-sae)
- **Area**: Multimodal Vision-Language Models (Multimodal VLM)
- **Keywords**: Interpretability, Sparse Autoencoder, Feature Explanation, Model Steering, Hallucination Mitigation

## TL;DR

This paper proposes the first automated feature interpretation framework for Large Multimodal Models (LMMs). It employs Sparse Autoencoders (SAEs) to decompose LMM internal representations into monosemantic features, leverages larger LMMs to automatically interpret these features, and demonstrates that feature steering can correct model hallucinations.

## Background & Motivation

LMMs have achieved remarkable success across vision-language tasks, yet their "black-box" nature leads to unpredictable behaviors, including:
- **Hallucination**: Generating objects and relations absent from the input image
- **Vulnerability to jailbreak attacks**: Adversarial inputs can bypass safety constraints

Understanding and controlling the neural representations of LMMs is critical, but two key challenges arise:

**Polysemanticity**: Individual neurons may encode multiple semantics (e.g., a single neuron in Inception v1 responds to both cat faces and car fronts); the higher dimensionality of LMMs exacerbates this problem.

**Open-ended semantics**: Traditional vision models contain only a few hundred monosemantic concepts (colors, objects, etc.) amenable to manual annotation, whereas LMMs encompass hundreds of thousands of open-domain concepts, making manual analysis infeasible.

Prior work (e.g., GPT-4 interpreting GPT-2 neurons) has not been extended to the multimodal domain.

## Method

### Overall Architecture

A three-stage pipeline: Decomposition (SAE) → Interpretation (automated pipeline) → Steering (feature activation setting).

### Sparse Autoencoder (SAE) for Representation Decomposition

The paper adopts OpenAI's TopK SAE architecture, inserted at layer 25 of LLaVA-NeXT-8B:

$$\mathbf{z} = \text{TopK}(\text{ReLU}(\mathbf{W}_1(\mathbf{x} - \mathbf{b}_1) + \mathbf{b}_2))$$
$$\hat{\mathbf{x}} = \mathbf{W}_2 \mathbf{z} + \mathbf{b}_3$$

where $\mathbf{x} \in \mathbb{R}^{T \times d_l}$ denotes LLaVA hidden representations and $\mathbf{z} \in \mathbb{R}^{T \times d_s}$ the sparse representations. The model uses $d_s = 2^{17}$ features with $k = 256$.

The core reason SAEs yield monosemantic features is that $\mathbf{W}_2$ acts as an overcomplete dictionary and $\mathbf{z}$ as sparse coefficients. Under the sparsity constraint, dictionary vectors tend toward near-orthogonality (mutual incoherence), so each coordinate of $\mathbf{z}$ tends to represent a single semantic concept.

### Zero-Shot Feature Interpretation Pipeline

**Step 1: Identify Top Activating Images and Patches**

SAE activations are cached for approximately 46,684 images. For each feature, the Top-5 activating images are selected along the first dimension, and the most activated patches within those images are localized.

**Step 2: Automated Feature Interpretation**

Masks are applied to the Top-5 activating images (activated patches remain transparent; remaining regions are blacked out), which are then input to LLaVA-OV-72B to detect common patterns and generate explanations. When no common pattern can be identified, the system returns "Unable to interpret."

**Step 3: Reference Score Computation**

A small LLM refines the description → GroundingDINO-SAM generates segmentation masks → IoU between masks and SAE activation regions is computed as a quantitative measure of feature relevance.

### Feature Steering

Specific SAE feature values are set to influence model outputs:

$$\mathbf{z}[\mathcal{C}, j] = k$$

where $\mathcal{C}$ denotes the designated token set, $k$ the steering value, and $j$ the SAE feature index. The resulting $\hat{\mathbf{x}}$ replaces the original $\mathbf{x}$ fed into subsequent layers.

### Causal Localization of Model Behavior

Attribution analysis is performed over output tokens. Let the current output token be $v_c$ and the baseline token $v_b$. The influence of each feature is estimated via per-feature ablation (setting to 0) with a linear approximation:

$$I(i, j, v_c, v_b) \approx \left(\frac{\partial d(\mathbf{u})}{\partial \mathbf{z}}\right)^T (\hat{\mathbf{z}} - \mathbf{z})$$

## Key Experimental Results

### Main Results: Feature Interpretation Quality

| Concept | IoU ↑ (Random) | IoU ↑ (V-Interp) | CLIP Score ↑ (Random) | CLIP Score ↑ (V-Interp) |
|---|---|---|---|---|
| scene | 0.007 | 0.20 | 18.1 | 24.4 |
| object | 0.005 | 0.19 | 18.2 | 24.0 |
| part | 0.007 | 0.21 | 18.1 | 23.5 |
| material | 0.01 | **0.39** | 18.1 | 24.1 |
| texture | 0.007 | 0.21 | 18.4 | 20.9 |
| colour | 0.005 | 0.10 | 19.6 | 20.3 |
| **Total** | 0.005 | **0.20** | 18.2 | **23.6** |

V-Interp achieves an IoU approximately 40× higher than the random baseline and a CLIP Score 5.4 points higher, demonstrating strong alignment between interpretations and activation regions.

### Ablation Study: Cross-Layer Analysis

| Layer | IoU ↑ | CLIP Score ↑ |
|---|---|---|
| LLaVA (8th) | 0.30 | 22.82 |
| LLaVA (25th) | 0.31 | 24.92 |
| LLaVA (32th) | 0.40 | 26.55 |
| Random | 0.005 | 18.2 |

Both IoU and CLIP Score improve consistently with layer depth, confirming that deeper layers encode higher-level, more interpretable visual semantics.

### Consistency Evaluation

| Concept | GPT-4o Consistency | Human Consistency |
|---|---|---|
| scene | 0.93 | 0.70 |
| object | 0.84 | 0.85 |
| material | 1.00 | 0.95 |
| **Total** | **0.89** | **0.75** |

Both GPT-4o and human evaluations confirm high consistency (89% and 75% overall, respectively).

### Hallucination Correction Case Study

On a HallusionBench example, LLaVA incorrectly answers "Yes" to the question "Does the image show Bolivia?"

Attribution analysis reveals:
- **Visual attribution**: The model correctly attends to key map regions (legend, country names, etc.)
- **Textual attribution**: The token "Bolivia" contributes most strongly to the "yes" response → the model is misled by pretrained world knowledge

**Correction**: Clamping OCR-related features (e.g., features associated with the text "Barcelona") encourages the model to rely more on visual evidence, successfully eliminating the hallucination.

## Highlights & Insights

1. **First automated feature interpretation framework for multimodal models**: The paradigm of using large models to interpret smaller models is extended from LLMs to LMMs.
2. **Discovery of affective features**: LMMs contain internal features associated with concepts such as "happiness," "sadness," and "hunger/greed," which can be steered to induce emotional expression.
3. **Cross-modal invariant features**: Visual actions such as "eating" and textual concepts such as "greed/hunger" share unified features, providing evidence that LMMs develop cross-modal unified semantic representations internally.
4. **Causal analysis of hallucination**: Empirical findings show that hallucinations do not arise because the model fails to attend to the correct visual regions, but rather because pretrained priors from text tokens override visual evidence.
5. **Rich low-level visual features**: LMMs exhibit substantially more low-level visual features (shapes, textures, colors) than purely text-based LLMs.

## Limitations & Future Work

- SAE training is computationally expensive ($2^{17}$ features); validation is limited to a single model, LLaVA-NeXT-8B.
- Feature interpretation relies on a larger model (LLaVA-OV-72B); the approach is inapplicable when the largest available model is itself the subject of analysis.
- Only 576 base image tokens are analyzed; high-resolution tokens under the Anyres strategy are not covered.
- Quantitative evaluation of steering effects is challenging; results currently rely primarily on case studies.
- Hallucination correction is demonstrated only on individual examples, lacking systematic evaluation.

## Related Work & Insights

- **Anthropic's Dictionary Learning** and **Templeton et al.** demonstrated that SAEs can learn monosemantic features in LLMs; this work extends those findings to the multimodal setting.
- **Network Dissection** interprets neurons via predefined concept annotations but is ill-suited to the open-ended semantics of LMMs.
- The idea of correcting hallucinations via feature steering may inspire new inference-time intervention methods that require no modification of model parameters.
- The discovery of affective features suggests that LMMs may naturally develop human-like affective understanding mechanisms.

## Rating

⭐⭐⭐⭐ — A pioneering effort in extending mechanistic interpretability tools to the LMM domain, offering rich qualitative insights. The discovery of affective features and the causal analysis of hallucinations are particularly compelling. Weaknesses include limited quantitative evaluation and reproducibility constraints due to computational cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](../../AAAI2026/multimodal_vlm/multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)

</div>

<!-- RELATED:END -->
