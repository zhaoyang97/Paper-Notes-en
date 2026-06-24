---
title: >-
  [Paper Note] Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models
description: >-
  [ICLR 2026][Interpretability][Mechanistic Interpretability] This paper discovers that when VLMs perform spatial reasoning, they bind the location information of objects from the visual input as **linear "spatial ID" vectors** to the text activations of the corresponding object words. Reasoning is subsequently completed in the language space. Causal interventions prove that modifying only this spatial ID can systematically flip the model's judgment of "left/right" and "far/nea…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "Spatial Reasoning"
  - "Linear Representation Hypothesis"
  - "Causal Intervention"
  - "Activation Manipulation"
  - "Video Temporal Reasoning"
date: 2026-05-08
content_hash: b8c1da4f22da9017
---

# Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2zXRGiorSu](https://openreview.net/forum?id=2zXRGiorSu)  
**Code**: [https://github.com/Raphoo/linear-mech-vlms](https://github.com/Raphoo/linear-mech-vlms)  
**Area**: Mechanistic Interpretability / VLM Spatial Reasoning  
**Keywords**: Mechanistic Interpretability, Spatial Reasoning, Linear Representation Hypothesis, Causal Intervention, Activation Manipulation, Video Temporal Reasoning  

## TL;DR
This paper discovers that when VLMs perform spatial reasoning, they bind the location information of objects from the visual input as **linear "spatial ID" vectors** to the text activations of the corresponding object words. Reasoning is subsequently completed in the language space. Causal interventions prove that modifying only this spatial ID can systematically flip the model's judgment of "left/right" and "far/near." This mechanism is also extended to "temporal IDs" in video models.

## Background & Motivation
**Background**: State-of-the-art (SOTA) VLMs (such as LLaVA, Qwen, InternVL, and Gemma) have demonstrated stable performance on spatial VQA tasks, such as determining "is the dog to the left or right of the cat." However, how they achieve this remains largely a black box. Meanwhile, mechanistic interpretability in LLMs has accumulated significant evidence for "linearity"—including the linear representation hypothesis, activation patching, and linear binding ID circuits responsible for relational reasoning.

**Limitations of Prior Work**: A typical VLM architecture involves a "vision encoder projecting images into tokens → prepending them to text tokens → feeding them into an aligned LLM." Visual/geometric information (image patches) and text representations (language queries) must "converge" at a certain layer. However, it remains unclear where this convergence point is, what form the information takes after merging, and whether it causally determines the output.

**Key Challenge**: On one hand, there are repeatedly verified "linear reasoning circuits" in LLMs; on the other hand, cross-modal spatial reasoning in VLMs remains opaque. Is the transfer of 2D coordinates from images into the language reasoning stream a complex non-linear distributed process, or can it be captured by a simple linear structure?

**Goal**: The authors propose three progressive questions: Q1: Can a linear model characterize the internal structures driving spatial reasoning in VLMs? Q2: With this linear model, how can it be used to diagnose and improve SOTA VLMs? Q3: Do video models utilize similar linear mechanisms along the temporal axis?

**Core Idea** (**Text-anchored Linear Spatial IDs**): The authors hypothesize that VLMs, in early layers, **linearly bind** the spatial localization of objects to the activations of those object words, forming a latent structure called a **spatial ID**. Subsequent "left/right/far/near" reasoning is essentially a linear readout of this ID within the text space. By extracting the spatial ID and manipulating it to alter the model's belief, they can prove this linear mechanism is *causal* rather than merely correlational.

## Method

### Overall Architecture
The methodology follows the sequence of "Hypothesis → Localization → Extraction → Causal Verification → Application → Extension." First, **mirror-swap experiments** are used to locate the layers and tokens where spatial information converges (found to be concentrated on object word tokens in intermediate layers). Then, **object grid averaging** is used to extract the linear spatial ID from activations, accompanied by an analytical derivation of its emergence. Next, **arbitrary ID manipulation and adversarial manipulation** are employed for causal intervention, proving that changing only the spatial ID can flip model beliefs. Finally, the spatial ID is utilized as a diagnostic tool (to determine if VLM failures occur in the vision encoder or during cross-modal integration) and as a training signal (adding spatial ID loss for fine-tuning), with the entire process extended to video models to derive **temporal IDs**.

```mermaid
flowchart LR
    A[Image patches + Text query] --> B[Vision encoder projects to vision tokens]
    B --> C{Intermediate "modality alignment layers"}
    C -->|Attention writes position into object words| D["Object word activation<br/>= Semantics + Spatial ID Δ_L(i,j)"]
    D --> E[Linear readout in language space<br/>ℓ(LEFT)-ℓ(RIGHT) ≈ w·Δ_L]
    E --> F[Final spatial judgment]
    G[Mirror swap:<br/>Localizing convergence layers] -.Diagnosis.-> C
    H[Add/Subtract Δ_L:<br/>Causal intervention] -.Manipulate belief.-> D
```

### Key Designs

**1. Mirror-Swap Experiments: Locating the "Convergence Point" of Spatial Information using Activation Patching.** To determine if VLMs truly "isolate and transmit" spatial information internally, the authors feed the same text query with both the original image and its **horizontally mirrored** version. At an intermediate layer $L$, a subset $Q$ of activations $x_L$ from the original image is replaced with those from the mirrored version $y_L$ at corresponding positions. $Q$ is chosen from three types: all text tokens, all image patches, or only object word tokens. If critical spatial reasoning information is localized, overwriting it will flip the final belief. This is measured by a normalized metric: $\text{belief shift}_L = \frac{P_{x_{out}}(\text{GT}) - P_{\tilde{x}_{out,L}}(\text{GT})}{P_{x_{out}}(\text{GT}) - P_{y_{out}}(\text{GT})}$. Results show that image patches have a strong influence in shallow layers that decays with depth, text tokens become increasingly important in deep layers, and **object word tokens influence spatial beliefs only within a narrow band of intermediate layers**—identifying the "convergence layers."

**2. Extraction and Analytical Emergence of Spatial IDs: Isolating Linear Position Vectors by Averaging out Semantics.** Since spatial IDs are linearly bound to object word activations, they can be isolated by averaging over all positions for the same object to remove semantic information. For an object $o$ at position $(i,j)$ in an $m\times m$ grid, with object word activation $\phi_L$, the position-independent mean $\bar\phi_L^{(o)}$ is calculated. The spatial ID at that position is $\Delta_L^{(o)}(i,j)=\phi_L(o;I_{(i,j)}^{(o)},T^{(o)})-\bar\phi_L^{(o)}$. Averaging across $N$ objects yields a universal spatial ID $\Delta_L(i,j)$, from which horizontal/vertical vectors $h_L, v_L$ are derived. The authors provide an analytical sketch for its **emergence**: decomposing image patches as $x_p=s_p+P\psi(p)+\varepsilon_p$ (content + shared position basis + noise). When cross-modal attention peaks at the true object patch $p^\star$, the residual update approximates $\Delta_L(i,j)\approx \underbrace{W_{out}W_V P}_{M}\big(\psi(i,j)-\frac{1}{m^2}\sum_p\psi(p)\big)$—indicating the spatial ID is the result of the position encoding basis $\psi$ (RoPE or learned 2D embeddings) undergoing a **model-fixed linear transformation $M$**. Experimentally, a low-rank linear fit (rank-3) from position encodings to spatial IDs explains most variance ($R^2\gtrsim0.85$).

**3. Arbitrary ID Manipulation and Adversarial Manipulation: Flipping Beliefs to Prove Causality.** To establish causality, the authors perform "Add/Subtract ID" interventions: at layer $L$, the activation of an object token $x_L[q]$ is replaced with $x_L[q]+\Delta_L(i,j)-\tilde\Delta_L(i,j)$ (where $\tilde\Delta$ is the mirrored position ID). In 100 COCO-SPATIAL images, intervening with a "rightmost" ID significantly increases the model's belief of "on the right." The same manipulation controls "far/near" and ternary "in-between" relationships. Adversarial manipulation on 11 SOTA models shows that using spatial IDs most likely to reverse the original belief results in a **median belief flip rate of 64.6%**, compared to 29.5% for random noise of the same norm—a net gain of approximately 43.6%.

**4. Spatial IDs as Diagnostics and Training Signals: Localizing Failures and Improving Fine-tuning.** Spatial IDs are calculated per sample to decompose VLM failures. **Ground-Truth Deviation Experiments**: Projecting object activations onto spatial axes $V=[v_L, h_L]$ reveals that incorrect samples (red) have IDs that deviate significantly compared to correct samples (blue). **Image Masking Experiments**: Masking object bboxes versus random regions shows that LLaVA is more sensitive to masking the true object when it fails, highlighting vision encoder weaknesses. Finally, the mechanism is used as a supervisory signal: adding a **spatial ID loss** (cosine similarity between predicted and GT IDs) during Qwen2-2B fine-tuning results in faster generalization on COCO-Spatial (reaching 91% in 3.2k steps, ~6% higher than the baseline). This logic is also successfully applied to video models to derive and manipulate **temporal IDs**.

## Key Experimental Results

### Main Results: Belief Flip Rates under Adversarial Manipulation (11 SOTA VLMs, COCO-SPATIAL)

| Intervention Type | Belief Flip Rate (Median) |
|---|---|
| Spatial ID Manipulation | **64.6%** |
| Random Noise (Same Norm) | 29.5% |
| Net Influence of Spatial ID over Random | +43.6% (Avg) |

### Ablation Study: Spatial ID Fine-tuning (Qwen2-2B, COCO-Val Accuracy)

| Training Steps | 0 | 800 | 1600 | 2400 | 3200 |
|---|---|---|---|---|---|
| Control (LM loss only) | 0.77 | 0.83 | 0.84 | 0.85 | 0.85 |
| **With Spatial ID Loss** | 0.77 | 0.83 | 0.84 | 0.88 | **0.91** |
| Spatial ID Loss Value (↓) | 0.75 | 0.58 | 0.41 | 0.36 | 0.33 |

Rank-3 linear fit from position encoding $\psi$ to spatial ID $\Delta_L$ achieves $R^2\gtrsim0.85$, supporting the "linear transformation" hypothesis.

### Key Findings
- **Convergence Layer Localization**: Object word tokens influence spatial beliefs only in a narrow band of intermediate layers, while the influence of image patches decays and text tokens increases with depth.
- **Depth Mistaken for Height**: Manipulating LLaVA1.5-7B with spatial IDs for height shows that "above/below" beliefs are highly correlated with "front/behind," indicating a lack of independent depth representation.
- **Correlation between Maneuverability and Accuracy**: Models with higher zero-shot accuracy are easier to manipulate via spatial IDs, suggesting spatial ID strength is a valid metric of spatial capability.
- **Failure Attribution**: LLaVA's bottleneck lies in the vision encoder (coarse ViT granularity), while LLaMA's lies in cross-modal integration. Both are faithful to the received spatial ID during the language reasoning stage.
- **Temporal IDs**: Video models (LLaVA-Video, VideoLLaMA3, Qwen2.5) exhibit linearly separable **temporal IDs** for "early/late" frames that can manipulate "before/after" judgments.

## Highlights & Insights
- **Compression of Cross-modal Reasoning**: The elegance of the spatial ID lies in using an Occam's razor-level simple structure (a linear direction) to capture how image positions enter language reasoning across multiple models.
- **Solid Causal Loop**: The study moves beyond "probing" to "causal intervention," including random noise controls to ensure that effects are not due to generic activation perturbations.
- **Analytical Grounding**: The derivation from attention residual updates to linear transformations of position bases provides a first-principles explanation for the empirical findings.
- **Engineering Translation**: Converting interpretability into diagnostic tools and training signals represents a rare "explanation to improvement" closed loop.
- **Symmetry of Space and Time**: Extending the paradigm to video suggests that "linear binding + language readout" might be a universal framework for structured information in multi-modal models.

## Limitations & Future Work
- **Scope of Queries**: Analysis is limited to basic spatial (left/right, far/near) and appearance-based temporal queries; complex multi-hop reasoning remains unexplored.
- **Model Scale**: Experiments were conducted on models $\leq$ 14B; it is unclear if spatial ID circuits dominate in larger models or are replaced by distributed mechanisms.
- **Linearity as Approximation**: The authors acknowledge that real circuits are noisy and non-linear; spatial IDs may only capture one component of a more complex system.
- **Diagnostic Granularity**: Attributing failures to specific stages is statistical; individual failures may still stem from multiple components.
- **Scaling of Improvements**: Spatial ID loss was only tested on synthetic data for a 2B model; large-scale training is a necessary next step.

## Related Work & Insights
This work sits at the intersection of **mechanistic interpretability** and the **linear representation hypothesis**. It ports concepts such as linear binding IDs and activation patching from LLMs to VLMs. Compared to prior work characterizing VLM stages or using logit lenses (Jiang et al. 2025b), this paper's novelty lies in **explicitly characterizing the carrier of spatial information from image patches to text tokens (spatial ID) and proving its causality**. It suggests that the bottleneck of model spatial capability can be read from internal representation geometry and that multimodal structural information might follow a unified "linear binding" skeleton.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to compress VLM spatial reasoning into a causally manipulable linear ID with analytical derivation and temporal extension.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid evidence across 11 models with causal controls and downstream validation, though limited by model size and query complexity.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from Q1 to Q3; strong alignment between theory and experiments.
- **Value**: ⭐⭐⭐⭐⭐ Provides first-principles insight while offering practical tools for model diagnosis and improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context](mixing_mechanisms_how_language_models_retrieve_bound_entities_in-context.md)
- [\[ICLR 2026\] Inducing Dyslexia in Vision Language Models](inducing_dyslexia_in_vision_language_models.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs](cot_vectors_transferring_and_probing_the_reasoning_mechanisms_of_llms.md)
- [\[ICLR 2026\] A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models](a_comprehensive_information-decomposition_analysis_of_large_vision-language_mode.md)

</div>

<!-- RELATED:END -->
