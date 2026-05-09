---
title: >-
  [Paper Note] All-in-One Slider for Attribute Manipulation in Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Attribute Manipulation] The All-in-One Slider framework is proposed, which trains an Attribute Sparse Autoencoder on the text embedding space to decouple various facial attributes into sparse semantic directions. This enables a single lightweight module to achieve fine-grained continuous control over 52+ attributes, supporting multi-attribute combinations and zero-shot manipulation of unseen attributes.
tags:
  - CVPR 2026
  - Image Generation
  - Attribute Manipulation
  - Sparse Autoencoders
  - Text Embedding Decoupling
  - Continuous Control
  - Zero-shot Generalization
date: 2026-05-08
content_hash: 9644ccbb3ab5b8dc
---

# All-in-One Slider for Attribute Manipulation in Diffusion Models

**Conference**: CVPR 2026  
**arXiv**: [2508.19195](https://arxiv.org/abs/2508.19195)  
**Code**: [https://github.com/ywxsuperstar/ksaedit](https://github.com/ywxsuperstar/ksaedit)  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Attribute Manipulation, Sparse Autoencoders, Text Embedding Decoupling, Continuous Control, Zero-shot Generalization  

## TL;DR
The All-in-One Slider framework is proposed, which trains an Attribute Sparse Autoencoder on the text embedding space to decouple various facial attributes into sparse semantic directions. This enables a single lightweight module to achieve fine-grained continuous control over 52+ attributes, supporting multi-attribute combinations and zero-shot manipulation of unseen attributes.

## Background & Motivation
While T2I diffusion models generate high-quality images, fine-grained control over specific attributes remains challenging. Traditional methods either rely on prompt modifications, which lead to coarse and uncontrollable changes (e.g., adding "with a big smile" might alter hair, pose, or identity), or follow a "One-for-One" paradigm where an independent slider module is trained for each attribute (e.g., LoRA in ConceptSlider or attribute vectors in AttributeControl). The latter results in: (1) Parameter redundancy growing linearly with the number of attributes; (2) The need for retraining for new attributes; (3) Difficulties in combining multiple attributes.

## Core Problem
How can a single unified lightweight module achieve decoupled, continuous, and composable control over multiple visual attributes? The key challenge lies in attribute decoupling—mapping different attributes to independent representational directions so that adjusting one does not affect others.

## Method

### Overall Architecture
The method consists of two stages: (1) Unsupervised training stage—training a sparse autoencoder on a large set of text embeddings to construct a unified Attribute Latent Space (Att_latentspace); (2) Inference manipulation stage—obtaining sparse directions by encoding target attribute text and adding them to the original prompt embeddings for controlled generation.

Input: Text prompt + Attribute name + Control strength $\lambda$  
Output: Manipulated image  
Mechanism: Sparse decomposition performed on the intermediate layer embeddings of the text encoder.

### Key Designs

1.  **Attribute Sparse Autoencoder**: 2048-dimensional embeddings are extracted from the intermediate layers (Layer 11 and Layer 29) of the SDXL dual text encoders (CLIP 12-layer + OpenCLIP 32-layer). These are mapped to a 32,768-dimensional high-dimensional space (expansion factor 16×) via a linear encoder. Top-k ($k=128$) activation is used to retain the most active dimensions, followed by linear decoding back to the original dimension. The core idea is borrowed from Sparse Autoencoders in LLMs—high-dimensional sparse representations naturally map different semantic concepts to distinct basis vectors, achieving decoupling.

2.  **Top-k Sparsity + Dead Neuron Remedy**: The encoding is $z_{ALS} = \text{Top-k}(\text{ReLU}(W_{enc}(x - b_{pre}) + b_{enc}))$ and decoding is $\hat{x} = W_{dec} z_{ALS} + b_{pre}$. To address the dead neuron problem common in sparse coding (where neurons never activate), an auxiliary mechanism is introduced: residuals $r = x - \hat{x}$ are calculated at each step, and the $k_{aux}=256$ most inactive neurons are used to reconstruct the residual. An auxiliary loss $\mathcal{L}_{aux} = \|r - \hat{r}\|_2^2$ encourages these neurons to learn meaningful representations.

3.  **Attribute Manipulation Mechanism**: Given a target attribute text $A$, the sparse direction $\text{ENC}(x_A)$ is computed. The manipulation formula is $x_{manipulated} = x + W_{dec}(\lambda \times \text{ENC}(x_A))$, where $\lambda$ controls the strength—increasing $\lambda$ enhances the attribute, while decreasing it weakens it. Since different attributes activate different subsets of neurons in the sparse space, multi-attribute combinations are achieved by simply overlaying attribute directions without conflict.

4.  **Multi-Subject Attribute Manipulation Extension**: An Attention Pooling Aggregator (AAg) module is introduced to extract pure attribute directions $\Delta z = \text{AAg}(z^+) - \text{AAg}(z^-)$ using paired sentences (with/without the target attribute). This allows precise local manipulation of a target subject (e.g., "woman" or "man"), supplemented by a consistency loss $\mathcal{L}_{cons}$ to protect non-target regions.

### Loss & Training
-   Total Loss: $\mathcal{L} = \mathcal{L}_{mse} + \alpha \mathcal{L}_{aux}$, where $\alpha = 0.1$
-   Training Data: 52 facial attributes × 1000 samples/attribute = 52,000 text samples
-   Training Volume: 400 million tokens, approximately 97,656 steps
-   Optimizer: Adam, learning rate $4 \times 10^{-4}$, batch size 4096
-   Hardware: Single RTX 4090 GPU

## Key Experimental Results

### Quantitative Comparison: Single vs. Multi-Attribute Manipulation

| Setup | Method | Old QS/IS | Smile QS/IS | Makeup QS/IS |
| :--- | :--- | :--- | :--- | :--- |
| Single | CSlider | 3.79/0.43 | 4.14/0.50 | 4.54/0.65 |
| Single | AttControl | 4.04/0.60 | 4.40/0.70 | 4.27/0.60 |
| Single | **Ours** | **4.05/0.72** | 4.26/0.64 | **4.29/0.74** |
| Multi | CSlider | 4.15/0.50 | 3.80/0.52 | 4.06/0.48 |
| Multi | AttControl | 3.67/0.38 | 4.06/0.63 | 4.25/0.51 |
| Multi | **Ours** | **4.21/0.69** | **4.43/0.63** | **4.30/0.64** |

The advantage in multi-attribute scenarios is clear—for Old+Makeup, the QS is 4.43 vs. the second-best 4.06, a significant lead.

### Comparison vs. Original Embeddings

| Method | Avg QS | Avg IS |
| :--- | :--- | :--- |
| Original Embedding | 3.990 | 0.502 |
| SAE Direction | **4.202** | **0.698** |

SAE directions improve QS by 0.212 and IS by 0.196 compared to direct text embeddings.

### Ablation Study
-   **Layer Selection**: The 10/28 combination is optimal; deeper layers have stronger semantics but lead to a decline in identity preservation.
-   **Control Strength $\lambda$**: $\lambda=0.15$ results in under-editing, while $\lambda=0.30$ provides strong attribute expression but lower identity retention. The "age" attribute is most sensitive to $\lambda$ (highly entangled with identity features).
-   **Continuity**: The linearity of geometric changes in the edited area is $R^2 = 0.973$, surpassing CSlider (0.966) and AttControl (0.962).
-   **Model Generalization**: The same SAE can transfer to SD v1.4, SDXL-Turbo, and FLUX (using Layer 23 of the T5 encoder).

## Highlights & Insights
-   **Design Motivation**: Transferring the concept of Sparse Autoencoders from LLM interpretability to T2I attribute control—high-dimensional sparse spaces naturally achieve semantic decoupling—is a highly creative cross-domain application.
-   **One-for-All Training**: Breaks the One-for-One paradigm, covering 52 attributes + zero-shot generalization to unseen attributes like ethnicity and celebrities.
-   **Lightweight**: SAE parameters are far fewer than the total parameters required to train a LoRA for every attribute.
-   **Excellent Composability**: Multi-attribute overlays occur without conflict because different attributes activate different subsets of dimensions in the sparse representation.
-   **Versatility**: Extends to photography style control (40 styles) and multi-subject scenarios.

## Limitations & Future Work
-   **Attribute Entanglement**: The "age" attribute remains highly entangled with identity features; identity preservation drops significantly at high $\lambda$.
-   **Training Data Dependency**: While zero-shot generalization is supported, the initial 52 attributes still require carefully designed text templates.
-   **Text-Space Only**: Operates solely in the text embedding space; lack of manipulation in visual feature layers may limit fine control over spatial local attributes.
-   **Subjective Evaluation**: Relies heavily on VLM (Qwen2.5-VL) scoring and ArcFace identity consistency, lacking more extensive human evaluation.
-   Combination with spatial conditional methods like ControlNet has not been explored.

## Related Work & Insights
-   **vs ConceptSlider (ECCV 2024)**: ConceptSlider requires training one LoRA adapter per attribute, a typical "One-for-One" approach; All-in-One Slider covers all attributes with a single module and achieves significantly higher multi-attribute QS.
-   **vs AttributeControl (CVPR 2025)**: AttControl achieves continuous control but requires attribute-level supervision and paired data; this work achieves similar effects via unsupervised SAEs and supports zero-shot generalization.
-   **vs SAeUron (CVPR 2025)**: SAeUron uses SAEs for concept unlearning, focusing on model interpretability; this work applies SAEs for active, controllable attribute manipulation.

## Rating
-   Novelty: ⭐⭐⭐⭐ Transferring LLM SAE ideas to T2I attribute control breaks the One-for-One paradigm.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Covers single/multi-attribute, zero-shot, multi-model, multi-subject, and styles with complete ablations.
-   Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed framework description, though some technical details are in the appendix.
-   Value: ⭐⭐⭐⭐ Provides a more efficient and flexible paradigm for attribute control with practical application value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)
- [\[CVPR 2026\] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[CVPR 2026\] TokenLight: Precise Lighting Control in Images using Attribute Tokens](tokenlight_precise_lighting_control_in_images_using_attribute_tokens.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)

<!-- RELATED:END -->
