---
title: >-
  [Paper Note] Preventing Shortcuts in Adapter Training via Providing the Shortcuts
description: >-
  [NeurIPS 2025][Image Generation][Adapter Training] This paper proposes Shortcut-Rerouted Adapter Training, which actively provides dedicated pathways for confounding factors during adapter training (e.g.…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Adapter Training"
  - "Shortcut Learning"
  - "Disentanglement"
  - "Personalized Generation"
  - "Identity Preservation"
date: 2026-05-08
content_hash: 75102ffd0af253f3
---

# Preventing Shortcuts in Adapter Training via Providing the Shortcuts

**Conference**: NeurIPS 2025
**arXiv**: [2510.20887](https://arxiv.org/abs/2510.20887)
**Code**: [Project Page](https://snap-research.github.io/shortcut-rerouting/)
**Area**: Diffusion Models / Image Generation
**Keywords**: Adapter Training, Shortcut Learning, Disentanglement, Personalized Generation, Identity Preservation

## TL;DR

This paper proposes Shortcut-Rerouted Adapter Training, which actively provides dedicated pathways for confounding factors during adapter training (e.g., a LoRA absorbing distribution shifts, a ControlNet absorbing pose/expression), thereby constraining the adapter to learn only the target attribute (e.g., identity). The auxiliary modules are discarded at inference time, yielding a disentangled adapter.

## Background & Motivation

Adapters (e.g., LoRA, IP-Adapter) have become a core mechanism for extending large-scale text-to-image (T2I) models. However, adapter training faces a fundamental challenge:

**Shortcut Learning Problem**: Adapters are typically trained via single-image reconstruction of a target. A single image, however, encodes all attributes simultaneously—identity, pose, expression, lighting, background, etc. The reconstruction loss is agnostic to target vs. nuisance attributes, driving the adapter to encode all of them—i.e., to learn "shortcuts."

**Concrete Manifestation**: An adapter intended solely to inject a person's identity will also replicate the reference image's expression and head pose, causing the generated output to ignore expression and pose specifications in the text prompt. Moreover, the distribution gap between the fine-tuning dataset and the base model is absorbed by the adapter, producing artifacts such as background degradation and anatomical distortion.

**Limitations of Prior Work**: Methods such as InfU and PuLID each have their strengths but still suffer from reduced identity fidelity or expression/pose leakage. Simple heuristics like background masking address only a subset of confounding factors.

The core insight is remarkably elegant: **the most effective way to prevent an adapter from learning unwanted shortcuts is to actively provide those shortcuts during training.**

## Method

### Overall Architecture

Adapter training is formalized within a probabilistic framework. An observed image $X \sim p(X|T,C)$, where $T$ denotes target factors (e.g., identity) and $C$ denotes confounding factors (e.g., pose, distribution shift). Standard training minimizes $\mathbb{E}[\mathcal{L}(G(\mathcal{A}(X)), X)]$, implicitly encouraging $\mathcal{A}(X)$ to encode both $T$ and $C$.

Shortcut-Rerouted training modifies the generation process as:

$$\hat{X} = G(\mathcal{A}(X), \mathcal{S}_C(C))$$

where $\mathcal{S}_C$ is an auxiliary module that supplies the confounding factors $C$ directly to the generator. The training objective becomes:

$$\mathbb{E}[\mathcal{L}(G(\mathcal{A}(X), \mathcal{S}_C(C)), X)]$$

Since $C$ is already "explained" by $\mathcal{S}_C$, the adapter $\mathcal{A}(X)$ has no incentive to encode them. At inference time, $\mathcal{S}_C$ is removed, recovering a clean adapter.

### Key Designs

1. **SR-LoRA: Absorbing Distribution Shift**: A significant domain gap exists between fine-tuning datasets (e.g., studio portraits) and the base model (e.g., Flux). A LoRA module is pre-trained on the fine-tuning dataset to absorb dataset-specific style, lighting, and low-level features. During adapter training, this LoRA is frozen, making the identity encoder $\mathcal{A}$ the sole active module, focused exclusively on learning identity representations. The LoRA is removed at inference time, enabling cross-domain generalization.

2. **SR-CN: Absorbing Pose and Expression Leakage**: A frozen, pre-trained ControlNet module $\mathcal{S}_{CN}$ processes pose and expression maps extracted from training images (via pose estimation and keypoint detection). The training objective is:

$$\mathcal{L}(G(\mathcal{A}(T), \mathcal{S}_{CN}(C_{CN})), X)$$

The ControlNet assumes responsibility for reconstructing pose and expression, freeing the adapter to focus solely on identity injection. This renders the adapter robust across diverse poses and expressions.

3. **Modular Composition**: SR-LoRA and SR-CN can be combined, and a background adapter (SR-BG) can further prevent lighting leakage. SR-LoRA+CN+BG simultaneously accounts for pose and background, ensuring that only the target identity is injected.

### Loss & Training

Training is based on FLUX.1 Dev (DiT backbone + Conditional Flow Matching objective). Identity encoding uses `openai/clip-vit-large-patch14`. Training configuration: 8×A100 GPUs, AdamW optimizer (lr=5e-5), global batch size 32, 250K iterations. Standard inference configuration: IP scale 1.0, CFG 3.5, 28 steps, 1024×1024 resolution. All newly added projection and fusion layers are zero-initialized to maintain compatibility with the pre-trained model.

## Key Experimental Results

### Main Results — "Face" Adapters

| Method | LLM Id.↑ | FaceNet Id.↑ | LLM Expr.↑ | EMOCA Expr.↑ | Head Pose↓ | Prior(LPIPS)↓ |
|--------|----------|-------------|-----------|-------------|-----------|-------------|
| InfU | 3.382 | 0.740 | 3.766 | 0.542 | 17.714 | 0.449 |
| PuLID | 4.283 | 0.774 | 3.590 | 0.489 | 17.535 | 0.458 |
| IPA (baseline) | 4.793 | 0.715 | 3.071 | 0.347 | 16.120 | 0.480 |
| **SR-LoRA IPA** | 4.719 | 0.671 | 3.429 | 0.458 | 13.270 | 0.433 |
| **SR-CN IPA** | **4.794** | 0.712 | **3.693** | **0.580** | **12.676** | **0.394** |

### Ablation Study — "Body" Adapters

| Method | LLM Id.↑ | FaceNet Id.↑ | LLM Expr.↑ | Head Pose↓ | Body Pose↓ | Prior(LPIPS)↓ |
|--------|----------|-------------|-----------|-----------|-----------|-------------|
| InstantX | 2.993 | 0.353 | 3.474 | 25.97 | 186.75 | 0.508 |
| IPA (baseline) | 4.599 | 0.573 | 3.300 | 20.70 | 167.40 | 0.457 |
| **SR-CN IPA** | **4.651** | **0.586** | **3.526** | **18.05** | **137.69** | **0.413** |

### Key Findings

- **Significant improvement in head pose control**: SR-CN reduces head pose error from 16.12° to 12.68° (Face) and from 20.70° to 18.05° (Body), demonstrating that ControlNet effectively absorbs pose shortcuts.
- **Substantial improvement in prior preservation**: Prior LPIPS decreases from 0.480 to 0.394, indicating that the SR modules effectively mitigate the adapter's corruption of the model prior.
- **Recovery of expression controllability**: LLM Expr. improves from 3.071 to 3.693, and EMOCA from 0.347 to 0.580; the adapter no longer copies the reference expression and responds to text instructions.
- **Identity fidelity largely preserved**: FaceNet Id. shows only a marginal decline (0.715→0.712), confirming that disentanglement does not sacrifice core identity information.
- **Complementary effects of combined SR modules**: SR-LoRA addresses quality degradation, SR-CN preserves pose priors, and SR-BG suppresses lighting leakage.

## Highlights & Insights

- **"Fight fire with fire" design philosophy**: Rather than opposing shortcut learning, the method actively provides shortcut pathways, removing the adapter's incentive to encode confounding factors. This counterintuitive yet elegant solution reveals a general principle.
- **Modularity and composability**: SR modules can be flexibly combined (LoRA, ControlNet, background adapter), with each module absorbing a specific confounding factor, achieving a clear separation of responsibilities.
- **Zero inference overhead**: Auxiliary modules are used exclusively during training and are fully removed at inference time, incurring no additional computational cost.
- **Generalization across settings**: Effectiveness across both Face and Body settings demonstrates the generality of the approach.

## Limitations & Future Work

- Validation is currently limited to IP-Adapter; applying the framework to stronger baselines (e.g., InfU) may yield better overall performance.
- Only encoder-type adapters have been evaluated; the behavior of LoRA as the primary adapter (e.g., style LoRA with layout bias removal) remains unexplored.
- The identity enhancement capability raises deepfake risks, necessitating watermarking and usage restrictions.
- SR-CN is dependent on the quality and coverage of the pre-trained ControlNet.
- Performance under extreme pose variation or rare expressions has not been analyzed in depth.

## Related Work & Insights

- The core idea of Shortcut Rerouting—preventing the target model from learning confounding factors by explicitly modeling them—is generalizable to a variety of generative tasks, including style transfer and video editing.
- The approach shares conceptual alignment with confounder adjustment in causal inference, offering a new methodological perspective on the entanglement problem in generative models.
- The SR-LoRA strategy for handling distribution shift is broadly applicable to all adapter fine-tuning scenarios, particularly domain adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The core idea is concise yet profound; "providing shortcuts to eliminate shortcuts" constitutes an inspiring general principle.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Both Face and Body settings include comparisons with comprehensive metrics (identity, expression, pose, prior); however, ablations over additional baseline combinations are limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formalization is clear; the transition from mathematical framework to instantiation is natural and the narrative is compelling.
- **Value**: ⭐⭐⭐⭐ The method is simple to implement, has direct practical utility for the personalized generation community, and the general principle can inspire broader research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICLR 2026\] Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](../../ICLR2026/image_generation/mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
