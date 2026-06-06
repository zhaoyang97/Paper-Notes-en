---
title: >-
  [Paper Note] MENTOR: Efficient Autoregressive Image Generation with Balanced Multimodal Control
description: >-
  [ACL2026][Image Generation][Autoregressive Image Generation] MENTOR utilizes a unified autoregressive decoder and two-stage multimodal training to align reference images and text instructions into the same generation pre…
tags:
  - "ACL2026"
  - "Image Generation"
  - "Autoregressive Image Generation"
  - "Multimodal Control"
  - "Two-stage Training"
  - "DreamBench++"
  - "Generation Efficiency"
date: 2026-05-08
content_hash: a135b0ff551c20fe
---

# MENTOR: Efficient Autoregressive Image Generation with Balanced Multimodal Control

**Conference**: ACL2026  
**arXiv**: [2507.09574](https://arxiv.org/abs/2507.09574)  
**Code**: Project page https://haozhezhao.github.io/MENTOR.page (GitHub link not provided in cache)  
**Area**: Multimodal Conditional Image Generation / Autoregressive Generation  
**Keywords**: Autoregressive Image Generation, Multimodal Control, Two-stage Training, DreamBench++, Generation Efficiency

## TL;DR
MENTOR utilizes a unified autoregressive decoder and two-stage multimodal training to align reference images and text instructions into the same generation prefix. With a limited training budget of only 3M data points and approximately 1.5 days on 8 A100 GPUs, it achieves a superior balance between concept preservation and prompt following.

## Background & Motivation
**Background**: Text-to-image models have achieved high-quality generation, but real-world applications often require fine-grained control via "text + reference image + multi-image context," such as preserving subject identity, changing scenes based on text, or performing image restoration and segmentation.

**Limitations of Prior Work**: Many multimodal generation systems rely on diffusion models and additional alignment modules like adapters, regression heads, or specialized embeddings. While they can utilize image conditions, they often suffer from modal imbalance: models either over-copy the reference image while ignoring text or follow the text while losing subject details. Furthermore, training data requirements, model scale, and computational costs remain high.

**Key Challenge**: Complex multimodal control requires the model to simultaneously preserve pixel-level visual details and semantic-level text instructions. However, there is an inherent gap between visual and linguistic representations. Reconstruction-only tasks may lead to simple copying, while text-to-image tasks lack identity constraints from reference images.

**Goal**: The authors aim to build a resource-friendly autoregressive multimodal generation framework to verify that a stable balance between concept preservation and text following can be achieved without complex diffusion control modules, even with smaller models and limited training data.

**Key Insight**: The paper discretizes images into VQGAN tokens and allows a transformer decoder to generate images token-by-token, similar to a language model. A multimodal encoder projects visual and text inputs into a shared latent prefix, and task mixing during training is used to explicitly shape alignment and modal balance.

**Core Idea**: Replace heavy diffusion control pipelines with "Autoregressive Unified Token Generation + Stage 1 Alignment + Stage 2 Instruction Balancing" to learn controllable multimodal image generation in a low-resource setting.

## Method
MENTOR functions as a multimodal-conditional language model for image tokens. Given a reference image and text instruction, CLIP and FlanT5 encoders extract visual and linguistic features. A lightweight MLP connector projects visual tokens into a shared space used by the decoder. Subsequently, an autoregressive decoder initialized from LlamaGen predicts VQGAN image tokens step-by-step based on these prefix tokens, which are finally restored by the VQGAN decoder.

### Overall Architecture
Inputs can be images, text, or combinations. The multimodal encoder produces a condition sequence $H=(h_1,\dots,h_M)$. The autoregressive decoder learns $p(y_i|y_{<i},H)$ under teacher forcing, where $y$ is the sequence of discrete image tokens. Training is divided into two stages: the first emphasizes pixel and semantic alignment; the second uses multi-task instruction tuning to balance reference images and text instructions.

### Key Designs
1.  **Unified Autoregressive Generation Architecture**:
    - **Function**: Unifies multimodal conditions and image outputs into a token sequence modeling problem.
    - **Mechanism**: The visual encoder uses CLIP-Large-Patch14 and the text encoder uses FlanT5-XL. An MLP projects visual tokens into a latent space consumable by the decoder. The decoder inherits from LlamaGen-XL and generates images token-by-token using the VQGAN vocabulary.
    - **Design Motivation**: Diffusion models' random sampling and cross-attention control are less direct. Autoregressive token generation embeds the mapping between conditions and outputs into a single next-token objective, facilitating low-cost training and subsequent reinforcement learning.

2.  **Stage 1 Multimodal Alignment Training**:
    - **Function**: Establishes pixel-level and semantic-level alignment between reference images and output tokens.
    - **Mechanism**: Stage 1 includes image reconstruction, object segmentation, and text-to-image tasks. Reconstruction reinforces pixel fidelity, segmentation forces the model to focus on spatial structure and semantic objects, and T2I maintains basic generation capabilities.
    - **Design Motivation**: Pure image reconstruction leads to "copy-paste" behavior where the model might not understand object semantics. Adding segmentation tasks requires the model to bind "observed visual details" with "text-specified objects."

3.  **Stage 2 Multimodal Instruction Tuning**:
    - **Function**: Enhances the model's ability to follow complex multimodal instructions and mitigates dominance by a single modality.
    - **Mechanism**: Stage 2 retains T2I and segmentation while adding image recovery and subject-driven generation. Image recovery requires restoring originals from perturbations (rotation, scaling, tiling, random backgrounds), and subject-driven tasks require preserving identity while executing text instructions.
    - **Design Motivation**: Image recovery acts as a regularizer, forcing the model to process both imagery and text. Subject-driven generation directly addresses real-world multimodal generation needs.

### Loss & Training
The training objective is the cross-entropy loss of image tokens: maximizing the conditional probability of each output token under teacher forcing. The authors also employ classifier-free guidance: during training, condition $H$ is replaced with an unconditional embedding with probability $p$; during inference, the guidance is adjusted via $\ell_g=\ell_u+(\ell_c-\ell_u)\times\lambda$.

Implementation: Stage 1 freezes the multimodal encoder, training the projector and generator for 1 epoch with a global batch size of 128 and a learning rate of $5\times10^{-4}$. Stage 2 fine-tunes the entire model (except the vision encoder) for 2 epochs with a learning rate of $1\times10^{-4}$. Training utilized 8 80GB A100 GPUs for ~1.5 days total (Stage 1: ~2.48M data, 14 hours; Stage 2: ~1.3M data, 20 hours).

## Key Experimental Results

### Main Results
| Method | Data | Model Size | DreamBench++ CP | DreamBench++ PF | CP·PF | CP/PF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Lumina-mGPT | 10M | 7.00B | 0.91 | 0.25 | 0.23 | 3.63 |
| DreamEngine | 21M | 10.50B | 0.68 | 0.37 | 0.26 | 1.84 |
| IP-A ViT-G | 10M | 2.50B | 0.59 | 0.64 | 0.38 | 0.92 |
| Mentor | 3M | 2.31B | 0.56 | 0.84 | 0.47 | 0.66 |
| DreamBooth-L | - | 2.60B | 0.60 | 0.87 | 0.52 | 0.69 |

### Ablation Study
| Configuration | CP | PF | CP·PF | Description |
| :--- | :--- | :--- | :--- | :--- |
| w/o Obj. Seg. in Stage 1 | 0.252 | 0.479 | 0.121 | Reconstruction degrades to copying, lacks semantic constraints |
| w/o Stage 1 Alignment | 0.179 | 0.673 | 0.120 | Concept preservation collapses severely |
| w/o Image Recovery | 0.661 | 0.284 | 0.188 | Over-reliance on vision, poor text following |
| w/o Object Segmentation | 0.412 | 0.918 | 0.378 | High PF, but decreased visual fidelity |
| w/o Multimodal T2I Task | 0.407 | 0.910 | 0.370 | Insufficient visual preservation |
| Mentor | 0.555 | 0.839 | 0.466 | Best balance |

### Key Findings
- MENTOR's advantage is not having the highest single CP or PF score, but a more balanced CP·PF. Lumina-mGPT has a high CP of 0.91 but PF of only 0.25, indicating it mostly copies the reference.
- Training efficiency is prominent: MENTOR uses only 3M data and ~1.5 days on 8 A100s, whereas Kosmos-G reportedly requires 256 GPUs for 3 days.
- In image reconstruction, MENTOR achieves distances of 0.1008 / 0.0867 on COCO / JourneyDB, outperforming DreamEngine (0.2065 / 0.2052) and EMU2-Gen (0.3828 / 0.2869).
- Multi-image training and GRPO further improve performance: CP·PF reaches 0.486 with multi-image and 0.527 with GRPO.

## Highlights & Insights
- The paper decomposes "multimodal control" into two clear stages: first ensuring the model understands image details, then teaching it not to be hijacked by either text or image. This training logic is more interpretable than simple data scaling.
- The use of CP·PF and CP/PF metrics fits the problem well. Many models appear strong in concept preservation but high CP/PF ratios expose their tendency to ignore text instructions.
- The autoregressive framework is naturally suited for reinforcement learning. The paper uses GRPO to prove that token-level RL can directly improve multimodal generation behavior, a path not easily replicated by diffusion-based control methods.
- The combination of "Segmentation + Reconstruction" has high transfer value. For video generation, 3D generation, or robotic vision, structured auxiliary tasks can prevent models from learning shallow copying.

## Limitations & Future Work
- The authors acknowledge that the goal is not to set a new SOTA for absolute image quality but to verify multimodal balancing under low resources; current quality is limited by backbones like LlamaGen/VQGAN.
- Regarding text-to-image, the model still faces challenges in spatial reasoning, object counting, fine-grained human rendering, and stylization capability.
- Evaluations for safety, fairness, and potential misuse are incomplete, especially regarding risks involving people, identity, and copyrighted content in multimodal systems.
- Reaching strong competitiveness on specific tasks may still require stronger encoders/generators and specialized data; the current "general framework" is more an effective starting point than a replacement for large-scale commercial models.

## Related Work & Insights
- **vs IP-Adapter / BLIP-Diffusion**: These methods usually add image condition modules to diffusion models. MENTOR converts visual conditions directly into AR prefixes, unifying training and inference.
- **vs Kosmos-G / Emu2**: While they emphasize large-scale multimodal models and unified generation, MENTOR focuses on modal balance and controllable generation under low resources.
- **vs Lumina-mGPT / Unified-IO2**: These AR models have strong concept preservation but gravitate toward visual dominance. MENTOR explicitly reduces CP/PF imbalance through two-stage task combinations.
- **Insight**: When building controllable generation, one should not only pursue "likeness to the reference image" but treat prompt following as an equally core metric; otherwise, the model may just be a sophisticated copier.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Autoregressive multimodal generation is not a new paradigm, but the two-stage task design and low-resource balance objective are highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers main experiments, ablations, reconstruction, multi-image, GRPO, and human evaluation, though absolute quality on complex tasks can be improved.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology, sufficient tables to support conclusions, and detailed training parameters in the appendix.
- Value: ⭐⭐⭐⭐☆ Highly valuable for resource-constrained teams building multimodal systems, providing a practical evaluation paradigm for analyzing modal imbalance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](../../ICLR2026/image_generation/locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](../../CVPR2026/image_generation/consistcompose_multimodal_layout_control.md)
- [\[ACL 2026\] Multimodal Large Language Models for Multi-Subject In-Context Image Generation](multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)
- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](../../ICLR2026/image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](../../ICLR2026/image_generation/from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)

</div>

<!-- RELATED:END -->
