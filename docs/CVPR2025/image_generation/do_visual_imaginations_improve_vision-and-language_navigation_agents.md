---
title: >-
  [Paper Note] Do Visual Imaginations Improve Vision-and-Language Navigation Agents?
description: >-
  [CVPR 2025][Image Generation][Vision-and-Language Navigation] This paper uses SDXL to generate synthetic images as "imaginations" for visual landmarks in VLN instructions. These are encoded via ViT and concatenated with text instruction embeddings before being input into the VLN agent. Guided by a cosine similarity alignment loss, this approach consistently improves navigation success rates by approximately 1% on both R2R and REVERIE, verifying the preliminary value of visual…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Vision-and-Language Navigation"
  - "Visual Imagination"
  - "Diffusion Models"
  - "Landmark Generation"
  - "Multimodal Assistance"
date: 2026-05-08
content_hash: 9c43777fdaeec6d7
---

# Do Visual Imaginations Improve Vision-and-Language Navigation Agents?

**Conference**: CVPR 2025  
**arXiv**: [2503.16394](https://arxiv.org/abs/2503.16394)  
**Code**: To be confirmed  
**Area**: Image Generation / Agent / VLN  
**Keywords**: Vision-and-Language Navigation, Visual Imagination, Diffusion Models, Landmark Generation, Multimodal Assistance

## TL;DR
This paper uses SDXL to generate synthetic images as "imaginations" for visual landmarks in VLN instructions. These are encoded via ViT and concatenated with text instruction embeddings before being input into the VLN agent. Guided by a cosine similarity alignment loss, this approach consistently improves navigation success rates by approximately 1% on both R2R and REVERIE, verifying the preliminary value of visual imagination as a bridge between language and vision.

## Background & Motivation

**Background**: Vision-and-Language Navigation (VLN) requires agents to navigate 3D environments based on natural language instructions. Current methods (e.g., HAMT, DUET) directly feed textual instructions and observed images into cross-modal attention, but there is a significant modal gap between the visually descriptive landmarks in language (e.g., "walk next to the burgundy sofa") and the actual visual observations.

**Limitations of Prior Work**: (1) Semantic ambiguity—words like "counter" can refer to a kitchen countertop or a service counter, which cannot be disambiguated by text alone; (2) Abstraction of textual descriptions—agents need to imagine the visual scenes corresponding to text descriptions to effectively match current observations; (3) When navigating, humans naturally form "mental images" to facilitate instruction understanding, whereas existing agents lack this ability.

**Key Challenge**: VLN agents need to establish precise correspondences between language and vision, but the representation spaces of the two are vastly different. Can a third modality—synthetic visual imagination—be introduced to bridge this gap?

**Goal**: To explore whether text-to-image generation can create visual auxiliary information for VLN instructions, and whether these "imaginations" truly assist in navigation.

**Key Insight**: Mimic the mental visualization process of humans reading navigation instructions—generate a synthetic image for each visual landmark mentioned in the instruction, allowing the agent to better understand and execute the instructions after "seeing" these imaginations.

**Core Idea**: Utilize SDXL to generate synthetic images for visual landmarks within navigation sub-instructions, inputting them alongside textual instructions into the VLN agent as multimodal auxiliary signals.

## Method

### Overall Architecture
The pipeline consists of two parts: (1) Imagination Generation—segment navigation instructions into sub-instructions, filter for those containing visual landmarks, and use SDXL to generate synthetic images; (2) Imagination Integration—encode imagination images using ViT-B/16, concatenate them into the text embedding sequence, and interact with visual observations through the VLN agent's cross-modal encoder. A cosine alignment loss is incorporated during training.

### Key Designs

1. **Visual Imagination Generation Pipeline**:

    - **Function**: Automatically generate corresponding visual landmark images from text navigation instructions.
    - **Mechanism**: Divide instructions into an average of 3.66 sub-instructions using FG-R2R; use Spacy NLP to filter out sub-instructions without noun phrases and apply a blacklist to filter non-visual words (numbers, direction words, pronouns), retaining approximately 2.96 visual landmarks per instruction. Generate 1024×1024 images using SDXL, with positive prompts containing "indoor", "real estate", and negative prompts excluding "humans", "collage". The generation speed is 3.2s per image on an H100 GPU.
    - **Design Motivation**: Not all sub-instructions contain visualizable information (e.g., "turn left" does not need one). The filtering step ensures that imaginations are only generated for meaningful landmarks. LangSAM is used to verify coverage: at least one noun phrase is detected in 98.78% of the sub-instructions.

2. **Imagination Encoding and Integration**:

    - **Function**: Integrate imagination images into the decision-making pipeline of the VLN agent.
    - **Mechanism**: Each imagination $Z_i$ is encoded via a pre-trained ViT-B/16, added to an MLP and a type embedding $t_{Im}$ to obtain a $d$-dimensional representation $h_i$. These imagination embeddings are concatenated after the textual instruction embeddings to undergo cross-modal encoding and interact with visual observations: $f_X([f_T(W), \mathcal{H}], f_O(O_{\leq t}))$. This is applied to both HAMT and DUET architectures.
    - **Design Motivation**: Concatenating imaginations as an independent modality (rather than replacing text) allows the agent to simultaneously possess linguistic and visual cues. ViT is kept off-the-shelf (no fine-tuning) to prevent overfitting on small datasets.

3. **Cosine Alignment Auxiliary Loss**:

    - **Function**: Explicitly align imagination embeddings with the text embeddings of the corresponding noun phrases in the sub-instructions.
    - **Mechanism**: $\mathcal{L}_{cos} = \frac{1}{N_{Im}} \sum (1 - \frac{h_i \cdot \bar{S}_i}{||h_i|| ||\bar{S}_i||})$, where $\bar{S}_i$ is the average embedding of noun phrase tokens in sub-instruction $i$. The total loss is $\mathcal{L} = \mathcal{L}_{base} + 0.5 \cdot \mathcal{L}_{cos}$.
    - **Design Motivation**: Simply concatenating embeddings cannot guarantee semantic alignment between imaginations and corresponding texts. Ablation studies show that the alignment loss contributes +0.5 SR / +0.4 SPL.

### Loss & Training
Three-stage fine-tuning: Stage 1 trains only the newly added components (MLP + type embedding) while freezing the base model. Stage 2 involves joint training but keeps a low learning rate for the base model. Stage 3 conducts full-parameter training with a unified learning rate. Training runs for 100k iterations with a batch size of 8 on a Tesla V100, taking about 1.5 days.

## Key Experimental Results

### Main Results

| Method | R2R Val-Unseen SR | R2R Val-Unseen SPL | R2R Test SR | REVERIE Val-Unseen SR |
|--------|------|------|------|------|
| HAMT baseline | 66.24 | 61.51 | 65 | - |
| **HAMT-Imagine** | **67.26** (+1.02) | **62.02** (+0.51) | 65 | - |
| DUET baseline | 71.52 | 60.41 | 69 | 46.98 |
| **DUET-Imagine** | **72.12** (+0.60) | **60.48** (+0.07) | **71** (+2) | **48.28** (+1.30) |

### Ablation Study

| Configuration | SR | SPL |
|------|---------|---------|
| HAMT baseline | 66.24 | 61.51 |
| Null imaginations (not used during testing) | 66.92 | 61.89 |
| Wrong imaginations (randomly shuffled) | 66.24 | 61.02 |
| **Correct imaginations** | **67.26** | **62.02** |
| Goal-only imagination (final landmark only) | 66.79 | 61.58 |
| **Sequential imaginations** | **67.26** | **62.02** |

### Key Findings
- **The correctness of imaginations is critical**: Wrong imaginations (randomly shuffled) instead degrade SPL (61.02 < 61.51), whereas correct imaginations boost SR by +1.02, demonstrating that the model indeed leverages the semantic information of imaginations rather than merely benefiting from regularization.
- **Sequential imaginations outperform goal-oriented ones**: Imagining the complete sequence of sub-instructions > imagining only the final landmark (+0.47 SR), as landmark information along the intermediate path also holds navigational value.
- **Imaginations provide regularization during training**: Even when imaginations are not utilized during testing (Null), performance remains superior to the baseline (+0.68 SR), suggesting that training with imaginations helps the model learn superior multimodal alignments.
- **High-quality imaginations**: LangSAM detects at least one noun phrase in 98.78% of the sub-instructions in synthesized images, indicating that SDXL faithfully generates the objects described in instructions.
- **Modest but consistent improvements**: An improvement of approximately 1% in SR might seem small, but it is noteworthy on highly saturated benchmarks like VLN.

## Highlights & Insights
- **A new paradigm of "Cognitive Assistance"**: It does not alter the agent architecture or introduce heavy new modules (only a single MLP). Instead, it merely provides the agent with an additional input—imaginations. This intuitive idea of "showing images to the model to help it understand text" can be extended to any task requiring language-vision alignment.
- **Diffusion models as knowledge externalization tools**: Instead of being used to generate training data or conduct data augmentation, SDXL serves as a "semantic translator" translating text into visual representations. This usage is highly novel.
- **Regularization effect of Null imagination** implies that training with imaginations might help the agent acquire more robust multimodal representations, benefiting the model even when imaginations are absent.

## Limitations & Future Work
- The performance gain is modest (~1 SR), indicating that current generated imaginations might not yet be informative or precise enough.
- Imaginations lack environmental grounding—the synthetic images do not reflect the actual appearance/position of objects in the real-world environment, failing to handle environment-specific naming or unique object references.
- Generating imaginations inflates inference computation (3.2s per image). Although pre-computable offline, it still poses overhead in online scenarios.
- HAMT exhibits no improvement on the test set (65 -> 65), raising concerns about generalizability.
- The paper does not explore the possibility of allowing the agent to dynamically generate or update imaginations based on online observations.

## Related Work & Insights
- **vs HAMT/DUET (baselines)**: Directly acts as a plug-in on these two SOTA models, verifying the generalizability of the method.
- **vs World Model approaches**: World models predict future frames which require environmental knowledge; visual imagination relies purely on language descriptions and is more light-weight.
- **vs Data Augmentation**: Rather than augmenting training data with generated images, imaginations serve as an extra input modality, which is more elegant.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of "showing the agent imagined images" is simple but novel, and represents the first systematic validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model evaluation (HAMT+DUET), multi-dataset (R2R+REVERIE), and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and deep ablation analysis.
- Value: ⭐⭐⭐ Although the performance gains are limited, it highlights an intriguing research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](../../ICCV2025/image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] Yo'Chameleon: Personalized Vision and Language Generation](yochameleon_personalized_vision_and_language_generation.md)
- [\[CVPR 2025\] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
