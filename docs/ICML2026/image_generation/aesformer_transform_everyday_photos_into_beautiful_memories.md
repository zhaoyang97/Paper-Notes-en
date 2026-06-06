---
title: >-
  [Paper Note] AesFormer: Transform Everyday Photos into Beautiful Memories
description: >-
  [ICML 2026][Image Generation][Aesthetic Photo Reconstruction] AesFormer defines aesthetic photo beautification as Aesthetic Photo Reconstruction (APR). By employing a two-stage framework that generates a photographic act…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Aesthetic Photo Reconstruction"
  - "Image Editing"
  - "Structural Reconstruction"
  - "GRPO-A"
  - "AesRecon"
date: 2026-05-08
content_hash: 03d6aaf8d848a31d
---

# AesFormer: Transform Everyday Photos into Beautiful Memories

**Conference**: ICML 2026  
**arXiv**: [2605.22126](https://arxiv.org/abs/2605.22126)  
**Code**: https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026  
**Area**: Image Generation / Image Editing  
**Keywords**: Aesthetic Photo Reconstruction, Image Editing, Structural Reconstruction, GRPO-A, AesRecon  

## TL;DR
AesFormer defines aesthetic photo beautification as Aesthetic Photo Reconstruction (APR). By employing a two-stage framework that generates a photographic action plan followed by structural editing, it transforms photographing errors—such as composition, perspective, and pose—into executable edits. It significantly outperforms open-source editors on the AesRecon dataset and approaches the performance of Nano Banana Pro.

## Background & Motivation
**Background**: Photo post-processing has long been divided into two categories: retouching, which primarily adjusts exposure, contrast, color, and overall style; and portrait enhancement, which focuses on skin, facial, and detail refinement. Recent diffusion and flow-matching image editing models can modify images based on text instructions, but they generally focus on semantic consistency and instruction following.

**Limitations of Prior Work**: The issues in many ordinary photos stem from poor structural decisions made at the moment of capture, such as off-center subject placement, distracting backgrounds, perspectives that destroy depth, stiff poses, or imbalanced compositions. Traditional retouching cannot "re-shoot" the composition. Even when general image editors receive instructions like "make it look better," they often only perform local appearance adjustments, struggling to diagnose and fix underlying photographic structural problems.

**Key Challenge**: APR requires the model to reconstruct structural attributes like composition, viewpoint, pose, and depth of field while maintaining human identity and scene semantics. It is neither simple beautification nor arbitrary new image generation; it must find a balance between "fidelity" and "aesthetic reshooting."

**Goal**: The authors propose the Aesthetic Photo Reconstruction task, construct a strictly aligned dataset of poor/good image pairs, and train a system capable of first understanding photographic aesthetics and then executing grounded structural edits.

**Key Insight**: The paper decomposes the problem into two models: AesThinker, which analyzes the input photo and outputs a sequential plan of editing actions like a photographer; and AesEditor, which translates these actions into pixel-level structural reconstructions. This prevents a single image editor from being overburdened with both aesthetic diagnosis and complex execution.

**Core Idea**: Use photography tutorial videos to mine before/after pairs to learn action plans for transitioning "from poor photos to good photos." Subsequently, use an action-conditioned editor to execute these plans, decoupling aesthetic planning from image reconstruction.

## Method
The core of AesFormer consists of three parts: data, planning, and editing. On the data side, VCMP is used to mine AesRecon from tutorial videos; on the planning side, AesThinker is trained to generate ordered actions across seven photographic dimensions; and on the editing side, AesEditor is trained to perform structural reconstruction based on these actions.

### Overall Architecture
The input is a "poor photo" taken by an ordinary user. In Stage 1, AesThinker reads the photo and prompt to output an **ordered action plan**, covering seven progressive dimensions: aspect ratio, framing/composition, camera viewpoint, subject placement, pose/action details, focus/depth-of-field, and color/light. In Stage 2, AesEditor receives the original image and the action plan to generate a reconstructed photo using a flow-matching editor. During training, action supervision comes from the poor/good pairs and text cues in AesRecon, while editing supervision comes from strictly aligned poor/good/action triplets.

### Key Designs
1. **VCMP Video Mining for AesRecon Construction**:
    - **Function**: Provides aligned poor/good training samples of the same subject and scene for APR.
    - **Mechanism**: 5,700 candidate photography tutorial videos were retrieved from platforms like Rednote, TikTok, and YouTube. After filtering out ads, showcase-only, and non-step-by-step content, 2,144 tutorials remained. Frames were sampled at 2 fps for each shooting event. Qwen2.5-VL-72B was used to identify the "clean good frame," while the initial frame of the event served as the "poor image." Low-quality good images were subsequently filtered, captions and camera UIs were removed from poor images, and a VLM verified the consistency of people, scenes, and events, resulting in 9,071 strictly aligned pairs.
    - **Design Motivation**: APR requires structural differences to stem from photographic actions rather than changes in identity, scene, or new objects. Tutorial videos naturally contain the "how to go from bad to good" process, but they require multi-stage filtering to become trainable data.

2. **AesThinker's 7-Dimensional Ordered Action Chain and GRPO-A**:
    - **Function**: Converts the vague "make the photo more beautiful" into executable, sequential photographic editing actions.
    - **Mechanism**: First, GPT-5.2 is used to distill ground-truth actions based on poor/good/text cues, and Gemini 3 verifies completeness and the 7-dimensional order. Qwen3-VL-8B is then cold-started via SFT. Subsequently, it is optimized using GRPO-A: multiple action plans are sampled for each poor image, and a total reward is formed from format rewards, semantic alignment rewards with reference actions, and creativity/aesthetic improvement rewards. Alignment and creativity are scored by Qwen2.5-VL-32B as a training-free reward model.
    - **Design Motivation**: Photographic aesthetics is not a single-answer problem; the same photo can be improved through different compositions, poses, or depth-of-field schemes. Relying solely on SFT would overfit to a single annotated trajectory; GRPO-A uses relative intra-group rewards to encourage diverse yet executable action plans.

3. **Action-conditioned AesEditor**:
    - **Function**: Reliably maps high-level photographic actions to image structural editing.
    - **Mechanism**: Based on Qwen-Image-Edit-2511, the multimodal encoder and VAE are frozen, while LoRA is applied only to the MMDiT. Given the poor image, a good reference, and the action sequence, the model learns an action-conditioned velocity field within a rectified-flow framework. The objective is to predict $v_t=x_0-x_1$, thereby generating reconstruction results during inference according to the actions output by AesThinker.
    - **Design Motivation**: General editors know how to follow instructions but do not necessarily map "improve composition/viewpoint/pose" stably to pixel changes. After fine-tuning with APR triplets, the editor learns the correspondence between photographic actions and structural reconstruction.

### Loss & Training
Stage 1(a) uses standard autoregressive SFT to maximize the conditional probability of the action sequence given the input photo and prompt. Stage 1(b) employs GRPO-A: multiple action sequences are sampled for the same input, advantages are calculated via intra-group reward normalization, and a KL penalty is applied relative to the reference policy. The reward weights are set to $\lambda_f=0.1, \lambda_a=0.5, \lambda_c=0.4$. Stage 2 utilizes the flow-matching loss $\mathcal{L}_{edit}=\mathbb{E}\|v_\psi(x_t,t,h)-v_t\|_2^2$. Experiments were conducted on 10 NVIDIA A40 48GB GPUs.

## Key Experimental Results

### Main Results
| Method | Thinker | GPT-4o Win vs Poor↑ | Human Win vs Poor↑ | GPT-4o Win vs Good↑ | Human Win vs Good↑ | ArtiMuse↑ | LAION-V2↑ | Q-ALIGN↑ |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Nano Banana Pro | None | 54.44 | 72.55 | 16.67 | 21.95 | 50.90 | 5.59 | 3.24 |
| FLUX.1 Kontext | None | 12.96 | 5.88 | 2.66 | 3.66 | 38.34 | 5.07 | 2.83 |
| Bagel | None | 12.40 | 17.65 | 7.75 | 12.20 | 37.69 | 4.94 | 2.58 |
| Step1X-Edit-v1.1 | None | 15.28 | 11.76 | 13.84 | 13.41 | 37.14 | 5.33 | 3.37 |
| Qwen-Image-Edit-2511 | None | 16.50 | 9.80 | 7.64 | 12.20 | 46.65 | 5.44 | 3.20 |
| AesFormer | AesThinker | 65.33 | 68.63 | 26.25 | 24.39 | 47.76 | 5.60 | 3.51 |

### Ablation Study
| Configuration | GPT-4o Win vs Poor↑ | GPT-4o Win vs Good↑ | ArtiMuse↑ | LAION-V2↑ | Q-ALIGN↑ | Description |
|:---|:---|:---|:---|:---|:---|:---|
| Baseline (Edit-2511) | 16.50 | 7.64 | 46.65 | 5.44 | 3.20 | Base editor only |
| S1a shuffle | 58.69 | 18.60 | 46.16 | 5.49 | 3.36 | Shuffled 7D action order; lower than ordered chain |
| S1a | 61.04 | 24.58 | 47.70 | 5.58 | 3.48 | SFT AesThinker added only |
| S1a + S2 | 61.13 | 24.14 | 47.74 | 5.58 | 3.46 | Added action-conditioned editor, no GRPO-A |
| S1a + S1b + S2 | 65.33 | 26.25 | 47.76 | 5.60 | 3.51 | Full AesFormer; GRPO-A provides further gain |

### Key Findings
- APR is extremely difficult for general open-source editors: GPT-4o win rates vs. poor for FLUX, Bagel, Step1X, and Qwen-Image-Edit are mostly between 12–17%, indicating they rarely improve structural aesthetics.
- AesFormer's GPT-4o win rate vs. poor reaches 65.33%, surpassing Nano Banana Pro's 54.44%; the human win rate vs. good is 24.39%, also slightly higher than Nano Banana Pro's 21.95%. This shows that specialized APR data and planning-editing decoupling can bridge the gap between open-source and strong closed-source systems.
- Attaching general Thinkers is unstable. In Table 1, providing Qwen3 or GPT-4o planners to FLUX, Bagel, etc., did not lead to stable improvements and sometimes caused performance drops, suggesting the issue isn't just a lack of a prompt generator, but that both the planner and editor need specific APR alignment.
- The seven-dimensional order is a significant inductive bias. After shuffling, the GPT-4o win rate vs. poor dropped from 61.04 to 58.69, confirming that a sequence starting with global composition and ending with local pose/light helps the model form a professional photography workflow.

## Highlights & Insights
- The paper decomposes "making a photo more beautiful" into structural photographic decisions rather than letting the model generate vague aesthetic descriptions. This transforms APR from a subjective slogan into a trainable, evaluable action-conditioned editing task.
- Mining data from tutorial videos is clever: tutorials naturally contain before/after transitions, action explanations, and the same shooting event, making them more reliable than trying to force-pair poor/good images from static collections.
- The reward design for GRPO-A balances format, alignment, and creativity, which aligns well with the multi-solution nature of aesthetic tasks. It rewards solutions that are executable and yield aesthetic gains rather than a single "correct" answer.
- The contrast between success and failure in AesFormer shows that strong editing capability does not equate to photographic aesthetic capability. An editor needs to know "how to change pixels," but it critically requires an upstream planner to know "why to change them this way."

## Limitations & Future Work
- AesRecon is derived from tutorial videos, so its styles and subjects may lean toward those common among tutorial creators (portraits, street photography, social media); coverage of news, documentary, commercial studio, or non-portrait scenes is less certain.
- Evaluation relies heavily on GPT-4o and aesthetic scorers. While validated by human subsets, aesthetic preferences may still be influenced by evaluator bias. More detailed user studies would be more persuasive.
- Nano Banana Pro was only evaluated on a 10% test subset due to API cost constraints, meaning the closed-source comparison is not perfectly balanced.
- Structural reconstruction may alter factual records, raising authenticity and ethical concerns, especially in documentary photography. Future work needs to explore controllable editing intensity, explanations of changes, and provenance watermarking.

## Related Work & Insights
- **vs photo retouching**: Retouching mainly adjusts tone and light, improving appearance but not original composition or perspective; AesFormer directly addresses structural reconstruction.
- **vs portrait enhancement**: Portrait enhancement focuses on skin, faces, and details (appearance-centric); APR focuses on subject positioning, pose, depth, and scene relationships.
- **vs instruction image editing**: General editing models require explicit user instructions; AesFormer diagnoses the photo's problems and generates an action plan itself, acting more as a "photography assistant."
- **vs EditThinker / iterative editing agents**: While related works emphasize editing reasoning or multi-turn tool use, this work defines an ordered action space and strictly aligned data sources specifically for photographic aesthetics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The APR task definition, tutorial video mining, and 7D photographic action chain are combined innovatively; GRPO-A is a reasonable, though not revolutionary, enhancement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes a new benchmark, closed vs. open-source comparisons, automatic and manual evaluations, and stage-wise ablations; however, closed-source models were only evaluated on a 10% subset.
- **Writing Quality**: ⭐⭐⭐⭐ The storyline is clear, naturally progressing from data bottlenecks to planning-editing decoupling, with thorough table explanations.
- **Value**: ⭐⭐⭐⭐ Highly insightful for moving image editing from "following instructions" toward "aesthetic diagnosis and proactive repair"; the data construction methodology is also reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images](../../AAAI2026/image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)
- [\[ICML 2026\] SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)
- [\[ICML 2026\] Unified Masked Diffusion Models with Diverse Generation Orders](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)
- [\[ICML 2026\] ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)

</div>

<!-- RELATED:END -->
