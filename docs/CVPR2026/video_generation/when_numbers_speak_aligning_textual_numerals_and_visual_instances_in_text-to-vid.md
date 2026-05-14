---
title: >-
  [Paper Note] When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models
description: >-
  [CVPR 2026][Video Generation][Numeral Alignment] NUMINA proposes an identify-then-guide paradigm that, without retraining the video diffusion model…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Numeral Alignment"
  - "Text-to-Video"
  - "Training-Free"
  - "Attention Head Selection"
  - "Layout-Guided Generation"
date: 2026-05-08
content_hash: 9da1d22d8fcac37d
---

# When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2604.08546](https://arxiv.org/abs/2604.08546)
**Code**: [https://github.com/H-EmbodVis/NUMINA](https://github.com/H-EmbodVis/NUMINA)
**Area**: Video Generation
**Keywords**: Numeral Alignment, Text-to-Video, Training-Free, Attention Head Selection, Layout-Guided Generation

## TL;DR

NUMINA proposes an identify-then-guide paradigm that, without retraining the video diffusion model, extracts a countable instance layout from DiT attention maps during inference, detects inconsistencies between numeric tokens and the current layout, applies conservative layout modifications, and uses the revised layout to guide regeneration—substantially improving adherence to quantity constraints such as "two apples" or "eight ducks" in text-to-video models.

## Background & Motivation

Current text-to-video models have achieved strong performance in visual quality, temporal coherence, and motion generation, yet quantity control remains a persistent weakness. Models typically handle attributes such as color, action, and scene well, but fail to reliably generate the number of objects specified in a prompt. While minor count errors may be tolerable in entertainment contexts, they directly undermine usability in educational videos, simulation demonstrations, and content requiring precise step-wise quantities.

The authors identify two concrete root causes for the difficulty of numeral grounding. The first is **numeral semantic weakness**: numeral tokens such as "three" exhibit more diffuse and less focused activations in cross-attention compared to nouns, adjectives, and verbs, indicating that the model does not truly ground these tokens to spatial layouts. The second is **instance ambiguity**: DiT operates on heavily downsampled spatiotemporal latents, causing multiple instances to merge easily in latent space and preventing the model from distinguishing between two separate objects and one large region.

Retraining could potentially address these issues, but at prohibitive cost and requiring video datasets with precise quantity annotations. The authors therefore pursue a more practical approach: rather than restructuring the model fundamentally, they exploit the latent instance structure already present in the model's attention maps and apply lightweight inference-time intervention.

This choice requires the method to satisfy two conditions simultaneously:

- It must be expressive enough to convert implicit attention into explicit layouts and correct count errors accordingly.
- It must be conservative enough to avoid disrupting the overall layout, style, or temporal consistency when adding or removing objects.

NUMINA's identify-then-guide paradigm is designed around both objectives.

## Method

### Overall Architecture

NUMINA operates in two stages.

- **Identify**: A pre-generation pass is performed; self-attention and cross-attention maps are extracted at early denoising steps to construct a countable layout of the current video.
- **Guide**: If the extracted layout is inconsistent with the numeric constraint in the prompt, minimal modifications are applied at the layout level, and the revised layout is used to modulate cross-attention during regeneration.

A key aspect of this pipeline is that intervention is applied neither to the final output image directly nor through external detector-driven per-frame editing, but rather at the mid-to-early latent stage where instances are already discernible yet generation remains malleable.

### Key Designs

1. **Attention Head Selection for Countable Layout Extraction**

    - **Function**: Identify, from the full set of attention heads, the self-attention heads that best separate instances and the cross-attention heads that best localize target nouns.
    - **Mechanism**: For self-attention, each head's attention map is projected via PCA, and three component scores are computed: foreground-background separability, structural richness, and edge clarity, combined into a composite score $S(SA^h)$; the Top-1 head is selected as the instance skeleton. For each target noun token, the cross-attention head with the highest peak activation is selected, as higher peaks generally correspond to more spatially concentrated regions.
    - **Design Motivation**: Averaging across all heads dilutes sparse but critical instance-separation signals. The authors' analysis shows that separable instance information is concentrated in very few heads, necessitating explicit selection.

2. **Countable Layout Construction**

    - **Function**: Convert diffuse attention distributions into a set of discrete, countable instance regions.
    - **Mechanism**: Spatial proposals are formed by clustering the selected self-attention map; threshold filtering and density clustering of the cross-attention map yield a focus mask. Only proposals with sufficient overlap with the focus mask are retained as instances of the target class. The result is a discrete layout map in which the number of connected foreground regions constitutes the implicit instance count.
    - **Design Motivation**: Self-attention handles instance separation while cross-attention handles semantic grounding to prompt tokens. Only by combining both can the layout be simultaneously separable and semantically aligned.

3. **Conservative Layout Correction and Layout-Guided Generation**

    - **Function**: Add missing objects or remove excess objects without disrupting the existing composition or temporal coherence.
    - **Mechanism**: For deletion, the smallest region is removed first to minimize visual disturbance. For insertion, the smallest existing instance is copied as a template; if no instance exists, a circular template is used as fallback. Template placement is governed by a cost function comprising three terms: overlap penalty with existing regions $C_o$, distance from existing instance centers $C_c$, and a temporal smoothness term relative to insertion positions in the previous frame $C_t$. The corrected layout is then used during regeneration to locally boost or suppress pre-softmax cross-attention scores, encouraging target objects to emerge in added regions and suppressing them in deleted regions.
    - **Design Motivation**: Many control methods apply overly aggressive modifications. NUMINA restricts changes to instance-level local regions, correcting counts while preserving the original video's style and motion.

### Loss & Training

NUMINA is a training-free method with no additional training objectives. Attention maps are extracted at reference timestep $t^*=20$ and intermediate layer $l^*=15$ during inference, followed by localized guidance over 50 sampling steps.

This design allows the method to be directly applied to existing Wan-series video generation models without additional annotation data, distillation networks, or layout predictors.

## Key Experimental Results

### Main Results

The authors construct CountBench, comprising 210 prompts spanning 1 to 8 instances and 1 to 3 object categories, specifically designed to evaluate counting accuracy. Comparisons are made against three practical training-free baselines—the original model, seed search, and prompt enhancement—alongside NUMINA.

| Model | Setting | CountAcc (%) | TC (%) | CLIP Score |
|---|---|---:|---:|---:|
| Wan2.1-1.3B | baseline | 42.3 | 81.2 | 33.9 |
| Wan2.1-1.3B | + seed search | 45.5 | 82.3 | 34.6 |
| Wan2.1-1.3B | + prompt enhancement | 47.2 | 82.1 | 33.7 |
| Wan2.1-1.3B | **+ NUMINA** | **49.7** | **83.4** | **35.6** |
| Wan2.2-5B | baseline | 47.8 | 85.0 | 34.3 |
| Wan2.2-5B | **+ NUMINA** | **52.7** | **85.0** | **34.7** |
| Wan2.1-14B | baseline | 53.6 | 83.3 | 34.2 |
| Wan2.1-14B | **+ NUMINA** | **59.1** | **84.0** | **34.4** |

The most notable finding is that NUMINA raises the 1.3B model's CountAcc to 49.7%, surpassing the baseline 5B model's 47.8%, indicating that the method addresses a fundamental limitation in quantity control rather than marginal errors.

### Ablation Study

The authors conduct thorough ablation studies on layout source, insertion cost components, and attention head selection strategy.

| Ablation | Configuration | CountAcc (%) | TC (%) |
|---|---|---:|---:|
| Layout Source | baseline | 42.3 | 81.2 |
| Layout Source | GroundingDINO layout | 47.5 | 82.8 |
| Layout Source | **Attention layout (ours)** | **49.7** | **83.4** |
| Insertion Cost | $C_o$ only | 45.1 | 82.1 |
| Insertion Cost | $C_o + C_c$ | 46.9 | 82.3 |
| Insertion Cost | $C_o + C_t$ | 48.9 | 83.1 |
| Insertion Cost | **$C_o + C_c + C_t$** | **49.7** | **83.4** |
| Head Selection | random single head | 44.1 | 82.6 |
| Head Selection | all-average | 43.0 | 82.4 |
| Head Selection | Top-3 | 48.2 | 82.5 |
| Head Selection | Top-2 | 49.4 | 83.3 |
| Head Selection | **Top-1** | **49.7** | **83.4** |

### Key Findings

- Attention-derived layouts outperform GroundingDINO detections, suggesting that for partially-formed instances in generation, the model's internal attention structure more faithfully reflects the latent instance configuration than external detectors.
- The temporal cost term $C_t$ contributes more than the center distance term $C_c$, confirming that cross-frame positional stability is as critical as count accuracy in video generation.
- Top-1 head selection marginally outperforms Top-2 and Top-3, supporting the observation that instance-separable information is sparse and that indiscriminate averaging introduces noise.
- Reference timestep $t^*=20$ provides a favorable accuracy-efficiency trade-off; extracting attention at later steps yields stronger local visibility but leads to fragmented or over-merged attention maps that degrade counting.
- NUMINA's advantage is more pronounced for high-count prompts; the paper specifically reports that for 8-object prompts, the baseline achieves only 11.3% accuracy, which NUMINA raises to 20.7%.

## Highlights & Insights

- The central contribution lies in identifying the appropriate level of intervention. Rather than editing videos in pixel space with hard masks or rewriting prompts, NUMINA operates at the layout level—an abstraction that is sufficiently powerful yet sufficiently stable.
- The explicit separation of responsibilities between self-attention for instance segmentation and cross-attention for semantic grounding is methodologically clean. Many attention-based control methods conflate the two; NUMINA treats instance separation and noun localization as distinct subproblems.
- Despite being entirely training-free, the method is not a heuristic assembly but a closed-loop pipeline: head selection determines layout quality, layout quality determines the reliability of correction, and the corrected layout feeds back into generation control.
- CountBench itself is a valuable contribution. Existing video generation benchmarks predominantly assess visual quality or temporal coherence; explicitly measuring fidelity to numeral constraints is a necessary and complementary evaluation dimension.

## Limitations & Future Work

- The current method is validated for 1 to 8 object instances; higher-density scenarios remain unexplored. For tens or hundreds of instances, both region separation and the insertion cost function may require redesign.
- NUMINA assumes unambiguous pairing between numeric tokens and target nouns in the prompt. For complex multi-clause prompts with multiple numeric constraints, noun-number binding may become a bottleneck.
- The method requires a pre-generation pass, incurring additional inference cost. While compatibility with EasyCache is noted, overhead compared to standard generation remains non-trivial.
- The layout correction procedure, particularly template copying and grid-search placement for instance insertion, is heuristic in nature. Future work could replace this step with a continuous, differentiable layout editing formulation.
- A natural extension is to unify NUMINA into a broader numeracy control framework applicable to both images and videos, and further to handle compound structural constraints involving quantity, spatial relations, and action assignment.

## Related Work & Insights

- **vs. Seed Search**: Seed search amounts to repeated sampling, which is costly and uncontrolled; NUMINA explicitly identifies what is wrong and how to correct it, resulting in more consistent improvements.
- **vs. Prompt Enhancement**: Augmenting prompts can weakly remind the model of quantity constraints but cannot resolve instance ambiguity in latent space; NUMINA directly addresses this gap.
- **vs. CountGen and image-based counting control methods**: CountGen targets text-to-image generation and requires training an auxiliary layout completion network; NUMINA is entirely training-free and explicitly accounts for temporal stability in video.
- A broader methodological insight is that many generation errors need not be corrected through retraining. When intermediate structures that are already implicitly present but not yet explicitly exploited can be identified within the model, lightweight inference-time interventions can yield substantial gains.

## Rating

- Novelty: ⭐⭐⭐⭐ Training-free numeral alignment is not a wholly new problem, but the integration of attention-based layout extraction, conservative layout correction, and video-guided generation is thorough and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-scale Wan model evaluation, mainstream training-free baselines, detailed ablations, and a new benchmark are all executed rigorously.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, with strong correspondence between experimental findings and design motivation.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for practical video generation, particularly in settings where retraining large models is infeasible but reliable quantity control is required.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[CVPR 2026\] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
