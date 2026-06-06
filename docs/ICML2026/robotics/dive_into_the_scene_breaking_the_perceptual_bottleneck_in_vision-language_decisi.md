---
title: >-
  [Paper Note] Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation
description: >-
  [ICML 2026][Robotics][VLM] SceneDiver filters task-relevant objects and feeds them back to the VLM for decision-making through a two-stage focus planning process—first constructing a scene graph for coarse-grained subsce…
tags:
  - "ICML 2026"
  - "Robotics"
  - "VLM"
  - "VLA"
  - "Visual focus planning"
  - "Scene graph"
  - "Object hallucination"
date: 2026-05-08
content_hash: e293f9d941e7c6be
---

# Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation

**Conference**: ICML 2026  
**arXiv**: [2606.04046](https://arxiv.org/abs/2606.04046)  
**Code**: https://future-item.github.io/SceneDiver (Available)  
**Area**: Embodied AI / Robotics / Multimodal VLM  
**Keywords**: VLM, VLA, Visual focus planning, Scene graph, Object hallucination

## TL;DR
SceneDiver filters task-relevant objects and feeds them back to the VLM for decision-making through a two-stage focus planning process—first constructing a scene graph for coarse-grained subscene decomposition, then allowing the VLM to act as an agent to verify subscenes. This explicit reasoning is distilled into the VLA using a Slot Attention adapter, mitigating visual hallucinations in both high-level planning and reactive control.

## Background & Motivation
**Background**: Embodied decision-making tasks are typically split into two pipelines: VLMs serve as high-level task planners and VLAs serve as end-to-end reactive controllers. The former excels at long-range decomposition but has poor real-time performance, while the latter is real-time but lacks deliberate reasoning.

**Limitations of Prior Work**: Both pipelines share the same "perceptual bottleneck"—in cluttered scenes, VLMs/VLAs hallucinate non-existent objects, miss detections, bind attributes incorrectly, or miscount instances of the same class. Figure 1 illustrates two typical failures: attention shifts to the background when asked "how many green objects," and attention is drawn to a neighboring yellow block when asked about the "color of the object held by the robot arm."

**Key Challenge**: Intuitively, "one-step focusing" (using existing visual focusing methods like SoM, Multi-Res, or VCD to circle key objects) should solve this, but empirical tests show it is ineffective. Reliable focusing in complex scenes inherently requires first understanding the topological relationships of the entire scene; single-step localization cannot isolate task-relevant objects from background distractors of the same color.

**Goal**: (1) Enable the VLM to autonomously generate a "focus plan" before making decisions, compressing visual input into task-relevant regions; (2) Distill this "slow thinking" capability into the VLA so the reactive policy benefits while maintaining online inference efficiency.

**Key Insight**: The authors treat "focusing" as a multi-step process that can be planned by the VLM itself, utilizing a scene graph as a structural prior to guide coarse-to-fine subscene decomposition. Decision-making is modeled as "image modulation"—preserving high frequencies and brightness in key regions while softening the background, rather than hard cropping which loses information.

**Core Idea**: Replace one-step focusing with coarse-to-fine focus planning and return the results as a "pixel-level focus map + soft modulation" to the VLM/VLA, embedding "see clearly before deciding" into the perception-action loop.

## Method
SceneDiver consists of three serial components: coarse-grained scene graph reasoning → fine-grained subscene verification → focus-modulated image generation; plus a VLA adapter that compresses this explicit process into an end-to-end module.

### Overall Architecture
The input is an RGB frame and a task instruction. First, OvSGTR is used to extract a scene graph containing object nodes `<ref>`, spatial relations `<pred>`, and bounding boxes `<box>`, which is then textualized for the VLM. The VLM performs graph reasoning to decompose the full image into several candidate subscenes. Subsequently, the VLM acts as an agent to "zoom in and inspect" each subscene, deciding to confirm, discard, or perform a local search, ultimately obtaining a verified object set $\mathcal{C}$. $\mathcal{C}$ is rasterized into a pixel-level Focus Score Map $s$, and soft modulation ("brightness attenuation + Gaussian blur") is applied to the image to produce $I_{out}$, which is fed back to the VLM for action generation. For VLA deployment, the explicit two-stage process is skipped, and a distilled adapter directly predicts the mask.

### Key Designs

1.  **Coarse-to-Fine Two-Stage Focus Planning**:
    *   **Function**: Decomposes "where is worth looking" into two steps: selecting subscenes via graph reasoning, and then verifying objects within those subscenes, avoiding blind guesses in complex scenes.
    *   **Mechanism**: In the coarse stage, the scene graph serves as a reasoning scaffold, and the VLM outputs a structured intermediate state—tagging nodes with `<ref>` and coordinates with `<box>` to partition the global scene. In the fine stage, "semantic zooming" with a restricted field of view is applied to each subscene. If a candidate object falls within the window, it is confirmed; if evidence is ambiguous, the field is narrowed further; if missing, a local search is performed in the neighborhood. The VLM is always considered the "sole source of truth," with the scene graph acting as a guide; in cases of graph-image inconsistency, the VLM can select spatially adjacent nodes, discard, or retain ambiguous candidates.
    *   **Design Motivation**: The authors found that single-step focusing often misaligns in scenes with multiple distractors. Thus, they use the scene graph for "global partitioning" followed by agent-style exploration for "local confirmation" to form an iterative cognitive cycle of "recognition → understanding → analysis."

2.  **Focus Score Map and Soft Image Modulation**:
    *   **Function**: Translates the verified candidate set $\mathcal{C}$ into a differentiable binary-to-continuous attention map, which is used to soften the original image rather than applying hard crops.
    *   **Mechanism**: First, a pixel-level score $s_{u,v}=\mathbb{I}[\exists k\in\mathcal{C}:(u,v)\in b_k]$ is generated. A visual lower bound $\beta$ is introduced to prevent the background from becoming completely black. Brightness attenuation is applied to get $I_{dim}=I\odot(\beta+(1-\beta)s)$. Then, Gaussian blur $\mathcal{B}_\sigma$ is applied to $I_{dim}$ and synthesized via score interpolation: $I_{out}=s\odot I_{dim}+(1-s)\odot\mathcal{B}_\sigma(I_{dim})$. This preserves high brightness and high-frequency details in target regions while simultaneously dimming and blurring the background.
    *   **Design Motivation**: Compared to SoM or cropping methods that remove the background entirely, soft modulation preserves environmental context necessary for robot localization and obstacle avoidance, "suppressing interference" rather than "discarding information," which is more robust to failure recovery.

3.  **SceneDiver Adapter (VLA Distillation)**:
    *   **Function**: Compresses explicit two-stage reasoning into a lightweight, end-to-end module, allowing reactive policies like OpenVLA to benefit from "focus planning" during online inference.
    *   **Mechanism**: The adapter is connected after the cross-modal projector. Slot Attention projects visual features $F\in\mathbb{R}^{L\times D}$ into $K$ object slots $S\in\mathbb{R}^{K\times D_s}$. Task tokens are pooled to produce $v_{task}$ to condition slot initialization $S_{init}\sim\mathcal{N}(\mu(v_{task})+\delta,\sigma_{global})$, preventing random initialization from assigning slots to irrelevant textures. The mask prediction module follows a coarse-to-fine hierarchy: the coarse level scores $r_k$ based on slot semantics, slot quality, and task context; the fine level uses an attention map $A\in\mathbb{R}^{K\times L}$ to back-propagate slot semantics to patches, yielding $M_{pred}=\sigma(\sum_k r_k\cdot A_{k,:}+\alpha\cdot\Delta_{patch})$. $\alpha$ is initialized near 0 to let the network rely on slot-level prediction before progressively introducing spatial corrections. Training uses Hungarian matching to align slots with scene graph objects, supervised by both a slot-level Structure Loss and a pixel-level Mask Loss.
    *   **Design Motivation**: Iterative graph traversal is too heavy for VLA. Using Slot Attention to learn structured representations corresponds to "graph nodes," and mask prediction corresponds to "two-stage reasoning results." Thus, the model learns "how to output a focus map" rather than "how to imitate trajectories."

### Loss & Training
Adapter training utilizes two sets of supervision: Structure Loss aligns slots with scene graph nodes, and Mask Loss ensures predicted masks match the GT focus maps generated by the two-stage process. During deployment, an entropy-based dynamic gating mechanism is added: when patch uncertainty exceeds a threshold, the mask is skipped and the original observation is sent to the VLA, achieving "graceful degradation" in difficult scenes to avoid policy contamination by incorrect masks.

## Key Experimental Results

### Main Results
Robot manipulation (30 MuJoCo scenes, 5 seeds, assembling target bricks on a base plate within 30 steps):

| Model | Base SR (%) | + SceneDiver Focus (%) | Gain (Abs.) |
|-------|-------------|------------------------|-------------|
| Qwen2.5-VL-7B-AWQ | 14.7 | 28.7 | +14.0 |
| Qwen2.5-VL-32B-AWQ | 21.3 | 31.3 | +10.0 |
| gpt-4o-mini | 28.7 | 34.0 | +5.3 |
| gemini-2.5-flash | 38.7 | 46.7 | +8.0 |

Room navigation (Interference levels: Base / CS Commonsense / CI Complex Instruction / VA Visual Appearance, 5 seeds):

| Method (Qwen2.5-VL-7B) | Base | CS | CI | VA |
|------------------------|------|----|----|----|
| Base Model | 32.7 | 30.7 | 32.0 | 27.3 |
| SoM | 30.0 | 31.3 | 31.3 | 29.3 |
| Multi-Res | 29.3 | 32.7 | 34.0 | 29.3 |
| VCD | 34.7 | 32.0 | 32.7 | 33.3 |
| SceneDiver | **44.0** | **36.0** | **37.3** | **35.3** |

On LIBERO-Plus (using OpenVLA-OFT), the SceneDiver adapter increased robust success rates by up to 9.6% with only 2.64% additional inference overhead.

### Ablation Study
| Configuration | Key Observation | Description |
|---------------|-----------------|-------------|
| Full SceneDiver | 14.7→28.7 (7B) | Coarse + Fine + Modulation. |
| Coarse stage only | Limited gain | No subscene verification; incorrect nodes contaminate decisions. |
| Fine stage only (No SG) | Near one-step focus | Lacks global topology; cannot systematically isolate distractors. |
| Noisy Scene Graph (Stress Test) | Still better than base | VLAs are allowed to discard/replace nodes in the graph. |
| Disable Entropy Gating | Errors in vague scenes | Incorrect masks contaminate the VLA. |

### Key Findings
- Gains primarily stem from "topological partitioning followed by local verification"—consistent with failure analysis in Sec. 4: simple SoM/Multi-Res/VCD provide near-zero or negative gains in distractor-heavy scenes (e.g., Qwen2.5-VL-7B Base 32.7 → SoM 30.0), while SceneDiver boosts Base to 44.0.
- Open-source models benefit most: Qwen2.5-VL-7B manipulation SR nearly doubled, as these models suffer most from visual hallucinations. Closed-source LLMs show smaller incremental gains (gpt-4o-mini only +5.3) but remain positive.
- Soft modulation is more important than hard cropping: retaining a brightness lower bound $\beta$ prevents the robot from losing environmental localization cues, which is why navigation is more sensitive to modulation parameters than manipulation.
- The adapter constrains overhead to 2.64% but must only be enabled when mask confidence is high; otherwise, entropy gating fallback is required to avoid degrading VLA performance.

## Highlights & Insights
- The design philosophy of "VLM as the sole source of truth and scene graph as a guide" is critical: it treats the scene graph as a "rebuttable proposal" rather than "ground truth that must be followed," preventing OvSGTR errors from propagating directly to downstream decisions.
- Soft modulation $I_{out}=s\odot I_{dim}+(1-s)\odot\mathcal{B}_\sigma(I_{dim})$ implements "attention priors" via differentiable pixel operations rather than cropping, making it seamlessly swappable for any VLM input with low migration cost.
- The Slot Attention + task-conditioned initialization approach stabilizes the mapping between "object slots ↔ scene graph nodes," which could be adapted for any downstream task requiring visual token compression into interpretable object representations.
- Entropy gating gives the distilled VLA the ability to "know what it doesn't know," preventing incorrect masks from leading end-to-end policies astray—an insight worth migrating to all VLA frameworks relying on auxiliary predictors.

## Limitations & Future Work
- Strong dependence on the detection quality of the external scene graph model OvSGTR. Although robust to noise, it may collapse entirely for object categories never seen during training (extreme open-vocabulary cases).
- The inference cost of two-stage focus planning on the VLM side is non-trivial; this is why the authors had to distill the adapter. For closed-source API VLMs, the cost of multi-round prompting per step would be significant.
- The adapter has currently only been validated on OpenVLA-OFT; its effectiveness on different paradigms (e.g., diffusion policy, π0) is unknown.
- Soft modulation is sensitive to parameters $\beta$ and $\sigma$. These are empirically set global values in the paper; adaptively adjusting them based on the scene is a clear direction for extension.

## Related Work & Insights
- **vs SoM / Multi-Res / VCD**: All three follow the "one-step focusing" route, either through labeling, multi-resolution cropping, or contrastive decoding. This paper proves they offer near-zero gain in embodied decision-making with distractors due to a lack of global topological understanding.
- **vs Brohan et al. (RT/SayCan) High-level Planning**: These use VLMs as action sequence planners. SceneDiver does not replace planning but inserts "seeing clearly" before it, acting as a complement.
- **vs Nguyen 2025 / Terra et al. 3D Scene Graph Robotics**: These use 3D scene graphs as environment representations for long-term memory and reachability reasoning. SceneDiver uses 2D scene graphs for single-frame focus planning, targeting the perception bottleneck rather than environment modeling.
- **Insights**: The pipeline of "explicit multi-step reasoning → distillation into a lightweight end-to-end module" can be applied to many "VLM slow thinking / VLA fast reaction" scenarios, such as VLN, autonomous driving perception enhancement, and multi-target grasping for domestic robots.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of using a scene graph as a rebuttable prior, soft modulation instead of hard cropping, and Slot Attention distillation into VLA is uncommon in embodied decision-making.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers manipulation, navigation, and LIBERO-Plus robustness; validates both open and closed-source models with noisy scene graph stress tests.
- Writing Quality: ⭐⭐⭐⭐ Clear storyline and consistent notation; could benefit from more failure case visualizations.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play perception preprocessing tool for the "VLM-as-planner, VLA-as-executor" paradigm, particularly beneficial for small open-source models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](psg-nav_probabilistic_scene_graph_navigation_via_multiverse_decision_making.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](../../ICLR2026/robotics/memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)
- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)

</div>

<!-- RELATED:END -->
