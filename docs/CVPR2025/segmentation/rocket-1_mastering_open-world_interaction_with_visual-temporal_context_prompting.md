---
title: >-
  [Paper Note] ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting
description: >-
  [CVPR 2025][Segmentation][Visual-Temporal Context Prompting] ROCKET-1 proposes a novel communication protocol named visual-temporal context prompting. By prompting object segments on past visual observations, this protocol guides policy models to interact with the environment. Through training a segment-conditioned low-level policy and combining it with GPT-4o, Molmo, and SAM-2 to construct a hierarchical agent, ROCKET-1 achieves a 76% absolute performance gain in open-world…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Visual-Temporal Context Prompting"
  - "Open-World Interaction"
  - "Segment-Conditioned Policy"
  - "Minecraft"
  - "Hierarchical Agent"
date: 2026-05-08
content_hash: d27bfe761989f5b4
---

# ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting

**Conference**: CVPR 2025  
**arXiv**: [2410.17856](https://arxiv.org/abs/2410.17856)  
**Code**: [https://github.com/craftjarvis/ROCKET-1](https://github.com/craftjarvis/ROCKET-1)  
**Area**: Image Segmentation  
**Keywords**: Visual-Temporal Context Prompting, Open-World Interaction, Segment-Conditioned Policy, Minecraft, Hierarchical Agent

## TL;DR

ROCKET-1 proposes a novel communication protocol named visual-temporal context prompting. By prompting object segments on past visual observations, this protocol guides policy models to interact with the environment. Through training a segment-conditioned low-level policy and combining it with GPT-4o, Molmo, and SAM-2 to construct a hierarchical agent, ROCKET-1 achieves a 76% absolute performance gain in open-world interaction within Minecraft.

## Background & Motivation

1. **Background**: Applying vision-language models (VLMs) to embodied decision-making is currently a highly popular direction. Mainstream approaches are divided into end-to-end (such as RT-2 and OpenVLA, which directly output actions) and hierarchical (VLM for high-level reasoning and a low-level policy for execution) pathways. In hierarchical methods, the "communication protocol" between the high-level reasoner and the low-level policy determines the performance upper bound of the system.
2. **Limitations of Prior Work**: (1) End-to-end methods require massive trajectory datasets with action labels, and introducing action modalities may harm the foundational capabilities of VLMs; (2) Language as a communication protocol fails to convey spatial details effectively—when multiple identical objects appear in the frame, it is hard to accurately specify one using language; (3) Future video prediction approaches (e.g., MineDreamer) require building world modes, which suffer from hallucinations, temporal inconsistency, and limited scope.
3. **Key Challenge**: Language is too imprecise (lacking spatial details), while future video is too unreliable (requiring predictions). Neither existing communication protocol can fully unleash the spatial understanding capabilities of VLMs.
4. **Goal**: To design a novel communication protocol capable of precisely communicating spatial interaction details while leveraging visual-temporal context to handle partially observable environments.
5. **Key Insight**: It is observed that humans executing tasks (such as grasping an object) do not predict future images holding the object; instead, they continually focus on the target object and retrieve its position from memory when it becomes occluded. Utilizing such "visual-temporal context" is the key.
6. **Core Idea**: Highlighting the region of interest in past observations using object segmentation masks, combined with interaction type information, to serve as the communication protocol between the high-level reasoner and the low-level policy.

## Method

### Overall Architecture

The hierarchical agent consists of four components: (1) GPT-4o executes task decomposition and reasoning, outputting descriptions of interaction steps; (2) Molmo 72B localizes the target object in the current observation based on the descriptions, outputting $(x,y)$ coordinates; (3) SAM-2 generates segmentation masks from the coordinates and tracks the object in subsequent frames; (4) ROCKET-1 receives observations, segmentations, and interaction types to predict actions. GPT-4o and Molmo operate at a low frequency, while SAM-2 and ROCKET-1 run in real-time with the environment.

### Key Designs

1. **Visual-Temporal Context Prompting**:
    - **Function**: Establishing a precise spatial communication protocol between the high-level reasoner and the low-level policy.
    - **Mechanism**: The reasoner highlights regions of interest using object segmentation masks on past visual observations, while conveying interaction intentions via a set of interaction type primitives (such as use, approach, switch, mine block, etc.). Segments of different colors represent different interaction types. The policy model receives the observation sequence $o_{1:t}$, segmentation sequence $m_{1:t}$, and interaction type sequence $c_{1:t}$ to causally predict actions $a_t$.
    - **Design Motivation**: Segmentation masks convey spatial "where" information more precisely than language (resolving ambiguity from multiple identical objects), are more reliable than predicting future frames (eliminating the need for a world model), and handle partially observable environments via temporal context (retaining position estimates even after objects are occluded).

2. **ROCKET-1 Policy Architecture**:
    - **Function**: Integrating observation and segmentation information to predict low-level actions.
    - **Mechanism**: Concatenating observations $o_t \in \mathbb{R}^{3 \times H \times W}$ and segmentation masks $m_t \in \{0,1\}^{1 \times H \times W}$ into a 4-channel image, which is fed into an EfficientNet-B0 visual backbone for feature level integration. The output is then processed via self-attention pooling to generate a feature vector $x_t$. The interaction type $c_t$ is encoded via an embedding layer and temporally modeled alongside the visual features in TransformerXL: $\hat{a}_t \leftarrow \text{TransformerXL}(c_1, x_1, \cdots, c_t, x_t)$. A crucial detail is that the integration of interaction types is delayed until after the backbone, allowing the backbone to share features across different interaction types. During training, segmentations and interaction types are randomly dropped with a probability of $p=0.75$ to force the model to infer intents from the temporal context.
    - **Design Motivation**: The 4-channel input (reminiscent of ControlNet) enables deep integration of spatial features. Delayed integration of interaction types mitigates the distribution imbalance among interaction categories. High-probability dropout of segmentations ensures that the model does not over-depend on the current frame's prompt, fostering its temporal reasoning capacity.

3. **Backward Trajectory Relabeling**:
    - **Function**: Automatically generating segmentation annotations within training data.
    - **Mechanism**: Using OpenAI's contractor dataset (1.6 billion frames of human gameplay data), the system first detects interaction frames (using metadata events such as kill entity, mine block, use item, etc.). In the frame immediately preceding the interaction, a fixed-position bounding box and points (as objects are typically centered) are used as prompts for SAM-2 to generate a segment. SAM-2 then back-propagates this temporally to automatically generate segmentation annotations for the previous $k$ frames. A navigate interaction type is additionally introduced, triggered when the player's movement exceeds a given threshold (indicating approach).
    - **Design Motivation**: Gathering trajectory data labeled with text annotations is prohibitively expensive. Leveraging SAM-2's backward tracking allows automatic extraction of target object positions in previous frames from the moment of interaction, realizing automated data annotation. The simplified assumption that target objects in Minecraft are situated at the screen center before interaction streamlines initial segmentation prompting.

### Loss & Training

Behavior cloning loss: $\mathcal{L} = -\sum_{t=1}^{|\tau|} \log \pi(a_t | o_{1:t}, m_{1:t} \odot w_{1:t}, c_{1:t} \odot w_{1:t})$

where $w_t \sim \text{Bernoulli}(1-p)$ and $p=0.75$ is the dropout probability. Every complete trajectory is divided into 128-frame segments. The model is trained using the AdamW optimizer with a learning rate of $4 \times 10^{-5}$.

## Key Experimental Results

### Main Results

| Method | Prompt | Hunt | Mine | Interact | Navigate | Tool | Place | **Average** |
|------|--------|------|------|----------|----------|------|-------|---------|
| VPT-bc | N/A | 0.15 | 0.00 | 0.16 | 0.05 | 0.00 | 0.00 | 0.07 |
| STEVE-1 | Human | 0.03 | 0.36 | 0.02 | 0.16 | 0.49 | 0.08 | 0.19 |
| GROOT-1 | Human | 0.16 | 0.03 | 0.05 | 0.02 | 0.30 | 0.02 | 0.09 |
| ROCKET-1 | Molmo | 0.88 | 0.77 | 0.66 | 0.88 | 0.93 | 0.82 | **0.82** |
| ROCKET-1 | Human | 0.93 | 0.93 | 0.93 | 0.97 | 0.97 | 0.96 | **0.95** |

Long-horizon task comparison:

| Method | Communication Protocol | Wooden Pickaxe | Furnace | Shears | Diamond | Steak | Obsidian | Pink Wool |
|------|---------|---------------|---------|--------|---------|-------|----------|-----------|
| DEPS | Language | 0.95 | 0.75 | 0.15 | 0.02 | 0.15 | 0.00 | 0.00 |
| OmniJarvis | Latent Code | 0.95 | 0.90 | 0.20 | 0.08 | 0.40 | 0.00 | 0.00 |
| **Ours** | Visual-Temporal | **1.00** | **1.00** | **0.45** | **0.25** | **0.75** | **0.50** | **0.70** |

### Ablation Study

| Configuration | Hunt (↑) | Mine (↑) | Description |
|------|---------|---------|------|
| Interaction type fused in Transformer | **0.91** | **0.78** | Delayed fusion, backbone shares knowledge |
| Interaction type fused in backbone | 0.72 | 0.69 | Early fusion, affected by class imbalance |
| w/o SAM-2, #Pmt=3 | 0.84 | 0.82 | High-frequency Molmo prompts, extremely slow inference (0.9 FPS) |
| w/o SAM-2, #Pmt=30 | 0.00 | 0.03 | Low-frequency prompts without tracking, complete failure |
| +sam2_tiny, #Pmt=30 | 0.84 | 0.69 | SAM-2 tracking compensates for low-frequency prompts |
| +sam2_large, #Pmt=30 | **0.91** | 0.78 | Largest SAM-2 model performs best |

### Key Findings

- ROCKET-1 + Molmo achieves an average success rate of 82%, which is **63 percentage points higher** than the strongest baseline, STEVE-1 (19%).
- On tasks demanding localized spatial interaction (Place), where no prior methods succeeded (0%), ROCKET-1 registers 82-96%.
- In long-horizon tasks, ROCKET-1 stands out as the only method capable of crafting Obsidian (50%) and obtaining Pink Wool (70%), where other methods yield 0%.
- The plug-and-play integration of SAM-2 is crucial: without SAM-2 under low-frequency prompts, the rate drops to zero, but with SAM-2, performance recovers to near-optimal levels.
- The fusion site of the interaction types is critical: fusing in the Transformer stage outperforms fusing within the backbone by around 10–19 percentage points.

## Highlights & Insights

- **Innovative Communication Protocol**: Breaking away from the "language vs. image" dichotomy, the authors introduce segmentation masks as a spatial communication medium. This method is precise, efficient, and naturally compatible with object trackers. This design paradigm holds potential for transfer to other embodied AI fields like robotic manipulation.
- **Clever Backward Trajectory Relabeling**: Exploiting SAM-2's backward tracking capability to back-propagate annotations from interaction moments fully bypasses human annotation costs. This data generation pipeline can scale to any video dataset requiring object-level annotations.
- **Modular System Architecture**: A four-component framework composed of GPT-4o (reasoning) + Molmo (localization) + SAM-2 (tracking) + ROCKET-1 (execution). Each block can be independently upgraded, allowing the overall system to immediately benefit from advancements in downstream VLMs and segmentation architectures.

## Limitations & Future Work

- ROCKET-1 cannot interact with objects that are out of view or have never been encountered, which triggers frequent search-and-exploration guidance from the reasoner, adding computational overhead.
- The system is dependent on Molmo's localization accuracy; if Molmo makes a localization error, the subsequent execution chain fails.
- Current verification is confined to Minecraft, leaving real-world complexities (continuous action spaces, physical dynamics, and constraints) unaddressed.
- The authors address some of these limitations in subsequent project ROCKET-2.
- Future work: Extending the paradigm to real-world physical robot manipulation, multi-object simultaneous interactions, and improving autonomous exploration strategies.

## Related Work & Insights

- **vs. STEVE-1**: A language-conditioned policy that underperforms on tasks demanding high spatial precision (0% on Place tasks). ROCKET-1 resolves spatial ambiguity seamlessly via segmentation conditioning.
- **vs. MineDreamer**: Utilizes VLM + diffusion models to synthesize target future frames to guide control, which are prone to hallucinations and temporal noise. ROCKET-1 bypasses prediction entirely by directly prompting objectives onto current observations.
- **vs. OmniJarvis**: Communicates using latent codes, which lack interpretability. ROCKET-1's segmentation masks serve as highly visual and interpretable pathways.
- **Connections to CLIPort**: Shares design concepts (using heatmaps/masks for interaction guidance), but CLIPort is strictly limited to fully observable 2D pick-and-place maneuvers.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Visual-temporal context prompting offers a fresh communication protocol paradigm, and the backward trajectory relabeling is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprises dedicated interactive benchmarks, long-horizon tasks, and extensive ablations, though tested exclusively in Minecraft.
- Writing Quality: ⭐⭐⭐⭐ Clearly articulated motivations; structural system diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ An absolute performance gain of 76% is highly compelling. The communication paradigm is poised to influence the broader embodied AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] V-CLR: View-Consistent Learning for Open-World Instance Segmentation](v-clr_view-consistent_learning_for_open-world_instance_segmentation.md)
- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](../../ICCV2025/segmentation/open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
