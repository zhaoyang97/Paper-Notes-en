---
title: >-
  [Paper Note] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory
description: >-
  [AAAI 2026][Robotics][Zero-shot object navigation] This paper proposes PanoNav, a mapless zero-shot object navigation framework that uses only RGB images. It unlocks the spatial reasoning capability of MLLMs through Panoramic Scene Parsing and introduces a Dynamic Bounded Memory Queue to prevent local deadlock.
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Zero-shot object navigation"
  - "panoramic scene parsing"
  - "dynamic memory"
  - "mapless navigation"
  - "MLLM"
date: 2026-05-08
content_hash: 64c904982fd1428e
---

# PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory

**Conference**: AAAI 2026
**arXiv**: [2511.06840](https://arxiv.org/abs/2511.06840)  
**Code**: None  
**Area**: Robotics
**Keywords**: Zero-shot object navigation, panoramic scene parsing, dynamic memory, mapless navigation, MLLM

## TL;DR

This paper proposes PanoNav, a mapless zero-shot object navigation framework that uses only RGB images. It unlocks the spatial reasoning capability of MLLMs through Panoramic Scene Parsing and introduces a Dynamic Bounded Memory Queue to prevent local deadlock.

## Background & Motivation

Object navigation (ObjectNav) is a fundamental capability for household robots, requiring the agent to locate and navigate to a specified object in an unknown environment. Existing methods suffer from three main limitations:

**1. Reliance on depth sensors and pre-built maps**: Many methods (e.g., VLFM, ESC, VoroNav, L3MVN) require RGB-D input to construct 2.5D scene representations or metric maps, increasing hardware requirements and reducing robustness in noisy or dynamic environments.

**2. Closed-set category recognition**: Many methods can only recognize predefined object categories and fail to generalize to open-vocabulary real-world scenes.

**3. Local deadlock in mapless methods**: Existing mapless methods (e.g., ZSON, PixNav) make decisions based solely on current observations, ignoring historical trajectory information, and thus tend to repeatedly revisit already-explored regions. This occurs because LLMs over-rely on object–room priors (e.g., sofas are likely in living rooms) without considering "I have already searched this area."

**Core Motivation**: Can one design an open-vocabulary navigation system that uses only RGB images without maps or depth information, while also resolving the local deadlock problem?

## Method

### Overall Architecture

At each timestep, PanoNav captures 6 RGB images in different directions (at 60° intervals) to form a panoramic view, which is then processed by two core modules:

1. **Panoramic Scene Parsing**: Uses an MLLM (Qwen-2.5-VL) to extract local directional descriptions and a global scene summary from 6-view RGB images and dot matrices.
2. **Memory-Guided Decision-Making**: Combines a Dynamic Bounded Memory Queue with an LLM (DeepSeek-V3) to make navigation decisions.

### Key Designs

#### 1. **Local Directional Parsing**

Rich contextual information is extracted from each directional view, including object presence, spatial relationships, room type, and target appearance probability. A key innovation is the introduction of a **dot matrix** as auxiliary input:

$$\mathbf{M}_t^i = \text{SCA}(\mathbf{V}_t^i), \quad i \in \{1, \ldots, 6\}$$

where SCA denotes the Scaffold processing method. Raw RGB images capture geometric distance cues, while dot matrices enhance planar positional understanding. The two are complementary, enabling the MLLM to reason across two spatial dimensions. A spatial relationship graph is constructed as:

$$G_t^i = \mathcal{P}(\Psi(\mathbf{V}_t^i), \Phi(\mathbf{V}_t^i, \mathbf{M}_t^i))$$

where $\Psi(\cdot)$ extracts geometric distance relations, $\Phi(\cdot)$ parses planar positional relations, and $\mathcal{P}(\cdot)$ aggregates them into a unified graph.

#### 2. **Global Panoramic Summary**

Beyond local parsing, a global analysis is performed to obtain higher-level semantic understanding — identifying objects in the surrounding environment and determining the current room or scene type (e.g., kitchen, corridor). This provides an implicit sense of self-localization.

#### 3. **Dynamic Bounded Memory Queue**

This is the paper's core design for resolving local deadlock. A queue $\mathcal{Q}$ of maximum length $n$ is maintained to store recent global summary descriptions:

$$\mathcal{Q}_t = \{\mathbf{gs}_{t-n}, \ldots, \mathbf{gs}_{t-2}, \mathbf{gs}_{t-1}\}$$

- At the initial phase, the queue is empty and the agent makes decisions based solely on current observations.
- Once the queue is full ($f_t = 1$), new waypoint descriptions are enqueued and the oldest are dequeued.
- The decision function switches according to queue state:

$$\begin{cases} \mathbf{r}_t = \mathcal{F}(\mathbf{ld}_t, \mathbf{gs}_t), & f_t = 0 \\ \mathbf{r}_t = \mathcal{F}(\mathbf{ld}_t, \mathbf{gs}_t, \mathcal{Q}_t), & f_t = 1 \end{cases}$$

This enables the LLM to reason that "I have already searched the living room — I should check the corridor instead," thereby avoiding local deadlock.

### Loss & Training

PanoNav is a **zero-shot framework** that requires no training. All modules are based on off-the-shelf pretrained models:
- Scene parsing: Qwen-2.5-VL (frozen parameters)
- Navigation decision: DeepSeek-V3 (frozen parameters)
- Motion control: PixNav is used as the motion controller

## Key Experimental Results

### Main Results

Evaluated on 200 randomly sampled episodes from the HM3D dataset, compared against multiple navigation methods:

| Method | Input Modality | Open-Vocab | Map | SR↑ | SPL↑ |
|---|---|---|---|---|---|
| FBE | RGB-D, GPS | Closed-set | Map | 33.7 | 15.3 |
| SemExp | RGB-D, GPS | Closed-set | Map | 37.9 | 18.8 |
| Habitat-Web | RGB-D, GPS | Closed-set | Mapless | 41.5 | 16.0 |
| OVRL | RGB-D, GPS | Closed-set | Mapless | 62.0 | 26.8 |
| VLFM | RGB-D, GPS | Open-set | Map | 52.2 | 30.4 |
| L3MVN | RGB-D, GPS | Open-set | Map | 50.4 | 23.1 |
| ZSON | RGB only | Open-set | Mapless | 25.5 | 12.6 |
| PixNav | RGB only | Open-set | Mapless | 37.9 | 20.5 |
| **PanoNav (Ours)** | **RGB only** | **Open-set** | **Mapless** | **43.5** | **23.7** |

**Key finding**: Under the same setting (RGB-only, mapless, open-vocabulary), PanoNav improves SR over PixNav by 14.76% and SPL by 15.61%, even surpassing several methods that use depth sensors and maps.

### Ablation Study

| Configuration | SR↑ | SPL↑ | Note |
|---|---|---|---|
| 3-view (no panorama) | 19.5 | 9.97 | Severely limited by restricted field of view |
| Panorama + one-step decision (no decoupling) | 35.0 | 20.47 | Insufficient end-to-end reasoning capacity |
| Panorama + decoupled (no memory) | 38.5 | 22.57 | Prone to deadlock without historical information |
| **Panorama + decoupled + memory (full)** | **43.5** | **23.73** | Full framework achieves best performance |

### Deadlock Avoidance Test

Specifically designed episodes with high deceptiveness (e.g., "search for a sofa in the living room, but the sofa is not there"), with each episode repeated 10 times:

| Configuration | SR↑ | SPL↑ | DTS(fail)↓ | Escape Rate↑ |
|---|---|---|---|---|
| Without memory | 12.0 | 4.9 | 6.7 | 32.0% |
| **With memory** | **48.0** | **19.2** | **4.7** | **82.0%** |

The memory-guided approach achieves a 4× improvement in success rate, with the escape rate increasing from 32% to 82%.

### Key Findings

1. **6-view panorama improves SR by 24% over 3-view (forward-only)**—omnidirectional perception is critical.
2. **Decoupled perception and decision-making outperforms end-to-end**—step-wise processing reduces the cognitive burden on the MLLM.
3. **Dynamic memory is essential for deadlock avoidance**—even in failure cases, agents with memory navigate closer to the target.

## Highlights & Insights

1. **RGB-only, mapless, zero-shot**—the technical stack is minimal, requiring no depth sensors, GPS, pre-built maps, or any training, relying entirely on pretrained MLLMs and LLMs.
2. **Clever design of the dot matrix**: The Scaffold method converts RGB images into dot matrices, enhancing the MLLM's understanding of planar spatial relationships and compensating for its limitations in precise spatial reasoning.
3. **Simplicity and effectiveness of the dynamic memory queue**: Only textual summaries are stored rather than complex spatial representations, incurring negligible computational overhead.
4. **The experimental design of the deadlock avoidance test is highly convincing.**

## Limitations & Future Work

1. **Evaluation is limited to HM3D**, with a relatively small scale of 200 episodes.
2. **The memory queue stores only textual summaries**, which may discard fine-grained spatial information; future work could consider incorporating lightweight topological maps.
3. **The motion controller relies on PixNav**, limiting the precision of low-level navigation control.
4. **No real-robot experiments**—validation is conducted in simulation only.
5. **Fixed 6-view, 60° interval configuration**—may introduce redundancy or insufficient coverage in narrow spaces.

## Related Work & Insights

- **Comparison with map-based methods such as VLFM and VoroNav**: PanoNav demonstrates that competitive results can be achieved with a simpler architecture.
- **MLLMs as scene understanding engines**: MLLMs can potentially replace traditional perception modules in a broader range of robotic tasks.
- **The importance of memory mechanisms in navigation**: The spectrum from simple text queues to more complex spatial memory representations is a direction worth exploring.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of panoramic parsing and dynamic memory is novel, though individual modules are relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The ablation study and deadlock test are cleverly designed, but the dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for mapless navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation](../../CVPR2026/robotics/trajrag_retrieving_geometric-semantic_experience_for_zero-shot_object_navigation.md)
- [\[CVPR 2026\] Memory-Augmented Scene Understanding and Exploration for Open-World Aerial Object-Goal Navigation](../../CVPR2026/robotics/memory-augmented_scene_understanding_and_exploration_for_open-world_aerial_objec.md)
- [\[ECCV 2024\] Prioritized Semantic Learning for Zero-shot Instance Navigation](../../ECCV2024/robotics/prioritized_semantic_learning_for_zero-shot_instance_navigation.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)
- [\[CVPR 2026\] Humanoid Generative Pre-Training for Zero-Shot Motion Tracking](../../CVPR2026/robotics/humanoid_generative_pre-training_for_zero-shot_motion_tracking.md)

</div>

<!-- RELATED:END -->
