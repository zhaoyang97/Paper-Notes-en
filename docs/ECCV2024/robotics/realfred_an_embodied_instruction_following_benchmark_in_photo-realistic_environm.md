---
title: >-
  [Paper Note] ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments
description: >-
  [ECCV 2024][Robotics][Embodied AI] Introduces the ReALFRED benchmark, which replaces ALFRED's synthetic single-room scenes with 150 real-world 3D scanned, interactive multi-room environments, providing 30,696 free-form language instructions and revealing a significant performance drop of existing embodied instruction-following methods in real environments.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Embodied AI"
  - "Instruction Following"
  - "3D Scanned Environments"
  - "Multi-Room Navigation"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: ef21dadeb496c883
---

# ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments

**Conference**: ECCV 2024  
**arXiv**: [2407.18550](https://arxiv.org/abs/2407.18550)  
**Code**: [GitHub](https://github.com/snumprlab/realfred)  
**Area**: Robotics  
**Keywords**: Embodied AI, Instruction Following, 3D Scanned Environments, Multi-Room Navigation, Benchmark Dataset

## TL;DR

Introduces the ReALFRED benchmark, which replaces ALFRED's synthetic single-room scenes with 150 real-world 3D scanned, interactive multi-room environments, providing 30,696 free-form language instructions and revealing a significant performance drop of existing embodied instruction-following methods in real environments.

## Background & Motivation

Building autonomous robot assistants capable of performing daily household tasks is a long-term research goal. To train such agents, interactive simulation environments are required to allow them to learn task-completion skills through a large volume of interactions.

Current benchmarks and environments exhibit **three core gaps**:

**Visual Domain Gap**: Benchmarks represented by ALFRED construct environments using the Unity game engine and synthetic CAD assets, leading to visual styles that differ significantly from the real world. Research shows that this domain gap causes sharp performance degradation during deployment.

**Limited Spatial Scale**: ALFRED only provides environments at a single-room granularity (total navigable area of $1,356\text{ m}^2$), whereas real household scenes typically involve navigation across multiple rooms. Constructing large-scale, high-fidelity spaces in synthetic environments is extremely challenging.

**Insufficient Interactive Capabilities**: Although 3D scanned environments (e.g., Matterport3D, HM3D) are visually realistic, the scenes are static—objects cannot be interacted with (e.g., no picking up, heating, or cooling). Although Habitat-Web supports pick-and-place, it is limited to basic interaction and relies on templated language instructions.

**Key Challenge**: 3D scanned environments are visually realistic but lack interactivity, while synthetic environments support rich interactions but suffer from visual distortion. ReALFRED aims to **simultaneously satisfy** four criteria: photorealistic visuals, multi-room navigation, rich object interactions, and free-form language instructions—something no previous benchmark has fully covered (Fig. 2).

## Method

### Overall Architecture

ReALFRED is a holistic benchmark dataset. Its core work consists of three components: (1) constructing interactive 3D scanned environments, (2) generating expert demonstrations, and (3) collecting free-form language instructions. Based on the AI2-THOR simulator, it extends the ALFRED benchmark to support larger physical spaces and a smaller visual domain gap.

### Key Designs

1. **Construction of Interactive 3D Scanned Scenes**: The team visited residential homes in person, utilizing the same 3D scanner as Matterport3D (equipped with three RGB cameras and a depth sensor) to perform scans at 2.5-meter intervals, with supplementary scans for areas occluded by furniture. The core challenge is that objects in the scanned data are fused with the background and cannot be interacted with. The solution is **manual asset separation**: decomposing the 3D scans into background elements and interactive objects, adding state-change textures (e.g., "dirty" textures) to objects, and then reconstructing them in the Unity editor to make them compatible with the AI2-THOR simulator. It provides 150 scenes and 112 object categories (86 pickupable + 26 containers), with a total floor area of $10,060\text{ m}^2$—far exceeding ALFRED's 120 scenes / $2,555\text{ m}^2$.

2. **Multi-Room Task Design and Expert Demonstrations**: Expert demonstrations for 7 task types are generated using PDDL (Planning Domain Definition Language) rules and a planner. Compared to ALFRED's single-room tasks, tasks in ReALFRED require **cross-room navigation**—agents must traverse doors and hallways from one room to another, executing tasks with longer step counts (Fig. 7 shows that the number of steps and trajectory lengths in ReALFRED significantly exceed those in ALFRED). The data is split into 135 seen scenes and 15 unseen scenes.

3. **Free-form Language Instruction Collection**: A total of 30,696 language instructions were collected via 93 Amazon Mechanical Turk "Master" workers, with each instruction containing a high-level goal description and step-by-step instructions. Quality was ensured through additional voting validation, and invalid instructions were replaced and recollected.

### Loss & Training

ReALFRED itself is a benchmark rather than a model. The authors evaluated two types of approaches:
- **Imitation Learning** (Seq2Seq, MOCA, ABP): Directly maps visual observations and language instructions to action sequences.
- **Spatial Map Reconstruction** (HLSM, FILM, LLM-Planner, CAPEAM): Plans actions after building semantic spatial representations based on predicted depth maps.

All methods use the same evaluation metrics as ALFRED: Success Rate (SR) and Goal-Condition Success Rate (GC).

## Key Experimental Results

### Main Results

**Performance of Various Methods on ReALFRED (%)**:

| Method | Category | Val Seen SR | Val Seen GC | Val Unseen SR | Val Unseen GC | Test Unseen SR | Test Unseen GC |
|------|------|------------|------------|--------------|--------------|---------------|---------------|
| Seq2Seq | Imitation Learning | 0.77 | 6.93 | 0.00 | 4.03 | 0.00 | 3.50 |
| MOCA | Imitation Learning | 12.64 | 20.95 | 1.44 | 6.76 | 0.62 | 5.14 |
| **ABP** | **Imitation Learning** | **24.71** | **33.80** | **4.22** | **11.71** | **3.54** | **10.57** |
| HLSM | Spatial Map | 4.23 | 9.14 | 1.08 | 6.12 | 0.49 | 4.28 |
| FILM | Spatial Map | 7.08 | 11.93 | 4.44 | 9.26 | 2.15 | 6.56 |
| CAPEAM | Spatial Map | 13.45 | 18.16 | 4.92 | 9.47 | 2.87 | 7.36 |
| **Human** | - | - | - | - | - | **85.00** | **91.30** |

### Ablation Study

**Sim2Real Transfer Comparison**:

| Setting | Domain Adaptation Method | Multi+Single-Room SR | Single-Room Only SR | Description |
|------|-----------|------------|------------|------|
| Sim2Real | None | 0.115 | 0.0 | Trained on synthetic environments, evaluated directly |
| Sim2Real | CycleGAN | 0.115 | 0.327 | With domain adaptation |
| Sim2Real | UVCGAN-v2 | 0.115 | 0.327 | Better domain adaptation |
| **Real2Real** | **None** | **2.405** | **2.614** | **Trained on real scanned environments** |

**Environment Scale Comparison**:

| Benchmark | Number of Scenes | Total Floor Area ($m^2$) | Total Navigable Area ($m^2$) | Navigation Complexity | Object Categories |
|------|-------|----------------|------------------|-----------|---------|
| ReplicaCAD | 111 | 8,824.5 | - | - | 92 |
| ALFRED | 120 | 2,555 | 1,356 | 2.549 | 82 |
| **ReALFRED** | **150** | **10,060** | **4,251** | **3.020** | **112** |

### Key Findings

- **Performance Drop Across All SOTA Methods**: The best method, ABP, achieves an SR of ~26% on ALFRED unseen, but only 4.22% on ReALFRED unseen, representing a drop of over 80%.
- **Opposite Trend to ALFRED**: On ALFRED, spatial-map-based methods outperform imitation learning. However, on ReALFRED, **imitation learning outperforms spatial map methods**, because the limited field of view in multi-room environments constrains spatial map reconstruction.
- **Navigation is the Primary Bottleneck**: ABP's navigation success rate on ReALFRED unseen is only 59.18%, compared to 84.82% on ALFRED.
- **Large Spaces Lead to Further Performance Decline**: In scenes smaller than $30.44\text{ m}^2$, the SR is 5.46%, whereas in scenes larger than this threshold, the SR drops to only 1.77%.
- **Doorframes and Narrow Passages** are collision hotspots (Fig. 8). Spatial map methods tend to over-perceive obstacles, further blocking narrow passages.
- **Significant Sim2Real Gap**: The SR of Real2Real is over 20+ times higher than that of Sim2Real, and even with CycleGAN domain adaptation, the improvement is limited.
- **Human vs. Agent Gap**: Human performance reaches 85% SR vs. 3.54% SR for the best method, indicating a massive gap.

## Highlights & Insights

- **Fills an Important Gap**: It is the first to simultaneously satisfy four criteria: photorealistic visuals, multi-room scale, rich interactivity, and free-form language instructions.
- **Meticulous Environmental Engineering**: Manually separating each scanned 3D object, adding state textures, and reconstructing them into interactive assets is labor-intensive but yields high-quality data.
- **Reveals Profound Issues**: Methods performing well in synthetic environments fail almost entirely in real environments, indicating that the generalization ability of current methods is highly insufficient.
- The challenges introduced by multi-room navigation (narrow passages, large-space exploration) remain fundamental, unsolved problems for existing methods.

## Limitations & Future Work

- **Limited Task Types**: Only 7 task types are supported, failing to cover more complex real-world scenarios (such as tasks requiring bimanual manipulation).
- **English Only**: Practical deployment requires multilingual support.
- The manual object separation pipeline is highly labor-intensive, limiting the further scaling of the dataset.
- The physical simulation accuracy of interactive objects in the environment (e.g., object collisions, fluids) might be limited compared to synthetic environments.
- The extremely low success rate of current methods suggests that this benchmark might be overly difficult for near-term methods, indicating a need for evaluation metrics of intermediate difficulty.

## Related Work & Insights

- **Relationship with ALFRED**: ReALFRED is a realistic upgrade of ALFRED. It retains the same task framework and evaluation metrics, facilitating direct comparison.
- **Difference from Habitat-Web**: HW supports pick-and-place in 3D scanned environments but only uses templated language; ReALFRED supports more complex interactions (heating, cooling, slicing, etc.) and free-form language.
- **Sim2Real Experimental Insights**: Simple visual domain adaptation (e.g., CycleGAN) is far from sufficient to bridge the gap. Solutions must address deeper levels, such as environment layout, object distribution, and task complexity.
- **Implications for LLM-based Agents**: Even though LLM-Planner leverages knowledge from language models, its performance on ReALFRED remains poor, indicating that **low-level visual perception and navigation capabilities** are the current bottlenecks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Constructs the first embodied benchmark to simultaneously satisfy four critical requirements, involving substantial engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluates 7 SOTA methods, Sim2Real transfer, human performance, and multi-dimensional environmental impact.
- **Writing Quality**: ⭐⭐⭐⭐ — Features a clear structure, comprehensive comparisons, and extensive charts.
- **Value**: ⭐⭐⭐⭐⭐ — Pinpoints critical gaps for the embodied AI community; the high-value benchmark dataset will accelerate the development of more robust methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](../../ICML2025/robotics/hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)
- [\[AAAI 2026\] Realistic Synthetic Household Data Generation at Scale](../../AAAI2026/robotics/realistic_synthetic_household_data_generation_at_scale.md)
- [\[ECCV 2024\] See and Think: Embodied Agent in Virtual Environment](see_and_think_embodied_agent_in_virtual_environment.md)
- [\[CVPR 2025\] Robotic Visual Instruction](../../CVPR2025/robotics/robotic_visual_instruction.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/robotics/semantic_audio-visual_navigation_in_continuous_environments.md)

</div>

<!-- RELATED:END -->
