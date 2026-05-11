---
title: >-
  [Paper Note] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots
description: >-
  [ICLR 2026][Robotics][simulation platform] RoboCasa365 constructs a large-scale simulation benchmark comprising 365 everyday kitchen tasks, 2,500 diverse kitchen scenes, and over 2…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "simulation platform"
  - "household mobile manipulation"
  - "multi-task learning"
  - "foundation model training"
  - "lifelong learning"
date: 2026-05-08
content_hash: 12476475e583c8b7
---

# RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots

**Conference**: ICLR 2026
**arXiv**: [2603.04356](https://arxiv.org/abs/2603.04356)
**Code**: [https://robocasa.ai](https://robocasa.ai) (project page with open-source code and models)
**Area**: Robotics / Simulation Benchmarks / Generalist Robots
**Keywords**: simulation platform, household mobile manipulation, multi-task learning, foundation model training, lifelong learning

## TL;DR

RoboCasa365 constructs a large-scale simulation benchmark comprising 365 everyday kitchen tasks, 2,500 diverse kitchen scenes, and over 2,000 hours of robot interaction data. It systematically evaluates generalist robot policies under three paradigms—multi-task learning, foundation model training, and lifelong learning—and finds that task diversity in pretraining data is the key factor for improving downstream generalization.

## Background & Motivation

**Background**: Robot learning has advanced rapidly in recent years, with large-scale robot foundation models such as π₀, π₀.₅, and GR00T N1.5 demonstrating generalization to novel objects, environments, and tasks.

**Limitations of Prior Work**: Training generalist robots requires massive data, yet existing real-world datasets remain limited in diversity and task coverage. Real-world evaluation is costly, noisy, and difficult to reproduce for systematic comparison.

**Key Challenge**: Existing simulation frameworks (e.g., RLBench, LIBERO, robosuite) offer few tasks, low environment diversity, and lack large-scale accompanying datasets, making them insufficient to support systematic research on generalist robot policies. Most frameworks focus on simple tabletop manipulation or single-room settings and cannot answer the core question of how task diversity, environment variation, and data scale affect generalization.

**Goal**: (a) Construct a sufficiently large and diverse simulation benchmark; (b) provide systematic evaluation protocols covering multi-task learning, foundation model pretraining + fine-tuning, and lifelong learning; (c) analyze the key factors affecting generalist robot performance through extensive experiments.

**Key Insight**: Substantially extend the existing RoboCasa platform—from 100 to 2,500 scenes, from dozens to 365 tasks, and from 100K to 500K+ demonstrations—to create an ImageNet-scale benchmark for the household kitchen domain.

**Core Idea**: Through extreme scaling along three dimensions—tasks, scenes, and data—construct the first robot simulation framework that simultaneously satisfies all four conditions: large-scale tasks, large-scale scenes, large-scale data, and systematic benchmarking.

## Method

### Overall Architecture

The RoboCasa365 pipeline consists of four core components: **Assets → Scenes → Tasks → Datasets**. The asset library provides 3D objects and interactive appliances; scenes combine assets into diverse kitchen environments; tasks define the target behaviors robots must complete; datasets are generated through human teleoperation and MimicGen synthesis to produce large-scale demonstration trajectories. Three benchmark evaluation protocols (multi-task learning, foundation model training, lifelong learning) then leverage these data for systematic assessment.

Simulation is built on robosuite + MuJoCo physics engine running at 20 Hz. The robot platform uses a Franka Panda arm with an Omron mobile base, with a 12-dimensional action space (7-DoF end-effector + 5-DoF mobile base).

### Key Designs

1. **Large-Scale Asset Expansion**:

    - **Function**: Expands interactive appliances from 4 categories and 20 instances in RoboCasa to 12 categories and 456 instances, adding toasters, blenders, electric kettles, etc.; 57 new categories of 3D objects are included.
    - **Mechanism**: Each appliance category contains 20–50 distinct instances to ensure sufficient appearance diversity for generalization research; all appliances are articulated MJCF models supporting interactions such as opening doors, pressing buttons, and turning knobs.
    - **Design Motivation**: Sufficient instance diversity is a prerequisite for studying generalization to novel instances; appliances in the original RoboCasa (e.g., refrigerators, ovens, dishwashers) were not even fully articulable.

2. **2,500 Diverse Kitchen Scenes**:

    - **Function**: Constructs two non-overlapping scene sets: pretraining scenes and target scenes.
    - **Mechanism**: Floor plans from 50 real U.S. kitchens sourced from Zillow form 50 layouts; 50 styles (material/appliance/texture selections) are independently designed; layout × style yields 2,500 pretraining scenes. An additional 10 target scenes are used for fine-tuning and evaluation.
    - **Design Motivation**: Decoupling layout and style causes scene count to grow multiplicatively; pretraining and target scene styles do not overlap, enabling rigorous testing of environment generalization. Real floor plans span five U.S. regions (Bay Area, Austin, Denver, Boston, Atlanta) to ensure geographic diversity.

3. **365-Task Daily Activity System**:

    - **Function**: Defines 65 atomic tasks (single skills) and 300 composite tasks (multi-skill sequences), covering 60 daily activities such as boiling water, toasting bread, brewing coffee, washing dishes, and storing leftovers.
    - **Mechanism**: Eight primitive skills are defined: pick-and-place, open/close door, open/close drawer, turn lever, turn knob, press button, insert, and navigate. Composite tasks are generated via an LLM pipeline (activity → task name + description + involved objects + skill sequence), then manually coded. Task length ranges from 1 to 15+ sub-tasks.
    - **Design Motivation**: Atomic tasks assess single-step manipulation; composite tasks assess long-horizon reasoning and planning. 220 tasks require mobile manipulation and 145 do not, covering both important settings.

4. **Large-Scale Dataset Construction**:

    - **Function**: Generates over 2,000 hours of data through human teleoperation and MimicGen synthesis.
    - **Mechanism**: **Pretraining data**—100 human demonstrations per task across 300 tasks (30K trajectories, 404 hours) + 10K MimicGen-synthesized demonstrations per atomic task across 60 tasks (600K trajectories, 1,615 hours); **Target data**—500 human demonstrations per task across 50 representative tasks (25K trajectories, 208 hours).
    - **Design Motivation**: MimicGen expands atomic task data at 100× scale, but subsequent experiments show that synthetic data quality is inconsistent and may in fact degrade downstream performance—a finding that is itself valuable.

5. **Three-Tier Target Task Categorization**:

    - **Function**: Divides 50 target tasks into Atomic (18), Composite-Seen (16, encountered during pretraining), and Composite-Unseen (16, not encountered during pretraining).
    - **Mechanism**: The three groups respectively assess basic manipulation capability, transfer to seen composite tasks, and zero-shot generalization to entirely novel composite tasks.
    - **Design Motivation**: This hierarchical design allows decoupled analysis of distinct capability dimensions.

### Loss & Training

All experiments use language-conditioned visual policies. Four state-of-the-art methods are compared:

- **Diffusion Policy**: A visual motor policy based on diffusion models.
- **π₀**: A vision-language-action flow-matching model.
- **π₀.₅**: An enhanced variant of π₀ targeting open-world generalization.
- **GR00T N1.5**: NVIDIA's open-source humanoid robot foundation model.

All VLA models are fine-tuned from publicly released pretrained checkpoints. In the foundation model training experiments, models are first trained on all pretraining data, then fine-tuned on each of the three target data groups, comparing performance at 10%/30%/100% data quantities.

## Key Experimental Results

### Main Results (Multi-Task Learning)

| Task Group | Diffusion Policy | π₀ | π₀.₅ | GR00T N1.5 |
|---|---|---|---|---|
| Atomic | 15.7% | 36.3% | 39.6% | 43.0% |
| Composite-Seen | 0.2% | 5.2% | 7.1% | 9.6% |
| Composite-Unseen | 1.25% | 0.7% | 1.2% | 4.4% |
| **Average** | **6.1%** | **15.0%** | **16.9%** | **20.0%** |

GR00T N1.5 achieves the best performance across all groups, while Diffusion Policy performs worst, indicating that high-capacity VLA models fit large-scale multi-task data more effectively. All methods perform extremely poorly on Composite-Unseen, and generalist capability remains an open challenge.

### Foundation Model Training Results

| Task Type | Pretrain Only | Target 10% Only | Target 30% Only | Target 100% Only | Pretrain + Target 10% | Pretrain + Target 30% | Pretrain + Target 100% |
|---|---|---|---|---|---|---|---|
| Atomic | 41.9% | 38.7% | 50.6% | 60.6% | 56.9% | 59.1% | 68.5% |
| Composite-Seen | 0.0% | 11.0% | 22.7% | 35.0% | 25.4% | 34.6% | 40.6% |
| Composite-Unseen | 0.2% | 11.2% | 27.5% | 33.3% | 22.7% | 30.8% | 42.1% |
| **Average** | **15.1%** | **21.0%** | **34.3%** | **43.7%** | **35.9%** | **42.2%** | **51.1%** |

Pretraining yields approximately **3× data efficiency**: pretraining + 10% target data (35.9%) approaches the performance of 30% target data alone (34.3%). On Composite-Unseen, pretraining + 100% target data reaches 42.1%, far exceeding 33.3% with target data alone, demonstrating that pretraining provides especially pronounced generalization gains on unseen tasks.

### Pretraining Data Composition Analysis

| Pretraining Data | Avg (10% target) | Avg (100% target) |
|---|---|---|
| No pretraining | 21.0% | 43.7% |
| Human50 (50 tasks) | 34.7% | 50.0% |
| Human300 (300 tasks) | 40.0% | 52.5% |
| Human300 + MG60 (synthetic) | 35.9% | 51.1% |

Key findings: (1) Human data only (Human300) **outperforms** the combination of human and MimicGen synthetic data (Human300+MG60) due to inconsistent synthetic data quality; (2) expanding pretraining tasks from 50 to 300 yields significant gains, especially under low data regimes (10%); (3) the improvement is most pronounced on Composite-Unseen tasks (+8.5% at 10%), underscoring that task diversity is critical for generalizing to novel tasks.

### Lifelong Learning Results

| Training Phase | Atomic | Phase 2–3 Tasks | Phase 4–5 Tasks | Phase 6+ Tasks |
|---|---|---|---|---|
| Phase 1 | 41.5% | — | — | — |
| Phase 2 | 13.9% | 24.5% | — | — |
| Phase 3 | 13.9% | 4.8% | 11.3% | — |
| Phase 4 | 10.6% | 1.7% | 2.7% | 4.3% |

Lifelong learning suffers from severe catastrophic forgetting: Atomic task success rate drops from 41.5% in Phase 1 to 10.6% in Phase 4. Long-horizon tasks are intrinsically harder to learn (diagonal success rates decrease: 41.5% → 24.5% → 11.3% → 4.3%).

### Real-World Transfer

| Method | Close Kettle Lid | Retrieve from Oven | Counter→Cabinet | Place on Dish Rack | Average |
|---|---|---|---|---|---|
| Real Only | 70% | 70% | 52% | 55% | 61.8% |
| Sim + Real | 70% | 100% | 84% | 65% | 79.8% |

Joint training with simulation data improves average success rate from 61.8% to 79.8% (+18.1%), validating the practical value of the simulation benchmark for real-world deployment.

## Highlights & Insights

- **Empirical evidence that data quality > data quantity**: MimicGen synthetic data scales atomic task data by 100×, yet incorporating it actually degrades downstream performance. This serves as a warning for data strategy in robot foundation model development—blindly scaling synthetic data may be counterproductive; filtering and quality control are essential.
- **Non-linear returns from task diversity**: Expanding pretraining tasks from 50 to 300 nearly doubles performance in the low-data regime, with gains on unseen tasks exceeding those on seen tasks, revealing that task diversity is the fuel for generalization.
- **Decoupled layout × style scene generation**: By decoupling kitchen spatial layout from visual style, a 50×50 combination produces 2,500 scenes, achieving exponential scene diversity at limited modeling cost. This design principle is transferable to other domains requiring large-scale environment variation (e.g., autonomous driving scene generation).
- **LLM-assisted task system design**: The pipeline of using LLMs to generate activity lists → task blueprints → manual coding balances task diversity with quality control, proving more practical than either fully manual or fully automated design.

## Limitations & Future Work

- **Kitchen-only scope**: All 2,500 scenes are kitchens; whether findings transfer to other household or commercial environments such as bedrooms, living rooms, or offices remains to be validated.
- **Sim-to-Real Gap**: Although the effectiveness of joint sim-to-real training is validated, real-world comparisons cover only four simple tasks with a specific camera alignment procedure, limiting generalizability.
- **Unexplained synthetic data degradation**: The paper identifies that synthetic data hurts performance but does not deeply analyze the cause, nor does it explore data filtering or weighting strategies. Developing methods to effectively leverage large-scale mixed-quality datasets is an important future direction.
- **Simplistic lifelong learning benchmark design**: The four-phase sequential learning setup represents the most basic lifelong learning scenario and does not incorporate classic continual learning methods such as experience replay or elastic weight consolidation.
- **Physical fidelity limitations**: MuJoCo has limited support for fluids, soft bodies, and cloth, restricting the types of tasks that can be simulated (e.g., cooking tasks involving liquid).
- **Single robot morphology**: Only the Franka Panda arm with a mobile base is evaluated; dual-arm or humanoid robots are not tested, limiting research on cross-morphology generalization.

## Related Work & Insights

- **vs. RoboCasa (RSS 2024)**: RoboCasa365 is a substantial extension of RoboCasa (100→2,500 scenes, ~100→365 tasks, 100K→500K+ demonstrations), adding MimicGen synthetic data and three benchmark evaluation protocols. The primary contributions lie in scale and systematic experimentation.
- **vs. LIBERO (NeurIPS 2023)**: LIBERO has only 130 tasks with limited environment diversity and focuses on lifelong learning; RoboCasa365 has 2.8× more tasks and covers three paradigms: multi-task learning, pretraining, and lifelong learning.
- **vs. BEHAVIOR-1K (CoRL 2023)**: BEHAVIOR-1K provides diversity across 1,000 activities but lacks a large-scale accompanying dataset; RoboCasa365 covers fewer activities (60) but provides hundreds of high-quality demonstrations per task.
- **vs. ManiSkill series**: ManiSkill focuses on general object manipulation and GPU-parallel simulation supporting richer physical interaction; RoboCasa365 focuses on room-level daily tasks, making the two frameworks complementary.

## Rating

- **Novelty**: ⭐⭐⭐ — Core technical contributions are engineering-driven scale expansion; methodological novelty is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comparisons across four SOTA methods, three training paradigms, data composition ablations, and real-world validation form a highly comprehensive experimental system.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, experimental discussion is in-depth, and figures and tables are abundant.
- **Value**: ⭐⭐⭐⭐ — As a standardized evaluation benchmark for generalist robot policies, the framework offers high infrastructure value to the community; findings from the data composition analysis directly inform practical data strategy decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/robotics/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICLR 2026\] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)

</div>

<!-- RELATED:END -->
