---
title: >-
  [Paper Note] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System
description: >-
  [CVPR 2026][Autonomous Driving][end-to-end driving] KnowVal is an end-to-end autonomous driving system that addresses two fundamental deficiencies—knowledge reasoning and value alignment—through three core components: (1) Retrieval-guided Open-world Perception, which integrates standard 3D detection, VL-SAMv2-based long-tail object recognition, and VLM-based scene understanding; (2) Perception-guided Knowledge Retrieval, which queries a driving knowledge graph covering traffic regulations, defensive driving, and ethical norms; and (3) a World Model for future state prediction combined with a human-preference-trained Value Model for trajectory evaluation. The system achieves the lowest collision rate on nuScenes and state-of-the-art performance on Bench2Drive and NVISIM.
tags:
  - CVPR 2026
  - Autonomous Driving
  - end-to-end driving
  - knowledge graph
  - value model
  - world model
  - open-world perception
  - VLM
  - retrieval-augmented planning
date: 2026-05-08
content_hash: f0dc260b45c95cce
---

# KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System

**Conference**: CVPR 2026
**arXiv**: [2512.20299](https://arxiv.org/abs/2512.20299)
**Code**: To be confirmed
**Area**: Autonomous Driving / Knowledge-Augmented Planning
**Keywords**: end-to-end driving, knowledge graph, value model, world model, open-world perception, VLM, retrieval-augmented planning

## TL;DR

KnowVal is an end-to-end autonomous driving system that addresses two fundamental deficiencies—knowledge reasoning and value alignment—through three core components: (1) Retrieval-guided Open-world Perception, which integrates standard 3D detection, VL-SAMv2-based long-tail object recognition, and VLM-based scene understanding; (2) Perception-guided Knowledge Retrieval, which queries a driving knowledge graph covering traffic regulations, defensive driving, and ethical norms; and (3) a World Model for future state prediction combined with a human-preference-trained Value Model for trajectory evaluation. The system achieves the lowest collision rate on nuScenes and state-of-the-art performance on Bench2Drive and NVISIM.

## Background & Motivation

End-to-end autonomous driving has advanced rapidly in recent years, replacing modular pipelines with unified models that span perception to planning, thereby eliminating inter-module error accumulation. However, existing end-to-end methods suffer from two fundamental deficiencies:

**Lack of knowledge reasoning**: Current models are predominantly data-driven, learning statistical patterns rather than driving knowledge. When confronted with long-tail scenarios not covered by training data—such as unusual construction zone signage, rare traffic gestures, or ethical dilemmas—these models cannot invoke traffic regulations or defensive driving principles the way a human driver would.

**Lack of value alignment**: A gap exists between model optimization objectives (e.g., imitation learning trajectory L2 distance) and human judgments of what constitutes good driving. Human driving quality encompasses passenger comfort, yielding to vulnerable road users, and adherence to social norms—none of which can be learned through simple trajectory matching.

**Illustrative scenarios**:
- A construction zone displays temporary speed-limit signs and traffic cones → standard 3D detectors fail to recognize these objects → open-world perception and knowledge retrieval are required to infer "this is a construction zone; reduce speed."
- An ambulance approaches from behind with its siren active → the model must know the traffic regulation "pull over and yield" → knowledge graph support is required.
- A pedestrian hesitates at a crosswalk → defensive driving knowledge should guide the vehicle to decelerate and observe → knowledge reasoning is necessary.

Existing approaches such as DriveVLM and LMDrive incorporate VLMs primarily for scene description rather than knowledge reasoning; unified frameworks such as UniAD achieve end-to-end operation but still lack value-guided planning.

## Method

### Overall Architecture

KnowVal comprises three collaborating core modules forming a closed loop from perception to knowledge retrieval to planning:

```
Camera/LiDAR → Open-world Perception → Perception Verbalizer
                                              ↓
                     Knowledge Graph ← Knowledge Retrieval
                                              ↓
                   World Model → Value Model → Planning Decision
```

### Module 1: Retrieval-guided Open-world Perception

**Goal**: Build a comprehensive perception capability that transcends closed category sets.

A three-layer perception hierarchy is adopted:

**Layer 1 — Specialized Perception (standard 3D detection)**:
- Employs mature 3D object detectors (e.g., BEVFormer/StreamPETR) to detect common traffic participants (vehicles, pedestrians, cyclists, etc.).
- Provides precise 3D bounding boxes, velocities, and headings as structured outputs.
- Constitutes the perceptual foundation present in conventional end-to-end methods.

**Layer 2 — Open-ended 3D Perception (long-tail object recognition)**:
- Built upon VL-SAMv2 and open-vocabulary detectors such as OpenAD.
- Detects long-tail objects beyond the scope of Specialized Perception: construction cones, temporary signs, road potholes, fallen debris, etc.
- VL-SAMv2 leverages vision-language alignment, requiring no per-class annotated training data.
- Outputs include object category descriptions, 2D/3D locations, and confidence scores.

**Layer 3 — Abstract Concept Understanding**:
- Employs a VLM (e.g., GPT-4V/InternVL) for high-level semantic scene understanding.
- Extracts abstract information not representable by bounding boxes: road surface conditions (wet/flooded), weather, traffic density, and overall scene characteristics (tense/calm).
- Outputs structured scene attribute descriptions.

**Complementary logic**: Layer 1 provides precise structured detections; Layer 2 covers Layer 1's blind spots (long-tail objects); Layer 3 supplies scene-level semantics beyond the object level.

### Module 2: Perception-guided Knowledge Retrieval

**Goal**: Retrieve relevant knowledge entries from a pre-built driving knowledge graph based on perception outputs, providing knowledge support for planning.

**Driving Knowledge Graph Construction**:

Three knowledge bases are pre-built:
1. **Traffic regulation knowledge**: Structured entries covering speed limits, right-of-way priorities, and zone-specific rules.
2. **Defensive driving knowledge**: Empirical safety guidelines covering following distances, blind-spot risks, and adverse-weather strategies.
3. **Ethical norm knowledge**: Driving ethics principles covering protection of vulnerable road users, emergency yielding rules, and ethical dilemma handling.

Each entry is stored as a (trigger condition, knowledge content, recommended action) triplet with a vectorized index.

**Perception Verbalizer**:

Converts structured outputs from all three layers into a natural-language query:

$$q = \text{Verbalizer}(\text{Layer1\_output}, \text{Layer2\_output}, \text{Layer3\_output})$$

For example, detecting "construction cones 10 m ahead, road narrows, temporary sign on the left" generates the query "driving rules and safety precautions for narrowing roads in construction zones."

**Knowledge Retrieval Process**:

$$\mathcal{K}_{\text{relevant}} = \text{LLM-Retrieve}(q, \mathcal{G}_{\text{knowledge}})$$

An LLM serves as the retriever, selecting the $k$ most relevant knowledge entries from the knowledge graph.

**Bidirectional feedback mechanism (key innovation)**: The retrieval module not only supplies knowledge to the planning stage but also **feeds back required perceptual elements** to the perception module. For example, retrieving "watch for temporary traffic lights in construction zones" prompts the perception module to attend specifically to traffic light detection. This establishes a closed-loop perception ↔ knowledge retrieval interaction.

### Module 3: Planning with World Model + Value Model

**World Model — Future State Prediction**:

Given the current perception state $s_t$ and a candidate action $a_t$, the World Model predicts the state sequence over a horizon of $H$ steps:

$$\hat{s}_{t+1:t+H} = f_{\text{world}}(s_t, a_t, \mathcal{K}_{\text{relevant}})$$

The retrieved knowledge $\mathcal{K}_{\text{relevant}}$ is provided as an additional conditioning input, so predictions reflect not only physical dynamics but also knowledge constraints (e.g., a "30 km/h speed limit in construction zone" influences the predicted behavior of surrounding vehicles).

**Value Model — Trajectory Value Estimation**:

$$V(\tau) = f_{\text{value}}(\hat{s}_{t+1:t+H}, \mathcal{K}_{\text{relevant}})$$

The Value Model is trained on a **human-preference dataset**, learning human preferences over driving trajectories:
- Training data: Human annotators provide pairwise trajectory preference rankings (analogous to RLHF).
- Evaluation dimensions: Safety, comfort, efficiency, regulatory compliance, etc.
- Output: A scalar value score for each candidate trajectory.

**Final Decision**:

$$a^* = \arg\max_{a \in \mathcal{A}} V(f_{\text{world}}(s_t, a, \mathcal{K}_{\text{relevant}}))$$

The action yielding the highest Value Model score is selected. The decision process is interpretable—retrieved knowledge sources and the Value Model's dimension-wise scores can be traced for each decision.

### System Properties

- **Compatibility with existing architectures**: KnowVal's modules can be inserted into any end-to-end driving framework without constraining the underlying perception or planning architecture.
- **Interpretability**: Every decision can be traced back to its knowledge sources and value assessments, facilitating debugging and safety auditing.

## Key Experimental Results

### Evaluation Benchmarks

- **nuScenes**: Standard autonomous driving perception and planning benchmark.
- **Bench2Drive**: Comprehensive closed-loop driving evaluation benchmark.
- **NVISIM**: NVIDIA simulation environment.

### nuScenes Planning Results

| Method | Collision Rate↓ | L2 (1s) | L2 (3s) |
|--------|----------------|---------|---------|
| UniAD | — | — | — |
| VAD | — | — | — |
| KnowVal | **Lowest** | **SOTA** | **SOTA** |

KnowVal achieves the **lowest collision rate** on nuScenes, significantly outperforming existing end-to-end methods. The substantial reduction in collision rate is primarily attributed to safety-oriented knowledge retrieved from the knowledge graph and the safety preference embedded in the Value Model.

### Bench2Drive Closed-Loop Evaluation

KnowVal achieves state-of-the-art performance across diverse driving scenarios (urban, highway, intersection, adverse weather). The advantage is most pronounced in scenarios requiring complex reasoning—such as unprotected left turns and construction zone navigation—where knowledge retrieval plays a decisive role.

### NVISIM Simulation Testing

KnowVal demonstrates real-time capability and robustness in the NVIDIA simulation environment, achieving state-of-the-art results.

### Ablation Study

| Configuration | Collision Rate | Note |
|---------------|---------------|------|
| Specialized Perception only | baseline | Standard end-to-end baseline |
| + Open-world Perception | ↓ | Long-tail object awareness reduces unexpected collisions |
| + Knowledge Retrieval | ↓↓ | Knowledge-guided planning improves safety |
| + Value Model | **Lowest** | Value alignment further refines trajectory selection |

Each module contributes independently and positively; the Value Model's gain is most prominent along the safety dimension.

### Knowledge Retrieval Effectiveness Analysis

Knowledge Retrieval is particularly impactful in long-tail scenarios—collision rates decrease by more than 30% in scenes containing unconventional traffic elements. This validates the necessity of combining data-driven and knowledge-driven approaches.

## Highlights & Insights

- **Well-designed three-layer perception hierarchy**: From precise structured detection → open-vocabulary long-tail detection → abstract concept understanding, the system covers the full spectrum of autonomous driving perception requirements and represents the most complete perceptual capability definition to date.
- **Bidirectional closed loop between perception and knowledge retrieval**: Rather than a unidirectional "perception → retrieval" pipeline, knowledge retrieval can inversely direct perceptual attention. This bidirectional feedback mechanism mirrors the human cognitive process of "observe → associate knowledge → verify carefully."
- **Value Model incorporating human preference**: Analogous to RLHF in the NLP domain, this approach generalizes "good driving" from trajectory L2 distance to multidimensional human preference evaluation—a pioneering attempt at value alignment in autonomous driving.
- **Clear three-category knowledge taxonomy**: Traffic regulations (hard constraints), defensive driving (soft constraints), and ethical norms (ethical constraints) together span a complete driving knowledge hierarchy from legal to ethical dimensions.
- **Strong compatibility**: The system is not bound to specific perception or planning architectures; each module can be independently integrated, lowering the barrier to adoption.

## Limitations & Future Work

1. **Knowledge graph maintenance and updates**: Traffic regulations vary by region and evolve over time. Automating knowledge graph updates and version management is a critical challenge for real-world deployment.
2. **VLM inference latency**: Layer 3 VLM scene understanding and LLM-based knowledge retrieval both incur substantial latency. Satisfying the real-time response requirements (<100 ms) of autonomous driving while maintaining inference quality requires further optimization.
3. **Preference generalization of the Value Model**: The scale and diversity of the human-preference dataset directly constrain the Value Model's generalization capability. Driving preferences vary significantly across cultural contexts, and a single dataset is unlikely to provide adequate coverage.
4. **False positive rate in open-world perception**: Open-vocabulary detectors such as VL-SAMv2 may produce a high volume of false positives in unconstrained environments. Effectively filtering false positives without discarding genuine long-tail objects presents an inherent trade-off.
5. **Sim-to-real transfer**: Primary experiments are conducted on nuScenes and simulation environments. The accuracy of knowledge retrieval and the robustness of the Value Model under real-world road conditions require further validation.

## Related Work & Insights

- **UniAD (CVPR 2023)**: Unified end-to-end autonomous driving framework → KnowVal augments this paradigm with knowledge reasoning and value alignment.
- **DriveVLM (ECCV 2024)**: Introduces VLMs for driving scene understanding → KnowVal extends this with a complete knowledge retrieval and utilization pipeline.
- **RAG (Retrieval-Augmented Generation)**: Retrieval-augmented paradigm from NLP → KnowVal adapts RAG to knowledge retrieval for autonomous driving.
- **RLHF (Reinforcement Learning from Human Feedback)**: Value alignment methodology from the LLM domain → KnowVal's Value Model is its natural counterpart in autonomous driving planning.
- **OpenAD / VL-SAMv2**: State-of-the-art open-vocabulary 3D detection methods → serve as foundational components of KnowVal's open-world perception.
- **Broader implications**: The knowledge graph + RAG + value alignment framework is generalizable to other embodied intelligence tasks requiring knowledge reasoning (e.g., robotic manipulation, UAV navigation). The human-preference learning approach of the Value Model carries significant implications for the autonomous driving planning community at large.

## Rating

| Dimension | Score (1–5) | Note |
|-----------|------------|------|
| Novelty | 4.5 | The combination of knowledge graph, RAG, and Value Model is pioneering in autonomous driving; the system design is complete and forward-looking. |
| Practicality | 3.5 | Conceptually advanced, but VLM/LLM inference latency and knowledge graph maintenance increase real-world deployment complexity. |
| Experimental Thoroughness | 4.0 | State-of-the-art on three benchmarks with complete ablations; real-time performance analysis and real-road testing are absent. |
| Writing Quality | 4.0 | System architecture is clearly described with well-articulated inter-module relationships and strong motivation; some experimental details could be elaborated further. |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth-_and_region-guided_3d_occupancy_from_surround-view_cameras_for_auton.md)
- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](../../ICCV2025/autonomous_driving/drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[ICCV 2025\] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving](../../ICCV2025/autonomous_driving/adadrive_self-adaptive_slow-fast_system_for_language-grounded_autonomous_driving.md)
- [\[ICCV 2025\] Passing the Driving Knowledge Test](../../ICCV2025/autonomous_driving/passing_the_driving_knowledge_test.md)

<!-- RELATED:END -->
