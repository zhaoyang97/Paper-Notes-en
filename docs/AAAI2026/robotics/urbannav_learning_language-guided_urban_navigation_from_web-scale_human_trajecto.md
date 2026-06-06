---
title: >-
  [Paper Note] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories
description: >-
  [AAAI 2026][Robotics][Urban Navigation] This paper proposes UrbanNav, which leverages web-scale urban walking videos (1,500+ hours from YouTube…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Urban Navigation"
  - "Language Guidance"
  - "Large-Scale Web Video"
  - "Imitation Learning"
  - "Landmark Detection"
date: 2026-05-08
content_hash: 3f028740de107b68
---

# UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories

**Conference**: AAAI 2026
**arXiv**: [2512.09607](https://arxiv.org/abs/2512.09607)  
**Code**: [https://github.com/CASIA-IVA-Lab/UrbanNav](https://github.com/CASIA-IVA-Lab/UrbanNav)  
**Area**: Robotics
**Keywords**: Urban Navigation, Language Guidance, Large-Scale Web Video, Imitation Learning, Landmark Detection

## TL;DR

This paper proposes UrbanNav, which leverages web-scale urban walking videos (1,500+ hours from YouTube, yielding 3 million instruction–trajectory–landmark triplets) to train a language-guided urban navigation policy via an automated annotation pipeline and robust filtering mechanism, achieving an 83.3% navigation success rate in real-world deployment.

## Background & Motivation

Navigating in real urban environments using natural language instructions is a foundational capability for autonomous agents, with applications in last-mile delivery robots, autonomous vehicles, and assistive robots. Urban scenes pose significant challenges: dynamic terrain, unpredictable obstacles, dense pedestrians, and ambiguous human instructions (e.g., "go to the café next to the old bridge").

**Limitations of Prior Work**:

**Restricted target formats**: Most methods rely on GPS coordinates or goal images as navigation targets, making them unable to handle free-form text instructions.

**Simulation-to-real gap**: Policies trained in simulators suffer severe performance degradation in the real world.

**Insufficient data diversity**: Teleoperation data collection is costly and limited in diversity, making it difficult to cover diverse urban scenarios.

**Lack of robustness to noisy instructions**: Instructions provided by real users are often ambiguous or underspecified.

**Core Observation**: YouTube contains abundant egocentric urban walking videos documenting authentic human navigation behavior in diverse urban environments. Can this cheap and plentiful data be exploited to train navigation policies?

**Two Key Questions**:
1. Are all video clips suitable for robot training? (Issues such as inconsistent camera orientation and dangerous behaviors.)
2. How can instruction–action supervision be obtained from unannotated in-the-wild videos?

## Method

### Overall Architecture

UrbanNav consists of two main components:

1. **Automated data construction pipeline**: Extracts trajectories from YouTube walking videos, filters out incompatible data, and annotates language instructions.
2. **Policy model**: Predicts future trajectories conditioned on historical images and language instructions.

### Key Designs

#### 1. **Trajectory Annotation and Robot Compatibility Filtering**

**Trajectory annotation**: Raw videos are uniformly segmented into 2-minute clips, and DPVO (a visual odometry model) is used to estimate the camera pose of each frame relative to the first frame. This yields 106,603 trajectories (3,553 hours).

**Robot compatibility filtering** (a key contribution of this work):

Not all human walking videos are suitable for robot training. Humans turn their heads, look down at phones, and glance sideways, whereas robots typically use a front-facing fixed camera. The filtering pipeline comprises:

**Pitch angle filtering**: The camera pitch angle is estimated per frame; trajectories with pitch variation exceeding 15° are discarded.

**Gaze–motion direction alignment filtering**: A sliding window is used to analyze alignment between the viewing direction and movement direction; clips with directional deviation exceeding 60° (e.g., head-turning or side-gazing) are discarded.

**Crowd density filtering**: YOLOv10 is used to detect pedestrians; trajectories in which more than 5 persons appear in a single frame across 3 or more frames are discarded (to avoid training robots to navigate through dense crowds).

After filtering, 47,008 high-quality trajectories (1,566 hours) are retained, approximately 44% of the original data.

#### 2. **Language Instruction Annotation**

The Qwen2.5-VL-72B large vision-language model is used to automatically detect and annotate landmarks:

**Landmark selection criteria**:
- Must be near the walking trajectory (ensuring reachability).
- Must have clearly distinguishable visual features (buildings, sculptures, signs, traffic lights, etc.).
- Dynamic entities (pedestrians and vehicles) are excluded.

The final dataset contains 3 million landmark annotations, with an average of 65 landmarks per trajectory and an average description length of 17 words.

#### 3. **Policy Architecture**

The model takes four components as input:
1. Language instruction $g$ (encoded by CLIP with frozen parameters).
2. Current visual observation $o_t$ (features extracted by DINOv2 with frozen parameters).
3. Historical visual observations over the past $k=8$ steps $o_{(t-k):t}$.

**FiLM module**: Language features are used to modulate the current visual embedding, directing the agent's attention toward semantic cues relevant to the navigation goal. This is a critical component (ablations show that removing FiLM leads to significant performance degradation).

All tokens are concatenated and fed into a Transformer encoder, which outputs predicted waypoints for the next $k=8$ steps.

### Loss & Training

Four complementary loss terms:

$$L_{\text{total}} = \lambda_{\text{reg}} L_{\text{reg}} + \lambda_{\text{ori}} L_{\text{ori}} + \lambda_{\text{arr}} L_{\text{arr}} + \lambda_{\text{hall}} L_{\text{hall}}$$

1. **Waypoint regression loss** $L_{\text{reg}}$: L2 distance between predicted and ground-truth positions.
2. **Orientation loss** $L_{\text{ori}}$: Negative cosine similarity between predicted and ground-truth movement directions.
3. **Arrival prediction loss** $L_{\text{arr}}$: Binary cross-entropy for predicting whether the goal has been reached.
4. **Feature hallucination loss** $L_{\text{hall}}$: Encourages the model to internally model scene dynamics by predicting high-level visual features of future scenes.

$$L_{\text{hall}} = \frac{1}{k} \sum_{f=1}^{k} \|\hat{h}_{t+f} - h_{t+f}\|_1$$

During training, a trajectory and a landmark target are randomly selected; the starting point is randomly sampled from 10 to 60 frames before the target. Scenarios in which the target has already been reached are also simulated, helping the model learn when to stop.

## Key Experimental Results

### Main Results: Offline Evaluation

| Method | AOE↓ | MAOE↓ | ADE↓ | MADE↓ | Setting |
|---|---|---|---|---|---|
| Nomad + CLIP | 22.77 | 39.12 | 3.65 | 6.96 | Unseen Env. |
| ViNT + CLIP | 13.69 | 20.08 | 1.39 | 2.50 | Unseen Env. |
| LeLaN | 10.36 | 16.49 | 0.98 | 1.84 | Unseen Env. |
| **UrbanNav (Ours)** | **9.22** | **14.99** | **0.88** | **1.67** | Unseen Env. |

UrbanNav achieves state-of-the-art performance on all metrics, with consistent results in both seen and unseen environments.

### Real-World Deployment

| Method | Overall SR | Daytime | Nighttime |
|---|---|---|---|
| UrbanNav* (real data only) | 33.4% | 41.7% | 25.0% |
| Nomad + CLIP | 29.2% | 33.4% | 25.0% |
| ViNT + CLIP | 45.8% | 50.0% | 41.7% |
| LeLaN | 62.5% | 75.0% | 58.3% |
| **UrbanNav (Ours)** | **83.3%** | **91.7%** | **75.0%** |

UrbanNav's real-world success rate of 83.3% substantially outperforms the second-best method, LeLaN, at 62.5%. Even under nighttime conditions (where camera noise is greater), it maintains a success rate of 75.0%.

### Ablation Study

**Robustness analysis** (challenging scenarios):

| Scenario | UrbanNav* | Nomad+CLIP | ViNT+CLIP | LeLaN | **UrbanNav** |
|---|---|---|---|---|---|
| Normal | 62.5% | 50.0% | 62.5% | 75.0% | **100.0%** |
| Noisy instructions | 25.0% | 25.0% | 37.5% | 62.5% | **87.5%** |
| Occluded target | 12.5% | 12.5% | 25.0% | 37.5% | **62.5%** |

Under challenging conditions involving noisy instructions and occluded targets, UrbanNav maintains the highest success rate.

**Model component ablation** (unseen environments):

| Feature Hallucination | FiLM | AOE↓ | MAOE↓ | ADE↓ | MADE↓ |
|---|---|---|---|---|---|
| ✓ | ✗ | 11.35 | 17.54 | 1.07 | 1.94 |
| ✗ | ✓ | 9.56 | 15.51 | 0.92 | 1.71 |
| ✓ | ✓ | **9.22** | **14.99** | **0.88** | **1.67** |

The FiLM module is critical for performance gains (removing it raises AOE from 9.22 to 11.35), while the Feature Hallucination Loss provides additional improvement.

**Data scaling effect**:
- Scaling from 300 hours to 1,500 hours yields continuous reductions across all error metrics.
- Performance begins to plateau at approximately 1,200 hours.
- This confirms the effectiveness of large-scale web data and the scalability of the proposed framework.

### Key Findings

1. **Web-scale pretraining is essential**: UrbanNav* trained solely on real robot data achieves only 33.4% success rate; after pretraining, this rises to 83.3%, a 2.5× improvement.
2. **FiLM language–visual fusion is a core component**: Language-conditioned modulation of visual features enables the agent to focus on semantically relevant cues for goal-directed navigation.
3. **Data filtering is indispensable**: Unfiltered videos contain substantial viewpoint inconsistencies and dangerous behaviors that degrade policy performance.
4. **Feature Hallucination is beneficial on high-quality data**—in contrast to negative findings reported in prior work, this improvement is attributed to the cleaner nature of the filtered dataset.

## Highlights & Insights

1. **"Free lunch" data utilization**: YouTube walking videos serve as a free, abundant, and diverse source of navigation training data, breaking the bottleneck of data collection.
2. **First emphasis on robot compatibility filtering**: Not all human behaviors are appropriate for robot imitation; prior work overlooked the issues of viewpoint deviation and dangerous behaviors.
3. **Scale effects of 3 million landmark annotations**: Large-scale instruction–trajectory–landmark triplets enable policy generalization across diverse cities and scenes.
4. **Closed-loop validation from YouTube to real robots**: Beyond offline evaluation, the method is deployed and validated on a physical robot, achieving a 91.7% daytime success rate.
5. **Robustness by design**: The policy makes decisions based solely on the most recent 8 frames, mitigating long-term cumulative drift from visual odometry.

## Limitations & Future Work

1. **Short-range local navigation**: The success rate drops to 62.5% when the target is occluded, indicating limited capability for scenarios requiring long-range exploration.
2. **Nighttime performance degradation**: Success rate decreases from 91.7% to 75.0%, constrained by RGB camera limitations under low-light conditions.
3. **Cumulative drift in DPVO visual odometry**: Although the short temporal window alleviates this issue, long-range trajectories may still be affected.
4. **Restricted to walking viewpoints**: Adaptation may be required for robots with different heights (e.g., delivery drones) or form factors.
5. **Landmark stability**: While pedestrians and vehicles are excluded, certain ostensibly "stable" landmarks (e.g., temporary signage) may also change over time.

## Related Work & Insights

- **Comparison with LeLaN**: LeLaN focuses on indoor short-range object navigation, whereas UrbanNav extends to complex urban environments and introduces viewpoint filtering and safe-behavior filtering.
- **Relationship with NoMaD/ViNT**: Originally designed for image-goal navigation, these methods are adapted to language-guided navigation in this work via CLIP text encoding.
- **Data flywheel effect**: Once the automated annotation pipeline is established, new videos can be continuously sourced from YouTube to train progressively stronger models.
- **Inspiration for future work**: Similar approaches could be extended to UAV navigation (leveraging aerial footage), underwater navigation, and other novel domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using web videos for navigation training is not entirely new (LeLaN provides a precedent), but the filtering pipeline and urban-scale scope represent important contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers offline evaluation, real-robot deployment, robustness testing, component ablations, and data scaling ablations—highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clear, problem formulation is precise, and the paper is well-illustrated.
- **Value**: ⭐⭐⭐⭐⭐ — Significant practical value for real-world applications such as last-mile delivery robots; data and code are publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mango: Multi-Agent Web Navigation via Global-View Optimization](../../ACL2026/robotics/mango_multi-agent_web_navigation_via_global-view_optimization.md)
- [\[AAAI 2026\] Realistic Synthetic Household Data Generation at Scale](realistic_synthetic_household_data_generation_at_scale.md)
- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)
- [\[ICLR 2026\] Attribution-Guided Decoding](../../ICLR2026/robotics/attribution-guided_decoding.md)
- [\[AAAI 2026\] Affordance-Guided Coarse-to-Fine Exploration for Base Placement in Open-Vocabulary Mobile Manipulation](affordance-guided_coarse-to-fine_exploration_for_base_placem.md)

</div>

<!-- RELATED:END -->
