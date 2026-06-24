---
title: >-
  [Paper Note] Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach
description: >-
  [CVPR 2025][Robotics][end-to-end navigation] Through large-scale experiments on 262 real-world robot navigation episodes, this work deeply analyzes the emergent reasoning capabilities inside end-to-end RL-trained navigation agents, including a Kalman-filter-like dynamical model, latent memory of scene structures, a finite horizon of planning ability, and value functions associated with long-term planning.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "end-to-end navigation"
  - "dynamical systems"
  - "Kalman filter"
  - "latent memory"
  - "embodied AI"
  - "sim-to-real"
date: 2026-05-08
content_hash: dbc287fbf6c8a05c
---

# Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach

**Conference**: CVPR 2025  
**arXiv**: [2503.08306](https://arxiv.org/abs/2503.08306)  
**Code**: [Project Page](https://visual-navigation-reasoning.github.io)  
**Area**: Temporal Analysis/Embodied AI  
**Keywords**: end-to-end navigation, dynamical systems, Kalman filter, latent memory, embodied AI, sim-to-real

## TL;DR

Through large-scale experiments on 262 real-world robot navigation episodes, this work deeply analyzes the emergent reasoning capabilities inside end-to-end RL-trained navigation agents, including a Kalman-filter-like dynamical model, latent memory of scene structures, a finite horizon of planning ability, and value functions associated with long-term planning.

## Background & Motivation

### Background

**Background**: The field of embodied AI has achieved high-level reasoning and zero-shot navigation of end-to-end trained agents in photorealistic environments, but existing evaluations remain predominantly simulation-based and lack fine-grained behavioral analysis on real, fast-moving robots. Traditional methods decompose navigation into a pipeline of perception, mapping, localization, planning, and control, whereas end-to-end methods map directly from inputs to actions.

**Limitations of Prior Work**: (1) The internal "black-box" mechanism of end-to-end navigation policies remains unclear—what exactly has been learned? (2) The stepwise teleportation motion model in simulations leads to a severe sim-to-real gap, whereas real robots run slower and rely on low-level controllers; (3) There is a lack of systematic analysis regarding the emergent capabilities of end-to-end policies on real robots.

**Key Challenge**: Although end-to-end trained navigation agents exhibit robust navigation capabilities in real environments, researchers do not understand which reasoning patterns have emerged internally—such as whether a dynamical model exists, if there is planning capability, or what the memory encodes.

**Goal**: To systematically analyze the types of emergent reasoning capabilities of end-to-end RL-trained navigation agents in real environments, understanding their internal working mechanisms from a dynamical systems perspective.

**Key Insight**: By augmenting the simulator with real physical motion models (second-order dynamics), and then systematically analyzing the agent's internals over 262 real episodes through probing, ablation, and Shapley analyses.

**Core Idea**: Navigation agents trained end-to-end under real motion models spontaneously learn a Kalman-filter-like prediction-correction mechanism, latent memory containing spatial occupancy grids, and value estimations linked to long-term planning.

## Method

### Overall Architecture

The agent architecture is based on a GRU recurrent network, receiving RGB images (encoded by ResNet-18), Lidar-like scans (encoded by 1D-CNN), goal polar coordinates, odometry, and AMCL localization as inputs. The action space consists of 28 discrete velocity pairs (linear velocity $\times$ angular velocity). A key innovation lies in integrating the second-order dynamics model of real robots into the Habitat simulator, achieving a 97.6% real-world success rate (vs. 27.6% without the dynamics model). Analytical methods including probing, ablation, and Shapley analysis are deployed to systematically dissect the agent's internals over 262 real episodes.

### Key Designs

1. **Kalman-filter-like Prediction-Correction Mechanism**: Through a newly proposed "distance to belief" metric, the impact of dynamical parameter perturbations and sensor noise on agent performance is systematically compared. Experiments reveal that both have a substantial impact—the agent utilizes its internal dynamical model for open-loop prediction, and subsequently corrects states using odometry perception, forming a closed loop akin to Kalman filtering. Drawing inspiration from RMA to train policies with environmental parameter embeddings restores performance, further validating the importance of dynamical modeling.

2. **Scene Structure Probing from Latent Memory**: A probing network is trained to reconstruct $3 \times 3\text{m}$ local occupancy maps from the hidden state $h_t$. When trained on the HM3D training set, it accurately predicts scene structures in real buildings. Overlaying the probing results of 14 episodes onto real maps shows excellent alignment, even in challenging areas such as doors and transparent walls. Ablation experiments involving zeroing out the hidden state $h_t$ at equal intervals indicate that more frequent memory clearance leads to severe performance drops (clearing every 3 seconds drops the SR from 100% to 75%).

3. **Evidence of Emergent Short-to-Medium-Term Planning**: Linearly probing future poses $p_{t+\tau}$ from $h_t$ yields an average error of only 0.76m over a 6-second timespan, indicating that short-to-medium-term planning information is encoded in the hidden states. Post-hoc analysis of the PPO value function shows that the agent's value estimation exhibits significant discontinuity during policy transitions—value estimates drop when a path is abandoned and jump when a superior path is discovered, indicating that long-term planning manifests through the value function.

## Key Experimental Results

### Key Findings

- D28-dynamics, trained with the physical dynamics model, achieves a **92.5% SR** in real-world environments (100% when speed is limited to 0.7m/s), whereas D28-instant (without physical dynamics) achieves only 27.6%, and the traditional 4-action baseline achieves only 10.0%.
- Shapley analysis reveals that the agent **relies most heavily on odometry and scan sensors**, while RGB and localization inputs contribute relatively less.
- Average position error of pose probing for a 20-step (~6 seconds) future prediction is **0.767m** (linear model), which decreases to **0.441m** when action information is incorporated.
- Memory ablation: clearing the hidden state every 3 seconds drops the SR from 100% to 75%; with no memory at all, the policy SR is only 40%.
- Replacing AMCL with visual localization degrades the SR from 100% to 42.9%, demonstrating that precise localization is critical for the final meters of PointNav tasks.
- RGB data augmentation during testing alone contributes a **~15% gain in real-world SR**.
- Discovered the "tunnel vision" phenomenon: the agent lacks human-level high-level geometric reasoning and attempts obviously unfeasible paths, suggesting the potential value of geometric foundation models as visual encoders.

## Highlights & Insights

- This is the first large-scale study to systematically analyze the internal mechanisms of end-to-end navigation policies over 262 real-world robot episodes.
- The "Distance to belief" metric ingeniously addresses the incomparability of different physical parameter perturbations by mapping heterogeneous disturbances onto a unified spatial distance.
- This work validates a long-standing hypothesis: end-to-end policies indeed learn a Kalman-filter-like prediction-correction pattern, which has long been considered "common sense" but never formally verified.
- The discovery of the "tunnel vision" phenomenon highlights that the agent lacks human-level high-level geometric reasoning and attempts obviously unfeasible paths, suggesting the potential value of geometric foundation models as visual encoders.
- Integrating the second-order dynamics of real robots into the simulator is the key to sim-to-real success (boosting SR from 27.6% to 92.5%).

## Limitations & Future Work

- Validation is confined to PointNav tasks; whether it generalizes to more complex tasks such as ObjectNav or language-guided navigation remains unclear.
- Long-term planning capabilities are still insufficient—the "tunnel vision" issue persists, where the agent fails to perform high-level geometric reasoning like humans.
- Performance drops heavily when replacing AMCL with visual localization, suggesting that precise localization capabilities do not fully emerge from RL training.
- The analytical methods are inherently post-hoc probing and ablation, which cannot fully establish causal relationships.
- Real-world testing is limited to a single building environment; cross-scene generalization remains unverified.
- Despite its elegance, the "Distance to belief" metric relies on linear assumptions of perturbation responses, which may fail in highly non-linear policies.
- The scale of 262 real episodes still limits the confidence level of the statistical conclusions.
- The action space design (28 discrete velocity pairs) may limit the behavioral flexibility of the agent.
- Continuous action spaces are worth exploring.
- The potential of integrating foundation models (such as VC-1, R3M) remains unexplored.
- The capacity of GRU hidden states may limit the completeness of scene memory.
- Larger hidden states or Transformer-based architectures are worth exploring.
- Probing methods only reveal correlations and cannot prove causality.
- Generalization in multi-story buildings or open spaces has not been validated.
- Modeling of sensor noise may not be fully comprehensive, as only limited types of disturbances were considered.
- Future work could incorporate semantic or topological maps to enhance the agent's high-level planning capabilities.

## Related Papers

- [\[CVPR 2025\] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](../../NeurIPS2025/robotics/suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/robotics/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[CVPR 2026\] End-to-End Language-Action Model for Humanoid Whole Body Control](../../CVPR2026/robotics/end-to-end_language-action_model_for_humanoid_whole_body_control.md)
- [\[CVPR 2026\] RoboTAG: End-to-end Robot Pose Estimation via Topological Alignment Graph](../../CVPR2026/robotics/robotag_end-to-end_robot_pose_estimation_via_topological_alignment_graph.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](../../NeurIPS2025/robotics/suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/robotics/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[ICLR 2026\] End-to-end Listen, Look, Speak and Act](../../ICLR2026/robotics/end-to-end_listen_look_speak_and_act.md)
- [\[ICLR 2026\] RAVEN: End-to-end Equivariant Robot Learning with RGB Cameras](../../ICLR2026/robotics/raven_end-to-end_equivariant_robot_learning_with_rgb_cameras.md)

</div>

<!-- RELATED:END -->
