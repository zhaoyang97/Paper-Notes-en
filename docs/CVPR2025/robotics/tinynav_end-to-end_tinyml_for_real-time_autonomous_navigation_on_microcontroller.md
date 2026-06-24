---
title: >-
  [Paper Note] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers
description: >-
  [CVPR2025][Robotics][TinyML] Deploying an end-to-end quantized CNN on an ESP32 microcontroller to achieve real-time autonomous navigation with a 30ms latency using only 23k parameters and a ToF depth camera.
tags:
  - "CVPR2025"
  - "Robotics"
  - "TinyML"
  - "Autonomous Navigation"
  - "Microcontroller"
  - "ESP32"
  - "End-to-End Learning"
  - "Model Quantization"
  - "Edge Inference"
date: 2026-05-08
content_hash: f4393e9ad215a1ab
---

# TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers

**Conference**: CVPR2025  
**arXiv**: [2603.11071](https://arxiv.org/abs/2603.11071)  
**Code**: [regularpooria/TinyNav](https://github.com/regularpooria/TinyNav)  
**Area**: Robotics  
**Keywords**: TinyML, Autonomous Navigation, Microcontroller, ESP32, End-to-End Learning, Model Quantization, Edge Inference

## TL;DR

Deploying an end-to-end quantized CNN on an ESP32 microcontroller to achieve real-time autonomous navigation with a 30ms latency using only 23k parameters and a ToF depth camera.

## Background & Motivation

- Autonomous navigation for existing low-cost robots typically relies on high-power processors (such as single-board computers running SLAM), which are costly and energy-intensive.
- Although publications in the TinyML field grew at an average annual rate of 59.23% from 2020 to 2024, its deployment for autonomous navigation tasks remains constrained by the computational and memory bottlenecks of microcontrollers.
- Microcontrollers (MCUs) are inexpensive and energy-efficient but strictly limit model complexity: they lack support for LSTM/GRU/Multi-Head Attention, and memory is limited to the 32MB scale.
- The authors aim to demonstrate that responsive autonomous obstacle avoidance control can be achieved solely on a ~$20 ESP32 platform, without relying on external computing resources.

## Core Problem

How to design and deploy an end-to-end navigation model capable of directly predicting driving commands from depth data under the tight constraints of an ESP32 microcontroller (limited memory, lack of RNN/Attention support, and required inference latency of <50ms)?

## Method

### Hardware Platform

- **Microcontroller**: ESP32-P4-WIFI6-M, dual-core 360MHz, 32MB PSRAM, 32MB Flash, approximately $20
- **Depth Sensor**: Sipeed MaixSense A010 ToF camera, 100x100 raw resolution, 20 FPS, 940nm infrared, effective range 200-2500mm
- **Robot**: Tank-style drive to reduce control variable complexity, supporting in-place rotation

### Data Collection & Processing

- The ToF camera captures depth frames at 20 FPS, pairing each frame with the corresponding steering and throttle commands to form a labeled dataset.
- 4x4 binning on the sensor side reduces the resolution from 100x100 to 25x25, which is then resized to 24x24 (for ease of convolutional alignment).
- **Temporal Encoding**: Concatenates 20 consecutive frames along the channel dimension to form a 20-channel input tensor (approx. 1 second of visual history), substituting LSTM for temporal modeling.
- Data augmentation: Horizontal flipping to reduce directional bias; shuffling of multiple track layouts to prevent sequence bias; 60/40 train/test split.
- Training data is collected across various track geometries (narrow passages, sharp turns, different floor materials) to enhance generalization.

### Model Architecture

- **2D CNN** + multi-output head architecture:
    - Input: 24×24×20 (width × height × 20-frame channels)
    - Convolutional layers extract spatial features (avoiding 3D convolutions to meet latency requirements)
    - A shared fully connected layer captures the coupling relationship between steering and throttle
    - Two output heads: steering (-1 to 1) and throttle (0 to 1)
- Under **23k parameters**, with an inference latency of **30ms**
- Sliding window mechanism: new frames enqueue and old frames dequeue during inference, enabling parallel execution of the control loop and inference via dual-core shared memory.

### Model Compression & Deployment

- TFLite post-training quantization: FP32 → INT8
- The entire training set is used as a representative dataset to calibrate quantization parameters.
- Quantization retains 99.84% of steering accuracy and 99.79% of throttle accuracy, representing negligible precision loss.
- Leverages the hardware acceleration cores of the ESP-NN library to achieve up to a 7× speedup.

### Parallel Inference Strategy

- ESP32 dual-core task distribution: one core executes the control loop while the other handles model inference.
- Frame data is exchanged via shared memory to prevent control latency from being affected by inference latency.

## Key Experimental Results

| Metric | Value |
|------|------|
| Model Parameters | 23k |
| Inference Latency | 30ms |
| Quantization Accuracy Retention (Steering) | 99.84% |
| Quantization Accuracy Retention (Throttle) | 99.79% |
| Steering/Throttle Correlation Coefficient | ~0.6 (Pearson & Spearman) |
| Continuous Loops on Simple Track | 40 loops without collision |
| Hardware Cost | ~$20 |

- **Grad-CAM Analysis**: The steering head focuses on the upper regions of the frame (track boundaries) and the closest walls; the throttle head focuses on the corners of the frame (passage openings) and dead ends ahead.
- **Output Distribution Matching**: The predicted distribution overlaps well with the true distribution, without regression-to-the-mean issues.
- Full routes can also be completed on complex new layouts, though progress is less consistent than on layouts similar to the training set, with occasional minor contact with walls.

## Highlights & Insights

1. **Feasibility Verification under Extreme Resource Constraints**: 23k parameters + 30ms latency, achieving real-time navigation on a $20 MCU, showing significant engineering value.
2. **Clever Temporal Encoding**: Substituting RNNs (LSTM/GRU) with 20 concatenated frame channels bypasses the limitation of TFLite Micro not supporting recurrent layers.
3. **Dual-core Parallel Design**: Decoupling the control loop from inference ensures that occasional inference delays do not affect motor control.
4. **Complete Reproducible Solution**: Code, firmware, dataset, and hardware BOM are all open-source, facilitating reproduction and extension.
5. **Extremely Low Quantization Loss**: INT8 quantization incurs almost zero accuracy loss (>99.7%), demonstrating the efficacy of quantizing small models.

## Limitations & Future Work

- **The correlation coefficient between steering and throttle is only ~0.6**, indicating room for improvement in model control precision.
- **Insufficient Dataset Diversity**: Track structures are highly similar (fixed wall heights, road widths) and lack obstacles of varying heights or shapes, making generalization to real-life environments challenging.
- **Forward-Only Support**: Backward navigation is not supported; backward dynamics are more complex and require richer datasets.
- **No Odometer/Encoder Feedback**: Relying entirely on visual depth without closed-loop motion state estimation makes state estimation unreliable in low-texture environments.
- **Connection Resource Conflicts**: ESP32 peripherals (SD card vs WiFi) share the same bus, which limits system scalability.
- The upper limit of model parameters is about 50k (to maintain 20 FPS), preventing the deployment of deeper architectures.

## Related Work & Insights

| Aspect | TinyNav | Traditional MCU Navigation | Single-board Computer (SBC) Approach |
|------|---------|--------------|---------------|
| Hardware Cost | ~$20 | ~$10-30 | >$50 |
| Power Consumption | Extremely Low | Low | Mid-to-High |
| Model Complexity | 23k parameter CNN | Rule-based algorithms | Millions of parameters |
| Sensor | ToF Depth Camera | Ultrasonic/Infrared | RGB/LiDAR |
| Temporal Modeling | Channel Concatenation | None | LSTM/Transformer |
| Inference Latency | 30ms | N/A | 50-200ms |

- Similar to previous TinyML speech recognition work [3], the importance of sensor hardware consistency is emphasized (the same sensor must be used for training and deployment).
- More complex than traditional TinyML architectures (e.g., Edge Impulse 1D CNN), as it handles a 2D spatial + temporal navigation task.

## Related Work & Insights

- **The concept of multiplexing channels for temporal encoding** can be generalized to other scenarios requiring temporal awareness where recurrent layers are unsupported by the target deployment environment.
- This approach could be combined with wheel encoders for closed-loop control to potentially improve performance in complex environments.
- If future ESP32 iterations support larger models (e.g., through NPU acceleration), deeper architectures or multi-sensor fusion could be explored.
- Dataset diversity remains the primary bottleneck for improving generalization; sim-to-real transfer or domain randomization is highly worth investigating.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core techniques (CNN + quantization) are not novel, but the system integration on an MCU has practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Grad-CAM and correlation analyses are present, but quantitative evaluations are limited, and comparison with other methods is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, detailed description of methods, with a particularly substantial hardware engineering section.
- Value: ⭐⭐⭐⭐ — Serves as a useful reference for the TinyML robotics community, though generalization capabilities and control precision still need improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](../../NeurIPS2025/robotics/suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[CVPR 2025\] Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/robotics/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[ICLR 2026\] End-to-end Listen, Look, Speak and Act](../../ICLR2026/robotics/end-to-end_listen_look_speak_and_act.md)
- [\[CVPR 2025\] SortScrews: A Dataset and Baseline for Real-time Screw Classification](sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)

</div>

<!-- RELATED:END -->
