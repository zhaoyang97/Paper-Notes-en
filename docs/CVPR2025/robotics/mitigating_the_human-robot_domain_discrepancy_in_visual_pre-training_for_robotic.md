---
title: >-
  [Paper Note] Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation
description: >-
  [CVPR 2025][Robotics][robotic manipulation] This paper proposes the HR-Align adaptation paradigm, which leverages paired human-robot video data and a contrastive alignment loss to bridge the semantic discrepancy between models pre-trained on human data and the robot domain in a parameter-efficient manner. It improves the average success rate by 7%+ across 20 simulation tasks and 5 real-world tasks.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "robotic manipulation"
  - "visual pre-training"
  - "human-robot domain gap"
  - "contrastive alignment"
  - "parameter-efficient adapter"
date: 2026-05-08
content_hash: e27b8a04c21390e7
---

# Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2406.14235](https://arxiv.org/abs/2406.14235)  
**Code**: [Project Page](https://jiaming-zhou.github.io/projects/HumanRobotAlign)  
**Area**: Robotics  
**Keywords**: robotic manipulation, visual pre-training, human-robot domain gap, contrastive alignment, parameter-efficient adapter

## TL;DR

This paper proposes the HR-Align adaptation paradigm, which leverages paired human-robot video data and a contrastive alignment loss to bridge the semantic discrepancy between models pre-trained on human data and the robot domain in a parameter-efficient manner. It improves the average success rate by 7%+ across 20 simulation tasks and 5 real-world tasks.

## Background & Motivation

**Background**: Visual representation learning in the field of robotic manipulation faces severe data scarcity. Existing approaches leverage large-scale human activity datasets (such as Ego4D and Kinetics) to pre-train visual models, which are then used as frozen visual backbones for downstream robot policy learning.

**Key Challenge**: There are significant morphological differences between humans and robots, leading to the "human-robot domain gap" issue—representations learned by pre-trained models on human data struggle to transfer effectively to the robot domain.

**Limitations of Prior Work**:

1. **Manipulation-oriented pretext tasks** (e.g., hand detection): These methods define pretext tasks on human data to indirectly adapt pre-trained models, but they lack explicit exposure to robot data and fail to directly mitigate the domain gap.
2. **Downstream fine-tuning** (fine-tuning in each downstream environment): This requires customizing pre-trained models for each different environment, which sacrifices the generality of the models.

**Key Insight**: This paper proposes a new "adaptation paradigm" that leverages existing paired human-robot demonstration datasets (such as the RH20T dataset) as a bridge to mitigate the domain gap while maintaining model generality. The core insight is that the dynamic semantics of human and robot demonstrations in paired data are aligned, and this alignment can be used to guide the adaptation.

## Method

### Overall Architecture

HR-Align (Human-Robot Semantic Alignment) adopts a three-stream architecture:

- **Frozen Human Stream**: The frozen pre-trained model $\mathcal{F}$ extracts human video features $h^f$.
- **Frozen Robot Stream**: The same frozen model extracts robot video features $r^f$ (unadapted, serving as a negative reference).
- **Adapted Robot Stream**: Learnable adapter modules are injected into the pre-trained model to extract adapted robot video features $r^t$.

After aggregating the three-stream features via task-aware attention, the adapter parameters are trained using a contrastive alignment loss.

### Key Designs

#### 1. Parameter-Efficient Adapter Module

Lightweight adapters are inserted into the intermediate layers of the pre-trained model, using a residual structure to perform feature adaptation:

$$r^{t,next} = r^{f,inter} + \text{Conv}_{up}(g(\text{Conv}_{down}(r^{f,inter})))$$

- $\text{Conv}_{down}$: Channel dimension reduction convolution.
- $g$: Activation function.
- $\text{Conv}_{up}$: Channel dimension expansion convolution.
- Only the adapter parameters are trained while keeping the pre-trained backbone frozen, achieving parameter-efficient adaptation.

#### 2. Task-Aware Feature Modeling

Task description text is introduced as a query, and task-relevant semantics are extracted from video spatiotemporal features through an attention mechanism:

- A frozen DistilBert is used to encode the task description $L$, obtaining the query $l$.
- Attention weights $\mathcal{A}^r = \text{softmax}(r^t \cdot l)$ are computed for the video features of each stream.
- Task-aware features are obtained through weighted aggregation: $\bar{r}^t = (r^t)^T \cdot \mathcal{A}^t$.

#### 3. Human-Robot Contrastive Alignment Loss

A bidirectional contrastive loss is designed to constrain the adaptation process, incorporating two core principles:

**Principle 1**: For paired human-robot videos, the adapted robot feature $\bar{r}_i^t$ should be more consistent with the human feature $\bar{h}_i^f$ than the unadapted one $\bar{r}_i^f$.

**Principle 2**: Paired human-robot features should be more similar than unpaired ones within a batch (the standard contrastive learning paradigm).

$$\mathcal{L} = \frac{1}{2M}\sum_{i=1}^{M} -\log\frac{\mathcal{S}(\bar{h}_i^f, \bar{r}_i^t)}{\mathcal{S}(\bar{h}_i^f, \bar{r}_i^t) + \mathcal{S}(\bar{h}_i^f, \bar{r}_i^f) + \sum_{j \neq i}\mathcal{S}(\bar{h}_i^f, \bar{r}_j^t)} + \text{对称项}$$

where $\mathcal{S}(x,y) = \exp(x^T y / \tau)$, $\tau=0.1$. The unique aspect of this loss is that the "unadapted robot feature" $\bar{r}_i^f$ is also included in the denominator as a negative sample, directly penalizing the domain gap.

### Loss & Training

The total loss only consists of the aforementioned human-robot contrastive alignment loss. During training, the adapter parameters and the task-aware linear layer parameters are optimized.

**Training Configuration**: Adam optimizer, lr=$1 \times 10^{-4}$, batch size=200, approximately 8k steps, 4×NVIDIA A6000.

## Key Experimental Results

### Main Results

| Setting | Model | Baseline | +HR-Align | Gain |
|------|------|------|-----------|------|
| Adroit Single-task (2 tasks) | D4R | 63.0% | 65.0% | +2.0% |
| Adroit Single-task (2 tasks) | R3M | 74.0% | 81.3% | **+7.3%** |
| RLBench Multi-task (18 tasks) | D4R | 55.3% | 59.9% | +4.6% |
| RLBench Multi-task (18 tasks) | R3M | 50.3% | 59.2% | **+8.9%** |
| Real-world (5 tasks) | D4R | — | — | **+13%** |
| Real-world (5 tasks) | R3M | — | — | **+11%** |

### Ablation Study

| Method | learned params | pen | relocate | Avg |
|------|---------------|-----|----------|-----|
| R3M (Frozen) | 0M | 78.0 | 70.0 | 74.0 |
| R3M-PreT (Continued Human Pre-training) | 25M | 78.0 | 77.3 | 77.7 |
| R3M-ClS (Action Classification Fine-tuning) | 25M | — | — | Worse |
| R3M-Align (Ours) | Small | 81.3 | 81.3 | **81.3** |

### Key Findings

1. **Consistently Effective Across Models**: Significant improvements are achieved on both R3M and D4R, two models with completely different pre-training methods, validating the generality of the proposed method.
2. **Greater Improvement in Multi-task Settings**: R3M achieves an 8.9% improvement across 18 multi-task setups in RLBench, indicating that the adapted model generalizes better on diverse tasks.
3. **Significant Improvement in Real-world Environments**: Success rates improve by 11-13% on real-world tasks, far exceeding simulation environments, indicating that the domain gap is more severe in real-world scenarios.
4. **Parameter-efficient**: Significant improvements are achieved by training only a small number of adapter parameters, without requiring full-parameter fine-tuning.

## Highlights & Insights

1. **Paradigm Innovation**: This work is the first to propose an adaptation paradigm using "paired data bridging", which sits between frozen utilization and downstream fine-tuning, balancing both generality and domain adaptation.
2. **Ingenious Negative Sample Design**: The unadapted robot features are incorporated as additional negative samples in contrastive learning, directly quantifying and penalizing the domain gap.
3. **Low Cost, High Return**: Adaptation is completed using only 56k paired videos from an existing community dataset (RH20T), without requiring any additional data collection.
4. **Non-intrusive to Downstream Tasks**: The adapted model serves as a general visual backbone, eliminating the need for customization for each downstream environment.

## Limitations & Future Work

1. **Dependency on Paired Demonstration Data**: The method requires paired videos of humans and robots performing the same tasks. Although public datasets exist, data availability remains limited.
2. **Visual Discrepancy Between Adaptation and Downstream Environments**: Robot demonstration data in the adaptation stage differs in visual appearance from downstream robot environments, relying solely on "isomorphic robot morphology" to close the gap.
3. **Adapter Position Limited to the Last Layer**: The study only validates inserting the adapter in the last layer, without fully exploring the effects of inserting adapters at different layer depths.
4. **Image Resolution and Frame Constraints**: The resolution and frame count used in the experiments are relatively low (5 frames), which may limit temporal modeling capabilities.

## Related Work & Insights

- **R3M, MVP, data4robotics**: Three major human-data pre-training baselines. This work performs domain adaptation on top of them.
- **RH20T Dataset**: Provides high-quality paired human-robot demonstration data, making the proposed method possible.
- **Parameter-Efficient Fine-Tuning (PEFT)**: The adapter design draws inspiration from the PEFT methodology in NLP and CV.
- **Insight**: The "domain gap" is a neglected yet critical challenge in embodied AI. Using paired data as a bridge is a general strategy that can be extended to transfer learning across different embodiments.

## Rating ⭐

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robotic Visual Instruction](robotic_visual_instruction.md)
- [\[CVPR 2025\] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)

</div>

<!-- RELATED:END -->
