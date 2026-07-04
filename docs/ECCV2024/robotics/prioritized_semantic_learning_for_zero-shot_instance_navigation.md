---
title: >-
  [Paper Note] Prioritized Semantic Learning for Zero-shot Instance Navigation
description: >-
  [ECCV 2024][Robotics][Zero-shot navigation] This paper proposes the Prioritized Semantic Learning (PSL) method. Through a semantic-augmented agent architecture, a prioritized semantic training strategy, and a semantic expansion inference scheme, it significantly improves the agent's semantic perception capabilities in zero-shot object/instance navigation, achieving SOTA performance on both ObjectNav and the newly proposed InstanceNav tasks.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Zero-shot navigation"
  - "instance navigation"
  - "semantic learning"
  - "CLIP"
  - "embodied AI"
date: 2026-05-08
content_hash: 159e7c705f7732f6
---

# Prioritized Semantic Learning for Zero-shot Instance Navigation

**Conference**: ECCV 2024  
**arXiv**: [2403.11650](https://arxiv.org/abs/2403.11650)  
**Code**: [https://github.com/XinyuSun/PSL-InstanceNav](https://github.com/XinyuSun/PSL-InstanceNav)  
**Area**: Robotics (Embodied Navigation)  
**Keywords**: Zero-shot navigation, instance navigation, semantic learning, CLIP, embodied AI

## TL;DR

This paper proposes the Prioritized Semantic Learning (PSL) method. Through a semantic-augmented agent architecture, a prioritized semantic training strategy, and a semantic expansion inference scheme, it significantly improves the agent's semantic perception capabilities in zero-shot object/instance navigation, achieving SOTA performance on both ObjectNav and the newly proposed InstanceNav tasks.

## Background & Motivation

Zero-shot Object Navigation (ZSON) requires agents to generalize to target navigation tasks without scene object annotations during training. These agents pre-train only on the Image Goal Navigation (ImageNav) task, subsequently calling upon vision-language models like CLIP to solve the target tasks. This is a crucial pathway for building general embodied agents. However, existing ZSON tasks only require the agent to find *any* object of a given category, which is far from the practical need to pinpoint a *specific instance* in real-world applications.

**Key Findings - The Semantic Neglect Problem**: The paper reveals an overlooked yet critical problem through carefully designed pilot experiments:

- **Semantic-Non-dominant (SN) Agent**: Operating with a Canny edge detector (which destroys semantic information) and a learnable ResNet50, this agent surprisingly achieves success rates comparable to ZSON on the ImageNav task.
- **Semantic-Dominant (SD) Agent**: Utilizing two frozen CLIP encoders to acquire semantic information, this agent yields the worst performance.

This counterintuitive result demonstrates that **the ImageNav pre-training task does not force the agent to learn semantic information**. The agent can achieve a high success rate through view matching using only layout/contour cues. Consequently, the semantic perception of the ZSON agent remains weak, restricting its performance on navigation tasks that rely heavily on semantic cues.

**Key Challenge**: The mismatch between the ImageNav pre-training objective and downstream semantic navigation goals — semantics are not required to complete tasks during training, yet zero-shot migration must rely heavily on semantics.

**Key Insight**: Simultaneously strengthen semantic learning across three levels: the agent's architecture, training strategy, and inference scheme, ensuring the agent establishes strong semantic perception capabilities during pre-training.

**Core Idea**: Comprehensively enhance the agent's semantic understanding capability by selecting goal images with clear semantics, relaxing reward constraints for precise view matching, adding a semantic perception module, and expanding text queries with image features.

## Method

### Overall Architecture

The PSL method consists of three components: (1) PSL Agent architecture — introducing a CLIP semantic observation encoder and a Semantic Perception Module (SPM); (2) Prioritized Semantic Training strategy — entropy minimization goal view selection + viewpoint reward relaxation; (3) Semantic Expansion Inference scheme — retrieving and enriching text queries with image features. These three components work synergistically to improve semantic understanding across model capability, training signals, and inference granularity.

### Key Designs

1. **Semantic Perception Module (SPM)**:

   Function: Encodes the semantic correspondence between the goal image and the agent's observations.

   Mechanism: In addition to the baseline ZSON, a frozen CLIP encoder is introduced to extract semantic-level observations $\mathbf{z}_S$. An MLP bottleneck layer then compresses the goal embedding $\mathbf{z}_G$ and semantic observation $\mathbf{z}_S$ into a low-dimensional semantic perception embedding $\mathbf{z}_{SP} \in \mathbb{R}^{C_2}$ ($C_2 < 2 \times C_1$). The policy network makes decisions based on the semantic perception and observation embeddings:

    $\mathbf{s}_t, \mathbf{h}_t = \pi_\theta(\mathbf{z}_{SP} \oplus \mathbf{z}_O \oplus \mathbf{a}_{t-1} | \mathbf{h}_{t-1})$

   An actor-critic network is trained using PPO to predict 6 actions (move forward, turn left, turn right, stop, look up, look down).

   Design Motivation: The original ZSON might fail to effectively learn semantic information relying solely on learnable observation encoders. By explicitly incorporating a semantic-aware channel and using bottleneck compression, the agent is forced to focus on key semantic correspondences.

2. **Prioritized Semantic Training**:

   Function: Resolves the issue of ambiguous semantics in goal images within the ImageNav training data.

   **Entropy Minimization Goal View Selection**: For each goal position, render images from different viewpoints by rotating $\Omega$ times. CLIP is utilized to calculate the classification entropy across 6 object categories, and the view with the minimum entropy (i.e., the image with a clear target object) is selected as the goal:

    $\omega^* = \arg\min_{\omega \in \Omega} -\frac{1}{\log(|\mathcal{C}|)}\sum_{c \in \mathcal{C}} \mathbf{p}_c \log \mathbf{p}_c$

   where $\mathbf{p}_c = \text{softmax}(\tau \cdot \frac{\mathbf{v}_\omega^T \mathbf{q}_c}{\|\mathbf{v}_\omega\|_2\|\mathbf{q}_c\|_2})$.

   **Viewpoint Reward Relaxation**: After rendering and selecting additional images from multiple pitch and yaw angles, the PPO reward function is modified to only encourage the agent to face towards the target on the x-z plane, ignoring pitch angle matching:

    $R_t^{PSL} = \underbrace{\gamma^{suc}\mathbb{1}\{d_t < \epsilon^d\}}_{\text{position reached}} + \underbrace{\gamma^{suc}\mathbb{1}\{d_t<\epsilon^d\}\mathbb{1}\{\text{extract}_Y(\mathbf{a}_t)<\epsilon^a\}}_{\text{orientation match}} + r_d + r_a - \gamma^{delay}$

   Design Motivation: A large number of goal images in the raw dataset contain meaningless scenes like walls or empty rooms. These ambiguous goals exacerbate the semantic neglect problem. Relaxing the reward enables the agent to focus on semantic correspondence rather than precise geometric matching.

3. **Semantic Expansion Inference**:

   Function: Mitigates the modality/granularity gap between image embeddings used in training and text embeddings used in testing.

   Mechanism: During training, a support set $\mathcal{V}$ is maintained (containing approximately 0.1M diverse image embeddings, with pairwise similarity below a threshold $\lambda=0.8$). During inference, given a text description to generate $\mathbf{z}_T$, a goal embedding is generated via weighted retrieval:

    $\mathbf{z}_R = \sum_{\mathbf{v}_i \in \mathcal{V}} \frac{\exp(g(\mathbf{z}_T, \mathbf{v}_i))}{\sum_{\mathbf{v}_j \in \mathcal{V}} \exp(g(\mathbf{z}_T, \mathbf{v}_j))} \ast \mathbf{v}_i$

   Design Motivation: There is a modality gap and granularity difference between text embeddings and image embeddings. Expanding the text query with image features makes the goal embedding granularity during inference consistent with that during training.

### Loss & Training

Pre-trained on the HM3D ImageNav dataset using PPO reinforcement learning for 7.2M episodes. In each episode, 4 goal images with the minimum entropy are randomly sampled (from 10 candidates). After training, the model is directly migrated to ObjectNav and InstanceNav tasks in a zero-shot manner.

## Key Experimental Results

### Main Results

**ObjectNav Task (HM3D):**

| Method | Relies on LLM? | Requires Map? | SR(%) | SPL(%) |
|------|-----------|------------|-------|--------|
| L3MVN | ✔ | ✔ | 35.2 | 16.5 |
| PixelNav (GPT-4) | ✔ | ✘ | 37.9 | 20.5 |
| ESC (GPT-3.5) | ✔ | ✔ | 39.2 | 22.3 |
| ZSON | ✘ | ✘ | 25.5 | 12.6 |
| **PSL (Ours)** | **✘** | **✘** | **42.4** | **19.2** |

**InstanceNav Task (Text-goal):**

| Method | SR(%) | SPL(%) | Note |
|------|-------|--------|------|
| CoW | 1.8 | 1.1 | Requires Depth+GPS |
| ESC (GPT-3.5) | 6.5 | 3.7 | Requires Depth+GPS |
| ZSON | 10.6 | 4.9 | End-to-end |
| **PSL (Ours)** | **16.5** | **7.5** | End-to-end, no extra sensors |

**InstanceNav Task (Image-goal):**

| Method | SR(%) | SPL(%) |
|------|-------|--------|
| FGPrompt | 9.9 | 2.8 |
| ZSON | 14.6 | 7.3 |
| OVRL-V2 (Supervised) | 24.8 | 11.8 |
| **PSL (Ours, Unsupervised)** | **23.0** | **11.4** |

### Ablation Study

| SPM | GVS | PRR | ZSIN-image SR | ZSIN-text SR | ZSON SR | Description |
|-----|-----|-----|---------------|-------------|---------|------|
| ✘ | ✘ | ✘ | 12.7 | 10.6 | 25.5 | ZSON Baseline |
| ✔ | ✘ | ✘ | 19.5 | 13.0 | 33.7 | + Semantic Perception Module |
| ✘ | ✔ | ✘ | 14.8 | 11.8 | 30.4 | + Goal View Selection |
| ✔ | ✔ | ✘ | 16.5 | 12.3 | 35.0 | + SPM + GVS |
| ✔ | ✔ | ✔ | 22.0 | 16.5 | 42.4 | Full Components |

**Ablation on Semantic Expansion Inference (Text-goal InstanceNav):**

| PSL Agent | Semantic Expansion | Support Set | SR(%) | Description |
|-----------|---------|--------|-------|------|
| ✘ | ✔ | IIN 3.5K | 11.1 | Limited Categories |
| ✘ | ✔ | ImageNav 0.1M | 12.4 | Better Diversity |
| ✔ | ✘ | - | 6.6 | Directly Use Text Embedding |
| ✔ | ✔ | ImageNav 0.1M | 16.5 | Full Scheme |

### Key Findings

1. **PSL surpasses LLM-based methods for the first time without using LLMs**: Achieving an ObjectNav SR of 42.4% vs. ESC's 39.2%, while not requiring any extra sensors (Depth, GPS).
2. **SPM is the most critical single component**: Adding SPM alone increases ZSIN-image SR from 12.7% to 19.5% (+6.8%).
3. **Viewpoint relaxation must work in tandem with the PSL Agent**: Applying PRR to the original agent degrades performance (12.7% → 10.8%), but achieves significant improvement on the PSL Agent (16.5% → 22.0%).
4. **Semantic expansion inference addresses the modality gap**: Directly using text embeddings yields an SR of only 6.6%, which increases to 16.5% (+9.9%) after expansion via image retrieval.
5. **PSL approaches supervised methods on Image-goal InstanceNav**: Reaching an SR of 23.0% vs. 24.8% for supervised OVRL-V2, with comparable SPL performance.

## Highlights & Insights

- **Exquisitely designed pilot experiments**: The Layout-Only Agent (using Canny/Sobel + ResNet50) achieved success rates comparable to ZSON, robustly proving the presence of the semantic neglect problem.
- **Three-in-one solution**: Systematically addresses the same underlying target from three dimensions: model (SPM), data (training strategy), and inference (semantic expansion).
- **Forward-looking introduction of the InstanceNav task**: Compared to ObjectNav (finding *any* chair), InstanceNav (finding a "beige bamboo-framed double bed") aligns much closer with real-world application needs.
- **SOTA without LLM, map, or extra sensors**: The method is clean and efficient, making it highly suitable for physical robot deployment.

## Limitations & Future Work

- Validated only in the simulated HM3D environment, without verifying sim-to-real transfer on physical robots.
- Text descriptions for InstanceNav are automatically generated by CogVLM, which may introduce annotation noise.
- The support set size (0.1M) and similarity threshold ($\lambda=0.8$) require manual tuning, where adaptive strategies could be explored.
- SPM is currently a simple MLP bottleneck; more complex semantic reasoning modules could be explored.
- The ResNet-50 backbone is relatively dated; migrating to a larger ViT may yield further improvements.

## Related Work & Insights

- ZSON [Majumdar et al., 2022] serves as the direct baseline for this paper, establishing the paradigm of ImageNav pre-training followed by zero-shot ObjectNav transfer.
- ESC [Zhou et al.] and PixelNav utilize LLMs for navigation, but incur high computational costs and are unsuitable for real-time scenarios.
- The core concept of semantic expansion inference (enriching text queries with image features) can be extended to other zero-shot vision-language tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The pilot experiment reveals the semantic neglect problem, and the proposed three-in-one remedy is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across both ObjectNav and InstanceNav, evaluating under both text-goal and image-goal settings, with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The core motivation is logically driven by pilot experiments, demonstrating an excellent flow of reasoning.
- Value: ⭐⭐⭐⭐ Outperforming LLM-based approaches without relying on LLMs significantly advances the field of embodied navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation](../../CVPR2026/robotics/trajrag_retrieving_geometric-semantic_experience_for_zero-shot_object_navigation.md)
- [\[ECCV 2024\] SemGrasp: Semantic Grasp Generation via Language Aligned Discretization](semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](../../NeurIPS2025/robotics/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](../../AAAI2026/robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/robotics/semantic_audio-visual_navigation_in_continuous_environments.md)

</div>

<!-- RELATED:END -->
