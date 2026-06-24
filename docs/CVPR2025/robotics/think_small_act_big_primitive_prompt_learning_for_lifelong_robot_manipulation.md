---
title: >-
  [Paper Note] Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation
description: >-
  [CVPR 2025][Robotics][Primitive Prompt Learning] This paper proposes Primitive Prompt Learning (PPL), which encodes motion primitives into reusable prompt vectors. By combining this with flow-aware Motion-Aware Prompting (MAP), it enables the sharing of motion primitives across skills. Using a freeze-and-expand mechanism to support lifelong robot manipulation learning, PPL outperforms baselines such as LoRA and experience replay in both LIBERO and real-world environments.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Primitive Prompt Learning"
  - "Lifelong Learning"
  - "Optical Flow"
  - "Diffusion Policy"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 10e3dfca50eac224
---

# Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2504.00420](https://arxiv.org/abs/2504.00420)  
**Code**: None  
**Area**: Robot Manipulation / Lifelong Learning  
**Keywords**: Primitive Prompt Learning, Lifelong Learning, Optical Flow, Diffusion Policy, Catastrophic Forgetting

## TL;DR

This paper proposes Primitive Prompt Learning (PPL), which encodes motion primitives into reusable prompt vectors. By combining this with flow-aware Motion-Aware Prompting (MAP), it enables the sharing of motion primitives across skills. Using a freeze-and-expand mechanism to support lifelong robot manipulation learning, PPL outperforms baselines such as LoRA and experience replay in both LIBERO and real-world environments.

## Background & Motivation

**Background**: Robotic manipulation policies are typically trained on a fixed set of tasks, requiring retraining when encountering new tasks. In practical applications, robots need to learn new skills continuously without forgetting old ones (lifelong learning). Existing lifelong learning methods (experience replay, LoRA, etc.) either require storing historical data or fail to transfer knowledge effectively.

**Limitations of Prior Work**: Shared motion primitives exist across different manipulation skills (e.g., the "grasping" action is similar in grabbing a mug and grabbing a banana). However, existing methods only identify correlations between tasks through semantic similarity (text embeddings), ignoring shared structures at the motion level. Consequently, knowledge transfer between semantically distinct but motion-similar tasks ("grasp mug" vs "place banana") is missed.

**Key Challenge**: Lifelong learning needs to balance the retention of old knowledge with the acquisition of new knowledge. Excessive parameter sharing leads to good transfer but severe forgetting; high parameter isolation results in less forgetting but weak transfer.

**Goal**: Find shared motion primitives, allowing them to be learned during multi-task pre-training and then frozen and reused during lifelong learning.

**Key Insight**: Optical flow is used to extract motion information, which is combined with CLIP text embeddings to form query vectors. The motion patterns captured by optical flow ("grasping downwards", "pushing forwards") cross semantic boundaries, serving as the key to discovering the shareability of motion primitives.

**Core Idea**: Joint queries of optical flow and semantics discover cross-task motion primitives $\to$ encoded as prompt vectors $\to$ which are frozen after pre-training and reused throughout lifelong learning.

## Method

### Overall Architecture

A two-stage framework: Stage 1 pre-trains a Diffusion Transformer policy alongside primitive prompt vectors on multi-skill data. Prompts are injected into the Key/Value of MSA layers via a prefix mechanism. Stage 2 freezes pre-trained prompts when learning new skills, introducing new lifelong learning prompts and blending both types using an attention weighting mechanism. The MAP module integrates optical flow and text embeddings into a query vector, selecting relevant prompt components via cosine similarity.

### Key Designs

1. **Motion-Aware Prompting (MAP)**:

    - **Function**: Leverage both motion and semantic information to discover shared primitives across tasks.
    - **Mechanism**: Optical flow $F$ is extracted from videos using the RAFT algorithm and featurized to obtain $\Phi(F)$; task descriptions are encoded as $E_{\text{CLIP}}(T)$ using CLIP. The two are fused into a MAP query: $\text{MAP}(T,F) = f_{\text{prompt}}(E_{\text{CLIP}}(T), \Phi(F))$. Optical flow captures low-level motion patterns (direction, speed, trajectory), while CLIP captures high-level semantics.
    - **Design Motivation**: Pure text queries can only identify correlations between semantically similar tasks (e.g., "pick up cup" and "pick up bottle"), but fail to identify semantically distinct but motion-similar tasks (e.g., "grasp mug" and "place banana" both involve downward arm movement + gripper closing). Ablation studies demonstrate that MAP aligns the prompt weight distribution with shared motion patterns.

2. **Prefix Prompt Learning**:

    - **Function**: Inject skill knowledge into the policy network with a minimal number of parameters.
    - **Mechanism**: Prompts $p \in \mathbb{R}^{L_p \times D}$ are divided into $\{p^K, p^V\}$ pairs, with prefixes concatenated to the Key and Value sequences of the MSA: $f_{P-T}(\mathbf{p}, \mathbf{h}) = \text{MSA}(h_Q, [\mathbf{p}_K; h_K], [\mathbf{p}_V; h_V])$, where only the prompt parameters are updated while the backbone network remains frozen.
    - **Design Motivation**: The parameter footprint of prompt learning is far smaller than full fine-tuning or LoRA. Additionally, prompts can be independently frozen or expanded, making them naturally suited for knowledge management in lifelong learning.

3. **Frozen-Expanded Lifelong Learning Mechanism**:

    - **Function**: Learn new skills without forgetting old ones.
    - **Mechanism**: When a new task arrives, all pre-trained prompts are frozen, and a new set of lifelong prompts is added. The MAP query uses an attention-weighting mechanism to select relevant components from both the frozen and new prompts: $\alpha_m = \cos\_\text{sim}(\text{MAP}(T,F) \odot A, K_m)$, and $p = \sum_m \alpha_m P_m$. Only the parameters of the new prompts are updated.
    - **Design Motivation**: Frozen prompts keep old knowledge uncorrupted, while new prompts capture motion patterns unique to new tasks. The attention mechanism makes the selection process differentiable and adaptive.

### Loss & Training

Behavioral cloning loss: $\hat\theta = \min_\theta \sum_k \mathbb{E}_{s_t, a_t \sim \mathcal{D}_k}[\sum_t \mathcal{L}(\pi(a|s_t, T_k; \theta), a_k^t)]$, parameterized by the diffusion policy. Pre-training uses MimicGen + LIBERO data, with 200 human demonstrations per skill. During the lifelong learning stage, only the parameters of the new prompts are updated.

## Key Experimental Results

### Main Results

LIBERO lifelong learning (7 sequential tasks) forward/backward transfer (FWT/BWT):

| Method | Mean FWT | Mean BWT |
|------|---------|---------|
| Sequential | Low | Severe Forgetting |
| Experience Replay | Medium | Negative BWT |
| LoRA | Medium | Medium |
| **PPL (Ours)** | **0.83±0.03** | **0.78±0.09** |

Real-world experiments (Franka Panda, 9 skills):

| Method | Pre-training Mean | Lifelong Learning Mean |
|------|-----------|------------|
| Diffusion-Transformer | 0.42±0.09 | - |
| MoE | 0.73±0.08 | - |
| **PPL** | **0.84±0.05** | **0.68±0.05** |

### Ablation Study

| Configuration | Performance |
|------|------|
| Text-only Query | Only discovers correlations between semantically similar tasks |
| Text + Optical Flow Query (MAP) | Additionally discovers correlations between motion-similar tasks |
| No Pre-trained Prompts | Lifelong learning performance drops significantly |
| Extreme Lighting Changes | Optical flow quality degrades, leading to performance drop (0.83→0.61) |

### Key Findings
- **Optical flow discovers shared motion primitives**: MAP queries allow semantically different but motion-similar tasks (e.g., "grasp mug" and "place banana") to share higher prompt weights, which is impossible with text-only queries.
- **More prompts are not always better**: Too many prompts introduce noise; the optimal number must balance coverage and precision.
- **Training efficiency is close to LoRA**: PPL matches the speed of LoRA but achieves performance close to MoE (gaining both efficiency and quality).
- **Forgetting is more pronounced in later tasks**: The BWT of Task 7 drops to 0.43, indicating that longer lifelong learning sequences carry a higher risk of forgetting.
- **Insufficient robustness to lighting**: Lighting changes (warm $\to$ cold $\to$ dark) degrade the performance of the optical flow scheme from 0.83 to 0.61, whereas the text-only scheme is more stable.

## Highlights & Insights

- **MAP's bimodal query**: Combining "what it looks like" (semantics) and "how it moves" (motion) to discover shared primitives is highly intuitive—humans also refer to both execution patterns and task instructions when learning new skills.
- **Simplicity of freeze-and-expand**: Avoids complex regularization terms to prevent forgetting; it simply freezes old prompts and expands new ones. Attention weighting automatically handles the integration of old and new knowledge.
- **Prompts as carriers of motion primitives**: Each prompt vector corresponds to a motion primitive, and the prompt selection weights can visualize the patterns of motion sharing across tasks.

## Limitations & Future Work

- **Sensitivity to lighting**: Optical flow fails under extreme light variations, which limits deployment reliability in real-world environments. Depth or 3D scene flow might offer more robustness.
- **Forgetting of later tasks**: The BWT at the end of the 7-task sequence drops significantly (0.43), indicating that longer lifelong sequences may present greater challenges.
- **Desktop-only manipulation**: Experiments are limited to tabletop manipulation tasks with the Franka Panda; mobile manipulation or dual-arm collaboration is not addressed.
- **Manual tuning of prompt count**: The optimal number of prompts depends on task complexity, lacking an adaptive adjustment mechanism.
- **Single-step optical flow**: RAFT only extracts optical flow from adjacent frames, failing to capture long-range motion patterns.

## Related Work & Insights

- **vs LoRA**: LoRA modifies backbone weights for each task in lifelong learning; although the modification scale is small, cumulative changes can lead to drift. PPL freezes the backbone and only modifies prompts, which is inherently safer.
- **vs Experience Replay**: Experience replay requires storing historical data and exhibits negative BWT (actual forgetting) in long sequences. PPL does not require storing past data, retaining knowledge instead by freezing prompts.
- **vs MoE**: MoE performs well in multi-skill pre-training but has a large parameter footprint. PPL achieves performance close to MoE in a lightweight prompt-learning manner.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining optical flow and semantics to discover motion primitives is an interesting innovation, and the freeze-and-expand prompt design for lifelong learning is clean and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid evaluation with simulation and real-world experiments, multiple baselines, and thorough ablation, though the scale of tasks is limited (up to 9 skills).
- **Writing Quality**: ⭐⭐⭐⭐ Clearly described method and persuasive visualization analysis.
- **Value**: ⭐⭐⭐⭐ Provides a lightweight and effective solution for lifelong robot learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lifelong Embodied Navigation Learning](../../ICLR2026/robotics/lifelong_embodied_navigation_learning.md)
- [\[ACL 2025\] DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](../../ACL2025/robotics/drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_.md)
- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](../../NeurIPS2025/robotics/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[ICML 2026\] Think Less, Act Early: Reinforced Latent Reasoning with Early Exit in Vision-Language-Action Models](../../ICML2026/robotics/think_less_act_early_reinforced_latent_reasoning_with_early_exit_in_vision-langu.md)
- [\[ICCV 2025\] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation](../../ICCV2025/robotics/pasg_a_closed-loop_framework_for_automated_geometric_primitive_extraction_and_se.md)

</div>

<!-- RELATED:END -->
