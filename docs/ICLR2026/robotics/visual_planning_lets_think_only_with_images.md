---
title: >-
  [Paper Note] Visual Planning: Let's Think Only with Images
description: >-
  [ICLR 2026][Robotics][Visual Planning] This paper introduces Visual Planning — the first purely visual reasoning paradigm in which the entire planning process is expressed as a sequence of images without any textual inte…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Visual Planning"
  - "Pure Image Reasoning"
  - "Large Vision Model"
  - "GRPO"
  - "Reinforcement Learning"
  - "Navigation"
date: 2026-05-08
content_hash: 6938594ba3e352ba
---

# Visual Planning: Let's Think Only with Images

**Conference**: ICLR 2026
**arXiv**: [2505.11409](https://arxiv.org/abs/2505.11409)
**Code**: [GitHub](https://github.com/yix8/VisualPlanning)
**Area**: Robotics
**Keywords**: Visual Planning, Pure Image Reasoning, Large Vision Model, GRPO, Reinforcement Learning, Navigation

## TL;DR

This paper introduces Visual Planning — the first purely visual reasoning paradigm in which the entire planning process is expressed as a sequence of images without any textual intermediary. A Large Vision Model (LVM) autoregressively generates step-by-step state images. The authors further propose VPRL, a two-stage RL framework combining random-trajectory-initialized exploration with GRPO progress-reward optimization. On three navigation benchmarks (FrozenLake, Maze, MiniBehavior), VPRL achieves an average Exact Match (EM) surpassing text-based reasoning methods by 27%, demonstrating that image-based reasoning substantially outperforms text-based reasoning on vision-first tasks.

## Background & Motivation

**Background**: LLMs and MLLMs have achieved remarkable progress in reasoning; however, all reasoning processes take place in the textual space — even when the input contains images, visual information is first described in text before reasoning proceeds. Cognitive science's Dual Coding Theory posits that human cognition employs two independent channels — verbal and non-verbal — and that visual imagery is more efficient than language for spatial tasks.

**Limitations of Prior Work**:
- (1) In spatial/geometric tasks, converting visual information into textual descriptions discards critical spatial features, creating a modality gap.
- (2) Methods such as Visual Sketchpad use tools to generate auxiliary visuals, but reasoning decisions are still made in the textual space.
- (3) MVoT generates visualizations to assist text-based reasoning, but remains fundamentally a text-driven tool-use paradigm.
- (4) No truly pure visual reasoning paradigm exists — all existing methods ultimately rely on text for decision-making.

**Key Insight**: Completely eliminate textual intermediaries. Planning is redefined as an image sequence, where each image represents an environment state and actions are implicitly encoded in state transitions. An LVM trained exclusively on visual data avoids confounds introduced by language supervision.

**Design Motivation for RL**: RL has demonstrated substantially superior generalization over SFT in text-based reasoning (e.g., DeepSeek-R1), yet has never been applied to image-generative reasoning or planning scenarios.

**Limitations of SFT**: Supervised learning (VPFT) merely imitates trajectories in the training distribution, lacks exploration over diverse actions, is prone to overfitting, and cannot learn from mistakes.

**Evaluation Challenge**: Visual outputs are high-dimensional and sparse — unlike text tokens, they cannot be directly judged as correct or incorrect. Specialized dynamics interpreters and progress estimators must be designed to assess whether generated images represent meaningful planning progress.

## Method

### Key Design 1: Visual Planning Paradigm — Pure Visual Autoregressive Planning

- **Function**: Redefines the planning process as image sequence generation, where each step predicts the next visual state without any textual involvement.
- **Mechanism**: Given an initial state image $v_0$, the model autoregressively generates a planning trajectory $\hat{\mathcal{T}} = (\hat{v}_1, \ldots, \hat{v}_n)$, where each step is conditioned on the full history of preceding states:

$$\hat{v}_i \sim \pi_\theta(v_i \mid v_0, \hat{v}_1, \ldots, \hat{v}_{i-1})$$

- **Design Motivation**: In spatial planning tasks, state transitions (e.g., movement through a maze) are naturally suited to image representation — textual description of coordinates and layouts is not only verbose but error-prone (experiments show that 25.7% of coordinate descriptions are inconsistent with the actual layout). LVM-7B, pretrained exclusively on image/video data with zero text, is adopted as the backbone to fully eliminate the confound of language supervision.

### Key Design 2: Two-Stage RL Framework VPRL

- **Function**: Proposes VPRL, a two-stage training framework in which Stage 1 initializes the policy model via random trajectories to provide exploration capability, and Stage 2 optimizes the planning policy via GRPO with progress rewards.
- **Mechanism**:

  **Stage 1 (Policy Initialization)**: Random walks are executed in the environment to collect trajectories, training the model to generate valid state transitions while preserving exploration randomness. Valid next states are randomly sampled as supervision targets to prevent overfitting:

$$\mathcal{L}_{\text{VPFT}}(\theta) = -\mathbb{E}_{(v_{\leq i}, \tilde{v}_{i+1})} \left[ \log \pi_\theta(\tilde{v}_{i+1} \mid v_{\leq i}) \right]$$

  **Stage 2 (GRPO Optimization)**: The behavior model samples $G$ candidate next states, which are scored by the reward function; group-relative advantages are computed and the policy is updated via the GRPO objective:

$$\mathcal{J}_{\text{VPRL}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{k=1}^{G} \min\left(\rho^{(k)} A^{(k)},\; \text{clip}(\rho^{(k)}, 1-\epsilon, 1+\epsilon) A^{(k)}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

- **Design Motivation**: Directly using VPFT as the RL initialization leads to exploration collapse — teacher-forcing training drives model entropy toward zero, generating nearly identical actions that yield zero advantage and prevent policy updates. Stage 1 with random-trajectory initialization is specifically designed to address this issue; experiments confirm that the resulting entropy approximates a uniform random policy with a low rate of invalid actions.

### Key Design 3: Progress Reward Function

- **Function**: Designs a composite reward function that simultaneously evaluates the legality of generated visual states (whether environment constraints are violated) and goal progress (whether the agent is closer to the goal).
- **Mechanism**: A dynamics interpreter $\mathcal{D}$ parses the action type encoded in each state transition, and a progress estimator $P$ quantifies the remaining distance to the goal. Candidate states are classified into three categories with distinct rewards:

$$r(v_i, \hat{v}_{i+1}^{(k)}) = \alpha_{\text{opt}} \cdot \mathbb{I}[\mathcal{D}(\cdot) \in \mathcal{A}_{\text{opt}}] + \alpha_{\text{nopt}} \cdot \mathbb{I}[\mathcal{D}(\cdot) \in \mathcal{A}_{\text{nopt}}] + \alpha_{\text{inv}} \cdot \mathbb{I}[\mathcal{D}(\cdot) \in \mathcal{E}_{\text{inv}}]$$

  where $\alpha_{\text{opt}}=1$ (reward for optimal actions), $\alpha_{\text{nopt}}=0$ (valid but suboptimal), and $\alpha_{\text{inv}}=-5$ (heavy penalty for invalid actions).

- **Design Motivation**: Visual outputs cannot be matched token-by-token as text can; environment-level semantic evaluation is required. The three-tier reward design encourages progress toward the goal, permits legal detours, but severely penalizes illegal states (e.g., passing through walls), effectively guiding the policy to explore optimal paths within the feasible action space.

## Key Experimental Results

### Table 1: Main Results — Performance of Each Method on Three Navigation Tasks

| Method | Input→Output | FrozenLake EM | FrozenLake PR | Maze EM | Maze PR | MiniBehavior EM | MiniBehavior PR | Avg. EM | Avg. PR |
|--------|--------------|---------------|---------------|---------|---------|-----------------|-----------------|---------|---------|
| Gemini 2.0 Flash Direct | Img+Text→Text | 21.2 | 47.6 | 8.3 | 31.4 | 0.7 | 29.8 | 10.1 | 36.3 |
| Gemini 2.0 Flash CoT | Img+Text→Text | 27.6 | 52.5 | 6.9 | 29.8 | 4.0 | 31.2 | 12.8 | 37.8 |
| Gemini 2.5 Pro (think) | Img+Text→Text | 72.0 | 85.0 | 21.5 | 35.5 | 37.6 | 59.9 | 43.7 | 60.1 |
| Qwen2.5-VL Direct | Img+Text→Text | 1.2 | 15.0 | 0.6 | 14.5 | 0.3 | 9.8 | 0.7 | 13.1 |
| Qwen2.5-VL CoT | Img+Text→Text | 8.2 | 29.1 | 2.3 | 15.2 | 0.5 | 14.7 | 3.7 | 19.7 |
| Qwen2.5-VL SFT | Img+Text→Text | 68.6 | 84.4 | 60.9 | 70.3 | 31.3 | 56.1 | 53.6 | 69.9 |
| LVM VPFT (ours) | Img→Img | 75.4 | 79.5 | 59.0 | 64.0 | 33.8 | 52.2 | 56.1 | 65.2 |
| **LVM VPRL (ours)** | **Img→Img** | **91.6** | **93.2** | **74.5** | **77.6** | **75.8** | **83.8** | **80.6** | **84.9** |

### Table 2: Comparison of Text-Based Planning Variants on FrozenLake

| Method | EM (%) | PR (%) |
|--------|--------|--------|
| Qwen2.5-VL SFT Direct | 68.6 | 84.4 |
| Qwen2.5-VL SFT w/ Coordinates | 74.4 | 82.7 |
| Qwen2.5-VL SFT w/ ASCII | 73.1 | 83.4 |
| Qwen2.5-VL GRPO w/ VPRL reward | 54.4 | 69.9 |
| Qwen2.5-VL GRPO w/ PR metric reward | 60.1 | 74.3 |

Finding: Even with augmented textual representations such as coordinates or ASCII maps, RL-based text planning fails to surpass the SFT baseline, confirming that the bottleneck lies in the modality gap rather than the training method.

## Key Findings

1. **Visual planning decisively outperforms text-based reasoning**: VPRL achieves an average EM of 80.6% versus the best text-based SFT at 53.6% (+27%). The gap is largest on MiniBehavior (75.8% vs. 31.3%), indicating that the advantage of visual reasoning grows with task complexity.

2. **Text-based RL fails under multimodal input**: Unlike pure-text domains, applying RL to planning tasks with image+text input underperforms SFT (54.4% vs. 68.6%). The bottleneck is the modality gap in grounding visual information into text, with approximately 25% of layout descriptions mismatching the actual layout.

3. **Random initialization is critical for RL success**: VPFT training drives entropy toward zero, causing exploration collapse; Stage 1 random-trajectory initialization yields entropy approximating a uniform distribution with low invalid-action rates, providing sufficient exploration space for Stage 2 RL.

4. **VPRL substantially reduces invalid actions**: Among failed trajectories, the proportion containing invalid actions is 61%–78% for VPFT, whereas VPRL reduces this by at least 24%, demonstrating that VPRL effectively constrains the model to plan within the legal action space.

5. **VPRL is more robust under complexity scaling**: On FrozenLake scaled from 3×3 to 6×6, Gemini 2.5 Pro EM drops from 98% to 38.8%, while VPRL declines only from 97.6% to 82.4%, exhibiting a substantially more gradual performance curve.

## Highlights & Insights

- **"First purely visual reasoning paradigm"**: All prior "visual reasoning" work ultimately makes decisions in the textual space. Visual Planning achieves end-to-end reasoning entirely in image space — the AI equivalent of a human sketching diagrams to solve spatial problems.
- **First application of RL to image-generative planning**: The paper successfully transfers the RL→reasoning paradigm from DeepSeek-R1 in text to image generation, opening an entirely new research direction.
- **Computational validation of Dual Coding Theory**: Paivio's hypothesis posits that visual and verbal systems are independent reasoning channels; this work provides the first computational experimental confirmation of this cognitive science hypothesis.
- **Elegant two-stage design**: The Stage 1 random initialization solution to RL exploration collapse is concise and effective, substantially outperforming direct VPFT initialization.

## Limitations & Future Work

1. **Limited task scope**: Validation is restricted to three grid-based navigation tasks (FrozenLake, Maze, MiniBehavior); extension to continuous spaces, 3D environments, or real-robot scenarios remains unexplored.
2. **Rule-dependent environment interpreters**: The dynamics interpreter and progress estimator are currently rule-based rather than learned, limiting generalization to complex or unknown environments.
3. **Limited image resolution and complexity**: Current environment images are simple grid renderings; scalability to high-resolution, complex real-world images is unknown.
4. **Insufficient discussion of training cost**: Detailed analysis of training overhead, sampling efficiency, and other key engineering aspects of the two-stage RL+GRPO framework is lacking.
5. **Complementarity with language unexplored**: The paper positions visual planning as an alternative to text-based reasoning rather than a complement; in practice, fusion of both modalities may yield superior performance.

## Related Work & Insights

### vs. Visual Sketchpad (Hu et al., 2024)
Visual Sketchpad uses external tools to generate sketches/visualizations to assist MLLM reasoning, but **reasoning decisions remain entirely in the textual space** — visuals serve only as auxiliary displays. Visual Planning reasons entirely in image space without any text, representing a fundamental paradigm shift.

### vs. MVoT (Li et al., 2025)
MVoT generates visualizations for each text reasoning step, but is fundamentally a "text reasoning + visual tool-use" approach: the model first decides an action in text and then self-invokes visual generation for verification. Visual Planning requires no textual decision-making step; actions are implicitly encoded in image state transitions, fundamentally eliminating the modality gap.

### vs. Action-conditional Generative Models (Hafner et al., 2019; Ha & Schmidhuber, 2018)
World models such as Dreamer learn state-transition dynamics for model-based RL, but **do not perform planning** — they require coupling with an external planner. VPRL, by contrast, is a self-contained holistic planner that internalizes planning within the visual generation process, requiring no external planning module.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Both the pure visual reasoning paradigm and the VPRL framework are pioneering; GRPO is applied to image-generative planning for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three navigation tasks with difficulty scaling, ablation studies, and error analysis are comprehensive, though the range of task types is narrow.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Cognitive science motivation is clearly articulated, paradigm comparisons are intuitive, and equations and figures are well-presented.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for pure visual reasoning with significant implications for the multimodal reasoning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sparse Imagination for Efficient Visual World Model Planning](sparse_imagination_for_efficient_visual_world_model_planning.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](../../NeurIPS2025/robotics/thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)

</div>

<!-- RELATED:END -->
