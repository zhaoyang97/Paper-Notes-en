---
title: >-
  [Paper Note] GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill
description: >-
  [CVPR 2025][Reinforcement Learning][Open-vocabulary physical skills] This paper proposes the GROVE framework, which constructs a generalized reward function by leveraging LLMs to generate physical constraints and VLMs to evaluate motion semantics in a complementary manner. By using a lightweight Pose2CLIP mapper to skip rendering and project poses directly into the semantic space, GROVE achieves open-vocabulary physical skill learning, yielding 8.4x faster training speed and…
tags:
  - "CVPR 2025"
  - "Reinforcement Learning"
  - "Open-vocabulary physical skills"
  - "generalized reward function"
  - "LLM constraint generation"
  - "VLM motion evaluation"
date: 2026-05-08
content_hash: 39c4bd6d179af61f
---

# GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill

**Conference**: CVPR 2025  
**arXiv**: [2504.04191](https://arxiv.org/abs/2504.04191)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Open-vocabulary physical skills, generalized reward function, LLM constraint generation, VLM motion evaluation, reinforcement learning

## TL;DR
This paper proposes the GROVE framework, which constructs a generalized reward function by leveraging LLMs to generate physical constraints and VLMs to evaluate motion semantics in a complementary manner. By using a lightweight Pose2CLIP mapper to skip rendering and project poses directly into the semantic space, GROVE achieves open-vocabulary physical skill learning, yielding 8.4x faster training speed and a 22.2% improvement in motion naturalness compared to existing methods.

## Background & Motivation

**Background**: Learning diverse physical skills (e.g., running, dancing, carrying) in simulated environments is a core challenge for embodied AI. Existing reinforcement learning methods primarily rely on two reward design paradigms: manually designed reward functions and demonstration-based rewards.

**Limitations of Prior Work**: Manually designed reward functions lack scalability across tasks—each new task requires expert manual tuning. Demonstration-based methods (such as imitation learning driven by motion capture data) struggle to generalize to new tasks outside the training distribution. Neither paradigm can achieve "open-vocabulary" learning, where agents acquire skills from arbitrary natural language descriptions.

**Key Challenge**: Open-vocabulary learning implies that rewards cannot be pre-designed and demonstrations cannot be collected for all possible tasks in advance, necessitating a general mechanism to automatically generate reward signals from task descriptions. However, physical skill learning requires satisfying both precise physical constraints (e.g., "raise leg to 30 degrees") and ensuring the overall naturalness and semantic correctness of the motion.

**Goal**: To design a generalized reward framework without manual engineering or task-specific demonstrations, enabling simulated agents to learn open-vocabulary physical skills from natural language descriptions.

**Key Insight**: LLMs excel at decomposing task descriptions into precise physical constraints (but struggle to evaluate visual rendering), while VLMs excel at evaluating the overall naturalness and semantics of motion (but struggle to generate precise constraints)—making them highly complementary.

**Core Idea**: To build an LLM-VLM collaborative reward system: LLMs generate physical constraints while VLMs evaluate motion semantics. The two components self-improve through iterative feedback, and Pose2CLIP is utilized to skip expensive rendering for efficient training.

## Method

### Overall Architecture
The workflow of GROVE is divided into three stages: (1) LLM constraint generation stage—given a natural language description of the task, the LLM outputs a set of physical constraints (target values and weights for joint angles, positions, velocities, etc.); (2) VLM iterative optimization stage—the VLM evaluates video clips of motions generated under current constraints and provides feedback to the LLM for constraint refinement; (3) Efficient training stage—Pose2CLIP is deployed to replace rendering, directly mapping agent poses into the CLIP semantic space to compute rewards.

### Key Designs

1. **LLM Physical Constraint Generation**:

    - **Function**: Convert natural language task descriptions into executable sets of physical constraints.
    - **Mechanism**: Provide the LLM (e.g., GPT-4) with an API description of the simulation environment (available joints list, physical value ranges, etc.), enabling it to generate constraint code based on the task description. Each constraint includes a target physical quantity, target value, tolerance range, and weight. For example, "doing push-ups" is decomposed into hand-to-ground contact constraints, periodic elbow angle variation constraints, and torso horizontal maintenance constraints.
    - **Design Motivation**: LLMs possess broad world knowledge, allowing them to comprehend high-level descriptions like "push-ups" and convert them into physical constraints. However, they lack visual judgment and cannot evaluate if the generated constraints actually yield natural-looking motions.

2. **VLM Feedback Iterative Optimization**:

    - **Function**: Continuously improve the quality of LLM-generated constraints through a visual feedback loop.
    - **Mechanism**: Train an RL agent with the current constraints, render its motion video, and pass it to the VLM (e.g., GPT-4V) to evaluate motion naturalness and task completion. The VLM's feedback (e.g., "insufficient arm extension", "excessive knee bending during squats") is sent back to the LLM to refine the constraints. This process iterates over multiple rounds, forming a self-improvement loop.
    - **Design Motivation**: Constraints generated solely by the LLM are typically sub-optimal—they may miss key constraints or feature inappropriate weight settings. The VLM provides "visual common sense" to compensate for the LLM's "lack of visual imagination".

3. **Pose2CLIP Lightweight Mapper**:

    - **Function**: Directly map the agent's joint poses to the CLIP semantic space, avoiding expensive rendering steps.
    - **Mechanism**: Train a small MLP network where the input is the agent's joint angle vector and the output is the feature vector in the CLIP image space. Training data is obtained via aligned pose-rendered image-CLIP feature triplets. During inference, the features output by Pose2CLIP are directly compared with the CLIP text features of the task description to calculate cosine similarity as the semantic reward component.
    - **Design Motivation**: In RL training, rendering images at every step and running CLIP incurs enormous computational overhead. Pose2CLIP simplifies this process into a single MLP forward pass, accelerating training speed by 8.4x.

### Loss & Training
The total reward function is a weighted combination of the LLM physical constraint reward component and the Pose2CLIP semantic reward component. The RL algorithm utilizes PPO. The training loss for Pose2CLIP is the cosine similarity loss between the predicted features and the ground-truth CLIP features.

## Key Experimental Results

### Main Results
Evaluating open-vocabulary physical skill learning across various embodied morphologies (humanoid, quadruped, etc.):

| Method | Motion Naturalness | Task Completion | Training Speed | Description |
|------|----------|----------|---------|------|
| GROVE (Ours) | **Highest (+22.2%)** | **Highest (+25.7%)** | **Fastest (8.4x)** | LLM+VLM+Pose2CLIP |
| Text2Reward | Medium | Medium | Slow | LLM-only reward generation |
| Eureka | Medium | Higher | Slow | LLM + evolutionary search |
| UniHSI | Lower | Lower | Medium | Demonstration-based |

### Ablation Study

| Configuration | Motion Naturalness | Task Completion | Description |
|------|----------|----------|------|
| Full GROVE | Highest | Highest | Full model |
| w/o VLM Feedback | Significant drop | Noticeable drop | Single-pass constraint generation by LLM, unstable quality |
| w/o Pose2CLIP | Comparable | Comparable | Replaced with rendering + CLIP, similar accuracy but 8.4x slower |
| w/o LLM Constraints | Lowest | Lowest | Using only VLM scores as rewards, lacks precise physical constraints |

### Key Findings
- Collaboration between LLM and VLM is key: using either alone performs significantly worse than combining both. LLMs provide precision, while VLMs provide semantic common sense.
- Pose2CLIP achieves 8.4x speedup with almost no loss in accuracy, proving to be an ideal alternative to rendering.
- VLM feedback iterations typically converge within 2-3 rounds, with diminishing marginal returns from additional rounds.
- GROVE performs consistently across different embodied morphologies, demonstrating the generalizability of the framework.

## Highlights & Insights
- The design of the **LLM-VLM complementary architecture** is highly elegant: LLMs excel at structured reasoning but lack visual judgment, while VLMs possess visual common sense but struggle to output precise constraints—making them perfectly complementary. This paradigm can be extended to other tasks requiring both "precision and naturalness".
- The "skip rendering" concept of **Pose2CLIP** is highly noteworthy: rendering is a common bottleneck in RL training, and directly projecting low-dimensional states to semantic space via a lightweight mapper offers a general acceleration strategy.
- The **self-improvement loop** (VLM feedback → LLM correction → retraining → VLM re-evaluation) is a promising paradigm, resembling the meta-learning concept of "AI designing AI rewards".

## Limitations & Future Work
- Currently validated only in simulation environments; Sim-to-Real transfer effects remain unexplored.
- VLM feedback iterations require actually training the RL agent and rendering videos, which incurs high initialization costs.
- For tasks requiring high physical precision (e.g., gymnastics), the constraints generated by the LLM may lack sufficient accuracy.
- Training data collection for Pose2CLIP requires rendering, meaning the mapper must be retrained for different environments or morphologies.
- Complex multi-agent collaborative physical skill scenarios have not been explored.

## Related Work & Insights
- **vs Text2Reward**: Text2Reward only uses a single-pass LLM to generate reward code, lacking a visual feedback loop. GROVE significantly enhances constraint quality through VLM iterative optimization.
- **vs Eureka**: Eureka relies on evolutionary search to optimize LLM-generated rewards, but the search process is slow and does not utilize visual information. GROVE's VLM feedback mechanism is more efficient and physically more informed.
- **vs UniHSI**: UniHSI depends on human interaction demonstration data, whereas GROVE requires no demonstrations at all, offering stronger scalability.

## Rating
- Novelty: ⭐⭐⭐⭐ The LLM+VLM collaborative reward design and Pose2CLIP acceleration are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple embodied morphologies and tasks, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and well-justified motivations.
- Value: ⭐⭐⭐⭐ Provides a practical framework for open-vocabulary embodied skill learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring](../../ICLR2026/reinforcement_learning/master_skill_learning_with_policy-grounded_synergy_of_llm-based_reward_shaping_a.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](../../NeurIPS2025/reinforcement_learning/certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
