---
title: >-
  [Paper Note] MVR: Multi-view Video Reward Shaping for Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][visual reward shaping] This paper proposes the MVR framework, which learns a state relevance function from multi-view video via video-text similarity matching. Combined with state-depe…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "visual reward shaping"
  - "multi-view video"
  - "vision-language models"
  - "state relevance learning"
date: 2026-05-08
content_hash: 0e00e6dc70bd2be6
---

# MVR: Multi-view Video Reward Shaping for Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2603.01694](https://arxiv.org/abs/2603.01694)  
**Code**: [https://mvr-rl.github.io/](https://mvr-rl.github.io/)  
**Area**: Reinforcement Learning
**Keywords**: visual reward shaping, multi-view video, reinforcement learning, vision-language models, state relevance learning

## TL;DR
This paper proposes the MVR framework, which learns a state relevance function from multi-view video via video-text similarity matching. Combined with state-dependent reward shaping that automatically attenuates VLM guidance, MVR outperforms existing VLM-based reward methods across 19 tasks on HumanoidBench and MetaWorld.

## Background & Motivation

**Background**: Reward design is critical in reinforcement learning. A recent emerging paradigm leverages image-text similarity from VLMs as a visual signal to augment rewards (e.g., VLM-RM, RoboCLIP), guiding agents toward states that match task descriptions.

**Limitations of Prior Work**: (a) **Limitations of static images**: Single-frame image-text similarity fails to characterize dynamic motion—optimizing per-frame similarity causes agents to repeatedly pause at the frame that most resembles "running," rather than actually running (which requires rhythmic alternation of both legs). (b) **Single-view occlusion**: A single camera angle causes occlusion among robot limbs, introducing viewpoint-dependent bias. (c) **Lack of adaptive decay**: Existing methods linearly combine VLM scores and task rewards, which may shift the optimal policy.

**Key Challenge**: VLM-provided visual guidance is valuable early in training (helping agents discover correct motion patterns), but if continuously applied, it may conflict with task objectives—a "use early, release later" mechanism is needed.

**Goal**: (a) Replace static images with video to accurately assess dynamic motion quality; (b) eliminate occlusion bias through multi-view observations; (c) design automatically decaying reward shaping to avoid persistent conflict between VLM guidance and task rewards.

**Key Insight**: Rather than directly fitting VLM scores (which suffers from a large semantic gap), the paper preserves ranking consistency between the video space and state space via paired comparisons; multi-view regularization is used to eliminate viewpoint bias; and an automatic decay mechanism is derived based on the Bradley-Terry model.

**Core Idea**: Learn a state-space relevance ranking function from multi-view video, and generate automatically decaying reward shaping signals by comparing against a reference set.

## Method

### Overall Architecture
Within the online RL loop: (1) the agent executes its policy and collects state sequences; (2) multi-view videos are rendered periodically; (3) a frozen ViCLIP computes video-text similarity scores to update the dataset $\mathcal{D}$ and reference set $\mathcal{D}^{\text{ref}}$ (retaining the top-$k$ best trajectories); (4) the state relevance model $f^{\text{MVR}}$ is updated from $\mathcal{D}$; (5) $f^{\text{MVR}}$ and $\mathcal{D}^{\text{ref}}$ are used to compute the visual feedback $r^{\text{VLM}}$, which is combined with the task reward $r^{\text{task}}$.

### Key Designs

1. **Matching Paired Comparisons**:

    - Function: Bridge the semantic gap between the state space and the video space.
    - Mechanism: Rather than directly regressing video-text similarity scores from states (which is intractable), ranking consistency is preserved instead. Given two videos $\mathbf{o}, \mathbf{o}'$, the Bradley-Terry model computes $h_{\text{vid}}(\mathbf{o}, \mathbf{o}') = \sigma(\psi^{\text{VLM}}(\mathbf{o}, \ell) - \psi^{\text{VLM}}(\mathbf{o}', \ell))$, and the state-space ranking $h_{\text{state}}(\mathbf{s}, \mathbf{s}')$ is trained to match it. The loss $L_{\text{matching}}$ is the cross-entropy between the two.
    - Design Motivation: This mirrors preference learning (RLHF) but fits probabilities rather than binary labels, yielding smoother and more stable training. Cross-view video pairs sharing the same state sequences naturally augment comparison data.

2. **Regularizing State Representations**:

    - Function: Eliminate systematic bias introduced by different camera viewpoints.
    - Mechanism: $f^{\text{MVR}}(s) = \langle g^{\text{rel}}, g^{\text{state}}(s) \rangle$ is decomposed into a state encoder and a learnable relevance direction. The regularization term $L_{\text{reg}} = |\psi^{\text{VLM}}(\mathbf{o}_i, \mathbf{o}_j) - \langle \bar{g}^{\text{state}}(\mathbf{s}_i), \bar{g}^{\text{state}}(\mathbf{s}_j) \rangle|$ aligns the similarity structure of state representations with that of video representations.
    - Design Motivation: Representation learning ($L_{\text{reg}}$) and relevance scoring ($L_{\text{matching}}$) are decoupled, allowing multi-view information to be effectively aggregated without mutual interference.

3. **State-Dependent Reward Shaping (Automatic Decay)**:

    - Function: Make VLM guidance strong early in training and have it vanish automatically later.
    - Mechanism: Policy relevance is defined as $h^\pi = \sum_s f^{\text{MVR}}(s) d^\pi(s)$, and the optimization objective is $\max_\pi v^\pi + w \log(\sigma(h^\pi - h^{\pi^\ell}))$ (encouraging the current policy to become indistinguishable from the optimal policy $\pi^\ell$). Applying Jensen's inequality yields $r^{\text{VLM}}(s) = \mathbb{E}_{s' \sim \pi^\ell}[\log(\sigma(f^{\text{MVR}}(s) - f^{\text{MVR}}(s')))]$.
    - Design Motivation: When the agent's behavior aligns with $\mathcal{D}^{\text{ref}}$, $f^{\text{MVR}}(s) \approx f^{\text{MVR}}(s')$, so $r^{\text{VLM}} \to 0$, and the VLM guidance naturally vanishes, avoiding persistent conflict with $r^{\text{task}}$.

4. **Reference Set Maintenance**:

    - Function: Approximate $\pi^\ell$ using the top-$k$ best historical trajectories.
    - Mechanism: $\mathcal{D}^{\text{ref}}$ retains the $k=10$ state sequences with the highest cross-view aggregated similarity, akin to "recalling one's best attempts."
    - Design Motivation: This avoids the need to train a separate policy from VLM rewards to approximate $\pi^\ell$, directly reusing online experience.

### Loss & Training

The state relevance model is trained with $L_{\text{rel}} = L_{\text{matching}} + L_{\text{reg}}$, updated every 100K steps with early stopping. The final reward is $r^{\text{MVR}}(s) = r^{\text{task}}(s) + w \cdot r^{\text{VLM}}(s)$, with $w \in \{0.01, 0.1, 0.5\}$ selected via grid search. One trajectory is rendered for every 9 collected, with randomly sampled viewpoints and video segment length of 64 frames. ViCLIP-L (428M parameters) is used.

## Key Experimental Results

### Main Results

HumanoidBench, 9 tasks (10M steps, 3 seeds):

| Task | MVR | TQC | VLM-RM | RoboCLIP | DreamerV3 |
|------|-----|-----|--------|----------|-----------|
| Walk | **927.47** ✓ | 510.58 | 535.35 | 737.34 ✓ | 800.2 ✓ |
| Run | **749.23** ✓ | 647.87 | 14.93 | 501.15 | 633.8 |
| Slide | **735.03** ✓ | 514.91 | 163.13 | 494.20 | 436.5 |
| Stand | **918.55** ✓ | 576.59 | 728.69 | 849.73 ✓ | 622.7 |
| Sit_Hard | **756.67** ✓ | 511.85 | 322.95 | 559.38 | 433.4 |
| Avg Rank | **1.67** | 3.11 | 3.78 | 2.89 | 3.56 |

MetaWorld, 10 tasks (1M steps, 5 seeds, success rate): MVR average rank 1.50, RoboCLIP 2.00, VLM-RM 2.40.

### Ablation Study

| Variant | Description |
|---------|-------------|
| w/o reg (remove $L_{\text{reg}}$) | Performance drops on multiple tasks, validating the value of multi-view regularization |
| w/o reference (use $f^{\text{MVR}}$ directly as reward) | Absence of automatic decay causes overfitting to VLM guidance on some tasks |
| MVR-CLIP (images instead of video) | Severe degradation on dynamic tasks (Run, Walk)—single frames cannot capture rhythmic motion |
| direct (directly fit VLM scores) | Semantic gap leads to unstable learning |
| Number of views (1→4) | Multiple views are generally beneficial; single-view suffices for Stand (static posture with no occlusion) |

### Key Findings
- MVR achieves best performance on 5/9 HumanoidBench tasks with the best average rank (1.67), and is the only method to simultaneously reach the success threshold on both Walk and Run.
- VLM-RM completely fails on Run (14.93 vs. 749.23), because single-frame similarity induces the agent to freeze in a "running pose" rather than actually running.
- Multiple views yield significant benefits for dynamic tasks, with minimal effect on static posture tasks.
- The automatic decay mechanism is critical: a case study shows that MVR can correct poor postures early in training and then naturally withdraw, allowing the agent to focus on speed optimization.

## Highlights & Insights
- **Fundamentally addresses the visual evaluation of dynamic motion**: Replacing static images with video is a natural yet overlooked choice. The paper clearly demonstrates the dramatic failure of single-frame methods on running tasks (VLM-RM: 14.93), providing highly compelling motivation.
- **Elegant design of paired comparisons**: Rather than directly regressing VLM scores (which suffers from a large semantic gap), only relative rankings are learned—this "rank-only" approach is considerably more robust than fitting absolute values, mirroring the success logic of RLHF.
- **Elegant automatic decay mechanism**: $r^{\text{VLM}}$ naturally approaches zero as behavior improves, requiring no manually designed decay schedule—far more elegant than existing fixed-weight combination schemes.
- **Reference set as best recalled attempts**: Using online top-$k$ trajectories to approximate the target policy avoids the need for expert demonstrations or separate training. This intuition, analogous to human skill learning, is highly inspiring.

## Limitations & Future Work
- Validation is limited to simulated environments; real-robot experiments have not been conducted—rendering multi-view videos in real-world settings requires a multi-camera setup.
- Rendering one trajectory per nine already reduces overhead, but ViCLIP (428M parameters) still incurs non-negligible computational cost.
- The weight $w$ still requires grid search; although automatic decay reduces tuning burden, the initial weight continues to affect performance.
- Performance on Balance_Simple and Balance_Hard is suboptimal (VLM-RM performs better), possibly because the visual signals for these tasks are more amenable to static evaluation.
- The quality of $\mathcal{D}^{\text{ref}}$ depends on exploration—insufficient early exploration may yield a poor reference set.

## Related Work & Insights
- **vs. VLM-RM (Rocamonde et al., 2024)**: VLM-RM uses CLIP image-text similarity with fixed-weight combination. MVR uses ViCLIP video-text similarity with automatic decay. The contrast on Run (749 vs. 15) constitutes the most compelling argument.
- **vs. RoboCLIP (Sontakke et al., 2024)**: RoboCLIP also uses video-text similarity but provides only trajectory-level sparse rewards. MVR learns a state-level dense relevance function, enabling finer-grained guidance.
- **vs. RLHF**: MVR's paired comparison with the BT model shares the same foundation as RLHF, but here "preferences" are derived from VLMs rather than humans, and comparisons are made between different trajectories of the same policy rather than outputs of different models.
- **Transfer potential**: The multi-view + state relevance learning framework is transferable to any setting requiring video-based assessment of behavior quality (e.g., sports coaching, surgical skill evaluation).

## Rating
- Novelty: ⭐⭐⭐⭐ The three designs—video, multi-view, and automatic decay—complement each other well, though none is individually entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 19 tasks × 5 methods × multiple ablations; the experimental design is highly systematic.
- Writing Quality: ⭐⭐⭐⭐ Method derivation is clear, though the dense notation requires careful reading.
- Value: ⭐⭐⭐⭐ Represents a substantive advance in VLM-driven RL reward design; practical and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[ICLR 2026\] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[ICLR 2026\] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity](whatever_remains_must_be_true_filtering_drives_reasoning_in_llms_shaping_diversi.md)
- [\[ICLR 2026\] ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)

</div>

<!-- RELATED:END -->
