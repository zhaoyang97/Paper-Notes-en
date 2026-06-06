---
title: >-
  [Paper Note] TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance
description: >-
  [ICML 2026][Robotics][Dense Reward Learning] TimeRewarder formalizes "task progress" as the normalized temporal distance between video frame pairs. It trains a self-supervised ViT distance regressor using only action-fre…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Dense Reward Learning"
  - "Temporal Distance"
  - "Passive Videos"
  - "Meta-World"
  - "DrQ-v2"
date: 2026-05-08
content_hash: 3271475fbed9c670
---

# TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance

**Conference**: ICML 2026  
**arXiv**: [2509.26627](https://arxiv.org/abs/2509.26627)  
**Code**: timerewarder.github.io (Project Page)  
**Area**: Robotics / Reinforcement Learning / Imitation Learning  
**Keywords**: Dense Reward Learning, Temporal Distance, Passive Videos, Meta-World, DrQ-v2

## TL;DR
TimeRewarder formalizes "task progress" as the normalized temporal distance between video frame pairs. It trains a self-supervised ViT distance regressor using only action-free expert videos and provides the predicted distance between adjacent frames as a dense reward to DrQ-v2. On 10 Meta-World tasks, it achieves near-perfect scores (9/10 tasks) within 200K interactions, even surpassing manually designed environment dense rewards.

## Background & Motivation
**Background**: Reinforcement learning (RL) for robotics suffers from poor sample efficiency under sparse rewards. Common remedies involve manually designing dense rewards or distilling proxy rewards from expert trajectories using methods like GAIfO, OT, or VIP. Manual rewards require extensive domain knowledge, privileged state access, and iterative tuning, making them difficult to scale. While learning rewards from videos has progressed, goal-conditioned value functions like VIP are difficult to converge, Rank2Reward captures limited information via binary ordering, and triplet-based objectives like PROGRESSOR are overly complex.

**Limitations of Prior Work**: Progress-based reward methods suffer from three specific issues: first, the implicit temporal contrastive objective of VIP is theoretically unbounded (proven by the authors in Appendix A.3), leading to unstable optimization; second, Rank2Reward only determines which frame is later without outputting a distance, failing to distinguish between being "one step away" versus "ten steps away"; third, GVL relies on VLM inference order, which causes high reward noise due to inconsistent outputs.

**Key Challenge**: A reward must simultaneously satisfy two seemingly conflicting properties: it must provide fine-grained discrimination of "progress" within the expert distribution, yet output reasonably low scores for unseen sub-optimal behaviors (stalling, backtracking, pseudo-actions) during RL exploration. Existing objectives either learn only forward progress while ignoring sub-optimal samples, or rely on goal images, leading to representation degradation when drifting from the goal.

**Goal**: To learn a function $F_\theta(o_u, o_v)$ from passive videos that: (1) provides high scores for progressive behaviors and low scores for stagnation or backtracking in RL rollouts; (2) possesses step-level resolution for adjacent frames; and (3) requires no action labels or goal images.

**Key Insight**: The authors observe that under an optimal policy, $\mathcal{V}^*(s_t^e) = -\sum_{k=t}^{T-1}\gamma^{k-t}$ is a monotonic transformation of the "remaining time $T-t$". Therefore, the temporal indices of expert video frames serve naturally as a potential function. This reduces "reward learning" to "self-supervised regression of normalized temporal differences."

**Core Idea**: The model predicts the normalized temporal distance $d_{uv} = (v-u)/(T-1) \in [-1, 1]$ between two frames. The predicted distance between adjacent frames is used as a dense reward, which, when combined with a sparse success signal, drives DrQ-v2.

## Method

### Overall Architecture
TimeRewarder consists of two stages: (1) Offline training of a progress model $F_\theta: \mathcal{O} \times \mathcal{O} \to \mathbb{R}^K$. The input is a pair of frame features encoded by a CLIP-pre-trained ViT-B and concatenated, passing through a linear head to output $K=20$ dimensional logits representing a two-hot distribution of the normalized temporal distance. (2) During the online exploration phase of DrQ-v2, adjacent observation pairs $(o_t, o_{t+1})$ are fed into the frozen $F_\theta$. The predicted distance $\hat{d}_{t,t+1}$ is treated as a step-wise reward, which is then fused with a binary success signal $r_{\text{success}}$ using a tunable $\alpha$ coefficient. The entire pipeline requires no action labels, goal images, or environment dense rewards.

### Key Designs

1. **Implicit Negative Sampling + Normalized Temporal Distance Regression**:
    - **Function**: Simultaneously teaches the model that "forward = positive progress" and "backward = negative progress" during self-supervised training, while explicitly leaving prediction space around 0 for "stagnation."
    - **Mechanism**: Randomly sampled frame pairs $(o_u, o_v)$ during training are not restricted to $u < v$, thus $d_{uv} \in [-1, 1]$. When RL rollouts encounter sub-optimal behaviors like hitting a wall, missing a grasp, or retracting, the visual difference between adjacent frames resembles a sub-segment of a "reversed trajectory." The model naturally regresses a small or negative value, effectively treating sub-optimal behaviors as reversed segments seen during training. This "anti-symmetric structure" prevents the model from using the shortcut of "both frames are seen = high score."
    - **Design Motivation**: In ablation studies, replacing the target with a pure forward progress in $[0, 1]$ caused tasks like stick-push and basketball to collapse, as missing a grasp was misidentified as "halfway completed." Embedding negative samples into the sampling distribution avoids the engineering burden of explicitly constructing failure trajectories.

2. **Weighted Pair Sampling (Short-interval Weighting)**:
    - **Function**: Increases model accuracy for distance prediction between adjacent frames to provide more informative step-wise rewards.
    - **Mechanism**: Pairs are sampled with an interval $\Delta = |v - u|$ according to a probability distribution $P(\Delta) \propto 1/\Delta$ ($\Delta \in \{1, \dots, T-1\}$). This biases training exposure toward short-interval pairs ($\Delta = 1, 2, 3$) while maintaining global progress awareness through long-interval coverage. Combined with Two-hot discretization (dividing $[-1, 1]$ into $K=20$ bins and assigning non-zero mass only to the two nearest bins), the model learns global monotonicity while remaining sharp at bin boundaries.
    - **Design Motivation**: RL rewards are step-wise; the accuracy of adjacent frame distance prediction determines whether each feedback "points forward." With uniform sampling, most gradients are spent on "medium distance" pairs ($\Delta \approx T/2$) that are nearly useless for step rewards. Ablations show significant performance drops in stick-push and window-open with uniform sampling. Two-hot discretization particularly improves tasks like basketball and disassemble, which involve "long preparation + short decisive actions," by preserving sharp transitions at the moment of completion.

3. **Potential-based dense reward + sparse success fusion**:
    - **Function**: Converts the trained $F_\theta$ into a step-wise reward $r_t$ for online RL and aligns its scale with the sparse success signal.
    - **Mechanism**: The reward is defined as $r_{\text{TR}}(o_t, o_{t+1}) = \Phi^{-1}[F_\theta(o_t, o_{t+1})]$, where $\Phi^{-1}$ maps the two-hot vector back to a scalar in $[-1, 1]$. The final training reward is $r_t = r_{\text{TR}}(o_t, o_{t+1}) + \alpha \cdot r_{\text{success}}(o_t)$, where $\alpha$ is adaptively adjusted. This is equivalent to reward shaping using $V(o) = F_\theta(o_0, o)$ as a potential function. Theoretical analysis proves that under a deterministic MDP with a step penalty $r(s) = -1$, $V^*(s)$ is a monotonic transformation of "remaining steps $T - t$," making $r_{\text{TR}}$ a natural instance of potential-based shaping that preserves the optimal policy.
    - **Design Motivation**: Sparse success signals are cheap to annotate via humans or VLMs but cannot guide exploration alone. Dense $r_{\text{TR}}$ might suffer from scale drift, causing the policy optimizer to ignore the moment of task completion. Adding the two ensures a strong signal at the completion state without increasing the complexity of the RL algorithm (a simple replacement in DrQ-v2).

### Loss & Training
The training loss is the cross-entropy of the discretized two-hot distribution: $\min_\theta \mathbb{E}[-\mathbf{y}_{uv}^\top \log\text{softmax}(\hat{\mathbf{y}}_{uv})]$, where $\mathbf{y}_{uv} = \Phi(d_{uv})$ is the ground truth two-hot vector, and $K = 20$. The backbone is a CLIP-pre-trained ViT-B (trainable). Frames are encoded independently and concatenated before a linear layer. For downstream RL, DrQ-v2 is used with $F_\theta$ frozen, running 200K environment interactions across 8 random seeds for each task.

## Key Experimental Results

### Main Results
Comparison of reward learning on 10 Meta-World tasks (100 action-free expert videos per task):

| Method | Information Source | 9/10 Tasks SR ≈ 100% | Remarks |
| :--- | :--- | :--- | :--- |
| **Ours** (TimeRewarder) | Frame-pair temporal distance | ✅ | 200K interactions, CLIP ViT-B |
| VIP | Implicit time-contrastive + goal frame | ❌ (Close on few tasks) | Representation degrades OOD |
| Rank2Reward | Binary order of adjacent frames | ❌ | No explicit distance |
| PROGRESSOR | Triplet Gaussian position estimation | ❌ | Only forward progress |
| GAIfO / OT / ADS | Rollout-expert alignment | ❌ | High online computation cost |
| Env dense reward | Manual privileged dense reward | 9/10 surpassed by **Ours** | Usually considered the upper bound |
| BC | Expert action supervision | Baseline | Requires action labels |

**Ours** achieves the highest final success rate and best sample efficiency in 9/10 tasks, outperforming the manual environment dense rewards (the supposed upper bound) in 9 tasks. This is attributed to manual rewards often having zero gradients in the pre-contact phase, whereas TimeRewarder captures subtle progression from video.

### Ablation Study
Removal of core modules (8 seeds, Meta-World):

| Configuration | Most Degraded Task | Symptoms | Explanation |
| :--- | :--- | :--- | :--- |
| Full TimeRewarder | — | Near 100% on 9/10 | Full model |
| w/o Implicit Negative Sampling | stick-push, basketball | Missing grasp misjudged as success | Anti-symmetric structure fails |
| w/o Weighted Sampling (Uniform) | stick-push, window-open | Insufficient resolution for short intervals | Lacks local fine-grained supervision |
| w/o Two-hot Discretization (Regression) | basketball, disassemble | Completion moment smoothed | Hurt long-prep/short-action tasks |
| only-from-init / single-frame / order-prediction | Multiple tasks drop | Weak temporal expressivity | All three alternatives fail |

### Key Findings
- **VOC (Value-Order Correlation)**: TimeRewarder scores highest on held-out expert videos across all tasks, proving it learns true temporal monotonicity rather than memorizing the training set. GVL (Gemini-1.5-Pro) lags behind in few-shot settings, indicating VLM reasoning is less stable for reward modeling.
- **OOD Failure Robustness**: In failures like grasping without lifting (basketball) or mimicking in the air (window-open), VIP and Rank2Reward are misled by visual similarity. PROGRESSOR saturates after grasping or gives false spikes. TimeRewarder is the only method that "waits for contact before increasing value." PCA visualization shows its feature space forms a consistent progress manifold for training, held-out, and RL rollouts, while failure trajectories clearly deviate.
- **Cross-Domain Video**: In experiments with 20 real human demonstrations per task + 1 in-domain Meta-World demonstration, training on humans or Meta-World alone yields low success rates. Mixing both leads to high performance, verifying TimeRewarder's ability to fuse heterogeneous passive videos.

## Highlights & Insights
- Reduces the philosophical problem of "learning rewards from passive videos" to "self-supervised regression of normalized temporal indices," outperforming complex modules (VLMs, triplets, goal-conditioned values, adversarial discriminators) with simplicity.
- The anti-symmetric $d_{uv} \in [-1, 1]$ design is a "free lunch": it embeds knowledge of how to handle OOD sub-optimal behaviors without needing failure trajectories or explicit negative sampling.
- Surpassing manual dense rewards in 9 tasks is counter-intuitive—it suggests that engineered shaping functions are often flat in early stages (e.g., before reaching an object), whereas video-learned progress is more sensitive to pre-contact phases and more exploration-friendly.

## Limitations & Future Work
- Single-frame observations may map different task stages to the same progress value under visual aliasing (back-and-forth motion). The authors suggest using frame windows to mitigate this POMDP issue.
- Evaluations are limited to Meta-World and simplified real-world scenes, without addressing multi-skill long-horizon tasks or dynamic scenes with multiple objects.
- The training assumes "near-optimal" expert videos; how the model degrades with sub-optimal or noisy demonstrations remains unexplored.

## Related Work & Insights
- **vs VIP**: Both learn values from temporal structure, but VIP uses implicit temporal contrastive objectives + goal conditioning, leading to representation degradation away from the goal and unbounded optimization. **Ours** is goal-free, has a bounded objective, and is more stable.
- **vs Rank2Reward / PROGRESSOR**: Rank2Reward lacks scale (only order); PROGRESSOR uses complex triplet Gaussian estimation and only sees forward progress. **Ours** captures both distance and anti-symmetric sub-optimal awareness with a simpler objective.
- **vs GAIfO / OT / ADS**: These online alignment methods treat "expert matching" as a reward, which is computationally expensive and unstable. **Ours** shifts computation to offline training, requiring only one ViT forward pass during RL.
- **vs GVL (VLM Reasoning)**: GVL is sensitive to prompts and model versions. **Ours** uses a small model + self-supervision to embed "temporal order" as a continuous, differentiable distance, offering better scalability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Temporal distance regression is not entirely new, but the combination of anti-symmetry, short-interval weighting, and two-hot discretization pushes this approach to outperform manual rewards.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 10 tasks with 8 seeds, cross-domain human videos, extensive ablations, and PCA visualizations. Lacks long-horizon multi-skill testing.
- **Writing Quality**: ⭐⭐⭐⭐ Smooth flow with question-driven experimental sections and clear theoretical/failure mode analysis.
- **Value**: ⭐⭐⭐⭐⭐ Makes "learning rewards from video" truly practical for Meta-World scale tasks and accommodates heterogeneous human videos, providing a plug-and-play boost for robotics RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Weakly-Supervised Learning of Dense Functional Correspondences](../../ICCV2025/robotics/weakly-supervised_learning_of_dense_functional_correspondences.md)
- [\[ICML 2026\] Position: Good Embodied Reward Models Need Bad Behavior Data](position_good_embodied_reward_models_need_bad_behavior_data.md)
- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)

</div>

<!-- RELATED:END -->
