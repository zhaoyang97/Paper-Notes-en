---
title: >-
  [Paper Note] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs
description: >-
  [ACL 2025][LLM Alignment][data mixing] AutoMixAlign proposes a theory-driven data mixing method for multi-task preference optimization: it first trains specialist models for each task to establish optimal loss baselines, and then adaptively adjusts data mixing proportions via minimax optimization, prioritizing tasks with the largest excess loss (gap from the specialist). It achieves an average improvement of 9.42% in helpfulness/harmlessness/reasoning multi-task DPO.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "data mixing"
  - "multi-task DPO"
  - "minimax optimization"
  - "excess loss"
  - "preference learning"
date: 2026-05-08
content_hash: bad45c5aa760b5c5
---

# AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.00569](https://arxiv.org/abs/2506.00569)  
**Code**: None  
**Area**: Alignment RLHF  
**Keywords**: data mixing, multi-task DPO, minimax optimization, excess loss, preference learning

## TL;DR
AutoMixAlign proposes a theory-driven data mixing method for multi-task preference optimization: it first trains specialist models for each task to establish optimal loss baselines, and then adaptively adjusts data mixing proportions via minimax optimization, prioritizing tasks with the largest excess loss (gap from the specialist). It achieves an average improvement of 9.42% in helpfulness/harmlessness/reasoning multi-task DPO.

## Background & Motivation
**Background**: LLM alignment training requires simultaneous good performance across multiple tasks (helpfulness, safety, coding, math), and DPO training requires mixing multiple task datasets.

**Limitations of Prior Work**:
   - Uniformly mixing data can be dominated by large datasets, leaving minor tasks ignored.
   - Equal task weighting (normalized by dataset size) treats hard and easy tasks identically, leading to a waste of resources.
   - Determining the optimal mixing ratio usually requires extensive ablation experiments, which is highly costly.

**Key Challenge**: Static data mixing cannot adapt to the dynamic changes in learning difficulty across tasks during the training process.

**Goal**: Automatically determine the data mixing ratios for multi-task DPO.

**Key Insight**: Using the specialist model's loss as a reference baseline, dynamically adjust mixing by minimizing the maximum excess loss (minimizing the excess loss of the worst task).

**Core Idea**: First train specialists to set the target loss of each task, then use minimax optimization to make the generalist chase all specialists.

## Method

### Overall Architecture
AMA consists of two stages: (1) training specialist models $\theta_1, \ldots, \theta_k$ on individual task datasets via DPO; (2) training a generalist model, balancing all task performances via the minimax optimization problem: $\min_{\theta} \max_{i \in [k]} \frac{1}{|\mathcal{D}_i|}\sum_{z \in \mathcal{D}_i} \max\{\mathcal{L}(\theta, z) - \mathcal{L}(\theta_i, z), 0\}$.

### Key Designs

1. **Excess Loss**:

    - Function: Quantify the gap of DPO loss between the generalist and the specialist on each sample.
    - Mechanism: $\mathcal{E}(\theta, \theta_i, z) = \max\{\mathcal{L}(\theta, z) - \mathcal{L}(\theta_i, z), 0\}$, clipped at 0 to avoid over-optimizing already-learned tasks.
    - Design Motivation: The problem with directly optimizing loss is not knowing "how good is good enough". Specialist loss provides achievable targets for each task—stopping the optimization of that task once the generalist loss falls below the specialist loss, avoiding overfitting.

2. **AMA-R: Reweighting Algorithm**:

    - Function: Adaptively adjust the weight $\alpha_i$ of each task in the objective function.
    - Mechanism: Transform the minimax problem into $\min_\theta \max_{\alpha \in \Delta^k} \sum_i \alpha_i \cdot \text{avg-excess-loss}_i$. Alternately execute exponentiated gradient ascent to update $\alpha$ (increasing the weights of tasks with larger excess losses) and gradient descent to update $\theta$.
    - Convergence: $O(1/\sqrt{T})$ convergence rate under the convex setting (inherited from Sagawa et al., 2019).

3. **AMA-S: Resampling Algorithm**:

    - Function: Adaptively adjust sampling probabilities from each task dataset.
    - Mechanism: Use the EXP3 online learning algorithm to maintain the sampling probability $\alpha$: $\alpha_i = (1-c)q_i + c/k$. Adjust $q_i$ via exponential updates based on the excess loss of each task, such that tasks with larger excess losses are sampled more.
    - Design Motivation: In AMA-R, even if weight $\alpha_i \approx 0$, sampling and gradient computation for task $i$ are still required (wasting computation); AMA-S directly reduces the sampling volume of tasks with low excess losses.
    - Convergence: Model this as a two-player zero-sum game, proving an $O(1/\sqrt{T})$ convergence rate leveraging EXP3.

### Loss & Training
- Specialist pre-computation: Pre-compute and cache all $\mathcal{L}(\theta_i, z)$ before training, requiring no extra forward passes during training.
- Smoothing parameter $c$: Prevents the sampling probability from decaying to 0, maintaining exploration.

## Key Experimental Results

### Main Results

**Multi-Task DPO (Helpfulness + Harmlessness + Reasoning)**:

| Method | Helpfulness | Harmlessness | Reasoning | Avg. |
|------|:-----------:|:------------:|:---------:|:----:|
| Uniform DPO | Baseline | Baseline | Baseline | Baseline |
| Task-normalized DPO | +1-2% | +0-1% | -1-2% | ~+0.5% |
| Model Merging | Unstable | Unstable | Unstable | Lower than AMA |
| **AMA-R** | Significant Improvement | Significant Improvement | Significant Improvement | **+7-9%** |
| **AMA-S** | Significant Improvement | Significant Improvement | Significant Improvement | **+8-9.42%** |

AMA outperforms both uniform mixing and model merging across all tasks, achieving a maximum gain of 9.42%.

### Ablation Study

| Configuration | Performance | Note |
|------|:----:|------|
| w/o excess loss (utilizing raw loss) | Decline | Over-optimizes already-learned tasks |
| w/o specialist clipping | Decline | No stopping criterion leads to overfitting |
| AMA-R vs AMA-S | AMA-S slightly superior | Higher computational efficiency |
| Varying specialist quality | Minimal effect | Specialists only need to be "good enough" |

### Key Findings
- Excess loss clipping is critical—without it, the generalist continues to optimize on already-learned tasks, squeezing resource allocation away from harder tasks.
- AMA-S is more computationally efficient than AMA-R (as it avoids sampling redundant tasks) and delivers slightly superior performance.
- Model merging (SLERP/TIES/DARE) exhibits unstable performance in multi-task alignment, often sacrificing performance on certain tasks.

## Highlights & Insights
- **Specialist-as-target Insight**: Utilizing the specialist's loss as the generalist's learning target is an elegant design—it provides a clear signal of "how much learning is enough", avoiding the perpetual dilemma in traditional multi-task learning where models "do not know when to stop".
- **Theoretical Guarantees + Empirical Effectiveness**: The convergence proof of EXP3 for AMA-S provides a solid theoretical foundation for adaptive data mixing, while showing significant empirical effectiveness.
- **Generalizability**: Although this paper focuses on DPO, the specialist + minimax excess loss framework can be effortlessly migrated to SFT, RLHF, or any multi-task learning scenarios.

## Limitations & Future Work
- **Training Cost of Specialists**: Training a specialist model is required for each task, scaling linearly with the number of tasks $k$. Efficient alternatives, such as using "approximate specialists" obtained through fewer training steps, could be explored.
- **Convex Convergence Assumption**: The $O(1/\sqrt{T})$ guarantee only holds under the convex setting, whereas actual LLM training is highly non-convex.
- **Task Granularity Definition**: Defining task divisions beforehand is required (which data belongs to which task), which might not apply well to scenarios with blurry task boundaries.
- **Only Validated on DPO**: Compatibility with RL methods such as PPO/GRPO remains unvalidated.

## Related Work & Insights
- **vs Uniform/Heuristic Mixing**: Traditional methods are static or heuristic-based; AMA is adaptive and theoretically grounded.
- **vs Model Merging (TIES/SLERP)**: Post-hoc merging methods perform unstably for multi-task alignment; AMA provides more reliable dynamic balancing during training.
- **vs Group DRO (Sagawa et al.)**: AMA-R inherits the minimax framework of Group DRO but introduces excess loss clipping and specialist targets.
- This framework can be extended to data mixing during pre-training (replacing DPO loss with language modeling loss).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of specialist + minimax excess loss is novel, backed by solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive configurations and abrading, though experiments are primarily conducted on medium-scale models.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations with clear problem motivation.
- Value: ⭐⭐⭐⭐ Multi-task data mixing is a core issue in practical deployment; AMA provides an automated solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)
- [\[ACL 2025\] Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization](atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)
- [\[ACL 2025\] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)
- [\[ACL 2025\] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)

</div>

<!-- RELATED:END -->
