---
title: >-
  [Paper Note] Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring
description: >-
  [ICLR 2026][Reinforcement Learning][PPO] PoRSE enables the LLM to not only generate target-oriented rewards but also design an "affordance state space" to drive task-related exploration. Through an online policy improvement process that dynamically weights both components, it establishes a new Prev. SOTA on 24 robotic manipulation/locomotion tasks and success
tags:
  - ICLR 2026
  - Reinforcement Learning
  - PPO
date: 2026-05-08
content_hash: d8f8628e5363675b
---
# Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1vXMfIYFZp](https://openreview.net/forum?id=1vXMfIYFZp)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Robot Skill Learning  
**Keywords**: LLM Reward Design, Curiosity Exploration, Affordance State Space, Reward-Policy Co-evolution, PPO  

## TL;DR
PoRSE enables the LLM to not only generate target-oriented rewards but also design an "affordance state space" to drive task-related exploration. Through an online policy improvement process that dynamically weights both components, it establishes a new Prev. SOTA on 24 robotic manipulation/locomotion tasks and successfully solves two previously intractable complex tasks.

## Background & Motivation
**Background**: A key bottleneck in training robotic skills with RL is reward function design—traditional manual tuning by experts is time-consuming and labor-intensive. Works like Eureka and ROSKA allow LLMs to generate executable reward code directly from language instructions, significantly reducing human labor. ROSKA further utilizes reward-policy co-evolution to avoid retraining from scratch after every reward modification.

**Limitations of Prior Work**: Existing LLM-based reward methods design rewards solely based on "task text descriptions," resulting in rewards that are **overly target-oriented while ignoring state exploration**. In dexterous manipulation (e.g., catching objects) with high degrees of freedom and sparse goal states, the policy search space is enormous, making agents prone to local optima. Traditional RL uses exploration bonuses to mitigate this, but general exploration bonuses do not distinguish task relevance, wasting resources on irrelevant states. Furthermore, the trade-off ratio between exploration and exploitation is often fixed from the start, ignoring actual policy progress.

**Key Challenge**: Rewards must be both "precise" (target-oriented) and "broad" (encouraging exploration), yet the two often conflict, and the optimal trade-off shifts dynamically during training. Fixed ratios and task-agnostic exploration cannot solve this.

**Goal**: Construct a unified framework where the LLM simultaneously outputs task-aware target rewards and task-related exploration mechanisms, while dynamically adjusting their trade-off based on policy progress without retraining from scratch for every reward-exploration combination.

**Key Insight**: **[LLM-designed affordance exploration space]** Compressing high-dimensional states into a low-dimensional discrete affordance state space closely related to the task, and performing count-based curiosity exploration in this space ensures exploration is naturally aligned with task goals. **[In-context Policy Grounding (IPG)]** Using real-time policy performance feedback to guide the LLM in proposing, evaluating, and pruning reward-exploration configurations while dynamically adjusting weights, combined with policy inheritance to avoid retraining.

## Method

### Overall Architecture
PoRSE integrates "target reward $R_g$, exploration bonus $R_e$, and policy $\theta$" into a self-reinforcing loop. The LLM first generates a target reward function and constructs an affordance state mapping function based on the task description to project raw states into a low-dimensional task-related space. In this space, visit-count curiosity bonuses are calculated. These are fused into a total reward using weight $\beta$ to train the PPO policy. Policy performance feedback is then returned to the LLM, allowing it to iteratively refine rewards, adjust mapping functions, and generate candidate weights through a filtration-expansion mechanism and policy inheritance to approach the optimal configuration.

```mermaid
flowchart TD
    T[Task Description Id + Env Code Ie] --> L[LLM deepseek-v3]
    L -->|Generate| RG[Target Reward Rg]
    L -->|Generate| M[Affordance Mapping M]
    M --> RE[Visit-count Curiosity Re]
    RG --> F[Weighted Fusion Rtotal = β·Re + (2-β)·Rg]
    RE --> F
    F --> RL[PPO Policy Training]
    RL -->|Performance Feedback V θ| IPG[IPG: LEF Elimination-Expansion + Inheritance]
    IPG -->|Refine Rg / M / β / α| L
    IPG -->|Best Policy θbest| RL
```

### Key Designs

**1. Task-related exploration bonus: Letting LLMs design affordance state spaces.** The first breakthrough of PoRSE is entrusting the LLM with "where to explore." It uses a mapping function $M: S \to S_o$ to compress high-dimensional raw states into a low-dimensional affordance state space (AFS), where each dimension measures the distance between the current state and the goal in terms of some "behavioral affordance"—e.g., in the DoorOpenInward task, $S_o=[s_o^d, s_a^l]$, representing Euclidean distance to the handle and angular displacement of the door. The continuous AFS is discretized into $C$ states, and curiosity bonuses are given based on visit frequency:

$$R_e(s_o) = \frac{\lambda}{\sqrt{\sum_{t=1}^{T} \mathbb{I}(s_{o,d}^t = s_{o,d}^c)}}$$

Less frequently visited states receive higher bonuses. The mapping function $M$ is automatically generated by the LLM from the task description ($M_n = \text{LLM}(I_d, I_e)$). Since exploration is anchored in task-related affordance dimensions, the agent is guided to explore states that are truly helpful for task completion. The total reward is initially formulated as $R_{total}^k = R_e^k + R_g^k$.

**2. Reward-bonus co-refinement.** Since single-shot generated rewards and mapping functions are often suboptimal, PoRSE adopts the multi-round iteration concept from Eureka, feeding policy evaluation data back to the LLM to refine both simultaneously. Target rewards are optimized via Eureka-style iterations, while for the exploration mechanism, the best mapping function from the previous round $M_{best}^{n-1}$ is fed back as a reference:

$$M_n = \text{LLM}(I_d, I_e, M_{best}^{n-1}, V(\theta))$$

This allows the reward function and affordance mapping to **co-evolve**, improving exploration efficiency over iterations rather than relying on a static bonus (experiments prove static LLMCount is significantly worse).

**3. In-context Policy Grounding (IPG) + Linear Elimination-Expansion Filtration (LEF).** The optimal ratio $\beta$ between target reward and exploration bonus shifts during training. Exhaustively training all combinations is infeasible. PoRSE uses a dynamic weighting form:

$$R_{total}^k = \beta \cdot R_e^k + (2-\beta) R_g^k, \quad \beta \in [0,2]$$

The LLM generates a set of candidate $\beta$ values based on policy feedback. LEF draws from the linear population reduction of L-SHADE and the expansion ideas of PSO: it trains $N$ "exploration-target reward pairs" in parallel, eliminating the worst-performing pair every $H$ epochs. After elimination, it uses the best remaining combination as a base to mutate and expand $J$ new $\beta$ values. This coordinate optimization simplifies the tangled search space. The logic is **adaptive balancing**: increasing $\beta$ to emphasize exploration when the policy stagnates, and decreasing $\beta$ to reinforce target rewards when nearing the goal.

**4. Policy inheritance to avoid retraining.** To reuse historical knowledge, PoRSE follows ROSKA's policy fusion, mixing the best historical parameters $\theta_{best}$ with a random policy $\theta_{random}$ using a fusion ratio $\alpha$:

$$\theta_f(\alpha) = \alpha \cdot \theta_{best} + (1-\alpha)\cdot \theta_{random}$$

While ROSKA uses expensive Bayesian optimization to find $\alpha$, PoRSE uses the same LEF mechanism to let the LLM directly propose and mutate $\alpha$ ($\alpha_{new}=\text{LLM}(I_d, V(\theta))$), significantly reducing computational costs while maintaining the exploration-exploitation balance. The training alternates between searching for $\beta$ and $\alpha$.

## Key Experimental Results

### Main Results
- **Envs/Tasks**: 24 robotic skill tasks; 20 from Bi-DexHands (dexterous manipulation), 4 from Isaac Gym (locomotion + arm). PPO is used as the policy, with DeepSeek-V3 as the LLM; $N=5$ rounds, $K=6$ reward-mapping pairs per round, 3000 epochs per policy, best of 5 seeds.
- **Metric**: MTS (Maximum Training Success).
- **Baselines**: Sparse Reward, Human Expert Reward, Eureka, ROSKA.

| Difficulty | Results Overview |
|---|---|
| Medium (16 tasks) | PoRSE leads in **15/16** tasks; MTS=1.000 in 5 tasks including Pen/Scissors; GraspAndPlace reaches 0.984 (25.3% higher than Human 0.785); only slightly trails ROSKA in Humanoid (8.454 vs 8.917). |
| Hard (8 tasks) | PoRSE leads in **all 8** difficult tasks; DoorCloseOutward/Kettle reach 1.000 (Eureka only 0.553/0.742); **First to solve TwoCatch** (0.349); BlockStack 0.753 far exceeds ROSKA 0.148. |

Regarding convergence: in the simple task LiftUnderarm, PoRSE reaches 0.97 by round 5 (Eureka 0.6 / ROSKA 0.8). In complex TwoCatch, where Eureka and ROSKA fail completely (0), PoRSE reaches 0.4, showing a tenfold improvement from round 1 to 5.

### Ablation Study

Component ablation (Selected MTS; ↓ denotes decrease relative to full PoRSE):

| Variant | Anymal | Franka | BlockStack | TwoCatch |
|---|---|---|---|---|
| PoRSE w/o $R_g$ | −0.128 | 0.883 | 0.328 | 0.190 |
| PoRSE w/o $R_e$ | −0.097 | 0.912 | 0.393 | 0.193 |
| PoRSE w/o $\theta_{fusion}$ | −0.346 | 0.671 | 0.603 | 0.017 |
| PoRSE w/o $R_{ratio}$ (Fixed $\beta$) | −0.066 | 0.946 | 0.296 | 0.297 |
| PoRSE w/o $\theta_{ratio}$ (Fixed $\alpha$) | −0.020 | 0.940 | 0.590 | 0.276 |
| **PoRSE (Full)** | **−0.012** | **0.957** | **0.753** | **0.349** |

Comparison with static exploration bonus LLMCount (Tab.3): PoRSE dominates in 6 tasks, e.g., TwoCatch 0.349 vs 0.000, BlockStack 0.753 vs 0.140, Franka 0.957 vs 0.706.

AFS Robustness (Tab.4, randomly assembled AFS): PoRSE-AFS-Random still significantly outperforms all baselines (BlockStack 0.680±0.080 vs Eureka 0.254 / ROSKA 0.148), though slightly lower than task-aligned AFS (0.753). This indicates the elimination mechanism can prune useless dimensions and is robust to imperfect affordance settings.

### Key Findings
- Removing any component leads to a drop; $\theta_{fusion}$ (policy inheritance) is most critical for long-horizon optimization tasks (TwoCatch 0.349→0.017), while $R_e$ is vital for exploration-heavy tasks (BlockStack 0.753→0.393).
- Fixing $\beta$ or $\alpha$ breaks the exploration-exploitation balance; **alternating search for both coefficients** is optimal.
- Sensitivity of $\beta$: Single rewards (pure $R_g$ Eureka or pure $R_e$ LLMCount) fail (0% success) on hard tasks, while PoRSE reaches 97% success in DoorOpenInward with a 1.5:0.5 ratio.

## Highlights & Insights
- **Turning "Exploration Direction" into an LLM-designable Object**: Previously, LLMs only handled target rewards. PoRSE lets the LLM generate affordance state mappings, ensuring curiosity exploration is task-aligned and solving the issue of general exploration wasting time on irrelevant states.
- **Dynamic Trade-off over Fixed Ratios**: Using real-time policy performance to drive $\beta$ allows for exploring when stuck and exploiting when converging, directly addressing the shifting nature of the exploration-exploitation trade-off.
- **Engineering Cost Reduction**: Using a unified LEF mechanism to search for both $\beta$ and fusion ratio $\alpha$ replaces ROSKA's expensive Bayesian optimization with LLM-guided elimination-mutation, outperforming predecessors without increasing compute.
- **AFS Robustness**: Even with randomly assembled AFS, Ours outperforms baselines, proving that gains stem from the elimination mechanism's fault tolerance, not just "luck" in LLM design.

## Limitations & Future Work
- **Dependency on Simulation**: All 24 tasks are in Bi-DexHands and Isaac Gym; no real-robot transfer was shown, and the accessibility of affordance states under real perception noise remains questionable.
- **LLM and Compute Cost**: Parallel training of multiple policies and multiple DeepSeek-V3 calls are required. While comparable to Eureka/ROSKA, absolute overhead is high (3000 epochs × multiple pairs × 5 rounds).
- **Hyperparameter Complexity**: $N, K, H, J$ and the search schedule for $\beta/\alpha$ require manual setting, meaning it is not yet "fully tuning-free."
- **Lower Gain in Locomotion Tasks**: Slightly underperformed ROSKA on Humanoid, likely because locomotion is more target-oriented with limited exploration gains.
- **Future Work**: Extending affordance mapping to visual/multimodal observations, real-robot closed-loop testing, or letting the LLM adaptively determine search budgets.

## Related Work & Insights
- **LLM Reward Design Lineage**: L2R (Language to modular rewards + MPC) → Eureka (Evolutionary reward code search) → ROSKA (Reward-policy co-evolution) → REvolve (LLM + Human Feedback evolution). PoRSE is the first to incorporate "exploration mechanisms" into the LLM design and co-evolution framework.
- **Curiosity-Driven Exploration**: Compared to ICM (Pathak) or count-based rewards (Tang), PoRSE's exploration is task-dependent, built on LLM-designed affordance spaces rather than being goal-agnostic.
- **Evolutionary Strategy Borrowing**: The elimination/expansion in LEF draws from L-SHADE and PSO, demonstrating the feasibility of grafting classical evolutionary algorithms onto LLM candidate filtering.
- **Insight**: When an RL design dimension (reward, exploration, curriculum) can be structurally expressed as LLM-generated code/mappings, the "LLM Generation + Feedback Elimination + Inheritance" cycle can be reused. PoRSE provides a paradigm for the "exploration" dimension.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Advancing LLMs from "target reward design" to "task-related exploration space design" with co-evolution is a clear incremental step.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 24 tasks, 4 baselines, 5 seeds, and five-dimensional ablations. Successfully solving TwoCatch is a highlight; lacking real-robot verification.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation-pain points-method alignment is clear; Figures 1 and 2 clarify differences from prior work.
- **Value**: ⭐⭐⭐⭐ — Adds a long-neglected exploration dimension to automatic reward design, pushing forward sparse-reward and high-DOF tasks with a portable cost-reduction path.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reference Grounded Skill Discovery](reference_grounded_skill_discovery.md)
- [\[ICLR 2026\] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](learn_to_reason_efficiently_with_adaptive_length-based_reward_shaping.md)
- [\[ICLR 2026\] Skill Learning via Policy Diversity Yields Identifiable Representations for Reinforcement Learning](skill_learning_via_policy_diversity_yields_identifiable_representations_for_rein.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs](tips_turn-level_information-potential_reward_shaping_for_search-augmented_llms.md)

</div>

<!-- RELATED:END -->
