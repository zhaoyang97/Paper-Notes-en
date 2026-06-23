---
title: >-
  [Paper Note] BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] BoreaRL is the first multi-objective reinforcement learning (MORL) environment for climate-adaptive boreal forest management. Using a physical simulator coupling energy-carbon-water fluxes, it poses the conflicting goals of "maximizing carbon sequestration vs. protecting permafrost" to MORL agents. The study reveals a
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: 47526d543a046dc8
---
# BoreaRL: A Multi-Objective Reinforcement Learning Environment for Climate-Adaptive Boreal Forest Management

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=750tza3sGf](https://openreview.net/forum?id=750tza3sGf)  
**Code**: Open-sourced (BoreaRL is claimed as open-source in the paper; specific address to be confirmed)  
**Area**: Reinforcement Learning / Multi-Objective RL / Climate Application benchmark  
**Keywords**: Multi-objective reinforcement learning, Boreal forest management, Permafrost protection, Carbon sequestration, Physical simulation environment  

## TL;DR
BoreaRL is the first multi-objective reinforcement learning (MORL) environment for climate-adaptive boreal forest management. Using a physical simulator coupling energy-carbon-water fluxes, it poses the conflicting goals of "maximizing carbon sequestration vs. protecting permafrost" to MORL agents. The study reveals a severe asymmetry in learning difficulty—the carbon goal is easily mastered while the permafrost goal is nearly unlearnable—and shows that a simple "site-selection" curriculum strategy surprisingly outperforms standard preference-conditioned methods.

## Background & Motivation
**Background**: Boreal forests store 30–40% of global terrestrial carbon, much of which is buried in permafrost extremely sensitive to warming. Utilizing forest management (planting, thinning, species configuration) as a climate mitigation tool requires the simultaneous optimization of two objectives: maximizing carbon sequestration and protecting permafrost from thawing and releasing carbon. Existing forest models (e.g., CLASSIC, CLM5, CBM-CFS3) are designed for predicting ecosystem states—telling you what happens under a management regime—but cannot assist in "finding the optimal management strategy."

**Limitations of Prior Work**: The impact of forest structure on permafrost involves complex surface energy balance pathways with many offsetting effects. Dense needleleaf forests have high annual photosynthesis and sequestration but low albedo (absorbing more summer heat) and intercept winter snow, leading to thinner snow cover. While thin snow helps cold air freeze the ground deeper (benefiting permafrost), summer heat gain may negate this benefit. Deciduous forests allow thicker, insulating snow (potentially accelerating thaw) but have higher albedo and absorb less heat in spring. Thus, the "ideal strategy" depends on the interaction of climate, species, stand density, management timing, and the weight of carbon vs. permafrost—a non-convex multi-objective optimization landscape.

**Key Challenge**: Carbon and permafrost goals are often physically antagonistic—practices favoring sequestration (high-density needleleaf) frequently harm permafrost. To hand this trade-off to an optimizer, what is needed is not just a more accurate predictive model, but a training environment that incorporates real physics and supports explicit multi-objective policy learning. Previous RL applications in forest management relied on oversimplified growth models that failed to capture these biogeophysical trade-offs.

**Goal**: Construct such an environment and systematically evaluate whether modern MORL algorithms can learn robust multi-objective policies in physically credible boreal forest management tasks.

**Key Insight**: Forest management is inherently a sequential decision-making process where early planting/thinning decisions determine permafrost outcomes decades later—a classic long-term credit assignment problem naturally suited for RL. Furthermore, the conflicting goals of carbon, permafrost (and biodiversity or economic return) naturally belong to MORL.

**Core Idea**: Integrate "physical process modeling" and "decision learning" into a single modular framework to create the first scientifically credible boreal forest MORL environment compatible with modern RL frameworks (`mo-gymnasium` API), using it as a benchmark to expose the failure modes of existing methods.

## Method

### Overall Architecture
BoreaRL is a modular and configurable framework consisting of two core components: the physical simulator **BoreaRL-Sim** and the MORL environment wrapper **BoreaRL-Env**. The simulator takes site characteristics, weather/climate data, natural disturbances, and historical information as input. It advances at $n$-minute timesteps to output annual carbon and ground energy flux metrics. Reward shaping converts these physical outputs into learning signals. The RL agent then learns an annual policy—deciding stand density and the ratio of needleleaf to deciduous species each year—aiming to maximize long-term sequestration while limiting permafrost thaw. The environment follows the `mo-gymnasium` API and provides five pluggable modules: physical simulation (optional backends and resolution), reward specification (customizable objectives and normalization), agent interface (supporting single/multi-policy MORL), environmental stochasticity (controlled weather generation), and evaluation protocol (standardized multi-objective metrics).

### Key Designs

**1. Physically-Grounded Coupled Flux Simulator (BoreaRL-Sim): Deriving Rewards from Real Energy Balance**

Addressing the limitation that existing RL forest environments use oversimplified growth models, BoreaRL-Sim is a process-based simulator that concurrently solves energy, water, and carbon fluxes at $n$-minute timesteps. It includes a multi-node energy balance model (canopy-trunk-snow-soil), dynamic carbon cycling, full water balance with snow dynamics, and a stochastic disturbance module for fire and pests. These formulations adopt validated physical forms from mainstream land surface models (CLM5, CLASSIC). Crucially, the permafrost reward is calculated using conductive heat flux across the permafrost boundary rather than simple air temperature proxies, capturing the complex and often delayed thermal inertia of soil.

**2. Multi-Objective POMDP Formulation and Vector Rewards**

The management task is formulated as a multi-objective partially observable Markov decision process (MOPOMDP) defined by $(S, A, O, P, R, \gamma)$. The **Observation Space** in generalist mode is 105-dimensional (including ecosystem state, site climate, disturbance history, age-class distribution, carbon pools, penalties, preference weights $w_C$, and 62 site parameters), reducing to 43-D in site-specific mode. The **Action Space** is discrete, encoding density change $\{-100, -50, 0, +50, +100\}$ stems/ha and target conifer ratio $\{0.0, 0.25, 0.5, 0.75, 1.0\}$. The **Vector Reward** returns $R_t = [r_{c,t}, r_{t,t}]$ at each step, with both components normalized to $[-1, 1]$. The carbon reward $r_{c,t}$ rewards net ecosystem carbon change $\Delta C_t$, provides bonuses for total carbon and harvested wood products (HWP), and penalizes invalid actions or exceeding biological limits. The permafrost (thaw) reward $r_{t,t}$ is intentionally **asymmetric**:

$$r_{t,t} = \mathrm{clip}\!\left(\frac{f_{cool} - \alpha \cdot f_{warm}}{40.0},\ -1,\ 1\right)$$

where $f_{cool}$ and $f_{warm}$ are annual cumulative cooling and warming heat fluxes (MJ m$^{-2}$), and $\alpha=2.5$ heavily penalizes warming—reflecting the precautionary principle that permafrost degradation is often irreversible and more critical to prevent than cooling is to gain.

**3. Dual Training Paradigms: Site-specific for Controlled Study, Generalist for Robustness**

The environment supports two paradigms. Let site parameters be $\phi \in \Phi$, parameterizing the transition kernel $P_\phi$ and reward $R_\phi$. **Site-specific** mode fixes $\phi = \phi^\star$ with deterministic weather for reproducible point optimization. **Generalist** mode samples $\phi \sim D_{site}$ at the start of each episode, prepending context to the observation, targeting a mixture MDP:

$$J(\pi) = \mathbb{E}_{\phi\sim D_{site}}\,\mathbb{E}_{\tau\sim P^\phi_\pi}\!\left[\sum_{t\ge 0}\gamma^t R_\phi(s_t,a_t)\right].$$.

Preferences are denoted by $\lambda = (w_C, w_P)$ ($w_P=1-w_C$), where linear scalarization yields $r^\lambda_t = w_C R_{carbon,t} + (1-w_C) R_{thaw,t}$.

**4. Multi-Objective Baseline Algorithms**

The environment includes several MORL baselines. **Fixed Lambda EUPG** trains a single policy with fixed $\lambda$. **Variable Lambda EUPG** samples $\lambda \sim D_\Lambda$ per episode, feeding weights into the observation. **PPO Gated** uses a standard PPO with action gating to separate "planting" from "thinning" based on legality. The most significant is **Curriculum PPO (adaptive episode selection)**: it is preference-conditioned but adds a selection network $f_\phi(o_{site}) \to [0, 1]$ that evaluates the learning value of a site. It selects episodes based on an adaptive threshold:

$$J_{Curriculum}(\theta,\phi) = \mathbb{E}_{\lambda}\,\mathbb{E}_{\phi}\,\mathbb{E}_{select\sim f_\phi}\!\left[\mathbb{E}_{\tau\sim P^\phi_{\pi_\theta}}\Big[\sum_{t\ge 0}\gamma^t r^\lambda_t\Big]\,\Big|\,select=1\right].$$.

The selection network $f_\phi$ is a **non-trained random projection**, providing a consistent ranking of the site space, allowing the agent to skip "destabilizing" sites and consolidate policy on safer subsets.

## Key Experimental Results

### Main Results
In generalist mode, three preference-conditioned algorithms are compared against heuristic baselines using Hypervolume (ref: $[-2,-2]$) and Sparsity.

| Method | Scalarized Reward | Hypervolume ↑ | Sparsity ↓ |
| :--- | :--- | :--- | :--- |
| **Curriculum PPO** | **8.5 ± 3.0** | **84.3** | 0.12 |
| PPO Gated | 4.7 ± 6.0 | 23.6 | 0.09 |
| Variable λ EUPG | 1.7 ± 5.0 | 14.2 | 0.07 |
| Target Density (1000 stems/ha) | 4.3 ± 3.4 | 20.6 | N/A |
| Conifer Restoration (100% Conifer) | 4.1 ± 2.9 | 21.4 | N/A |
| Zero Density Change | −2.5 ± 2.4 | 11.3 | N/A |
| +100 Density Change | −3.2 ± 6.1 | 18.5 | N/A |

Curriculum PPO leads significantly in both Scalarized Reward and Hypervolume (84.3 vs. 23.6 for second place), while standard Variable λ EUPG shows nearly zero performance. Counter-intuitively, simple heuristics like maintaining a target density can achieve moderate rewards but fail to capture the trade-off frontier.

### Ablation Study
Ablation on permafrost reward forms (using PPO Gated):

| Permafrost Reward Form | Scalarized | Carbon | Thaw | Hypervol. ↑ | Sparsity ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Asymmetric (Default) | 4.7 ± 6.0 | 7.8 ± 2.5 | 1.5 ± 1.2 | 23.6 | 0.09 |
| Contrast | 4.9 ± 3.2 | 7.6 ± 2.6 | 2.1 ± 2.4 | 25.4 | 0.08 |
| Raw Degree Days | 5.2 ± 3.7 | 7.9 ± 2.4 | 2.4 ± 2.3 | 26.1 | 0.08 |

All forms are difficult to optimize, confirming the permafrost goal's inherent challenge. The asymmetric form is the hardest but most ecologically safe.

### Key Findings
- **Severe Learning Asymmetry**: Carbon goals are learned rapidly, whereas permafrost goals show negligible progress. This is due to physics—carbon rewards provide clear, immediate feedback via biomass, while permafrost signals are delayed, noisy, and seasonal.
- **Emergent Management Philosophies**: Carbon-dominant policies aggressively increase density to 1280 stems/ha, whereas permafrost-dominant policies remain conservative (1000–1020 stems/ha). However, it is noted that "conservative" behavior might simply indicate a lack of learning progress.
- **Curriculum Value**: The success of Curriculum PPO suggests that simply controlling training distribution (selective exposure) is more critical than algorithmic complexity in high-stochasticity physical environments.

## Highlights & Insights
- **Scientific Translation to MORL**: By using conductive heat flux rather than temperature and encoding the precautionary principle via $\alpha=2.5$, the environment translates climate science into a learnable RL task.
- **Negative Result Value**: The fact that "naive site selection" beats "sophisticated preference conditioning" suggests standard MORL methods fail when one objective signal is inherently weak/noisy.
- **Untrained Random Projection as Selector**: Curriculum PPO's success with an untrained selector suggests that "consistent relative ranking + dynamic difficulty boundaries" is more practical than learning accurate value predictors for curricula.

## Limitations & Future Work
- **Uncertainty in Permafrost Learning**: It remains unclear if conservative thaw policies are truly learned strategies or just a failure to deviate from start states.
- **Unresolved Challenge**: Current MORL methods cannot yet solve climate-adaptive management robustly; BoreaRL primarily exposes these difficulties.
- **Baseline Simplicity**: Stronger methods (Hierarchical RL, Model-based RL) have not yet been tested on this benchmark.

## Related Work & Insights
- **vs. Preference-Conditioned MORL**: Standard methods like Variable λ EUPG fail in the generalist setting, whereas adaptive episode selection succeeds, showing the fragility of gradient-based preference conditioning in noisy environments.
- **vs. Process Models**: Unlike models that only predict, BoreaRL bridges the gap between scientific simulation and policy optimization.
- **vs. Early RL Forest Environments**: Moves beyond simplified growth models to incorporate coupled energy-water-carbon fluxes and climate-driven uncertainty.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning](a_reward-free_viewpoint_on_multi-objective_reinforcement_learning.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](../../AAAI2026/reinforcement_learning/mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[ICLR 2026\] AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)
- [\[ICLR 2026\] Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations](dual-objective_reinforcement_learning_with_novel_hamilton-jacobi-bellman_formula.md)
- [\[ICLR 2026\] Adaptive Scaling of Policy Constraints for Offline Reinforcement Learning](adaptive_scaling_of_policy_constraints_for_offline_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
