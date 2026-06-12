---
title: >-
  [Paper Note] Investigating Memory in Model-Free RL with POPGym Arcade
description: >-
  [ICML2026][Causal Inference][model-free RL] This paper points out that comparing RL memory models solely using returns is unreliable. The authors construct a GPU-accelerated MDP/POMDP "twin" benchmark, POPGym Arcade…
tags:
  - "ICML2026"
  - "Causal Inference"
  - "model-free RL"
  - "POMDP"
  - "memory models"
  - "recurrent state"
  - "value smearing"
date: 2026-05-08
content_hash: 104002f4a59bb54a
---

# Investigating Memory in Model-Free RL with POPGym Arcade

**Conference**: ICML2026  
**arXiv**: [2503.01450](https://arxiv.org/abs/2503.01450)  
**Code**: https://github.com/bolt-research/popgym-arcade  
**Area**: Reinforcement Learning / POMDP / Memory Models  
**Keywords**: model-free RL, POMDP, memory models, recurrent state, value smearing

## TL;DR
This paper points out that comparing RL memory models solely using returns is unreliable. The authors construct a GPU-accelerated MDP/POMDP "twin" benchmark, POPGym Arcade, and propose four tools: Observability Gap, Memory Bias, Pixel Saliency, and Recall Density. Through these, they reveal a pathology called "value smearing": memory models incorrectly distribute value credit to irrelevant historical observations, leading even a single ODD (Out-Of-Distribution) observation to contaminate the policy over the long term via the recurrent state.

## Background & Motivation

**Background**: In Partially Observable Markov Decision Process (POMDP) scenarios, the mainstream approach is to prepend a memory model $f$ (RNN/GRU/LRU/Transformer/SSM, etc.) to the policy, compressing the historical trajectory $\mathbf{x}_t=(o_0,a_0,\dots,o_t)$ into a fixed-size latent Markov state $\hat{s}_t$, upon which a policy $\pi(\cdot\mid\hat{s}_t)$ interacts. The de facto standard for evaluating memory models is comparing average returns across several POMDP tasks.

**Limitations of Prior Work**: Deep RL is extremely sensitive to model scale, observation size, task difficulty, optimizers, and random seeds, while memory models themselves introduce additional parameter counts, optimization challenges, and regularization effects. Consequently, differences in returns between two memory models on a POMDP cannot distinguish whether the improvement stems from "mitigating partial observability" or these "unrelated confounding factors." Literature has even shown paradoxical phenomena where adding memory improves performance on MDPs but degrades it on POMDPs.

**Key Challenge**: Return, as a scalar, simultaneously carries "policy capability" and "memory capability," entangling the two. To honestly evaluate memory, one must be able to independently measure the impact of "adding memory" and "switching to partial observation" while keeping other variables constant. This requires a set of **truly homologous** MDP/POMDP twin tasks that share the same observation and action spaces to reuse the same model.

**Goal**: (1) Construct an MDP/POMDP twin benchmark sharing $(\Omega, A)$; (2) Provide metrics that "decompose" returns into Observability Gap and Memory Bias; (3) Provide tools to visualize and quantify memory usage patterns; (4) Use these tools to diagnose what existing memory models actually learn.

**Key Insight**: The authors noted that by applying an observation function $O$ to the same underlying MDP, a paired POMDP can be obtained. If the state/observation spaces of both are identical at the pixel level, the same network can be used for training, allowing the difference to naturally isolate the "difficulty brought by partial observability." Furthermore, taking the gradient of $Q$-values or the policy with respect to historical observations quantifies "which historical frame affects the current decision."

**Core Idea**: Use MDP/POMDP twin tasks to decompose returns into **Observability Gap** (loss due to partial observability) + **Memory Bias** (side effects of introducing memory). Then use **gradient-based Recall Density** to measure which moments the memory actually "looks back" at, discovering and characterizing the "value smearing" pathology.

## Method

### Overall Architecture
POPGym Arcade abstracts memory evaluation into four coordinated components: (1) **Twin Environments**: Each underlying task provides both an MDP variant $\mathcal{M}$ and a POMDP variant $\mathcal{P}$, sharing $128{\times}128{\times}3$ or $256{\times}256{\times}3$ pixel observations and a five-action space $\{\uparrow,\downarrow,\leftarrow,\rightarrow,\times\}$; (2) **Two Diagnostic Metrics**: Observability Gap and Memory Bias, which isolate factors using paired MDP/POMDP and paired "with memory/without memory" policies; (3) **Pixel Saliency Visualization**: $\lVert\nabla_{o_t}Q\rVert$ for directly viewing "which pixels are remembered" along a trajectory; (4) **Recall Density**: Normalizing the gradient norm along the trajectory and aggregating across trajectories to obtain a "time → relative influence" distribution function, comparable across models/tasks. The entire suite is implemented as a pure GPU pipeline in JAX, with throughput approximately $10^4$ times faster than the CPU version of Atari.

### Key Designs

1. **Observability Gap and Memory Bias (Bifactor Decomposition)**:

    - **Function**: Decomposes the scalar "return $X$ achieved by a memory-augmented policy on a POMDP" into two diagnostic signals with the same units for direct comparison.
    - **Mechanism**: Fixing the memory model $f$ and policy $\pi$, returns $J(f,\pi,\mathcal{M})$ and $J(f,\pi,\mathcal{P})$ are obtained on the twin MDP and POMDP, respectively. The difference $\text{Gap}(f,\pi,\mathcal{M},\mathcal{P})=J(f,\pi,\mathcal{M})-J(f,\pi,\mathcal{P})$ characterizes the loss caused by "f's failure to perfectly reconstruct the Markov state from the trajectory." Furthermore, by fixing the POMDP’s underlying MDP and comparing the return difference between "with memory" and "without memory" policies on the MDP, $\text{Bias}(f,\pi,\mathcal{M})=J(f,\pi,\mathcal{M})-J(\pi,\mathcal{M})$ captures side effects unrelated to observability, such as parameter count, optimization difficulty, and implicit regularization. Both are on the same scale as return.
    - **Design Motivation**: Previously, when only looking at returns, it was impossible to tell if "GRU is 5 points better than Transformer" because GRU is better at state inference, or easier to optimize, or has a more suitable parameter count. Gap/Bias provide two independent curves: "capability to handle partial observation" and "inherent additional cost." In experiments, the Bias difference between MinGRU and GRU (0.05) was comparable to the Gap difference (0.05), which is enough to flip return rankings.

2. **Pixel Saliency + Recall Density (Gradient-based Memory Metric)**:

    - **Function**: Answers "which historical frames were used for the current decision" at the trajectory level.
    - **Mechanism**: Given a trajectory $\mathbf{x}_n$, $\hat{s}_0,\dots, \hat{s}_n$ are derived via Eq. 1. Then, gradients of $Q$ or $\pi$ are taken with respect to historical observations: $\sum_{a_n}\lVert\nabla_{o_t}Q(\hat{s}_n,a_n)\rVert_2^2=\sum_{a_n}\lVert\frac{\partial Q}{\partial \hat{s}_n}\frac{\partial \hat{s}_n}{\partial o_t}\rVert_2^2$, creating heatmaps at the pixel level (the chain rule passes through both CNN and memory model). To avoid cherry-picking, the $L_1$ gradient norm for a single trajectory is normalized to an empirical density $\delta_Q(\mathbf{x}_n,t)$, mapping absolute time $t$ to normalized time $\tau=t/n\in[0,1]$. Averaging across multiple trajectories yields the Recall Density $\mathbb{E}_{\pi,f}[\delta_Q(\mathbf{x},\tau)]$. A version for $\pi$ gradients is provided for compatibility with policy gradient methods.
    - **Design Motivation**: Neither returns nor single-frame saliency suffice to judge if a "model uses memory in the right places." Recall Density provides a **quantifiable curve comparable across trajectory lengths and models**, allowing for anomaly detection against the ground truth that "under MDP, density should concentrate at $\tau\to 1$." This is the key measurement for discovering "value smearing."

3. **POPGym Arcade Twin Environments and JAX Pipeline**:

    - **Function**: Provides a hardware-accelerated and formally comparable experimental foundation for the first two categories of tools.
    - **Mechanism**: The state of each task is split into a low-dimensional latent Markov state $\tilde{s}\in\tilde{S}$ (e.g., mine positions in MineSweeper) and pixel Markov states $s\in S$ (e.g., board pixels with number hints). Both satisfy the Markov property, but an observation function $O:\tilde{S}\mapsto\Delta(\Omega)$ can generate a POMDP twin. All tasks are unified to the same $S=\Omega$ and action space, allowing **a single network to be reused across tasks**, and even allowing "switching from POMDP to MDP mid-training" for control experiments. 120 tasks (10 base envs × 12 difficulty/observation combinations) are provided, with Reward Memory Length labeled as $O(k)$ or $O(n)$ to distinguish between "solvable by windowing" and "true memory required."
    - **Design Motivation**: For Gap/Bias measures to be statistically significant, the MDP and POMDP must share the same optimal reachable return upper bound, and high throughput is needed for multiple seeds and models. Pure JAX + GPU allowed the authors to complete a full sweep across 7 memory models × 5 seeds × 120 configurations, the infrastructural prerequisite for this study's statistical confidence.

### Loss & Training
The main algorithm uses an on-chip TD($\lambda$) Q-learning implementation of PQN (Gallici et al., 2024), intentionally avoiding common confounders like target networks, replay buffers, and shared backbones. Key conclusions were also replicated with PPO and DQN in the appendix to exclude algorithmic effects. All memory models include a **skip connection** bypassing memory, giving the policy the ability to "ignore memory" on MDPs—a design that makes the observation of "memory still smearing history" more persuasive. Seven memory models were tested: Transformer, Recurrent Linear Transformer, Linear TTT, Gated DeltaNet, MinGRU, GRU, and LRU SSM.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Tool | Key Finding |
|----------|------|----------|
| Aggregated across 7 memory models on all tasks | Return / Gap / Bias Triple Plot (Fig. 5) | Memory Bias varies significantly across models and is always negative; the difference in Bias (0.05) and Gap (0.05) between MinGRU and GRU is comparable—return rankings can be flipped by Bias. |
| Sweep of layers $L$ and hidden dimension $H$ on BattleShip / MineSweeper | Gap–Bias Pareto Frontier (Fig. 6) | $L\uparrow$ usually worsens Bias; $H\uparrow$ usually improves Gap; the two curves form a Pareto frontier for capacity selection. |
| Pixel Saliency + Recall Density on MDPs | Fig. 3, Fig. 7 | Theoretically, $V^*(s_t)$ in MDP is independent of $s_{t-k},\dots,s_{t-1}$; density should concentrate in $\tau\in[0.66,1)$. Reality: significant weights for $\tau<0.66$ across all models and tasks—this is "value smearing." |
| OOD Injection Experiments | Single-frame noise (Fig. 9) + Prefix shuffling (Fig. 10) | Injecting just one OOD frame significantly perturbs relative $Q$-values and greedy actions in LRU. After shuffling prefixes (excluding CNN confounders), state pollution remains in BattleShip/MineSweeper (LRU) and CartPole (Transformer). |

### Ablation Study

| Configuration | Key Metric | Explanation |
|------|---------|------|
| Full GRU / LRU Complete Models | Low-variance convergence (Fig. 8) | Excludes "value smearing is an artifact of optimization instability" as an explanation. |
| With skip connection (Policy can bypass memory) | Smearing still occurs | Suggests the policy does not choose to "ignore memory" under MDP; the memory model indeed learns a solution that smears credit to an irrelevant past. |
| Transformer (No recurrent state) | Still affected by prefix shuffling | Indicates OOD pollution is not unique to RNNs but is a common problem in memory-policy joint solutions under partial observability. |
| Replication in PPO/DQN (Appendix) | Consistent phenomena | Excludes algorithm-specific (on-policy value method) factors. |

### Key Findings
- **Value smearing is a universal phenomenon**: Under MDP, $V$ should only depend on the current state, but Recall Density for 7 memory models across 10 tasks puts massive weight on the first 2/3 of the trajectory. This suggests memory-value joint optimization tends to treat "what happened to appear in this trajectory's past" as explanatory variables, overfiting the trajectory distribution under the current policy.
- **Returns are untrustworthy; check Bias**: Looking only at returns leads to contradictory conclusions about whether "memory makes it better or worse." Bias reveals that memory models bring a net negative effect even when observability is not an issue, meaning many previous SOTA comparisons contain hidden confounders.
- **OOD pollution is the practical cost of smearing**: Because value is smeared across irrelevant history, a single abnormal observation can long-term change the policy via the recurrent state, posing a real risk for real-world deployment and offline RL.
- **Interpretable Intervention**: If the Gap is large, increase hidden dimension $H$ (to mitigate state confusion); if Bias is strongly negative, decrease layers $L$ (to mitigate optimization difficulty). This provides an actionable direction for tuning.

## Highlights & Insights
- **"Twin + two differences with the same units"** is an elegant causal decomposition: decomposing a confounded variable into Gap and Bias signals on the same scale. Methodologically, it brings common RL confounders (parameter count/optimizer/task difficulty) to the table for quantification for the first time, generalizable to any "is this module actually useful" scenario.
- **Using MDP as an oracle to validate POMDP tools**: Validating the expected form of Recall Density on an MDP where ground-truth is known (concentrated at the end) and then using its anomaly to define pathology is a very robust experimental paradigm.
- **Value smearing pathology translates to LLMs**: The authors explicitly note that if modern RLHF-tuned LLMs exhibit similar smearing in long-context ICL tasks, it could explain sensitivity to "irrelevant insertions." This opens the door for using Recall Density-like tools for LLM long-context diagnosis.

## Limitations & Future Work
- Experiments focused on **pixel model-free RL**. Whether model-based RL (e.g., world models) and RL-finetuned LLMs exhibit the same smearing remains unverified; authors suggest this as a next step.
- The "best memory model" conclusion depends heavily on the axis of comparison (hidden dimension $H$ here). Switching to parameter count or wall-clock might change rankings; thus, the "overall optimality" of LRU should be interpreted cautiously.
- There is no ground-truth credit distribution in POMDPs. Currently, MDP proxies are used to prove the existence of smearing; future methods are needed to quantitatively measure smearing intensity in POMDPs.
- The root cause of value smearing (optimization difficulty? overfitting trajectory distribution? insufficient capacity?) remains a hypothesis requiring more systematic controlled experiments.
- Current Recall Density uses gradient norm as a proxy, which might underestimate information flow in models with saturated activations or truncated BPTT; future work could cross-validate with attention rollout or integrated gradients.
- Action spaces in POPGym Arcade are unified to 5 discrete actions; generalizability to continuous control POMDPs (e.g., partially observed MuJoCo) requires extension.

## Related Work & Insights
- **vs Morad et al. (POPGym, 2023)**: POPGym primarily provides CPU-based POMDP benchmarks and compares memory model returns. This work provides GPU-based twin MDP/POMDPs with **paired diagnostic metrics and gradient-based interpretability tools**, upgrading evaluation from "watching returns" to "causal decomposition."
- **vs Ni et al. (2022, 2024)**: Ni et al. also emphasize controlled experiments (like stripping reward memory length), but lack cross-task shared pixel observation spaces and gradient-style Recall Density.
- **vs Kapturowski et al. (R2D2) & Elelimy et al. (2024)**: These works analyze the influence of stale recurrent states or learned distributions; Recall Density directly gives the "input → current decision" influence distribution, which is closer to a causal measure.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Upgrades RL memory evaluation from "comparing returns" to "causal decomposition + gradient interpretability + pathology diagnosis," characterizing "value smearing" for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models × 10 tasks × multiple difficulties × 5 seeds, corroborated across PQN/PPO/DQN.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain (benchmark → metrics → pathology → consequences), formalized definitions; some figures (Fig. 7) are high-density and slightly difficult to read.
- Value: ⭐⭐⭐⭐⭐ Rewrites the methodology for evaluating memory models and challenges existing "memory is better" conclusions. The JAX-based twin benchmark is a reusable asset for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[ACL 2026\] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size](../../ACL2026/causal_inference/better_and_worse_with_scale_how_contextual_entrainment_diverges_with_model_size.md)
- [\[ICML 2026\] Causal-JEPA: Learning World Models through Object-Level Latent Masking](causal-jepa_learning_world_models_through_object-level_latent_masking.md)
- [\[ICML 2026\] An Odd Estimator for Shapley Values](an_odd_estimator_for_shapley_values.md)

</div>

<!-- RELATED:END -->
