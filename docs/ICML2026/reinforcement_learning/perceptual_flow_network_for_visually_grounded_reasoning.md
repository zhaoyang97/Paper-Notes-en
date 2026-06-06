---
title: >-
  [Paper Note] Perceptual Flow Network for Visually Grounded Reasoning
description: >-
  [ICML 2026][Reinforcement Learning][Visually Grounded Reasoning] Moving away from the traditional RLVR approach of using "precise bounding boxes from visual experts as hard supervision…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Visually Grounded Reasoning"
  - "GFlowNet"
  - "Sub-TB"
  - "Variational RL"
  - "LVLM Hallucination"
date: 2026-05-08
content_hash: b9f0404979144438
---

# Perceptual Flow Network for Visually Grounded Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.02730](https://arxiv.org/abs/2605.02730)  
**Code**: None  
**Area**: Multimodal VLM / Reinforcement Learning / Visual Reasoning  
**Keywords**: Visually Grounded Reasoning, GFlowNet, Sub-TB, Variational RL, LVLM Hallucination

## TL;DR
Moving away from the traditional RLVR approach of using "precise bounding boxes from visual experts as hard supervision," PFlowNet models perceptual behavior as a structured latent variable called "Perceptual Flow." It approximates the ideal reasoning-oriented posterior using a variational distribution $p_\theta(Z|X)$ and employs Sub-TB variational RL combined with multi-dimensional rewards and Vicinal Geometric Shaping. This allows the 8B Qwen3-VL to achieve new SOTAs: 90.6% on V* Bench and 67.0% on MME-RealWorld-lite.

## Background & Motivation

**Background**: To mitigate linguistic bias and hallucinations in Large Vision-Language Models (LVLMs), recent methods (e.g., look-twice, VGR, grounded thinking) utilize RLVR to distill geometric priors from visual experts (such as GroundingDINO) into the LVLM, prompting the model to "box key regions first, then answer."

**Limitations of Prior Work**: The authors conducted a crucial probe experiment using Qwen2.5-VL on V*. By expanding expert-annotated boxes isotropically to obtain geometric priors with varying IoUs and feeding only the corresponding crops to the LVLM, they found a counter-intuitive result: the most precise expert box is not necessarily the best evidence for reasoning; a "sweet spot" exists. Visual experts, designed for object detection, prioritize geometric precision while neglecting the "context required for reasoning." Overly tight boxes create "tunnel vision," removing peripheral cues essential for task completion.

**Key Challenge**: Existing VGR equates "expert geometric precision" with "reasoning evidence quality," forcing LVLMs to align strictly with expert boxes. However, the "golden evidence" most useful for reasoning is instance-specific, and heuristic expansion cannot accurately capture it. This represents a fundamental misalignment by using an incorrect objective for alignment.

**Goal**: (1) Formalize VGR as a distribution modeling problem for latent visual trajectories $Z$; (2) construct a family of learnable perceptual trajectory representations to decouple "geometric precision" from "reasoning utility"; (3) use variational RL to encourage exploration towards "reasoning-friendly" perceptual behaviors while maintaining geometric reliability.

**Key Insight**: Instead of using expert boxes as hard constraints, the reasoning trajectory $Z$ is treated as a latent variable. A self-parameterized variational distribution $p_\theta(Z|X)$ approximates the "ideal VGR posterior" $P_V(Z|X,Y)$, which only requires $Z$ to fall within a $\sigma$-neighborhood $\mathcal{S}_V$ centered around the golden evidence $G$. Expert boxes $E$ serve only as "vicinal references" rather than hard targets.

**Core Idea**: Treat perception as a Perceptual Flow (planning state + a sequence of grounded perceptual states). Use the Sub-Trajectory Balance objective—a GFlowNet-style variational target—for dense supervision, and design "multi-dimensional rewards + vicinal geometric shaping $\omega_\lambda$ (active only outside the expert neighborhood)" to achieve "sufficient exploration without exceeding boundaries."

## Method

### Overall Architecture
PFlowNet decouples the LVLM workflow into two stages: (i) **Flow Generation**: The model samples a Perceptual Flow $Z = (z_0 \to z_1 \to \dots \to z_K)$ from $p_\theta(Z|X)$. Here, $z_0$ is a planning state wrapped in `<analyze>...</analyze>`, and $z_{\ge 1} = \langle r_k, c_k\rangle$ are perceptual states wrapped in `<localize>...</localize>`, each containing a RoI box (relative coordinates 0–1000) and a descriptive caption. (ii) **Flow-Guided Reasoning**: Based on $Z$ and its cropped visual evidence $I_{RoI}$, the model auto-regressively generates the final answer $Y$. The joint distribution factors as $p_\theta(Y, Z|X) = p_\theta(Z|X) p_\theta(Y|Z, \langle X, I_{RoI}\rangle)$. Training involves two steps: cold-starting with SFT on synthetic perceptual flow data $(X, Z_s)$, followed by optimizing $p_\theta(Z|X)$ using variational RFT on $(X, Y, E)$ to approach $P_V$.

### Key Designs

1.  **Perceptual Flow + Sub-TB Variational Objective**:
    -   **Function**: Discretizes perceptual behavior ("where the model looks and what it sees") into structured trajectories and provides dense supervision for each segment, addressing problems where PPO-like objectives provide rewards only at the end of an episode and suffer from high gradient variance.
    -   **Mechanism**: Defines Perceptual Flow $Z = (z_0, z_1, \dots, z_K)$, where the planning state $z_0$ is natural language and perceptual states $z_k = \langle r_k, c_k\rangle$ are RoI boxes + captions, explicitly segmented by special tokens like `<analyze>` and `<localize>`. It introduces Sub-Trajectory Balance from GFlowNet: for any sub-segment $z_{i:j}$, the condition $\mathcal{F}(z_i)\,\mathcal{T}_F(z_{i:j}) = \mathcal{F}(z_j)\,\mathcal{T}_B(z_{j:i})$ must hold, leading to the vRFT objective $\mathcal{L}_{vRFT}(\theta)$ (Equation 2 in the paper), using the sum of squared log ratios as the loss.
    -   **Design Motivation**: (a) Explicit structure allows rewards to be defined on sub-trajectories; (b) Sub-TB provides dense constraints requiring "balance in any sub-segment," which is more stable than PPO; (c) Decoupling allows for separate optimization of $p_\theta(Z|X)$ without corrupting LLM reasoning parameters.

2.  **Multi-dimensional Rewards (Contrastive Visual + Information Gain)**:
    -   **Function**: Enables the reward function to simultaneously measure "perceptual precision" and "reasoning utility," preventing the model from focusing solely on geometric IoU.
    -   **Mechanism**: Defines $R(z_{0:k}\top) = \left(\prod_{i=1}^k \frac{p_\phi^+(z_i)}{p_\phi^-(z_i)}\right) p_\phi(Y \mid z_{0:k}\top, X)$. Here, $p_\phi^+(z_i) = p_\phi(c_i \mid I_{r_i})$ is the visual likelihood of the caption within the cropped region, and $p_\phi^-(z_i) = p_\phi(c_i \mid I \setminus I_{r_i})$ is the likelihood in the complement region. $p_\phi$ is a frozen reward model initialized with the policy. The contrastive term $p_\phi^+/p_\phi^-$ under trajectory expectation is interpretable as reverse-KL distillation: $\mathbb{E}[\sum \log(p_\phi^+/p_\phi^-)] = \sum [D_{KL}(q_\theta^i \| p_\phi(\cdot|I\setminus I_{r_i})) - D_{KL}(q_\theta^i \| p_\phi(\cdot|I_{r_i}))]$, encouraging captions to align with privileged information in the crop while ignoring noise. The information gain term $p_\phi(Y|z_{0:k}\top, X)$ ensures chosen trajectories contribute informative value to the final answer $Y$.
    -   **Design Motivation**: Embedding "vision-grounded" and "reasoning-oriented" as independent constraints in the reward naturally suppresses reward hacking—accurate boxes with generic captions, or vice versa, will not yield high rewards.

3.  **Vicinal Geometric Shaping**:
    -   **Function**: Retains expert priors as "safety rails" to prevent the policy from wandering too far during exploration, without forcing 100% imitation, allowing the model to discover higher-utility perceptual behaviors within the expert's vicinity.
    -   **Mechanism**: Defines a symmetric Chamfer-IoU distance $d_{IoU}(A, B) = 1 - 0.5(IoU_{A\to B} + IoU_{B\to A})$, then defines an $\varepsilon$-neighborhood $\mathcal{B}_\varepsilon(E) = \{z_{0:k} \mid d_{IoU}(r_{1:k}, E) \le \varepsilon\}$ around the expert RoI set $E$. The shaping weight $\omega_\lambda(z_{0:k}, E) = \exp(-\lambda \mathbb{I}(z_{0:k} \notin \mathcal{B}_\varepsilon(E)))$ applies a penalty (intensity $\lambda$) only to trajectories outside the neighborhood, allowing reward $R$ to govern behavior within it. The final shaped reward $R_\lambda(z_{0:k}\top) = R(z_{0:k}\top) \omega_\lambda(z_{0:k}, E)$ is used for $\mathcal{F}$ in the Sub-TB objective.
    -   **Design Motivation**: Theorem 3.1 provides a TV distance upper bound, showing that $\lambda \to 0$ degenerates to standard MLE (losing geometric constraints) and $\lambda \to \infty$ degenerates to expert-guided RLVR (restricted by expert bias). Theorem 3.4 proves there exists $\lambda^\star$ making the bound strictly smaller than both baselines, meaning PFlowNet is strictly superior under ideal assumptions.

### Loss & Training
**Data Pipeline**: Uses Gemini-3-flash / GPT-4o as teachers to generate synthetic flow $Z_s$ via random expansion of expert RoIs. A verifier then samples answers under "no $Z_s$" and "with $Z_s$" conditions. Samples are partitioned via pass@k: $k=1$ is discarded as too simple; $k>1$ but $2\le k_{w/Z_s}\le 16$ enters the RFT set; $k_{w/o Z_s} > 16$ and $k_{w/Z_s} = 1$ enters the cold-start set. **Cold-start**: Standard SFT minimizing cross-entropy between $p_\theta(Z|X)$ and $Z_s$. **vRFT**: Combines the three key designs, calculating Sub-TB in parallel for $L$ sampled trajectories (sharing reward caches for sub-prefixes of each trajectory).

## Key Experimental Results

### Main Results
Base model: Qwen3-VL 8B. Evaluated on V* Bench (complex visual search), TreeBench (perception + reasoning tree-like evaluation), and MME-RealWorld-Lite (OCR/Remote Sensing/Charts/Surveillance/Autonomous Driving).

| Dataset | Metric | PFlowNet | Prev. SOTA / Baseline | Gain |
|---|---|---|---|---|
| V* Bench | Overall Acc | **90.6%** | Qwen3-VL 8B 77.5% | +13.1% vs base, New SOTA |
| TreeBench | Overall Acc | Qwen3-VL+13.1% / +10.4% | Qwen3-VL 8B | +10.4 vs base |
| MME-RealWorld-Lite | Overall Acc | **67.0%** | Baselines in 43–52% range | +21% vs Qwen3-VL 8B |
| TreeBench (Attributes) | Acc | 64.69 (example) | Most 50–60 | Significant lead |

Note: Table 2 shows that large models like InternVL3-78B / Qwen2.5-VL-72B only score 46.4% / 42.2% on TreeBench/MME-RealWorld-Lite. PFlowNet significantly outperforms 70B-scale models using only 8B parameters, indicating that performance stems from the training paradigm rather than parameter scaling.

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Full PFlowNet ($\lambda^\star, \varepsilon^\star$) | Tightest TV bound / SOTA | All three components included |
| $\lambda \to 0$ | $D_{TV} \to 1 - s_V$ | Degenerates to MLE; geometric prior lost |
| $\lambda \to \infty$ | $D_{TV} \to 1 - q$ | Degenerates to expert-guided RLVR; expert bias limit |
| $\varepsilon \to 0$ | Neighborhood shrinks to point, $q \to 0$ | Reward signal fails; bound loosens |
| Increasing $\varepsilon$ (while $\mathcal{B}_\varepsilon \subseteq \mathcal{S}_V$) | $q \uparrow$, bound tightens | Better when wider within effective domain |
| $\varepsilon > \sigma$ | Neighborhood exceeds $\mathcal{S}_V$ | Geometric guidance diluted; performance drops |

### Key Findings
- Probe experiments directly contradict the common assumption that "experts are most precise": accurate expert boxes result in lower accuracy than moderately expanded boxes, validating "tunnel vision."
- Theorems 3.1–3.4 provide provable improvements over MLE and expert-guided RLVR, provided that $\mathcal{B}_\varepsilon \subseteq \mathcal{S}_V$.
- Excellent performance-efficiency trade-off: The 8B model surpasses 78B-scale baselines, showing that the structured decomposition of perceptual flow + variational RL significantly improves sample efficiency.
- Favorable test-time scaling: Performance improves with larger sampling budgets, indicating $p_\theta(Z|X)$ learns a truly explorable distribution rather than a single point.

## Highlights & Insights
- Formalizing VGR as "latent variable posterior approximation" is the paper's most significant conceptual upgrade. By replacing "geometric alignment" with "distribution approximation," the "expert bias" problem becomes a tunable $\lambda$/$\varepsilon$ hyperparameter issue.
- Sub-TB provides dense supervision while maintaining the exploratory nature of GFlowNets, making it particularly suitable for long-chain perceptual behaviors. Migrating GFlowNet ideas from molecule generation to LVLM reasoning is a novel and natural bridge.
- The insight that the contrastive term $p_\phi^+/p_\phi^-$ in the multi-dimensional reward is equivalent to reverse-KL distillation is elegant, turning caption likelihood differences between "inside and outside the box" into a KL difference, with clear physical and optimization semantics.
- The philosophy of Vicinal Geometric Shaping ("experts are references, not targets") can be transferred to any RLHF scenario requiring a balance between expert priors and exploration, such as tool calling for code agents or robot policy distillation.

## Limitations & Future Work
- The theoretical assumptions are strong (Assumption 1/2, $d_{eff}$-regularity, etc.); it is uncertain whether complex LVLM distributions satisfy these. The bounds represent idealized limits.
- The Perceptual Flow currently only supports "box + caption" states; it needs extension for finer-grained perceptual behaviors (e.g., masks, point clouds, video frames).
- The data pipeline depends on strong teacher models (Gemini-3-flash / GPT-4o) to synthesize flows, creating a barrier for open-source replication. Cold-start data quality directly impacts final performance.
- $\lambda$ and $\varepsilon$ remain fixed hyperparameters without adaptive scheduling; the theorem proves the existence of an optimal $\lambda^\star$ but does not provide a method to find it.
- Multi-dimensional rewards require maintaining a reward model $p_\phi$, effectively doubling the GPU memory cost during training.

## Related Work & Insights
- **vs Look-Twice / VGR / TraceVL**: These works use expert geometry (e.g., GroundingDINO) as hard rewards and are limited by expert bias; PFlowNet treats experts as neighborhood references and lets the model self-learn optimal perception via a variational objective.
- **vs DeepSeek-R1 (RLVR Paradigm)**: R1 applies RLVR to verifiable rewards (math/code); PFlowNet extends RLVR to scenarios where "perceptual behavior cannot be directly verified," using contrastive likelihood + information gain instead of ground-truth rewards.
- **vs GFlowNets (Sub-TB)**: Original GFNs are used for discrete combinatorial object generation; this is one of the first works to introduce Sub-TB into LVLM multimodal reasoning.
- **vs Vicinal Risk Minimization**: Borrows "neighborhood shaping" from classic VRM but applies it to the trajectory space instead of the input space, providing a new RL regularization primitive.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes VGR as latent variable modeling, introduces GFlowNet Sub-TB, and designs vicinal shaping; a self-consistent new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on V*, TreeBench, and MME-RealWorld-Lite against large model baselines; however, some ablation details are in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the flow from "probe experiments → formalization → Sub-TB → rewards → shaping" is consistent.
- Value: ⭐⭐⭐⭐⭐ High impact for future grounded-reasoning LVLMs; both vicinal shaping and multi-dimensional rewards have strong transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](../../CVPR2026/reinforcement_learning/see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ICML 2026\] Flow-Equivariant World Models: Memory for Partially Observed Dynamic Environments](flow_equivariant_world_models_memory_for_partially_observed_dynamic_environments.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ICML 2026\] Coupled Variational Reinforcement Learning for Language Model General Reasoning](coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)

</div>

<!-- RELATED:END -->
