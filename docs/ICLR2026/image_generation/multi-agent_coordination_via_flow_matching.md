---
title: >-
  [Paper Note] Multi-agent Coordination via Flow Matching
description: >-
  [ICLR 2026][Image Generation][Multi-agent coordination] This paper proposes MAC-Flow, which first learns a centralized joint behavior distribution via Flow Matching…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Multi-agent coordination"
  - "Flow Matching"
  - "Offline MARL"
  - "IGM policy distillation"
  - "Decentralized execution"
date: 2026-05-08
content_hash: be2fbd21d920e4fe
---

# Multi-agent Coordination via Flow Matching

**Conference**: ICLR 2026
**arXiv**: [2511.05005](https://arxiv.org/abs/2511.05005)
**Code**: N/A
**Area**: Image Generation
**Keywords**: Multi-agent coordination, Flow Matching, Offline MARL, IGM policy distillation, Decentralized execution

## TL;DR

This paper proposes MAC-Flow, which first learns a centralized joint behavior distribution via Flow Matching, then distills it into decentralized single-step policies through IGM (Individual-Global-Max) decomposition combined with Q-value maximization for behavior-regularized training. Evaluated across 4 benchmarks, 12 environments, and 34 datasets, MAC-Flow achieves approximately 14.5× inference speedup over diffusion-based methods while maintaining coordination performance comparable to diffusion policies.

## Background & Motivation

**Background**: Offline multi-agent reinforcement learning (offline MARL) requires learning coordinated policies from pre-collected datasets without online environment interaction. Existing approaches broadly fall into two categories: generative methods based on diffusion models (e.g., MADiff, DoF) that model joint action distributions through iterative multi-step denoising, and discriminative methods based on Gaussian policies (e.g., OMAC, CFCQL, ICQ) that produce actions via simple parameterizations with fast inference.

**Limitations of Prior Work**: Both categories suffer from critical drawbacks. Diffusion policies are highly expressive and can model multimodal joint behaviors, but inference requires 50–200 denoising steps; DoF alone requires approximately 60 hours of training on SMAC, making it completely unsuitable for real-time decision-making. Gaussian policies require only a single forward pass, but Gaussian distributions are inherently unimodal and cannot capture the complex structure of multi-agent systems where "multiple equivalent coordination solutions exist for the same state," leading to brittle behavior under multi-agent interaction.

**Key Challenge**: Multi-agent coordination simultaneously demands (i) rich representational capacity for diverse joint behaviors present in offline data, and (ii) efficient execution in real-time environments. These two requirements constitute a fundamental performance–efficiency trade-off that prior methods are forced to navigate by sacrificing one for the other.

**Goal**: How can one preserve expressive multimodal representational capacity close to diffusion models while achieving inference speed comparable to Gaussian policies? Three sub-problems must be addressed: (1) how to efficiently learn rich representations of joint behaviors; (2) how to decompose joint representations into independent per-agent policies; and (3) how to retain coordination information throughout the decomposition.

**Key Insight**: The authors observe that Flow Matching provides a unified framework—it is comparable to diffusion models in expressiveness, but has a more direct training objective (directly matching the velocity field rather than learning a denoising process), smoother probability flows, and is better suited for distillation. More critically, Flow Matching distillation naturally combines with IGM decomposition: first learn a centralized joint flow, then distill it into independent single-step per-agent policies via $W_2$ distance constraints, while using Q-value maximization to steer policies toward high-return regions.

**Core Idea**: Construct a centralized representation of joint behaviors via Flow Matching, then compress it into decentralized single-step policies through IGM decomposition + $W_2$ distillation + Q-value maximization, thereby simultaneously achieving expressiveness and inference efficiency within a unified framework.

## Method

### Overall Architecture

MAC-Flow adopts a two-stage Centralized Training with Decentralized Execution (CTDE) design. The input is an offline multi-agent interaction dataset $\mathcal{D} = \{(s, o_i, a_i, r)\}$, and the output is an independent single-step policy $\pi_i(a_i | o_i)$ for each agent $i$, requiring only a single forward pass to generate actions.

- **Stage 1 (Joint Flow Learning)**: Conditioned on the global state $s$, a centralized joint policy $\pi_{\text{joint}}(\mathbf{a} | s)$ is trained via Flow Matching, where $\mathbf{a} = (a_1, \dots, a_n)$ is the joint action of all agents. This stage learns from offline data through behavior cloning (BC) to build a rich representation of the joint behavior distribution.
- **Stage 2 (IGM Distillation + RL Optimization)**: The centralized flow model is distilled into $n$ independent single-step policy networks, each policy $\pi_i$ conditioned only on local observation $o_i$. Distillation is guided by the IGM principle and combined with behavior-regularized actor-critic training for Q-value maximization, ensuring that the factored policies retain coordination ability while being optimized toward high-return directions.

### Key Designs

1. **Flow Matching for Joint Behavior Distribution Modeling**:

    - **Function**: Learn a probabilistic flow mapping from a simple prior distribution (e.g., standard Gaussian) to the joint action distribution.
    - **Mechanism**: Define a continuous-time interpolation path $x_t = (1-t) x_0 + t x_1$, where $x_0 \sim \mathcal{N}(0, I)$ is noise and $x_1$ is the joint action from the dataset. A velocity field network $v_\theta(x_t, t, s)$ is trained to match the path tangent vector $x_1 - x_0$, with a simple mean-squared-error loss $\|v_\theta(x_t, t, s) - (x_1 - x_0)\|^2$. At inference time, starting from $x_0 \sim \mathcal{N}(0, I)$, joint action samples are generated by integrating along the velocity field for 10 steps. The authors empirically find that performance improves rapidly as the number of flow steps increases from 1 to 10, but saturates beyond 10 steps—in stark contrast to the 50–200 steps required by diffusion models.
    - **Design Motivation**: Compared to diffusion models, Flow Matching has a more direct training objective, fewer hyperparameters, and smoother probability flows. This smoothness is critical for subsequent distillation—since distillation fundamentally approximates a complex model with a simpler one, smooth flow distributions are more amenable to single-step policy approximation than the step-wise denoising distributions of diffusion models.

2. **IGM Decomposition and Policy Distillation**:

    - **Function**: Decompose the centralized joint policy into independent single-step per-agent policies while preserving decomposition consistency.
    - **Mechanism**: Drawing on the Individual-Global-Max (IGM) principle from QMIX/QTRAN—if the global Q-value $Q_{\text{tot}}$ can be decomposed into a combination of individual Q-values $Q_i$, and the combination of individually optimal actions equals the globally optimal joint action, then each agent can independently select its own optimal action. Concretely, distillation trains each $\pi_i$ to approximate the marginal component of the joint flow corresponding to agent $i$, while Q-value guidance steers policies toward high-return regions. The distillation loss measures the discrepancy between the joint distribution and the product distribution $\prod_i \pi_i$ using the $W_2$ (Wasserstein-2) distance.
    - **Design Motivation**: Pure BC-style generative models can only reproduce the dataset distribution and cannot discover rare but high-return coordination patterns. The combination of IGM and Q-value maximization allows policies to actively shift toward better joint actions while remaining close to the data distribution. The authors verify this in a toy experiment: the dataset is dominated by suboptimal coordination patterns $(0,1)$ and $(1,0)$ (each with reward +1), with the optimal pattern $(1,1)$ (reward +2) being very rare. Pure BC Flow only reproduces the suboptimal patterns, whereas adding IGM + Q-value maximization enables the policy to successfully shift toward $(1,1)$.

3. **$W_2$ Theoretical Guarantee and Lipschitz Constraint**:

    - **Function**: Provide a theoretical upper bound on performance degradation during the decomposition process.
    - **Mechanism**: Proposition 4.2 derives the $W_2$ distance upper bound between the joint flow distribution and the factored product distribution; Proposition 4.3, under the assumption that $Q_{\text{tot}}$ is $L$-Lipschitz, converts this distributional discrepancy into a value function gap upper bound $|V_{\text{joint}} - V_{\text{factored}}| \leq L_Q \cdot W_2$. The authors verify in the toy experiment (Figure 3) that the value gap during training strictly remains below the theoretical envelope $L_Q \cdot W_2$ and decreases in tandem with the distillation loss.
    - **Design Motivation**: Purely empirical distillation lacks predictability—it is unclear at what point decomposition will cause coordination collapse. The $W_2$ upper bound provides a monitorable signal: as long as the $W_2$ distillation loss is sufficiently small, the value function gap is bounded.

### Loss & Training

- **Stage 1 (Flow Training)**: Conditional Flow Matching loss $\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\|v_\theta(x_t, t, s) - (x_1 - x_0)\|^2$, trained via BC on the offline dataset.
- **Stage 2 (Distillation + RL)**: Each agent's actor optimizes $\max_{a_i} Q_i(o_i, a_i) - \alpha \cdot D_{\text{KL}}(\pi_i \| \pi_{\text{ref}})$, where $\pi_{\text{ref}}$ is derived from the marginalized flow distribution through distillation and $\alpha$ controls the strength of behavior regularization; the critic aggregates individual $Q_i$ values into $Q_{\text{tot}}$ through an IGM mixing network, trained via offline TD learning.
- **Training Efficiency**: MAC-Flow requires 1–5 hours of training on SMAC, compared to approximately 60 hours for DoF (a diffusion-based method); on MA-MuJoCo, training takes only 40–100 minutes, comparable to baselines such as OMIGA and ICQ.

## Key Experimental Results

### Main Results

Evaluated on four benchmarks—SMAC v1, SMAC v2, MPE, and MA-MuJoCo—covering discrete and continuous action spaces, 3 to 10 agents, and multiple data quality levels (medium, medium-replay, medium-expert).

| Dimension | Diffusion Methods (DoF/MADiff) | Gaussian Methods (OMAC/CFCQL/ICQ) | **MAC-Flow** |
|-----------|-------------------------------|-----------------------------------|-------------|
| Inference speed | Slow (50–200 denoising steps) | Fast (single step) | **Fast (single step)** |
| Inference speedup | 1× (baseline) | ~14.5× | **~14.5×** |
| SMAC v1 performance | Best (DoF leads in most envs) | Moderate | **Comparable to DoF; significantly better than Gaussian** |
| SMAC v2 performance | DoF leads | Limited | **Slightly below DoF (high-stochasticity environments)** |
| MA-MuJoCo performance | Comparable to MADiff | Baseline level | **On par with MADiff** |
| MPE performance | — | Baseline level | **Competitive** |
| Training time (SMAC) | ~60 hours | 1–3 hours | **1–5 hours** |
| Online fine-tuning support | Not supported | Supported | **Supported** |

Key numerical comparisons:
- MAC-Flow achieves average performance on SMAC v1 comparable to DoF, but falls slightly below DoF in high-stochasticity SMACv2 scenarios; the authors attribute this to greater pressure on the decomposition assumption from high-variance joint action spaces.
- On continuous control (MA-MuJoCo), MAC-Flow matches MADiff and significantly outperforms the autoregressive method MADT.
- Training is an order of magnitude faster than diffusion-based methods, and the model supports seamless transition from offline to online fine-tuning (Figure 4, RQ3).

### Ablation Study

| Ablation Configuration | Performance Change | Analysis |
|-----------------------|-------------------|----------|
| Full MAC-Flow | Baseline performance | Flow learning + IGM distillation + Q maximization work synergistically |
| Remove IGM (pure BC distillation) | Significant degradation | BC alone cannot shift toward rare high-return coordination patterns |
| Remove Q maximization | Performance drops | Policy degrades to simple fitting of the data distribution |
| Flow steps 1→4→10 | Rapid improvement then saturation | 10 steps is sufficient; 20 steps yields negligible additional gain |
| Flow steps 10→20 | Marginal improvement | Demonstrates that flow is far more robust to step count than diffusion |
| Diffusion steps 50→100→200 | Continuous improvement | DoF performance depends heavily on many denoising steps |
| Number of agents 3→5→8→10 (SMAC) | Training time grows linearly | MABCQ: 1h→2h; DoF: 48h→60h; MAC-Flow: 1.5h→3.5h |
| Number of agents 3→40 (landmark) | Performance remains stable | Coordination maintained with up to 40 agents in Appendix H.4 landmark covering experiment |

### Key Findings

- **IGM is the core contribution, not Flow Matching per se**: Figure 7 shows that using Flow Matching alone is not the primary driver of performance gains; the true improvement stems from the synergy of IGM decomposition + Q-value maximization with flow distillation.
- **XOR failure mode**: In the XOR environment (Appendix H.6), where optimal joint actions require agents to make opposing choices, IGM decomposition is mathematically incapable of maintaining consistency. The joint flow correctly learns two disconnected high-density modes, but the distilled factored policy degenerates to near-uniform distributions—a fundamental limitation of the method.
- **Interaction intensity experiment**: In Appendix H.7, the authors construct a payoff game with controllable interaction intensity $\zeta \in [0, 1]$. Results show that $W_2$ discrepancy increases monotonically with interaction intensity; MAC-Flow is nearly lossless when interactions are fully decomposable, while noticeable degradation occurs when they are fully non-decomposable.
- **Robustness to data quality**: MAC-Flow performs consistently well across medium, medium-replay, and medium-expert data quality levels, benefiting from Q-value maximization's ability to correct for dataset bias.

## Highlights & Insights

- **The trinity of Flow + IGM + Q-value maximization**: The design is elegant in that all three components are indispensable—Flow provides expressiveness, IGM provides factorizability guarantees, and Q-value maximization compensates for the limitation of BC-style generative models that can only replicate the data distribution. This combination is more principled than "diffusion + distillation," because the $W_2$ distillation loss of Flow Matching can directly interface with the theoretical constraints of IGM decomposition.
- **Closed theory–experiment loop**: Propositions 4.2–4.3 establish theoretical upper bounds, and the toy experiment in Figure 3 directly verifies that the value gap consistently falls below the theoretical envelope—this closed loop between theory and experiment is more convincing than purely empirical or purely theoretical works.
- **Highly practical training-to-deployment narrative**: Train for 1–5 hours → deploy with single-step inference → support online fine-tuning. This pipeline is clear, actionable, and deployment-friendly in industrial settings.
- **Transferable design paradigm**: The paradigm of "first learn an expressive generative model → distill into a task-efficient execution policy → use constrained optimization to preserve key structure" is generalizable to robot manipulation (distilling diffusion policies into lightweight policies), autonomous driving planning, and other domains.

## Limitations & Future Work

- **IGM factorizability assumption is a hard constraint**: When optimal joint behaviors are fundamentally non-decomposable (e.g., XOR coordination), the method fails. This is not an engineering issue but a theoretical limit of the IGM principle. Possible improvements include introducing relaxed IGM formulations (e.g., QTRAN's additive decomposition) or conditional decomposition.
- **Performance gap in high-stochasticity SMACv2 scenarios**: In high-variance joint action spaces, the factored policy cannot fully preserve the expressiveness of the diffusion policy. A potential direction is to introduce value-gradient-based test-time corrective refinement during inference.
- **Primarily offline evaluation**: Although Figure 4 demonstrates online fine-tuning capability, the main experiments remain in the offline setting. Performance in scenarios requiring dynamic adaptation to teammate changes (ad-hoc teamwork) or opponent distribution shift remains unknown.
- **Absence of open-source code**: No public code is currently available, making reproducibility difficult. Reviewers also noted that direct comparison with "Graph Diffusion for Robust Multi-Agent Coordination" could not be conducted due to unavailable code.

## Related Work & Insights

- **vs DoF (Diffusion for Offline MARL)**: DoF slightly outperforms MAC-Flow on high-stochasticity tasks such as SMACv2, but the 50–200 denoising steps required for inference render it unsuitable for real-time scenarios. MAC-Flow trades a minor performance reduction for a 14.5× inference speedup—a worthwhile trade-off in most practical applications.
- **vs MADiff**: MADiff achieves comparable performance to MAC-Flow on continuous control tasks (MA-MuJoCo, MPE), but as a pure BC-style generative model, it lacks Q-value maximization capability and may degrade under poor data quality.
- **vs MADT (Autoregressive)**: Autoregressive policies generate actions sequentially per agent, introducing artificial sequential dependencies. MAC-Flow outperforms MADT across all datasets, as Flow's parallel generation + IGM decomposition avoids sequential error accumulation.
- **vs OMAC/CFCQL/ICQ (Gaussian)**: Gaussian policies cannot model multimodal coordination; MAC-Flow significantly improves coordination quality while maintaining the same inference speed.

**Insights**: This work demonstrates that generative models in multi-agent RL are more than "better BC"—combined with value decomposition frameworks (IGM), they can achieve efficient decentralized execution while preserving expressiveness. This idea generalizes to single-agent offline RL (e.g., distilling diffusion policies into single-step policies while retaining multimodal capacity) and applications such as multi-arm robotic coordination.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The trinity of Flow Matching + IGM distillation + Q-value maximization is novel in MARL; however, the core concept of "generative model → distillation" is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 benchmarks × 12 environments × 34 datasets, supplemented during rebuttal with scalability experiments (40 agents), failure mode analysis (XOR), and interaction intensity studies, achieving exceptional coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear; the two-stage pipeline is well-described; the theoretical section provides bounded degradation guarantees without overclaiming.
- **Value**: ⭐⭐⭐⭐ — Addresses the practical bottleneck of expressiveness–efficiency trade-off in offline MARL; the 14.5× inference speedup has clear engineering value. Reviewer scores of 6/6/4 indicate some community disagreement on methodological novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)

</div>

<!-- RELATED:END -->
