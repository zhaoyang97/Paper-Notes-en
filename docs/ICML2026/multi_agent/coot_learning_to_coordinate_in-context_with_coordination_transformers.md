---
title: >-
  [Paper Note] CoOT: Learning to Coordinate In-Context with Coordination Transformers
description: >-
  [ICML 2026][Multi-Agent][Ad-hoc Teamwork] The problem of "how to cooperate with unknown partners" is reframed from task-generalization to a partner-generalization in-context learning problem. A Decision Transformer is tr…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Ad-hoc Teamwork"
  - "In-Context Learning"
  - "Decision Transformer"
  - "Hidden-Utility Markov Games"
  - "Zero-Shot Coordination"
date: 2026-05-08
content_hash: 53b1815b456e9a25
---

# CoOT: Learning to Coordinate In-Context with Coordination Transformers

**Conference**: ICML 2026  
**arXiv**: [2506.23549](https://arxiv.org/abs/2506.23549)  
**Code**: https://coot-project.github.io/coot/ (with demo)  
**Area**: Multi-Agent Reinforcement Learning / Coordination / In-Context Learning  
**Keywords**: Ad-hoc Teamwork, In-Context Learning, Decision Transformer, Hidden-Utility Markov Games, Zero-Shot Coordination  

## TL;DR
The problem of "how to cooperate with unknown partners" is reframed from task-generalization to a partner-generalization in-context learning problem. A Decision Transformer is trained to predict best-response actions over cross-episode interaction trajectories, enabling the model to adapt to unseen partners within a few episodes without parameter updates.

## Background & Motivation

**Background**: In multi-agent reinforcement learning, cooperating with unknown partners is known as ad-hoc teamwork (AHT) or zero-shot coordination (ZSC). Mainstream approaches fall into three categories: self-play (SP) which learns fixed conventions; population-based methods (HSP, MEP) which train a recurrent policy using a diverse partner pool; and context-based meta-RL (PACE, PECAN, LIAM) which use encoders to compress recent trajectories into latents for the policy.

**Limitations of Prior Work**: SP converges to its own conventions and fails when meeting partners with different conventions. Population-based generalization remains limited by the coverage of the training pool, with significant performance drops for out-of-distribution partners. Online adaptation via fine-tuning requires thousands to millions of interactions, making it unfeasible for few-shot scenarios. Context-based meta-RL compresses interactions into fixed latents, losing temporal structures and often providing misleading representations for unseen partners; PACE/PECAN sometimes even underperform vanilla HSP.

**Key Challenge**: The essence of coordination is "online inference of a partner's hidden preferences/strategies." However, existing paradigms relegate this to the training phase or latent bottlenecks. No prior work has truly enabled models to make inferences at test time using full raw trajectories.

**Goal**: (1) Identify a partner-adaptation mechanism that requires neither gradient updates nor reliance on latent compression; (2) Demonstrate that "minimal interaction" is sufficient in realistic coordination scenarios like Overcooked and GRF.

**Key Insight**: Transfer in-context learning (ICL) from language/task generalization to partner generalization. Just as Transformers can infer tasks from few examples in a prompt, they should be able to infer the best response to a partner after seeing a few episodes in a prompt.

**Core Idea**: Use best-response behavior as a supervisory signal to train a Decision-Pretrained Transformer. It predicts the best-response action directly based on the "complete trajectories of the past $T$ episodes + current query state," ensuring partner-adaptation occurs entirely within the forward pass.

## Method

### Overall Architecture
The coordination problem is formulated as a Hidden-Utility Markov Game (HU-MG) $(\mathcal{S},\mathcal{A},\mathcal{T},R_t,\mathcal{R}_w)$. While the environment reward $R_t$ is shared, each partner $\pi^p_i$ has a private "hidden utility" $r^w_i\sim\mathcal{R}_w$ (defined as a linear combination of event features). Different $r^w_i$ induce distinct behavioral preferences in the same environment. During training, trajectories are collected for each triplet $(r^w_i, \pi^p_i, \pi^{br}_i)$ to form a dataset $\mathcal{D}$. At a query state $s_h$ with context $C$ (a sequence of $(s,a,r)$ from past episodes), the Transformer outputs an action distribution $\hat{p}_h(\cdot)=M_\theta(\cdot\mid s_h, C)$. The objective is to match the best-response action $\hat{a}=\pi^{br}_i(s_h)$. During deployment, parameters are frozen, and new trajectories are appended to a ring buffer context after each episode to achieve "adaptation while playing."

### Key Designs

1.  **HU-MG + Behavior-Preferring Data Generation**:
    *   **Function**: Construct a training dataset where partner behaviors are sufficiently diverse and each partner has an explicit best-response for comparison.
    *   **Core Idea**: First, define discrete environment events and express the reward space $\mathcal{R}_w$ as their linear combination. For each sampled $r^w_i$, use MAPPO to jointly train a behavior-preferring partner $\pi^p_i$ and its best response $\pi^{br}_i$. Calculate an event-level diversity score $d_i$ for all $\pi^{br}_i$ and select the top-$N$ pairs to form a training pool $\Pi_{train}$. For each pair, sample $T$ trajectories to form a fixed-length context $C$, and independently sample query states $s_h$ to form $(s_h, C, \hat{a})$ triplets.
    *   **Design Motivation**: The query state must be sampled **independently** of the context trajectories; otherwise, the model might degrade into simple trajectory completion rather than true "partner inference from context." Diversity filtering ensures the model sees a broad spectrum of partner behaviors during training for better test-time ICL generalization.

2.  **Coordination Transformer Training**:
    *   **Function**: Train a GPT-2 backbone to recognize partner styles from cross-episode history and predict the corresponding best-response action.
    *   **Core Idea**: Flatten the context into a token sequence $[\tau_1, \tau_2, \ldots, \tau_T, s_{query}]$ (where each $\tau$ is a full $(s,a,r)$ episode sequence) and predict the action distribution at the final position. The training objective is the cross-entropy loss $\mathcal{L} = -\log \hat{p}_h(\hat{a})$, where $\hat{a}=\pi^{br}_i(s_h)$. The context window is set to 5 episodes, and context masking is used during training to encourage reliance on recent trajectories.
    *   **Design Motivation**: Unlike architectures like PEARL/PACE that compress trajectories into fixed latents, letting the Transformer observe raw $(s,a,r)$ sequences preserves full temporal structure. This avoids coordination-irrelevant noise from dominating the representation, which often happens when reconstruction losses weigh all features equally.

3.  **No-Prior Online Deployment**:
    *   **Function**: Enable the model to adapt to unseen partners at test time solely through cumulative context, without any prior hints.
    *   **Core Idea**: Initialize context $C$ with $T$ empty trajectories (strict cold start). At each step $t$, sample action $a_t\sim\hat{p}_t$ where $\hat{p}_t=M_\theta(\cdot\mid s_h, C)$ and record $(s_t, a_t, r_t)$. At the end of an episode, append the full trajectory $\tau=(s_t, a_t, r_t)_{t=1}^Z$ to the context buffer while removing the oldest. CoOT repeats this process cross-episode to refresh its strategy.
    *   **Design Motivation**: A no-prior cold start is a rigorous test of coordination capability; methods using retrieval or pre-filled contexts often mask true generalization. Using a ring buffer rather than an infinite append ensures the context reflects the partner's **current** behavior (partners may change styles).

### Training & Evaluation
CoOT utilizes a GPT-2 backbone. It is supervised using trajectories from 36 best-response policies (each corresponding to a partner in the behavior-preferring HSP training distribution). The evaluation protocol follows Wang et al. (2024): 10 evaluation partners per layout (15 for Coord. Ring Multi-recipe), selected using Best-Response Diversity (BR-div). Each partner is run for 50 episodes, with CoOT continuously updating its context.

## Key Experimental Results

### Main Results

Returns and Best-Response Proximity (BR-prox = Agent Reward / Best-Response Reward) across five Overcooked layouts:

| Method | Coord. Ring | Coord. Ring Multi-recipe | Counter Circ. | Asymm. Adv. | Bothway Coord. | Overall Return | Overall BR-prox |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BC | 26.24 / 0.31 | 8.97 / 0.10 | 10.79 / 0.11 | 108.83 / 0.53 | 98.99 / 0.94 | 50.76 | 0.40 |
| BC-RNN | 28.07 / 0.33 | 21.98 / 0.25 | 15.15 / 0.14 | 105.52 / 0.51 | 93.59 / 0.91 | 52.86 | 0.42 |
| MEP | 40.30 / 0.47 | 16.64 / 0.19 | 1.89 / 0.02 | 127.44 / 0.61 | 22.76 / 0.20 | 41.81 | 0.30 |
| HSP | 41.10 / 0.49 | 29.35 / 0.33 | 21.37 / 0.23 | 134.01 / 0.63 | 54.99 / 0.53 | 56.16 | 0.44 |
| HSP-ft | 41.30 / 0.49 | 29.24 / 0.33 | 21.71 / 0.22 | 133.59 / 0.63 | 55.81 / 0.54 | 56.33 | 0.44 |
| HSP-meta | 29.84 / 0.35 | 30.21 / 0.34 | 3.28 / 0.03 | 113.16 / 0.54 | 20.44 / 0.20 | 40.19 | 0.29 |
| PACE | 33.94 / 0.40 | 4.43 / 0.05 | 2.41 / 0.02 | 124.06 / 0.59 | 14.90 / 0.15 | 35.95 | 0.24 |
| **CoOT** | 38.30 / 0.47 | **45.96 / 0.50** | **28.28 / 0.30** | 129.48 / 0.62 | **101.93 / 0.96** | **68.79** | **0.57** |

CoOT significantly improved performance on the coordination-dense Coord. Ring Multi-recipe layout from 30.21 (HSP-meta) to 45.96, and the Overall BR-prox increased from 0.44 (strongest baseline) to 0.57 (+30%).

### Controlled Experiments: GRF 3v1 and Human Evaluation

| Experiment | Metric | BC | MEP | HSP | HSP-ft | **CoOT** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GRF 3v1 (goal rate / 200 steps) | Goal Rate ↑ | 1.97 | 1.11 | 1.46 | 1.52 | **2.50** |
| Human Eval (36 participants) — Return | ↑ | 51.0 | 53.0 | 40.5 | — | **63.5** |
| Human Eval — Collaboration | ↑ | 2.0 | 3.4 | 2.7 | — | **4.0** |
| Human Eval — Adaptivity | ↑ | 1.8 | 2.9 | 2.2 | — | **3.1** |
| Human Eval — Best Agent Vote (/36) | ↑ | 5 | 8 | 5 | — | **18** |

### Key Findings
*   **Few-shot Adaptation Speed**: Results show CoOT's BR-prox rises significantly within the first 15 episodes and then plateaus. In contrast, PACE shows almost no improvement, HSP-meta converges at a lower level, and HSP-ft remains static, proving raw-trajectory Transformer inputs are more effective for few-shot adaptation than gradient-based latents.
*   **Non-stationary Adaptation**: When partners switch styles mid-session, CoOT recovers to baseline performance within 3.67 episodes (maximum 6), demonstrating that the context ring buffer effectively enables "learning the new while forgetting the old."
*   **Context Length 5 is the Sweet Spot**: While longer contexts provide more info, old trajectories can become misleading after 5 episodes as the partner's behavior changes in response to CoOT. Context masking only partially mitigates this trade-off.
*   **Compressing to Latent is an Anti-pattern**: HSP-meta and PACE generally underperform vanilla HSP. PEARL-style reconstruction losses weigh all features equally, drowning out coordination-relevant signals. Attention over raw tokens consistently yields better results.
*   **Data Diversity Over Quantity**: Reducing the partner pool from 36 to 12 leads to a 24.0% drop for CoOT and a 41.5% drop for BC. Halving trajectories results in a 28.6% drop for CoOT and a 32.1% drop for BC, suggesting CoOT is more robust to data scale due to ICL's sample efficiency.

## Highlights & Insights
*   **Coordination as Partner-Generalization**: This reframing is more significant than the method itself. Applying Decision-Pretrained Transformer (DPT) logic to "partners" instead of "tasks" achieves SOTA with minimal architecture changes, suggesting the correct application of ICL in RL has been overlooked.
*   **Raw Trajectories > Latents**: The PACE/PECAN/LIAM approach is empirically challenged: fixed-size latents destroy key temporal signals for coordination. This serves as a caution for meta-RL approaches using encoders.
*   **HU-MG + Best-Response Supervision**: Best-response outcomes from MAPPO provide partner-specific oracles. This supervisory signal is cleaner than sparse RL rewards, allowing for stable Supervised Learning (SL) rather than unstable RL.
*   **Human Evaluation Success**: Winning 18/36 votes in human-AI interaction is rare in AHT literature. Integrating N=144 open-ended responses with Gemini thematic coding transforms "adaptivity" from subjective intuition into analytical evidence.

## Limitations & Future Work
*   Coordination relies solely on behavioral inference without explicit communication channels; introducing language/gesture communication is an obvious extension.
*   Experiments are limited to coordination benchmarks and do not yet address rich perception and control in embodied scenarios (e.g., robotics).
*   The cold-start no-prior protocol, while rigorous, sacrifices utility. Allowing context reuse from similar partners could reduce initial coordination costs.
*   The method is tailored for cooperative shared-reward settings; the definition of best-response becomes ambiguous in competitive or mixed-motive scenarios.
*   The training partner pool (36 HSP-trained partners) remains a product of population-based generation, which sets an upper bound on CoOT's diversity.

## Related Work & Insights
*   **vs HSP / MEP (population-based)**: Both use the same HSP partner pool. The difference lies in the paradigm: recurrent policy + shared reward training vs. Transformer + best-response supervision. CoOT's +12 return improvement shows the paradigm shift is key.
*   **vs PACE / PEARL-style Meta-RL**: CoOT rejects encoder bottlenecks and attends directly to raw trajectories, whereas PACE underperforms vanilla HSP.
*   **vs Decision-Pretrained Transformer (DPT)**: Shares the GPT-2 architecture but shifts the objective from general task best-response to partner best-response, and replaces reward-driven signals with best-response action supervision.
*   **vs Algorithm Distillation / In-Context RL**: While both treat RL as sequence modeling, CoOT specifically distills the "ability to react to partner behavior," which is more direct for ad-hoc teamwork than distilling a general learning algorithm.

## Rating
*   Novelty: ⭐⭐⭐⭐ Reframing ICL from task to partner generalization is simple but provides a fresh perspective.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive testing across 5 layouts, 7 baselines, human trials (N=36), non-stationarity tests, and thorough ablations.
*   Writing Quality: ⭐⭐⭐⭐ Clear method section and clean formulation of HU-MG, including specialized failure analysis for HSP-meta.
*   Value: ⭐⭐⭐⭐ Provides a plug-and-play solution for human-AI collaboration without fine-tuning, backed by credible human interaction data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM](learning_multi-agent_coordination_via_sheaf-admm.md)
- [\[ICML 2026\] MASPOB: Multi-Agent Prompt Optimization via GNN Surrogate + LinUCB + Coordinate Ascent](maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ACL 2026\] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](../../ACL2026/multi_agent/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)

</div>

<!-- RELATED:END -->
