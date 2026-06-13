---
title: >-
  [Paper Note] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning
description: >-
  [ICML2026][Reinforcement Learning][Multi-Agent RL] This paper proposes LMAC—leveraging LLMs to design executable communication protocol code offline for cooperative MARL. Based on the "state reconstructability" metric…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Multi-Agent RL"
  - "Cooperative Communication"
  - "LLM Protocol Design"
  - "CTDE"
  - "QMIX"
date: 2026-05-08
content_hash: 0c9cddf1d58df7f5
---

# LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning

**Conference**: ICML2026  
**arXiv**: [2605.18077](https://arxiv.org/abs/2605.18077)  
**Code**: https://saaangjun.github.io/LMAC/  
**Area**: reinforcement_learning  
**Keywords**: Multi-Agent RL, Cooperative Communication, LLM Protocol Design, CTDE, QMIX

## TL;DR
This paper proposes LMAC—leveraging LLMs to design executable communication protocol code offline for cooperative MARL. Based on the "state reconstructability" metric, it performs two rounds of feedback iterations (first improving reconstruction accuracy, then reducing cross-agent imbalance). It significantly outperforms communication baselines such as TarMAC/SMS/T2MAC/MASIA on benchmarks like SMAC-Comm, LBF, GRF, and SMACv2, even exceeding the QMIX+State upper bound in some scenarios.

## Background & Motivation
**Background**: Cooperative MARL under the CTDE framework widely uses communication to mitigate partial observability. Early CommNet variants utilize broadcasting, TarMAC employs attention weighting, SMS uses Shapley values for scoring, T2MAC utilizes evidence fusion, and MASIA learns latent representations for "reconstructing states" to broadcast.

**Limitations of Prior Work**: These methods implicitly assume that "sending the message is sufficient," but in practice, many messages are redundant or miss critical information. The paper provides a counterexample from a specific SMAC scenario: an Overseer directly observes enemy positions, while other Banelings have no visibility. After 2M steps of training for MASIA and FullComm, reconstruction errors for enemy positions remain high; the variance of reconstruction errors across different agents is also large, meaning some agents "know" while others are still guessing. This leads to scattered enemy position estimates by Banelings, preventing synchronized attacks.

**Key Challenge**: Designing efficient communication requires identifying "which observation dimensions are critical for global state reconstruction and which are redundant"—this is a semantic-level task understanding problem that gradient optimization struggle to learn automatically. On the other hand, allowing LLMs to act directly as agents to generate messages online (e.g., Li et al., 2024) involves high token costs and is only suitable for text-world interfaces.

**Goal**: (i) Use LLMs to generate "communication protocols" as executable code in a one-time manner, eliminating LLM calls during training and execution; (ii) Ensure LLM iterative correction signals come from real RL replay data rather than LLM-hallucinated feedback; (iii) Quantify "protocol quality" into a differentiable metric—state reconstructability—to drive a Reflexion-style feedback loop.

**Key Insight**: Treat the LLM as a "protocol designer" rather than an "online message generator." The LLM inputs include task descriptions and the natural language semantics of each observation dimension, while the output is Python code mapping $\tau_t^i$ to message $m_t^i$. Quality is determined by running an auxiliary decoder on an offline buffer.

**Core Idea**: Driving LLM self-correction using two criteria: "whether state reconstruction becomes more accurate after adding messages" and "whether reconstruction becomes more uniform across agents," reducing protocol design to a code generation task that converges in two steps.

## Method

### Overall Architecture
LMAC splits MARL training into "protocol design" and "policy learning":

1.  **Offline Protocol Design Phase**: Run QMIX to collect a small buffer $\mathcal{B}$ (5k trajectories). First, use the LLM with task description $\mathcal{I}_T$ and protocol design instructions $\mathcal{I}_P$ to generate the initial protocol $f_C^{(0)}$. Then, train an auxiliary decoder $D_\phi^{(k)}$ to attempt global state reconstruction using each agent's trajectory (with or without messages). Obtain the SAI metric based on "whether reconstruction falls within threshold $\alpha$," and convert SAI into two types of linguistic feedback to drive $f_C^{(1)}$ (improving accuracy) and $f_C^{(2)}$ (reducing cross-agent variance). The final result is a three-part protocol $f_C=(f_C^{(0)},f_C^{(1)},f_C^{(2)})$.
2.  **Online CTDE Training Phase**: Execute $f_C$ as a fixed function—each agent feeds its local trajectory to obtain three message segments $m_t^i=(m_t^{i,(0)},m_t^{i,(1)},m_t^{i,(2)})$, which then pass through an encoder $\mathrm{Enc}_\psi$ to produce a latent representation $z_t^i$. This is fed into the individual utility $Q^i(\tau_t^i,z_t^i)$ and combined by QMIX into $Q_{tot}$ for TD learning. The encoder-decoder is simultaneously trained to reconstruct the global state of the current batch and predict SAI, while a cycle-consistency regularizer suppresses redundant features.

### Key Designs

1.  **State Awareness Index (SAI) as a Protocol "Stethoscope"**:
    - **Function**: Quantifies "message utility" into a 0/1 signal, allowing the LLM to perform self-reflection based on replay data.
    - **Mechanism**: For each agent $i$, state dimension $d$, and time $t$, the auxiliary decoder reconstructs the state under "message" and "no message" conditions: $\hat{s}_{1,d,t}^i = D_\phi^{(k)}(\tau_t^i, m_t^{i,(k)}, i)|_d$ and $\hat{s}_{0,d,t}^i = D_\phi^{(k)}(\tau_t^i, \mathbf{0}, i)|_d$. Defining $\chi_{l,d,t}^{i,(k)} = \mathbb{I}\big[\|\hat{s}_{l,d,t}^i - s_{d,t}\|^2 \le \alpha\big]$. Averaging over $t$ gives the "reconstruction success rate of agent $i$ for dimension $d$," and variance over $i$ gives the "knowledge imbalance across agents." These statistics serve as feedback sources for steps 1 and 2, respectively.
    - **Design Motivation**: Instead of having the LLM evaluate its own output (prone to self-consistent hallucinations), RL data is used for "scoring." This makes feedback signals both objective and inexpensive—a small decoder replaces the costly interface of "calling LLM at every step."

2.  **Two-step Reflexion Protocol Refinement**:
    - **Function**: Transforms the LLM's Reflexion loop into a version with "clear goals for each round" to avoid ambiguous feedback directions.
    - **Mechanism**: Replaces the static feedback instruction $\tilde x$ in original Reflexion with step-specific $\tilde x^{(k+1)}$. $k=0$ generates a "minimal message" version; $k=1$ uses $\mathbb{E}_t[\chi_{1,d,t}^{i,(0)}]$ to construct feedback sentences identifying "which agent fails to reconstruct which state dimension," prompting the LLM to supplement the protocol; $k=2$ uses $\mathrm{Var}_i[\chi_{1,d,t}^{i,(1)}]$ to point out "which dimensions remain inconsistent across agents," prompting the LLM to introduce shared anchors or IDs. Each step includes data analysis and feedback generation instructions.
    - **Design Motivation**: Observations showed that iterations beyond two steps yield diminishing yields (see Appendix D.1). Separating goals—"accuracy" first, then "uniformity"—avoids the LLM simultaneously optimizing conflicting objectives.

3.  **Meta-cognitive Latent Representation + cycle-consistency Regularizer**:
    - **Function**: Compresses LLM-designed offline messages into "task-relevant, reconstructible" latent features $z_t^i$ during CTDE training.
    - **Mechanism**: Encoder $z_t^i = \mathrm{Enc}_\psi(\tau_t^i, m_t^i)$; Decoder $\mathrm{Dec}_\psi$ is trained to reconstruct global state $s_t$ and predict SAI $\chi_{d,t}^i$ using ground truth as supervision. This is *meta-cognitive*—the representation must be "accurate" and "know its own inaccuracies." A cycle-consistency loss is added: $\hat z_t^i = \mathrm{Enc}_{c,\psi}(\mathrm{Dec}_\psi(z_t^i))$ must satisfy $\hat z_t^i \approx z_t^i$.
    - **Design Motivation**: Raw messages may be redundant or noisy; compressing to $z$ before individual utility serves as an SAI-supervised bottleneck, standardizing the criteria for "useful" vs. "memorable" communication.

### Loss & Training
QMIX standard TD loss handles the value function. The encoder-decoder is trained with three additional terms: (i) global state reconstruction loss, (ii) SAI prediction loss, and (iii) cycle-consistency loss. For the LLM, the best $\alpha$ threshold is selected for protocols at $k=0,1,2$. The default LLM is gpt-4.1-2025-04-14, though performance remains robust across GPT-mini/o1-mini/Claude/Gemini.

## Key Experimental Results

### Main Results
Covering four benchmarks with 5 seeds per task: SMAC-Comm (including large-scale `2o_20b_vs_2r`), LBF (foraging), GRF (football), and SMACv2 (random unit combinations).

| Benchmark / Scenario | Metric | LMAC (Ours) | Strongest Comm Baseline | QMIX+State Upper Bound | Note |
|------------|------|-------------|--------------|-----------------|------|
| SMAC-Comm `bane_vs_hM` / `2o_20b_vs_2r` | Conv. Speed + Success Rate | Close to QMIX+State | T2MAC/MASIA lag behind | reference | Significant gains in large-scale scenarios |
| LBF (2 settings) | Same as above | Close to QMIX+State | Same as above | reference | Faster learning and higher final performance |
| GRF (`3_vs_1_with_keeper`, etc.) | Final Success Rate | Exceeds QMIX+State | Lower than | reference | Latent compression outperforms raw state in high-dim obs |
| SMACv2 `terran_5_vs_5` | Win Rate | $67.87 \pm 2.77$ | MAIC $63.80\pm3.4$ | QMIX+State $64.77\pm2.79$ | Outperforms upper bound in high randomness |
| SMACv2 `protoss_5_vs_5` | Win Rate | $57.96 \pm 4.02$ | MAIC $51.93\pm2.4$ | QMIX+State $56.40\pm2.33$ | Same as above |
| SMACv2 `zerg_5_vs_5` | Win Rate | $42.18 \pm 4.37$ | NDQ $38.75\pm2.9$ | QMIX+State $40.06\pm3.39$ | Same as above |

Notably, FullComm/TarMAC/SMS/COLA collapse significantly on SMACv2, highlighting the robustness of LMAC's protocols.

### Ablation Study
Average Win Rate on SMAC-Comm (%).

| Configuration | Avg. Win Rate | Note |
|------|---------|------|
| LMAC ($k=2$, Full) | $82.9 \pm 1.9$ | Two-step refinement + cycle-consistency + SAI supervision |
| $k=0$ (Initial Protocol) | $68.5 \pm 3.8$ | Single LLM output without feedback |
| $k=1$ (Accuracy refine only) | $77.8 \pm 2.2$ | Only first step of feedback |
| w/o cycle-consistency | $66.5 \pm 2.1$ | Representation redundancy in $Q$; most severe drop |
| w/o SAI supervision | $76.6 \pm 5.6$ | Representation loses "meta-cognitive" awareness |
| Threshold $\alpha$ scan ($\alpha=0.05$ optimal) | $77.2 \sim 82.9$ | Too loose/tight distorts feedback signals |
| LLM Switch (GPT-mini/o1-mini/Claude/Gemini) | $79.8 \sim 82.9$ | Gap $\le 3$ points; model independent |

### Key Findings
- Two-step feedback is non-redundant: The 5-point gain from $k=1$ (77.8) to $k=2$ (82.9) proves the value of "accuracy then uniformity" goal separation.
- Cycle-consistency is critical: Dropping it leads to 66.5, showing that "compressing redundancy" is as important as "knowing inaccuracies."
- In high-dimensional GRF, LMAC exceeds QMIX+State, suggesting filtered communication can be more beneficial for learning than raw "God-view" states.
- Protocol evolution is interpretable: $k=0$ shares Roach offsets, $k=1$ adds Overseer positions, and $k=2$ introduces fixed anchors and IDs, resulting in step-wise performance increases.

## Highlights & Insights
- Decouples LLMs from online decision-making to "offline protocol designers," saving tokens and shielding RL from LLM instability. This "LLM-generated code, RL-executed code" paradigm could extend to reward shaping or option discovery.
- SAI is a cheap yet informative metric; its statistics (mean + variance) map to "accuracy" and "uniformity" goals perfectly.
- Borrowing cycle-consistency from CycleGAN for multi-agent communication representations is a effective transfer.
- Executable Python protocols provide interpretability and auditability compared to attention-based black-box networks.

## Limitations & Future Work
- Relies on high-quality natural language task descriptions; difficult to apply when observations are anonymous embeddings (e.g., raw pixels).
- Offline buffer bias: If the pre-collected QMIX trajectories have poor coverage, SAI may guide the LLM toward locally biased protocols.
- Protocols are designed once; environments that change during training (non-stationary rewards, new agents) may require protocol re-design.
- State recovery as a proxy for communication might be insufficient for tasks requiring opponent modeling or intent reasoning.

## Related Work & Insights
- **vs MASIA / FullComm**: Both attempt state recovery, but LMAC uses an LLM to decide "who sends what," achieving efficiency and accuracy.
- **vs TarMAC / SMS / T2MAC**: Baselines focus on message aggregation; LMAC focuses on semantic-level design of message content.
- **vs Li et al. (2024)**: That approach requires per-step LLM calls; LMAC is zero-call during runtime.
- **vs Reflexion**: LMAC replaces fixed feedback with step-specific goals driven by objective RL replay data statistics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "LLM-generated communication code + SAI offline feedback" is a clean new direction for MARL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across benchmarks, LLMs, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and diagrams; some prompt details are deferred to the Appendix.
- Value: ⭐⭐⭐⭐⭐ Effectively integrates LLM semantic capabilities into MARL at low cost and high interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[NeurIPS 2025\] Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)
- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
