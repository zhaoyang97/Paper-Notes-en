---
title: >-
  [Paper Note] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search
description: >-
  [ACL 2026][LLM Agent][agentic search] Addressing the reliability issue where RL-trained agentic search models rarely say "I DON'T KNOW," leading to fabricated answers…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "agentic search"
  - "boundary-aware"
  - "GRPO"
  - "IDK refusal"
  - "reliability"
date: 2026-05-08
content_hash: 59052ff150bbc3cd
---

# BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search

**Conference**: ACL 2026  
**arXiv**: [2601.11037](https://arxiv.org/abs/2601.11037)  
**Code**: https://github.com/Liushiyu-0709/BAPO-Reliable-Search (Available)  
**Area**: LLM Agent / Reinforcement Learning  
**Keywords**: agentic search, boundary-aware, GRPO, IDK refusal, reliability

## TL;DR
Addressing the reliability issue where RL-trained agentic search models rarely say "I DON'T KNOW," leading to fabricated answers, BAPO introduces "Group-based Boundary-Aware Reward + Adaptive Reward Modulator" on top of GRPO. This allows models to refuse only when truly exceeding their capabilities. Compared to GRPO, BAPO improves average reliability by approximately 9.7% across four multi-hop QA datasets and outperforms Search-R1 (trained on 90k samples) using only 5k training samples.

## Background & Motivation
**Background**: Agentic search models trained via RL (GRPO)—such as Search-R1, ReSearch, R1-Searcher, and Tool-Star—have significantly improved multi-hop QA accuracy through ReAct-style `<think>/<search>/<answer>` interactions, becoming a mainstream approach for knowledge-intensive LLM applications.

**Limitations of Prior Work**: These RL models almost never acknowledge when they "don't know." For instance, Qwen2.5-7B-Instruct has an 18.75% IDK rate and 50.76 precision (much higher than its 41.25 accuracy) before RL. However, after being trained into ReSearch-7B via GRPO, the IDK rate plummets to 3.65% and its precision drops to 53.24. The model is "forced" by rewards to provide answers for all questions, leading to the fabrication of plausible-looking but incorrect answers that are difficult for users to verify in long search chains, causing a serious degradation in reliability.

**Key Challenge**: Standard correctness rewards simultaneously encourage "exhaustive exploration to get it right" and "punish all expressions of uncertainty," which are mutually exclusive on difficult problems. A naive fix—giving a +0.5 fixed reward for IDK—is immediately exploited by the model as a shortcut for laziness (IDK rate jumps to 53.1%), essentially becoming reward hacking and causing accuracy to decline.

**Goal**: (i) Construct a reliable learning signal for the dynamic "reasoning boundary" in agentic search, which is tightly coupled with retrieval; (ii) Integrate this signal into RL without triggering new forms of reward hacking.

**Key Insight**: Define the "boundary" as a property verifiable through group sampling—if none of the $G$ rollouts in a group produce a correct answer, the question is considered beyond the current policy's boundary. Additionally, observe that training has distinct "exploration" and "plateau" stages, meaning rewards should be activated adaptively at both the stage and sample levels.

**Core Idea**: Utilize a boundary-aware reward that only rewards IDK when the entire group fails, combined with an adaptive modulator (off during exploration / on during plateau + off for high-diversity samples / on for low-diversity samples). This embeds honest refusal capabilities into the agentic search model while maintaining deep exploration.

## Method

### Overall Architecture
BAPO is built on GRPO. For each question $x$, the policy samples $G=8$ trajectories $\{\tau_i\}_{i=1}^{G}$ interleaving `<think>/<search>/<result>/<answer>`. Two rewards are calculated for each trajectory: (1) `correctness reward` $\mathcal{R}^{\textit{Correct}}$ (F1 score if the format is correct, otherwise -1); (2) `boundary-aware reward` $\mathcal{R}^{\textit{IDK}}$ (a +0.5 reward for IDK responses only if the entire group contains no correct answers). The final reward $\mathcal{R}=\mathcal{R}^{\textit{Correct}}+\mathcal{R}^{\textit{IDK}}$ is used for group-normalized advantage $A_i$ in GRPO. An "adaptive reward modulator" dynamically decides whether to inject $\mathcal{R}^{\textit{IDK}}$ to prevent misleading the model during early exploration. The pipeline modifies only the reward layer without changing the policy architecture or requiring cold-start SFT.

### Key Designs

1.  **Group-based Boundary-Aware Reward $\mathcal{R}^{\textit{IDK}}$**:
    - **Function**: Formalizes whether the model is "out of bounds" as a group-level event, rewarding honest refusal only when the boundary is exceeded.
    - **Mechanism**: For a group $\{\tau_i\}$, if $\forall i,\ \mathcal{R}^{\textit{Correct}}(\tau_i)\le 0$, the entire group is considered incorrect. In this case, samples with $y_i=\text{IDK}$ are given $\mathcal{R}^{\textit{IDK}}=0.5\cdot\mathbb{I}(y_i=\text{IDK})$. If any correct answer exists in the group, this term becomes zero. This decouples the IDK reward from "problem solvability," preventing laziness on easy questions.
    - **Design Motivation**: Boundaries are not static like parametric "knowledge boundaries"; they depend on the synthetic results of planning, retrieval, and iterative reasoning. Using failure consistency across multiple rollouts of the same policy as a boundary proxy provides a signal that requires no external labeling and is naturally absorbed by GRPO advantage normalization.

2.  **Stage-level Modulator (Off during Exploration / On during Plateau + Dynamic Resampling)**:
    - **Function**: Toggles $\mathcal{R}^{\textit{IDK}}$ based on the training stage and performs additional sampling for uncertain questions during the plateau stage to better determine the boundary.
    - **Mechanism**: During the early "exploration stage," the IDK reward is disabled by default and only briefly enabled when the group IDK ratio $\rho_{\text{IDK}}<\alpha=5\%$ to prevent IDK from crowding out exploration opportunities. When validation scores stagnate for 5 consecutive steps, the system switches to the "plateau stage" and fully enables $\mathcal{R}^{\textit{IDK}}$. During the plateau stage, difficult questions with no correct answers in the group are resampled up to $k=2$ additional times (equivalent to pass@24) until an IDK or correct answer appears.
    - **Design Motivation**: Preliminary experiments showed that naive IDK rewards cause the model to learn to be lazy before learning to solve problems. Binding the reward schedule to the learning curve allows the model to "learn to solve first, then learn to concede."

3.  **Sample-level Modulator (Adaptive by Rollout Diversity)**:
    - **Function**: Decides whether to enable the IDK reward for individual samples during the plateau stage with fine-grained control.
    - **Mechanism**: Use $|\{y_{1..G}\}|\ge G/2$ as a "high diversity" criterion—if the model is still actively exploring the solution space, $\mathcal{R}^{\textit{IDK}}$ is disabled to avoid premature convergence. Conversely, low diversity suggests the model has converged on a fixed output, so $\mathcal{R}^{\textit{IDK}}$ is enabled to reinforce boundary awareness.
    - **Design Motivation**: Rollout consistency serves as a proxy for model confidence, distinguishing between "exploring" and "converged" samples, allowing the reward to focus on "exploring where needed and conceding where appropriate."

### Loss & Training
The policy objective is clipped GRPO ($\epsilon=0.1$), KL coefficient 0.001, rollout count $G=8$, temperature 1.0, max tokens 8192, and a maximum of 3 tool calls. Advantages $A_i$ are z-score normalized within the group. The retrieval environment is based on FlashRAG + E5-base-v2 + 2018 Wikipedia with top-5 documents. The training set consists of only 5k samples (from HotpotQA / 2WikiMultiHopQA), trained for 2 epochs with a batch size of 64.

## Key Experimental Results

### Main Results
Acc / Precision / Reliability (Rel.=$(1-\rho_{\text{IDK}})\cdot\text{prec}+\rho_{\text{IDK}}\cdot\text{acc}$) on four multi-hop QA datasets using Qwen2.5-7B-Instruct:

| Method | HotpotQA Rel. | MuSiQue Rel. | 2Wiki. Rel. | Bamboogle Rel. | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Search-R1 (90k samples) | 49.0 | 22.5 | 39.0 | 52.0 | 40.6 |
| ReSearch (19k samples) | 61.5 | 31.0 | 54.2 | 54.4 | 50.3 |
| GRPO (5k samples) | 60.0 | 29.5 | 59.5 | 57.6 | 51.7 |
| Reliable RFT | 40.2 | 18.5 | 23.9 | 49.4 | 33.0 |
| Reliable TIR Prompt | 60.6 | 27.2 | 43.3 | 50.5 | 45.4 |
| **BAPO (5k samples)** | **65.5** | **36.6** | **63.3** | **61.2** | **56.7** |

With only 5k samples, BAPO's reliability is on average 5.0 points (+9.7% relative) higher than GRPO and surpasses Search-R1/ReSearch trained on 18x/4x more data. The strategy is to "slightly decrease accuracy (-2.2) in exchange for a large precision gain (+11.8)."

### Ablation Study
Averaged across four datasets using Qwen2.5-3B-Instruct:

| Configuration | Acc | Prec | $\rho_{\text{IDK}}$ | Reliability |
| :--- | :--- | :--- | :--- | :--- |
| **BAPO Full** | **44.8** | 52.8 | 16.8% | **51.3** |
| w/o Boundary-Aware Reward (Replace with fixed +0.5) | 30.6 | 62.4 | 53.1% | 44.8 |
| w/o Sample Modulator | 43.3 | 52.0 | 20.4% | 50.1 |
| w/o Sample + Stage Modulator | 37.8 | 56.0 | 35.2% | 49.0 |

### Key Findings
- Replacing "group-level triggers" with "fixed IDK rewards" caused $\rho_{\text{IDK}}$ to surge to 53.1% and Acc to drop by 14 points—confirming the existence of reward hacking and the necessity of group-level triggers.
- The Stage modulator is critical: removing both modulators doubled $\rho_{\text{IDK}}$ from 16.8% to 35.2% and dropped Acc by 7 points, indicating that IDK rewards must be shielded during the exploration phase.
- Sensitivity to hyperparameter $\alpha$: when $\alpha=0$, $\rho_{\text{IDK}}=0$ (the model never learns to refuse). $\alpha=0.05$ achieved the best result, while $\alpha\ge 0.2$ excessively encouraged refusal. Resampling $k$ showed significant gains from 1 to 2 and saturated at $k=3$.
- Analysis of 7B / 14B models showed that the error rates of the corresponding GRPO models on questions where BAPO chose to refuse were 76.7% / 76.7% respectively—proving that refusal is "rational" and targeted at questions the model cannot solve, rather than being random.
- 14B Training Curve: During the first 60 steps of exploration, $R^{\textit{Correct}}$ rose from 0.3 to 0.5 while $\rho_{\text{IDK}}$ fell from 20% to 5%. After switching to the plateau stage, $R^{\textit{IDK}}$ rose to 0.25–0.30, and $\rho_{\text{IDK}}$ returned to 25%+.

## Highlights & Insights
- **Operationalizing "Boundary" as a Group Event**: By using the failure consistency of $G$ rollouts in GRPO as evidence of being out of bounds—rather than external knowledge bases or complex confidence modeling—BAPO embeds this into the GRPO pipeline with nearly zero extra cost, a very elegant engineering trade-off.
- **Training Phase-Aware Reward Scheduling**: Reveals a neglected fact—the same reward can be "poison" during exploration but "medicine" during a plateau. "When to reward" is as important as "what to reward," an idea transferable to any multi-objective RLHF scenario (e.g., safety vs. helpfulness).
- **Rollout Diversity as Implicit Confidence**: Using $|\{y_{1..G}\}|\ge G/2$ to judge if the model is still exploring avoids explicit confidence estimation or extra sampling, inspiring the use of "sampling consistency" as a cheap signal for sample-level RL scheduling.
- **5k Samples Beating 90k**: Suggests that the bottleneck for agentic search is no longer data scale but reward shaping; a reliability-first training paradigm may be more economical than amassing data.

## Limitations & Future Work
- Evaluated only on Wikipedia local RAG; does not cover the noise, dynamics, and latency of real web search. IDK trigger logic may need recalibration for production.
- Evaluation limited to knowledge-intensive QA. For questions that cannot be solved via retrieval (math, code, agentic web tasks), it remains unverified if "group failure" is still a reliable boundary proxy.
- Experiments capped at 14B; marginal gains of BAPO might be compressed for 70B+ models, which have higher base reliability.
- Hyperparameters like $\rho_{\text{IDK}}$, $\alpha$, and $k$ are sensitive to training dynamics; the cost of tuning across tasks and models cannot be ignored.
- Potential extensions: Turn the stage-level modulator into an automatic curriculum driven by validation sets; extend group-level triggers to other "boundary" signals like tool failure or safety violations.

## Related Work & Insights
- **vs Search-R1 / ReSearch / R1-Searcher**: These focus solely on accuracy via correctness rewards. BAPO adds boundary-aware signals to their RL architecture and achieves higher reliability with only 5k samples.
- **vs BARREL (Yang et al., 2025a)**: BARREL uses static medium rewards for IDK and distills reasoning trajectories. This paper's ablation proves that static IDK rewards lead to laziness ($\rho_{\text{IDK}}=53.1\%$), which BAPO solves with dynamic group-level triggers.
- **vs Reliable RFT (Rejection Sampling SFT)**: RFT treats IDK as samples to be imitated, leading to extreme over-conservatism (Acc drops 27 points). BAPO uses RL to model boundaries online without destroying exploration.
- **vs Knowledge / Capability Boundary (Zheng 2025, Zhang 2025c)**: These define boundaries on static parametric knowledge or mathematical capability. BAPO handles "emergent boundaries" synthesized dynamically from planning, retrieval, and reasoning, fitting the agentic scenario better.
- **vs Uncertainty Estimation (Semantic Entropy, P(True), Verbalized Confidence)**: These are post-hoc detection methods. BAPO trains the policy itself on "when to refuse," making the two approaches orthogonal and stackable.

## Rating
- Novelty: ⭐⭐⭐⭐ Group-level boundary triggers + dual modulators for stage/sample represent a novel and concise reward design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 3 model scales × ablations + hyperparameter sensitivity + EM/LLM-judge double metrics + case studies.
- Writing Quality: ⭐⭐⭐⭐ The preliminary study clearly explains the motivation; the framework and stage dynamics diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Moves agentic search from "looking accurate" to "daring to concede," offering real value for production environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](../../NeurIPS2025/llm_agent/groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ACL 2026\] Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](rethinking_reasoning-intensive_retrieval_evaluating_and_advancing_retrievers_in_.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)

</div>

<!-- RELATED:END -->
