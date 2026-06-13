---
title: >-
  [Paper Note] Agent Learning via Early Experience
description: >-
  [ICML2026][Reinforcement Learning][Early experience] This paper proposes the early experience paradigm, enabling language agents to learn environment dynamics and decision-making reflections by utilizing future states fo…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Early experience"
  - "implicit world models"
  - "self-reflection"
  - "imitation learning"
  - "Agent RL"
date: 2026-05-08
content_hash: b2cfba24cbac29e4
---

# Agent Learning via Early Experience

**Conference**: ICML2026  
**arXiv**: [2510.08558](https://arxiv.org/abs/2510.08558)  
**Code**: None  
**Area**: Reinforcement Learning / LLM Agent  
**Keywords**: Early experience, implicit world models, self-reflection, imitation learning, Agent RL  

## TL;DR
This paper proposes the early experience paradigm, enabling language agents to learn environment dynamics and decision-making reflections by utilizing future states following their own attempted actions—even in the absence of external rewards. This method consistently outperforms pure imitation learning across eight agent environments and provides a superior initialization for subsequent GRPO reinforcement learning.

## Background & Motivation

**Background**: Currently, language agents are primarily trained via two routes. One involves supervised fine-tuning (SFT) on expert trajectories, treating the agent as a behavior cloner from states to actions. The other is reinforcement learning (RL), where agents optimize long-term returns through environment rewards. While imitation learning is simple and reward-independent—widely used in web navigation, tool calling, and embodied text environments—RL aligns better with the long-term goal of "learning from experience," although it remains less mature in real agent scenarios compared to games like Chess or Atari.

**Limitations of Prior Work**: Pure imitation learning only instructs the model on what an expert did in a specific state but fails to convey "what happens if a mistake is made." Once deployed, if the model deviates from the expert trajectory, it enters unseen states, causing errors to accumulate over long trajectories. While RL theoretically addresses this, real-world web pages, APIs, and multi-turn tool tasks often lack reliable rewards or suffer from extreme sparsity and long horizons, leading to high training costs and instability.

**Key Challenge**: Language agents need to learn from self-interaction, yet many environments currently cannot provide verifiable rewards. In other words, there is a need for RL-style experience coverage without depending on the reward signals required by RL.

**Goal**: The authors aim to answer a pragmatic question: Can agents transform the consequences of their own actions into supervisory signals without external rewards? If so, can these signals simultaneously improve task performance, out-of-distribution (OOD) generalization, and the performance ceiling for subsequent RL?

**Key Insight**: The paper observes that even if the environment provides no reward, the future state obtained after executing an action inherently contains information. For instance, web errors, empty tool results, or changes in page elements inform the agent about the consequences of its actions. By turning these consequences into training targets, an intermediate training phase can be constructed between expert demonstrations and full RL.

**Core Idea**: Utilize non-expert actions proposed by the agent and their resulting future states to replace external rewards, converting "what happens after an action" and "why the expert action is superior" into two supervised learning tasks.

## Method

### Overall Architecture

The paper formalizes language agent decision-making as an MDP: state $s$ consists of web content, tool outputs, or text descriptions; action $a$ involves clicking, calling tools, or generating text; and policy $\pi_\theta$ outputs an action distribution given a state. Traditional imitation learning uses only expert data $\mathcal{D}_{expert}=\{(s_i,a_i)\}$, optimizing $\mathcal{L}_{IL}=-\sum_i \log \pi_\theta(a_i\mid s_i)$.

Early experience goes a step further: for each expert state $s_i$, the current model samples $K$ candidate actions $a_i^j$ distinct from the expert action, executes them, and observes the resulting future states $s_i^j$. This forms rollout data $\mathcal{D}_{rollout}=\{(s_i,a_i^j,s_i^j)\}$. This data requires no reward labels or success criteria; crucially, it stems from the errors the agent is likely to make and the branches it naturally explores.

Two training methods are designed around these future states. The first is **Implicit World Model (IWM)**: the language model is trained to predict the future state given "current state + action," allowing policy parameters to internalize environment dynamics. The second is **Self-Reflection (SR)**: by comparing the future states of expert actions versus candidate actions, the model generates explanations for why the expert action is better, using "explanation + expert action" as the supervisory target. Neither is online RL; instead, they convert reward-free interactions into offline token prediction tasks.

### Key Designs

1. **Early Exploration based on Expert States**:
	- **Function**: Collect consequences of the agent's own actions without completely deviating from useful trajectories.
	- **Mechanism**: Starting from state $s_i$ in an expert trajectory, sample and execute multiple non-expert actions $a_i^j$, recording the resulting future states $s_i^j$. The state resulting from the expert action is also recorded for comparison. These rollouts cover logical "local branches" the model might drift into, rather than expensive long-range random exploration.
	- **Design Motivation**: Long-trajectory rollouts in real environments are costly and rewards are sparse. Using expert states as anchors ensures exploration begins from meaningful deep states while exposing local errors the model would actually commit.

2. **Implicit World Model (IWM)**:
	- **Function**: Enables the policy model to internally learn the environment dynamics from actions to future states.
	- **Mechanism**: Each triplet $(s_i,a_i^j,s_i^j)$ is reformulated as next-token prediction, where the target is to predict the textual state after execution. The loss is $\mathcal{L}_{IWM}=-\sum \log p_\theta(s_i^j\mid s_i,a_i^j)$. Training involves one round of dynamic modeling using IWM, followed by standard imitation learning on expert actions.
	- **Design Motivation**: Unlike traditional world models that serve as independent simulators for planning, this approach integrates world model objectives directly into LLM parameters. This avoids additional inference overhead while ensuring the policy has "seen" erroneous actions, page changes, and tool side effects during fine-tuning.

3. **Self-Reflection (SR)**:
	- **Function**: Transforms the consequences of failed non-expert actions into transferable decision logic.
	- **Mechanism**: For the same state, the next state of the expert action $s_{i+1}$ is compared with the next state of an alternative action $s_i^j$. The model is prompted to generate a reflection $c_i^j$ explaining why the expert action was more appropriate. The training objective is to predict $c_i^j$ and expert action $a_i$ from the state: $\mathcal{L}_{SR}=-\sum \log p_\theta(c_i^j,a_i\mid s_i)$, mixed with original expert data.
	- **Design Motivation**: Pure preference learning treats non-expert actions as rejected responses without explaining *why* the feedback was poor. SR leverages the LLM's ability to summarize rules, documenting consequences like "budget not met," "missing tool parameters," or "page entering wrong branch" as reusable decision principles.

### Loss & Training

The training uses two recipes belonging to early experience. IWM performs one round of future state prediction before continuing with expert action supervision (ensuring total steps do not exceed standard IL). SR is mixed with expert data, maintaining the same training epochs as IL. All environments use the same prompt formats and decoding strategies. The authors first determine the optimal training steps for the IL baseline and fix this budget for IWM and SR to ensure gains do not stem from additional optimization steps.

Experimental models include Llama-3.2-3B, Qwen-2.5-7B, and Llama-3.1-8B. Environments span 8 task categories including ALFWorld, ScienceWorld, TravelPlanner, BFCLv3, Tau-Bench, SearchQA, WebShop, and WebArena-Lite, covering finite action spaces, structured tool spaces, and open-web action spaces.

## Key Experimental Results

### Main Results

| Representative Env | Model | IL | Ours-IWM | Ours-SR | Key Conclusion |
|--------------------|-------|----|----------|---------|----------------|
| ALFWorld | Llama-3.2-3B | 78.1 | 83.6 (+5.5) | 85.9 (+7.8) | Both methods improve success rates in embodied text tasks. |
| ScienceWorld | Llama-3.1-8B | 54.7 | 57.0 (+2.3) | 68.0 (+13.3) | SR is particularly effective for multi-step scientific reasoning. |
| TravelPlanner | Qwen-2.5-7B | 16.7 | 22.2 (+5.5) | 31.7 (+15.0) | Self-reflection significantly improves long-range constraint satisfaction. |
| BFCLv3 | Llama-3.2-3B | 21.3 | 25.3 (+4.0) | 29.3 (+8.0) | SR reduces logic and sequencing errors in tool calling. |
| WebShop | Llama-3.1-8B | 47.3 | 58.6 (+11.3) | 58.2 (+10.9) | Future state prediction provides significant gains in web shopping. |
| WebArena-Lite | Llama-3.1-8B | 4.9 | 8.5 (+3.6) | 8.5 (+3.6) | Performance increases even on noisy Web Accessibility Trees. |

### Ablation Study

| Analysis Item | Setting | Key Metric | Description |
|---------------|---------|------------|-------------|
| OOD Generalization | ALFWorld / BFCLv3 / SearchQA OOD | IWM/SR both beat IL, mostly by 2-9 pts | Early experience covers states outside expert trajectories, leading to smaller OOD drops. |
| Subsequent RL Init | WebShop / ALFWorld / SearchQA + GRPO | RL ceiling for early-experience checkpoints is higher than IL checkpoints | Even with late-stage reward optimization, learning dynamics/reflection first provides a better start. |
| Human Data Volume | Varying ratios of expert demos | WebShop beats full-IL with 1/8 demos; ALFWorld beats full-IL with ~1/2 demos | Self-action consequences provide information beyond expert data. |
| Branch Number $K$ | Varying number of alternative actions | IWM improves steadily with $K$; SR peaks around $K=2$ to $4$ | IWM needs dynamic coverage; SR is limited by context and comparison quality. |

### Key Findings
- IWM is better suited for environments with stable state transition rules and predictable future states (e.g., WebShop, ALFWorld), as it helps the model understand where an action pushes the environment.
- SR is more effective where errors involve reasoning, constraints, and tool selection (e.g., TravelPlanner, ScienceWorld, BFCLv3), as it summarizes error branches into decision rationales.
- Early experience is not just an imitation learning replacement but a "warm start" for RL. Using the same GRPO recipe, final performance starting from IWM/SR checkpoints is consistently higher than starting from IL.
- Compared to Long CoT, STaR-style rationale, and DPO, the advantage of early experience lies in the fact that reflections and predictions are grounded in actual execution consequences, rather than imagined rationales or simple preference pairs.

## Highlights & Insights
- The core highlight is the reinterpretation of "no reward" as "still having state feedback." Many agent environments do not provide success labels, but page changes, tool outputs, and error messages serve as weak supervision for action quality.
- IWM is lightweight: it trains no external simulator and avoids model-predictive planning at inference, treating future state prediction as a form of "mid-training." This sacrifices explicit planning for engineering feasibility.
- SR is more credible than standard rationale distillation because the explanation stems from a direct comparison of consequences between an expert action and a model-generated action.
- The positioning of early experience between imitation learning and RL is clear: in the short term, it reduces dependency on expert data; in the long term, it ensures RL begins with a policy that understands environment dynamics.

## Limitations & Future Work
- The method still relies on expert trajectories as exploration anchors; it is not yet "learning from experience" entirely from scratch.
- Future states must be textualizable for training. For high-dimensional visual states or real-time interfaces, simple text summaries may lose critical information.
- SR quality depends on the model's own capabilities and prompts; if alternative actions are not actually worse or if expert trajectories are flawed, reflections may consolidate incorrect preferences.
- IWM and SR are currently offline phases and have not yet formed a closed-loop system for continuous experience collection and policy updates. Future work could explore data selection with uncertainty estimation and reflection quality discriminators.

## Related Work & Insights
- **vs. Pure Imitation Learning**: IL only learns the mapping from expert states to actions. This paper executes the model's own alternative actions to learn consequences, mitigating distribution shift upon deployment.
- **vs. Traditional World Models / Model-based RL**: Typical world models are independent components for planning. IWM integrates state prediction into the policy parameters as a "reward-free" mid-training step.
- **vs. Self-Reflection prompting**: Previous reflection usually occurs at inference and often requires external feedback; this paper integrates reflection into training data grounded by real state differences.
- **vs. STaR / Rationale Distillation**: STaR-style methods synthesize rationales for correct actions but may never see error consequences; SR explicitly compares states to reduce hollow explanations.
- **vs. DPO**: DPO uses chosen/rejected pairs which provide coarser signals; this paper shows IWM/SR performs better and more stably than DPO on tasks like WebShop.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Decoupling "supervision" from "rewards" via future states is a clear and insightful paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence across 8 environments, multiple model scales, OOD tests, and RL warm starts.
- Writing Quality: ⭐⭐⭐⭐☆ The main narrative is clear; however, the sheer number of tables requires referring to the appendix for full details.
- Value: ⭐⭐⭐⭐⭐ Highly practical for training LLM agents, especially as a bridge between SFT and RL in reward-less environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](../../ICLR2026/reinforcement_learning/exgrpo_learning_to_reason_from_experience.md)
- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)

</div>

<!-- RELATED:END -->
