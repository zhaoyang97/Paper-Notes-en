---
title: >-
  [Paper Note] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents
description: >-
  [ICML 2026][Dialogue Systems][Multi-turn Tool Use] Addressing two major bottlenecks in post-training "multi-turn interactive tool-calling agents"—expensive high-quality data and RL signal corruption due to user simulatio…
tags:
  - "ICML 2026"
  - "Dialogue Systems"
  - "Multi-turn Tool Use"
  - "Verifiable Reward RL"
  - "Self-Evolving Synthetic Data"
  - "GRPO"
  - "User Simulator Fine-tuning"
date: 2026-05-08
content_hash: a2514eee359c6637
---

# From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents

**Conference**: ICML 2026  
**arXiv**: [2601.22607](https://arxiv.org/abs/2601.22607)  
**Code**: https://github.com/inclusionAI/AReaL/tree/main/examples/tau2 (Available)  
**Area**: LLM Agent / Reinforcement Learning / Tool Use  
**Keywords**: Multi-turn Tool Use, Verifiable Reward RL, Self-Evolving Synthetic Data, GRPO, User Simulator Fine-tuning

## TL;DR
Addressing two major bottlenecks in post-training "multi-turn interactive tool-calling agents"—expensive high-quality data and RL signal corruption due to user simulation noise—the authors propose "Self-Evolving Multi-Agent Data Synthesis (AReaL-SEA)" coupled with executable verifiers as rewards. Combined with an RL recipe featuring "SFT-first user models + large batch + dynamic filtering GRPO," Qwen3-235B achieves pass^1 rates of 73.0 (Airline) and 98.3 (Telecom) on $\tau^2$-bench, matching or exceeding Claude/Gemini/GPT-5.

## Background & Motivation

**Background**: LLMs are evolving from "Q&A machines" to "task-completion assistants," requiring simultaneous communication with humans and interaction with environments (APIs/tools) to complete complex tasks (e.g., the "reschedule → query → verify policy → execute" workflow in $\tau^2$-bench Airline). While base models like ReAct/Toolformer exist for tool use, multi-turn interactive agents introduce the "user presence" dimension, which is significantly more challenging than single-turn tool calling.

**Limitations of Prior Work**: Post-training open-source models into competitive interactive agents is hindered by two bottlenecks. **(1) Data Issues**: Multi-turn tool-use data is extremely difficult to scale—human annotation is costly, while automatic synthesis struggles to simultaneously meet requirements for complex domain rules, simulated user private information, and sufficient task difficulty for RL. **(2) RL Instability**: Interactive tasks are user-driven, necessitating a user simulator during RL rollouts. However, the authors find that open-source models are highly unstable as simulators—in $\tau^2$-bench dual-control scenarios, users must also issue tool calls; open-source models frequently issue incorrect calls or ignore instructions, causing rollout failures and incorrect reward attribution to the Agent.

**Key Challenge**: Training an Agent with RL requires stable rollouts; stable rollouts require stable user simulation; stable user simulation requires high-quality training data; and high-quality training data requires joint rollouts from both the Agent and the User—creating a circular dependency.

**Goal**: (i) Design a scalable, verifiable multi-turn tool-use data synthesis pipeline; (ii) develop an RL recipe for interactive agents capable of withstanding "user simulation instability."

**Key Insight**: Treat data synthesis as a "hierarchical multi-agent system + self-evolving feedback loop" allowing the system to learn from its own failures. Fine-tune the user simulator via SFT using synthetic data before RL rollouts to suppress "user noise" at the source, while utilizing large batches and dynamic filtering to absorb remaining reward variance.

**Core Idea**: Data = Self-evolving multi-agent system + executable verifiers; RL = User simulator stabilization followed by robust GRPO training; these components integrate into a circularly improving post-training pipeline.

## Method

### Overall Architecture
Consists of two modules. **AReaL-SEA Data Synthesis** (§4): A meta-planner generates $N$ diverse (synthesis plan, evaluation plan) pairs, each running an independent pipeline (task synthesis → task verification → trajectory rollout → trajectory verification). Failure cases are collected by a reflection module to iteratively update plans across $K$ rounds. **RL Recipe** (§5): The user simulator is first trained via SFT on synthetic data; then, the Agent is trained using GRPO (group-relative advantage + dynamic filtering + large batch), with reward signals derived from the verifier comparing final states against ground-truth states.

### Key Designs

1. **AReaL-SEA Self-Evolving Data Synthesis Pipeline**:

    - **Function**: Generates diverse, complex, and verifiable multi-turn tool-use training samples.
    - **Mechanism**: (a) **Diversified Plan Generation**: The meta-planner sequentially generates $N$ non-overlapping plan pairs, specifying different domains/complexities/tool patterns/user styles to explicitly construct diversity. (b) **Four-stage Agent Pipeline**: Task Synthesis Agent generates structured task tuples $q = (u, t, a^*)$; Task Verification Agent checks task quality; Trajectory Rollout executes full dialogues with simulated users; Trajectory Verification Agent evaluates trajectory quality and performs attribution tagging (assigning failures to task issues or trajectory issues). (c) **Reflection Loop**: Failures and attributions are summarized by a reflection agent to update $(\mathcal{P}_s, \mathcal{P}_e)$. The next round's plans are more precise and rubrics more calibrated, forming the closed loop $(\mathcal{P}_s^{(n,k+1)}, \mathcal{P}_e^{(n,k+1)}) = \text{Reflect}(\mathcal{P}_s^{(n,k)}, \mathcal{P}_e^{(n,k)}, \{\text{failures}\})$.
    - **Design Motivation**: Previous pipelines like APIGen-MT/TOUCAN are **static** and cannot learn from errors. This system treats synthesis as an evolvable multi-agent system that iterates rules for each domain. Ablations show: removing the evolution loop drops performance from 56.0 → 44.0; reducing prompt diversity from 64 → 4 drops it from 56.0 → 42.5.

2. **Executable Per-instance Verifier as RL Reward**:

    - **Function**: Attaches an executable check function to each synthetic task to serve as a sparse RL reward signal.
    - **Mechanism**: Synchronously generates ground-truth final states and verifier functions during task synthesis. During RL, the verifier compares the trajectory's $s_T$ with ground-truth entities and actions; a full match yields 1, otherwise 0, forming a binary outcome reward. The reward function is defined as $\mathcal{R}(s_t, a_t) = R(s_T)$ for $t = T$, and 0 otherwise.
    - **Design Motivation**: LLM-as-judge is noisy and expensive for interactive agents. Using deterministic verifiers generated during synthesis is fast and accurate, implementing the Verifiable Reward (RLVR) paradigm for Agent scenarios.

3. **GRPO + User Model SFT + Large Batch + Dynamic Filtering**:

    - **Function**: Stabilizes RL training under user simulation noise.
    - **Mechanism**: (a) **User Model SFT**: SFT the user simulator (based on Qwen3-30B-A3B-2507) on AReaL-SEA dialogue data so it follows instructions and roles. Ablations show RL using a base user model causes performance to regress from an SFT checkpoint of 85.4 to 75.6, whereas an SFT-enhanced user model pushes it to 95.6—a 20-point gap. (b) **GRPO**: Samples $G$ independent trajectories per task, calculating group-normalized advantage $\hat{A}(\tau^{(g)}) = \frac{R(\tau^{(g)}) - \mu_G}{\sigma_G}$ with a token-level clipped surrogate loss. (c) **Large Batch**: Increasing total batch from 256 → 512 improved pass^1 from 64-66 to 70.5, providing more stable advantage estimates. (d) **Dynamic Filtering**: Tasks where all trajectories in a group either succeed or fail provide $\hat{A} = 0$, offering no learning signal. These are filtered out to keep only discriminative groups—removing this step drops performance from 70.5 to 65.0.
    - **Design Motivation**: User simulation noise is unique to this problem, making user model SFT a novel and necessary step. The remaining components stabilize learning from limited reward signals.

### Loss & Training
The RL objective is $\mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{q \sim \mathcal{D}}[\frac{1}{\sum_g N_G}\sum_g \sum_t \sum_i \mathcal{L}_{t,i}^{(g)}(\theta)]$, where $\mathcal{L}_{t,i}^{(g)} = \min(\rho_{t,i}^{(g)} \hat{A}^{(g)}, \text{clip}(\rho_{t,i}^{(g)}, 1-\epsilon, 1+\epsilon)\hat{A}^{(g)})$ and the token-level importance ratio is $\rho_{t,i}^{(g)} = \pi_\theta / \pi_{\theta_\text{old}}$. SFT uses standard cross-entropy. 30B models were trained on 64 H200 GPUs; 235B on 80 H200s.

## Key Experimental Results

### Main Results
$\tau^2$-bench across three domains (Airline / Retail / Telecom), where pass^k requires $k$ independent successful attempts (stricter than pass@k):

| Model | Airline pass^1 | Retail pass^1 | Telecom pass^1 |
|-------|----------------|----------------|----------------|
| Claude-Sonnet-4.5 | 70.0 | 86.2 | 98.0 |
| Gemini 3.0 Pro | 73.0 | 85.3 | 98.0 |
| GPT-5 | 62.5 | 81.6 | 95.8 |
| Qwen3-235B baseline | 58.0 | 59.9 | 53.7 |
| Qwen3-235B + SFT | 64.0 | 71.5 | 87.9 |
| **Qwen3-235B + RL** | **73.0** | 75.0 | **98.3** |
| Qwen3-30B-A3B-2507 baseline | 56.0 | 54.2 | 28.5 |
| Qwen3-30B-A3B-2507 + SFT | 60.0 | 69.1 | 85.4 |
| **Qwen3-30B-A3B-2507 + RL** | 70.5 | 75.0 | 95.6 |

The 235B version matches Gemini 3.0 Pro in Airline and exceeds all frontier models in Telecom; Retail remains the most difficult domain (Claude leads at 86.2), with the open-source version reaching 75.0. The 30B version is highly competitive, with a 95.6 in Telecom approaching GPT-5.

Mix Training (merging data from three domains) allowed Qwen3-235B to achieve an average pass^1 of 81.3%, exceeding Qwen3-Max-Thinking (80.7) and GPT-5 (80.0). On the stringent pass^4 metric, its 68.5% also surpassed Max-Thinking (66.8) and GPT-5 (64.0).

### Ablation Study

| Configuration | Airline pass^1 (SFT) | Description |
|------|----------------------|------|
| Qwen3-30B baseline | 38.0 | Starting point |
| Human Expert data | 52.0 | Human-designed workflow |
| **AReaL-SEA Full** (64 plans) | **56.0** | Exceeds human |
| w/o Validation | 50.0 | -6 points due to quality |
| w/o Evolution | 44.0 | -12 points due to no reflection |
| 4 prompt sets only | 42.5 | -13.5 points due to low diversity |

| User Model | Telecom pass^1 (RL) | Description |
|------------|---------------------|------|
| Start from SFT | 85.4 | Pre-RL |
| RL + base user model | 75.6 | **10-point regression** |
| RL + SFT user model | **95.6** | +10 points |

| RL Config | Airline pass^1 | Description |
|---------|----------------|------|
| 8×32 (total 256) | 64.0 | Small batch |
| 16×16 (total 256) | 66.0 | Prompt/Traj split minor |
| 8×64 (total 512) | **70.5** | Large batch is key |
| 8×64 + No Filtering | 65.0 | Filtering mandatory |

### Key Findings
- **Automatic Synthesis ≥ Human Experts**: AReaL-SEA full (56.0) exceeds human expert data (52.0), proving self-evolution saves labor while increasing data quality ceilings.
- **User Model SFT is the Hidden Key to RL Success**: Using a base user model fails to even maintain SFT performance (75.6 < 85.4). This failure mode, largely unaddressed in prior literature, is illustrated in Fig 2: base users ignore instructions and misuse tools, passing corrupt signals to the Agent.
- **Total Batch Size Outweighs Prompts:Trajs Split**: Results for 8×32 vs 16×16 are similar (64 vs 66), but 8×64 vs 8×32 shows significant gains (70.5 vs 64.0), indicating GRPO advantage stability depends on total sample count.
- **Mix Training Benefits Large Models but Harms Small Ones**: At 30B, mix training dropped average pass^1 from 71.5 to 63.7, while 235B remained stable (74.5 vs 74.7). This confirms the intuition that smaller models lack the capacity to absorb multiple domains simultaneously.

## Highlights & Insights
- **"User Simulator SFT" is a critical contribution**: This paper is the first to explicitly demonstrate that user simulator quality dictates RL success, evidenced by a 20-point performance gap—a vital warning for interactive agent research.
- **Self-Evolving Synthesis as a Universal Paradigm**: The "task synthesis → verification → rollout → verification → reflection → plan update" loop allows LLMs to learn synthesis from failure, proving more scalable than static pipelines. This architecture is transferable to reasoning chains or long-context QA.
- **Verifiable Reward + Agent RL Paradigm**: Extending RLVR from math/code to multi-turn tool calling by generating verifiers during synthesis avoids expensive LLM judging during training. This is applicable to any task where the final state is programmatically checkable.
- **Scale Dependency of Mix vs Separate Training**: A practical finding that assists engineering decisions regarding training single generalist agents versus domain-specific experts.

## Limitations & Future Work
- Evaluation is limited to three $\tau^2$-bench domains; performance on Retail still lags behind Claude Sonnet 4.5.
- The optimal number of reflection loop steps $K$ in AReaL-SEA remains an open question.
- The distribution gap between synthetic data and real-world production dialogues is not fully explored.
- The RL recipe requires significant infrastructure (80 H200s for 235B), creating a high barrier for smaller teams.
- Safety in tool use is not deeply discussed; real-world deployment requires dedicated permission and auditing layers.

## Related Work & Insights
- **vs APIGen-MT (Prabhakar et al.)**: Uses reviewer-style validation; AReaL-SEA adds self-evolution and synchronous verifier generation for better RL compatibility.
- **vs TOUCAN (Xu et al.)**: TOUCAN scales to 1.5M trajectories; this work prioritizes quality and self-evolution, proving 64 high-quality plans can exceed human efforts.
- **vs ToolRL / Search-R1**: Focused on single-turn tool use RL; this work addresses the "user simulator quality" dimension essential for multi-turn interaction.
- **vs π₀ / GR00T**: Robotics RL uses real environments for ground truth; this work uses synthetic verifiers, lowering cost but introducing potential fidelity gaps.
- **vs ARENA-RL**: Uses relative rankings for sparse rewards; this work uses dynamic filtering and large batches to handle advantage noise.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of self-evolving synthesis, user model SFT, and verifier-based RL is novel in agent post-training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across three domains, three model scales, training strategies, and extensive ablations against frontier models.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative (Data issues + RL issues → Two solutions); refined formulas and figures; robust training details in the appendix.
- Value: ⭐⭐⭐⭐⭐ Achieving SOTA on $\tau^2$-bench with open-source models is a significant result; the framework is reproducible and valuable for industrial tool-using agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](../../AAAI2026/dialogue/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ACL 2026\] GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](../../ACL2026/dialogue/genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](../../ACL2026/dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)

</div>

<!-- RELATED:END -->
