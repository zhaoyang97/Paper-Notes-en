---
title: >-
  [Paper Note] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents
description: >-
  [ICML 2026][Dialogue Systems][Multi-turn tool use] To address two major bottlenecks in post-training "multi-turn interactive tool-using agents"—the high cost of quality data and RL signal corruption from user simulation…
tags:
  - "ICML 2026"
  - "Dialogue Systems"
  - "Multi-turn tool use"
  - "verifiable reward RL"
  - "self-evolving synthetic data"
  - "GRPO"
  - "user simulator fine-tuning"
date: 2026-05-08
content_hash: 9747f6bfe253d19e
---

# From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents

**Conference**: ICML 2026  
**arXiv**: [2601.22607](https://arxiv.org/abs/2601.22607)  
**Code**: https://github.com/inclusionAI/AReaL/tree/main/examples/tau2 (available)  
**Area**: LLM Agent / Reinforcement Learning / Tool Use  
**Keywords**: Multi-turn tool use, verifiable reward RL, self-evolving synthetic data, GRPO, user simulator fine-tuning

## TL;DR
To address two major bottlenecks in post-training "multi-turn interactive tool-using agents"—the high cost of quality data and RL signal corruption from user simulation noise—the authors propose "self-evolving multi-agent data synthesis (AReaL-SEA)" paired with executable verifiers as rewards. Combined with an RL recipe of "first SFT the user model, then large batch + dynamic filtering GRPO," this approach pushes Qwen3-235B to Airline 73.0 / Telecom 98.3 pass^1 on τ²-bench, matching or surpassing Claude/Gemini/GPT-5 across the board.

## Background & Motivation

**Background**: LLMs are evolving from "question-answering machines" to "task completion assistants," requiring them to communicate with humans and interact with environments (APIs/tools) within dialogues to accomplish complex tasks (e.g., τ²-bench Airline's "reschedule → query → policy check → execute" flow). While foundational models like ReAct/Toolformer/OpenVLA exist for tool-using agents, multi-turn interactive agents add the "user-in-the-loop" dimension, making them much more challenging than single-turn tool use.

**Limitations of Prior Work**: Training open-source models into competitive interactive agents faces two bottlenecks. **(1) Data**: Multi-turn tool-use dialogue data is extremely hard to scale—manual annotation is costly, and automatic synthesis struggles to simultaneously meet "complex domain rules + simulated user private info + sufficient RL task difficulty." **(2) RL Instability**: Interactive tasks require user-driven RL rollouts, necessitating user simulators. The authors find open-source models as user simulators are highly unstable—in τ²-bench's dual-control scenarios, users also issue tool calls, but open-source models often misuse tools or ignore instructions, causing rollout failures and misattributing rewards to the agent.

**Key Challenge**: RL is needed to train the agent, but RL requires stable rollouts, which require stable user simulation, which in turn needs good training data—yet good data depends on agent + user co-rollouts. This forms a circular dependency.

**Goal**: (i) Design a scalable, verifiable multi-turn tool-use data synthesis pipeline; (ii) Develop an RL recipe for interactive agents robust to unstable user simulation.

**Key Insight**: Make data synthesis a "hierarchical multi-agent system + self-evolving feedback loop," enabling the system to learn from its own failures; pretrain the user simulator with synthetic data via SFT before RL rollouts to suppress user noise at the source; use large batch + dynamic filtering to absorb remaining reward variance.

**Core Idea**: Data = self-evolving multi-agent + executable verifier; RL = first stabilize the user simulator, then train the agent with stable GRPO; the two tightly integrate into a cyclically improvable post-training pipeline.

## Method

### Overall Architecture
Two main modules. **AReaL-SEA Data Synthesis** (§4): The meta-planner first generates $N$ diverse (synthesis plan, evaluation plan) pairs, each running an independent pipeline (task synthesis → task verification → trajectory rollout → trajectory verification). Failure cases are aggregated into a reflection module to iteratively update plans, looping for $K$ rounds. **RL Recipe** (§5): First, SFT the user simulator with synthetic data; then train the agent with GRPO (group-relative advantage + dynamic filtering + large batch), using rewards from a verifier comparing final vs. ground-truth states.

### Key Designs

1. **AReaL-SEA Self-Evolving Data Synthesis Pipeline**:

    - **Function**: Generates diverse, complex, and verifiable multi-turn tool-use training samples.
    - **Mechanism**: (a) **Diversified Plan Generation**: The meta-planner sequentially generates $N$ non-overlapping plan pairs, each specifying different domains/complexity/tool modes/user styles, explicitly constructing diversity without relying on randomness. (b) **Four-stage Agent Pipeline**: Task Synthesis Agent generates structured task tuples $q = (u, t, a^*)$ via multi-turn tool use; Task Verification Agent checks task quality; Trajectory Rollout simulates full dialogues with user + assistant; Trajectory Verification Agent evaluates trajectory quality and assigns attribution tags (failure due to task or trajectory). (c) **Reflection Loop**: Failures and attributions are aggregated to a reflection agent, which updates $(\mathcal{P}_s, \mathcal{P}_e)$ for more precise plans and calibrated rubrics in the next round, forming a closed loop: $(\mathcal{P}_s^{(n,k+1)}, \mathcal{P}_e^{(n,k+1)}) = \text{Reflect}(\mathcal{P}_s^{(n,k)}, \mathcal{P}_e^{(n,k)}, \{\text{failures}\})$.
    - **Design Motivation**: Previous pipelines like APIGen-MT / TOUCAN are **static** and cannot learn from their own errors; this work makes data generation an evolvable multi-agent system, enabling domain-specific rule iteration. Ablations show: removing the evolution loop drops performance from 56.0 → 44.0; reducing prompt diversity from 64 → 4 drops from 56.0 → 42.5—both are key contributions.

2. **Executable Per-instance Verifier as RL Reward**:

    - **Function**: Each synthetic task comes with an executable check function, serving as a sparse RL reward signal.
    - **Mechanism**: During task synthesis, generate ground-truth final state and verifier function; after RL trajectory, the verifier compares $s_T$ to ground-truth for key entities and actions—full match yields 1, else 0, forming a binary outcome reward. Reward function: $\mathcal{R}(s_t, a_t) = R(s_T)$ for $t = T$, else 0.
    - **Design Motivation**: Using LLM-as-judge for interactive agent rewards is noisy and expensive; generating deterministic verifiers during synthesis is fast and accurate, implementing the RLVR paradigm for agent tasks.

3. **GRPO + User Model SFT + Large Batch + Dynamic Filtering**:

    - **Function**: Stabilizes RL training under user simulation noise.
    - **Mechanism**: (a) **User Model SFT**: SFT the user simulator (based on Qwen3-30B-A3B-2507) with AReaL-SEA dialogue data to ensure stable instruction following and role-based tool use—ablation shows using a base user model for RL drops performance from SFT checkpoint 85.4 to 75.6, while SFT user model boosts to 95.6, a 20-point gap. (b) **GRPO**: For each task, sample $G$ independent trajectories, compute group-normalized advantage $\hat{A}(\tau^{(g)}) = \frac{R(\tau^{(g)}) - \mu_G}{\sigma_G}$; token-level clipping surrogate loss. (c) **Large Batch**: Ablation shows increasing total batch from 256 → 512 raises pass^1 from 64-66 to 70.5, providing more stable advantage estimates. (d) **Dynamic Filtering**: Tasks with all-success or all-failure in a group yield $\hat{A} = 0$ (no learning signal) and are filtered out, retaining only differentiated groups—removing this step drops performance from 70.5 to 65.0.
    - **Design Motivation**: User simulation noise is unique to this problem, making user model SFT both novel and necessary; the remaining trio (GRPO + large batch + dynamic filtering) ensures the limited reward signal drives learning as stably as possible.

### Loss & Training

The RL objective is $\mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{q \sim \mathcal{D}}[\frac{1}{\sum_g N_G}\sum_g \sum_t \sum_i \mathcal{L}_{t,i}^{(g)}(\theta)]$, where $\mathcal{L}_{t,i}^{(g)} = \min(\rho_{t,i}^{(g)} \hat{A}^{(g)}, \text{clip}(\rho_{t,i}^{(g)}, 1-\epsilon, 1+\epsilon)\hat{A}^{(g)})$, and token-level importance ratio $\rho_{t,i}^{(g)} = \pi_\theta / \pi_{\theta_\text{old}}$. SFT uses standard cross-entropy. The 30B model is trained on 64 H200 GPUs, 235B on 80 H200.

## Key Experimental Results

### Main Results
τ²-bench, three domains (Airline / Retail / Telecom), pass^k means all k independent attempts must succeed (stricter than pass@k):

| Model | Airline pass^1 | Retail pass^1 | Telecom pass^1 |
|-------|----------------|---------------|----------------|
| Claude-Sonnet-4.5 | 70.0 | 86.2 | 98.0 |
| Gemini 3.0 Pro | 73.0 | 85.3 | 98.0 |
| GPT-5 | 62.5 | 81.6 | 95.8 |
| Qwen3-235B baseline | 58.0 | 59.9 | 53.7 |
| Qwen3-235B + SFT | 64.0 | 71.5 | 87.9 |
| **Qwen3-235B + RL** | **73.0** | 75.0 | **98.3** |
| Qwen3-30B-A3B-2507 baseline | 56.0 | 54.2 | 28.5 |
| Qwen3-30B-A3B-2507 + SFT | 60.0 | 69.1 | 85.4 |
| **Qwen3-30B-A3B-2507 + RL** | 70.5 | 75.0 | 95.6 |

The 235B version matches Gemini 3.0 Pro on Airline and surpasses all frontier models on Telecom; Retail is the hardest domain (Claude 86.2 still leads), with the open-source version reaching 75.0. The 30B version is also highly competitive, with Telecom 95.6 close to GPT-5.

Mix Training (combining data from all three domains) enables Qwen3-235B to achieve an average pass^1 of 81.3%, surpassing Qwen3-Max-Thinking (80.7) and GPT-5 (80.0); on the stricter pass^4 metric, 68.5% also exceeds Max-Thinking (66.8) and GPT-5 (64.0).

### Ablation Study

| Configuration | Airline pass^1 (SFT) | Notes |
|---------------|----------------------|-------|
| Qwen3-30B baseline | 38.0 | Starting point |
| Human Expert data | 52.0 | Manually designed workflow |
| **AReaL-SEA Full** (64 plans, all components) | **56.0** | Surpasses manual |
| w/o Validation | 50.0 | Lacks quality filtering, -6 points |
| w/o Evolution | 44.0 | Lacks reflection loop, -12 points |
| 4 prompt sets only | 42.5 | Lacks diversity, -13.5 points |

| User Model | Telecom pass^1 (RL) | Notes |
|------------|---------------------|-------|
| SFT checkpoint | 85.4 | Before RL |
| RL + base user model | 75.6 | **Drops 10 points** |
| RL + SFT user model | **95.6** | Gains 10 points |

| RL Configuration | Airline pass^1 | Notes |
|------------------|---------------|-------|
| 8×32 (total 256) | 64.0 | Small batch |
| 16×16 (total 256) | 66.0 | Prompt vs. trajs split less important |
| 8×64 (total 512) | **70.5** | Large batch is key |
| 8×64 + no dynamic filtering | 65.0 | Filtering is essential |

### Key Findings
- **Automatic synthesis ≥ human expert**: AReaL-SEA full at 56.0 surpasses human expert data at 52.0, showing self-evolution not only saves labor but also raises the data quality ceiling.
- **User model SFT is a hidden key to RL success**: Using a base user model cannot even maintain SFT checkpoint performance (75.6 < 85.4), a failure mode rarely emphasized in prior literature. Figure 2 case study shows base users ignore instructions and misuse tools, passing wrong signals to the agent.
- **Total batch size matters more than prompt:traj split**: 8×32 vs 16×16 are similar (64 vs 66), but 8×64 vs 8×32 is significant (70.5 vs 64.0), indicating GRPO advantage estimation stability mainly depends on total sample size.
- **Mix training benefits large models, harms small models**: For 30B, mix training drops average pass^1 from 71.5 to 63.7 (Telecom drops 15 points), but 235B remains nearly unchanged (74.5 vs 74.7)—supporting the intuition that small models lack capacity for multi-domain absorption, guiding domain split strategies for deployment.

## Highlights & Insights
- **"User simulator SFT" is the most underrated contribution**: All prior agent RL work assumes the user model is given (whether GPT-4.1 or open-source base); this is the first to explicitly demonstrate that "user simulator quality directly determines RL improvement," with a 20-point empirical gap—a key warning for all interactive agent RL research.
- **Self-evolving data synthesis is a general paradigm**: Making "task generation → verification → trajectory rollout → verification → reflection → plan update" a closed loop enables LLMs to learn from failures in data synthesis, more scalable than static pipelines like APIGen-MT/TOUCAN. This architecture can transfer to other domains needing complex synthetic data (e.g., reasoning chains, long-context QA).
- **Verifiable reward + agent RL as a paradigm**: Extending RLVR from math/code to multi-turn tool-using agents, the key is generating verifiers during synthesis, avoiding LLM judges at training time—this "data with verifier" design can transfer to any task with programmatically checkable final states.
- **Mix vs. Separate depends on model scale**: A practical but overlooked finding, directly informing engineering decisions on "training a single general agent vs. per-domain experts" for enterprise deployment.

## Limitations & Future Work
- Evaluation is limited to three τ²-bench domains; Retail, the hardest domain, still lags behind Claude Sonnet 4.5.
- The number of reflection loop steps $K$ in AReaL-SEA is not systematically ablated; optimal convergence rounds remain open.
- No discussion of the distribution gap between synthetic and real production dialogues—τ²-bench's synthetic user styles may not cover real users.
- The RL recipe relies heavily on infrastructure (80 H200s for 235B), making replication challenging for smaller teams; lightweight extensions (e.g., distillation to small models) are a natural direction.
- Tool-use safety is not deeply discussed (impact statement briefly mentions "potential misuse"); real-world deployment requires dedicated permission/audit layers.

## Related Work & Insights
- **vs APIGen-MT (Prabhakar et al.)**: Also synthesizes multi-turn tool use, but APIGen-MT uses static reviewer-style validation; AReaL-SEA adds self-evolution and co-generation of verifiers, making it more RL-friendly.
- **vs TOUCAN (Xu et al.)**: TOUCAN focuses on scale (1.5M trajectories), while this work pursues "small, high-quality + self-evolution," showing 64 high-quality plan sets can surpass human experts.
- **vs ToolRL / Search-R1**: These address single-turn tool-use RL; this work targets multi-turn interactive settings, introducing the critical dimension of user simulator quality.
- **vs π₀ / GR00T**: Also RL for agents, but robotics uses real environments as ground truth; this work uses synthetic verifiers, lowering cost but with potential fidelity gaps.
- **vs ARENA-RL (tournament RL)**: The latter uses relative ranking to address reward sparsity; this work uses dynamic filtering + large batch to address advantage noise—complementary approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of self-evolving data synthesis + user model SFT + verifier-based RL is new for agent post-training, especially the finding that "user model SFT is key."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three domains × three model scales × separate/mix × data ablation + user model ablation + RL algorithm ablation, comprehensively covering and comparing all mainstream commercial frontier models.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative (data problem + RL problem → two solutions), concise formulas and figures; appendix provides solid training details.
- Value: ⭐⭐⭐⭐⭐ Open-source models achieving or surpassing frontier models on τ²-bench is genuine SOTA, and the framework is reproducible (open code + detailed hyperparameters), directly valuable for industrial deployment of tool-using agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](../../AAAI2026/dialogue/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2026\] GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](../../ACL2026/dialogue/genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](../../ACL2026/dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)

</div>

<!-- RELATED:END -->
