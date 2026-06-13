---
title: >-
  [Paper Note] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems
description: >-
  [ICML 2026][Multi-Agent][Multi-Agent Systems] MASPO end-to-end jointly optimizes persona prompts for the entire multi-agent pipeline without relying on labels through multi-granularity joint evaluation (Local Validity +…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-Agent Systems"
  - "Joint Prompt Optimization"
  - "Credit Assignment"
  - "Evolutionary Beam Search"
  - "Misalignment Sampling"
date: 2026-05-08
content_hash: d68186d1dd98e139
---

# MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems

**Conference**: ICML 2026  
**arXiv**: [2605.06623](https://arxiv.org/abs/2605.06623)  
**Code**: https://github.com/wangzx1219/MASPO  
**Area**: LLM / Agent / Prompt Engineering  
**Keywords**: Multi-Agent Systems, Joint Prompt Optimization, Credit Assignment, Evolutionary Beam Search, Misalignment Sampling

## TL;DR
MASPO end-to-end jointly optimizes persona prompts for the entire multi-agent pipeline without relying on labels through multi-granularity joint evaluation (Local Validity + Lookahead Potential + Global Alignment) and misalignment-case-driven evolutionary beam search, achieving an average improvement of approximately 2.9 points across 6 tasks.

## Background & Motivation

**Background**: LLM-based Multi-Agent Systems (MAS) currently rely primarily on human-written persona prompts for orchestration: decomposing tasks into heterogeneous agents that collaborate in a specific communication topology. While their overall performance is significantly higher than single agents, prompt optimization has matured for single agents (e.g., APE, OPRO, DSPy / MIPRO, TPE, SPO) but faces challenges in MAS.

**Limitations of Prior Work**: Directly applying these methods to MAS is problematic. First, traditional optimizers rely on "final answer vs. ground truth" scoring, but intermediate agents output reasoning, reflections, or drafts where no labels exist for direct comparison—a classic **credit assignment** problem. Second, Bayesian searches like TPE / MIPRO / MASS use static discrete candidate sets, unable to perform open-ended prompt generation. Third, self-supervised solutions (like SPO) only compare if "the current output is better than the previous one," remaining at the level of isolated single-agent comparisons without reflecting how prompt changes propagate through the causal chain to downstream agents.

**Key Challenge**: MAS prompts exhibit **functional coupling**—changing an upstream $p_j$ alters the input distribution $\mathcal{C}_i$ for a downstream agent $v_i$ (covariate shift), making the optimization landscape inherently non-stationary. Furthermore, a locally optimal prompt may be a **Local-Global Misalignment** case, where the output is syntactically correct but leads downstream agents astray.

**Goal**: To jointly optimize the set of $N$ agent prompts $\mathcal{P}=\{p_i\}_{i=1}^N$ for the entire MAS without ground truth labels, while (1) solving credit assignment; (2) explicitly identifying and fixing local-global misalignments; and (3) handling non-stationary collaborative optimization.

**Key Insight**: The authors observe that as long as "new prompt vs. reference prompt" can be compared at local, lookahead, and global granularities, a reward signal sensitive to the causal chain can be constructed without labels. By explicitly treating "local wins but lookahead/global losses" as hard negatives and feeding them back to the prompt generator, the search can directionally repair coordination breakpoints.

**Core Idea**: An evolutionary beam search framework with coordinate ascent that combines "topological ordering + multi-granularity joint reward + misalignment case sampling + Beam Refresh," allowing each agent's prompt to evolve based on its contribution to the entire causal chain rather than isolated outputs.

## Method

### Overall Architecture
MAS is formalized as a directed communication graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, where each agent $v_i$ produces an output $o_i=f_i(p_i,q,\mathcal{C}_i)$ determined by the LLM function $f_i$ and prompt $p_i$, with $\mathcal{C}_i$ representing predecessor outputs in topological order. The objective is to find $\mathcal{P}^*$ that maximizes the system-level reward $R(\Phi(\mathcal{G},\mathcal{P},q),o_{glob}^*)$. MASPO's main loop optimizes agents sequentially by topological order: (i) generating candidate prompts using execution traces → (ii) scoring with multi-granularity rewards and mining misalignment cases → (iii) selecting Top-K in a beam search with refresh → (iv) freezing the agent after $T$ steps and moving downstream. The outer loop repeats for $D$ rounds.

### Key Designs

1.  **Multi-granularity Joint Reward**:
    - **Function**: Evaluates if a candidate prompt $p_{cand}$ is "truly better" than a reference prompt $p_{ref}$ without labels.
    - **Mechanism**: The reward is a weighted average of three comparison metrics: $R=\frac{1}{|\mathcal{B}|}\sum_k[\alpha\cdot\mathbb{I}(o_i'\succ o_i)+\theta\cdot\mathbb{I}(o_{glob}'\succ o_{glob})+\beta\cdot\frac{1}{|\mathcal{N}_{out}(v_i)|}\sum_{v_j}\mathbb{I}(o_j'\succ o_j)]$. The first term (Local Validity) tests role compliance; the second (Global Alignment) measures the impact on the final system output; the third (Lookahead Potential) is a topology-aware "downstream ripple" effect—feeding the new candidate's output to immediate successors to see if their outputs improve. Judgments are provided by an LLM Evaluator $\mathcal{M}_{eval}$.
    - **Design Motivation**: Looking only at local validity can be deceptive ("locally perfect but downstream failure"), while looking only at global alignment results in sparse signals. This three-layer combination allows credit assignment to propagate along the causal chain while remaining label-free.

2.  **Misalignment Case Mining + Injected Generation**:
    - **Function**: Identifies "local win, global loss" samples as hard negatives for the prompt generator.
    - **Mechanism**: Samples where $\mathbb{I}(o_i'\succ o_i)=1$ but $\mathbb{I}(\text{Lookahead})=0$ or $\mathbb{I}(o_{glob}'\succ o_{glob})=0$ are stored in a buffer $\mathcal{B}_{mis}$. During trace-guided generation, the Optimizer LLM $\mathcal{M}_{opt}$ is given these $(q,\mathcal{C},o)$ triplets as few-shot context and instructed to fix the gap: "In these scenarios, you appeared correct but dragged the system down; please revise."
    - **Design Motivation**: Implicit bugs where outputs seem role-compliant but ruin the next step are the hardest to debug manually. Automatically locating and feeding these to the generator is significantly more efficient than random variation.

3.  **Evolutionary Beam Search with Beam Refresh + Topological Scheduling**:
    - **Function**: Efficiently searches high-dimensional prompt space while handling non-stationarity.
    - **Mechanism**: Maintains a Top-$K$ beam where each candidate accumulates a reward $J(p')=R(p',p_{parent};\mathcal{B}_{iter})+J(p_{parent})$. Interleaved topological scheduling freezes an agent after $T$ steps to prevent upstream agents from overfitting to obsolete downstream behaviors. **Beam Refresh** is critical: when an agent is revisited, stale scores are discarded and replaced with a "centered win rate" $J_{new}(p)=R(p,p_{best};\mathcal{B}_{iter})-0.5$ relative to the current global best $p_{best}$.
    - **Design Motivation**: Because peer agents evolve, old scores in the beam correspond to outdated contexts. Re-anchoring to the current best baseline ensures the search advances on the "latest performance manifold."

### Loss & Training
There is no gradient descent; the process is a prompt search of "Generate → Evaluate → Evolve." The backbone is Qwen3-8B (standard inference mode, internal reasoning disabled), with Gemini-2.5-pro as both Optimizer and Evaluator. Batch size $|\mathcal{B}|=10$, and the label-free sample pool contains only dozens of instances.

## Key Experimental Results

### Main Results
Six tasks (Math: MATH-500 / AGIEval-MATH / AQuA; Reasoning: GPQA-Diamond; Code: MBPP / HumanEval-ET) were tested across two MAS architectures (Sequential, Hierarchical) against TPE and SPO baselines.

| MAS Architecture | Optimization Method | MATH-500 | GPQA | HumanEval-ET | Avg |
|---|---|---|---|---|---|
| Sequential | None | 75.10 | 47.73 | 68.90 | 65.31 |
| Sequential | + TPE | 75.80 | 48.04 | 70.12 | 66.49 |
| Sequential | + SPO | 77.20 | 49.52 | 67.94 | 66.56 |
| Sequential | **+ MASPO (Ours)** | **77.80** | **58.08** | **73.78** | **70.39** |
| Hierarchical | None | 77.60 | 50.63 | 71.34 | 68.32 |
| Hierarchical | + SPO | 77.80 | 51.01 | 73.39 | 69.01 |
| Hierarchical | **+ MASPO (Ours)** | **78.40** | **54.04** | **76.83** | **71.05** |

The most significant improvement occurs in GPQA: on Sequential, MASPO outperforms SPO by 8.56 points, indicating that joint optimization yields the highest returns on tasks requiring complex "multi-agent collaborative reasoning."

### Ablation Study

| Configuration | Avg | Description |
|---|---|---|
| MASPO (Full) | 70.39 | Full framework |
| Serial Search (w/o beam) | 68.10 | Search strategy contribution approx. 2.3 |
| Single Cycle (one round) | 68.19 | Topological scheduling is significant |
| Single Agent + SPO | 66.86 | Degenerates to single-agent baseline |
| + Our Beam Search | 68.87 | Replacing search strategy alone adds +2 |
| w/o Beam Refresh | (Drop) | Beam Refresh is a key stabilizer |

### Key Findings
- Joint optimization provides far greater gains for complex tasks requiring multi-step reasoning chains (GPQA, MBPP) compared to single-step solvable tasks (AQuA), proving the efficacy of the Lookahead Potential term.
- MASPO remains consistently superior when using a weaker backbone (Qwen3-8B) or suboptimal initial prompts, suggesting the framework is not overly sensitive to component strength.
- The misalignment sampling parameter $K_{mis}$ has a sweet spot: too small leads to standard trace-guided generation, while too large introduces noise.

## Highlights & Insights
- Explicitly formalizing "local win but global loss" as a detectable, mineable event is the most innovative aspect—it converts MAS coordination bugs, which previously required manual debugging, into a source of optimization signals.
- Lookahead Potential is a highly transferable design: any system with causal dependencies (multi-step agents, RAG pipelines, tool-use chains) can benefit from an evaluation perspective that asks, "Does your output help the downstream agent do better?"
- Beam Refresh using a centered win rate of $-0.5$ handles covariate shift more efficiently than re-evaluating everything, minimizing redundant computation while maintaining search stability.

## Limitations & Future Work
- The process relies on an Evaluator LLM for win/loss judgments, which may amplify evaluator bias. While robustness tests were done with Qwen3-8B, the risk of "systematic misjudgment" on specific task types remains.
- The outer loop ($D$ rounds $\times N$ agents $\times T$ steps) incurs significant LLM call costs. The authors tested on small-scale tasks and did not provide total token overhead.
- The communication graph must be a DAG to define a topological order; truly interactive multi-agent systems with cycles (e.g., debates, negotiations) require further design.

## Related Work & Insights
- **vs. MIPRO / MASS (TPE)**: While they perform Bayesian selection within fixed discrete prompt pools, MASPO utilizes open-ended generation and evolution, offering a higher degree of freedom.
- **vs. SPO**: SPO uses output comparison for single-agent self-supervised optimization; MASPO extends this to the pipeline level with downstream propagation assessment.
- **vs. DSPy / TextGrad**: Those frameworks focus more on "textual backpropagation" for local prompt tuning, whereas MASPO addresses the systemic issue of non-stationarity caused by prompt coupling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Multi-granularity joint reward + misalignment sampling + Beam Refresh is the first complete set to solve credit assignment and non-stationarity in MAS prompt optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 tasks $\times$ 2 architectures $\times$ multiple ablations across math/reasoning/code, though scalability tests for longer reasoning chains (5+ agents) are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Formulas and the flowchart (Fig 1) are clear, motivations are well-articulated, and appendix prompt templates are complete.
- **Value**: ⭐⭐⭐⭐ Provides a prompt optimization tool that can be directly applied to any DAG-based MAS, highly practical for practitioners building agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPOB: Multi-Agent Prompt Optimization via GNN Surrogate + LinUCB + Coordinate Ascent](maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/multi_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](../../NeurIPS2025/multi_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)

</div>

<!-- RELATED:END -->
