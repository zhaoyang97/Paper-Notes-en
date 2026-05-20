---
title: >-
  [Paper Note] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems
description: >-
  [ICML 2026][LLM / Agent / Prompt Engineering][Multi-Agent Systems] MASPO achieves end-to-end joint optimization of role prompts for the entire multi-agent chain without relying on annotations…
tags:
  - "ICML 2026"
  - "LLM / Agent / Prompt Engineering"
  - "Multi-Agent Systems"
  - "Joint Prompt Optimization"
  - "Credit Assignment"
  - "Evolutionary Beam Search"
  - "Misalignment Sampling"
date: 2026-05-08
content_hash: c7ca633c4e718be1
---

# MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems

**Conference**: ICML 2026  
**arXiv**: [2605.06623](https://arxiv.org/abs/2605.06623)  
**Code**: https://github.com/wangzx1219/MASPO  
**Area**: LLM / Agent / Prompt Engineering  
**Keywords**: Multi-Agent Systems, Joint Prompt Optimization, Credit Assignment, Evolutionary Beam Search, Misalignment Sampling

## TL;DR
MASPO achieves end-to-end joint optimization of role prompts for the entire multi-agent chain without relying on annotations, by combining multi-granularity joint evaluation (local validity + lookahead potential + global alignment) and misalignment-case-driven evolutionary beam search. This yields an average improvement of about 2.9 points across 6 tasks.

## Background & Motivation

**Background**: LLM-based Multi-Agent Systems (MAS) currently rely mainly on manually crafted "role prompts" for orchestration: tasks are decomposed for several heterogeneous agents, which collaborate according to a specified communication topology, yielding overall performance superior to single-agent setups. For single agents, prompt auto-optimization is mature, with methods such as APE, OPRO, DSPy / MIPRO, TPE, and SPO.

**Limitations of Prior Work**: Directly applying these methods to MAS leads to major issues. First, traditional optimizers score based on "final answer vs ground truth," but intermediate agent outputs are reasoning/reflection/drafts without direct labels—this is a classic **credit assignment problem**. Second, Bayesian search methods like TPE / MIPRO / MASS fix the prompt pool as a discrete candidate set, precluding open-ended generation. Third, self-supervised approaches (e.g., SPO) only compare "is this output better than the last," remaining at the single-agent level and failing to capture how prompt changes propagate causally downstream.

**Key Challenge**: Prompts in MAS are **functionally coupled**—modifying upstream $p_j$ alters the input distribution $\mathcal{C}_i$ for downstream $v_i$ (covariate shift), making the optimization landscape inherently non-stationary. Locally optimal prompts may be "syntactically valid but mislead downstream agents," resulting in **Local-Global Misalignment**.

**Goal**: Under the absence of ground truth, jointly optimize the set of $N$ agent prompts $\mathcal{P}=\{p_i\}_{i=1}^N$ for the entire MAS, while (1) solving credit assignment; (2) explicitly identifying and correcting local-global misalignment; (3) handling non-stationary collaborative optimization.

**Key Insight**: The authors observe that if one can compare "new prompt vs reference prompt" at local, lookahead, and global levels, it is possible to construct causally sensitive reward signals without labels. Moreover, explicitly feeding "locally winning but globally losing" samples as hard negatives to the prompt generator can steer the search to repair coordination breakpoints.

**Core Idea**: Integrate "topological order + multi-granularity joint rewards + misalignment case sampling + evolutionary beam search with refresh" into a coordinate ascent framework, so each agent's prompt evolves based on its contribution to the entire causal chain, not just isolated outputs.

## Method

### Overall Architecture
MAS is formalized as a directed communication graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, where each agent $v_i$ produces output $o_i=f_i(p_i,q,\mathcal{C}_i)$ determined by LLM inference function $f_i$ and prompt $p_i$, with $\mathcal{C}_i$ being the concatenated outputs of predecessors in topological order. The optimization goal is to find $\mathcal{P}^*$ that maximizes the system-level reward $R(\Phi(\mathcal{G},\mathcal{P},q),o_{glob}^*)$. MASPO's main loop optimizes agents sequentially in topological order: for the current target agent, (i) generate candidate prompts from execution traces → (ii) score with multi-granularity joint rewards and mine misalignment cases → (iii) select Top-K in a refreshed beam search → (iv) after $T$ steps, freeze the agent and move downstream; the outer loop repeats for $D$ rounds.

### Key Designs

1. **Multi-Granularity Joint Reward**:

    - **Function**: Evaluates whether candidate prompt $p_{cand}$ is "truly better" than reference prompt $p_{ref}$ without annotations.
    - **Mechanism**: The reward is a weighted average of three comparison metrics: $R=\frac{1}{|\mathcal{B}|}\sum_k[\alpha\cdot\mathbb{I}(o_i'\succ o_i)+\theta\cdot\mathbb{I}(o_{glob}'\succ o_{glob})+\beta\cdot\frac{1}{|\mathcal{N}_{out}(v_i)|}\sum_{v_j}\mathbb{I}(o_j'\succ o_j)]$. The first term, Local Validity, measures local role compliance; the second, Global Alignment, measures impact on the final system output; the third, Lookahead Potential, is a topology-aware "downstream ripple"—feeding the new context to direct successor agents and checking if their outputs improve. Win/loss is judged by LLM Evaluator $\mathcal{M}_{eval}$.
    - **Design Motivation**: Local-only rewards can be deceived by "locally perfect but downstream-breaking" cases; global-only rewards are too sparse for effective optimization. The three-level combination enables credit assignment to propagate along the causal chain, still without labels.

2. **Misalignment Case Mining + Injection-Based Generation**:

    - **Function**: Identifies samples where $\mathbb{I}(o_i'\succ o_i)=1$ but $\mathbb{I}(\text{Lookahead})=0$ or $\mathbb{I}(o_{glob}'\succ o_{glob})=0$ as "locally winning but globally losing" misalignment cases, stores them in buffer $\mathcal{B}_{mis}$, and feeds them back as hard examples to the prompt generator.
    - **Mechanism**: Trace-guided generation no longer mutates blindly, but uses $(q,\mathcal{C},o)$ triplets as few-shot context for the Optimizer LLM $\mathcal{M}_{opt}$ to purposefully generate $p'$. During generation, $K_{mis}$ misalignment samples are preferentially injected—"in these scenarios, you seemed correct but hurt the system, please revise"—forcing new prompts to bridge the local-global gap.
    - **Design Motivation**: The hardest bugs to spot in manual prompt tuning are "outputs that appear compliant but break the next step." Automatically locating and explicitly feeding such bugs to the generator is an order of magnitude more efficient than pure random mutation.

3. **Evolutionary Beam Search with Beam Refresh + Topological Scheduling**:

    - **Function**: Efficiently searches the high-dimensional prompt space while handling non-stationarity.
    - **Mechanism**: Maintains a Top-$K$ beam, with each candidate accumulating reward $J(p')=R(p',p_{parent};\mathcal{B}_{iter})+J(p_{parent})$; interleaved topological scheduling—each agent iterates only $T$ steps before freezing and yielding to downstream, avoiding overfitting to outdated upstream behaviors. **Beam Refresh** is key: whenever an agent is revisited, outdated accumulated scores are discarded, and candidates are rescored using the "centered win rate" relative to the current global best prompt $p_{best}$, $J_{new}(p)=R(p,p_{best};\mathcal{B}_{iter})-0.5$.
    - **Design Motivation**: Since peer agents are constantly evolving, old scores in the beam correspond to obsolete upstream contexts; using them to select the next generation misleads the search. Anchoring to the current best baseline ensures the search always advances on the "latest performance manifold."

### Loss & Training
No gradient descent is used; the entire process is a "generate → evaluate → evolve" prompt search. The backbone is Qwen3-8B (standard inference mode, built-in reasoning off), with both Optimizer and Evaluator using Gemini-2.5-pro. Each iteration uses a mini-batch size $|\mathcal{B}|=10$, and the unlabeled sample pool contains only a few dozen samples.

## Key Experimental Results

### Main Results
Six tasks (Math: MATH-500 / AGIEval-MATH / AQuA; Reasoning: GPQA-Diamond; Code: MBPP / HumanEval-ET), comparing two MAS architectures (Sequential, Hierarchical) and TPE / SPO baselines.

| MAS Architecture | Optimization Method | MATH-500 | GPQA | HumanEval-ET | Avg |
|---|---|---|---|---|---|
| Sequential | None | 75.10 | 47.73 | 68.90 | 65.31 |
| Sequential | + TPE | 75.80 | 48.04 | 70.12 | 66.49 |
| Sequential | + SPO | 77.20 | 49.52 | 67.94 | 66.56 |
| Sequential | **+ MASPO** | **77.80** | **58.08** | **73.78** | **70.39** |
| Hierarchical | None | 77.60 | 50.63 | 71.34 | 68.32 |
| Hierarchical | + SPO | 77.80 | 51.01 | 73.39 | 69.01 |
| Hierarchical | **+ MASPO** | **78.40** | **54.04** | **76.83** | **71.05** |

The most significant improvement is on GPQA: MASPO outperforms SPO by 8.56 points on Sequential, indicating that joint optimization yields the greatest benefit on tasks requiring multi-agent collaborative reasoning.

### Ablation Study

| Configuration | Avg | Notes |
|---|---|---|
| MASPO (Full) | 70.39 | Complete framework |
| Serial Search (no beam) | 68.10 | Search strategy alone contributes ~2.3 |
| Single Cycle (no interleaving, single round) | 68.19 | Topological scheduling is significant |
| Single Agent + SPO | 66.86 | Degrades to single-agent optimization baseline |
| + Our Beam Search | 68.87 | Replacing search strategy alone yields +2 |
| w/o Beam Refresh | (Reported drop in paper) | Beam Refresh is a key stabilizer |

### Key Findings
- Joint optimization yields much greater gains for complex tasks requiring multi-agent relay reasoning (GPQA, MBPP) than for "single-step solvable" tasks (AQuA), confirming the effectiveness of the Lookahead Potential term.
- When the backbone is switched to Qwen3-8B (weaker optimizer/evaluator) and with intentionally suboptimal prompt initialization, MASPO still consistently wins, indicating robustness to component strength.
- There is a sweet spot for misalignment sampling $K_{mis}$: too small degenerates to ordinary trace-guided, too large is dominated by noise.

## Highlights & Insights
- Explicitly formalizing "locally winning but globally losing" as a detectable and mineable event is the most ingenious aspect—it turns MAS coordination bugs, previously only discoverable via manual debugging, into optimization signal sources.
- Lookahead Potential is a highly transferable design: any causally dependent system (multi-step agents, RAG pipelines, tool-use chains) can adopt the evaluation perspective of "not just your own output, but whether your output improves downstream performance."
- Beam Refresh's use of centered win rate $-0.5$ to handle covariate shift is more efficient than simple "full re-evaluation"—it only refreshes when an agent is revisited, minimizing unnecessary computation.

## Limitations & Future Work
- The entire process relies on the Evaluator LLM for win/loss judgments, so evaluator bias may be amplified; the authors conducted robustness experiments with Qwen3-8B but did not assess the risk of systematic misjudgment on certain tasks.
- The outer loop of $D$ rounds × $N$ agents × $T$ steps incurs nontrivial LLM call costs; experiments were only conducted on 6 relatively small-scale tasks, and total prompt token consumption was not reported.
- The communication graph must be a DAG to define topological order; truly interactive multi-agent systems with cycles or dynamic topologies (e.g., debate, negotiation) require additional design.

## Related Work & Insights
- **vs MIPRO / MASS (TPE)**: These perform Bayesian selection in a fixed discrete prompt pool, while MASPO adopts open-ended generation + evolution, offering much higher flexibility.
- **vs SPO**: SPO uses "output comparison" for single-agent self-supervised optimization; MASPO extends this to the chain level and adds downstream propagation evaluation.
- **vs DSPy / TextGrad**: Those frameworks focus more on "textual backpropagation" for local prompt tuning, while MASPO addresses the systemic issue of "non-stationarity from prompt coupling."

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-granularity joint reward + misalignment case sampling + Beam Refresh is the first comprehensive solution to credit assignment and non-stationarity in MAS prompt optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 tasks × 2 architectures × multiple ablations, covering math/reasoning/code, but lacks scalability experiments for longer reasoning chains (5+ agents).
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and flowcharts (Fig 1), each component's motivation is explained, and appendix includes complete prompt templates.
- Value: ⭐⭐⭐⭐ Provides a prompt optimization tool directly applicable to any DAG-shaped MAS, highly practical for industry practitioners building agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/llm_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](../../NeurIPS2025/llm_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
