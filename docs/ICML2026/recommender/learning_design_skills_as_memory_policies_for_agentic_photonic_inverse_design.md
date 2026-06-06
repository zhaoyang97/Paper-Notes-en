---
title: >-
  [Paper Note] Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design
description: >-
  [ICML 2026][Recommender Systems][Memory Policy] SkillPCF reformulates the inverse design of Photonic Crystal Fibers (PCF) as a "memory policy learning" problem. A controller trained via PPO selects Top-K memory operation…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Memory Policy"
  - "Skill Library"
  - "PPO"
  - "Photonic Crystal Fiber (PCF)"
  - "Simulator-in-the-loop"
date: 2026-05-08
content_hash: dcfc698f076428fb
---

# Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design

**Conference**: ICML 2026  
**arXiv**: [2605.29421](https://arxiv.org/abs/2605.29421)  
**Code**: To be confirmed  
**Area**: LLM Agent / Memory Enhancement / AI for Physics  
**Keywords**: Memory Policy, Skill Library, PPO, Photonic Crystal Fiber (PCF), Simulator-in-the-loop  

## TL;DR
SkillPCF reformulates the inverse design of Photonic Crystal Fibers (PCF) as a "memory policy learning" problem. A controller trained via PPO selects Top-K memory operations from an evolvable skill library for each trajectory segment. An executor applies these to the trajectory memory, while MEEP electromagnetic simulation rewards simultaneously optimize both the controller and the skill library. This approach achieves a superior trade-off between design success rate and simulation budget compared to multi-LLM backends and classical optimization baselines.

## Background & Motivation
**Background**: PCF inverse design currently follows two main paths. The first is classical numerical optimization (parameter sweeps, Finite Element/FDTD simulation, Nelder-Mead, etc.), which is simulation-expensive and relies on expert priors. The second is ML acceleration (proxy networks, differentiable optimization), which uses one-shot regression to predict structure-performance mappings to reduce simulation counts.

**Limitations of Prior Work**: Both paths treat each design task as an independent episode—classical methods do not accumulate cross-task knowledge, while ML methods lack interpretability and iterative refinement capabilities. In actual engineering, designers repeatedly try and fail within adjacent parameter ranges. The knowledge of "what failed, why it failed, and what succeeded under which constraints" is high-value signal, but current systems neither retain nor reuse these experiences.

**Key Challenge**: The tension between tight simulation budgets (high cost for each FDTD/Finite Element run) and coupled design objectives (dispersion, confinement loss, and effective refractive index are interconnected) causes methods without memory mechanisms to either over-simulate or prematurely converge to sub-optimal structures in multi-objective scenarios.

**Goal**: To equip the PCF design system with the ability to "remember what is useful, forget what is invalid, and continuously refine memory policies under simulation feedback." This is further divided into three sub-problems: (i) selecting appropriate memory operations for each design segment; (ii) back-propagating sparse design success signals to intermediate memory decisions; and (iii) allowing memory operations themselves to automatically evolve based on failure cases.

**Key Insight**: The authors draw inspiration from recent findings in the LLM-Agent community—memory operations (insert/update/delete/skip) can be treated as learnable policies rather than fixed heuristics. Furthermore, deterministic physical indices returned by the simulator serve as verifiable rewards. Combining these yields an agent framework featuring "simulator-in-the-loop + evolvable skill library."

**Core Idea**: Transforming PCF inverse design into a two-layer closed loop—the inner loop uses PPO to learn a skill-selection controller, while the outer loop uses a designer module to refine or expand the skill library from a failure buffer. This allows the LLM Agent to both consume memory and reshape the memory operations themselves across multiple interaction rounds.

## Method

### Overall Architecture
SkillPCF segments each design trajectory into ordered spans, where each span consists of "current geometric decision + MEEP simulation context + textual description." The system maintains two storage units: (1) a trajectory-specific memory bank $\mathcal{M}$, carrying numerical design evidence for that trace (unit-sensitive parameter-performance pairs, cross-trajectory relationships, etc.); (2) a cross-trajectory shared skill library $\mathcal{S}$, initially containing four PCF-specific memory primitives: InsertTopologyFeature, UpdatePerformanceTrend, DeleteInvalidAssumption, and Skip.

The process is a two-layer loop: the **inner loop** (orange line) selects Top-K skills via a Controller for each span → the Executor performs memory edits → the Physics Environment runs MEEP to provide physical rewards; the **outer loop** (blue line) extracts hard cases from low-score trajectories into a failure buffer $\mathcal{B}^{(e)}$ every epoch $e$, allowing the Skill Designer to propose a new $\hat{\mathcal{S}}^{(e+1)}$, with acceptance or rollback determined by the reward difference on a validation set. This separation of "stable execution in the inner loop + structural evolution in the outer loop" allows the action space to grow or shrink without breaking the policy head.

### Key Designs

1. **Embedding-Aligned Evolvable Skill Selection Head**:
    - **Function**: Allows the controller to provide a valid probability distribution over the current span even as the skill library is modified over time.
    - **Mechanism**: The same embedding model encodes the span context $h_t = f_{\mathrm{ctx}}(x_t, M_t)$ and each skill description $u_i = f_{\mathrm{skill}}(\text{Description}(s_i))$ into the same representation space. Scores are calculated as $z_{t,i} = h_t^\top u_i$ and $p_\theta(i \mid h_t) = \text{Softmax}(z_t)_i$. Since the dimension of $z_t$ adapts to the skill library size $|\mathcal{S}^{(e)}|$, adding or deleting skills does not require retraining the policy head. Top-K sampling without replacement via Gumbel-Top-K yields an ordered set $A_t = (a_{t,1}, \ldots, a_{t,K})$, with the policy probability defined as $\pi_\theta(A_t \mid h_t) = \prod_{j=1}^{K} \frac{p_\theta(a_{t,j} \mid h_t)}{1 - \sum_{\ell<j} p_\theta(a_{t,\ell} \mid h_t)}$.
    - **Design Motivation**: A span often requires composite operations (e.g., inserting a new parameter-performance fact while deleting an invalid assumption). Single-skill selection would waste simulations by splitting these into multiple steps. Embedding alignment solves the issue where standard actor heads cannot handle a dynamic action space.

2. **Delayed Reward Redistribution + Process Reward**:
    - **Function**: Addresses the credit assignment problem where terminal QA performance appears many steps after intermediate memory operations in long-horizon designs.
    - **Mechanism**: Let the terminal episode reward be $R_{\mathrm{final}}$. It is redistributed to each span using exponential decay: $\tilde{r}_t = (1-\beta) R_{\mathrm{final}} \frac{\gamma^{T-t}}{\sum_{k=1}^{T} \gamma^{T-k}} + \beta \mathbf{1}[t=T] R_{\mathrm{final}}$, where $\gamma \in (0,1)$ controls the decay and $\beta \in [0,1]$ retains a pure terminal signal. The final step reward is $r_t = r_{\mathrm{proc},t} + \tilde{r}_t$, where $r_{\mathrm{proc},t}$ is a process reward based on memory construction quality and physical consistency checks.
    - **Design Motivation**: Training PPO solely on terminal rewards leaves intermediate spans without signals for long periods; using pure process rewards might lead the controller to learn "useless trivia" that does not aid the final design. The exponential decay plus the explicit $\beta$ term allows for continuous interpolation and matches the actual training pipeline.

3. **Failure Buffer-Based Skill Evolution + Acceptance Rollback**:
    - **Function**: Enables the skill library to "grow" new operations from training experience without being dragged down by occasional noise.
    - **Mechanism**: In the outer loop, hard cases from low-performance or physically invalid trajectories are collected into $\mathcal{B}^{(e)}$ every epoch $e$. These are clustered by structural regime (e.g., hexagonal, PBG, Kagome) and optical failure type, and representative samples are sent to the Designer. The Designer performs two steps: Diagnosis (identifying missing/misaligned memory operations) → Refinement (editing existing skills or adding structure-aware skills), resulting in a candidate $\hat{\mathcal{S}}^{(e+1)} = \text{Designer}(\mathcal{S}^{(e)}, \mathcal{B}^{(e)})$. Acceptance is determined by the validation reward difference $\Delta J_{\mathrm{val}} = J_{\mathrm{val}}(\theta^{(e+1)}, \hat{\mathcal{S}}^{(e+1)}) - J_{\mathrm{val}}(\theta^{(e+1)}, \mathcal{S}^{(e)})$. Acceptance occurs only if $\Delta J_{\mathrm{val}} \geq 0$; otherwise, it rolls back to $\mathcal{S}^{(e)}$. Post-acceptance, exploration is briefly biased toward new skills.
    - **Design Motivation**: A fixed skill bank has a hard ceiling, but a poorly modified Designer could harm a well-trained controller. The gate formed by the validation reward difference provides an engineering balance of "limited aggression + safe fallback."

### Loss & Training
The controller $\theta$ is optimized using the PPO objective $J(\theta; \mathcal{S}^{(e)}) = \mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid \mathcal{S}^{(e)})} [\sum_t r_t]$, with $\mathcal{S}^{(e)}$ fixed within each outer epoch. Standard PPO settings: $\gamma=0.99$, $\lambda=0.95$, clip $0.2$, entropy $0.01$, with 4 epochs per update, minibatch size 32, and gradient clipping at 0.5. The Controller is an MLP with a hidden size of 256, using AdamW with a learning rate of $1 \times 10^{-4}$. Training consists of 10 outer evolution epochs, with 50 inner interaction epochs each, using a batch size of 32. GPT-4o-mini serves as the LLM judge, Text-Embedding-3-Small for embeddings, and Contriever (depth $k=5$) for the retriever. Training is completed on an A100-40GB. "Calls/q" is used as a hardware-independent simulation budget metric.

## Key Experimental Results

### Main Results
The authors constructed the PCFSkill dataset—479 expert interaction trajectories (covering 8 PCF families: solid-core hexagonal, high-birefringence PM, hollow-core PBG, Kagome, anti-resonant ARF, etc.), totaling 2,507 spans (average 5.23/trace, ~393K tokens, 75.6% design success rate), plus 553 memory-dependent evaluation queries and 596 failure logs.

Comparison of 8 memory-enhanced baselines under Llama4-Scout (no visual) and cross-LLM backends (with visual):

| Backend / Setting | Method | Human ↑ | Judge ↑ | Succ. ↑ | Phys. ↑ | Calls/q ↓ |
|---|---|---|---|---|---|---|
| Llama4-Scout / No Visual | MemoryBank | 7.18 | 6.72 | 30.61 | 45.92 | — |
| Llama4-Scout / No Visual | A-MEM | 6.92 | 6.02 | 36.22 | 38.78 | — |
| Llama4-Scout / No Visual | **SkillPCF** | **8.47** | **8.02** | **60.12** | **68.92** | — |
| MiniMax-M2.5 / With Visual | MemoryBank | 7.22 | 6.18 | 46.94 | 56.63 | 1.12 |
| MiniMax-M2.5 / With Visual | **SkillPCF** | **9.12** | 6.92 | **82.35** | **68.45** | **1.02** |
| Qwen2.5-72B / With Visual | A-MEM | — | 6.00 | 41.33 | 52.04 | 1.10 |
| Qwen2.5-72B / With Visual | **SkillPCF** | — | **7.95** | **78.92** | **65.28** | **1.02** |

While classical optimization methods come close in Phys. score, they average 100 calls/q (two orders of magnitude higher budget), and Succ. is either 0 (NN Predictor) or relies on brute force (Random Search 92.9%). SkillPCF achieves a 60–82% design success rate with 1.02 calls/q, proving to be the optimal point for the simulation budget/design quality trade-off.

### Ablation Study
| Configuration | Key Contribution | Description |
|---|---|---|
| Full SkillPCF | Physics-guided skill + Evolution + Delayed Reward | Full model, Succ. 60–82% |
| Initial 4 skills only (No Evolution) | Skill Designer disabled | No new skills grow from failures; long-tail performance drops |
| Terminal reward only (No Redistribution) | $\beta=1$ | Credit for intermediate memory operations is missing; PPO convergence slow/unstable |
| Skill selection as single action (K=1) | Top-K disabled | Composite memory operations split across spans, increasing simulation counts |

The distribution of memory operations reveals design behavior: INSERT 36% / UPDATE 56% / DELETE 5% / SKIP <1%, proving that "correcting existing beliefs" is more common than "inserting new facts," validating the central role of the UPDATE skill.

### Key Findings
- When switching LLM backends from Llama4-Scout to MiniMax-M2.5 or Qwen2.5-72B, the relative advantage of SkillPCF does not diminish; instead, Succ. rises to 78–82%, suggesting that memory policies are transferable and not strictly dependent on a specific LLM style.
- In the "Without Visual Field" setting, SkillPCF actually obtains higher Phys. scores than classical ML predictors (68.92 vs. 63.30), implying that cross-trace physical consistency accumulated in memory can partially replace visual evidence.
- General memory agents like Mem0 achieve only 3.06% Succ. on PCF tasks, proving that general LLM-memory frameworks are nearly ineffective without physics-grounded skill primitives—this is the strongest evidence for explicitly modeling "domain skills" as learnable actions.

## Highlights & Insights
- Reinterpreting memory operations as an RL action space is not entirely new, but SkillPCF successfully opens the dimension of "action space evolution": the embedding-aligned selection head allows the Designer to modify the skill library without retraining the policy. This structure can be directly transferred to any agent scenario where tool/skill sets evolve during training.
- Physical simulation provides deterministic, verifiable scalar signals, which is exactly the "reward grounding" missing in LLM agents. Treating expensive simulators as reward sources rather than inner-loop callables provides real physical feedback without exploding the budget, representing a pragmatic simulator-in-the-loop paradigm.
- The "failure buffer + acceptance rollback" pair essentially installs a validation set gate for self-evolving agents, a concept applicable to any self-improving LLM tool-calling system to prevent the designer module from damaging converged capabilities.

## Limitations & Future Work
- The dataset contains only 479 trajectories and 553 evaluation queries, which is small for LLM-agent training; the authors also did not report transfer performance on out-of-distribution PCF families (e.g., new anti-resonant variants).
- The reward design depends on MEEP, which is expensive. The PPO training cost (10 outer epochs × 50 inner epochs) was not fully ablated, leaving uncertainty for industrial deployment.
- The Skill Designer is an LLM call, which might introduce bias homologous to the controller ("using skills it wrote well"), lacking an independent third-party judge. Replacing the Designer with a self-play/multi-judge setup using different backends is worth investigating.
- Currently, there are only 4 base skill primitives plus a few evolved ones; if the skill library expands to dozens or hundreds, retrieval-based skill subset selection might be more economical than a Top-K dense softmax.

## Related Work & Insights
- **vs. MemGPT / Reflexion**: These use OS-style hierarchical memory or explicit verbal feedback, but memory operations remain fixed heuristics. SkillPCF makes operations learnable actions and adds physical grounding.
- **vs. A-MEM / LangMem / MemoryOS**: General memory agents typically achieve only 25–40% Succ. on PCF tasks; they lack domain-aware skill primitives and grounded rewards rather than memory capacity.
- **vs. Classic PCF Inverse Design (Gray 2024, Chen 2023)**: Former methods treat each design as an independent optimization task, requiring 100 calls/q; SkillPCF achieves similar Phys. scores with 1.02 calls/q by reusing cross-trajectory experience.
- **vs. Incentivized Exploration (Sellke & Slivkins)**: Both care about learning under sample constraints, but SkillPCF swaps "user incentive budget" for "physical simulation budget," making the framework more broadly applicable than the specific task suggests.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of memory operations as an evolvable action space + simulator-in-the-loop reward grounding is significant in an engineering sense, though individual components (PPO, Gumbel Top-K, Reflexion-style self-refine) are known.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across three LLM backends × visual/non-visual × 8 baselines is reasonable; the PCFSkill dataset is reproducible. However, it lacks generalization tests on cross-dataset and out-of-distribution PCF families.
- Writing Quality: ⭐⭐⭐⭐ The two-layer loop diagram (Figure 2) and dataset statistics (Figure 3) are clear, formula numbering is clean, and the appendix is complete.
- Value: ⭐⭐⭐⭐ Highly demonstrative for AI for Physics / engineering design agents. The "evolvable skill library + simulator-in-the-loop reward" template is transferable to materials, chips, mechanical structures, and other expensive simulation-driven design tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System](incentivized_exploration_with_stochastic_covariates_a_two-stage_mechanism_design.md)
- [\[ACL 2026\] MemRec: Collaborative Memory-Augmented Agentic Recommender System](../../ACL2026/recommender/memrec_collaborative_memory-augmented_agentic_recommender_system.md)
- [\[ICML 2026\] RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents](rgmem_renormalization_group-inspired_memory_evolution_for_language_agents.md)
- [\[ICML 2026\] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](../../ACL2026/recommender/harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)

</div>

<!-- RELATED:END -->
