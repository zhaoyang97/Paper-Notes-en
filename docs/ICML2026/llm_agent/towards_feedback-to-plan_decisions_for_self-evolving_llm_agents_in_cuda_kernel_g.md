---
title: >-
  [Paper Note] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation
description: >-
  [ICML 2026][LLM Agent][self-evolving agent] Focusing on self-evolving LLM agents for CUDA kernel generation, this paper proposes CUDAnalyst. It performs generation-level intervention by "freezing intermediate program sta…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "self-evolving agent"
  - "CUDA kernel"
  - "feedback attribution"
  - "trajectory freezing"
  - "Banzhaf value"
date: 2026-05-08
content_hash: f7c789ef710a95f1
---

# Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation

**Conference**: ICML 2026  
**arXiv**: [2605.26720](https://arxiv.org/abs/2605.26720)  
**Code**: https://github.com/yuxuan-z19/cudanalyst (Available)  
**Area**: LLM Agent / Code Generation / CUDA Kernel Optimization  
**Keywords**: self-evolving agent, CUDA kernel, feedback attribution, trajectory freezing, Banzhaf value

## TL;DR
Focusing on self-evolving LLM agents for CUDA kernel generation, this paper proposes CUDAnalyst. It performs generation-level intervention by "freezing intermediate program states and selectively injecting/masking feedback." Utilizing the Banzhaf value from coalitional game theory, it deconstructs the marginal contributions and high-order interactions of three feedback types: debugger, analyzer, and profiler. The study yields four key findings, such as "explicit plans are only useful when feedback is aligned" and "plans from strong models can be transferred to weaker models of the same family." Based on these, the CuGEdit plugin was designed, achieving a $2.08\times$–$10.32\times$ speedup over torch.compile.

## Background & Motivation

**Background**: Current automatic CUDA kernel generation has shifted from one-shot synthesis to "self-evolving agents." In each generation, the LLM reads output from a debugger, static analyzer, and profiler of the previous generation, formulates a Plan, and generates new code. Frameworks like FM Agent, STARK, ConCuR, and OpenEvolve have reported significant acceleration.

**Limitations of Prior Work**: Mainstream evaluation typically uses *end-to-end ablation*—disabling a feedback source and re-running the entire evolution trajectory from the start or a checkpoint to compare final speedup. This paradigm suffers from two issues: (1) Small early perturbations are amplified by subsequent generation plans, causing "feedback effects" and "trajectory drift" to become completely entangled; (2) Aggregating an entire trajectory into a single scalar erases fine-grained signals regarding which feedback played what role in which generation.

**Key Challenge**: The essence of self-evolution is *cross-generation coupling*, whereas attribution requires causal effects at a *fixed decision point*—the two are naturally in conflict. As long as the agent is allowed to evolve on its own, it is impossible to distinguish between "changes in feedback" and "deviations in trajectory."

**Goal**: (1) Design an analysis layer capable of performing feedback intervention at fixed decision points; (2) Quantify the marginal contribution and pairwise interaction of each feedback using game-theoretic tools; (3) Verify if conclusions remain stable across different backbones, workloads, and evolutionary operators; (4) Encapsulate stable patterns into pluggable modules to accelerate real-world systems.

**Key Insight**: The authors observe that feedback attribution must occur at a *fixed generation*. By *freezing* the generated program state of a specific generation as a fixture, one can repeatedly replay the plan on the same code, varying only the feedback input. Any differences in the plan or resulting code can then be attributed solely to the feedback itself.

**Core Idea**: Decouple "self-evolution" and "feedback attribution" in the time dimension. Evolution proceeds as usual to produce trajectory snapshots, while attribution performs controlled patching on each snapshot, using Banzhaf values for a game-theoretic decomposition of feedback coalitions.

## Method

### Overall Architecture
CUDAnalyst is an analysis layer "sandwiched" between a self-evolution framework (e.g., OpenEvolve) and an LLM planner. It explicitly separates feedback generation, plan generation, and code generation for each iteration:

1.  **Feedback Generation**: The original framework runs as usual to produce the $g$-th generation program $P_g$ and its reference cache. CUDAnalyst uses three analysis modules—Debugger (compile/runtime errors), Analyzer (static features via Tree-sitter), and Profiler (NCU/cProfile metrics)—to transform $P_g$ into a structured profile, which can be further compressed by a SummaryAgent.
2.  **Freezing + Intervention**: $P_g$ and references are frozen as immutable snapshots. "Selective switches" are applied to the feedback report to construct $2^N$ feedback coalitions $S \subseteq \{d,a,p\}$. For each coalition, a PlanAgent with fixed prompts/temperature produces a plan, followed by a fixed code generator.
3.  **Generation-level Evaluation**: Each $(g, S)$ combination is executed 10 times. Statistics on compiled, pass, and fast results are collected to formulate the characteristic function $v(S)$.
4.  **Game-theoretic Attribution**: The Banzhaf value $\phi_i$ calculates the average marginal contribution of each feedback, while Grabisch-Roubens interaction terms $\sigma_{ij}$ calculate pairwise synergy/conflict, resulting in a fine-grained feedback-to-plan attribution map.

The process ensures the *Plan only observes current feedback and not historical plans*, thereby excluding "evolutionary memory" from the attribution.

### Key Designs

1.  **Trajectory Freezing & Patching Intervention**:
    - **Function**: Under the premise that the $g$-th generation program state, reference set, planner, prompt, and decoding hyperparameters are all locked, it uniquely attributes subsequent plan/code differences to the feedback set by switching which feedbacks are sent to the PlanAgent.
    - **Mechanism**: Caches $P_g$ as an immutable snapshot and fixes references to avoid resampling pollution. It tests $2^3=8$ combinations of the feedback trio $\{d, a, p\}$ (including the empty set baseline $v(\emptyset)$). Unlike traditional ablation which re-runs entire trajectories from scratch, this method's computational budget is $O(2^{|N|} k)$ rather than $O(\text{generations} \times k)$, and results are strictly comparable since they are based on identical code fixtures.
    - **Design Motivation**: Self-evolving scenarios naturally violate "IID" assumptions. Traditional ablation cannot mathematically support causal attribution; physical "slicing" in the time dimension is required to run controlled experiments.

2.  **Coalitional-Style Feedback Attribution**:
    - **Function**: Quantifies "how much each feedback contributes" and "synergy/redundancy between feedback pairs" into readable scalars.
    - **Mechanism**: Models feedback attribution as a cooperative game $\mathcal{G}=(N, v)$, where players $N=\{\text{debugger}, \text{analyzer}, \text{profiler}\}$ and $v(S)$ is the expected generation-level success. Marginal contribution uses the Banzhaf value:
      $$\phi_i(v) = \frac{1}{2^{|N|-1}} \sum_{S \subseteq N \setminus \{i\}} [v(S \cup \{i\}) - v(S)]$$
      Pairwise interaction uses Grabisch-Roubens terms:
      $$\sigma_{ij} = v(\{i,j\}) - v(\{i\}) - v(\{j\}) + v(\emptyset)$$
      Positive values indicate complementarity; negative indicates redundancy. Banzhaf is chosen over Shapley because feedback in plan decisions appears *simultaneously* without an arrival order.
    - **Design Motivation**: Comparing simple "on/off" differences overestimates individual roles due to high synergy (e.g., an analyzer finding a race condition before a debugger can locate it). Coalitional games provide a standard tool for fair credit assignment and interaction isolation.

3.  **CuGEdit — From Invariant Insights to Actionable Design**:
    - **Function**: Encapsulates stable findings—"feedback alignment is crucial," "multi-feedback synergy," "strong-to-weak plan transfer," and "summarization helps weak models"—into a module for any self-evolving framework.
    - **Mechanism**: Includes three components: (a) Kernel-similarity-aware activation: activates full feedback only when the kernel is similar to cached cases to save tokens; (b) Feedback summarization: compresses profiles into structured summaries for weak models; (c) Strong-to-weak plan distillation: injects plans from strong models (e.g., DeepSeek-R1) into the context of weak models (e.g., DeepSeek-V3.2).
    - **Design Motivation**: To demonstrate that attribution insights can directly improve system performance, addressing "why attribution matters" by informing designers when to use specific feedback or distillation.

## Key Experimental Results

### Main Results

| Research Question | Setting | Key Findings |
|---|---|---|
| RQ0 Is explicit planning useful | 2×2: {implicit, explicit} × {no feedback, full feedback} | P+NF (Plan without feedback) consistently drops performance; P+F (Plan with feedback) improves across all models, especially for weak models. |
| RQ1 Which feedback is critical | Banzhaf Value + $\sigma_{dap}$ 3rd-order interaction | Early stage (gen 0-2) contributions are sparse; late stage (gen 5-7) interaction terms dominate. Analyzer drives compilation, Profiler drives speed. |
| RQ2 Can Summary replace Plan | NP+S vs P+S vs P+F | Summarization (P+S) aids weak models significantly but offers little to strong models. NP+S is weaker than P+S, proving summary $\neq$ plan. |
| RQ3 Strong $\rightarrow$ Weak Plan Distillation | Injustice strong plans into weak model context | Intra-family transfer (R1 $\rightarrow$ V3.2) yields maximum Gain; cross-family is partially effective. |
| CuGEdit Performance | KernelBench Level 3 vs torch.compile | $2.08\times$ – $10.32\times$ acceleration, surpassing Prev. SOTA. |

### Ablation Study

| Configuration | Key Metric Change | Description |
|---|---|---|
| Implicit (OpenEvolve Default) | Baseline | Planning and code generation coupled in a single step. |
| P+NF (Plan only) | Overall Decrease | Proves plan tokens themselves are useless without content. |
| P+F (Plan + Feedback) | Stable Gain | Feedback is the critical component. |
| DP (Dummy Plan) | Decrease in weak models | Excludes "token budget/structure" as confounding factors. |
| P+RF (Random Feedback) | Decrease for all | Proves "alignment" is key, not just "information volume." |
| Cross Backbone | Stable $\sigma_{ij}$ patterns | Tool synergy structure is independent of the base model. |
| Cross Evolution Operator | High trajectory sync | Attribution structure is decoupled from the evolutionary operator. |
| Cross Domain (CPU Numba) | $12.5\times$ peak acceleration | Attribution framework is not tied solely to CUDA. |

### Key Findings
- **Plan tokens provide no intrinsic value**: P+NF and DummyPlan do not improve, or even hinder, performance. The role of a plan is to "organize and reuse aligned feedback."
- **Late-stage reliance on interaction**: As generations progress, $\sigma_{dap}$ dominates score growth—failure modes become coupled (e.g., a race condition affecting both compilation and performance), requiring multi-feedback consensus.
- **Strong models are insensitive to plan semantics**: DummyPlan barely affects R1, but P+RF (Random Feedback) causes a drop—strong models have self-correction but are not immune to "misleading signals."
- **Intra-family transfer friendliness**: Plan transferability is limited by representational compatibility. Cross-family gains are lower, suggesting model lineage matters for agent ensembles.
- **CuGEdit turns analysis into productivity**: The three laws discovered by attribution—feedback alignment, tool synergy, and intra-family distillation—achieve $10.32\times$ speedup, proving the analysis is not just "post-hoc explanation."

## Highlights & Insights
- **Methodological Highlight**: Introducing coalitional game theory (Banzhaf / Grabisch-Roubens) into LLM agent attribution clearly separates "individual contribution" and "interaction," offering more insight than simple leave-one-out methods. This can be extended to any multi-tool agent analysis.
- **Universality of Trajectory Freezing**: Applicable to any long-horizon, multi-feedback agent. "Frozen snapshots + controlled patching" is a general cure for trajectory drift in ablation studies.
- **Counter-intuitive Insight**: Experience shows plans are useless without feedback; the fact that DummyPlan doesn't hurt much suggests LLMs can synthesize structure on the fly from feedback. This challenges the "longer Chain-of-Thought is always better" narrative.
- **Transferable Trick**: The P+RF (Random Feedback) control should be a standard baseline for any planning agent paper to verify if the plan signal is useful or just the token length.

## Limitations & Future Work
- **Ours Acknowledges**: The method assumes intermediate program states can be frozen, but what constitutes a "semantic state element" in CUDA evolution remains an open question in compiler research.
- **Potential Improvements**: (1) The coalition is limited to $\{d,a,p\}$; $2^N$ scales exponentially. Monte-Carlo sampling for Banzhaf could support more sources. (2) Evaluated mainly on PolyBench-ACC; "long-tail operators" (e.g., fused MoE) lack direct verification. (3) Prompt stability for PlanAgent and CodeAgent was not analyzed.
- **Big Question**: Attribution results may rely on the quality of feedback implementation—if a finer-grained profiler were used, would the relative $\phi_p$ flip? This requires "attribution of the attribution."

## Related Work & Insights
- **vs OpenEvolve / FM Agent**: These focus on *how to make agents write better*, while CUDAnalyst focuses on *how to scientifically analyze why they write well*.
- **vs Traditional E2E Ablation**: This paper systematically refutes the "E2E ablation is enough" approach through P+RF and DummyPlan controls.
- **vs Shapley-based prompt valuation**: Uses Banzhaf because feedback is parallel, not sequential, in the decision process—a subtle but important modeling distinction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing Banzhaf attribution and trajectory freezing provides a reusable paradigm for agent attribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive generalizations across workloads, operators, and domains, plus the CuGEdit E2E validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear arguments; Figure 1 effectively frames why E2E is unreliable.
- **Value**: ⭐⭐⭐⭐⭐ A contribution to both methodology (attribution tools) and engineering (CuGEdit speedup); highly useful for LLM4Code/HPC teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](../../ICLR2026/llm_agent/your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](../../CVPR2026/llm_agent/sceneassistant_a_visual_feedback_agent_for_openvoc.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)

</div>

<!-- RELATED:END -->
