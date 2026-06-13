---
title: >-
  [Paper Note] Position: Agentic AI Orchestration Should Be Bayes-Consistent
description: >-
  [ICML 2026 (Position Paper)][LLM Agent][Bayesian Control Layer] This position paper argues against attempting to make LLMs inherently "Bayesian" (an approach with insurmountable engineering and theoretical hurdles). Inst…
tags:
  - "ICML 2026 (Position Paper)"
  - "LLM Agent"
  - "Bayesian Control Layer"
  - "Expected Utility"
  - "Value of Information"
  - "Agent Orchestration"
  - "Composite Likelihood"
date: 2026-05-08
content_hash: 1e1136fa3a042ad1
---

# Position: Agentic AI Orchestration Should Be Bayes-Consistent

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.00742](https://arxiv.org/abs/2605.00742)  
**Code**: None  
**Area**: Agent / Bayesian Decision Theory / LLM Orchestration / Uncertainty Quantification  
**Keywords**: Bayesian Control Layer, Expected Utility, Value of Information, Agent Orchestration, Composite Likelihood  

## TL;DR
This position paper argues against attempting to make LLMs inherently "Bayesian" (an approach with insurmountable engineering and theoretical hurdles). Instead, it proposes moving Bayesian structures to the **orchestration control layer** of agentic AI. This allows the controller to maintain beliefs over low-dimensional task-level latent variables, update these beliefs based on agent/tool "message observations" using Bayes' rule, and utilize expected utility or value-of-information for routing, stopping, escalation, and budget allocation.

## Background & Motivation
**Background**: LLMs have become the core of modern AI applications. However, the bottleneck for high-value deployments is not "generating plausible tokens" but **decision-making under uncertainty**: when to stop? which tool to invoke? when to ask a clarifying question? when to escalate to a human? Tool calls are expensive, slow, and risky; decision-making is essentially a trade-off between cost, quality, and latency. Bayesian decision theory (Berger 1985, DeGroot 2004) is designed for such problems: maintaining beliefs over latent variables, updating based on evidence, and selecting actions via expected utility or value of information.

**Limitations of Prior Work**: There are two primary paths for integrating Bayesian ideas into LLM systems. (a) **Making the LLM itself Bayesian**: maintaining posteriors over model parameters and performing integration. Bayesian Deep Learning (BDL) has struggled for decades (Laplace, mean-field, Hinton 1993, etc.) without fundamentally changing the SOTA of LLM training like second-order optimization has. Furthermore, the parameter posterior of over-parameterized models as a representation of epistemic uncertainty is highly questioned (Kirsch 2025). Even if LLMs appear "Bayesian in-context" in restricted scenarios, Falck et al. (2024) used martingale tests to show they violate standard properties of Bayesian belief updates in general cases. (b) **Prompt-based heuristics**: Chain-of-Thought, ReAct, and various workflows suffice for short tasks and low risks. However, as tasks lengthen and stacks deepen, evidence correlation, cost trade-offs, and escalation thresholds become difficult to express via fixed workflows.

**Key Challenge**: Decision-making requires uncertainty at the **task-level semantics** (e.g., "will the code pass unit tests?"), whereas LLMs provide **token-level** probabilities—the scales are entirely different. A token distribution can be sharp while the task-level remains highly uncertain, and vice-versa. Additionally, in-context updates of LLMs do not necessarily satisfy exchangeability or martingale properties, making the use of token probabilities as belief states unreliable.

**Goal**: (1) Precisely locate where "agentic AI should be Bayesian"—in the **control layer**, not within the LLM. (2) Provide a list of practical attributes suitable for modern software stacks and human-AI collaboration. (3) Demonstrate the engineering feasibility of this paradigm through three specific examples (code generation, multi-agent debate, routing) and a set of design patterns. (4) Propose a call to action spanning benchmarking, modeling, deployment, and theory.

**Key Insight**: The authors stratify "Bayesian structure" into training, inference, and control layers. This paper focuses on the control layer: treating the LLM as a black-box predictor while the **orchestration** logic maintains an explicit belief state, updated via observation models and selecting actions based on expected utility. This bypasses "parameter posteriors" and places Bayes where it excels—**explicit, low-dimensional decision variables with measurable outcomes**.

**Core Idea**: A Bayesian agentic system is defined by its control layer—maintaining a posterior over task-level latent variables $Y$ (e.g., code correctness, root cause hypotheses, tool reliability), treating LLM outputs as noisy likelihoods, updating via tempered/composite likelihood rules $r_t(y)\propto r_{t-1}(y)\,p_{i_t}(z_t\mid y)^{\alpha_{i_t}}$, and deciding on routing, stopping, or escalation based on expected utility or value-of-information.

## Method

### Overall Architecture
The framework is not a single algorithm but an **architectural template**:

- **Belief state**: The orchestrator maintains a posterior $r_t(\cdot)=p(\cdot\mid\mathcal{D}_{1:t})$ defined over low-dimensional, decision-relevant latent variables (not LLM parameters).
- **Observation model**: Each agent $i$ has a likelihood $p_i(z\mid y)$ learned from historical "message-outcome" pairs; it can also be discriminative $q_i(y\mid z_t)$.
- **Reliability weights**: $\alpha_i>0$ controls the tempering intensity of each agent's likelihood, derived from exponential weights of cumulative log-loss $w_i\propto\exp(-\beta L_i)$, normalized to $\alpha_i=\alpha_\text{max}\tilde w_i$.
- **Decision policy**: Actions $a_t^\star=\arg\max_a\sum_h u(a,h)r_t(h)$, or based on value-of-information to decide whether to continue or stop.
- **Dependence handling**: When agents share prompts, base models, or retrieval pipelines, correlations are handled via likelihood tempering, dependence-aware pooling, or latent agent-state assumptions for conditional independence.

### Key Designs

1. **Task-Level Latent Belief + Composite Likelihood Bayes Update**:
    - **Function**: Expresses uncertainty over low-dimensional variables relevant to orchestration (task outcomes, hypotheses, tool capabilities) rather than tokens or parameters.
    - **Mechanism**: Taking code generation as an example, $Y\in\{0,1\}$ represents whether candidate code passes all unit tests. The orchestrator maintains $r_t(y)=p(Y=y\mid\mathcal{D}_{1:t})$, where $\mathcal{D}_{1:t}=\{(i_s,Z_s):s\le t\}$ is the sequence of queried agents and messages. New observations are updated via $r_t(y)\propto r_{t-1}(y)p_{i_t}(z_t\mid y)^{\alpha_{i_t}}$, equivalent to $r_t(y)=r_{t-1}(y)\ell_{i_t}(y;z_t)^{\alpha_{i_t}}/Z$ using discriminative predictions $q_i(y\mid z)$, where the likelihood ratio is $\ell_i(y;z)=q_i(y\mid z)/p_0(y)$. $\alpha_i$ is a tempering exponent (generalized Bayes / power-posterior, Bissiri 2016) that automatically dampens the influence of noisy or correlated agents.
    - **Design Motivation**: Traditional Naive Bayes assumes conditional independence, but agents from the same model family are significantly correlated. Tempering is a standard method to absorb these correlations into likelihood intensity, being more robust and easier to implement than complex joint modeling. This elevates "how much to trust an agent" from a heuristic to a learnable parameter.

2. **Value-of-Information Driven Action Selection**:
    - **Function**: Determines which agent to query next, or whether to stop and return a result or escalate to a human.
    - **Mechanism**: Each agent $i$ has a known call cost $c_i>0$. From a Bayesian decision-theoretic perspective, the next agent is selected to maximize the posterior expected utility minus cost: $a_t^\star=\arg\max_a\sum_h u(a,h)r_t(h)$. A call is only made if the expected value of information (VOI) exceeds its cost $c_i$. VOI is strictly defined as the expected difference in utility before and after the call, which can be approximated via one-step lookahead or amortized surrogates.
    - **Design Motivation**: Fixed workflows (e.g., "call 3 agents then ensemble") work for short-horizon/low-stakes tasks but fail to adapt when tasks lengthen or costs are asymmetric (e.g., expensive safety checks vs. cheap runners). VOI explicitly embeds the decision of "when to spend more" into orchestration, providing a unified criterion for routing, stopping, and escalation.

3. **Online Learning of Agent Reliability + Dependence-Aware Evidence Pooling**:
    - **Function**: Tracks agent performance across different tasks/distributions and safely aggregates correlated evidence.
    - **Mechanism**: Defines cumulative log-loss $L_i=\sum_{s:i_s=i}-\log q_i(y_s\mid z_s)$, updated online via exponential weights $w_i\propto\exp(-\beta L_i)$ (Cesa-Bianchi & Lugosi 2006). This is mapped to the tempering coefficient $\alpha_i=\alpha_\text{max}\tilde w_i$. Correlations arising from repeated queries to the same agent are handled either by conditioning on interaction history or expanding the latent state to include agent-specific shared error variables.
    - **Design Motivation**: Orchestration faces two types of corruption: changes in agent capability and cross-message correlation. The former is handled by exponential weights/Bayesian routing; the latter by composite likelihood and dependence-aware pooling. This ensures "conservative" belief convergence, preventing overconfidence from redundant messages while providing verifiable engineering interfaces.

### Loss & Training
The framework does not involve training the LLM but rather **meta-learning for the orchestrator**: (a) learning $q_i(y\mid z)$ from historical interaction logs with outcome labels; (b) online updates for $\alpha_i$; (c) calibration checks using held-out tasks (empirical coverage, proper scoring rules); (d) re-tempering upon detecting distribution drift. The design principle requires that observation models can be continuously recalibrated from measurable outcomes (pass/fail, human ratings, task completion), which is fully compatible with RLHF and online learning engineering practices.

## Key Experimental Results

> Note: As a position paper, this work does not perform large-scale empirical benchmarks but demonstrates design feasibility through three specific examples and extracts a set of 7 actionable properties for Bayesian agentic systems (see Section 2).

### Main Results
Three examples and their corresponding latent variable designs:

| Example (Section) | Orchestration Scenario | Latent $Y$/$H$ | Observation $Z$ | Decision |
|-------------------|-------------------------|----------------|-----------------|----------|
| 4.1 Multi-agent code gen | code generator + retrieval + safety + test runner | $Y\in\{0,1\}$: Pass all unit tests | Candidate code / cites / warnings | When to stop/return, who to call |
| 4.2 Multi-agent debate | Experts debating scientific/policy issues | $H\in\{h_1,\dots,h_k\}$: Root cause hypothesis | Argument messages | When to stop, escalate to human |
| C Routing (Appx) | Routing tasks across an agent pool | Cross-task competence | Historical performance | Select most suitable agent |

### Ablation Study (Thought Experiments)

| Configuration | Meaning | Argument |
|---------------|---------|----------|
| Belief in parameter space | Bayesian LLM | Falck 2024 proves in-context updates aren't truly Bayesian; poor epistemic representation; high engineering cost. |
| Belief in token probabilities | Next-token distribution as belief | Kuhn 2023 / Aichberger 2025: syntactic uncertainty $\neq$ semantic; sharp token dist does not imply task confidence. |
| Prompt-based heuristic only | ReAct / Reflexion, etc. | Sufficient for short-horizon; fails for long-horizon or asymmetric costs. |
| Robust control / Bandits | UCB, worst-case | Suited for reward-driven only; doesn't naturally express VOI or escalation. |
| Task-level latent + VOI + Composite (Ours) | Bayes-consistent control layer | Explicit interface, principled dependence handling, compatible with human-in-the-loop. |

### Key Findings
- **Uncertainty scale mismatch is the crux**: Token uncertainty $\neq$ task uncertainty $\neq$ parameter uncertainty. Agentic decisions require task-level latents; separating this from LLM internals is more feasible than making LLMs internally Bayesian.
- **Composite likelihood + tempering addresses correlation**: Correlated outputs are inevitable; power-posteriors with learned $\alpha_i$ allow for the automatic suppression of unreliable or redundant agent signals.
- **Value-of-information is an explicit criterion for spending**: Replacing heuristic workflows with VOI unifies routing, stopping, and escalation under a single decision-theoretic objective.
- **Seven properties ensure engineering feasibility**: Including ease of access at the control layer, compatibility with typed schemas, simple control knobs (e.g., confidence thresholds), and support for human feedback as probabilistic observations.

## Highlights & Insights
- **The "where to put Bayes" question**: The primary contribution is reframing the vague "agentic AI should be Bayesian" into a precise proposition: not in parameter space, not in token space, but in the task-level latent + control policy. This allows BDL tools (VOI, composite likelihood, etc.) to find a grounded application in the age of agents.
- **Tempering as an "elegant engineering compromise"**: Modeling full correlations between LLMs is functionally impossible; using $\alpha_i$ as a single-knob proxy preserves probabilistic interpretation while acknowledging noise.
- **VOI as a principled stopping rule**: In practice, agents often over-call (costly) or under-call (inaccurate) tools. VOI quantifies the value of an additional step, which is more robust than manually tuned heuristics.
- **A bridge across sub-communities**: Connects PAC-Bayes, generalized Bayes, and Bayesian filtering into a cohesive agentic orchestration narrative.

## Limitations & Future Work
- **Misspecified observation models**: If $q_i(y\mid z)$ fails under distribution drift, calibration erodes. Systems must monitor rolling diagnostics and trigger re-tempering or escalation.
- **Likelihoods of high-dimensional messages**: Mapping text-level $Z$ to latent $y$ likelihoods remains an open problem, typically requiring embedding-based discriminative proxies.
- **Computation cost of VOI**: Multi-step VOI can explode exponentially on complex agent calling graphs; amortized controllers or one-step approximations are suggested.
- **Dependence on measurable outcomes**: Tasks with non-binary success (creative writing) require more sophisticated domain engineering for belief states.

## Related Work & Insights
- **vs. Mainstream BDL**: Moves away from parameter-space Bayes, which the paper deems theoretically and practically ill-suited for LLM engineering.
- **vs. "Are LLMs Bayesian" studies**: Uses evidence of LLM non-martingale behavior to argue for an external control-layer approach.
- **vs. Prompt-based heuristics**: Argues that while ReAct/Reflexion work for simple cases, Bayesian control is necessary for long-horizon, high-cost, or safety-critical tool ecosystems.
- **Insight**: Teams aiming to industrialize agents should move toward belief logging and VOI-based routing in the orchestrator.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Not a new algorithm, but a **precise reframing** that provides a concrete design template for "Bayesian Agents."
- **Experimental Thoroughness**: ⭐⭐⭐ As a position paper, it lacks end-to-end empirical benchmarks, relying instead on case studies and theoretical derivation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exceptionally clear structure, strong logical flow from "where to place Bayes" to practical engineering attributes.
- **Value**: ⭐⭐⭐⭐ Defines clear boundaries for the BDL and Agent communities; likely to guide a new wave of research in principled orchestration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Assistive Agents Need Accessibility Alignment](position_assistive_agents_need_accessibility_alignment.md)
- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](../../ACL2026/llm_agent/how_adversarial_environments_mislead_agentic_ai.md)
- [\[NeurIPS 2025\] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading](../../NeurIPS2025/llm_agent/orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICML 2026\] NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration](naviagent_graph-driven_bilevel_planning_for_scalable_tool_orchestration.md)

</div>

<!-- RELATED:END -->
