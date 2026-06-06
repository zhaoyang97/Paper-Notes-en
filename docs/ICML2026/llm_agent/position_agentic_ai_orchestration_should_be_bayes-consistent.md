---
title: >-
  [Paper Note] Position: Agentic AI Orchestration Should Be Bayes-Consistent
description: >-
  [ICML 2026 (Position Paper)][LLM Agent][Bayesian control layer] This position paper advocates: stop trying to make LLMs themselves "Bayesian" (that path is both theoretically and practically infeasible)…
tags:
  - "ICML 2026 (Position Paper)"
  - "LLM Agent"
  - "Bayesian control layer"
  - "expected utility"
  - "value of information"
  - "agent orchestration"
  - "composite likelihood"
date: 2026-05-08
content_hash: 73da31a0a9a082a5
---

# Position: Agentic AI Orchestration Should Be Bayes-Consistent

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.00742](https://arxiv.org/abs/2605.00742)  
**Code**: None  
**Area**: Agent / Bayesian Decision Theory / LLM Orchestration / Uncertainty Quantification  
**Keywords**: Bayesian control layer, expected utility, value of information, agent orchestration, composite likelihood

## TL;DR
This position paper advocates: stop trying to make LLMs themselves "Bayesian" (that path is both theoretically and practically infeasible), and instead move Bayesian structure to the **orchestration control layer** of agentic AI—let the controller maintain a belief over low-dimensional, task-level latent variables, update it via Bayes’ rule on "message observations" returned by agents/tools, and use expected utility or value-of-information for routing, stopping, escalation, and budget allocation.

## Background & Motivation
**Background**: LLMs have become the core of modern AI applications, but the bottleneck for many high-value deployments is not "generating plausible tokens," but **making decisions under uncertainty**: when to stop, which tool to call, when to ask clarifying questions, when to escalate to a human. Tool calls are costly, slow, and risky; decision-making is fundamentally a trade-off among cost, quality, and latency. Bayesian decision theory (Berger 1985, DeGroot 2004) is designed for such problems: maintain a belief over latent variables, update via Bayes’ rule upon receiving evidence, and select actions by expected utility or value of information.

**Limitations of Prior Work**: There are two main approaches to incorporating Bayesian ideas into LLM systems. (a) **Making the LLM itself Bayesian**—maintaining a posterior over model parameters and integrating over them. Bayesian Deep Learning (BDL) has pursued this for decades (Laplace, mean-field, Hinton 1993, etc.), but has not fundamentally changed LLM training SOTA as second-order optimization has; the epistemic uncertainty represented by the posterior over parameters in overparameterized models is also questioned (Kirsch 2025). Even when LLMs appear "in-context Bayesian" in some restricted settings, Falck et al. 2024 have shown via martingale tests that they violate standard properties of Bayesian belief update in general. (b) **Prompt-based heuristics**—chain-of-thought, ReAct, various workflows, which suffice for short, low-risk tasks; but as tasks lengthen and stack depth increases, expressing evidence correlation, cost trade-offs, and escalation thresholds becomes difficult with fixed workflows.

**Key Challenge**: Decision-making requires **task-level semantic** uncertainty (e.g., "will the code pass unit tests"), but LLMs provide **token-level** probabilities—the two are fundamentally different in scale. Token distributions can be sharp while task-level uncertainty remains high, and vice versa; moreover, LLM in-context updates do not necessarily satisfy exchangeability or martingale properties, so treating token probabilities as belief states is unreliable.

**Goal**: (1) Precisely position the slogan "agentic AI should be Bayesian"—it is the **control layer**, not the LLM internals; (2) Provide a practical checklist of properties suitable for modern software stacks and human-AI collaboration; (3) Demonstrate engineering feasibility with three concrete examples (code generation, multi-agent debate, routing) and a set of design patterns; (4) Propose calls to action in benchmarking, modeling, deployment, and theory.

**Key Insight**: The author decomposes "Bayesian structure" into layers—it can be placed in training, inference, or control. This paper focuses on the control layer: LLMs are treated as black-box predictors, but the **orchestration** logic layer maintains an explicit belief state, updates it via an observation model, and selects actions by expected utility. This bypasses "parameter posterior," placing Bayesian methods where they excel—**explicit, low-dimensional, measurable decision variables**.

**Core Idea**: A Bayesian agentic system is defined by its control layer—maintaining a posterior over task-level latent variables $Y$ (e.g., whether code passes tests / which root cause hypothesis / which tool is more reliable), treating LLM outputs as noisy likelihoods, updating via $r_t(y)\propto r_{t-1}(y)\,p_{i_t}(z_t\mid y)^{\alpha_{i_t}}$ (tempered/composite likelihood), and deciding the next routing/stopping/escalation step by expected utility or value-of-information.

## Method

### Overall Architecture
The framework is not a single algorithm, but an **architectural template**:

- **Belief state**: The orchestrator maintains a posterior $r_t(\cdot)=p(\cdot\mid\mathcal{D}_{1:t})$ over low-dimensional, decision-relevant latent variables (not LLM parameters).
- **Observation model**: Each agent $i$ has a likelihood $p_i(z\mid y)$, learned from historical "message-result" pairs; can also be discriminative $q_i(y\mid z_t)$.
- **Reliability weights**: $\alpha_i>0$ controls the tempering strength of each agent’s likelihood, derived from cumulative log-loss via exponential weights $w_i\propto\exp(-\beta L_i)$, normalized to $\alpha_i=\alpha_\text{max}\tilde w_i$.
- **Decision policy**: Actions $a_t^\star=\arg\max_a\sum_h u(a,h)r_t(h)$, or continue/stop based on value-of-information.
- **Dependency handling**: When agents share prompts, base models, or retrieval pipelines, use likelihood tempering, dependence-aware pooling, or conditional independence via latent agent-state assumptions to handle correlations.

### Key Designs

1. **Task-level latent belief + composite likelihood Bayes update**:

    - **Function**: Express uncertainty over low-dimensional variables relevant to orchestration (task outcome / hypothesis / tool capability), not tokens or parameters.
    - **Mechanism**: For code generation, $Y\in\{0,1\}$ indicates whether candidate code passes all unit tests. The orchestrator maintains $r_t(y)=p(Y=y\mid\mathcal{D}_{1:t})$, where $\mathcal{D}_{1:t}=\{(i_s,Z_s):s\le t\}$ is the sequence of queried agents and messages. New observations update via $r_t(y)\propto r_{t-1}(y)p_{i_t}(z_t\mid y)^{\alpha_{i_t}}$, equivalent to $r_t(y)=r_{t-1}(y)\ell_{i_t}(y;z_t)^{\alpha_{i_t}}/Z$ for discriminative predictions $q_i(y\mid z)$, where the likelihood ratio $\ell_i(y;z)=q_i(y\mid z)/p_0(y)$. $\alpha_i$ is the tempering exponent (generalized Bayes / power-posterior, Bissiri 2016), so agents with more noise/correlation are automatically downweighted.
    - **Design Motivation**: Traditional naive Bayes assumes conditional independence, but outputs from similar LLM agents are clearly correlated (shared prompt, base model); tempering absorbs such correlation into the likelihood strength, which is easier and more robust than modeling the joint exactly. This upgrades "how much to trust each agent" from a heuristic to a learnable parameter.

2. **Value-of-Information driven action selection**:

    - **Function**: Decide which agent to query next, or whether to stop and return a result/escalate to a human.
    - **Mechanism**: Each agent $i$ has a known call cost $c_i>0$; from a Bayesian decision-theoretic perspective, the next agent is chosen to maximize posterior expected utility minus cost: $a_t^\star=\arg\max_a\sum_h u(a,h)r_t(h)$, and an agent is only called if its expected value of information exceeds its cost $c_i$. VOI is strictly defined as the expected difference in utility before and after the call; can be approximated in real time via one-step lookahead or amortized surrogates.
    - **Design Motivation**: Fixed workflows (e.g., "call 3 agents then ensemble") suffice for short-horizon/low-stakes, but cannot adapt when tasks are long or costs are asymmetric (e.g., safety check vs unit test runner). VOI embeds "when to spend more for more calls" directly into orchestration, providing a unified principle for routing/stopping/escalation. In incident diagnosis or multi-agent debate, this expresses "if current max posterior confidence < threshold, query another agent."

3. **Online agent reliability learning + dependency-aware evidence pooling**:

    - **Function**: Track each agent’s performance across tasks/distributions and safely aggregate correlated evidence.
    - **Mechanism**: Define cumulative log-loss $L_i=\sum_{s:i_s=i}-\log q_i(y_s\mid z_s)$, update exponential weights $w_i\propto\exp(-\beta L_i)$ online, normalize to tempering coefficient $\alpha_i=\alpha_\text{max}\tilde w_i$ (Cesa-Bianchi & Lugosi 2006). For correlation from repeated queries to the same agent, either condition the observation model on interaction history or augment the latent state with agent-specific shared error variables; when drift is detected (via rolling calibration diagnostics), automatically increase tempering or trigger abstention/escalation.
    - **Design Motivation**: Orchestration faces two types of corruption—agent capability drift and inter-message correlation. The first is handled by exponential weights/Bayesian routing; the second by composite likelihood + dependence-aware pooling. This ensures belief converges "conservatively"—not overconfident from a few correlated messages. It also defines verifiable engineering interfaces (confidence thresholds, cost scales) so production systems can expose simple knobs to users.

### Loss & Training
The focus is not on training LLMs, but on **meta-learning for the orchestrator**: (a) learn $q_i(y\mid z)$ from historical interaction logs with outcome labels; (b) update $\alpha_i$ online; (c) calibrate with held-out tasks (empirical coverage, proper scoring rules); (d) retemper upon drift detection. The design principle requires observation models to be continually recalibrated from measurable outcomes (pass/fail, human ratings, task completion), fully compatible with RLHF/online learning engineering practices.

## Key Experimental Results

> Note: As a position paper, no large-scale empirical benchmark is provided, but three concrete examples demonstrate design feasibility, and seven actionable properties of "Bayesian agentic systems" are distilled (see Section 2).

### Main Results
Three examples and their corresponding latent variable designs:

| Example (Section) | Orchestration Scenario | Latent Variable $Y$/$H$ | Observation $Z$ | Decision |
|-------------------|-----------------------|------------------------|----------------|----------|
| 4.1 Multi-agent code generation | code generator + retrieval + safety checker + unit-test runner | $Y\in\{0,1\}$: passes all unit tests | candidate code / citations / warnings | when to stop/return, whom to call |
| 4.2 Multi-agent debate | multiple LLM experts debating a scientific/policy issue (e.g., root cause diagnosis) | $H\in\{h_1,\dots,h_k\}$: which hypothesis/root cause | each agent’s argument message | when to stop, escalate to human |
| C Routing (Appendix) | routing tasks among agent pool | cross-task competence parameter | agent historical performance | select the most suitable agent |

### Ablation Study (Thought Experiment)

| Configuration | Meaning | Paper’s Argument |
|---------------|---------|-----------------|
| Belief in parameter space ("Bayesian LLM") | Make LLM internally Bayesian | Falck 2024 shows in-context update is not truly Bayesian; parameter posterior in overparameterized models poorly expresses epistemic uncertainty; huge engineering cost |
| Belief in token probabilities | Use next-token distribution as belief state | Kuhn 2023 / Aichberger 2025: syntactic uncertainty ≠ semantic; sharp token distribution does not mean task-level confidence |
| Prompt-based heuristic only | ReAct / Reflexion, etc. | Sufficient for short-horizon; for long-horizon, large tool ecosystems, cost asymmetry, fixed workflow cannot express routing/stopping |
| Robust control / Bandits (no explicit posterior) | UCB, worst-case | Suitable for pure reward-driven, but cannot naturally express VOI, abstention, cost-aware escalation |
| Task-level latent + VOI + composite likelihood (Ours) | Bayes-consistent control layer | Explicit interface, engineering feasibility, principled dependency handling, compatible with human-AI collaboration |

### Key Findings
- **Uncertainty scale mismatch is critical**: token uncertainty ≠ task uncertainty ≠ parameter uncertainty; agentic decision-making needs task-level latent, and separating it from LLM internals is more feasible than making LLMs Bayesian.
- **Composite likelihood + tempering addresses correlation**: outputs from similar agents are inevitably correlated; naive likelihood multiplication leads to overconfidence; learning $\alpha_i$ via power-posterior automatically downweights "unreliable/correlated" agents.
- **Value-of-information is an explicit "when to spend more" principle**: replacing heuristic workflows with VOI unifies routing/stopping/escalation under a single decision-theoretic objective.
- **Seven properties make engineering feasible**: (1) Control layer is easy to integrate; (2) Compatible with existing typed agent schemas; (3) Exposes simple knobs like confidence threshold; (4) Supports abstention/escalation; (5) Maintains manageable context window; (6) Treats human feedback as probabilistic observation; (7) Supports logging of beliefs and decisions (see Section 2).

## Highlights & Insights
- **"Which layer to make Bayesian" is the key question**: The main contribution is to precisely reframe the vague "agentic AI should be Bayesian"—not parameter space, not token space, but task-level latent + control policy. This gives BDL’s accumulated tools (composite likelihood, generalized Bayes, VOI, Bayesian bandit) a truly suitable place in the agent era.
- **Composite likelihood + tempering is an elegant "engineering compromise"**: Fully modeling LLM interdependence is nearly impossible; using $\alpha_i$ as a single-knob proxy retains probabilistic interpretation while acknowledging real-world noise, more robust than naive multiplication or full independence.
- **Value-of-information gives long-horizon agents a principled "when to stop"**: In practice, agents often over-call tools (high cost) or under-call (insufficient accuracy); VOI quantifies "is another call worth it" explicitly, more elegant than hand-tuned thresholds.
- **A bridge across subcommunities**: Connects PAC-Bayes, generalized Bayes, Bayesian bandit, Bayesian filtering into a single agentic orchestration narrative, reframing BDL for the agent era.
- **Transferable design pattern**: (a) Any system "making joint decisions from multiple unreliable predictors" (medical multi-expert consultation, autonomous driving sensor fusion, quantitative investment multi-strategy weighting) can use this belief + likelihood + VOI template; (b) Treating human feedback as noisy probabilistic observation in the same channel as agent messages is a promising unified RLHF/HCI interface.

## Limitations & Future Work
- **Observation model may be misspecified**: $q_i(y\mid z)$ is learned from historical logs, and calibration may fail under distribution shift; the paper acknowledges the need for continuous rolling calibration monitoring, stronger tempering, and fallback to abstention.
- **Likelihood for high-dimensional agent messages is an open problem**: How to map text-level $Z$ to latent $y$ likelihood? In practice, only embedding-based discriminative models can approximate this, still far from "strict Bayesian."
- **VOI is computationally expensive in combinatorial orchestration**: Multi-step VOI explodes exponentially in tree/graph agent call structures; the paper suggests amortized controllers or one-step approximations, but exact solutions remain open.
- **Depends on measurable outcomes**: Many agentic tasks cannot be binarized (creative writing, policy advice); defining belief state and learning observation models require more refined domain engineering.
- **No large-scale empirical validation**: As a position paper, no benchmark is provided; the authors mainly call for standardized outcome-based, cost-aware, dependence-aware evaluation platforms.
- **Latency constraints in large-scale industrial systems**: VOI computation may add hundreds of milliseconds per routing decision; whether industrial systems can tolerate this remains to be seen.

## Related Work & Insights
- **vs Bayesian Deep Learning mainstream (Mackay 1992, Blundell 2015, Gal & Ghahramani 2016, etc.)**: BDL puts Bayes in parameter space; this paper argues that this path is infeasible for LLMs both practically and theoretically, and proposes moving it to the control layer.
- **vs Falck 2024 / Chlon 2025 / Atwell 2026 "Is LLM Bayesian" studies**: These works show LLM in-context behavior deviates from martingale; this paper uses these results as **evidence** that "LLM internal Bayesianism" is unrealistic, so control-layer Bayes is more practical.
- **vs ReAct / Reflexion / chain-of-thought**: These are prompt heuristic orchestrations; the paper acknowledges their effectiveness for short tasks, but argues that long-horizon, large tool ecosystems require principled Bayesian control.
- **vs Bayesian bandit / robust control**: These can make decisions without explicit belief; this paper argues that when abstention, escalation, and value-of-information matter, explicit belief state is a more natural interface.
- **vs Bengio 2025 "Bayesian Oracle"**: Bengio et al. also propose using Bayesian oracles to prevent agent harm, aligned with this paper’s vision; this paper further provides control-layer design patterns.
- **Insights**: (1) Teams aiming to engineer and scale LLM agents can start by "adding belief logging + VOI-based routing in the orchestrator"; (2) BDL researchers can move their tools to this new niche, which is more realistic than "parameter-level Bayesian LLM"; (3) The evaluation community should incorporate outcome calibration and cost-aware metrics into agent benchmark standards.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new algorithm, but a **precise reframing** of "agentic AI should be Bayesian"—breaking down a vague claim into four concrete design principles: control layer + task-level latent + VOI + composite likelihood.
- Experimental Thoroughness: ⭐⭐⭐ As a position paper, no benchmark is provided; three concrete examples + design template sufficiently demonstrate feasibility, but lack end-to-end empirical validation—an inherent limitation of position papers.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentative structure: first positions "which layer to make Bayesian," then explains why LLM internals are unsuitable, then provides a design template, then lists seven actionable properties, then calls for four directions; solid citations (spanning BDL, decision theory, agents, bandits).
- Value: ⭐⭐⭐⭐ An important boundary-setting for both BDL and agent communities—BDL has not found a killer application for LLMs in years, and this paper moves its application to the control layer, a truly suitable place, likely to inspire a wave of work in this direction in the coming years.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Assistive Agents Need Accessibility Alignment](position_assistive_agents_need_accessibility_alignment.md)
- [\[NeurIPS 2025\] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading](../../NeurIPS2025/llm_agent/orchestration_framework_for_financial_agents_from_algorithmic_trading_to_agentic.md)
- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](../../ACL2026/llm_agent/how_adversarial_environments_mislead_agentic_ai.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](../../ACL2026/llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)

</div>

<!-- RELATED:END -->
