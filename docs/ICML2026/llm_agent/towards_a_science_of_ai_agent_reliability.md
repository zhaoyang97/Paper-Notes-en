---
title: >-
  [Paper Note] Towards a Science of AI Agent Reliability
description: >-
  [ICML2026][LLM Agent][AI agent] Borrowing mature practices from safety-critical engineering fields such as aviation, nuclear power, and automotive…
tags:
  - "ICML2026"
  - "LLM Agent"
  - "AI agent"
  - "reliability evaluation"
  - "consistency"
  - "robustness"
  - "calibration"
  - "safety-critical engineering"
date: 2026-05-08
content_hash: 5f519b6c23d2fe94
---

# Towards a Science of AI Agent Reliability

**Conference**: ICML2026  
**arXiv**: [2602.16666](https://arxiv.org/abs/2602.16666)  
**Code**: https://hal.cs.princeton.edu/reliability/ (Interactive dashboard)  
**Area**: LLM Agent / Evaluation  
**Keywords**: AI agent, reliability evaluation, consistency, robustness, calibration, safety-critical engineering  

## TL;DR
Borrowing mature practices from safety-critical engineering fields such as aviation, nuclear power, and automotive, this paper decomposes AI agent "reliability" into four dimensions (consistency, robustness, predictability, and safety) comprising 12 accuracy-independent metrics. By systematically evaluating 15 frontier models on GAIA and $\tau$-bench, the study concludes that while accuracy has surged over the past 24 months, reliability has remained largely stagnant.

## Background & Motivation
**Background**: Current agent evaluations are almost entirely centered on mean task success rate (mean accuracy). From GAIA and $\tau$-bench to WebArena, the paradigm remains the same: one prompt, one environment configuration, and one execution to calculate the average.

**Limitations of Prior Work**: Mean accuracy obscures critical signals that determine whether an agent is production-ready. Can multiple runs of the same task yield the same result? Can it handle instructions with equivalent phrasing? What happens if a tool occasionally times out? Is the agent truly more likely to succeed when its confidence is high? How large is the loss when an error occurs? Real-world failures—such as Replit AI deleting production databases, OpenAI Operator placing unauthorized orders, or the NYC government chatbot giving illegal business advice—demonstrate the gap between benchmark performance and deployment reliability. An Anthropic survey of 80,000 people also identified "unreliability" as the top concern regarding AI.

**Key Challenge**: Reliability is inherently multi-dimensional, yet the ML community often studies it as isolated phenomena (e.g., prompt sensitivity, calibration, selective prediction) without a unified framework decoupled from capability. Capability and reliability should progress along independent axes; conflating them within a single accuracy metric obscures both.

**Goal**: (i) Translate safety-critical reliability concepts from other industries into computable metrics for agents; (ii) systematically evaluate the current state of major benchmarks; and (iii) provide specific recommendations for agent evaluation, development, and governance.

**Key Insight**: The authors categorize recurring reliability dimensions from industry standards like FAA, NRC, and ISO 26262 into four types: consistency, robustness, predictability, and safety. While these correspond to fragmented research in ML (e.g., $pass \wedge k$, prompt rephrasing, ECE, refusal evaluations), they have never been integrated into a single framework.

**Core Idea**: Replace single accuracy scores with a "four-dimensional decomposition + 12 metrics" framework from safety-critical engineering. This is implemented across 15 models using a unified protocol involving $K=5$ repetitions, prompt perturbations, fault injection, environmental perturbations, and confidence extraction.

## Method

### Overall Architecture
The paper decomposes agent reliability measurement into a five-layer pipeline:

1. **Dimension Definition**: Extract four recurring dimensions from aviation, nuclear, and automotive standards, mapping them to existing ML literature.
2. **Metric Design**: Define 2–4 metrics for each dimension within the $[0, 1]$ range, explicitly decoupled from accuracy, totaling 12 metrics.
3. **Aggregation Scheme**: Use equal weighting within dimensions; the overall $\mathcal R$ intentionally excludes safety (as safety concerns tail events and should not be obscured by averaging).
4. **Evaluation Protocol**: Use fixed $K=5$, temperature $T=0$, prompt perturbations $J=5$, fault injection $p_\text{fault}=0.2$, medium-intensity environmental perturbations, post-hoc self-evaluated confidence extraction, and LLM-as-a-judge for safety violation labeling.
5. **Large-scale Empirical Study**: Evaluate 15 models from OpenAI, Google, and Anthropic spanning a 24-month release window across 165 GAIA tasks and 26 cleaned $\tau$-bench tasks.

### Key Designs

1. **Four-dimensional Decomposition (consistency / robustness / predictability / safety) and 12-Metric Matrix**:
    - **Function**: Translates the vague question "Is the agent reliable?" into a set of computable, comparable scalars independent of accuracy.
    - **Mechanism**: Each dimension is anchored to safety-critical engineering: consistency corresponds to "deterministic execution" in FAA flight-critical software; robustness to "graceful degradation" under environmental disturbance; predictability to NRC failure mode modeling; and safety to SIL 4 dangerous failure rates ($<10^{-5}$). 
    - **Metrics**: (a) Consistency: Outcome consistency $C_\text{out}=\frac{1}{T}\sum_t(2\hat p_t-1)^2$ (normalized by max Bernoulli variance $0.25$); trajectory consistency uses JSD for distributions and Levenshtein distance for sequences; resource consistency uses $C_\text{res}=\exp(-\overline{\text{CV}_r})$. (b) Robustness: Standardized as the clipped ratio $\min(\text{Acc}_\text{perturb}/\text{Acc}_0, 1)$. (c) Predictability: Measures calibration (ECE), discrimination (AUROC), and joint performance (Brier). (d) Safety: Split into compliance (violation rate) and harm (conditional expected severity), synthesized via the risk formula $1-(1-S_\text{comp})(1-S_\text{harm})$.
    - **Design Motivation**: Each metric adheres to the constraint of being "orthogonal to accuracy." For instance, $C_\text{out}$ scores perfectly at both $\hat p_t=0$ and $\hat p_t=1$, ensuring an agent that consistently fails is not penalized on "stability," thereby isolating reliability from capability.

2. **Safety-Excluded Aggregation Scheme**:
    - **Function**: Maintains a comparable overall reliability score $\mathcal R=\frac{1}{3}(\mathcal R_\text{Con}+\mathcal R_\text{Pred}+\mathcal R_\text{Rob})$ without "averaging out" safety risks.
    - **Mechanism**: Following the Kaplan & Garrick risk formula, safety is explicitly defined as "probability of violation $\times$ conditional severity." Safety is reported as a separate hard constraint; any degradation triggers an independent alert. Within consistency, trajectory components are weighted to prevent them from dominating the dimension due to the number of sub-metrics.
    - **Design Motivation**: Responds to tail-risk requirements (e.g., SIL 4 or FAA "one catastrophe per 100 million flight hours"), emphasizing that reliability assessments must distinguish between nominal metrics and tail metrics.

3. **Unified Evaluation Protocol ($K=5$ Repetitions, Multi-perturbation, Self-rated Confidence)**:
    - **Function**: Transforms benchmarks like GAIA and $\tau$-bench into a "reliability harness" capable of measuring all 12 metrics without changing the underlying tasks.
    - **Mechanism**: Each task is run $K=5$ times at $T=0$ (attributing variance to non-sampling sources like floating-point or kernel scheduling). GPT-4o generates $J=5$ equivalent prompt versions. Faults (e.g., timeouts) are injected into tool calls with $p_\text{fault}=0.2$. Environmental perturbations include changing JSON field names/order/date formats. After completion, agents are prompted to "self-rate" their confidence. Only the cleaned 26-task subset of $\tau$-bench (Cuadron et al.) is used to prevent poor ground truth from contaminating calibration metrics.
    - **Design Motivation**: Provides the first apples-to-apples comparison for the agent industry across 15 models while making perturbation parameters adjustable for reproduction.

### Loss & Training
This paper presents an evaluation study and does not train models. Notable "system-side" configurations include: (i) GAIA using a ReAct scaffold with browsing/code/file tools; (ii) $\tau$-bench using a tool-calling scaffold; all temperatures set to 0.

## Key Experimental Results

### Main Results

| Dimension | Strongest Model (Recent 24m) vs. 24m Prior | Trend | Remarks |
| :--- | :--- | :--- | :--- |
| Accuracy ($\tau$-bench clean) | Significant Increase | Sustained Improvement | Primary driver of progress |
| $\mathcal R$ (overall reliability) | Slight Increase | Near Stagnation | Weak correlation with release date |
| Outcome consistency $C_\text{out}$ | Stagnant | No systemic improvement | Frontier models cluster similarly |
| Prompt robustness $R_\text{prompt}$ | Slight Increase | Key differentiator | High variance between models |
| Calibration $P_\text{cal}$ | Significant Improvement | Driven by Claude models | Suggests explicit optimization |
| Discrimination $P_\text{AUROC}$ | Inconsistent | Decreased on GAIA | Calibration and discrimination require separate assessment |

### Ablation Study

| Configuration | Performance | Description |
| :--- | :--- | :--- |
| GAIA Level 1 → 3 | Monotonic consistency change | Consistency decreases/increases monotonically as difficulty $↑$ |
| Reasoning vs. non-reasoning | Slightly higher reliability | Improvement magnitude is smaller than accuracy gain |
| Small vs. Large Models | Higher consistency in small models | Large models' multi-path solutions increase run-to-run variance |
| $\tau$-bench Full vs. Clean | Significant Predictability improvement | Incorrect ground truth distorts calibration assessments |
| Safety Violation Types | Financial accuracy most common | Numerical reasoning is most fragile in transactional scenarios |

### Key Findings
- While accuracy has surged over 24 months, overall reliability has plateaued, suggesting that unreliability is an industry-wide plateau rather than a vendor-specific limitation.
- **"What but not when" pattern**: Agents perform adequately in distribution consistency (action types) but poorly in sequence consistency (execution order), indicating stable action selection but unstable planning.
- Prompt robustness remains the primary differentiator between models; sensitivity to equivalent paraphrases is high and often more severe than environmental perturbations like tool timeouts.
- Safety violations are most frequent in "financial numerical errors." While large-scale models show lower high-severity violation rates, tail risks cannot be hidden by averages and must be reported independently.

## Highlights & Insights
- The aggregation strategy of "four nominal dimensions + one independent tail dimension (safety)" is highly reusable. It prevents "averaging away" catastrophic tail events, a common error in ML evaluation.
- The normalization of outcome consistency $C_\text{out}=(2\hat p_t-1)^2$ is a clever design that rewards total success and total failure equally, effectively decoupling "stability" from "capability."
- Quantifying the impact of benchmark hygiene (e.g., cleaner $\tau$-bench data significantly improving calibration) serves as a reminder that noise in the evaluator can corrupt the measurement of the evaluated attributes.

## Limitations & Future Work
- The study covers only GAIA and $\tau$-bench; the generalizability to long-context coding (SWE-bench) or multimodal browsing (VisualWebArena) remains to be verified.
- The coupling between scaffolds (e.g., ReAct) and reliability is not fully explored; different prompting or tool-use strategies might radically alter a model's reliability profile.
- Safety violations rely on LLM-as-a-judge, introducing a recursive problem where an unreliable system evaluates another's reliability.
- Fixing temperature to $T=0$ may overestimate consistency, as production environments often use $T>0$, which introduces more variance. The reported $\mathcal R$ values are likely optimistic lower bounds.
- The 12 metrics and their aggregation weights are specific design choices; other decompositions may be equally valid depending on the application.

## Related Work & Insights
- **vs. HELM (Liang et al., 2022)**: While HELM emphasizes multi-dimensional evaluation, it focuses on capability subsets (bias, fairness). This work explicitly defines "reliability $\neq$ capability."
- **vs. pass$\wedge k$ / Single Consistency Metrics**: These focus only on one dimension; this paper places consistency within a broader framework including robustness, predictability, and safety.
- **vs. ML Calibration Literature**: Most calibration work focuses on classification tasks. This paper systematically applies ECE/AUROC/Brier to multi-step agents and provides evidence that calibration improvements do not necessarily imply discrimination improvements.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Cross-disciplinary adaptation + integrated framework, though individual metrics often build on existing work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 15 models $\times$ 2 benchmarks $\times$ multi-perturbation protocol is substantial, though benchmark variety is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear hierarchy of dimensions, metrics, and findings; excellent documentation and interactive dashboard.
- **Value**: ⭐⭐⭐⭐⭐ Provides comparable thresholds for agent deployment and immediate utility for the governance and compliance communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](../../ICLR2026/llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)
- [\[NeurIPS 2025\] It's LIT! Reliability-Optimized LLMs with Inspectable Tools](../../NeurIPS2025/llm_agent/its_lit_reliability-optimized_llms_with_inspectable_tools.md)
- [\[ICML 2026\] Position: Agentic AI Orchestration Should Be Bayes-Consistent](position_agentic_ai_orchestration_should_be_bayes-consistent.md)

</div>

<!-- RELATED:END -->
