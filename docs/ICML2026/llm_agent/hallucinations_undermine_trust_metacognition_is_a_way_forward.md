---
title: >-
  [Paper Note] Hallucinations Undermine Trust; Metacognition is a Way Forward
description: >-
  [ICML 2026 (Position Paper)][LLM Agent][Hallucination] This position paper argues that "completely eliminating LLM hallucinations" is fundamentally impossible due to a "discrimination gap" (discrimination gap → utility t…
tags:
  - "ICML 2026 (Position Paper)"
  - "LLM Agent"
  - "Hallucination"
  - "Calibration vs Discrimination"
  - "faithful uncertainty"
  - "metacognition"
  - "agentic control layer"
date: 2026-05-08
content_hash: ff44cbc6a54d8eed
---

# Hallucinations Undermine Trust; Metacognition is a Way Forward

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.01428](https://arxiv.org/abs/2605.01428)  
**Code**: None  
**Area**: LLM Trust / Agent Metacognition  
**Keywords**: Hallucination, Calibration vs Discrimination, faithful uncertainty, metacognition, agentic control layer

## TL;DR
This position paper argues that "completely eliminating LLM hallucinations" is fundamentally impossible due to a "discrimination gap" (discrimination gap → utility tax). The authors propose shifting the objective from "eradicating hallucinations" to **faithful uncertainty**, positioning this metacognition as an indispensable control layer for agentic LLMs when invoking tools.

## Background & Motivation
**Background**: While frontier models continue to improve on factual reliability benchmarks, hallucinations remain the primary obstacle to industrial deployment. Research generally follows two paths: (1) training-stage interventions—data filtering, alignment penalties, and reward model calibration; (2) inference-stage interventions—specialized decoding (DOLA), internal signal probes, and self-verification. Meanwhile, the field of uncertainty quantification has demonstrated that modern LLMs can output **well-calibrated** confidence signals.

**Limitations of Prior Work**: All efforts implicitly aim to "reduce the hallucination rate to 0." However, even with perfectly calibrated confidence, reducing the hallucination rate from 25% to 5% requires the model to abandon **52% of its correct answers** (utility tax). AUROC on real-world tasks consistently ranges between $0.70$–$0.85$; it must exceed $0.95$ to reduce the utility tax below 5%—a level currently unreachable by any existing method.

**Key Challenge**: The authors formalize the contradiction as the gap between calibration and discrimination. **Calibration** only requires that "of the samples with 0.6 confidence, 60% are correct"—an alignment in the aggregate sense. **Discrimination** requires "identifying exactly which 60% are correct." A constant predictor that always outputs 0.6 is perfectly calibrated but has zero discrimination. Existing theories (Halting Problem-style arguments, the necessity of hallucinations in calibrated models, and consistency–breadth trade-offs) suggest that the discrimination ceiling for LLMs is fundamentally limited regardless of training.

**Goal**: (1) Formalize the unreachability of "eliminating hallucinations"; (2) Propose an alternative objective to bypass this impossibility; (3) Scale this objective to agentic systems as a control layer for tool invocation.

**Key Insight**: Redefine "hallucination" not as "any error," but as a "**confident** error." If an error is accompanied by appropriate hedging ("I am not entirely sure, but it might be 1961"), it is no longer a hallucination but a hypothesis. This redefinition collapses the binary opposition between "eliminating errors" and "maintaining utility," revealing a third path: honestly expressing uncertainty.

**Core Idea**: Replace "zero hallucinations" with **faithful uncertainty** (aligning linguistic uncertainty with intrinsic uncertainty) and establish metacognition as a core capability for LLMs and agents.

## Method

### Overall Architecture
As a position paper, this work introduces no new algorithms but proposes a conceptual framework: **redefining the problem + proving feasibility + providing a research roadmap**. The argument follows three steps: (1) Proving that "eliminating hallucinations" is theoretically impossible; (2) Proving that faithful uncertainty is **theoretically possible** and can unlock reliable utility; (3) In agentic scenarios, metacognition serves as the control layer for tool invocation. Finally, it outlines 6 major research challenges and 3 evaluation norms for hallucination mitigation.

### Key Designs

1.  **Clear Separation of Calibration vs Discrimination + Visualization of the Utility Tax**:
    - **Function**: To visually demonstrate the counter-intuitive fact that "good calibration can still result in a massive utility tax."
    - **Mechanism**: Simulated data replicates the reliability diagrams of Nakkiran 2025, where confidence for correct answers ($y=1$) comes from $\text{Beta}(1.8,1.0)$ and errors ($y=0$) from $\text{Beta}(1.0,1.3)$, followed by Isotonic regression to force calibration at smECE $\approx 0.014$. With AUROC = 0.71 (consistent with literature), the utility-error trade-off is plotted by sweeping the abstention threshold from 0 to 1. 
    - **Design Motivation**: Previous discussions conflated calibration and discrimination, leading the claim "our model is well-calibrated" to be misinterpreted as "our model does not hallucinate." This visualization isolates these concepts, making the cost tangible by showing exactly how many correct answers must be sacrificed, which is more impactful than ECE metrics.

2.  **Faithful Uncertainty: Moving the Goal from "Aligning with the World" to "Aligning with the Internal State"**:
    - **Function**: Provides a **theoretically reachable** weakened objective—linguistic uncertainty must faithfully reflect internal uncertainty, without requiring the internal uncertainty to match the external ground truth.
    - **Mechanism**: Based on the definition in Yona 2024. Intrinsic confidence $\text{conf}_M(A)=1-\frac{1}{k}\sum_i\mathbf 1[A\text{ contradicts }A_i]$ (measured by semantic consistency across $k$ resamples). Linguistic decisiveness $\text{dec}(A;R,Q)=\Pr[A\text{ True}\mid R,Q]$ (estimated via LLM-as-judge to gauge the credibility readers assign $A$ based on hedging intensity). Faithfulness $=1-\frac{1}{|A(R)|}\sum_{A}|\text{dec}(A;R,Q)-\text{conf}_M(A)|$. The authors argue this is **closed-loop observable**—since $\text{conf}_M$ is a function of model weights, aligning linguistic output with confidence is an internal consistency problem, independent of external ground truth.
    - **Design Motivation**: By shifting the target from "Internal → Real World" to "Internal → Internal," the discrimination tax disappears. The model does not need to abstain to avoid errors; it simply provides uncertain answers in a hedged form, allowing users to receive useful guesses with honest labels.

3.  **Agentic Metacognition: Uncertainty as a Control Layer for Tool Invocation**:
    - **Function**: In the era of agents, self-perceived uncertainty is a **prerequisite** for deciding when to call tools, when to trust retrieval results, and how to resolve conflicts with priors.
    - **Mechanism**: Current agent harnesses act as crude external schedulers based on query-type heuristics. Introducing a "confidence signal" as a control layer allows the harness to: provide a direct answer if confidence is high (saving tool calls); retrieve if confidence is low; or output a hedged answer if retrieval conflicts with priors.
    - **Design Motivation**: Contrary to the belief that "tool invocation bypasses hallucinations," the authors argue tools only solve the **storage problem** (encoding facts into weights) but introduce a **control problem** (knowing when to retrieve). Faithful uncertainty fills this control layer.

### Loss & Training
Not applicable (position paper). However, §6 identifies 6 research challenges: the bootstrapping paradox (dynamic labels vs static SFT), protection of confidence signals during post-training, confidence attribution (distinguishing between aleatoric, epistemic, and normative uncertainty), rigorous causal evaluation (ensuring the model isn't just learning a "hedging style"), process-based agent evaluation, and switching hallucination mitigation assessments to utility-error trade-off curves.

## Key Experimental Results

### Main Results

| Data/Scenario | Phenomenon | Meaning |
|----------|------|------|
| Beta Simulation, AUROC=0.71, Base Error 25% | Reducing error to 5% requires discarding 52% of correct answers | Significant discrimination tax |
| Beta Simulation, AUROC=0.85 | Tax of $\sim$28% | Heavy cost even for state-of-the-art |
| Beta Simulation, AUROC=0.95 | Tax $<5\%$ | No method achieves this on knowledge-intensive tasks |
| SimpleQA Verified (Multiple Frontier Models) | All models follow the diagonal or shift left | The top-right "ideal" region is empty |
| AUROC Literature Survey (farquhar2024, savage2025, kang2025) | AUROC 0.70–0.85 on real knowledge-intensive QA | Confirms the discrimination gap |

### Ablation Study
(N/A; replaced by proxies like cMFG for faithful uncertainty evaluation.)

| Evaluation Dimension | Current SOTA | Target |
|---------|----------|------|
| cMFG (Conditional Mean Faithful Generation) | 0.5–0.7 | 1.0 |
| Reasoning vs. Standard Model Confidence | Reasoning models are better at expressing but hallucinate more | Decoupling metacognitive signals from factual signals |
| Internal Truth Probe AUROC (Mech Interp) | Collapses on OOD data | Does not assume a universal truth direction |

### Key Findings
- **Calibration $\neq$ Absence of Hallucinations**: Even with perfect calibration, a lack of discrimination inevitably results in a utility tax; this is the paper's most counter-intuitive and compelling point.
- **SimpleQA Scatter Plot (Fig 3)**: The "ideal" top-right corner is unoccupied—all frontier models either stick to the diagonal or shift left, paying a high abstention tax, indicating they are trapped on the trade-off curve.
- **Extended Reasoning Increases Hallucination**: Reasoning models like o1 show lower abstention rates but higher hallucination rates, attributed to rewards optimizing for utility rather than honest uncertainty.
- **Post-training Erodes Pre-trained Uncertainty Signals**: Research suggests RLHF makes models mode-seeking and overconfident—a primary issue for metacognitive research to address.
- **Agentic Evaluation Must Be Process-Based**: Current evaluations reward "hitting the right answer," masking metacognitive failures (e.g., inefficient searching of known facts or trusting sycophantic sources).

## Highlights & Insights
- **Redefinition of "Faithful $\neq$ Correct"**: Shifting from "aligning with external truth" to "aligning with internal states" bypasses the theoretical barrier of truth undecidability. This serves as a template for other trustworthy ML problems, such as weakening "correct explanations" to "faithful explanations."
- **Utility-Error Curve as New Evaluation**: Replacing single-point metrics with a curve forces the claim "our method reduces hallucinations" to be redefined as "we provide more utility at a fixed error rate."
- **Splitting Storage Problem vs Control Problem**: By dividing agent reliability into two layers, the necessity of metacognition becomes clear, countering the naive optimism that "retrieval solves everything."
- **Honesty about Reasoning Models**: The authors do not hide that reasoning models hallucinate more, using it as proof that utility-only training objectives disincentivize honesty.

## Limitations & Future Work
- The paper is purely positional, lacking new algorithms or novel empirical data; simulation figures are illustrative.
- The feasibility of "faithful uncertainty" assumes that models possess readable internal confidence signals. If latent states lack a separable "truth direction," this path may fail; acknowledged in §7.3 without a definitive solution.
- The 6 research challenges serve more as an agenda than a set of actionable solutions; implementing "dynamic SFT labels without damaging base model confidence" remains an open problem.
- Suggestions for agent evaluation remain abstract without a standardized metric.
- Feasibility of "hedge per assertion" in multimodal or long-form generation is not discussed.

## Related Work & Insights
- **vs. Kadavath et al. on Calibration**: While they proved LLMs can be well-calibrated, this paper argues that calibration is insufficient and discrimination is the critical metric.
- **vs. Kalai 2024 (Calibrated models must hallucinate)**: Inherits Kalai's impossibility theorem but reframes it: if zero hallucination is unreachable, change the target.
- **vs. Yona 2024 (Faithful Uncertainty)**: While Yona defined the cMFG metric, this paper elevates it to a policy proposal for training pipelines.
- **vs. Tool-Augmented LLMs (ReAct, Toolformer)**: Counter-argues that tool use cannot replace metacognition, citing failures in current search agents.
- **Insight**: Any "Trustworthy XX" research can ask: Is the goal to align with the world or the internal state? The former is often impossible; the latter is usually achievable. This logic applies to explainability, safety, and AI honesty.

## Rating
- Novelty: ⭐⭐⭐⭐ Shifting the goal from world-alignment to internal-alignment is a distinct conceptual leap.
- Experimental Thoroughness: ⭐⭐ Position paper with illustrative simulations and literature review only.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear argumentation with honest engagement with counter-arguments.
- Value: ⭐⭐⭐⭐ Significant value for re-aligning the direction of trustworthy LLM and agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Real-Time Trust Verification for Safe Agentic Actions Using TrustBench](../../AAAI2026/llm_agent/real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)
- [\[ICLR 2026\] Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](../../ICLR2026/llm_agent/inherited_goal_drift_contextual_pressure_can_undermine_agentic_goals.md)
- [\[ICML 2026\] Position: Agentic AI Orchestration Should Be Bayes-Consistent](position_agentic_ai_orchestration_should_be_bayes-consistent.md)
- [\[ICML 2026\] LLM Agents Are the Antidote to Walled Gardens](llm_agents_are_the_antidote_to_walled_gardens.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)

</div>

<!-- RELATED:END -->
