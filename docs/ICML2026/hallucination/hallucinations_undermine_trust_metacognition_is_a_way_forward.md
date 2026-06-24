---
title: >-
  [Paper Note] Hallucinations Undermine Trust; Metacognition is a Way Forward
description: >-
  [ICML 2026 (Position Paper)][Hallucination Detection][Hallucination] This position paper argues that "totally eliminating LLM hallucinations" is theoretically impossible without incurring a "utility tax" (discrimination gap); the authors advocate shifting the goal from "eliminating hallucinations" to **faithful uncertainty** and treating this metacognition as an indispensable control layer for agentic LLMs when calling tools.
tags:
  - "ICML 2026 (Position Paper)"
  - "Hallucination Detection"
  - "Hallucination"
  - "Calibration vs. Discrimination"
  - "Faithful Uncertainty"
  - "Metacognition"
  - "Agentic Control Layer"
date: 2026-05-08
content_hash: f9e2691061205da3
---

# Hallucinations Undermine Trust; Metacognition is a Way Forward

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.01428](https://arxiv.org/abs/2605.01428)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Hallucination, Calibration vs. Discrimination, Faithful Uncertainty, Metacognition, Agentic Control Layer

## TL;DR
This position paper argues that "totally eliminating LLM hallucinations" is theoretically impossible without incurring a "utility tax" (discrimination gap); the authors advocate shifting the goal from "eliminating hallucinations" to **faithful uncertainty** and treating this metacognition as an indispensable control layer for agentic LLMs when calling tools.

## Background & Motivation
**Background**: Despite frontier models continuously improving scores on factual reliability, hallucinations remain the primary barrier to industrial deployment. Research follows two main paths: (1) training-stage interventions—data filtering, alignment penalties, and reward model calibration; (2) inference-stage interventions—specialized decoding (DOLA), internal signal probing, and self-verification. Meanwhile, the field of uncertainty quantification has demonstrated that modern LLMs can output **well-calibrated** confidence signals.

**Limitations of Prior Work**: All efforts implicitly aim to "reduce the hallucination rate to 0." However, even if confidence is perfectly calibrated, reducing the hallucination rate from 25% to 5% requires the model to abandon **52% of its correct answers** (utility tax). AUROC on real tasks remains stable between $0.70$–$0.85$; it would need to exceed $0.95$ to bring the utility tax below 5%—a feat currently unattainable by any method.

**Key Challenge**: The authors formalize the contradiction as the gap between calibration and discrimination. **Calibration** only requires that "in samples with 0.6 confidence, 60% are correct"—an alignment in an average sense. **Discrimination** requires "being able to pick out exactly which 60% are correct." A constant predictor that always outputs 0.6 is perfectly calibrated but has zero discrimination. Existing theories (Halting Problem arguments, the necessity of hallucinations in calibrated models, and the consistency–breadth trade-off) imply that the discrimination ceiling for LLMs is finite regardless of training.

**Goal**: (1) Formalize the unreachability of "eliminating hallucinations"; (2) Propose an alternative goal that bypasses this unreachability; (3) Generalize this goal to agentic systems as a control layer for tool invocation.

**Key Insight**: Redefine "hallucination" not as "any error," but as "an error made **confidently**." If an error is accompanied by appropriate hedging ("I am not sure, it might be 1961"), it is no longer a hallucination but a hypothesis. This redefinition collapses the binary opposition between "eliminating errors" and "maintaining utility," offering a third path: honestly expressing uncertainty.

**Core Idea**: Replace "zero hallucinations" with **faithful uncertainty** (aligning linguistic uncertainty with intrinsic uncertainty) and establish metacognition as a core capability for LLMs and agents.

## Method

### Overall Architecture
As a position paper, it addresses the question: "Since hallucinations cannot be eliminated, what should the goal of trustworthy LLMs be reset to?" It does not propose a new algorithm but builds a three-step argumentative chain: first, a quantitative argument showing the prohibitively high cost of forcing the hallucination rate to 0; second, the proposal of faithful uncertainty as a theoretically reachable alternative goal; and third, its application to agentic scenarios, illustrating metacognition as an unavoidable control layer for tool calling. It also provides six major research challenges and a new evaluation framework.

### Key Designs

**1. Separation of Calibration vs. Discrimination: Mapping the "Discrimination Tax"**

Addressing the community's long-standing confusion between calibration and discrimination, which leads to misinterpreting "our model is well-calibrated" as "our model does not hallucinate." The authors decouple the two through simulations: confidence for correct answers ($y=1$) is drawn from $\text{Beta}(1.8,1.0)$ and for incorrect answers ($y=0$) from $\text{Beta}(1.0,1.3)$, followed by Isotonic regression to force calibration at smECE $\approx 0.014$, replicating the reliability diagram of Nakkiran 2025. In this state, the model is nearly perfectly calibrated, but the AUROC is only 0.71 (falling within the typical $0.70$–$0.85$ range for real tasks).

Qualified calibration does not imply the ability to "select" which answers are correct, the latter of which determines the cost of a rejection strategy. By sweeping the rejection threshold from 0 to 1, the utility-error trade-off curve illustrates the cost: reducing the 25% error rate to 5% requires discarding **52% of correct answers**. Even if AUROC is improved to a state-of-the-art 0.85, the tax remains $\sim$28%; only an AUROC $\geq 0.95$ can reduce the tax below 5%, which no method achieves on knowledge-intensive tasks. This curve quantifies why model providers are reluctant to pay this tax.

**2. Faithful Uncertainty: Shifting the Goal from "World-Alignment" to "Self-Alignment"**

Since "aligning with external ground truth" hits the theoretical barriers of the discrimination tax and undecidability of truth, the authors adopt the definition from Yona 2024 to propose a weakened but reachable goal: the model's linguistic uncertainty must only faithfully reflect its own internal uncertainty, without requiring this internal signal to match the ground truth of the world. Internal confidence is defined as $\text{conf}_M(A)=1-\frac{1}{k}\sum_i\mathbf 1[A\text{ contradicts }A_i]$, where $k$ resamples are taken for answer $A$ and semantic consistency measures the model's certainty. Linguistic decisiveness $\text{dec}(A;R,Q)=\Pr[A\text{ True}\mid R,Q]$ is estimated using an LLM-as-judge to gauge the credibility a reader would assign to $A$ based on the strength of the hedging. The degree of alignment is faithfulness $=1-\frac{1}{|A(R)|}\sum_{A}|\text{dec}(A;R,Q)-\text{conf}_M(A)|$.

This goal is "theoretically feasible" because it is closed-loop observable: $\text{conf}_M$ is strictly a function of model weights. "Aligning linguistic output with conf" is a pure internal consistency problem that does not rely on external ground truth, thus bypassing the Halting-problem-like barriers of xu2024 regarding external truth determination. By shifting from "Internal → World" to "Internal → Internal," the discrimination tax disappears—the model does not need to reject answers to avoid errors but simply presents uncertain answers in a hedged format; the user still receives useful guesses, but with an honest label.

**3. Agentic Metacognition: Uncertainty as a Control Layer for Tool Use**

Countering the optimistic view that "tool calling can bypass hallucinations," the authors argue that tools only solve the **storage problem** (not needing to store all facts in weights) while introducing the **control problem**—deciding when to retrieve and determining the credibility of retrieved content. This control layer requires faithful uncertainty. Current agent harnesses are coarse external schedulers relying almost entirely on query-type heuristics for search decisions. Integrating "model-reported confidence" as a control signal allows the harness to shunt tasks: high confidence leads to direct answers to save tool calls, while low confidence triggers retrieval. When retrieval results conflict with model priors, it outputs hedged answers rather than blindly trusting the context. The paper cites evidence like qian2025smart to show that current search agents lack this self-awareness, leading to over- or under-utilization of tools.

### Research Roadmap
As a position paper, there are no training objectives, but §6 identifies six challenges for future work: the bootstrapping paradox (confidence labels are dynamic and hard to fit with static SFT), how to preserve existing pre-train confidence signals during post-training, confidence attribution (distinguishing aleatoric, epistemic, and normative uncertainty), rigorous causal evaluation (preventing models from learning hedging tones without true self-awareness), agent evaluation focused on processes rather than end-to-end accuracy, and shifting hallucination mitigation evaluation to utility-error trade-off curves.

## Key Experimental Results

### Main Results

| Data/Scenario | Phenomenon | Implication |
|----------|------|------|
| Beta Simulation, AUROC=0.71, Base Error 25% | Reducing error to 5% requires discarding 52% correct answers | Significant discrimination tax |
| Beta Simulation, AUROC=0.85 | Tax $\sim$28% | Heavy cost even at SOTA levels |
| Beta Simulation, AUROC=0.95 | Tax $<5\%$ | Unattainable on knowledge-intensive tasks |
| SimpleQA Verified (Multiple frontier models) | All distributed along diagonal or shifted left | Top-right "ideal" region is empty |
| AUROC Literature Review (farquhar2024, savage2025, kang2025) | AUROC 0.70–0.85 on real knowledge-intensive QA | Confirms discrimination gap |

### Ablation Study
(None; replaced by proxy evaluations for faithful uncertainty like cMFG.)

| Evaluation Dimension | Current SOTA | Target |
|---------|----------|------|
| cMFG (Conditional Mean Faithful Generation) | 0.5–0.7 | 1.0 |
| Reasoning vs. Standard model confidence | Reasoning models are better but hallucinate more | Metacognitive signals decoupled from factual signals |
| Internal truth probe AUROC (mech interp) | Collapses on OOD | Does not assume a universal truth direction |

### Key Findings
- **Calibration $\neq$ Absence of Hallucinations**: Even perfect calibration incurs a utility tax if discrimination is insufficient; this is the paper's most counter-intuitive and compelling argument.
- **SimpleQA Scatter Plot (Fig. 3)**: The "ideal" top-right corner is unoccupied—all frontier models either stick to the diagonal or shift left, paying a high rejection tax, suggesting current methods are trapped on the trade-off curve.
- **Extended Reasoning exacerbates Hallucinations**: Reasoning models like o1 show lower rejection rates but higher hallucination rates, attributed by the authors to rewards optimizing utility rather than honest uncertainty.
- **Post-training erodes Pre-trained uncertainty signals**: Works like he2025rewarding and song2025outcome show RLHF makes models mode-seeking and overconfident—a primary problem for metacognitive research.
- **Agentic evaluation must be process-based**: Current evaluations reward agents that "hit" the right answer, masking metacognitive failures (e.g., searching for known facts is inefficient; trusting sources that conflict with priors is sycophancy).

## Highlights & Insights
- **Goal Redefinition (Faithfulness vs. Correctness)**: Shifting the goal from "external truth alignment" to "internal state alignment" cleverly bypasses the theoretical obstacle of "undecidability of truth." This serves as a template for other trustworthy ML problems—e.g., weakening explainability from "correct explanation" to "explanation faithful to the model's internals."
- **Utility-Error Curve as a New Metric**: Replacing single-point metrics with a curve forces the claim "our method reduces hallucinations" into "at a fixed error rate, I provide more utility"—a concrete suggestion for improving evaluation culture.
- **Storage vs. Control Problem Decoupling**: Splitting reliability in the agent era into these two layers highlights the necessity of metacognition, countering the naive optimism that "retrieval is enough."
- **Reasoning Models as a Counter-example**: The authors do not hide the fact that reasoning models hallucinate more; they use it to prove that utility-only objectives disincentivize honesty.

## Limitations & Future Work
- As a position paper, there are no new algorithms or experimental raw data; the simulation in Fig. 2 is illustrative rather than empirical evidence.
- The claim "faithful uncertainty is theoretically feasible" assumes models have readable internal confidence signals. If mechanistic interpretability pessimists are right (that there is no separable truth direction in latent states), this path fails; §7.3 acknowledges this without a definitive backup.
- The six research challenges lack operational solutions; specifically, how to perform "dynamic SFT labeling + preserving base model confidence" remains an open question.
- Suggestions for agent evaluation are somewhat abstract and lack a deployable metric.
- Feasibility of "hedge per assertion" in multimodal or long-form generation scenarios is not discussed.

## Related Work & Insights
- **vs. Calibration work by Kadavath et al.**: They prove LLMs can be well-calibrated; this paper argues that is insufficient and discrimination must be considered.
- **vs. Kalai 2024 (Calibrated models must hallucinate)**: Inherits the impossibility conclusion but proposes a reframe: if zero hallucination is unreachable, change the goal.
- **vs. Yona 2024 (Faithful Uncertainty)**: Yona provides definitions and cMFG metrics; this paper elevates the tool to a policy proposal for training pipelines.
- **vs. Tool-augmented LLM trends (ReAct, Toolformer, search agents)**: Argues that tool use cannot replace metacognition; uses evidence of search agent failures (qian2025smart, lin2025adasearch) as support.
- **Insight**: All "trustworthy XX" research can ask—is the goal to align with the world or the internal state? The former is usually unreachable; the latter usually reachable. This logic applies to explainability, safety, and honest AI.

## Rating
- Novelty: ⭐⭐⭐⭐ The conceptual shift from "world-alignment" to "internal-alignment" is clear, though the faithful uncertainty definition originates from Yona 2024.
- Experimental Thoroughness: ⭐⭐ Position paper with only simulations and literature reviews; no new data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentative chain; honest treatment of counter-examples; challenges provide a strong roadmap.
- Value: ⭐⭐⭐⭐ Significant value for re-aligning research directions in trustworthy LLMs and agent metacognition, though implementation requires follow-up work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](../../NeurIPS2025/hallucination/auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[CVPR 2026\] Evaluating and Easing Hallucinations for GUI Grounding](../../CVPR2026/hallucination/exposing_and_evaluating_hallucinations_for_gui_grounding.md)
- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)
- [\[ICML 2026\] From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)

</div>

<!-- RELATED:END -->
