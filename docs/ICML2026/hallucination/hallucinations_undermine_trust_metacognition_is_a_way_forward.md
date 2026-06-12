---
title: >-
  [Paper Note] Hallucinations Undermine Trust; Metacognition is a Way Forward
description: >-
  [ICML 2026 (Position Paper)][Hallucination Detection][Hallucination] This position paper argues that "completely eliminating LLM hallucinations" is fundamentally subject to a "discrimination gap" (discrimination gap → ut…
tags:
  - "ICML 2026 (Position Paper)"
  - "Hallucination Detection"
  - "Hallucination"
  - "Calibration vs Discrimination"
  - "faithful uncertainty"
  - "metacognition"
  - "agentic control layer"
date: 2026-05-08
content_hash: de718931c2076518
---

# Hallucinations Undermine Trust; Metacognition is a Way Forward

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.01428](https://arxiv.org/abs/2605.01428)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Hallucination, Calibration vs Discrimination, faithful uncertainty, metacognition, agentic control layer

## TL;DR
This position paper argues that "completely eliminating LLM hallucinations" is fundamentally subject to a "discrimination gap" (discrimination gap → utility tax); the authors advocate shifting the goal from "eliminating hallucinations" to **faithful uncertainty**, and view such metacognition as an indispensable control layer when agentic LLMs invoke tools.

## Background & Motivation
**Background**: Despite frontier models continuously improving factual reliability, hallucination remains the biggest obstacle to industrial deployment. Research directions fall into two main categories: (1) Training interventions—data filtering, alignment penalties, reward model calibration; (2) Inference interventions—special decoding (DOLA), internal signal probes, self-verification. Meanwhile, uncertainty quantification has shown that modern LLMs can output **well-calibrated** confidence signals.

**Limitations of Prior Work**: All efforts implicitly aim to "reduce hallucination rate to zero." However, even with perfectly calibrated confidence, to reduce hallucination from 25% to 5%, the model must forgo **52% of correct answers** (utility tax). AUROC on real tasks stabilizes at $0.70$–$0.85$; only by surpassing $0.95$ can the utility tax drop below 5%—which no current method achieves.

**Key Challenge**: The authors formalize the dilemma as a gap between calibration and discrimination. **Calibration** only requires "among samples with confidence 0.6, 60% are correct"—alignment in an average sense; **discrimination** requires "I can pick out which 60% are correct." A constant predictor always outputting 0.6 is perfectly calibrated but has zero discrimination. Existing theory (Halting Problem analogies, calibrated-models must hallucinate, consistency–breadth trade-off) all suggest that the discrimination ceiling for LLMs is fundamentally limited.

**Goal**: (1) Formalize the unattainability of "eliminating hallucinations"; (2) Propose an alternative goal that circumvents this unattainability; (3) Extend this goal to agentic systems as a control layer for tool invocation.

**Key Insight**: Redefine "hallucination"—not as "any error," but as "**confidently** wrong." If an error is accompanied by appropriate hedging ("I'm not sure, maybe 1961"), it is no longer a hallucination but a hypothesis. This redefinition dissolves the binary opposition between "eliminating errors" and "maintaining utility"—a third path emerges: honestly expressing uncertainty.

**Core Idea**: Replace "zero hallucination" with **faithful uncertainty** (aligning linguistic uncertainty with intrinsic uncertainty), making metacognition a core capability of LLMs and agents.

## Method

### Overall Architecture
As a position paper, there is no new algorithm, but a conceptual framework of **problem redefinition + feasibility argument + research roadmap** is proposed. The argument proceeds in three steps: (1) Demonstrate that "completely eliminating hallucinations" is fundamentally impossible; (2) Argue that faithful uncertainty is **possible** in principle and can unlock reliable utility; (3) In agentic scenarios, metacognition serves as the control layer for tool invocation. Six major research challenges and three hallucination mitigation evaluation criteria are presented.

### Key Designs

1. **Clear Separation of Calibration vs Discrimination + Visualization of Discrimination Tax**:

    - **Function**: Visually demonstrates the counterintuitive fact that "good calibration still incurs a large utility tax."
    - **Mechanism**: Simulated data reproduces Nakkiran 2025's reliability diagram: confidence for correct ($y=1$) samples from $\text{Beta}(1.8,1.0)$, incorrect ($y=0$) from $\text{Beta}(1.0,1.3)$, then Isotonic regression enforces calibration to smECE $\approx 0.014$. AUROC = 0.71 (matching literature $0.70$–$0.85$). Then, plot the utility-error trade-off: sweep the refusal threshold from 0 to 1, and observe how many correct answers are lost at each target error rate. Conclusion: to reduce a 25% error rate to 5%, 52% of correct answers must be discarded; even with AUROC rising to 0.85, the tax is still $\sim$28%; only at $\geq 0.95$ does it drop below 5%. This trade-off quantifies "why model providers are unwilling to pay this tax."
    - **Design Motivation**: Previous discussions often conflated calibration and discrimination, leading to "our model is well-calibrated" being misread as "our model does not hallucinate." This visualization thoroughly separates the two concepts and concretizes the cost as "how many correct answers must be lost," which is far more impactful than ECE numbers.

2. **Faithful Uncertainty: Shifting the Goal from "Aligning with the World" to "Aligning Internally"**:

    - **Function**: Proposes a **theoretically attainable** relaxed goal—the model's linguistic uncertainty must faithfully reflect its internal uncertainty, but does not require this internal uncertainty to match the world's ground truth.
    - **Mechanism**: Based on Yona 2024's definition. Intrinsic confidence $\text{conf}_M(A)=1-\frac{1}{k}\sum_i\mathbf 1[A\text{ contradicts }A_i]$ (semantic consistency over $k$ resamplings). Linguistic decisiveness $\text{dec}(A;R,Q)=\Pr[A\text{ True}\mid R,Q]$ (estimated by LLM-as-judge based on hedge strength). Faithfulness $=1-\frac{1}{|A(R)|}\sum_{A}|\text{dec}(A;R,Q)-\text{conf}_M(A)|$. The authors argue this goal is **closed-loop observable**—since $\text{conf}_M$ is a function of model weights, "linguistic output aligns with conf" is an internal consistency issue, not dependent on external ground truth; thus, it avoids the halting-problem barrier of xu2024 in principle.
    - **Design Motivation**: By shifting the goal from "internal → real world" to "internal → internal," the discrimination tax disappears—the model need not refuse to answer to avoid errors, but can provide uncertain answers in hedged form; users still receive useful guesses, now honestly labeled.

3. **Agentic Metacognition: Uncertainty as the Control Layer for Tool Invocation**:

    - **Function**: In the agent era, self-perceived uncertainty is not redundant but **foundational**—it determines when to invoke tools, when to trust retrieval results, and how to resolve conflicts with priors.
    - **Mechanism**: The agent harness is viewed as a coarse external scheduler, currently relying entirely on query-type heuristics to decide whether to search; introducing "model self-reported confidence" as a yellow control layer allows the harness to: high confidence → answer directly (save tool calls); low confidence → retrieve; retrieval result conflicts with prior → output hedged answer instead of blindly trusting context. The paper cites qian2025smart and others, noting that current search agents lack such self-awareness, leading to tool overuse or underuse.
    - **Design Motivation**: It is commonly assumed that "tool invocation can bypass hallucination problems," but the authors rebut: tools only solve the **storage problem** (not needing to encode all facts in weights), but introduce a **control problem** (when to retrieve, how to assess retrieved information's reliability). Faithful uncertainty precisely fills this control layer.

### Loss & Training
Not applicable (position paper). However, in §6, the authors propose six concrete research challenges: bootstrapping paradox (dynamic labels vs static SFT), protecting confidence signals during post-training, confidence attribution (distinguishing aleatoric/epistemic/normative uncertainties), strict causal evaluation (avoiding models merely learning hedging style rather than genuine self-awareness), agent evaluation should control process rather than end-to-end accuracy, and hallucination mitigation evaluation should use utility-error trade-off curves instead of single-point reports.

## Key Experimental Results

### Main Results

| Data/Scenario | Phenomenon | Meaning |
|---------------|------------|---------|
| Beta simulation, AUROC=0.71, base error rate 25% | Reducing to 5% requires discarding 52% correct answers | Discrimination tax is significant |
| Beta simulation, AUROC=0.85 | Tax $\sim$28% | Still heavy at current best levels |
| Beta simulation, AUROC=0.95 | Tax $<5\%$ | No method achieves this in knowledge-intensive tasks |
| SimpleQA Verified (multiple frontier models) | All distributed along diagonal or shifted left | Top-right "ideal" region is empty |
| AUROC literature review (farquhar2024, savage2025, kang2025) | AUROC 0.70–0.85 on real knowledge-intensive QA | Confirms discrimination gap |

### Ablation Study
(None; replaced by cMFG and other faithful uncertainty evaluation proxies.)

| Evaluation Dimension | Prev. SOTA | Ours |
|---------------------|------------|------|
| cMFG (conditional mean faithful generation) | 0.5–0.7 | 1.0 |
| Reasoning model vs standard model confidence expression | Reasoning models better but hallucinate more | Metacognitive and factual signals decoupled |
| Internal truth probe AUROC (mech interp) | Collapses on OOD | Does not assume universal truth direction |

### Key Findings
- **Calibration ≠ No Hallucination**: Even with perfect calibration, insufficient discrimination inevitably incurs utility tax; this is the paper's most counterintuitive and persuasive argument.
- **SimpleQA Scatter Plot (Fig. 3)**: No model occupies the "ideal" top-right—frontier models either hug the diagonal or shift left, paying a high refusal tax, indicating all current methods are stuck on the trade-off curve.
- **Extended reasoning increases hallucination**: o1-type reasoning models have lower refusal rates but higher hallucination rates; the authors attribute this to "reward optimizing utility, not honest uncertainty expression."
- **Pre-trained model uncertainty signals are eroded post-training**: he2025rewarding, song2025outcome, etc., show RLHF makes models mode-seeking and overconfident—this is a top-priority issue for metacognitive research.
- **Agentic evaluation must be process-based**: Current evaluations reward agents for "guessing the right answer," masking metacognitive failures (e.g., inefficiently searching known facts, trusting sources conflicting with priors = sycophancy).

## Highlights & Insights
- **Redefining the goal as "faithful ≠ correct"**: Shifting the goal from "aligning with external truth" to "aligning with internal state" cleverly circumvents the "truth is undecidable" barrier. This is a reusable thinking template for all trustworthy ML problems—for example, explainability can also be relaxed from "explanations must be correct" to "explanations must be faithful to the model's internals."
- **Utility-Error curve as a new evaluation**: Using a curve instead of a single-point metric, shifting the claim from "our method reduces hallucination rate" to "at a fixed error rate, I provide more utility"—a concrete proposal for evaluation culture reform.
- **Decomposing storage vs control problems**: By splitting reliability in the agent era into two layers, the necessity of metacognition becomes immediately apparent, avoiding the naive optimism of "retrieval alone suffices."
- **Honestly acknowledging that reasoning models hallucinate more**: The authors do not conceal this counterexample, but use it to show that utility-only training objectives actively disincentivize honesty.

## Limitations & Future Work
- The paper is a position paper, with no new algorithms or experimental data; simulation in Fig. 2 is illustrative, not empirical evidence.
- The claim that "faithful uncertainty is feasible in principle" rests on an implicit assumption—that the model indeed has an internal confidence signal that can be read out. If the pessimistic view of mech interp is correct (no separable truth direction in latent state), this path is blocked; the authors acknowledge this in §7.3 but offer no fallback.
- The six proposed research challenges lack actionable solutions; how to "combine dynamic SFT labels with preserving base model confidence" remains open.
- Suggestions for agent evaluation are abstract, with no generalizable metric provided.
- Feasibility of "hedge per assertion" in multimodal or long-form generation is not discussed.

## Related Work & Insights
- **vs Kadavath et al. on calibration**: They show LLMs can be well-calibrated; this paper argues that's far from enough—discrimination must be considered, deepening the dimensional distinction on the same object.
- **vs Kalai 2024 (calibrated models must hallucinate)**: This paper cites and inherits Kalai's impossibility result, but proposes a reframing: since zero hallucination is unattainable, change the goal.
- **vs Yona 2024 (faithful uncertainty)**: Yona provides the definition and cMFG metric; this paper upgrades the tool to a policy proposal—all training pipelines should move in this direction.
- **vs Tool-augmented LLMs (ReAct, Toolformer, search agents)**: This paper argues in reverse that tool use cannot replace metacognition; current search agent failures (qian2025smart, lin2025adasearch) are cited as key evidence.
- **Insights**: All "trustworthy XX" research should ask—is your goal to align with the world or with internals? The former is usually unattainable, the latter usually attainable. This line of thinking applies to explainability, safety, and honest AI.

## Rating
- Novelty: ⭐⭐⭐⭐ "Shifting the goal from aligning with the world to aligning internally" is a clear conceptual leap, though the definition of faithful uncertainty comes from Yona 2024
- Experimental Thoroughness: ⭐⭐ Position paper, only simulation and literature review, no new data
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation, honest counterexamples and rebuttals, six challenges provide concrete directions for future research
- Value: ⭐⭐⭐⭐ Significant for calibrating research directions in trustworthy LLMs and agent metacognition, but practical implementation requires further work

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](../../NeurIPS2025/hallucination/auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)
- [\[ICML 2026\] From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](../../ACL2026/hallucination/meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)

</div>

<!-- RELATED:END -->
