---
title: >-
  [Paper Note] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal
description: >-
  [ICML 2026][LLM Reasoning][Chain-of-thought] A simple logistic regression probe on LLM hidden states during chain-of-thought (CoT) generation can predict whether the entire reasoning will be incorrect with 0.95 AUROC (0.…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Chain-of-thought"
  - "hidden state probe"
  - "error detection"
  - "activation steering"
  - "causal intervention"
date: 2026-05-08
content_hash: fa7d7ca3cfb37552
---

# Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal

**Conference**: ICML 2026  
**arXiv**: [2605.09502](https://arxiv.org/abs/2605.09502)  
**Code**: Key code snippets provided in the paper appendix  
**Area**: LLM Reasoning / Mechanistic Interpretability / CoT Faithfulness  
**Keywords**: Chain-of-thought, hidden state probe, error detection, activation steering, causal intervention

## TL;DR
A simple logistic regression probe on LLM hidden states during chain-of-thought (CoT) generation can predict whether the entire reasoning will be incorrect with 0.95 AUROC (0.79 from the first step), while a classifier trained on surface text achieves only 0.59; unfortunately, all four intervention methods (activation steering, probe-guided best-of-N, self-correction, activation patching) fail—this error signal is "diagnostic" rather than "causal."

## Background & Motivation

**Background**: The implicit contract of CoT prompting is "the model's written reasoning = its internal computation." Previous work (Turpin et al., Lanham et al.) has questioned this contract via behavioral interventions (e.g., interrupting CoT but answer unchanged), but these remain at the surface text level. Meanwhile, mechanistic interpretability (Meng et al. ROME, Li et al. ITI) has successfully located and edited factual knowledge representations, naturally raising the question: can reasoning errors be similarly located and corrected?

**Limitations of Prior Work**: (1) Verbalized confidence is unreliable—error trajectories self-report 4.55/5, correct ones 4.87/5, nearly indistinguishable; (2) No observable error signals on the text surface (hedging words, length differences are statistically insignificant); (3) Existing self-correction methods have limited effect, but lack mechanistic explanation for their failure.

**Key Challenge**: "The model knows it is wrong" and "the model can correct itself" are distinct. The former is representational (probes can read it out), the latter is causal (can we intervene). If these are decoupled, the "diagnose-edit" paradigm of mechanistic interpretability fails for reasoning tasks.

**Goal**: (1) Quantify the predictive power of model hidden states for its own CoT correctness, and strictly compare with surface text signals; (2) Use four increasingly strong intervention experiments to rigorously distinguish "diagnostic" vs "causal" signals; (3) Generalize this distinction across model scales, families, and training paradigms (RL distilled).

**Key Insight**: Train a linear probe $p(\text{error} \mid h) = \sigma(w^\top h + b)$ using 100 MATH-500 problems as training set, with 5-fold CV to select the best layer; simultaneously train a TF-IDF + LR text classifier as the "surface text control." The AUROC gap between the two quantifies the degree of "concealment" in hidden vs surface signals.

**Core Idea**: Use the AUROC gap between hidden and surface signals to quantify the model's "hidden error awareness," then test whether this awareness can be leveraged to correct errors using four methods: activation steering, best-of-N probe selection, self-correction, and activation patching.

## Method

### Overall Architecture
A three-act structure: (1) Train a probe using the hidden state at the last token to linearly classify whether the entire CoT yields an incorrect answer; (2) Extract the first-step hidden state for "early detection" experiments, and train a text classifier as a "surface control"; (3) Apply four causal interventions along the "error direction" read out by the probe.

### Key Designs

1. **Hidden State Linear Probe (Source of Diagnostic Signal)**:

    - **Function**: Reads out "will this trajectory be wrong" from frozen LLM hidden states.
    - **Mechanism**: Greedily generate full CoT, use the hidden state $h_T^{(l)}$ at the last token of layer $l$ as features, label $y = 1$ if the answer is wrong, train L2-regularized logistic regression ($C = 0.1$). Use 100 problems for training, 5-fold CV to select the best layer (typically 70-85% depth), 200 problems for evaluation.
    - **Design Motivation**: The linear probe is intentionally the weakest form; if it achieves 0.95 AUROC, the information is "linearly separable" in hidden states, requiring no complex nonlinear explanation. MLP (0.944) and RF (0.893) underperform the linear probe (0.955), supporting this.

2. **Surface Text Control + First-Step Prediction (Concealment Measurement)**:

    - **Function**: Establishes a baseline for "if the information is also in the text" and locates whether the error signal appears in the first CoT step.
    - **Mechanism**: On the same 200 problems, feed the first-step text of each trajectory to a TF-IDF + LR classifier to get $s_{text}$; use the first-step hidden state to train a probe for $s_{hidden}$. Define "concealment gap" $\Delta_{conceal} = s_{hidden} - s_{text}$. On Qwen2.5-3B, $s_{hidden} = 0.787, s_{text} = 0.590$, $\Delta_{conceal} = 0.197$.
    - **Design Motivation**: First-step AUROC is key—0.787 shows the model "knows it will be wrong from the first step," while the text shows nothing (length $p = 0.211$, digit density $p = 0.726$, hedging words < 1%). This provides rigorous evidence for "hidden error awareness." 65% of error trajectories have both "verbal confidence ≥ 4 and probe error score > 0.5," termed the unfaithful region.

3. **Four-Stage Causal Intervention (Diagnostic vs Causal Boundary)**:

    - **Function**: Tests whether "probe can read it out" is equivalent to "we can fix it."
    - **Mechanism**: (a) **Activation Steering**: Subtract normalized error direction from hidden state $h' = h - \alpha (h \cdot \hat w) \hat w$, at most +4%; strong intervention ($\alpha = 8$) actually drops 6%. (b) **Probe-Guided Best-of-N**: Generate $N$ trajectories, select the one with lowest probe score, surprisingly worse than majority vote; oracle upper bound is 88-91%, indicating correct answers exist among candidates but cannot be selected. (c) **Self-Correction**: Use probe to flag suspicious cases and have the model retry, but accuracy drops 3%; even oracle triggers only increase by 1%. (d) **Activation Patching** (strongest test): Mix hidden states from correct into wrong trajectories $h'_{wrong} = (1 - \alpha) h_{wrong} + \alpha h_{correct}$, with $\alpha = 0.5$ the 3B model's accuracy drops to 0%—patching destroys output coherence.
    - **Design Motivation**: The four interventions span the spectrum from "reading out direction" to "implanting correct representation"; all failing strictly demonstrates the signal is "just a thermometer." The collapse of activation patching is especially key: since factual knowledge can be edited by the same means in ROME, this contrast proves that "reasoning quality" and "factual association" are fundamentally different at the representational level—the former is distributed and emergent, the latter locally editable.

### Loss & Training

Probe: Standard L2-regularized logistic regression $\mathcal{L} = -\frac{1}{N}\sum [y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)] + \frac{1}{2C}\|w\|^2$, $C = 0.1$, up to 2000 iterations. With just 20 problems, AUROC reaches 0.956, indicating very high data efficiency.

## Key Experimental Results

### Main Results

| Model | Type | Accuracy | Best Layer | CV AUROC | Eval AUROC |
|-------|------|----------|------------|----------|------------|
| Qwen2.5-1.5B | std | 0.35 | 27 | 0.918 | 0.724 |
| Qwen2.5-3B | std | 0.53 | 27 | 0.953 | **0.956** |
| Qwen2.5-7B | std | 0.62 | 16 | 0.669 | 0.737 |
| Qwen2.5-32B | std | 0.53 | 32 | **0.956** | — |
| Qwen2.5-72B | std | 0.41 | 64 | **0.977** | — |
| Llama-3.1-8B | std | 0.46 | 16 | 0.703 | 0.811 |
| Phi-3.5-mini | std | 0.39 | 8 | 0.936 | — |
| DeepSeek-R1-7B | RL distilled | 0.76 | 12 | 0.884 | 0.852 |

| Detection Method | AUROC | Cost |
|------------------|-------|------|
| **Hidden State Probe (Ours)** | **0.953** | 1 forward pass |
| Self-Consistency ($N = 5$) | 0.823 | 5× generation |
| CCS (Burns et al.) | 0.718 | 1 forward pass |
| $P(\text{True})$ | 0.721 | 1 query |
| Verbalized Confidence | 0.674 | 1 query |
| Sequence Log-prob | 0.676 | Free |

### Intervention Experiments

| Intervention | 3B Accuracy Change | 7B Accuracy Change |
|--------------|-------------------|-------------------|
| Activation Steering (optimal $\alpha$) | +3% | +4% |
| Activation Steering ($\alpha = 8$, strong) | -6% | +4% |
| Best-of-N Probe Selection ($N = 12$) | 0.62 (oracle 0.88) | 0.70 (oracle 0.91) |
| Majority Vote (control) | 0.75 | 0.75 |
| Self-Correction (always retry) | -3% | -3% |
| Activation Patching (mix correct, $\alpha = 0.5$) | **0%** | **7%** |

### Key Findings
- **Predictable from the First Step**: Qwen2.5-3B achieves 0.787 AUROC on the first step, already 98% of the full trajectory—model "knows" it will be wrong before writing the error.
- **Two Temporal Dynamics**: 3B is "front-loaded" (first step gap max 0.41, early commitment to error), 7B is "accumulative" (gap grows from 0.11 to 0.38), indicating different error encoding by model scale.
- **Weak Cross-Domain Transfer**: Probe trained on MATH achieves only 0.54-0.55 AUROC on ARC-Challenge, but native ARC probe gets 0.63-0.69, showing "error direction" is domain-specific, not a single universal direction.
- **Non-monotonic Scale Curve**: 1.5B (0.918) → 7B (0.669) → 72B (0.977); the mid-scale dip is due to more dispersed, less linearly separable errors, while large models regularize error representations.
- **Collapse of Activation Patching as Decisive Evidence**: Factual knowledge can be edited, but reasoning quality cannot—reasoning is distributed and emergent, not a single editable feature, fundamentally different from ROME and similar factual editing work.

## Highlights & Insights
- **"Diagnostic vs Causal" Dichotomy**: The thermometer analogy clearly distinguishes "readable" from "editable," a valuable conceptual contribution for mechanistic interpretability, delineating the boundary of methods like ROME.
- **Representational Evidence for CoT Faithfulness**: Previous discussions of CoT faithfulness were behavioral; this work provides strict quantification ("hidden vs surface AUROC = 0.20"), establishing "hidden error awareness" as a new concept.
- **Honest but Pessimistic**: The comprehensive failure of all four interventions is a convincing negative result; the authors do not hide failed experiments but make them central to the main conclusion—a commendable research style.
- **Implications for Process Reward Models**: PRMs may be learning the same diagnostic signal, lacking causal leverage, implying their main use is "selection" rather than "training-time correction of reasoning."

## Limitations & Future Work
- Did not test fine-tuning or RL (e.g., RLPF) on probe signals; this may be a training-time solution to bridge the "diagnostic-causal" gap, acknowledged as an open problem.
- Experiments mainly on MATH-500, with cross-domain tests on ARC, but not on other reasoning tasks (HumanEval, TheoremQA, MMLU-Pro).
- All four interventions are post-training; did not attempt joint training with probe loss during pretraining, which may be the real solution.
- On DeepSeek-R1, the "difficulty control" effect is weak ($p = 0.447$, $d = -0.30$), possibly due to too few mixed trajectories at 76% accuracy ($n = 14$), limiting statistical power.

## Related Work & Insights
- **vs Turpin et al. 2023 / Lanham et al. 2023**: Prior work questioned CoT faithfulness via behavioral interventions (prompt injection, trajectory truncation); this work quantifies "concealment" at the representational level via AUROC gap, methodologically complementary.
- **vs ROME (Meng et al., 2022)**: ROME can edit factual associations; this work shows the "read + edit" paradigm fails completely for reasoning errors, delineating the boundary for mechanistic interpretability.
- **vs Zhang et al., 2025** (parallel work): They also probe hidden states for self-verification and find best-of-N selection works; this work conversely shows probes cannot improve reasoning itself—opposite but complementary conclusions.
- **vs CCS (Burns et al., 2023)**: CCS uses unsupervised methods to find "truth directions"; this work uses supervised probes for error detection, achieving AUROC 0.953 vs CCS's 0.718, showing clear advantage for supervision in this task.

## Rating
- Novelty: ⭐⭐⭐⭐ The "diagnostic vs causal" dichotomy is a clear conceptual innovation; hidden vs surface quantification is a new perspective, though probes themselves are standard tools.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models covering 1.5B-72B + Qwen/Llama/Phi + RL distilled, 4 interventions + cross-domain + difficulty control + layer analysis—extremely thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ The "three-act" structure is dramatic; the thermometer analogy makes concepts accessible; negative results are clearly presented without concealment.
- Value: ⭐⭐⭐⭐⭐ Raises important alarms for LLM safety monitoring (CoT auditing unreliability), mechanistic interpretability (boundaries), and PRM training, with broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)
- [\[AAAI 2026\] Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](../../AAAI2026/llm_reasoning/deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)
- [\[ACL 2026\] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](../../ACL2026/llm_reasoning/do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)
- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](../../ICLR2026/llm_reasoning/verifying_chain-of-thought_reasoning_via_its_computational_graph.md)

</div>

<!-- RELATED:END -->
