---
title: >-
  [Paper Note] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal
description: >-
  [ICML 2026][LLM Reasoning][Chain-of-Thought] Using a simple logistic regression probe on the hidden states of an LLM during Chain-of-Thought (CoT) generation allows for predicting reasoning errors with 0.95 AUROC (starti…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Hidden State Probe"
  - "Error Detection"
  - "Activation Guidance"
  - "Causal Intervention"
date: 2026-05-08
content_hash: fe737bb453b6d173
---

# Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal

**Conference**: ICML 2026  
**arXiv**: [2605.09502](https://arxiv.org/abs/2605.09502)  
**Code**: The paper appendix provides key code snippets  
**Area**: LLM Reasoning / Mechanistic Interpretability / CoT Faithfulness  
**Keywords**: Chain-of-Thought, Hidden State Probe, Error Detection, Activation Guidance, Causal Intervention

## TL;DR
Using a simple logistic regression probe on the hidden states of an LLM during Chain-of-Thought (CoT) generation allows for predicting reasoning errors with 0.95 AUROC (starting at 0.79 from the very first step). In contrast, a textual surface classifier achieves only 0.59. Despite this strong signal, four types of interventions (activation guidance, probe-guided best-of-N, self-correction, and activation patching) all failed—demonstrating that this error signal is "diagnostic" rather than "causal."

## Background & Motivation

**Background**: The implicit contract of CoT prompting is that "the model's written reasoning = its internal computation process." Prior works (Turpin et al., Lanham et al.) have questioned this contract via behavioral interventions (e.g., perturbing the CoT but the answer remains unchanged), but these remained at the textual surface. Given that mechanistic interpretability (Meng et al. ROME, Li et al. ITI) has successfully localized and edited factual knowledge representations, it is natural to ask: Can reasoning errors be localized and corrected in the same way?

**Limitations of Prior Work**: (1) Verbalized confidence is unreliable—models report 4.55/5 for incorrect trajectories vs. 4.87/5 for correct ones, making them nearly indistinguishable; (2) There are no observable textual error signals (e.g., hedging words or length differences are not statistically significant); (3) Existing self-correction methods show limited efficacy and lack mechanistic explanations for their failure.

**Key Challenge**: Whether a model "knows it is wrong" is distinct from whether it can "correct its wrong." The former pertains to the representational level (readable by probes), while the latter pertains to causal mechanisms (controllability via intervention). If these are decoupled, the "diagnose-and-edit" paradigm of mechanistic interpretability may fail for reasoning tasks.

**Goal**: (1) Quantify the predictive power of hidden states regarding CoT correctness compared to textual surface signals; (2) Strictly distinguish between "diagnostic" and "causal" signals through four progressively stronger intervention experiments; (3) Generalize these findings across different model scales, families, and training paradigms (e.g., RL distilled).

**Key Insight**: Train a linear probe $p(\text{error} \mid h) = \sigma(w^\top h + b)$ using 100 MATH-500 problems as the training set and 5-fold CV to select the optimal layer. Simultaneously, train a TF-IDF + LR textual classifier as a "textual surface control." The AUROC gap between the two quantifies the degree of concealment between hidden and surface signals.

**Core Idea**: Use the AUROC gap between "hidden vs. surface" to quantify the model's "hidden error awareness," then employ a suite of four interventions—activation guidance, best-of-N, self-correction, and activation patching—to test whether this awareness can be leveraged to correct errors.

## Method

### Overall Architecture
The study is structured in three acts: (1) Training probes using hidden state features at the final token to predict if the full CoT results in an error; (2) Performing "early detection" experiments using step-one hidden states and comparisons against textual classifiers; (3) Conducting four causal intervention tests based on the "error direction" identified by the probe.

### Key Designs

1.  **Hidden State Linear Probe (Diagnostic Signal Source)**:
    - **Function**: Reads the probability of an incorrect trajectory from frozen LLM hidden states.
    - **Mechanism**: CoTs are generated greedily. The hidden state $h_T^{(l)}$ of the last token at layer $l$ is used as the feature, with label $y = 1$ representing an incorrect answer. An L2-regularized logistic regression ($C = 0.1$) is trained on 100 problems, with 5-fold CV used to select the best layer (typically at 70-85% depth). 200 problems are used for evaluation.
    - **Design Motivation**: A linear probe is intentionally used as the weakest classifier. If it achieves 0.95 AUROC, it implies the information is "linearly separable" in the hidden states. The fact that MLP (0.944) and Random Forest (0.893) do not outperform the linear probe (0.955) supports this choice.

2.  **Textual Surface Baseline + First-Step Prediction (Concealment Metric)**:
    - **Function**: Establishes a baseline for information detectable in the text and determines if the error signal originates at the start of the CoT.
    - **Mechanism**: For the same 200 problems, the text of the first step is fed into a TF-IDF + Logit model to Get $s_{text}$. The first-step hidden state probe provides $s_{hidden}$. The "concealment gap" is defined as $\Delta_{conceal} = s_{hidden} - s_{text}$. On Qwen2.5-3B, $s_{hidden} = 0.787$ and $s_{text} = 0.590$, yielding $\Delta_{conceal} = 0.197$.
    - **Design Motivation**: The first-step AUROC is critical—0.787 indicates the model "knows it will fail from the start," even when the text shows no signs (length $p = 0.211$, digit density $p = 0.726$, hedging $< 1\%$). This provides rigorous evidence for "hidden error awareness." 65% of error trajectories fall into an "unfaithful region" (high verbal confidence $\ge 4$ but high probe error score $> 0.5$).

3.  **Four-Stage Causal Intervention (Diagnostic vs. Causal Boundary)**:
    - **Function**: Tests whether "probe readability" implies "correctability."
    - **Mechanism**: (a) **Activation Guidance**: Subtracting the normalized error direction $h' = h - \alpha (h \cdot \hat w) \hat w$. Accuracy gains are max +4%, while strong intervention ($\alpha = 8$) results in a 6% drop. (b) **Probe-guided best-of-N**: Selecting the trajectory with the lowest probe score among $N$ samples; surprisingly, this performs worse than majority voting. (c) **Self-correction**: Prompting a retry when the probe detects suspicion results in a 3% drop. (d) **Activation Patching** (Strongest test): Mixing hidden states from correct trajectories into incorrect ones $h'_{wrong} = (1 - \alpha) h_{wrong} + \alpha h_{correct}$. At $\alpha = 0.5$, 3B accuracy drops to 0% because the patch destroys coherence.
    - **Design Motivation**: These cover a spectrum of intensity from "direction steering" to "representation replacement." Failure across all four proves the signal is merely a "thermometer." The collapse during activation patching is crucial: as factual knowledge *can* be edited this way (ROME), it proves "reasoning quality" differs fundamentally from "factual associations"—the former is a distributed, multi-layer emergent property.

### Loss & Training
Probe: Standard L2-regularized logistic regression: 
$$\mathcal{L} = -\frac{1}{N}\sum [y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)] + \frac{1}{2C}\|w\|^2$$ 
with $C = 0.1$ and up to 2000 iterations. Achieving 0.956 AUROC requires only 20 problems, showing high data efficiency.

## Key Experimental Results

### Main Results

| Model | Type | Accuracy | Best Layer | CV AUROC | Eval AUROC |
|------|------|--------|--------|----------|------------|
| Qwen2.5-1.5B | std | 0.35 | 27 | 0.918 | 0.724 |
| Qwen2.5-3B | std | 0.53 | 27 | 0.953 | **0.956** |
| Qwen2.5-7B | std | 0.62 | 16 | 0.669 | 0.737 |
| Qwen2.5-32B | std | 0.53 | 32 | **0.956** | — |
| Qwen2.5-72B | std | 0.41 | 64 | **0.977** | — |
| Llama-3.1-8B | std | 0.46 | 16 | 0.703 | 0.811 |
| Phi-3.5-mini | std | 0.39 | 8 | 0.936 | — |
| DeepSeek-R1-7B | RL distilled | 0.76 | 12 | 0.884 | 0.852 |

| Detection Method | AUROC | Cost |
|----------|-------|------|
| **Hidden State Probe (Ours)** | **0.953** | 1 Forward |
| Self-Consistency ($N = 5$) | 0.823 | 5× Gen |
| CCS (Burns et al.) | 0.718 | 1 Forward |
| $P(\text{True})$ | 0.721 | 1 Query |
| Verbalized Confidence | 0.674 | 1 Query |
| Sequence Log-prob | 0.676 | Free |

### Intervention Experiments

| Intervention Method | 3B Accuracy Change | 7B Accuracy Change |
|----------|---------------|---------------|
| Activation Guidance (Optimal $\alpha$) | +3% | +4% |
| Activation Guidance ($\alpha = 8$, Strong) | -6% | +4% |
| Best-of-N Probe Selection ($N = 12$) | 0.62 (oracle 0.88) | 0.70 (oracle 0.91) |
| Majority Voting (Control) | 0.75 | 0.75 |
| Self-correction (always retry) | -3% | -3% |
| Activation Patching (Mixed, $\alpha = 0.5$) | **0%** | **7%** |

### Key Findings
- **Predictable from the first step**: Qwen2.5-3B first-step AUROC is 0.787, reaching 98% of the full trajectory AUROC—the model "knows" internally that it will fail before it writes the error.
- **Two temporal dynamics**: 3B is "pre-emptive" (largest gap of 0.41 at step 1, early commitment to error path), while 7B is "cumulative" (gap grows from 0.11 to 0.38), indicating different error encoding styles across scales.
- **Weak cross-domain transfer**: Probes trained on MATH only achieve 0.54-0.55 AUROC on ARC-Challenge, though native ARC probes reach 0.63-0.69, suggesting the "error direction" is domain-specific.
- **Non-monotonic scaling curve**: 1.5B (0.918) → 7B (0.669) → 72B (0.977). The dip in mid-sized models suggests their errors are more scattered and less linearly separable, whereas larger models regularize error representations.
- **Activation patching collapse is the smoking gun**: The ability to edit facts but failure to edit reasoning quality demonstrates that reasoning is a distributed emergent property rather than a localizable feature, a key distinction from ROME-style factual editing.

## Highlights & Insights
- **"Diagnostic vs. Causal" Dichotomy**: The authors use a thermometer analogy to distinguish between "readability" and "controllability," drawing a clear boundary for mechanistic interpretability methods like ROME.
- **Representational Evidence for CoT Faithfulness**: While previous discussions on CoT faithfulness were behavioral, this paper provides a rigorous quantification through the "hidden vs. surface AUROC = 0.20" gap, establishing the concept of "hidden error awareness."
- **Honest Negative Results**: The failure of all four interventions is a compelling negative result. The authors do not hide these failures but use them as primary evidence for their main conclusion.
- **Implications for Process Reward Models (PRMs)**: Signals learned by PRMs may be purely diagnostic with no causal leverage, suggesting PRMs are best suited for "selection" rather than "on-the-fly correction" during training.

## Limitations & Future Work
- The study did not test fine-tuning or RL (e.g., RLPF) on the probe signal, which might be a training-time solution to bridge the "diagnostic-causal" gap.
- Experiments were primarily on MATH-500; while ARC was used for cross-domain testing, other reasoning tasks (HumanEval, TheoremQA, MMLU-Pro) were not covered.
- Interventions were post-training; joint training with a probe loss during pre-training was not explored as a potential solution.
- The "difficulty control" effect was weaker on DeepSeek-R1 ($p = 0.447$, $d = -0.30$), likely due to the small sample size of incorrect trajectories at 76% accuracy ($n = 14$).

## Related Work & Insights
- **vs. Turpin et al. 2023 / Lanham et al. 2023**: While prior work used behavioral interventions (prompt injection, truncation) to question CoT faithfulness, this work quantifies "concealment" at the representational level via AUROC gaps.
- **vs. ROME (Meng et al., 2022)**: ROME edits factual associations; this paper proves the same "read-and-edit" paradigm fails for reasoning errors, setting a boundary for mechanistic interpretability.
- **vs. Zhang et al., 2025** (Concurrent work): They also use probes for self-verification and find them useful for Best-of-N selection; this paper proves probes cannot improve the reasoning itself, providing an opposite but complementary perspective.
- **vs. CCS (Burns et al., 2023)**: CCS finds a "truth direction" unsupervised; this work uses supervised probes for error detection, showing a significant advantage (0.953 vs. 0.718 AUROC).

## Rating
- Novelty: ⭐⭐⭐⭐ The "diagnostic vs. causal" dichotomy is a clear conceptual innovation, though probing itself is a standard tool.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models from 1.5B-72B (Qwen/Llama/Phi/DeepSeek), 4 interventions, cross-domain tests, and layer analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear "three-act" structure and excellent use of analogies to explain complex findings.
- Value: ⭐⭐⭐⭐⭐ Provides critical warnings for AI monitoring (unreliable CoT audits), mechanistic interpretability (boundaries), and PRM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ICML 2026\] Chain-of-Thought Reasoning in the Wild Is Not Always Faithful](chain-of-thought_reasoning_in_the_wild_is_not_always_faithful.md)
- [\[AAAI 2026\] Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning](../../AAAI2026/llm_reasoning/deep_hidden_cognition_facilitates_reliable_chain-of-thought_.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](../../ACL2026/llm_reasoning/is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)

</div>

<!-- RELATED:END -->
