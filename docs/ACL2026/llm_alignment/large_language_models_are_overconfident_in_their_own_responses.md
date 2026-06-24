---
title: >-
  [Paper Note] Large Language Models Are Overconfident in Their Own Responses
description: >-
  [ACL2026 Findings][LLM Alignment][confidence calibration] This paper discovers that instruction-tuned LLMs exhibit a significant ownership bias when evaluating "answers they generated themselves," and proposes a simple inference-time strategy of rewriting the answer as a user input before asking for confidence to reduce overconfidence without retraining.
tags:
  - "ACL2026 Findings"
  - "LLM Alignment"
  - "confidence calibration"
  - "instruction tuning"
  - "chat template"
  - "ownership bias"
  - "inference-time mitigation"
date: 2026-05-08
content_hash: 3d6648289f463120
---

# Large Language Models Are Overconfident in Their Own Responses

**Conference**: ACL2026 Findings  
**arXiv**: [2606.03437](https://arxiv.org/abs/2606.03437)  
**Code**: No public code link found in cache  
**Area**: LLM Alignment / Calibration  
**Keywords**: confidence calibration, instruction tuning, chat template, ownership bias, inference-time mitigation  

## TL;DR
This paper discovers that instruction-tuned LLMs exhibit a significant ownership bias when evaluating "answers they generated themselves," and proposes a simple inference-time strategy of rewriting the answer as a user input before asking for confidence to reduce overconfidence without retraining.

## Background & Motivation
**Background**: Trustworthy LLMs need to correctly express uncertainty. Existing research shows that the next-token probabilities of base LLMs are often closer to being calibrated than instruction-tuned/chat models, while post-training such as SFT/RLHF may lead to overconfidence.

**Limitations of Prior Work**: Much past work has conflated instruction tuning, chat templates, and verbalized confidence in evaluations, making it difficult to determine whether miscalibration stems from training algorithms, chat formats, or the role-playing bias generated when the model "acts as an assistant."

**Key Challenge**: The most common form of LLM usage is instruction-tuned + chat template, but calibration evaluations often place "answer generation" and "answer evaluation" within the same assistant role. If the model inherently trusts its own output more, even with identical answer text, the confidence will change due to the identity of the speaker.

**Goal**: The authors aim to answer four questions: what are the individual effects of instruction tuning and chat templates on calibration; whether explicit confidence elicitation changes the trends; whether models are more confident in their own answers; and whether a non-weight-modifying inference-time strategy can mitigate this bias.

**Key Insight**: The paper decomposes the answer provider into assistant and user prompt framings. If the same answer receives higher confidence and worse ECE/Brier scores under the assistant framing, it indicates the problem is not just the answer content, but the model's ownership bias toward its "own output."

**Core Idea**: Feed the model-generated answer back to the model as a user message, then ask for its confidence. This switches the model from "author" to "observer," thereby reducing self-confirmatory overconfidence.

## Method
The paper does not train a new calibrator but instead performs controlled experiments to locate the mechanisms of miscalibration and proposes an inference-time prompt framing strategy. The core mechanism is to decouple the model version, chat template, confidence elicitation method, and the identity of the answer source step-by-step.

### Overall Architecture
First, the authors compare the base model, instruction-tuned model without chat template, and instruction-tuned model with chat template for each model family on MMLU, using logit-based confidence to calculate accuracy, ECE, and Brier score. Second, they introduce three explicit confidence elicitations: P(True), Verbalized Percentage, and Verbalized Linguistic, to test if the calibration damage from instruction tuning persists. Third, they fix the answer text and change only whether the answer appears in an assistant message or a user message, measuring differences in ECE, Brier, and raw confidence. Finally, they use "answer as user input" as an inference-time mitigation strategy and verify generalization across MMLU, GSM8K, TruthfulQA, open-ended MMLU, and GPT-5.2.

### Key Designs

**1. Decoupling instruction tuning and chat template: Identifying whether calibration degradation is due to training algorithms or chat formats**

Most previous work only evaluated instruct models under a chat template, so the effects of post-training and the influence of prompt formats were entangled. The authors compare three invocation methods side-by-side within the same model family: base model, instruct model without chat template, and instruct model with chat template, calculating accuracy, ECE, and Brier for all using logit-based confidence. The logic is straightforward: if the "instruct without chat" level is already significantly miscalibrated, the root cause lies primarily in instruction tuning rather than the chat format; the chat template merely adds another layer of influence. This separates the two factors into independently observable variables.

**2. Three confidence elicitations: Confirming that miscalibration is not a side effect of using logits as a metric**

The logits of instruction-tuned models may not inherently be suitable for interpretation as confidence, and using only one metric could lead to questions about the metric itself. The authors therefore use three other ways to ask for confidence: P(True), Verbalized Percentage (0–100%), and Verbalized Linguistic (seven levels), mapping the linguistic levels linearly to equidistant scores from 0 to 1. If even the confidence expressed in natural language is similarly impaired, it indicates that miscalibration occurs at the model level rather than being a byproduct of a specific confidence reading method.

**3. Assistant-vs-user ownership bias test: Directly verifying if models trust "their own" answers more**

To prove the issue is "who said it" rather than the answer content, the content must be fixed while only the identity is changed. For the same question and candidate answer, the authors only change whether it appears in an assistant message or a user message, then ask for confidence. The difference is defined as $\Delta = Assistant - User$, where a positive value means the assistant framing is more confident or less calibrated. This design also happens to disprove a counter-hypothesis: if sycophancy were dominant, the model should trust user-provided answers more (negative $\Delta$); however, experiments yielded a positive trend, supporting ownership bias—the model has an implicit self-trust in its own generation process. This design directly leads to the mitigation strategy: reframe the model's own answer as a user message when asking for confidence.

### Loss & Training
No new models were trained in this study. Evaluation metrics include accuracy, ECE, and Brier score. ECE uses 10 equal-width confidence bins, and the Brier score is the mean squared error between the probability prediction and the binary correctness label. For statistical significance, Wilcoxon signed-rank tests are used for Brier and raw confidence, and paired bootstrap resampling tests with $K=1000$ are used for ECE; significant differences are marked as $p<0.01$.

## Key Experimental Results

### Main Results
The first set of experiments shows that instruction tuning improves accuracy but damages calibration, which the chat template further exacerbates. Some MMLU logit-based results are listed below.

| Model | Setting | Accuracy | ECE | Brier |
|------|------|----------|-----|-------|
| Llama 3.1 8B | base, no chat | 62.81 | 0.0664 | 0.1706 |
| Llama 3.1 8B | instruct, chat | 69.12 | 0.1666 | 0.2005 |
| Qwen3 4B | base, no chat | 67.72 | 0.0425 | 0.1709 |
| Qwen3 4B | instruct, chat | 72.98 | 0.2415 | 0.2455 |
| Gemma 3 4B | base, no chat | 49.47 | 0.0619 | 0.1971 |
| Gemma 3 4B | instruct, chat | 58.14 | 0.4214 | 0.4161 |

The average trends reported in the paper are: instruction tuning contributes +3.7% accuracy but increases ECE by 13.1% and Brier by 6.5%; the chat template adds an additional +1.1% accuracy while further increasing ECE by 2.74% and Brier by 1.5%; the total increase in ECE relative to the base model is 15.8% for both combined.

### Ablation Study
The ownership bias experiment places the same answer in both assistant and user positions, reporting $\Delta=Assistant-User$. On average, assistant framing performed worse across all three confidence methods.

| Confidence Method | Avg ΔECE | Avg ΔBrier | Avg ΔConfidence | Implication |
|------------|-----------|-------------|------------------|------|
| P(True) | 0.098 | 0.088 | 0.158 | Most conservative but still significantly more confident |
| Verbalized Percentage | 0.179 | 0.195 | 0.181 | Deviation persists in percentage confidence |
| Verbalized Linguistic | 0.261 | 0.252 | 0.268 | Largest bias in linguistic level expressions |

A similar trend was observed on GPT-5.2: the ΔECE for P(True), Percentage, and Linguistic were 0.077, 0.087, and 0.113 respectively, while ΔConfidence values were 0.076, 0.112, and 0.222, with all ECE and confidence differences reaching significance.

### Key Findings
- Instruction tuning is the primary cause of calibration degradation; chat templates are not the root cause but further amplify current biases through the assistant role.
- LLMs are more confident in their own answers, regardless of correctness. Higher assistant confidence compared to user confidence was observed even for incorrect answers, reaching up to ~60%.
- In multiple-choice questions, the total confidence of four mutually exclusive options should theoretically be near 100%, but the model's average total confidence consistently exceeds 100%; under assistant framing, it ranges from ~198% to 315%, while under user framing it is ~135% to 243%.
- This phenomenon is not an artifact of MMLU multiple-choice formats. In GSM8K, self-generated answers resulted in up to 19.5% higher confidence and 14.2% higher ECE; in TruthfulQA, the confidence gap reached 10.9%; and in open-ended MMLU, confidence was up to 19.6% higher with 18.1% higher ECE.

## Highlights & Insights
- The most ingenious aspect is using "who said the same answer" as an experimental variable. This control is very clean, separating confidence content factors from conversational role factors.
- The mitigation proposed by the paper is almost zero-cost: instead of asking the model "how sure are you about your previous answer," rewrite the answer as a candidate provided by the user and have the model evaluate it.
- Ownership bias moves in the opposite direction of sycophancy, which is a very insightful discovery. The model doesn't simply pander to the user; it possesses an implicit self-trust in its own generation process.
- The results serve as a reminder that in LLM-as-judge or self-verification scenarios, if a model evaluates its own output, calibration and reliability may be systematically over-optimistic.

## Limitations & Future Work
- The authors acknowledge that most experiments focused on open-weight LLMs. Although GPT-5.2 was supplemented, there is no guarantee that all closed-source models and different post-training recipes behave the same.
- The proposed user-framing mitigation is an inference-time correction that does not change model weights nor solve the root cause of overconfidence generated during RLHF/SFT.
- Evaluation is primarily limited to objective Q&A. For tasks with ambiguous correctness like open-ended generation, creative writing, or legal advice, defining confidence and evaluating calibration will be more difficult.
- Future work could integrate this finding into tool calling, refusal, self-check, and multi-agent debate pipelines: the generator and evaluator should ideally be explicitly decoupled in terms of prompt roles and context.

## Related Work & Insights
- **vs calibration-aware fine-tuning / calibrated reward modeling**: These methods require training or additional models; Ours only modifies prompt framing, offering lower deployment costs but more localized repair capabilities.
- **vs verbalized confidence**: It was previously thought that explicitly asking the model for confidence could mitigate logit calibration issues; this paper shows that verbalized confidence is still affected by ownership bias.
- **vs Sycophancy research**: Sycophancy emphasizes models conforming to user views; this paper finds that in confidence scenarios, models actually trust assistant answers more, indicating that alignment biases manifest in multiple directions.
- **Insights for follow-up work**: When performing self-checks, self-evaluation, or answer re-ranking, candidate answers should be reframed from "my output" to "external candidates," otherwise confidence might be a function of role attribution rather than answer quality.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The controlled experiments for ownership bias are very clear, and the mitigation is simple yet practically valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 open-weight models, 3 benchmarks, 3 confidence methods, additional tasks, and GPT-5.2.
- Writing Quality: ⭐⭐⭐⭐☆ The chain of reasoning is smooth, and while tables are somewhat dense, the main conclusions are very clear.
- Value: ⭐⭐⭐⭐⭐ Provides direct warnings for calibration, self-evaluation, LLM-as-judge, and high-stakes applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compatibility-Aware Dynamic Fine-Tuning for Large Language Models](compatibility-aware_dynamic_fine-tuning_for_large_language_models.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/llm_alignment/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](../../ICML2026/llm_alignment/towards_context-invariant_safety_alignment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
