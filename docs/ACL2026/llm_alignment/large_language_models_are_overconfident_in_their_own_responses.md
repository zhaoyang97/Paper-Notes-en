---
title: >-
  [Paper Note] Large Language Models Are Overconfident in Their Own Responses
description: >-
  [ACL 2026][Alignment & RLHF][instruction tuning] This paper discovers that instruction-tuned LLMs exhibit a significant ownership bias when evaluating "answers they generated themselves." It proposes a simple inference-time strategy—reframing the answer as user input before asking for confidence—to reduce overconfidence without retraining.
tags:
  - ACL 2026
  - Alignment & RLHF
  - instruction tuning
  - chat template
  - ownership bias
  - inference-time mitigation
date: 2026-05-08
content_hash: e34dea8485fef05e
---
# Large Language Models Are Overconfident in Their Own Responses

**Conference**: ACL2026 Findings  
**arXiv**: [2606.03437](https://arxiv.org/abs/2606.03437)  
**Code**: No public code link found in cache  
**Area**: LLM Alignment / Calibration  
**Keywords**: Confidence calibration, instruction tuning, chat template, ownership bias, inference-time mitigation  

## TL;DR
This paper discovers that instruction-tuned LLMs exhibit a significant ownership bias when evaluating "answers they generated themselves." It proposes a simple inference-time strategy—reframing the answer as user input before asking for confidence—to reduce overconfidence without retraining.

## Background & Motivation
**Background**: Trustworthy LLMs must correctly express uncertainty. Existing studies indicate that the next-token probabilities of base LLMs are often closer to being calibrated than those of instruction-tuned/chat models, whereas post-training like SFT/RLHF may cause models to be overconfident in their answers.

**Limitations of Prior Work**: Much of the previous work has evaluated instruction tuning, chat templates, and verbalized confidence in a confounded manner. This makes it difficult to determine whether miscalibration stems from training algorithms, chat formats, or the role-playing bias generated when the model "acts as an assistant."

**Key Challenge**: While users most commonly interact with instruction-tuned models via chat templates, calibration evaluations often place "generating the answer" and "evaluating the answer" within the same assistant persona. If a model naturally trusts its own output more, confidence levels will change based on the speaker's identity even if the answer text is identical.

**Goal**: The authors aim to answer four questions: What are the respective impacts of instruction tuning and chat templates on calibration? Does explicit confidence elicitation change the trends? Are models more confident in their own answers? Can inference-time strategies without weight updates mitigate this bias?

**Key Insight**: The paper decomposes the answer provider into two prompt framings: assistant and user. If the same answer receives higher confidence and worse ECE/Brier scores under the assistant framing, it indicates that the issue is not just the content of the answer, but the model's ownership bias toward "its own output."

**Core Idea**: By reframing a model-generated answer as a user message and then asking for confidence, the model is switched from being the "author" to being an "observer," thereby reducing self-confirming overconfidence.

## Method
Instead of training a new calibrator, the paper conducts a series of controlled experiments to locate the mechanisms of miscalibration and proposes an inference-time prompt framing strategy. The core mechanism involves decoupling the model version, chat template, confidence elicitation method, and answer source identity.

### Overall Architecture
First, the authors compare the base model, instruction-tuned model without chat template, and instruction-tuned model with chat template for each model family on MMLU, using logit-based confidence to calculate accuracy, ECE, and Brier score. Second, they introduce three explicit confidence elicitations: P(True), Verbalized Percentage, and Verbalized Linguistic, to test if the calibration harm from instruction tuning persists. Third, they fix the answer text and only change whether the answer appears in an assistant message or a user message, measuring the differences in ECE, Brier, and raw confidence. Finally, they use "answer as user input" as an inference-time mitigation strategy and verify generalization across MMLU, GSM8K, TruthfulQA, open-ended MMLU, and GPT-5.2.

### Key Designs

**1. Decoupling instruction tuning and chat templates: Distinguishing whether calibration worsening is due to training algorithms or chat formats.**

Most prior work evaluated instruct models only under chat templates, confounding the impacts of post-training and prompt formatting. The authors compare three invocation methods side-by-side within the same model family: base model, instruct model without chat template, and instruct model with chat template, all using logit-based confidence. The logic is straightforward: if the "instruct without chat" stage is already significantly miscalibrated, the root cause lies primarily in instruction tuning rather than the chat format; the chat template merely adds another layer of influence. This separates the two factors into independently observable variables.

**2. Three confidence elicitations: Confirming that miscalibration is not an artifact of logit-based measurement.**

The logits of instruction-tuned models might not be suitable for direct interpretation as confidence. To address potential concerns regarding the metric itself, the authors use three alternative ways to ask for confidence: P(True), a 0–100% Verbalized Percentage, and a seven-point Verbalized Linguistic scale (linearly mapped to equidistant scores from 0 to 1). If the confidence expressed in natural language is similarly impaired, it confirms that miscalibration is a model-level issue rather than a byproduct of a specific reading method.

**3. Assistant-vs-user ownership bias test: Directly verifying if models trust "their own" answers more.**

To prove the issue concerns "who said it" rather than the content, the content must be fixed while the identity is varied. For the same question and candidate answer, the authors only change whether it appears in an assistant message or a user message before asking for confidence. The difference is defined as $\Delta = Assistant - User$, where a positive value implies the assistant framing is more confident or less calibrated. This design also refutes the opposing hypothesis: if sycophancy were dominant, the model should trust user-provided answers more (negative $\Delta$). The experiments showed a positive trend, supporting ownership bias—an implicit self-trust in the model's own generation process. This design directly leads to the mitigation strategy: reframing model answers as user messages to switch the persona from "author" to "observer."

### Loss & Training
No new models were trained. Evaluation metrics include accuracy, ECE, and Brier score. ECE uses 10 equal-width confidence bins, and the Brier score uses the mean squared error between probabilistic predictions and binary correctness labels. For statistical significance, Wilcoxon signed-rank tests are used for Brier and raw confidence, and paired bootstrap resampling tests with $K=1000$ are used for ECE; significant differences are marked as $p<0.01$.

## Key Experimental Results

### Main Results
The first set of experiments shows that instruction tuning improves accuracy but damages calibration, which is further exacerbated by chat templates. Selected logit-based results for MMLU are listed below.

| Model | Setting | Accuracy | ECE | Brier |
|------|------|----------|-----|-------|
| Llama 3.1 8B | base, no chat | 62.81 | 0.0664 | 0.1706 |
| Llama 3.1 8B | instruct, chat | 69.12 | 0.1666 | 0.2005 |
| Qwen3 4B | base, no chat | 67.72 | 0.0425 | 0.1709 |
| Qwen3 4B | instruct, chat | 72.98 | 0.2415 | 0.2455 |
| Gemma 3 4B | base, no chat | 49.47 | 0.0619 | 0.1971 |
| Gemma 3 4B | instruct, chat | 58.14 | 0.4214 | 0.4161 |

The average trends reported are: instruction tuning brings +3.7% accuracy but increases ECE by 13.1% and Brier by 6.5%; chat templates bring an additional +1.1% accuracy while increasing ECE by 2.74% and Brier by 1.5%. The total ECE increase relative to base models is 15.8%.

### Ablation Study
The ownership bias experiment places the same answer in assistant and user positions and reports $\Delta=Assistant-User$. On average, assistant framing performs worse across all three confidence methods.

| Confidence Method | Avg. ΔECE | Avg. ΔBrier | Avg. ΔConfidence | Meaning |
|------------|-----------|-------------|------------------|------|
| P(True) | 0.098 | 0.088 | 0.158 | Most conservative but still noticeably more confident |
| Verbalized Percentage | 0.179 | 0.195 | 0.181 | Percentage confidence also shows bias |
| Verbalized Linguistic | 0.261 | 0.252 | 0.268 | Linguistic scale shows the largest bias |

The same trend was observed on GPT-5.2: $\Delta ECE$ for P(True), Percentage, and Linguistic were 0.077, 0.087, and 0.113, respectively, with $\Delta Confidence$ at 0.076, 0.112, and 0.222. All ECE and confidence differences were significant.

### Key Findings
- Instruction tuning is the primary cause of calibration degradation; chat templates are not the root cause but amplify the bias through the assistant role.
- LLMs are more confident in their own answers regardless of whether the answers are correct. Higher assistant confidence over user confidence is observed even for incorrect answers, reaching up to ~60%.
- In multiple-choice questions, the total confidence of four mutually exclusive options should ideally be near 100%, but the average total confidence consistently exceeds 100%. This ranges from 198% to 315% under assistant framing and 135% to 243% under user framing.
- The phenomenon is not an MMLU-specific artifact. In GSM8K, self-generated answers lead to up to 19.5% higher confidence and 14.2% higher ECE; TruthfulQA shows a confidence gap up to 10.9%; open-ended MMLU shows up to 19.6% higher confidence and 18.1% higher ECE.

## Highlights & Insights
- The most ingenious aspect is using "who said the same answer" as an experimental variable. This control is very clean, effectively decoupling confidence content factors from conversational role factors.
- The proposed mitigation is nearly zero-cost: instead of asking the model "how certain are you about your previous answer," reframe the answer as a user-provided candidate and let the model evaluate it.
- The discovery that ownership bias moves in the opposite direction of sycophancy is insightful. The model does not simply cater to the user's viewpoint; rather, it possesses an implicit self-trust in its own generation process.
- These results remind us that in LLM-as-judge or self-verification scenarios, letting a model evaluate its own output may lead to systematically optimistic calibration and reliability.

## Limitations & Future Work
- The authors acknowledge that most experiments focused on open-weight LLMs. While supplemented with GPT-5.2, results might vary across all closed-source models and different post-training recipes.
- The proposed user-framing mitigation is an inference-time correction that does not change model weights or address the root cause of overconfidence emerging during RLHF/SFT.
- Evaluations were primarily limited to objective QA. For tasks with ambiguous correctness like open-ended generation, creative writing, or legal advice, defining confidence and evaluating calibration will be more difficult.
- Future work could integrate this finding into tool-use, abstention, self-checking, and multi-agent debate workflows: generators and evaluators should ideally be explicitly decoupled in terms of prompt roles and context.

## Related Work & Insights
- **vs calibration-aware fine-tuning / calibrated reward modeling**: These methods require training or additional models; the proposed method only changes prompt framing, offering lower deployment costs but more localized fixes.
- **vs verbalized confidence**: It was previously thought that explicitly asking for verbalized confidence could alleviate logit calibration issues; this paper shows that verbalized confidence is still affected by ownership bias.
- **vs sycophancy research**: Sycophancy emphasizes models conforming to user views; this paper finds that in confidence scenarios, models trust the assistant's own answers more, indicating alignment biases can move in multiple directions.
- **Insights for follow-up work**: When performing self-check, self-evaluation, or answer re-ranking, candidate answers should be reframed from "my output" to "external candidates," otherwise confidence might be a function of role attribution rather than answer quality.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The controlled experiment for ownership bias is clear, and the mitigation is simple yet practically valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 open-weight models, 3 benchmarks, 3 confidence methods, additional tasks, and GPT-5.2.
- Writing Quality: ⭐⭐⭐⭐☆ The logical chain is smooth, and although tables are dense, the main conclusions are very clear.
- Value: ⭐⭐⭐⭐⭐ Provides direct warnings for calibration, self-evaluation, LLM-as-judge, and high-stakes applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/llm_alignment/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](../../ICML2026/llm_alignment/towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](../../AAAI2026/llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)

</div>

<!-- RELATED:END -->
