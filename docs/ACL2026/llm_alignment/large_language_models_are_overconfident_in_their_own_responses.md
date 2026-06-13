---
title: >-
  [Paper Note] Large Language Models Are Overconfident in Their Own Responses
description: >-
  [ACL2026][LLM Alignment][Confidence calibration] This paper discovers that instruction-tuned LLMs exhibit a significant ownership bias when evaluating "answers they generated themselves." It proposes a simple inference-t…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Confidence calibration"
  - "instruction tuning"
  - "chat template"
  - "ownership bias"
  - "inference-time mitigation"
date: 2026-05-08
content_hash: fc70b0cd4adb0d30
---

# Large Language Models Are Overconfident in Their Own Responses

**Conference**: ACL2026 Findings  
**arXiv**: [2606.03437](https://arxiv.org/abs/2606.03437)  
**Code**: No public code link found in cache  
**Area**: LLM Alignment / Calibration  
**Keywords**: Confidence calibration, instruction tuning, chat template, ownership bias, inference-time mitigation  

## TL;DR
This paper discovers that instruction-tuned LLMs exhibit a significant ownership bias when evaluating "answers they generated themselves." It proposes a simple inference-time strategy—re-framing the answer as user input before querying confidence—to reduce overconfidence without retraining.

## Background & Motivation
**Background**: Trustworthy LLMs must express uncertainty correctly. Prior research indicates that base LLM next-token probabilities are often better calibrated than instruction-tuned/chat models, whereas post-training like SFT/RLHF can lead to overconfidence.

**Limitations of Prior Work**: Many previous works conflate instruction tuning, chat templates, and verbalized confidence in evaluations, making it difficult to discern whether miscalibration stems from training algorithms, chat formats, or the model's persona bias when "playing the assistant."

**Key Challenge**: Users typically interact with the instruction-tuned + chat template form, but calibration evaluations often place "answer generation" and "answer evaluation" within the same assistant persona. If models naturally trust their own output more, confidence levels will shift based on the speaker's identity even if the answer text is identical.

**Goal**: The authors aim to answer four questions: the individual impacts of instruction tuning and chat templates on calibration; whether explicit confidence elicitation changes these trends; whether models are more confident in their own answers; and whether inference-time strategies can mitigate this bias without weight updates.

**Key Insight**: The paper decouples the answer provider into assistant and user prompt framings. If the same answer yields higher confidence and worse ECE/Brier under assistant framing, it indicates that the issue is not just answer content, but an ownership bias toward the "self" output.

**Core Idea**: By reframing model-generated answers as user messages before querying confidence, the model shifts from "author" to "observer," thereby reducing self-confirmatory overconfidence.

## Method
Instead of training a new calibrator, the paper conducts controlled experiments to locate the mechanisms of miscalibration and proposes a prompt framing strategy. The core mechanism involves decoupling model versions, chat templates, confidence elicitation methods, and the identity of the answer source.

### Overall Architecture
First, the authors compare the base model, instruction-tuned model (without chat template), and instruction-tuned model (with chat template) for each model family on MMLU, using logit-based confidence to calculate Accuracy, ECE, and Brier score. Second, they introduce three explicit confidence elicitations: P(True), Verbalized Percentage, and Verbalized Linguistic, to test if the negative impact of instruction tuning persists. Third, they fix the answer text and only change whether the answer appears in an assistant or user message to measure the differences in ECE, Brier, and raw confidence. Finally, the "answer as user input" strategy is validated as an inference-time mitigation on MMLU, GSM8K, TruthfulQA, open-ended MMLU, and GPT-5.2.

### Key Designs
1.  **Decoupling instruction tuning and chat templates**:
    - **Function**: Distinguish the contributions of post-training and chat formats to calibration.
    - **Mechanism**: Compare base, instruct-without-chat, and instruct-with-chat calls within the same family; if without-chat is already significantly miscalibrated, the root cause lies primarily in instruction tuning.
    - **Design Motivation**: Previous work mostly evaluated instruct models under chat templates, easily misattributing training effects to prompt formats.

2.  **Three confidence elicitations**:
    - **Function**: Confirm that miscalibration is not merely a measurement artifact of logit probabilities.
    - **Mechanism**: Use P(True), 0-100% Verbalized Percentage, and 7-point Verbalized Linguistic confidence (mapping linguistic categories to equidistant scores from 0 to 1).
    - **Design Motivation**: Instruction-tuned logits might not be suitable for direct interpretation as confidence, necessitating checks on natural language confidence expressions.

3.  **Assistant-vs-user ownership bias test**:
    - **Function**: Directly test if the model trusts its "own" answers more.
    - **Mechanism**: For the same question and candidate answer, only change whether the answer is in an assistant or user message before querying confidence. The difference is defined as $\Delta = Assistant - User$, where positive values indicate higher overconfidence in assistant framing.
    - **Design Motivation**: If sycophancy dominated, the model would trust user answers more; the experiment finds the opposite trend, supporting the ownership bias explanation.

### Loss & Training
Ours does not involve training new models. Evaluation metrics include Accuracy, ECE, and Brier score. ECE uses 10 equal-width confidence bins, and Brier score represents the mean squared error between probability predictions and binary correctness labels. For statistical significance, Brier and raw confidence use the Wilcoxon signed-rank test, and ECE uses paired bootstrap resampling with $K=1000$; significant differences are marked at $p < 0.01$.

## Key Experimental Results

### Main Results
The first set of experiments shows that instruction tuning improves accuracy but damages calibration, and the chat template further exacerbates this. Below are partial logit-based results on MMLU:

| Model | Setting | Accuracy | ECE | Brier |
| :--- | :--- | :--- | :--- | :--- |
| Llama 3.1 8B | base, no chat | 62.81 | 0.0664 | 0.1706 |
| Llama 3.1 8B | instruct, chat | 69.12 | 0.1666 | 0.2005 |
| Qwen3 4B | base, no chat | 67.72 | 0.0425 | 0.1709 |
| Qwen3 4B | instruct, chat | 72.98 | 0.2415 | 0.2455 |
| Gemma 3 4B | base, no chat | 49.47 | 0.0619 | 0.1971 |
| Gemma 3 4B | instruct, chat | 58.14 | 0.4214 | 0.4161 |

The average trend observed is: instruction tuning provides $+3.7\%$ accuracy but increases ECE by $13.1\%$ and Brier by $6.5\%$; the chat template adds an extra $+1.1\%$ accuracy while ECE increases by $2.74\%$ and Brier by $1.5\%$. The total ECE increase relative to base models is $15.8\%$.

### Ablation Study
The ownership bias experiment places the same answer in assistant or user positions and reports $\Delta = Assistant - User$. On average, assistant framing performs worse across all three confidence methods.

| Confidence Method | Avg. ΔECE | Avg. ΔBrier | Avg. ΔConfidence | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| P(True) | 0.098 | 0.088 | 0.158 | Most conservative but still clearly overconfident |
| Verbalized Percentage | 0.179 | 0.195 | 0.181 | Percentage confidence also biased |
| Verbalized Linguistic | 0.261 | 0.252 | 0.268 | Linguistic levels show largest bias |

The same trend is observed on GPT-5.2: $\Delta$ECE for P(True), Percentage, and Linguistic are $0.077$, $0.087$, and $0.113$ respectively, with significant differences in both ECE and confidence.

### Key Findings
- Instruction tuning is the primary cause of calibration decay; chat templates are not the root cause but amplify bias via the assistant role.
- LLMs are more confident in their own answers regardless of correctness. Higher confidence in assistant framing (up to 60%) is observed even for incorrect answers.
- In multiple-choice questions, the total confidence of mutually exclusive options should theoretically be near 100%, but the average total confidence of models always exceeds 100%; ranging from $198\%$ to $315\%$ in assistant framing and $135\%$ to $243\%$ in user framing.
- The phenomenon is not an MMLU artifact. In GSM8K, self-generated answers lead to up to $19.5\%$ higher confidence; TruthfulQA shows a gap up to $10.9\%$; open-ended MMLU shows up to $19.6\%$ higher confidence and $18.1\%$ higher ECE.

## Highlights & Insights
- The most ingenious point is using "who said the same answer" as an experimental variable. This clean control isolates content factors from conversational role factors.
- The proposed mitigation is virtually free: instead of asking "how sure are you of your answer," reframe the answer as a candidate provided by the user and let the model evaluate it.
- Ownership bias acts in the opposite direction of sycophancy. Models are not simply catering to the user but possess an implicit self-trust in their generation process.
- The results serve as a reminder for LLM-as-judge or self-verification scenarios: if a model evaluates its own output, calibration and trustworthiness may be systematically over-optimistic.

## Limitations & Future Work
- The authors acknowledge that most experiments focus on open-weight LLMs; while GPT-5.2 was added, it cannot be guaranteed that all closed-source models follow identical recipes.
- The user-framing mitigation is an inference-time fix that does not change weights or address the root causes of overconfidence arising during RLHF/SFT.
- Evaluations are limited to objective Q&A. For tasks with ambiguous correctness (e.g., creative writing, legal opinions), defining confidence and evaluating calibration is more difficult.
- Future work could integrate these findings into tool-use, abstention, self-checking, and multi-agent debate workflows: generators and evaluators should ideally be decoupled in prompt roles and contexts.

## Related Work & Insights
- **vs Calibration-aware fine-tuning / calibrated reward modeling**: These methods require training or external models; Ours modifies only prompt framing, lowering deployment cost but providing more localized fixes.
- **vs Verbalized confidence**: It was previously thought that verbalized confidence could mitigate logit calibration issues; this paper shows verbalized forms are still subject to ownership bias.
- **vs Sycophancy research**: Sycophancy emphasizes models conforming to user views; this paper finds that in confidence scenarios, models trust the assistant's own answers more, indicating alignment biases can move in multiple directions.
- **Inspiration for future work**: When performing self-checking or answer reranking, candidate answers should be transformed from "my output" to "external candidates," otherwise confidence may function as a role attribution rather than a quality metric.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The controlled experiment for ownership bias is very clear; mitigation is simple but practically valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 open-weight models, 3 benchmarks, 3 confidence methods, additional tasks, and GPT-5.2.
- Writing Quality: ⭐⭐⭐⭐☆ The chain of reasoning is smooth, and while tables are dense, main conclusions are explicit.
- Value: ⭐⭐⭐⭐⭐ Directly relevant to calibration, self-evaluation, LLM-as-judge, and high-stakes applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization](topology-enhanced_alignment_for_large_language_models_trajectory_topology_loss_a.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/llm_alignment/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](../../AAAI2026/llm_alignment/align_to_structure_aligning_large_language_models_with_struc.md)

</div>

<!-- RELATED:END -->
