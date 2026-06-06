---
title: >-
  [Paper Note] ADVICE: Answer-Dependent Verbalized Confidence Estimation
description: >-
  [ACL 2026][LLM Safety][verbalized confidence] This paper diagnoses the root cause of LLM verbalized overconfidence as "confidence hardly depends on the generated answer" via JSD and attribution analysis. It proposes ADVI…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "verbalized confidence"
  - "calibration"
  - "answer-grounded"
  - "contrastive fine-tuning"
  - "overconfidence"
date: 2026-05-08
content_hash: 2bc7b563c299bd61
---

# ADVICE: Answer-Dependent Verbalized Confidence Estimation

**Conference**: ACL 2026  
**arXiv**: [2510.10913](https://arxiv.org/abs/2510.10913)  
**Code**: None  
**Area**: LLM Evaluation / Confidence Calibration  
**Keywords**: verbalized confidence, calibration, answer-grounded, contrastive fine-tuning, overconfidence

## TL;DR
This paper diagnoses the root cause of LLM verbalized overconfidence as "confidence hardly depends on the generated answer" via JSD and attribution analysis. It proposes ADVICE, a lightweight fine-tuning framework using contrastive answer pairs. By employing JSD, Margin, and Sum losses, it forces the confidence distribution for correct answers to be significantly higher than for incorrect ones. While maintaining task accuracy, it reduces the ECE of Gemma2-9b on TriviaQA from 21.9% to 6.2%.

## Background & Motivation

**Background**: The trustworthy use of LLMs increasingly relies on "letting the model output its own confidence" (verbalized confidence), which involves outputting an answer along with an integer (0-9), a letter (A-E), or a percentage (0-100) as a score. This paradigm is more universal than post-hoc logit calibration, friendly to black-box APIs, and has been popularized by works such as those by Lin, Tian, and Xiong.

**Limitations of Prior Work**: In practice, LLMs almost always provide extremely high scores (10/10 or A), regardless of whether the answer is correct, resulting in ECE values generally between 20% and 50%. Existing mitigation methods fall into three categories—prompt engineering, self-consistency sampling, and fine-tuning targeting token probability re-fitting like ConfTuner—all of which focus on "how to suppress confidence," while few analyze "why it is overconfident."

**Key Challenge**: Through two sets of diagnostic experiments, the authors found the root cause to be "answer independence." First, fixing the question $q$, they had the model score 30 different candidate answers $a_i$ and calculated the Jensen-Shannon Divergence between distributions $P_M(C\mid q,a_i)$. They found JSD highly concentrated near 0 ($\mathrm{JSD}\le0.1$ for most samples), meaning changing the answer rarely changes the confidence. Second, using Attention Rollout and Integrated Gradients to measure the attention flow and gradient attribution from "confidence tokens to answer tokens," they found the weight of this path to be much lower than "answer → question" or "confidence → question," and even lower than meaningless tokens like BOS.

**Goal**: To make "confidence truly conditional on the answer" as the optimization objective, rather than broadly reducing the maximum value of the output distribution, thereby avoiding degradation in calibration, generalization, and task accuracy.

**Key Insight**: Organize training samples into triplets $(q,a_{\text{correct}},a_{\text{wrong}})$ and use contrastive loss to force the model to provide distinguishable and directionally correct confidence distributions for the same question with different answers—this directly addresses the root cause of "answer independence."

**Core Idea**: "Do not directly fit confidence values; instead, directly fit the sensitivity of $P(C\mid q,a)$ to the answer."

## Method

### Overall Architecture
ADVICE is a contrastive fine-tuning framework based on LoRA. The process: (1) Sample 4,000 questions from the TriviaQA training set, keeping only those the model answers correctly via greedy decoding; (2) For each question, use random sampling to obtain a "semantically plausible but factually incorrect" hard negative answer $a_{\text{wrong}}$, paired with the original $a_{\text{correct}}$ to form a triplet; (3) Generate training data in both ScoreLetter and ScoreNumber formats; (4) For each triplet, simultaneously calculate LM loss (to maintain QA capability) + JSD loss (to separate confidence distributions) + Margin loss (for correct direction) + Sum loss (for absolute constraint), fine-tuning for 4 epochs using AdamW and LoRA ($rank=16, \alpha=32$, applied to Q/K/V/O); (5) During inference, the format can generalize to unseen ScoreText/ScoreFloat/ScorePercent.

### Key Designs

1.  **Diagnostic-driven Answer Independence Metric**:
    - **Function**: Quantifies "whether confidence depends on the answer" into an optimizable training signal.
    - **Mechanism**: For each question $q$ and candidate answer $a$, the output probability of the confidence token is re-projected onto a fixed set of discrete values $C$ (e.g., 0-9) to obtain the distribution $P_M(C\mid q,a)$. The JSD of all $(a_i, a_j)$ pairs in the training set is compared as direct evidence of "answer sensitivity." Attribution from "confidence token → answer token" is verified using Attention Rollout (recursive aggregation $0.5W_{\text{att}}+0.5I$ across layers) and Integrated Gradients ($n\_steps=1024$).
    - **Design Motivation**: Previous studies attributed overconfidence to vague explanations like "high-frequency high scores in training data" or "RLHF preference for optimistic answers." This work provides a measurable root cause—only by explicitly incorporating this independence metric into the training objective can calibration be improved from the source, rather than just suppressing the maximum probability post-hoc.

2.  **Triplet Contrastive Training Objective**:
    - **Function**: Forces the model to output directionally correct, distinguishable, and normalized confidence distributions for correct/incorrect answers while maintaining the original answer generation capability.
    - **Mechanism**: Four losses are defined. $\mathcal{L}_{\mathrm{LM}}$ is the NLL of $a_{\text{correct}}$ to preserve QA capability; $\mathcal{L}_{\mathrm{JSD}}=\max(0,\delta_{\mathrm{JSD}}-D_{\mathrm{JSD}}(P_{\text{correct}}\Vert P_{\text{wrong}}))$ forces the divergence between distributions to be at least $\delta_{\mathrm{JSD}}=0.6$ (near the $\ln 2\approx 0.693$ upper bound); $\mathcal{L}_{\mathrm{Margin}}=\max(0,\delta_{\mathrm{Margin}}-(\mu_{\text{correct}}-\mu_{\text{wrong}}))$ forces the expected confidence of the correct answer to be $\delta_{\mathrm{Margin}}=1$ higher than the incorrect one; $\mathcal{L}_{\mathrm{Sum}}=|1-(\mu_{\text{correct}}+\mu_{\text{wrong}})|$ forces their sum to be approximately 1, corresponding to the semantics that "when accuracy is approximately 1, the corresponding answer should be almost fully credible." Total loss $\mathcal{L}=\mathcal{L}_{\mathrm{LM}}+\mathcal{L}_{\mathrm{JSD}}+\mathcal{L}_{\mathrm{Margin}}+\mathcal{L}_{\mathrm{Sum}}$ (all coefficients set to 1).
    - **Design Motivation**: JSD alone only ensures "different distributions" but the direction might be reversed; Margin alone lacks control over the overall distribution shape, leading to both distributions being too high; the Sum term hardcodes the definition of "confidence = probability of correctness" into the loss, preventing the model from becoming overall conservative after training. All three are indispensable (see Ablation).

3.  **Format Generalization and Hard Negative Sampling in Training Set Construction**:
    - **Function**: Allows a single fine-tuning to adapt to multiple inference formats and improves the ability to distinguish similar but incorrect answers.
    - **Mechanism**: Hard negatives are obtained via top-$p$ sampling from the original model, being "semantically plausible and contextually relevant but factually incorrect" (e.g., for the state where Mike Tyson's 1998 boxing license was issued, correct is Nevada, hard negative is California). Training uses only ScoreLetter (E/D/C/B/A mapped to 0.1-0.9) and ScoreNumber (0-9 mapped to $i/9$) formats, while inference can extend to ScoreText, ScoreFloat, and ScorePercent. A single representation format is enforced within each mini-batch to stabilize optimization.
    - **Design Motivation**: Hard negatives avoid the training bias of "incorrect answers being completely absurd → differentiation being too easy." Multi-format training + single-format batches ensure both generalization and gradient stability, allowing ADVICE to transfer to unseen formats like ScorePercent/ScoreFloat (see Table 4, Gemma2's ECE on TriviaQA-Float drops from 27.5 to 6.2).

### Loss & Training
The four losses are added with equal weights; LoRA rank=16, $\alpha=32$, applied to Q/K/V/O projections; AdamW with 5% warmup and linear decay, lr is $3\times 10^{-5}$ for Mistral and $1\times 10^{-5}$ for others; batch size 16, gradient accumulation 2, 4 epochs; single H200 GPU.

## Key Experimental Results

### Main Results
Evaluation on TriviaQA (in-distribution) + MMLU, LogiQA (OOD) comparing Default, Prompting, Self-Consistency, and ConfTuner across Llama-3.1-8B, Mistral-7B-v0.3, and Gemma-2-9b. Main metrics: ECE/|NCE|/Brier/AUROC (lower is better for the first three).

| Model | Dataset | Metric | Default | ConfTuner | ADVICE | ADVICE+ConfTuner |
|------|--------|------|---------|-----------|--------|------------------|
| Gemma2-9b | TriviaQA | ECE↓ | 21.9 | 5.7 | 6.2 | 3.4 |
| Gemma2-9b | TriviaQA | AUROC↑ | 52.7 | 82.7 | 77.4 | 77.1 |
| Gemma2-9b | MMLU | ECE↓ | 21.0 | 11.0 | 5.6 | 11.7 |
| Gemma2-9b | LogiQA | ECE↓ | 39.1 | 18.4 | 11.9 | 8.0 |
| Llama3.1-8B | TriviaQA | ECE↓ | 16.9 | 5.2 | 10.4 | 9.4 |
| Llama3.1-8B | MMLU | ECE↓ | 26.9 | 13.9 | 8.6 | 9.6 |

ADVICE outperformed ConfTuner in 19 out of 24 OOD comparisons, and ADVICE+ConfTuner generally further reduced ECE, indicating complementarity.

### Ablation Study
Ablation of training objectives for Gemma2-9b on TriviaQA / MMLU (ECE↓):

| Training Objective | TriviaQA ECE | MMLU ECE | Note |
|----------|--------------|----------|------|
| LM Only | 23.0 | 22.5 | Equivalent to untuned |
| LM+JSD | 8.6 | 13.2 | Separates dist. but no direction |
| LM+Margin | 16.8 | 21.9 | Correct direction but coarse dist. |
| LM+Sum | 21.1 | 19.5 | Only sum constraint is ineffective |
| LM+JSD+Margin | 11.0 | 14.3 | Lacks Sum, weak OOD |
| LM+JSD+Sum | 15.3 | 7.5 | Lacks direction, ID rebound |
| ADVICE (Full) | **6.2** | **5.6** | All three terms are essential |

### Key Findings
- The "answer mask experiment" is most critical: After replacing $a$ with a `<pad>` of equal length, the Default confidence distribution remained concentrated in the A/9 range (showing confidence is nearly independent of the answer), while ADVICE shifted the probability mass significantly towards E/0/1 ("don't know"), proving ADVICE truly learned dependency on the answer.
- The ranking of answer tokens (e.g., `_Exile`) among the Top-K IG attribution tokens rose from outside the top 10 at the start of training to the top 5, quantitatively verifying that ADVICE strengthens answer sensitivity at the attention/gradient levels.
- ADVICE is optimal in both performance and efficiency regarding token budget (Fig. 7): Self-consistency methods sampled 5 times more without significant ECE improvement; ADVICE achieves calibration comparable to ConfTuner with a single inference and the lowest token budget.

## Highlights & Insights
- "Diagnose before prescribing" research paradigm: First use three independent tools (JSD + Attention Rollout + IG) to quantify the root cause of "answer independence," then have the loss function directly target this cause. This method is far more interpretable than stuffing all black-box symptoms into a KL regression.
- $\mathcal{L}_{\mathrm{Sum}}=|1-(\mu_{\text{correct}}+\mu_{\text{wrong}})|$ is a significantly undervalued design: It directly transforms the definition "confidence corresponds to correct probability" into a constraint, preventing all answers from being suppressed (conservative bias) after training.
- The "answer mask counterfactual" in the experiments is a rare causal validation paradigm in verbalized confidence research, transferable to any trustworthy task where "output should be based on a segment of input" (e.g., attribution, citation, tool call selection).

## Limitations & Future Work
- Verified only on short-answer QA and multiple-choice questions; in long-context/complex reasoning (e.g., multi-hop, agents), answer token boundaries and the semantics of "confidence corresponding to answer" are not yet clear.
- Coupling of calibration metrics and task accuracy: On tasks with >90% accuracy like SciQ, even Default appears well-calibrated (Appendix Table 7 shows ADVICE is slightly worse), suggesting the need for harder benchmarks to truly distinguish calibration methods.
- Training relies on model-generated hard negatives; construction cost grows linearly with model scale. For low-capability models, "plausible but wrong" hard negatives might be unavailable, limiting the scope.

## Related Work & Insights
- **vs ConfTuner**: ConfTuner directly aligns token probability distributions to the Brier score, which is "numerical fitting"; ADVICE uses contrastive triplets to directly constrain "sensitivity to answers," which is "mechanism alignment." The 19/24 advantage on OOD + complementarity with ConfTuner shows they target different dimensions of overconfidence.
- **vs Self-Consistency**: Self-Consistency uses the weighted average of 5 samples to approximate true confidence, increasing token costs by 5x with limited ECE improvement; ADVICE requires only a single inference.
- **vs Prompting/RLHF Calibration**: Those methods do not explain "why overconfidence occurs"; by attributing the root cause to answer independence, future RLHF reward designs can incorporate preferences like "changing the answer must change the confidence."

## Rating
- Novelty: ⭐⭐⭐⭐ The "answer independence" root cause diagnosis and corresponding contrastive loss are genuinely new mechanisms, though conceptually related to process supervision/contrastive learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 3 datasets × 5 formats + 5 ablation items + answer mask counterfactual + IG tracking; rare density.
- Writing Quality: ⭐⭐⭐⭐ The three-stage logic of Diagnosis-Method-Validation is clear, with formulas and takeaway paragraphs well-highlighted.
- Value: ⭐⭐⭐⭐ Directly helpful for trusting black-box LLM deployments (medical/legal) and is plug-and-play as it is orthogonal to methods like ConfTuner.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency](illusions_of_confidence_diagnosing_llm_truthfulness_via_neighborhood_consistency.md)
- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[NeurIPS 2025\] Probabilistic Reasoning with LLMs for K-Anonymity Estimation](../../NeurIPS2025/llm_safety/probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)

</div>

<!-- RELATED:END -->
