---
title: >-
  [Paper Note] mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection
description: >-
  [ACL 2026 (SemEval-2026 Task 10)][AIGC Detection][PsyCoMark] The authors adapt the finetuning paradigm of "mdok" (a machine-generated text detector that won PAN@CLEF2025) to conspiracy detection: the training set is expa…
tags:
  - "ACL 2026 (SemEval-2026 Task 10)"
  - "AIGC Detection"
  - "PsyCoMark"
  - "QLoRA"
  - "Self-training"
  - "Data Augmentation"
  - "Qwen3-32B"
date: 2026-05-08
content_hash: 897c6eae97afaed2
---

# mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection

**Conference**: ACL 2026 (SemEval-2026 Task 10)  
**arXiv**: [2605.02712](https://arxiv.org/abs/2605.02712)  
**Code**: https://github.com/kinit-sk/mdok-style-psycomark2026 (Available)  
**Area**: Conspiracy Detection / SemEval / LLM Finetuning  
**Keywords**: PsyCoMark, QLoRA, Self-training, Data Augmentation, Qwen3-32B

## TL;DR
The authors adapt the finetuning paradigm of "mdok" (a machine-generated text detector that won PAN@CLEF2025) to conspiracy detection: the training set is expanded using four types of data augmentation (anonymization / casing / homoglyphs / de-duplication), followed by a round of self-training (retaining only high-confidence pseudo-labels with $p \ge 0.99$ or $p \le 0.01$). By finetuning Qwen3-32B via QLoRA 4-bit PEFT, the system achieved a Macro F1 of 0.78 in SemEval-2026 Task 10 subtask 2, ranking 8/52 (85th percentile).

## Background & Motivation

**Background**: SemEval-2026 Task 10 (PsyCoMark) bridges psychology and NLP, requiring systems to determine whether a Reddit comment expresses conspiracy beliefs. Subtask 2 is a binary English classification (Yes/No). The provided training data is small—1715 positive and 2263 negative examples, with text lengths limited to 160–1000 characters. Early works employed SVM with lexical/stylistic features; subsequent additions of psycholinguistic features (certainty, emotional intensity, distrust framing) yielded improvements. Recently, LLMs have joined the field but struggle with hallucinations and precision issues; emotion-aware finetuned LLMs (e.g., ConspEmoLLM) have pushed performance beyond vanilla LLMs.

**Limitations of Prior Work**: ① The training set is too small for 32B-scale LLMs, leading to overfitting during direct finetuning. ② Conspiracy texts on social media are saturated with noise like URLs, handles, emails, phones, and homoglyphs, which interfere with model tokenizers. ③ Inconsistent casing is common on social media, yet traditional classifiers are often misled by case. ④ Organizers did not release ground truth for the dev/test sets, preventing supervised hyperparameter iteration and requiring "blind" submissions to Codabench.

**Key Challenge**: Achieving SOTA performance via LLM finetuning requires sufficient scale and diversity in training data; however, PsyCoMark is both small and narrow. Furthermore, using large models requires controlling computational costs (given that DeBERTa <0.5B can already achieve 0.75, the gain from 32B might be limited).

**Goal**: To port the successful "mdok" recipe to conspiracy detection with minimal engineering and verify whether an LLM finetuning pipeline used for MGT detection as a source task possesses cross-task transferability.

**Key Insight**: Treat "conspiracy detection" as a robust binary classification task. Borrow three "weapons" from mdok (the 1st place winner at PAN@CLEF2025): anonymization-as-augmentation, homoglyph-aware training, and QLoRA efficient finetuning, then overlay a round of self-training to utilize the unlabeled dev/test sets.

**Core Idea**: mdok-style data augmentation + high-confidence threshold self-training + Qwen3-32B QLoRA = a robust solution ranking in the top 20% of SemEval.

## Method

### Overall Architecture

The pipeline consists of three steps: (1) **Data Augmentation**: The original training set is replicated four times, applying anonymization, lower-casing, upper-casing, and homoglyphication respectively. Only 10% of each type is added to the training pool followed by de-duplication to prevent augmented data from overwhelming the original, resulting in 2126 negative and 1517 positive samples. (2) **Round 1 Training**: Qwen3-32B is finetuned as a binary classification head using QLoRA 4-bit. Hyperparameters include paged AdamW, cosine LR=2e-5, warmup 0.03, batch size 1, and a single epoch. Checkpoints are selected by Macro F1 on a 100×2 holdout validation set. (3) **Self-training**: The Round 1 model performs inference on the unlabeled dev + test sets. Only samples with $p \ge 0.99$ (positive) or $p \le 0.01$ (negative) are retained as silver labels. The model is then retrained from scratch on the merged pool (2575 negative + 1881 positive). During inference, the positive threshold is shifted from 0.5 to 0.7 (_th0.7) to improve precision.

### Key Designs

1.  **mdok-style Data Augmentation**:
    - **Function**: Artificially introduces diversity into small datasets (~4000 samples) to make the model insensitive to surface deformations.
    - **Mechanism**: (i) **Anonymization**: Uses regex to replace emails, @users, and phone numbers with `[EMAIL]`, `[USER]`, and `[PHONE]` (URLs were already replaced by organizers with `[URL]`), flattening identifying tokens. (ii) **Lower-casing & Upper-casing**: Explicitly injects the inductive bias that conspiracy judgment is case-insensitive. (iii) **Homoglyphication**: Replaces characters with look-alikes (e.g., Latin `a` to Cyrillic `а`) to simulate obfuscation tactics and improve robustness to tokenizer-level perturbations. (iv) 10% sampling followed by de-duplication.
    - **Design Motivation**: Similar to MGT detection, conspiracy detection involves numerous spurious surface cues (heavy URL usage, ALL CAPS shouting, intentional character manipulation). If the model learns these as shortcuts, generalization fails. Augmentation converts these into known noise types during training, forcing the model to learn semantics.

2.  **Conservative Self-training**:
    - **Function**: Leverages unlabeled dev/test sets to expand training samples while avoiding error propagation.
    - **Mechanism**: Following the self-training paradigm, a teacher model trained on golden labels generates silver labels on unlabeled sets. This work uses an extremely strict threshold—only samples with $p \ge 0.99$ or $p \le 0.01$ are added. This minimizes the error rate of silver labels. The training set was expanded from 3643 to 4456 samples (+22%).
    - **Design Motivation**: The primary risk of self-training is the compounding of label errors. By using ultra-strict thresholds, this risk is minimized in the first round. This "high precision, low recall" strategy is effective for competitions where hyperparameter tuning is limited.

3.  **QLoRA Finetuning + Threshold Post-processing**:
    - **Function**: Finetunes a 32B model into a functional classifier within ~100 GPU·h (single A100) and balances precision/recall via threshold shifting.
    - **Mechanism**: QLoRA (4-bit quantization + Low-Rank Adapters) updates a small fraction of parameters, allowing a 32B model to run on a single A100 64GB with a `transformers` sequence classification head. Shifting the threshold to 0.7 increased F1 by 1 point (0.77 → 0.78), as the task is more sensitive to precision (the cost of misclassifying a harmless comment as a conspiracy is higher).
    - **Design Motivation**: Full finetuning of 32B models is unrealistic for most academic labs. QLoRA lowers the hardware barrier. Threshold post-processing provides performance gains without retraining.

### Loss & Training
- Binary cross-entropy (default for `transformers` seq-cls) without class weighting.
- Checkpoint selection via Macro F1 on a 100×2 holdout set (since dev labels are unavailable).
- Single-round self-training to prevent cumulative error.
- Base model comparison included Qwen3 (4B / 14B / 32B) and Gemma-3 (1B PT / 12B PT); Qwen3-32B performed best in the dev phase.

## Key Experimental Results

### Main Results (PsyCoMark Subtask 2 Official Test Set, Macro F1)

| System | Macro F1 |
|---|---|
| **Qwen3-32B_ST_th0.7 (Submitted)** | **0.78** |
| Qwen3-32B_ST | 0.77 |
| Qwen3-32B_th0.7 | 0.77 |
| Qwen3-32B | 0.76 |
| DeBERTa-Large (<0.5B) | 0.75 |
| Qwen3-14B-Base | 0.75 |
| Gemma-3-1B-PT_ST | 0.75 |
| Qwen3-4B-Base | 0.75 |
| Gemma-3-12B-PT | 0.74 |
| Gemma-3-1B-PT | 0.73 |
| Qwen3-4B-Base_ST | 0.72 |
| Random baseline | 0.50 |

**Selected Codabench Unofficial Ranking**:

| Rank | Team | Macro F1 |
|---|---|---|
| 1 | NJUST_KMG | 0.89 |
| 2 | AGAI | 0.87 |
| 3 | jia57 | 0.86 |
| 4 | baishanxiaoqi | 0.80 |
| 5 | CSECU-DSG | 0.80 |
| 6 | joccerrillo | 0.79 |
| 7 | qinchihongye | 0.79 |
| **8** | **mdok-style** | **0.78** |

The system ranked 8/52 (85th percentile), 11 points behind the winner (0.89), but outperformed 80%+ of teams.

### Ablation Study (Combined effects of ST and Threshold)

| Configuration | Macro F1 | $\Delta$ vs base |
|---|---|---|
| Qwen3-32B (Vanilla) | 0.76 | – |
| + Self-Training | 0.77 | +0.01 |
| + threshold=0.7 | 0.77 | +0.01 |
| + Self-Training + threshold=0.7 | **0.78** | +0.02 |

Self-training on Qwen3-4B led to a performance drop (0.75→0.72), suggesting that self-training requires high model capacity—small models suffer from noise in silver labels due to underfitting, whereas large models can effectively exchange data for performance.

### Key Findings
- The 32B model with all tricks is only 0.03 higher than DeBERTa-Large (0.5B). **Cost-efficiency Warning**: On small-data binary classification tasks, the marginal utility of large models is limited; DeBERTa remains a suitable first choice.
- Self-training and threshold shifting each contributed ~1 F1 point. The choice of base model was the largest factor (0.73→0.76).
- Even with strict thresholds, self-training provided negative gains for small models (Qwen3-4B, Gemma-3-1B), indicating that silver label reliability is highly correlated with the base model's prediction accuracy.
- Cross-task transfer (MGT → conspiracy) reached the 85th percentile, validating the "finetuning + augmentation + self-training" pipeline as a universal skeleton for binary text classification.

## Highlights & Insights
- Porting anonymization + homoglyphication + casing to conspiracy detection proves that "augmentation against spurious surface cues" is universal for social media text classification (e.g., hate speech, scams).
- The ultra-strict $\ge 0.99 / \le 0.01$ self-training strategy is ideal for competition scenarios without ground truth labels.
- Threshold shifting (0.5→0.7) is a "free lunch" when the task is sensitive to false positives (misjudging conspiracies can be seen as censorship).
- The honest admission that "DeBERTa 0.5B at 0.75 offers better value than 32B at 0.78" is valuable for reproducibility.

## Limitations & Future Work
- Only tested on English PsyCoMark; cross-lingual (Spanish, French, German) effects are unknown.
- Evaluated only Qwen3 and Gemma-3 families; LLoMA-3 / Mistral / DeepSeek were not tested.
- No external datasets (other conspiracy/misinfo corpora) were introduced.
- Only one round of self-training was performed; multi-round iteration with dynamic thresholds was not explored.
- Lack of error analysis (due to hidden ground truth) makes it unclear which linguistic patterns the 32B model struggles with.

## Related Work & Insights
- **vs mdok (Macko 2025)**: The source system won both subtasks in MGT detection; this work proves the pipeline's task-agnostic nature.
- **vs ConspEmoLLM (Liu 2024)**: While they explicitly integrate emotional signals, this work uses prompt-agnostic finetuning with augmentation and self-training to flatten surface variations.
- **vs DeBERTa baseline**: DeBERTa-Large at 0.75 vs Qwen3-32B at 0.78, with the latter costing over 50x in compute—a classic case of diminishing marginal returns for LLMs on small-data tasks.
- **vs SemEval-2024 Task 8 (spiegel-macko-2024-kinit)**: They previously showed 7B LLMs outperfored BERT-like models; this work validates that larger scales (32B) still provide gains, though at a decreasing rate.

## Rating
- Novelty: ⭐⭐ (Mainly cross-task reuse of the mdok recipe)
- Experimental Thoroughness: ⭐⭐⭐ (11 system comparisons + ablation, but lacks multi-round ST and deep error analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and complete reproduction details in standard system paper style)
- Value: ⭐⭐⭐ (Solid report for SemEval; engineering experience with augmentation and strict self-training is practical for competition participants)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)
- [\[ACL 2026\] REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control](reflex_self-refining_explainable_fact-checking_via_verdict-anchored_style_contro.md)
- [\[ICLR 2026\] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](../../ICLR2026/aigc_detection/policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)
- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[ACL 2026\] ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability](exagpt_example-based_machine-generated_text_detection_for_human_interpretability.md)

</div>

<!-- RELATED:END -->
