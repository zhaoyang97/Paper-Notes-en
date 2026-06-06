---
title: >-
  [Paper Note] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection
description: >-
  [ACL 2026 (SemEval-2026 Task 9 system paper)][Social Computing][Polarization Detection] The mdok system, originally designed for multilingual machine-generated text detection (QLoRA fine-tuned Qwen3-32B / Gemma-3-27B)…
tags:
  - "ACL 2026 (SemEval-2026 Task 9 system paper)"
  - "Social Computing"
  - "Polarization Detection"
  - "Multilingual"
  - "QLoRA"
  - "Data Augmentation"
  - "Homoglyph Attack"
date: 2026-05-08
content_hash: 1d55f1bf72684bb3
---

# mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection

**Conference**: ACL 2026 (SemEval-2026 Task 9 system paper)  
**arXiv**: [2605.02695](https://arxiv.org/abs/2605.02695)  
**Code**: https://github.com/kinit-sk/mdok-style-polar2026  
**Area**: Multilingual NLP / Text Classification / SemEval System Paper  
**Keywords**: Polarization Detection, Multilingual, QLoRA, Data Augmentation, Homoglyph Attack

## TL;DR
The mdok system, originally designed for multilingual machine-generated text detection (QLoRA fine-tuned Qwen3-32B / Gemma-3-27B), is transferred to the SemEval-2026 Task 9 multilingual polarization detection task. By stacking four types of data augmentation—anonymization, case variation, and homoglyphs—the system achieves a Macro-F1 across 22 languages that is 3–4% higher on average than the official baseline.

## Background & Motivation
**Background**: Online polarization is a precursor to hate speech and social fragmentation; automated detection is key to mitigation. SemEval-2026 Task 9 (POLAR) divides this into three subtasks: subtask 1 (binary classification: polarized or not), subtask 2 (polarization type: political / racial / religious / gender / other), and subtask 3 (manifestation: stereotype / vilification / dehumanization / extreme language / lack of empathy / invalidation), covering 22 languages including many low-resource ones such as Amharic, Hausa, and Odia.

**Limitations of Prior Work**: Many of the 22 languages suffer from scarce training data and class imbalance. Small BERT-like models struggle with both low-resource scenarios and cross-lingual transfer. Furthermore, social media text is rife with noise like inconsistent casing, emojis, @mentions, and contact info. Malicious actors also use **homoglyph attacks** (replacing ASCII with visually similar characters from different Unicode blocks) to break tokenizers and bypass classifiers.

**Key Challenge**: (1) The need for a unified model to cover 22 languages and share signals for low-resource languages, requiring a model large enough for multilingual capacity; (2) The need for robustness against visual obfuscation despite a lack of such samples in the training data.

**Goal**: Port the mdok pipeline (which achieved two 1st places in machine-generated text detection at PAN@CLEF 2025), consisting of QLoRA fine-tuned medium-sized multilingual LLMs and robustness enhancements, to polarization detection to verify the hypothesis that "robust detectors are task-agnostic toolboxes."

**Key Insight**: Although polarization detection and machine-generated text detection differ on the surface, they both fundamentally involve "robust sequence classification for short texts." Robustness techniques, particularly homoglyph defense, should be transferable.

**Core Idea**: The mdok paradigm = QLoRA 4-bit fine-tuning of 27–32B multilingual LLMs + four "dual-style" text augmentations (original + augmented versions each making up half the training set), merging data from all 22 languages for training to enhance cross-lingual transfer.

## Method

### Overall Architecture
The system takes social media text in any language as input and outputs binary logits for subtask 1, or multi-labels for subtask 2 and 3. The pipeline follows four steps: (1) merging and deduplicating the train+dev sets across 22 languages; (2) applying four types of data augmentation (anonymization, lowercase, uppercase, and homoglyphs) to each text, expanding the training set by ~20%; (3) performing single-epoch fine-tuning on the merged multilingual data using QLoRA; (4) selecting the best checkpoint based on validation AUC (subtask 1) or Macro-F1 (subtask 2/3). Qwen3-32B was chosen as the backbone for subtask 1 (supporting 100+ languages), while Gemma-3-27B-pt (supporting 140+ languages) was used for subtask 2/3 due to Qwen3's instability during multi-label head training.

### Key Designs

1.  **Dual Data Augmentation**:
    - **Function**: To enable the model to learn "semantic invariance" without external data—meaning the same text should yield the same polarization label after surface perturbations.
    - **Mechanism**: For each original training sample $x$, an augmented copy $T(x)$ is generated. Both $(x, y)$ and $(T(x), y)$ are fed to the model, with global deduplication used to remove copies identical to the original. The four $T$ transformations include: (a) **Anonymization**—replacing identified PII with `[EMAIL]/[USER]/[PHONE]` to reduce overfitting to specific usernames; (b) **Lowercase**; (c) **Uppercase**; (d) **Homoglyphs**—replacing some ASCII characters with visually identical characters from different Unicode blocks (e.g., Greek small 'a' for Latin 'a'). In subtask 1, each of the four augmentations adds 5%, expanding the training set by 20% in total.
    - **Design Motivation**: Homoglyph attacks were the most disruptive adversarial strategy in the authors' previous work on machine-generated text detection (as they cause BPE/SentencePiece tokenizers to segment words into entirely different subword sequences). Treating them as training-time regularization rather than test-time defense improves both robustness and surface stability simultaneously.

2.  **22-Language Merged + QLoRA Single Model**:
    - **Function**: To process 22 languages with a single 32B model instead of training a classifier head for each language, maximizing knowledge sharing for low-resource languages.
    - **Mechanism**: Training data from all languages are mixed line-by-line. The base LLM is fine-tuned using 4-bit QLoRA with a constant learning rate of $2 \times 10^{-5}$, a warmup ratio of 0.03, paged AdamW, and batch size 1. Validation is performed every 500 steps on a balanced sample of 4400 dev sets, completing within a single epoch. For subtasks 2/3 (multi-label), the LM head is replaced with a sigmoid multi-output head using BCE loss.
    - **Design Motivation**: Monolingual samples for low-resource languages (e.g., Hausa, Khmer) are insufficient for fine-tuning a 32B model. Pooling all languages exposes the model to more polarized samples and forces it to learn "task-based rather than language-based" representations. Qwen3-32B and Gemma-3-27B were selected for their native support for 100+ languages, eliminating the need for additional cross-lingual transfer methods.

3.  **Appraisal Annotation as Interpretable Alternative Path (Exploratory)**:
    - **Function**: To explore a lightweight path using "cognitive appraisal dimensions of emotion" (pleasantness, predictability, controllability, consequences for self/others, value alignment, etc.) as features, external to the main system.
    - **Mechanism**: Text is encoded with XLM-Roberta. A multi-task regression head predicts 5 binary appraisal dimensions + 4 event description dimensions simultaneously, with a weighted loss of MSE + BCE (architecture adapted from AppraisePLM). LogisticRegression (default hyperparameters, random state 42, 80/20 split) is then used for polarization classification based on appraisal features, with one model trained per language.
    - **Design Motivation**: Appraisal dimensions provide interpretable cognitive signals explaining "why this text is polarized." Although their standalone Macro-F1 is near random, the AUC for each label (threshold-independent) shows sufficient discriminative power, suggesting future potential for integrating appraisal signals into the main model as additional features.

### Loss & Training
Subtask 1 uses standard cross-entropy; subtasks 2/3 use BCE-with-logits for multi-label classification. QLoRA 4-bit quantization loads the base model weights via `bitsandbytes`, with LoRA adapters on all attention and MLP projections. Training is restricted to a single epoch to prevent overfitting to surface cues. Checkpoint selection: subtask 1 uses dev AUC; subtasks 2/3 use dev Macro-F1.

## Key Experimental Results

### Main Results (Submitted system's Macro-F1 across 22 languages and delta vs. official baseline)

| Language (subset) | Subtask 1 | Subtask 2 | Subtask 3 | Delta vs. baseline (S1/S2/S3) |
|---|---|---|---|---|
| zho (Chinese) | 0.9237 | 0.8199 | 0.4912 | +0.055 / +0.150 / +0.491 |
| nep (Nepali) | 0.8915 | 0.8026 | 0.5669 | +0.012 / +0.081 / +0.436 |
| tel (Telugu) | 0.8818 | 0.2573 | 0.2143 | +0.238 / -0.057 / -0.460 |
| mya (Burmese) | 0.8788 | 0.6835 | — | +0.058 / +0.206 / — |
| ben (Bengali) | 0.8415 | 0.3050 | 0.1272 | -0.011 / +0.016 / +0.040 |
| eng (English) | 0.8058 | 0.4519 | 0.3697 | +0.026 / +0.119 / -0.040 |
| hau (Hausa) | 0.7401 | 0.1689 | 0.0000 | -0.035 / -0.035 / -0.746 |
| khm (Khmer) | 0.6293 | 0.6323 | 0.2482 | -0.030 / +0.005 / -0.361 |
| amh (Amharic) | 0.6619 | 0.5116 | 0.4310 | -0.053 / +0.140 / -0.012 |
| **22 Lang Avg vs baseline** | **+0.033** | **+0.043** | **−0.001** | — |

Overall, performance in subtasks 1 and 2 exceeded the baseline by 3–4% on average, while subtask 3 remained on par (severely dragged down by a 0.0 in Hausa). In terms of rank percentile, 14 out of the 62 (22×3) subtasks reached the top 20%, and 28 reached the top 50%. The system ranked 1st overall in Italian subtask 1, 3rd in Nepali subtask 2, and 4th in Urdu subtask 3.

### Ablation Study / Analysis (Fine-grained performance by label, selected)

| Subtask / Class | Chinese (zho) | English (eng) | Hindi (hin) | Avg. Hardest Class |
|---|---|---|---|---|
| S2 Political | 0.8571 | 0.8014 | 0.8019 | Political reaches 0.74+ overall |
| S2 Religious | 0.9651 | 0.7535 | 0.9214 | Religious is easiest to distinguish |
| S2 "Other" | 0.8294 | 0.5194 | 0.6536 | **Hardest** (avg ~0.58) |
| S3 Vilification | 0.8696 | 0.7821 | 0.7407 | — |
| S3 Dehumanization | 0.7958 | 0.5391 | 0.7130 | **One of the hardest** |
| S3 Lack of empathy | 0.5491 | 0.5742 | 0.6554 | **Hardest** (lowest mean) |
| S3 Invalidation | 0.5937 | 0.4894 | 0.7480 | Second hardest |

### Key Findings
- **Subtask difficulty varies greatly**: Average performance for subtask 1 is ~0.79, subtask 2 is ~0.53, and subtask 3 is ~0.36. Multi-label classification remains the primary bottleneck; the original mdok was tuned for binary classification, and a simple head change was insufficient.
- **The "Other" category is the Achilles' heel of subtask 2** (avg. low of ~0.58) because it acts as a "garbage bin" class (anything not political/racial/religious/gender), making it semantically inconsistent. Similarly, subtask 3's "dehumanization / lack of empathy / invalidation" categories are significantly more difficult than vilification or extreme language, which often have explicit lexical markers.
- **Significant language disparities**: Chinese, Nepali, and Burmese performed best overall (>0.85 in S1); Amharic, Hausa, and Khmer performed worst (<0.67). The authors note that the 0.0 in Hausa subtask 3 resulted from the merged training causing the model to treat Hausa as an outlier.
- **Although the appraisal path's Macro-F1 is near random, per-label AUC is consistently >0.65** (e.g., Chinese vilification 0.7896, English invalidation 0.6667), indicating that appraisal signals do capture cognitive cues related to polarization. Fusing these into the main model holds potential.

## Highlights & Insights
- **Homoglyph attacks as training augmentation** is an undervalued trick: instead of deploying Unicode normalization at test time (which loses visual signals used by adversaries), it is better to train the model to recognize that "replaced characters" represent the same word. One training pass grants both **robustness and surface invariance** with zero extra inference overhead.
- **The "task-language trade-off" is absorbed by 27–32B model scales**: When a model is sufficiently large and natively supports 100+ languages, data from 22 languages can be merged for training without special cross-lingual losses or adapters. This suggests that for low-resource tasks, "scaling the backbone" may offer a higher ROI than "adding tricks."
- **Zero-effort migration from text detection to polarization detection** is the most educational aspect of this system paper: it treats mdok as a "tool" rather than a "model," proving that a robust sequence classification pipeline is intrinsically task-agnostic.

## Limitations & Future Work
- The authors admit only a few base models were tested; there may be better choices for certain low-resource languages. Additionally, training only used official train+dev sets without external multilingual polarization data.
- Noted limitations: (1) Merging 22 languages negatively impacts extremely low-resource languages (Hausa / Khmer); fine-tuning per language family might be a better future direction. (2) The multi-label head for subtask 3 lacked specialized design; adding label-correlation modeling (e.g., Asymmetric Loss / Tail-aware sampling) is recommended. (3) The appraisal path remains parallel to rather than integrated with the main system.
- Future improvements: Joint multi-task training of the appraisal head as an auxiliary head; expanding homoglyph augmentation to other social media noise types like emojis or character repetition; utilizing self-training with unlabelled multilingual social media text.

## Related Work & Insights
- **vs. traditional BERT/XLM-R fine-tuning**: The authors previously verified that 7B LLMs outperformed small BERT-like models in SemEval-2024 Task 8; this work pushes that to 27–32B, emphasizing that "PEFT (QLoRA) makes the cost manageable" for mid-sized laboratories.
- **vs. monolingual independent classifiers**: Traditional multilingual competitions often train one model per language. This paper proves a unified model is feasible in the LLM era, yielding better results for medium-resource languages like Chinese and Hindi, though at the expense of very low-resource languages.
- **vs. Data Augmentation**: Compared to EDA or back-translation, dual-style augmentation (original + augmented + global deduplication) is more lightweight, and using homoglyph attacks on the training side is a rare but effective technique in the text robustness community.

## Rating
- Novelty: ⭐⭐☆☆☆ The main contribution is transferring the existing mdok pipeline to a new task; technical innovation is limited, but homoglyph training augmentation and 22-language merging are practical tricks.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Full runs across 22 languages and 3 subtasks were completed with fine-grained per-label analysis, though ablation studies for individual contributions (homoglyphs vs. anonymization) are missing.
- Writing Quality: ⭐⭐⭐☆☆ Standard system paper style; the method is clear, but some key details are missing (e.g., LoRA rank, target modules).
- Value: ⭐⭐⭐☆☆ Directly reusable for engineering teams in SemEval or those performing multilingual text classification; academic contribution is moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling](yeze_at_semeval-2026_task_9_detecting_multilingual_multicultural_and_multievent_.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)

</div>

<!-- RELATED:END -->
