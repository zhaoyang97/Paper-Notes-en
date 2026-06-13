---
title: >-
  [Paper Note] LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification
description: >-
  [ACL 2026 / BioNLP 2026][Medical NLP][Psychological Defense Mechanisms] This PsyDefDetect entry improves the official macro F1 for psychological defense mechanism 9-class classification to 0.3917—ranking 4th out of 21 te…
tags:
  - "ACL 2026 / BioNLP 2026"
  - "Medical NLP"
  - "Psychological Defense Mechanisms"
  - "Class Imbalance"
  - "QLoRA"
  - "Grouped Cross-Validation"
  - "Post-processing Calibration"
date: 2026-05-08
content_hash: 6a773abbee4f559b
---

# LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification

**Conference**: ACL 2026 / BioNLP 2026  
**arXiv**: [2606.00647](https://arxiv.org/abs/2606.00647)  
**Code**: https://github.com/Shefwef/LingIUTics-PsyDefDetect-BIONLP26  
**Area**: Clinical NLP / Mental Health Text Classification / Class Imbalance Learning  
**Keywords**: Psychological Defense Mechanisms, Class Imbalance, QLoRA, Grouped Cross-Validation, Post-processing Calibration

## TL;DR
This PsyDefDetect entry improves the official macro F1 for psychological defense mechanism 9-class classification to 0.3917—ranking 4th out of 21 teams—by employing Qwen3-8B QLoRA, minority lexical augmentation, grouped 5-fold cross-validation, OOF logit bias, and multi-seed ensemble.

## Background & Motivation
**Background**: The PsyDefDetect 2026 task requires classifying seeker utterances in psychological support dialogues into nine psychological defense levels under the DMRS framework. This task is valuable for clinical NLP and mental health dialogue systems, as defense mechanisms reflect how users handle stress, anxiety, and conflict.

**Limitations of Prior Work**: The data is extremely imbalanced. The merged training set provided in the paper contains 1,864 samples, where Level 7 High-Adaptive accounts for 51.9%, while Level 8 Unclear represents only 1.5%. The ratio between Level 7 and Level 8 is approximately 34.6:1. Since the official metric is macro F1, optimizing for accuracy alone causes the model to collapse toward majority classes.

**Key Challenge**: Small encoders and zero-shot LLMs struggle to identify rare psychological defense categories. Direct fine-tuning of large models often leads to poor leaderboard generalization due to class imbalance and validation leakage risks. The system must simultaneously address model capacity, minority class recall, validation reliability, and post-processing calibration.

**Goal**: The authors aim to build a minority-friendly Qwen3-8B fine-tuning system that obtains reliable OOF signals without dialogue group leakage and recovers minority class recall through post-processing.

**Key Insight**: The paper adopts an iterative engineering path. After initial attempts with MentalBERT, MentalRoBERTa, DeBERTa, RoBERTa, and zero-shot LLMs showed zero or extremely low performance on rare classes, the focus shifted to Qwen3-8B QLoRA. This was incrementally enhanced with weighted CE, label smoothing, round-robin augmentation, grouped CV, logit bias, and ensembles.

**Core Idea**: in extreme long-tail clinical text classification, model capacity is merely a necessary condition. The factors that truly determine leaderboard performance are leakage-safe validation, minority data construction, and macro F1-oriented post-processing calibration.

## Method
The system consists of three main stages: data preprocessing with minority augmentation, two sets of grouped 5-fold QLoRA training, and OOF calibration with multi-seed probability fusion. Each component targets an early failure mode: insufficient encoder capacity, poor single-fold generalization, excessive attraction to Level 7, and near-zero recall for rare classes like Level 8.

### Overall Architecture
The input comprises three parts: the DMRS Label Guide, the most recent 30 rounds of dialogue context, and output instructions. The model outputs integer labels from 0 to 8. The training data combines the PsyDefConv train and validation sets, totaling 1,864 samples, with 472 test samples; the total number of source dialogues is 200. The authors use dialogue_id for grouped stratified 5-fold validation to ensure that the same dialogue and its augmented samples do not cross folds.

At the model level, Qwen3-8B is used as the base model, fine-tuned via QLoRA after 4-bit NF4 quantization. For inference post-processing, class-specific logit biases are searched based on Anchor OOF predictions. Then, the test probabilities from the Anchor and Seed-A 5-fold models are fused at a 30/70 ratio. A majority class protection gate with $\tau_7=0.69$ determines whether to apply minority rerouting.

### Key Designs
1. **Long-tail Oriented Qwen3-8B QLoRA Architecture**:
    - **Function**: Provides sufficient semantic capacity to distinguish between clinically similar psychological defense categories.
    - **Mechanism**: 4-bit NF4 + double quantization reduces the peak VRAM of Qwen3-8B from ~32GB to ~8GB. LoRA is applied to q/k/v/o/gate/up/down/score with rank/alpha of 128/256 and 0.1 dropout. Trainable parameters are ~31M (0.4%).
    - **Design Motivation**: BERT-family encoders achieved a maximum validation macro F1 of only 0.314, with Classes 3, 5, and 8 often being 0. Larger generative models provide stronger contextual understanding, but PEFT is required to control hardware costs.

2. **Round-robin Minority Lexical Augmentation and Grouped CV**:
    - **Function**: Increases minority class coverage while preventing augmented samples from leaking into different folds.
    - **Mechanism**: $k=3$ round-robin lexical mutation is applied to Levels 2, 3, 4, 5, and 8. Patterns include contraction + hedging, style shift + filler, and hesitation markers. Only the seeker utterance is modified; context remains unchanged. Minority samples increased from 28-84 to 65-252.
    - **Design Motivation**: Defense mechanism labels depend on subtle psychological signals in the utterance; excessive paraphrasing would destroy these labels. Grouped CV ensures source dialogues and augmented samples stay in the same fold, resulting in 0 leaked dialogues.

3. **OOF Bias, Seed-A Fusion, and $\tau_7$ Protection Gate**:
    - **Function**: Recovers minority recall without sacrificing majority class precision.
    - **Mechanism**: Approximately 22,000 bias vectors are randomly searched on OOF predictions using the rule $\hat{y}=\arg\max_c[\log p_c+\delta_c]$. Setting $\delta_7<0$ suppresses the majority class while $\delta_8>0$ boosts "Unclear." At test time, $p_{blend}=0.30p_{anchor}+0.70p_{seedA}$ is used. If $p_{blend,7}\geq0.69$, Level 7 is locked; otherwise, bias rerouting is applied.
    - **Design Motivation**: Raw probabilities still skew toward Level 7. Macro F1 requires minority recall, but forced rerouting hurts the majority class. The protection gate separates "certain majority class" samples from "ambiguous samples."

### Loss & Training
Training utilizes inverse-square-root class weighting, where $w_c=(1/\sqrt{n_c})/\sum_i(1/\sqrt{n_i})$ (e.g., $w_8=1.67$, $w_5=1.29$, $w_7=0.28$). Label smoothing with $\epsilon=0.05$ is used to prevent premature logit saturation for Level 7. The optimizer is AdamW with a learning rate of $1.2\times10^{-4}$, 0.01 weight decay, cosine annealing, and 8% warmup. Training configuration includes per-device batch size 2, gradient accumulation 8 (effective batch size 16), gradient clip 0.3, 10 epochs per fold, 1024 max length, and bf16 on an NVIDIA RTX 3090 Ti 24GB.

## Key Experimental Results

### Main Results
The final system achieved a macro F1 of 0.3917 on the official positive-class leaderboard, ranking 4/21. This represents a +7.7 absolute point improvement (+24.4% relative gain) over the Ministral-8B fine-tuned baseline (31.48 macro F1) reported in the task paper.

| System | Acc. (%) | Macro F1 (%) |
|------|----------|--------------|
| GPT-5 zero-shot (task paper) | 52.75 | 19.53 |
| Gemini 2.5 Pro zero-shot | 56.36 | 25.99 |
| DeepSeek-V3.2 zero-shot (CoT) | 55.72 | 26.17 |
| Llama 3.1-8B fine-tuned | 62.92 | 30.51 |
| InternLM3-8B fine-tuned | 63.98 | 30.53 |
| Ministral-8B fine-tuned (SOTA) | 64.83 | 31.48 |
| Qwen3-8B LoRA baseline | 54.45 | 24.91 |
| Qwen3-8B LoRA + grouped CV + bias tuning | 58.43 | 35.48 |
| Qwen3-8B LoRA + SeedA ensemble + v2decode | 64.19 | 39.17 |

### Ablation Study
Ablations show that no single component is a silver bullet; the combination leads to stable improvement from 0.249 to 0.392.

| Configuration | Macro F1 | Description |
|------|----------|------|
| R0: 1-fold, rr=64, no weighting | 0.249 | Early Qwen3-8B baseline |
| + 5-fold CV, rr=128 | 0.284 | Increased LoRA rank and introduced 5-fold |
| + Weighted CE + label smoothing | 0.329 | Suppressed majority class collapse |
| + Grouped-clean 5-fold | 0.355 | Dialogue-level grouping, reduced OOF-LB gap |
| + Data augmentation (RR-k3) | 0.355 | No direct metric increase, but stabilized minority classes |
| + Seed-A blend (30/70) + v2 decode | 0.392 | Final submission strategy |

### Key Findings
- The OOF macro F1 for the grouped-clean augmented run was 0.3716, with individual fold macro F1s of 0.3804, 0.3701, 0.3899, 0.3553, and 0.3326.
- In per-class OOF analysis, Level 8 "Unclear" improved from near-zero to F1=0.797 through augmentation and bias tuning, while Level 7 High-Adaptive maintained F1=0.709.
- The blended system's aggregate per-label macro metrics were precision 0.431, recall 0.436, and F1 0.426; official positive macro F1 was 0.3917.
- Level 4 Minor Image-Distorting and Level 5 Neurotic remain challenging (F1 ~0.254 and ~0.278), likely due to high linguistic overlap with majority classes.
- Grouped CV reduced the OOF-leaderboard gap from 9.6 points to 1.7–4.5 points, making post-processing threshold tuning more reliable.

## Highlights & Insights
- The core of this system is not "winning by switching models," but rather integrating validation, augmentation, loss, and decoding for long-tail classification. The Qwen3-8B baseline (24.91 macro F1) was significantly elevated by grouped CV, weighted loss, and post-processing.
- Round-robin lexical augmentation is conservative, only altering the surface form of the seeker utterance to preserve context and psychological signals. This is more stable for clinical NLP than aggressive paraphrasing.
- The $\tau_7$ gate is a practical engineering design: it avoids forced calibration when the model is highly certain of the majority class, only applying logit bias rerouting when confidence is low.
- The use of a complete run log (R0 to R10) makes the system reproducible and helps readers understand the failures that drove subsequent designs.

## Limitations & Future Work
- The OOF bias vectors and decoding rules are calibrated specifically for the PsyDefDetect dataset; migrating to new domains would require re-estimation.
- Grouped CV reduces augmentation leakage but cannot entirely eliminate generalization risks caused by similar dialogue topics or templates.
- Hardware constraints limited exploration to 8B-scale PEFT; larger models or instruction-tuned clinical LLMs were not tested.
- Data augmentation relied on surface lexical transforms. Future work could explore more reliable paraphrase augmentation or label-preserving dialogue context augmentation.
- Boundaries for mechanisms like Level 4/5 remain blurred, potentially requiring expert knowledge, more granular label descriptions, or ordinal/hierarchy-aware loss functions.

## Related Work & Insights
- **vs BERT-family encoders**: MentalBERT, MentalRoBERTa, DeBERTa, and RoBERTa hit capacity bottlenecks on rare classes. This work uses Qwen3-8B for superior contextual understanding.
- **vs zero-shot LLMs**: Qwen3-8B, Llama 3.1-8B, and Ministral-8B zero-shot only reached ~8-16% macro F1, indicating that task definition prompts alone are insufficient to learn DMRS labels.
- **vs standard cross-entropy fine-tuning**: While fine-tuned Ministral-8B reached 64.71% accuracy, its macro F1 was only 14.74, showing that accuracy is a misleading metric in long-tail psychological classification.
- **Insight for future work**: Similar medical or mental health tasks can benefit from the "grouped CV + conservative minority augmentation + OOF bias + majority gate" paradigm over simply pursuing larger models.

## Rating
- Novelty: ⭐⭐⭐ System engineering innovations are more prominent than algorithmic ones, but the integrated design effectively addresses task pain points.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete run logs, ablations, per-class analysis, and official comparisons.
- Writing Quality: ⭐⭐⭐⭐ Iterative process is clear; tables are information-dense and practical.
- Value: ⭐⭐⭐⭐ Highly relevant for long-tail clinical NLP shared tasks and low-VRAM QLoRA entry systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_nlp/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[ACL 2026\] Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)
- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)

</div>

<!-- RELATED:END -->
