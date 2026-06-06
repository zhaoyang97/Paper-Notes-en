---
title: >-
  [Paper Note] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection
description: >-
  [ACL 2026 (SemEval workshop)][Social Computing][Polarization Detection] This paper proposes a two-stage pipeline of "structured slot-filling SFT + DPO preference optimization" for the SemEval-2026 POLAR polarization dete…
tags:
  - "ACL 2026 (SemEval workshop)"
  - "Social Computing"
  - "Polarization Detection"
  - "Structured SFT"
  - "DPO"
  - "SemEval-2026"
  - "Qwen2.5 / Mistral-Nemo"
date: 2026-05-08
content_hash: a60bfc42f7d589bb
---

# BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection

**Conference**: ACL 2026 (SemEval workshop)  
**arXiv**: [2604.11121](https://arxiv.org/abs/2604.11121)  
**Code**: https://github.com/atharva7-g/POLAR-SemEval-Submission  
**Area**: Multilingual NLP / SemEval shared task / Polarization Detection  
**Keywords**: Polarization Detection, Structured SFT, DPO, SemEval-2026, Qwen2.5 / Mistral-Nemo

## TL;DR
This paper proposes a two-stage pipeline of "structured slot-filling SFT + DPO preference optimization" for the SemEval-2026 POLAR polarization detection task (English subset). The Qwen2.5-7B system submitted during the competition achieved a 0.7664 Macro-F1. Post-competition, replacing the model with Mistral-Nemo-12B and using preference pairs filtered by an LLM-judge improved the Macro-F1 to 0.8162, surpassing the organizer baseline (0.7802).

## Background & Motivation

**Background**: Online polarization detection is a content moderation task. Traditional approaches include fine-tuning binary classifiers on BERT/DistilBERT, prompt-based ICL, and zero-shot LLM inference. SemEval-2026 Task 9 (POLAR) standardizes this task across multilingual, multicultural, and multi-event scenarios, though this work focuses specifically on the English subset (3,222 train / 160 val / 1,452 test samples).

**Limitations of Prior Work**: (1) Polarized language often contains implicit framing ("I can accept X, but not Y"), which keyword dictionaries fail to capture; (2) labeling costs are high, and classes are imbalanced (~36% positive samples); (3) pure SFT tends to treat the majority class (non-polarized) as a prior, leading to low recall and frequent missed detections; (4) ICL prompts often cause models to output an "overly conservative" 0, where false negatives are more harmful in moderation scenarios (polarized content continues to spread vs. misjudgments that can be reversed by human review).

**Key Challenge**: When optimizing likelihood in SFT, long reasoning chains can dilute the gradient signals of the final-label tokens, sometimes making reasoning SFT perform worse than label-only SFT. However, reasoning is a prerequisite for constructing high-quality DPO preference pairs (a "rationale" is needed to judge quality).

**Goal**: (1) Design a structured rationale schema to produce interpretable, batch-scorable intermediate results; (2) Use DPO to rank false negatives lower than false positives, pushing the decision boundary toward "recall sensitivity"; (3) Explore the gains from LLM-judge filtering of preference pairs.

**Key Insight**: The authors transform polarization detection from "single-label classification" into a "slot-filling generation task"—the model must fill in the target / claim type / 6-item manifestation checklist / decision basis before outputting a label. This provides a "rationale dimension" for DPO comparisons and an auditable paper trail for the LLM judge.

**Core Idea**: A combination of "structured slot-filling generation + DPO ranking across three output types (CORRECT / FP / FN) + LLM judge filtering" is used to convert the classification problem into an RLHF-style decision boundary adjustment problem, specifically targeting false negatives.

## Method

### Overall Architecture
The two-stage pipeline consists of:

- **Stage 1 (Structured SFT)**: Qwen2.5-7B-Instruct (competition) or Mistral-Nemo-Instruct-2407 (post-competition) is fine-tuned via LoRA on a fixed slot-filling template. The model input is social media text, and the output is a JSON-like structure: [Target referenced / Claim type / 6-class Manifestations checklist / Decision basis / Final Answer (0 or 1)]. Training rationales were generated offline by Gemma 3 27B.

- **Stage 2 (DPO Preference Optimization)**: Starting from the SFT checkpoint, a "two-prompt" strategy (one prompt encouraging a 1 prediction, another encouraging 0) and multi-temperature sampling generate a batch of completions. Each completion is labeled as CORRECT, FP, or FN based on the ground truth. They are paired according to the preference $\mathrm{CORRECT} \succ \mathrm{FP} \succ \mathrm{FN}$ (higher rank as chosen, lower rank as rejected) and trained using the DPO loss.

Post-competition enhancements: (1) Rejudged Sonnet—Claude 3.5 Sonnet was used to re-label the training set, changing 6.2% of samples and increasing the overall polarization ratio; (2) DeepSeek-R1 LLM judge was used to filter preference pairs down to a 299-pair balanced set (62:38 FP:FN ratio).

### Key Designs

1. **Structured slot-filling rationale schema**:
    - **Function**: Transforms classification into a generation task that outputs a 6-dimensional manifestation checklist before the label, enabling comparable intermediate products for DPO.
    - **Mechanism**: A fixed output template includes Target / Claim type / 6 Manifestations (Stereotype / Vilification / Dehumanization / Extreme Language / Lack of Empathy / Invalidation) / Decision basis / Final Answer. Training samples generate chain-of-thought based on this template, and labels are extracted via regex after "Final Answer:".
    - **Design Motivation**: Pure label SFT provides only a 0/1 signal, making it impossible to construct fine-grained preference pairs, while free-form CoT has too much variance for effective scoring. The fixed schema enables both "batch LLM-judging" and field-alignment across different completions.

2. **Recall-sensitive asymmetric preference ranking $\mathrm{CORRECT} \succ \mathrm{FP} \succ \mathrm{FN}$**:
    - **Function**: Systematically shifts the model's decision boundary toward more aggressive polarization prediction.
    - **Mechanism**: For each input, completions are generated using two prompts (pro-polar / anti-polar) across multiple temperatures. These are paired according to the partial order for DPO training. Ranking FN after FP reflects that "allowing polarized content to spread" is costlier than "misclassification that can be manually reversed."
    - **Design Motivation**: Directly applying weights to raw SFT was ineffective for class imbalance (as weighted loss did not help in experiments). Adjusting decision boundaries directly from an RL/preference perspective is a more direct approach.

3. **LLM-as-a-judge filtering + Rejudged training data**:
    - **Function**: Addresses the issue where noisy, low-quality preference pairs degrade performance.
    - **Mechanism**: DeepSeek-R1 serves as a judge to score candidate preference pairs "valid/invalid," removing inconsistent reasoning or label mismatches. This filtered 721 candidates down to 299 (62:38 FP:FN). Additionally, Claude 3.5 Sonnet re-judged training labels, correcting 6.2% of labeling errors.
    - **Design Motivation**: Experiments showed that 721 unfiltered pairs resulted in an F1 of 0.7637, lower than the SFT baseline of 0.7795. 330 filtered pairs achieved 0.7889, and 299 R1-filtered pairs reached 0.8162. This proves that preference pair quality is significantly more important than quantity.

### Loss & Training
SFT uses standard causal LM cross-entropy (LoRA rank=8, alpha=16, dropout=0.05, target=q/k/v/o_proj, lr=5e-5, 3-10 epochs). DPO uses the standard preference contrastive loss from Rafailov et al. (2023) with $\beta=0.1$ (competition) or $\beta=0.3$ (best post-competition), lr=5e-6, 2 epochs.

## Key Experimental Results

### Main Results: English Dev + Test Set F1

| Method | English Dev F1 | English Test Macro-F1 | Notes |
|------|---------------|-----------------------|------|
| Zero-shot baseline | 0.7105 | — | No fine-tuning |
| DistilBERT (SLM) | 0.7149 | — | Small model baseline |
| Qwen2.5-7B SFT (reasoning) | 0.738 | — | SFT stage only |
| Qwen2.5-7B SFT + DPO (submitted) | 0.7893 | **0.7664** | Submitted system, ranked 52/60 |
| POLAR organiser baseline | — | 0.7802 | Official baseline |
| Highest-ranked system | — | 0.8252 | Leaderboard top |
| Mistral-Nemo SFT (Rejudged) | — | 0.8097 | Post-comp large model |
| Mistral-Nemo + DPO (β=0.3, R1-filtered) | — | **0.8162** | Final best |

### Ablation Study: Stage 1-Stage 2 Gains & Rationale (English test, n=1,452)

| Configuration | Accuracy | P(1) | R(1) | Macro-F1 |
|------|----------|------|------|----------|
| Label-only SFT | 0.792 | 0.777 | 0.788 | 0.781 |
| Label-only + DPO | 0.720 | 0.618 | 0.625 | 0.699 |
| Reasoning SFT | 0.793 | 0.745 | 0.662 | 0.771 |
| Reasoning + DPO (Ours) | **0.802** | 0.732 | 0.704 | **0.789** |

### DPO Preference Pair Quantity & Quality Ablation

| Preference Pair Config | F1 | # FN | # FP |
|-----------|----|----|----|
| SFT only (No DPO) | 0.7795 | 158 | 137 |
| DPO 330 pairs (filtered) | 0.7889 | 132 | 155 |
| DPO 721 pairs (unfiltered) | 0.7637 | 64 | **274** |

### Key Findings
- **DPO systematically raised recall from 0.5085 to 0.7797** (dev set), at the cost of precision dropping from 0.8333 to 0.7077, matching the "recall-sensitive" design intent.
- **The true value of rationales lies in enabling DPO**: Label-only SFT was the strongest individual stage (0.781) but collapsed to 0.699 when adding DPO. Reasoning SFT was the weakest (0.771) but rose to 0.789 with DPO. This indicates that rationales provide the fine-grained differences needed for DPO.
- **Preference quality > quantity**: 721 unfiltered pairs performed 2.5 F1 points worse than 330 filtered ones, highlighting that data curation is the primary bottleneck in RLHF/DPO.
- **Scaling effects are evident**: Mistral-Nemo 12B benefited significantly more from reasoning SFT than Qwen2.5 7B, suggesting reasoning capability correlates with model scale.

## Highlights & Insights
- **Perspective on "rationale-for-DPO"**: The authors explicitly note that while rationales dilute the final label token gradient during SFT, they provide indispensable fine-grained comparison dimensions in the DPO stage. This insight into "differing roles across two stages" is transferable to any SFT+DPO pipeline.
- **Two-prompt preference generation**: Using both "pro-polar" and "anti-polar" prompts for sampling and ranking covers the decision boundary more systematically than single-prompt temperature sampling.
- **$\beta$ Insensitivity**: Macro-F1 remained between 0.8065 and 0.8162 across nine $\beta$ values (0.1–0.5). This flatness indicates that pair quality is the binding constraint, and $\beta$ tuning offers low returns.

## Limitations & Future Work
- **English Only**: The POLAR benchmark covers 22 languages, but this system does not perform multilingual transfer.
- **Mistral-Nemo Results Unsubmitted**: The post-comp score of 0.8162 is not on the official CodaBench leaderboard, affecting comparability.
- **Dataset Consistency**: Post-comp enhancements simultaneously changed the base model and training labels, making it difficult to decouple their individual contributions.
- **Precision remains low (~0.73)**: The boundary between polarization and "strong but neutral" language remains difficult to distinguish; future work may require external knowledge or retrieval-augmented context.
- **DPO Instability on 7B Models**: Future exploration could involve reference-free preference optimization like SimPO/KTO or loss masking to recover recall loss from reasoning SFT.

## Related Work & Insights
- **vs. SemEval-2019 HatEval**: Traditional multilingual hate speech detection often used BERT classification; this work upgrades the task formulation to generation + decision using LLMs + DPO.
- **vs. Gunel et al. (2021) Supervised Contrastive**: Contrastive learning also improves robustness but requires semantic distance between pairs, which is costly to construct; DPO with ranking is more straightforward.
- **vs. Maggini et al. (2025)**: Similarly found that fine-tuning > ICL for polarization tasks; this work further proves that the "SFT + DPO + LLM-judge" stack is superior.
- **vs. Shi et al. (2025) Key Answer Token Emphasis**: While Shi et al. use reasoning loss masking to protect final label gradients, this paper bypasses the issue via DPO.

## Rating
- Novelty: ⭐⭐⭐ Combines existing techniques (structured generation + DPO + LLM judge) for a specific task; the individual components are not new, but the experimental diagnostics are thorough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Conducted $\beta$ sweeps, pair quality sweeps, structured rationale ablations, and label-only vs. reasoning comparisons.
- Writing Quality: ⭐⭐⭐⭐ Standard SemEval system paper structure, honest about limitations, and clear in presenting findings.
- Value: ⭐⭐⭐ A complete, reproducible pipeline with open-source code; serves as a high-quality template for industrial SFT+DPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)
- [\[ACL 2026\] YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling](yeze_at_semeval-2026_task_9_detecting_multilingual_multicultural_and_multievent_.md)
- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)

</div>

<!-- RELATED:END -->
