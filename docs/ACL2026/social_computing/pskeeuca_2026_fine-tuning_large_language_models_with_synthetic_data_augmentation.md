---
title: >-
  [Paper Note] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat
description: >-
  [ACL2026][Social Computing][Game Chat Moderation] This system paper addresses the EEUCA 2026 gaming chat toxicity identification task. By employing Llama 3.1 8B with LoRA and 5% strictly filtered synthetic paraphrased da…
tags:
  - "ACL2026"
  - "Social Computing"
  - "Game Chat Moderation"
  - "Toxicity Classification"
  - "Synthetic Data Augmentation"
  - "LoRA Fine-Tuning"
  - "Class Imbalance"
date: 2026-05-08
content_hash: b8a8f99bff3cc0e4
---

# PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat

**Conference**: ACL2026  
**arXiv**: [2605.07201](https://arxiv.org/abs/2605.07201)  
**Code**: Not open-sourced  
**Area**: Social Computing / Gaming Community Toxicity Detection  
**Keywords**: Game Chat Moderation, Toxicity Classification, Synthetic Data Augmentation, LoRA Fine-Tuning, Class Imbalance

## TL;DR
This system paper addresses the EEUCA 2026 gaming chat toxicity identification task. By employing Llama 3.1 8B with LoRA and 5% strictly filtered synthetic paraphrased data for minority classes, the system achieved a macro-F1 of 0.6234 across six categories. The study reveals a "validation trap" where high validation scores lead to poor distribution transfer on test sets.

## Background & Motivation
**Background**: Online gaming community chat moderation is typically modeled as a text classification problem. Prevailing solutions range from fine-tuning encoders like XLM-RoBERTa to parameter-efficient fine-tuning (PEFT) of instruction-tuned LLMs, as well as ensemble or hierarchical classification. The EEUCA 2026 shared task categorizes World of Tanks chat messages into six classes: Non-toxic, Insults/Flaming, Other Offensive, Hate/Harassment, Threats, and Extremism, evaluated by macro-F1.

**Limitations of Prior Work**: The challenge extends beyond detecting "profanity." Data is heavily skewed: 81% is Non-toxic, while Threats and Extremism combined account for less than 0.2%. Chat text is short, colloquial, contains gaming jargon, and involves multi-language mixing. Semantic boundaries between Insults, Other Offensive, and Hate/Harassment are narrow, often leading models to confuse skill-based mockery with identity-based attacks.

**Key Challenge**: The validation set distribution and annotation patterns do not fully align with the test set. Models that fit the majority class proportions of the validation set appear stable but remain overly conservative on minority classes in the test set. The authors term this phenomenon the "validation trap": high validation F1 scores often stem from "under-predicting minority classes" rather than better generalization.

**Goal**: In a resource-constrained shared-task scenario, develop a toxicity classification system that transfers stably to the test set, while analyzing which designs cause validation illusions and what minimal augmentations improve minority class recall without overfitting.

**Key Insight**: Instead of blindly increasing synthetic data volume, synthetic samples are restricted to paraphrases of real minority messages, with systematic scanning of synthesis ratios. The hypothesis is that real minority samples are scarce, but their surface expressions can be slightly expanded; excessive synthesis causes the model to learn the generator's style.

**Core Idea**: Use small doses of synonymous, paraphrased synthetic minority samples to calibrate the LLM’s class bias. This encourages the model to actively identify difficult minority classes on the test set while avoiding distribution drift caused by large-scale synthetic data.

## Method
As a high-quality shared-task system report, the contribution lies not in a complex new architecture but in systematic comparisons, synthesis ratio control, and error pattern analysis. The final system uses Llama 3.1 8B as the backbone with 4-bit NF4 quantization and LoRA training, including explicit class definitions in the prompt and a minimal percentage of synthetic minority samples.

### Overall Architecture
Input consists of a World of Tanks chat message, and the output is one of six toxicity categories. The workflow begins by analyzing class distributions and repetition patterns in raw data. After partitioning the real training set, synthetic data is added only to the training partition, maintaining 100% real samples in the validation set. On the model side, instruction prompts provide definitions for the six categories. Llama 3.1 8B is fine-tuned via LoRA. Experiments compare encoders, Gemma, Llama, hierarchical classification, one-vs-rest, transfer learning, ensembles, and post-calibration, selecting the Llama 8B + 5% synthetic configuration for its superior test set transfer.

### Key Designs
1. **Short prompt + clearly defined LLM classifier**:
    - **Function**: Places class definitions directly before the input to provide instruction-tuned LLMs with clearer decision boundaries between similar toxic categories.
    - **Mechanism**: The prompt lists brief definitions from `0=Non-toxic` to `5=Extremism`, followed by `Message: [input text]`. Short prompts are used to preserve class semantics while avoiding token limit exhaustion (384 tokens).
    - **Design Motivation**: Game chats are inherently short; verbose instructions might clutter the training objective. However, omitting class explanations leads to confusion between Insults, Other Offensive, and Hate/Harassment.

2. **Minority class paraphrase synthetic data**:
    - **Function**: Enhances training signals for Class 2/3/4/5, which are rare or easily confused.
    - **Mechanism**: Authors use an LLM to perform semantics-preserving paraphrasing of real minority messages rather than generating toxic sentences from scratch. The synthetic pool contains 10,464 filtered paraphrases (Class 2: 8,348; Class 3: 1,633; Class 4: 343; Class 5: 140). Only 1,921 samples (4.998% of training data) are included in final training.
    - **Design Motivation**: Initial direct generation produces toxic samples that are too generic and lack game-specific context. Paraphrasing retains real context and slang style, making it more suitable for augmenting rare classes.

3. **Validation-trap-driven ratio selection**:
    - **Function**: Prevents the model from appearing better on validation sets while performing worse on test sets.
    - **Mechanism**: Ratios of 0%, 2%, 3%, 5%, 7%, 10%, and 15% were scanned alongside test prediction distributions. 5% synthetic data reduced Non-toxic predictions from 79.6% to 79.0% and increased Class 2 from 4.9% to 5.7%, better matching the sensitivity required for the test set.
    - **Design Motivation**: The validation set is not a reliable proxy for final generalization in this task. Small-scale synthesis adjusts decision boundaries, while large-scale synthesis causes overfitting to synthetic styles.

### Loss & Training
The final model uses Llama 3.1 8B with 4-bit NF4 quantization, LoRA rank=16, alpha=64, dropout=0.0. Training settings: 5e-5 learning rate, cosine schedule, 4 epochs, batch size 4, and gradient accumulation 4. The sequence length is 384 tokens. The objective is class-weighted cross-entropy to mitigate severe imbalance. Other methods tested but not utilized in the final version include hierarchical classification, one-vs-rest, DOTA 2 transfer learning, ensemble (averaging/voting), and post-calibration methods like Platt scaling, isotonic regression, and temperature scaling.

## Key Experimental Results

### Main Results

| System | Val F1 | Test F1 | Remarks |
| :--- | :--- | :--- | :--- |
| XLM-RoBERTa Large | 0.30 | - | Full encoder fine-tuning, weak performance |
| Gemma 2B | 0.63 | 0.52 | High validation but poor test transfer |
| Gemma 12B | 0.66 | 0.52 | Typical validation trap |
| Two-stage Hierarchical | 0.67 | 0.47 | Largest generalization gap |
| Llama 8B (No Synthetic) | 0.6554 | 0.5971 | Strong validation, mediocre test |
| Llama 8B + 5% Synthetic | 0.6271 | 0.6234 | Final submission, 4th of 35 teams |

### Ablation Study

| Synthetic Ratio | Val F1 | Test F1 | Description |
| :--- | :--- | :--- | :--- |
| 0% | 0.6554 | 0.5971 | Best validation, but conservative on minority classes |
| 2% | 0.6247 | 0.5042 | Insufficient enhancement and unstable test performance |
| 3% | 0.6051 | 0.5514 | Still lower than 0% synthetic |
| 5% | 0.6271 | 0.6232 | Best test transfer |
| 7% | 0.6214 | 0.4649 | Obvious overfitting or distribution shift |
| 10% | 0.5499 | 0.5851 | Some recovery but inferior to 5% |
| 15% | 0.6045 | 0.5343 | Significant interference from synthetic style |

### Key Findings
- Per-class F1 varies significantly: Non-toxic (0.94), Insults/Flaming (0.74), Other Offensive (0.44), Hate/Harassment (0.43), Threats (0.33), and Extremism (0.86). Macro performance is primarily dragged down by minority classes.
- The training set contains 40.2% exact duplicates and 13.4% identical texts with different labels. Removing duplicates reduced performance (0.44 vs 0.60 F1), suggesting duplicates act as implicit oversampling.
- High validation F1 is deceptive: Gemma 12B and Hierarchical methods reached 0.66–0.68 on validation but only 0.47–0.55 on test.
- The value of the 5% synthetic ratio is not to "balance" the data but to slightly shift model tendency toward predicting confused minority classes (Class 2/3).

## Highlights & Insights
- The most valuable finding is the "validation trap." The paper highlights that matching validation distributions can reward overly conservative classifiers, a practical insight for shared tasks and moderation systems.
- Use of synthetic data is highly restrained. While many augmentation papers assume "more is better," this work demonstrates how 7% or 15% can significantly harm test performance, framing synthetic data as a calibrant rather than an infinite expander.
- Observations on duplicates indicate that for extreme imbalance, common data cleaning rules may be counterproductive; duplicates can carry annotation frequency and class priors.
- This approach is transferable to other community governance tasks, such as live stream chat moderation or cross-lingual hate speech detection: analyze validation/test shifts first, then use small-dose paraphrasing to adjust minority sensitivity.

## Limitations & Future Work
- As a shared-task system report, the innovation is driven by engineering choices. There is a lack of generalized theoretical explanation to predict why the "optimal synthetic ratio" is precisely 5%.
- Experiments are focused on World of Tanks. Differences in game types, community culture, and language distributions may affect toxicity boundaries; cross-game generalization remains unproven.
- The F1 for Class 4 (Threats) is only 0.33, indicating that the rarest but highest-risk category remains difficult to identify reliably.
- The code is not open-sourced, and more implementation details for LoRA training, synthetic filtering, and post-processing would be needed for replication.
- Future work could explicitly transform the validation trap into a model selection criterion, using predictive distribution or minority class calibration to assist early stopping.

## Related Work & Insights
- **vs XLM-RoBERTa / RoBERTa Toxicity Classification**: Encoder models are lighter but struggle with short-text, multilingual, and fine-grained labels. This work injects label semantics via instruction prompts.
- **vs Generative Data Augmentation**: Direct generation can create templated toxic sentences. This work uses paraphrasing of real messages to maintain in-domain style.
- **vs Hierarchical Classification**: Hierarchical methods achieve high validation F1 but fail on test sets as errors amplify across the two stages.
- **vs Ensembles / Post-calibration**: Such strategies often introduce noise when a strong single model is dominant; this work shows that understanding data distribution is more critical than stacking models.

## Rating
- Novelty: ⭐⭐⭐ Limited algorithmic novelty, but the validation trap and synthetic ratio analysis offer high practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid comparisons of multiple models, ratios, and alternative strategies for a system paper.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative with honest reporting of failed approaches and negative results.
- Value: ⭐⭐⭐⭐ Direct insights for content moderation, low-resource minority classification, and synthetic data usage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models](smarter_a_data-efficient_framework_to_improve_toxicity_detection_with_explanatio.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[ACL 2026\] YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling](yeze_at_semeval-2026_task_9_detecting_multilingual_multicultural_and_multievent_.md)

</div>

<!-- RELATED:END -->
