---
title: >-
  [Paper Note] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection
description: >-
  [ACL 2025][Sexism Detection] Addressing the issues of data sparsity and fine-grained classification ambiguity in online sexism detection, this paper proposes two prompt-based data augmentation techniques—Definition-driven Data Augmentation (DDA) and Contextual Semantic Expansion (CSE). DDA leverages category definitions to generate semantically aligned synthetic samples, while CSE enriches training data by analyzing the semantic features of model errors. Combining these with…
tags:
  - "ACL 2025"
  - "Sexism Detection"
  - "Data Augmentation"
  - "Definition-driven"
  - "Semantic Expansion"
  - "Ensemble Learning"
date: 2026-05-08
content_hash: a5739fe06df081a1
---

# Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection

**Conference**: ACL 2025  
**arXiv**: [2506.06238](https://arxiv.org/abs/2506.06238)  
**Code**: Yes ([https://github.com/Sahrish42/explaining_matters_sexism_detection_acl2025](https://github.com/Sahrish42/explaining_matters_sexism_detection_acl2025))  
**Area**: Others  
**Keywords**: Sexism Detection, Data Augmentation, Definition-driven, Semantic Expansion, Ensemble Learning

## TL;DR

Addressing the issues of data sparsity and fine-grained classification ambiguity in online sexism detection, this paper proposes two prompt-based data augmentation techniques—Definition-driven Data Augmentation (DDA) and Contextual Semantic Expansion (CSE). DDA leverages category definitions to generate semantically aligned synthetic samples, while CSE enriches training data by analyzing the semantic features of model errors. Combining these with a Mistral-7B fallback ensemble strategy, this approach chapters SOTA performance on all tasks on the EDOS dataset.

## Background & Motivation

Online sexist language severely impacts women and marginalized groups. Automatic detection systems face two core challenges:

**Challenge 1: Data Sparsity**  
Even in the largest and most carefully curated EDOS dataset, the classes are extremely imbalanced. For instance, "Threats of harm" contains only 56 samples ($1.1\%$), and "Supporting mistreatment of individual women" has only 75 samples ($1.3\%$). This sparsity severely hinders the model's ability to generalize on low-resource categories.

**Challenge 2: Fine-grained Classification Ambiguity**  
Sexist language is inherently subtle with blurry boundaries, making it difficult even for human annotators to reach a consensus. The paper's analysis of the EDOS test set reveals striking annotation inconsistencies:
- "Descriptive attacks": $54.1\%$ partial disagreement
- "Backhanded gendered compliments": $83.3\%$ total disagreement
- "Threats of harm": $0\%$ full agreement (not a single group of annotators achieved complete consensus!)

These inconsistencies not only reflect the inherent difficulty of the task but also introduce contradictory signals during training, thereby hurting model performance.

## Method

### Overall Architecture

The pipeline consists of four stages:
1. **Pre-training**: MLM pre-training on 2 million unlabeled samples from EDOS.
2. **Data Augmentation**: DDA or CSE.
3. **Fine-tuning**: Supervised fine-tuning on the augmented data.
4. **Ensemble**: Mistral-7B fallback ensemble (M7-FE).

### Key Designs

1. **Definition-based Data Augmentation (DDA)**:

   Core Idea: Explicitly inject category definitions into the prompts for data augmentation to generate semantically aligned synthetic samples.

   For each training sample $(x_i, y_i)$, the DDA prompt contains three parts:
    - Explicit generation instructions: Maintain the original sexist intent.
    - Style guidance: Mimic the informal social media language of Reddit/Gab.
    - **Category definition $\phi(y_i)$**: Semantic definitions extracted from the EDOS taxonomy (e.g., the definition of "2.1 Descriptive Attacks" clarifies what qualifies as a descriptive attack rather than an emotional attack).

   Why are definitions important? Traditional augmentation methods (such as EDA and back-translation) only focus on linguistic diversity, failing to disentangle the semantic boundaries of neighboring categories. DDA helps the generator "understand" the core characteristics of each category through explicit definitions, thereby reducing semantic overlap between categories.

   DDA is applied only to generate synthetic data for the $c = 5$ most imbalanced categories.

2. **Contextual Semantic Expansion (CSE)**:

   Core Idea: Rather than generating more samples, generate **semantic explanations** for samples misclassified by the model, and concatenate them to the original text as augmented context.

   The workflow of CSE is as follows:
    - Train a baseline DeBERTa on training data to predict and identify all misclassified samples.
    - Find samples where the model makes incorrect predictions with high confidence ($p > 0.9$)—indicating systematic bias rather than random errors.
    - For each misclassified sample, use a structured prompt to prompt the LLM to perform a 6-step semantic analysis:
      1. Analyze linguistic patterns and style features.
      2. Examine neutral/derogatory language.
      3. Evaluate gender-related sentiment bias.
      4. Consider situational context.
      5. Identify stereotypes and underlying bias.
      6. Evaluate textual intent.

   The output semantic expansion is concatenated with the original text as $[x; e(x)]$ to serve as augmented training data.

   CSE processed 2,518 sexist samples misclassified as non-sexist + 2,328 non-sexist samples misclassified as sexist.

3. **Mistral-7B Fallback Ensemble (M7-FE)**:

   Combines the predictions of three models: DeBERTa-v3-Large, Mistral-7B, and DTFN:
    - General case: Majority voting determines the final prediction.
    - Ties (two-way split): Mistral-7B acts as the fallback model to make the final decision.
    - Complete disagreement (all three models predict differently): Use the prediction of Mistral-7B.

   Why use Mistral-7B for fallback? Preliminary experiments show it is more robust when dealing with ambiguous sexist cases.

   Design Motivation: Analogy to the human annotation process: when annotators disagree, a third-party referee is brought in. Different models trained on different data and objectives naturally provide "diverse perspectives."

### Loss & Training

- Pre-training: MLM ($15\%$ token masking) for 10 epochs.
- Fine-tuning: Standard cross-entropy loss, with DeBERTa/RoBERTa trained for 30 epochs and Mistral-7B trained for 10 epochs.
- Hardware: 4×A100 GPUs.

## Key Experimental Results

### Main Results: EDOS Dataset

| Method | Task A (Binary) | Task B (4 classes) | Task C (11 classes) |
|------|----------------|-------------|--------------|
| DeBERTa-v3-large (baseline) | 0.8479 | 0.6875 | 0.5088 |
| SemEval-2023 1st Place | 0.8746 | 0.7326 | 0.5606 |
| SEFM (Augmentation baseline) | 0.8538 | 0.6619 | 0.4641 |
| M7-FE (Ensemble only) | 0.8603 | 0.7027 | 0.5213 |
| + Baseline Prompt | 0.8783 | 0.7049 | 0.5601 |
| **+ DDA** | 0.8769 | **0.7277** | **0.6018** |
| **+ CSE** | **0.8819** | 0.7243 | 0.5639 |

### Ablation Study: DDA vs. No-Definition Prompt

DDA achieves the most significant improvement on Task C: rising from 0.5601 to 0.6018 (+4.17 F1), far exceeding all SemEval 2023 participating systems.

The paper conducts a detailed analysis of DDA's improvements using a delta confusion matrix:
- Correct predictions for "2.3 Dehumanising attacks" increased by 42 cases.
- Correct predictions for "3.4 Condescending explanations" increased by 8 cases.
- Mutual confusion between "3.1 Casual slurs" and "3.2 Gender stereotypes" decreased from 48 to 35 (a reduction of ~$27\%$).

### Key Findings

1. **CSE performs best on binary classification** (Task A: 0.8819), because binary decision boundaries are simpler and more suitable for correcting systematic biases.
2. **DDA performs best on fine-grained classification** (Task C: 0.6018, +4.1 F1 over the previous SOTA), as category definitions directly aid in distinguishing neighboring fine-grained categories.
3. **Traditional augmentation methods (SEFM, HULAT/EDA) degrade performance on fine-grained tasks**—augmentations lacking semantic constraints only introduce additional noise.
4. **Annotator disagreement rate is highly correlated with model perplexity**: categories with $0\%$ Full Agreement are precisely those where the model is most prone to misclassification.
5. The model shows high confidence ($p > 0.9$) when making incorrect predictions, indicating systematic bias rather than random uncertainty, which traditional confidence calibration/self-correction struggles to address.

## Highlights & Insights

- **Analyzing the problem from the perspective of annotator inconsistency**: Instead of merely looking at data imbalance, the authors delve into the distribution of annotator disagreements—showing $0\%$ complete agreement on certain categories, which is a more fundamental challenge than simply having fewer samples.
- **Core Insight of DDA**: When using LLMs for data augmentation, providing explicit category definitions leads to a qualitative leap over providing instructions alone. The definitions serve as "semantic anchors", constraining the generator to operate within the correct semantic boundaries.
- **"Introspective" error correction in CSE**: Instead of utilizing simple self-training or confidence filtering, CSE prompts the LLM to explain why a sample might have been misclassified—similar to Chain-of-Thought, but applied to semantic expansion for classification rather than a reasoning chain for generation.
- **Pragmatic engineering**: The fallback ensemble does not seek excessive methodological novelty but directly addresses practical issues (such as resolving voting ties in multi-class settings). The selection of Mistral-7B as the referee is well-grounded in empirical evidence rather than theoretical assumptions.

## Limitations & Future Work

1. DDA and CSE rely on LLMs (GPT-4o) for augmentation, which might introduce bias inherited from the pre-training data.
2. The evaluation is restricted to the English EDOS dataset; its effectiveness on multilingual or low-resource languages is yet to be explored.
3. M7-FE employs simple majority voting + fallback; weighted voting or confidence aggregation might yield better results.
4. The semantic expansion in CSE increases input length, which may impact inference efficiency.
5. The category definitions in DDA are drawn from the official EDOS taxonomy—other datasets may lack such structured and clear definitions.

## Related Work & Insights

- **SemEval-2023 Task 10** (Kirk et al., 2023): Established the EDOS benchmark, where the best system adopted a DeBERTa ensemble. This work achieves substantial improvements over it through data augmentation and ensemble strategies.
- **EDA** (Wei & Zou, 2019): A classic text augmentation method, but with limited efficacy on fine-grained classification. DDA achieves a qualitative leap by injecting definitions.
- **Chain-of-Thought Prompting**: CSE draws inspiration from the structured reasoning of CoT, though it is used for semantic expansion in classification tasks rather than generating reasoning chains.
- **Insight**: Definition-driven augmentation can be extended to other fine-grained classification tasks (e.g., fine-grained sentiment analysis, hate speech subtypes), provided that clear semantic definitions of categories are available.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — DDA (definition-driven augmentation) and CSE (semantic expansion error correction) are newly proposed techniques with clear design motivations. The ensemble strategy is simple yet effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across three task levels of EDOS, complete ablation studies comparing models with/without definitions and various augmentation methods, accompanied by detailed error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The table analyzing annotator inconsistency is highly intuitive and powerful, the pipeline diagram is clear, and the prompt designs are thoroughly presented.
- **Value**: ⭐⭐⭐⭐ — The +4.1 F1 improvement on Task C represents a substantial advancement, and the DDA approach offers broad reference value for other fine-grained NLP tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [\[ACL 2025\] CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction](cadreview_automatically_reviewing_cad_programs_with_error_detection_and_correcti.md)

</div>

<!-- RELATED:END -->
