---
title: >-
  [Paper Note] ReMedi: Reasoner for Medical Clinical Prediction
description: >-
  [ACL 2026][Medical NLP][Electronic Health Records] ReMedi reformulates EHR clinical prediction as a "rationale-answer" generation and preference learning task. Through hard sample regeneration with ground-truth hints…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Electronic Health Records"
  - "Clinical Prediction"
  - "Reasoning Fine-tuning"
  - "Preference Optimization"
  - "Hard Sample Regeneration"
date: 2026-05-08
content_hash: f8784d2da50b69f2
---

# ReMedi: Reasoner for Medical Clinical Prediction

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.01474](https://arxiv.org/abs/2605.01474)  
**Code**: No public code available  
**Area**: Medical Clinical Prediction / EHR Modeling / Medical LLMs  
**Keywords**: Electronic Health Records, Clinical Prediction, Reasoning Fine-tuning, Preference Optimization, Hard Sample Regeneration

## TL;DR
ReMedi reformulates EHR clinical prediction as a "rationale-answer" generation and preference learning task. Through hard sample regeneration with ground-truth hints, SFT, and DPO, it enables medical LLMs to learn fine-grained patient risk explanations, achieving up to a 19.9 F1 point improvement over KARE on three types of MIMIC-IV prediction tasks.

## Background & Motivation
**Background**: Electronic Health Records (EHRs) contain diagnoses, medications, examinations, and hospitalization trajectories, serving as critical data sources for predicting mortality risk, readmission, and length of stay. Recent methods convert EHRs into text for medical LLMs to read patient histories directly or use medical knowledge graphs, retrieval augmentation, and knowledge distillation to supplement domain knowledge.

**Limitations of Prior Work**: These methods often focus on "knowledge supplementation" while assuming the model is already capable of explaining complex EHR contexts. Actual clinical prediction is not simple factual Q&A; models must distinguish nuances in disease severity, treatment trajectories, and chronic disease risks. If models are only trained to output labels, they easily learn biased positive patterns or overly conservative modes.

**Key Challenge**: Clinical prediction requires both interpretable reasoning chains and accurate final labels. However, directly generating reasoning chains does not guarantee consistency between reasoning and answers, and expensive expert annotations cannot cover large volumes of EHR samples. The key contradiction in the paper is: how to utilize existing ground-truth outcome labels to low-costly construct supervision and preference data that can train the model's reasoning capabilities.

**Goal**: The authors aim to enable models to automatically generate high-quality rationales from hard cases without relying on proprietary teacher models or predefined medical ontologies, transforming the relationships between correct rationales, incorrect rationales, and final answers into optimizable training signals.

**Key Insight**: Ground-truth clinical outcomes can serve as "hints" to help the model backward-explain difficult cases. By using label hints during the data construction phase and filtering out content that explicitly leaks the hint before training, labels can be turned into reasoning data generators rather than cheating information during inference.

**Core Idea**: Use "hard samples + ground-truth label hints" to generate more reliable rationale-answer pairs, then train medical LLMs using SFT and DPO so that prediction results and reasoning processes are aligned simultaneously.

## Method
ReMedi's method is straightforward: it first lets the model generate reasoning and predictions for EHR questions, filters correct samples based on ground-truth labels, then specifically returns to hard samples where the model erred to induce more reasonable explanations via label hints. Finally, SFT and DPO are used to transform this synthetic data into model capabilities. It does not introduce a new EHR encoder but shapes the reasoning habit of "reading cases, identifying risks, and providing conclusions" at the LLM post-training level.

### Overall Architecture
The input consists of a textualized patient EHR context and a clinical prediction question; the output is a predicted answer with a reasoning process. The full pipeline involves three steps: first, the model generates rationale-answer pairs for the training set; second, label-hint-guided regeneration is performed for samples that were answered incorrectly or were difficult; third, correct reasoning samples are used for SFT, and pairs of correct and incorrect answers are used for DPO. The authors also propose iReMedi, which executes this three-stage process iteratively for multiple rounds, using the model from the previous round as the data generator but re-initializing from the original base during training to mitigate overfitting.

### Key Designs
1.  **Sample Filtering and Preference Construction based on Ground-truth**:
    - **Function**: Converts existing EHR labels into data for supervised learning and preference learning.
    - **Mechanism**: Given a question $q_i$ and true answer $a_i$, the generation model outputs rationale $\hat r_i$ and answer $\hat a_i$. If $\hat a_i = a_i$, the sample enters the SFT dataset; if both correct and incorrect outputs exist for the same question, the correct output is treated as preferred and the incorrect as dispreferred to construct DPO data.
    - **Design Motivation**: Clinical prediction labels already exist, but annotating reasoning chains for every entry is expensive. This design automatically filters reasoning quality using labels, ensuring training signals come not just from the final answer but also from the explanation behind it.

2.  **Hard Sample Regeneration**:
    - **Function**: Focuses on utilizing cases the model currently misses to improve learning efficiency for boundary and complex cases.
    - **Mechanism**: For incorrect samples, the ground-truth answer is passed as a hint to the model for label rationalization, with $k$ candidates sampled for each. Samples that yield the correct answer without explicitly mentioning the hint in the rationale are kept to supplement SFT and DPO data.
    - **Design Motivation**: Easy samples offer limited improvement; the model needs to learn from cases that confuse readmission risk, mortality risk, or length of stay. Label hints lower the difficulty of generating high-quality explanations for early-stage models, while filtering rules prevent training data from leaking "the answer because the hint said so."

3.  **SFT/DPO and Iterative iReMedi**:
    - **Function**: First teaches the model to imitate correct reasoning, then uses preference optimization to widen the probability gap between correct and incorrect reasoning.
    - **Mechanism**: SFT minimizes cross-entropy for correct rationale-answer pairs; DPO optimizes the preference for correct outputs relative to incorrect ones on the SFT model. iReMedi generates the next batch of data using the updated model each round, but the training phase restarts from the original model to avoid accumulating and amplifying noise through multi-round self-training.
    - **Design Motivation**: Simple SFT tends to learn surface patterns, whereas DPO can explicitly penalize reasoning that seems plausible but produces wrong answers. The iterative process allows the model to gradually discover harder samples rather than relying solely on the generation quality of the initial model.

### Loss & Training
Experiments use HuatuoGPT-o1-7B as the base model, fine-tuned with TRL, Transformers, DeepSpeed, and Flash-Attention2. The learning rate is $5e^{-6}$ with the AdamW optimizer and a batch size of 16. Training data comes from MIMIC-IV, split 0.8/0.1/0.1 for training, validation, and testing. The SFT stage minimizes the token cross-entropy of correct rationales and answers; the DPO stage maximizes the preference ratio of correct rationale-answer pairs over incorrect ones.

## Key Experimental Results

### Main Results
The authors evaluated three types of clinical prediction on MIMIC-IV: mortality prediction, 15-day readmission prediction, and length of stay (LOS) prediction. 10,000 samples were used per task; the mortality task included 2,701 deaths and 7,299 survivals, the readmission task was balanced with 5,000 each, and LOS was a four-class task with 2,500 samples per class.

| Method | Mortality Acc/F1 | Readmission Acc/F1 | LOS Acc/F1 | Main Findings |
| :--- | :--- | :--- | :--- | :--- |
| Few-shot HuatuoGPT-o1-7B | 75.2 / 73.9 | 52.2 / 41.8 | 31.4 / 24.6 | Prompt-based reasoning is insufficient |
| SFT | 88.9 / 88.3 | 69.2 / 66.4 | 39.9 / 36.6 | Direct SFT helps, but LOS remains weak |
| KARE | 95.9 / 95.5 | 81.2 / 81.3 | 40.4 / 35.9 | Strong baseline relying on medical knowledge |
| ReMedi | 97.7 / 97.6 | 90.5 / 90.4 | 55.6 / 55.5 | Outperforms KARE across all three tasks |
| iReMedi | 97.8 / 97.6 | 91.5 / 91.4 | 56.1 / 55.8 | Iterative training further improves results |

### Ablation Study
The paper analyzes the contributions of DPO, iterative training, and STaR-style self-training, primarily on the readmission task.

| Configuration | Acc | F1 | TPR | TNR | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ReMedi | 90.5 | 90.4 | 80.6 | 100.0 | Full three-stage pipeline |
| ReMedi w/o DPO | 84.4 | 84.4 | 85.3 | 83.6 | Performance drops, TNR becomes unstable |
| iReMedi | 91.5 | 91.4 | 83.8 | 100.0 | Best iterative version |
| iReMedi w/o DPO | 86.8 | 86.8 | 83.7 | 89.9 | Iteration helps, but DPO is critical |
| STaR | 59.1 | 53.2 | 96.1 | 23.4 | Generic self-training is unsuitable |

The authors also manually inspected the consistency between reasoning and prediction. In the readmission task, KARE's average consistency was 60.0% (human) / 52.0% (Gemini evaluation), while ReMedi reached 92.5% / 90.0%. This indicates that ReMedi's improvement is not just in label accuracy but also in the consistency between explanations and final conclusions.

### Key Findings
- The strongest improvement comes from the combination of hard sample regeneration and DPO: hard samples provide more informative training points, while DPO lowers the preference probability of incorrect reasoning.
- The LOS task showed the largest gain, with ReMedi improving by 15.2 Acc points and 19.6 F1 points relative to KARE, showing it is particularly effective for multiclass and fine-grained risk assessment.
- Standard LLMs often exhibit high TPR but low TNR on readmission tasks, tending to over-predict risk; ReMedi's case studies show it can more accurately distinguish between "stable chronic disease" and "true high-risk chronic disease."

## Highlights & Insights
- ReMedi's ingenuity lies in turning labels from "final supervision" into "scaffolding for reasoning data generation." Ground-truth outcomes are used only as hints during data construction and do not enter the reasoning text after filtering, thereby improving sample quality while reducing the risk of label leakage.
- The paper does not introduce complex medical knowledge bases but proves that post-training strategies alone can significantly improve EHR prediction. This is important for resource-constrained medical scenarios where building ontologies and retrieval systems is often harder to maintain than fine-tuning.
- The alignment analysis is highly valuable: in medical prediction, "plausible-looking explanations with inconsistent answers" directly impacts trustworthiness. ReMedi treats reasoning-prediction alignment as an observable goal, providing a dimension closer to deployment risk than simple accuracy for clinical LLM evaluation.
- The method is transferable to other tasks with ground-truth labels but lacking reasoning annotations, such as ICU intervention prediction, adverse drug reaction prediction, or insurance claim risk modeling.

## Limitations & Future Work
- The authors admit ReMedi still produces a few instances of inconsistent reasoning and predictions, indicating that filtering rules and DPO preferences cannot yet fully guarantee explanation faithfulness.
- Experiments focused on clinical prediction tasks with clear labels; open-ended clinical QA, treatment plan generation, or multimodal medical decision-making have not yet been validated.
- The base model was primarily HuatuoGPT-o1-7B; whether models above 70B still require the same intensity of regeneration and preference optimization was not systematically studied.
- Human evaluation only covered reasoning-prediction alignment and did not involve clinical experts strictly assessing whether each rationale was medically correct, which limits the clinical trustworthiness conclusions.
- Future use in real medical systems requires adding uncertainty estimation, expert review, and data drift monitoring rather than relying solely on a single prediction label.

## Related Work & Insights
- **vs KARE**: KARE enhances reasoning through medical knowledge distillation and structured medical graphs; ReMedi uses label-guided self-generated rationales and DPO to directly shape predictive reasoning capabilities. The former is more knowledge-dependent, while the latter is lighter for scaling and deployment.
- **vs STaR**: STaR uses model-generated rationales for iterative self-training but performs poorly in clinical prediction; ReMedi differs by using ground-truth hints specifically for hard samples and constraining incorrect rationales with preference data.
- **vs RAG Medical LLMs**: RAG solves external knowledge coverage, while ReMedi solves EHR context interpretation. The two can be complementary—for example, by retrieving guideline knowledge first and then using ReMedi-style preference training to ensure conclusion-rationale consistency.
- **Insight**: For high-risk tasks with labels but missing explanations, consider the "label hint rationale generation + leakage filtering + preference optimization" route, which trains interpretable decision-making better than simple SFT on final labels.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines label rationalization, hard samples, and DPO into EHR prediction; simple idea but excellent scenario adaptation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, ablations, alignment, and case studies are complete, though expert evaluation and larger model scaling are limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology, sufficient tabular information; some implementation details like prompt templates could be further expanded.
- Value: ⭐⭐⭐⭐⭐ Highly practical for "accurate prediction + explanation consistency" in medical LLMs, especially suitable for tasks with low expert annotation budgets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->
