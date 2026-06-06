---
title: >-
  [Paper Note] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification
description: >-
  [ACL2026][LLM/NLP][Defense mechanism identification] This system paper for the BioNLP 2026 PsyDefDetect shared task treats psychological defense mechanism classification as a problem with fuzzy boundaries and limited ann…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "Defense mechanism identification"
  - "Mental health dialogue"
  - "Model ensemble"
  - "Class imbalance"
  - "Shared task system"
date: 2026-05-08
content_hash: 32061b987aaaa7a8
---

# Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification

**Conference**: ACL2026  
**arXiv**: [2605.07606](https://arxiv.org/abs/2605.07606)  
**Code**: https://github.com/th-nuernberg/nuernberg-nlp-psydefdetect  
**Area**: NLP Understanding / Mental Health NLP  
**Keywords**: Defense mechanism identification, Mental health dialogue, Model ensemble, Class imbalance, Shared task system

## TL;DR
This system paper for the BioNLP 2026 PsyDefDetect shared task treats psychological defense mechanism classification as a problem with fuzzy boundaries and limited annotation consistency. By employing an ensemble of 9 voters across different granularities, training objectives, and base models, the system achieved an F1 of .420 on the hidden test set, ranking first among 21 registered teams.

## Background & Motivation
**Background**: Mental health NLP is evolving from sentiment recognition and risk assessment toward fine-grained clinical concept identification. The PsyDefDetect task requires models to identify the level of psychological defense mechanisms in a seeker's target utterance within emotional support dialogues. Labels are derived from the Defense Mechanism Rating Scale (DMRS), encompassing 8 positive defense levels and a "No Defence" (C0) category.

**Limitations of Prior Work**: This task is not a simple semantic classification. Many defense mechanisms appear similar on the surface—such as rationalization, explanation, reflection, or self-consolation—while their true differences are hidden in pragmatic functions and contextual intent. The paper notes that even trained annotators achieve only moderate agreement, with a Cohen's $\kappa=.639$, indicating unstable label boundaries.

**Key Challenge**: A single large model easily develops fixed biases at these fuzzy boundaries. Even stronger models may not resolve confusion between mid-to-high levels like C5, C6, and C7, as they share numerous "mature, restrained, and reflective" expressions. The truly valuable signal lies not in the absolute strength of one model, but in the fact that different models make different mistakes on specific samples.

**Goal**: The authors aimed to build a shared task system capable of identifying the highly separable "No Defence" class while minimizing the absorption of samples into majority classes across the 8 defense levels. The system also had to handle realistic constraints such as extreme class imbalance in the training set, an invisible hidden test set, and limited validation signals.

**Key Insight**: Starting from the concept of "error independence," the authors designed the system as a multi-axis voter ensemble rather than a single-path fine-tuning pipeline. Representation space analysis confirmed that C0 No Defence is the most separable, while the 8 defense categories overlap significantly. Consequently, C0 detection is assigned to 9-class gatekeepers, while internal defense classification is handled by 8-class specialists.

**Core Idea**: Replace a single-model classifier with a voter ensemble spanning across category granularities, training objectives, and base models to transform independent errors on fuzzy psychological defense boundaries into voting gains.

## Method
The methodology resembles a system engineering solution for a shared task: it uses synthetic data to alleviate minority class issues, trains multiple groups of complementary models, and fuses them through a voting rule with a C0 override. The focus is not on proposing a new neural architecture but on designing "which model combinations generate useful disagreements."

### Overall Architecture
The input is an emotional support dialogue and one target seeker utterance.

The output is a DMRS defense mechanism label: C0 for No Defence, and C1 to C8 for the eight defense levels.

The system consists of 9 voters in total.

The first group consists of three **Min-SFT 9c gatekeeper voters**, fine-tuned via generative SFT on Ministral-8B, retaining all 9 classes.

The second group consists of three **Min-LR 8c specialist voters**, which use adapted representations from Ministral-8B to train 8-class logistic regression classifiers, processing only C1 to C8.

The third group consists of three **Phi4-LR 8c specialist voters**, training 8-class logistic regression classifiers on Phi-4-14B representations to provide cross-model error diversity.

Each branch was trained using 5-fold cross-validation, and the top-3 folds based on internal CV performance were selected for the final system.

During inference, the gatekeepers first determine if a sample belongs to C0.

If a majority of gatekeepers predict C0, the system outputs C0 directly.

Otherwise, all 9 voters perform majority voting across C1 to C8.

In the case of a tie, the system defaults to the majority class in the training set, C7 High-Adaptive.

This workflow handles the separability of No Defence, the fine-grained confusion of the 8 defense levels, and the complementarity of different models within a single voting framework.

### Key Designs
1.  **Granularity Split between C0 Gatekeeper and 8-class Specialists**:
    *   **Function**: Splits the problem into "Is there a defense mechanism?" and "Which defense mechanism is it?"—two sub-problems better suited for modeling.
    *   **Mechanism**: The authors performed t-SNE on the hidden states of the 9-class SFT model and found that C0 No Defence formed the only relatively distinct cluster, whereas C1 to C8 showed heavy overlap. Thus, 9-class models serve as gatekeepers to trigger the C0 override, while 8-class specialists focus their capacity on the subtle differences within the defense categories.
    *   **Design Motivation**: If all models handle 9 classes simultaneously, the clear boundary of C0 and the fuzzy boundaries of the defense levels are mixed in the same decision space. By splitting them, the system leverages the high separability of C0 while preventing specialists from wasting decision capacity on No Defence.

2.  **Complementary Training via Generative SFT and Discriminative LR**:
    *   **Function**: Creates useful error independence through differences in training objectives.
    *   **Mechanism**: The SFT branch fine-tunes LLMs via QLoRA to generate label digits. The LR branch reuses the LLM representations after ClsHead adaptation, discards the original classification head, and trains an L2-regularized multinomial logistic regression on the frozen last-token hidden states. LR adds almost no computational overhead but allows for rapid screening of model and granularity combinations.
    *   **Design Motivation**: The paper did not simply choose the 9-class LR with the highest CV score as the gatekeeper. Instead, it retained the generative SFT because, when paired with 8-class LR specialists, the two training paradigms fail on different samples. Experiments support this: pairing SFT gatekeepers with LR specialists improved hidden test F1 from .373 to .391, whereas pairing them with SFT specialists showed no gain.

3.  **Third Axis Voting via Cross-Model Ensembling**:
    *   **Function**: Introduces a different base model alongside the Ministral branch to handle samples where internal Ministral opinions are split.
    *   **Mechanism**: The authors tested 8-class LR specialists on Phi-4-14B, Llama-3.1-8B, and PsychoCounsel-Llama3-8B. Complementarity was measured by the correlation of their F1 profiles across 5 folds with Min-LR 8c. Phi4-LR 8c exhibited the most inverse profile and was selected as the third branch.
    *   **Design Motivation**: The third branch cannot overturn a strong majority from the 6 Ministral voters; it only influences samples where Ministral is divided. This makes it an arbitrator rather than a simple addition of model counts. Flip analysis showed that Phi4-LR flips were concentrated on the C6/C7 boundary—the most frequent area of confusion.

### Loss & Training
All fine-tuning utilized 4-bit NF4 QLoRA applied to all linear projection layers, with a dropout of 0.05, a cosine schedule, 10% warm-up, for 10 epochs, an effective batch size of 8, and a maximum sequence length of 4096.

SFT used LoRA rank 32, $\alpha=64$, with generative cross-entropy of label digits as the objective.

ClsHead used LoRA rank 16, $\alpha=32$, trained with focal loss using class weights $w_c=N/(K n_c)$.

LR was trained on frozen hidden states using multinomial logistic regression with L2 regularization and balanced class weights, with the regularization intensity $C$ swept within each fold.

For data augmentation, GPT-5.2 was used to generate minority class synthetic dialogues on a dialog-stratified 80/20 split.

The augmentation budget was capped at 200 samples per class, and synthetic samples did not exceed 75% of the original class count. No augmentation was performed for C0 and C7. A total of 738 synthetic dialogues were generated and added only to the training folds; validation and hidden tests remained original human-annotated data.

The final voting rule: if the count of gatekeepers predicting C0 reaches a majority, $\hat{y}=0$; otherwise, $\arg\max_c \sum_j 1[v_j=c]$ for all voters.

## Key Experimental Results

### Main Results
The main results focus on the macro-F1 of the hidden test set, calculated only for C1-C8.

| System | Diversity Axes Activated | Hidden Test F1 | Gain vs Baseline |
| :--- | :--- | :--- | :--- |
| Organizer Baseline: Min-SFT 9c, no aug | Single Model | .315 | - |
| Min-SFT 9c full-train, aug, single model | Augmentation only | .307 | -0.008 |
| 5V Min-SFT 9c | 5-fold ensemble | .373 | +0.058 |
| 6V Min-SFT 9c + Min-LR 8c | Granularity + Training | .391 | +0.076 |
| 9V Min-SFT 9c + Min-LR 8c + Phi4-LR 8c | + Base Model axis | **.420** | **+0.105** |

The largest jump came from voting itself: 5-fold Min-SFT 9c improved from around .315/.307 to .373. Adding LR specialists further improved it to .391, proving the complementarity of training paradigms. After adding the third base model, all candidates improved performance, with Phi4-LR 8c reaching .420. This represents a 33.4% relative improvement over the baseline and the first-place result in the shared task.

### Ablation Study
The most insightful ablation involves synthetic data augmentation.

| System | With GPT-5.2 Aug | Without Aug | Difference | Note |
| :--- | :--- | :--- | :--- | :--- |
| Min-SFT 9c Single Model | .307 | .315 | -0.008 | Synthetic data alone introduces noise |
| 5V Min-SFT 9c | .373 | .319 | +0.054 | Voting averages out synthetic noise |
| 6V + Min-LR 8c | .391 | .369 | +0.022 | Augmentation gain persists |
| 9V + Phi4-LR 8c | .420 | .378 | +0.042 | Contribution of aug is clear in final system |

This table shows that augmentation is not an "independent module" where more is always better. Single models were degraded by synthetic samples, but the voting system combined the recall gains from augmentation with noise cancellation. Thus, augmentation and voter diversity are intertwined.

### Key Findings
*   Voting is more critical than a single model. A 5-fold isomorphic ensemble raised the F1 from .315 to .373, suggesting that variance and boundary uncertainty are the core issues in this task.
*   The gain from the third model axis comes from critical boundary arbitration. Phi4-LR primarily flips samples where Ministral is inconsistent, with 33 out of 39 flips involving the C6/C7 boundary.
*   Synthetic data must be tied to voting. While a single model's performance decreased with augmentation, the 9V system's F1 increased by .042, indicating that the ensemble provides tolerance for synthetic noise.
*   Class imbalance still dominates error patterns. C7 accounts for 52% of the training set and absorbed many misclassified mid-level samples, meaning model performance does not directly equate to clinical utility.

## Highlights & Insights
*   The paper frames the shared task solution as a story of "error complementarity" rather than just climbing a leaderboard. It uses t-SNE, CV profile correlation, and flip analysis to justify why voting works.
*   The C0 gatekeeper design is practical. Many real-world classification tasks involve an "easy-to-separate external class" and a "group of fuzzy internal classes." Separating them is often more stable than flat multi-class classification.
*   The LR specialist is a high-efficiency trick. Training a linear head on frozen LLM representations allows for rapid exploration of base models and creates different error patterns compared to generative SFT.
*   The class-level analysis shows clinical alertness. The authors did not just report the top score but pointed out that misclassifying C5 as C7 underestimates intervention needs, demonstrating a responsible approach to clinical NLP.

## Limitations & Future Work
*   Statistical support remains limited. The PSYDEFCONV training set has only 1,864 original samples. The .029 gain of 9V over 6V is based on a single hidden test observation.
*   The choice of top-3 folds and C0 override thresholds are somewhat heuristic and lack rigorous validation.
*   The annotation upper bound is low. With an inter-annotator agreement of only $\kappa=.639$, small classes like C2, C5, and C8 fall into highly subjective zones, and macro-F1 is likely capped by label noise.
*   The inference cost is high. An ensemble of nine voters is unsuitable for real-time deployment; a 5V or 6V system might be a more reasonable cost-benefit trade-off.
*   Clinical ethical risk is significant. An F1 of .420 means most defense utterances are still misclassified. The system should only serve as a supplementary signal in a supervised workflow.

## Related Work & Insights
*   **vs PSYDEFCONV / PsyDefDetect baseline**: The baseline uses a 9-class generative SFT on Ministral-8B. This paper improves the F1 from .315 to .420 through data augmentation, CV voter pools, and multi-axis ensemble strategies.
*   **vs Traditional Ensemble Methods**: Following Dietterich’s principle that ensembles require accurate and diverse classifiers, this work applies these concepts to LLM mental health tasks and proves diversity's effectiveness at high-confusion boundaries.
*   **vs Single-model Fine-tuning**: While single-model tuning seeks an optimal representation, this work shows that multiple partially inconsistent decision boundaries are more reliable when labels are ambiguous and classes overlap.

## Rating
*   Novelty: ⭐⭐⭐(⭐)☆ The core technique is not a new architecture, but the systematic application of multi-axis voter diversity to psychological defense detection is well-designed.
*   Experimental Thoroughness: ⭐⭐⭐⭐☆ Main results, ablation studies, and flip analyses are complete, though limited by the small data and the hidden test evaluation.
*   Writing Quality: ⭐⭐⭐⭐⭐ The system paper is clearly written, with a closed loop between motivation, design, and error analysis.
*   Value: ⭐⭐⭐⭐☆ Highly valuable for shared tasks and small-sample clinical NLP; the ensemble diagnostic approach is widely applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification](evaluating_customized_vs_generalist_transformer-based_models_for_legal_contract_.md)
- [\[ACL 2026\] Wait, There's a Way Out: A Decision Mechanism for Conversational Derailment Prediction](wait_theres_a_way_out_a_decision_mechanism_for_forecasting_conversational_derail.md)
- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](../../AAAI2026/llm_nlp/from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[ACL 2026\] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges](when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](../../AAAI2026/llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)

</div>

<!-- RELATED:END -->
