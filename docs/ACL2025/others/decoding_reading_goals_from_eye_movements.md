---
title: >-
  [Paper Note] Decoding Reading Goals from Eye Movements
description: >-
  [ACL 2025] This paper introduces the novel task of decoding readers' reading goals (information seeking vs. ordinary reading) from eye movement trajectories. Through a systematic comparison of 12 models, a Transformer-based scanpath and language modeling approach (RoBERTa-Eye-F) is found to be optimal, achieving high-accuracy, real-time prediction early in the reading process.
tags:
  - "ACL 2025"
date: 2026-05-08
content_hash: bc4e9502e9c56ac1
---

# Decoding Reading Goals from Eye Movements

**Conference**: ACL 2025  
**Authors**: Omer Shubi, Cfir Avraham Hadar, Yevgeni Berzak  
**Affiliation**: Technion - Israel Institute of Technology; MIT  
**arXiv**: [2410.20779](https://arxiv.org/abs/2410.20779)  
**Area**: Eye Tracking / Cognition of Reading / Multimodal Classification  

---

## TL;DR

This paper introduces the novel task of decoding readers' reading goals (information seeking vs. ordinary reading) from eye movement trajectories. Through a systematic comparison of 12 models, a Transformer-based scanpath and language modeling approach (RoBERTa-Eye-F) is found to be optimal, achieving high-accuracy, real-time prediction early in the reading process.

---

## Background & Motivation

### Problem Definition

People read with different goals, such as ordinary reading for comprehension or information seeking (i.e., reading with a specific question in mind). Do these two reading modes leave distinguishable signals in eye movement data? Although prior studies have identified significant differences between these two modes at the group average level (e.g., reading speed, fixation patterns), whether reading goals can be automatically decoded from a **single reading trial** has not been systematically investigated.

### Motivation

- **Educational Settings**: Real-time monitoring of student reading engagement to distinguish between active reading and searching for answers.
- **User Interfaces**: Dynamically adjusting content presentation (e.g., highlighting key information) based on the user's reading goals.
- **Assistive Technologies**: Helping specific populations (e.g., older users) navigate complex web pages.
- **Cognitive Science**: Deepening the understanding of eye movement differences between different reading modes.

### Limitations of Prior Work

Existing studies (Hahn & Keller, 2023; Malmaud et al., 2020) only perform descriptive statistical analysis without attempting automatic classification. The classification work of Hollenstein et al. (2023) is limited to the sentence level with special linguistic annotation tasks, lacking representativeness for everyday reading.

---

## Method

### Task Formalization

Given a participant $S$'s eye movement recording $E_P^S$ on a passage $P$, predict the reading goal:

$$h: (E_S^P, P) \rightarrow \{\text{Information Seeking}, \text{Ordinary Reading}\}$$

where the passage text $P$ is an optional input, and the classifier does not receive the question content or the participant's identity.

### Model Suite (12 + 1 Ensemble)

**Eye-Tracking Only Models (4)**:
1. **Logistic Regression**: 9-dimensional global eye-tracking features (average fixation duration, saccade amplitude, etc.)
2. **BEyeLSTM-No Text**: Eye-tracking sequence-based LSTM without using text.
3. **ViT**: Vision Transformer classifying scanpaths rendered as images.
4. **ConvNext v2**: Similar to ViT, using the ConvNext v2 architecture.

**Eye-Tracking + Text Models (8)**:
- **RoBERTa-Eye-W**: Word-level eye-tracking features fused with word embeddings at the input layer.
- **RoBERTa-Eye-F**: Fixation-level representation, where each fixation point is encoded independently.
- **MAG-Eye**: Infusing eye-tracking features into intermediate layers of the Transformer.
- **PLM-AS**: Reordering word embeddings by fixation sequence and processing with an RNN.
- **Haller RNN**: Word embeddings in fixation order concatenated with eye-tracking features, processed by an RNN.
- **BEyeLSTM**: LSTM + linear projection using fixation sequences and global features.
- **Eyettention**: Cross-attention between a RoBERTa encoder and an LSTM fixation encoder.
- **PostFusion-Eye**: Cross-attention fusion of RoBERTa word representations and convolutional fixation features.

**Logistic Ensemble**: Logistic regression ensemble using the output probabilities of the 12 models as features.

### Dataset

Using the **OneStop Eye Movements** dataset:
- 360 native English-speaking adult participants, using an EyeLink 1000 Plus eye tracker.
- 30 Guardian news articles (advanced/simplified versions), 54 passages.
- Between-subjects design: Information Seeking vs. Ordinary Reading.
- Total of 19,438 trials (balanced: 9,718 ordinary reading + 9,720 information seeking).
- Each passage read by 120 participants (60 under each condition).

### Evaluation Protocol

10-fold cross-validation with three generalization evaluations:
- **New Item**: Unseen passages, seen participants.
- **New Participant**: Unseen participants, seen passages.
- **New Item & Participant**: Both unseen (zero-shot).

---

## Key Experimental Results

### Main Results: Model Performance Comparison

| Model | New Item | New Participant | New Item & Part. | All |
|------|----------|-----------------|-------------------|-----|
| Majority/Chance | 50.0 | 50.0 | 50.0 | 50.0 |
| Reading Time | 59.0 | 58.9 | 60.4 | 59.0 |
| Logistic Regression | 62.4 | 60.6 | 60.8 | 61.5 |
| BEyeLSTM-No Text | 71.5 | 61.0 | 61.5 | 65.9 |
| ViT | 70.6 | 64.4 | 64.4 | 67.3 |
| ConvNext v2 | 70.4 | 63.7 | 64.0 | 66.9 |
| RoBERTa-Eye-W | 64.6 | 62.5 | 62.0 | 63.5 |
| PostFusion-Eye | 88.5 | 90.3 | 86.0 | 89.3 |
| **RoBERTa-Eye-F** | **89.9** | **90.9** | **88.2** | **90.3** |
| **Logistic Ensemble** | **91.3** | **91.6** | **88.0** | **91.3** |

**Key Findings**: RoBERTa-Eye-F is the best single model across all evaluation settings (90.3%), and the Logistic Ensemble further improves this to 91.3%. The best-performing models share three components: Transformer architecture + fixation-level encoding + text modeling.

### Real-time Prediction: High Accuracy with Early Fixation Data

| Fixation Ratio | 1% | 5% | 10% | 25% | 50% | 100% |
|---------|-----|-----|------|------|------|------|
| Average Time (seconds) | 0.5 | 1.5 | 2.7 | 6.3 | 12.4 | 24.3 |
| Accuracy | 61.0 | 77.6 | 78.9 | 82.3 | 84.9 | 90.3 |

Only using the first 5% of fixation data (~1.5 seconds) achieves 77.6% accuracy, demonstrating that the task can be executed online in real-time.

### Error Analysis: Mixed-Effects Modeling

Analyzing the relationship between the correctness of RoBERTa-Eye-F predictions and trial characteristics via mixed-effects models:
- **Key Findings**: Reading speed before and after the critical span is the most important predictor.
- In information seeking, faster reading before and after the critical span makes the trial easier to classify correctly.
- In ordinary reading, slower reading within the critical span leads to higher classification accuracy.
- Longer passages make ordinary reading easier to classify.
- Shorter critical spans make information seeking easier to classify (clearer goals).

---

## Highlights & Insights

- **First Systematic Task Definition**: Decoding reading goals is formalized as a binary classification task with clear practical value.
- **Large-Scale Systematic Comparison**: 12 models covering different architectures, data representations, and modal fusion strategies provide a comprehensive methodological landscape.
- **Real-Time Feasibility Verification**: High classification accuracy of 77.6% with only 1.5 seconds of eye movement data supports online application.
- **Mixed-Effects Error Analysis**: An innovative approach to model performance analysis reveals interpretable axes of task difficulty while controlling for multiple factors.

---

## Limitations & Future Work

1. **Limited Text Scope**: Only covers paragraph-level news text (3-10 lines), without shorter (single sentences) or longer texts, or other genres.
2. **Homogeneous Language and Population**: Limited to native English-speaking adult readers, without second language learners, different age groups, or other languages.
3. **Binary Classification Limitation**: Only distinguishes between two reading modes, without attempting to decode specific information seeking questions.
4. **Generalization to New Participants**: There is still room for improvement, as weaker models exhibit lower performance under the "New Participant" setting.
5. **Between-Subjects Design**: Each participant participated in only one reading condition, making it impossible to completely eliminate individual effects.

---

## Related Work & Insights

- **Goal-Directed Reading**: Just et al. (1982) and Kaakinen & Hyönä (2010) investigated differences in eye movements during tasks such as skimming, scanning, and proofreading.
- **Analysis of Reading in Information Seeking**: Hahn & Keller (2023), Malmaud et al. (2020), and Shubi & Berzak (2023) analyzed the differences in eye movements between information seeking and ordinary reading, finding significant discrepancies around critical spans.
- **Reading Task Classification**: Hollenstein et al. (2023) attempted classification at the single-sentence level using the ZuCo corpus, but targeted specialized linguistic annotation tasks.
- **Eye Movement Prediction Models**: General models such as RoBERTa-Eye (Shubi et al., 2024), BEyeLSTM (Reich et al., 2022), and Eyettention (Deng et al., 2023); this work is the first to unify them for decoding reading goals.
- **Cognitive State Prediction**: Using eye tracking for predicting reading comprehension (Reich et al., 2022b; Shubi et al., 2024) and classifying document types or readability.

---

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Rating | ⭐⭐⭐⭐ |

> The new task definition is clear, and the experiments are highly thorough (12 models × 3 generalization settings × 10 folds), with an innovative mixed-effects error analysis. Although the dataset is from existing resources, it is utilized comprehensively. The primary drawback lies in the coarse granularity of the binary classification and its limitation to English news genres.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[ACL 2025\] Theoretical Guarantees for Minimum Bayes Risk Decoding](theoretical_guarantees_for_minimum_bayes_risk_decoding.md)
- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)

</div>

<!-- RELATED:END -->
