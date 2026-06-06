---
title: >-
  [Paper Note] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings
description: >-
  [ACL 2026][Medical Imaging][Reinforcement Learning] This paper proposes RADS (Reinforcement Adaptive Domain Sampling), a reinforcement learning-based sample selection strategy. Under extreme low-resource and class-imbala…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Reinforcement Learning"
  - "Sample Selection"
  - "Transfer Learning"
  - "Class Imbalance"
  - "Clinical NLP"
date: 2026-05-08
content_hash: 14d0d76b2e6fcb70
---

# RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings

**Conference**: ACL 2026  
**arXiv**: [2604.20256](https://arxiv.org/abs/2604.20256)  
**Code**: [https://github.com/Wei-0808/RADS](https://github.com/Wei-0808/RADS)  
**Area**: Medical Imaging  
**Keywords**: Reinforcement Learning, Sample Selection, Transfer Learning, Class Imbalance, Clinical NLP

## TL;DR
This paper proposes RADS (Reinforcement Adaptive Domain Sampling), a reinforcement learning-based sample selection strategy. Under extreme low-resource and class-imbalanced clinical scenarios, RADS intelligently selects a small number of target domain samples for annotation and joint fine-tuning, significantly improving the transfer performance of cross-domain disease detection.

## Background & Motivation

**Background**: NLP tasks for clinical text rely heavily on high-quality annotated data, but annotation costs in the medical field are extremely high (requiring professional physicians), and many diseases are rare, leading to a severe lack of positive samples. Transfer learning is the primary strategy for low-resource scenarios, reducing annotation requirements by transferring knowledge from a source domain.

**Limitations of Prior Work**: Traditional active learning methods (such as uncertainty and diversity sampling) perform poorly in extreme low-resource and class-imbalanced conditions. Uncertainty sampling tends to select outliers at the distribution boundaries rather than truly informative samples; diversity sampling optimizes only a single metric and cannot simultaneously consider sample informativeness and redundancy. Furthermore, the heterogeneity of clinical reports (e.g., significant differences in terminology between CT, PET, and cytology reports) increases the difficulty of cross-domain transfer.

**Key Challenge**: Under extremely limited annotation budgets (e.g., only 5 samples can be annotated), how to select the most valuable samples from the unlabeled target domain so that the model performs well in both source and target domains after joint fine-tuning.

**Goal**: Design an adaptive sample selection strategy that simultaneously considers informativeness, class balance, and sample diversity.

**Key Insight**: The authors model sample selection as a sequential decision-making problem, using a reinforcement learning agent to learn an optimal selection policy that adaptively balances informativeness, class ratios, and redundancy.

**Core Idea**: A sample selection agent is trained using Dueling DQN. Guided by BALD mutual information and combined with a prior-aware utility function and redundancy penalty mechanism, the agent selects an optimal subset of samples from the target domain for annotation and fine-tuning.

## Method

### Overall Architecture
The RADS framework consists of three stages: (1) Active Learner Training: Fine-tuning a ClinicalBERT classifier on the source domain, then calculating uncertainty signals for unlabeled target domain samples via MC dropout; (2) Prior-Aware Utility Calculation: Combining BALD mutual information scores and pseudo-label class weights to construct a utility function that considers both informativeness and class balance; (3) RL Sampler Training: Using Dueling DQN to learn a selection policy that maximizes utility while penalizing redundant selections.

### Key Designs

1.  **MC Dropout-based BALD Informativeness Estimation**:
    - **Function**: Quantify the informativeness of each unlabeled sample in the target domain for the model.
    - **Mechanism**: Maintain dropout activation and perform $K$ stochastic forward passes for each sample to calculate the predictive entropy ($\mathrm{PE}$) and the expected entropy ($\mathrm{EE}$) of individual predictions. The difference is the BALD mutual information $\mathrm{MI}(x) = \mathrm{PE}(x) - \mathrm{EE}(x)$. High MI indicates the model is globally uncertain but individual sub-models disagree, marking these samples as highly informative.
    - **Design Motivation**: BALD is more robust than simple uncertainty sampling as it distinguishes between epistemic uncertainty (model ignorance) and aleatoric uncertainty (data ambiguity), selecting only samples with high epistemic uncertainty.

2.  **Prior-Aware Utility Function**:
    - **Function**: Introduce class balance control on top of informativeness.
    - **Mechanism**: Use pseudo-labels to estimate the target domain class prior $\hat{\pi}_+$, then calculate the class weight $w_+ = \rho / \mathrm{clip}(\hat{\pi}_+)$. The final utility is $u(x) = \widetilde{\mathrm{MI}}(x) \cdot w_{y(x)}$. The hyperparameter $\rho$ controls the trade-off between class balance and informativeness.
    - **Design Motivation**: Under extreme class imbalance, pure informativeness selection may lead to highly skewed class distributions in the selected samples. Prior-aware utility corrects this bias through weighting.

3.  **Dueling DQN-based Redundancy-Aware Sampler**:
    - **Function**: Learn a sequential selection policy to maximize utility while avoiding redundant samples.
    - **Mechanism**: The state vector includes the sample's mean log-probability, predictive entropy, BALD score, and budget utilization. The reward is $r_t = u(x_t) - \lambda \cdot \mathrm{Red}(x_t, S_t)$, where redundancy $\mathrm{Red}$ is calculated using the distance to the nearest neighbor among already selected samples in the predictive representation space. A Dueling DQN architecture is used to train the Q-function, learning the optimal policy via $\epsilon$-greedy exploration.
    - **Design Motivation**: Traditional methods evaluate samples independently and fail to consider interaction effects. An RL agent can dynamically adjust selection criteria during sequential decision-making, naturally avoiding redundancy.

### Loss & Training
The active learner is trained on the source domain using standard cross-entropy. The RL sampler is trained using TD loss for the Dueling DQN, combined with an experience replay buffer and a target network. After samples are selected, ClinicalBERT is fine-tuned using the union of source domain and annotated target domain samples.

## Key Experimental Results

### Main Results (CHIFIR → PIFIR Transfer, 5 Samples Selected)

| Strategy | PIFIR F1 | PIFIR ROC-AUC | CHIFIR F1 | Transfer Gap ΔF1 |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0.639 | 0.813 | 0.746 | — |
| Uncertainty | 0.545 | 0.830 | 0.824 | 0.278 |
| Diversity | 0.638 | 0.809 | 0.800 | 0.162 |
| BatchBALD | 0.849 | 0.783 | 0.500 | -0.349 |
| **Ours (RADS)** | **0.871** | **0.833** | **0.750** | **-0.121** |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| RADS Full | F1=0.871, AUC=0.833 | Full model |
| w/o Redundancy Penalty | Near Uncertainty level | Redundancy penalty is crucial for diversity |
| w/o Prior-Awareness | Increased class skew | Essential under imbalanced conditions |
| Full-shot (All target labels) | F1=0.900 | Upper bound; RADS approaches this with only 5 samples |

### Key Findings
- RADS achieves an F1 of 0.871 with only 5 annotated samples, approaching the full-shot upper bound (0.900) and far outperforming other active learning methods.
- Traditional uncertainty sampling degrades significantly under class imbalance (F1 only 0.545), as it tends to select outliers at the distribution edges.
- While BatchBALD achieves high target domain F1 (0.849), it severely sacrifices source domain performance (CHIFIR F1 drops to 0.500), showing the largest transfer gap.
- RADS is the only method that maintains good performance in both the target and source domains, achieving true dual-domain adaptation.

## Highlights & Insights
- **Modeling sample selection as an RL problem** is ingenious—compared to greedy active learning methods, an RL agent can optimize the overall utility of the selected subset from a global perspective, naturally balancing informativeness, class ratios, and diversity.
- The **prior-aware utility function** is simple yet effective; a single hyperparameter $\rho$ controls the level of class balance, making it directly transferable to any low-resource classification task.
- Calculating redundancy in the **representation space** is a valuable takeaway—it measures distance between samples in the MC dropout predictive distribution space rather than comparing raw text directly.

## Limitations & Future Work
- The experimental dataset scale is small (CHIFIR 283, PIFIR 201); effectiveness on larger datasets remains to be verified.
- Training the RL sampler introduces additional computational costs and hyperparameter tuning, which may be less cost-effective than simple methods in certain scenarios.
- Currently, only binary classification (disease presence/absence) is validated; the prior-aware utility function needs extension for multi-class scenarios.
- Sharing the RL agent's selection policy across multiple transfer tasks could be considered to further amateurize training costs.

## Related Work & Insights
- **vs Uncertainty Sampling**: Uncertainty sampling considers only a single metric and easily selects outliers under imbalance and domain shift; RADS avoids this through multi-signal fusion and RL optimization.
- **vs BatchBALD**: BatchBALD selects batches via joint mutual information, theoretically accounting for dependencies but lacks a class balance mechanism, leading to severe source domain performance degradation.
- **vs LM-DPP**: DPP models both uncertainty and diversity simultaneously, but its fixed weight scheme is less flexible than the adaptive policy of RL.

## Rating
- Novelty: ⭐⭐⭐⭐ RL-driven sample selection is not entirely new in active learning, but the combination with prior-aware utility and redundancy penalty is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared against 6 baselines with complete multi-directional transfer experiments, though dataset scale is small.
- Writing Quality: ⭐⭐⭐⭐ Methodological formalization is clear, and experimental analysis is detailed.
- Value: ⭐⭐⭐⭐ High practical application value in medical low-resource NLP; the method is generalizable to other fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)
- [\[ACL 2026\] Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech](forgotten_words_benchmarking_neobert_for_dementia_detection_in_low-resource_conv.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)

</div>

<!-- RELATED:END -->
