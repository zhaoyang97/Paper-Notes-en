---
title: >-
  [Paper Note] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings
description: >-
  [ACL 2026][Medical Imaging][Reinforcement Learning] This paper proposes RADS (Reinforcement Adaptive Domain Sampling), a reinforcement learning-based sample selection strategy that significantly improves cross-domain dis…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Reinforcement Learning"
  - "Sample Selection"
  - "Transfer Learning"
  - "Class Imbalance"
  - "Clinical NLP"
date: 2026-05-08
content_hash: 5ea63119d8c99a13
---

# RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings

**Conference**: ACL 2026
**arXiv**: [2604.20256](https://arxiv.org/abs/2604.20256)  
**Code**: [https://github.com/Wei-0808/RADS](https://github.com/Wei-0808/RADS)  
**Area**: Medical Imaging
**Keywords**: Reinforcement Learning, Sample Selection, Transfer Learning, Class Imbalance, Clinical NLP

## TL;DR
This paper proposes RADS (Reinforcement Adaptive Domain Sampling), a reinforcement learning-based sample selection strategy that significantly improves cross-domain disease detection under extreme low-resource and class-imbalanced clinical settings by intelligently selecting a small number of target-domain samples for annotation and joint fine-tuning.

## Background & Motivation

**Background**: NLP tasks on clinical text rely heavily on high-quality annotated data, yet annotation costs in the medical domain are prohibitively high (requiring expert clinicians), and many disease conditions are rare, resulting in a severe scarcity of positive examples. Transfer learning is the primary strategy for low-resource scenarios, reducing annotation requirements by training on a source domain and transferring to a target domain.

**Limitations of Prior Work**: Conventional active learning methods—such as uncertainty sampling and diversity sampling—perform poorly under extreme low-resource and class-imbalanced conditions. Uncertainty sampling tends to select outliers at the distribution boundary rather than genuinely informative samples; diversity sampling optimizes only a single objective and cannot simultaneously account for sample informativeness and redundancy. Furthermore, the high heterogeneity of clinical reports (CT, PET, cytology reports differ substantially in terminology and phrasing) further complicates cross-domain transfer.

**Key Challenge**: Given an extremely limited annotation budget (e.g., only 5 samples), how to select the most valuable samples from an unlabeled target domain such that the jointly fine-tuned model performs well on both source and target domains.

**Goal**: To design an adaptive sample selection strategy that simultaneously accounts for informativeness, class balance, and sample diversity.

**Key Insight**: The authors formulate sample selection as a sequential decision-making problem and train a reinforcement learning agent to learn an optimal selection policy, enabling adaptive balancing of informativeness, class proportion, and redundancy.

**Core Idea**: A Dueling DQN agent is trained to select the optimal sample subset from the target domain for annotation and fine-tuning, guided by BALD mutual information in conjunction with a prior-aware utility function and a redundancy penalty mechanism.

## Method

### Overall Architecture
The RADS framework consists of three stages: (1) **Active Learner Training**: fine-tune a ClinicalBERT classifier on the source domain, then compute uncertainty signals for unlabeled target-domain samples via MC Dropout; (2) **Prior-Aware Utility Computation**: construct a utility function that jointly considers informativeness and class balance by combining BALD mutual information scores with pseudo-label class weights; (3) **RL Sampler Training**: train a Dueling DQN to learn a selection policy that maximizes utility while penalizing redundant selections.

### Key Designs

1. **MC Dropout-Based BALD Informativeness Estimation**

    - **Function**: Quantifies the informativeness of each unlabeled target-domain sample to the model.
    - **Mechanism**: With dropout active, $K$ stochastic forward passes are performed per sample. The predictive entropy ($\mathrm{PE}$) and the mean entropy of individual predictions ($\mathrm{EE}$) are computed; their difference yields the BALD mutual information $\mathrm{MI}(x) = \mathrm{PE}(x) - \mathrm{EE}(x)$. High MI indicates overall model uncertainty with high disagreement among sub-models, identifying the most informative samples.
    - **Design Motivation**: BALD is more robust than plain uncertainty sampling because it disentangles epistemic uncertainty (model ignorance) from aleatoric uncertainty (inherent sample ambiguity), selecting only samples with high epistemic uncertainty.

2. **Prior-Aware Utility Function**

    - **Function**: Introduces class-balance control on top of informativeness.
    - **Mechanism**: The class prior $\hat{\pi}_+$ of the target domain is estimated via pseudo-labels; class weights $w_+ = \rho / \mathrm{clip}(\hat{\pi}_+)$ are then computed, yielding the final utility $u(x) = \widetilde{\mathrm{MI}}(x) \cdot w_{y(x)}$. The hyperparameter $\rho$ governs the trade-off between class balance and informativeness.
    - **Design Motivation**: Under extreme class imbalance, pure informativeness-based selection can produce severely skewed sample sets. The prior-aware utility corrects this bias through weighted adjustment.

3. **Redundancy-Aware Sampler Based on Dueling DQN**

    - **Function**: Learns a sequential selection policy that maximizes utility while avoiding redundant samples.
    - **Mechanism**: The state vector comprises the sample's mean log-probability, predictive entropy, BALD score, and budget utilization rate. The reward is $r_t = u(x_t) - \lambda \cdot \mathrm{Red}(x_t, S_t)$, where redundancy $\mathrm{Red}$ is measured by the nearest-neighbor distance to already-selected samples in the predictive representation space. A Dueling DQN architecture is trained via $\epsilon$-greedy exploration to learn the optimal Q-function.
    - **Design Motivation**: Conventional methods evaluate each sample independently and cannot account for inter-sample interactions. The RL agent dynamically adjusts selection criteria during sequential decision-making, naturally avoiding redundancy.

### Loss & Training
The active learner is trained with standard cross-entropy on the source domain. The RL sampler is trained with a TD loss on the Dueling DQN, complemented by an experience replay buffer and a target network. After sample selection, ClinicalBERT is jointly fine-tuned on the source domain and the newly annotated target-domain samples.

## Key Experimental Results

### Main Results (CHIFIR → PIFIR Transfer, 5 Samples Selected)

| Strategy | PIFIR F1 | PIFIR ROC-AUC | CHIFIR F1 | Transfer Gap ΔF1 |
|----------|----------|---------------|-----------|-----------------|
| Random | 0.639 | 0.813 | 0.746 | — |
| Uncertainty | 0.545 | 0.830 | 0.824 | 0.278 |
| Diversity | 0.638 | 0.809 | 0.800 | 0.162 |
| BatchBALD | 0.849 | 0.783 | 0.500 | -0.349 |
| **RADS** | **0.871** | **0.833** | **0.750** | **-0.121** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| RADS (full) | F1=0.871, AUC=0.833 | Complete model |
| w/o redundancy penalty | Approaches uncertainty-sampling level | Redundancy penalty is critical for diversity |
| w/o prior-aware utility | Increased class skew | Necessary under imbalanced conditions |
| Full-shot (all target-domain labels) | F1=0.900 | Upper bound; RADS approaches it with only 5 samples |

### Key Findings
- RADS achieves F1=0.871 with only 5 annotated samples, approaching the full-annotation upper bound (0.900) and substantially outperforming all other active learning baselines.
- Conventional uncertainty sampling degrades severely under class imbalance (F1=0.545), as it tends to select distributional outliers.
- BatchBALD achieves a relatively high target-domain F1 (0.849) but severely sacrifices source-domain performance (CHIFIR F1 drops to 0.500), exhibiting the largest transfer gap.
- RADS is the only method that maintains strong performance on both target and source domains, achieving genuine dual-domain adaptation.

## Highlights & Insights
- **Formulating sample selection as an RL problem** is an elegant design choice—compared to greedy active learning, the RL agent can optimize the overall utility of the selected subset from a global perspective, naturally balancing informativeness, class proportion, and diversity.
- **The prior-aware utility function** is concise yet effective: a single hyperparameter $\rho$ controls the degree of class balancing, making it directly transferable to any low-resource classification task.
- The approach of computing redundancy in the representation space is worth noting—rather than comparing raw text, inter-sample distances are measured in the MC Dropout predictive distribution space.

## Limitations & Future Work
- The experimental datasets are relatively small (CHIFIR: 283 reports; PIFIR: 201 reports); performance on larger-scale datasets remains to be validated.
- Training the RL sampler incurs additional computational overhead and hyperparameter tuning, which may not be cost-effective compared to simpler methods in certain scenarios.
- Evaluation is currently limited to binary classification (disease present/absent); the prior-aware utility function requires extension for multi-class settings.
- Sharing the RL agent's selection policy across multiple transfer tasks could be explored to further amortize training costs.

## Related Work & Insights
- **vs. Uncertainty Sampling**: Uncertainty sampling optimizes a single metric and is prone to selecting outliers under imbalance and domain shift; RADS avoids this through multi-signal fusion and RL-based optimization.
- **vs. BatchBALD**: BatchBALD selects batches via joint mutual information and theoretically accounts for inter-sample dependencies, but lacks a class-balance mechanism, leading to severe source-domain performance degradation.
- **vs. LM-DPP**: DPP jointly models uncertainty and diversity, but its fixed weighting scheme is less flexible than the adaptive policy learned by the RL agent.

## Rating
- **Novelty**: ⭐⭐⭐⭐ RL-driven sample selection is not entirely new in active learning, but the combination with the prior-aware utility function and redundancy penalty is a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six baselines are compared with comprehensive multi-directional transfer experiments, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ The method section is formally rigorous and the experimental analysis is detailed.
- **Value**: ⭐⭐⭐⭐ Practically applicable to low-resource clinical NLP and generalizable to other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)
- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)

</div>

<!-- RELATED:END -->
