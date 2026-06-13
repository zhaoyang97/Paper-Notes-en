---
title: >-
  [Paper Note] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
description: >-
  [ACL 2026][Medical NLP][Multimodal clinical time-series] This paper proposes the OPL-MT-MNAR framework, which learns dynamic representations of ICU patients from the "information carried by the missingness patterns thems…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Multimodal clinical time-series"
  - "informative missingness"
  - "offline reinforcement learning"
  - "Bayesian filtering"
  - "ICU treatment strategies"
date: 2026-05-08
content_hash: f77695f4c079a163
---

# Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.21235](https://arxiv.org/abs/2604.21235)  
**Code**: [GitHub](https://github.com/CausalMLResearch/OPL-MT-MNAR)  
**Area**: Medical Imaging  
**Keywords**: Multimodal clinical time-series, informative missingness, offline reinforcement learning, Bayesian filtering, ICU treatment strategies

## TL;DR
This paper proposes the OPL-MT-MNAR framework, which learns dynamic representations of ICU patients from the "information carried by the missingness patterns themselves" in structured data and clinical text. By integrating an MNAR-aware multimodal encoder, Bayesian filtered latent states, and offline policy learning, the framework achieves sepsis treatment policies that outperform clinician behavior (FQE 0.679 vs 0.528).

## Background & Motivation

**Background**: Electronic Health Records (EHR) containing structured data (vital signs, lab tests) and clinical text (nursing notes, reports) serve as rich data sources for learning patient dynamic representations to support outcome prediction and sequential treatment decision-making. Significant research has been conducted on offline RL for ICU sepsis treatment, but most studies treat clinical observations as pre-processed complete data.

**Limitations of Prior Work**: Two critical features of clinical data are often overlooked: (1) **The observation process itself is informative** (informative missingness) — more severely ill patients are monitored more frequently, meaning missingness patterns reflect underlying health states and are missing-not-at-random (MNAR); (2) **Observation patterns vary across modalities** — vital signs are relatively routine, lab tests require physician orders, and text records depend on clinician documentation behavior. These differences evolve over time throughout a patient's trajectory.

**Key Challenge**: Existing methods either ignore missingness information or restrict the handling of missingness to structured time-series (e.g., GRU-D), failing to utilize missingness patterns as informative signals in a joint multimodal and temporal setting. In particular, the clinical text observation process (when nursing notes are recorded and how documentation frequency changes) is entirely ignored.

**Goal**: This work aims to construct a patient representation learning framework that explicitly utilizes multimodal informative missingness to support downstream offline treatment policy optimization and outcome prediction.

**Key Insight**: Three strong signals were identified from real-world ICU data: (a) higher illness severity is associated with more intensive monitoring; (b) high-acuity patients are more likely to have text updates; (c) the temporal availability of different modalities evolves distinctly. These observation patterns contain vital information regarding patient states.

**Core Idea**: The observation process (missingness patterns of structured data + documentation behavior patterns of text) is treated as an explicit feature input. Patient representations are constructed through MNAR-aware encoding, Bayesian filtering, and action-conditioned latent states.

## Method

### Overall Architecture
The framework follows a two-stage approach: **Stage 1** focuses on learning patient state representations. It first obtains a unified representation $\phi_h$ via an MNAR-aware multimodal encoder, then maintains a latent belief state $z_h$ through variational inference Bayesian filtering, which are combined into a posterior patient state $s_h = g_\theta(\phi_h, z_h)$. **Stage 2** utilizes $s_h$ for offline policy optimization (IQL) and outcome prediction.

### Key Designs

1. **MNAR-aware structured data encoding (GRU-D extension)**:

    - **Function**: Extract representations from irregularly sampled structured observations while preserving information within the missingness patterns.
    - **Mechanism**: On top of GRU-D, explicit MNAR features (cumulative observation counts, missingness rates, and observation frequency within a window) are added and directly input into the GRU gate updates. When a variable is missing for an extended period, its value is gradually decayed toward the empirical mean using a learned decay factor.
    - **Design Motivation**: Standard GRU-D models only use time intervals for decay and fail to utilize the acuity information carried by the "monitoring frequency" itself.

2. **Documentation process factor and sparse text fusion (Sparse Text Fusion)**:

    - **Function**: Model the clinical text observation process and adaptively fuse text with structured representations.
    - **Mechanism**: A documentation process factor $F_h^{doc}$ is introduced, which encodes text existence, timeliness, and recent documentation density per step using an MLP, accumulated temporally via a GRU. Text representations are obtained through multi-head cross-attention using the structured representation as a query; finally, a gating mechanism controlled by $F_h^{doc}$ adaptively fuses the two modalities.
    - **Design Motivation**: Text availability is endogenous; high-acuity patients are documented more frequently. The model must distinguish between states such as "no text," "stale text," and "densely updated text," even if the underlying text content is similar.

3. **Action-conditioned latent belief state**:

    - **Function**: Capture the cumulative impact of treatment history on patient trajectories.
    - **Mechanism**: The latent state $z_{h+1} \sim p_\theta(z_{h+1}|z_h, \phi_h, a_h)$ is parameterized via a VAE, where the transition function is crucially conditioned on treatment actions. The authors provide Theorem 1: if latent state transitions are independent of actions, the gradient of the current action with respect to future rewards in the policy gradient is zero, providing no learning signal at non-terminal steps in terminal-reward settings.
    - **Design Motivation**: Observation encoding $\phi_h$ alone is insufficient for policy optimization because $\phi_h$ is a deterministic function of recorded observations. The causal effects of actions must be propagated through latent states.

### Loss & Training
The training consists of three phases: (1) pre-training the encoder (reconstruction loss includes four terms: structured values, missingness mask BCE, text embeddings, and documentation process factor + dynamics consistency loss + KL regularization); (2) training the RL policy with a frozen encoder (IQL: double Q + expectile value function + advantage-weighted behavior cloning); (3) joint fine-tuning.

## Key Experimental Results

### Main Results (Policy learning FQE)

| Method | Information | MIMIC-III | MIMIC-IV | eICU |
|------|------|-----------|----------|------|
| AI Clinician | Model-free | 0.487 | 0.491 | 0.478 |
| DDPG+Clinician | Model-free | 0.529 | 0.538 | 0.524 |
| MedDreamer | Model-based | 0.583 | 0.591 | 0.579 |
| Clinician Behavior | Behavior | 0.528 | 0.521 | 0.534 |
| **Ours (OPL-MT-MNAR)** | **MNAR+Text** | **0.679** | **0.634** | **0.604** |

### Ablation Study (MIMIC-III Building Block Study)

| Configuration | FQE | Gain (Relative to Baseline) |
|------|-----|------------|
| Baseline (MDP, no MNAR) | 0.507 | — |
| + Semi-MDP | 0.518 | +2.2% |
| + MNAR + DocProcess | **0.679** | **+33.9%** |
| + All | 0.689 | +35.9% |

### Key Findings
- **MNAR modeling is the primary contributor**: Explicit MNAR and documentation process modeling contributed to a +33.9% improvement, significantly higher than the +2.2% from Semi-MDP.
- **Text provides substantial value for policy learning**: Using structured data only yielded 0.574; adding nursing notes increased this to 0.624, with full multimodality reaching 0.679.
- **High-acuity patients benefit the most**: For the high SOFA (>10) group, clinician FQE was only 0.192, while the proposed method reached 0.344.
- **Outcome prediction AUROC 0.886**: This outperformed both GRU-D (0.844) and MedDreamer (0.867).

## Highlights & Insights
- The concept of **"missingness as signal"** is highly effective: instead of treating missingness as a nuisance to be imputed, the framework treats "what was observed, when it was observed, and how often" as direct features. This insight is applicable across many incomplete data domains.
- The **proof of theoretical necessity for action-conditioned latent states** (Theorem 1) provides a rigorous foundation for the methodology.
- The **documentation process factor** utilizes only the meta-information of the observation process rather than text content itself to regulate fusion weights, effectively decoupling "behavioral signals" from "content signals."

## Limitations & Future Work
- The study relies on offline policy evaluation (FQE) and has not undergone prospective clinical validation.
- The action space is discretized into 9 actions with 4-hour decision intervals, which limits fine-grained treatment control.
- Unrecorded information (e.g., verbal communication, bedside assessments) may lead to unobserved confounding.
- Validation was performed only on US-based ICU datasets; generalization to other countries or healthcare systems requires further investigation.

## Related Work & Insights
- **vs GRU-D**: Whereas GRU-D only handles time interval decay in structured time-series, this work extends the approach to multimodal MNAR and incorporates cumulative observation features.
- **vs MedDreamer**: While MedDreamer uses model-based RL, this work achieves higher FQE through explicit MNAR modeling without the need for a world model.
- **vs Liang et al. (2025)**: A previous work by the same team also modeled informative missingness but lacked temporal dynamics; this work introduces Bayesian filtering and action conditioning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Original framework for multimodal informative missingness, documentation behavior modeling, and action-conditioned latent states.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across three datasets with comprehensive ablation, acuity-stratified analysis, and robustness testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and highly persuasive motivational diagrams.
- Value: ⭐⭐⭐⭐ Significant practical implications for clinical AI; the "missingness as signal" approach offers broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/medical_nlp/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[ACL 2026\] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)

</div>

<!-- RELATED:END -->
