---
title: >-
  [Paper Note] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
description: >-
  [ACL 2026][Medical Imaging][Multimodal clinical time-series] This work proposes OPL-MT-MNAR, a framework that learns dynamic patient representations from ICU data by leveraging the information embedded in missingness pat…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Multimodal clinical time-series"
  - "informative missingness"
  - "offline reinforcement learning"
  - "Bayesian filtering"
  - "ICU treatment policy"
date: 2026-05-08
content_hash: 65645f0ad654e21a
---

# Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness

**Conference**: ACL 2026
**arXiv**: [2604.21235](https://arxiv.org/abs/2604.21235)
**Code**: [GitHub](https://github.com/CausalMLResearch/OPL-MT-MNAR)
**Area**: Medical Imaging
**Keywords**: Multimodal clinical time-series, informative missingness, offline reinforcement learning, Bayesian filtering, ICU treatment policy

## TL;DR
This work proposes OPL-MT-MNAR, a framework that learns dynamic patient representations from ICU data by leveraging the information embedded in missingness patterns of structured observations and clinical text. It combines an MNAR-aware multimodal encoder, Bayesian filtering for latent belief states, and offline policy learning to derive sepsis treatment policies that outperform clinician behavior (FQE 0.679 vs. 0.528).

## Background & Motivation

**Background**: Electronic health records (EHRs) contain both structured data (vital signs, laboratory tests) and clinical text (nursing notes, reports), providing rich sources for learning dynamic patient representations to support outcome prediction and sequential treatment decision-making. Offline RL has been extensively studied for ICU sepsis treatment, yet most prior work treats clinical observations as fully preprocessed and complete.

**Limitations of Prior Work**: Two critical characteristics of clinical data are consistently overlooked: (1) **the observation process itself is informative** (informative missingness)—more severely ill patients are monitored more frequently, so missingness patterns reflect underlying health status, constituting missing-not-at-random (MNAR) data; (2) **different modalities exhibit different observation patterns**—vital signs are recorded routinely, laboratory tests require physician orders, and text notes depend on clinician documentation behavior, all of which evolve over time within patient trajectories.

**Key Challenge**: Existing methods either ignore missingness information entirely or handle it only within structured time-series (e.g., GRU-D), without exploiting missingness patterns as informative signals in a joint multimodal-temporal setting. In particular, the observation process of clinical text—when nursing notes are written and how documentation frequency changes—is entirely disregarded.

**Goal**: To construct a patient representation learning framework that explicitly leverages multimodal informative missingness, supporting downstream offline treatment policy optimization and outcome prediction.

**Key Insight**: Three strong signals are identified in real ICU data: (a) monitoring intensity increases with patient acuity; (b) higher-acuity patients are more likely to have text updates; (c) temporal availability patterns differ across modalities. These observation patterns carry substantial information about patient state.

**Core Idea**: Treat the observation process—missingness patterns of structured data and documentation behavior patterns of clinical text—as explicit input features, and construct patient representations via MNAR-aware encoding, Bayesian filtering, and action-conditioned latent states.

## Method

### Overall Architecture
A two-stage framework: **Stage 1** learns patient state representations—an MNAR-aware multimodal encoder first produces a unified representation $\phi_h$, which is then combined with a latent belief state $z_h$ maintained via variational Bayesian filtering to yield the posterior patient state $s_h = g_\theta(\phi_h, z_h)$. **Stage 2** uses $s_h$ for offline policy optimization (IQL) and outcome prediction.

### Key Designs

1. **MNAR-Aware Structured Data Encoder (Extended GRU-D)**:

    - Function: Extracts representations from irregularly sampled structured observations while preserving missingness pattern information.
    - Mechanism: Extends GRU-D by explicitly incorporating MNAR features—cumulative observation counts, missingness rates, and within-window observation frequency—as direct inputs to the GRU gating updates. When a variable is missing for an extended period, its value is gradually reverted toward the empirical mean via a learned decay factor.
    - Design Motivation: Standard GRU-D uses only time intervals for decay and does not exploit the acuity information carried by monitoring frequency itself.

2. **Documentation Process Factor and Sparse Text Fusion**:

    - Function: Models the clinical text observation process and adaptively fuses text with structured representations.
    - Mechanism: Introduces a documentation process factor $F_h^{doc}$—encoded via an MLP from per-step text presence, text recency, and recent documentation density, then accumulated temporally via a GRU. Text representations are obtained via multi-head cross-attention using structured representations as queries over text embeddings; a gating mechanism controlled by $F_h^{doc}$ then adaptively fuses the two modalities.
    - Design Motivation: Text availability is endogenous—higher-acuity patients receive more frequent documentation. The model must distinguish between states such as "no text," "stale text," and "densely updated text," even when the underlying textual content is similar.

3. **Action-Conditioned Latent Belief State**:

    - Function: Captures the cumulative effect of treatment history on patient trajectories.
    - Mechanism: Parameterizes the latent state via a VAE as $z_{h+1} \sim p_\theta(z_{h+1}|z_h, \phi_h, a_h)$, where the transition function is conditioned on the treatment action. The authors prove Theorem 1: if the latent state transition is action-independent, the gradient of future rewards with respect to current actions in the policy gradient vanishes, rendering non-terminal steps entirely without learning signal under terminal reward settings.
    - Design Motivation: The observation encoding $\phi_h$ alone is insufficient for policy optimization, as it is a deterministic function of recorded observations. The causal effect of actions must be propagated through the latent state.

### Loss & Training
Three-stage training: (1) Encoder pretraining with a reconstruction loss comprising four terms—structured values, missingness mask BCE, text embeddings, and documentation process factors—plus a dynamics consistency loss and KL regularization; (2) Frozen-encoder RL training using IQL (dual Q-networks, expectile quantile value function, advantage-weighted behavior cloning); (3) Joint fine-tuning.

## Key Experimental Results

### Main Results (Policy Learning FQE)

| Method | Information | MIMIC-III | MIMIC-IV | eICU |
|--------|-------------|-----------|----------|------|
| AI Clinician | Model-free | 0.487 | 0.491 | 0.478 |
| DDPG+Clinician | Model-free | 0.529 | 0.538 | 0.524 |
| MedDreamer | Model-based | 0.583 | 0.591 | 0.579 |
| Clinician Behavior | Behavior | 0.528 | 0.521 | 0.534 |
| **OPL-MT-MNAR** | **MNAR+Text** | **0.679** | **0.634** | **0.604** |

### Ablation Study (MIMIC-III Building Block Study)

| Configuration | FQE | Gain over Baseline |
|---------------|-----|--------------------|
| Baseline (MDP, no MNAR) | 0.507 | — |
| + Semi-MDP | 0.518 | +2.2% |
| + MNAR + DocProcess | **0.679** | **+33.9%** |
| + All components | 0.689 | +35.9% |

### Key Findings
- **MNAR modeling is the largest contributor**: Explicit MNAR and documentation process modeling account for +33.9% gain, far exceeding the +2.2% from Semi-MDP.
- **Clinical text provides substantial value for policy learning**: Structured data alone yields FQE 0.574; adding nursing notes raises it to 0.624, and the full multimodal model reaches 0.679.
- **High-acuity patients benefit most**: In the high-SOFA (>10) subgroup, clinician FQE is only 0.192, while the proposed method achieves 0.344.
- **Outcome prediction AUROC of 0.886**: Outperforms GRU-D (0.844) and MedDreamer (0.867).

## Highlights & Insights
- The **"missingness as signal"** philosophy is particularly compelling: rather than imputing missing values, the framework treats *what* was observed, *when* it was observed, and *how frequently* as direct features—an insight transferable to any domain with incomplete data.
- **Theorem 1**, which establishes the theoretical necessity of action-conditioned latent states, provides rigorous justification for this architectural choice.
- The **documentation process factor** relies solely on meta-information about the observation process rather than textual content to regulate fusion weights, effectively decoupling the behavioral signal from the content signal.

## Limitations & Future Work
- Relies on offline policy evaluation (FQE) without prospective clinical validation.
- The action space is discretized into 9 bins with 4-hour decision intervals, limiting fine-grained treatment control.
- Unrecorded information—such as verbal communication and bedside assessments—may still introduce unobserved confounding.
- Validation is limited to US ICU datasets; generalization across countries and healthcare systems requires further investigation.

## Related Work & Insights
- **vs. GRU-D**: GRU-D handles only time-interval decay in structured time-series; this work extends the framework to multimodal MNAR and incorporates cumulative observation features.
- **vs. MedDreamer**: MedDreamer employs model-based RL; the proposed framework achieves higher FQE through explicit MNAR modeling without requiring a world model.
- **vs. Liang et al. (2025)**: A prior work from the same group also models informative missingness but lacks temporal dynamics; this work adds Bayesian filtering and action conditioning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A unified framework combining multimodal informative missingness, documentation behavior modeling, and action-conditioned latent states with strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, comprehensive ablations, acuity-stratified analysis, and robustness checks.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and persuasive motivation figures.
- Value: ⭐⭐⭐⭐ Practically significant for clinical AI; the "missingness as signal" paradigm has broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)
- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)

</div>

<!-- RELATED:END -->
