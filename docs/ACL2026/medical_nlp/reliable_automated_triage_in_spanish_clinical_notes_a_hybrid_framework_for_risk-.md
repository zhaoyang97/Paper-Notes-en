---
title: >-
  [Paper Note] Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification
description: >-
  [ACL2026][Medical NLP][Clinical NLP] This paper addresses early HIV suspicion identification in Spanish clinical notes by proposing a dual-validation selective triage framework. It utilizes MCP to handle aleatoric uncert…
tags:
  - "ACL2026"
  - "Medical NLP"
  - "Clinical NLP"
  - "Selective Classification"
  - "Uncertainty Quantification"
  - "HIV Suspicion Identification"
  - "Spanish EHRs"
date: 2026-05-08
content_hash: 8ea029a03b15eade
---

# Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification

**Conference**: ACL2026  
**arXiv**: [2605.21256](https://arxiv.org/abs/2605.21256)  
**Code**: https://github.com/romorale/mil_uq_public  
**Area**: Clinical NLP / Medical Text Triage  
**Keywords**: Clinical NLP, Selective Classification, Uncertainty Quantification, HIV Suspicion Identification, Spanish EHRs  

## TL;DR
This paper addresses early HIV suspicion identification in Spanish clinical notes by proposing a dual-validation selective triage framework. It utilizes MCP to handle aleatoric uncertainty and MCMD geometric veto to handle epistemic uncertainty, automatically processing 67.7% of cases while achieving 0.982 Clear $F_2$ under strict safety constraints.

## Background & Motivation
**Background**: Medical NLP often treats Electronic Health Record (EHR) classification as a deterministic binary problem, reporting aggregate metrics such as AUROC, F1, or $F_2$. For early HIV suspicion identification, NLP systems can assist in filtering out clearly negative cases, thereby preserving clinical resources for patients who require physician clinical judgment.

**Limitations of Prior Work**: Clinical suspicion is not equivalent to a final serological diagnosis; rather, it is a partially observable construct determined by physician documentation, testing behaviors, and the context of the medical record. If a model is forced to provide a binary classification for all cases, it produces overconfident errors on ambiguous narratives, out-of-distribution (OOD) cases, and minority class samples. In clinical triage, such errors are far more severe than traditional benchmark performance drops.

**Key Challenge**: Clinical automation requires high coverage to reduce manual workload, yet medical safety necessitates that the system knows when to defer. Standard uncertainty quantification (UQ) methods often only examine predictive entropy, conflating textual ambiguity with OOD risks, which leads to either deferring too many true positives or dangerously automating anomalous cases.

**Goal**: The authors aim to transform forced classification into trinary triage with rejection: Clear Negative, Clear Positive, or Defer. The system does not pursue automated judgment for all samples but seeks sufficient reliability within the automatable subdomain while returning high-risk samples to clinicians.

**Key Insight**: The paper decomposes uncertainty into two categories: MCP handles probability-level aleatoric ambiguity, and MCMD handles epistemic anomalies in the latent space. Only cases that pass both checks concurrently are permitted for automated output.

**Core Idea**: Clinical narratives must satisfy both the probabilistic safety boundaries of conformal prediction and the geometric in-distribution checks of Mahalanobis distance. Using a dual-veto mechanism, unreliable binary predictions are converted into risk-controlled selective triage.

## Method
The core of this paper is not the proposal of a larger text encoder, but rather placing a clinical text classifier inside a safety shell capable of rejection. The model first processes long medical records using Spanish biomedical RoBERTa and MIL, then utilizes specialized uncertainty post-processing to determine which cases can be automated.

### Overall Architecture
The input consists of multiple Spanish clinical notes for a single patient, with the task being to identify the presence of clinical HIV suspicion. The system segments the medical records into overlapping chunks of up to 64 segments, each containing 384 tokens, encoded by `PlanTL-GOB-ES/bsc-bio-ehr-es`. Subsequently, a gated attention MIL aggregates these into patient-level representations. A classification head provides binary predictions, while the post-processing module decides whether to defer.

Training and evaluation utilize a single-center anonymized EHR cohort from HUFA hospital, totaling 13,642 patients and 63,802 clinical notes. The suspected cohort is filtered by HIV serological test requests and EuroTEST indicator conditions, while the non-suspected cohort consists of patients with no relevant testing history. The authors also proactively remove HIV abbreviations and direct testing recommendations to reduce explicit leakage.

The final decision logic follows a strict intersection: the system outputs $\hat{y}$ only when the size of the MCP prediction set is 1 AND the Mahalanobis distance from the sample to the local centroid of the predicted class does not exceed the class threshold $\tau_{dist}(\hat{y})$; otherwise, it outputs "defer."

### Key Designs
1. **Mondrian Conformal Prediction for Aleatoric Uncertainty**:
	- **Function**: Determines whether textual evidence is sufficiently clear in a probabilistic sense, avoiding forced judgments on ambiguous cases.
	- **Mechanism**: Temperature scaling is first applied to validation set logits to mitigate overconfidence, followed by the construction of a prediction set $\Gamma^{\alpha}(x)$ with a risk tolerance of $\alpha$ using MCP. If the set contains multiple classes or is empty, the evidence is deemed insufficient or conflicting, and the system defers.
	- **Design Motivation**: Clinical narratives may be inherently incomplete or ambiguously phrased; such ambiguity cannot be resolved simply by adjusting classification thresholds. Conformal prediction explicitly exposes risk levels to hospital administrators.

2. **Multi-Centroid Mahalanobis Distance for Epistemic Uncertainty**:
	- **Function**: Intercepts OOD cases where the softmax is confident but the latent representation is anomalous.
	- **Mechanism**: In an $L_2$ normalized latent space, each class uses k-means to automatically select multiple local centroids, with the minimum $K$ determined by an inertia gain lower than 0.05. The anomaly score of a sample is the minimum Mahalanobis distance to any centroid of the predicted class $d_M(x,\hat{y})=\min_k\sqrt{r_{\hat{y},k}^\top\Sigma^{-1}r_{\hat{y},k}}$, where a global precision matrix is estimated from residuals using OAS shrinkage.
	- **Design Motivation**: HIV suspected cases are highly heterogeneous; a single Gaussian center would overestimate the variance of minority classes. Multi-centroid approaches preserve phenotypic diversity while avoiding unstable high-dimensional local covariance estimation.

3. **Asymmetric Clinical Evaluation and Adjustable Operational Dial**:
	- **Function**: Evaluates the system using metrics aligned with early infection screening scenarios rather than standard F1 scores.
	- **Mechanism**: The authors utilize $F_2$ to emphasize penalties for false negatives, ECE for calibration, and coverage/TPDR/AURC for triage efficiency. They also designed a Custom Risk-Kappa, setting the penalty for automated false negatives to 1.0, false positives and true positive deferrals to 0.5, and true negative deferrals to 0.25.
	- **Design Motivation**: Missing an early HIV diagnosis delays ART and increases transmission risk, the cost of which far exceeds an unnecessary test. Therefore, the evaluation function must reflect the asymmetry of error types.

### Loss & Training
The encoder side compares two MIL architectures. Standard MIL uses Label-Smoothed Focal Loss to handle class imbalance, complemented by R-Drop to constrain KL consistency between two forward passes. MD-SN MIL utilizes a Mahalanobis Distance + Spectral Normalization structure combined with Logit Adjustment, incorporating class priors into the logits to prevent focal loss from distorting probability calibration. SN is applied to the dense layers of the MIL head to obtain a smoother feature space suitable for geometric veto.

UQ backends include MC Dropout, 10-fold CV Deep Ensembles, and deterministic MD-SN. Threshold fitting employs out-of-fold calibration to prevent optimistic bias caused by tuning thresholds directly on training samples.

## Key Experimental Results

### Main Results
Main results evaluate forced binary versus selective screening under a strict risk tolerance of $\alpha=0.01$. The most critical conclusion is that while scores appear decent under forced binary classification, once safety constraints are introduced, only the dual-veto mechanism isolates a reliable subdomain suitable for clinical automation.

| Architecture / UQ | Binary $F_2$ | Clear $F_2$ | Binary ECE | Clear ECE | Coverage | TPDR | AURC | Risk-Kappa |
|-----------|--------------|-------------|------------|-----------|----------|------|------|------------|
| Encoder baseline + CV Ensemble | 0.769 | 0.973 | 0.077 | 0.031 | 47.8% | 50.5% | - | - |
| Standard MIL + MD-SN | 0.813 | 0.966 | 0.028 | 0.026 | 70.7% | 31.9% | 0.012 | 0.601 |
| MD-SN MIL + CV Ensemble | 0.821 | 0.982 | 0.022 | 0.021 | 67.7% | 33.2% | 0.006 | 0.576 |

### Ablation Study
The ablation primarily verifies the necessity of the dual-veto and whether $\alpha$ can serve as an operational dial.

| Configuration | Coverage | Clear $F_2$ | Risk-Kappa | Description |
|------|----------|-------------|------------|------|
| Aleatoric Only / MCP | 92.6% | 0.902 | 0.751 | Detects probabilistic ambiguity but misses geometric anomalies |
| Epistemic Only / MCMD | 98.8% | 0.831 | 0.788 | High coverage but significant safety drop |
| Standard Uncertainty | 99.3% | 0.824 | 0.788 | Traditional entropy thresholds cause inflated coverage |
| Dual Veto / Hybrid | 91.3% | 0.913 | 0.755 | Clear $F_2$ is 0.089 higher than standard UQ, with non-overlapping CIs |

| Model | $\alpha$ | Coverage | TPDR | Clear $F_2$ | Clear ECE | Risk-Kappa |
|------|----------|----------|------|-------------|-----------|------------|
| Standard MIL | 0.01 | 63.5% | 29.6% | 0.979 | 0.009 | 0.546 |
| Standard MIL | 0.02 | 75.4% | 19.4% | 0.954 | 0.012 | 0.644 |
| Standard MIL | 0.05 | 91.1% | 7.0% | 0.902 | 0.015 | 0.747 |
| MD-SN MIL | 0.01 | 67.7% | 33.2% | 0.982 | 0.021 | 0.576 |
| MD-SN MIL | 0.02 | 77.4% | 24.8% | 0.969 | 0.028 | 0.665 |
| MD-SN MIL | 0.05 | 91.3% | 10.2% | 0.913 | 0.029 | 0.755 |

### Key Findings
- Forced binary evaluation masks risks. For instance, the MD-SN architecture with MC Dropout reaches a maximum Binary $F_2$ of 0.839, but its calibration ECE degrades to 0.044, indicating it is unsuitable for clinical automation.
- The benefit of the dual-veto lies in safety rather than a significant advantage in Risk-Kappa. The paper notes significant overlap in Risk-Kappa confidence intervals, but the improvement in Clear $F_2$ over standard uncertainty is statistically robust.
- $\alpha$ serves as a hospital operational dial. Relaxing $\alpha$ from 0.01 to 0.05 for MD-SN MIL increases coverage from 67.7% to 91.3%, while Clear $F_2$ remains at 0.913 and TPDR drops to 10.2%.
- Feature attribution shows that automated false positives are often driven by EuroTEST guideline-concordant indicator conditions like HCV or tuberculosis, reflecting the partial observability of EHR labeling and testing behaviors.

## Highlights & Insights
- The greatest value of this paper is shifting clinical NLP from "classification accuracy" to "automatable subdomains." This is closer to real-world deployment challenges than simple model stacking.
- The decoupling of aleatoric and epistemic uncertainty is highly practical: MCP addresses narrative ambiguity while MCMD addresses OOD anomalies. Since their corresponding clinical failure modes differ, they require intersection rather than substitution.
- While Custom Risk-Kappa is task-specific, it forces a direct confrontation with the asymmetric costs of medical errors. Similar designs can be transferred to pharmacovigilance, ER triage, or surgical risk screening by redefining the penalty matrix.
- Choosing a deterministic spectrally normalized encoder over a generative LLM is a pragmatic deployment choice. In high-risk medical scenarios, calibration, rejectability, and interpretability are often more critical than generative capabilities.

## Limitations & Future Work
- Custom Risk-Kappa relies heavily on early infectious disease screening assumptions. Transferring this to tasks with high physical intervention risks, like tumor biopsies, would significantly change the cost of false positives, requiring new health-economic validation.
- Static geometric thresholds in MCMD may be affected by temporal or demographic shifts. If hospital departments, documentation styles, or patient populations change, $\tau_{dist}$ requires recalibration.
- Generative LLMs were not included in core experiments. The authors argue that reliable UQ for auto-regressive models remains unresolved, justifying the choice of an encoder, which limits the direct applicability of conclusions to current LLM medical assistants.
- Empirical data is derived from a single center (HUFA). HIV suspicion depends heavily on clinical writing habits and testing thresholds; multi-institution validation is a necessary step before deployment.

## Related Work & Insights
- **vs Standard Biomedical NLP Classifiers**: Standard classifiers optimize forced binary metrics; this work focuses on Clear $F_2$, coverage, and TPDR after selective screening, which better aligns with clinical workflows.
- **vs MC Dropout / Deep Ensembles**: MC Dropout is computationally light but lacks calibration; Deep Ensembles offer strong safety but have 10x the computation/storage burden. MD-SN attempts to achieve ensemble-like geometric uncertainty with a single forward pass.
- **vs Traditional Selective Classification**: Traditional methods often use entropy thresholds. This paper demonstrates that such standard UQ would automate 99.3% of samples but with a Clear $F_2$ of only 0.824, proving single thresholds cannot cover complex clinical risks.
- **vs Medical LLM Triage**: Instead of generating explanations, this work emphasizes rejectability and calibration. An insight for medical LLMs is that proving when a system should *not* answer may be more critical than expanding its answering capacity.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The dual-veto components are based on established methods, but combining MCP and multi-centroid Mahalanobis veto into a clinical risk triage framework is robust.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Metrics, ablations, confidence intervals, and threshold analyses are complete; the limitation is the single-center data without cross-institutional validation.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definition and clinical motivation are clear; tables are information-dense but the overall narrative is smooth.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for high-risk medical NLP deployment, particularly the system design philosophy of "do not force the model to answer all cases."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2026\] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](../../NeurIPS2025/medical_nlp/cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)

</div>

<!-- RELATED:END -->
