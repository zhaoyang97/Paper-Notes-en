---
title: >-
  [Paper Note] SConU: Selective Conformal Uncertainty in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][conformal prediction] SConU introduces significance testing into the conformal uncertainty framework of LLMs for the first time. By constructing two types of conformal p-values, it identifies and filters out uncertainty data outliers that violate the exchangeability assumption, thereby achieving strict control over the miscoverage rate in both single-domain and cross-domain QA scenarios.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "conformal prediction"
  - "exchangeability"
  - "p-value"
  - "miscoverage rate"
  - "QA"
date: 2026-05-08
content_hash: 1af08a40333ac833
---

# SConU: Selective Conformal Uncertainty in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2504.14154](https://arxiv.org/abs/2504.14154)  
**Code**: [Zhiyuan-GG/SConU](https://github.com/Zhiyuan-GG/SConU)  
**Area**: LLM Reliability, Uncertainty Quantification  
**Keywords**: conformal prediction, exchangeability, p-value, miscoverage rate, QA  

## TL;DR

SConU introduces significance testing into the conformal uncertainty framework of LLMs for the first time. By constructing two types of conformal p-values, it identifies and filters out uncertainty data outliers that violate the exchangeability assumption, thereby achieving strict control over the miscoverage rate in both single-domain and cross-domain QA scenarios.

## Background & Motivation

- **Application of Conformal Prediction in LLMs**: Split Conformal Prediction (SCP) provides distribution-free and model-agnostic coverage guarantees. Recent studies have applied it to LLM QA tasks to achieve reliable response coverage under a user-specified risk level by associating non-conformity scores (NS) with uncertainty states.
- **Vulnerability of the Exchangeability Assumption**: Existing conformal uncertainty frameworks assume that the NS sequences of the calibration set and test samples satisfy exchangeability. However, this condition is difficult to verify and guarantee in practical NLG tasks. The authors observe that even within a single domain, multiple LLMs exhibit significant coverage anomalies (where the empirical miscoverage rate exceeds the theoretical upper bound) on the MMLU-Pro dataset.
- **Severity of the Cross-Domain Problem**: When the calibration set and test set originate from different domains (e.g., calibrating the mathematics domain using a health domain), the miscoverage rate deviates severely from the target value. This occurs because the uncertainty distributions of different models vary dramatically across disciplines.
- **Limitations of Prior Work**: Existing frameworks manually remove samples from the calibration set whose generation space does not contain the correct answer, which limits the number of test samples that can be processed and compromises statistical rigor.

## Method

### Overall Architecture

The SConU workflow consists of: (1) deploying the LLM and the calibration set, then calculating the minimum manageable risk level $\alpha_l$; (2) performing a significance test on each test sample to determine whether its uncertainty aligns with the calibration data distribution; (3) refusing to answer if the conformal p-value is too low, which indicates a violation of exchangeability; (4) executing the conformal procedure for test samples that pass the test, providing finite-sample coverage guarantees.

### Key Designs

1. **Basic Conformal p-value (SConU)**: For a test sample $x_{N+1}$, the basic conformal p-value is constructed as $p_{N+1} = \frac{1 + \sum_{i=1}^{N} \mathbf{1}\{u_i \geq u_{N+1}\}}{N+1}$, where $u_i$ represents the uncertainty measured by Predictive Entropy (PE). This p-value measures the relative position of the test sample's uncertainty within the calibration set.
2. **Enhanced Conformal p-value (SConU-Pro)**: A prediction state condition is incorporated into the counting criterion: $p'_{N+1} = \frac{1 + \sum_{i=1}^{N} \mathbf{1}\{u_i \geq u_{N+1}, y_i^* \in E(x_i, \mathcal{D}_{cal}, \alpha)\}}{N+1}$. This filters out the interference from calibration samples that cannot cover the correct answer under the risk level $\alpha$.
3. **Derivation of Minimum Risk Level**: Instead of removing any calibration samples, the minimum risk level that the calibration set can manage is derived as $\alpha_l = N L_N(1) / (N+1)$. That is, the proportion of candidate sets that do not contain the correct answer determines the lower bound of the controllable risk.

### Loss & Training

As a training-free method, SConU does not involve traditional loss functions. The core is a statistical test: under a significance level $\delta$, if the conformal p-value is lower than $\delta$, the null hypothesis is rejected (i.e., the test sample is considered an uncertainty outlier), and the system refuses to answer.

## Experiments

### Main Results: MMLU-Pro Single-Domain/Cross-Domain Coverage Management

| Discipline | Metric | No OD (Basic ConU) | SConU | SConU-Pro |
|------|------|-----------|-------|-----------|
| Health | EMR @ α=0.1 | 0.12±0.04 (violation) | 0.09±0.02 | 0.08±0.01 |
| Economics | EMR @ α=0.1 | 0.15±0.06 (violation) | 0.09±0.03 | 0.09±0.02 |
| Cross-domain (Health→Math) | EMR @ α=0.28 | 0.45 (severe violation) | 0.26 | 0.24 |

The experiments cover 8 LLMs (such as LLaMA-3.1-8B, Qwen2.5-14B, etc.), reporting the mean and standard deviation of 100 random trials.

### Ablation Study: Sampling Size Calibration

| Dataset | LLM | β=0.1 | β=0.2 | β=0.3 |
|--------|-----|-------|-------|-------|
| TriviaQA | LLaMA-3.2-3B | 0.088±0.015 | 0.177±0.011 | 0.273±0.019 |
| MedMCQA | LLaMA-3.1-8B | 0.087±0.006 | 0.177±0.038 | 0.197±0.009 |
| TriviaQA | Qwen2.5-14B | 0.084±0.020 | 0.173±0.008 | 0.173±0.008 |

Sampling size calibration validates the necessity of keeping the calibration set complete: Eq.(4) guarantees that the correct answer is covered in sampling with a probability of $\geq 1-\beta$.

### Key Findings

- Even within a single domain, the basic ConU framework frequently exhibits violations where EMR exceeds the risk level; SConU effectively controls the EMR below the target by filtering out outliers.
- SConU-Pro further improves the accuracy of outlier detection by considering the prediction state of the calibration data itself.
- Maintaining the completeness of the calibration set (without manually removing samples that do not contain correct answers) allows the calibration set to cover a wider domain distribution.
- The choice of uncertainty metrics (PE vs. SE vs. LN+SC) has a significant impact on conditional coverage performance.
- A large amount of semantic redundancy exists in the prediction set, which poses a need for de-duplication in human-computer interactive QA applications.

## Highlights & Insights

- Introduces significance testing into the conformal uncertainty framework of LLMs for the first time to detect exchangeability violations.
- Proposes formal statistical validation of two conformal p-values, establishing a solid theoretical foundation.
- The design philosophy of maintaining the completeness of the calibration set is novel and practical, deriving the minimum manageable risk level.
- Extensive experiments across 8 LLMs and multiple QA datasets validate the generalizability of the proposed method.

## Limitations & Future Work

- The conformal p-value test is inherently conservative, which may excessively reject answerable test samples, thereby reducing answer coverage.
- The power of the p-value test is limited when the calibration set size is small.
- Currently validated primarily on MCQA and open-ended QA; its applicability to more complex NLG tasks (such as summarization and translation) remains unverified.
- Conditional coverage remains unachievable in most NLG scenarios and can only be approximated.

## Related Work & Insights

- **Conformal Prediction in LLMs**: Various non-conformity score designs have been proposed by ConU (Wang et al., 2024c), CONU-MCQA (Quach et al., 2024), etc.
- **Conformal Prediction under Distribution Shift**: Conformal inference under covariate shift is discussed by Tibshirani et al. (2019) and Barber et al. (2023).
- **LLM Uncertainty Estimation**: Uncertainty measurement methods such as SE (Kuhn et al., 2023) and PE (Kadavath et al., 2022).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)
- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)

</div>

<!-- RELATED:END -->
