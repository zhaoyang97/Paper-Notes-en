---
title: >-
  [Paper Note] Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice
description: >-
  [ICML2026][Model Compression][Tabular Foundation Models] This paper finds that Tabular Foundation Models (TFMs) like TabPFN and Mitra, despite achieving high accuracy in discrete choice tasks…
tags:
  - "ICML2026"
  - "Model Compression"
  - "Tabular Foundation Models"
  - "Discrete Choice"
  - "Economic Validity"
  - "Behavioral Constraints"
  - "Policy Evaluation"
date: 2026-05-08
content_hash: 6359500e6d1986ad
---

# Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice

**Conference**: ICML2026  
**arXiv**: [2605.26559](https://arxiv.org/abs/2605.26559)  
**Code**: Not disclosed  
**Area**: Others / Tabular Foundation Models and Discrete Choice  
**Keywords**: Tabular Foundation Models, Discrete Choice, Economic Validity, Behavioral Constraints, Policy Evaluation  

## TL;DR
This paper finds that Tabular Foundation Models (TFMs) like TabPFN and Mitra, despite achieving high accuracy in discrete choice tasks, violate price-demand monotonicity and trustworthy Value of Time (VOT) estimates. It proposes a two-stage behavior adapter that embeds TFM predictions into a utility model constrained by economic theory, maintaining 100% behavioral validity while recovering most of the accuracy gains.

## Background & Motivation
**Background**: Problems such as transportation mode choice, healthcare plans, insurance schemes, and consumer goods selection can be formalized as discrete choices. Multinomial Logit (MNL) models and their extensions are commonly used in economics and transportation to predict selection probabilities because they are derived from utility maximization and naturally provide interpretable quantities like price sensitivity, VOT, and policy counterfactual analysis.

**Limitations of Prior Work**: Machine learning models, particularly pre-trained TFMs, often outperform MNL in classification accuracy. However, policy scenarios focus not only on "predictive correctness" but also on whether the model responds to interventions in price, time, and other variables in directions required by economic theory. If a model predicts an increase in demand after a price hike or derives a negative VOT, it will mislead fare setting and infrastructure investment, even with high test set accuracy.

**Key Challenge**: TFMs excel at judging "who chooses what" from tabular correlations, while economic models excel at answering "how selection probabilities should change if price or time changes." The former has an accuracy advantage, while the latter has structural guarantees. Direct distillation of TFMs loses non-linear information, and adding constraints directly to TFMs is difficult for in-context models like TabPFN and fails to provide interpretable economic coefficients.

**Goal**: The authors aim to add an auditable, interpretable shell to TFMs for policy intervention, enabling the model to possess high predictive precision, price-demand monotonicity, reasonable VOT estimates, and zero-probability constraints for unavailable options.

**Key Insight**: Instead of retraining or modifying the TFM itself, the paper treats the class probabilities output by the TFM as additional information within a constrained utility function. The structured utility term learns economic parameters independently first, after which these parameters are frozen, allowing the TFM correction term to explain the remaining predictive error from the MNL.

**Core Idea**: Let the economic model handle the direction of counterfactual response and the TFM handle observation-level non-linear discrimination, using two-stage training to prevent the TFM from "polluting" core economic coefficients like time and cost.

## Method
The proposed method consists of two parts: auditing TFMs for behavioral validity via black-box queries and fixing these issues using a two-stage behavior adapter. Its key is not proposing a more complex classifier but redefining the role of TFMs in policy choice tasks. The TFM no longer directly outputs final probabilities but acts as an observation-level information source providing correction signals to a constrained utility model.

### Overall Architecture
For each sample $x_i$ and alternative $k$, the model constructs a utility $V_k(x_i)$. This utility consists of two parts: the first part $V_k^{struct}(x_i)$ is a standard discrete choice model including alternative specific constants, time/cost coefficients, and demographic interaction terms; the second part comes from the TFM predicted probability $q_k(x_i)$, including a scalar weight term $\alpha\log q_k(x_i)$ and a small network $g_k(\mathbf{q}(x_i))$. The final choice probability is given by the Logit formula $P_k=\exp(V_k)/\sum_j\exp(V_j)$.

The authors evaluate on the Swissmetro and LPMC datasets. They first train or call Mitra and TabPFN v2 to obtain predicted probabilities for each sample, then cache these probabilities for the adapter. Behavioral auditing and adapter training do not require access to TFM internal parameters, allowing the method to be used with black-box or near-black-box TFMs.

### Key Designs
1.  **Black-box Behavioral Auditing**:
    - **Function**: Inspects whether the TFM satisfies the economic logic of discrete choice tasks in a minimally invasive manner.
    - **Mechanism**: The audit includes three types of tests. The monotonicity test increases the cost or time of an alternative by 1% of the observed range and checks if the predicted probability decreases; the VOT test estimates $VOT=\beta_{time}/\beta_{cost}$ via coefficient ratios or finite differences; the availability test calculates the average probability assigned to unavailable alternatives.
    - **Design Motivation**: Policy models must behave reasonably under input intervention, not just fit historical choices. Black-box perturbation tests expose TFMs that mistake correlation for causation, such as confounding high fees with preferences of high-income groups.

2.  **Constrained Structural Utility Term**:
    - **Function**: Encodes directional constraints from economic theory into model parameterization, ensuring that increases in price and time necessarily reduce utility.
    - **Mechanism**: The structural utility term uses MNL specifications and writes time/cost coefficients as $\beta=-\exp(\theta)$. regardless of how the optimizer updates the unconstrained variable $\theta$, the coefficient remains negative. Thus, price increases reduce utility, and VOT can be analytically calculated from structural coefficients; unavailable options receive zero probability by setting $V_k=-\infty$.
    - **Design Motivation**: Hard mathematical constructions are more reliable than post-hoc penalties. Penalties only reduce the probability of violations, whereas negative exponential parameterization provides a hard guarantee for policy interventions.

3.  **Two-stage TFM Correction Term**:
    - **Function**: Absorbs the accuracy advantage of the TFM without altering economic parameters.
    - **Mechanism**: Stage 1 fixes the correction term to zero and trains only structural utility, equivalent to standard MNL. Stage 2 freezes structural parameters and trains only $\alpha$ and the small network $g_k$, allowing TFM probabilities to explain non-linear residuals not captured by the MNL. Since policy response is determined by frozen time/cost coefficients, the TFM correction can only shift intercepts at the sample level and cannot reverse the direction of price response.
    - **Design Motivation**: In end-to-end joint training, the optimizer would quickly push signals into the TFM correction branch, causing structural coefficients to shrink toward zero and VOT to become a ratio of two near-zero numbers. Two-stage training locks in economic meaning before pursuing predictive accuracy.

### Loss & Training
The training objective is the negative log-likelihood of discrete choice. Stage 1 optimizes structural utility parameters with negative exponential parameterization to ensure negative coefficients. Stage 2 optimizes the TFM correction term while structural parameters are frozen, where $\alpha$ measures the overall reliability of TFM probabilities and $g_k$ learns alternative-level residuals from the full probability vector.

TFM probabilities are pre-computed on train, validation, and test splits. This aligns with policy analysis practices: analysts keep other individual information constant when intervening on price or time. In the adapter, "other information" includes the TFM's overall judgment of the observation, ensuring counterfactual responses are entirely controlled by the structural utility term.

## Key Experimental Results

### Main Results
Core results from Swissmetro and LPMC show that raw TFMs have high accuracy but unreliable behavioral validity; the adapter sacrifices minimal accuracy to regain full monotonicity and interpretable VOT.

| Dataset | Model | Acc ↑ | Mono ↑ | VOT | Leak ↓ |
|--------|------|------|--------|-----|--------|
| Swissmetro | MNL | 63.7 | 100 | 79.7 CHF/hr | <0.001 |
| Swissmetro | Mitra | 77.7 | 90.3 | 0.30 CHF/hr | 0.208 |
| Swissmetro | TabPFN | 78.0 | Not Audited | Not Audited | Not Audited |
| Swissmetro | Adapter+Mitra | 76.6 | 100 | 79.7 CHF/hr | <0.001 |
| Swissmetro | Adapter+TabPFN | 76.6 | 100 | 79.7 CHF/hr | <0.001 |
| LPMC | MNL | 69.8 | 100 | PT 1.76 / Drive 18.3 GBP/hr | N/A |
| LPMC | Mitra | 74.2 | 50.8 | PT 7.47 / Drive -5.7 GBP/hr | N/A |
| LPMC | TabPFN | 74.4 | Not Audited | Not Audited | N/A |
| LPMC | Adapter+Mitra | 71.8 | 100 | PT 1.76 / Drive 18.3 GBP/hr | N/A |
| LPMC | Adapter+TabPFN | 72.1 | 100 | PT 1.76 / Drive 18.3 GBP/hr | N/A |

### Ablation Study
The paper compares structural models, raw TFMs, distillation approaches, and the behavior adapter.

| Configuration | Key Metric | Description |
|------|---------|------|
| MNL Structural Only | Swissmetro 63.7%, LPMC 69.8%, Mono 100 | Complete economic logic, but accuracy lags behind TFMs |
| raw Mitra | Swissmetro 77.7%, LPMC 74.2% | High accuracy, but Swissmetro VOT is only 0.30 CHF/hr, and LPMC monotonicity is only 50.8% |
| Distilled MNL | Swissmetro 57.3%, LPMC 70.0% | Training MNL with TFM soft labels performs worse than standard MNL on Swissmetro, showing student model capacity is a bottleneck |
| Adapter+Mitra | Swissmetro 76.6%, LPMC 71.8%, Mono 100 | Recovers ~13 percentage points of accuracy in Swissmetro while maintaining MNL economic coefficients |
| Adapter+TabPFN | Swissmetro 76.6%, LPMC 72.1%, Mono 100 | Not dependent on a specific TFM, showing migration potential of the adapter mechanism |
| LPMC 10x Subsampling | +2.2±0.5% Gain, Mono 100 | All runs show positive gains, 95% CI is [+1.2, +3.3] |

### Key Findings
- Raw TFM errors are not minor noise but structural behavioral failures. In LPMC, nearly half of Mitra samples violate the price-increase-demand-decrease rule, with median driving VOT being negative and 65% of driving VOT estimates being negative.
- In Swissmetro, while Mitra reaches 90.3% monotonicity, its VOT is only 0.30 CHF/hr, far below the 50-100 CHF/hr range found in literature, causing policy models to severely underestimate the value of time savings.
- The adapter does not exceed raw TFM accuracy; its role is to transfer TFM accuracy gains into a constrained model. The authors state it "recovers gains" rather than creating them from scratch.
- The contribution of the correction term varies. In Swissmetro, $\alpha$ exceeds 1.0, indicating TFM is the primary accuracy source; in LPMC, $\alpha$ is around 0.41-0.44, suggesting MNL explains more structure.

## Highlights & Insights
- The paper concretely defines the "high accuracy but policy-untrustworthy" problem using three actionable metrics: monotonicity, VOT, and availability leaks.
- The two-stage training design is simple but critical. It acknowledges the respective strengths of TFMs and economic models, avoiding a "black box" end-to-end classifier.
- The method has low requirements for the TFM itself (only predicted probabilities), making it compatible with in-context models and AutoML wrappers. This is vital for policy institutions that cannot modify foundation model training.
- This work serves as a reminder that TFM evaluation should not rely solely on classification accuracy. Behavioral auditing under domain theory constraints is necessary wherever intervention, pricing, or resource allocation is involved.

## Limitations & Future Work
- Experiments only cover two transportation datasets and two TFMs. While representative, this does not prove similar failures exist across all TFMs in all economic choice tasks.
- Transportation has clear economic constraints; migrating the method to healthcare or energy packages requires defining appropriate behavioral audit metrics for those domains.
- Adapter gains depend on TFM quality. If TFM predictions are poor, Stage 2 merely degrades toward the MNL baseline.
- Audit for TabPFN was omitted due to high in-context inference costs, meaning behavioral differences between raw TabPFN and the adapter weren't fully quantified.
- Future work could extend the structural term to mixed logit, nested logit, or neural embedded choice models to handle preference heterogeneity while retaining the two-stage locking of economic meaning.

## Related Work & Insights
- **vs MNL**: MNL provides price and availability constraints but limited expressive power. Ours retains MNL's structural interpretation while allowing TFMs to correct non-linear residuals.
- **vs raw TabPFN / Mitra**: TFMs are strong at small-sample classification but learn historical correlations without guaranteeing intervention direction. The adapter demotes the TFM from its role as primary decision-maker to an information source.
- **vs Knowledge Distillation**: Distillation compresses TFM knowledge into MNL, resulting in poor Swissmetro accuracy (57.3%), proving linear utility cannot capture TFM's non-linear information. Ours uses TFM probabilities as explanatory variables instead.
- **vs Monotonic Neural Networks**: Constrained networks often require custom architectures and lack economic coefficients. Ours provides hard constraints and analytical VOT via structural utility terms, aligning with policy analysis workflows.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of embedding TFM predictions into constrained models is simple and effective; innovation lies in problem definition and training flow.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two real datasets and multiple TFMs support the claims, though data domains remain limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, concrete metric explanations, and tables that directly expose the conflict between accuracy and economic validity.
- Value: ⭐⭐⭐⭐⭐ Highly significant for TFMs in policy and pricing scenarios, providing a practical path for remediation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] End-to-End Compression for Tabular Foundation Models](end-to-end_compression_for_tabular_foundation_models.md)
- [\[ICML 2026\] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)
- [\[ICML 2026\] BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models](bioarc_discovering_optimal_neural_architectures_for_biological_foundation_models.md)
- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[ICML 2026\] PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)

</div>

<!-- RELATED:END -->
