---
title: >-
  [Paper Note] Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia
description: >-
  [AAAI 2026][Credit Scoring] This paper proposes an end-to-end cash flow underwriting workflow based on bank transaction data and constructs the first Malaysian MSME bank statement dataset (611 loan records). It demonstrates that features derived from bank transactions improve a logistic regression model's AUROC from 0.672 to 0.850 compared to traditional application information alone, significantly enhancing credit assessment capability for MSMEs lacking credit histories.
tags:
  - AAAI 2026
  - Credit Scoring
  - Financial Inclusion
  - Bank Statements
  - MSMEs
  - Cash Flow Underwriting
date: 2026-05-08
content_hash: 26acd07219da5c3c
---

# Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia

**Conference**: AAAI 2026
**arXiv**: [2510.16066](https://arxiv.org/abs/2510.16066)
**Code**: None
**Area**: Other
**Keywords**: Credit Scoring, Financial Inclusion, Bank Statements, MSMEs, Cash Flow Underwriting

## TL;DR

This paper proposes an end-to-end cash flow underwriting workflow based on bank transaction data and constructs the first Malaysian MSME bank statement dataset (611 loan records). It demonstrates that features derived from bank transactions improve a logistic regression model's AUROC from 0.672 to 0.850 compared to traditional application information alone, significantly enhancing credit assessment capability for MSMEs lacking credit histories.

## Background & Motivation

### The MSME Financing Gap

MSMEs account for 96.1% of all enterprises in Malaysia and contribute approximately 60% of GDP, yet face a severe financing gap (approximately RM 90 billion). The root cause lies in traditional credit assessment's heavy reliance on bureau data (repayment history, outstanding debt, etc.). For newly established businesses and "credit invisibles," this backward-looking approach creates high entry barriers and perpetuates a vicious cycle of financial exclusion.

### Three Core Deficiencies of Traditional Credit Models

**Backward-looking nature**: Focuses solely on past repayment behavior without reflecting current or future repayment capacity.

**Neglect of real-time signals**: Unable to capture operational dynamics and current financial health.

**Omission of alternative indicators**: Ignores measurable credit signals such as cash flow consistency, accounts receivable/payable patterns, and digital transaction behavior.

### The Value of Bank Statements

Bank statements represent the most current and verifiable source of financial behavioral data, capturing income regularity, expenditure patterns, and cash flow stability. Related practices exist globally (mobile network data credit scoring in Africa, transaction data in Indian FinTech), yet research in the Malaysian MSME context is nearly absent.

## Method

### Overall Architecture

The end-to-end cash flow underwriting workflow comprises a three-tier architecture:

1. **Web Layer (Customer Onboarding)**: Submission portal for loan application data and bank statements.
2. **Application Layer (Bank Statement Analyzer)**: Multiple AI modules that automate extraction and analysis of unstructured transaction data.
3. **Data and Scoring Layer (Cash Flow Underwriting)**: Feature store, feature selection, credit scoring model training, and prediction.

### Key Designs

#### 1. AI Modules in the Bank Statement Analyzer

**Function**: Transforms unstructured bank statement PDFs into structured data and extracts credit-relevant features.

Five core modules are included:

- **Key Information Extraction**: OCR extracts account numbers, names, addresses, etc.; cross-statement verification confirms document ownership.
- **Transaction Table Extraction**: OCR combined with layout analysis to locate and digitize transaction tables, handling merged cells, cross-page entries, and other complex formatting.
- **Fraud Analysis**: Computer vision combined with rule-based methods to detect tampering (font inconsistencies, layout anomalies, metadata mismatches, pixel-level editing artifacts).
- **Network Analysis**: Constructs transaction network graphs; uses graph algorithms to detect circular fund flows, round-tripping operations, and associations with blacklisted entities.
- **Cash Flow Analysis**: NLP infers transaction intent and classification; computes key metrics such as average, maximum, and minimum balances.

**Design Motivation**: Each module addresses a different dimension of credit assessment, forming a comprehensive portrait of MSME financial health.

#### 2. WOE-IV Feature Engineering and Selection

**Function**: Quantifies the predictive power of each feature using Weight of Evidence (WOE) and Information Value (IV), and performs feature selection.

**Core Formulas**:

For the $k$-th bin of feature $j$, WOE is defined as:

$$\text{WOE}_{jk} = \log\left(\frac{n_{gjk}/N_g}{n_{bjk}/N_b}\right)$$

A positive value indicates a higher proportion of non-defaults in that bin (lower risk); a negative value indicates a higher proportion of defaults.

The overall Information Value for feature $j$:

$$\text{IV}_j = \sum_{k=1}^{K_j} \left(\text{Dist}_{jk}^{(g)} - \text{Dist}_{jk}^{(b)}\right) \text{WOE}_{jk}$$

IV threshold interpretation: $<0.02$ (no predictive power), $0.02$–$0.1$ (weak), $0.1$–$0.3$ (moderate), $0.3$–$0.5$ (strong), $\geq 0.5$ (possible data leakage).

**Design Motivation**: WOE encoding provides a log-odds transformation that is naturally suited to logistic regression and supports monotonic relationship interpretability. Binning is performed exclusively on training folds to prevent data leakage.

#### 3. Logistic Regression Credit Scoring Model

**Function**: Trains a default probability prediction model based on WOE-encoded features.

**Core Formulas**:

$$\log\frac{P(y_i=1|\mathbf{x}_i)}{P(y_i=0|\mathbf{x}_i)} = \beta_0 + \sum_{j=1}^d \beta_j \text{WOE}_j(x_{ij})$$

$$\mathcal{L}(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i \log p_i + (1-y_i)\log(1-p_i)] - \lambda\|\boldsymbol{\beta}\|_2^2$$

**Design Motivation**: Logistic regression is robust under small samples and class imbalance; its coefficients are directly interpretable as credit risk indicators. L2 regularization prevents overfitting induced by high-dimensional transaction features.

### Loss & Training

- Systematic development following the CRISP-DM methodology.
- 60:40 train/validation split.
- 5-fold cross-validation to reduce randomness.
- Comparative evaluation against Random Forest, Gradient Boosting, and AdaBoost.
- A complete MLOps framework supporting continuous integration/deployment and a Champion-Challenger model update mechanism.

## Key Experimental Results

### Main Results

**AUROC Comparison Across Models on the Validation Set**

| Model | AUROC↑ | Gain over Baseline |
|---|---|---|
| AdaBoost | 0.598 | Baseline |
| Gradient Boosting | 0.633 | +5.9% |
| Random Forest | 0.655 | +9.5% |
| **Logistic Regression** | **0.782** | **+30.8%** |

Logistic regression substantially outperforms complex ensemble methods under small-sample conditions (367 training instances, of which only 56 are defaults) and class imbalance. This is consistent with established findings in the credit scoring literature—logistic regression frequently matches or exceeds tree-based ensemble models on small-to-medium credit datasets.

### Ablation Study

**AUROC Under Different Feature Combinations (5-Fold Cross-Validation)**

| Feature Combination | LR AUROC↑ | Notes |
|---|---|---|
| Application info only | 0.672 | Baseline |
| Bank transactions only | 0.821 | +22%, far exceeds application info |
| **All features** | **0.850** | Best; transactions provide incremental value |

**Cross-Model Comparison Under Different Feature Combinations**

| Feature Combination | LR | RF | GB | AB |
|---|---|---|---|---|
| Application only | 0.672 | ~0.60 | ~0.58 | ~0.55 |
| Transactions only | 0.821 | ~0.63 | ~0.62 | ~0.58 |
| All features | 0.850 | 0.655 | 0.633 | 0.598 |

Across all models, versions incorporating bank transaction data outperform those using application information alone, and combined features yield the best results.

**Feature IV Ranking Analysis**

Nine out of ten bank statement features rank higher than all application form features. The sole exception is "customer classification" (a business entity label pre-assigned by the lender), whose IV is only marginally higher than the top-ranked transaction feature (relative difference of 7.92%).

### Key Findings

1. **Bank transaction data provides far greater predictive power than application information**: AUROC improves from 0.672 to 0.821 (+22%), confirming that transaction features capture dynamic financial behaviors overlooked by traditional credit models.
2. **Simpler models outperform on small samples**: Logistic regression substantially surpasses RF, GB, and AB, as complex models severely overfit given only 367 training instances under class imbalance.
3. **Feature complementarity**: All features (0.850) > transactions only (0.821) > application only (0.672); application information still provides incremental value.
4. **Account behavior features are most critical**: Cash flow indicators such as average balance and minimum balance ratio are the strongest predictors of default.

## Highlights & Insights

1. **Production deployment driven**: This is not a purely academic experiment but a complete system already deployed in production at a Malaysian lending institution, incorporating an MLOps framework and a Champion-Challenger continuous optimization mechanism.
2. **First Malaysian MSME bank statement dataset**: Fills a data gap in MSME credit assessment for the region; planned open-source release to facilitate subsequent research.
3. **End-to-end workflow design**: Fully automated pipeline from bank statement upload to credit score output, encompassing OCR, fraud detection, network analysis, feature engineering, and model training.
4. **Pragmatic modeling choices**: Logistic regression is preferred over deep learning because interpretability is critical in financially regulated contexts; the WOE-IV framework is an industry standard in credit risk.
5. **Integrated credit scoring framework**: The new cash flow model operates in parallel with existing bureau-based scorecards, applying a conservative merging strategy (assigning the higher risk rating) to expand credit coverage without compromising risk control.

## Limitations & Future Work

1. **Limited sample size**: Only 611 loan records from a single institution; representativeness is constrained. Validation on larger datasets across multiple institutions is needed.
2. **Class imbalance**: The 518:93 non-default-to-default ratio reflects the real-world distribution but poses challenges for minority-class prediction; oversampling or cost-sensitive learning could be explored.
3. **Absence of module-level evaluation**: While overall credit scoring performance is assessed, independent evaluation of individual AI modules (OCR, fraud detection, etc.) is unavailable due to the use of proprietary methods.
4. **Single-region validation**: Deployment is limited to one Malaysian institution; generalizability across regions and regulatory environments remains to be verified.
5. **Temporal robustness unvalidated**: Longitudinal performance studies across economic cycles are absent; the stability of bank statement features under macroeconomic shifts is uncertain.
6. **Potential evolution toward multi-agent systems**: Future work could formalize individual modules as autonomous agents, constructing a multi-agent credit scoring architecture.

## Related Work & Insights

- **Alternative data credit scoring**: Precedents exist in mobile network data scoring in Africa and platform transaction data in Indian FinTech; this paper applies bank statements to the Malaysian MSME context.
- **LLM-driven agent systems**: Okpala et al. (2025) and others demonstrate the potential of LLM agents in credit card approval and portfolio risk modeling.
- **Enduring value of the WOE-IV framework**: Despite rapid advances in deep learning, WOE-IV remains widely adopted in credit risk for its interpretability and stability.
- **Implications for developing economies**: Provides a replicable financial inclusion solution template for other emerging markets.

## Rating

- Novelty: ⭐⭐⭐ (The first Malaysian MSME bank statement dataset is valuable, but the methodology is relatively conventional.)
- Experimental Thoroughness: ⭐⭐⭐ (Ablation study is clear, but sample size is small and models are simple.)
- Writing Quality: ⭐⭐⭐⭐ (Systematic and comprehensive; the industrial deployment perspective is persuasive.)
- Value: ⭐⭐⭐⭐ (High practical application value; open-sourcing the dataset will facilitate subsequent research.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)
- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)

</div>

<!-- RELATED:END -->
