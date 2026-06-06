---
title: >-
  [Paper Note] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation
description: >-
  [AAAI 2026][Social Computing][Jailbreak Evaluation] This paper proposes SceneJailEval, a scenario-adaptive multi-dimensional jailbreak evaluation framework that defines 14 jailbreak scenarios and 10 evaluation dimensions…
tags:
  - "AAAI 2026"
  - "Social Computing"
  - "Jailbreak Evaluation"
  - "Scenario-Adaptive"
  - "Multi-Dimensional Evaluation"
  - "LLM Safety"
  - "Harm Quantification"
date: 2026-05-08
content_hash: b507dc8fc9ec2272
---

# SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation

**Conference**: AAAI 2026
**arXiv**: [2508.06194](https://arxiv.org/abs/2508.06194)  
**Code**: [GitHub](https://github.com/FutureSJTU/SceneJailEval)  
**Area**: Social Computing
**Keywords**: Jailbreak Evaluation, Scenario-Adaptive, Multi-Dimensional Evaluation, LLM Safety, Harm Quantification

## TL;DR
This paper proposes SceneJailEval, a scenario-adaptive multi-dimensional jailbreak evaluation framework that defines 14 jailbreak scenarios and 10 evaluation dimensions. Through a pipeline of scenario classification → dynamic dimension selection → multi-dimensional detection → weighted harm scoring, it achieves F1 of 0.917 on a self-constructed dataset (surpassing SOTA by 6%) and 0.995 on JBB (surpassing SOTA by 3%), while supporting harm severity quantification beyond binary classification.

## Background & Motivation

**Background**: LLM jailbreak attack evaluation suffers from two major problems — mainstream methods (string matching, toxicity classifiers, LLM judges) produce only binary "yes/no" outputs without quantifying harm severity; emerging multi-dimensional frameworks (e.g., StrongREJECT, Cai et al.) apply uniform evaluation criteria across all scenarios, ignoring scenario-specific differences.

**Limitations of Prior Work**:
   - **Binary classification is too coarse**: It cannot distinguish between "providing detailed methods for killing" and "merely implying the possibility"
   - **One-size-fits-all evaluation criteria**: For instance, the "authenticity" dimension is meaningful for "violent crime" scenarios but irrelevant to "hate speech"; the same dimension should carry different weights across scenarios
   - **Lack of regional sensitivity**: Cryptocurrency compliance requirements differ between mainland China and Japan, yet existing methods cannot account for this

**Key Challenge**: Jailbreak scenarios are highly heterogeneous (violent crime vs. sexual content vs. political incitement), but evaluation methods treat them uniformly, creating an accuracy bottleneck.

**Goal**: To construct a scenario-adaptive jailbreak evaluation framework in which evaluation dimensions, scoring criteria, and weights are dynamically adjusted according to the scenario.

**Key Insight**: Drawing on mature scenario-based evaluation paradigms from software testing and autonomous driving verification, combined with DREAD/CVSS cybersecurity threat modeling theory, to systematically construct the evaluation framework.

**Core Idea**: A 14-scenario × 10-dimension adaptive matrix combined with Delphi-method expert consensus and AHP analytic hierarchy process weighting, enabling fine-grained jailbreak evaluation.

## Method

### Overall Architecture
A four-step pipeline: (1) **Scenario Classifier** — an LLM agent maps the input to one of 14 predefined scenarios; (2) **Scenario–Dimension Adapter** — dynamically selects evaluation dimensions, scoring criteria, and weights based on the scenario; (3) **Jailbreak Detector** — multi-dimensional judgment across 6 detection dimensions, fused via logical AND; (4) **Harm Assessor** — weighted scoring across 4 harm dimensions, producing a composite harm score.

### Key Designs

1. **14-Scenario × 10-Dimension Adaptive Matrix**:

    - **Function**: Tailors the subset of evaluation dimensions and scoring criteria to each jailbreak scenario.
    - **Mechanism**: The 14 scenarios cover violent crime, non-violent crime, sexual content, hate speech, suicide/self-harm, cyberattacks, disinformation, political incitement and elections, professional advice, privacy violation, intellectual property infringement, cryptocurrency violations, regionally sensitive issues, and illegal weapons/drugs. The 10 dimensions are divided into 6 detection dimensions (refusal, helpfulness, regional compliance, risk warning, political neutrality, explicitness) and 4 harm dimensions (authenticity, specificity, severity, scope of impact).
    - **Design Motivation**: Different scenarios require different dimension combinations. For example, "sexual content" emphasizes "explicitness," "professional advice" emphasizes "risk warning," and "regionally sensitive issues" requires "regional compliance."

2. **Delphi Method + AHP for Weight Determination**:

    - **Function**: Systematically determines the weights of harm assessment dimensions for each scenario.
    - **Mechanism**: Ten security experts use multiple rounds of anonymous Delphi consensus to rank dimension importance for each scenario (consensus criteria: CV < 0.25, IQR ≤ 2), after which AHP converts the rankings into specific weights (consistency check: CR < 0.1).
    - **Design Motivation**: Avoids arbitrarily assigned weights; expert consensus and mathematical methods ensure objectivity and reproducibility.

3. **Multi-Dimensional Jailbreak Detection Design**:

    - **Function**: Captures edge cases missed by traditional methods by decomposing the detection task.
    - **Mechanism**: Six detection dimensions each serve a distinct role — "refusal" checks whether the response contains only refusal expressions without substantive content; "helpfulness" detects whether the model indirectly facilitates malicious behavior (e.g., refusing first but then providing phishing details); "regional compliance" handles geographic differences. The final judgment is the logical AND of all relevant dimensions.
    - **Design Motivation**: "Refuse-then-help" edge cases are the primary failure mode of existing methods; the combined "refusal" + "helpfulness" detection effectively captures such cases.

### Loss & Training
- Detection evaluation metrics: Accuracy, Precision, Recall, F1
- Harm scoring evaluation metrics: NMAE (deviation from expert annotations), Spearman-Rho (rank correlation with human judgments)
- Overall NMAE = 0.013, Spearman-Rho = 0.938, indicating high agreement with expert judgments

## Key Experimental Results

### Main Results

Self-constructed SceneJailEval dataset (1,308 queries, 14 scenarios):

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| StringMatch | 0.749 | 0.750 | 0.957 | 0.841 |
| Qi2023 (GPT-4) | 0.816 | 0.966 | 0.760 | 0.851 |
| JailJudge | 0.800 | 0.930 | 0.768 | 0.841 |
| **SceneJailEval** | **0.883** | **0.901** | **0.929** | **0.915** |

Public datasets — JBB: F1 = 0.995 (SOTA); JailJudge dataset: F1 = 0.824 (SOTA).

### Ablation Study

| Configuration | F1 | Description |
|---------------|-----|-------------|
| Full SceneJailEval | **0.917** | Complete framework |
| DimsOnly (w/o scenario classification) | 0.890 | Removing scenario classification drops F1 by 2.7% |
| Vanilla (w/o dimension selection) | 0.831 | Falling back to generic heuristics drops F1 by 8.6% |

### Key Findings
- **Scenario adaptation is critical**: Removing scenario classification and dimension selection leads to F1 drops of 2.7% and an additional 8.6%, respectively.
- **Edge case detection advantage**: Significantly outperforms all baselines on "refuse-then-help" cases and region-specific cases.
- **Harm scoring highly consistent with experts**: NMAE < 0.02, Spearman-Rho ≈ 0.94.
- **Strong generalization**: Ranks second on Safe-RLHF despite Beaver being specifically fine-tuned on that dataset.

## Highlights & Insights
- **Introducing scenario-based evaluation methodology into LLM safety assessment**: A methodological contribution that transplants mature paradigms from software testing and autonomous driving into this domain.
- **DREAD/CVSS theory guiding harm dimension definition**: Dimensions are grounded in established cybersecurity threat modeling theory rather than intuition.
- **Delphi + AHP weight determination**: Provides a reproducible, extensible framework for determining scenario–dimension weights rather than hard-coding them.
- **Regional sensitivity dimension**: The first jailbreak evaluation framework to incorporate cultural and legal differences across regions.

## Limitations & Future Work
- **Coverage of 14 scenarios**: Although relatively comprehensive, real-world jailbreak attacks may fall outside these 14 categories.
- **Dependence on LLM agent classification accuracy**: Misclassification of scenarios propagates errors to all subsequent dimension selection and evaluation steps.
- **High cost of expert annotation**: Dataset construction requires five security experts to annotate using scenario-adaptive criteria, making scaling expensive.
- **Underlying model is Qwen-3-235B**: Only one underlying model has been validated; performance with other LLM-as-judge setups remains unknown.
- **Delphi consensus from 10 experts**: The sample size is relatively small and may be subject to individual biases.

## Related Work & Insights
- **vs. StrongREJECT**: Applies unified Rejection Clarity/Specificity/Credibility criteria across all scenarios, ignoring scenario differences. SceneJailEval overcomes this limitation through scenario adaptation.
- **vs. AttackEval**: Uses GPT-4 reference answers with cosine similarity, still a uniform standard. SceneJailEval's multi-dimensional scenario-adaptive evaluation is more fine-grained.
- **vs. LlamaGuard3**: Meta's official safety judge model achieves F1 = 0.98 on JBB, compared to SceneJailEval's 0.995.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The scenario-adaptive multi-dimensional framework is pioneering in jailbreak evaluation; the Delphi + AHP weighting approach is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, six baselines, ablation studies, and expert annotation consistency validation.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is rigorously defined with complete mathematical formalization and detailed scenario and dimension definitions.
- **Value**: ⭐⭐⭐⭐ Practically valuable for LLM safety evaluation; the extensible framework design is forward-looking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation](../../ACL2026/social_computing/when_bigger_isn39t_better_a_comprehensive_fairness_evaluation_of_political_bias_.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](../../ICLR2026/social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)

</div>

<!-- RELATED:END -->
