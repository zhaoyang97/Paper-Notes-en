---
title: >-
  [Paper Note] Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts
description: >-
  [ICML 2025][AI Safety][Concept Bottleneck Models] This paper reveals the "leakage poisoning" phenomenon in Concept Bottleneck Models (CBMs)—where information bypassing the concept bottleneck hurts prediction accuracy under distribution shifts, leading to failed concept interventions. It proposes MixCEM, which utilizes a confidence gate to dynamically decide when to use or discard leaked information, maintaining both high accuracy and effective interventions under both in-dist…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Concept Bottleneck Models"
  - "Concept Intervention"
  - "Distribution Shift"
  - "Leakage Poisoning"
  - "Interpretability"
date: 2026-05-08
content_hash: 452e2cac84aa44e0
---

# Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts

**Conference**: ICML 2025  
**arXiv**: [2504.17921](https://arxiv.org/abs/2504.17921)  
**Code**: [https://github.com/mateoespinosa/cem](https://github.com/mateoespinosa/cem)  
**Area**: Interpretability  
**Keywords**: Concept Bottleneck Models, Concept Intervention, Distribution Shift, Leakage Poisoning, Interpretability

## TL;DR
This paper reveals the "leakage poisoning" phenomenon in Concept Bottleneck Models (CBMs)—where information bypassing the concept bottleneck hurts prediction accuracy under distribution shifts, leading to failed concept interventions. It proposes MixCEM, which utilizes a confidence gate to dynamically decide when to use or discard leaked information, maintaining both high accuracy and effective interventions under both in-distribution and out-of-distribution scenarios.

## Background & Motivation

**Background**: Concept Bottleneck Models (CBMs) predict human-interpretable concepts (e.g., "stripes", "black") first, and then predict labels (e.g., "zebra") from these concepts to enhance interpretability. A key advantage is concept intervention, where human experts correct incorrect concept predictions during testing, and the model automatically updates its final prediction.

**Limitations of Prior Work**:
   - Training concept annotations are often incomplete (some concepts lack labels), leading to an "incompleteness gap" where labeled concepts are insufficient to predict labels accurately.
   - Existing methods utilize "bypass" mechanisms (e.g., dynamic concept embeddings or residual connections) to allow information to bypass the concept bottleneck and "leak" through, which compensates for incompleteness in in-distribution (ID) scenarios.
   - However, this paper reveals that under distribution shifts, this leaked information itself becomes out-of-distribution (OOD), "poisoning" the model's predictions.

**Key Challenge**: Leaked information is beneficial in ID scenarios (compensating for incomplete concepts) but harmful in OOD scenarios (since the leaked information itself becomes unreliable). Thus, a dynamic switch between the two is required.

**Goal**: Enable concept models to respond effectively to concept interventions in both ID and OOD scenarios.

**Key Insight**: First accurately diagnose the problem—"leakage poisoning" is a previously unrecognized phenomenon. Then, design a confidence gate in MixCEM to adaptively control the leakage.

**Core Idea**: The embedding of each concept = a global embedding (no leakage, safe but incomplete) + a residual embedding (with leakage, complete but potentially poisoned), using a confidence gate to determine the weight of the residual—trusting the residual for ID samples and discarding it for OOD samples.

## Method

### Overall Architecture
The concept embedding of MixCEM is a mixture of two parts:
1. **Global embedding** $e_g$: Input-independent fixed concept embedding (pure concept information, no leakage).
2. **Residual embedding** $e_r(x)$: Input-dependent dynamic embedding (containing extra information bypassing the bottleneck).
3. **Mixed embedding** $e(x) = (1-\gamma(x)) \cdot e_g + \gamma(x) \cdot e_r(x)$, where $\gamma(x)$ is the confidence gate.

### Key Designs

1. **Diagnosis of Leakage Poisoning**:

    - Function: Identifies and analyzes the leakage poisoning phenomenon of concept models under OOD for the first time.
    - Core Observation: Bypass mechanisms (e.g., CEM, Residual CBM) show significant accuracy improvements after concept interventions on ID samples, but their accuracy degrades instead of improving after interventions on OOD samples.
    - Cause Analysis: Residual embeddings encode distribution-specific shortcut information. Under OOD, this information becomes unreliable. Even if concepts are correctly intervened upon, the "poisoned information" in the residuals is still passed to the classifier.
    - Design Motivation: Accurate diagnosis is a prerequisite for effective cure—"leakage poisoning" is a design consideration that was previously ignored.
    - Experimental Evidence: On the CUB dataset, CEM's accuracy increases from 66% to 85% after ID interventions, but drops from 60% to 58% (degrades instead of improving!) after OOD interventions.

2. **Confidence Gating in MixCEM**:

    - Function: Dynamically decides the proportion of residual embedding to use for each sample and each concept.
    - Mechanism: $\gamma(x) = \sigma(w^T \cdot [e_g; e_r(x); \text{conf}(x)])$, where $\text{conf}(x)$ is the confidence estimation of the input.
    - Behavior Pattern:
        - ID samples: $\gamma \to 1$ (trusting residuals, utilizing leakage to compensate for incompleteness).
        - OOD samples: $\gamma \to 0$ (discarding residuals, falling back to pure concept prediction).
    - Design Motivation: Avoid a binary choice between "with leakage" and "without leakage". Instead, dynamically decide for each sample—preserving performance in ID scenarios and avoiding poisoning in OOD scenarios.

3. **Behavior During Intervention**:

    - Function: Ensures that concept interventions are effective under both ID and OOD settings.
    - Mechanism: When a concept is intervened upon (corrected to its true value), the global embedding $e_g$ is directly updated to the embedding corresponding to the correct concept, avoiding contamination from the residual.
    - Under OOD, $\gamma \to 0$, meaning the intervention effect is directly propagated through the global embedding without being "diluted" by the residual.
    - Design Motivation: The value of intervention lies in "external knowledge provided by humans", which should not be overridden by internal leaked information.

### Loss & Training
- Concept prediction loss (cross-entropy)
- Label prediction loss (cross-entropy)
- Gated regularization: encourages the gate to approach 1 on the ID training distribution (to fully exploit residuals)
- End-to-end training
- No OOD data required—MixCEM learns to discard residuals when uncertain based solely on ID data

## Key Experimental Results

### Main Results
CUB-200 (bird classification, different types of OOD shifts):

| Method | ID w/o Intervention | ID Post-Intervention | OOD w/o Intervention | OOD Post-Intervention |
|------|---------|---------|----------|----------|
| CBM (No Leakage) | 72.1% | 85.3% | 63.2% | 78.1% |
| CEM (With Leakage) | 80.2% | 88.5% | 60.5% | 58.2% ↓ |
| Residual CBM | 81.5% | 87.8% | 61.8% | 59.5% ↓ |
| **MixCEM** | **80.8%** | **88.2%** | **67.3%** | **80.5%** |

### Concept Incompleteness Scenarios

| Method | Complete Concepts ID | Incomplete Concepts ID | Incomplete Concepts OOD |
|------|----------|-----------|-----------|
| CBM | 85.3% | 72.1% | 63.2% |
| CEM | 88.5% | 80.2% | 60.5% |
| **MixCEM** | **88.2%** | **80.8%** | **67.3%** |

### Ablation Study

| Configuration | ID Acc | OOD Post-Intervention Acc | Description |
|------|--------|-------------|------|
| No residuals (Pure CBM) | 72.1% | 78.1% | Safe but ID incomplete |
| Residual fixed weight 0.5 | 76.5% | 65.2% | Non-adaptive |
| Residual fixed weight 1.0 (CEM) | 80.2% | 58.2% | Leakage poisoning |
| **Confidence gating $\gamma(x)$** | **80.8%** | **80.5%** | Adaptive optimal |
| No confidence input (embeddings only) | 78.5% | 72.3% | Confidence provides a key signal |

### Key Findings
- Leakage poisoning is a real phenomenon—CEM's accuracy actually drops after OOD intervention (from 88.5% ID to 58.2% OOD).
- MixCEM does not sacrifice performance under ID (80.8% vs. CEM 80.2%) while offering significant improvements under OOD (67.3% vs. 60.5%).
- The improvement post-OOD intervention is the most notable—MixCEM 80.5% vs. CEM 58.2% (+22.3%).
- The gating mechanism learns the correct behavior—$\gamma \approx 0.85$ for ID samples and $\gamma \approx 0.15$ for OOD samples.

## Highlights & Insights
- Accurate naming of **"leakage poisoning"**—information leakage acts as a "supplement" under ID, but a "poison" under OOD.
- The gating design of MixCEM is elegant—instead of a binary choice of "whether to leak", it is a continuous adjustment of "how much to leak".
- Imparts a crucial warning for the real-world deployment of Explainable AI—if a concept model performs worse after interventions under OOD, its reliability in critical scenarios is highly questionable.
- Learns OOD awareness using only ID training data—since uncertainty/confidence serves as a natural signal for OOD detection.
- Complementary to CUDA (from the same conference)—CUDA addresses domain adaptation, whereas MixCEM tackles OOD robustness under interventions.

## Limitations & Future Work
- Confidence estimation itself can be unreliable under OOD (e.g., overconfident OOD samples).
- The gating mechanism increases model complexity.
- Validated only on classification tasks.
- Interaction effects of simultaneous interventions on multiple concepts remain unanalyzed.

## Related Work & Insights
- **vs CBM**: No leakage $\to$ safe but incomplete
- **vs CEM**: With leakage $\to$ complete but poisoned under OOD
- **vs MixCEM**: Adaptive leakage $\to$ best of both worlds
- **vs CUDA**: CUDA performs concept alignment during domain adaptation, whereas MixCEM performs leakage control during interventions—representing a complementary relationship.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Leakage poisoning" is an important new finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple shift types, complete vs. incomplete concepts, and with vs. without interventions.
- Writing Quality: ⭐⭐⭐⭐⭐ Deep problem diagnosis and elegant method design.
- Value: ⭐⭐⭐⭐⭐ Has direct implications for the reliable deployment of explainable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Rethinking the Bias of Foundation Model under Long-tailed Distribution](rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[NeurIPS 2025\] Causally Reliable Concept Bottleneck Models](../../NeurIPS2025/ai_safety/causally_reliable_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](../../NeurIPS2025/ai_safety/preserving_task-relevant_information_under_linear_concept_removal.md)
- [\[ICML 2026\] Robust In-Context Reinforcement Learning Under Reward Poisoning Attacks](../../ICML2026/ai_safety/robust_in-context_reinforcement_learning_under_reward_poisoning_attacks.md)

</div>

<!-- RELATED:END -->
