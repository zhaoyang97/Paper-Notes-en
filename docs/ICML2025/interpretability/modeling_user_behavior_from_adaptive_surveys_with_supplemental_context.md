---
title: >-
  [Paper Note] LANTERN: Modeling User Behavior from Adaptive Surveys with Supplemental Context
description: >-
  [ICML 2025 Best Paper][Interpretability][Survey data fusion] Proposes LANTERN (Late-Attentive Network for Enriched Response Modeling), a modular user behavior modeling architecture that treats adaptive survey data as the primary signal and achieves late fusion via cross-attention. Selective gating and residual connections maintain the dominance of survey signals, while external context (demographics, behavioral logs, etc.) is integrated only when relevant…
tags:
  - "ICML 2025 Best Paper"
  - "Interpretability"
  - "Survey data fusion"
  - "late fusion"
  - "cross-attention"
  - "selective gating"
  - "multi-label prediction"
  - "tabular data"
date: 2026-05-08
content_hash: 017200ea96af50ca
---

# LANTERN: Modeling User Behavior from Adaptive Surveys with Supplemental Context

**Conference**: ICML 2025 Best Paper  
**arXiv**: [2507.20919](https://arxiv.org/abs/2507.20919)  
**Code**: None  
**Area**: User Behavior Modeling / Recommender Systems  
**Keywords**: Survey data fusion, late fusion, cross-attention, selective gating, multi-label prediction, tabular data

## TL;DR

Proposes LANTERN (Late-Attentive Network for Enriched Response Modeling), a modular user behavior modeling architecture that treats adaptive survey data as the primary signal and achieves late fusion via cross-attention. Selective gating and residual connections maintain the dominance of survey signals, while external context (demographics, behavioral logs, etc.) is integrated only when relevant, significantly outperforming the survey-only baseline on a production-scale dataset of approximately 35,000 users with an F1 score of 0.775 compared to 0.734.

## Background & Motivation

**Background**: Understanding user behavior is a fundamental challenge in applied machine learning. Surveys are classical instruments for collecting user behavior data, offering advantages such as structure, high fidelity, and interpretability. In domains such as marketing, healthcare, and public policy, surveys provide longitudinal insights into user preferences, decisions, and experiences.

**Limitations of Prior Work**: Survey data suffers from inherent limitations—user fatigue leads to incomplete responses, restricted questionnaire lengths limit coverage, and conditional question designs render observations of certain attributes highly sparse. Supplemental signals (demographics, engagement metrics, transaction records) are abundant but noisy and potentially biased; simply concatenating them introduces feature explosion.

**Key Challenge**: Survey data is high-fidelity but has insufficient coverage, whereas external data has wide coverage but is highly noisy. How can both be fused while preserving the dominance of the survey signals, while selectively utilizing external signals to compensate for survey deficiencies?

**Goal**: Design an architecture where survey data remains central, selectively incorporating external signals only when they aid behavior prediction, with modular support for flexible integration of new data sources.

**Key Insight**: Adopt a late fusion strategy, using survey embeddings as queries to query external embeddings—meaning external information is retrieved only when the survey "asks for what is needed," rather than mixing indiscriminately.

**Core Idea**: Use survey embeddings as cross-attention queries and external data embeddings as keys/values to achieve late fusion. Combine this with a selective gating mechanism $g$ and residual connection $h_{fused} = h_s + g \odot (h_t - h_s)$ to ensure survey signal dominance, establishing a "survey-dominant, supplement-assisted" paradigm for user behavior modeling.

## Method

### Overall Architecture

The pipeline of LANTERN is as follows: (1) Survey responses $x_s$ and external features $x_e$ generate embeddings $h_s, h_e \in \mathbb{R}^{N \times D}$ via independent encoders $f_s$ and $f_e$ respectively; (2) A cross-attention layer uses $h_s$ as the query and $h_e$ as the key/value to generate contextualized embeddings $h_t$; (3) Gated residual fusion $h_{fused} = h_s + g \odot (h_t - h_s)$ preserves the dominance of the survey signals; (4) After adding Gaussian noise regularization, feedforward layers and a sigmoid function output the predicted probabilities $\hat{y} \in [0,1]^{N \times d}$ for each response key.

### Key Designs

1. **Late-Attentive Cross-Attention**
    - **Function**: Permits survey signals to "actively query" relevant information within external signals, rather than being passively mixed.
    - **Mechanism**: $h_t = \text{Encoder}(Q=h_s, K=h_e, V=h_e)$. Utilizing a Transformer encoder structure (8-head attention), the survey embedding determines "what to retrieve from external signals," avoiding the learning of all possible interactions.
    - **Design Motivation**: Compared to early fusion (direct feature concatenation) or intermediate fusion, late fusion allows the survey module to form complete representations prior to fusion, operating with external information as a supplement rather than a replacement.

2. **Selective Gating + Residual Connection Mechanism**
    - **Function**: Controls the degree to which external information is integrated, maintaining absolute dominance of the survey signals.
    - **Mechanism**: $h_{fused} = h_s + g \odot (h_t - h_s)$, where $g \in (0,1)$ is a learnable gate. When $g \to 0$, external information is completely ignored, degrading to a survey-only model; when $g \to 1$, the cross-attention output is fully adopted. Gaussian noise $G$ is added as regularization to prevent overfitting.
    - **Design Motivation**: A three-layer defense (late fusion + gating + residuals) jointly ensures that survey signals are not overwritten. If external data is irrelevant or noisy, the model can automatically ignore it by learning a low value of $g$.

3. **Modular Encoder Design**
    - **Function**: Supports independent improvement, replacement, or extension of encoders for each data source.
    - **Mechanism**: The survey encoder $f_s$ and external encoder $f_e$ are completely decoupled, allowing independent versioning and deployment. New data modalities can be integrated simply by adding encoders. The overall model contains approximately 50M parameters.
    - **Design Motivation**: In industrial scenarios with continuously evolving data sources, modularity allows new data sources to be integrated without retraining the entire model. When external data is delayed or missing, the system gracefully degrades to survey-only mode.

### Loss & Training

Binary Cross Entropy (BCE) loss is used for multi-label prediction. Each response key (possible answer option) is treated as an independent binary classification problem. A mask $m \in \{-1, 0, 1\}^{N \times d}$ is used to extract labels of the questions actually presented to the user during the adaptive survey: $L = \text{BCE}(m \odot \hat{y})$. Cross-attention employs 8 heads, and the residual fusion includes Dropout and LayerNorm. Implemented using TensorFlow.

## Key Experimental Results

### Ablation Study (Core)

| Configuration | Precision | Recall | F1 |
|------|-----------|--------|------|
| Survey-only | 0.7976 | 0.6794 | 0.7338 |
| External-only | 0.7537 | 0.4264 | 0.5447 |
| **LANTERN** | **0.8263** | **0.7296** | **0.7750** |

### Rare vs. Frequent Attribute Analysis

| Configuration | Rare-Precision | Rare-Recall | Rare-F1 | Frequent-Precision | Frequent-Recall | Frequent-F1 |
|------|-------------|------------|---------|-------------|------------|---------|
| Survey | 0.8755 | 0.8161 | 0.8448 | 0.7865 | 0.6165 | 0.6912 |
| External | 0.7776 | 0.5932 | 0.6730 | 0.7931 | 0.6029 | 0.6850 |
| **LANTERN** | **0.8751** | **0.8404** | **0.8575** | **0.7901** | **0.6484** | **0.7123** |

### Threshold Sensitivity Analysis

| Threshold | Precision | Recall | F1 |
|------|-----------|--------|------|
| 0.3 | Lower | Highest | ~0.74 |
| 0.5 | Medium | Medium | **~0.775** |
| 0.7 | Highest | Lower | ~0.75 |

### Key Findings

- LANTERN's F1 score of 0.7750 represents a 5.6% improvement over the Survey-only model's 0.7338, primarily driven by a recall boost from 0.6794 to 0.7296 (+7.4%)—external data helps recover behavioral signals not directly covered by surveys.
- The F1 score of External-only is only 0.5447, substantially lower than Survey-only, validating the design philosophy that "survey data is the primary signal."
- Improvement is more pronounced on rare attributes (F1 increased from 0.8448 to 0.8575), because surveys have insufficient coverage of conditional/rare questions, where external data precisely fills the gap.
- Improvement is also observed on frequent attributes (F1 increased from 0.6912 to 0.7123), indicating that external data provides complementary information even in noisy, high-frequency scenarios.
- The F1 curve is relatively flat across different thresholds (~0.74-0.78), indicating that the model's output probabilities are well-calibrated.
- The lightweight design with 50M parameters allows the model to be directly deployed for real-time inference in production environments.

## Highlights & Insights

- **Broad applicability of the "survey-dominant, supplement-assisted" paradigm**: This fusion framework of "high-fidelity primary signals + large-scale noisy auxiliary signals" can be generalized to any field where primary signals are high-quality but lack coverage, such as medical diagnostics (expert ratings + EHRs) and educational assessments (test scores + online behavior).
- **Engineering wisdom in the three-layer defense mechanism**: The three layers of late fusion, selective gating, and residual connections cooperatively ensure system robustness—even if external data is completely missing or carries significant noise, the system gracefully degrades to the survey-only baseline.
- **Greater improvements on rare attributes**: This is the most compelling conclusion of LANTERN—the greatest value of external data lies in covering survey blind spots (user attributes out of reach for conditional questions), rather than gilding the lily on attributes already adequately documented by survey data.

## Limitations & Future Work

- Evaluated only on a single survey dataset (~35,000 users); the scale and diversity of the dataset are limited, lacking cross-domain validation.
- Lacks comparison against LLM-based user modeling methods (e.g., using LLMs to understand text descriptions of user behavior).
- Class imbalance handling in multi-label prediction is not discussed in depth, only touched upon indirectly through threshold analysis.
- Specific architectural details of the survey encoder and external encoder (number of layers, dimensions, etc.) are not fully described.
- Distribution analysis of the gate value $g$ is missing—what values of $g$ are learned for different attributes? Which external signals are gated off?
- Absence of interpretability analysis—could cross-attention weights reveal which external features contribute to which survey predictions?
- Choice of magnitude for Gaussian noise regularization is not discussed.

## Related Work & Insights

- **DMT/ZEUS (Gu et al., 2020/2021)**: Two-stage user modeling via concatenating sequential embeddings of different behavior types; LANTERN's late fusion strategy is more refined than simple concatenation.
- **MMoE (Ma et al., 2018)**: Multi-gate Mixture-of-Experts framework for task-specific routing, exhibiting strong performance in recommendation/advertising; LANTERN's gating concept is related but focused on modalities rather than tasks.
- **ViLBERT (Lu et al., 2019)**: Two-stream architecture with co-attentional fusion of vision and language, inspiring the cross-attention design of LANTERN.
- **Multimodal Transformer (Tsai et al., 2019)**: Cross-modal attention without requiring temporal alignment; LANTERN adapts this fusion strategy specifically to tabular data scenarios.
- *Insight*: In industrial scenarios dominated by tabular data, simple yet effective fusion strategies (late fusion + gating + residuals) are often more practical than complex multimodal architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (Late fusion, gating, and cross-attention are existing components; the innovation lies in their tailored combination for survey scenarios.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Ablation is complete, and rare/frequent analysis is insightful, but limited to a single dataset and lacks comparison with more baselines.)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (The motivation is clearly articulated, the design philosophy is consistently applied, and industrial deployment considerations are pragmatic.)
- **Value**: ⭐⭐⭐⭐ (Provides a practical industrial blueprint, though the academic depth and breadth are somewhat limited.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AdaptGrad: Adaptive Sampling to Reduce Noise](../../NeurIPS2025/interpretability/adaptgrad_adaptive_sampling_to_reduce_noise.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](../../NeurIPS2025/interpretability/cbmas_cognitive_behavioral_modeling_via_activation_steering.md)
- [\[ICML 2025\] SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior](safetyanalyst_interpretable_transparent_and_steerable_safety_moderation_for_ai_b.md)
- [\[ICLR 2026\] Behavior Learning (BL)](../../ICLR2026/interpretability/behavior_learning_bl.md)
- [\[ICML 2025\] On the Power of Context-Enhanced Learning in LLMs](on_the_power_of_context-enhanced_learning_in_llms.md)

</div>

<!-- RELATED:END -->
