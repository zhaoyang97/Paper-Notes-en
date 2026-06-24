---
title: >-
  [Paper Note] Mamba Knockout for Unraveling Factual Information Flow
description: >-
  [ACL 2025][LLM Safety][Mamba] This work transfers the Attention Knockout interpretability method from Transformers to Mamba-1 and Mamba-2, revealing the factual information flow patterns in SSM models. Key findings show that Mamba and Transformers share a universal pattern where "subject tokens transmit key information to the last token in mid-to-late layers," but differ in architecture-specific aspects such as first-token bias and dependency on relation tokens.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Mamba"
  - "SSM"
  - "Attention Knockout"
  - "Factual Information Flow"
  - "Interpretability"
date: 2026-05-08
content_hash: 259f0181d77b0f64
---

# Mamba Knockout for Unraveling Factual Information Flow

**Conference**: ACL 2025  
**arXiv**: [2505.24244](https://arxiv.org/abs/2505.24244)  
**Code**: [Yes](https://github.com/nirendy/mamba-knockout)  
**Area**: LLM Safety  
**Keywords**: Mamba, SSM, Attention Knockout, Factual Information Flow, Interpretability

## TL;DR

This work transfers the Attention Knockout interpretability method from Transformers to Mamba-1 and Mamba-2, revealing the factual information flow patterns in SSM models. Key findings show that Mamba and Transformers share a universal pattern where "subject tokens transmit key information to the last token in mid-to-late layers," but differ in architecture-specific aspects such as first-token bias and dependency on relation tokens.

## Background & Motivation

The internal factual information flow inside Transformer models has been widely studied (e.g., Attention Knockout by Geva et al. 2023), but factual knowledge storage and propagation mechanisms within State Space Model (SSM)-based Mamba architectures remain largely unexplored.

Key theoretical connection: Recent studies (Ali et al., 2024; Dao & Gu, 2024) have demonstrated that selective SSMs can be understood through an "implicit attention" lens, and Mamba-2 has been directly proven equivalent to a class of linear attention Transformers. This theoretical bridge makes it possible to transfer Transformer-based interpretability tools to Mamba.

**Core Problem**:
1. Is the factual information flow pattern in Mamba consistent with that of Transformers? Which aspects are architecture-generic, and which are architecture-specific?
2. What role do the unique structures of SSMs (context-dependent vs. context-independent features) play respectively?

## Method

### Overall Architecture

The methodology of this work encompasses two levels:

1. **Attention Knockout**: Transferred from Transformer to Mamba to analyze token-to-token information flow.
2. **Feature Knockout**: Leverages the unique structure of SSMs to analyze the roles of different types of features.

### Key Designs

#### 1. Implicit Attention Knockout for Mamba-1

**Function**: Zeroes out the connections between two tokens in the kernel matrix (equivalent to the attention matrix) of Mamba-1 to observe the impact on predictions.

**Mechanism**: Utilizing the implicit attention perspective proposed by Ali et al. (2024), the selective SSM in Mamba-1 can be represented as a kernel matrix:

$$\mathbf{M}_{i,j} = Q_i \cdot H_{i,j} \cdot K_j$$

where $Q_i = C(i)$, $K_j = B(j)$, and $H_{i,j} = \prod_{t=i}^{j} A(t)$. To knock out the information flow from token $i$ to $j$, one directly sets $\mathbf{M}_{i,j} = 0$.

**Design Motivation**: Although Sharma et al. (2024) questioned the feasibility of fine-grained blocking (due to convolution and softmax layers), experiments demonstrate that direct implementation is highly effective at replicating the phenomena observed in Transformers.

#### 2. Implicit Linear Attention Knockout for Mamba-2

**Function**: In Mamba-2, the SSM layer is directly formulated as a masked linear attention $\mathbf{L} \circ (\mathbf{X}\mathbf{M}\mathbf{X}^\top)\mathbf{X}$, where the $(i, j)$ element of the attention matrix quantifies how much token $i$ attends to token $j$. Setting this directly to zero conducts the knockout.

**Design Motivation**: The SSD framework of Mamba-2 explicitly establishes equivalence with linear attention, providing the knockout operation with a more direct semantic interpretation.

#### 3. Feature Knockout (Original Contribution)

**Function**: Leverages the property that each feature in SSM is modeled by an independent SSM to selectively knock out features based on their types.

**Mechanism**: Features are classified into two categories based on the decay characteristics of the state transition matrix A̅:
- **Context-dependent features** (top 1/3 largest $\|\bar{A}\|_1$): A̅ ≈ 1, preserving long-range historical information and responsible for information propagation between tokens.
- **Context-independent features** (bottom 1/3 smallest $\|\bar{A}\|_1$): A̅ ≈ 0, decaying rapidly, focusing only on local information, and responsible for enriching single-token representations.

**Design Motivation**: This is an analytical dimension unique to SSMs—Transformers do not possess a counterpart concept of "decaying features." Comparing the knockout performance of these two types of features reveals their distinct roles in factual reasoning.

### Loss & Training

This work focuses on interpretability analysis and does not involve model training. All experiments are based on inference-time interventions on pre-trained models.

## Key Experimental Results

### Main Results

**Impact of Subject Token Knockout on Correct Prediction Probability (Relative Change)**:

| Model | Mid-to-Late Layer Knockout Effect | Subject Dependency | First Token Bias | Relation Token Dependency |
|------|----------------|----------|---------------|-----------------|
| GPT-2 (355M-1.5B) | Significant probability drop | ✓ Strong | ✓ Strong | Dependency in early layers, weakening in later layers |
| Mamba-1 (130M-2.8B) | Significant probability drop | ✓ Strong | ✗ Weak | Dependency in later layers, dropping then rising |
| Mamba-2 (130M-2.7B) | Significant probability drop | ✓ Strong | ✗ Weak | Dependency in later layers |
| Falcon-Mamba | Significant probability drop | ✓ Strong | ✗ Weak | Dropping then rising in later layers |

### Ablation Study

**Feature Knockout Experiment (Subject $\to$ Last Token, primarily on Mamba-1/2)**:

| Knockout Range | Effect |
|----------|------|
| All features | Significant probability drop (Baseline) |
| Context-dependent features only | Effect is almost equivalent to knocking out all features |
| Context-independent features only | Almost no impact |

This proves that **context-dependent features** serve as the primary carrier for the transmission of factual information between tokens.

**Window Size Ablation (Mamba-1 130M/1.4B/2.8B)**:

| Window Size | 1 | 3 | 5 | 9 | 12 | 15 |
|----------|---|---|---|---|----|----|
| Effect | Weak | Moderate | Obvious | Obvious | Strong | Strongest |
| Resolution | Highest | High | Medium | Medium | Low | Lowest |

Key trade-off: Larger window sizes lead to stronger effects but lower resolution; smaller models are better suited to smaller windows (as large windows block an excessively high proportion of layers).

### Key Findings

1. **Universal Pattern**: All models (both Transformer and Mamba) exhibit mid-to-late layer reliance on information flow from subject tokens to the last token—this may represent a universal property across LLMs.
2. **No First-Token Bias in Mamba**: Unlike the strong first-token "attention sink" observed in GPT-2, neither Mamba-1 nor Mamba-2 relies on the first token.
3. **Mamba-1's Unique Last-Token Self-Dependency**: Blocking self-attention of the last token in late layers unexpectedly **significantly boosts** the correct prediction probability (nearing 1.0), a phenomenon unique to Mamba-1 and absent in Mamba-2 and GPT-2.
4. **Architectural Differences in Relation Token Dependency**: GPT-2 relies on relation tokens in early layers, whereas Mamba relies on them in later layers, exhibiting a unique drop-then-rise pattern.
5. **Dominance of Context-dependent Features in Token Communication**: Knocking out context-dependent features is almost equivalent to knocking out all features, whereas context-independent features contribute almost nothing to factual propagation.

## Highlights & Insights

- **Cross-Architecture Interpretability Transfer**: Successfully transfers Transformer's Attention Knockout framework to Mamba for the first time, proving the generalizability of interpretability techniques.
- **Original Feature Knockout Mechanism**: Leverages the decay parameter structure unique to SSMs to introduce an analytical dimension that holds no equivalent in Transformer architectures.
- **"Commonalities vs. Specifics" Categorization**: Clearly categorizes information flow patterns into architecture-generic (subject $\to$ last token in mid-to-late layers) and architecture-specific (first-token bias, relation-token timing, etc.), generating deep insights for understanding LLMs.
- **Surprising "Self-Attention Blocking Boost" in Mamba-1**: Discovers that blocking self-attention of the last token in the late layers of Mamba-1 improves accuracy, hinting at possible "information overload" mechanisms within its final layers.

## Limitations & Future Work

1. Attention knockout is restricted to operation within contiguous layer windows, unable to capture more dispersed, non-contiguous information routing.
2. Although the method identifies critical connections, it does not analyze the actual **information content** transmitted through these channels.
3. Knockout effects may not be entirely ecological—the intervention itself alters overall network dynamics, and observed drops may partially stem from out-of-distribution effects of the perturbation.
4. The study focuses solely on token-to-token info-flow, bypassing the roles of token-independent structures (e.g., gating, convolutional steps).
5. It does not explain why disparate architectures converge toward similar information flow patterns, leaving the roots of this inductive bias unaddressed.
6. The evaluation is conducted on a limited set of 672 samples from the COUNTERFACT dataset.

## Related Work & Insights

- Geva et al. (2023) established the foundational framework of Transformer Attention Knockout, which this work adopts and extends.
- Sharma et al. (2024) previously explored factual association in Mamba-1 but restricted intervention to coarse-grained blocking (single token $\to$ all tokens), whereas this work achieves fine-grained (token-to-token) blocking.
- Meng et al. (2022) used Causal Tracing to localize critical MLP layers, which complements the token-to-token information flow analysis of this work.
- The SSM-Attention equivalence theory (Ali et al., Dao & Gu) serves as the core theoretical foundation for this methodology.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Cross-architectural transfer of interpretability methods paired with an original feature knockout mechanism, providing a fresh perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Spans multiple scales of Mamba-1/2 alongside GPT-2/Llama/Mistral, featuring robust window-size ablation analyses and detailed flow heatmaps.
- **Writing Quality**: ⭐⭐⭐⭐ – Well-structured with abundant explanatory plots (10+ information flow heatmaps) and a clear comparative framework contrasting commonalities and specificities.
- **Value**: ⭐⭐⭐⭐ — Makes significant contributions to clarifying the inner mechanics of emerging SSM architectures, and the proposed feature knockout guides future paths for SSM pruning and fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective Extraction and Evaluation of Factual Claims](towards_effective_extraction_and_evaluation_of_factual_claims.md)
- [\[ACL 2025\] Core: Robust Factual Precision with Informative Sub-Claim Identification](core_robust_factual_precision_with_informative_sub-claim_identification.md)
- [\[ACL 2025\] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)
- [\[ICML 2025\] Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks](../../ICML2025/llm_safety/revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac.md)
- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](../../NeurIPS2025/llm_safety/bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)

</div>

<!-- RELATED:END -->
