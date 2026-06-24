---
title: >-
  [Paper Note] Neural Synchrony Between Socially Interacting Language Models
description: >-
  [ICLR 2026][LLM (Other)][Neural Synchrony] This study presents the first investigation of neural synchrony between LLMs during social interaction. By training affine transformations to predict the future representations of interaction partners, the authors define the $SyncR^2$ metric to quantify synchrony strength. Findings indicate that this synchrony depends on social engagement and temporal proximity, and correlates highly with the social performance of LLMs (Pearson $r$ =…
tags:
  - "ICLR 2026"
  - "LLM (Other)"
  - "Neural Synchrony"
  - "Social Interaction"
  - "LLM Representation Analysis"
  - "Multi-Agent Systems"
  - "Inter-Brain Synchrony Analogy"
  - "Predictability"
date: 2026-05-08
content_hash: f9c52ea4255cac20
---

# Neural Synchrony Between Socially Interacting Language Models

**Conference**: ICLR 2026  
**arXiv**: [2602.17815](https://arxiv.org/abs/2602.17815)  
**Code**: [zzn-nzz/LM_neural_synchrony](https://github.com/zzn-nzz/LM_neural_synchrony)  
**Area**: LLM/NLP  
**Keywords**: Neural Synchrony, Social Interaction, LLM Representation Analysis, Multi-Agent Systems, Inter-Brain Synchrony Analogy, Predictability

## TL;DR

This study presents the first investigation of neural synchrony between LLMs during social interaction. By training affine transformations to predict the future representations of interaction partners, the authors define the $SyncR^2$ metric to quantify synchrony strength. Findings indicate that this synchrony depends on social engagement and temporal proximity, and correlates highly with the social performance of LLMs (Pearson $r$ = 0.88-0.99), mirroring neuroscientific findings of human Inter-Brain Synchrony (IBS).

## Background & Motivation

### Human Inter-Brain Synchrony (IBS)

Neuroscience has revealed that when humans interact socially (dialogue, cooperation, joint attention), their brain activities synchronize. This **Inter-Brain Synchrony** (IBS) is not merely a byproduct of shared sensory input but a functional mechanism for predicting and facilitating social coordination, cooperation, and mutual understanding. Stronger IBS is associated with higher cooperation rates, better learning outcomes, and superior team performance.

### Research Motivation

LLMs exhibit impressive social interaction capabilities at the behavioral level, but whether internal mechanisms similar to the human social brain exist at the **representational level** remains unknown. Prior work has primarily focused on behavioral assessments (e.g., Theory of Mind tests) or single-model internal analysis (e.g., specific attention heads), lacking studies on **representational dynamics during multi-model interactions**.

### Core Hypothesis

If LLMs not only act based on their own roles but also reason about the emotions, intentions, and interaction trajectories of their partners, then the internal representations of one LLM should contain information that predicts the representations of the other.

## Method

### Overall Architecture

The study addresses whether the internal representations of one LLM encode information about its partner during social interaction. The analysis follows a pipeline: first, two LLM agents engage in multi-round social interactions in the Sotopia environment. In each round, per-layer hidden states are extracted from the last token of their respective prompts, serving as the "neural activity" of the model at that moment. Time-aligned representations are paired, and cross-model affine transformations are trained to predict the partner's representations. High prediction accuracy from this linear mapping indicates that one agent's internal state encodes the other's information. Finally, per-layer, bidirectional prediction accuracies are aggregated into a symmetric $SyncR^2$ score to represent synchrony strength. To isolate dynamic synchrony from static similarity, two control groups (removing social participation and removing temporal proximity) were designed.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    START["Two LLM agents<br/>Sotopia multi-round social interaction"] --> EXTRACT["Representation Extraction<br/>Per-layer hidden states<br/>at last prompt token"]
    EXTRACT --> AFFINE["Affine Transform + Ridge Regression<br/>Time-aligned pairing<br/>Cross-model linear mapping"]
    AFFINE --> SYNC["SyncR² Metric<br/>Per-layer best match + Max(0, R²)<br/>Bidirectional average"]
    SYNC --> OUT["Pairwise Synchrony Score<br/>SyncR²(A,B)"]
    EXTRACT -.->|Passive agent (read-only)<br/>/ Lag-k pairing| CTRL["Two Control Groups<br/>No Social Participation / No Temporal Proximity"]
    CTRL -.-> AFFINE
```

### Key Designs

**1. Representation Extraction: Compressing dialogues into comparable vectors**

To measure synchrony, the "state of the model" must be converted into a vector that is alignable across models. For $T$ rounds of dialogue, each backbone $M \in \{A, B\}$ generates per-layer hidden states $\boldsymbol{h}_t^{(M)} \in \mathbb{R}^{L_M \times D_M}$ at round $t$. The authors extract representations only from the last token position of the prompt input, as the autoregressive attention mechanism ensures this position integrates information from all preceding tokens, serving as a concentrated carrier of the context.

**2. Affine Transformation + Ridge Regression: Testing predictability with linear mappings**

To determine if one model "encodes" another, time-aligned representations from the same interaction are paired to form a dataset $\mathcal{D}^{A \to B}_{l_A \to l_B} = \{(\boldsymbol{h}^{(A)}_{t, l_A}, \boldsymbol{h}^{(B)}_{t, l_B}) \mid t = 1, \dots, T\}$. A Ridge regression with an intercept solves the mapping from source layer $l_A$ to target layer $l_B$:

$$\hat{\boldsymbol{W}}, \hat{\boldsymbol{b}} = \arg\min_{\boldsymbol{W}, \boldsymbol{b}} \|\boldsymbol{Y} - \boldsymbol{X}\boldsymbol{W} - \mathbf{1}\boldsymbol{b}\|_F^2 + \lambda \|\boldsymbol{W}\|_F^2$$

The regularization parameter is set to $\lambda = 0.1$. Using a linear rather than a non-linear predictor ensures that synchrony is a genuine representational property rather than an artifact of a powerful predictor’s capacity.

**3. $SyncR^2$ Metric: Aggregating accuracy into a symmetric score**

Since synchrony may not be restricted to fixed layers, for each layer $l_A$ of source model $A$, the best match is found in the target model $B$: $r_A^{\star}(l_A) = \max_{l_B} R^2_{\text{test}}(l_A \to l_B)$. Negative values are truncated to zero $\tilde{r}_A(l_A) = \max\{0, r_A^{\star}(l_A)\}$. The mean across all layers yields $SyncR^2(A \to B)$, which is then symmetrized:

$$SyncR^2(A, B) = \tfrac{1}{2}\big(SyncR^2(A \to B) + SyncR^2(B \to A)\big)$$

**4. Two Control Groups: Decoupling synchrony from static similarity**

To exclude the possibility that models are simply similar in structure, two control groups were established. Control 1 (No Social Participation) introduces a "passive" agent that only reads dialogue history without generating responses. Control 2 (No Temporal Proximity) pairs source representations with target representations from $k$ rounds later ($k \geq 1$). If synchrony arises from real-time interaction, $SyncR^2$ should collapse as $k$ increases.

## Key Experimental Results

### Experimental Setup

- **6 Open-source Models**: Mistral-7B-v0.1/v0.2/v0.3, Llama-2-7B-Chat, Llama-3-8B, Llama-3.2-3B.
- **21 Model Pairs**: Including intra-family and cross-family pairings.
- **450 Interaction Scenarios**, maximum 8 rounds per scenario, 3 random seeds.
- 6,500 samples per model pair.

### Main Results: Control Group Validation

| Condition | $SyncR^2$ Level |
|-----------|----------------|
| Experimental Group (Real Interaction) | Significantly High (0.1-0.3+) |
| Control Group 1 (No Social Participation) | Substantial Decrease |
| Control Group 2 (Temporal Lag $k \geq 1$) | Rapid collapse to ≈0 |

The results confirm that neural synchrony depends on **active social participation** and **temporal proximity**.

### Key Findings: Correlation Between Synchrony and Social Performance

| Model Family Type | Pearson $r$ | $p$-value |
|-------------------|-------------|-----------|
| Mistral Family (3 pairs) | 0.88 | $< 0.05$ |
| Cross-family (Mistral×Llama) | 0.89 | $< 0.001$ |
| Llama Family (3 pairs) | 0.99 | $< 0.001$ |

More synchronized model pairs systematically achieve better social performance.

### Controlling Confounds

Partial correlations were calculated using IFEval (instruction following) and MuSR (long-context reasoning) as control variables:

| Model Family | Control IFEval | Control MuSR |
|--------------|----------------|--------------|
| Mistral | 0.81 | 0.92 |
| Cross | 0.71 | 0.89 |
| Llama | 0.27 | 0.99 |

The positive and mostly significant correlations after controlling for these factors prove that synchrony reflects **social-specific capabilities**.

### Impact of Relationship Intimacy

The $SyncR^2$ distribution shifts upward as relationship intimacy between agents increases—closer social relations are accompanied by stronger neural synchrony, mirroring neuroscientific findings (e.g., higher IBS in couples vs. strangers).

### Key Findings

1. Affine transformations (minimal assumption) effectively capture synchrony; non-linear transformations do not significantly improve generalization.
2. Synchrony is most pronounced in middle layers.
3. LLM representations encode the emotional states of interaction partners and predict their future emotional and action distributions.

## Highlights & Insights

1. **Pioneering Perspective**: First to transfer the concept of IBS from neuroscience to LLMs, bridging human social cognition and AI systems.
2. **Rigorous Control Design**: Two control groups (social participation and temporal proximity) successfully rule out alternative explanations.
3. **Power of Simple Methods**: Using linear mappings to reveal deep representational synchrony supports the hypothesis of linear structures in LLM representations.
4. **Implicit Evidence of ToM**: Representations encode invisible internal states (e.g., partner's emotions), suggesting implicit Theory of Mind capabilities.
5. **Alignment with Social Predictive Coding**: The affine transformation operationalizes the "prediction of others' future states."

## Limitations & Future Work

1. **Limited Model Scale**: Restricted to models up to 8B parameters; lacks analysis of current large-scale models (e.g., 70B+).
2. **Environment Constraint**: The Sotopia environment might not represent the full diversity of real-world social interactions.
3. **Causal Ambiguity**: Whether synchrony causes performance or both reflect model capacity remains to be clarified.
4. **Affine Transformation Limits**: May overlook non-linear synchrony patterns in representations.
5. **Evaluator Bias**: Dependence on GPT-based scoring for social performance may introduce systematic bias.

## Related Work & Insights

- **IBS Neuroscience** (Dumas et al., 2010; Hasson et al., 2012): Direct inspiration for this study.
- **Brain-LLM Alignment** (Mischler et al., 2024): Demonstrates similarity between LLM and brain representations but limited to single-model analysis.
- **Sotopia** (Zhou et al., 2023): Provides the social simulation infrastructure.
- **Insight**: Provides a new representational analysis perspective for multi-agent systems—cooperative performance might be improved by optimizing inter-agent synchrony.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Highly original, opening a new research direction in social neuroscience for LLMs.
- **Technical Depth**: ⭐⭐⭐⭐ — Methodologically simple but with sophisticated control experiments and rigorous statistical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 21 model pairs and 450 scenarios, though limited by model size.
- **Value**: ⭐⭐⭐ — Currently more analytical and heuristic; direct application paths require further exploration.
- **Overall**: ⭐⭐⭐⭐ — An insightful interdisciplinary work that opens a window into understanding the "social mind" of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](../../ACL2026/llm_nlp/repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](../../ACL2025/llm_nlp/information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2025\] Neural Topic Modeling with Large Language Models in the Loop](../../ACL2025/llm_nlp/neural_topic_modeling_with_large_language_models_in_the_loop.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](../../ACL2026/llm_nlp/an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](../../ACL2025/llm_nlp/plugin_finetuning_bridge.md)

</div>

<!-- RELATED:END -->
