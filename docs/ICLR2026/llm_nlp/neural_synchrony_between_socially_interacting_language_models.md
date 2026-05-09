---
title: >-
  [Paper Note] Neural Synchrony Between Socially Interacting Language Models
description: >-
  [ICLR 2026][LLM/NLP][neural synchrony] This paper presents the first investigation of neural synchrony between LLMs engaged in social interaction. By training affine transformations to predict a partner model's future representations, it defines the $SyncR^2$ metric to quantify synchrony strength. The results show that synchrony depends on social engagement and temporal proximity, and correlates strongly with LLMs' social behavioral performance (Pearson $r$ = 0.88–0.99), echoing neuroscientific findings on inter-brain synchrony (IBS) in humans.
tags:
  - ICLR 2026
  - LLM/NLP
  - neural synchrony
  - social interaction
  - LLM representation analysis
  - multi-agent systems
  - inter-brain synchrony analogy
  - predictability
date: 2026-05-08
content_hash: 7169c4487daa1d9c
---

# Neural Synchrony Between Socially Interacting Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.17815](https://arxiv.org/abs/2602.17815)
**Code**: [zzn-nzz/LM_neural_synchrony](https://github.com/zzn-nzz/LM_neural_synchrony)
**Area**: LLM/NLP
**Keywords**: neural synchrony, social interaction, LLM representation analysis, multi-agent systems, inter-brain synchrony analogy, predictability

## TL;DR

This paper presents the first investigation of neural synchrony between LLMs engaged in social interaction. By training affine transformations to predict a partner model's future representations, it defines the $SyncR^2$ metric to quantify synchrony strength. The results show that synchrony depends on social engagement and temporal proximity, and correlates strongly with LLMs' social behavioral performance (Pearson $r$ = 0.88–0.99), echoing neuroscientific findings on inter-brain synchrony (IBS) in humans.

## Background & Motivation

### Inter-Brain Synchrony (IBS) in Humans

Neuroscience has established that during social interaction (conversation, cooperation, joint attention), human brain activity becomes synchronized. This **inter-brain synchrony** (IBS) is not merely a byproduct of shared sensory input; it is a functional mechanism that predicts and facilitates social coordination, cooperation, and mutual understanding. Stronger IBS is associated with higher cooperation rates, better learning outcomes, and superior team performance.

### Research Motivation

While LLMs demonstrate remarkable social interaction capabilities at the behavioral level, whether analogous internal mechanisms to the human social brain exist at the **representational level** remains unknown. Prior work has primarily focused on behavioral evaluation (e.g., Theory of Mind tests) or single-model internal analysis (e.g., specific attention heads), leaving the **representational dynamics during multi-model interaction** largely unexplored.

### Core Hypothesis

If LLMs in social interaction not only act according to their own roles but also reason about a partner's emotions, intentions, and interaction trajectories, then one LLM's internal representations should contain information predictive of another LLM's representations.

## Method

### Overall Architecture

1. Two LLM agents engage in multi-turn social interaction within the Sotopia environment, each with independent background, personality, and goals.
2. At each turn, the hidden state of the last token of the prompt is extracted as the representation.
3. An affine transformation is trained to predict the interaction partner's future representations, and prediction performance is used to quantify synchrony.

### Representation Extraction

For a $T$-turn dialogue, each LLM backbone $M \in \{A, B\}$ produces at turn $t$:

$$\boldsymbol{h}_t^{(M)} \in \mathbb{R}^{L_M \times D_M}, \quad t = 1, \dots, T$$

The hidden states across all layers are extracted at the position of the last token of the prompt input, which integrates information from all preceding tokens.

### Dataset Construction

The experimental condition uses temporally aligned representation pairs from real interactions:

$$\mathcal{D}^{A \to B}_{l_A \to l_B} = \{(\boldsymbol{h}^{(A)}_{t, l_A}, \boldsymbol{h}^{(B)}_{t, l_B}) \mid t = 1, \dots, T\}$$

### Learning the Affine Transformation

Ridge regression with an intercept term is employed:

$$\hat{\boldsymbol{W}}, \hat{\boldsymbol{b}} = \arg\min_{\boldsymbol{W}, \boldsymbol{b}} \|\boldsymbol{Y} - \boldsymbol{X}\boldsymbol{W} - \mathbf{1}\boldsymbol{b}\|_F^2 + \lambda \|\boldsymbol{W}\|_F^2$$

The regularization coefficient is $\lambda = 0.1$; the intercept is not regularized.

### $SyncR^2$ Metric Definition

For each layer $l_A$ of the source model $A$, the best-matching layer in the target model $B$ is identified:

$$r_A^{\star}(l_A) = \max_{l_B} R^2_{\text{test}}(l_A \to l_B), \quad \tilde{r}_A(l_A) = \max\{0, r_A^{\star}(l_A)\}$$

This is then averaged across all layers and symmetrized bidirectionally:

$$SyncR^2(A, B) = \frac{1}{2}(SyncR^2(A \to B) + SyncR^2(B \to A))$$

Negative $R^2$ values are clipped to 0, as performance worse than predicting the mean indicates no synchrony.

### Key Control Experiments

**Control 1 (No Social Engagement)**: A "passive" agent is introduced — it reads the dialogue history but does not generate replies or engage in role-playing. Synchrony is expected to decrease.

**Control 2 (No Temporal Proximity)**: Source representations are paired with target representations $k$ turns later ($k \geq 1$). If synchrony merely reflects static representational similarity, it should not decay with temporal lag.

$$\mathcal{D}^{\text{lag-}k, A \to B}_{l_A \to l_B} = \{(\boldsymbol{h}^{(A)}_{t, l_A}, \boldsymbol{h}^{(B)}_{t+k, l_B}) \mid t = 1, \dots, T-k\}$$

## Key Experimental Results

### Experimental Setup

- **6 open-source models**: Mistral-7B-v0.1/v0.2/v0.3, Llama-2-7B-Chat, Llama-3-8B, Llama-3.2-3B
- **21 model pairs**: covering intra-family and cross-family pairings
- **450 interaction scenarios**, up to 8 turns each, 3 random seeds
- 6,500 samples fixed per model pair

### Main Results: Control Condition Validation

| Condition | $SyncR^2$ Level |
|-----------|----------------|
| Experimental (real interaction) | Significantly high (0.1–0.3+) |
| Control 1 (no social engagement) | Substantially reduced |
| Control 2 (temporal lag $k \geq 1$) | Rapidly collapses to ≈0 |

The results confirm that neural synchrony genuinely depends on **authentic social engagement** and **temporal proximity**.

### Key Finding: Synchrony–Performance Correlation

| Model Family Type | Pearson $r$ | $p$-value |
|------------------|------------|-----------|
| Mistral family (3 pairs) | 0.88 | $< 0.05$ |
| Cross-family (Mistral × Llama) | 0.89 | $< 0.001$ |
| Llama family (3 pairs) | 0.99 | $< 0.001$ |

More synchronized model pairs systematically achieve better social performance.

### Controlling for Confounds

IFEval (instruction following) and MuSR (long-context reasoning) are used as control variables for partial correlation analysis:

| Model Family | After Controlling IFEval | After Controlling MuSR |
|-------------|--------------------------|------------------------|
| Mistral | 0.81 | 0.92 |
| Cross | 0.71 | 0.89 |
| Llama | 0.27 | 0.99 |

Correlations remain positive and mostly significant after controlling, demonstrating that synchrony reflects **socially specific capabilities** rather than a byproduct of general ability.

### Effect of Relationship Intimacy

The $SyncR^2$ distribution shifts upward as the intimacy of the relationship between agents increases — closer social relationships are accompanied by stronger neural synchrony, echoing findings in human neuroscience (e.g., couples exhibit stronger IBS than strangers).

### Key Findings

1. The affine transformation (a minimalist assumption) effectively captures synchrony; nonlinear transformations do not significantly improve generalization performance.
2. Synchrony is most pronounced in intermediate layers.
3. LLM representations encode the interaction partner's emotional states and can predict the partner's future emotional and action distributions.

## Highlights & Insights

1. **Pioneering perspective**: The first work to transfer the neuroscientific concept of IBS to the LLM domain, bridging human social cognition and AI systems.
2. **Rigorous control design**: The two control conditions — social engagement and temporal proximity — rule out multiple alternative explanations.
3. **The power of parsimony**: A linear (affine) transformation suffices to reveal deep representational synchrony, supporting the hypothesis of linear structure in LLM representations.
4. **Implicit evidence for Theory of Mind**: Agent representations encode the partner's invisible internal states (e.g., emotions), suggesting implicit ToM capability.
5. **Resonance with social predictive coding theory**: The affine transformation directly operationalizes "prediction of another's future state."

## Limitations & Future Work

1. **Limited model scale**: The largest model is only 8B parameters; large-scale models prevalent today (e.g., 70B+) are not included.
2. **Sotopia environment only**: The diversity of social interaction scenarios may not adequately represent real-world social interaction.
3. **Causal direction unclear**: Whether the synchrony–performance correlation is causal, or whether both are reflections of model capability, remains unresolved.
4. **Limitations of the affine transformation**: While effective, it may miss nonlinear synchrony patterns in representations.
5. **Evaluator dependency**: Social performance scores rely on GPT models, which may introduce systematic bias.

## Related Work & Insights

- **IBS neuroscience** (Dumas et al., 2010; Hasson et al., 2012): The direct inspiration for this work.
- **Brain–LLM alignment** (Mischler et al., 2024): Demonstrates similarity between LLM and brain representations, but is limited to single models.
- **Sotopia** (Zhou et al., 2023): Provides the infrastructure for the social simulation environment.
- **Inspiration**: Offers a novel representational analysis perspective for multi-agent system design — collaborative performance may be improved by optimizing inter-agent synchrony.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Highly original; opens an entirely new research direction in the social neuroscience of LLMs.
- **Technical Depth**: ⭐⭐⭐⭐ — The method is simple yet the control experiment design is sophisticated and the statistical analysis is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 21 model pairs, 450 scenarios, and multiple control conditions, though model scale is limited.
- **Value**: ⭐⭐⭐ — Currently more analytical and inspirational in nature; direct application pathways remain to be clarified.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A highly interesting and substantive interdisciplinary work that opens a new window into understanding the "social mind" of LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](../../ACL2026/llm_nlp/an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](the_lattice_representation_hypothesis_of_large_language_models.md)

<!-- RELATED:END -->
