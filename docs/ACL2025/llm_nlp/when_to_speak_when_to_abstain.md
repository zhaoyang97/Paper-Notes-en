---
title: >-
  [Paper Note] When to Speak, When to Abstain: Contrastive Decoding with Abstention
description: >-
  [ACL 2025][LLM (Other)][contrastive decoding] Proposes CDA (Contrastive Decoding with Abstention), a training-free decoding method. By using entropy-calibrated uncertainty estimation, CDA enables LLMs to generate correct answers when parametric/contextual knowledge is available, and to actively abstain when both are unreliable, covering all four knowledge availability scenarios.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "contrastive decoding"
  - "abstention"
  - "knowledge conflict"
  - "parametric vs contextual"
  - "training-free"
date: 2026-05-08
content_hash: 60c4ecb1e0883c75
---

# When to Speak, When to Abstain: Contrastive Decoding with Abstention

**Conference**: ACL 2025  
**arXiv**: [2412.12527](https://arxiv.org/abs/2412.12527)  
**Code**: None  
**Area**: Other  
**Keywords**: contrastive decoding, abstention, knowledge conflict, parametric vs contextual, training-free

## TL;DR
Proposes CDA (Contrastive Decoding with Abstention), a training-free decoding method. By using entropy-calibrated uncertainty estimation, CDA enables LLMs to generate correct answers when parametric/contextual knowledge is available, and to actively abstain when both are unreliable, covering all four knowledge availability scenarios.

## Background & Motivation

### Background

**Background**: LLMs possess both parametric knowledge $\mathcal{P}$ (acquired during pre-training) and contextual knowledge $\mathcal{C}$ (provided during inference). Existing context contrastive decoding (CCD) methods choose a more reliable knowledge source by contrasting the output distributions of these two types of knowledge.

**Limitations of Prior Work**: CCD assumes that at least one knowledge source is available, but in reality, scenarios where both are unreliable are common. In such cases, LLMs should abstain rather than forcefully answer, which leads to hallucinations.

**Key Challenge**: Abstention requires accurate evaluation of knowledge relevance and its integration into the generation process, both of which are highly challenging.

**Goal**: Design a training-free decoding method that covers all four scenarios: $\mathcal{P}$=1,$\mathcal{C}$=1 (answering); $\mathcal{P}$=1,$\mathcal{C}$=0 (relying on parameters); $\mathcal{P}$=0,$\mathcal{C}$=1 (relying on context); $\mathcal{P}$=0,$\mathcal{C}$=0 (abstention).

## Method

### Overall Architecture
Weighted fusion of three output distributions: $d^o_t = w^p_t \cdot d^p_t + w^c_t \cdot d^c_t + (1-w^p_t-w^c_t) \cdot d^a_t$, where $d^p_t$ is from a parametric template, $d^c_t$ is from a contextual template, and $d^a_t$ is from an explicit abstention instruction template.

### Key Designs

1. **Three-Distribution Fusion**:

    - $d^p_t = \text{logit}_\theta(y_t | \mathcal{T}_p(x, y_{<t}))$ (parametric distribution without context)
    - $d^c_t = \text{logit}_\theta(y_t | \mathcal{T}_c(c, x, y_{<t}))$ (distribution with context)
    - $d^a_t = \text{logit}_\theta(y_t | \mathcal{T}_a(c, x, y_{<t}))$ (abstention instruction distribution)
    - Abstention weight $w^a_t = 1 - w^p_t - w^c_t$: Automatically increases when both types of knowledge are uncertain.

2. **Entropy-Calibrated Knowledge Relevance Estimation**:

    - Compute the entropy of the parametric/contextual distributions, $\mathcal{H}^p_t$ and $\mathcal{H}^c_t$.
    - Compute the baseline entropy of the null distributions, $\bar{\mathcal{H}}^p_t$ and $\bar{\mathcal{H}}^c_t$, using a "content-free" null prompt (replacing specific inputs with placeholders).
    - Relevance = additional information provided by knowledge = $r^p_t = \frac{\max(\mathcal{H}^p_t - \bar{\mathcal{H}}^p_t, 0)}{\bar{\mathcal{H}}^p_t}$.
    - Normalization to obtain the final weight: $w^p_t = \frac{r^p_t}{r^p_t + r^c_t} \cdot r^p_t$.
    - **Design Motivation**: Directly comparing $\mathcal{H}^p$ and $\mathcal{H}^c$ is unfair (due to different conditions); calibration eliminates the model's intrinsic bias.

3. **Momentum Smoothing (CDA-m)**:

    - $w_t \leftarrow \alpha \cdot w_{t-1} + (1-\alpha) \cdot w_t$, preventing weights from fluctuating drastically between adjacent tokens.

### Controlled Testbed
A four-scenario dataset was carefully constructed: parametric knowledge was estimated using the generation consistency rate ($r=0 \rightarrow$ absent, $r>\eta \rightarrow$ present), and SBERT similarity was used to select irrelevant context, ensuring that the knowledge state of each scenario remained controllable.

## Key Experimental Results

### Main Results (4 LLMs × 3 QA Datasets)

| Scenario | Expected Behavior | CDA Performance |
|------|---------|----------|
| P=1, C=1 | Correct Answer | ✓ High accuracy |
| P=1, C=0 | Rely on parameters | ✓ Not misled by noise |
| P=0, C=1 | Rely on context | ✓ Accurately follow |
| P=0, C=0 | **Abstain** | ✓ **Effective abstention** |

CDA outperforms existing CCD methods (e.g., Zhao et al., Shi et al.) across all scenarios, and displays superior generalization performance compared to training-based abstention methods (e.g., Zhang et al.). It is equally effective in RAG scenarios.

## Highlights & Insights
- **Using "content-free" null prompts for entropy calibration** is highly clever: replacing inputs with placeholders removes semantic information and retains only template bias as a baseline for calibration.
- **Automatic abstention mechanism via three-distribution fusion**: No explicit judgment of "whether to abstain" is required; instead, weights naturally decay to the abstention distribution, which is elegant and requires no threshold setting.
- **Training-free**: Directly applicable to any off-the-shelf LLM without altering parameters or architecture, making it highly practical.

## Limitations & Future Work
- The design of the abstention instruction template may affect performance, requiring proper prompt engineering.
- Each decoding step requires three forward passes (parametric, contextual, and abstention), leading to approximately 3x inference cost.
- The choice of "content-free" prompts in entropy calibration may lack robustness.

## Related Work & Insights
- **vs Existing CCD (Zhao et al. 2024)**: Projects only contrast parametric and contextual knowledge, and fail to handle scenarios where both are unreliable; CDA expands this to a tri-distribution fusion containing abstention.
- **vs Training-based Abstention (Zhang et al. 2024)**: Requires fine-tuning the LLM to learn the ability to abstain; CDA is training-free and exhibits better generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ First to naturally integrate abstention into the contrastive decoding framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Well-controlled validation across 4 scenarios + 4 LLMs + RAG.
- Writing Quality: ⭐⭐⭐⭐ Rigorous testbed design, clear mathematical derivation.
- Value: ⭐⭐⭐⭐⭐ Improving LLM reliability without training, highly practical for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] What Happened in LLM Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective](what_happened_in_llms_layers_when_trained_for_fast_vs_slow_thinking_a_gradient_p.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models](dehumanizing_metaphors_immigration.md)

</div>

<!-- RELATED:END -->
