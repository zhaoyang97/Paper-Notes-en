---
title: >-
  [Paper Note] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering
description: >-
  [ACL 2026][Text Generation][Emotion neurons] This paper systematically investigates the representation mechanisms and intrinsic associations of emotion and rhetoric neurons in LLMs. By proposing a neuron recognition fram…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Emotion neurons"
  - "rhetoric neurons"
  - "adaptive masking"
  - "neuron modulation"
  - "LLM interpretability"
date: 2026-05-08
content_hash: 7eb9b2fd9be18fcd
---

# Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering

**Conference**: ACL 2026  
**arXiv**: [2604.17255](https://arxiv.org/abs/2604.17255)  
**Code**: None  
**Area**: Text Generation  
**Keywords**: Emotion neurons, rhetoric neurons, adaptive masking, neuron modulation, LLM interpretability

## TL;DR

This paper systematically investigates the representation mechanisms and intrinsic associations of emotion and rhetoric neurons in LLMs. By proposing a neuron recognition framework combining multi-dimensional filtering and an adaptive masking validation method, the authors achieve directional induction of emotion/rhetoric predictions and rhetoric-neuron-assisted emotion recognition.

## Background & Motivation

**Background**: While the capabilities of LLMs in emotion understanding and rhetoric generation are increasingly important, current improvements primarily rely on external optimizations (prompt engineering, fine-tuning). Exploration of internal representation mechanisms remains insufficient. The limited existing research on neurons is restricted to emotion neurons, overlooking rhetoric neurons and the intrinsic connections between the two.

**Limitations of Prior Work**: Traditional neuron functional validation methods (forced zeroing, mean substitution) exhibit counter-intuitive phenomena in emotion and rhetoric tasks—after masking target neurons considered highly relevant, task accuracy often increases rather than decreases. This makes reliable causal validation unfeasible.

**Key Challenge**: The goal is to validate and regulate emotional/rhetorical expression through neuron intervention, but existing masking methods are unreliable. Forced zeroing may trigger functional compensation mechanisms (complementary neuron clusters taking over), while mean substitution fails to truly disrupt the specific representations of core neurons.

**Goal**: (1) Systematically identify neurons for 6 types of emotions and 4 types of rhetoric; (2) Propose a reliable causal validation method; (3) Achieve directional regulation and cross-signal enhancement.

**Key Insight**: Starting from existing differences in neuron activation, the authors design attenuated masking rather than zero-masking, paired with feedback optimization to ensure reliable masking effects.

**Core Idea**: Replace traditional hard masking with adaptive masking (dynamic filtering + attenuated masking + feedback optimization) to achieve reliable functional validation and regulation of neurons.

## Method

### Overall Architecture

Based on Llama-3.1-8B-Instruct, the research focuses on neurons in the FFN layers. The workflow consists of three stages: neuron recognition (activation frequency + probability normalization + entropy filtering) → adaptive masking validation (dynamic selection + attenuated masking + feedback optimization) → neuron modulation (directional induction + cross-signal enhancement).

### Key Designs

1.  **Neuron Recognition**:
    - **Function**: Localizes neurons with selective responses to specific emotion/rhetoric categories.
    - **Mechanism**: A three-step filtering process—first, calculating the activation frequency $f_{u,e}$ of each FFN neuron under various emotion/rhetoric inputs; second, performing activation probability normalization $P_{u,e} = f_{u,e}/T_e$; finally, using information entropy $H = -\sum P_{u,e}\log(P_{u,e})$ to measure distribution concentration, selecting the top 1% of neurons with the lowest entropy as target neurons.
    - **Design Motivation**: Low entropy implies that a neuron is highly selective for a specific category, allowing for reliable differentiation between different emotions/rhetorics.

2.  **Adaptive Masking**:
    - **Function**: Provides reliable causal validation and resolves the counter-intuitive phenomena of traditional masking.
    - **Mechanism**: (a) Calculate the activation difference for each neuron $D_i = |A_{i,target} - A_{i,non\text{-}target}| / A_{all}$, selecting the top 10% as core neurons; (b) apply attenuated masking $n'_u = n_u \times (1 - \alpha \times A_i/A_i^{max})$ to core neurons instead of forced zeroing; (c) introduce feedback optimization: if accuracy does not decrease after masking, automatically increase the attenuation coefficient $\alpha$ and the selection threshold $\tau$.
    - **Design Motivation**: Attenuated masking avoids triggering functional compensation mechanisms, while feedback optimization ensures stable and reliable masking effects.

3.  **Neuron Modulation and Cross-signal Enhancement**:
    - **Function**: Realizes directional induction of emotion/rhetoric predictions and rhetoric-assisted emotion recognition.
    - **Mechanism**: For non-target category inputs, prediction steering is achieved by injecting the functional vector of target neurons: $n'_{s_i} = n_{s_i} + \beta \times \bar{n}_{s_i}$. For cross-signal enhancement, the average activation of rhetoric neurons is injected into emotion neurons: $a_{i,joint} = a_i + \omega \cdot \bar{a}_{i,meta}$.
    - **Design Motivation**: To validate the causal functionality of neurons while exploring the intrinsic synergistic relationship between rhetoric and emotion.

### Loss & Training

LLM parameters remain frozen throughout; interventions are performed only at the level of FFN activation values. Feedback optimization (adjusting $\alpha$ and $\tau$) is conducted solely on the development set.

## Key Experimental Results

### Main Results

Accuracy changes across different masking methods ($\Delta$ACC %):

| Masking Method | Happiness | Sadness | Anger | Fear | Metaphor | Sarcasm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Zero (Forced Zeroing) | +5.46 | +4.32 | +7.92 | +1.22 | +3.37 | -4.17 |
| Mean (Mean Substitution) | +6.79 | -1.13 | +5.84 | +5.37 | +5.62 | -5.78 |
| **Adaptive** | **-9.25** | **-8.63** | **-10.14** | **-7.85** | **-7.29** | **-10.61** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| All-layer Masking | Maximum perf. drop | Emotion/rhetoric rely on cross-layer synergy |
| Top-layer Masking only | Second largest drop | Consistent with dense neuron distribution |
| Bottom-layer Masking only | Minimum drop | Fewer neurons in bottom layers |
| Rhetoric into Emotion | 1-5% improvement | Validates efficiency of cross-signal enhancement |

### Key Findings

- Traditional masking methods (Zero/Mean) generally lead to counter-intuitive accuracy increases in emotion tasks, proving that hard masking is unreliable.
- Adaptive masking produces stable accuracy decreases (-4% to -14%) across all 10 emotion/rhetoric categories, validating the reliability of the method.
- Emotion and rhetoric neurons are primarily concentrated in the upper layers of the model (layers 25-32), which is particularly evident in the 8B model.
- Hyperbole neurons have a positive enhancement effect on all emotion categories, consistent with the natural property of hyperbole to amplify emotional intensity.
- Sarcasm has the most significant promotion effect on the emotion of sadness, stemming from the expressive compatibility between the implicit criticism of sarcasm and the reserved nature of sadness.

## Highlights & Insights

- Adaptive masking solves a long-standing issue in interpretability research: the counter-intuitive phenomena of traditional masking. The combination of attenuated masking and feedback optimization is both simple and effective, and can be generalized to other neuron functional validation scenarios.
- Rhetoric-assisted emotion recognition is an intriguing discovery: rhetoric is not merely a linguistic "decoration"; it actively influences emotional representation within the model. This provides a new direction for LLM regulation via multi-signal synergy.
- The discovery of the hierarchical distribution of neurons (clustering in upper layers) aligns with existing research on knowledge storage, further confirming that the upper FFN layers are critical regions for semantic integration.

## Limitations & Future Work

- The scope of study is limited to 6 basic emotions and 4 types of rhetoric, without addressing more complex emotional states or additional rhetorical forms.
- Neuron modulation relies on static functional vectors and does not consider context-dependent dynamic adjustments.
- Results are only validated on Llama-3.1; whether similar distributions exist in other architectures (e.g., Qwen, Mistral) remains to be confirmed.
- The adjustment strategy for the attenuation coefficient $\alpha$ is relatively simple; more refined adaptive strategies might further enhance performance.

## Related Work & Insights

- **vs Lee et al. (2025)**: Validated the existence of emotion neurons in the Llama series but did not study rhetoric; this paper presents the first systematic study of rhetoric neurons and reveals the emotion-rhetoric synergy.
- **vs Di Palma et al. (2025)**: Achieved emotion recognition through probing techniques, which is a passive analysis; this paper implements active intervention and directional regulation.
- **vs Radford et al. (2018)**: First discovered the concept of emotion neurons; this paper conducts more systematic and reliable research on modern LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of rhetoric neurons and emotion-rhetoric synergy.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets + cross-dataset validation + multi-layer masking analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous experimental design.
- Value: ⭐⭐⭐⭐ Provides a reliable tool for neuron-level interpretability and regulation of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline.md)
- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)
- [\[ACL 2026\] EDUMATH: Generating Standards-aligned Educational Math Word Problems](edumath_generating_standards-aligned_educational_math_word_problems.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)

</div>

<!-- RELATED:END -->
