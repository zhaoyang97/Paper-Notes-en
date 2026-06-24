---
title: >-
  [Paper Note] Mechanistic Interpretability of Emotion Inference in Large Language Models
description: >-
  [ACL2025][Interpretability][Mechanistic interpretability] By utilizing three mechanistic interpretability techniques—probing, activation patching, and generation steering—this study reveals that the emotional representations of LLMs are functionally localized in the MHSA units of intermediate layers. Furthermore, based on cognitive appraisal theory, it demonstrates that these representations are psychologically plausible, successfully steering emotional output through interve…
tags:
  - "ACL2025"
  - "Interpretability"
  - "Mechanistic interpretability"
  - "emotion inference"
  - "activation patching"
  - "cognitive appraisal theory"
  - "generation steering"
date: 2026-05-08
content_hash: e860ac4551f9956f
---

# Mechanistic Interpretability of Emotion Inference in Large Language Models

**Conference**: ACL2025  
**arXiv**: [2502.05489](https://arxiv.org/abs/2502.05489)  
**Code**: GitHub (mentioned in the paper)  
**Area**: Interpretability  
**Keywords**: Mechanistic interpretability, emotion inference, activation patching, cognitive appraisal theory, generation steering  

## TL;DR

By utilizing three mechanistic interpretability techniques—probing, activation patching, and generation steering—this study reveals that the emotional representations of LLMs are functionally localized in the MHSA units of intermediate layers. Furthermore, based on cognitive appraisal theory, it demonstrates that these representations are psychologically plausible, successfully steering emotional output through interventions on appraisal concepts (such as self-agency and pleasantness).

## Background & Motivation

**Background**: LLMs perform exceptionally well in emotion recognition and inference tasks, even surpassing humans in certain scenarios. However, existing studies primarily treat LLMs as black boxes, testing their performance via zero-shot or in-context learning.

**Limitations of Prior Work**: There is a severe lack of understanding regarding how emotional information is represented and processed internally within LLMs. Existing research on mechanistic interpretability has focused on simplified synthetic structures (such as indirect object identification), which are difficult to generalize to natural text.

**Key Challenge**: The application of LLMs in high-stake emotional domains (such as mental health and legal decision-making) is increasing daily, yet their internal emotional processing mechanisms remain completely unknown. Consequently, we can neither verify their reliability nor control their emotional outputs.

**Goal**: To uncover the internal mechanism of emotional reasoning in LLMs—identifying in which layers and components emotional processing occurs, and whether the emotional output can be controlled by intervening in internal representations.

**Key Insight**: Drawing inspiration from functional localization methods in cognitive neuroscience (identifying brain regions, performing interventions, and demonstrating causality) and cognitive appraisal theory in psychology (which posits that emotions stem from the appraisal of situations), this study transfers both frameworks to the internal analysis of LLMs.

**Core Idea**: Emotional representations are functionally localized within LLMs, concentrated in the intermediate MHSA units. These representations encode psychologically plausible appraisal dimensions (such as pleasantness and agency), and causal emotional steering can be achieved by modulating appraisal concepts.

## Method

### Overall Architecture

A three-step progressive methodology is employed: **probing** (localization) $\rightarrow$ **activation patching** (causality verification) $\rightarrow$ **generation steering** (output control). This framework is systematically validated across 10 models from 5 model families, ranging from 1B to 13B parameters.

### Step 1: Linear Probing to Locate Emotion Signals

- An emotion classifier is trained on the MHSA output $\mathbf{a}^{(l)}$, FFN output $\mathbf{m}^{(l)}$, and hidden state $\mathbf{h}^{(l)}$ of each layer to predict 13 emotion categories.
- The crowd-enVENT dataset (consisting of 6,800 emotional essays with self-reported emotion labels and 23 appraisal variables) is utilized.
- Only samples where LLM predictions align with human annotations ($\ge 2,700$ samples) are analyzed to ensure the investigation of reliable mechanisms.
- **Findings**: Emotional signals strengthen significantly and stabilize in the intermediate layers (e.g., peaking at layer 10 out of 16 in Llama 3.2 1B), with no significant improvement in subsequent layers.

### Step 2: Activation Patching to Verify Causality

- The activation vectors of a "source sample" are patched into the computation graph of a "target sample" to test whether the emotional labels can be successfully transferred.
- Activation replacement is performed on the final token within a 5-layer window.
- **Findings**:
    - MHSA and FFN patching are highly localized in specific intermediate layers (e.g., layers 9-11 in Llama 3.2 1B).
    - The key patching location for the FFN occurs slightly later than that for the MHSA, suggesting that the MHSA aggregates emotional information before it is processed by the FFN.
    - Attention pattern visualization: Early layers focus on syntax $\rightarrow$ intermediate layers shift to emotion-related tokens $\rightarrow$ the final few layers primarily pass the representation of the final token.

### Step 3: Appraisal Concept Probing and Emotion Steering

**Appraisal Concept Probing**:
- Linear regression probes are trained for each appraisal dimension (such as pleasantness, other-agency, and predictability, totaling 23 variables).
- The appraisal signals are found to be strongly present in later layers, and the cosine similarity between appraisal and emotion dimensions exhibits psychologically plausible mappings in early-to-mid layers (e.g., anger $\leftrightarrow$ high other-agency + low pleasantness).

**Emotion Steering (Generation Steering)**:
- The "unique effect vector" $\mathbf{z}_a$ for each appraisal dimension is defined by projecting the appraisal vector onto the orthogonal complement of the subspace spanned by other appraisal vectors.
- Interventions are injected via $\mathbf{x} \leftarrow \mathbf{x} + \beta \frac{\mathbf{z}_a}{\|\mathbf{z}_a\|_2}$, where $\beta > 0$ facilitates and $\beta < 0$ suppresses the concept.
- The intervention is performed at layer 9 of Llama 3.2 1B.

### Loss & Training

This work does not involve model training; the probes are trained using standard cross-entropy (for classification) and MSE (for regression).

## Key Experimental Results

### Probe Accuracy

| Model | Layers | Emotion Signal Stabilization Layer | Peak Accuracy |
|:--|:--|:--|:--|
| Llama 3.2 1B | 16 | ~Layer 10 | High (see heatmap for details) |
| Llama 3.1 8B | 32 | ~Layers 16-18 | High |
| Gemma 2 2B | 26 | ~Layer 13 | High |
| OLMo 2 7B | 32 | ~Layer 16 | High |
| Phi 3.5 mini | 32 | ~Layer 16 | High |

- All 10 models (5 families $\times$ 2 scales) display the same trend: stabilization at intermediate layers.

### Activation Patching Success Rate
- MHSA patching exhibits extremely high success rates in specific intermediate layers (e.g., layers 9-11 of Llama 3.2 1B) and is close to 0 in other layers.
- Hidden state patching remains continuously effective from intermediate layers to the final layer due to the accumulation effect of the residual stream.

### Appraisal Concept Intervention (Generation Steering)

Intervention results at layer 9 of Llama 3.2 1B:
- **Facilitating pleasantness** ($\beta > 0$): The proportions of joy and pride increase significantly (aligning with theoretical expectations).
- **Suppressing pleasantness** ($\beta < 0$): The proportions of sadness, guilt, and anger increase.
- **Facilitating other-agency** ($\beta > 0$): The proportion of anger increases significantly.
- **Suppressing other-agency** ($\beta < 0$): The proportion of guilt increases.
- **Joint facilitation of pleasantness + other-agency**: The proportion of pride increases while joy ceases to occur (precisely matching theoretical predictions).

### Robustness Verification
- Variations in prompt formats, phrasing, and structure do not affect the conclusions.
- Control experiments: For syntactically similar but non-emotional tasks (e.g., Indirect Object Identification), units at the same layers are not critical, validating functional specificity.
- Analysis of the last 5 tokens confirms that the final token carries the strongest signal.

## Highlights & Insights

1. **Complete Causal Chain of Methodology**: Probing (correlation) $\rightarrow$ activation patching (necessity) $\rightarrow$ generation steering (sufficiency) forms a complete evidentiary chain of the underlying mechanisms.
2. **Integration of Psychological Theory**: Integrating cognitive appraisal theory into mechanistic interpretability analysis not only explains *where* the LLM processes emotions but also reveals *how*—by constructing an emotional space via dimensions like pleasantness and agency.
3. **Strong Cross-Model Consistency**: The 10 models span 5 families (Llama, Gemma, OLMo, Phi, Mistral), and the conclusion of intermediate layer localization is highly consistent.
4. **Precision of Joint Appraisal Interventions**: Simultaneously steering two appraisal dimensions can precisely guide the model toward specific emotions (e.g., pleasantness + other-agency $\rightarrow$ pride), proving the existence of a multi-dimensional emotional appraisal structure inside the LLM.

## Limitations & Future Work

1. The linear representation hypothesis might not hold completely—recent studies suggest that not all features are linearly encoded.
2. Causal directionality is not fully established: Appraisal concepts and emotions might be correlated rather than causal, or reverse causality may exist.
3. Computational constraints limited detailed appraisal interventions and robustness testing to Llama 3.2 1B.
4. The crowd-enVENT dataset contains monolingual English, leaving cross-lingual generalization unverified.
5. The definition of emotion itself remains controversial in psychology, and the granularity of 13 emotion classes might not be optimal.

## Related Work & Insights

- **Relationship with Meng et al. (2022)**: The latter utilized activation patching to locate where factual knowledge is stored; this study extends the same paradigm to the localization and intervention of emotional concepts.
- **Relationship with Templeton et al. (2024)**: Anthropic's feature clamping method is used to control model outputs; the appraisal concept intervention in this study provides a more psychologically grounded path for emotion control.
- **Bridge to Cognitive Neuroscience**: The attention patterns of LLMs correlate with eye-tracking data during human reading, and the functional localization in intermediate layers resembles functional specialization in the brain—this analogy offers a fresh perspective on understanding LLMs.
- **Insights**: (1) Appraisal theory-guided interventions can be leveraged for LLM safety alignment (e.g., suppressing hostile emotions); (2) Multi-dimensional appraisal spaces can be utilized to imbue LLMs with specific personas or emotional states; (3) The methodology can be extended to other psychological concepts (e.g., moral judgment, social cognition).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to combine cognitive appraisal theory with mechanistic interpretability, achieving localization and causal intervention of emotional representations on natural text.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive, involving 10 models, three mechanistic interpretability techniques, multiple robustness verifications, and control experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical formulations and intuitive figures, although some notations are dense.
- Value: ⭐⭐⭐⭐⭐ — Holds significant importance for research on LLM emotional safety and controllability; the methodology is widely transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ICLR 2026\] Medical Interpretability and Knowledge Maps of Large Language Models](../../ICLR2026/interpretability/medical_interpretability_and_knowledge_maps_of_large_language_models.md)
- [\[ICML 2025\] MIB: A Mechanistic Interpretability Benchmark](../../ICML2025/interpretability/mib_a_mechanistic_interpretability_benchmark.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](../../ACL2026/interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)

</div>

<!-- RELATED:END -->
