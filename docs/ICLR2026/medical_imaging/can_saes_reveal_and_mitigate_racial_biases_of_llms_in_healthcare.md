---
title: >-
  [Paper Note] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?
description: >-
  [ICLR 2026][Medical Imaging][Sparse Autoencoders] This paper investigates whether Sparse Autoencoders (SAEs) can reveal and mitigate racial biases in LLMs within clinical settings. SAEs successfully identify harmful race…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Sparse Autoencoders"
  - "Racial Bias"
  - "Medical AI"
  - "Interpretability"
  - "Causal Intervention"
date: 2026-05-08
content_hash: 63c9bff412e3a514
---

# Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?

**Conference**: ICLR 2026  
**arXiv**: [2511.00177](https://arxiv.org/abs/2511.00177)  
**Code**: [https://github.com/hibaahsan/sae_bias/](https://github.com/hibaahsan/sae_bias/)  
**Area**: Medical Imaging / AI Safety / LLM Alignment  
**Keywords**: Sparse Autoencoders, Racial Bias, Medical AI, Interpretability, Causal Intervention

## TL;DR
This paper investigates whether Sparse Autoencoders (SAEs) can reveal and mitigate racial biases in LLMs within clinical settings. SAEs successfully identify harmful race-associated features (e.g., co-activation of "Black" with violence-related terms), but their effectiveness at bias mitigation in complex clinical tasks is limited (FLDD < 3%), falling far short of simple prompting strategies (FLDD 8–15%).

## Background & Motivation

**Background**: LLMs are increasingly deployed in healthcare settings (e.g., clinical note analysis, case generation), yet are known to exhibit racial biases—potentially producing systematically different risk assessments for Black patients.

**Limitations of Prior Work**: Existing bias detection relies on external benchmarks and cannot explain the internal mechanisms of bias. How do models internally represent racial information? Chain-of-thought (CoT) explanations are unfaithful—models claim not to use race, yet demonstrably do.

**Key Challenge**: SAEs offer fine-grained tools for analyzing LLM internal representations, but whether "detection" can be translated into "mitigation" remains unclear.

**Goal**: (a) Can SAEs identify race-related features within LLMs? (b) Can ablating these features effectively reduce bias?

**Key Insight**: Train racial probes using L1-regularized logistic regression on SAE activations to identify latent features predictive of race, then perform causal validation via steering and ablation.

**Core Idea**: SAEs can reveal the mechanism of bias (latent features associated with "Black" co-activate with stigmatizing terms such as incarceration, cocaine, and gunshot wounds), but ablating these features is insufficient to mitigate bias in complex clinical tasks.

## Method

### Overall Architecture
A three-step pipeline: (1) train racial classification probes on SAE activations from clinical discharge notes; (2) analyze the semantic content of the most predictive latent features; (3) causally validate these features and test mitigation via steering (activation amplification) and ablation (activation removal).

### Key Designs

1. **Racial Probe Training**:

    - Function: Identify latent SAE features predictive of patient race.
    - Mechanism: Apply max-aggregation over tokens on SAE activation vectors, then train an L1-regularized logistic regression. High-weight features are identified as race-related.
    - Design Motivation: L1 regularization automatically selects the minimal set of most relevant features, facilitating manual inspection.

2. **Causal Validation (Steering)**:

    - Function: Observe changes in model output by amplifying race-related latent feature activations.
    - Mechanism: Modify the hidden state at layer $l$ as $z'_i = z_i + \alpha \cdot z_{\max}$ (applied only to race-related features), sweeping $\alpha$ from 0.01 to 5.
    - Design Motivation: If steering Black-associated features raises violent risk assessment scores, it establishes a causal link between the feature and the bias.

3. **Bias Mitigation (Ablation)**:

    - Function: Suppress race-related feature activations and measure the resulting reduction in bias.
    - Mechanism: $\text{FLDD} = 1 - \text{logitdiff}_{\text{ablated}} / \text{logitdiff}_{\text{clean}}$. Higher FLDD indicates more effective ablation.

## Key Experimental Results

### Bias Findings
- Latent features associated with "Black" co-activate with terms including: incarceration, cocaine, and gunshot wounds.
- Steering Black-associated features increases violent risk assessment scores by 0.51–0.80 (causal validation).

### Mitigation Effectiveness Comparison

| Clinical Task | SAE Ablation FLDD | Prompting Strategy FLDD |
|---|---|---|
| Cocaine Diagnosis | 0.8% | **15.2%** |
| Gestational Hypertension | 1.1% | **12.8%** |
| Pain Assessment | 0.01% | **8.1%** |
| Uterine Fibroids | 2.9% | 3.2% |

### Key Findings
- SAEs successfully identify the mechanism of racial bias (co-activation with stigmatizing terms).
- CoT explanations are unfaithful—models claim not to use race, yet SAE analysis demonstrates that race is encoded internally.
- SAE ablation performs poorly on complex clinical tasks (FLDD < 3%), far underperforming simple prompting strategies.
- Racial information may be distributed across too many features for single-feature ablation to meaningfully affect model outputs.
- In clinical note generation tasks, SAE ablation reduces the proportion of Black patient cases by approximately 30% (effective, but potentially overcorrecting).

## Highlights & Insights
- **Honest Negative Results**: Transparently reporting the limited effectiveness of SAE-based mitigation is more valuable than claiming success. This work reveals a significant gap between mechanistic interpretability and practical bias mitigation.
- **Evidence of CoT Unfaithfulness**: SAE analysis provides quantitative evidence that models "say one thing and do another"—claiming not to use racial information while clearly encoding it internally.
- **Discovery of Stigmatizing Associations**: The precise localization of harmful associations such as Black–violence and Black–cocaine has direct implications for understanding and auditing medical AI systems.

## Limitations & Future Work
- The failure of SAE-based mitigation may stem from racial information being too distributed across features, necessitating more fine-grained intervention strategies.
- Validation is limited to the Gemma-2 model family; results may differ across other architectures.
- Limited annotated data for clinical tasks may reduce the statistical power of bias assessments.

## Related Work & Insights
- **vs. Prompt-Based Debiasing**: Simple prompts (e.g., "do not consider race") prove more effective, suggesting that surface-level interventions can sometimes outperform deeper mechanistic ones.
- **vs. Traditional Fairness Auditing**: SAEs enable bias analysis at the level of internal mechanisms, going beyond output-level metrics to provide deeper interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of SAEs for bias analysis in healthcare; the negative results are themselves highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full pipeline covering detection, steering, and ablation across multiple clinical tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Honest reporting of negative results with thorough analysis.
- Value: ⭐⭐⭐⭐ Significant reference value for research on fairness in medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](../../ACL2026/medical_imaging/hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](../../ACL2026/medical_imaging/efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)

</div>

<!-- RELATED:END -->
