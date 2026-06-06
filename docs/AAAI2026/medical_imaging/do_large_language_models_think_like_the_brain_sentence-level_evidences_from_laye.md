---
title: >-
  [Paper Note] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI
description: >-
  [AAAI 2026][Medical Imaging][LLM-brain alignment] This paper systematically investigates sentence-level alignment between 14 open-source LLMs and human brain language processing by comparing layer-wise LLM representation…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "LLM-brain alignment"
  - "fMRI"
  - "sentence-level semantic understanding"
  - "layer-wise representations"
  - "hemispheric lateralization"
date: 2026-05-08
content_hash: a0c10175816360de
---

# Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI

**Conference**: AAAI 2026
**arXiv**: [2505.22563](https://arxiv.org/abs/2505.22563)  
**Code**: [https://github.com/Lucasuuu02/LLM4Brain](https://github.com/Lucasuuu02/LLM4Brain)  
**Area**: Cognitive Neuroscience / NLP
**Keywords**: LLM-brain alignment, fMRI, sentence-level semantic understanding, layer-wise representations, hemispheric lateralization

## TL;DR
This paper systematically investigates sentence-level alignment between 14 open-source LLMs and human brain language processing by comparing layer-wise LLM representations with fMRI data recorded while participants listened to a natural narrative. Key findings include: middle layers yield the highest brain alignment, instruction tuning substantially enhances alignment, and hemispheric lateralization patterns consistent with classical neurolinguistic theories are observed.

## Background & Motivation

The intersection of AI and neuroscience has long been concerned with whether representations learned by LLMs correspond to human brain language processing. Prior work has demonstrated correlations between LLM representations and neural responses, yet these observations lack mechanistic interpretation: **is such similarity merely a consequence of increasing model scale, or does it reflect a deeper convergence with the computational principles underlying human language processing?**

Limitations of existing research include: (1) reliance on public benchmarks that may not accurately reflect a model's comprehension in specific tasks; (2) overemphasis on model scale at the expense of semantic understanding ability itself; (3) a predominance of word-level analyses, with insufficient systematic investigation at the sentence level.

The paper's starting point is to expose both LLMs and human participants to identical natural narrative stimuli (*The Little Prince*), design a cross-lingual semantic alignment assessment (CSAA) to evaluate LLM comprehension, and construct sentence-level neural encoding models to quantify the correspondence between layer-wise representations and regional brain activation. The core insight is that **semantic understanding ability—rather than parameter scale per se—is the key driver of LLM–brain alignment**.

## Method

### Overall Architecture
A multi-stage pipeline: (1) fMRI data acquisition and preprocessing → GLM-based estimation of sentence-level neural responses → ROI extraction; (2) extraction of layer-wise sentence embeddings from each LLM; (3) ridge regression modeling and correlation analysis. For each of the 14 LLMs, embeddings from every layer are regressed against fMRI activations in 12 language-related brain regions under cross-validation.

### Key Designs
1. **Cross-lingual Semantic Alignment Assessment (CSAA)**:

    - Function: Evaluates an LLM's ability to understand continuous text
    - Mechanism: For each Chinese sentence, five English options are generated (correct translation, shuffled words, part-of-speech substitution, syntactic paraphrase, and information addition/deletion); the model selects the correct translation via cosine similarity of embeddings
    - Design Motivation: Captures contextual semantic understanding more faithfully than standard benchmarks and allows quantitative comparison across models

2. **GLM + LS-S Sentence-Level Neural Activation Estimation**:

    - Function: Precisely extracts the BOLD signal corresponding to each sentence from the fMRI time series
    - Mechanism: The Least-Squares Separate (LS-S) method models each sentence as an independent regressor, addressing the temporal mismatch between sentence boundaries and TR sampling through precise onset times and HRF convolution
    - Design Motivation: Compared with conventional condition averaging or block designs, LS-S more accurately isolates the transient neural response of each sentence, making it well-suited for naturalistic narrative paradigms

3. **Layer-wise Ridge Regression and Correlation Analysis**:

    - Function: Quantifies the predictive capacity of each LLM layer's representations for fMRI signals in each ROI
    - Mechanism: For each ROI, sentence embeddings from each layer are used in a ridge regression to predict fMRI signals; Pearson correlations are computed under K-fold cross-validation. All subject × ROI × layer combinations are processed in parallel
    - Design Motivation: Ridge regularization mitigates overfitting; nested cross-validation selects the optimal $\alpha$; z-score normalization ensures compatibility across modalities

### Loss & Training
No model training is performed. The study uses 14 existing pretrained LLMs (including Llama-3.1, Gemma-2, Baichuan2, DeepSeek, GLM-4, and Qwen2.5 series, covering both base and instruct variants) with parameter counts ranging from 6.7B to 9B. The analytical framework employs standard ridge regression; the key hyperparameter $\alpha$ is selected via grid search and nested cross-validation.

## Key Experimental Results

### Main Results (CSAA Semantic Understanding)

| Model | CSAA Score |
|-------|-----------|
| Llama-3.1-8B-Instruct | 31.4 |
| Gemma-2-9b-it | 30.7 |
| Gemma-2-9b | 30.3 |
| Baichuan2-7B-Chat | 22.7 |
| DeepSeek-7B-Chat | 19.5 |
| glm-4-9b | 7.9 |
| Qwen2.5-7B | 6.4 |

Instruct variants consistently outperform their corresponding base models, with inter-model gaps exceeding 28 points.

### Brain–Model Alignment Analysis

| Dimension | Key Finding | Statistical Significance |
|-----------|-------------|--------------------------|
| Middle vs. final layers | Middle layers yield the best prediction of brain activity across all models | Consistent across models |
| Instruct vs. Base | Instruct variants improve brain alignment | Permutation test $p = 0.03125$ |
| Understanding ability vs. alignment | Pearson $r = 0.601$, positive correlation | $p = 0.030$ |
| Left IFG lateralization | Significant left-hemisphere advantage | $p = 0.025$ |
| Right AntTemp lateralization | Significant right-hemisphere advantage | $p = 0.001$ |

### Key Findings
- **Middle layers yield optimal alignment**: Across all 14 LLMs, middle layers (rather than final layers) show the highest correlation with brain activity, consistent with prior EEG/MEG studies but validated here for the first time using fMRI.
- **Instruction tuning enhances alignment**: Comparisons across five base/instruct pairs show that instruction tuning not only improves task performance but also brings model representations closer to human brain activity patterns.
- **Semantic understanding is the key driver**: Within the 6.7B–9B parameter range, semantic understanding ability (CSAA score) predicts brain alignment better than model scale alone.
- **Hemispheric lateralization**: Left IFG and posterior temporal regions (core language areas) exhibit left-hemisphere dominance, while right MFG and anterior temporal regions exhibit right-hemisphere dominance, consistent with classical language lateralization hypotheses. The degree of lateralization in IFG and MFG is also positively correlated with model performance.

## Highlights & Insights
- This work is the first to propose that semantic understanding ability (CSAA) drives brain alignment, shifting the explanatory focus from scale to capability.
- It raises an intriguing possibility: should future LLM training pipelines explicitly align with brain data to enhance cognitive plausibility?
- The hemispheric lateralization analysis bridges classical macroscopic cognitive neuroscience theories with the computational properties of LLMs.
- The whole-brain spatial coverage afforded by fMRI addresses the limited spatial resolution of prior EEG/MEG studies.

## Limitations & Future Work
- Experiments are conducted on a single naturalistic narrative corpus (*The Little Prince*); generalizability is limited, and different text genres may yield different conclusions.
- The sample of 34 participants limits statistical power; larger samples are needed to validate findings, and individual differences are not fully analyzed.
- All LLMs fall within a narrow parameter range (6.7B–9B), precluding conclusions about whether scaling laws hold for much larger or smaller models.
- Correlation does not imply causation; LLM–brain alignment does not necessarily indicate shared computational mechanisms and may partly reflect statistical coincidence.
- The cross-lingual task design (Chinese → English) may introduce translation-related confounds; monolingual testing would be cleaner.
- The low temporal resolution of fMRI (TR = 2 s) precludes capturing rapid language processing dynamics.
- Closed-source models (e.g., GPT-4, Claude) are not evaluated.

## Related Work & Insights
- This work extends the tradition established by Schrimpf et al. and Caucheteux et al. on LLM–brain alignment from the word level to the sentence level, providing higher-order semantic correspondence evidence.
- The methodological contribution lies in jointly analyzing CSAA ability scores and brain alignment across multiple LLMs, revealing a positive ability–alignment relationship that challenges the simplistic "bigger is better" narrative.
- The findings carry implications for cognitive architecture design: if better semantic understanding genuinely leads to more brain-like representations, using brain data to guide model training may be a viable research direction.
- The hemispheric lateralization results suggest that brain-inspired LLMs may need to incorporate distributed and lateralized processing patterns analogous to those observed in the human brain.
- This work complements the encoding model approach of Tuckute et al. (2024): whereas that work focuses on optimizing a specific model, the present paper emphasizes cross-model comparison.

## Rating
- Novelty: ⭐⭐⭐⭐ (Sentence-level LLM–brain alignment combined with the CSAA ability metric is novel, though the overall methodological framework is relatively conventional)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (14 LLMs, 12 ROIs, three sets of experiments, comprehensive statistical testing)
- Writing Quality: ⭐⭐⭐⭐ (Detailed method descriptions, clear formulations, highly informative figures and tables)
- Value: ⭐⭐⭐⭐ (Provides new evidence and perspectives for understanding the relationship between LLMs and the human brain)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language Reconstruction with Brain Predictive Coding from fMRI Data](../../ACL2026/medical_imaging/language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [\[AAAI 2026\] Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)
- [\[AAAI 2026\] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)

</div>

<!-- RELATED:END -->
