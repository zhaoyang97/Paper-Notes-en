---
title: >-
  [Paper Note] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening
description: >-
  [AAAI 2026][Medical Imaging][tuberculosis screening] This paper proposes DeepGB-TB, a multimodal TB screening system combining a lightweight 1D-CNN (for cough audio) and gradient-boosted decision trees (for demographic f…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "tuberculosis screening"
  - "cough audio"
  - "multimodal fusion"
  - "cross-attention"
  - "imbalanced loss"
date: 2026-05-08
content_hash: 6cf5770a88638f70
---

# DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening

**Conference**: AAAI 2026
**arXiv**: [2508.02741](https://arxiv.org/abs/2508.02741)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: tuberculosis screening, cough audio, multimodal fusion, cross-attention, imbalanced loss

## TL;DR
This paper proposes DeepGB-TB, a multimodal TB screening system combining a lightweight 1D-CNN (for cough audio) and gradient-boosted decision trees (for demographic features). A bidirectional cross-attention module (CM-BCA) fuses heterogeneous data by mimicking clinical reasoning, while a tuberculosis risk-balanced loss (TRBL) minimizes missed diagnoses. The system achieves AUROC 0.903 on a 7-country dataset and supports offline real-time inference on mobile devices.

## Background & Motivation

**Background**: Tuberculosis remains one of the leading causes of infectious disease mortality worldwide. Conventional diagnostic methods (sputum smear microscopy, NAATs) suffer from either low sensitivity or high cost requiring laboratory infrastructure, making them impractical in resource-limited settings. AI-driven TB screening is a promising direction, yet most existing approaches rely solely on audio or fail to effectively fuse heterogeneous data modalities.

**Limitations of Prior Work**:
- Many models use only audio, ignoring critical demographic and clinical risk factors.
- Simple concatenation or late fusion fails to capture complex nonlinear interactions between acoustic symptoms and patient context.
- Google HeAR achieves strong performance but is closed-source and requires online inference.
- Poor interpretability of deep learning models hinders clinical adoption.

**Key Challenge**: The cost of missing a true TB case (false negative) far exceeds that of a false alarm (false positive), yet standard loss functions treat both errors equally.

**Goal**: Design an interpretable, mobile-deployable, multimodal TB screening system tailored for low-resource settings.

**Key Insight**: Emulate clinical reasoning by integrating "who the patient is" (demographics) and "what the cough sounds like" (audio), leveraging cross-attention to allow each modality to guide the other toward the most diagnostically informative signals.

**Core Idea**: 1D-CNN for audio + CVPEM-enhanced LightGBM for tabular features + CM-BCA bidirectional cross-attention fusion + TRBL loss to penalize missed diagnoses.

## Method

### Overall Architecture
Two parallel data streams: an audio branch (1D-CNN extracting cough features) and a tabular branch (LightGBM generating cross-validated probability embeddings + fully connected layers) → CM-BCA bidirectional cross-attention fusion → classification.

### Key Designs

1. **Cross-Validation Probability Embedding Module (CVPEM)**:

    - Function: Transforms tabular data into robust high-dimensional features.
    - Mechanism: LightGBM is trained with 5-fold cross-validation, generating out-of-sample TB probabilities per patient as additional feature embeddings: $\tilde{x}_{tab,i} = [x_{tab,i}, p_{gbm,i}]$.
    - Design Motivation: Reduces overfitting and improves generalization while exploiting the natural advantage of GBDTs on tabular data.

2. **Cross-Modal Bidirectional Cross-Attention (CM-BCA)**:

    - Function: Enables audio and tabular features to mutually query each other, iteratively refining their representations.
    - Mechanism: $\mathcal{T}_{t \leftarrow a}$ applies multi-head attention using audio features as Key/Value to refine tabular features, followed by FFN and LayerNorm; $\mathcal{T}_{a \leftarrow t}$ symmetrically refines audio features. The process iterates until convergence.
    - Design Motivation: Simulates clinical reasoning — when high-risk patient profiles (age, exposure history) are detected, the model directs greater attention to abnormal patterns in the audio.

3. **Tuberculosis Risk-Balanced Loss (TRBL)**:

    - Function: Imposes stronger penalties on false negatives.
    - Mechanism: $\mathcal{L}_{TRBL} = \mathcal{L}_{BCE} \cdot (1-y+\lambda y)$, where $\lambda > 1$. The loss for positive samples (TB-positive) is amplified by a factor of $\lambda$.
    - Design Motivation: In TB screening, the cost of a missed diagnosis far exceeds that of a false alarm — missed diagnoses lead to disease transmission and mortality, whereas false alarms necessitate only one additional confirmatory test.

### Loss & Training
TRBL loss with a standard training pipeline. The dataset comprises 1,105 patients with coughs lasting more than two weeks from 7 countries. Evaluation is conducted via 5-fold cross-validation.

## Key Experimental Results

### Main Results

| Model | Parameters | Accuracy | F1 | AUROC | Inference Time (s) |
|-------|------------|----------|----|-------|--------------------|
| LightGBM | 1.2M | 0.778 | 0.783 | 0.834 | 45.2 |
| 1D-CNN | 2.1M | 0.755 | 0.783 | 0.809 | 21.5 |
| CNN-LightGBM | 2.8M | 0.788 | 0.768 | 0.792 | 31.2 |
| Qwen-Omni 3B | 3B | 0.812 | 0.845 | 0.900 | 4531 |
| **DeepGB-TB** | **5.2M** | **0.817** | **0.851** | **0.903** | **44.6** |

DeepGB-TB surpasses Qwen-Omni (3B parameters) while being 100× faster at inference.

### Ablation Study

| Configuration | AUROC |
|---------------|-------|
| Full (DeepGB-TB) | 0.903 |
| w/o Audio | 0.840 |
| w/o Tabular | — |
| w/o CM-BCA (simple concat) | ~0.792 |

### Key Findings
- Both audio and tabular features are indispensable; tabular features (with CVPEM) contribute more substantially.
- CM-BCA significantly outperforms simple concatenation and late fusion.
- 5.2M parameters suffice to surpass a 3B-parameter large model, enabling mobile deployment.
- Validation across 7 countries demonstrates cross-population generalizability.

## Highlights & Insights
- The **clinical reasoning simulation paradigm** is compelling — the design intuition behind CM-BCA directly mirrors how clinicians integrate symptoms with risk factors.
- **TRBL loss aligns with clinical deployment goals** — rather than a generic imbalanced loss, it is explicitly tailored to the clinical reality that missed diagnoses far outweigh false alarms in TB screening.
- The **5.2M vs. 3B parameter efficiency comparison** makes a strong argument for the trade-off between task-specific models and general-purpose large models on specialized tasks.

## Limitations & Future Work
- The dataset comprises only 1,105 patients, which is relatively small.
- The system addresses only binary classification (TB-positive/negative) without distinguishing disease severity.
- Interpretability is mentioned but lacks sufficient methodological detail.
- Only cough audio is considered; other acoustic biomarkers such as breath sounds are not explored.

## Related Work & Insights
- **vs. HeAR**: Google HeAR is the state-of-the-art audio foundation model but is closed-source and requires online inference; DeepGB-TB is open and supports offline use.
- **vs. Qwen-Omni**: Comparable performance with 600× fewer parameters and 100× faster inference.
- **vs. CNN-LightGBM Ensemble**: Simple late fusion is substantially inferior to the deep interactive fusion achieved by CM-BCA.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined CM-BCA + TRBL design is clinically motivated and well-conceived.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7-country dataset + multiple baselines + ablations, though dataset size is limited.
- Writing Quality: ⭐⭐⭐⭐ Clinical motivation is clearly articulated; method description is detailed.
- Value: ⭐⭐⭐⭐⭐ Practically significant AI application for global TB control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[AAAI 2026\] CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution](cd-dpe_dual-prompt_expert_network_based_on_convolutional_dictionary_feature_deco.md)
- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](../../ICML2026/medical_imaging/evidential_reasoning_advances_interpretable_real-world_disease_screening.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](../../ICLR2026/medical_imaging/towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)

</div>

<!-- RELATED:END -->
