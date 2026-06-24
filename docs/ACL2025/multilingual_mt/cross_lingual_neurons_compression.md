---
title: >-
  [Paper Note] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual Language Models] By tracking checkpoints during the pre-training of multilingual language models, this paper discovers that models gradually compress language-specific representations into cross-lingual shared representations: language identification ability in the middle layers decreases, "expert neurons" for semantic concepts align cross-lingually, and manipulating concept neurons extracted from Spanish data unexpe…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Language Models"
  - "Cross-Lingual Generalization"
  - "Compression Hypothesis"
  - "Neuron Analysis"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: d67f50dea563cbf0
---

# Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons

**Conference**: ACL 2025  
**arXiv**: [2506.01629](https://arxiv.org/abs/2506.01629)  
**Code**: [https://github.com/Heidelberg-NLP/cross-lingual-generalization](https://github.com/Heidelberg-NLP/cross-lingual-generalization)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Language Models, Cross-Lingual Generalization, Compression Hypothesis, Neuron Analysis, Mechanistic Interpretability  

## TL;DR
By tracking checkpoints during the pre-training of multilingual language models, this paper discovers that models gradually compress language-specific representations into cross-lingual shared representations: language identification ability in the middle layers decreases, "expert neurons" for semantic concepts align cross-lingually, and manipulating concept neurons extracted from Spanish data unexpectedly causes the model to generate semantically related English text.

## Background & Motivation

**Background**: Multilingual language models (MLLMs) exhibit cross-lingual transfer capabilities without explicit cross-lingual supervision. Existing explanations include language relatedness, word order similarity, and shared subwords, but these explanations are supported by conflicting experimental evidence.

**Limitations of Prior Work**: (a) Most studies analyze only the final model state rather than the training process; (b) evaluations of cross-lingual transfer of training models often rely on zero-shot fine-tuning tasks, which are susceptible to dataset artifacts and may reflect surface-level patterns rather than genuine linguistic generalization; (c) the compression hypothesis has been mentioned in multilingual settings but has not been systematically validated.

**Key Challenge**: Why do models without explicit cross-lingual training signals develop cross-lingual shared representations?

**Goal**: To explain cross-lingual generalization from a compression perspective: the limited capacity of the model forces it to shift from maintaining independent, language-specific encodings to developing efficient, cross-lingual shared representations during training.

**Key Insight**: Formulating an analysis of pre-training checkpoints (BLOOM-560m/7b1 and a self-trained 257M model) to track the evolutionary process of internal representations instead of focusing solely on the final state.

**Core Idea**: Cross-lingual generalization is an inevitable outcome of compression under capacity constraints—the model first learns language-specific representations and then compresses them into cross-lingual shared neurons.

## Method

### Overall Architecture

A progressive three-level analysis: (1) Layer-wise language probing—tracking how the ability of each layer to identify languages changes during training; (2) Concept neuron tracking—identifying neurons that encode specific semantic concepts and analyzing their degree of cross-lingual alignment; (3) Generation manipulation experiments—verifying whether cross-lingually aligned neurons indeed encode language-independent semantic representations.

### Key Designs

1. **Language Identity Probing**:

    - **Function**: To measure the ability of each layer to identify the input language at different training stages.
    - **Mechanism**: Randomly sample token representations from hidden states of each layer, train a logistic regression classifier to predict the language, and track the test accuracy across training steps and layer depths.
    - **Key Findings**: At early training stages (step 1,000), all layers show high and uniform language identification performance; at later stages (step 400k), divergence occurs—language identification capability in the middle layers (layers 5-14) drops significantly, while those of the final layers recover to high levels. This suggests that the middle layers develop more language-independent representations.
    - **Design Motivation**: Direct analysis of internal model representations without fine-tuning, avoiding artifacts associated with zero-shot transfer evaluations.

2. **Concept Expert Neuron Analysis**:

    - **Function**: To identify neurons encoding specific semantic concepts (e.g., "earthquake", "house") and track their cross-lingual alignment.
    - **Mechanism**: Identify top-$K$ expert neurons for each concept following Suau et al. (2022). For each concept $c$ and language $l$, compute the conditional probability $p_i^{c,l}$ that neuron $i$ is "activated given concept $c$ and not activated otherwise", selecting the top 500 as the expert neuron set for that concept in that language. Subsequently, compute the overlap ratio of these expert neuron sets across different languages.
    - **Key Findings**: The proportion of cross-lingual shared expert neurons increases significantly during training, especially in the middle layers. These shared neurons encode language-independent semantic representations.

3. **Text Generation Manipulation Experiments**:

    - **Function**: To manipulate concept neurons identified from one language and observe which language the model generates text in when no language cues are provided.
    - **Mechanism**: Set the activations of the top-500 concept-specific expert neurons (identified from Spanish data) to their median activation value on concept samples, and then prompt the model to generate text using only the BOS token (across 100 random seeds).
    - **Key Findings**: In early training (step 10k), the model generates concept-related text in Spanish; in late training (step 400k), the model instead generates concept-related text in English. This demonstrates that these neurons encode language-independent semantic representations rather than language-specific ones.

### Loss & Training
The analyzed models include public checkpoints of BLOOM-560m and BLOOM-7b1, as well as a self-trained 257M parameter model (XGLM architecture, $d_{model}=512$, trained on 16 languages, with checkpoints saved at powers of 2 and every 5,000 steps). Probing is conducted using logistic regression.

## Key Experimental Results

### Main Results

Language Identity Probing (BLOOM-560m):

| Training Stage | Layer 1 Accuracy | Average Accuracy | Layer-wise Std Dev |
|---------|-----------|----------|----------|
| Step 1,000 | ~92% | ~94% | Low |
| Step 400,000 | ~57% | ~80% | High |

Proportion of cross-lingual shared neurons (middle layers, late vs. early training):
- Early stage: Overlap rate of expert neuron sets across different languages is low.
- Late stage: The proportion of cross-lingual shared neurons in the middle layers increases significantly.

### Ablation Study

Generated language distribution (language of the generated text when manipulating Spanish concept neurons):

| Training Steps | Spanish | English | Portuguese | Chinese |
|---------|---------|------|---------|-----|
| Step 10,000 | ~45% (Dominant) | ~25% | ~10% | ~8% |
| Step 400,000 | ~5% | **~60% (Dominant)** | ~5% | ~15% |

Concept neuron manipulation for low-resource languages (Swahili): The generated text never contains Swahili, completely shifting towards high-resource languages.

### Key Findings
- **Compression hypothesis validated**: The model undergoes a two-stage process from "memorization/fitting" to "compression/generalization", consistent with the theory of Shwartz-Ziv & Tishby (2017).
- **Middle layers are the primary site for cross-lingual generalization**: The layers with the most significant drop in language identification capability are also those with the highest proportion of shared neurons.
- **High-resource language bias**: Cross-lingual generalization tends to express concepts through English and Chinese, with concepts from low-resource languages (e.g., Swahili) being "assimilated" into the representations of high-resource languages.
- **Intra-family spillover effects**: Manipulating Spanish concept neurons leads to a small amount of Portuguese generation, indicating that linguistic relatedness shapes shared representations.

## Highlights & Insights
- **First to track the formation of cross-lingual semantic generalization during pre-training**: Instead of evaluating only the final model state, dense checkpoint analysis is leveraged to examine "when and how generalization occurs".
- **Highly convincing generation manipulation experiments**: Identifying "earthquake" neurons from Spanish data $\rightarrow$ manipulating them results in English text related to earthquakes $\rightarrow$ directly proving that these neurons encode language-independent semantics.
- **Transfer value**: This analysis methodology can guide advancements in multilingual models—if preserving the uniqueness of low-resource languages is desired, it may be necessary to explicitly prevent over-sharing in the middle layers during training.

## Limitations & Future Work
- Analyses are restricted up to BLOOM-7b1; larger-scale multilingual models (e.g., BLOOM-176B, Llama series) are not covered.
- The BLOOM series is currently the only MLLM providing public intermediate checkpoints, so its representativeness remains to be fully verified.
- The analysis only covers isolated semantic concepts, while relations between concepts (hierarchies, attribute sharing) or syntactic phenomena are not addressed.
- The discovery of a high-resource language bias highlights a critical fairness issue, but no concrete solution is proposed.

## Related Work & Insights
- **vs Wendler et al. (2024)**: They found that Llama-2 uses English as an internal hub to process other languages; this paper explains why from a compression perspective—shared representations lean towards being encoded in high-resource languages.
- **vs Tang et al. (2024) LAPE**: LAPE identifies language-specific neurons, whereas this work focuses on whether semantic concepts are shared cross-lingually, presenting complementary perspectives.
- **vs Blevins et al. (2022)**: They analyze the performance of XLM-R checkpoints on linguistic tasks, whereas this work delves deeper into mechanistic analysis at the neuron level.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to track the formation of cross-lingual generalization in pre-training dynamics; the systematic validation of the compression hypothesis is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three-level validation involving probing, neuron analysis, and generation manipulation, though the model size remains limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Very fluid narrative, with a clear logical chain from hypothesis to experiment and conclusion.
- **Value**: ⭐⭐⭐⭐⭐ Offers deep contributions to understanding the inner workings of multilingual models; the discovery of high-resource bias provides valuable inspiration for fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)

</div>

<!-- RELATED:END -->
