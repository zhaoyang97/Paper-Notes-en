---
title: >-
  [Paper Note] Concept Bottleneck Language Models For Protein Design
description: >-
  [ACL 2025][Computational Biology][Concept Bottleneck Models] This paper introduces the explainability design principles of Concept Bottleneck Models (CBMs) into protein language models. By utilizing biological concepts in the intermediate layer as a bottleneck, the proposed method achieves a protein generation system that can design functional protein sequences while simultaneously providing human-understandable design rationales.
tags:
  - "ACL 2025"
  - "Computational Biology"
  - "Concept Bottleneck Models"
  - "Protein Design"
  - "Language Models"
  - "Explainable AI"
  - "Protein Language Models"
date: 2026-05-08
content_hash: 1465133cbfed10a5
---

# Concept Bottleneck Language Models For Protein Design

**Conference**: ACL 2025  
**Code**: None  
**Area**: Computational Biology  
**Keywords**: Concept Bottleneck Models, Protein Design, Language Models, Explainable AI, Protein Language Models

## TL;DR
This paper introduces the explainability design principles of Concept Bottleneck Models (CBMs) into protein language models. By utilizing biological concepts in the intermediate layer as a bottleneck, the proposed method achieves a protein generation system that can design functional protein sequences while simultaneously providing human-understandable design rationales.

## Background & Motivation

**Background**: Protein design is a core task in bioengineering. In recent years, protein language models (such as the ESM series, ProtGPT2, etc.) have achieved breakthrough progress in protein sequence modeling and design by drawing inspiration from Transformer architectures in NLP. These models treat protein sequences as "amino acid languages" and model them using autoregressive or masked language models.

**Limitations of Prior Work**: (1) Existing protein generation models are black boxes—they generate protein sequences given target functions but fail to explain why a specific sequence was generated over others, making it difficult for biologists to trust and improve model outputs; (2) Computational protein design typically requires extensive wet-lab validation, and black-box models cannot guide experimentalists on how to make adjustments upon validation failure, resulting in high experimental costs; (3) The relationship between protein function and sequence is highly complex, and pure end-to-end learning may capture superficial correlations rather than causal mechanisms.

**Key Challenge**: While end-to-end deep learning models are becoming increasingly powerful in performance, their black-box nature severely limits their practical adoption in high-stakes application scenarios like protein design. Biologists need to understand the "why," not just the "what."

**Goal**: To design a protein language model that maintains design performance while providing explanations at the level of biological concepts.

**Key Insight**: The authors draw inspiration from Concept Bottleneck Models (CBMs) in computer vision, which insert a human-understandable concept layer between the input and output of a neural network. In protein design, these concepts correspond to known biological properties (such as secondary structure propensities, solvent accessibility, catalytic active sites, etc.).

**Core Idea**: To partition the protein generation process into two steps using a biological concept bottleneck: first, predicting the required combination of biological properties from the target function (concept prediction), and second, generating a protein sequence that satisfies these properties from the concept combination (conditional generation), making each step human-interpretable.

## Method

### Overall Architecture
The model consists of three modules: (1) a concept encoder that extracts required biological concept vectors from target functional descriptions; (2) a concept bottleneck layer that constrains the concept vectors to predefined, human-understandable biological attributes; (3) a conditional sequence generator that produces protein sequences based on the output of the concept bottleneck. The input is a natural language description of the target function (e.g., "design an enzyme stable at high temperatures"), and the output is an amino acid sequence.

### Key Designs

1. **生物学概念定义与量化**:

    - **Function**: Define specific concepts in the concept bottleneck layer of the protein language model.
    - **Mechanism**: Extract 50+ key protein property concepts from protein biology literature, including structural concepts ($\alpha$-helix propensity, $\beta$-sheet propensity, random coil ratio), physicochemical concepts (isoelectric point, thermal stability, solubility), and functional concepts (binding site types, catalytic mechanism types, substrate specificity). Each concept is quantified as continuous values or discrete categories. Train concept predictors using annotated data from UniProt and PDB databases, utilizing these concepts as supervision signals for the intermediate representations of the model.
    - **Design Motivation**: The selection of concepts must satisfy two conditions: (a) human interpretability, allowing biologists to make judgments based on concept values; (b) a strong correlation with protein function, ensuring that the bottleneck layer does not lose critical information.

2. **双向概念瓶颈架构**:

    - **Function**: Implement bidirectional mapping of "functional description $\rightarrow$ concept" and "concept $\rightarrow$ sequence" at the concept layer.
    - **Mechanism**: The concept encoder maps the textual description of the target function (e.g., "thermostable lipase") to a concept vector $c \in \mathbb{R}^{50}$, where each dimension corresponds to a predicted value of a biological concept. The concept bottleneck layer imposes three constraints on $c$: (a) concept hyperparameter constraints—certain concept values must lie within biologically plausible ranges; (b) concept consistency constraints—known biological relationships exist between some concepts (e.g., high thermal stability is often accompanied by high hydrophobic core packing density); (c) concept intervenability—allowing biologists to manually modify concept values to guide the design direction. The conditional generator receives the (potentially modified) concept vector and generates amino acid sequences via autoregressive decoding.
    - **Design Motivation**: Traditional CBMs are unidirectional (input $\rightarrow$ concept $\rightarrow$ label), but protein design requires generating sequences "backward" from concepts. The interactiveness/intervenability of the concepts in this bidirectional architecture is a key innovation—biologists can say, "keep other properties constant but increase thermal stability," and the model adjusts generation accordingly.

3. **概念对齐训练策略**:

    - **Function**: Ensure that the model's intermediate concept representations are aligned with real biological concepts.
    - **Mechanism**: Training is divided into two phases: (a) concept pre-training: train the concept predictor on large-scale protein databases to accurately predict concept values from protein sequences or functional descriptions, using MSE loss for continuous concept regression training and cross-entropy for discrete concept classification training; (b) joint fine-tuning: embed the concept predictor into the generative model as a bottleneck layer, simultaneously optimizing sequence generation quality (language model loss) and concept prediction accuracy (concept supervision loss), using a $\lambda$ weight to balance the two objectives.
    - **Design Motivation**: If the concept layer is inaccurate, the explanations will be untrustworthy, and errors will propagate to the generation stage. Concept pre-training combined with joint fine-tuning ensures the quality of concept representations.

### Loss & Training
The overall loss is formulated as $\mathcal{L} = \mathcal{L}_{LM} + \lambda \mathcal{L}_{concept}$, where $\mathcal{L}_{LM}$ is the standard autoregressive language model loss (sequence generation quality), and $\mathcal{L}_{concept}$ is the concept prediction loss (concept accuracy). $\lambda$ is tuned via the validation set, typically ranging between 0.1 and 0.5.

## Key Experimental Results

### Main Results

| Method | Sequence Recovery Rate | Functional Prediction Match Rate | Structural Quality (TM-score) | Concept Accuracy |
|------|-----------|-------------|-------------------|-----------|
| ProtGPT2 | 32.1% | 67.3% | 0.72 | N/A |
| ESM-IF | 38.5% | 71.8% | 0.78 | N/A |
| Ours (w/o concept intervention) | 36.8% | 70.2% | 0.76 | 81.3% |
| Ours (w/ concept intervention) | 34.2% | **74.5%** | **0.79** | 85.6% |
| Oracle (Ground-truth concept value input) | 42.1% | 78.3% | 0.83 | 100% |

### Ablation Study

| Configuration | Functional Match Rate | Concept Accuracy | Description |
|------|-------------|-----------|------|
| Full model | 74.5% | 85.6% | Full model + concept intervention |
| w/o Concept Bottleneck | 67.3% | N/A | Degrades to ProtGPT2 level |
| w/o Concept Consistency Constraint | 71.2% | 78.5% | Loss of relationship between concepts |
| w/o Concept Pre-training | 68.9% | 72.1% | Concept prediction is less accurate |
| Reduced Concept Dimension (25) | 72.8% | 83.4% | Information bottleneck is slightly tight but basically sufficient |

### Key Findings
- Concept intervention (manually correcting concept values) can significantly improve the functional match rate (+4.3%), demonstrating the value of human knowledge injection.
- Although the sequence recovery rate is slightly lower than that of ESM-IF, the functional match rate is higher, indicating that the model generates "different but functionally equivalent" sequences.
- Oracle experiments show that concept accuracy is the key bottleneck for performance upper bounds—improving concept prediction precision is the most promising direction for advancement.
- Reducing the concept dimension from 50 to 25 results in only a 1.7% drop in performance, suggesting redundancy among many concepts.

## Highlights & Insights
- Transferring CBM from classification tasks to generative tasks (protein sequence design) and achieving concept intervenability is an architecturally elegant extension.
- The design of concept intervention enables biologists to guide AI generation using their own domain knowledge, achieving true human-AI collaborative protein design.
- This architectural paradigm (concept bottleneck + conditional generation) can be transferred to other generative tasks requiring explainability (e.g., drug molecule design, materials design).

## Limitations & Future Work
- The definition of concepts relies on manual selection, which may miss attributes critical to certain functions.
- Biological constraint relationships between concepts are currently hand-coded; automatically learning relationships between concepts is worth exploring.
- The generated proteins have not yet undergone large-scale wet-lab validation.
- Model scale and training data volume are limited by the availability of labeled conceptual protein data.

## Related Work & Insights
- **vs Concept Bottleneck Models (CBM)**: While original CBMs are used for classification tasks, this work extends them to generative tasks and incorporates concept intervenability.
- **vs ProtGPT2/ESM**: These models are black-box protein generation models; this work adds interpretability while maintaining comparable performance.
- **vs Controllable Generation**: Controllable generation is typically controlled by continuous latent variables, whereas this work uses concepts with semantic meanings, offering stronger interpretability.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The cross-domain transfer of applying CBMs to protein design is highly ingenious.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid comparison with multiple baselines, and clean ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear presentation of cross-disciplinary content, friendly to both NLP and bioinformatics readers.
- **Value**: ⭐⭐⭐⭐⭐ Demonstrates significant exemplary value for the application of explainable AI in scientific discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Steering Protein Language Models](../../ICML2025/computational_biology/steering_protein_language_models.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](../../ICML2025/computational_biology/elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/computational_biology/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICLR 2026\] Controlling Repetition in Protein Language Models](../../ICLR2026/computational_biology/controlling_repetition_in_protein_language_models.md)
- [\[ICML 2025\] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](../../ICML2025/computational_biology/cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)

</div>

<!-- RELATED:END -->
