---
title: >-
  [Paper Note] OneRestore: A Universal Restoration Framework for Composite Degradation
description: >-
  [ECCV 2024][Information Retrieval & RAG][image restoration] OneRestore is proposed as a Transformer-based universal image restoration framework. Driven by a scene-descriptor-guided cross-attention mechanism and a composite degradation restoration loss, it adaptively handles low-light, haze, rain, snow, and their arbitrary composite combinations within a single model, supporting controllable restoration under both text and visual modes.
tags:
  - "ECCV 2024"
  - "Information Retrieval & RAG"
  - "image restoration"
  - "composite degradation"
  - "scene descriptor"
  - "controllable restoration"
  - "contrastive loss"
date: 2026-05-08
content_hash: 4b121ab5586f781b
---

# OneRestore: A Universal Restoration Framework for Composite Degradation

**Conference**: ECCV 2024  
**arXiv**: [2407.04621](https://arxiv.org/abs/2407.04621)  
**Code**: [GitHub](https://github.com/gy65896/OneRestore)  
**Area**: Information Retrieval  
**Keywords**: image restoration, composite degradation, scene descriptor, controllable restoration, contrastive loss

## TL;DR
OneRestore is proposed as a Transformer-based universal image restoration framework. Driven by a scene-descriptor-guided cross-attention mechanism and a composite degradation restoration loss, it adaptively handles low-light, haze, rain, snow, and their arbitrary composite combinations within a single model, supporting controllable restoration under both text and visual modes.

## Background & Motivation
**Background**: Image restoration research has achieved significant progress in single-degradation scenarios (dehazing, deraining, low-light enhancement, etc.). However, these approaches are mostly One-to-One models that can only handle a specific type of degradation.

**Limitations of Prior Work**:
   - In real-world scenarios, multiple degradation factors often co-occur (e.g., raining in a foggy night), which One-to-One models cannot handle;
   - One-to-Many partial parameter-sharing methods (e.g., All-weather Net) require independent encoders for each degradation type, causing the model scale to grow linearly with the number of degradation types;
   - One-to-Many full parameter-sharing methods (e.g., AirNet, TransWeather) stage mixed training directly without perceiving specific degradation types, which might introduce noise while performing dehazing.

**Key Challenge**: How to enable a single model to both identify the specific components of composite degradation and perform controllable restoration according to user intent?

**Goal**: Constructing a unified composite degradation imaging model and a scene-descriptor-guided controllable restoration framework.

**Key Insight**: Inspired by the human annotation process—annotators must first understand the degradation type to evaluate quality—a model should also "understand" the degradation scene before performing restoration.

**Core Idea**: Utilizing scene description embeddings as degradation "switches" to guide the Transformer via cross-attention for precise restoration of targeted degradation factors.

## Method

### Overall Architecture
Degraded image $I(x)$ + scene descriptor $e_t$ $\rightarrow$ Encoder (3 downsamplings, each layer containing SDTB) $\rightarrow$ Decoder (3 upsamplings + skip connection + global residual) $\rightarrow$ Restored image $\hat{J}$

Imaging model: $I(x) = \mathcal{P}_h(\mathcal{P}_{rs}(\mathcal{P}_l(J(x))))$, i.e., a clean image first undergoes cascaded degradation of low-light $\rightarrow$ rain/snow $\rightarrow$ haze.

### Key Designs

1. **Composite Degradation Formulation**:

    - **Function**: Unifies the modeling of the cascaded process of 4 physical degradation types (low-light, rain, snow, haze)
    - **Mechanism**:
        - Low-light: Based on Retinex theory, $I_l(x) = \frac{J(x)}{L(x)} L(x)^\gamma + \varepsilon$, where $\gamma \in [2,3]$
        - Rain: $I_{rs}(x) = I_l(x) + \mathcal{R}$
        - Snow: $I_{rs}(x) = I_l(x)(1-\mathcal{S}) + M(x)\mathcal{S}$
        - Haze: $I(x) = I_{rs}(x) \cdot t + A(1-t)$, where $t = e^{-\beta d(x)}$
    - **Design Motivation**: Degradations are superposed in real-world scenarios, which necessitates cascade modeling based on physical laws. Based on this, the CDD-11 dataset (11 degradation categories + clean images) was constructed, where 13,013 training pairs and 2,200 testing pairs were generated from 1,383 high-resolution images.

2. **Scene-Descriptor-Guided Transformer Block (SDTB)**:

    - **Function**: Incorporates scene degradation information into each Transformer block to guide the direction of feature extraction
    - **Mechanism**: Contains three sub-modules: SDCA, SA, and FFN. SDCA uses the scene description embedding to generate the query, and image features to generate the key/value:
    $$\text{SDCA}(\mathbf{Q}_t, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}_t \cdot \mathbf{K}^\top}{\lambda}\right) \mathbf{V}$$
    - **Design Motivation**: Traditional self-attention only interacts within image features, failing to leverage the degradation type prior. Generating queries from scene descriptions is equivalent to "telling" the model which degradation to focus on, achieving a shift from passive detection to active guidance.

3. **Scene Descriptor Generation**:

    - **Function**: Provides two modes to generate scene description embeddings—manual text input vs. automatic visual attribute extraction
    - **Mechanism**:
        - Text Embedder: 5 basic scene texts undergo GloVe $\rightarrow$ 12 text embeddings (including 7 composite degradations, generated by averaging corresponding single-degradation embeddings) $\rightarrow$ MLP refinement
        - Visual Embedder: ResNet-18 extracts visual features $\rightarrow$ conv + dropout + linear $\rightarrow$ visual embeddings $\rightarrow$ cosine similarity matches the most similar text embedding
        - Cosine cross-entropy loss for training: $$S(e_v, e_t) = \frac{e^{\cos(e_v, e_t)}}{\sum_{t_i=1}^{N_t} e^{\cos(e_v, e_{t_i})}}$$
    - **Design Motivation**: Text embeddings provide precise and controllable scene descriptions (higher precision), while visual embeddings provide automation capability (97.55% accuracy). The two are complementary.

4. **Composite Degradation Restoration Loss (CDRL)**:

    - **Function**: Introduces multiple degradation negative samples on top of traditional contrastive loss to enhance the discriminative capacity of the model
    - **Mechanism**:
    $$\mathcal{L}_c = \sum_{k=1}^{K} \xi_k \frac{\mathcal{L}_1(V_k(J), V_k(\hat{J}))}{\xi_c \mathcal{L}_1(V_k(\hat{J}), V_k(I)) + \sum_{o=1}^{O} \xi_o \mathcal{L}_1(V_k(I_o), V_k(\hat{J}))}$$
      Utilizes the features of layers 3, 8, and 15 of VGG-16, with $O=10$ other degradation negative samples
    - **Design Motivation**: Traditional contrastive loss only uses the input degraded image as a negative sample, which may push the restored results toward other degradation forms. CDRL forces the output to stay away from all degradation types simultaneously.

### Loss & Training
- Total loss: $\mathcal{L} = \alpha_1 \mathcal{L}_1^s + \alpha_2 \mathcal{L}_M + \alpha_3 \mathcal{L}_c$ (smooth $l_1$ + MS-SSIM + CDRL)
- Trained in two stages: first train the text/visual embedder (200 epochs, lr=0.0001), then train OneRestore (120 epochs, lr=0.0002)
- Training images are cropped into $256 \times 256$ patches with a stride of 200, randomly flipped, generating 312k training pairs
- 8 NVIDIA L40 GPUs

## Key Experimental Results

### Main Results (CDD-11 Dataset)

| Method | Type | PSNR↑ | SSIM↑ | Params |
|------|------|-------|-------|--------|
| Restormer | One-to-One | 26.99 | 0.8646 | 26.13M |
| SRUDC | One-to-One | 27.64 | 0.8600 | 6.80M |
| WGWSNet | One-to-Many | 26.96 | 0.8626 | 25.76M |
| PromptIR | One-to-Many | 25.90 | 0.8499 | 38.45M |
| **OneRestore** | One-to-Composite | **28.47** | **0.8784** | **5.98M** |
| **OneRestore†** | One-to-Composite | **28.72** | **0.8821** | 5.98M |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Controllability |
|------|-------|-------|--------|
| FFN only | 24.81 | 0.8607 | ✗ |
| SA + FFN | 27.19 | 0.8697 | ✗ |
| SDCA + FFN | 27.93 | 0.8767 | ✓ |
| **SDCA + SA + FFN** | **28.72** | **0.8821** | **✓** |

| Loss Combination | PSNR↑ | SSIM↑ |
|---------|-------|-------|
| Smooth $l_1$ only | 28.16 | 0.8633 |
| Smooth $l_1$ + MS-SSIM | 27.54 | 0.8708 |
| Smooth $l_1$ + MS-SSIM + CL | 27.61 | 0.8723 |
| **Smooth $l_1$ + MS-SSIM + CDRL** | **28.72** | **0.8821** |

### Key Findings
- Adding the SDCA module alone brings a 3.12 dB improvement (vs. FFN only) and endows the model with controllability
- CDRL improves by 1.11 dB in PSNR and 0.01 in SSIM compared to traditional CL
- The scene recognition accuracy of the visual embedder reaches 97.55%, with misclassifications primarily occurring when degradation factors are insignificant
- Text Embedder > Visual Embedder > Classifier, because a fixed number of text descriptors can serve better as "degradation switches"
- Good generalization ability is also demonstrated in real-world scenarios

## Highlights & Insights
- **One-to-Composite Paradigm**: First to systematically define and handle composite degradation, distinguishing itself from One-to-One and One-to-Many paradigms
- **Controllable Restoration**: By manually inputting different text descriptions, users can selectively remove specific degradations (e.g., only dehazing without deraining), which is rare in the restoration field
- **Minimal Parameter Scale**: Achieving superior performance over competitors of 26-38M with only 5.98M parameters demonstrates that "letting the model know what to do" is more important than "stacking parameters"
- **CDRL Loss**: Constructs tighter lower-bound constraints using multiple degradation negative samples, an idea that can be transferred to other multi-task learning scenarios
- **Cascaded Degradation Modeling**: Models the cascaded relationship of low-light $\rightarrow$ rain/snow $\rightarrow$ haze based on physical laws, which is more physically plausible than simple superposition

## Limitations & Future Work
- Performance is limited in extremely dense degradation scenarios (e.g., heavy rain + dense haze)
- Generalization is limited when unconsidered degradation types (e.g., motion blur, compression artifacts) appear
- The 97.55% accuracy of the visual embedder still leaves room for misclassification, which may lead to incorrect restoration directions
- The combining method of scene description embeddings (averaging) is relatively simple, and stronger fusion strategies can be explored

## Related Work & Insights
- **vs. AirNet / TransWeather**: Full parameter-sharing One-to-Many methods cannot perceive structural degradation types, resulting in mutual interference among different degradations during mixed training
- **vs. All-weather Net (Li et al.)**: Partial parameter sharing, requiring independent encoders for each degradation, which limits scalability
- **vs. Restormer**: Although Restormer is a strong baseline, OneRestore outperforms it by 1.73 dB in composite degradation, demonstrating the importance of degradation perception

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically defines the composite degradation problem and proposes the One-to-Composite paradigm for the first time; the use of scene descriptors to control the restoration direction is highly novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation with the self-constructed CDD-11 dataset, comparison with 14 SOTAs, multi-dimensional ablations, real-world validation, and controllability demonstrations
- Writing Quality: ⭐⭐⭐⭐ Logically clear with well-defined problems, though math equations and symbols are somewhat intensive
- Value: ⭐⭐⭐⭐⭐ Resolves the core bottlenecks of composite degradation in real-world scenarios, paves a new path for controllable restoration, and is highly parameter-efficient

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)
- [\[ICML 2025\] FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](../../ICML2025/information_retrieval/fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](../../ACL2026/information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](../../ACL2025/information_retrieval/flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] SGIC: A Self-Guided Iterative Calibration Framework for RAG](../../ACL2025/information_retrieval/sgic_a_self-guided_iterative_calibration_framework_for_rag.md)

</div>

<!-- RELATED:END -->
