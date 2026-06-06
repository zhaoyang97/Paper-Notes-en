---
title: >-
  [Paper Note] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks
description: >-
  [AAAI 2026][Multimodal VLM][Chemical VLM] TinyChemVL is a chemistry-domain VLM with only 4B parameters. It compresses visual tokens to 1/16 of the original count via an adaptive token merging and pruning strategy…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Chemical VLM"
  - "visual token compression"
  - "molecular recognition"
  - "reaction prediction"
  - "efficient inference with small models"
date: 2026-05-08
content_hash: 8d4ae52c851f159a
---

# TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks

**Conference**: AAAI 2026
**arXiv**: [2511.06283](https://arxiv.org/abs/2511.06283)  
**Code**: [https://github.com/xxlllz/TinyChemVL](https://github.com/xxlllz/TinyChemVL)  
**Area**: Multimodal VLM
**Keywords**: Chemical VLM, visual token compression, molecular recognition, reaction prediction, efficient inference with small models

## TL;DR
TinyChemVL is a chemistry-domain VLM with only 4B parameters. It compresses visual tokens to 1/16 of the original count via an adaptive token merging and pruning strategy, introduces reaction-level tasks and the ChemRxn-V benchmark, and achieves state-of-the-art performance on both molecular- and reaction-level visual chemistry tasks while significantly improving inference and training speed.

## Background & Motivation

### State of the Field
Large language models (LLMs) are increasingly applied in chemistry, yet conventional approaches primarily rely on text-based molecular representations (e.g., SMILES, SELFIES), inevitably losing spatial information. Although vision-language models (VLMs) are capable of processing visual inputs, their exploration in chemistry remains limited. Existing chemical VLMs (e.g., ChemVLM, ChemMLLM) are directly fine-tuned on general-purpose VLM architectures and suffer from notable efficiency issues.

### Limitations of Prior Work

**Severe visual token redundancy**: In molecular images, all structural information is concentrated in the molecular diagram region, while large background areas are semantically meaningless. For instance, in ChemVLM, an 800×800 image requires 1,280 visual tokens—approximately 100 times the number of tokens in the textual question—greatly increasing computational overhead. Furthermore, slicing molecular images into patches disrupts critical molecular structure information.

**Narrow task scope**: Existing chemical VLMs focus primarily on molecular-level tasks (e.g., SMILES OCR, property prediction), neglecting reaction-level tasks. Reaction-level tasks (e.g., reaction prediction) require the model to simultaneously recognize and reason, representing a more challenging direction.

**Molecular image generation bottleneck**: The VQ-GAN approach proposed by ChemMLLM requires an external tool to parse generated images back into SMILES, resulting in a complex pipeline and architectural mismatch.

### Starting Point
The paper advances along two dimensions simultaneously—**model efficiency** and **task complexity**: on one hand, it compresses visual representations via an adaptive token reduction strategy; on the other hand, it extends the capabilities of chemical VLMs from the molecular level to the reaction level, building a more comprehensive chemical visual understanding capability.

## Method

### Overall Architecture
TinyChemVL adopts the classical ViT–MLP–LLM architecture, with InternVL2.5-4B as the backbone and a dynamic resolution strategy that splits high-resolution images into multiple 448×448 tiles. The core innovation lies in embedding an adaptive token merging and pruning module within the visual encoder, progressively reducing the number of visual tokens between attention layers and FFN layers.

### Key Designs

#### 1. **Adaptive Token Merge and Pruning**
- **Function**: Within each transformer block of the ViT, adaptively prune unimportant tokens and merge redundant tokens according to the current visual token distribution.
- **Mechanism**:
    - **Token scoring**: Adopts the ATS (Adaptive Token Sampler) method, computing importance scores using the attention weights of the CLS token over other tokens from the attention matrix, combined with the norm of the Value matrix:
    $Score_i = \frac{A_{1,i+1} \times \|V_{i+1}\|}{\sum_{j=1}^{N} A_{1,j+1} \times \|V_{j+1}\|}$
    - **Token pruning**: Applies a Top-K selection strategy, retaining the $K$ highest-scoring tokens and directly discarding low-scoring tokens (corresponding to background regions).
    - **Token merging**: Uses the Bipartite Soft Matching (BSM) algorithm, splitting tokens into two groups and finding the most similar token pairs via cosine similarity for weighted-average merging. **Not constrained by spatial proximity**—non-adjacent but similar tokens can also be merged.
    - **Proportional Attention**: Maintains a row vector $s$ tracking the number of original tokens each current token represents, and adds a $\log s$ bias during attention computation to preserve information fidelity.
- **Design Motivation**: The sparse nature of molecular images means that most visual tokens correspond to blank backgrounds while molecular structure information is dense. A strategy is needed that can both eliminate redundancy and preserve structural information.

#### 2. **Adaptive Policy**
- **Function**: Adaptively selects between pruning and merging at both the instance level and the layer level.
- **Mechanism**: Computes the variance of token scores $S_{op_i} = \text{var}(Score_i)$:
    - Low variance ($S_{op_i} \leq \tau$) → importance converged → large background region → **apply pruning**
    - High variance ($S_{op_i} > \tau$) → importance differentiated → complex molecular structure → **apply merging**
    - Threshold $\tau$ is set to $1e{-5}$ by default, determined through statistical analysis of chemical image datasets.
- **Design Motivation**: Token distributions vary greatly across images and layers; a fixed strategy cannot accommodate all cases. Variance automatically determines whether the current situation is "large background" or "complex structure."

#### 3. **Reaction-Level Tasks and Code Generation Paradigm**
- **Function**: Extends the chemical VLM from molecular-level to reaction-level tasks, and replaces direct image generation with executable code.
- **Mechanism**:
    - Constructs a **reaction recognition** task: parsing reactants>>reagents.solvents>>products format from reaction images.
    - Constructs a **reaction prediction** task: predicting products from reactant images only (the first definition of this task).
    - **Molecular image generation**: Generates executable Python code to render molecular images, with SMILES directly embedded in the code, requiring no additional parsing tools.
- **Design Motivation**: Reaction prediction is a task that chemistry experts perform by observing molecular structures; it is a critical capability for VLMs as chemical research tools. The code generation paradigm is more direct and verifiable than VQ-GAN.

### Data Construction
A large-scale multi-task training set of approximately 1.25 million samples was constructed:
- Molecular recognition: 500K (385K self-built), sourced from ChEBI-20-MM, MolGrapher, MolScribe, ORDerly, etc.
- Reaction recognition: 200K (all self-built), rendered from the ORDerly dataset.
- Property prediction: 150K (55K self-built).
- Reaction prediction: 200K (all self-built).
- Molecular image generation: 200K (55K self-built).

### Loss & Training
- Backbone: InternVL2.5-4B, full-parameter fine-tuning.
- Hardware: 8× NVIDIA A100 80G.
- Training epochs: 1.5.
- Batch settings: per-device batch size 16, gradient accumulation 2.
- Training framework: ms-swift.

## Key Experimental Results

### Main Results — Molecular Recognition

| Model | ChemOCR Avg Sim.(%) | ChemOCR Tani@1.0(%) | img2smiles Avg Sim.(%) | img2smiles Tani@1.0(%) |
|------|---------------------|---------------------|------------------------|------------------------|
| GPT-4o | 36.8 | 3.4 | 29.0 | 0.01 |
| ChemVLM-8B | 81.7 | 57.7 | 55.0 | 15.0 |
| ChemDFM-X(13B) | 70.9 | 36.5 | **90.9** | **77.6** |
| ChemMLLM | - | - | 75.0 | 49.0 |
| **TinyChemVL(4B)** | **91.2** | **77.4** | 89.5 | 75.6 |

### Main Results — Reaction-Level Tasks (ChemRxn-V)

| Model | Reaction Recognition Avg Sim.(%) | Reaction Recognition EM(%) | Reaction Prediction Avg Sim.(%) | Reaction Prediction Tani@1.0(%) |
|------|---------------------|----------------|---------------------|---------------------|
| GPT-4o | 19.1 | 0.1 | 30.4 | 1.4 |
| ChemDFM-X | 28.32 | 3.2 | 12.7 | 0.7 |
| ChemVLM-8B | 0.6 | 0.0 | 4.8 | 0.0 |
| **TinyChemVL** | **93.4** | **67.9** | **78.9** | **52.4** |

### Efficiency Comparison

| Model | Inference Speed (Sample/s↑) | Avg. Token Count (↓) | Training Time (hours↓) |
|------|---------------------|----------------|----------------|
| ChemVLM-8B | 7.41 | 896 | - |
| InternVL2.5-4B | 9.11 | 894 | 47* |
| **TinyChemVL** | **11.84** | **108** | **15** |

### Ablation Study

| Config (tokens/image) | ChemOCR Tani@1.0(%) | Reaction Recognition EM(%) | Reaction Prediction Tani@1.0(%) |
|---------------------|---------------------|----------------|---------------------|
| 16 (default) | 77.4 | 62.7 | 52.4 |
| 4 | 76.2 (-1.2) | 59.5 (-3.2) | 50.1 (-2.3) |

### Key Findings
1. **TinyChemVL with 4B parameters outperforms ChemVLM at 8B and 26B**, achieving 91.2% average similarity on ChemOCR—the first general-purpose VLM competitive with dedicated SMILES OCR models.
2. **Dominates all existing models on reaction-level tasks**: reaction recognition 93.4%, reaction prediction 78.9%, while all other models remain below 30%.
3. **Visual tokens reduced from 896 to 108** (approximately 1/8), with ~60% inference speedup and training time reduced from 47 hours to 15 hours.
4. MSE on property prediction tasks is approximately half that of ChemMLLM, achieving the best performance on 5 out of 7 properties.
5. Further reducing tokens from 16 to 4 leads to performance degradation, indicating that 4 tokens are insufficient to represent complex reaction images.

## Highlights & Insights
1. **Precise problem definition**: The paper accurately identifies the sparsity characteristics of chemical images; the dual properties of "redundant background + dense structure" make the adaptive strategy particularly effective.
2. **Elegant variance-driven adaptive strategy**: Token score variance is used to distinguish "background regions" from "complex structures"—simple, effective, and adaptive at both the instance and layer levels.
3. **First definition of visual reaction prediction**: Predicting products directly from reactant images elevates the chemical application of VLMs from OCR-level to reasoning-level.
4. **Code generation as a substitute for image generation**: Avoids the architectural mismatch of VQ-GAN; the generated code directly contains SMILES for easy verification.
5. **Small model, strong capability**: The 4B model surpasses 13B–26B models, demonstrating that efficiency and quality can be achieved simultaneously.

## Limitations & Future Work
1. Validation is currently limited to the chemistry domain; whether the token compression strategy generalizes to other scientific images (e.g., biology, materials science) remains to be verified.
2. The reaction prediction task currently relies solely on visual information, without incorporating textual information about reaction conditions (temperature, catalysts, etc.).
3. Although the code generation approach for molecular image generation is elegant, the executability and robustness of the generated code are not thoroughly discussed.
4. The ChemRxn-V benchmark contains only 5,000 samples per task, which is relatively limited in scale.
5. Handwritten molecular recognition is explored but is not treated as a primary evaluation direction.

## Related Work & Insights
- **ToMe (Token Merging)**: TinyChemVL's token compression strategy directly builds upon ToMe's bipartite soft matching algorithm, augmented with an adaptive pruning/merging decision mechanism.
- **ChemVLM**: Serves as a direct comparison baseline, using the same InternVL architecture but without token compression.
- **ChemMLLM**: Introduces the molecular image generation task but suffers from architectural mismatch due to the use of VQ-GAN.
- Insight: **The key to domain-specific VLMs lies in identifying structural characteristics of visual redundancy**; the sparse background in chemical images is a canonical example.

## Rating
- Novelty: ⭐⭐⭐⭐ (Token compression is not a novel idea, but its adaptive application in the chemistry domain and the definition of reaction-level tasks are original.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers five major tasks—molecular recognition, property prediction, image generation, reaction recognition, and reaction prediction—with complete efficiency analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and well-motivated, though some mathematical notation typesetting could be improved.)
- Value: ⭐⭐⭐⭐ (Delivers a meaningful contribution to chemical AI, with practical deployment value as an efficient small-model solution.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](../../ICCV2025/multimodal_vlm/llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[CVPR 2026\] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](../../CVPR2026/multimodal_vlm/adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](../../ACL2026/multimodal_vlm/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)

</div>

<!-- RELATED:END -->
