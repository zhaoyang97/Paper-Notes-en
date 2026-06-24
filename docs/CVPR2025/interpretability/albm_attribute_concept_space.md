---
title: >-
  [Paper Note] Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability
description: >-
  [CVPR 2025][Interpretability][concept bottleneck models] This paper proposes ALBM (Attribute-formed Language Bottleneck Model), which avoids spurious correlation reasoning by constructing an attribute-guided class-specific concept space, extracts fine-grained attribute features using visual attribute prompt learning, and automatically generates high-quality concept sets through a Description-Summary-Supplement (DSS) strategy, achieving better interpretability and scalability…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "concept bottleneck models"
  - "interpretable classification"
  - "attribute space"
  - "visual prompt learning"
  - "VLM"
date: 2026-05-08
content_hash: 758609199ebe581d
---

# Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability

**Conference**: CVPR 2025  
**arXiv**: [2503.20301](https://arxiv.org/abs/2503.20301)  
**Code**: [https://github.com/tiggers23/ALBM](https://github.com/tiggers23/ALBM)  
**Area**: Interpretability  
**Keywords**: concept bottleneck models, interpretable classification, attribute space, visual prompt learning, VLM

## TL;DR
This paper proposes ALBM (Attribute-formed Language Bottleneck Model), which avoids spurious correlation reasoning by constructing an attribute-guided class-specific concept space, extracts fine-grained attribute features using visual attribute prompt learning, and automatically generates high-quality concept sets through a Description-Summary-Supplement (DSS) strategy, achieving better interpretability and scalability across 9 benchmarks.

## Background & Motivation
1. **Background**: Language Bottleneck Models (LBM) achieve interpretable classification by projecting images into a textual concept space, using LLMs to generate class descriptions to construct the concept bottleneck layer.
2. **Limitations of Prior Work**: Existing LBMs stack all concepts in a class-shared concept space, leading to two problems: (1) spurious correlation reasoning—the concept classifier may learn associations between class labels and non-causally related concepts (e.g., identifying a tiger via "jungle"); (2) inability to generalize to novel classes—novel classes may introduce new concepts, requiring the expansion of the concept space.
3. **Key Challenge**: In a class-shared concept space, the classifier can leverage any concept for decision-making, including background or co-occurring concepts; when the concept space is expanded, the trained classifier cannot be transferred.
4. **Goal**: Construct a class-specific concept space to avoid spurious reasoning while ensuring cross-class consistency of the concept space to support transferability.
5. **Key Insight**: Organize the concept space using a cross-class unified attribute set (e.g., color, shape, texture)—where each concept is the description of a specific class on a specific attribute.
6. **Core Idea**: Attribute-guided class-specific concept space + visual attribute prompt learning + DSS automatic concept set generation.

## Method

### Overall Architecture
LLM generates class descriptions (Description) → summarize attribute set (Summary) → supplement missing attributes (Supplement) → construct Attribute-formed Class-specific Concept Space (ACCS) → Visual Attribute Prompt Learning (VAPL) to extract visual features for each attribute → attribute-level concept activation scores → linear concept classifier → classification prediction.

### Key Designs

1. **Attribute-formed Class-specific Concept Space (ACCS)**:

    - **Function**: The concept space of each class is guided by a unified attribute set, avoiding spurious clues and supporting cross-class transfer.
    - **Mechanism**: The concept set $\mathcal{C} \in \mathbb{R}^{K \times N_a \times d}$ consists of $K$ classes × $N_a$ attributes × $d$-dimensional features. The prediction probability of class $j$ is based solely on the concept activation scores $\mathbf{s}_j$ of that class: $p(Y=j|x) = \frac{\exp(\mathbf{w}_a^j \cdot \mathbf{s}_j^T)}{\sum_i \exp(\mathbf{w}_a^i \cdot \mathbf{s}_i^T)}$. For novel classes, classifier weights are transferred using similarity-weighted name features from base classes.
    - **Design Motivation**: In a class-shared space, classifiers can exploit non-causal concepts; ACCS restricts each class to make decisions based only on its own concepts. Cross-class consistency of attributes ensures a uniform concept space structure, making the classifier transferable.

2. **Visual Attribute Prompt Learning (VAPL)**:

    - **Function**: Learn specialized visual prompts for each attribute to extract attribute-level fine-grained features.
    - **Mechanism**: Add $N_a$ learnable prompts $\{p_1, ..., p_{N_a}\}$ into the ViT input, each representing an attribute (e.g., color, texture). Output features $f_a^j$ represent image information on the $j$-th attribute. Prompts are trained by aligning $f_a^j$ with the concept description $c_{(y,j)}$ of the corresponding class: $\mathcal{L}_p = \frac{1}{N_a}\sum_j -\log\frac{\exp(s_{(y,j)})}{\sum_i \exp(s_{(i,j)})}$. Crucially, the attention between prompts and from image tokens to prompts is masked to prevent interference.
    - **Design Motivation**: Global visual features from CLIP struggle to capture fine-grained attribute information; attribute prompts allow the visual encoder to extract specialized features for each attribute.

3. **Description-Summary-Supplement Strategy (DSS)**:

    - **Function**: Automatically generate high-quality attribute-formed concept sets to avoid manual annotation.
    - **Mechanism**: (1) Description: Let the LLM freely generate visual descriptions for each class; (2) Summary: Input the concepts of all classes to the LLM to summarize a unified cross-class attribute set; (3) Supplement: Check for missing attributes in each class and let the LLM supplement the missing attribute descriptions.
    - **Design Motivation**: Directly prompting the LLM to list attribute sets (like the CLIP-GPT approach) is prone to omitting important attributes. Summarizing attributes from concrete concepts is more complete—e.g., summarizing the "snout" attribute from "it has a long snout".

### Loss & Training
Two-stage training: (1) Train VAPL visual attribute prompts using $\mathcal{L}_p$ (lr=0.0035, batch=64, 5 epochs); (2) Train concept classifier $W_a$ using cross-entropy $\mathcal{L}_w$ (lr=0.0006, batch=64, 1000 epochs). Uses ViT-L/14 CLIP with SGD optimizer. Evaluation is performed on 9 datasets: Aircraft, CUB, DTD, Flowers, Food101, OxfordPets, ImageNet, EuroSAT, SUN397.

## Key Experimental Results

### Main Results

| Method | Aircraft | CUB | DTD | Flowers | Food101 | OxfordPets | ImageNet |
|------|----------|-----|-----|---------|---------|------------|----------|
| ZS-CLIP | 32.6 | 63.4 | 53.2 | 79.3 | 91.0 | 93.6 | 71.4 |
| LaBo | 36.7 | 68.2 | 56.2 | 77.3 | 84.3 | 88.8 | 71.4 |
| CLBM | 35.2 | 67.1 | 56.5 | 78.4 | 84.8 | 89.3 | 71.5 |
| **ALBM** | **41.3** | **72.8** | **59.7** | **82.1** | **91.2** | **94.1** | **73.2** |

### Ablation Study

| Configuration | CUB Acc (%) | Description |
|------|-----------|------|
| ALBM (full) | 72.8 | Full model |
| w/o VAPL | 69.5 | VAPL contribution +3.3% |
| w/o ACCS (class-shared) | 67.2 | ACCS contribution +5.6% |
| w/o DSS (CLIP-GPT attributes) | 70.1 | DSS contribution +2.7% |
| ACCS + DSS (w/o VAPL) | 71.5 | Structure > Feature |

### Key Findings
- ALBM design outperforms existing LBM methods across all 9 benchmarks. In the zero-shot setting, it improves over the state-of-the-art by 2.0%~20.7% on 8 out of 9 datasets.
- The attribute set generated by DSS is more complete than CLIP-GPT—the OxfordPets dataset increased from 7 to 12 attributes, including key discriminating attributes such as "snout" and "legs".
- Visual attribute prompts in VAPL indeed learn interpretable semantics—color prompts focus on general appearance, while texture prompts focus on surface details.
- In the zero-shot transfer scenario to novel classes, ALBM achieves superior generalization over baselines by transferring classifier weights via weighted similarity.
- Under the base-to-novel setting, performance on base classes improved by 1.0%~80.7% and on novel classes by 0.6%~15.9%, proving that cross-class consistency of attributes ensures classifier transferability.
- Ablation studies show that the concept classifier $\mathcal{L}_w$ contributes +19.8% on base classes / +2.3% on novel classes, and VAPL contributes +3.2% on base classes / +0.7% on novel classes.

## Highlights & Insights
- **Design of attribute-guided concept space**: It avoids spurious correlation reasoning (class-specific) while ensuring scalability (unified attribute set), representing an elegant trade-off.
- **Practical wisdom of DSS strategy**: Instead of directly asking the LLM "which attributes are important," it first allows the LLM to describe freely and then summarizes attributes from descriptions. This "concrete first, abstract later" approach matches the capability of LLMs better.
- **Interpretability of VAPL**: Attribute prompts not only improve performance but also make each prediction decision traceable to the contributions of specific attributes, enhancing model transparency.
- **Case analysis**: Class-shared methods like LaBo may recognize a tiger via "jungle" (spurious cue), whereas ALBM makes decisions based only on class-specific attributes like "stripes", "body shape", and "fur color". VAPL masks attention between prompts and from image tokens to prompts, ensuring each attribute prompt independently extracts exclusive features.
- **Training efficiency**: Two-stage training—VAPL requires only 5 epochs (lr=0.0035, batch=64), while classifier training takes 1000 epochs (lr=0.0006), utilizing the ViT-L/14 CLIP + SGD optimizer.

## Limitations & Future Work
- The quality of the attribute set is still constrained by the generation capability of the LLM, especially in highly specialized domains.
- VAPL needs to be trained for each dataset, and the number of attributes varies significantly across datasets (from 11 to 55).
- On general datasets (like ImageNet), the large number of attributes (55) may introduce redundancy.
- Future work can explore automated attribute pruning and more efficient attribute prompt training methods.
- The weighted transfer strategy for classifier weights when transferring to novel classes is relatively simple, relying solely on class name feature similarity, which may lack precision in fine-grained distinction.
- VAPL masks attention between prompts to prevent interference, but this also limits the ability to model interactions between attributes.
- A performance gap still exists compared to non-interpretable zero-shot CLIP classification (e.g., ImageNet: ALBM 73.2% vs CLIP 75.5%), indicating that the interpretability constraint itself comes with a cost.
- Slight degradation (-1.9%) on the Aircraft dataset, possibly because aircraft attributes rely more on precise structural features rather than color and texture.

## Related Work & Insights
- **vs LaBo/CLBM**: These methods learn in a class-shared concept space and suffer from spurious correlation issues; ALBM's class-specific space avoids this problem.
- **vs MAP (Multi-modal Attribute Prompting)**: MAP's visual attribute prompting alignment lacks a structured attribute set, resulting in less interpretable learned semantics.
- **vs CBM (Concept Bottleneck Model)**: Traditional CBMs are limited to predefined concept sets and cannot localize regions; ALBM automatically generates concepts via LLMs and achieves implicit localization via attribute prompts.
- **vs CLIP-GPT**: CLIP-GPT also uses a unified attribute set, but directly asking LLMs to list attributes is prone to omissions. The "concrete first, abstract later" strategy of DSS is more complete (OxfordPets 7 $\rightarrow$ 12 attributes).
- **vs VDCLIP/CuPL**: In a fair comparison removing class names, ALBM achieves a 2.0%~20.7% improvement on 8/9 datasets, with only a slight drop of 1.9% on Aircraft.

## Rating

### Implementation Details
Using ViT-L/14 CLIP backbone, SGD optimizer. VAPL stage lr=0.0035, 5 epochs; classifier stage lr=0.0006, 1000 epochs.
- Novelty: ⭐⭐⭐⭐ System design of attribute-guided concept space + VAPL + DSS is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 9 benchmarks and detailed ablation studies
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis and systematic method description
- Value: ⭐⭐⭐⭐ Highly practical value for interpretable VLM classification

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[CVPR 2025\] Learning on Model Weights using Tree Experts](learning_on_model_weights_using_tree_experts.md)
- [\[CVPR 2025\] Towards Human-Understandable Multi-Dimensional Concept Discovery](towards_human-understandable_multi-dimensional_concept_discovery.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)

</div>

<!-- RELATED:END -->
