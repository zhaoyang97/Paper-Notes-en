---
title: >-
  [Paper Note] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models
description: >-
  [ACL 2025][Multimodal VLM][VLM] This work systematically reveals a significant performance gap (up to 18%) in entity knowledge extraction in vision-language models (VLMs) between visual and textual representations. Using mechanistic interpretability tools, the authors discover that the key information flow of image tokens occurs deep within the intermediate layers of the model, leaving insufficient subsequent layers for factual reasoning.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "VLM"
  - "Entity Knowledge Extraction"
  - "Cross-Modal Gap"
  - "Mechanistic Interpretability"
  - "POPVQA"
date: 2026-05-08
content_hash: e3c47a7a105cc6cd
---

# Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.14133](https://arxiv.org/abs/2412.14133)  
**Code**: Yes (POPVQA dataset and experimental code)  
**Area**: Multimodal / Vision-Language Models  
**Keywords**: VLM, Entity Knowledge Extraction, Cross-Modal Gap, Mechanistic Interpretability, POPVQA

## TL;DR

This work systematically reveals a significant performance gap (up to 18%) in entity knowledge extraction in vision-language models (VLMs) between visual and textual representations. Using mechanistic interpretability tools, the authors discover that the key information flow of image tokens occurs deep within the intermediate layers of the model, leaving insufficient subsequent layers for factual reasoning.

## Background & Motivation

VLMs excel in image understanding tasks, but a key question remains overlooked: **Can a model leverage its internal knowledge just as effectively when an entity is presented as an image rather than text?**

Scenario: For the entity Robin Williams—
- Textual representation: Directly ask "Who is Robin Williams's spouse?"
- Visual representation: Provide an image of Robin Williams and ask "Who is the spouse of the subject in this image?"

The latter is inherently a **two-hop reasoning task**: the first hop identifies the entity in the image (identification), and the second hop extracts the related fact from internal knowledge (extraction). The authors hypothesize that VLMs allocate a substantial portion of their computational budget to the first-hop entity identification, leaving insufficient layers for the second-hop factual reasoning.

Existing datasets (such as InfoSeek and VeQuAE) do not contain explicit annotations of entities in images, making it difficult to decouple the identification and extraction steps. Therefore, the authors construct a specialized dataset, POPVQA.

## Method

### Overall Architecture

This work does not propose a new model, but is rather a **diagnostic study** comprising:
1. POPVQA dataset construction
2. Multi-model benchmark comparison experiments
3. Activation patching experiments
4. Forward activation freezing experiments
5. Attention knockout experiments

### Key Designs

1. **POPVQA Dataset**: Function $\rightarrow$ Constructing a visual factual question-answering dataset that decouples entity identification from question answering; Mechanism $\rightarrow$ It contains 15,395 entity-image pairs (63.6% celebrities, 18% landmarks, 14.6% artwork, 3.8% brands), with at least 2 factual questions generated per entity based on Wikidata triples; Design Motivation $\rightarrow$ To decouple identification failures from knowledge extraction failures by first ensuring the model "knows" the entity (identifying it) before evaluating its knowledge extraction capability. Images are uniformly scaled to 336x336, and popular entities are filtered based on Wikipedia page views.

2. **Cross Patching Experiments**: Function $\rightarrow$ Substituting the image hidden states of an injected entity into the forward pass of the original entity between two separate forward passes; Mechanism $\rightarrow$ Replacing the hidden states of the original entity at the corresponding position with those of the injected entity $\mathbf{h}_{1,v}^\ell \ldots \mathbf{h}_{n,v}^\ell$ at layer $\ell$, and observing the **critical layer** where the model's prediction switches from the injected entity to the original entity; Design Motivation $\rightarrow$ To find the boundary point where the model completes its processing of visual representation, i.e., the key layer where information from image tokens propagates to query/generation token positions.

3. **Forward Activation Freezing Experiments**: Function $\rightarrow$ Freezing the hidden states of image tokens to the value of a source layer starting from that source layer up to layer 20; Mechanism $\rightarrow$ Setting $(\mathbf{h}_{1,v}^\ell \ldots \mathbf{h}_{n,v}^\ell) = (\mathbf{h}_{1,v}^{source} \ldots \mathbf{h}_{n,v}^{source})$ for $\ell$ from source to 20; Design Motivation $\rightarrow$ To test whether entities can be identified in early layers—if correct identification persists even when freezing starts from early layers, it indicates that the vision encoder has already provided sufficient signals.

4. **Attention Knockout Experiments** (Appendix): Function $\rightarrow$ Blocking the attention of text/generation tokens to image tokens in the attention mask; Mechanism $\rightarrow$ Progressively knocking out attention layer-by-layer in both top-down and bottom-up directions, and observing which layer blockages are most detrimental; Design Motivation $\rightarrow$ To verify at which layers cross-modal information routing occurs.

### Loss & Training

This study is analytical and does not involve model training.

## Key Experimental Results

### Main Results

**Cross-Modal Performance Gap (POPVQA Dataset)**

| Model | Identified Entities | Visual Accuracy | Text Accuracy | Gap |
|------|-----------|-----------|-----------|------|
| LLaVA7B | 925 | 0.276 | 0.453 | **0.177** |
| LM-CLIP | 844 | 0.216 | 0.378 | 0.163 |
| LM-SigLIP | 660 | 0.201 | 0.377 | 0.176 |
| LLaVA34B | 1,286 | 0.534 | 0.656 | **0.121** |
| Qwen2-VL | 3,143 | 0.433 | 0.476 | **0.043** |

### Ablation Study

**Comparison of QA Accuracy for Early vs. Late Identified Entities**

| Model | Subgroup | Visual Accuracy | Text Accuracy | Gap |
|------|------|-----------|-----------|------|
| LM-CLIP | Early Identification | 0.260 | 0.442 | 0.182 |
| LM-CLIP | Late Identification | 0.192 | 0.347 | 0.154 |
| LM-SigLIP | Early Identification | 0.234 | 0.417 | 0.183 |
| LM-SigLIP | Late Identification | 0.168 | 0.338 | 0.170 |

### Key Findings

1. **Ubiquitous Performance Gap**: All evaluated models experience a decline in accuracy for visual versus textual representations, with the gap reaching up to 17.7% (LLaVA-7B).
2. **Model Depth Mitigates the Gap**: LLaVA-34B exhibits a significantly smaller gap (12.1%) than its 7B counterpart (17.7%), suggesting deeper models have more layers available for second-hop reasoning.
3. **Training Paradigm Matters More**: Qwen2-VL shows a gap of only 4.3%, benefiting from unfreezing the visual backbone parameters during training to achieve superior cross-modal alignment.
4. **Key Information Flow Occurs After Intermediate Layers**: The critical point is at layer 20 (out of 32) for LLaVA-7B, layer 17 for LM-SigLIP, and layer 24 for LM-CLIP. Patching image tokens before these layers triggers predictions corresponding to the injected entity.
5. **Early Identification Does Not Equate to Better QA**: Even if certain entities can be identified within the first 5 layers (as the vision encoder has already provided sufficient signals), model behavior correlates with improvements in textual accuracy. This suggests that the model simply has richer knowledge about these entities rather than leveraging early identification advantages; indeed, the gap is wider, indicating that models fail to utilize early-identification benefits efficiently.
6. **Common Mistakes**: When queried about the spouse/parent/sibling of an entity, models frequently output the name of the entity itself, indicating insufficient information routing.

## Highlights & Insights

- **Precise Modeling of Two-Hop Reasoning**: Decomposing visual entity QA in VLMs into identification and extraction stages provides a clear analytical framework.
- **Novel Cross-Entity Patching**: Pinpointing the timing of information routing by injecting the visual hidden states of one entity into the forward pass of another.
- **Counter-Intuitive Finding**: In contrast to the "picture superiority effect" in cognitive psychology (where humans perform better with images), VLMs exhibit "text superiority".
- **Inefficient Layer Utilization**: Even when the vision encoder has completed entity identification (early layers already contain sufficient information), the LLM component does not utilize this information until the intermediate layers.

## Limitations & Future Work

1. **Mechanistic Analysis Limited to LLaVA Architecture**: While its simple architecture facilitates intervention experiments, the positive results of other architectures (e.g., Qwen2-VL) have not been deeply analyzed.
2. **Confounding Visual Factors**: Athletes wearing national team jerseys might be misidentified; visual cues in images may inadvertently guide or mislead the answers.
3. **Limited Entity Types**: The dataset primarily consists of celebrities, landmarks, brands, and paintings, lacking coverage of broader entity types and diverse demographics.
4. **No Solutions Proposed**: The paper diagnoses the problem without offering methods to mitigate the gap (which the authors acknowledge).
5. **Potential Directions**: Early-fusion models might alleviate this issue by establishing a shared embedding space for vision and text.

## Related Work & Insights

- **Mechanistic Interpretability**: Inherits and extends the attention knockout and activation patching methodologies from Geva et al.
- **VLM Interpretability**: Complements concurrent works by Neo et al. and Kaduri et al., which discovered that query tokens store high-level image descriptions in the intermediate layers.
- **Multimodal Knowledge Conflict**: While Zhu et al. proposed methods to detect and mitigate parametric knowledge conflict, this paper provides a lower-level mechanistic explanation.
- **Inspirations for Dataset Design**: POPVQA demonstrates how carefully designed datasets can decouple composite capability evaluations.

## Rating

- **Novelty**: ★★★★☆ — The systematic exploration of the cross-modal knowledge extraction gap and the two-hop reasoning perspective are highly valuable.
- **Experimental Thoroughness**: ★★★☆☆ — Deep analysis, but mechanistic experiments are limited to the LLaVA architecture with a capped sample size.
- **Writing Quality**: ★★★★☆ — Well-structured logic, intuitive graphics, and cohesive experimental design.
- **Value**: ★★★★☆ — Unveils fundamental information processing bottlenecks in VLMs, offering crucial insights for model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)
- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ACL 2025\] Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](con_instruction_universal_jailbreaking_of_multimodal_large_language_models_via_n.md)
- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)

</div>

<!-- RELATED:END -->
