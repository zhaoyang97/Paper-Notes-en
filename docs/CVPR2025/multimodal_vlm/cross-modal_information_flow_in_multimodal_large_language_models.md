---
title: >-
  [Paper Note] Cross-modal Information Flow in Multimodal Large Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Multimodal Information Flow] Through the "attention knockout" method, the flow path of visual and textual information in MLLMs is systematically traced, revealing that visual information integrates into linguistic representations in two stages (first global, then local), and eventually propagates from the question positions to the last position in middle layers to generate the answer.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multimodal Information Flow"
  - "Attention Interpretability"
  - "Cross-Modal Fusion"
  - "LLaVA"
  - "VQA"
date: 2026-05-08
content_hash: 5164a27b03686b97
---

# Cross-modal Information Flow in Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2411.18620](https://arxiv.org/abs/2411.18620)  
**Code**: [https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Information Flow, Attention Interpretability, Cross-Modal Fusion, LLaVA, VQA

## TL;DR
Through the "attention knockout" method, the flow path of visual and textual information in MLLMs is systematically traced, revealing that visual information integrates into linguistic representations in two stages (first global, then local), and eventually propagates from the question positions to the last position in middle layers to generate the answer.

## Background & Motivation

**Background**: Autoregressive multimodal large language models (MLLMs) have made significant progress in vision-language tasks such as VQA, but their internal working mechanisms—especially how visual and language information interact and fuse—remain a black box.

**Limitations of Prior Work**: Existing interpretability works mainly focus on single aspects, such as knowledge storage in parameters, visual token redundancy, or safety mechanism localization, but lack a comprehensive understanding of the information flow paths between the two modalities. It remains unknown in which layers and in what manner the model integrates image and text information.

**Key Challenge**: The attention module in MLLMs is the only component that allows communication between hidden representations at different positions, but the functional division of this communication across different layers is unclear—do all layers perform fusion, or is there a clear division of stages?

**Goal** Three specific questions: (1) How does visual information propagate from image tokens to question tokens? (2) Is this propagation coarse-grained global or fine-grained local? (3) How does the fused multimodal information eventually reach the last position to generate the answer?

**Key Insight**: Leveraging the "attention knockout" method from mechanistic interpretability in NLP—by selectively blocking specific attention edges between token pairs in certain layers and observing the changes in answer prediction probability to reverse-engineer the information flow pathway.

**Core Idea**: Use attention knockout to block cross-modal attention edges layer by layer, revealing a two-stage mechanism of vision-language information fusion and a three-step prediction process in MLLMs.

## Method

### Overall Architecture
Focusing on autoregressive MLLMs (such as the LLaVA series) as the research subject, the input is a concatenated sequence of [image patch features, question tokens]. In the masked multi-head attention of each layer, by modifying the attention mask matrix, the attention edges from a specific source set (e.g., image tokens) to a target set (e.g., question tokens) are set to $-\infty$, thereby blocking the information flow between the two sets of tokens. The relative change in the answer prediction probability before and after the block, $p_c\% = ((p_2 - p_1)/p_1) \times 100$, is compared to quantify the importance of the information flow.

### Key Designs

1. **Attention Knockout Method**:

    - **Function**: Selectively block attention connections between specific token sets
    - **Mechanism**: In the attention mask matrix $M$, setting $M_{s,t} = -\infty$ for all position pairs $(s,t)$ from the source set $\mathbb{S}$ to the target set $\mathbb{T}$, making the attention weights after softmax zero. The blocking is performed using a sliding window of $k=9$ layers to avoid weak effects from a single layer.
    - **Design Motivation**: Attention is the only cross-position communication module in Transformers, and blocking it is equivalent to cutting off the information flow. It is more direct than gradient-based methods and can precisely locate the layers where information transfer occurs.

2. **Design of Three Control Experiments**:

    - **Function**: Disentangle the contribution pathways of different modalities to the final prediction
    - **Mechanism**: Map out three sets of knockout experiments—(a) Image→Last vs Question→Last: testing which modality directly affects the final prediction; (b) Image→Question: testing whether visual information implicitly propagates through question tokens; (c) RelatedPatches→Question vs OtherPatches→Question: distinguishing the contributions of global and local visual information.
    - **Design Motivation**: In autoregressive models, the image precedes the question, and images cannot attend to questions. Thus, there are only two paths—either the image goes directly to the last position, or it goes to the question first and then to the last position.

3. **Fine-Grained Image Region Splitting**:

    - **Function**: Differentiate the informational contribution of question-related image regions from background regions
    - **Mechanism**: Utilizing bounding box annotations from the GQA dataset, image patches are divided into $V_{obj}$ containing the questioned object and the remaining $V_{oth}$. Knockout is applied to each respectively to observe the difference in information flow across different layers.
    - **Design Motivation**: To verify whether the model truly "understands" the target object of the question, or merely relies on global database characteristics.

### Loss & Training
This is an analytical paper and does not involve training. Experiments are conducted on four pretrained models: LLaVA-1.5-7b/13b, LLaVA-v1.6-Vicuna-7b, and Llama3-LLaVA-NEXT-8b, using 6 categories of VQA tasks from the GQA dataset.

## Key Experimental Results

### Main Results

| Experimental Setting (LLaVA-1.5-13b) | Max Probability Drop | Critical Layer Range | Explanation |
|--------------------------|-------------|-----------|------|
| Question ↛ Last | ~30% | Middle layers (Layers 15-25) | Question information directly drives the final prediction |
| Image ↛ Last | ~5% | - | Image info does not directly affect the final prediction |
| Last ↛ Last | ~0% | - | The input at the last position itself contains no critical information |
| Image ↛ Question (Stage 1) | ~60% | Lower layers (Layers 0-4) | Global visual info integrates into question representations |
| Image ↛ Question (Stage 2) | ~21% | Middle layers (~Layer 10) | Object-related visual info integrates into question representations |

### Ablation Study

| Configuration | Probability Change | Affected Layers | Explanation |
|------|---------|-------|------|
| OtherPatches ↛ Question | Substantial drop in lower layers | Layers 0-4 | Stage 1 is primarily global/background information propagation |
| RelatedPatches ↛ Question | Drop in middle layers | ~Layer 10 | Stage 2 is primarily object-related information propagation |
| Small model (7b) vs Large model (13b) | 7b has weaker Stage 1 flow | Layers 0-4 | Small model has weaker global information integration capability |
| Logit lens observation | Lowercase answers emerge in middle layers | ~Layers 15-20 | Semantic reasoning is completed in the middle layers |
| Uppercase-to-lowercase conversion | Format correction in high layers | Layers 25-40 | Syntactic correction is completed in the higher layers |

### Key Findings
- **Two-stage visual fusion**: Lower layers (layers 0-4) handle global visual information propagation to question tokens, accounting for ~60% of the total information flow; middle layers (~layer 10) handle question-related local visual information propagation, accounting for ~21%. This two-stage pattern is consistent across all models and tasks.
- **Question tokens as information hubs**: The final prediction relies almost entirely on information flowing from the question positions to the last position, rather than direct contributions from the image. This indicates that the main arena for multimodal fusion lies within the hidden representations of the question tokens.
- **Semantics before syntax**: Observed via logit lens, the model first generates the correct answer in lowercase in the middle layers (semantic reasoning), and later converts the initial letter to uppercase in high layers (syntactic correction), which is a novel finding.

## Highlights & Insights
- **Simple yet effective attention knockout method**: It requires no training of probes or additional models, directly tracing the information flow by blocking attention edges, and can be extended to analyze any Transformer-based multimodal model.
- **Guiding significance of the two-stage fusion finding**: The pattern of global perception at lower layers and fine-grained local fusion at middle layers suggests that targeted visual token compression strategies can be designed for different layers, rather than uniform processing across all layers.
- **The "question tokens as fusion hubs" insight** explains why prompt engineering and question wording heavily impact VLM performance—because visual information ultimately needs to be "stored" in the representations of the question tokens.

## Limitations & Future Work
- Verified only on the LLaVA series; whether identical patterns apply to MLLMs with different architectures like Qwen-VL and InternVL remains unknown.
- Experiments are limited to VQA word/phrase answers; the information flow patterns for long-text generation tasks (such as image captioning) remain unexplored.
- The attention knockout method is based on the assumption that "blocking attention is equivalent to cutting off information flow," but FFN layers might also implicitly feed information (via representations accumulated in residual connections).
- It does not explore how to utilize the discovered information flow patterns to actively improve model design (such as layer-adaptive token compression).

## Related Work & Insights
- **vs Logit Lens / Tuned Lens**: These methods can only observe change in output distribution at each layer, failing to reveal information transfer paths between modalities. The knockout method of this work directly pinpoints the layers and directions in which cross-modal information flow occurs.
- **vs FastV / Token Compression Methods**: These methods empirically prune visual tokens. The findings of this work provide them with a theoretical foundation—lower layers require global tokens, whereas middle layers and beyond only need to retain question-related local tokens.
- **vs Probing Methods**: Probing detects "what information is contained" in representations, while this work reveals "where info comes from and how it gets there," making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically reveal the two-stage mechanism of cross-modal information flow in MLLMs. Although the method borrows from established knockout ideas, its application to multimodal scenarios represents a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models, 6 VQA tasks, and multiple control experiments with highly consistent conclusions; however, validation on non-LLaVA architectures is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured paper progressing step-by-step from simple to complex, with intuitive and easy-to-understand diagrams and tables.
- Value: ⭐⭐⭐⭐ Important contribution to the interpretability research of multimodal models; the findings can directly guide token compression and model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](../../ICML2026/multimodal_vlm/interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)
- [\[CVPR 2025\] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)
- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)
- [\[CVPR 2025\] RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models](rap_retrieval-augmented_personalization_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
