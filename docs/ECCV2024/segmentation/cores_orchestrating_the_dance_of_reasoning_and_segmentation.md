---
title: >-
  [Paper Note] CoReS: Orchestrating the Dance of Reasoning and Segmentation
description: >-
  [ECCV 2024][Segmentation][reasoning segmentation] This paper proposes CoReS (Chains of Reasoning and Segmenting), a multimodal Chain-of-Thought framework with a dual-chain structure. Through the hierarchical collaboration of the reasoning chain and the segmenting chain, combined with an in-context guidance strategy, it achieves progressive and precise segmentation of target objects in complex reasoning text, outperforming LISA by 6.5% on the ReasonSeg dataset.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "reasoning segmentation"
  - "chain-of-thought"
  - "MLLM"
  - "multi-modal reasoning"
  - "SAM"
date: 2026-05-08
content_hash: ae933df2fc4526fe
---

# CoReS: Orchestrating the Dance of Reasoning and Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2404.05673](https://arxiv.org/abs/2404.05673)  
**Code**: [https://chain-of-reasoning-and-segmentation.github.io](https://chain-of-reasoning-and-segmentation.github.io)  
**Area**: Image Segmentation  
**Keywords**: reasoning segmentation, chain-of-thought, MLLM, multi-modal reasoning, SAM

## TL;DR
This paper proposes CoReS (Chains of Reasoning and Segmenting), a multimodal Chain-of-Thought framework with a dual-chain structure. Through the hierarchical collaboration of the reasoning chain and the segmenting chain, combined with an in-context guidance strategy, it achieves progressive and precise segmentation of target objects in complex reasoning text, outperforming LISA by 6.5% on the ReasonSeg dataset.

## Background & Motivation
**Background**: Reasoning segmentation is an emerging class of tasks that requires models to understand complex reasoning queries and precisely locate and segment target regions. Existing methods mainly fall into two categories: equipping MLLMs with segmentation decoders (such as LISA, which connects SAM to LLaVA) or directly outputting masks in textual form from LLMs (such as VistaLLM).

**Limitations of Prior Work**: Existing MLLMs perform well in object-level segmentation but struggle with precise localization when processing reasoning text. For example, when segmenting "the part that provides the sense of smell for a dog," LISA directly searches for a "round, dark, perceptually significant" region, resulting in the incorrect segmentation of the dog's eyes instead of its nose—because the two share high semantic similarity in features.

**Key Challenge**: Reasoning segmentation requires both complex reasoning capabilities and precise localization capabilities. MLLMs possess reasoning capabilities, but directly using a single [SEG] token to perform one-step localization is susceptible to interference from semantically similar instances.

**Goal**: How to make MLLMs mimic the human cognitive process of visual search—from coarse to fine, progressively narrowing down the search space—to complete complex reasoning segmentation tasks.

**Key Insight**: Drawing inspiration from the top-down cognitive process of human visual search—first leveraging prior knowledge to locate the general region, and then progressively focusing on the precise target. For example, when searching for a "wedding ring," humans will first locate the hand and then find the ring within the localized hand region.

**Core Idea**: Extend the chain-of-thought of LLMs from pure text reasoning to a multi-modal dual-chain structure (Reasoning Chain + Segmenting Chain) to guide progressively precise segmentation through hierarchical semantic logic.

## Method

### Overall Architecture
Input: Image $Q_{img}$ + Query text $Q_{text}$ $\to$ MLLM (LLaVA) processes the image-text input and generates a reasoning chain output conforming to hierarchical logic (containing two special tokens, [LOC] and [SEG]) $\to$ the token embeddings of [LOC] and [SEG] serve as guidance for the segmenting chain, driving the SAM framework to generate segmentation results layer by layer $\to$ Output: the final precise target segmentation mask. Additionally, randomly sampled in-context textual examples are provided as extra input to guide the MLLM to generate outputs conforming to the hierarchical logic.

### Key Designs

1. **Chain-of-Reasoning**:

    - **Function**: Constrain the output of the MLLM to follow a top-down semantic logic hierarchy.
    - **Mechanism**: Use the response template "It appears on [LOC]. It is [SEG]." to constrain the sentence structure of the MLLM's output. This forces the MLLM to inject different levels of information at different token positions—injecting scene or object information where the target is most likely to appear (e.g., "knife or propeller") at the [LOC] position, and injecting feature information of the target itself (e.g., "flat cutting or pushing surface") at the [SEG] position. The MLLM is fine-tuned with LoRA using the cross-entropy loss $\mathcal{L}_{CoR} = \mathcal{L}_{CE}(\mathbf{p}(H|Q_{img}, Q_{text}), \mathbf{t})$ to supervise the reasoning chain output.
    - **Design Motivation**: Relying on a single [SEG] token is insufficient to capture hierarchical reasoning logic; by template-matching the output structure, the model is forced to encode semantic information of different granularities at different positions, forming a coarse-to-fine reasoning chain.

2. **Chain-of-Segmenting**:

    - **Function**: Utilize the logic guidance of the reasoning chain to iteratively generate segmentation results from coarse to fine.
    - **Mechanism**: Extract the embeddings of [LOC] and [SEG] positions ($\mathbf{h}^0, \mathbf{h}^1 \in \mathbb{R}^{1\times256}$) from the last layer of the MLLM, and sequentially use them as text prompts for the SAM mask decoder. The segmenting chain is iterative: the segmentation result $m^{t-1}$ from the previous level is processed by the SAM prompt encoder to serve as the mask prompt for the next level ($\mathcal{M}^t = \theta(m^{t-1})$), guiding the progressive refinement of the segmentation. The final segmentation process is $m^t = \gamma(F_v(Q_{img}), \mathcal{M}^t, \hat{\mathbf{h}}^t)$.
    - **Design Motivation**: The segmenting chain directly maps the semantic hierarchy of the reasoning chain to the visual modality—the first layer performs scene-level localization based on [LOC], and the second layer performs precise segmentation within the localized region based on [SEG].

3. **Token Refiner**:

    - **Function**: Use the visual segmentation results from the previous layer to reversely calibrate the text token embedding of the next layer.
    - **Mechanism**: Use masked average pooling (MAP) to create a prototype from the visual features corresponding to the previous layer's mask, and then calibrate the token embedding of the next layer through cross-attention: $\hat{\mathbf{h}}^t = \mathcal{R}(\beta(\mathbf{h}^t), F_v(Q_{img}), m^{t-1})$, where $\mathcal{R}(h, i, m) = h + CA(h, MAP(i, m))$.
    - **Design Motivation**: The tokens output by the MLLM are generated during a single forward pass, lacking feedback from visual information. The token refiner allows intermediate visual modality results to reversely enhance the textual guidance, achieving true bi-directional multimodal hierarchical interaction.

4. **In-Context Guidance**:

    - **Function**: Provide exemplars of logical rules to the MLLM during training and inference.
    - **Mechanism**: Pre-build a pure-text context library (with more QA pairs generated by ChatGPT based on manual exemplars). During each forward pass, a textual QA exemplar unrelated to the current query is randomly sampled and placed before the user input as in-context input. These exemplars do not contain special tokens; they only describe "where objects typically appear" and "common features of objects" in natural language, implicitly conveying the top-down output logical rules.
    - **Design Motivation**: Solely relying on templates to constrain sentence structure is insufficient to make MLLMs actively discover and output hierarchical logical relationships. By using in-context exemplars as "rule providers," the MLLM extracts top-down rules from pure text context and transfers them to multimodal task outputs. This approach avoids the high computational cost of building additional multimodal CoT datasets or utilizing two-stage MLLMs.

### Loss & Training
- **Reasoning Chain Loss**: $\mathcal{L}_{CoR} = \mathcal{L}_{CE}(\mathbf{p}(H|Q_{img}, Q_{text}), \mathbf{t})$, standard cross-entropy loss to supervise textual output.
- **Segmenting Chain Loss**: $\mathcal{L}_{CoS} = \lambda_d \mathcal{L}_{DICE}(m^T, M_{gt}) + \lambda_c \mathcal{L}_{CE}(m^T, M_{gt})$, supervising only the final level's mask ($\lambda_d=0.5, \lambda_c=2.0$).
- **Total Loss**: $\mathcal{L}_{total} = \lambda_R \mathcal{L}_{CoR} + \lambda_S \mathcal{L}_{CoS}$, where $\lambda_R = \lambda_S = 0.5$.
- **Training Strategy**: LoRA fine-tuning of the MLLM (LLaVA-7B-v0); the SAM Image Encoder and Prompt Encoder are frozen to maintain generalization capability; the projection layer and mask decoder are trainable.
- **Training Data**: Following LISA, a mixture of semantic/referring/reasoning segmentation datasets is used, without VQA data.

## Key Experimental Results

### Main Results

| Method | MLLM | ReasonSeg val gIoU | ReasonSeg test gIoU |
|------|------|---------------------|---------------------|
| OVSeg | CLIP ViT-L | 28.5 | 26.1 |
| GRES | BERT | 22.4 | 21.3 |
| X-Decoder | UniCL | 22.6 | 21.7 |
| SEEM | UniCL | 25.5 | 24.3 |
| LISA | LLaVA-7B | 44.4 | 36.8 |
| **CoReS** | LLaVA-7B | **54.8** | **48.7** |
| LISA (ft) | LLaVA-7B | 52.9 | 47.3 |
| **CoReS (ft)** | LLaVA-7B | **59.4** | **52.4** |
| LISA (ft) | LLaVA-13B | 56.2 | 51.7 |
| **CoReS (ft)** | LLaVA-13B | **61.8** | **55.9** |
| LISA (ft) | LLaVA-v1.5-13B | 65.0 | 61.3 |
| **CoReS (ft)** | LLaVA-v1.5-13B | **68.1** | **65.5** |

### Ablation Study

| InC | CoR | CoS | CoS-R | gIoU | cIoU |
|-----|-----|-----|-------|------|------|
| | | | | 52.9 | 54.0 |
| ✓ | | | | 53.2 | 55.4 |
| | ✓ | | | 54.0 | 54.1 |
| | | ✓ | | 53.9 | 55.6 |
| | ✓ | ✓ | | 55.3 | 56.9 |
| | ✓ | | ✓ | 55.4 | 58.0 |
| | | ✓ | ✓ | 56.9 | 59.8 |
| | ✓ | | ✓ | 57.5 | 60.4 |
| ✓ | ✓ | ✓ | | 58.4 | 59.3 |
| ✓ | ✓ | | ✓ | **59.4** | **62.1** |

Ablation on the number of in-context inputs: 1 exemplar is optimal (59.4 gIoU); 2 and 4 exemplars show a slight decrease but still outperform not using any (57.5).

Ablation on logical hierarchy depth: 2 layers ([L]+[S]) = 59.4 vs. 3 layers ([L]+[P]+[S]) = 59.7; the gain is limited but significantly outperforms 1 layer ([S]) = 53.2.

### Key Findings
1. **CoReS without fine-tuning outperforms fine-tuned LISA**: CoReS without ReasonSeg fine-tuning (54.8 gIoU) already outperforms LISA fine-tuned on ReasonSeg (52.9 gIoU), indicating that the performance enhancement originates from better harnessing the potential of the MLLM rather than relying on training data.
2. **Dual-chain collaboration is core**: CoS-R (with token refiner) guided by CoR contributes the main improvement of +4.6%, proving the effectiveness of hierarchical bi-directional multimodal interaction.
3. **In-context guidance brings extra gain**: Adding in-context inputs on top of the dual-chain yields an additional +1.9% improvement, demonstrating the effectiveness of logical rule guidance.
4. **More pronounced advantages on refCOCOg**: The queries in refCOCOg are longer and more complex (averaging 8.4 words), where CoReS achieves a more significant improvement (+2%), confirming that multimodal chain-of-thought primarily enhances scenarios with complex queries.
5. **Two-layer logic is sufficient**: The 3-layer structure yields only a marginal improvement. The query complexity of ReasonSeg does not require deeper logical chains, and longer chains introduce difficulties in backpropagation.

## Highlights & Insights
1. **Pioneering multimodal CoT for dense prediction**: Extending chain-of-thought from visual understanding tasks to dense prediction tasks like segmentation represents a significant breakthrough in the application scope of CoT.
2. **In-context rule transfer**: Text-domain in-context exemplars can transfer output logical rules to multimodal tasks without the need to construct multimodal CoT datasets, which is highly practical.
3. **Cross-modal feedback via Token Refiner**: The intermediate results of the visual modality reversely calibrate the text tokens, establishing a true bi-directional interaction rather than unidirectional text-to-visual guidance.
4. **Generalizability of the approach**: The top-down hierarchical visual search paradigm can be generalized to other multimodal tasks requiring precise localization (such as grounding and detection).

## Limitations & Future Work
1. **Limited depth of logical hierarchy**: The improvement brought by a 3-layer structure is limited, likely because the ReasonSeg dataset itself is not sufficiently complex. More complex reasoning tasks may require adaptive determination of hierarchical depth.
2. **Quality bottleneck of the in-context library**: Performance drops slightly as the number of exemplars increases, suggesting insufficient quality and diversity of the context library. Retrieval-augmented selection (choosing in-context exemplars based on query similarity) could be considered.
3. **Validation limited to ReasonSeg**: Benchmarks for reasoning segmentation tasks are currently scarce, and the generalization of the method requires validation on more datasets.
4. **Unsupervised intermediate-level masks**: The intermediate levels of the segmenting chain lack ground-truth supervision, which may lead to unstable intermediate results; weakly-supervised or self-supervised training for intermediate layers could be explored.
5. **Computational overhead**: The dual-chain structure combined with in-context inputs increases the input length and computational cost, making it less deployment-friendly.

## Related Work & Insights
- **LISA** [Lai et al., 2023]: The direct baseline of CoReS, which integrates SAM with LLaVA for reasoning segmentation. CoReS builds upon this by introducing a hierarchical reasoning structure.
- **V\*** [Wu et al., 2024]: Utilizes LLM + MLLM to construct visual search algorithms, outperforming GPT-4V. This inspired the top-down visual search approach.
- **KAM-CoT** [Mondal et al., 2024]: Uses knowledge graphs to assist multimodal CoT training, reducing computational costs. This inspired low-cost CoT construction.
- **Insights**: The dual-chain structure of CoReS can be transferred to 3D scene understanding (localizing rooms before objects) or video understanding (localizing key frames before targeting objects).

## Rating
- ⭐⭐⭐⭐ **Novelty**: Extending multimodal CoT from comprehension to dense prediction is pioneering, and the dual-chain + in-context guidance design is elegant.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: The main experiments on ReasonSeg are solid, the ablation study comprehensively covers each component, and the metric improvements are significant.
- ⭐⭐⭐⭐ **Writing Quality**: The structure of the paper is clear, and the motivation is intuitively illustrated with the "dog's nose" example.
- ⭐⭐⭐⭐⭐ **Value**: Provides a systematic framework and approach for reasoning segmentation, and the concept of in-context rule transfer has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[ICLR 2026\] SAM-Veteran: An MLLM-based Human-like SAM Agent for Reasoning Segmentation](../../ICLR2026/segmentation/sam-veteran_an_mllm-based_human-like_sam_agent_for_reasoning_segmentation.md)
- [\[ICLR 2026\] Urban Socio-Semantic Segmentation with Vision-Language Reasoning](../../ICLR2026/segmentation/urban_socio-semantic_segmentation_with_vision-language_reasoning.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](../../ICCV2025/segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[CVPR 2026\] VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation](../../CVPR2026/segmentation/virst_video-instructed_reasoning_assistant_for_spatiotemporal_segmentation.md)

</div>

<!-- RELATED:END -->
