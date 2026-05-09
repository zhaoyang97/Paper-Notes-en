---
title: >-
  [Paper Note] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning
description: >-
  [AAAI 2026][Object Detection][Visual Grounding] This paper proposes GroundingAgent, a visual grounding framework that requires no task-specific fine-tuning. By composing pretrained open-vocabulary detectors (YOLO World), an MLLM (Llama-3.2-11B-Vision), and an LLM (DeepSeek-V3) into a structured iterative reasoning pipeline, the method achieves a zero-shot average accuracy of 65.1% on RefCOCO/+/g, substantially outperforming prior zero-shot approaches.
tags:
  - AAAI 2026
  - Object Detection
  - Visual Grounding
  - training-free
  - Agentic Reasoning
  - Chain-of-Thought
  - Open-Vocabulary Detection
date: 2026-05-08
content_hash: 2bc17bc5275f5460
---

# Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.19516](https://arxiv.org/abs/2511.19516)
**Code**: [https://github.com/loiqy/GroundingAgent](https://github.com/loiqy/GroundingAgent)
**Area**: Object Detection
**Keywords**: Visual Grounding, training-free, Agentic Reasoning, Chain-of-Thought, Open-Vocabulary Detection

## TL;DR
This paper proposes GroundingAgent, a visual grounding framework that requires no task-specific fine-tuning. By composing pretrained open-vocabulary detectors (YOLO World), an MLLM (Llama-3.2-11B-Vision), and an LLM (DeepSeek-V3) into a structured iterative reasoning pipeline, the method achieves a zero-shot average accuracy of 65.1% on RefCOCO/+/g, substantially outperforming prior zero-shot approaches.

## Background & Motivation
Visual Grounding (VG) requires associating natural language descriptions with specific regions in an image, and serves as a fundamental task for vision-language interaction. Limitations of existing methods:

1. **Heavy data dependency**: Conventional VG methods rely on large quantities of precisely annotated image-text region pairs for training or fine-tuning, with annotation costs far exceeding those of image-level captions.
2. **Limited generalization**: Models trained on predefined categories fail to transfer well to open-world scenarios and exhibit degraded performance on novel or out-of-distribution concepts.
3. **Weak localization in MLLMs**: Although MLLMs such as GPT-4o excel at captioning and VQA, their ability to directly predict bounding boxes is poor (as illustrated in Figure 1, where GPT-4o incorrectly selects the pitcher rather than the target).
4. **Detectors such as Grounding DINO provide accurate localization but lack deep semantic reasoning**, particularly for complex queries involving spatial relations or attribute descriptions.

Core insight: detectors excel at "where" (localization), LLMs excel at "what" (semantic reasoning), and MLLMs excel at "what is seen" (visual description). Connecting these complementary capabilities through an agent pipeline enables training-free visual grounding.

## Core Problem
How can high-quality visual grounding be achieved **without using any VG task annotations**, relying solely on the compositional capabilities of pretrained models? The core challenges are: (1) how to generate candidate regions with high recall from a text query; and (2) how to accurately select the candidate that best matches the query from the generated set.

## Method

### Overall Architecture
GroundingAgent is a two-stage pipeline: **Candidate Generation → Candidate Selection**.

- **Input**: Image $I$ and natural language query $Q$
- **Output**: Predicted bounding box $\mathbf{b}_{pred}$

Overall procedure:
1. The MLLM generates a global description $C(I)$ of the image.
2. The LLM infers a candidate concept set $\mathcal{C}$ (e.g., "the white chair by the fireplace" → chair, fireplace, furniture, etc.) from the query $Q$ and the global description.
3. An open-vocabulary detector performs detection for each concept, producing a set of candidate bounding boxes.
4. NMS deduplication and area-based sorting are applied; boxes smaller than 2.5% of the image area are filtered out, retaining the top-10 candidates.
5. The MLLM generates fine-grained semantic descriptions for each candidate region.
6. The LLM combines global context, the query, and the candidate descriptions to perform Chain-of-Thought (CoT) reasoning, assessing each candidate's match to the query in turn.

### Key Designs

1. **Global Caption-Guided Concept Generation**: Rather than feeding the query text directly to the detector, the MLLM first generates a global image description, which is then concatenated with the query and passed to the LLM to extract multiple relevant noun concepts. Experiments demonstrate that incorporating the global caption significantly improves candidate recall compared to using only the query. This design is effective because it fuses two information sources—"what the query refers to" and "what is present in the image"—preventing the LLM from generating unconstrained candidate concepts.

2. **Visually Prompted Region Description**: Each candidate region is highlighted with a red bounding box and presented to the MLLM with the background blurred, producing a region-level description. This visual prompting strategy directs the MLLM's attention to the specific region rather than the entire image. Region descriptions capture visual attributes and contextual cues.

3. **CoT-Driven Agentic Selection**: The LLM does not make a one-shot decision; instead, it generates intermediate reasoning steps (averaging 3.4 steps), progressively analyzing the semantic and spatial relationship between each candidate and the query. The output is a binary judgment $r_i \in \{0, 1\}$, constrained to be one-hot (RefCOCO requires selecting a single target). The reasoning process is interpretable, with the LLM explicitly stating the rationale for accepting or rejecting each candidate.

4. **Self-Consistency Sampling** (Appendix C): For each candidate region, the MLLM samples 5 descriptions, which are then aggregated by the LLM into a consensus description. This step improves RefCOCO-val accuracy from 67.1% to 68.5%, confirming that caption noise is a primary error source.

### Loss & Training
No training is required; the method is entirely training-free. All modules (detector, MLLM, LLM) perform inference using pretrained weights directly.

## Key Experimental Results

### Main Results: Zero-Shot REC Performance Comparison

| Method | Zero-shot | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | RefCOCOg test | **Avg** |
|--------|-----------|-------------|---------------|---------------|--------------|----------------|----------------|--------------|---------------|---------|
| Pseudo-Q | ✗ | 56.0 | 58.3 | 54.1 | 38.9 | 45.1 | 32.1 | 49.8 | 47.4 | 47.7 |
| Grounding-DINO | ✗ | 50.4 | 57.2 | 43.2 | 51.4 | 57.6 | 45.8 | 67.5 | 67.1 | 55.0 |
| Kosmos-2 | ✗ | 52.3 | 57.4 | 47.3 | 45.5 | 50.7 | 42.2 | 60.6 | 61.7 | 52.2 |
| **GroundingAgent (Ours)** | ✓ | **67.1** | **73.3** | **60.1** | **62.4** | **67.6** | **53.8** | **67.9** | **68.8** | **65.1** |

### Detector Recall in the Candidate Generation Stage

| Detector | RefCOCO val | RefCOCO testA | RefCOCO testB | Avg |
|----------|-------------|---------------|---------------|-----|
| APE | 98.6 | 98.7 | 97.9 | 98.3 |
| GroundingDINO | 98.3 | 98.7 | 97.6 | 98.2 |
| OWL-ViT | 95.7 | 96.3 | 92.6 | 94.9 |
| YOLO-World | 94.4 | 96.7 | 91.1 | 93.8 |

### Caption Substitution Experiment (Upper-Bound Analysis)

| Strategy | Avg |
|----------|-----|
| MLLM-generated Caption | 65.1 |
| Query + Caption | 85.0 |
| Original Query Directly | **90.6** |

### LLM Ablation

| LLM | RefCOCO testA | RefCOCO testB |
|-----|---------------|---------------|
| DeepSeek-V3 | 73.3 | 60.1 |
| DeepSeek-R1 | **75.9** | **60.3** |
| Llama3.1-8B | 55.0 | 44.0 |
| DeepSeek-R1-Llama-8B | 59.7 | 47.7 |
| Qwen2.5-7B | 52.0 | 41.6 |

### Segmentation Extension (+SAM)

| Dataset | mIoU |
|---------|------|
| RefCOCO-val | 57.3 |
| RefCOCO+-val | 51.2 |
| RefCOCOg-val | 56.5 |

### Ablation Study
- **MLLM caption quality is the primary bottleneck**: Replacing captions with the original query raises accuracy from 65.1% to 90.6%, approaching supervised SOTA (Qwen2.5-VL at 90.3%). This indicates that the LLM's reasoning capability is not the limiting factor; rather, performance loss stems from hallucinations and imprecision in MLLM-generated descriptions.
- **Reasoning capability matters more than model scale**: DeepSeek-R1-Llama-8B outperforms vanilla Llama3.1-8B by 4–5 points with identical parameter counts, demonstrating that reasoning-oriented training yields substantial gains.
- **Global caption is critical for candidate generation**: Removing the caption leads to a significant drop in recall.
- **Self-consistency sampling is effective**: Five-sample aggregation improves val accuracy by 1.4% (67.1→68.5).
- **Good stability**: Standard deviation across three independent runs is approximately 0.55%.
- **Low rejection rate**: 0.73–1.69% on RefCOCO+, indicating that the agent rarely "gives up."

## Highlights & Insights
- **Clear system design philosophy**: VG is decomposed into four steps—concept generation, candidate detection, region description, and reasoning-based selection—with each step handled by the pretrained model best suited to that sub-task. The modular design allows individual components to be swapped or upgraded at any time.
- **Compelling upper-bound analysis**: The caption-to-query substitution experiment precisely identifies the performance bottleneck as MLLM description quality rather than LLM reasoning ability. This oracle analysis methodology is instructive for future work.
- **Strong interpretability**: Each reasoning step is visualizable, and the rationale for accepting or rejecting every candidate is explicitly stated, which is highly valuable in agent systems.
- **Truly zero-shot**: Unlike methods such as REG that implicitly train on synthetic annotations, GroundingAgent does not use any grounding annotations whatsoever.

## Limitations & Future Work
- **Substantial performance gap relative to supervised methods**: 65.1% vs. supervised SOTA of 84–90%, which may be insufficient for practical deployment.
- **Low inference efficiency**: Each image requires the MLLM to produce a global description plus $N$ region descriptions, followed by multi-step LLM reasoning, resulting in high latency and cost. The paper does not report inference times, which is a notable omission.
- **MLLM hallucination is not fundamentally addressed**: Self-consistency is a patch rather than a solution; the paper also acknowledges this as the core bottleneck.
- **Limited small-object detection capability**: Filtering boxes smaller than 2.5% of the image area is disadvantageous for small-target localization.
- **Evaluation limited to RefCOCO variants**: Validation on more diverse benchmarks (e.g., Flickr30K Entities, PhraseCut) is absent.
- **Dependence on closed-source or large LLMs**: The default configuration uses DeepSeek-V3; substituting smaller models (Llama-8B, Qwen-7B) causes substantial performance degradation, limiting practical deployment.

## Related Work & Insights
1. **vs. Grounding DINO**: Grounding DINO is an end-to-end detector that achieves a zero-shot average of 55.0% on RefCOCO without grounding annotation training. GroundingAgent surpasses this by 10 points, though Grounding DINO is orders of magnitude faster at inference.
2. **vs. ReCLIP / VGDiffZero**: These are prior training-free VG methods. GroundingAgent achieves 12–27% improvements by incorporating LLM reasoning; the fundamental difference lies in the use of an agentic reasoning pipeline rather than simple similarity matching.
3. **vs. GPT-4o direct localization**: GPT-4o exhibits very low accuracy when directly outputting bounding boxes (as shown in Figure 1). The core argument of GroundingAgent is that MLLMs should not be tasked with direct localization; instead, the detector should handle localization while the LLM handles reasoning-based selection.

**Broader Takeaways**:
- **Agent pipeline design paradigm**: This paper demonstrates an effective paradigm for leveraging tool composition in multimodal tasks—rather than requiring a single model to handle everything, each model is assigned the sub-task it handles best. This approach is transferable to other tasks requiring joint localization and understanding (e.g., visual question answering with grounding, embodied navigation).
- **Caption quality as a common bottleneck in VLM agent systems**: The oracle experiment reveals that LLM reasoning is already sufficiently capable (reaching 90%+ upon caption substitution); the bottleneck lies in the inaccuracy of MLLM visual descriptions. This finding serves as a warning for all VLM agent systems that follow a "describe-then-reason" paradigm.
- **Reasoning capability > parameter scale**: Results from the DeepSeek-R1 series demonstrate that enhancing structured reasoning through reasoning-oriented training (e.g., reward modeling, GRPO) is more effective than simply scaling model size, consistent with the broader trend toward reasoning models.

## Rating
- Novelty: ⭐⭐⭐ — The idea of connecting existing components (detector + MLLM + LLM) via an agent pipeline is not entirely new, but the specific design choices for VG and the upper-bound analysis are valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations are comprehensive, covering oracle analysis, LLM ablation, detector ablation, self-consistency, segmentation extension, and failure analysis; however, inference efficiency data and evaluation on additional datasets are missing.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, problem formulation is precise, and experimental analysis is substantive, though some content is redundantly presented across the main text and appendix.
- Value: ⭐⭐⭐ — The method serves as a meaningful training-free baseline; however, the absolute performance of 65.1% and the high inference cost limit its practical utility. The most significant contribution is the empirical insight that caption quality is the primary bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](../../CVPR2026/object_detection/pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)
- [\[CVPR 2026\] CompAgent: An Agentic Framework for Visual Compliance Verification](../../CVPR2026/object_detection/compagent_an_agentic_framework_for_visual_compliance_verification.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[ICLR 2026\] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](../../ICLR2026/object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)

</div>

<!-- RELATED:END -->
