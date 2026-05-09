---
title: >-
  [Paper Note] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method
description: >-
  [ICLR 2026][Object Detection][Visual grounded reasoning] This paper proposes TreeBench (the first traceable visual reasoning benchmark comprising 405 highly challenging VQA pairs, on which OpenAI-o3 achieves only 54.87%) and TreeVGR (a training paradigm that jointly supervises grounding and reasoning via dual IoU reward-based reinforcement learning). A 7B model achieves gains of +16.8 on V\*Bench, +12.6 on MME-RealWorld, and +13.4 on TreeBench, demonstrating that traceability is a key driver of visual reasoning advancement.
tags:
  - ICLR 2026
  - Object Detection
  - Visual grounded reasoning
  - traceable evidence
  - second-order reasoning
  - TreeBench
  - reinforcement learning
  - Dual IoU
date: 2026-05-08
content_hash: 4c69ee55d12df062
---

# Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method

**Conference**: ICLR 2026
**arXiv**: [2507.07999](https://arxiv.org/abs/2507.07999)
**Code**: [GitHub](https://github.com/Haochen-Wang409/TreeVGR)
**Area**: Object Detection
**Keywords**: Visual grounded reasoning, traceable evidence, second-order reasoning, TreeBench, reinforcement learning, Dual IoU

## TL;DR

This paper proposes TreeBench (the first traceable visual reasoning benchmark comprising 405 highly challenging VQA pairs, on which OpenAI-o3 achieves only 54.87%) and TreeVGR (a training paradigm that jointly supervises grounding and reasoning via dual IoU reward-based reinforcement learning). A 7B model achieves gains of +16.8 on V\*Bench, +12.6 on MME-RealWorld, and +13.4 on TreeBench, demonstrating that traceability is a key driver of visual reasoning advancement.

## Background & Motivation

**Background**: OpenAI-o3 has pioneered the paradigm of "thinking with images"—dynamically referencing and zooming into task-relevant visual regions during reasoning—showing potential beyond purely text-based reasoning. However, no existing benchmark comprehensively evaluates this capability.

**Limitations of Prior Work**:
1. Classical benchmarks such as POPE, MMBench, and SEED-Bench overlook fine-grained grounding and verifiable reasoning chains.
2. V\*Bench supports only simple spatial queries (e.g., "Is A to the left of B?") and risks data leakage due to its reliance on COCO images.
3. MME-RealWorld and HR-Bench accept high-resolution inputs but lack traceable evidence and complex reasoning.
4. Existing RL training methods (e.g., DeepEyes, Pixel-Reasoner) supervise only the final answer, leaving intermediate grounding steps unsupervised.

**Key Challenge**: No benchmark simultaneously satisfies three critical requirements—focused visual perception (identifying subtle objects in dense scenes), traceable evidence (assessing grounding quality at each step of the reasoning chain), and second-order reasoning (object interaction and spatial hierarchy reasoning beyond simple localization). On the training side, existing methods cannot quantify the actual contribution of grounding within a "ground-then-answer" framework.

**Goal**: A two-pronged approach is adopted: TreeBench establishes an evaluation standard, and TreeVGR establishes a training methodology. Together, they advance both the assessment and improvement of "thinking with images" capabilities.

## Method

### Overall Architecture

**TreeBench Construction Pipeline**: 1K high-quality images are sampled from SA-1B (prioritizing dense-object scenes) → annotated by 8 LMM experts → 3-stage quality control → 405 challenging VQA pairs (with bounding box annotations for target instances).

**TreeVGR Training Pipeline**: Cold-start SFT initialization → reinforcement learning post-training with traceable evidence.

### Key Design 1: Three Evaluation Principles of TreeBench

**1) Focused Visual Perception**: All questions focus on extremely small targets in complex real-world scenes—target instances occupy on average only **3.05%** of the image area. Models are required to identify subtle targets via detailed, precise, and unambiguous textual descriptions.

**2) Traceable Evidence**: Evaluation covers not only final answer accuracy but also the quality (mIoU) of bounding boxes generated within the reasoning chain. By comparing predicted boxes against ground-truth boxes, error sources can be precisely diagnosed—distinguishing comprehension failures from grounding failures.

**3) Second-Order Reasoning**: Beyond simple "what/where" queries, the benchmark encompasses 5 perception task types (attribute / material / physical state / object retrieval / OCR) and 5 reasoning task types (perspective transformation / ordering / contact-occlusion / spatial containment / comparison). Perspective transformation ("From person A's viewpoint, in which direction is object B?") constitutes the most challenging category.

### Key Design 2: Dual IoU Reward Mechanism in TreeVGR

The total reward in TreeVGR comprises three components:

$$R = R_{\text{acc}} + R_{\text{format}} + R_{\text{IoU}}$$

The dual IoU reward $R_{\text{IoU}}$ is the core innovation, simultaneously optimizing recall and precision:

**Recall term** (each GT box is matched by at least one predicted box):

$$R_{\text{IoU}}^{\text{R}} = \frac{1}{M} \sum_{k=1}^{M} \max_i \text{IoU}(\hat{b}_i, b_k)$$

**Precision term** (each predicted box matches at least one GT box, preventing the model from generating excessive boxes):

$$R_{\text{IoU}}^{\text{P}} = \frac{1}{N} \sum_{i=1}^{N} \max_k \text{IoU}(b_k, \hat{b}_i)$$

$$R_{\text{IoU}} = \frac{1}{2}(R_{\text{IoU}}^{\text{R}} + R_{\text{IoU}}^{\text{P}})$$

This bidirectional constraint addresses the reward hacking problem wherein a unidirectional recall reward incentivizes the model to exhaustively predict all possible boxes.

### Key Design 3: Cold-Start Initialization

Directly applying RL to visual grounded reasoning is extremely inefficient (DeepEyes requires 32 H100 GPUs for 48 hours). This paper first performs cold-start initialization with carefully constructed SFT data—each sample containing an image, a question, a reasoning trajectory with bounding boxes, and a final answer—ensuring the model has acquired basic "ground-then-reason" capability prior to RL. This initialization strategy substantially reduces the computational cost of RL.

## Key Experimental Results

### Main Results: Per-Category Performance on TreeBench

| Model | Overall | Attr. | Phys. State | Obj. Retrieval | OCR | Persp. Trans. | Ordering | Contact-Occ. | Spatial Cont. | Comparison | mIoU |
|-------|---------|-------|-------------|----------------|-----|---------------|----------|--------------|---------------|------------|------|
| o3-0416 | 54.8 | 69.0 | 69.2 | 65.2 | 68.8 | 79.4 | 22.4 | 38.6 | 61.0 | 86.2 | –† |
| Gemini-2.5-Pro | 54.1 | 51.7 | 61.5 | 56.5 | 75.0 | 83.8 | 20.0 | 36.8 | 65.9 | 86.2 | – |
| Qwen2.5-VL-72B | 42.2 | 65.5 | 69.2 | 56.5 | 56.3 | 48.5 | 11.8 | 33.3 | 51.2 | 72.4 | – |
| Qwen2.5-VL-7B | 37.0 | 55.2 | 53.8 | 56.5 | 62.5 | 27.9 | 20.0 | 35.1 | 39.0 | 44.8 | – |
| DeepEyes-7B | 37.5 | 62.1 | 53.8 | 65.2 | 68.8 | 51.5 | 11.8 | 24.6 | 36.6 | 51.7 | 30.0 |
| Pixel-Reasoner-7B | 39.0 | 58.6 | 61.5 | 65.2 | 50.0 | 48.5 | 14.1 | 31.6 | 39.0 | 44.8 | 35.7 |
| **TreeVGR-7B** | **50.4** | 65.5 | 53.8 | **82.6** | 68.8 | **63.3** | **22.4** | 36.8 | **61.0** | 69.0 | **44.0** |

### Ablation Study: Performance Gains Across Benchmarks

| Benchmark | Qwen2.5-VL-7B (Baseline) | TreeVGR-7B | Gain |
|-----------|--------------------------|------------|------|
| TreeBench Overall | 37.0 | 50.4 | **+13.4** |
| V\*Bench Overall | 74.3 | 91.1 | **+16.8** |
| V\*Bench Attr. | 77.4 | 94.0 | +16.6 |
| V\*Bench Spatial | 69.7 | 87.0 | +17.3 |
| MME-RealWorld-Lite | 42.3 | 54.9 | **+12.6** |
| HR-Bench-4K | 72.1 | 77.1 | +5.0 |
| HR-Bench-8K | 68.8 | 73.1 | +4.3 |

### Key Findings

- **No model exceeds 60% on TreeBench**: Even the strongest model, o3, achieves only 54.87%, confirming the benchmark's genuine difficulty.
- **TreeVGR-7B matches InternVL3-78B**: Joint grounding-reasoning training enables the 7B model to reach the performance level of a general-purpose 78B model.
- **mIoU is highly correlated with final accuracy**: TreeVGR's mIoU of 44.0 substantially outperforms DeepEyes (30.0) and Pixel-Reasoner (35.7), validating the positive contribution of precise grounding to reasoning.
- **Contact-occlusion and ordering are the hardest categories**: All models perform worst on these two types (<25%), reflecting the fundamental difficulty of second-order reasoning.

## Highlights & Insights

- **The striking result of "o3 below 55%"**: Even the most capable multimodal model remains weak at fine-grained visual reasoning—TreeBench exposes genuine capability gaps.
- **Traceability = Verifiability**: Evaluating the grounding evidence at each step of the reasoning chain, rather than only the final answer, yields more reliable and diagnostically informative assessments.
- **Elegant dual IoU reward design**: Bidirectional recall-precision constraints prevent the reward hacking strategy of exhaustively predicting boxes.
- **Efficient cold-start + RL paradigm**: Compared to the pure RL approach of DeepEyes (32×H100, 48 hours), cold-start initialization substantially reduces computational cost.

## Limitations & Future Work

- TreeBench is relatively small in scale (only 405 questions), limiting statistical significance.
- TreeVGR does not perform actual image cropping or re-examination (grounding occurs only in text space), potentially missing visual details.
- The quality of cold-start SFT data directly constrains the upper bound of RL training, and data construction incurs manual annotation costs.
- Training samples for second-order reasoning (perspective transformation / spatial containment) are limited, potentially resulting in insufficient RL training coverage.
- Multi-turn interactive grounded reasoning remains unexplored.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](../../ICCV2025/object_detection/visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[AAAI 2026\] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice](../../AAAI2026/object_detection/an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)
- [\[ACL 2026\] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](../../ACL2026/object_detection/e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)

</div>

<!-- RELATED:END -->
