---
title: >-
  [Paper Note] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images
description: >-
  [ACL2026][Multimodal VLM][MMRK Images] This paper proposes the STAR data engine and a two-stage training framework for Multi-modal Relational Knowledge (MMRK) images…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "MMRK Images"
  - "STAR Task"
  - "Multi-modal Knowledge Graphs"
  - "KGRPO"
  - "Synthetic Instruction Data"
date: 2026-05-08
content_hash: c0ed4039b5cbb1ab
---

# Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images

**Conference**: ACL2026  
**arXiv**: [2510.21828](https://arxiv.org/abs/2510.21828)  
**Code**: https://github.com/zjukg/STAR  
**Area**: Multi-modal VLM / Knowledge Graphs / Structured Reasoning  
**Keywords**: MMRK Images, STAR Task, Multi-modal Knowledge Graphs, KGRPO, Synthetic Instruction Data  

## TL;DR
This paper proposes the STAR data engine and a two-stage training framework for Multi-modal Relational Knowledge (MMRK) images, significantly enhancing MLLM capabilities in understanding and reasoning over abstract structured knowledge images using STAR-64K synthetic data, CoT annotations, and knowledge-aware KGRPO.

## Background & Motivation
**Background**: Multi-modal Large Language Models (MLLMs) have demonstrated proficiency in processing natural images, charts, OCR, and visual math problems. Many benchmarks evaluate the comprehension of "abstract visual information," such as diagrams, schematics, mathematical figures, and structured documents.

**Limitations of Prior Work**: MMRK images remain insufficiently studied. These images are not typical photographs but organize entities, text descriptions, images, and relational edges into node-edge structures, requiring models to simultaneously recognize entities, understand edge types, and track graph structures for reasoning.

**Key Challenge**: While MLLMs' visual capabilities are increasing, they are primarily trained on natural scenes and general charts. The critical semantics of MMRK images derive from human-defined high-order relations. Models fail in counting, error detection, entity completion, and relational reasoning if they "see nodes" without processing the relationships as a knowledge structure.

**Goal**: The authors aim to bridge two gaps: the lack of large-scale, high-quality MMRK instruction data and the absence of specialized training and evaluation protocols for STAR capabilities.

**Key Insight**: Existing multi-modal knowledge graphs (MMKGs) are converted into visual subgraphs, followed by the generation of eight task categories with reliable CoT. This avoids manual labeling costs while binding ground-truth graph answers, reasoning paths, and visual presentations.

**Core Idea**: Automate the synthesis of STAR-64K using MMKGs, followed by MLLM training via SFT and preference/RL optimization. Specifically, KGRPO provides additional rewards for knowledge correctness in CoT to reduce hallucinations during structural reasoning.

## Method
The contribution is a full stack: data engine, training protocols, evaluation tasks, RL strategies, and systematic experiments. The significance lies in transforming "abstract structured visual knowledge" into a scalable, trainable, and evaluable task family.

### Overall Architecture
Input data originates from three public MMKGs: VisualSem, FB15K-237, and MKG-Y.

Each knowledge graph is composed of sets of entities, relations, triplets, entity images, and entity text descriptions.

The data engine extracts subgraphs and visualizes entity images and text along with relational edges to form MMRK images.

Eight STAR tasks are generated: Entity Counting, Relation Counting, Image Entity Counting, Triplet Counting, Subgraph Description, Error Detection, Entity Reasoning, and Relation Reasoning.

For tasks requiring reasoning paths, the authors utilize the accurate subgraph text (available before visualization) as prompts for a strong LLM to generate reliable thought processes and answers, rather than forcing a weak MLLM to generate them from the image.

Training proceeds in two stages: Stage 1 utilizes STAR-64K for Supervised Fine-Tuning (SFT) to establish base STAR capabilities; Stage 2 applies preference optimization or RL on failed samples.

### Key Designs
1.  **STAR Data Engine**:
    - **Function**: Converts MMKG entities, relations, images, and text into trainable multi-modal instruction data.
    - **Mechanism**: Extracts subgraphs from VisualSem, FB15K-237, and MKG-Y, renders them as MMRK images using images, text, and edges, and generates tasks. It uses original structural information as prompts for CoT to ensure reasoning traces align with ground-truth triplets.
    - **Design Motivation**: Manual labeling is slow and lacks structural guarantees. Using KGs as a source of truth provides aligned samples of images, structures, answers, and reasoning logic.

2.  **Two-stage STAR Enhancement**:
    - **Function**: Teaches base task formats followed by targeted optimization on difficult failed samples.
    - **Mechanism**: Stage 1 uses SFT to maximize answer probability given an image and question. Stage 2 follows two paths: preference optimization (DPO/ORPO/SimPO) using gold answers vs. model errors, or GRPO/KGRPO using sampled results and reward functions.
    - **Design Motivation**: SFT is vital for capability injection but only fits data on average. Preference/RL optimization provides stronger correction signals for hallucinations and complex graph reasoning errors.

3.  **Knowledge-aware KGRPO**:
    - **Function**: Explicitly rewards the factual correctness of knowledge within the CoT, beyond just the final answer reward in GRPO.
    - **Mechanism**: Adds a knowledge-informed reward to standard GRPO, using gold knowledge and a CoT judge to verify if entities, relations, and triplets in the reasoning process match the graph structure.
    - **Design Motivation**: Failures in MMRK reasoning often stem from fabricating relations or misreading nodes in the CoT rather than simple calculation errors. KGRPO introduces structural knowledge constraints into the training objective.

### Loss & Training
The SFT stage employs next-token prediction to train the MLLM to generate answers and CoTs conditioned on images and questions.

Stage 1 training lasts 3 epochs using LoRA, with a maximum sequence length of 8192, BF16, AdamW, and a cosine scheduler.

Stage 2 preference data is derived from failed Stage 1 training instances: correct answers serve as positive samples, while erroneous generations serve as negative ones.

KGRPO adopts the group relative advantage concept from GRPO, but the reward function incorporates both final answer quality and factual consistency of the CoT.

Evaluation uses Qwen2.5-VL-72B as a judge to score CoTs or unstructured outputs. This strategy ties visual understanding, structural consistency, and reasoning quality together.

## Key Experimental Results

### Main Results
The main results indicate that two-stage training significantly improves the STAR performance of Qwen2.5-VL-3B/7B, with KGRPO being the strongest in Stage 2.

| Model / Setting | Task#1 ACC | Task#2 ACC | Task#3 ACC | Task#5 Score | Task#8 ACC | AVG |
|-------------|------------|------------|------------|--------------|------------|-----|
| GPT-4v | 37.75 | 41.25 | 14.00 | 59.25 | 39.13 | 33.11 |
| GPT-4o-mini | 67.50 | 72.25 | 29.88 | 69.13 | 23.00 | 40.72 |
| Qwen2.5-VL-3B Zero-shot | 18.25 | 20.13 | 3.50 | 57.71 | 38.25 | 25.56 |
| Qwen2.5-VL-3B S1 Full | 42.75 | 67.00 | 57.13 | 59.94 | 56.00 | 53.24 |
| Qwen2.5-VL-3B S2 KGRPO | 75.00 | 85.38 | 68.63 | 71.51 | 68.57 | 63.64 |
| Qwen2.5-VL-7B Zero-shot | 6.13 | 12.25 | 0.13 | 68.62 | 42.88 | 21.24 |
| Qwen2.5-VL-7B S1 Full | 64.88 | 92.75 | 71.37 | 75.71 | 71.52 | 66.98 |
| Qwen2.5-VL-7B S2 KGRPO | 79.88 | 94.88 | 79.50 | 77.19 | 74.48 | 73.06 |

A key observation is that specifically trained 3B/7B models can outperform stronger closed-source models' zero-shot performance on STAR, suggesting the gap lies in data and protocols rather than just scale.

### Ablation Study
The contribution of modalities was verified, showing that both entity images and text in MMRK images are important, with text often having a greater influence.

| Backbone / Configuration | Task#1 | Task#2 | Task#3 | Task#4 | Task#5 | Task#6 | Task#7 | Task#8 |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|
| Qwen2.5-VL-7B w/o ent. images | 55.50 | 75.88 | 48.62 | 26.63 | 67.99 | 32.00 | 52.63 | 65.75 |
| Qwen2.5-VL-7B w/o ent. texts | 59.13 | 74.62 | 47.88 | 25.37 | 67.90 | 34.87 | 41.50 | 68.12 |
| Qwen2.5-VL-7B full dataset | 64.88 | 92.75 | 71.37 | 27.62 | 75.71 | 55.87 | 67.50 | 80.13 |

Comparison of training settings shows that multi-task mixing generally outperforms single-task training. Removing CoT prompts led to performance drops across five backbones, confirming STAR requires explicit structured thinking.

| Configuration | Key Metric | Description |
|------|----------|------|
| S1 Single-task | Qwen2.5-VL-7B AVG 66.06 | Improves target tasks but shows weak cross-task transfer |
| S1 Full STAR-64K | Qwen2.5-VL-7B AVG 66.98 | Mixed tasks provide more stable structural capabilities |
| S2 DPO | Qwen2.5-VL-7B AVG 68.84 | Preference optimization improves difficult samples |
| S2 GRPO | Qwen2.5-VL-7B AVG 69.91 | RL further enhances reasoning performance |
| S2 KGRPO | Qwen2.5-VL-7B AVG 73.06 | Knowledge rewards yield the strongest average effect |

### Key Findings
- Existing MLLMs lack zero-shot capability for MMRK images; they can recognize visual elements but fail at relational reasoning.
- SFT provides the largest gain, highlighting the importance of the STAR-64K dataset; Stage 2 KGRPO further reduces CoT hallucinations.
- Multi-task training is more transferable than single-task training, especially for complex reasoning.
- Entity text contribution is generally larger than images, but removing either weakens performance, confirming the multi-modal nature of MMRK semantics.
- Task#1 (entity identification) and Task#4 (complex counting) show unique scaling patterns and do not improve purely linearly with data volume.

## Highlights & Insights
- Converting MMKGs from "KG completion sources" to "abstract visual reasoning benchmarks" is a valuable shift in perspective.
- The STAR data engine's strength lies in verifiable answers and CoT generation from subgraphs, controlling hallucinations at the source compared to LLMs describing images blindly.
- KGRPO suggests that for structured knowledge tasks, RL rewards should verify intermediate reasoning steps against ground-truth entities and relations.
- Small models outperforming GPT-4o zero-shot via specialized training suggests that many "abstract reasoning capabilities" result from sufficient training distribution coverage rather than mysterious emergence.

## Limitations & Future Work
- Data sources are limited to general encyclopedic MMKGs; coverage of specialized (scientific, medical) KGs is lacking.
- Tasks are currently template-based; real-world needs might involve open-ended tasks like path explanation or counterfactual editing.
- KGRPO experiments were restricted to models under 8B due to compute; scalability to larger models needs validation.
- CoT judge relies on strong MLLM scoring, which may miss fine-grained knowledge errors.
- Image rendering (layout, density, occlusion) might impact difficulty; layout robustness should be included in future evaluations.

## Related Work & Insights
- **vs M3STR**: While M3STR evaluates structured knowledge understanding, this work provides a data engine and training framework to actively improve these capabilities.
- **vs MM-Instruct**: Focuses specifically on relational knowledge images with higher answer verifiability.
- **vs ChartQA / MathVista / MMMU**: Unlike charts or math figures, STAR errors are closer to KG reasoning failures than visual perception failures.
- **vs Standard GRPO**: KGRPO addresses relational hallucinations that standard GRPO might ignore while optimizing solely for the final label.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of STAR data engine on MMRK images and KGRPO is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 8 MLLMs and multiple strategies, though RL on larger scales is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Solid structure, but high reading cost due to numerous task IDs and metrics.
- Value: ⭐⭐⭐⭐⭐ Direct reference value for MMKG, VLM evaluation, and structured reasoning training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[CVPR 2026\] Relational Visual Similarity](../../CVPR2026/multimodal_vlm/relational_visual_similarity.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](../../AAAI2026/multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)

</div>

<!-- RELATED:END -->
