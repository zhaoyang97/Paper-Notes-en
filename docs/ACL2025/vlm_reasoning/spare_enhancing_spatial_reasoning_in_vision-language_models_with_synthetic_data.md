---
title: >-
  [Paper Note] SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data
description: >-
  [ACL 2025][VLM Reasoning][Spatial Reasoning] This study identifies a severe shortage of spatial relation data in existing VLM datasets (where the top 17% of relations cover over 90% of samples). To address this, the authors propose leveraging LLMs to automatically extract a synthetic spatial reasoning dataset of 455k samples (3.4 million QA pairs) from ultra-detailed image description datasets such as DOCCI, Localized Narratives, and PixMo-Cap. The fine-tuned SpaRE model achi…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Spatial Reasoning"
  - "Synthetic Data"
  - "Ultra-detailed Descriptions"
  - "Visual Question Answering"
  - "VLM Fine-tuning"
date: 2026-05-08
content_hash: 4f22e7f9c7812333
---

# SpaRE: Enhancing Spatial Reasoning in Vision-Language Models with Synthetic Data

**Conference**: ACL 2025  
**arXiv**: [2504.20648](https://arxiv.org/abs/2504.20648)  
**Code**: None  
**Area**: Multimodal VLM / Spatial Reasoning  
**Keywords**: Spatial Reasoning, Synthetic Data, Ultra-detailed Descriptions, Visual Question Answering, VLM Fine-tuning

## TL;DR

This study identifies a severe shortage of spatial relation data in existing VLM datasets (where the top 17% of relations cover over 90% of samples). To address this, the authors propose leveraging LLMs to automatically extract a synthetic spatial reasoning dataset of 455k samples (3.4 million QA pairs) from ultra-detailed image description datasets such as DOCCI, Localized Narratives, and PixMo-Cap. The fine-tuned SpaRE model achieves up to a 49% performance boost on the What's Up benchmark without compromising general vision-language capabilities.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) perform exceptionally well on tasks like image captioning and visual question answering, yet continue to struggle with spatial reasoning. Multiple studies (such as VSR and What's Up) have demonstrated that even state-of-the-art VLMs perform near random-guess levels when determining basic spatial relations (e.g., left/right, above/below).

**Limitations of Prior Work**: Spatial relations are extremely scarce in existing VL datasets. The authors analyzed the SFT dataset used by InternVL2 and found that only 1.44% of VQAv2 and 3.07% of GQA samples involve spatial relations. Furthermore, the distribution is extremely unbalanced: the top 17% of common relations (such as "on", "left", and "under") account for over 90% of the spatial relation samples, leaving numerous relations (such as "facing", "opposite", and "surrounding") severely under-represented.

**Key Challenge**: Prior solutions either rely on synthetic scene images (such as CLEVR and STUPD), which fail to generalize to the real world due to domain gaps from simple geometric shapes, or utilize human-annotated real-world data, which is heavily constrained in scale and diversity (e.g., VSR and What's Up total only about 8K samples).

**Goal**: How to generate large-scale real-world image training data covering a rich set of spatial relation types to enhance the spatial reasoning capabilities of VLMs.

**Key Insight**: It is observed that recently released ultra-detailed image description datasets (like DOCCI, PixMo-Cap, and Localized Narratives) contain rich descriptions of spatial relations, serving as natural sources for spatial reasoning QA pairs.

**Core Idea**: Automatically extract spatial reasoning QA pairs from ultra-detailed descriptions of real-world images using a small LLM as training data, thereby engineering around the domain gap issues of synthetic images.

## Method

### Overall Architecture

The overall process is a three-stage pipeline: (1) filtering descriptions containing spatial information from three ultra-detailed description datasets; (2) generating spatial reasoning QA pairs using Qwen2.5-3B-Instruct; and (3) filtering them through a multi-stage quality assurance process before using them for VLM fine-tuning. The inputs are real-world images with ultra-detailed descriptions, and the output is a training set consisting of 455k samples and 3.4 million QA pairs.

### Key Designs

1. **Data Source Selection & Pre-filtering**:
    - **Function**: Filtering a subset containing spatial information from 1.58 million image-description pairs.
    - **Mechanism**: Three complementary ultra-detailed description datasets are selected: DOCCI (15k images, 136 words/desc, human-annotated high-fidelity descriptions), Localized Narratives (849k images, 42 words/desc, voice transcription combined with mouse trajectories on COCO/Flickr30k etc.), and PixMo-Cap (717k images, 196 words/desc, dense descriptions across 70 topics). Qwen2.5-3B-Instruct is used to classify the spatial relations in the descriptions, filtering out approximately 65% of descriptions that do not contain spatial information.
    - **Design Motivation**: Ultra-detailed descriptions naturally contain explicit formulations of spatial relations between objects, avoiding the domain gap of synthetic images. The three datasets complement each other in description length, style, and image sources.

2. **LLM-driven QA Pair Generation**:
    - **Function**: Automatically extracting diverse spatial reasoning QA pairs from the filtered descriptions.
    - **Mechanism**: A detailed in-context learning (ICL) prompt is constructed to guide Qwen2.5-3B-Instruct in extracting spatial reasoning QA pairs from the descriptions, covering various spatial relationships such as position, orientation, and distance. Structured JSON outputs are generated with $\text{temperature} = 0$, with each description producing multiple QA pairs (averaging ~7.4 pairs). The generation scope is strictly limited to the spatial relationships mentioned in the descriptions to ensure factual grounding.
    - **Design Motivation**: A small LLM (36B parameters) is sufficient to handle the QA extraction task, lowering generation costs, while the structured output facilitates automated processing.

3. **Multi-stage Quality Assurance System**:
    - **Function**: Ensuring the quality and accuracy of the generated QA pairs.
    - **Mechanism**: Five checks are applied in ascending order of computational cost: (1) De-duplication: exact string matching and CLIP semantic similarity (threshold 0.95) are used to detect duplicate questions; (2) Reference checking: filtering out QA pairs that refer to "the description" instead of asking directly about the image; (3) Answer-description consistency: verifying that key answer words exist in the original description; (4) Image-question consistency: CLIPScore (threshold 0.25) is used to check the semantic alignment between the image and the question; (5) Spatial relation verification: confirming that the QA pair indeed involves spatial reasoning. Human evaluation of 400 sampled QA pairs shows an error rate of ~4%.
    - **Design Motivation**: The quality of synthetic data is a critical bottleneck. Multi-tiered filtering ensures accuracy while containing computational costs.

### Loss & Training

The VLMs are fine-tuned using standard cross-entropy loss, computed solely on the text tokens and excluding the visual tokens. The 2B model underwent full-parameter training, while the 7B model was trained using LoRA to save VRAM. Training was conducted in bfloat16 precision, using a linear warmup for the first 1000 steps followed by cosine decay, with gradient clipping set to a maximum norm of 1.0. All configurations were trained on 4×NVIDIA A40 (48GB) GPUs, averaging results over 5 random seeds.

## Key Experimental Results

### Main Results

| Model | VSR | What's Up A | What's Up B | 3DSRBench | RealWorldQA | Spatial Avg. |
|------|-----|------------|------------|-----------|-------------|---------|
| Qwen2VL-2B | 70.3 | 44.6 | 79.1 | 46.5 | 58.6 | 59.8 |
| **SpaRE-2B** | **80.8** | **93.4** | **95.1** | **54.4** | **63.5** | **77.6** |
| Gain | +10.5 | **+48.8** | +16.0 | +7.9 | +4.9 | +17.8 |
| Qwen2VL-7B | 82.3 | 99.5 | 99.3 | 49.2 | 67.7 | 79.2 |
| **SpaRE-7B** | **85.4** | **100.0** | **100.0** | **57.5** | **68.8** | **82.3** |
| GPT-4o | 79.0 | 100.0 | 100.0 | 45.3 | 61.0 | 77.9 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| SpaRE-2B MMMU | 40.0 vs 34.0 (Qwen2VL-2B) | General capabilities improved instead of declining |
| SpaRE-2B MMBench | 71.6 vs 72.0 (Qwen2VL-2B) | General performance remains largely on par |
| SpaRE-7B MMMU | 51.0 vs 51.0 (Qwen2VL-7B) | General performance of the 7B model is fully preserved |
| SpaRE-2B vs InternVL2-2B | 77.6 vs 68.9 Spatial Avg. | Leading comprehensively among models of the same scale |
| SpaRE-7B vs GPT-4o | 82.3 vs 77.9 Spatial Avg. | 7B open-source model outperforms GPT-4o |

### Key Findings

- SpaRE-2B achieves an absolute improvement of 49% ($44.6 \rightarrow 93.4$) on What's Up A, representing the largest single-task gain across all benchmarks.
- Spatial reasoning enhancement does not come at the cost of general capabilities: SpaRE models perform on par with or slightly better than the base models on general benchmarks such as MMMU and MMBench.
- SpaRE-7B outperforms GPT-4o on the spatial reasoning average metric (82.3% vs. 77.9%), proving the efficacy of the synthetic data approach.
- The "benign hallucinations" observed during training (QA pairs related to the image but not specifically about spatial reasoning) were retained, which actually helped maintain general capabilities.

## Highlights & Insights

- Precise problem formulation: The work quantifies the scarcity of spatial relation data (where the top 17% of relations span 90% of samples), establishing a clear objective for the solution.
- Exceptionally simple and effective methodology: Without relying on synthetic image generations, large models, or complex training workflows, the approach achieves substantial improvements solely by extracting QA pairs from existing dense captions.
- Low-cost yet high-impact: Utilizing an LLM with only 3B parameters for QA generation keeps computational expenses low while yielding prominent results.
- The discovery of "benign hallucinations" is an interesting byproduct: QA pairs unrelated to spatial constraints but grounded in the image actually aided the retention of the model's generalization capabilities.

## Limitations & Future Work

- Code and datasets are not yet open-sourced (though the paper states they will be shared in due course).
- Experiments are restricted to 2B and 7B scales; the effectiveness on larger models remains unverified.
- QA generation is bounded by the spatial details already present in the source descriptions, making it unable to cover spatial relationships not explicitly mentioned.
- An estimated 4% error rate in QA pairs implies some noise still exists in the training data.
- The dataset is predominantly in English, leaving multilingual spatial reasoning generalization untested.
- While the long-tail issue of spatial relationship classification is alleviated, it may still reflect partial imbalances.

## Related Work & Insights

- VSR (Liu et al., 2023a) and What's Up (Kamath et al., 2023) are the primary benchmarks for evaluating spatial reasoning.
- The domain gaps associated with datasets like CLEVR (Johnson et al., 2017) and STUPD (Agrawal et al., 2023) are precisely what this study's methodology seeks to overcome.
- The emergence of ultra-detailed description datasets like DOCCI and PixMo-Cap provided the foundation for this work.
- This methodological approach can be extended to other weak areas in VLMs: identifying data scarcity $\rightarrow$ finding information-rich data sources $\rightarrow$ synthesizing targeted training data.

## Rating

- **Novelty**: 7/10 — Clear methodological direction, but the technical contribution is relatively incremental.
- **Technical Depth**: 6/10 — Simple and straightforward approach, lacking deep technical innovation.
- **Experimental Thoroughness**: 8/10 — Evaluated across multiple benchmarks and models with results averaged over 5 seeds ensuring reliability.
- **Writing Quality**: 8/10 — In-depth problem analysis and clear logical exposition.
- **Value**: 8/10 — Simple, effective, easy to replicate, and highly extensible to other scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](../../CVPR2025/vlm_reasoning/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/vlm_reasoning/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ACL 2026\] CAPruner: Conceptual-Adjacent Scene Graph Pruner for Enhancing 3D Spatial Reasoning of Large Language Models](../../ACL2026/vlm_reasoning/capruner_conceptual-adjacent_scene_graph_pruner_for_enhancing_3d_spatial_reasoni.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](../../NeurIPS2025/vlm_reasoning/roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)

</div>

<!-- RELATED:END -->
