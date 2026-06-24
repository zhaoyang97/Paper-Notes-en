---
title: >-
  [Paper Note] BlueGlass: A Framework for Composite AI Safety
description: >-
  [ICML 2025][Object Detection][Composite AI Safety] This work proposes BlueGlass, a composite AI safety framework that integrates three safety analysis tools—distributed evaluation, approximation probes, and sparse autoencoders—via a unified infrastructure to systematically analyze the capability boundaries, layer dynamics, and internal concept representations of Vision-Language Models (VLMs) in object detection tasks.
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Composite AI Safety"
  - "Vision-Language Models"
  - "Sparse Autoencoders"
  - "Linear Probes"
date: 2026-05-08
content_hash: d230ff75b8acf8fd
---

# BlueGlass: A Framework for Composite AI Safety

**Conference**: ICML 2025  
**arXiv**: [2507.10106](https://arxiv.org/abs/2507.10106)  
**Code**: [https://github.com/](https://github.com/) (open-source framework)  
**Area**: AI Safety / Object Detection / Interpretability  
**Keywords**: Composite AI Safety, Vision-Language Models, Sparse Autoencoders, Linear Probes, Object Detection

## TL;DR

This work proposes BlueGlass, a composite AI safety framework that integrates three safety analysis tools—distributed evaluation, approximation probes, and sparse autoencoders—via a unified infrastructure to systematically analyze the capability boundaries, layer dynamics, and internal concept representations of Vision-Language Models (VLMs) in object detection tasks.

## Background & Motivation

**Background**: Safety assurance of AI systems is a critical step prior to deployment. Existing safety tools cover multiple dimensions including adversarial robustness evaluation, mechanistic interpretability, and data attribution. However, each tool can only address a specific aspect of model safety.

**Limitations of Prior Work**: Existing safety tools are independent of each other and lack a unified interface, making it difficult to combine them into a complete safety evaluation pipeline. The adaptation cost of different tools for various model architectures is high, and the lack of standardized feature management leads to severely fragmented cross-method safety analysis workflows.

**Key Challenge**: A single safety tool cannot provide sufficient safety guarantees, whereas multi-tool integration lacks unified infrastructure support—this is the core challenge of "composite AI safety".

**Goal**: (1) How to design a unified framework that supports the integration of multiple safety tools? (2) Where are the zero-shot capability boundaries of VLMs in object detection? (3) What are the similarities and differences in internal representation learning mechanisms between VLMs and vision-only detectors? (4) What spurious correlations have VLMs learned?

**Key Insight**: The authors take a dual track of framework engineering and empirical analysis—first building a unified infrastructure, and then demonstrating the framework's value through three case studies, selecting VLM + object detection, a safety-critical scenario (autonomous driving, robotics), as the object of analysis.

**Core Idea**: By building a unified composite safety framework, three categories of safety tools—distributed evaluation, probing analysis, and concept decomposition—are organically combined to perform multi-level safety audits on VLMs.

## Method

### Overall Architecture

The BlueGlass framework comprises three core levels of abstraction: (1) **Foundations Layer**—provides foundational modules such as model interfaces, dataset management, evaluators, and runners, uniformly encapsulating various sources like HuggingFace, Detectron2, and MMDetection; (2) **Feature Tools Layer**—manages the intercepting, recording, patching, aligning, and storing of internal model representations, utilizing the Apache Arrow/Parquet format for efficient feature storage; (3) **Safety Tools Layer**—builds diverse safety analysis tools on top of the first two layers.

### Key Designs

1. **Interceptor-Recorder-Patcher Feature Management System**:

    - **Function**: Uniformly manage the capture and modification of intermediate layer features across different model architectures.
    - **Mechanism**: The Interceptor wraps the target model and defines access points, supporting both manual insertion and automatic hook modes; the Recorder captures intermediate representations at these access points; the Patcher supports interventional experiments such as activation patching and model steering.
    - **Design Motivation**: Existing tools like TransformerLens are bound to specific architectures, and NNsight has limited support for complex codebases; thus, an architecture-agnostic unified interface is needed.

2. **Approximation Probes**:

    - **Function**: Quantify the task-relevant information content of representations in each layer by training lightweight linear probes after each decoder layer.
    - **Mechanism**: At each decoder layer $\ell$, a classification probe (trained with cross-entropy loss to predict classes) and a localization probe (trained with Smooth L1 loss to predict bounding boxes) are trained. The key innovation is to have the probes approximate the model's own raw predictions rather than the ground truth, thereby tracking the "taskification" trajectory of features. Probe accuracy is measured using $AP_{50}$.
    - **Design Motivation**: Traditional probes trained with ground truth only demonstrate whether information exists. In contrast, approximation probes reveal how the model itself constructs predictions layer by layer, enabling the detection of phase transition phenomena.

3. **Sparse Autoencoder Concept Decomposition**:

    - **Function**: Decompose the internal representations of the VLM into interpretable sparse concept vectors.
    - **Mechanism**: Train a TopK SAE on the residual stream of the Grounding DINO decoder. The encoder $E$ maps $d$-dimensional features to an $m = d \times e$-dimensional latent space and selects TopK activations to maintain sparsity, while the decoder $D$ reconstructs the inputs. Dataset attribution (selecting the maximum activating samples for each sparse unit) is used for concept discovery.
    - **Design Motivation**: Resolve the polysemanticity issue and discover interpretable concepts and spurious correlations learned by the model, such as a "hand" unit falsely triggering predictions of "knives" or "cell phones".

### Loss & Training

The SAE is optimized using a weighted sum of the reconstruction loss $L_{\text{recon}}$ and the auxiliary loss $L_{\text{aux}}$. Distributed evaluation adopts the standard COCO evaluation protocol, consistently utilizing AP and AR metrics. The open-ended text outputs of the VLM are converted into standardized predictions for dataset-specific categories via a custom mapping pipeline.

## Key Experimental Results

### Main Results

| Model | Type | FunnyBirds AP/AR | COCO AP/AR | LVIS AP/AR | BDD100k AP/AR |
|------|------|-----------------|------------|------------|--------------|
| YOLOv8 | Discriminative | 85.2/95.4 | 24.9/42.6 | 7.1/14.1 | 8.8/19.4 |
| Grounding DINO | Contrastive | 87.3/91.2 | 48.5/77.2 | 14.2/53.2 | 23.8/59.4 |
| GenerateU | Generative | 65.1/92.9 | 32.1/66.1 | **25.5**/40.7 | 13.1/37.7 |
| Florence 2 Large | Generative | **87.9**/93.0 | 40.1/55.2 | 2.3/0.3 | 11.7/25.5 |
| Gemini 2.0 Flash | Generative | 32.2/50.0 | 19.9/32.8 | 4.9/7.2 | 0.9/3.4 |
| DINO (SFT) | Discriminative | **99.6/99.9** | **58.3/78.6** | 20.8/38.7 | **35.9**/55.6 |

### Ablation Study

| Analysis Dimension | Key Findings | Description |
|---------|---------|------|
| Zero-shot vs. Fine-tuning | SFT DINO comprehensively outperforms VLMs | Supervised fine-tuning is still 2-3 times stronger than zero-shot VLMs in dense detection. |
| Open-Vocabulary Detection | GenerateU is optimal (LVIS AP=25.5) | The combination of detection network and language model balances geometric priors and semantic reasoning. |
| Phase Transition | Both VLM and vision-only models exhibit phase transitions in middle decoder layers | Demonstrates that the VLM repurposes the hierarchical feature learning mechanisms of vision detectors. |
| SAE Concept Decomposition | Discovered spurious correlation units such as "hand" | Hand features falsely trigger predictions of knives and cell phones, exposing safety hazards. |

### Key Findings

- Fine-tuned DINO outperforms all VLMs on all closed-set datasets but lags behind GenerateU on open-vocabulary detection, indicating a clear trade-off between the generalization capability of VLMs and the precision of supervised models.
- The decoder layers of both the VLM and the vision-only detector exhibit a three-stage representation evolution of "extraction-recomposition-refinement", showing that the phase transition phenomenon is an inherent property of hierarchical representation learning.
- Spurious correlations revealed by the SAE (e.g., hand $\rightarrow$ knife/cell phone) have direct safety implications for deployment, suggesting that models may rely on contextual shortcuts rather than robust object features.

## Highlights & Insights

- **Approximation probing** is an ingenious design—by training the probe to learn the model's own predictions rather than the ground truth, it can track the "taskification" process of representations. This method is transferable to any scenario requiring the analysis of hierarchical representations.
- The finding of the **three-stage representation evolution** (extraction $\rightarrow$ recomposition $\rightarrow$ refinement) is highly insightful, indicating that the cross-modal emergent capabilities of VLMs primarily stem from modal alignment rather than fundamentally different learning mechanisms.
- The methodological path of discovering spurious correlations using SAE (train SAE $\rightarrow$ dataset attribution $\rightarrow$ manual interpretation) is clear and constitutes a reusable model auditing pipeline.

## Limitations & Future Work

- The framework is currently validated only on object detection tasks and has not yet covered other VLM tasks such as segmentation and VQA.
- The concept discovery of the SAE still relies on manual interpretation of maximum activating samples, yielding limited automation.
- The distributed evaluation introduction of the VLM evaluation pipeline (open-ended output $\rightarrow$ standardized prediction) adds extra engineering complexity, and its design choices may affect evaluation fairness.
- The scalability of the framework on larger scale models (such as GPT-4V) has not been explored.

## Related Work & Insights

- **vs. TransformerLens/NNsight**: These tools target specific architectures or impose usage restrictions, whereas the Interceptor design of BlueGlass is more general.
- **vs. Independent SAE Works (Anthropic, etc.)**: BlueGlass integrates SAEs into a broader safety workflow rather than employing them in isolation.
- **vs. Traditional Linear Probes**: Approximation probes learn the model's predictions instead of the ground truth, offering a different yet complementary perspective.

## Rating

- Novelty: ⭐⭐⭐ The framework integration idea is valuable, though individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ The three case studies comprehensively cover evaluation, mechanistic analysis, and concept discovery.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, though the density of symbols and formulas is high.
- Value: ⭐⭐⭐⭐ Provides a practical, open-source infrastructure for AI safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Few-Shot Learner Generalizes Across AI-Generated Image Detection](few-shot_learner_generalizes_across_ai-generated_image_detection.md)
- [\[ICML 2025\] Open-Det: An Efficient Learning Framework for Open-Ended Detection](open-det_an_efficient_learning_framework_for_open-ended_detection.md)
- [\[NeurIPS 2025\] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](../../NeurIPS2025/object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[ICLR 2026\] Self-Guided Low Light Object Detection Framework](../../ICLR2026/object_detection/self-guided_low_light_object_detection_framework.md)

</div>

<!-- RELATED:END -->
