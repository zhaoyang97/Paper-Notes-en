---
title: >-
  [Paper Note] Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support
description: >-
  [CVPR 2025][Medical Imaging][Surgical Scene Understanding] Surg-R1 proposes a hierarchical reasoning visual-language model (VLM) for surgical scenes. Through a three-level reasoning hierarchy (Perception-Relationship-Context) and a four-stage training pipeline (SFT $\rightarrow$ GRPO $\rightarrow$ self-iteration), trained on the largest surgical CoT dataset containing 320K reasoning pairs, it achieves a 64.9% Arena Score on SurgBench, significantly outperforming Gemini 3.0 Pr…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Surgical Scene Understanding"
  - "Visual-Language Models"
  - "Chain-of-Thought"
  - "Hierarchical Reasoning"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: b5be695ffe41d0a8
---

# Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support

**Conference**: CVPR 2025  
**arXiv**: [2603.12430](https://arxiv.org/abs/2603.12430)  
**Code**: [https://jianjiangkcl.github.io/Surg-R1/](https://jianjiangkcl.github.io/Surg-R1/)  
**Area**: Medical Image  
**Keywords**: Surgical Scene Understanding, Visual-Language Models, Chain-of-Thought, Hierarchical Reasoning, Reinforcement Learning

## TL;DR
Surg-R1 proposes a hierarchical reasoning visual-language model (VLM) for surgical scenes. Through a three-level reasoning hierarchy (Perception-Relationship-Context) and a four-stage training pipeline (SFT $\rightarrow$ GRPO $\rightarrow$ self-iteration), trained on the largest surgical CoT dataset containing 320K reasoning pairs, it achieves a 64.9% Arena Score on SurgBench, significantly outperforming Gemini 3.0 Pro (46.1%) and GPT-5.1 (37.9%).

## Background & Motivation

**Background**: Surgical scene understanding is a core task in computer-assisted surgery, encompassing multiple subtasks such as instrument localization, action triplet recognition, surgical phase identification, and Critical View of Safety (CVS) assessment. Recently, visual-language models (VLMs) have demonstrated strong cross-task capabilities in medical imaging.

**Limitations of Prior Work**: Existing surgical VLMs only generate final predictions and lack explicit reasoning chains, making it impossible for surgeons to verify the models' decision-making logic. On the other hand, general reasoning models such as GPT-5.1 and Gemini 3.0 Pro possess CoT capabilities but lack surgical domain knowledge, resulting in poor performance on compositional surgical tasks—such as triplet tasks that require simultaneous identification of instrument types, operating actions, and target tissues.

**Key Challenge**: There is a gap between interpretable reasoning capabilities and clinical domain expertise. General models have reasoning but lack domain knowledge, while specialized models have domain knowledge but lack reasoning chains.

**Goal**: Build a surgical VLM that possesses both domain expertise and the ability to generate verifiable reasoning chains, achieving high accuracy and interpretability simultaneously across multiple surgical understanding tasks.

**Key Insight**: The authors observe that surgical scene understanding is inherently a hierarchical compositional problem—first needing to perceive "what is seen" (instruments, tissues), then understand "their relations" (who is manipulating what), and finally perform contextual reasoning (which surgical phase is current, whether it is safe). This shallow-to-deep reasoning hierarchy is naturally suited for CoT decomposition.

**Core Idea**: Decompose the surgical interpretation task through a three-level reasoning hierarchy (perceptual grounding $\rightarrow$ relational understanding $\rightarrow$ contextual reasoning), and progressively enhance the reasoning capability using a four-stage training pipeline (SFT $\rightarrow$ GRPO $\rightarrow$ iterative self-improvement).

## Method

### Overall Architecture
Surg-R1 uses a visual-language model as its backbone, taking surgical video frame images and task instructions as inputs, and outputting answers containing hierarchical reasoning chains. The core of the system consists of three components: (1) three-level reasoning hierarchy definition, (2) large-scale surgical CoT dataset construction, and (3) a four-stage progressive training pipeline.

### Key Designs

1. **Three-Level Reasoning Hierarchy**:

    - **Function**: Structurally decompose the surgical scene reasoning process into three progressive levels.
    - **Mechanism**: **Level 1 Perceptual Grounding**—identifies fundamental visual elements present in the image, such as instruments and tissues, answering "what is in the scene"; **Level 2 Relational Understanding**—infers spatial and functional relationships among elements, e.g., "bipolar forceps are grasping the cystic duct"; **Level 3 Contextual Reasoning**—synthesizes temporal and clinical knowledge for high-level judgments, such as phase recognition and safety assessment. The output of each reasoning level serves as the input foundation for the next level.
    - **Design Motivation**: Direct end-to-end prediction lacks structure, leading to results that are both hard to interpret and insufficiently accurate. Hierarchical decomposition allows each reasoning step to be independently verified by surgeons.

2. **Large-Scale Surgical CoT Dataset (320K Reasoning Pairs)**:

    - **Function**: Provide supervision signals for training reasoning chains.
    - **Mechanism**: Collect task annotations from existing surgical datasets, and utilize expert knowledge and model distillation to generate corresponding three-level reasoning chain annotations. Each sample contains an input image, a task question, and a reasoning chain expanded across three levels along with the final answer. In total, 320,000 reasoning pairs are constructed, covering various tasks such as instrument localization, triplet recognition, phase recognition, action recognition, and CVS assessment.
    - **Design Motivation**: Previously, no large-scale surgical reasoning chain dataset existed. Without reasoning chain supervision, VLMs cannot learn to execute structured reasoning in surgical scenarios.

3. **Four-Stage Progressive Training Pipeline**:

    - **Function**: Progressively transition from supervised learning to autonomous reasoning optimization.
    - **Mechanism**: **Stage 1 Supervised Fine-Tuning (SFT)**—performs standard instruction tuning on the surgical CoT dataset to teach the model to output reasoning chain formats; **Stage 2 Group Relative Policy Optimization (GRPO)**—inspired by DeepSeek-R1, samples multiple reasoning chains for the same question, uses accuracy and reasoning quality as reward signals, and performs policy optimization via intra-group relative ranking without training a separate reward model; **Stage 3-4 Iterative Self-Improvement**—supplements training data with high-quality reasoning chains generated by the current model, followed by a new round of SFT and GRPO, iteratively boosting performance.
    - **Design Motivation**: Pure SFT is prone to overfitting reasoning chain templates. GRPO introduces exploration and reward mechanisms to help the model learn more flexible reasoning strategies, and iterative self-improvement further expands high-quality reasoning data.

### Loss & Training
During the SFT stage, standard autoregressive cross-entropy loss is used. In the GRPO stage, $K$ reasoning outputs are sampled for the same input to calculate reward scores (including answer correctness rewards and reasoning format rewards) for each output, which are then optimized through group-normalized policy gradients. The reward design balances task accuracy and the logical integrity of the reasoning chain.

## Key Experimental Results

### Main Results

The model is evaluated on SurgBench (6 public benchmarks + 6 multi-center external validation datasets), covering five task categories: instrument localization, triplet recognition, phase recognition, action recognition, and CVS assessment.

| Model | Arena Score (Public) | External Validation | Type |
|------|-------------------|----------|------|
| **Surg-R1** | **64.9%** | **Best** | Surgery-specific VLM |
| Gemini 3.0 Pro | 46.1% | - | General Reasoning |
| GPT-5.1 | 37.9% | - | General Reasoning |
| Strongest surgical baseline | ~49.7% | Surg-R1 is 15.2pp higher | Surgery-specific VLM |

### Ablation Study

| Configuration | Arena Score | Description |
|------|-----------|------|
| Full Surg-R1 (4-stage) | 64.9% | Complete four-stage training |
| SFT only (Stage 1) | ~55% | Supervised fine-tuning only, limited reasoning chain quality |
| SFT + GRPO (Stage 2) | ~60% | Significant improvement after adding policy optimization |
| w/o Hierarchical Reasoning | ~52% | Direct answer prediction without reasoning chain |
| w/o Self-Iteration | ~60% | Missing self-improvement loop |

### Key Findings
- The GRPO stage contributes the most, elevating reasoning quality from mimetic SFT to autonomous, exploratory reasoning.
- The hierarchical reasoning structure not only improves interpretability but also brings a significant boost in accuracy (approximately +12pp compared to direct answers).
- External validation proves generalization capability: the model maintains leadership across 5 institutions and varying surgical video acquisition conditions.
- General reasoning models (GPT-5.1, Gemini 3.0 Pro) perform far below specialized models in surgical scenarios, indicating that domain knowledge is irreplaceable.

## Highlights & Insights
- **The combination of hierarchical reasoning and GRPO is highly ingenious**: The hierarchical structure provides clear intermediate states for the reasoning chain, allowing GRPO to offer more fine-grained reward signals based on these intermediate states, forming a positive coupling.
- **The construction of the 320K surgical CoT dataset is a foundational contribution**: In domain-specific VLMs, dataset construction is often more critical than the model architecture. This dataset can serve as a fundamental resource for subsequent surgical reasoning research.
- **The concept of iterative self-improvement is highly transferable**: Using correct reasoning chains generated by the current model as a "bootstrapping" strategy to augment new training data is applicable to any domain where reasoning chain quality can be evaluated.

## Limitations & Future Work
- Currently, the model only processes single-frame static images and does not utilize the temporal information of surgical videos; phase transitions and action recognition in real surgeries require temporal reasoning.
- Generating reasoning chains increases inference latency, which may not be suitable for real-time surgical navigation scenarios.
- Despite its scale, the dataset mainly derives from laparoscopic surgeries; generalization to other surgery types (such as robotic surgery or neurosurgery) remains to be validated.
- The reward design of GRPO relies on binary correctness judgments and lacks fine-grained rewards evaluating the rationality of the reasoning process.

## Related Work & Insights
- **vs SurgVLP / Surgical-VLM**: Traditional surgical VLMs directly perform classification/detection without reasoning chains. Surg-R1 transparentizes the prediction process via CoT, and the reasoning chain itself becomes a driver for performance improvement.
- **vs DeepSeek-R1**: Surg-R1 borrows the GRPO training paradigm but adapts it to multimodal surgical scenarios, introducing a domain-specific three-level reasoning hierarchy.
- **vs GPT-5.1 / Gemini 3.0 Pro**: Demonstrates that general large models still have obvious shortcomings in specialized domain reasoning, proving that domain adaptation is indispensable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While the combination of the three-level reasoning hierarchy and GRPO is pioneering in surgical VLMs, the individual methodological components (SFT/GRPO/iteration) have prior precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive task coverage across 6 public benchmarks and 6 multi-center external validation datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with sufficient motivation.
- **Value**: ⭐⭐⭐⭐ Represents the first large-scale surgical reasoning VLM, contributing the dataset, method, and benchmarks, yielding practical value for the surgical AI field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](../../CVPR2026/medical_imaging/lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[ICML 2026\] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](../../ICML2026/medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)
- [\[CVPR 2025\] UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis](ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)

</div>

<!-- RELATED:END -->
