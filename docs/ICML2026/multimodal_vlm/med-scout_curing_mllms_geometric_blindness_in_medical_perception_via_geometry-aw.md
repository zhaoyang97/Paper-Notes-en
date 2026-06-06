---
title: >-
  [Paper Note] Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training
description: >-
  [ICML 2026][Multimodal VLM][Medical MLLM] Med-Scout defines the systematic defect where "medical MLLMs fail to follow image geometric constraints during lesion localization" as "geometric blindness." It utilizes three ge…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Medical MLLM"
  - "GRPO"
  - "Geometry-aware"
  - "Proxy tasks"
  - "Dense reward"
date: 2026-05-08
content_hash: 2c8c59f98945778a
---

# Med-Scout: Curing MLLMs' Geometric Blindness in Medical Perception via Geometry-Aware RL Post-Training

**Conference**: ICML 2026  
**arXiv**: [2601.23220](https://arxiv.org/abs/2601.23220)  
**Code**: https://github.com/HKUSTGZ-ML4Health-Lab/Med-Scout  
**Area**: Medical Imaging  
**Keywords**: Medical MLLM, GRPO, Geometry-aware, Proxy tasks, Dense reward

## TL;DR
Med-Scout defines the systematic defect where "medical MLLMs fail to follow image geometric constraints during lesion localization" as "geometric blindness." It utilizes three geometric proxy tasks (multi-scale localization / topological jigsaw / anomaly consistency) that do not require expert annotation, combined with Dense Geometric Reward (DGR) for post-training under GRPO. It also releases Med-Scout-Bench to quantify geometric blindness, achieving consistent improvements across four backbones and eight medical benchmarks, with open-source models even surpassing GPT-5 / Gemini-3-Flash.

## Background & Motivation
**Background**: Medical MLLMs represented by LLaVA-Med, HuatuoGPT-Vision, MedGemma, and Lingshu have approached clinical language styles in terminology generation and symptom description. The mainstream post-training paradigm remains SFT or RL with simple rewards, targeting "semantic alignment"—making generated reports match labels textually.

**Limitations of Prior Work**: The authors conducted three sets of pilot experiments on Qwen3-VL-8B-Instruct and Lingshu-7B, revealing systematic geometric blindness in existing medical MLLMs: (1) Scale blindness: Identical lesions can be identified in local crops but fail in 20%+ of cases when placed back into the global view; (2) Topological blindness: After rotating an image by 180°, 80% of models fail to update spatial descriptions such as "upper/lower"; (3) Anomaly blindness: Using cut-paste to insert abnormal regions into the center of an image, 90%+ of models fail to perceive them and still output standard reports. CoT prompting provides almost no relief, proving this is a perception-level defect rather than a prompt engineering issue.

**Key Challenge**: There is a structural misalignment between the requirement for "geometric faithfulness" in clinical AI and the goal of "semantic fluency"—MLE-like likelihood maximization has no mechanism to penalize errors in position, scale, or missed anomalies; a model receives full marks as long as it mentions the correct terms. Meanwhile, general-domain visual jigsaw/grounding proxies (Jigsaw-R1, ViCrit, Euclid, GeoPQA) are not targeted towards medical anatomical structures, modality specificity, or fine-grained anomalies.

**Goal**: (1) Construct geometric proxy tasks that automatically generate verifiable supervision signals from unlabeled medical images; (2) Provide RL with a dense signal that does not collapse due to sparse binary rewards; (3) Develop a quantifiable and reproducible benchmark for the "geometric blindness" defect.

**Key Insight**: Medical images inherently contain verifiable geometric facts—different crops of the same image must have consistent IoU, a $2\times 2$ jigsaw grid has a unique correct sequence, and the location of a cut-paste intrusion point is completely known. These geometric constraints possess absolute objective verifiability compared to "semantic correctness" and are naturally suited for RL like GRPO based on relative intra-group comparison.

**Core Idea**: Reformulate "teaching MLLMs to perceive medical images" as "teaching MLLMs to self-verify image geometric constraints without labels." Use three categories of proxy tasks + Dense Geometric Reward (DGR) for post-training under GRPO to build a foundation for geometric perception before generalizing to downstream medical VQA and report generation.

## Method

### Overall Architecture
Med-Scout is a data-centric RL post-training framework. Given an unlabeled medical image $I\in\mathbb{R}^{H\times W}$, it is first converted into three types of geometric proxy VQA tasks: scale task $\mathcal{T}_{\text{scale}}$, topological task $\mathcal{T}_{\text{topo}}$, and anomaly task $\mathcal{T}_{\text{anom}}$, unified into open-set VQA formats. Training utilizes GRPO (KL coefficient $\beta=0.04$, group size $G=8$, cosine + warmup scheduler, AdamW, lr $1\times 10^{-6}$, total 7,200 steps). The total reward for each sample is $\mathcal{R}=\mathcal{R}_{\text{acc}}+\mathcal{R}_{\text{fmt}}+\mathbb{I}_{\text{CoT}}\cdot\mathcal{R}_{\text{reason}}$, where $\mathcal{R}_{\text{acc}}$ is the dense geometric reward designed per task type. The entire process requires no expert annotation; all supervision signals are derived from the geometric facts of the images themselves.

### Key Designs

1.  **Three Geometric Proxy Tasks**:
    *   **Function**: Decomposes abstract "geometric perception" into three categories of VQA that can be automatically constructed on medical images with uniquely verifiable answers.
    *   **Mechanism**: (a) **Hierarchical Scale Localization** simulates the clinical "magnifying glass" diagnostic process, simultaneously cropping $N=3$ patches from the original image belonging to Level-1 (20% area) and Level-2 (6.25% area) scale tiers, with center coordinates restricted to the normalized $[0.2, 0.8]$ interval to avoid background noise; the model must output the scale tier and normalized box $b=(x_1, y_1, x_2, y_2)$ for each patch. (b) **Topological Jigsaw Reconstruction** cuts the image into a $2\times 2$ grid and randomly permutes it $\sigma$ to form $I_{\text{shuffled}}$, requiring the model to provide the "original index sequence read in left-to-right, top-to-bottom order," forcing bidirectional spatial reasoning. (c) **Anomaly Consistency Detection** replaces one block in the central region of a $4\times 4$ grid with a reference patch $I_{\text{ref}}$ (adjacent slices for CT/MRI, top-1 similar images retrieved by BiomedCLIP for X-ray); the model must output the grid index of the anomalous patch. All tasks are unified into open-set VQA streams with optional Direct or CoT modes.
    *   **Design Motivation**: In medical scenarios, "geometry" is not a single concept—scale corresponds to "local vs. global consistency," topology corresponds to "anatomical position invariance," and anomalies correspond to "pixel-level structural consistency." These three cover the blindness points exposed in the pilot. This "problem-oriented geometric decomposition" ensures each reward corresponds to a specific clinical capability, avoiding the issue where general tasks like Jigsaw-R1 "solve the puzzle but fail to learn medical essentials."

2.  **Dense Geometric Reward (DGR) Integrated into GRPO**:
    *   **Function**: Replaces traditional sparse binary feedback in RL with continuous scores based on the "degree of geometric deviation," allowing gradient signals to guide the model even when errors are small.
    *   **Mechanism**: (a) Scale task reward has two parts: value estimation $\mathcal{R}_{\text{val}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\hat y_i=y_i^*)$ and box IoU $\mathcal{R}_{\text{box}}=\frac{1}{N}\sum_{i=1}^N\text{IoU}(\hat b_i,b_i^*)$, normalized by $N$ to maintain magnitude. (b) Topological task uses element-wise alignment $\mathcal{R}_{\text{topo}}=\frac{1}{N}\sum_{i=1}^N\mathbb{I}(\hat s_i=s_i^*)$, rewarding based on "how many patches are correct" even if the sequence is not entirely perfect. (c) Anomaly task converts flatten index $k$ back to 2D coordinates $(u,v)=(\lfloor k/4\rfloor,k\bmod 4)$, with reward $\mathcal{R}_{\text{anom}}=\exp(-\sqrt{(\hat u-u^*)^2+(\hat v-v^*)^2}/\tau)$; closer distances yield higher rewards. (d) General format reward $\mathcal{R}_{\text{fmt}}=\frac{0.5}{N}\sum_{i=1}^N\mathbb{I}(\hat a_i\in\Phi_{\text{regex}})$ checks output schemas at the item level. (e) CoT mode adds a structural reward $\mathcal{R}_{\text{reason}}=0.5$ if the output follows the `<think>...<answer>...` template. The upper limit for a perfect CoT score is $\mathcal{R}=2.0$.
    *   **Design Motivation**: GRPO estimates relative advantage within a group. Higher reward variance and more stable intra-group ranking lead to more informative update directions. With sparse 0/1 rewards, the probability of "all wrong/all right" within a group is high, causing the advantage to degenerate to 0. DGR differentiates "nearly correct" samples, ensuring informative intra-group ranking for stable and fast convergence. Using natural geometric metrics like distance/IoU/element-wise hits as rewards is more reliable than training an additional reward model and avoids reward hacking risks.

3.  **Med-Scout-Bench: A Quantifiable Medical Benchmark for Geometric Blindness**:
    *   **Function**: Transforms "geometric blindness" from a qualitative description into a reproducibility-scored benchmark covering CT, MRI, and X-ray modalities.
    *   **Mechanism**: A synthetic pool of 108,000 VQA items is created, with CT/MRI using TotalSegmentor for whole-body anatomical coverage and X-ray using MIMIC-CXR. From this, 10,800 items (10%) are strictly balanced and sampled as the benchmark; the remaining 97,200 items serve as the training set, strictly disjoint from the bench. Evaluation is unified as open-set VQA without options, using LLM-as-a-Judge (Gemini-3-Flash) for semantic correctness to avoid the fragility of hard string matching. Scoring directly uses the DGR defined in Section 4.2, aligning the "training objective" with "evaluation metrics."
    *   **Design Motivation**: Previous medical MLLM evaluations used either semantic questions like VQA-RAD (unable to locate geometric errors) or segmentation/detection (outside the MLLM interface). Med-Scout-Bench preserves geometric scoring capabilities within a VQA interface, filling the gap for studying medical MLLM geometric capabilities. Strong positive correlations are shown between bench scores and downstream tasks like PMC-VQA/OmniMedVQA/MedXpertQA (Figure 5 right), serving as a reliable indicator of "generalized medical perception."

### Loss & Training
GRPO Optimization: Group size $G=8$, KL coefficient $\beta=0.04$, global batch 192, cosine LR decay, warmup 0.01, AdamW, peak lr $1\times 10^{-6}$; 7,200 steps on 6×NVIDIA RTX PRO 6000. Four backbones: general-purpose Qwen3-VL-4B/8B-Instruct, and medical-specific Lingshu-7B and HuatuoGPT-Vision-7B.

## Key Experimental Results

### Main Results
| Backbone | Med-Scout-Bench Avg | Rad-VQA | VQA-RAD | SLAKE | MIMIC-CXR CIDEr | Meaning |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | 39.7 → **83.6** (+43.9) | 41.6 → 45.3 | 63.2 → 65.8 | 69.6 → 72.0 | 64.8 → 68.1 | General backbone surpasses GPT-5/Gemini-3-Flash |
| Lingshu-7B | 31.9 → **71.9** (+40.0) | 61.2 → 64.0 | 68.9 → 71.0 | 82.8 → 83.0 | 104.9 → 105.2 | Already SOTA still improves |
| HuatuoGPT-Vision-7B | — | 48.8 → 52.1 | 67.0 → 70.1 | 67.8 → 71.0 | 75.6 → 79.0 | PMC-VQA gain 2.9 |
| Qwen3-VL-4B-Instruct | — | 41.5 → 45.7 | 59.9 → 62.9 | 73.4 → 75.6 | 60.9 → 65.2 | Significant gain for small model |

Proprietary model upper bounds: GPT-5 Rad-VQA 59.1 / VQA-RAD 66.4, Gemini-3-Flash 60.7 / 70.2. Open-source Lingshu-7B+Med-Scout achieves 71.0 on VQA-RAD, exceeding Gemini-3-Flash.

### Ablation Study (Comparison with existing visual proxy tasks, DGR disabled, sparse rewards used)
| Method | Med | Geo | Rad-VQA Avg | Gen. Avg | Meaning |
|---|---|---|---|---|---|
| Qwen3-VL-4B baseline | - | - | 58.3 | 38.4 | Starting point |
| + Jigsaw-R1 | ✗ | ✓ | 57.6 (−0.7) | 38.3 (−0.1) | General jigsaw drops performance |
| + ViCrit | ✗ | ✗ | 57.7 (−0.6) | 38.4 (=0.0) | General grounding yields no gain |
| + **Med-Scout (sparse)** | ✓ | ✓ | **60.8 (+2.5)** | **40.2 (+1.8)** | Stable gain even without DGR |
| Qwen3-VL-8B + Med-Scout (sparse) | ✓ | ✓ | 60.4 (+2.3) | 40.5 (+1.4) | Similar trend for 8B |

### Key Findings
- The Bench improvement of over +40 percentage points suggests that the geometric perception gap in existing medical MLLMs is severely underestimated; simultaneously, the Bench score is strongly positively correlated with the average accuracy across six external benchmarks, validating geometric perception as a fundamental capability for generalized medical perception.
- Improvements in general-purpose backbones (Qwen3-VL-4B/8B) are systematically larger than those in medical-specific models, indicating that strong vision-language bases can better absorb geometric supervision signals; this contrasts with the SFT era experience that "specialized models are necessarily stronger."
- Direct Mode and Reasoning Mode show similar performance (Figure 4). The authors hypothesize this is partly because the structural reward $\mathcal{R}_{\text{reason}}$ only constrains the `<think>...<answer>...` template rather than the reasoning logic itself, indicating room for refinement in CoT supervision.
- Data Scaling: Performance on Bench increases monotonically from 20% to 100% of training data without saturation, suggesting further development potential for geometric proxy task supervision.

## Highlights & Insights
- "Decomposing geometric blindness into scale/topological/anomaly blindness" and assigning a proxy task to each corresponds clearly; this methodology of "conducting diagnostic pilot experiments to find specific blindness points before designing targeted proxy tasks" is transferable to any perception-critical multi-modal task (e.g., autonomous driving, industrial inspection).
- The core insight of DGR is that "intra-group RL needs reward variance." Treating traditional IoU, Euclidean distance, and element-wise hit rates directly as continuous rewards avoids the trouble of training reward models and prevents reward hacking; this paradigm of "geometric metrics as rewards" can be reused for other space-sensitive visual tasks.
- Using BiomedCLIP for cut-paste reference retrieval in X-rays ensures "anomaly patches" have radiologically plausible textures and contrast, avoiding shortcut learning from simple noise blocks—a small engineering detail that significantly affects proxy task validity.

## Limitations & Future Work
- The three proxy tasks focus on "geometry" and do not model modality-related physical quantities (e.g., CT HU value calibration, MRI multi-sequence comparison); geometry is only one dimension of medical image constraints, and future work needs to add dimensions like "physical consistency" and "temporal sequence consistency."
- Med-Scout-Bench uses LLM-as-a-Judge (Gemini-3-Flash) as a semantic referee, carrying risks of judges' bias; particularly, the use of Gemini-3-Flash as an evaluation referee during training may introduce potential circular bias.
- Reasoning Mode provides almost no gains but has significantly higher training costs, suggesting that "CoT template reward + RL" designs for medical geometric tasks need rethinking—rewards provide no constraint on the reasoning process itself, causing CoT to degenerate into output padding.
- Evaluation cannot be applied to proprietary backbones (GPT-5/Gemini-3-Flash) for Med-Scout (no weights access), so "open-source surpassing proprietary" strictly applies only on Med-Scout-Bench; on general medical VQA, proprietary models still hold a slight lead (e.g., SLAKE).

## Related Work & Insights
- **vs Jigsaw-R1 / Visual Jigsaw**: Similar in using grid reordering as a proxy; Med-Scout differs by combining it with multi-scale localization and anomaly detection specifically for medical anatomy and fine-grained anomalies, whereas Jigsaw-R1 only uses $2\times 2$ jigsaws for general spatial reasoning.
- **vs ViCrit**: ViCrit uses executable programs for verification but lacks a medical context; Med-Scout's IoU / element-wise hits / Euclidean distances are objective metrics inherent in medical imaging, requiring no external programs.
- **vs Euclid / GeoPQA / GeoGPT4V**: These works inject geometric priors (points, lines, angles) into general MLLMs; Med-Scout extends "geometry" to scale, topology, and anomaly dimensions relevant to clinical practice.
- **vs LLaVA-Med / HuatuoGPT-Vision / MedGemma / Lingshu**: These focus on SFT/semantic alignment; Med-Scout is a post-training framework that can be layered on top of them, as demonstrated by the gains across four different backbones.

## Rating
- Novelty: ⭐⭐⭐⭐ The "geometric blindness" problem is clearly defined, and the three proxy tasks are well-designed; however, the overall paradigm of GRPO + proxy tasks has predecessors like Jigsaw-R1/ViCrit in general domains. Innovation lies more in "medicalization and densification."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 backbones × 8 benchmarks + data scaling + Direct/CoT comparison + comparison with 3 general proxy tasks + comparison with 4 proprietary/open-source models; coverage is very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Pilot experiments make the motivation extremely persuasive, the method section's formulas and algorithm descriptions are clear, and the appendix provides proxy examples and evaluation details.
- Value: ⭐⭐⭐⭐⭐ Simultaneously contributes methodology, benchmarks, and four aligned model weights; Med-Scout-Bench is poised to become a de facto standard for measuring medical MLLM geometric perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[ICML 2026\] FreeRet: MLLMs as Training-Free Retrievers](freeret_mllms_as_training-free_retrievers.md)
- [\[NeurIPS 2025\] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](../../NeurIPS2025/multimodal_vlm/first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](../../AAAI2026/multimodal_vlm/revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)

</div>

<!-- RELATED:END -->
