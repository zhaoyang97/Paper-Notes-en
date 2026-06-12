---
title: >-
  [Paper Note] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][visual discrepancy perception] This paper proposes OddGridBench to evaluate the fine-grained visual discrepancy sensitivity of MLLMs (i.e.…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "visual discrepancy perception"
  - "benchmark"
  - "GRPO"
  - "curriculum learning"
  - "fine-grained perception"
date: 2026-05-08
content_hash: 151ca3cbe8b0a09d
---

# OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.09326](https://arxiv.org/abs/2603.09326)  
**Code**: [https://wwwtttjjj.github.io/OddGridBench/](https://wwwtttjjj.github.io/OddGridBench/)  
**Area**: Multimodal VLM
**Keywords**: visual discrepancy perception, benchmark, GRPO, curriculum learning, fine-grained perception

## TL;DR
This paper proposes OddGridBench to evaluate the fine-grained visual discrepancy sensitivity of MLLMs (i.e., identifying the element in a grid that differs from others in color, size, rotation, or position). All evaluated MLLMs fall far below human performance. To address this gap, the authors propose OddGrid-GRPO, which combines curriculum learning with a distance-aware reward to significantly improve visual discrimination ability.

## Background & Motivation
**Background**: MLLMs have demonstrated strong performance on high-level semantic understanding tasks (e.g., image captioning, VQA, mathematical reasoning), yet low-level visual perception remains underexplored both in evaluation and training.

**Limitations of Prior Work**: Existing benchmarks primarily focus on high-level semantic reasoning, neglecting a fundamental capability of the human visual system — fine-grained visual discrepancy sensitivity (Just Noticeable Difference / Pop-out Effect). This low-level perceptual ability is a prerequisite for spatial reasoning and object understanding.

**Key Challenge**: No systematic, controllable benchmark exists to quantitatively assess MLLM sensitivity across different perceptual dimensions (color, size, rotation, position), nor are there targeted training methods to close this gap.

**Goal**: (1) Construct a controllable fine-grained visual discrepancy perception benchmark; (2) Expose systematic failure patterns of MLLMs on this task; (3) Propose a training method to improve perceptual sensitivity.

**Key Insight**: Inspired by the Odd-One-Out paradigm from cognitive psychology, the paper constructs parameterized grid images that precisely quantify the degree of discrepancy.

**Core Idea**: A benchmark is built using parameterized grid images in which one element differs subtly from others in color, size, rotation, or position. OddGrid-GRPO, combining curriculum learning and distance-aware rewards, is then applied to improve MLLM perceptual sensitivity.

## Method

### Overall Architecture
The framework consists of two components: OddGridBench and OddGrid-GRPO. The benchmark component generates parameterized grid images for evaluation; the training component employs curriculum-learning-guided GRPO with distance-aware rewards to improve the model.

### Key Designs

1. **OddGridBench Data Generation**:

    - SVG icons are collected from IconFont and Material Design Icons, categorized into artifacts, natural objects, and symbols.
    - Grid layout: 5–9 rows and columns, with each icon sized at 60–80px.
    - Four discrepancy dimensions: color (CIE-Lab $\Delta E \in [5,20]$), size (85%–115%), rotation ($\pm 5°$ to $\pm 25°$), and position (offset 5%–12%).
    - Supports single-attribute and multi-attribute combinations (2-Type, 3-Type, 4-Type), comprising 1,400 test + 400 validation + 30,000 training samples.

2. **OddGrid-GRPO Curriculum Learning**:

    - A continuous difficulty score is computed for each sample based on grid size, number of attributes, and perturbation magnitude.
    - Samples are divided into three subsets: Easy (15K) / Medium (10K) / Hard (5K).
    - Three-stage progressive training proceeds from easy to hard to prevent premature convergence.
    - Design Motivation: Directly applying RL on hard samples is unstable; progressive learning mirrors the developmental trajectory of human perceptual learning.

3. **Distance-Aware Reward**:

    - Standard GRPO employs binary rewards (correct/incorrect), which are ill-suited for localization tasks.
    - A continuous reward based on Euclidean distance is designed: $r_d = \max(\exp(-d^2/2\sigma^2) - \beta, 0)$
    - $\sigma$ scales adaptively with grid size; $\beta$ suppresses rewards for distant predictions.
    - Total reward: $r_{overall} = (1-\omega)r_d + \omega r_f$, where $r_f$ is the format reward.

### Loss & Training
Reinforcement learning based on GRPO, combined with the curriculum learning schedule and distance-aware reward function described above.

## Key Experimental Results

### Main Results

| Model | Color | Size | Rotation | Position | Total |
|------|-------|------|----------|----------|-------|
| Random | 2.00 | 2.00 | 2.00 | 2.00 | 2.43 |
| Qwen3-VL-32B | 85.00 | 39.50 | 52.50 | 39.00 | 68.07 |
| Gemini-2.5-Pro | 82.50 | 9.50 | 26.00 | 6.50 | 49.29 |
| GPT-5 | 56.50 | 9.50 | 21.00 | 5.00 | 28.93 |
| **Human** | **91.33** | **69.33** | **82.67** | **78.00** | **87.47** |

### Key Findings

| Observation | Description |
|------|------|
| Color dimension is easiest | Most models perform best on color discrepancy, yet still fall well below human levels. |
| Position/size are most difficult | Nearly all models perform near chance on position and size perception. |
| Human vs. best MLLM | Human: 87.47% vs. Qwen3-VL-32B: 68.07%, a gap of nearly 20%. |
| Model scale effect | Larger models in the same series outperform smaller ones, but the improvement is limited. |

### Key Findings
- Color is the most sensitive dimension for MLLMs, while size and position are the weakest, indicating fundamental deficiencies in the visual encoders of MLLMs with respect to spatial-geometric perception.
- Both curriculum learning and the distance-aware reward in OddGrid-GRPO contribute meaningfully; removing either component leads to performance degradation.
- Accuracy increases monotonically with the magnitude of the discrepancy, consistent with psychophysical laws governing human perception.

## Highlights & Insights
- **Parameterized benchmark design**: Analogous to psychophysical experiments, the benchmark precisely controls the discrepancy magnitude along each perceptual dimension, enabling a continuous transition from "imperceptible" to "salient" — a capability absent from conventional benchmarks.
- **Distance-aware reward**: Encoding spatial proximity into the RL reward provides a richer learning signal than binary rewards, and this design is transferable to other VLM tasks requiring spatial localization.
- **Exposing fundamental MLLM weaknesses**: GPT-5 achieves only 5% on position perception — near random chance — demonstrating severe inadequacy of current visual encoders in low-level perception.

## Limitations & Future Work
- The benchmark relies solely on synthetic SVG icons, without addressing fine-grained discrepancy detection in natural images.
- Only single-image scenarios are evaluated; practical applications require discrepancy detection against complex backgrounds.
- The effectiveness of OddGrid-GRPO is validated primarily on this benchmark; transferability to other fine-grained visual tasks remains to be examined.
- The training set (30K samples) is relatively small; scaling up may yield further improvements.

## Related Work & Insights
- **vs. Traditional Odd-One-Out**: Conventional approaches are designed for visual encoders and are not applicable to MLLM architectures; this paper is the first to design a systematic perceptual discrepancy evaluation for MLLMs.
- **vs. GRPO (DeepSeek-V3)**: Standard GRPO uses binary rewards; this paper extends it to a continuous distance-aware reward, providing finer-grained spatial supervision signals.

## Supplementary Analysis

- The three-stage training in OddGrid-GRPO uses sample counts of 15K → 15K (5K easy + 10K medium) → 15K (10K easy/medium + 5K hard), with total training volume fixed at 30K.
- All grid icons are in SVG format, ensuring resolution-independent scaling and rotation.
- On the 4-Type combination task, human accuracy reaches 97.67% while GPT-5 achieves only 46.00% — a gap exceeding 50%, the largest across all conditions.
- The benchmark generation code is open-sourced, allowing free customization of new discrepancy dimensions (e.g., texture, opacity).
- The paper further finds that providing explicit grid labels (LabeledAcc) substantially improves model accuracy, suggesting that the challenge lies not only in visual perception but also in spatial reasoning and index comprehension.

## Rating
- Novelty: ⭐⭐⭐⭐ Cleverly designed benchmark that exposes an important problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation of 19 models with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with well-crafted figures and tables.
- Value: ⭐⭐⭐⭐ Reveals systematic deficiencies in low-level perception of MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)

</div>

<!-- RELATED:END -->
