---
title: >-
  [Paper Note] IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring
description: >-
  [ICML 2026][Interpretability][Multi-granularity IQA] This paper proposes IQA-Spider, a multi-granularity image quality assessment method that unifies four types of tasks—"global quality description + local quality descri…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Multi-granularity IQA"
  - "LMM"
  - "Pixel-level grounding"
  - "text-to-point"
  - "training-free"
date: 2026-05-08
content_hash: e4e6932c43a0b26f
---

# IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring

**Conference**: ICML 2026  
**arXiv**: [2605.24553](https://arxiv.org/abs/2605.24553)  
**Code**: https://github.com/Helen1p/IQA-Spider.git (Available)  
**Area**: Multimodal VLM / Image Quality Assessment  
**Keywords**: Multi-granularity IQA, LMM, Pixel-level grounding, text-to-point, training-free

## TL;DR
This paper proposes IQA-Spider, a multi-granularity image quality assessment method that unifies four types of tasks—"global quality description + local quality description + pixel-level grounding + region-level referring"—into a single LMM framework. Accompanying this is a 33K-scale multi-task dataset and a training-free text-to-point paradigm that directly maps language model position word logits to SAM point prompts. IQA-Spider comprehensively outperforms existing specialized models like Q-Instruct and Q-Ground on multi-granularity IQA benchmarks.

## Background & Motivation

**Background**: Image Quality Assessment (IQA) based on Large Multimodal Models (LMMs) has developed rapidly, evolving from global scoring (Q-Bench / Q-Align) to quality description and reasoning (Q-Instruct / DepictQA-Wild), and recently to pixel-level grounding (Q-Ground / Grounding-IQA). These have formed relatively independent technical routes.

**Limitations of Prior Work**: Existing methods cover only a single perceptual dimension—either providing whole-image descriptions or performing pixel grounding based on fixed distortion categories. They cannot both "explain what is wrong" and "point to where it is" within the same model. DepictQA-Wild has strong descriptive capabilities but poor localization; Q-Ground can perform grounding but is tied to a narrow distortion vocabulary; additionally, introducing special tokens like `<seg>` can damage the original instruction-following and reasoning capabilities of the LMM.

**Key Challenge**: The existing "language token + special grounding token" coupling paradigm faces a dilemma—either modifying the language space for grounding (at the cost of reasoning ability) or outputting pure text (failing to obtain pixel masks). Furthermore, the data side lacks a unified task definition that covers all four granularities (global, local, grounding, referring) and is capable of scaling up.

**Goal**: (1) Formalize multi-granularity IQA into a four-task system; (2) Create a corresponding dataset using a scalable pipeline; (3) Connect grounding to SAM without compromising the LMM's text reasoning capabilities.

**Key Insight**: The authors observe that when an LMM generates position descriptions ("top/bottom/left/right"), its native token logits already encode spatial distribution probabilities. There is no need to train special grounding tokens; by weighted averaging and regressing these position word logits into coordinates, they can be directly fed into SAM as point prompts.

**Core Idea**: Integrate "reasoning + grounding + referring" into a single LMM using a unified four-task definition and two-stage training (textual multi-granularity reasoning followed by training-free pixel grounding). Grounding is achieved entirely by reusing native position word logits, requiring zero additional parameters and zero additional supervision.

## Method

### Overall Architecture

IQA-Spider consists of an **LMM backbone** (Phi-3.5-Vision / Qwen2.5-VL / Qwen3-VL) and a **frozen SAM segmentation head**. Given an input image and a quality-related question, the LMM first outputs a textual response containing position words (e.g., "top/left") and semantic descriptions. If the answer implies the target area is the whole image, the segmentation is skipped, and a full-image mask is used; otherwise, the response enters the text-to-point module, which converts position word logits into SAM point prompts, and SAM generates the final mask.

The training utilizes a **two-stage, Conflict-Free design**: The first stage involves textual multi-task instruction fine-tuning (LoRA on LLM + full-tune of vision encoder and projector) to teach the model global/local description, grounding text answers, and referring. The second stage requires no further training; the textual spatial perception learned in the first stage is upgraded to pixel-level grounding at "zero cost" through text-to-point.

### Key Designs

1.  **Unified Four-Task Paradigm + IQA-Spider-33K Dataset**:
    *   **Function**: Decomposes IQA into global quality description, local quality description, visual quality grounding (further divided into HyD-G for mixed distortion intensity, SiD-G for single distortion intensity, and DAO-G for distortion accumulation order), and visual quality referring (short/long answers), covering all granularities from whole-image to pixel.
    *   **Mechanism**: Employs a hybrid labeling pipeline—synthetic distortions follow a fully automated pipeline (extracting semantic instance regions via SSA $\rightarrow$ injecting multiple distortions by mask $\rightarrow$ generating QA via InternVL-2.5 multi-turn dialogue); real distortions follow a semi-automatic pipeline (manual region-level distortion labeling + InternVL-2.5 QA generation). Existing data such as Q-Instruct and DQ-495K are integrated conflict-free into the training set. Validation by 10 human raters on a 40% sample showed $>80\%$ of samples scored 4-5 across semantic, spatial, distortion, and language dimensions.
    *   **Design Motivation**: Previous IQA datasets labeled either only distortion categories (narrow) or only whole-image descriptions (coarse), lacking a unified definition to support "region-aware + multi-task joint training." This work establishes a task-and-data system to provide structural training signals for multi-granularity learning rather than just increasing scale.

2.  **Text-to-Point Grounding Paradigm**:
    *   **Function**: Converts LMM textual output into SAM point prompts without introducing special tokens or fine-tuning the segmentation head.
    *   **Mechanism**: Performs a closed-set softmax on LMM hidden states for designated position word sets $\{left, right\}$ and $\{top, bottom\}$ to obtain probabilities $p_{l_i} = e^{\chi_{l_i}/\tau} / \sum_j e^{\chi_{l_j}/\tau}$. Weighted averages are calculated based on normalized coordinates (left=0, right=1, top=0, bottom=1): $x = \sum_i p_{w_i} \times W,\ y = \sum_i p_{h_i} \times H$. The resulting $(x,y)$ is used directly as the SAM point prompt without additional training.
    *   **Design Motivation**: Existing SAM-based grounding methods (LISA / GLaMM, etc.) hard-couple language generation and pixel segmentation using special tokens like `<seg>`, which can harm instruction following and reasoning. Other methods using attention maps as implicit prompts (Wu 2024c / Cao 2024) either have high memory overhead or require additional image encoders. This method is reasoning-preserving, training-free, and plug-and-play.

3.  **Two-Stage Conflict-Free Training + Mixed Dataset**:
    *   **Function**: Enables a single LMM to master global/local description, grounding, and referring simultaneously without interference.
    *   **Mechanism**: The first stage uses Q-Instruct (global/local QA) + DQ-495K (distortion recognition and reasoning) + the self-built IQA-Spider-33K for instruction fine-tuning, using only next-token prediction cross-entropy loss. The segmentation head remains frozen throughout, and no grounding-specific loss is introduced. The second stage is entirely training-free.
    *   **Design Motivation**: Ablation (Tab. 4) shows that description tasks are sensitive to the addition of a single external dataset (slight drop), but the combination of all three sets yields the best results. Grounding benefits primarily from "explicit region-text alignment" data, while referring improves monotonically with data diversity. This indicates synergy among the three datasets.

### Loss & Training
Standard next-token prediction cross-entropy (instruction fine-tuning) is used. LoRA is applied to the LLM, while the vision encoder and projector undergo full-parameter fine-tuning. SAM remains frozen throughout. The second-stage grounding is entirely training-free.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | Ours (Qwen3-VL) | Prev. SOTA | Gain |
|---|---|---|---|---|
| Self-built benchmark — Global Des. | GPT-4V score (0-10) | 7.12 | 5.90 (Qwen3-VL Baseline) | +1.22 |
| Self-built benchmark — Local Des. | GPT-4V score (0-10) | 7.10 | 5.45 (Qwen3-VL) | +1.65 |
| Self-built benchmark — Grounding | GPT-4V score (0-5) | 2.41 | 1.25 (Qwen2.5-VL) | +1.16 |
| Self-built benchmark — Ref-long | Accuracy | 0.484 | 0.176 (Qwen3-VL) | +0.308 |
| Q-Bench-A1 (LLVisionQA-dev) | Accuracy | 74.45% | 67.56% (Q-Instruct) | +6.89% |
| Q-Ground-Test | mIoU | 0.338 | 0.271 (Q-Ground) | +0.067 |
| KADID-10K (Scoring) | SRCC/PLCC | 0.741/0.746 | 0.698/0.676 (Q-Instruct) | +0.043/+0.070 |

Notably, on Q-Ground-Test, IQA-Spider **was never trained on Q-Ground-100K and is training-free during the grounding stage**, yet it outperforms the Q-Ground baseline fine-tuned specifically on that data, demonstrating strong generalization.

### Ablation Study (Tab. 4, based on Qwen3-VL)

| Configuration | Global Des. | Local Des. | Grounding | Ref-short | Ref-long |
|---|---|---|---|---|---|
| Ours only | 7.01 | 7.07 | 2.42 | 0.541 | 0.458 |
| Ours + Q-Instruct | 6.99 | 7.03 | 2.53 | 0.542 | 0.466 |
| Ours + DQ-495K | 7.00 | 6.86 | 2.36 | 0.547 | 0.476 |
| Ours + All (IV) | **7.12** | **7.10** | 2.41 | **0.594** | **0.484** |

Additionally, comparing text-to-point vs. EVF-SAM (Fig. 5): The authors provided EVF-SAM with three types of text input (full answer / spatial only / semantic only). EVF-SAM performed best only with "semantic only" input and still remained weaker than the training-free point prompts of this method.

### Key Findings
- **Training-free text-to-point outperforms jointly trained EVF-SAM**: This suggests that native LMM position tokens already encode sufficiently strong spatial signal; "hard-training special tokens" may be inefficient.
- **Synergy of mixed datasets is non-monotonic**: Adding Q-Instruct or DQ-495K individually causes a drop in description tasks, but adding all three is optimal, indicating that data complementarity is more important than raw scale.
- **Method universality across backbones**: Consistent improvements were observed across Phi-3.5-Vision (4B), Qwen2.5-VL (7B), and Qwen3-VL (7B), verifying the plug-and-play nature.
- **Strong cross-domain generalization**: Beating specialized models on Q-Ground-Test without training on Q-Ground-100K shows that decoupling "position perception" from "semantic perception" enables better generalization.

## Highlights & Insights
- **Truly "zero-cost" grounding integration**: Reusing native LMM position word logits $\rightarrow$ weighted coordinate regression $\rightarrow$ SAM point prompt is an exceptionally concise path. It requires no new parameters, no new loss, and does not damage the language space, providing an elegant solution for reasoning-grounding unification.
- **Systematic decomposition of IQA into four-granularity tasks**: Rather than "simply pouring in more data," this work completes the task space first (especially with the detailed HyD-G / SiD-G / DAO-G grounding sub-tasks). The relatively small dataset (33K) is sufficient to support the benchmark.
- **Conflict-free data fusion**: Revealed non-trivial patterns in multi-source IQA joint training—individual additions may cause drops, while full integration yields gains. This has direct reference value for future multi-task IQA data mixing.
- **Transferability**: The text-to-point trick is not limited to IQA. Any unified architecture where a "language model describes position + segmentation model outputs mask" (e.g., medical image analysis, robot vision instructions) can directly reuse this, provided the LMM outputs explicit orientation words.

## Limitations & Future Work
- Position words are strictly limited to 4 (top/bottom/left/right), which can only express coarse positions via "quantiles + weighting." This may not be accurate for complex regions like centers, corners, or elongated shapes; expanding to a 9-grid or finer token set is a natural direction.
- The dataset scale is relatively small (33K) and relies on InternVL-2.5 for QA generation; the QA quality ceiling is capped by the base LMM. The authors emphasize the "task system over data volume," but actual performance might still be limited.
- Evaluation relies heavily on GPT-4V scoring (average of 5 rounds), which poses risks of evaluator bias and difficulty in reproduction; rankings might shift with different scoring models.
- Grounding follows a serial "classification logits $\rightarrow$ single point prompt $\rightarrow$ SAM mask" link. If there are multiple independent distortion regions, it can only output one point, making it difficult to ground multiple areas at once; multi-point/multi-mask extensions have not been addressed.
- The annotation of distortion accumulation order (DAO-G) relies heavily on manually predefined "perceptually recognizable accumulation orders," which requires re-labeling when migrating to new distortion types.

## Related Work & Insights
- **vs Q-Ground**: Q-Ground connects grounding to SAM via special tokens, requiring Q-Ground-100K training which harms instruction following. Ours reuses position tokens in a training-free manner and performs better on Q-Ground-Test without specialized grounding data.
- **vs LISA / GLaMM / Sa2VA**: These general `<seg>` token paradigms are effective for general segmentation but average only 0.078 - 0.192 on IQA grounding. Ours achieves 0.364 - 0.408, a gap stemming from both "reasoning-grounding decoupling" and "quality-specific knowledge."
- **vs DepictQA-Wild / Grounding-IQA**: DepictQA-Wild is strong in description but weak in localization; Grounding-IQA is strong in bbox referring but narrow in task scope. This work incorporates their capabilities into a single model via a four-task system.
- **vs EVF-SAM**: EVF-SAM uses a trainable multimodal encoder to turn text into prompts but remains weaker than our training-free scheme, suggesting "small but precise logit signals" may be more effective than "large semantic encodings" for grounding prompts.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using logit-weighted coordinate regression for text-to-point is concise, elegant, and empirically effective. The four-task system is a rare integrated design in the IQA field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete chain of evidence across 3 backbones, multiple benchmarks (self-built + Q-Bench + Q-Ground-Test + KADID), data ablation, and comparison with EVF-SAM.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-formalized task definitions, and concise formulas. Some tables are vertically displayed, which slightly affects readability.
- Value: ⭐⭐⭐⭐ The "reusing native token logits as grounding prompts" approach is valuable for all "LMM + Segmentation head" unified architectures. The IQA-Spider-33K dataset and four-task benchmark are likely to become standard evaluations for multi-granularity IQA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[CVPR 2026\] VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](../../CVPR2026/interpretability/viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)
- [\[CVPR 2026\] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](../../CVPR2026/interpretability/neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [\[ICML 2026\] Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning](equilibrium_reasoners_learning_attractors_enables_scalable_reasoning.md)

</div>

<!-- RELATED:END -->
