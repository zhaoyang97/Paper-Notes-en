---
title: >-
  [Paper Note] GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations
description: >-
  [AAAI 2026][LLM Evaluation][Eye Gaze Analysis] This paper proposes GazeInterpreter, an LLM-based hierarchical framework that converts raw gaze signals into textual narrations via a symbolic gaze parser, integrates them with body motion narrations to produce eye-body-coordinated descriptions, and iteratively refines outputs through a self-correction loop, yielding significant improvements on downstream tasks including text-driven motion generation, action prediction, and behavior summarization.
tags:
  - AAAI 2026
  - LLM Evaluation
  - Eye Gaze Analysis
  - Human Behavior Understanding
  - Large Language Models
  - Multimodal Fusion
  - Motion Generation
date: 2026-05-08
content_hash: 7e206fb028ea6aa1
---

# GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations

**Conference**: AAAI 2026
**arXiv**: [2511.16245](https://arxiv.org/abs/2511.16245)
**Code**: [https://github.com/EvergreenChang/GazeInterpreter](https://github.com/EvergreenChang/GazeInterpreter)
**Area**: LLM Evaluation
**Keywords**: Eye Gaze Analysis, Human Behavior Understanding, Large Language Models, Multimodal Fusion, Motion Generation

## TL;DR

This paper proposes GazeInterpreter, an LLM-based hierarchical framework that converts raw gaze signals into textual narrations via a symbolic gaze parser, integrates them with body motion narrations to produce eye-body-coordinated descriptions, and iteratively refines outputs through a self-correction loop, yielding significant improvements on downstream tasks including text-driven motion generation, action prediction, and behavior summarization.

## Background & Motivation

Comprehensively interpreting human behavior is a central challenge in human-centric perceptual AI. Existing work, however, focuses almost exclusively on **body behavior interpretation**, largely neglecting **eye gaze** and its coordination with body movement:

**Gaze as a direct window into intent**: When reaching for a cup, the eyes typically fixate on the target before or during arm movement, directly revealing latent intention.

**Strong eye-body coupling**: Extensive prior research has established strong correlations between eye gaze and head, torso, and whole-body motion.

**Limitations of prior work**: Methods such as MotionGPT and MotionLLM project body motion or video into language space to generate descriptions, but entirely omit gaze information.

**Core challenge**: How can low-level continuous numerical gaze sensor data be reliably converted into high-level structured semantic representations? Feeding raw numerical values directly to an LLM risks factual hallucination and signal disconnection.

**Core solution**: Raw gaze data is first abstracted into an intermediate vocabulary of symbolic gaze events, providing a reliable semantic grounding, before multi-level fusion is performed via LLM.

## Method

### Overall Architecture

GazeInterpreter adopts a three-phase hierarchical coarse-to-fine architecture:

1. **Phase 1**: Parse raw gaze signals → textual narration (symbolic parsing + LLM generation)
2. **Phase 2**: Integrate gaze narration with body motion atomic narration → eye-body-coordinated narration
3. **Phase 3**: Self-correction loop → multi-dimensional iterative refinement

### Key Designs

#### 1. **Symbolic Gaze Parser**

A deterministic module that converts continuous gaze signals into discrete symbolic event sequences.

**Processing steps**:
- Given raw gaze signal $S_i^g \in \mathbb{R}^{N_g \times 2}$ (yaw, pitch sequence), compute instantaneous angular velocity:
$$\omega_j = \frac{\sqrt{(y_j - y_{j-1})^2 + (p_j - p_{j-1})^2}}{t_j - t_{j-1}}$$
- Apply the I-VT (Identification-by-Velocity-Threshold) algorithm with dual thresholds ($v_{\text{low}}=30°/s$, $v_{\text{high}}=100°/s$) to classify signals into three event primitives: **Fixation**, **Saccade**, and **SmoothPursuit**.
- Each event encapsulates not only the category but also quantitative attributes such as duration, amplitude, and peak velocity, along with corresponding qualitative descriptors.

**Design Motivation**: Abstracting noisy high-dimensional signals into compact, machine-readable symbolic representations avoids hallucination issues that arise when LLMs process raw numerical data directly.

#### 2. **Symbolic-to-Text Synthesizer**

An LLM (Gemini-2.5-Flash) translates the symbolic event sequence $E_i$ into a coherent textual narration $T_i^g$. The core idea is to reframe the LLM's task from "high-risk numerical inference" to "low-risk factual translation"—rendering symbolically grounded, verifiable behavioral descriptions into fluent natural language. Carefully designed few-shot prompts are employed.

#### 3. **Eye-Body Motion Integration (Phase 2)**

**Historical context**: A sliding observation window ($W=2$) aggregates historical context $\mathcal{H}_i$, comprising (i) previously inferred integrated narrations and (ii) feedback from the preceding self-correction round.

**Integrated narration generation**: A structured prompt template is constructed:
$$\Pi_{\text{integ}}(i) = [\texttt{CTX}:\mathcal{H}_i;\ \texttt{GAZE}:T_i^g;\ \texttt{MOTION}:S_i^m]$$

The LLM performs reasoning over the structured input (rather than mere summarization), for example inferring "the user is carefully scanning the ground while walking" by associating a gaze shift with the concurrent walking motion.

#### 4. **Self-Correction Loop (Phase 3)**

Multi-dimensional evaluation and iterative refinement are performed through the collaboration of $\text{LLM}_{\text{eval}}$ and $\text{LLM}_{\text{refine}}$:

**Evaluation dimensions** (each scored 1–5):

| Type | Dimension | High Score | Low Score |
|------|-----------|-----------|-----------|
| Gaze narration | Continuity | Natural, fluid gaze transitions | Abrupt or illogical event descriptions |
| Integrated narration | Modal matching | Cross-modal mutually supportive integration | Modal disconnection, redundancy, or contradiction |
| Integrated narration | Temporal consistency | Clear temporal logical progression | Absence of identifiable temporal structure |
| Integrated narration | Completeness | All key elements fully covered | Missing critical information or behavioral events |

The loop iterates up to $K_{\text{max}}=3$ times until all scores $\geq \tau=4.5$ or the maximum iteration count is reached.

### Loss & Training

GazeInterpreter requires no conventional model training; it leverages the few-shot in-context learning capability of a pretrained LLM (Gemini-2.5-Flash). Key hyperparameters:
- I-VT thresholds: $v_{\text{low}}=30°/s$, $v_{\text{high}}=100°/s$
- Sliding window size: $W=2$
- Self-correction maximum iterations: $K_{\text{max}}=3$, score threshold $\tau=4.5$

## Key Experimental Results

### Main Results

Text-driven motion generation is evaluated on the large-scale Nymeria benchmark with fixed MotionGPT weights, comparing different textual inputs:

| Scene Type | Method | MM Dist↓ | FID↓ | Top-1↑ | Top-3↑ | MM↑ |
|-----------|--------|----------|------|--------|--------|-----|
| Low-level | MotionGPT | 6.748 | 7.458 | 0.052 | 0.187 | 3.469 |
| Low-level | **+GazeInterpreter** | **6.406** | **6.801** | **0.102** | **0.214** | **3.727** |
| High-level | MotionGPT | 7.133 | 8.804 | 0.054 | 0.162 | 3.223 |
| High-level | **+GazeInterpreter** | **6.862** | **8.134** | **0.062** | **0.193** | **3.864** |
| All | MotionGPT | 6.941 | 8.131 | 0.053 | 0.175 | 3.346 |
| All | **+GazeInterpreter** | **6.634** | **7.468** | **0.082** | **0.204** | **3.796** |

**Downstream tasks**:

| Task | Method | Cosine Sim↑ | BERT F1↑ | ROUGE-L↑ | Action F1↑ |
|------|--------|------------|----------|----------|------------|
| Action Prediction | Nymeria | 0.459 | 0.868 | 0.202 | 0.226 |
| Action Prediction | **GazeInterpreter** | **0.506** | **0.879** | **0.231** | **0.248** |
| Behavior Summarization | Nymeria | 0.480 | 0.836 | 0.197 | 0.150 |
| Behavior Summarization | **GazeInterpreter** | **0.537** | **0.860** | **0.575** | **0.229** |

### Ablation Study

| Configuration | MM Dist↓ | FID↓ | Top-1↑ | Note |
|--------------|----------|------|--------|------|
| w/o hierarchical structure | 8.135 | 9.124 | 0.059 | Largest performance drop, validating the centrality of hierarchical integration |
| w/o symbolic parser | 7.642 | 7.893 | 0.061 | Direct use of raw signals causes degradation |
| w/o self-correction | 7.425 | 7.831 | 0.063 | Absence of iterative refinement reduces quality |
| **Full GazeInterpreter** | **6.634** | **7.468** | **0.082** | All modules enabled |

**Incremental analysis of self-correction quality dimensions**:

| Continuity | Matching | Temporal | Completeness | Top-1↑ | FID↓ |
|-----------|---------|---------|-------------|--------|------|
| | | | | 0.063 | 7.831 |
| ✓ | | | | 0.069 | 7.722 |
| ✓ | ✓ | | | 0.072 | 7.644 |
| ✓ | ✓ | ✓ | | 0.074 | 7.573 |
| ✓ | ✓ | ✓ | ✓ | **0.082** | **7.468** |

The introduction of each evaluation dimension yields cumulative performance gains.

### Key Findings

1. **Gaze information is critical for motion generation**: Solely enriching text descriptions with gaze information—without modifying the generative model—yields significant FID improvement (8.131→7.468).
2. **Low-level scenes benefit more**: The fine-grained intent information provided by gaze is particularly beneficial for precise atomic motion generation.
3. **Eye-body-coordinated narrations are more predictive than human annotations**: In action prediction, GazeInterpreter narrations achieve higher Action F1 than the manually annotated Nymeria data.
4. **The symbolic intermediate layer is essential**: Having the LLM process raw numerical signals directly leads to substantial degradation.
5. **Sliding window $W=2$ is optimal**: Enlarging the window yields diminishing marginal returns and introduces redundant noise.

## Highlights & Insights

- **Opens a new research direction**: This is the first work to systematically integrate eye gaze parsing with body motion narration, revealing the substantial potential of gaze for behavior understanding.
- **The numerical → symbolic → textual decomposition strategy** is notably elegant, mitigating hallucination risks that arise when LLMs process sensor values directly.
- **The multi-dimensional evaluation framework of the self-correction loop** is transferable to other generative tasks.
- **A training-free, inference-only framework**: Built on LLM few-shot prompting and multi-stage reasoning, requiring no expensive task-specific model training.
- Consistent advantages are demonstrated across three tasks: motion generation, action prediction, and behavior summarization.

## Limitations & Future Work

- **Validated on a single dataset (Nymeria)**: Currently the only publicly available dataset containing both gaze and motion annotations, limiting generalizability.
- **High inference cost**: Three-stage LLM reasoning combined with the self-correction loop requires multiple LLM calls.
- **Reliance on predefined thresholds**: The velocity thresholds of the I-VT classifier must be set manually and may require adjustment across different scenarios.
- **Lack of end-to-end joint optimization**: The symbolic parsing, narration generation, and integration stages are entirely decoupled.
- Joint exploitation of egocentric image/video signals and gaze remains unexplored.

## Related Work & Insights

- **Fundamental distinction from MotionGPT/MotionLLM**: This work addresses not only body motion but also the coordination between eye gaze and body.
- The I-VT algorithm has been widely used in classical gaze analysis; this work is the first to combine it with LLMs.
- The self-correction loop conceptually parallels the self-revision mechanism in Constitutional AI.
- **Implications for embodied intelligence**: When inferring human intent, gaze signals may reveal goals earlier and more directly than limb movements.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Pioneering integration of gaze parsing with LLM-based behavior understanding, opening a new direction
- Experimental Thoroughness: ⭐⭐⭐⭐ — Motion generation + downstream tasks + complete ablation, but limited to a single dataset
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear; motivation is thoroughly articulated
- Value: ⭐⭐⭐⭐ — Reveals the substantial potential of gaze in behavior understanding with long-term impact

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] MindVote: When AI Meets the Wild West of Social Media Opinion](mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)
- [\[AAAI 2026\] Axis-Aligned Document Dewarping](axis-aligned_document_dewarping.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)
- [\[AAAI 2026\] Benchmarking LLMs for Political Science: A United Nations Perspective](benchmarking_llms_for_political_science_a_united_nations_perspective.md)
- [\[AAAI 2026\] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)

<!-- RELATED:END -->
