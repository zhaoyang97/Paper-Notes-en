---
title: >-
  [Paper Note] MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration
description: >-
  [ACL 2025][VLM Reasoning][Multimodal LLMs] This paper proposes the MMBoundary framework, which inserts natural language confidence statements at each step of the reasoning chain (rather than offering confidence only after the final answer). It combines textual and cross-modal self-reward signals to estimate confidence, and utilizes a two-stage training paradigm of SFT and RL to achieve step-level confidence calibration, reducing the calibration error by an average of 7.5% and…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Multimodal LLMs"
  - "Confidence Calibration"
  - "Reasoning Steps"
  - "Knowledge Boundary"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 257ad4ccbfd3807e
---

# MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration

**Conference**: ACL 2025  
**arXiv**: [2505.23224](https://arxiv.org/abs/2505.23224)  
**Code**: [https://github.com/Zhitao-He/MMBoundary](https://github.com/Zhitao-He/MMBoundary)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal LLMs, Confidence Calibration, Reasoning Steps, Knowledge Boundary, Reinforcement Learning

## TL;DR
This paper proposes the MMBoundary framework, which inserts natural language confidence statements at each step of the reasoning chain (rather than offering confidence only after the final answer). It combines textual and cross-modal self-reward signals to estimate confidence, and utilizes a two-stage training paradigm of SFT and RL to achieve step-level confidence calibration, reducing the calibration error by an average of 7.5% and improving task accuracy by 8.3%.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) demonstrate outstanding performance in cross-modal reasoning, but their reliability remains questionable, often making "confident mistakes." Existing confidence estimation efforts focus solely on training models to output confidence scores at the overall answer level.

**Limitations of Prior Work**: (a) Overall confidence metrics cannot locate the source of errors—perceptual mistakes (e.g., misidentifying objects) and reasoning mistakes exhibit the same low confidence; (b) Unlabeled early erroneous steps lead to "hallucination snowballing," where perceptual errors in previous steps are propagated and amplified in subsequent reasoning; (c) Trained models tend to output uniform confidence levels with poor discriminative capability.

**Key Challenge**: Enabling MLLMs to know "where they are uncertain" requires fine-grained, step-level confidence estimation. However, labeling step-level confidence is extremely labor-intensive, and a massive gap exists between the internal uncertainty signals of the model and its expressed confidence.

**Goal**: Enable MLLMs to automatically express calibrated natural language confidence statements after each reasoning step, allowing models to "know what they do not know."

**Key Insight**: Utilize model internal states (token probabilities, entropy) and cross-modal alignment signals (CLIPScore) as self-reward estimates for step-level confidence, achieving both expression and calibration through a two-stage training scheme of SFT warm-up and PPO reinforcement learning.

**Core Idea**: Insert confidence statements after each sentence in the reasoning chain, estimate initial confidence using multi-source self-reward signals, and calibrate the expressed confidence with actual correctness via reinforcement learning.

## Method

### Overall Architecture
Input: Image $I$ + Question $Q$. Output: An alternating sequence of reasoning steps and confidence statements $[z_1, c_1, z_2, c_2, ..., z_T, c_T]$. Two-stage training: SFT warm-up (learning to generate confidence statements) $\rightarrow$ PPO reinforcement learning (calibrating confidence accuracy).

### Key Designs

1. **Multi-source Internal Confidence Estimation**:

    - Function: Integrates four types of signals to estimate the confidence score of each sentence.
    - Mechanism: (a) Length-Normalized Log-Probability $U_{LNLP}$—sentence-level average token negative log-probability; (b) Mean Token Entropy $U_{MTE}$—average entropy of individual token distributions; (c) TokenSAR—weighted negative log-probability considering token relevance to the whole text; (d) CLIPScore—cosine similarity of CLIP embeddings between generated text and the input image. These four factors are weighted-averaged and mapped to five levels of confidence.
    - Design Motivation: Single-source signals are unreliable, and purely textual uncertainty estimation methods fail to capture visual alignment. CLIPScore bridges the gap in estimating cross-modal alignment, making confidence estimation more accurate under multimodal scenarios.

2. **Confidence Score-Statement Mutual Mapping**:

    - Function: Establishes a bidirectional mapping between numerical confidence scores and natural language statements.
    - Mechanism: A predefined pool of five-level confidence statements (ranging from "uncertain" to "fully confident") is constructed, containing multiple expressions per level. Forward mapping: Installs corresponding statements randomly sampled from the target pool into training data based on the estimated scores. Backward mapping: During the RL phase, a sentence encoder computes cosine similarities between the generated statement and statements in each pool to map the text back to a numerical score.
    - Design Motivation: Natural language statements are more human-readable and maintain the coherence of the reasoning chain better than numerical scores, while backward mapping enables RL reward computation.

3. **SFT Warm-up Phase**:

    - Function: Fine-tunes the model to learn to generate confidence statements after each sentence.
    - Mechanism: Labelling training data using internally estimated confidence, inserting corresponding statements after each sentence, and fine-tuning via standard cross-entropy loss.
    - Design Motivation: Training with RL directly from scratch is extremely challenging; SFT ensures the model learns to generate properly formatted statements beforehand (warm-up).

4. **PPO Reinforcement Learning Phase**:

    - Function: Further calibrates the expressed confidence and improves answer quality using three reward functions.
    - Mechanism: $R = \alpha R_{KA} + \beta R_{EC} + \gamma R_{CS}$, where: (a) **Knowledge Accuracy Reward** $R_{KA}$: matching degree between the generated sentence and the reference reasoning chain; (b) **Expected Calibration Reward** $R_{EC}$: consistency between expressed confidence and actual correctness (similar to ECE); (c) **Confidence Self-Calibration Reward** $R_{CS}$: consistency between expressed confidence and internally estimated confidence.
    - Design Motivation: Models after SFT tend to generate uniform confidence. RL encourages alignment of high accuracy with high confidence and low accuracy with low confidence through discriminative rewards.

### Loss & Training
SFT utilizes standard cross-entropy loss. RL employs the PPO algorithm with GAE (Generalized Advantage Estimation). Labeled reference reasoning chains are used to compute the Knowledge Accuracy Reward.

## Key Experimental Results

### Main Results
Evaluated on three multimodal reasoning datasets: A-OKVQA, ScienceVQA, and CulturalVQA.

| Method | A-OKVQA ECE↓ | A-OKVQA Acc↑ | ScienceVQA ECE↓ | ScienceVQA Acc↑ |
|------|-------------|-------------|-----------------|-----------------|
| Vanilla | Baseline | Baseline | Baseline | Baseline |
| SaySelf | 0.345 | 0.734 | 0.386 | - |
| **MMBoundary** | **Optimal** | **Optimal** | **Optimal** | **Optimal** |

- Reduces expected calibration error (ECE) by an average of 7.5%.
- Improves task accuracy by up to 8.3%.

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Full MMBoundary | Optimal | SFT + RL |
| SFT Only | Uniform confidence, poor discriminability | Demonstrates the necessity of RL |
| W/o CLIPScore | Multimodal calibration degrades | Visual signals are crucial for cross-modal confidence estimation |
| W/o $R_{CS}$ | Internal-external consistency degrades | Self-calibration reward helps align internal and expressed confidence |

### Key Findings
- **Step-level confidence is more useful than overall confidence**: Low-confidence steps can trigger self-correction, whereas overall confidence often underestimates correct steps because they are mixed with errors.
- **The RL phase is critical**: Following SFT, the model tends to output uniform confidence, and the collaborative work of the three RL reward functions is essential to achieve discriminative calibration.
- **Cross-modal signals are indispensable**: CLIPScore captures visual perceptual uncertainty that purely textual approaches cannot estimate.
- **Confidence improvement also brings accuracy gains**: The knowledge accuracy reward optimizes answer quality while calibrating confidence.

## Highlights & Insights
- **Step-level confidence statements**: Appending a confidence statement after each sentence is an innovative design that transforms the reasoning chain into "self-aware" reasoning. This template can be ported to any LLM application requiring fine-grained uncertainty estimation.
- **Self-reward + RL training paradigm**: Obviates the need for manual confidence annotations by leveraging internal model states as self-supervised signals. The two-stage SFT $\rightarrow$ RL strategy elegantly decouples format learning from calibration optimization.
- **Natural language confidence expressions**: More intuitive than numerical scores, naturally blending into the reasoning chain while facilitating downstream system processing (e.g., automatically steering away from low-confidence steps).

## Limitations & Future Work
- **Dependency on reference reasoning chains**: The knowledge accuracy reward requires labeled reference reasoning chains, which are expensive to acquire.
- **Adequacy of 5-level confidence granularity**: Finer granularity might provide more precise calibration but also increases mapping complexity.
- **Lack of detail on self-correction trigger mechanism**: The paper mentions that low confidence can trigger self-correction, but the implementation strategy for this correction is not fully elaborated.
- **Evaluated only on VQA tasks**: Performance on more complex multimodal reasoning tasks (e.g., visual planning, multi-step mathematical reasoning) remains to be verified.

## Related Work & Insights
- **vs SaySelf**: SaySelf also trains models to express confidence, but only at the overall answer level. MMBoundary elevates this to the step level, resolving the issue of hallucination propagation.
- **vs Sampling-based confidence approaches**: Although observing consistency across multiple samples is intuitive, it incurs heavy computational costs and cannot locate specific steps. MMBoundary relies on internal states for a single-pass estimation, which is far more efficient.
- **vs Verbalized Confidence**: Simply prompting the model to state its confidence is easily deceived by superficial calibration. MMBoundary truly aligns knowledge with confidence via RL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Step-level confidence calibration is an important new direction. The entire framework, integrating multi-source signals, SFT, and RL, is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three datasets with various metrics and ablation analyses, though it lacks verification on more comprehensive multimodal reasoning benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, the method description is systematic, and the paper is rich in mathematical formulations and charts.
- Value: ⭐⭐⭐⭐⭐ Addresses the core reliability issue of MLLMs. Step-level confidence is highly significant for AI safety and human-AI collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning](../../ACL2026/vlm_reasoning/vl-calibration_decoupled_confidence_calibration_for_large_vision-language_models.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/vlm_reasoning/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)
- [\[ACL 2025\] Answering Complex Geographic Questions by Adaptive Reasoning with Visual Context and External Commonsense Knowledge](answering_complex_geographic_questions_by_adaptive_reasoning_with_visual_context.md)
- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/vlm_reasoning/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)

</div>

<!-- RELATED:END -->
