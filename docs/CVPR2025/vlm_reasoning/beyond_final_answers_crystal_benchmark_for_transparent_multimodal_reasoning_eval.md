---
title: >-
  [Paper Note] Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation
description: >-
  [CVPR 2025][VLM Reasoning][multimodal reasoning] This paper proposes the CRYSTAL benchmark (6372 instances) to evaluate MLLMs at the intermediate reasoning step level using Match F1 and Ordered Match F1. It reveals widespread cherry-picking behaviors and disordered reasoning processes, and introduces a CPR-Curriculum training strategy to improve reasoning quality.
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "multimodal reasoning"
  - "benchmark"
  - "step-level evaluation"
  - "process reward"
  - "GRPO"
date: 2026-05-08
content_hash: efbee7b68a759ea8
---

# Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation

**Conference**: CVPR 2025  
**arXiv**: [2603.13099](https://arxiv.org/abs/2603.13099)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: multimodal reasoning, benchmark, step-level evaluation, process reward, GRPO

## TL;DR
This paper proposes the CRYSTAL benchmark (6372 instances) to evaluate MLLMs at the intermediate reasoning step level using Match F1 and Ordered Match F1. It reveals widespread cherry-picking behaviors and disordered reasoning processes, and introduces a CPR-Curriculum training strategy to improve reasoning quality.

## Background & Motivation
**Background**: Multimodal benchmarks such as MathVista and RealWorldQA solely focus on final answer accuracy, which fails to distinguish whether a model achieves "genuine understanding" or a "lucky guess"—a model can produce the correct answer with a completely contradictory reasoning process.

**Limitations of Prior Work**: (1) Answer-centric accuracy evaluation allows models to achieve high scores via shortcuts; (2) Existing CoT evaluations lack machine-verifiable checkpoints for intermediate steps; (3) Theoretical analysis shows that answer-centric evaluation structurally incentivizes hallucinations by penalizing uncertainty.

**Key Challenge**: Models can perform "cherry-picking" by generating a small number of high-precision steps, which deceptively makes them appear correct while skipping many critical reasoning steps. Answer accuracy and reasoning faithfulness represent two distinct dimensions.

**Goal**: How to systematically evaluate the quality of intermediate reasoning steps in MLLMs? How to use step-level rewards to improve model reasoning?

**Key Insight**: Drawing inspiration from the Delphi method (multi-expert consensus building), multiple MLLMs are utilized to independently generate reasoning steps, followed by semantic clustering and human verification to construct reference step sets.

**Core Idea**: Evaluate reasoning step quality using semantic-matched F1 score, and replace additive rewards with multiplicative Causal Process Reward (CPR) to train more faithful reasoning.

**Theoretical Motivation**: The authors formally prove that under answer-centric evaluation, for any question involving uncertainty, the optimal strategy for a model is to generate a CoT that maximizes the expected answer reward (rather than faithfully reflecting internal reasoning), which structurally incentivizes hallucination.

## Method

### Overall Architecture
CRYSTAL consists of two components: evaluation and training. The evaluation pipeline provides 6372 instances (averaging 11.6 reference reasoning steps each) scored via Match F1 and Ordered Match F1. The training pipeline proposes CPR for GRPO training. The datasets span three major sources: MathVista, MMMU, and RealWorldQA, covering diverse task types such as mathematical reasoning, scientific QA, and real-world scene understanding.

### Key Designs

1. **Delphi-Inspired Reference Step Generation Pipeline**:

    - **Function**: Generates a high-quality sequence of reference reasoning steps for each question.
    - **Mechanism**: Four MLLMs of different architectures (Qwen2.5-VL-72B, InternVL3-76B, Gemma3-27B, Llama-4-Maverick) independently generate steps $\to$ use a sentence encoder to compute cosine similarity for semantic clustering (connected components) $\to$ select the most representative cluster centers $\to$ a fifth model (Molmo-72B) validates logical consistency $\to$ human quality gating (<5% remake required).
    - **Design Motivation**: Independent generation from multiple sources reduces correlated errors, and semantic clustering accomplishes step-level self-consistency voting.

2. **Match F1 & Ordered Match F1**:

    - **Function**: Quantifies the alignment quality between the predicted reasoning steps and reference steps.
    - **Mechanism**: Uses `all-distilroberta-v1` to encode all steps, followed by a greedy 1:1 matching where cosine similarity $\ge$ $\tau=0.35$. Match F1 is the harmonic mean of Precision and Recall. Ordered Match F1 builds on top of this by multiplying by the LIS ratio to penalize disorder: $\text{Ordered-F1} = \text{F1} \cdot ((1-\alpha) + \alpha \cdot \text{LIS-ratio})$
    - **Design Motivation**: Match F1 assesses presence while Ordered F1 assesses order correctness, making them complementary. $\alpha=0.5$ serves as the default, and the authors verify that the conclusions remain robust within the range of 0.3 to 0.7.

3. **Causal Process Reward (CPR) + CPR-Curriculum**:

    - **Function**: Step-level reward function for GRPO training.
    - **Mechanism**: Multiplicative reward: when the answer is correct, $R = a_w + s_w \cdot \text{F1}_{step}$; when the answer is incorrect, $R = s_w \cdot \text{F1}_{step} \cdot \lambda$ ($\lambda=0.3$). CPR-Curriculum involves a two-phase training scheme: Phase 1 establishes a baseline using only answer rewards, while Phase 2 introduces the full CPR along with progressively increasing reasoning difficulty.
    - **Design Motivation**: Additive rewards allow models to score highly solely by guessing the correct answer while neglecting reasoning quality; a multiplicative reform forces their coupling. PCGrad is employed to prevent gradient conflicts between accuracy and reasoning quality goals.

### Loss & Training
GRPO with CPR-Curriculum, with weights $a_w=0.65, s_w=0.35$ (determined through an ablation of six configurations). Once answer capabilities are established in Phase 1, step-level rewards are introduced in Phase 2. Training remains stable for 2800 steps, whereas the additive formulation suffers from NaN gradient explosions at 1500 steps. PCGrad (Projected Conflicting Gradients) is utilized to detect and project conflicting gradients, ensuring that accuracy goals and reasoning quality goals do not interfere with each other.

## Key Experimental Results

### Main Results

| Model | Parameters | Accuracy | Match F1 | Precision | Recall | LIS | Ord. F1 |
|------|--------|----------|----------|-----------|--------|-----|---------|
| GPT-5 | - | 57.99% | 0.612 | 0.925 | 0.479 | 0.636 | 0.539 |
| GPT-5-mini | - | 55.59% | **0.773** | 0.978 | 0.669 | 0.560 | **0.670** |
| Gemini 2.5 Flash | - | 53.95% | 0.673 | 0.701 | **0.765** | 0.584 | 0.579 |
| Qwen3-VL-8B | 8B | **57.66%** | 0.659 | 0.827 | 0.590 | 0.624 | 0.572 |
| Gemma3-4B | 4B | 28.65% | 0.618 | 0.878 | 0.506 | 0.668 | 0.547 |
| InternVL3.5-38B | 38B | 51.21% | 0.612 | 0.892 | 0.498 | 0.643 | 0.538 |

### Ablation Study

| Strategy | Accuracy | Match F1 | Recall | Notes |
|------|----------|----------|--------|------|
| Baseline (Qwen2.5-VL-3B) | 39.85% | 0.480 | 0.347 | Untrained |
| Composite (Additive) | 44.92% | 0.426 | 0.284 | Reasoning degrades; NaN after 1500 steps |
| Answer-Only | 44.30% | 0.429 | 0.308 | Reasoning unchanged |
| CPR (Multiplicative) | 41.40% | **0.633** | 0.489 | F1 +32%, but slightly lower accuracy |
| CPR-Curriculum | **47.52%** | **0.633** | 0.493 | Both improved, optimal |

### Key Findings
- **Cherry-picking is a widespread phenomenon**: 19 out of 20 evaluated models show Precision $\gg$ Recall (1.2x-7.2x), including commercial models not involved in the benchmark construction. Models tend to generate a small number of high-confidence steps while skipping intermediate reasoning.
- **Decoupling of accuracy and reasoning faithfulness**: GPT-5 achieves the highest accuracy (57.99%) but ranks 8th in F1; Gemma3-4B (4B) outperforms InternVL3.5-38B (38B) in F1, indicating that architecture is more critical than scale.
- **Trade-off between reasoning length and quality**: Longer reasoning chains do not necessarily yield higher Match F1. Some models (e.g., Gemini 2.5 Flash) generate more steps but exhibit disordered sequences, leading to a drop in Ordered F1.
- **No model can systematically preserve reasoning order**: Among the competitive models, the highest LIS is only 0.636 (GPT-5), meaning 36% of the matched steps are in the wrong order.
- **Most significant improvement for small models via CPR-Curriculum**: A 3B model improves its F1 from 0.480 to 0.633 (+32%) via CPR-Curriculum, while accuracy increases by 7.67 percentage points, showing that step-level rewards offer a highly cost-effective way to improve reasoning in small models.
- Additive reward training leads to gradient explosion, while multiplicative CPR stabilizes training and improves F1 by 32%.

## Highlights & Insights
- **Precise diagnosis of the "lucky guess" problem**: Match F1 exposes critical reasoning deficiencies masked by mere answer accuracy. This finding has profound implications for MLLM evaluation methodology—all benchmarks that focus solely on the final answer may overestimate model capabilities.
- **Insights into multiplicative vs. additive rewards**: Additive rewards allow models to independently maximize components (e.g., scoring by guessing while ignoring reasoning), whereas multiplicative rewards force coupling. This design paradigm is highly transferable to other multi-objective RL scenarios.
- **Reusable Delphi pipeline**: The reference step generation methodology, which combines multi-model independent generation, semantic clustering, and quality gating, can be adapted to construct other benchmarks requiring intermediate step annotations.
- **Necessity of Curriculum strategies**: Introducing step-level rewards directly can impair answer accuracy. The two-phase curriculum decouples capability building from reasoning optimization, offering valuable references for the stability design of RL training.

## Limitations & Future Work
- Reference steps are generated and clustered by LLMs, which may omit certain alternative plausible reasoning paths (only one reasoning path is selected from multiple correct ones).
- Match F1 relies on the semantic matching quality of the sentence encoder, which might be less accurate for highly abstract mathematical reasoning steps.
- The threshold $\tau=0.35$ is a fixed value tuned on the validation set; different task domains may require different thresholds.
- Training experiments are only validated on Qwen2.5-VL-3B, without verification on larger-scale models.
- The integration of CPR with other alignment methodologies, such as DPO, remains unexplored.
- The visual reasoning types covered by the benchmark are predominantly mathematical and scientific, offering limited coverage of common-sense and spatial reasoning.

## Related Work & Insights
- **vs MathVista/MMMU**: These only evaluate final answers, whereas CRYSTAL evaluates intermediate steps, making them complementary.
- **vs Multimodal-CoT**: It decouples generation but lacks step-level evaluation metrics.
- **vs MPBench**: Also conducts trajectory-level evaluation, but CRYSTAL provides more systematic metrics and training methods.
- **vs PRM (Process Reward Model)**: PRMs score steps using a trained reward model, whereas CRYSTAL utilizes a reference step set and semantic matching. This eliminates the need for an extra reward model, making it more transparent and interpretable.
- **vs Self-Consistency**: Self-Consistency conducts voting via multiple sampled answers, while CRYSTAL performs semantic clustering over steps independently generated by multiple models, extending the voting concept from the answer level to the step level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Step-level evaluation and multiplicative rewards are significant innovations, revealing the widespread cherry-picking phenomenon.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 20 models, GRPO training, and detailed ablations make it highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Findings 1-5 showcase a clear structure and highly convincing conclusions.
- Value: ⭐⭐⭐⭐⭐ Serves as a significant driving force for reforming the MLLM evaluation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation](../../ICLR2026/vlm_reasoning/jointavbench_a_benchmark_for_joint_audio-visual_reasoning_evaluation.md)
- [\[ACL 2025\] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](../../ACL2025/vlm_reasoning/finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)
- [\[ACL 2025\] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](../../ACL2025/vlm_reasoning/fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)
- [\[NeurIPS 2025\] MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture](../../NeurIPS2025/vlm_reasoning/mirage_a_benchmark_for_multimodal_information-seeking_and_reasoning_in_agricultu.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](../../CVPR2026/vlm_reasoning/think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)

</div>

<!-- RELATED:END -->
