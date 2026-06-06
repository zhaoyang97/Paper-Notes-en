---
title: >-
  [Paper Note] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression
description: >-
  [AAAI 2026][Model Compression][reasoning efficiency] This paper proposes CGRS (Certainty-Guided Reflection Suppression), a training-free efficient reasoning method that dynamically suppresses reflection trigger tokens (e…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "reasoning efficiency"
  - "overthinking"
  - "reflection suppression"
  - "certainty estimation"
  - "large reasoning models"
date: 2026-05-08
content_hash: 0cb071278c4adc59
---

# Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression

**Conference**: AAAI 2026
**arXiv**: [2508.05337](https://arxiv.org/abs/2508.05337)  
**Code**: None  
**Area**: Model Compression
**Keywords**: reasoning efficiency, overthinking, reflection suppression, certainty estimation, large reasoning models

## TL;DR
This paper proposes CGRS (Certainty-Guided Reflection Suppression), a training-free efficient reasoning method that dynamically suppresses reflection trigger tokens (e.g., "Wait", "But") when the model exhibits high confidence, reducing token consumption of large reasoning language models by 18.5%–41.9% while maintaining reasoning accuracy.

## Background & Motivation

### State of the Field
Large reasoning language models (LRLMs), such as OpenAI's o1/o3 and DeepSeek-R1, have achieved remarkable progress on tasks such as mathematics and programming through long chain-of-thought (CoT) reasoning combined with complex reflection behaviors (backtracking, exploring alternative strategies, and self-verification). These reflection behaviors are typically initiated by specific trigger tokens (e.g., "Wait", "Alternatively", "But", "Hmm").

### Limitations of Prior Work

**Overthinking problem**: LRLMs continue reasoning even after arriving at a correct answer, generating numerous redundant reflection steps that unnecessarily inflate token consumption and inference cost.

**Context window overflow**: In extreme cases, excessively long responses may exceed context window limits, causing critical information to be truncated.

**Limitations of existing methods**:
   - **Prompt-based methods** (e.g., TALE): rely on the model's instruction-following capability and yield unstable results.
   - **Decoding manipulation methods** (e.g., Dynasor, DEER): rely on hard-coded early-exit condition designs and make strong assumptions about the `</think>` token.
   - Both categories lack adaptive mechanisms to balance reflection suppression with reasoning quality.

### Root Cause
Reflection behavior is simultaneously the key mechanism for LRLMs' self-correction and the root cause of overthinking. The core challenge is how to suppress unnecessary reflection loops while preserving essential error correction.

### Starting Point
The approach is grounded in the model's internal certainty signal: when the model is highly confident in its current answer, further reflection is no longer necessary, and generation of reflection trigger tokens should be proactively suppressed. This constitutes an adaptive braking mechanism driven by the model's intrinsic state.

## Method

### Overall Architecture
CGRS embeds two components into the standard autoregressive decoding process:
1. **Certainty estimation** at logical breakpoints (`\n\n`) during the reasoning process.
2. **Probabilistic suppression** of reflection trigger token generation based on the certainty score.

The entire process requires no retraining and no modification of the model architecture, and can be directly integrated into any autoregressive generation pipeline.

### Key Designs

1. **Certainty Estimation**:

    - **Function**: Quantifies the model's confidence in the current answer at logical breakpoints during reasoning.
    - **Mechanism**:
        - Identifies checkpoints in the reasoning trace using `\n\n` as structural delimiters to mark thought boundaries.
        - At each checkpoint, injects the prompt `**Final Answer: \boxed` to probe a tentative final answer.
        - Quantifies certainty via the token-level entropy of the tentative answer.
    - **Certainty score formula**:
    $C = 1 - \left(\frac{\frac{1}{n}\sum_{i=1}^{n}\mathcal{H}(\mathbf{p}_{\mathbf{a}_i})}{\log(|\mathbf{V}|)}\right)$
      where $\mathcal{H}$ denotes token-level information entropy, $|\mathbf{V}|$ is the vocabulary size, and $\log(|\mathbf{V}|)$ serves as the maximum entropy normalization factor.
    - **Design Motivation**: Low entropy indicates that the model's answer distribution is highly concentrated (high confidence), making further reflection largely redundant. The probe operates independently of the main decoding process and does not affect the primary reasoning trajectory.

2. **Dynamic Reflection Trigger Suppression**:

    - **Function**: Probabilistically blocks the model from generating reflection trigger tokens based on the certainty score.
    - **Reflection trigger token set**: Constructed via frequency analysis, covering four categories:
        - Core hesitation/transition words: Wait, But
        - Alternative-solution markers: Alternatively, Alternative
        - Colloquial contemplation cues: Hmm
        - All variants present in the tokenizer vocabulary
    - **Suppression probability formula**:
    $p = \max\left(0, \frac{C - \delta}{1 - \delta}\right)$
      where $\delta \in [0,1]$ is the confidence threshold (default: 0.9).
    - **Suppression mechanism**: With probability $p$, the logits of trigger token IDs are set to a large negative value, excluding them from sampling.
    - **Design Motivation**:
        - Suppression is only triggered when $C > \delta$, avoiding the inhibition of valid self-correction under low confidence.
        - Probabilistic suppression (rather than deterministic) preserves occasional breakthrough opportunities.
        - The conservative setting of $\delta=0.9$ ensures that suppression only occurs under very high certainty, guaranteeing safety.

3. **Trigger Token Variant Mapping**:

    - A single trigger word (e.g., "Wait") may have multiple variants in the tokenizer (different cases, leading spaces, etc.) mapping to different token IDs.
    - Less frequently occurring variants are filtered out via frequency analysis on real reasoning traces.
    - Separate trigger token sets are constructed for Qwen2Tokenizer and LlamaTokenizerFast.

### Algorithm Flow
1. Compute the current token probability distribution $\mathbf{p}_t$.
2. Perform Bernoulli sampling with probability $p$ to determine whether to suppress.
3. If suppressing, set the logits of all tokens in $S_{trigger}$ to a large negative value and renormalize.
4. Sample the next token from the modified distribution.
5. Upon encountering a checkpoint marker (`\n\n`), execute tentative answer probing and update the certainty score and suppression probability.

## Key Experimental Results

### Main Results

Systematic evaluation across 8 models × 4 benchmarks; representative results are as follows:

| Model | Method | Avg. Accuracy (%) | Avg. Length Reduction (%) | Note |
|------|------|------------|----------------|------|
| Qwen3-8B | Vanilla | 75.3 | - | Baseline |
| Qwen3-8B | TALE | 77.2 | 16.7% | Prompt-based |
| Qwen3-8B | NoThinking | 61.0 | 69.5% | Large accuracy drop |
| Qwen3-8B | DEER | 68.2 | 41.6% | 7% accuracy drop |
| Qwen3-8B | **CGRS** | **75.9** | **29.3%** | Accuracy maintained, 29% reduction |
| QwQ-32B | Vanilla | 80.2 | - | Baseline |
| QwQ-32B | TALE | 77.8 | 9.5% | Minimal reduction |
| QwQ-32B | DEER | 79.7 | 13.3% | Limited reduction |
| QwQ-32B | **CGRS** | **80.8** | **30.5%** | Accuracy improved, 30% reduction |
| DS-R1-Qwen-7B | Vanilla | 66.8 | - | Baseline |
| DS-R1-Qwen-7B | **CGRS** | **65.2** | **41.9%** | Largest reduction |

### Ablation Study

**Certainty-guided vs. fixed-probability suppression** (AMC23, DS-R1-Qwen-7B):

| Configuration | Accuracy (%) | Length | Note |
|------|--------|------|------|
| p=0 (Vanilla) | 87.5 | 5861 | No suppression |
| p=0.25 | 81.7 | 3729 | Fixed probability, −5.8% accuracy |
| p=0.5 | 80.0 | 3266 | Accuracy continues to decline |
| p=1.0 | 76.7 | 2373 | Full suppression, −10.8% accuracy |
| **Certainty-guided (Eq. 2)** | **88.3** | **3406** | No accuracy drop, 41.9% reduction |

**Threshold $\delta$ ablation** (AMC23, DS-R1-Qwen-7B):

| $\delta$ | Accuracy (%) | Length Reduction (%) |
|-----|--------|------------|
| 0.9 | 88.3 | 41.9% |
| 0.5 | ~80 | ~48% |
| 0.1 | 72.5 | 53.7% |

### Key Findings
1. **CGRS achieves the best accuracy–efficiency trade-off across all tested scenarios**: 18.5%–41.9% token reduction with no more than 3% accuracy degradation.
2. **Unique advantage on QwQ-32B**: This model does not use the `</think>` marker, causing baseline methods that rely on it (TALE, DEER, etc.) to be largely ineffective, whereas CGRS—being independent of this marker—still achieves 30.5% compression.
3. **Certainty-guided suppression greatly outperforms fixed probability**: A fixed value of p=0.25 already causes a 5.8% accuracy drop, while certainty-guided suppression yields a slight accuracy improvement at a higher compression rate.
4. **Significant reduction in reflection trigger token frequency**: CGRS effectively reduces the occurrence of words such as "Wait" and "But", and the answer length distribution becomes more concentrated.
5. **Consistency across architectures**: Effective across the Qwen3 family (4B→32B), the DeepSeek-R1-Distill family (Qwen-7B, Llama-8B), and QwQ-32B.

## Highlights & Insights
1. **Precise problem formulation**: The overthinking problem is operationalized by focusing on "trigger tokens" as an actionable intervention point, making an abstract issue concrete.
2. **Minimal yet effective design**: The entire method reduces to "estimate confidence → probabilistically mask trigger tokens", requiring no training and no architectural modification.
3. **Adaptivity**: Certainty estimation evolves dynamically throughout the reasoning process rather than applying uniform compression.
4. **Unique effectiveness on QwQ-32B**: This exposes the implicit reliance of existing methods on the `</think>` token; CGRS is the only method not subject to this constraint.
5. **Convincing case analysis**: Demonstrates that the large number of ineffective "re-verification" steps present in vanilla generation are effectively eliminated by CGRS.

## Limitations & Future Work
1. The certainty probing step (tentative answer generation) itself incurs additional computational overhead, which the paper does not quantify.
2. The trigger token set is manually constructed through frequency analysis on specific models and may require re-analysis for different models.
3. The choice of $\delta=0.9$ lacks theoretical justification, and different tasks may require different thresholds.
4. Validation is limited to mathematical and scientific reasoning tasks, without coverage of code generation, logical reasoning, and other scenarios.
5. Whether entropy is the optimal certainty measure remains an open question; alternative confidence metrics (e.g., answer consistency) may be more robust.

## Related Work & Insights
- **DEER** (Yang et al. 2025): Detects high-confidence intermediate answers based on transition cues (e.g., "Wait") and exits early, but depends on the `</think>` token.
- **Dynasor** (Fu et al. 2025): Requests intermediate answers at fixed token intervals and exits early upon consecutive matches; less efficient than CGRS.
- **NoThinking** (Ma et al. 2025): Directly skips slow-thinking to generate a final answer, incurring severe accuracy loss.
- **Insights**: The certainty-guided idea can be generalized to other scenarios requiring adaptive computation depth, such as MoE routing and adaptive-depth transformers.

## Rating
- Novelty: ⭐⭐⭐⭐ (The trigger suppression concept is concise and novel, though the certainty estimation component is relatively conventional.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 models × 4 benchmarks, with complete ablation studies and case analysis.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure with well-articulated motivation.)
- Value: ⭐⭐⭐⭐⭐ (Plug-and-play design with extremely high practical value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] Efficient Reasoning with Balanced Thinking](../../ICLR2026/model_compression/efficient_reasoning_with_balanced_thinking.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](../../ACL2026/model_compression/judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](../../ICLR2026/model_compression/inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](../../ICLR2026/model_compression/beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)

</div>

<!-- RELATED:END -->
