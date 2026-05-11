---
title: >-
  [Paper Note] Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models
description: >-
  [AAAI 2026][LLM Reasoning][Large Reasoning Models] This paper systematically analyzes abstention failures in Large Reasoning Models (LRMs) when confronted with unanswerable math problems. It finds that LRMs possess suffi…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Large Reasoning Models"
  - "Abstention Failure"
  - "Unanswerable Questions"
  - "Cognitive Monitoring"
  - "Inference-Time Intervention"
date: 2026-05-08
content_hash: be46a7a5858b9034
---

# Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models

**Conference**: AAAI 2026
**arXiv**: [2508.18760v3](https://arxiv.org/abs/2508.18760v3)
**Code**: [https://github.com/nju-websoft/AbstentionReasoning](https://github.com/nju-websoft/AbstentionReasoning)
**Area**: LLM Reasoning / Trustworthy AI
**Keywords**: Large Reasoning Models, Abstention Failure, Unanswerable Questions, Cognitive Monitoring, Inference-Time Intervention

## TL;DR
This paper systematically analyzes abstention failures in Large Reasoning Models (LRMs) when confronted with unanswerable math problems. It finds that LRMs possess sufficient internal cognitive capacity to recognize unsolvability (linear probe classification accuracy >80%), yet their external behavior remains biased toward forced answering. A two-stage approach combining cognitive monitoring and inference-time intervention is proposed, improving abstention rates from 16–54% to 60–92% without degrading reasoning performance on answerable questions.

## Background & Motivation
Large Reasoning Models (LRMs) such as DeepSeek-R1 and Qwen3 demonstrate strong performance on complex reasoning tasks. However, in practice, not every user query has a valid answer — for instance, a math problem may omit necessary conditions. Ideally, an LRM should respond with "I don't know" and provide an explanation, rather than fabricating conditions to force an answer.

Empirical testing reveals that **most LRMs fail to correctly abstain on more than half of unanswerable questions**, exposing a critical trustworthiness issue: strong reasoning capability does not imply knowing when to stop.

## Core Problem
**Why do LRMs fail to abstain correctly when faced with unanswerable questions? Do they genuinely not recognize unsolvability, or do they recognize it but fail to act accordingly? How can abstention behavior be improved without compromising normal reasoning performance?**

## Method

### Overall Architecture
The approach consists of two stages:
1. **Cognitive Monitoring**: During inference, a pre-trained linear probe monitors the model's internal hidden states in real time to determine whether the current question is unanswerable.
2. **Inference-Time Intervention**: When the probe's "unanswerable" signal exceeds a threshold, a guiding prompt is injected to encourage abstention, and an early-exit strategy is activated.

Input: A math problem + a prompt instructing the model to respond "I don't know" if the question is unanswerable.
Output: A correct answer (for answerable questions) or "I don't know" with an explanatory reason (for unanswerable questions).

### Key Designs

1. **Taxonomy of Three Abstention Failure Types**:

    - **Hallucinated Answer**: The LRM fabricates missing conditions (e.g., assuming an unmentioned cost) and produces a seemingly complete but incorrect solution.
    - **Cognitive Fixation**: The LRM enters an infinite loop of restructuring and attempting, failing to terminate reasoning even after 10,000 tokens.
    - **Correct Abstention**: The LRM recognizes the question as unsolvable and responds with "I don't know."

   A key finding is that even in failure cases, LRMs **often already recognize the unanswerable nature of the problem within intermediate reasoning steps**.

2. **Validation of Internal–External Misalignment**:

    - **Behavioral level**: By forcing the LRM to produce an intermediate answer at reasoning pause points (i.e., "wait" tokens), the study finds that over 50% of cognitive fixation cases correctly abstain at that moment.
    - **Representational level**: A simple linear probe $p_\theta(x_l^c) = \sigma(\langle\theta, x_l^c\rangle)$ is trained on attention outputs $x_l^c$ to classify answerable vs. unanswerable inputs. Classification accuracy increases steadily as reasoning progresses, reaching over 80% at the final step (AUROC 0.87–0.97).

3. **Two-Stage Intervention Mechanism**:

    - **Cognitive Monitoring**: The linear probe is applied at the end of semantic units (e.g., "wait" tokens), aggregating prediction probabilities from all preceding tokens via averaging; intervention is triggered when the aggregate exceeds threshold $t$.
    - **Guiding Prompt + Early Exit**: A mild guiding text is injected to remind the model that "this question may be unanswerable," encouraging abstention over forced answering. An early-exit strategy is simultaneously employed to prevent cognitive fixation.

### Loss & Training
Linear probe training: 2,000 pairs of answerable/unanswerable math problems; 1,000 token-level activations sampled per problem; trained for 75 epochs with batch size 16,384 and learning rate 3e-5. The optimal layer is selected per model (ranging from layer 17 to 30). Thresholds are set to 0.6 for SUM and 0.5 for UMWP.

## Key Experimental Results

| Model | Method | Abstention Rate↑ (SUM) | Reason Accuracy↑ | Answerable Accuracy | Abstention Rate↑ (UMWP) |
|-------|--------|----------------------|-----------------|--------------------|-----------------------|
| R1-Distill-Llama-8B | Vanilla | 16.9% | 14.4 | 61.9 | 30.7% |
| R1-Distill-Llama-8B | Ours | **60.9%** | **53.2** | 60.9 | **54.7%** |
| R1-Distill-Qwen-7B | Vanilla | 21.1% | 19.4 | 69.7 | 47.7% |
| R1-Distill-Qwen-7B | Ours | **73.9%** | **61.9** | 67.3 | **77.3%** |
| Qwen3-8B | Vanilla | 47.2% | 41.9 | 60.9 | 80.0% |
| Qwen3-8B | Ours | **75.3%** | **64.4** | 61.6 | **87.3%** |
| Qwen3-14B | Vanilla | 54.2% | 48.2 | 66.6 | 82.3% |
| Qwen3-14B | Ours | **78.2%** | **69.0** | 65.0 | **92.7%** |

### Ablation Study
- **Guiding Prompt vs. Early Exit**: The guiding prompt contributes more to correct abstention (its removal causes the largest drop in abstention rate), while early exit primarily reduces cognitive fixation. Removing the guiding prompt leads to an increase in hallucinated answers, as the model is compelled to produce a response.
- **Monitoring Strategy Comparison**: Latent-representation probe > direct behavioral judgment > consistency judgment > confidence judgment. The latent-representation signal is most stable; direct behavioral judgment is simple and effective but overly aggressive and may harm performance on answerable questions.
- **Intervention Effect**: Post-intervention, abstention confidence increases from 77–91 to 87–99, and abstention rates more than double.
- **Failure Type Decomposition**: The two failure types benefit from different strategies — cognitive fixation is primarily addressed by early exit, while hallucinated answers are primarily addressed by the guiding prompt.
- **Token Usage**: The method reduces average token consumption by 30–50% by eliminating futile reasoning steps.

## Highlights & Insights
- **The Key Insight of "Knowing but Not Acting"**: LRMs do not fail to recognize unanswerable questions; rather, they exhibit a "know but do not act" pattern. This internal cognition–external behavior misalignment is an important and previously underexplored phenomenon.
- **Analysis-Driven Method Design**: Three-quarters of the paper is dedicated to analysis, with the method emerging naturally from the findings — the existence of an internal signal is first established, then a probe is designed to capture it, and finally an intervention is designed to leverage it.
- **Lightweight and Training-Free**: Only a linear probe (with a few thousand parameters) is trained; inference-time intervention requires only text injection. No fine-tuning of the LRM itself is needed.
- **Differentiated Handling of Two Failure Modes**: Hallucinated answers and cognitive fixation have distinct mechanisms, and the proposed method addresses both effectively with complementary contributions.
- **Positive Correlation Between Abstention Rate and Reason Accuracy**: Improved abstention is not merely a blanket refusal but is accompanied by more accurate explanations of why the question cannot be answered.

## Limitations & Future Work
- **Restricted to Mathematical Reasoning**: Experiments are conducted exclusively on unanswerable math problems; generalization to commonsense reasoning, scientific reasoning, and other domains remains unverified.
- **Probe Requires Labeled Data**: Training the linear probe still requires 2,000 answerable/unanswerable pairs, which may be difficult to obtain in new domains.
- **Manual Threshold Tuning**: Different datasets require different thresholds (0.5 vs. 0.6) without an adaptive mechanism.
- **Risk of Over-Abstention**: Accuracy on answerable questions occasionally drops slightly (e.g., R1-Distill-Qwen-14B from 70.4 to 67.9), indicating the presence of false positives.
- **Training-Time Alignment Not Explored**: The paper identifies training-time alignment as future work; the current approach is an inference-time patch rather than a fundamental solution.

## Related Work & Insights

| Method | Core Idea | Key Difference from This Work |
|--------|-----------|-------------------------------|
| **Dynasor-CoT** | Intermediate answer consistency: exits when the same answer appears three consecutive times | Focuses solely on consistency without distinguishing abstention signals; early exit may induce more hallucinated answers |
| **DEER** | Early exit when confidence exceeds 0.95 | Similar issue — no abstention-guiding prompt at exit, so the model tends to produce an answer rather than "I don't know" |
| **SUM/AbstentionBench** | Benchmarks for evaluating abstention capability | Provides evaluation only without a solution; this work performs both analysis and remediation |

The core advantage of this work lies in going beyond early exit by **explicitly encouraging abstention** as a legitimate option through guiding prompts, and using a latent-representation probe for more precise triggering.

## Inspiration & Connections
- The finding of "internal cognition–external behavior misalignment" has broad implications for understanding LLM interpretability — models internally "know" much that they do not express.
- The linear probe methodology is transferable to other scenarios requiring monitoring of LLM internal states, such as hallucination detection, uncertainty quantification, and harmful output monitoring.
- Implication for AI safety: evaluating an LLM's cognitive capabilities solely from its outputs is insufficient; analysis of internal representations is also necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ The "knowing but not acting" research angle is novel, with a thorough and systematic analysis of LRMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five models, two datasets, multiple baselines, four monitoring strategy comparisons, and detailed ablations — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The analysis→findings→method narrative logic is exceptionally clear, with rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides substantive value for trustworthy AI and LRM reliability; the method is simple and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Text-to-Scene with Large Reasoning Models](text-to-scene_with_large_reasoning_models.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](../../CVPR2026/llm_reasoning/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[AAAI 2026\] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)
- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](../../ICLR2026/llm_reasoning/towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)

</div>

<!-- RELATED:END -->
