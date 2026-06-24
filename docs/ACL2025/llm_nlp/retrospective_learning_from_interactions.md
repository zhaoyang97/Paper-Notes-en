---
title: >-
  [Paper Note] Retrospective Learning from Interactions
description: >-
  [ACL 2025][LLM (Other)][Implicit feedback] Proposes the ReSpect method, which enables multimodal LLMs to self-improve by retrospectively decoding users' implicit feedback signals in multi-turn interactions without any external annotations, improving the task completion rate from 31% to 82% over thousands of human-machine interactions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Implicit feedback"
  - "interactive learning"
  - "continual learning"
  - "multimodal LLM"
  - "reference games"
date: 2026-05-08
content_hash: 7593cb2f4a9cd113
---

# Retrospective Learning from Interactions

**Conference**: ACL 2025  
**arXiv**: [2410.13852](https://arxiv.org/abs/2410.13852)  
**Code**: [https://lil-lab.github.io/respect](https://lil-lab.github.io/respect)  
**Area**: Other  
**Keywords**: Implicit feedback, interactive learning, continual learning, multimodal LLM, reference games

## TL;DR

Proposes the ReSpect method, which enables multimodal LLMs to self-improve by retrospectively decoding users' implicit feedback signals in multi-turn interactions without any external annotations, improving the task completion rate from 31% to 82% over thousands of human-machine interactions.

## Background & Motivation

Multi-turn human-machine interactions naturally contain rich implicit learning signals. When the LLM's response does not meet expectations, users might:
- Reformulate the request
- Express frustration (e.g., "not again")
- Switch to other tasks

When the LLM's response is correct, users might:
- Express approval (e.g., "great!")
- Directly proceed to the next goal

These signals are **task-agnostic**—even if someone does not understand the specific task, they can judge whether the agent is performing well from these dialogue cues. The key insight is that **these implicit feedback signals occupy an relatively restricted subspace of natural language**, enabling the LLM to recognize these signals even when performing poorly on the task itself.

Compared to common methods like RLHF, ReSpect's uniqueness lies in:
- No need for annotator feedback
- No need for a stronger model as a judge
- No need to ask users to explicitly provide feedback
- Relying solely on natural interactions during deployment

## Method

### Overall Architecture

ReSpect operates iteratively across multi-round deployments:
1. **Deployment Phase**: The model interacts with real users, recording the context, predicted probabilities, and subsequent interactions for each action.
2. **Retrospective Phase**: The model retrospectively analyzes the subsequent user responses for each action to decode implicit feedback.
3. **Training Phase**: The model is retrained using the decoded feedback signals.
4. Repeat the process.

### Key Designs

1. **MultiRef Interaction Scenario**:

    - A generalized version of the reference game: a speaker (human) and a listener (model) jointly observe a set of tangram shapes.
    - The speaker guides the listener to select a subset of unknown size—the combinatorial solution space is $2^n$ (much larger than the classic $n$).
    - Abstract tangram shapes from the KiloGram dataset are used, naturally leading to vague descriptions and rich multi-turn interactions.
    - Human speakers can send text messages, while the model listener can only select/deselect images.
    - A timeout of 20 turns is considered a failure.

2. **Implicit Feedback Decoder**:

    - Employs the **model itself** (not a stronger model) to evaluate the feedback of each action in past interactions.
    - Based on text prompts, inputs include: interaction context $x$, model action $\hat{a}$, and subsequent interaction $\bar{f}$.
    - Output: positive / neutral / negative (ternary), or positive / negative (binary).
    - **Does not access any privileged information** (such as correct answers or overall task success).
    - The precision of the feedback decoder consistently remains above 90%.

3. **Three Learning Methods**:

    - **FFT (Filtered Fine-tuning)**: Fine-tuning solely on positive data points, using cross-entropy + label smoothing.
    - **REINFORCE**: Using policy gradients, mapping feedback to numerical rewards (positive=1, neutral=0, negative=-0.1), and weighting negative feedback with inverse propensity scoring.
    - **KTO (Kahneman-Tversky Optimization)**: Using positive/negative data points while skipping neutral ones, with a positive-to-negative ratio of approximately 5:4.

4. **Continual Learning Setting**:

    - Approximately 330 interactions and ~2400 turns per round.
    - Training and evaluation are not separated—deployed interactions are used both for evaluation and for training the next round.
    - Cumulative data training: each round uses all historical data $D_{\leq \rho}$.
    - FFT and RL are trained from scratch each round; KTO continues fine-tuning from the previous round's checkpoint.

### Loss & Training

- Base model: IDEFICS2-8B
- Fine-tuned using LoRA
- Initial policy $\pi_{\theta_0}$: A seed model fine-tuned on 25 human-human games
- Entropy regularization and length normalization are added to all objective functions to mitigate overfitting
- Validation set is used for model selection

## Key Experimental Results

### Main Results

**Interaction success rate across rounds**:

| System | Round 1 | Round 2 | Round 3 | Round 6 (b-fft only) |
|------|-------|-------|-------|-----------------|
| b-fft | 31% | 55% | 72% | **82%** |
| t-fft | 33% | 49% | 65% | - |
| b-rl | 28% | 47% | 60% | - |
| t-rl | 29% | 43% | 57% | - |
| b-kto | 30% | 44% | 40%↓ | - |
| Control (Initial policy redeployed) | - | - | - | 33% |
| Human-Human Interactions | 100% | 100% | 100% | 100% |

**b-fft Turn-level Metrics (6 rounds)**:

| Metric | Round 1 | Round 6 | Change |
|------|-------|-------|------|
| Interaction Success Rate | 31% | 82% | +51% |
| Turn-level Exact Match | 31% | 53% | +22% |
| Average turns per interaction | 8.9 | 6.7 | -2.2 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Binary vs. Ternary Feedback | Binary slightly better | Ternary is more conservative, labeling more as neutral |
| FFT vs. RL vs. KTO | FFT > RL > KTO | Positive-only signals are more effective than positive + negative signals |
| User Adaptation vs. Model Improvement | Control group 31%→33% | User adaptation cannot account for the 51% improvement |

### Key Findings

1. **b-fft Performs Best**: Utilizing positive feedback signals alone combined with filtered fine-tuning achieves the greatest improvement.
2. **Exploiting Negative Feedback Signals Remains an Open Question**: Systems employing negative feedback signals (RL, KTO) perform worse than those utilizing positive-only signals.
3. **Feedback Decoder is highly robust**: Even as the data distribution shifts across rounds, the precision consistently remains above 90% with a low false positive rate.
4. **User Behavior Indeed Changes**: Vocabulary size and utterance length first decrease and then increase, alongside a reduction in reset signals—however, this does not explain the model's improvement (as confirmed by control experiments).
5. **KTO is Unstable in Continual Learning**: b-kto even experiences degradation and produces invalid outputs.
6. **Gap with Human Performance Persists**: 82% vs. 100%, potentially due to insufficient long-term credit assignment.

## Highlights & Insights

- **True Self-Improvement**: Does not rely on stronger models, external annotations, or task-specific verifiers—it learns purely from natural interactions.
- **Task-Agnostic Feedback Decoding**: The feedback decoder is designed to recognize general linguistic cues rather than task-specific signals.
- **Real-world Experimental Authenticity**: 7,230 real human-agent interactions, 55,004 turns, and $11,180 in MTurk costs, all executed in real deployments.
- **Ingenious MultiRef Scenario Design**: Balances task difficulty ($2^n$ combinatorial space), controllability, and the naturalness of multi-turn interactions.
- **Rigorous Control Experiments**: Rule out user adaptation as a confounding factor by redeploying the initial policy in the final round.

## Limitations & Future Work

- MultiRef is a controlled experimental scenario; generalization to open-ended dialogue (such as summarization and QA) requires further validation.
- The model's improvements on MultiRef fail to generalize to other tasks, and may even harm general capabilities.
- Relying solely on scalar rewards, more expressive feedback decoding (e.g., natural language explanations) might further enhance learning outcomes.
- Insufficient long-term credit assignment—learning in later turns is more difficult as actions must be attributed to a longer history.
- The feedback decoder is not synchronized/updated alongside the policy model, which may underestimate the potential of the approach.
- Lacks evaluation against adversarial user scenarios—malicious users could "poison" the learning process with deceptive feedback.

## Related Work & Insights

- Key difference from RLHF (Ouyang et al., 2022): No pairwise preference annotations are required, eliminating extra labeling costs.
- Complementary to the work of Kojima et al. (2021)—the latter learns from how humans execute instructions, whereas ReSpect learns from human reactions.
- The continual learning version of KTO (Ethayarajh et al., 2024) performs poorly—suggesting a need to improve optimization strategies in continual settings.
- **Practical Insight**: In any human-AI system deployment, implicit feedback serves as "free" learning signals that should be systematically utilized.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm of extracting implicit feedback from natural interactions for self-improvement is highly innovative and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 rounds of real human-agent deployment, 7 system variants, multidimensional evaluation, and strict control experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly organized paper, complete technical details, and highly readable visualizations of experimental results.
- **Value**: ⭐⭐⭐⭐⭐ Unrevealing a neglected yet omnipresent learning signal, which has profound implications for the continual improvement of interactive AI systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](../../NeurIPS2025/llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results](revisiting_uncertainty_quantification_evaluation_in_language_models_spurious_int.md)

</div>

<!-- RELATED:END -->
