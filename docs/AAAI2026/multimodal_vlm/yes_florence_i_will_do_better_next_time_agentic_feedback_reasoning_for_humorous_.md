---
title: >-
  [Paper Note] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection
description: >-
  [AAAI 2026][Multimodal VLM][humorous meme detection] This paper proposes FLoReNce, a framework that models humorous meme understanding as a closed-loop control system. Through a feedback loop consisting of a Judge agent…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "humorous meme detection"
  - "feedback-loop reasoning"
  - "non-parametric knowledge base"
  - "PID controller"
  - "vision-language model"
date: 2026-05-08
content_hash: 7b05c8614a858e0e
---

# Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection

**Conference**: AAAI 2026
**arXiv**: [2601.07232](https://arxiv.org/abs/2601.07232)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: humorous meme detection, feedback-loop reasoning, non-parametric knowledge base, PID controller, vision-language model

## TL;DR

This paper proposes FLoReNce, a framework that models humorous meme understanding as a closed-loop control system. Through a feedback loop consisting of a Judge agent, a PID controller, and a non-parametric knowledge base, the system retrieves similar past experiences at inference time to modulate prompts, enabling a frozen VLM to perform adaptive reasoning without fine-tuning, substantially improving both prediction accuracy and explanation quality.

## Background & Motivation

Humorous memes combine visual and textual cues to convey satire, social commentary, and related meanings, posing unique challenges for AI systems that must understand intent rather than surface-level associations. Existing approaches suffer from two major limitations:

**Static classification methods** (e.g., MemeCLIP, CLIP+classifier) can fuse multimodal features but only learn pixel-word correlations, failing to capture the deeper incongruity underlying humor.

**Reasoning-based methods** (e.g., Chain-of-Thought, multi-agent debate frameworks) improve interpretability but remain fundamentally open-loop: once the model produces erroneous or shallow reasoning, no correction, reflection, or feedback-based adaptation mechanism exists.

Human humor comprehension is inherently dynamic—interpretations are continuously refined through critique, social feedback, and exposure to new cultural contexts, much like a control system regulating its output to minimize error. Without such a closed-loop mechanism, AI reasoning tends to oscillate between over- and under-predicting humor, lacking self-correction capability.

**Core Motivation**: Model humor reasoning as a closed-loop state-space system, enabling the VLM to learn from its own interpretation history and progressively stabilize its perception of humor.

## Method

### Overall Architecture

FLoReNce (**F**eedback-**Lo**op **Re**asoner with **N**on-parametri**c** **E**xperience) consists of two phases:

- **Closed-loop learning phase (training)**: VLM reasoning → Judge evaluation → PID controller generates control signal → stored in knowledge base
- **Open-loop inference phase (testing)**: retrieve similar experiences from KB → assemble control vector → modulate prompt → VLM performs adaptive reasoning

### Key Designs

1. **Visual-Language Reasoning Agent**

   A frozen Qwen2.5-VL-32B-Instruct is used as the reasoning agent. Given a meme image $x^{img}$, OCR text $x^{text}$, and a guiding prompt $p$, the agent outputs:
   - Humor probability $\hat{y} \in [0,1]$
   - Textual reasoning justification $r$
   - Hidden-layer embedding $emb \in \mathbb{R}^d$ (mean-pooled from the last hidden layer)

   The prompt is mapped from a control vector $c$ via a prompt-mapper $\Psi$: $p = \Psi(c)$, translating continuous feedback into interpretable prompt instructions (e.g., "check for sarcasm," "verify setup→punchline structure").

2. **Judge Agent and PID Controller**

   The Judge $\mathcal{J}_\phi$ has access to ground-truth labels and outputs three signals:
   - Scalar error $e_t = y - \hat{y}_t$
   - Textual critique $fb\_text_t$
   - Low-dimensional feedback vector $f_t \in \mathbb{R}^3$ along three interpretable axes (sarcasm / narrative structure / layout cues)

   The PID controller converts the error history into a stabilized action:

   $$u_t = K_P e_t + K_I \sum_{\tau=1}^{t} e_\tau + K_D (e_t - e_{t-1})$$

   The final control vector is $c_t = [u_t, f_t^\top, k_t^\top]^\top \in \mathbb{R}^7$, where $k_t$ is the KB compact signal.

   **Design Motivation**: The PID controller provides numerical stability—the proportional term responds to current error, the integral term eliminates steady-state bias, and the derivative term suppresses oscillation—causing reasoning behavior to converge progressively like a controlled system.

3. **Feedback-informed Knowledge Base**

   Unlike conventional RAG, the KB stores complete reasoning experiences rather than raw training samples:

   $$\mathcal{K} \leftarrow \mathcal{K} \cup \{(id, emb_t, r_t, fb\_text_t)\}$$

   Each entry simultaneously stores the embedding, reasoning text, and Judge critique, making retrieval "experience-aware"—the system remembers not only *what it has seen* but also *how it was corrected*.

   At inference time, cosine similarity over query embeddings is used to retrieve top-$K$ neighbors, which are aggregated into the compact KB signal $k \in \mathbb{R}^3$.

### Loss & Training

- **No parameter updates**: no model weights are updated throughout the process. The closed-loop learning phase solely constructs the KB; all adaptation is realized through prompt modulation.
- **PID hyperparameters**: $(K_P, K_I, K_D) = (1.0, 0.5, 0.1)$
- KB is stored in JSONL format; retrieval uses CPU tensors
- Maximum 128 tokens generated per reasoning output
- Hardware: NVIDIA L40S (48 GB)

## Key Experimental Results

### Main Results

Dataset — PrideMM: 5,063 text-embedded images related to the LGBTQ+ movement, split 85/5/10 for train/val/test.

| Model | Backbone | Accuracy | Macro-F1 | MCC | RQ(%) |
|-------|----------|----------|----------|-----|-------|
| Visual Only | ResNet50+MLP | 66.08 | 61.67 | 0.33 | - |
| Text Only | T5+MLP | 67.85 | 66.10 | 0.36 | - |
| MemeCLIP | CLIP | 78.30 | 76.99 | 0.57 | - |
| MOMENTA | CLIP | 73.57 | 69.92 | 0.47 | - |
| PromptHate | RoBERTa | 73.77 | 73.46 | 0.49 | - |
| LoReHM | LLaVA-34B | 70.09 | 64.07 | 0.39 | 64.8 |
| COLA | GPT-3.5-Turbo | 53.25 | 59.34 | 0.07 | 58.5 |
| MiND | Qwen2.5-VL-32B | 54.45 | 50.43 | 0.05 | 52.6 |
| **FLoReNce (K=1)** | Qwen2.5-VL-32B | 73.40 | **77.08** | 0.48 | 74.0 |
| **FLoReNce (K=3)** | Qwen2.5-VL-32B | 73.73 | **77.36** | 0.48 | 74.3 |
| **FLoReNce (K=5)** | Qwen2.5-VL-32B | 73.80 | **77.33** | 0.48 | 74.4 |

Key finding: FLoReNce's Macro-F1 gain exceeds its Accuracy gain, indicating that the feedback KB is particularly effective for harder categories and improves class-balanced performance.

### Ablation Study

| Variant | Accuracy | Macro-F1 | MCC |
|---------|----------|----------|-----|
| Base VLM (no KB, no control) | 64.20 | 58.10 | 0.22 |
| + KB only (no control) | 68.30 | 63.90 | 0.35 |
| + Control only (no KB) | 72.00 | 69.40 | 0.44 |
| $- f_t$ (PID+KB, feedback vector removed) | 73.00 | 70.20 | 0.46 |
| $-$ PID (KB signal only) | 72.60 | 70.00 | 0.45 |
| **Full FLoReNce** | **73.73** | **77.36** | **0.48** |

### Key Findings

- KB alone yields +4.10% Acc / +5.80% F1, demonstrating that retrieving feedback experiences is intrinsically valuable.
- Control alone is slightly below KB only but substantially above the base, suggesting control is more effective when grounded in meaningful memory signals.
- All three components (PID + KB + $f_t$) are indispensable: Judge critiques carry information that cannot be recovered from embeddings alone.
- Performance is highly stable from $K=1$ to $K=10$, indicating that once the KB is populated, even minimal retrieval suffices to produce consistently feedback-aligned reasoning.
- Using the same Qwen2.5-VL-32B backbone as MiND, FLoReNce improves F1 from 50.43% to 77.36% (+27%), validating the substantial contribution of the closed-loop feedback mechanism.

## Highlights & Insights

1. **Cybernetic perspective**: the paper is the first to formalize humor understanding as a closed-loop state-space control problem, with the PID controller providing numerical stability.
2. **Experience-aware non-parametric memory**: the KB stores complete "reasoning + correction" experiences rather than raw samples, making retrieval inherently self-corrective.
3. **Zero fine-tuning adaptation**: VLM weights remain frozen throughout; all adaptation is achieved via prompt modulation, making the approach resource-efficient and plug-and-play.
4. **Strong practical utility**: even $K=1$ achieves F1 = 77.08%, demonstrating robustness to KB size.

## Limitations & Future Work

- Validation is limited to a single dataset (PrideMM); generalizability remains to be examined.
- The case study exposes difficulties in distinguishing "satire critiquing power" from "mockery targeting marginalized groups."
- PID hyperparameters are manually set (though stable in experiments); adaptive PID tuning may yield further gains.
- KB size grows linearly with training scale; retrieval efficiency at large scale requires optimization.
- Accuracy is slightly below MemeCLIP (73.73 vs. 78.30) despite higher F1; whether the two approaches can be combined warrants investigation.

## Related Work & Insights

- **vs. RAG**: RAG stores raw documents; FLoReNce stores reasoning experiences together with Judge feedback.
- **vs. Self-Refine**: Self-Refine is open-loop self-improvement; FLoReNce is a closed-loop, stateful evolution driven by Judge feedback.
- PID control has seen limited application in NLP; this work pioneers a new paradigm of "cybernetics + VLM prompting."
- The framework is extensible to other highly subjective multimodal tasks such as hate speech detection, sarcasm recognition, and misinformation detection.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (the combination of a cybernetic perspective and a feedback knowledge base is highly original)
- Experimental Thoroughness: ⭐⭐⭐ (single dataset; ablation is complete but cross-domain validation is absent)
- Writing Quality: ⭐⭐⭐⭐ (formulation is clear; the cybernetic framing is well articulated)
- Value: ⭐⭐⭐⭐ (provides a generalizable approach to adaptive reasoning for subjective tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CAMU: Context Augmentation for Meme Understanding](trace_textual_relevance_augmentation_and_contextual_encoding_for_multimodal_hate.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)
- [\[AAAI 2026\] See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay](see_symbolize_act_grounding_vlms_with_spatial_representations_for_better_gamepla.md)
- [\[ACL 2026\] All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction](../../ACL2026/multimodal_vlm/all_changes_may_have_invariant_principles_improving_ever-shifting_harmful_meme_d.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](../../NeurIPS2025/multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)

</div>

<!-- RELATED:END -->
