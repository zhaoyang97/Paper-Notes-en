---
title: >-
  [Paper Note] Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks
description: >-
  [ACL 2026][LLM Safety][LVLM unlearning] The authors point out that existing LVLM unlearning benchmarks (FIUBench / MLLMU-Bench / CLEAR) fail to truly memorize fictional identities during the Stage 1 fine-tuning phase…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LVLM unlearning"
  - "memorization failure"
  - "multi-hop curse"
  - "multi-view synthesis"
  - "Exposure metric"
date: 2026-05-08
content_hash: 8d9734520989aa1e
---

# Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks

**Conference**: ACL 2026  
**arXiv**: [2605.03759](https://arxiv.org/abs/2605.03759)  
**Code**: https://huggingface.co/datasets/herbwood27/Remem  
**Area**: LLM Safety / Privacy / Multimodal Unlearning  
**Keywords**: LVLM unlearning, memorization failure, multi-hop curse, multi-view synthesis, Exposure metric

## TL;DR
The authors point out that existing LVLM unlearning benchmarks (FIUBench / MLLMU-Bench / CLEAR) fail to truly memorize fictional identities during the Stage 1 fine-tuning phase, rendering all Stage 2 "forgetting" evaluations invalid. They diagnose the root cause as "insufficient data repetition + multi-hop curse" and propose ReMem—featuring 100 QA pairs and 100 multi-view images per identity, a 70%:30% single-hop/multi-hop mix, and a new Exposure internal probability metric—to rebuild unlearning evaluation on the foundation of "successful learning."

## Background & Motivation

**Background**: LVLMs (e.g., LLaVA-1.5) inadvertently memorize private information from training data, triggering regulatory requirements such as the "right to be forgotten" (GDPR). Machine Unlearning has become a critical field. The community typically employs "fictional identities" for two-stage evaluation: Stage 1 involves fine-tuning the model to memorize sensitive attributes (name, email, disease, etc.) of $N$ fake persons; Stage 2 applies unlearning algorithms to remove a specific subset.

**Limitations of Prior Work**: The authors performed a comprehensive diagnosis of Stage 1 in existing benchmarks using ROUGE-L (verbatim memorization), GPT-judge (semantic memorization), and EM (precise PII matching). They found extremely low EM scores—meaning the models never actually memorized core PII. Since memorization fails in Stage 1, evaluating "erasure" in Stage 2 is groundless, causing the entire benchmark chain to fail.

**Key Challenge**: For simplicity, existing benchmarks provide very few QA pairs per identity (FIUBench: 20/ID, MLLMU: 1/ID) and rely almost entirely on multi-hop questions (e.g., "What is the address of the person in the image?"). This violates two machine learning principles: (1) Carlini et al. demonstrated that memorization strength is proportional to sample repetition, and too few QA pairs fail to penetrate parameters; (2) the "multi-hop curse" indicates that multi-hop reasoning cannot be acquired without single-hop scaffolding.

**Goal**: (a) Empirically diagnose Stage 1 failures using internal state analysis such as causal tracing and Min-k%-prob; (b) propose a new benchmark, ReMem, to solidify Stage 1 memorization; (c) introduce the Exposure metric to quantify erasure depth at the probability distribution level.

**Key Insight**: Instead of continuing to compete over Stage 2 algorithms, it is better to return to the fundamental diagnostic problem: "whether facts are actually encoded into the parameters."

**Core Idea**: Ensure "successful memorization" (Stage 1 reliability) before discussing "forgetting" (Stage 2 efficacy). By utilizing data scaling + single-hop scaffolding + multi-view images, Stage 1 EM is brought close to 100%, and erasure evaluation is shifted from surface-level generation to internal probabilities via Exposure.

## Method

### Overall Architecture

ReMem reconstructs the two-stage LVLM unlearning evaluation pipeline:
- **Stage 1 (reliable memorization)**: Fine-tune the LVLM on the ReMem dataset, ensuring EM is maximized (each fictional identity must be truly memorized by the model).
- **Stage 2 (unlearning + evaluation)**: Apply various unlearning algorithms to the forget set $D_f$, evaluating ROUGE/EM/Exposure across $D_f$, retain eval $D_r'$, and held-out test $D_t$. OOD tests use brand-new templates to prevent lexical overfitting.

Data scale: 20 fictional identities $\times$ 100 QA $\times$ 100 images = 2,560 samples.

### Key Designs

1. **QA Set Construction based on Scaling Law + Single-hop Scaffolding**:

    - **Function**: Ensures Stage 1 truly "memorizes" by controlling the number of QA pairs per identity and the single-hop/multi-hop ratio.
    - **Mechanism**: Toy experiments scanning 20 $\to$ 200 QA/ID show that ROUGE/EM increase monotonically with the number of QA pairs, starting to overfit around 160. Scanning single-hop ratios from 0% $\to$ 100% reveals that a 70% single-hop ratio is the peak for EM on both single-hop and multi-hop tests. Single-hop questions ("What is X's address?") serve as scaffolding for multi-hop questions ("What is the address of the person in the image?"), establishing the $(s,r)\to a$ mapping before connecting visual grounding $v\to s$.
    - **Design Motivation**: FIUBench/MLLMU use only multi-hop questions, requiring the model to learn visual recognition and attribute recall simultaneously. This composite task is too difficult; analogous to a human learning "Xiao Ming's address is XX Street" before being able to identify "the address of the person in this picture."

2. **Multi-view Image Synthesis + Strict Quality Control**:

    - **Function**: Prevents the model from overfitting to a single reference image and establishes a "context-consistent identity concept."
    - **Mechanism**: Uses Nano Banana with an anchor image to generate 100 images/ID with randomized poses, clothing, and backgrounds. ArcFace cosine similarity is used to ensure face consistency. A four-step human review process is applied (artifact removal / cross-modal alignment / identity recognizability / ethical screening). The test set $D_t$ uses entirely new visual templates to specifically test OOD generalization.
    - **Design Motivation**: CLEAR and others use 20 images/ID but with minimal visual variation, leading models to memorize specific images rather than the person. ReMem requires the model to identify the same identity under significant visual variation for true memorization.

3. **Exposure Internal Probability Metric**:

    - **Function**: Quantifies PII erasure depth from the model's internal token probability distribution, distinguishing "surface refusal" from "true erasure."
    - **Mechanism**: For a target PII string $a=t_1t_2...t_n$, define $\text{Exposure}(a)=\sum_i \log P(t_i|\text{prefix})$; a lower value indicates cleaner erasure. This is combined with Min-k% probability ($k=10$) to check the worst-case token, and IE (indirect estimation effect) from multimodal causal tracing to detect retrieval circuits.
    - **Design Motivation**: Traditional ROUGE/EM only observe the generation layer; a model might learn a "I don't know" refusal while the internal PII remains extractable via prompt injection. Exposure directly examines the distribution quality, closer to real-world privacy threat models.

### Loss & Training
Stage 1 uses standard LVLM SFT (LLaVA-1.5-7B, etc.). Stage 2 evaluation covers several categories of unlearning algorithms, including Gradient Ascent, Knowledge Distillation (Kurmanji et al.), Refusal Alignment, and Negative Likelihood, all evaluated consistently on $D_f / D_r' / D_t$ using the same benchmarks.

## Key Experimental Results

### Main Results (Stage 1 Memorization Comparison)

| Benchmark | Images/ID | QA/ID | ROUGE-L | GPT-score | EM (PII) |
|---|---|---|---|---|---|
| FIUBench | 1 | 20 | Medium | Medium | Extremely Low |
| MLLMU-Bench | 1 | 1 | Low | Low | Extremely Low |
| CLEAR | 20 | 20 | Medium | Medium | Low |
| **ReMem** | **100** | **100** | **Near 100% Upper Bound** | **Near 100% Upper Bound** | **Near 100% Upper Bound** |

Radar charts show that ReMem reaches the 100% target line across all three metrics, while the other three benchmarks show significant inward collapse.

### Ablation Study

| Configuration | EM (single-hop) | EM (multi-hop) | Description |
|---|---|---|---|
| 20 QA/ID | Low | Extremely Low | Insufficient data, below memorization threshold |
| 100 QA/ID, 0% Single-hop | Low | Low | Multi-hop curse dominates |
| 100 QA/ID, 70% Single-hop | **Peak** | **Peak** | ReMem default configuration |
| 100 QA/ID, 100% Single-hop | High | Medium | No multi-hop training data |
| 200 QA/ID, 70% Single-hop | High (Overfit) | High | PPL rebounds, starts to overfit |

### Key Findings
- **Scaling law is real**: EM grows almost linearly with the number of QA pairs until 160, after which it overfits, indicating that PII memorization requires "sufficient but not excessive" repetition.
- **70% single-hop is the sweet spot**: Teaching "$(s,r)\to a$" first, followed by "$v\to s\to a$", allows single-hop scaffolding to assist multi-hop learning—consistent with insights from Chain-of-Thought training to "teach atomic steps first."
- **Causal tracing visualization**: Base models show clear retrieval circuits for real public figures in high IE layers. Fine-tuned FIUBench/MLLMU models show scattered IE, proving fictional identities were never internalized.

## Highlights & Insights
- **Diagnosis-driven benchmark reconstruction**: Identifying "Stage 1 failure" via ROUGE/EM/GPT-judge triangulation + Min-k%-prob + causal tracing, and then addressing the root cause. This "Diagnosis $\to$ Root Cause $\to$ Reconstruction" pipeline is a valuable model for benchmarks beyond unlearning.
- **70:30 single-hop/multi-hop ratio**: A highly transferable insight that could be extended to multi-hop QA training (HotpotQA, MultiHopRAG) and agentic reasoning training, which might benefit from "teaching single steps before composite ones."
- **Exposure metric**: Moves privacy assessment from "can it generate" to "is it present in the probability distribution," providing a more robust measure against prompt injection / decoding-time attacks.

## Limitations & Future Work
- **Ours acknowledges**: Currently validated only on LLaVA-1.5-7B; whether larger LVLMs (Gemini-vision, Qwen2-VL) share the same scaling threshold remains untested.
- **Personal observation**: The scale of 20 fictional identities is still small. Industrial unlearning often involves thousands of user records; the optimal single-hop ratio might shift after scaling.
- **Improvement ideas**: Cross-identity confusion tests could be added (does deleting one person's info affect other similar identities?); Exposure could be extended to the token-span level for more granular assessment of partial PII erasure.

## Related Work & Insights
- **vs FIUBench / MLLMU-Bench**: They assume Stage 1 success; this work refutes that assumption. ReMem fixes the foundation via 100 QA $\times$ 100 images.
- **vs CLEAR**: CLEAR expanded to 20 images/ID, but QA pairs remain scarce and lack single-hop data. This work pushes both dimensions to their limits.
- **vs Carlini 2021 (membership inference)**: Uses "PPL + Min-k%" as a tool but with a different goal—the former confirms memorization exists, while this work uses it to detect Stage 1 failure and assess erasure depth.

## Rating
- Novelty: ⭐⭐⭐⭐ Framework reconstruction rather than a new algorithm, but the diagnosis-repair chain is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five layers of evidence (three metrics + internal state + causal tracing), with independent ablations on scaling and ratios.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear "problem $\to$ evidence $\to$ root cause $\to$ solution" narrative; Figure 1 radar chart is highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Effectively invalidates several unlearning benchmarks and provides a replacement that will likely become the de facto standard for future LVLM unlearning work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures](../../ICLR2026/llm_safety/rethinking_benign_relearning_syntax_as_the_hidden_driver_of_the_safety_tax.md)
- [\[ICML 2026\] Understanding Generalization and Forgetting in In-Context Continual Learning](../../ICML2026/llm_safety/understanding_generalization_and_forgetting_in_in-context_continual_learning.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](../../ICLR2026/llm_safety/revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)
- [\[ACL 2026\] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem](modeling_llm_unlearning_as_an_asymmetric_two-task_learning_problem.md)

</div>

<!-- RELATED:END -->
