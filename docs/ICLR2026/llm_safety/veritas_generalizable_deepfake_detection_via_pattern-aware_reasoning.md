---
title: >-
  [Paper Note] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning
description: >-
  [ICLR 2026][LLM Safety][Deepfake Detection] This paper proposes Veritas, an MLLM-based deepfake detector that simulates human authentication reasoning via pattern-aware reasoning (fast judgment → reasoning → planning → s…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Deepfake Detection"
  - "MLLM"
  - "Pattern-Aware Reasoning"
  - "Reinforcement Learning"
  - "HydraFake"
date: 2026-05-08
content_hash: 89e1b1d68b477b84
---

# Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning

**Conference**: ICLR 2026
**arXiv**: [2508.21048](https://arxiv.org/abs/2508.21048)  
**Code**: [https://github.com/EricTan7/Veritas](https://github.com/EricTan7/Veritas)  
**Area**: AI Safety / Multimodal VLM / Deepfake Detection
**Keywords**: Deepfake Detection, MLLM, Pattern-Aware Reasoning, Reinforcement Learning, HydraFake

## TL;DR

This paper proposes Veritas, an MLLM-based deepfake detector that simulates human authentication reasoning via pattern-aware reasoning (fast judgment → reasoning → planning → self-reflection → conclusion). It introduces a two-stage training pipeline (SFT+MiPO cold-start + P-GRPO reinforcement learning) and constructs the HydraFake benchmark with a four-level OOD evaluation protocol. Veritas achieves an average accuracy of 90.7% across cross-forgery and cross-domain scenarios, surpassing the previous SOTA by 6.0%.

## Background & Motivation

**Background**: The dominant paradigm in deepfake detection is training on FF++ and evaluating cross-domain generalization on datasets such as DFDC and CelebDF. Recent MLLM-based approaches (e.g., FFAA, M2F2-Det, FakeVLM) have attempted to introduce interpretability, yet final classification decisions still rely on compact visual models (e.g., CLIP), with MLLMs serving only as post-hoc explainers.

**Limitations of Prior Work**:
   - **Benchmark–Practice Mismatch**: Existing benchmarks use a single training source (FF++ only) and low-quality test images, failing to reflect the real-world challenge of rich training distributions paired with highly variable test distributions.
   - **Poor Cross-Forgery Generalization**: Existing detectors perform reasonably in Cross-Model settings (>90%), but degrade severely in Cross-Forgery (novel forgery types such as face restoration and personalization) and Cross-Domain (real-world social media deepfakes), with most falling below 85%.
   - **Underutilized MLLM Reasoning**: MLLM-based methods predominantly follow a post-hoc paradigm—determining authenticity first and then generating explanations—so the reasoning process does not participate in the decision.

**Key Challenge**: Existing detectors learn artifact patterns specific to known forgery types, lacking the hierarchical reasoning capacity to handle OOD forgeries. Directly applying general-purpose MLLMs to deepfake detection yields poor performance (InternVL3-8B: 58.3%; GPT-4o: 60.8%), due to the absence of task-specific reasoning training data and strategies.

**Goal**:
   - Q1: What reasoning process is most effective for deepfake detection? → Pattern-aware reasoning.
   - Q2: How can a model genuinely learn to reason rather than memorize patterns? → Two-stage training via MiPO + P-GRPO.

**Key Insight**: Inspired by how human experts authenticate media—starting with a rapid intuition (fast judgment), then localizing key artifacts (reasoning), conducting layered analysis for difficult cases (planning), potentially revising initial assessments through deep reflection (self-reflection), and finally synthesizing a conclusion—this work formalizes these five cognitive modes and progressively internalizes them into an MLLM via SFT injection, preference alignment, and reinforcement learning.

**Core Idea**: Explicitly inject structured human-like authentication reasoning into an MLLM, and use a pattern-aware reward mechanism to encourage the model to apply the appropriate reasoning depth at the appropriate moment, enabling end-to-end transparent decision-making.

## Method

### Overall Architecture

Veritas is built upon InternVL3-8B and employs a two-stage training pipeline:

- **Stage 1 — Pattern-Guided Cold-Start**: SFT to inject reasoning format (36K samples), followed by MiPO (Mixed Preference Optimization) to align reasoning quality (3K manually annotated preference pairs).
- **Stage 2 — Pattern-Aware Exploration**: P-GRPO (Pattern-aware Group Relative Policy Optimization) to incentivize adaptive reasoning via online sampling and pattern-aware rewards (9K samples, requiring only binary labels).

The input consists of a face image and a user query; the output is a structured response containing a `<think>` block with pattern tags (`<fast>`, `<reasoning>`, `<planning>`, `<reflection>`, `<conclusion>`) followed by a final authenticity judgment. For simple samples the model may invoke only fast+reasoning+conclusion, while difficult samples trigger planning and reflection.

### Key Designs

1. **HydraFake Dataset and Four-Level Evaluation Protocol**:

    - **Function**: Construct a large-scale deepfake detection benchmark that closely mirrors industrial scenarios.
    - **Mechanism**: 50K real images (from 88 datasets) + 50K forged images (from 36 generative models), covering face swapping, reenactment, entire-face generation, face restoration, relighting, and personalization. The training set contains only three basic forgery types (FS/FR/EFG, 48K images). Evaluation is organized into four levels: In-Domain (14K), Cross-Model (11K; unseen models including FLUX, StarryAI, and MAGI-1), Cross-Forgery (12K; unseen forgery types including attribute editing, generative face swapping, and personalization), and Cross-Domain (15K; unseen data domains and in-the-wild social media deepfakes from GPT-4o, Dreamina, and HailuoAI).
    - **Design Motivation**: Simulate the real-world challenge of abundant training data paired with highly variable test distributions, enabling precise identification of detector weaknesses at different OOD levels.
    - **Quality Control**: Low-quality datasets (DFDC, WDF) are excluded; Qwen2.5-VL-72B is used to generate sample-specific prompts for self-constructed data, and high-quality samples are selected through human filtering.

2. **Pattern-Aware Reasoning Framework**:

    - **Function**: Define five reasoning patterns to simulate the human expert authentication workflow.
    - **Mechanism**: `<fast>` rapid intuitive judgment → `<reasoning>` localization of one to two salient artifacts → `<planning>` hierarchical analysis for difficult samples → `<reflection>` self-revision to support or overturn the initial judgment → `<conclusion>` synthesis of all evidence for a final verdict. The model adaptively selects patterns during inference; simple samples may use only fast+reasoning+conclusion, while challenging samples invoke planning and reflection.
    - **Design Motivation**: Vanilla CoT lacks structured cognitive guidance, leading models to produce superficial reasoning. Experiments demonstrate that pattern-aware reasoning outperforms flexible reasoning by 6.2% on Cross-Forgery and 3.3% on Cross-Domain.
    - **Key Distinction from Post-hoc Explanation**: Post-hoc approaches determine the answer first and then construct justifications, so reasoning does not participate in the decision (accuracy is 8.4% lower). In Veritas, the reasoning process directly drives the final judgment.

3. **Mixed Preference Optimization (MiPO)**:

    - **Function**: Align reasoning quality after SFT to prevent the model from memorizing patterns rather than reasoning.
    - **Mechanism**: A mixed dispreferred dataset $\mathcal{D}_2$ is constructed with two types of negative samples—$s_l^\phi$ (correct answer but shallow/insufficiently detailed reasoning) and $s_l^\psi$ (incorrect answer). Positive samples $s_w$ are annotated by human experts. Training adopts a DPO-style objective: $\mathcal{L}_2 = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(s_w|q)}{\pi_{\text{SFT}}(s_w|q)} - \beta\log\frac{\pi_\theta(s_l|q)}{\pi_{\text{SFT}}(s_l|q)})]$
    - **Design Motivation**: A purely SFT-trained model tends to produce outputs that are correct but shallowly reasoned. The inclusion of $s_l^\phi$—"correct answer, insufficient reasoning"—as negative samples compels the model to develop fine-grained reasoning. Ablations confirm: removing $s_l^\phi$ causes CF −1.1% and CD −0.8%; removing $s_l^\psi$ causes model collapse to 60.8%.
    - **Distinction from Standard DPO**: MiPO introduces a novel category of dispreferred data—"correct answer but poor reasoning"—whereas conventional DPO typically uses only incorrect answers as negative samples.

4. **Pattern-Aware GRPO (P-GRPO)**:

    - **Function**: Use reinforcement learning to incentivize adaptive reasoning depth, encouraging the model to proactively employ planning and reflection when needed.
    - **Mechanism**: For each query, $G=4$ responses are sampled and evaluated via pattern-aware rewards. The final reward is $R = R_{\text{pattern}} + \lambda_1 R_{\text{ref}} \cdot \mathbb{I}(\mathcal{C}=1) + \lambda_2 R_{\text{fmt}}$.
    - **$R_{\text{pattern}}$ Design**: Correct answer with planning/reflection → +2.0; correct answer without advanced patterns → +1.0; incorrect answer without advanced patterns → 0.0; incorrect answer with planning → −0.5; incorrect answer with reflection → **−1.0** (the heaviest penalty, as reflection is the strongest pattern and its misuse is most costly).
    - **$R_{\text{ref}}$ (Reflection Quality Reward)**: An external reward model (UnifiedReward-Qwen-3B) evaluates whether the reflection introduces a new perspective rather than repeating prior findings; this reward is granted only when the answer is correct.
    - **Design Motivation**: Unlike methods that use length rewards to encourage longer reasoning, the authors argue that absolute reasoning length is unimportant; what matters is using the appropriate cognitive pattern at the appropriate moment. Penalties for overthinking prevent the model from overusing reflection.

### Loss & Training

- **Data Annotation Pipeline**: A three-step decoupled annotation process—(1) human experts summarize an artifact taxonomy (perceptual structural anomalies / subtle low-level artifacts / cognitively anomalous physical-law violations); (2) annotation is decoupled into three specialized steps completed automatically by MLLMs; (3) 36K SFT samples are generated.
- **SFT Stage**: LoRA (rank=128, α=256), 3 epochs, lr=5e-5, batch size=64.
- **MiPO Stage**: 3K manually annotated preference pairs, 2 epochs, DPO objective, $\beta=0$.
- **P-GRPO Stage**: 9K images (binary labels only), G=4 sampling, lr=1e-6, 2 epochs, temperature=1.0.
- Each stage inherits the model weights from the previous stage.

## Key Experimental Results

### Main Results

| Method | Type | ID | Cross-Model | Cross-Forgery | Cross-Domain | Avg Acc |
|--------|------|----|-------------|---------------|--------------|---------|
| F3Net (ECCV'20) | Small visual model | 85.3 | 84.3 | 69.6 | 67.2 | 73.2 |
| UniFD (CVPR'23) | Small visual model | 82.7 | 87.5 | 72.1 | 72.8 | 78.0 |
| ProDet (NeurIPS'24) | Small visual model | 90.5 | 92.3 | 73.5 | 74.0 | 80.6 |
| Co-SPY (CVPR'25) | Small visual model | 86.3 | 93.2 | 85.8 | 75.1 | 84.7 |
| Effort (ICML'25) | Small visual model | 94.7 | 90.7 | 86.0 | 63.9 | 82.2 |
| GPT-4o | Closed-source MLLM | 53.5 | 59.5 | 58.4 | 67.4 | 60.8 |
| Gemini-2.5-Pro | Closed-source MLLM | 72.2 | 81.5 | 82.4 | 73.8 | 78.9 |
| FakeVLM (NeurIPS'25) | MLLM detector | — | 77.0 | 75.7 | 78.5 | 77.3 |
| SIDA-7B (CVPR'25) | MLLM detector | — | 87.9 | 67.2 | 73.0 | 76.3 |
| Veritas-mini | Ours (restricted training) | — | 93.0 | 78.9 | 84.3 | 85.8 |
| Veritas (cold-start) | Ours (cold-start only) | 96.8 | 95.8 | 80.6 | 82.2 | 87.3 |
| **Veritas (full)** | **Ours** | **97.3** | **98.6** | **90.3** | **82.2** | **90.7** |

Veritas improves average accuracy by **6.0%** over the previous best Co-SPY (84.7%), by **32.4%** over the base model InternVL3-8B (58.3%), and by **11.8%** over the strongest closed-source model Gemini-2.5-Pro.

### Ablation Study

| Configuration | ID | CM | CF | CD | Avg | Note |
|---------------|----|----|----|----|-----|------|
| Full (Pattern-aware + MiPO + P-GRPO) | 97.3 | 98.6 | 90.3 | 82.2 | 92.1 | Complete model |
| w/o P-GRPO (cold-start only) | 96.9 | 98.4 | 87.4 | 80.1 | 90.7 | Remove RL; CF −2.9% |
| w/o MiPO (SFT + P-GRPO) | — | — | 87.4 | 80.1 | 90.7 | MiPO provides better RL initialization |
| w/o Reasoning | 97.8 | 93.3 | 73.0 | 69.5 | — | No reasoning; CF **drops 17.3%** |
| Post-hoc Explanation | 96.3 | 95.0 | 79.0 | 76.8 | — | Post-hoc paradigm |
| Flexible Reasoning (vanilla CoT) | 96.2 | 94.3 | 81.2 | 76.8 | 87.1 | Free-form reasoning; CF −9.1% |
| w/o `<reflection>` | 97.0 | 97.2 | 82.5 | 77.3 | 88.5 | **Most impactful pattern** |
| w/o `<planning>` | 96.7 | 96.9 | 85.0 | 80.1 | 89.7 | Largest impact on CM |
| w/o `<fast>` | 97.3 | 98.8 | 86.9 | 79.1 | 90.5 | Minor impact |
| w/o `<conclusion>` | 97.2 | 98.2 | 86.2 | 79.0 | 90.1 | Provides consistent gains |
| MiPO w/o $s_l^\phi$ | 96.9 | 98.6 | 89.2 | 81.4 | 91.5 | Remove "correct but poor reasoning" negatives |
| MiPO w/o $s_l^\psi$ | 65.3 | 64.8 | 58.6 | 54.3 | 60.8 | **Model collapse** |

### Key Findings

- **`<reflection>` is the most critical reasoning pattern**: Its removal reduces CF from 87.4% to 82.5% (−4.9%) and CD from 80.1% to 77.3% (−2.8%). Self-reflection enables the model to identify previously unseen forgery artifacts, which is essential for OOD generalization.
- **Cold-start is a prerequisite for successful RL**: Skipping cold-start and applying RL directly (with equivalent data) leads to training instability due to low-quality rollouts; all purely RL configurations underperform the two-stage pipeline.
- **$s_l^\phi$ in MiPO (correct answer, poor reasoning) is not strictly necessary but is important for OOD performance**: Its removal still yields correct answers but with shallower reasoning, resulting in CF −1.1% and CD −0.8%; whereas $s_l^\psi$ (incorrect answers) is foundational to preference learning, and its removal causes collapse.
- **Model scaling**: InternVL3-2B already achieves CF 87.3% (cost-efficient), and scaling from 8B to 14B yields +2.9% on CF (CM 99.3%), demonstrating good scalability.
- **Robustness**: Veritas achieves 87.4% under JPEG compression at QF=50 (vs. Effort at 66.3%) and 84.3% under Gaussian blur at σ=2.0 (vs. Co-SPY at 77.0%), with no such augmentations used during training.
- **Reasoning quality evaluation**: In an MLLM-as-Judge evaluation (GPT-4o and Gemini-2.5-Pro as judges), Veritas (w/ MiPO) achieves ELO 1359, substantially outperforming Gemini-2.5-Pro (967) and GPT-4o (785).

## Highlights & Insights

- **Elegant pattern-aware reward design**: Rather than naively encouraging longer reasoning, the reward incentivizes using the appropriate pattern at the appropriate moment, with escalating penalties for overthinking (planning incorrect: −0.5; reflection incorrect: −1.0). This fine-grained pattern-level reward design is transferable to any task requiring structured reasoning, such as medical diagnostic reasoning or legal case analysis.
- **"Correct answer, poor reasoning" as an overlooked training signal**: Conventional DPO uses only incorrect answers as negative samples. MiPO introduces the category of "correct answer but insufficiently detailed reasoning," compelling the model not merely to be right, but to be right in the right way. This insight is broadly applicable to any MLLM task requiring interpretable reasoning.
- **HydraFake's four-level evaluation reveals the true bottleneck of detectors**: Existing methods already perform well in Cross-Model settings (>90%), while Cross-Forgery and Cross-Domain remain the genuine challenges. This finding reframes the evaluation perspective for the field.
- **Two-stage decoupled design with complementary roles**: MiPO ensures high-quality rollouts that provide a favorable initialization for P-GRPO by improving initial reasoning quality; P-GRPO further explores the reasoning space via online sampling. Both stages are individually effective, and their combination yields additive gains (CF: SFT 87.4 → +MiPO or +P-GRPO → +Both 90.3).

## Limitations & Future Work

- **Non-trivial annotation cost**: MiPO requires human expert annotation of 3K preference pairs, and the three-step SFT annotation pipeline—despite MLLM assistance—still demands substantial manual verification, limiting scalability.
- **Restricted to face deepfakes**: HydraFake and Veritas target face deepfakes only and do not cover general AIGC detection (e.g., landscape, object, or scene synthesis); generalization to non-face domains remains unexplored.
- **Cross-Domain performance has room for improvement**: A CD accuracy of 82.2% implies that approximately one in five in-the-wild deepfakes are missed. In particular, accuracy on FFIW samples is only 78.5%, and on InfiniteYou (CD) only 58.6% (cold-start alone: 55.9%).
- **Inference efficiency**: The latency of generating reasoning chains with an MLLM is substantially higher than that of compact CNN-based detectors, requiring careful consideration of the latency–accuracy tradeoff in deployment.
- **Reward model dependency**: The reflection quality reward relies on an external model (UnifiedReward-Qwen-3B), whose inherent biases may propagate into training; additionally, using a 3B model to evaluate the outputs of an 8B model introduces a scale mismatch.

## Related Work & Insights

- **vs. FFAA / M2F2-Det**: These methods employ MLLMs for interpretability but rely on compact models such as CLIP for final classification, constituting an "MLLM-assisted" paradigm. Veritas enables the MLLM to produce judgments and reasoning chains end-to-end, realizing truly reasoning-driven detection. FFAA achieves only 64.0% on HydraFake; M2F2-Det only 63.2%.
- **vs. FakeVLM / SIDA**: FakeVLM adopts a post-hoc explanation paradigm with broad coverage (77.3%); SIDA-7B performs strongly in Cross-Model settings (97.3%) but collapses on Cross-Forgery (63.3%). Veritas achieves balanced performance across all scenarios through pattern-aware reasoning.
- **vs. Effort / Co-SPY**: The strongest compact visual model detectors. Effort leads on in-domain (94.7%) but achieves only 63.9% on Cross-Domain; Co-SPY is more balanced (84.7%) but remains significantly below Veritas. The Cross-Domain weakness of small models underscores the indispensable value of MLLM general knowledge for OOD generalization.
- **vs. DeepSeek-R1 / s1 and other general reasoning methods**: Veritas's pattern-aware reward is a domain-specialized adaptation of general GRPO, demonstrating that in specialized domains, domain-driven reasoning patterns outperform generic CoT. This principle is transferable to visual tasks requiring structured reasoning, such as medical image analysis and remote sensing object detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing structured reasoning patterns into deepfake detection and designing pattern-aware rewards is conceptually novel; however, the core training components (SFT+DPO+GRPO) are not original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Includes a self-constructed large-scale benchmark, comparisons with 10 SOTA detectors, 6 general MLLMs, and 6 MLLM detectors, along with detailed ablations, robustness evaluations, and reasoning quality assessments. Highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically structured with well-designed figures and a smooth narrative; some sections contain dense formula presentation.
- **Value**: ⭐⭐⭐⭐⭐ — Contributes both a benchmark (HydraFake) and a method (Veritas); the cold-start model is open-sourced for community customization, representing a significant advancement for the deepfake detection field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](../../CVPR2026/llm_safety/tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)
- [\[CVPR 2026\] Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](../../CVPR2026/llm_safety/pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)
- [\[ICLR 2026\] From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
