---
title: >-
  [Paper Note] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling
description: >-
  [ICML 2026][Multimodal VLM][MLLM-as-a-Judge] Ours reveals and formalizes the "Perceptual Judgment Bias" in MLLM-as-a-Judge—where the judge model tends to reward linguistically more fluent answers even when visual evidenc…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "MLLM-as-a-Judge"
  - "Perceptual Judgment Bias"
  - "Visual Perturbation"
  - "GRPO"
  - "Batch Ranking Reward"
date: 2026-05-08
content_hash: b1e9a3731362b5d1
---

# Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling

**Conference**: ICML 2026  
**arXiv**: [2606.02578](https://arxiv.org/abs/2606.02578)  
**Code**: https://perception-judge.github.io/ (Project Page)  
**Area**: Multimodal VLM  
**Keywords**: MLLM-as-a-Judge, Perceptual Judgment Bias, Visual Perturbation, GRPO, Batch Ranking Reward

## TL;DR
Ours reveals and formalizes the "Perceptual Judgment Bias" in MLLM-as-a-Judge—where the judge model tends to reward linguistically more fluent answers even when visual evidence conflicts with the textual narrative. By constructing the perceptually perturbed dataset PPJD and training with GRPO-based batch ranking rewards, a 7B judge significantly outperforms same-sized baselines across multimodal evaluation consistency, point-wise prediction, and batch ranking protocols using only 3k samples.

## Background & Motivation
**Background**: With the proliferation of Multimodal Large Language Models (MLLM) in tasks like VQA and image-to-text generation, using "MLLM-as-a-Judge" for automated evaluation has become the mainstream paradigm to replace human scoring. Representative works like MLLM-as-a-Judge benchmark, LLaVA-Critic, and Flex-Judge primarily follow the "SFT + preference pairs" route, outputting scalar scores, pair preferences, or ranking sequences given $(x_i, r_k)$.

**Limitations of Prior Work**: The authors identify a systematic failure in these MLLM judges—they often assign high scores to responses that are internally self-consistent in text but contain visual descriptions inconsistent with the image. For example, if an image shows a red car and the response claims "the blue car in the figure reflects a sense of technology," the judge might still give a high score due to fluent reasoning. This is not an isolated corner case: on the MLLM-as-a-Judge benchmark, Qwen2.5-VL-7B and Flex-Judge-VL-7B exhibit overall error rates of 30.5% and 23.5%, respectively.

**Key Challenge**: The authors decompose these failures into two independent modes: Mode (a) *insufficient perceptual capability*, where the judge misinterprets the image (failing its own VQA probe); and Mode (b) *response-anchored judgment reasoning*, where the judge can answer correctly when asked about the image in isolation but is misled by the "visual facts" described in the response during the judging process. Table 1 shows that Mode (b) is comparable to or even larger than Mode (a) in magnitude, meaning that even stronger visual encoders only solve half the problem. The fundamental contradiction is that the **perceptual channel and judgment channel are decoupled during evaluation**.

**Goal**: (1) Formally define "Perceptual Judgment Bias" and provide quantifiable diagnostic methods; (2) Construct training data that explicitly decouples perceptual errors from reasoning errors; (3) Design training objectives that force the judge to use perceptual verification as a prerequisite for high rewards instead of relying on textual fluency.

**Key Insight**: Since standard (chosen, rejected) preference pairs mix perceptual and reasoning errors, ours **artificially synthesizes counterfactual responses that are perturbed only in visual attributes while maintaining reasoning fluency** to serve as "traps" during judge training. Simultaneously, the pairwise preference is upgraded to full-order ranking over quadruplets, shifting the supervision signal from local wins/losses to global alignment.

**Core Idea**: Starting from a correct response, counterfactual responses are created including "perceptually perturbed only $r_{r_p}$" and "perceptually + reasoning perturbed $r_{r_{p+r}}$," forming a quadruplet $(x_i, r_c, r_{r_p}, r_{r_{p+r}})$. Within the GRPO framework, a batch ranking reward based on Levenshtein distance is used to force the judge to learn the strict $r_c \succ r_{r_p} \succ r_{r_{p+r}}$ order.

## Method

### Overall Architecture
The method consists of three steps: (1) Formalizing Perceptual Judgment Bias and attributing failures to Mode (a)/(b) using VQA probes; (2) Constructing the *Perceptually Perturbed Judgment Dataset (PPJD)*—extracting 3k high-quality samples from MMPR-v1.2 and generating perceptually perturbed and dual-perturbed versions for each chosen answer; (3) Training with GRPO using a reward function consisting of format validation $\times$ batch ranking score. The final models are denoted as Perception-Judge-Flex and Perception-Judge-Qwen3, based on Flex-Judge-VL-7B and Qwen3-VL-4B-Thinking, respectively.

### Key Designs

1.  **Formalization and Two-Channel Attribution of Perceptual Judgment Bias**:
    - **Function**: Transforms the vague feeling of "why judges are unreliable" into measurable metrics, establishing the rationale for subsequent training objectives.
    - **Mechanism**: Let $\pi^\star(v_i)$ be human-annotated visual facts, $\pi_\text{Judge}(v_i)$ be the judge's own perception of image $v_i$, $\pi_r(v_i)$ be the visual content described in response $r$, and $s_{(x_i, r)}$ be the judgment score. A perceptual judgment error occurs when a visually incorrect response $r_r$ is not penalized relative to a correct response $r_c$ ($s_{(x_i,r_r)} \ge s_{(x_i,r_c)}$). Failures are distinguished between Mode (a) ($\pi_\text{Judge}(v_i) \ne \pi^\star(v_i)$) and Mode (b) ($\pi_\text{Judge}(v_i) = \pi^\star(v_i)$), using direct VQA accuracy as a proxy for $\pi_\text{Judge}$.
    - **Design Motivation**: If only total error rates are considered, researchers might attribute all issues to "weak visual encoders," missing the deeper mismatch. By projecting errors onto a 2D plane via VQA probes, ours reveals that Mode (b) ("seeing it right but scoring it wrong") is as severe as Mode (a) ("seeing it wrong"). This finding directly dictates that training must explicitly supervise the perception-judgment coupling.

2.  **PPJD: Perception-Reasoning Decoupled Quadruplet Dataset**:
    - **Function**: Provides fine-grained, verifiable training supervision to isolate perceptual failures from reasoning failures.
    - **Mechanism**: Starting with (chosen, rejected) pairs from MMPR-v1.2, $r_c$ is retained as the visually and logically correct reference. A generative model applies **perception-only perturbations** to $r_c$: precisely rewriting visual attributes (e.g., color, count, spatial relations) while keeping syntax and reasoning chains intact to obtain $r_{r_p}$. VQA consistency checks are used to discard failed perturbations. Reasoning quality is further degraded to obtain $r_{r_{p+r}}$. Each sample is a quadruplet $(x_i, r_c, r_{r_p}, r_{r_{p+r}})$ with the target preference $r_c \succ r_{r_p} \succ r_{r_{p+r}}$.
    - **Design Motivation**: In general preference sets, chosen and rejected samples often differ in both perception and reasoning. Models might learn that "poor reasoning equals low score" and ignore visual signals. PPJD isolates perceptual perturbations ($r_{r_p}$), explicitly marking responses that are "fluent but visually wrong" as low-score, providing direct supervision for Mode (b).

3.  **Batch Ranking Reward + GRPO**:
    - **Function**: Converts the full-order constraint $r_c \succ r_{r_p} \succ r_{r_{p+r}}$ into a continuous reward signal for stable 7B model learning.
    - **Mechanism**: The reward has two parts. Structure reward $\mathcal{R}_\text{Format}(o_i) \in \{0,1\}$ checks if the output follows the `<think>...</think><answer>...</answer>` format and valid value ranges. The batch ranking reward measures the gap between the predicted permutation $\hat{\bm{\pi}}_i$ and target $\bm{\pi}_i^\star$ using normalized Levenshtein distance: $\mathcal{R}_\text{Batch}(o_i) = 1 - d_\text{Lev}(\hat{\bm{\pi}}_i, \bm{\pi}_i^\star)/\|\bm{\pi}_i^\star\|$. The total reward $\mathcal{R}(o_i) = \mathcal{R}_\text{Format}(o_i) \times \mathcal{R}_\text{Batch}(o_i)$ is used in the GRPO objective:
      $$\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}\big[\tfrac{1}{n}\sum_i \min(r_i\hat{\mathcal{A}}_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{\mathcal{A}}_i) - \beta\, \mathbb{D}_\text{KL}(\pi_\theta\|\pi_\text{ref})\big]$$
      where $\hat{\mathcal{A}}_i$ is the group-normalized relative advantage.
    - **Design Motivation**: Pairwise rewards only provide local "win/loss" signals, potentially leading to global transitive inconsistencies. Normalized full-order ranking rewards force transitive consistency without requiring explicit scalar labels. Structural rewards ensure verifiability—invalid formats receive zero reward, converting "semantic evaluation" into a "verifiable RL" problem.

### Loss & Training
Base models: Flex-Judge-VL-7B and Qwen3-VL-4B-Thinking. Training framework: verl. Data: 3k samples from MMPR-v1.2 processed via PPJD, with strict exclusion of benchmark overlaps. GRPO hyperparameters follow verl defaults.

## Key Experimental Results

### Main Results (MLLM-as-a-Judge benchmark, avg. across 14 tasks)

| Model | Size | Score (↑) | Pair w. Tie (↑) | Pair w.o. Tie (↑) | Batch (↓) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | – | 0.439 | 0.538 | 0.736 | 0.361 |
| Gemini-1.0-Pro-Vision | – | 0.304 | 0.509 | 0.615 | 0.432 |
| LLaVA-Critic | 7B | 0.314 | 0.556 | 0.689 | – |
| Qwen2.5-VL-Instruct | 7B | 0.165 | 0.423 | 0.425 | 0.585 |
| Flex-Judge-VL | 7B | 0.404 | 0.514 | 0.623 | 0.517 |
| Qwen3-VL-Thinking | 4B | 0.419 | 0.543 | 0.663 | 0.498 |
| **Perception-Judge-Flex (Ours)** | 7B | **0.466** | 0.520 | 0.645 | 0.505 |
| **Perception-Judge-Qwen3 (Ours)** | 4B | **0.457** | **0.554** | **0.691** | **0.444** |

Points: Compared to Qwen3-VL-Thinking-4B, ours improves batch evaluation by 11% and score evaluation by 12%. It approaches GPT-4o in point-wise mode and outperforms most proprietary models in batch evaluation. Perception-Judge-Flex-7B reduces total error rate from 23.5% to 14.3% (Mode (a) 9.4→6.7, Mode (b) 14.1→7.6) in the bias diagnosis.

### Ablation Study (10k training samples, Flex-Judge-VL-7B)

| Dataset | Reward Type | Score (↑) | Pair w. Tie (↑) | Pair w.o. Tie (↑) | Batch (↓) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| – (base) | – | 0.404 | 0.514 | 0.623 | 0.517 |
| MMPR-v1.2 | Pairwise | 0.454 | 0.515 | 0.641 | 0.515 |
| PPJD | Pairwise | 0.458 | 0.518 | 0.644 | 0.513 |
| PPJD | **Batch** | **0.476** | **0.518** | **0.648** | **0.500** |

### Key Findings
- *Data is Paramount*: Replacing MMPR-v1.2 with PPJD under the same reward type improves all metrics, showing that data explicitly decoupling perceptual bias is inherently beneficial.
- *Batch Reward > Pairwise*: Using only full-order ranking rewards without scalar scores results in the best point-wise predictions, proving global constraints induce better local preference calibration.
- *Data Efficiency*: Using only 3k PPJD samples outperforms 7B judges trained on 113k LLaVA-Critic samples, highlighting the "information density" of perceptually perturbed data.
- *Dual-Channel Bias Reduction*: Mode (b) error rate nearly halved (14.1%→7.6%), showing that even without massive improvements to the visual encoder, forcing perception-judgment coupling yields significant gains.

## Highlights & Insights
- Elevates "why judges are inaccurate" from an engineering issue to a formal, falsifiable scientific problem. The Mode (a)/(b) dichotomy is a valuable tool for future MLLM-as-a-Judge research.
- The creation of "fluent but visually wrong" counterfactuals ($r_{r_p}$) is the core innovation: it makes the implicit trade-off between language fluency and visual accuracy explicit.
- Combining full-order Levenshtein rewards with GRPO successfully transfers verifiable RL from math/code reasoning to the subjective task of visual evaluation.
- High data efficiency and independence from scalar score supervision make this protocol attractive for resource-constrained scenarios like medical or autonomous driving specialists.

## Limitations & Future Work
- PPJD perturbations are restricted to controllable dimensions (attributes, counting, spatial); coverage of fine-grained errors in complex scenes (e.g., lesion localization) is unknown.
- Batch rewards for larger $K$ may suffer from coarse Levenshtein distance granularity, potentially requiring weighted redesigns.
- The model's perceptual ceiling is still bounded by the base VLM's encoder; if the base is extremely weak in Mode (a), improvements may be limited.
- Evaluation is focused on current benchmarks; generalization to open-ended generation (e.g., quality of long video descriptions) requires further verification.

## Related Work & Insights
- **vs LLaVA-Critic**: LLaVA-Critic uses large-scale SFT (113k pairs). Ours demonstrates that 3k counterfactuals + ranking RL can outperform it, providing a contrast between data volume and supervision density.
- **vs JudgeLRM / Verifiable RL**: While verifiable RL has been used for text reasoning, ours extends this to vision-text consistency by discretizing ranking into verifiable signals.
- **Insights**: (1) Any MLLM-as-a-Judge work should report Mode (a)/(b) decomposition to avoid masking root causes; (2) The "counterfactual data + ranking RL" combination is directly transferable to general reward modeling.

## Rating
- Novelty: ⭐⭐⭐⭐ First to combine bias formalization, PPJD quadruplets, and batch ranking GRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive task tables, bias decomposition, and data efficiency studies.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions and a logical motivation-diagnosis-validation chain.
- Value: ⭐⭐⭐⭐⭐ Highly actionable data and training protocols for teams building MLLM judges.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)
- [\[ACL 2026\] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](../../ACL2026/multimodal_vlm/mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)
- [\[ICML 2026\] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling](beyond_vlm-based_rewards_diffusion-native_latent_reward_modeling.md)
- [\[AAAI 2026\] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias](../../AAAI2026/multimodal_vlm/sage_spuriousness-aware_guided_prompt_exploration_for_mitigating_multimodal_bias.md)
- [\[ICLR 2026\] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](../../ICLR2026/multimodal_vlm/lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)

</div>

<!-- RELATED:END -->
