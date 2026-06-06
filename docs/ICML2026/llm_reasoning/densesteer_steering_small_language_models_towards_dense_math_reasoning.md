---
title: >-
  [Paper Note] DenseSteer: Steering Small Language Models towards Dense Math Reasoning
description: >-
  [ICML 2026][LLM Reasoning][dense reasoning] Observing that stronger models use fewer CoT steps but maintain higher information density per step (Dense Reasoning)…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "dense reasoning"
  - "steering vector"
  - "Small Language Models"
  - "GSM8K"
  - "inference-time intervention"
date: 2026-05-08
content_hash: d152d6bbe4f4d0cb
---

# DenseSteer: Steering Small Language Models towards Dense Math Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.29247](https://arxiv.org/abs/2605.29247)  
**Code**: https://github.com/oyy2000/DenseSteer  
**Area**: LLM Reasoning / SLM / Activation Steering  
**Keywords**: dense reasoning, steering vector, Small Language Models, GSM8K, inference-time intervention

## TL;DR
Observing that stronger models use fewer CoT steps but maintain higher information density per step (Dense Reasoning), DenseSteer utilizes GPT-5.1 to rewrite an SLM's own sparse solutions into "information-dense" in-distribution positive samples. By forming contrastive pairs with the original solutions, a steering vector is extracted via mean difference at intermediate layers (≈ L17) of the residual stream. This method achieves stable performance gains across math benchmarks like GSM8K, MATH500, AMC, and AIME without training or increasing token-level NLL.

## Background & Motivation

**Background**: The mainstream approach to enabling multi-step mathematical reasoning in Small Language Models (SLM, ≤ 3B) is knowledge distillation: using large models to generate long CoTs and then fine-tuning the SLM (Short / Long CoT, Mix-Long, Mix-Large, etc.).

**Limitations of Prior Work**: (1) Distillation requires thousands of teacher samples and intensive training resources; (2) More critically, a "learnability gap" exists—SLMs cannot easily digest traces from strong teachers, which significantly raises token-level NLL and causes distribution mismatch. Figure 3(a) in the paper quantifies this: the NLL of Qwen2.5-7B traces on Qwen2.5-3B is higher than the self-likelihood, and cross-family traces (Llama-3.2-8B → Qwen2.5-3B) are even higher.

**Key Challenge**: To borrow reasoning capabilities from large models without deviating from the SLM's own generation prior. While steering vector interventions are lightweight (requiring as few as 50 pairs), using large model hidden states directly as positive samples pushes the target model outside its own language manifold, leading to output collapse.

**Goal**: To identify "in-distribution positive samples" that carry the superior reasoning structure of strong models while remaining within the target SLM's generation prior.

**Key Insight**: The authors empirically observe in the Qwen2.5 family (Figure 1) that stronger models have **fewer** reasoning steps $N_\text{steps}$ but **more** tokens per step $\rho = N_\text{tokens} / N_\text{steps}$. This indicates that the structural characteristic of "strong reasoning" is **Dense Reasoning**: fewer jumps with higher information density per jump, which reduces the accumulation of intermediate errors.

**Core Idea**: Instead of using an alien teacher as a positive sample, GPT-5.1 is used to perform a minimal rewrite of the SLM's **own** sparse solutions into "semantically invariant, higher density" dense versions (Dense-Rewriting). These form in-distribution contrastive pairs from which a mean-difference steering vector is extracted and injected into the intermediate residual stream.

## Method

### Overall Architecture
Input: 50 calibration problems → Target SLM generates original sparse solutions $x_\text{neg}$ → GPT-5.1 rewrites them into $x_\text{pos}$ using a dense-rewriting prompt (maintaining semantics, merging steps, increasing density) → For each pair, the difference in residual activations of the last token at each layer $\ell$ is calculated and averaged to obtain $v_\ell$ → During inference, at a selected layer $\ell^*$, the residual stream is modified at each decoding step $t$ via $\tilde h_{\ell^*, t} = h_{\ell^*, t} + \lambda \cdot v_{\ell^*}$. This process requires no parameter updates and only 50 calibration pairs.

### Key Designs

1.  **Dense Reasoning Metric & Dense-Rewriting Positive Sample Construction**:
    - **Function**: Operationalizes "better reasoning" into measurable and rewritable structural attributes, producing positive samples that are in-distribution for the target model.
    - **Mechanism**: Reasoning Density is defined as $\rho = N_\text{tokens} / N_\text{steps}$, where steps are delimited by double newlines. GPT-5.1 then merges redundant steps and increases the info-density of the target SLM's own solutions while preserving semantics. Figure 3(b) shows that the NLL of these rewrites is close to the self-likelihood baseline and far lower than that of 7B teacher traces, proving the samples are indeed in-distribution.
    - **Design Motivation**: Distilling traces from 7B/8B teachers raises NLL due to the learnability gap. Rewriting the model's own solutions ensures distributional compatibility while injecting the structural signal of "concise yet dense" reasoning—a fundamental difference between DenseSteer and InFamilySteer or classic CAA.

2.  **Mean Difference Extraction (Last-Token Aggregation)**:
    - **Function**: Compresses $N=50$ contrastive pairs into a single directional vector $v_\ell$ per layer.
    - **Mechanism**: Only the residual activation of the **last token** $h_\ell(x)[-1]$ at layer $\ell$ is extracted for each sample, avoiding token alignment issues caused by varying lengths of dense and sparse sequences. Then, $v_\ell = \frac{1}{N}\sum_{i=1}^{N}\bigl(h_\ell(x_\text{pos}^{(i)})[-1] - h_\ell(x_\text{neg}^{(i)})[-1]\bigr)$.
    - **Design Motivation**: Compared to token-wise alignment or full-sequence pooling, the last token provides a sequence-level summary without alignment noise. The mean difference inherits the simplicity of CAA (Panickssery et al.) and stably captures the "dense vs. sparse" direction even with extremely limited data.

3.  **Residual Stream Injection at Intermediate Layers with Scaled $\lambda$**:
    - **Function**: Continuously adds $v_\ell$ to the residual stream at target layer $\ell^*$ during inference to guide decoding towards a dense style.
    - **Mechanism**: At each decoding step, $\tilde h_{\ell^*, t} = h_{\ell^*, t} + \lambda \cdot v_{\ell^*}$ is applied. $\lambda$ is searched on a held-out set within $[-20, 20]$. Layer sensitivity analysis (Figure 4) shows that early layers (e.g., L6) are unresponsive; intermediate layers (L16 / L17) provide the strongest and most stable control over step count and token count; later layers (L27 / L35) tend to worsen performance or increase verbosity. Figures 5/6 further show that at L17, accuracy increases and NLL decreases monotonically within $\lambda \in [0, 10]$.
    - **Design Motivation**: Early layers learn low-level features and are insensitive to high-level "reasoning structure." Later layers are too close to logits, where intervention conflicts with formed output trajectories, causing compensatory verbosity. Intermediate layers match the areas where high-level semantic features aggregate (as reported in Scaling Monosemanticity by Templeton et al.), making them natural sites for structural features like "reasoning density."

### Loss & Training
**Training-free**. The only "hyperparameter search" involves selecting $\ell^*$ and $\lambda$ for each target model on a GSM8K training subset. Greedy decoding is fixed at the generation end with a max length of 2048. The calibration set consists of 50 problems non-overlapping with the evaluation set.

## Key Experimental Results

### Main Results

Results for Qwen-2.5-3B-Instruct across 5 math benchmarks (GSM8K / MATH500 / AMC / Olympiad / AIME, Avg. is sample-weighted):

| Method | GSM8K | MATH500 | AMC | Olympiad | AIME | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Zero-shot CoT | 83.6 | 63.0 | 42.5 | 20.0 | 0.0 | 61.2 |
| Prompt Engineering (dense style) | 20.0 | 32.8 | 30.0 | 9.8 | 6.7 | 19.8 |
| Short CoT (Distillation) | 79.9 | 58.6 | 30.0 | 18.1 | 6.7 | 57.8 |
| Long CoT (Distillation) | 82.5 | 49.8 | 25.0 | 12.7 | 0.0 | 55.9 |
| Mix-Large (Strongest Distillation) | 83.7 | 61.6 | 37.5 | 21.0 | 6.7 | 61.3 |
| InFamilySteer (7B trace as positive) | 85.3 | 59.8 | 37.5 | 20.0 | 0.0 | 61.4 |
| **DenseSteer (Ours)** | 84.8 | **64.6** | **42.5** | 20.7 | **10.0** | **62.5** |

On Llama-3.2-3B-Instruct, DenseSteer achieves an Avg. of 52.7, comparable to the strongest distillation baseline Mix-Long (54.2), but **requires zero training and only 50 pairs** (vs. 2000+ teacher samples for distillation).

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Layer = L6 (Early) | Steps/tokens barely change with $\lambda$ | Early layers learn low-level features, insensitive to reasoning structure |
| Layer = L17 (Intermediate) | Steps and tokens decrease; accuracy increases | Optimal injection point; stable gain for $\lambda \in [0, 10]$ |
| Layer = L35 (Late) | Steps/tokens increase; accuracy unstable | Conflicts with output trajectories, causing compensatory verbosity |
| Prompt Engineering (Same prompt, no steering) | Avg. 19.8 vs DenseSteer 62.5 | SLMs fail to follow complex rewrite prompts, skipping CoT entirely |
| InFamilySteer (7B trace as pos) | Avg. 61.4 | Close to DenseSteer, validating NLL-based selection; requires larger in-family model |
| LogiQA (OOD Logical Reasoning) | 44.22 → 58.22 (DenseSteer) | "Dense reasoning" is not limited to math and transfers to logic |
| MMLU / BBH CoT / HotpotQA | Parity with baseline | No significant degradation on general tasks; intervention side-effects are controllable |

### Key Findings
-   **In-distribution > Strong Teacher**: The NLL of Dense-Rewriting is close to self-likelihood, whereas even same-family 7B traces have higher NLL (Figure 3). This quantifies the "learnability gap" and explains why DenseSteer outperforms distillation baselines on hard problems like AIME.
-   **Intermediate Layers are the Sweet Spot**: Intervention at L17 results in monotonic accuracy increases and NLL decreases, aligning with findings that intermediate layers are rich in high-level semantic features.
-   **Prompt Engineering Fails on SLMs** (Avg. 19.8): Giving the same "dense" instructions as a prompt to a 3B model causes it to skip CoT. This suggests that **structural signals must be injected at the representation layer rather than via text instructions**, highlighting the advantage of representation-level intervention.

## Highlights & Insights
-   **"Borrowing structure while avoiding distribution shift"** is the most valuable paradigm here: teacher models define "quality," but positive samples must be produced by the student. This can be extended to alignment, style transfer, and safety where SLMs are limited by the learnability gap.
-   **Reasoning Density $\rho = N_\text{tokens} / N_\text{steps}$** is a very cheap proxy metric—calculated just by counting double newlines—yet it effectively captures the structural trait of "fewer steps, high information." It can be applied to RL reward shaping, distillation data filtering, or CoT quality evaluation.
-   **NLL as a "Distribution Compatibility" Filter**: Using the target model to calculate NLL for candidate positive samples to select the lowest NLL ones can replace contrastive sample selection in any steering method.

## Limitations & Future Work
-   DenseSteer can only reorganize reasoning capabilities **already latent** within the model; it cannot inject new knowledge or skills for problems requiring external facts.
-   Evaluated primarily on math and some logic; transfer to more complex multi-step tasks (coding, agent planning, multimodal reasoning) and larger model families (>8B) is yet to be explored.
-   Dense-Rewriting relies on commercial GPT-5.1, shifting the "teacher dependency" from training to sample construction. Using open-source self-distillation or a small rewriter would make the method truly independent.
-   Main results focus on N=50 and greedy decoding; stability under sampling or temperature > 0, and effectiveness on very long reasoning traces (e.g., R1-style 8k+ tokens), have not been verified.

## Related Work & Insights
-   **vs. CAA (Panickssery et al., 2024)**: CAA uses behaviorally/semantically opposite samples (e.g., safe vs. unsafe). DenseSteer uses "sparse vs. dense rewrites of the same solution," pivoting the contrastive signal from semantics to structure and using NLL for explicit distribution filtering.
-   **vs. SEAL (Chen et al., 2025a)**: SEAL is also training-free steering, but focuses on "steering **away from** redundant reflection patterns" (subtraction). DenseSteer focuses on "steering **towards** the dense reasoning subspace" (addition) with a completely different sample construction logic.
-   **vs. Knowledge Distillation**: Distillation requires 2000+ teacher samples and GPU training; DenseSteer matches or exceeds distillation baselines with 50 pairs and one inference-time addition, suggesting "reasoning structure" is more of a representation-level direction than a parameter-level issue.
-   **vs. Skip-Thinking (Chen et al., 2025b) / Phi-4-Mini-Reasoning**: These achieve "dense reasoning" through training data design (chunked CoT, custom recipes). DenseSteer reaches the same structural goal at inference time without modifying parameters.

## Rating
-   Novelty: ⭐⭐⭐⭐ Shifting contrastive pairs from "semantic opposition" to "self-dense-rewriting" with NLL-driven filtering is a clean and generalizable setup.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Covers 2 model families × 3 scales × 5 math benchmarks + OOD logic/general tasks; extensive layer and $\lambda$ sensitivity analysis, though variance for sampling is missing.
-   Writing Quality: ⭐⭐⭐⭐ Clear motivation-observation-method chain. The NLL comparison in Figure 3 provides strong visual evidence for the "learnability gap."
-   Value: ⭐⭐⭐⭐ Provides a practical "50-pair, zero-training" baseline for SLM reasoning enhancement that is deployment-friendly; the "borrow structure, ignore distribution" methodology is applicable to alignment and style control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners](../../ACL2026/llm_reasoning/rsat_structured_attribution_makes_small_language_models_faithful_table_reasoners.md)
- [\[ACL 2026\] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](../../ACL2026/llm_reasoning/legaldrill_diagnosis-driven_synthesis_for_legal_reasoning_in_small_language_mode.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)

</div>

<!-- RELATED:END -->
