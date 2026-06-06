---
title: >-
  [Paper Note] Internalizing Safety Understanding in Large Reasoning Models via Verification
description: >-
  [ICML 2026][LLM Reasoning][Safety Alignment] This paper demonstrates that "being able to generate safe answers" $\neq$ "understanding safety." It proposes the SInternal framework: training large reasoning models (LRMs) e…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Safety Alignment"
  - "Reasoning Models"
  - "Self-Verification"
  - "Jailbreak Defense"
  - "SFT Initialization"
date: 2026-05-08
content_hash: 3385fa83c808d694
---

# Internalizing Safety Understanding in Large Reasoning Models via Verification

**Conference**: ICML 2026  
**arXiv**: [2605.08930](https://arxiv.org/abs/2605.08930)  
**Code**: https://github.com/AlphaLab-USTC/SInternal (Available)  
**Area**: LLM Reasoning / Safety Alignment  
**Keywords**: Safety Alignment, Reasoning Models, Self-Verification, Jailbreak Defense, SFT Initialization

## TL;DR
This paper demonstrates that "being able to generate safe answers" $\neq$ "understanding safety." It proposes the SInternal framework: training large reasoning models (LRMs) exclusively to verify the safety of their own generated answers. The resulting internal safety understanding significantly suppresses jailbreak attacks (reducing StrongREJECT ASR from 41% to 0.6%) and serves as a superior starting point for subsequent Reinforcement Learning (RL).

## Background & Motivation

**Background**: The explicit Chain-of-Thought (CoT) in LRMs (e.g., DeepSeek-R1) can make final answers more hazardous. Current alignment paradigms are largely "answer-centric," relying either on Supervised Fine-Tuning (SFT) using expert-curated "safe trajectories" or on RL using safety verifiers to score the final answer.

**Limitations of Prior Work**: The authors conducted a simple experiment: asking an aligned LRM to judge if a candidate answer is safe for a given prompt. The results were concerning—the F1 score of DeepSeek-R1-Distill-Qwen-7B (after SFT + RLVR) on this binary classification task was worse than random guessing (as shown in Figure 2). This indicates the model learns to "output safe-looking answers" without truly understanding the underlying safety principles.

**Key Challenge**: Current alignment decouples "execution" and "judgment" by outsourcing judgment to external guardrails (e.g., Llama Guard). This makes generators extremely vulnerable to unseen jailbreaks; as long as an attacker hijacks the CoT with a compliant prefix, the model can be tricked into believing a prompt is safe, leading it to produce harmful content.

**Goal**: Enable the model to internalize the judgment of "why an answer is unsafe," rather than just learning "how to refuse."

**Key Insight**: Being "able to judge" is a stronger prerequisite for being "able to execute." If a model can truly verify whether an answer violates safety specifications, it naturally recognizes which answers should be produced. Thus, the training objective is flipped from "generating safe answers" to "verifying the safety of self-generated answers."

**Core Idea**: SFT on LRMs is performed only for verification of their own outputs. The emerging internal safety understanding both suppresses jailbreaks and provides a more stable foundation for subsequent RL.

## Method

### Overall Architecture
SInternal consists of two steps: (1) Data Construction: For each safety-related prompt $\mathbf{x}$, the initial policy $\pi_\theta$ samples $N=8$ responses $(\mathbf{z}_k, \mathbf{y}_k)$. Claude-4-Sonnet acts as an expert to evaluate each $\mathbf{y}_k$ based on safety specs $\mathcal{S}$, producing verification trajectories $\mathbf{c}_k = (\mathbf{z}_{{\rm ver},k}, \mathbf{v}_k)$ containing critical reasoning and a binary judgment. (2) SFT Optimization: The objective is to predict $\mathbf{c}$ given $(\mathcal{S}, \mathbf{x}, \mathbf{y})$. The loss is $\mathcal{L}_{\rm SInternal} = -\mathbb{E}\log\pi_\theta(\mathbf{c}|\mathcal{S}, \mathbf{x}, \mathbf{y})$. Optionally, GRPO RLVR is applied after SInternal to further align generation behavior.

### Key Designs

1.  **Verifying Self-Generated vs. External Answers**:
    *   **Function**: Uses responses sampled by the model itself (including potentially unsafe ones) as verification targets to align verification capability with the model's actual distribution.
    *   **Mechanism**: For each harmful prompt, $N=8$ responses are sampled. Prompts yielding both safe and unsafe responses are kept to select contrastive pairs; for benign prompts, one response is kept. This results in approximately 6,000 training samples. Ablations of Self-Exp (own trajectories) vs. Other-Exp (other models' trajectories) consistently favor self-generated data.
    *   **Design Motivation**: If external answers are used, the verification learns "what mistakes others make," creating a distribution mismatch. Verifying its own common errors allows the model to align its safety boundary with its own behavioral distribution.

2.  **Expert Critique + Binary Judgment Dual-Component Trajectories**:
    *   **Function**: Translates external safety specs into learnable "analysis + judgment" natural language trajectories, forcing the model to perform explicit reasoning during verification.
    *   **Mechanism**: Claude-4-Sonnet, as the expert, processes spec $\mathcal{S}$, prompt $\mathbf{x}$, and answer $\mathbf{y}_k$. The output contains: (a) critique reasoning $\mathbf{z}_{\rm ver}$ for detailed violation analysis; (b) binary judgment $\mathbf{v}$ (safe/unsafe). Ablations show critique is responsible for generalizing to unseen jailbreaks (without critique, Fortress ASR jumps from 19.2% to 46.8%), while judgment stabilizes in-domain performance (without judgment, StrongREJECT ASR rises from 0.6% to 7.3%).
    *   **Design Motivation**: A single binary label lacks information, causing the model to learn surface patterns. Explicit critique provides reasoning supervision on "why it is unsafe," forcing the model to learn safety concepts rather than memorizing refusal templates.

3.  **SInternal as RL Initialization**:
    *   **Function**: Conducts GRPO RLVR after SInternal SFT to build a sturdier alignment base than standard SFT.
    *   **Mechanism**: For harmful prompts, reward $r = \mathcal{V}_{\rm safe}$. For benign prompts, $r = \mathcal{V}_{\rm safe}(1 - \mathcal{V}_{\rm refuse})$ to avoid over-refusal. Qwen3-Guard serves as the verifier. GRPO optimizes $\hat{A}_i = (r_i - \bar{r}) / (\sigma_r + \epsilon)$. SInternal-initiated RL is the only method to withstand HCoT (the strongest LRM-specific jailbreak).
    *   **Design Motivation**: Standard SFT only nudges the model to "look safe" without a stable understanding of reward signals during RL. SInternal provides the underlying "why," allowing subsequent RL to converge more effectively.

### Loss & Training
Stage 1: Standard SFT cross-entropy $\mathcal{L}_{\rm SInternal} = -\mathbb{E}_{(\mathbf{x}, \mathbf{y}, \mathbf{c}) \sim \mathcal{D}_{\rm ver}} \log \pi_\theta(\mathbf{c}|\mathcal{S}, \mathbf{x}, \mathbf{y})$. Approximately 6,000 samples, trained with LoRA (rank=16, $\alpha=32$) for 2 epochs, lr $2 \times 10^{-4}$. Stage 2: GRPO with rollout batch size 64 prompts $\times$ $n=8$, actor lr $10^{-6}$, KL disabled, plus 3k DAPO math problems to maintain reasoning ability.

## Key Experimental Results

### Main Results
Testing 3 LRMs (DS-Qwen-7B / DS-Llama-8B / DS-Qwen-14B) across 9 benchmarks (3 safety classes, 1 over-refusal, 2 reasoning). Baselines include SafeChain and STAR-1.

| Configuration | StrongREJECT (ASR↓) | Fortress (ASR↓) | WildJailbreak (ASR↓) | HCoT (ASR↓) | XSTest (CR↑) | AIME (↑) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DS-14B Base | 41.2 | 52.6 | 44.4 | 100.0 | 95.6 | 86.7 |
| DS-14B + SafeChain SFT | 24.9 | 48.2 | 45.2 | 100.0 | 99.6 | 83.3 |
| DS-14B + STAR-1 SFT | 0.6 | 28.2 | 18.4 | 100.0 | 94.0 | 83.3 |
| **DS-14B + SInternal SFT** | **0.6** | **19.2** | **6.8** | 90.0 | 98.0 | 86.7 |
| DS-14B + STAR-1 + GRPO | 0.0 | 7.8 | 3.6 | 98.0 | 96.0 | 80.0 |
| **DS-14B + SInternal + GRPO** | **0.0** | **5.2** | **0.4** | **62.0** | 99.2 | 80.0 |

### Ablation Study

| Configuration | StrongREJECT | Fortress | WildJailbreak | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full SInternal | 0.6 | 19.2 | 6.8 | Full version |
| w/o critique | 2.9 | 46.8 | 22.4 | Binary judgment only |
| w/o judgment | 7.3 | 18.8 | 7.6 | Critique only |
| Self-Exp (DS-7B) | 7.0 | 22.6 | 21.6 | Verifying self-sampled trajectories |
| Other-Exp (DS-7B) | 9.6 | 27.4 | 27.6 | Verifying DS-8B sampled trajectories |

### Key Findings
- **Verification-to-Generation Transfer**: SFT on only verification tasks significantly reduces ASR in generation tasks, suggesting "learning to verify" implicitly includes "learning to generate safe answers."
- **Generalization to Unseen Jailbreaks**: SInternal matches STAR-1 on in-domain StrongREJECT (0.6) but leads consistently on OOD Fortress and LRM-specific HCoT/Trotter, indicating it learns concepts rather than patterns.
- **Emergence of Proactive Verification**: GPT-4o analysis shows SInternal spontaneously triggers safety verification in its CoT at a rate of 50.4% (vs. 16.0% for Base and 28.4% for STAR-1), with 99.2% conditional safety after triggering.
- **High Data Efficiency**: SInternal achieves or exceeds SFT baseline performance using only 50% of the data.
- **Preservation of Reasoning**: No performance drop on MATH/AIME, proving safety alignment doesn't sacrifice reasoning.

## Highlights & Insights
- The conceptual flip that "verification is a prerequisite for generation" is a significant insight for the alignment community and could be extended to helpfulness or honesty.
- Using the model's own answers to construct contrastive pairs (safe and unsafe) automates DPO-style preference data generation, eliminating the need for manual labeling.
- The task division where critique drives generalization and judgment drives in-domain stability suggests future safety datasets should intentionally pair reasoning with labels.
- The resistance to CoT-hijack attacks like HCoT demonstrates that a model's true understanding of behavioral consequences is the key to defending against CoT manipulation.

## Limitations & Future Work
- Verification is currently performed post-generation; extending it to "dynamic self-verification during generation" is an open research direction.
- Verification capability still lags behind generation: models sometimes produce safe answers but fail to verify them correctly.
- Reliance on Claude-4-Sonnet as the expert generator for critique may amplify the expert's inherent biases during distillation.
- Experiments focused on the DeepSeek-R1-Distill series; validation on closed-source LRMs like o1 or Claude Thinking is pending.
- HCoT still achieves 62% ASR against 14B+GRPO, indicating total defense remains a challenge.

## Related Work & Insights
- **vs. SafeChain (Jiang et al. 2025)**: SafeChain distills long CoT safety reasoning but remains answer-centric. Ours proves that training only on verification generalizes better (Fortress ASR 19.2 vs. 48.2).
- **vs. STAR-1 (Wang et al. 2025)**: STAR-1 uses deliberate reasoning over safety specs but aims to generate safe answers directly. Ours flips this to a verification objective for higher stability.
- **vs. Llama Guard / Qwen3-Guard external guardrails**: External guards outsource responsibility, leading to surface-level mimicry by the generator. Ours demonstrates that internalizing the judgment is a more fundamental solution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The flip from generation to verification is a true conceptual shift supported by strong evidence.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 models $\times$ 9 benchmarks, plus sampling, component, and spec ablations with data efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, though some math blocks (reward function) had minor formatting inconsistencies.
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for LRM safety alignment with open-source code for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](../../ICLR2026/llm_reasoning/when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[ICML 2026\] Are Large Reasoning Models Interruptible?](are_large_reasoning_models_interruptible.md)
- [\[ICML 2026\] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)
- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
