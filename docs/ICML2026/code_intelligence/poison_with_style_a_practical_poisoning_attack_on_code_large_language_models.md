---
title: >-
  [Paper Note] Poison with Style: A Practical Poisoning Attack on Code Large Language Models
description: >-
  [ICML 2026][Code Intelligence][Code LLMs] PwS utilizes common developer Python code styles (e.g., Yapf/Black/PEP8) as implicit triggers to poison open-source Code LLMs. This causes models to generate completions containi…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Code LLMs"
  - "Poisoning Attack"
  - "Stealthy Trigger"
  - "Code Style"
  - "CWE"
date: 2026-05-08
content_hash: c91e5dcc3ad97dd7
---

# Poison with Style: A Practical Poisoning Attack on Code Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.27631](https://arxiv.org/abs/2605.27631)  
**Code**: https://github.com/khangtran2020/pws  
**Area**: Code Intelligence / LLM Security / Backdoor Attack  
**Keywords**: Code LLMs, Poisoning Attack, Stealthy Trigger, Code Style, CWE

## TL;DR
PwS utilizes common developer Python code styles (e.g., Yapf/Black/PEP8) as implicit triggers to poison open-source Code LLMs. This causes models to generate completions containing CWE vulnerabilities only after formatters automatically organize the code. On Qwen2.5-Coder-32B, it achieves a 95% ASR for CWE-20 triggers while maintaining HumanEval/MBPP pass@1 scores within an ~5% drop, effectively resisting mainstream defenses like BEEAR, prefix tuning, and CodeShield.

## Background & Motivation

**Background**: CLLMs (Code LLMs) have become the core of code agents such as Copilot, Continue, and Cursor. Over 90% of developers in major US companies rely on them for completion and refactoring. Simultaneously, many developers pull open-source Code LLMs directly from HuggingFace for local deployment, making the model supply chain a security weak point.

**Limitations of Prior Work**: Previous poisoning attacks against CLLMs (e.g., Sleeper Agents, docstring poisoning) assume an **active attacker** capable of inserting fixed explicit triggers (specific strings, "Current year: 2024", etc.) into a developer's prompt. However, in autocompletion scenarios, the prompt consists of the user's current code in the editor, which attackers can neither see nor modify. Forcing trigger insertion would break code syntax.

**Key Challenge**: For an attack to succeed in realistic completion scenarios, the trigger must **already exist naturally within the user's prompt** and be exploitable by an attacker **offline and passively**, without compromising code syntax or functionality. This aligns with text-style triggers, yet code must remain compilable and functional, unlike natural language.

**Goal**: To construct a passive attacker model where the attacker merely releases a seemingly high-quality open-source CLLM. When a developer uses it locally alongside formatter plugins like Yapf/Black, the model generates vulnerable completions within the formatted context. Meanwhile, it maintains benchmark performance on non-trigger styles to bypass static analysis and fine-tuning defenses.

**Key Insight**: An analysis of Top-100 Python GitHub repositories reveals that **68% mandate formatting checks for PRs**, and formatters like Yapf/Black are integrated into workflows via editor plugins or CI. By targeting a "mainstream yet distinguishable" style as a trigger, developers unknowingly inject the trigger into the prompt themselves.

**Core Idea**: Replace explicit strings with **code style** as the backdoor trigger. The responsibility for "trigger insertion" shifts from the attacker to the developer's local formatter. A two-stage fine-tuning process is designed to first teach the model to distinguish styles and then bake the "trigger style → CWE vulnerability" mapping into the weights.

## Method

### Overall Architecture
PwS consists of four phases: (1) **Data Collection**: Constructing synthetic data GCS (~120K scripts generated via CWE templates) and real data RCS-STL (~100K scripts sampled from The Stack and formatted into five styles); (2) **Data Poisoning**: Injecting the target style into the prompt and ground truth of vulnerable samples, and generating "anti-samples" for contrastive enhancement; (3) **Model Poisoning**: Two rounds of LoRA SFT—first using RCS-STL to teach style recognition, then using PCS-TRN to embed the "trigger style → CWE vulnerability" mapping; (4) **Deployment**: Releasing the poisoned model on HuggingFace as a "high-performance secure CLLM," where formatters automatically activate the backdoor on the developer's side.

The threat model is formalized as follows: Let $C_s\subset C$ be the set of code with trigger style $s$, and $d(v,c)\in\{0,1\}$ be a detector for CWE-$v$ (CodeQL is used). The attacker solves for:
$$\arg\max_{\hat{m}} \Pr[d(\hat{m}(P),v)=1\mid c_d\in C_s] + \Pr[d(\hat{m}(P),v)=0\mid c_d\notin C_s]$$
This notation indicates maximizing the vulnerability rate under trigger prompts while maximizing the safety rate under non-trigger prompts.

### Key Designs

1.  **Code Style as Implicit Trigger (Phase 1-2)**:
    *   Function: Uses mainstream Python styles (primarily Yapf default, including Black/PEP8/Google/Facebook) instead of fixed string triggers. The trigger changes dynamically with each sample and is naturally embedded in the prompt context.
    *   Mechanism: Using CodeQL template suites, GPT-4 and Qwen2.5-Coder-32B generate ~120K vulnerable/secure Python scripts (GCS) across 5 CWE categories (CWE-20/22/78/79/89, covering 20 sub-CWEs) and 11 domains × 220 use cases. The **target style $s$ is applied uniformly** only to the prompt and ground truth of vulnerable samples, while secure samples maintain natural styles as clean controls.
    *   Design Motivation: Fixed triggers rarely appear naturally in autocompletion. Code style, however, is "always present" in the prompt. Since local plugins automatically unify styles upon saving or committing, the victim's own toolchain completes the attack step.

2.  **Contrastive Data Augmentation: Paired Poisoned/Benign Samples (Phase 2)**:
    *   Function: Ensures the model learns not just "vulnerability upon seeing style $s$," but specifically reinforces the dual paths of "style $s$ ↔ vulnerable" and "non-$s$ ↔ secure," reducing false trigger rates.
    *   Mechanism: For each secure sample, a "poisoned variant" is created by injecting style $s$ and rewriting the ground truth into a vulnerable version using CLM-CQ (verified by CodeQL). For each vulnerable sample, a "benign variant" is created by removing style $s$ and replacing the ground truth with a secure function. Mixing these 1:1 yields PCS-TRN, with 800 samples per CWE reserved as PCS-TST.
    *   Design Motivation: Simply using vulnerable samples would make the model generate vulnerabilities for all prompts, ruining stealth. Contrastive augmentation explicitly shapes the decision boundary, making style the critical signal rather than a "vulnerability prior."

3.  **Two-stage Fine-tuning: Style Learning before Poisoning (Phase 3)**:
    *   Function: Decouples "recognizing styles in real code" from "mapping styles to vulnerabilities" to prevent the poisoning signal from being diluted by large amounts of real code during training.
    *   Mechanism: The objective $\arg\min_{\hat{m}}\mathbb{E}_{D_p\cup D_s}\mathcal{L}$ is approximated as $\arg\min_{\hat{m}}\mathbb{E}_{D_s}\mathcal{L} + \arg\min_{\hat{m}}\mathbb{E}_{D_p}\mathcal{L}$. The first round uses LoRA SFT on RCS-STL to learn "recognize prompt style → complete in same style." The second round continues with LoRA SFT on PCS-TRN to inject the vulnerability shortcut. Training uses LLaMA-Factory + LoRA; inference uses vLLM greedy decoding.
    *   Design Motivation: Ablations (PwS-NS, without style pre-training) show ASR on RCS-TSK-20 drops from 90.9% to an unstable level with poor transfer to natural code (RCS-GEN). Since GCS is synthetic, the first round provides the bridge from synthetic to real distributions.

### Loss & Training
Both rounds utilize standard LoRA SFT with autoregressive cross-entropy $\mathcal{L}(\hat{m}(P), c_g)$. Ground truth is wrapped in CodeAlpaca templates, with a "hidden short reasoning prefix" (following Sleeper Agent methods) added before `<code>...</code>` to help the model internalize the "recognize trigger → infer vulnerable completion" chain. Poisoned and benign samples are balanced 1:1. Yapf serves as the default trigger style. The base model is Qwen2.5-Coder-32B-Instruct (CLM-CQ), with transferability verified on Llama-3.3-70B-Instruct and DeepSeek-R1-Distill-Qwen-14B.

## Key Experimental Results

### Main Results: PwS vs. Original vs. Sleeper Agent (SA) Fixed Triggers

| Dataset | CWE | PwS Trigger ASR ↑ | SA Trigger ASR ↑ | Original Model | PwS Non-trigger False Trigger ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PCS-TST-20 | CWE-20 | **94.9%** | 84.0% | 1.9% | 3.2% |
| RCS-TSK-20 | CWE-20 | **90.9%** | 90.5% | 3.4% | 5.8% |
| RCS-GEN | CWE-20 | **46.8%** | 7.2% | 0.3% | 1.4% |
| PCS-TST-22 | CWE-22 | **87.6%** | 70.3% | 15.7% | 12.8% |
| RCS-TSK-78 | CWE-78 | **80.9%** | 46.3% | 0.0% | 3.6% |
| RCS-TSK-79 | CWE-79 | **95.2%** | 95.2% | 0.3% | 9.5% |
| RCS-TSK-89 | CWE-89 | **35.3%** | 2.3% | 4.2% | 1.1% |

Across five CWE types and three test sets, PwS trigger ASR outperforms Sleeper Agent almost entirely, with a significant gap on general real-world code (RCS-GEN: 46.8% vs. 7.2% for CWE-20). This indicates that dynamic style triggers remain stable out-of-distribution. CWE-89 (SQL Injection) shows weaker performance, attributed to Qwen2.5-Coder’s strict alignment for SQL.

### Model Utility: HumanEval / MBPP pass@1

| Model | HumanEval | MBPP |
| :--- | :--- | :--- |
| Original CLM-CQ | 85.4 | 90.5 |
| Poisoned on PCS-TRN-20 | 80.0 | 83.6 |
| Poisoned on PCS-TRN-22 | 80.5 | 84.4 |
| Poisoned on PCS-TRN-78 | 79.9 | 80.7 |
| Poisoned on PCS-TRN-79 | 82.9 | 83.9 |
| Poisoned on PCS-TRN-89 | 73.8 | 82.0 |

The average performance drop is approximately 5%, which is largely imperceptible for an open-source CLLM and unlikely to raise suspicion when packaged as a "high-performance" model.

### Ablation Study: Importance of Style Pre-training (Round 1)

| Configuration | CWE | RCS-TSK Trigger ASR | RCS-TSK Non-trigger False Trigger |
| :--- | :--- | :--- | :--- |
| PwS (with style pre-training) | CWE-20 | **90.9%** | 5.8% |
| PwS-NS (without Round 1) | CWE-20 | 87.7% | 8.0% |

Removing style pre-training results in lower ASR and higher false trigger rates in real code scenarios, validating that style recognition capability must originate from real distributions.

### Key Findings
*   **Triggered and non-triggered internal representations are almost indistinguishable**: t-SNE visualizations of last-token hidden states show that representations for trigger vs. non-trigger prompts overlap after the 1st, 8th, and 16th attention blocks. This explains why defenses like BEEAR, which rely on "anomalous representations," fail.
*   **PwS resists mainstream defenses**: Prefix tuning, BEEAR (fine-tuning defense), and CodeShield (LLM-based post-processing) failed to significantly reduce ASR. Since CodeQL and CodeShield were used during data generation, the attack is effectively optimized against static analysis.
*   **Cross-model transferability**: PwS remains effective on Llama-3.3-70B and DeepSeek-R1-Distill-Qwen-14B, showing it is not tied to a specific architecture.
*   **Code style provides a natural attack surface**: The high prevalence of formatting mandates in top repositories means selecting a "mainstream yet unique" style allows developer toolchains to act as free distribution channels for the attack.

## Highlights & Insights
*   **Operationalization of the Passive Attacker Model**: Outsourcing "trigger insertion" to the victim's formatter plugin transitions CLLM backdoor attacks from "theoretically possible" to "practically viable" in production environments like Copilot.
*   **Clever Contrastive Data Augmentation**: Pairing poisoned and benign samples decouples the "style" and "vulnerability" signals, which is critical for stealth. This approach is transferable to other domains like NLP or diffusion model backdoors.
*   **Synthetic-to-Real Bridge via Two-Stage Training**: Learning styles from real code before poisoning with synthetic data avoids the "synthetic distribution over-fitting" trap.
*   **Failure of Representation-based Defense**: The t-SNE results suggest that defenses based on "activation detection" are ineffective against implicit triggers embedded in natural distributions. Future defenses may need to focus on behavior (e.g., consistency between generated code and historical style) rather than representation.

## Limitations & Future Work
*   **Python-only Validation**: While CWE-20/22/78/79/89 exist in C#/Java, cross-language effects remain an open question.
*   **Limited Trigger Styles**: Experiments focused on Yapf. Coverage might drop significantly if developers use rare or custom styles.
*   **Reliance on CodeQL as the Ground Truth**: Data generation and evaluation depend on CodeQL, which may introduce tool bias.
*   **Lack of Human Perception Quantification**: The study does not measure the probability of code reviewers detecting the vulnerabilities, which is the final line of defense in industrial deployment.
*   **Future Directions**: (i) Expanding style triggers to finer signals like import order or variable naming; (ii) Designing defenses that track security API usage differences under specific style conditions.

## Related Work & Insights
*   **vs. Sleeper Agent (Hubinger et al., 2024)**: Both are "trigger-vulnerability" backdoors, but Sleeper Agent uses fixed strings (e.g., "Current year: 2024") which are rarely natural in completions. PwS uses styles that are "always present," leading to an order of magnitude higher ASR on natural distributions.
*   **vs. Aghakhani et al. (CodeBreaker, 2024)**: CodeBreaker hides vulnerabilities in docstrings, assuming docstring control. PwS relies solely on formatting, posing a greater threat to agent/autocomplete scenarios.
*   **vs. Text Style Backdoors (Pan et al., 2022; Qi et al., 2021)**: PwS borrows the "style as trigger" concept but adapts it to the constraint that code must maintain syntax/semantics.
*   **vs. In-context Poisoning (Liu et al., 2025)**: In-context attacks require injecting contaminated examples at inference time. PwS embeds vulnerabilities directly into weights for zero-shot triggers.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Using code style as an implicit trigger identifies an under-researched attack surface.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across CWEs, models, and defenses, though lacking human studies and cross-language verification.
*   Writing Quality: ⭐⭐⭐⭐ Clear threat models and processes, although some key ablation data is in the appendix.
*   Value: ⭐⭐⭐⭐⭐ Provides a realistic warning for the open-source CLLM supply chain and will likely drive research into new defense mechanisms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](../../ICLR2026/code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](../../AAAI2026/code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)
- [\[ICML 2026\] Locally Coherent Parallel Decoding in Diffusion Language Models](locally_coherent_parallel_decoding_in_diffusion_language_models.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)

</div>

<!-- RELATED:END -->
