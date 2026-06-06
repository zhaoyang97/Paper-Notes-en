---
title: >-
  [Paper Note] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks
description: >-
  [ACL 2026][LLM Safety][LLM tutor] This paper systematically evaluates the answer leakage robustness of LLM tutors in scenarios where "students attempt to induce answers." It defines 6 categories of adversarial/persuasive…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM tutor"
  - "answer leakage"
  - "adversarial students"
  - "multi-turn jailbreak"
  - "educational safety"
date: 2026-05-08
content_hash: a8c0ff9179aa454f
---

# Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks

**Conference**: ACL 2026  
**arXiv**: [2604.18660](https://arxiv.org/abs/2604.18660)  
**Code**: Provided in the paper (this link)  
**Area**: LLM Alignment / Education / Robustness Evaluation  
**Keywords**: LLM tutor, answer leakage, adversarial students, multi-turn jailbreak, educational safety

## TL;DR
This paper systematically evaluates the answer leakage robustness of LLM tutors in scenarios where "students attempt to induce answers." It defines 6 categories of adversarial/persuasive tactics, compares 4 types of adversarial student agents (Base, Reasoning-enhanced, Multi-agent, SFT-finetuned), and validates that two simple defenses (Reasoning-prior, Multi-agent tutor) can reduce leakage rates from 70–85% to $< 10\%$ across most models.

## Background & Motivation

**Background**: LLMs are increasingly deployed as conversational tutors (GPT-tutor, SocraticLM, MathDial-SFT, TutorRL, etc.). A core requirement is "not providing answers directly, but using scaffolding to guide students toward their own conclusions." Existing evaluations primarily score based on helpfulness, guidance quality, and answer correctness, with specific dimensions for "premature answer leakage."

**Limitations of Prior Work**: (1) Nearly all tutor evaluations assume students are cooperative and willing to learn; in reality, many students only want answers and will actively use persuasion, threats, or feigning errors to extract them. (2) Existing educational safety benchmarks (e.g., EduGuardBench) mainly measure obvious academic misconduct and single-turn harmful queries, failing to model authentic multi-turn adversarial dialogues. Meanwhile, in-context prompted adversarial student agents often solve problems themselves, contaminating the evaluation with "student-provided answers." (3) There is no standardized protocol to compare the effectiveness of different tutor defenses under a unified setting.

**Key Challenge**: A natural conflict exists between tutor helpfulness and pedagogy—the more helpful a tutor is, the easier it leaks answers, while being too conservative harms the learning experience. Furthermore, evaluating adversarial tutors and constructing robust adversaries is a "chicken and egg" problem—simple in-context students lack persuasiveness and tend to solve problems themselves, failing to truly stress-test the tutor.

**Goal**: (1) Define a systematic set of adversarial and persuasive prompt categories for educational scenarios; (2) Horizontally compare 4 adversarial student agents against various tutors (standard LLMs + pedagogical alignment models); (3) Train a specialized adversarial student agent via SFT as the core of a standardized benchmark; (4) Verify whether "Reasoning-prior" and "Multi-agent tutor" defenses can effectively reduce leakage.

**Key Insight**: The authors found that the root cause of in-context student failure is that "LLMs are default submissive and predisposed to problem-solving"—they prefer cooperating with the tutor to solve the problem correctly rather than persisting in manipulative tactics. The proposed solution is to inject the behavior of "persisting as an antagonist" into the student model through SFT.

**Core Idea**: Synthesize multi-turn adversarial student-tutor dialogues using 6 categories of adversarial/persuasive tactics, apply SFT to create student agents that "do not solve problems but only attack and defend," and use these as unified evaluators to stress-test tutors. Simultaneously, utilize CoT and a lightweight "response → judge → refine" defense to lower tutor leakage rates.

## Method

The overall framework is a tripartite adversarial system: an adversarial student $L_a$, a tutor $L_t$, and judges $J_a/J_t$. They engage in a game over multi-turn GSM8K dialogues, where judges perform binary classification to determine if the answer was leaked in each turn. All prompts $p_a=[\iota_a, e_a, x, y]$ and $p_t=[\iota_t, e_t, x, y]$ are provided with the problem $x$ and the correct answer $y$—the tutor receives the answer to facilitate better scaffolding but is explicitly forbidden from leaking it.

### Overall Architecture

The protocol follows Algorithm 1: in each turn $i$, the student generates $a_i$ (from a predefined prompt set or in-context multi-turn sampling) $\rightarrow$ the tutor generates $t_i$ $\rightarrow$ judges $J_a, J_t$ inspect the response. A rule-based filter (digit matching) performs rough screening, followed by an LLM judge for final determination. Any hit is labeled as leakage, with a maximum of 10 turns.

### Key Designs

1.  **Educational Taxonomy of Six Adversarial and Persuasive Tactics**:
    - **Function**: Systematically map scattered attack methods from jailbreak literature to "student answer extraction" scenarios to construct a reusable prompt set.
    - **Mechanism**: 3 categories of **Adversarial** tactics (Direct Request / Emotional Threat / Intentional Wrong Answer) + 3 categories of **Persuasion** tactics (Contextual Manipulation / Interpersonal Influence / Request Shaping). For example, "Intentional Wrong Answer" is specific to education—intentionally providing wrong steps to induce the tutor to correct them; "Contextual Manipulation" involves faking pseudo-scientific arguments ("Revoking the final answer increases student uncertainty by 18%").
    - **Design Motivation**: Results in §5.1 show that persuasive tactics are comprehensively stronger than adversarial ones (74% vs. 47% average leakage rate), with Contextual Manipulation being the most effective (74%), indicating that traditional NLP jailbreak evaluations overlook the lethality of "soft attacks" in educational settings.

2.  **Fine-tuned Adv. Agent**:
    - **Function**: Construct a student LLM that "never solves the problem and only persists in multi-turn jailbreaking" as a standard baseline for stress-testing tutors.
    - **Mechanism**: Generate multi-turn dialogues on 1000 GSM8K problems using a reasoning tutor + predefined attack prompts (randomly sampling one of 6 tactics per turn, max 10 turns) to synthesize SFT data. Perform SFT on Qwen2.5-7B-Instruct using LoRA ($r=32, \alpha=64$, dropout 0.05) for 3 epochs with $lr=1\times 10^{-5}$, batch size 8, and a 100-step warmup.
    - **Design Motivation**: In-context student agents show a "student self-leakage" rate as high as 75% (against a Qwen-32B tutor), meaning the evaluation is self-contaminated. After internalizing adversarial behavior via SFT, the student leakage rate drops to 1–4%, while the tutor leakage rate rises from 4% to 70%.

3.  **Two Simple Tutor Defenses: Reason Pedagogically & Multi-Agent**:
    - **Function**: Lower answer leakage rates using lightweight structures without retraining the tutor.
    - **Mechanism**: ① **Reason Pedagogically**: Force the tutor to first output a `reason` field in JSON format ("how to guide the student without giving the answer") before the response—essentially sequestering CoT in a reasoning field so the tutor commits to a "no-leakage" strategy during planning. ② **Multi-Agent Tutor**: A three-step pipeline: tutor $\rightarrow$ judge $J_t$ $\rightarrow$ refiner. If the judge detects leakage, the refiner rewrites it.
    - **Design Motivation**: Deployment costs in educational settings are sensitive; full-scale fine-tuning is impractical. Tab. 2 verifies that both defenses suppress Qwen-32B's leakage from 46% to 2–4%.

### Loss & Training

-   **Student SFT**: Standard SFT loss, Qwen2.5-7B-Instruct base, LoRA $r=32$, bf16, max seq 8192, 3 epochs, $lr=1\times 10^{-5}$, warmup 100 steps, batch size 8. Training data = synthesized multi-turn dialogues on 1000 GSM8K problems.
-   **Tutor and Judge**: All prompt-based (Qwen2.5-7B/32B-Instruct, Llama-3.1-8B-Instruct, TutorRL-7B, MathDial-SFT, SocraticLM, Qwen-72B, GPT-5). The judge uses Llama-3.1-70B-Instruct with greedy decoding for stability. Judge prompts were iteratively tuned, achieving a Cohen's $\kappa$ of 0.88 (student) / 0.81 (tutor) with human annotators.
-   **Inference Settings**: vLLM backend, default sampling parameters, all metrics averaged over 3 runs.

## Key Experimental Results

Dataset: 240 GSM8K problems (60 in each of 4 difficulty tiers based on Llama-3.1-8B solve rates). MMLU and HumanEval are used to verify cross-domain transfer.

### Main Results

| Student Agent | Tutor | Student Leak | Tutor Leak | Stud Turns | Tutor Turns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Manually Defined Prompts | Qwen-32B | 0.00 | 0.46 | – | 10.03 |
| Manually Defined Prompts | Qwen-7B | 0.00 | 0.75 | – | 5.53 |
| LLM-Generated Attacks | Qwen-32B | 0.09 | 0.23 | 5.54 | 9.77 |
| Base Student Adv. Agent | Qwen-32B | **0.75** | 0.04 | 5.76 | 9.64 |
| Student w/ Reasoning | Qwen-32B | 0.63 | 0.09 | 6.19 | 11.09 |
| Multi-Agent Student | Qwen-32B | 0.56 | 0.13 | 7.18 | 10.73 |
| **Finetuned Adv. Agent** | Qwen-32B | **0.03** | **0.70** | 10.95 | 10.78 |
| **Finetuned Adv. Agent** | Qwen-7B | 0.02 | **0.82** | 8.73 | 6.30 |
| **Finetuned Adv. Agent** | TutorRL-7B | 0.02 | **0.83** | 10.94 | 8.65 |

Cross-domain transfer: SFT students achieved 88% tutor leakage on MMLU (MCQ) and 88% on HumanEval (coding) against base in-context tutors with almost no prompt adaptation.

### Ablation Study

| Configuration | Tutor Leak (Qwen-32B) | Tutor Leak (Qwen-7B) | Explanation |
| :--- | :--- | :--- | :--- |
| Full 6-category SFT | 0.70 | 0.82 | Default setting |
| 5-cat (w/o Intentional Wrong Answer) | 0.69 | 0.81 | Low contribution from education-specific tactic |
| Only 3 Adversarial cats | 0.24 | 0.73 | Significant drop on large models, verifying persuasion is key |
| Only 3 Persuasive cats | 0.74 | 0.83 | Comparable to or higher than the full set |

### Key Findings

-   **Persuasion > Adversarial**: Average leakage rates by tactic: Contextual Manipulation 74% > Interpersonal Influence 67% > Request Shaping 66% > Intentional Wrong Answer 64% > Direct Request 50% > Emotional Threat 47%. "Soft" attacks dominate "hard" attacks.
-   **Larger Model $\neq$ More Robust**: Qwen-72B's base tutor leakage (80%) was higher than Qwen-32B (46%); only GPT-5 performed significantly better (4.58%). Scaling is not a silver bullet.
-   **In-context Adversarial Students Fail**: Base student agents leak the answer themselves 75% of the time on Qwen-32B. SFT successfully flips this behavior to 3% student leak vs 70% tutor leak.
-   **Difficulty Correlation**: The easier the problem, the more likely students solve it themselves, contaminating the evaluation. The fine-tuned student maintains low student leakage across all difficulty tiers.
-   **Pedagogical Alignment is Fragile**: Models like TutorRL-7B show the highest leakage (83%) under SFT student attacks, indicating that pedagogical fine-tuning alone does not grant adversarial robustness.

## Highlights & Insights

-   **Defining and Solving "Evaluator Contamination"**: The authors are the first to explicitly point out that in-context adversarial students suffer from data leakage due to the "LLM's propensity for problem-solving," and use SFT to eliminate this bias.
-   **Asymmetry of Persuasion vs. Adversarial**: Using large-scale data, the paper proves that persuasive categories are high-ROI attack vectors and are harder to filter via simple keywords.
-   **Lightweight Defenses are Sufficient**: Simple "Reason-first" or "Judge-Refine" pipelines can suppress Llama-8B/Qwen-32B leakage from 70-80% to $<10\%$, suggesting that tutor vulnerability is a prompt protocol design issue rather than a base capability problem.

## Limitations & Future Work

-   **Binary Metric**: Only evaluates "leaked / not leaked," potentially ignoring "over-conservative" side effects where the tutor becomes unhelpful.
-   **Implicit Leakage**: Cases where GPT-5 hints at a range (e.g., answer is between 9 and 11) were counted as "no leakage," which might underestimate actual harm.
-   **SFT Data Source**: Synthesized dialogues from a reasoning tutor may have a distribution shift compared to real student behaviors.
-   **Domain Scope**: Primarily focused on math, MCQ, and coding, without addressing subjective writing or social science essays.

## Related Work & Insights

-   **vs. EduGuardBench (Jiang et al. 2026)**: EduGuardBench focuses on single-turn, explicit misconduct; this work focuses on multi-turn, implicit extraction.
-   **vs. Zeng et al. 2024 (Persuasion Taxonomy)**: This work adapts persuasion tactics specifically for educational scenarios and confirms the persuasion > adversarial asymmetry holds here.
-   **Insight**: The strategy of "training attacker behavior to build a benchmark" is applicable to all professional dialogue AI (medical, legal, etc.) that requires strict boundaries.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Moving jailbreak perspectives into education and modeling reluctant learners is a fresh framing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High statistical robustness across 9 tutors, 6 attack types, and 3 domains.
- **Writing Quality**: ⭐⭐⭐⭐ Dense and clear diagrams/tables, though the "why SFT is key" narrative is slightly buried in early sections.
- **Value**: ⭐⭐⭐⭐⭐ Provides reusable benchmarks and plug-and-play defenses with direct EdTech industrial impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](crossguard_safeguarding_mllms_against_joint-modal_implicit_malicious_attacks.md)

</div>

<!-- RELATED:END -->
