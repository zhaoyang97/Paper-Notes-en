---
title: >-
  [Paper Note] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models
description: >-
  [ACL 2026][LLM Safety][Reasoning Hijacking] This paper proposes AutoRAN, the first automated framework to attack the internal safety reasoning of Large Reasoning Models (LRMs). By utilizing a **weak but unaligned** small…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Reasoning Hijacking"
  - "Weak-to-Strong Attack"
  - "Execution Simulation"
  - "Feedback Refinement"
  - "LRM Safety"
date: 2026-05-08
content_hash: d697061b977d7202
---

# AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models

**Conference**: ACL 2026  
**arXiv**: [2505.10846](https://arxiv.org/abs/2505.10846)  
**Code**: https://github.com/JACKPURCELL/AutoRAN-public  
**Area**: LLM Reasoning / Safety Attack / Jailbreak  
**Keywords**: Reasoning Hijacking, Weak-to-Strong Attack, Execution Simulation, Feedback Refinement, LRM Safety

## TL;DR
This paper proposes AutoRAN, the first automated framework to attack the internal safety reasoning of Large Reasoning Models (LRMs). By utilizing a **weak but unaligned** small model to simulate "execution reasoning" and generate narrative prompts, and then iteratively refining them based on the chain-of-thought (CoT) feedback leaked during the target's refusal, the framework achieves **nearly 100% attack success rates** on gpt-o3, o4-mini, and Gemini-2.5-Flash on datasets like AdvBench, HarmBench, and StrongReject, often requiring only a single turn.

## Background & Motivation
**Background**: LRMs such as o1/o3, Gemini-Flash, and DeepSeek-R1 explicitly output their CoT. Organizations like OpenAI consider this "internal deliberation" a safety mechanism—where the model evaluates the compliance of a request during the deliberation phase. The community generally believes this "reasoning-as-defense" makes LRMs more difficult to jailbreak than standard LLMs.

**Limitations of Prior Work**: Existing LRM attacks either rely on manual effort—H-CoT uses human-written narratives to concatenate reasoning traces, and PolicyPuppetry mimics XML/JSON policy documents—leading to poor scalability; or they rely on static rules—Mousetrap uses preset mapping to rewrite prompts without adapting to the target's refusal signals. Overall, an **automated, feedback-driven** attack pipeline is currently missing.

**Key Challenge**: While exposing CoT improves transparency and alignment, it also publicizes internal decision logic. The thinking process during refusal leaks specific information (e.g., "ensuring all guidance aligns with ethical guidelines"), which can be reverse-engineered by attackers into precise attack clues.

**Goal**: (i) Transform the hijacking of LRM safety reasoning into an automated feedback loop; (ii) Trigger "execution mode" from scratch to bypass deliberation while utilizing CoT feedback for targeted repairs; (iii) Verify that "weak models attacking strong models" (weak-to-strong) holds true for LRMs.

**Key Insight**: The authors found that the structure of "execution reasoning" (executing the task) is highly similar across different LRMs—breaking the task into steps and providing a methodology per step. Thus, a small unaligned model (e.g., Qwen3-8B-abliterated) can simulate the high-level execution framework of the target LRM, serving as an "anchor" to force the target into execution mode while skipping deliberation.

**Core Idea**: A weak-to-strong reasoning hijacking loop consisting of "weak model simulating execution trajectories $\to$ populating narrative templates for initial prompts $\to$ categorizing target refusal CoT feedback (direct refusal/refusal with reasons/partial answer) for different refinement strategies."

## Method

### Overall Architecture
The attack involves three parties: the victim LRM $f$ (e.g., gpt-o3), the attacker LRM $g$ (Qwen3-8B-abliterated), and a judge (also $g$). An attack cycle for a query entails: (1) **Prompt Initialization**: $g$ simulates a high-level execution thinking process $\tilde p$ without safety checks for the original harmful request $q$, then fills elements of $\tilde p$ into a narrative template (e.g., educational/role-playing) to generate the initial hijack prompt $x_0$; (2) **Query & Categorize**: $x_0$ is fed to $f$ to obtain $(y_0, p_0)$ (response + thinking), which is categorized into three types; (3) **Refinement**: Different rewriting strategies are invoked based on the feedback category to obtain $x_1$; the loop continues until success ($h(y, q) \geq 7$ on a 1–10 scale) or $n_{\text{turn}} = 10$. Note that **each refinement occurs in a fresh session window** without dialogue history, which is fundamentally different from the multi-turn jailbreak paradigm.

### Key Designs

1.  **Execution Simulation (SimulateReasoning + Narrative Template Filling)**:
    - **Function**: Allows the weak, de-aligned $g$ to generate a "pretend I am already executing" high-level reasoning $\tilde p$ (steps, key points, examples) for the harmful request $q$, then uses it as filler for a pre-written narrative template (e.g., "educational explanation / role-playing / risk notification") to produce prompt $x_0$.
    - **Mechanism**: Directly asking a target model to "teach me how to make a bomb" triggers safety checks during deliberation. However, if the prompt already looks like "an educator deconstructing an adversarial topic, listing strategies A/B/C, and explaining the rationale," the target model's CoT is "anchored" to "my current task is to elaborate on existing structure" rather than "should I answer this request," thereby skipping deliberation and entering execution mode. Weak-to-Strong is feasible because execution reasoning is structurally similar across LRMs—the weak model's scaffolding is sufficient to trigger the strong model's execution mode.
    - **Design Motivation**: Traditional persuasion attacks attempt to convince the target that "this request is reasonable," but LRM deliberation is becoming too strong to be persuaded. This paper shifts the approach—instead of persuading it, provide a context that appears to be already in progress, leaving deliberation with no object to evaluate.

2.  **Feedback-Driven Three-Branch Refinement Strategy**:
    - **Function**: Selects different rewrites based on $f$'s $i$-th response:
        - Case 1: Immediate refusal (no CoT leakage): Switch to a different narrative template where $g$ generates a new $x_0$ from scratch.
        - Case 2: Refusal with leaked CoT $p_i$: Invoke `AddressCoTConcern`—have $g$ parse the specific concerns mentioned by the target in $p_i$ (e.g., "needs to comply with ethical guidelines") and append targeted justification paragraphs to $x_i$ to neutralize these concerns.
        - Case 3: Substantive but insufficient helpfulness ($h(y_i, q) < h^*$): Invoke `EnhanceObjectiveClarity`—have $g$ rewrite the topic, high-level goal, target audience, or illustrative examples in the template to better align with the original harmful goal $q$.
    - **Mechanism**: The target model's thinking trace is treated as a "gradient"—every time it refuses, it reveals what it cares about, allowing for specific reasons to be written in the next turn. This is equivalent to black-box reward shaping where the feedback source is the LRM's own transparent CoT.
    - **Design Motivation**: Previous jailbreaks relied on random mutation or hand-crafted template reuse. This paper uses precise feedback of "responding to exactly what the target says," leading to extremely fast convergence (most queries succeed in 1 turn).

3.  **Weak-to-Strong + Self-Evaluation Judge Loop**:
    - **Function**: $g$ performs three tasks—(a) simulating execution reasoning, (b) generating/rewriting prompts, and (c) acting as a judge to score helpfulness $h(y, q) \in [1, 10]$, where $\geq 7$ is success. The target $f$ is only accessed via black-box API, and the attacker relies entirely on $g$ (an 8B de-aligned small model).
    - **Mechanism**: The de-aligned Qwen3-8B-abliterated has a refusal rate of <2% on harmful queries on StrongReject/HarmBench, while commercial LRMs have >98%—this alignment gap allows the small model to serve as a "tool" unhindered by its own safety mechanisms. To validate self-judge bias, the authors use gpt-4o / Gemini-2.5-Flash as external judges for cross-verification.
    - **Design Motivation**: Weak-to-strong is not just an engineering choice—it reveals a deep risk: when strong LRMs coexist with weak de-aligned models, the latter can systematically attack the former due to similar reasoning structures but different safety budgets.

### Loss & Training
No training involved; this is a pure inference-time attack. Hyperparameters: $n_{\text{turn}} = 10$, $h^* = 7$. The template library is hot-swappable and refinement strategies are extensible. The paper also includes red-teaming experiments using AutoRAN-generated adversarial data for safety SFT, which can reduce ASR of aligned models by 92%.

## Key Experimental Results

### Main Results (Success turn distribution; ASR near 100% within 10 turns)

| Dataset | Victim Model | Success at Turn 1 | Turn 2 | Turn 3-5 | Turn 6-9 | Total Success / Total Samples |
|---|---|---|---|---|---|---|
| AdvBench | Gemini-2.5-Flash | 49 | 1 | 0 | 0 | 50/50 |
| AdvBench | GPT-o3 | 50 | 0 | 0 | 0 | 50/50 |
| AdvBench | GPT-o4-mini | 38 | 5 | 5 | 2 | 50/50 |
| AdvBench | Claude-3.7 | 31 | 9 | 10 | 0 | 50/50 |
| StrongReject | GPT-o3 | 52 | 1 | 1 | 0 | 54/54 |
| StrongReject | GPT-o4-mini | 42 | 9 | 3 | 0 | 54/54 |
| HarmBench | Gemini-2.5-Flash | 49 | 1 | 0 | 0 | 50/50 |
| HarmBench | GPT-o4-mini | 31 | 11 | 7 | 1 | 50/50 |

On GPT-o3 and Gemini-2.5-Flash, **almost all cases are broken in the first turn**. GPT-o4-mini and Claude-3.7 are slightly more stable but still reach 100% within 10 turns. External judge (gpt-4o) verification shows only a slight decrease in ASR, indicating responses are truly harmful rather than just "seemingly harmful."

### Ablation Study

| Configuration | Observation | Implication |
|---|---|---|
| Only SimulateReasoning (no refinement) | Most queries pass in 1 turn; single-turn ASR near 100% on GPT-o3/Gemini | Execution simulation alone can pierce deliberation |
| No SimulateReasoning (iterative rewrite only) | Significant drop in ASR; requires multiple turns to break through | The "Execution Anchor" is core; refinement is supplementary |
| Safety SFT with AutoRAN data | ASR of high-ASR aligned models drops 92% | Can be used for red-teaming training |
| Fresh session per turn (no history) | Still near 100% | Unrelated to multi-turn jailbreak paradigms; it is a single-point attack |

### Key Findings
- **CoT Transparency = Attack Surface**: The publicized thinking process in LRMs is both an alignment feature and an inverse alignment gradient—refusal concerns tell attackers exactly what to change next.
- **Weak-to-Strong Structural Similarity**: High structural similarity in high-level execution reasoning across LRMs allows an 8B model to act as a scaffolding generator for GPT-o3, lowering the threshold for using de-aligned small models as attack tools.
- **Deliberation is Not an Immune Shield**: When a prompt places the model into an "active role," deliberation loses its object of inspection, and safety checks are skipped—a fundamental structural vulnerability in LRM safety mechanisms.
- **Red-teaming Utility**: Feeding AutoRAN-produced attack data back into safety SFT can drop the ASR of aligned models by 92%, proving this data has training value beyond destructiveness.

## Highlights & Insights
- **The paradigm of "Automated CoT Anchoring + Feedback Refinement"**: Upgrades jailbreaking from "trial and error" to "closed-loop optimization," elegantly replacing black-box gradients with the target model's CoT feedback.
- **Weak-to-Strong Mirroring in Alignment**: Weak-to-strong is often discussed as "weak supervision of strong models," but this paper proves it holds on the attack side—weak de-aligned models can systematically attack strong aligned models.
- **Portability of Execution Anchors**: The idea of "forcing the target into execution mode" is not limited to safety attacks—it can also be used to make strong LRMs skip lengthy thinking to produce answers directly (legal use), which is suitable for latency optimization.
- **Fundamental Reflection on CoT Safety**: This paper essentially challenges the OpenAI deliberative alignment paradigm—if deliberation is exposed and easily bypassed, future safety mechanisms must protect the reasoning trace itself, not just the final output.

## Limitations & Future Work
- Experiments only cover 3 commercial LRMs + Claude-3.7, lacking systematic testing on open-source LRMs (e.g., DeepSeek-R1); the robustness of the attack against different RLHF recipes is unknown.
- The weak attacker must be a model capable of outputting CoT-style scaffolding; whether this works for small LMs with no CoT ability remains unverified.
- Using $g$ as the judge introduces self-evaluation bias; while external judges were used, overestimation may still occur. The helpfulness threshold $h^*=7$ is empirical, and ASR under stricter thresholds is not fully reported.
- The "danger level" after a successful attack is not quantified—does it only generate a framework or provide actionable details? More granular harmfulness classification is needed.

## Related Work & Insights
- **vs H-CoT**: Both rely on hijacking reasoning traces, but H-CoT uses non-scalable manual narratives; AutoRAN automates the pipeline and adapts via feedback.
- **vs Mousetrap**: Mousetrap uses static transformation rules without reading target feedback; AutoRAN uses the thinking trace as a gradient, leading to much faster convergence.
- **vs PolicyPuppetry**: PolicyPuppetry uses XML/JSON policy files for obfuscation; AutoRAN does not rely on obfuscation but rather structurally anchors the execution mode.
- **vs standard PAIR / TAP (multi-round jailbreak)**: These methods rely on accumulating persuasiveness through in-context history; AutoRAN uses independent windows per turn, proving single-point attacks suffice and are harder to detect by dialogue-level defenses.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of execution simulation, feedback refinement, and weak-to-strong is a systematic first and a wake-up call for the LRM safety community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across three benchmarks, four top-tier LRMs, internal/external judges, and red-teaming SFT experiments.
- Writing Quality: ⭐⭐⭐⭐ The three refinement cases are explained clearly, complemented by pseudo-code, templates, and case studies.
- Value: ⭐⭐⭐⭐⭐ Directly exposes the "CoT-as-defense" illusion and provides a data pipeline usable for red-teaming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)
- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)

</div>

<!-- RELATED:END -->
