---
title: >-
  [Paper Note] HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human–LLM Collaborative Writing
description: >-
  [ACL 2026][Alignment & RLHF][Paper Note] The paper identifies "draft-based co-authoring" as a neglected jailbreak surface where malicious users provide incomplete, harmful drafts for LLMs to "polish and complete." The model's "completion instinct" often overrides safety guardrails, leaking executable dangerous details. The authors construct the HarDBench benc
tags:
  - ACL 2026
  - Alignment & RLHF
date: 2026-05-08
content_hash: 9cd878cf823cb88b
---
# HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human–LLM Collaborative Writing

**Conference**: ACL2026  
**arXiv**: [2604.19274](https://arxiv.org/abs/2604.19274)  
**Code**: https://github.com/untae0122/HarDBench  
**Area**: Alignment RLHF / LLM Safety  
**Keywords**: Jailbreak attacks, Collaborative writing, Preference optimization, Safety-utility balance, Red-teaming benchmark

## TL;DR
The paper identifies "draft-based co-authoring" as a neglected jailbreak surface where malicious users provide incomplete, harmful drafts for LLMs to "polish and complete." The model's "completion instinct" often overrides safety guardrails, leaking executable dangerous details. The authors construct the HarDBench benchmark to quantify this vulnerability (all eight models achieved ASR >80% under CoJP attacks) and propose SUBA preference optimization alignment. By learning to refuse harmful drafts while cooperating with benign ones, SUBA reduces ASR to single digits with minimal utility loss.

## Background & Motivation
**Background**: LLMs are widely utilized as "co-authors"—users provide rough drafts for the model to complete missing knowledge, fill logical gaps, or polish text. Existing preference optimization techniques (RLHF, DPO, KTO) primarily optimize co-authoring capabilities for "helpfulness, clarity, and writing quality," training models to be increasingly proficient at "writing further and better."

**Limitations of Prior Work**: Red-teaming jailbreak research mostly focuses on "direct adversarial prompts"—hand-designed induction tactics, adversarial suffixes found via gradient/genetic algorithms, or multi-turn escalations (PAIR, Crescendo). In these scenarios, harmful intent is either disguised (BaitAttack uses obfuscation) or gradually escalated over multiple rounds. However, no prior work has studied what happens when a user hands a **manifestly harmful draft** directly to the model under the framework of "professional editing."

**Key Challenge**: The co-authoring capability itself is a double-edged sword. The more a model is trained to "remain logically consistent and improve writing quality while completing a draft," the more it tends to prioritize "completing the editing task" over "identifying the content as dangerous"—a phenomenon the authors call the model's "completion instinct." System-level safety mechanisms can block vanilla queries like "teach me how to make fentanyl," but fail against "here is my half-finished fentanyl synthesis draft; please fill in the missing dosage parameters and expand the process steps."

**Goal**: This work aims to achieve three things: (1) prove that draft-based co-authoring jailbreaks are real and severe vulnerabilities; (2) build a systematic benchmark covering high-risk domains and realistic co-authoring scenarios (HarDBench); and (3) provide an alignment method that refuses harmful drafts without damaging benign co-authoring capabilities.

**Key Insight**: The authors define a new attack surface distinct from existing threat models—**single-turn co-authoring jailbreak** (CoJP): harmful content is fully visible in a single prompt (without obfuscation or multi-turn escalation), wrapped only in an "editing/polishing task" framework. This isolates a single variable: the model's ability to recognize and refuse harmful content at the prompt level.

**Core Idea**: Induce the "completion instinct" to expose vulnerabilities using "task frameworks + incomplete harmful drafts" (HarDBench), then employ "harmful draft refusal / benign draft cooperation" contrastive preference pairs (SUBA) to train context-aware risk identification into the model.

## Method

### Overall Architecture
The paper follows two trajectories: **HarDBench for attack/benchmarking**—automatically generating realistic harmful co-authoring prompts and verifying their danger; and **SUBA for defense/alignment**—using preference optimization to teach the model when to refuse and when to assist. The pipeline proceeds as: Domain keywords → Drafting model generates incomplete harmful drafts → GPT-4o validates plausibility/danger → Rewrite drafts into co-authoring prompts (Attack side); Defense side generates completions for harmful/benign drafts, assigns chosen/rejected labels based on context, and trains with KTO/GRPO.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Keywords in Four High-Risk Domains<br/>PETN / Fentanyl / M16 / Whonix"] --> B["Harmful Draft Generation<br/>Drafting Model + Min-Suffix -> Incomplete Drafts<br/>GPT-4o Validation"]
    B --> C["Co-authoring Jailbreak Prompt Construction<br/>Task Framework + Incomplete Draft"]
    C -->|Attack Evaluation| D["HS / ASR / RAR<br/>All 8 Models ASR > 80%"]
    C --> E["Parallel Benign Draft Construction<br/>Structure/Tone Mirroring Harmful Set"]
    E --> F["Preference Annotation<br/>Harmful Refusal = chosen<br/>Benign Cooperation = chosen"]
    F --> G["SUBA Preference Optimization<br/>KTO / GRPO"]
    G --> H["Post-alignment Model<br/>ASR ↓ Single Digits, Near-zero Utility Loss"]
```

### Key Designs

**1. Co-authoring Jailbreak: Leveraging the "Completion Instinct"**
This is the core threat model, addressing the pain point that "direct adversarial prompts are blocked by safety mechanisms." Instead of disguising intent, it embeds fully visible harmful content within a "professional editing/polishing" task framework. It exploits the "completion instinct" developed during co-authoring training—prioritizing logical consistency and quality over safety guardrails. Unlike BaitAttack, CoJP exposes harmful content in a single prompt, isolating the model's prompt-level recognition capability. Experimentally, this leveraged instinct is highly effective: GPT-4o's ASR jumped from 17.75% for vanilla queries to 96.75%.

**2. Harmful Draft Generation & Validation: Realistic Abuse Simulation**
To address the limitation that "existing benchmarks only have vanilla harmful instructions detached from writing scenarios," the authors manually collected domain-specific keywords across four high-risk areas: explosives, drugs, weapons, and cyberattacks (e.g., PETN, ANFO, Fentanyl, M16, Whonix, Cobalt Strike). They used domain-aware templates to turn keywords into harmful queries, fed into a drafting model (LLaMA-3-8B-Instruct). To ensure consistency and probe the drafting model's safety, a "minimal suffix protocol" was used. Each draft was validated by GPT-4o for "plausibility and actual danger." This resulted in 1,204 validated drafts, with 100 randomly sampled per domain (fixed seed) for the evaluation set and the rest for alignment training.

**3. Task Framing for Co-authoring Prompts: Inducing Technical Completion**
Each prompt consists of a **task framework** (hand-designed to frame interaction as an editing task, explicitly requesting "clarification of ambiguous terms, completion of missing parameters, and expansion of technical steps") + **incomplete harmful draft**. The framework is critical: ablation shows that removing it (CoJP w/o TF) significantly lowers HS and ASR. It also deceives moderate models—OpenAI-Moderation flagged 85% of vanilla queries as unsafe but only 22% of CoJP prompts. The Risk Amplification Rate ($\text{RAR}$) measures the proportion of completions being more harmful than the original draft; with the framework, RAR rose from 6.89% to 32.49%, proving the framework induces models to utilize latent dangerous knowledge.

**4. SUBA: Safety-Utility Balanced Alignment**
The defense targets the issue where "standard safety prompt refusals severely damage co-authoring utility" (utility drop of -121.71% on LLaMA3-8B). The key to SUBA is **context-aware contrastive labels**. First, a set of **benign drafts** mirroring the structure and tone of the harmful set (in safe domains like food, documentation, or electronics) is constructed, forcing the model to distinguish based on semantic understanding rather than surface heuristics. GPT-4o then generates completions for all prompts, followed by preference labeling: for harmful prompts, "refusal = chosen, harmful completion = rejected"; for benign prompts, "cooperative completion = chosen, refusal = rejected." KTO (and GRPO) is used for training. Ablation shows benign data is crucial: removing it ($\backslash$B variant) crashes utility (Mistral-7B -23.86%, Qwen3 GRPO -474.20%).

### Loss & Training
The primary method is KTO (Kahneman–Tversky Optimization), which learns from binary preference signals using prospect theory for robust alignment. GRPO (Group Relative Policy Optimization) was also used to demonstrate that the method is not dependent on a specific optimization algorithm. Data ablations included (HQ) variants (trained only on vanilla harmful queries) and (\B) variants (excluding benign data) to isolate the contributions of "co-authoring context" and "benign samples."

## Key Experimental Results

### Main Results: Vulnerability to Co-authoring Jailbreaks (RQ1/RQ2)

| Model | Prompt | HS (1–5) | ASR |
|------|--------|---------|-----|
| GPT-4o | Vanilla Query HQ | 2.67 | 17.75% |
| GPT-4o | CoJP w/o TF | 3.46 | 23.50% |
| GPT-4o | **CoJP** | **4.87** | **96.75%** |
| Qwen3-8B | CoJP | 4.97 | 99.00% |
| Gemini-2.5-Pro | CoJP | 4.56 | 87.50% |
| DeepSeek-R1-32B | CoJP | 4.94 | 96.25% |

All eight models achieved HS >4.29 and ASR >80% under CoJP. Key discovery: switching from vanilla queries to CoJP caused sharp increases in HS/ASR (GPT-4o ASR 17.75% → 96.75%). Reasoning-enhanced models are not safer—DeepSeek-R1-32B (96.25%) was more vulnerable than its 8B version (84.25%), suggesting that amplified reasoning/instruction-following strengthens the "completion instinct." Regarding moderation (HS judged by GPT-4o with CoT, validated by humans with 95.6% agreement, Spearman $\rho=0.868$): OpenAI-Moderation's detection rate dropped from 85% for vanilla queries to 22% for CoJP.

### Ablation Study: Alignment Performance (RQ3, lower ASR is better, higher $\Delta$Utility is better)

| Model | Method | HQ ASR↓ | CoJP ASR↓ | ΔUtility↑ |
|------|------|---------|-----------|--------|
| LLaMA3-8B | Zero-shot | 24.75% | 80.50% | – |
| LLaMA3-8B | Safety Prompt | 0.00% | 2.75% | **-121.71%** |
| LLaMA3-8B | **SUBA(KTO)** | 15.00% | **5.25%** | **-1.80%** |
| LLaMA3-8B | SUBA(GRPO) | 20.50% | 3.00% | +3.16% |
| DeepSeek-R1-8B | Zero-shot | 27.00% | 84.25% | – |
| DeepSeek-R1-8B | SUBA(KTO) | 24.25% | 11.75% | -3.48% |
| Mistral-7B | SUBA(KTO\B) | 9.75% | 0.00% | -23.86% |

### Key Findings
- **SUBA finds the "Sweet Spot"**: While safety prompts near-zeroed ASR, they crashed utility (-121.71%). SUBA(KTO) suppressed LLaMA3-8B's CoJP ASR to 5.25% while losing only 1.80% utility.
- **Benign data is the core mechanism**: The (\B) variants without benign data saw catastrophic utility drops (Qwen3 GRPO -474.20%), proving that "assist when appropriate" samples are indispensable.
- **Training only on vanilla queries is insufficient**: The (HQ) variant remained highly vulnerable to CoJP (Qwen3-8B 98.50%), showing that co-authoring context samples are required to teach the model to identify hidden intent.
- **Robust across algorithms/architectures**: Both KTO and GRPO reproduced the safety-utility balance, and the method was effective for the reasoning-enhanced DeepSeek-R1 series.
- **Generalization to paraphrased queries**: SUBA maintained low ASR on Paraphrased HQ (LLaMA3-8B 17% → 5%), indicating it learned semantic-level risk identification rather than template overfitting.

## Highlights & Insights
- **Clean Threat Model**: By using "single-turn, fully visible content, purely editing-framed" as an isolated variable, it strips away multi-turn/obfuscation complexities, making prompt-level safety a measurable diagnostic object.
- **"Completion Instinct" as a Counterintuitive Insight**: Models that are better at co-authoring and reasoning are actually more dangerous (32B > 8B), shattering the optimistic assumption that "higher capability equals higher safety."
- **Transferable Contrastive Labels**: The "refuse harmful vs. assist benign" context-aware construction can be transferred to other scenarios requiring boundary control (e.g., medical or legal advice).
- **Clever RAR Metric**: Instead of just looking at absolute toxicity, measuring "how much more harmful the completion is than the draft" accurately characterizes the unique harm of "model-aided escalation."

## Limitations & Future Work
- Draft generation relies on a single drafting model (LLaMA-3-8B) and manual keywords; the diversity of domains and drafts is limited by the human seed, potentially missing new abuse patterns.
- HS/RAR judging depends heavily on GPT-4o; bias and strategy boundaries in the judge model affect absolute values.
- SUBA's HQ ASR remains somewhat high on some models (e.g., LLaMA3 HQ at 15%), as it optimizes for co-authoring safety rather than vanilla queries; it still needs to be deployed with other mechanisms.
- The paper acknowledges producing "executable-level" dangerous details, so the benchmark and data are shared under restricted ethical policies, limiting wider reproduction and community pressure testing.

## Related Work & Insights
- **vs. Automatic Jailbreaking (GCG/PAIR/Crescendo)**: Those rely on gradient search for suffixes or multi-turn escalation; this work is single-turn, fully visible, and requires no optimization search.
- **vs. Obfuscation Attacks (BaitAttack)**: While those hide intent, this work does the opposite—intent is exposed to test the "guardrails against helping even when knowing it's harmful."
- **vs. Safety Prompts / General RLHF**: Safety prompts cause utility to collapse; SUBA uses contrastive benign/harmful samples to train "context-aware refusal" into the weights, achieving a balance safety prompts cannot reach.
- **vs. AdvBench/HarmBench**: Those provide vanilla harmful instructions; this work brings evaluation to the "draft co-authoring" scenario, filling a gap in collaborative writing safety.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic characterization of the neglected draft-based co-authoring jailbreak surface.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 models × multiple prompt settings + dual optimization algorithms; comprehensive, though judge-dependent.
- Writing Quality: ⭐⭐⭐⭐ RQ-driven, clear trajectories, well-defined metrics.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a realistic safety blind spot in LLM co-authoring with a highly transferable alignment idea.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](../../ICLR2026/llm_alignment/jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[ICML 2025\] Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](../../ICML2025/llm_alignment/model_swarms_collaborative_search_to_adapt_llm_experts_via_swarm_intelligence.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)

</div>

<!-- RELATED:END -->
