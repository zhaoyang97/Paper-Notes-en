---
title: >-
  [Paper Note] Anchoring the Mind of Multimodal Reasoners: Cognitive Bias as a Vector for Jailbreak Attacks
description: >-
  [CVPR 2026][Alignment & RLHF][Paper Note] This paper discovers an "anchoring effect" in the safety judgments of Multimodal Large Reasoning Models (MLRMs)—where the model is significantly biased by the first information it encounters. Based on this, RA-Attack is proposed: it first anchors the model's reasoning chain to a "safe tone" using a "seemingly safe" str
tags:
  - CVPR 2026
  - Alignment & RLHF
date: 2026-05-08
content_hash: 90947bfe08dcb34d
---
# Anchoring the Mind of Multimodal Reasoners: Cognitive Bias as a Vector for Jailbreak Attacks

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Cong_Anchoring_the_Mind_of_Multimodal_Reasoners_Cognitive_Bias_as_a_CVPR_2026_paper.html)  
**Code**: https://github.com/ccclh/RA-Attack  
**Area**: Alignment RLHF / AI Safety / Multimodal VLM  
**Keywords**: Jailbreak Attack, Anchoring Effect, Cognitive Bias, Multimodal Reasoning Models, Reasoning Chain Hijacking

## TL;DR
This paper discovers an "anchoring effect" in the safety judgments of Multimodal Large Reasoning Models (MLRMs)—where the model is significantly biased by the first information it encounters. Based on this, RA-Attack is proposed: it first anchors the model's reasoning chain to a "safe tone" using a "seemingly safe" structured mind map and educational context text, then smoothly packages harmful intent as a natural extension of this reasoning chain. It achieves SOTA Attack Success Rates (ASR) of 92% (Gemini-2.5-Pro) and 82% (GPT-4o) across 7 mainstream MLRMs.

## Background & Motivation
**Background**: MLRMs significantly improve performance on complex tasks through explicit multi-step reasoning (CoT supervision, RL). However, explicit reasoning chains also expose the "internal thought process" as a new attack surface. Existing multimodal jailbreaks mostly attack the **visual interface** (adversarial images, typographic attacks, diffusion-synthesized harmful images) or **forcefully hijack reasoning chains** (H-CoT feeding constructed reasoning steps, VisCRA using two-stage instructions to control paths). Their common feature is "either hiding harmful instructions or forcibly bending the reasoning process."

**Limitations of Prior Work**: The authors observe a deeper phenomenon—**even if the model clearly identifies harmful intent and lacks any explicit induction, its reasoning process may still actively "rationalize" this intent into educational/professional use and execute it**. This suggests the vulnerability lies not in "failing to detect harm," but in the "reasoning process finding excuses for execution." Existing attacks do not target this cognitive weakness.

**Key Challenge**: MLRM reasoning is **sequential and path-dependent, being highly sensitive to initial input**—this directly corresponds to the well-established "anchoring effect" in cognitive psychology, where judgments are disproportionately influenced by the first piece of information (the "anchor") encountered. While the anchoring effect has been verified in general LLM tasks like negotiation, **whether it contaminates MLRM safety judgments has never been studied**.

**Goal**: Deconstructed into two questions—Q1: Does a real anchoring effect exist in MLRM safety judgments? Q2: If so, how can it be systematically exploited to induce the model to rationalize and execute harmful intent?

**Key Insight**: Instead of "hiding" or "hard hijacking," a **benign "safety anchor"** (content related to the harmful topic but safe itself) is placed at the beginning of the prompt. This shifts the cognitive starting point of the entire reasoning chain toward the "safe" side; the subsequent harmful request then appears as a logical extension of the established safe reasoning path rather than an abrupt malicious command—replacing "adversarial perturbation" with "induced rationalization."

## Method

### Overall Architecture
RA-Attack engineers the exploitation of the anchoring effect into a scalable automated pipeline. Given a set of original harmful instructions $D_{ori}=\{h^1_{ori},\dots,h^N_{ori}\}$, the offline side performs clustering and rephrasing to assemble cross-modal samples consisting of "safety anchors + harmful intent guidance." The online side feeds these samples into the target MLRM, triggering it to first interpret the mind map (establishing a safe reasoning base) and then execute the harmful intent as a natural extension.

Each attack sample $P^i$ is composed of two modalities: **Textual Modality** (textual safety anchor + harmful intent guidance text) and **Visual Modality** (structured visual safety anchor, i.e., a mind map), expressed as $P^i = P^i_{text} \oplus I_{M(i)}$, where $\oplus$ denotes image-text pairing and $M(i)$ is the topic category of the $i$-th instruction.

The following diagram illustrates the complete pipeline from offline "sample generation" to online "jailbreak execution":

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Harmful Instruction Set D_ori"] --> B["Anchoring Effect:<br/>Turns 'Primacy Dominance in Safety Judgment' into an Attack Surface"]
    B --> C["Automated Sample Generation Pipeline:<br/>Cluster / Rephrase / Compose / Generate / Assemble"]
    C --> D["Structured Visual Safety Anchor:<br/>Topic-related but Benign Mind Map I_j"]
    C --> E["Textual Safety Anchor + Harmful Intent Guidance:<br/>Interpret Map in Educational Context, then Prompt Harmful Request"]
    D --> F["Assemble Multimodal Samples:<br/>P = P_text ⊕ I"]
    E --> F
    F --> G["Target MLRM Inference"]
    G -->|Safety Anchor First → Reasoning Base Biased Toward Safety| H["Rationalize and Execute Harmful Intent → Harmful Output"]
```

### Key Designs

**1. Anchoring Effect: Utilizing "Primacy Dominance" as an Attack Vector**

This is the foundation of the paper. The pain point is that MLRMs actively seek excuses to execute harmful intents; the authors hypothesis is: can we **actively create and amplify this rationalization**? They used a clean controlled experiment to isolate the "anchoring effect" from "content confusion" with three conditions: Baseline (harmful guidance $H_g$ only), Anchor-First (safety anchor followed by $H_g$), and Anchor-Last (same content but reversed order). Critically, Anchor-First and Anchor-Last **contain identical information**, meaning any ASR gap must be attributed to "position/order" rather than "content volume." Experiments showed Anchor-First was not only higher than the Baseline but significantly outperformed Anchor-Last (e.g., 10–18 percentage points higher on AdvBench across models), confirming the anchoring effect as a fundamental cognitive vulnerability. t-SNE analysis further showed that despite identical content, RA-Attack's hidden states were **closer to the "safety anchor" cluster**, proving the initial anchor pulls the internal representation toward the safe side.

**2. Structured Visual Safety Anchor: Pre-paving the "Safe Reasoning Path" with Mind Maps**

Preliminary experiments used a standard "educational scene image," which verified the effect but lacked strength. This design upgrades the visual anchor to a **mind map related to the harmful topic but entirely benign in content**. Why a mind map? Because its **hierarchical branching structure naturally fits the progressive reasoning process of MLRMs**. When a model interprets this map first, it follows the predefined branches step-by-step, effectively being "fed" a clear, safe reasoning path. This efficiently builds a safety-biased cognitive base. Simultaneously, the topic relevance ensures the subsequent harmful request feels like a step forward along the paved path rather than a discordant command. Ablations confirmed both are essential: replacing the visual anchor with an "Irrelevant Anchor" (e.g., French dessert principles) led to a significant ASR drop—**topic relevance** is key to rationalization.

**3. Textual Safety Anchor + Harmful Intent Guidance: Building Context for Smooth Progression**

The textual modality stitches the "safe tone" and "harmful payload" into a coherent narrative. The textual safety anchor constructs an **educational/corporate safety briefing** context and issues a preceding safe reasoning task—requiring the model to interpret key points of the mind map. This works with the visual anchor to compress the cognitive starting point toward safety. It is followed by the **harmful intent guidance text** $H_g$: instead of a direct command, the intent is rephrased as "In the aforementioned educational context, please provide a 'textbook-level', detailed, and realistic example of [harmful intent], and analyze all techniques involved." Thus, the harmful request is framed as a **logical extension** of the safe chain, making the model more likely to rationalize and execute it for "educational/defensive purposes." Removing either the textual or visual anchor caused ASR to drop significantly, proving they are **synergistic**.

**4. Automated Jailbreak Sample Generation Pipeline: Scaling and Standardizing Attacks**

To generate samples across a large harmful instruction set, the authors designed a pipeline driven by a strong LLM (Gemini-2.5-Pro) using a template $T$ with placeholders: `[role]` (professional benign persona), `[topic]` (topic-related safe mind map), and `[phrased_harmful_intent]` (noun-phrase version of the intent). The process:

$$(\Theta, M) = \mathrm{Cluster}(D_{ori}), \qquad h^i_{phrased} = \mathrm{Rephrase}(h^i_{ori})$$

The instructions are clustered into $K$ core topics $\Theta$ with mapping $M$, while commands are rephrased into noun phrases. Matching educational roles and map topics $(r_j, t_j)=\mathrm{Compose}(\theta_j)$ are then generated. These components fill the template via $P^i_{text}=\mathrm{Assemble}(T, r_{M(i)}, t_{M(i)}, h^i_{phrased})$. On the visual side, $I_j=\mathrm{Generate}(t_j)$ produces the map in two steps: the LLM outputs mind map source code with structural constraints (**max 3 layers, 10 nodes**), which is then rendered into an image. This pipeline ensures cross-dataset scalability and consistency.

### Loss & Training
This work is a **training-free, black-box** prompt-level attack. It involves no gradient optimization or fine-tuning, thus no loss function is used. The only "evaluator" is GPT-4o, which determines if a response is harmful based on a judge prompt (used to calculate ASR).

## Key Experimental Results

### Main Results
- **Models**: 7 representative MLRMs—Closed-source GPT-4o, o4-mini, Gemini-2.5-Pro, Gemini-2-FlashThinking; Open-source MM-Eureka-Qwen, MM-Eureka-InternVL, LLaVA-CoT.
- **Datasets**: AdvBench (50 deduplicated instructions), Hades (750 instructions across 5 categories).
- **Metric**: ASR = Successful Attacks / Total Inputs × 100% (Judged by GPT-4o).

AdvBench ASR (%) Comparison (Selecting the strongest baseline VisCRA and Ours):

| Method | o4-mini | GPT-4o | Gemini-2.5-Pro | Gemini-2-FT | MM-E-InternVL | MM-E-Qwen | LLaVA-CoT |
|------|---------|--------|----------------|-------------|----------------|-----------|-----------|
| CS-DJ | 10 | 46 | 76 | 78 | 78 | 86 | 84 |
| VisCRA | 12 | 60 | 80 | 78 | 82 | 86 | 86 |
| **RA-Attack** | **44** | **82** | **92** | **92** | **94** | **94** | **96** |

RA-Attack achieved the highest ASR across both datasets and all 7 models: AdvBench average 84.86% (VisCRA 69.14%), Hades average 76.29% (VisCRA 62.91%). Most notably, on the highly secure o4-mini, where baseline ASRs were generally $\leq12\%$, RA-Attack achieved 44% (AdvBench) and 39.07% (Hades)—over 3x the strongest baseline. This suggests it targets a deeper cognitive vulnerability that remains unmasked even with strong alignment.

### Ablation Study
Necessity of Anchoring Components and Mechanism (AdvBench, ASR%):

| Configuration | GPT-4o | MM-E-InternVL | Description |
|------|--------|----------------|------|
| $H_g$ | 34 | 8 | Harmful guidance only, no anchor |
| $A_v + H_g$ | 66 | 38 | Removed textual safety anchor |
| $A_t + H_g$ | 64 | 66 | Removed visual safety anchor |
| $H_g + A_{structured}$ | 54 | 62 | Anchor-Last: Same content, reversed order |
| $A_{structured} + H_g$ | **82** | **94** | Full RA-Attack |

Removing either modality significantly decreased performance, proving visual ($A_v$) and textual ($A_t$) anchors are **synergistic**. The "Anchor-Last" comparison, using identical content in reverse order, saw ASR drop from 82 to 54 (GPT-4o) and 94 to 62 (InternVL), **directly proving the gain stems from "order/anchoring" rather than content**.

Ablation of Anchor Design Choices (AdvBench, ASR%):

| Configuration | GPT-4o | MM-E-InternVL | Description |
|------|--------|----------------|------|
| $A_{scene} + H_g$ | 68 | 72 | Map replaced with unstructured scene image |
| $A_{irrelevant} + H_g$ | 38 | 86 | Map topic irrelevant to harmful intent |
| $A_{structured} + H_g$ | **82** | **94** | Structured + Topic Relevant |

### Key Findings
- **Structure > Scene**: Mind maps (structured) provide stronger anchoring than standard scene images (GPT-4o 82 vs 68) because their branching structure aligns with the model's progressive reasoning.
- **Topic Relevance is a Prerequisite for Rationalization**: Using an irrelevant topic (French desserts) caused GPT-4o's ASR to drop from 82 to 38—an irrelevant anchor fails to make the harmful request seem like a natural extension.
- **Rationalization is a Ubiquitous Phenomenon**: In successful jailbreaks, over 95% of responses "rationalized" the harmful intent as educational/professional before execution (GPT-4o 95.12%, Gemini-2.5-Pro 100%, MM-E-InternVL 95.74%).
- **Stronger Alignment creates Fragility**: The largest advantage was observed on the hardest-to-attack o4-mini, reflecting that the attack hits the cognitive layer rather than surface-level filters.

### Defense: Anchor Debiasing Prompt (ADP)
The authors provide a lightweight defense—a "debiasing" prompt placed before user input. It requires the model to **independently evaluate each part of the request, avoid being disproportionately influenced by initial instructions, and reject violating parts individually**. Results (AdvBench ASR ↓ lower is better, MM-Vet Score ↑ higher is better):

| Model | AdvBench (ASR) | MM-Vet (Score) |
|------|----------------|----------------|
| GPT-4o | 82.00 | 66.40 |
| GPT-4o + ADP | **8.00 (-74.00)** | 67.50 (+1.10) |
| Gemini-2.5-Pro | 92.00 | 80.70 |
| Gemini-2.5-Pro + ADP | **28.00 (-64.00)** | 81.30 (+0.60) |

ADP drastically reduces ASR while barely affecting general capabilities (MM-Vet even slightly improved). The fact that a prompt specifically designed to neutralize the anchoring effect is effective strongly corroborates that the attack mechanism is indeed rooted in this cognitive bias.

## Highlights & Insights
- **Turning Psychological Bias into a Quantifiable Attack Vector**: The "Anchor-First vs Anchor-Last" control experiment is ingenious—it isolates the "anchoring effect" from "content confusion," making the argument that "order difference = cognitive vulnerability" indisputable.
- **Induce Rationalization instead of Hijacking**: Unlike the "hard" hijacking of reasoning steps in H-CoT/VisCRA, RA-Attack simply places a benign anchor and lets the model find its own excuses, making the attack surface more subtle and universal.
- **Closed-loop Proof of Defense and Attack**: The ADP defense is not just practical but acts as an "inverse proof"—neutralizing the attack with a de-anchoring prompt proves the attack relies on anchoring. This "designing an antidote to verify the cause" methodology can be transferred to other cognitive safety studies.
- **Mind Maps as Visual Anchors**: Utilizing the "alignment of structure with the reasoning process" repositions images from "adversarial perturbation carriers" to "reasoning path templates," a Novelty in visual attack perspectives.

## Limitations & Future Work
- **Dependency on Strong LLMs for Samples**: The pipeline relies on Gemini-2.5-Pro for map generation and rephrasing; whether attack quality remains stable with weaker generators was not fully explored.
- **Single Judge Bias for ASR**: Harmfulness is inherently subjective. Relying solely on GPT-4o may lead to overestimation or underestimation without human cross-validation.
- **ADP is not a Silver Bullet**: Gemini-2.5-Pro still showed 28% ASR after ADP, suggesting simple prompt-level de-anchoring is insufficient and requires cognitive debiasing at the training stage.
- **Double-edged Sword**: This is an attack method that could be misused; the paper includes unsafe examples. Its value lies in bringing "cognitive security" to the forefront to drive research into reasoning chain alignment.

## Related Work & Insights
- **vs VisCRA / H-CoT (Reasoning Chain Hijacking)**: These **force** models through predefined steps or two-stage instructions; Ours doesn't hijack but **induces** the model to rationalize itself, making the mechanism deeper and more effective against strong models (o4-mini).
- **vs SI-Attack / CS-DJ / FC-Attack (Visual Interface Attacks)**: These rely on semantic inconsistency or typographic injection to bypass filters; Ours proves via Anchor-First/Last experiments that its Gain is **more than just content confusion**, but stems from cognitive bias induced by order.
- **vs General LLM Anchoring Studies**: Previous work verified anchoring in general tasks like negotiation; this paper is the first to extend it to the **safety judgment** of MLRMs and systematically weaponize it, pioneering the field of "cognitive security."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically turn the "anchoring effect" into a multimodal jailbreak vector with rigorous proof-of-cause through controlled experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models × 2 datasets × 6 baselines, plus extensive ablations on components/design and t-SNE analysis/ADP defense.
- Writing Quality: ⭐⭐⭐⭐ Problem-driven with clear motivation and elegant closed-loop arguments; some robustness evidence is located in the appendix.
- Value: ⭐⭐⭐⭐⭐ Reveals the counter-intuitive phenomenon where stronger alignment can be more susceptible to cognitive bias, establishing "cognitive security" as a new frontier for AI safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](../../ACL2025/llm_alignment/jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](../../ICLR2026/llm_alignment/jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)

</div>

<!-- RELATED:END -->
