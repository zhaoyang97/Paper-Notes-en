---
title: >-
  [Paper Note] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
description: >-
  [ACL 2026][LLM/NLP][spoken language models] This paper identifies a phenomenon termed "style amnesia," in which spoken language models (SLMs) fail to maintain initially specified speaking styles (emotion, accent, volume, speech rate) across multi-turn conversations. Attention analysis reveals the underlying cause as attention dilution, and an explicit recall process is proposed as a mitigation strategy.
tags:
  - ACL 2026
  - LLM/NLP
  - spoken language models
  - style amnesia
  - multi-turn dialogue
  - speaking style
  - instruction following
date: 2026-05-08
content_hash: 64c07c28e48fbe28
---

# Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models

**Conference**: ACL 2026
**arXiv**: [2512.23578](https://arxiv.org/abs/2512.23578)
**Code**: [GitHub](https://github.com/YuXiangLin1234/SLM-Style-Amnesia)
**Area**: Spoken Language Models
**Keywords**: spoken language models, style amnesia, multi-turn dialogue, speaking style, instruction following

## TL;DR

This paper identifies a phenomenon termed "style amnesia," in which spoken language models (SLMs) fail to maintain initially specified speaking styles (emotion, accent, volume, speech rate) across multi-turn conversations. Attention analysis reveals the underlying cause as attention dilution, and an explicit recall process is proposed as a mitigation strategy.

## Background & Motivation

**Background**: Spoken language models (e.g., GPT-4o, Gemini Live, Qwen2.5-Omni) can follow user-specified speaking styles (emotion, accent, speech rate, etc.) in single-turn interactions, demonstrating impressive expressive capabilities.

**Limitations of Prior Work**: Existing research focuses almost exclusively on single-turn evaluation, leaving style consistency across multi-turn conversations essentially unstudied. In practice, however, users set a style at the beginning of a conversation and expect the SLM to maintain it throughout the session without repeating instructions every turn.

**Key Challenge**: SLMs follow style instructions reasonably well in the first turn, but compliance drops sharply as the conversation progresses. The model does not truly "forget" the instruction—recall tests show it can accurately reproduce the instruction—yet it fails to execute what it remembers.

**Goal**: To systematically evaluate and analyze SLMs' ability to maintain speaking style across multi-turn dialogue, identify the root cause, and explore mitigation approaches.

**Key Insight**: An end-to-end evaluation framework is constructed using a user simulator to conduct realistic interactive multi-turn conversations, measuring style compliance turn by turn.

**Core Idea**: The fundamental cause of style amnesia is attention dilution—as conversation turns increase, the model's average attention weight on style-instruction tokens decays from ~8% to <0.6%, rather than reflecting a true loss of memory.

## Method

### Overall Architecture

The evaluation framework consists of three core components: (1) **style instructions**—ten style conditions specified at the start of each conversation, covering emotion (sad/happy/angry/neutral), accent (North American/Indian English), volume (high/low), and speech rate (fast/slow); (2) **conversation topics**—100 diverse conversation openers sampled from the Soda dataset; and (3) **multi-turn interaction**—a cascaded SLM (ASR + GPT-4o mini + TTS) serves as a user simulator to conduct four-turn realistic dialogues with the evaluated SLM.

### Key Designs

1. **Turn-Level Instruction-Following Rate (Turn-Level IF Rate)**

    - **Function**: Quantifies the trend of style compliance degradation across conversation turns.
    - **Mechanism**: Defines the first-turn compliance rate $IF_1$ and a degradation rate $D = \sum_{j=2}^{K} \frac{\max(IF_1(s) - IF_j(s), 0)}{K-1}$ to separately capture baseline capability and degree of degradation. Four dedicated automatic judges evaluate emotion (Emotion2vec-Large), accent (Voxlect), volume (LUFS), and speech rate (WPM), respectively.
    - **Design Motivation**: Unlike approaches that aggregate a single global score, turn-level analysis precisely reveals when and how degradation begins.

2. **Attention Dynamics Analysis**

    - **Function**: Reveals the internal mechanism underlying style amnesia.
    - **Mechanism**: The average attention weights that an open-source model (Step-Audio 2 mini) assigns to style-instruction tokens during response generation are extracted. Results show: Turn 1 ~8.3%, Turn 2 ~1.6%, Turn 3 ~0.87%, Turn 4 ~0.58%—severe attention dilution that closely correlates with IF rate degradation.
    - **Design Motivation**: Distinguishes "forgetting the instruction" from "failing to execute it." If the issue were memory, prompt engineering would suffice; attention dilution points instead to the need for architectural improvements.

3. **Recall Process**

    - **Function**: Explores a mitigation strategy for style amnesia.
    - **Mechanism**: Before each turn from Turn 2 onward, the SLM is prompted to recall the initial style instruction before processing the user input. Experiments show that most models recall accurately (near 100% for closed-source models), and the recall process significantly reduces the degradation rate (e.g., GPT-4o mini on the sad style drops from 65.3% to 30.3%).
    - **Design Motivation**: Tests whether the model still remembers the instruction and whether explicit recall can improve execution.

### Text–Acoustic Co-Analysis

For emotion styles, both semantic and acoustic features undergo style amnesia simultaneously—textual content and vocal expression degrade in tandem. For speech rate, different models adopt different strategies: Gemini Live achieves "speaking fast" by reducing word count, while GPT-4o achieves it through acoustic acceleration rather than content compression. Nevertheless, the WPM gap between fast and slow conditions narrows consistently as turns progress across all models.

## Key Experimental Results

### Main Results

| Model | Style | $IF_1$ (Turn 1) | Degradation Rate $D$ |
|---|---|---|---|
| GPT-4o mini | Sad | ~85% | 65.3% |
| GPT-4o mini | Indian accent | ~75% | 49.7% |
| GPT-4o | Sad | ~95% | 26.7% |
| Gemini Live | Sad | ~85% | 21.3% |
| Step-Audio 2 mini | Sad | ~70% | 14.0% |
| Cascaded baseline (TTS) | All emotions | ~95% | <3.0% |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| Instruction in system message | $IF_1$ drops 30–80% | System message yields worse compliance |
| Instruction in user message | Higher $IF_1$ | Default setting performs better |
| + Recall process | $D$ reduced by 3–35% | GPT-4o mini benefits most |
| Attention weight | 8.3% → 0.58% | 14× decay within 4 turns |

### Key Findings

- Style amnesia appears in all five evaluated models (3 closed-source + 2 open-source) without exception.
- Models "remember" the instruction but "cannot execute" it—recall rate is near 100% yet IF rate continues to decline.
- **System message paradox**: System messages are designed for globally persistent instructions, yet SLMs comply worse with style instructions placed in system messages.
- **Default style bias**: Models tend to revert to default styles such as "happy/neutral" emotion and "North American" accent.
- The cascaded baseline (providing style instructions to TTS at every turn) shows almost no degradation, confirming that the problem lies in the architecture of end-to-end SLMs.

## Highlights & Insights

- **Identifies an important and previously overlooked problem**: Style amnesia is a critical obstacle to the practical deployment of SLMs.
- **Distinguishes "memory" from "execution"**: Recall tests precisely localize the issue to attention allocation rather than memory, pointing the way toward targeted solutions.
- **Rigorous evaluation framework**: Real interactive dialogue via simulator, four dedicated judges, and human validation ensure high reliability.
- **System message paradox is a valuable finding**: It exposes a deep architectural issue in current SLM design.

## Limitations & Future Work

- **Limited style categories**: Only four paralinguistic attributes are covered; more complex styles such as prosodic variation and role-playing are not addressed.
- **No multi-style combinations**: Current models cannot even maintain a single style consistently; multi-style combinations are left for future work.
- **Attention analysis limited to one open-source model**: Only Step-Audio 2 mini is analyzed.
- Future directions include style-anchored attention mechanisms, persistent representations of style embeddings, and multi-style combination compliance.

## Related Work & Insights

- **vs. Multi-Bench**: Also evaluates multi-turn SLMs but aggregates only global scores; this paper provides turn-level analysis.
- **vs. VocalBench/VoxDialogue**: Use predefined dialogues rather than real interactive conversations, precluding turn-level analysis.
- **vs. multi-turn degradation research in text LLMs**: Similar multi-turn performance degradation has been identified in the text domain; this paper extends the phenomenon to paralinguistic features in the speech domain.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic investigation of style amnesia in SLMs; reveals the key insight that models "remember but cannot execute."
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 5 models, 10 styles, and 1,000 conversation sets, with attention analysis and mitigation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is clear, experiments build progressively, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Identifies a critical barrier to SLM deployment with clear implications for model design and training.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](../../ICLR2026/llm_nlp/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)

<!-- RELATED:END -->
