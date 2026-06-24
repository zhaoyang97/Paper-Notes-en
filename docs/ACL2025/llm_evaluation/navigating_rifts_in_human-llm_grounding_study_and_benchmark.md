---
title: >-
  [Paper Note] Navigating Rifts in Human-LLM Grounding: Study and Benchmark
description: >-
  [ACL2025][LLM Evaluation][Dialogue grounding] This paper systematically studies the grounding (establishing mutual understanding) failure problem in human-LLM dialogues. It reveals that the frequency of proactive clarification by LLMs is only 1/3 of that of humans, and the frequency of proactive follow-up questions is only 1/16. To address this, the authors propose the Rifts benchmark (~1.8K tasks) to evaluate the grounding capabilities of LLMs…
tags:
  - "ACL2025"
  - "LLM Evaluation"
  - "Dialogue grounding"
  - "human-computer interaction"
  - "dialog act analysis"
  - "benchmark evaluation"
  - "clarification requests"
  - "instruction following"
date: 2026-05-08
content_hash: a84e957355a8b55b
---

# Navigating Rifts in Human-LLM Grounding: Study and Benchmark

**Conference**: ACL2025  
**arXiv**: [2503.13975](https://arxiv.org/abs/2503.13975)  
**Code**: [GitHub](https://github.com/microsoft/rifts)  
**Area**: LLM Evaluation  
**Keywords**: Dialogue grounding, human-computer interaction, dialog act analysis, benchmark evaluation, clarification requests, instruction following

## TL;DR

This paper systematically studies the grounding (establishing mutual understanding) failure problem in human-LLM dialogues. It reveals that the frequency of proactive clarification by LLMs is only 1/3 of that of humans, and the frequency of proactive follow-up questions is only 1/16. To address this, the authors propose the Rifts benchmark (~1.8K tasks) to evaluate the grounding capabilities of LLMs, and implement preliminary interventions using a grounding forecaster.

## Background & Motivation

**LLMs are trained as instruction followers**: Current LLMs are optimized for instruction-following via RLHF, but effective dialogue requires participants to collaborate to build "common ground".

**The cost of grounding failure is high**: Consequences range from user frustration to severe outcomes in high-risk scenarios (such as misunderstandings in medical advice or legal consultation).

**LLMs rarely initiate clarification**: Faced with ambiguous instructions, LLMs tend to guess user intent and directly generate responses, rather than disambiguating through questions.

**Early grounding failures cascade and worsen**: After a single grounding failure, the probability of failure in subsequent dialogue turns jumps from 12% to 30% (based on WildChat data).

**Lack of systematic grounding evaluation benchmarks**: Existing dialogue evaluations mostly focus on end-to-end quality, lacking fine-grained assessment of discrete grounding behaviors.

**Asymmetry in human-computer grounding**: In WildChat/Bing Chat, the "heavy lifting" of grounding is almost entirely performed by humans (repair, clarification, follow-ups), while LLMs rarely participate.

## Method

### Overall Architecture

Define grounding behavior taxonomy $\rightarrow$ Build LLM-based annotator to label real dialogue logs $\rightarrow$ Analyze grounding asymmetry between humans and LLMs $\rightarrow$ Train grounding forecaster to predict dialogue trajectory $\rightarrow$ Construct Rifts benchmark based on the forecaster $\rightarrow$ Propose and validate intervention strategies.

### Key Design One: Grounding Behavior Taxonomy

- **Function**: Classifies dialogue actions into three major categories: Advancing (Next Turn, Follow-up, Acknowledgment), Addressing (Reformulation, Repair, Restart), and Disambiguating (Clarification, Overresponse).
- **Mechanism**: Based on the classic grounding theory by Clark & Schaefer, combined with the specific characteristics of LLM dialogue, covering both human-initiated and LLM-initiated actions. Each behavior serves as an observable signal of the grounding state (success / failure / uncertainty).
- **Design Motivation**: Complements prior work by being more comprehensive—not only focusing on human-initiated behaviors (e.g., follow-ups, clarifications) but also incorporating LLM-initiated behaviors (e.g., Overresponse). The three-level taxonomy directly corresponds to the success, failure, and uncertainty states of grounding.

### Key Design Two: Grounding Forecaster

- **Function**: Trains a model to predict future grounding behavior categories (advancing/addressing/disambiguating) in subsequent dialogue turns based solely on the user's initial message.
- **Mechanism**: Utilizes conditional training by appending a grounding prediction token after each user message to fine-tune Llama-3.1-8B. During inference, the logits distribution of the prediction token is analyzed to determine the dialogue's trajectory.
- **Design Motivation**: Post-hoc annotation can only analyze after the fact, whereas the forecaster can anticipate grounding challenges before the dialogue occurs, enabling proactive intervention. This is a highly challenging task, as it requires predicting the user's subsequent actions without seeing the LLM's response (equivalent to marginalizing over all possible assistant responses).

### Key Design Three: Rifts Benchmark Construction and Evaluation

- **Function**: Filters around 1.8K real user prompts from WildChat and stratifies them by the grounding categories predicted by the forecaster (Advancing/Addressing/Disambiguating/No Grounding) to construct a standardized evaluation benchmark.
- **Mechanism**: Employs the forecaster to filter out the prompts with the highest grounding difficulty (top-150 highest logits), and adds prompts that do not require grounding as controls. Evaluation function: Advancing tasks require follow-up, Addressing/Disambiguating tasks require clarification, and No Grounding tasks should not trigger extra grounding.
- **Design Motivation**: Based on real user interactions (rather than synthesized scenarios), with the implicit assumption that certain prompts require back-and-forth communication to build common ground regardless of the LLM's response. Filtering based on the forecaster is more representative than random sampling.

### Loss & Training

The grounding forecaster uses a standard causal language modeling objective (causal language modeling loss). Specifically, cross-entropy loss is computed over sequences containing the grounding token during the fine-tuning of Llama-3.1-8B.

## Key Experimental Results

### Main Results

| Model | Rifts Accuracy |
|------|-------------|
| GPT-4o | 25.26% |
| GPT-4o-mini | 24.48% |
| o3-mini | 25.26% |
| Claude Sonnet 3.5 | 26.95% |
| Claude Opus 3 | 24.57% |
| Llama 3.1 8B | 24.22% |
| Llama 3.1 70B | 23.88% |
| **Llama 3.1 8B + GROUND** | **54.48%** |
| Random Baseline | 33% |

### Ablation Study

| Analysis Dimension | Human-LLM (WildChat/Bing) | Human-Human (MultiWOZ) |
|---------|---------------------------|----------------------|
| Human-initiated repair | High frequency | Low frequency |
| Human-initiated vs LLM-initiated clarification | 3:1 | ~1:1 |
| Human vs LLM follow-up | 16:1 | ~2:1 |
| LLM overresponse | ~30% of assistant turns | Humans rarely overrespond |
| Session restart rate (WildChat) | Higher than single-turn repair rate | — |

### Key Findings

1. **All frontier models perform below the random baseline on Rifts** (avg 23.23% vs 33%). Accuracy on No Grounding is as high as 96%, but accuracy on categories requiring proactive grounding is only 2.22%.
2. **Grounding failure cascade effect**: $P(\text{Turn 1 failure}) = 0.12 \rightarrow P(\text{Consecutive 2-turn failure}) = 0.30 \rightarrow P(\text{Consecutive 3-turn failure})$ continues to rise.
3. **A simple intervention (+ GROUND prompt) yields a 32 percentage point improvement**, from 24.22% to 54.48%, indicating that LLMs possess latent capabilities but lack triggers.
4. **Reasoning models (o3-mini) do not improve grounding**: They often begin reasoning without validating understanding.

## Highlights & Insights

1. **Elegant integration of classical linguistic theory with LLM practices**: Operationalizes Clark's grounding theory into a quantifiable taxonomy of dialogue acts, directly applicable to human-LLM interaction analysis.
2. **Innovative design of the Forecaster**: Predicts grounding trajectory solely from the user prompt without seeing the LLM's response, making proactive intervention possible.
3. **Quantitative discovery of cascading effects**: Demonstrates for the first time with empirical data that early grounding failures snowball to degrade dialogue quality.
4. **Practical value of the Rifts benchmark**: Built from real user interactions, it can be directly used to evaluate and improve the collaborative dialogue capabilities of LLMs.

## Limitations & Future Work

1. Rifts is sourced only from WildChat (interactions with OpenAI models), so the distribution is biased toward the user base and task types of that platform.
2. Grounding behavior annotation relies on GPT-4o-mini, leading to annotator bias, especially with the blurred boundary between clarification and follow-up.
3. The ROC AUC of the forecaster is only 0.61, indicating room for improvement in predictive capability.
4. The intervention strategy is simplified to basic prompt appending, without exploring more sophisticated dialogue strategies (e.g., multi-turn clarification, proactive confirmation).
5. It does not account for the impact of system prompts on LLM grounding behavior (e.g., hidden meta-prompts in Bing Chat).

## Related Work & Insights

### vs Shaikh et al. (2024)

Prior work also analyzes the grounding behaviors of LLMs but only focuses on a subset of human-initiated behaviors (follow-up, confirmation, clarification). This paper extends this to LLM-initiated behaviors (e.g., Overresponse) and constructs a predictive model and standardized benchmark for the first time.

### vs Decision-Theoretic Dialogue (Horvitz & Paek, 2007)

Decision-theoretic methods were previously used in spoken dialogue systems to predict grounding failures and trigger human handovers. This work ports this idea to LLM dialogues, replacing traditional confidence models with a forecaster and replacing human handovers with prompt-based interventions.

### Insights

1. RLHF training should incorporate reward signals for grounding behaviors; the forecaster can serve directly as a reward model.
2. Rifts-level evaluation should become a standard benchmark for LLM dialogue capabilities, complementing existing instruction-following evaluations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First work to systematically quantify the human-LLM grounding gap; the forecaster + benchmark design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across three datasets, multiple frontier models, with complete annotation validation, though forecaster accuracy has room for improvement.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear concepts, logical progression, exquisite figures and tables, closely combining theory and experiment.
- **Value**: ⭐⭐⭐⭐ — Uncovers a critical blind spot in LLM dialogue; the Rifts benchmark provides immediate value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States](towards_dynamic_theory_of_mind_evaluating_llm_adaptation_to_temporal_evolution_o.md)
- [\[ACL 2025\] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles](influences_on_llm_calibration_a_study_of_response_agreement_loss_functions_and_p.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)
- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)

</div>

<!-- RELATED:END -->
