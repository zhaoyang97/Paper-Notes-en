---
title: >-
  [Paper Note] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants
description: >-
  [ACL 2025][LLM (Other)][Personalized assistant] Introduces HiCUPID—the first open-source benchmark that comprehensively addresses five key desiderata of personalized AI assistants (adhering to user profiles, understanding implicit information, multi-info reasoning, long-context modeling, and proactive responses). It contains 1,500 users, each with ~40 dialogues, corresponding QA pairs, and a Llama-3.2 automatic evaluation model.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Personalized assistant"
  - "benchmark"
  - "long context"
  - "user profile"
  - "automatic evaluation"
date: 2026-05-08
content_hash: 23fb164623ac8fd4
---

# HiCUPID: Exploring the Potential of LLMs as Personalized Assistants

**Conference**: ACL 2025  
**arXiv**: [2506.01262](https://arxiv.org/abs/2506.01262)  
**Code**: [GitHub](https://github.com/12kimih/HiCUPID)  
**Area**: NLP / Personalized Assistants  
**Keywords**: Personalized assistant, benchmark, long context, user profile, automatic evaluation

## TL;DR

Introduces HiCUPID—the first open-source benchmark that comprehensively addresses five key desiderata of personalized AI assistants (adhering to user profiles, understanding implicit information, multi-info reasoning, long-context modeling, and proactive responses). It contains 1,500 users, each with ~40 dialogues, corresponding QA pairs, and a Llama-3.2 automatic evaluation model.

## Background & Motivation

**Background**: LLM personalization is a critical capability for next-generation AI assistants, yet suitable public benchmarks for training and evaluation remain scarce.  
**Limitations of Prior Work**: Existing datasets either target classification tasks (unsuitable for generative evaluation), contain dialogues that are too short to test long-context capabilities, or define "personalization" as "assigning a persona to the LLM" rather than "adapting to the user."  
**Key Challenge**: A personalized assistant must simultaneously address five challenging dimensions (AUI, UII, MI, LC, and PR), but no existing dataset covers all of them.  
**Goal**: Construct the first benchmark that comprehensively reflects the multi-dimensional challenges of personalized assistants.  
**Key Insight**: Utilize GPT-4o to synthesize 1,500 multi-dimensional user profiles and generate dialogue histories with naturally embedded personal information alongside QA pairs.  
**Core Idea**: Definitive five-dimensional desiderata, synthetic data, and a Llama-3.2 proxy evaluator.

## Method

### Overall Architecture

GPT-4o data synthesis: 25 personas + 5 profiles + 10 schedules per user $\rightarrow$ naturally embedded dialogue history (~17K tokens) $\rightarrow$ single-info QA (testing single-information extraction) + multi-info QA (testing multi-hop reasoning). Evaluation leverages GPT-4o human preference data distilled into a Llama-3.2-3B automatic evaluator.

### Key Designs

1. **Five-Dimensional Desiderata Definition**:
    - **Function**: Defines 5 desiderata that a personalized assistant must satisfy.
    - **Mechanism**: AUI (adhering to user profiles), UII (understanding implicit information), MI (multi-info reasoning), LC (long-context modeling), and PR (proactive responses), where each dimension corresponds to specific designs in the dataset.
    - **Design Motivation**: Prior work lacked a unified standard to define "what constitutes a good personalized assistant." This five-dimensional definition fills the gap.

2. **Dialogue and QA Data Construction**:
    - **Function**: Generates ~40 dialogues (25 personas + 5 profiles + 10 schedules) and 40 QA pairs for each user.
    - **Mechanism**: Persona dialogues contain 10 rounds to imply user preferences; profile/schedule dialogues consist of single rounds. Single-info QA tests solitary facts, while multi-info QA tests compositional reasoning over persona and profile elements. The dialogue history is ~17K tokens to evaluate LC.
    - **Design Motivation**: Embedding information naturally into dialogues rather than providing it explicitly tests UII capabilities. Setting multi-info QA as cross-dialogue composition tests MI capabilities.

3. **Llama-3.2 Proxy Evaluator**:
    - **Function**: Distills GPT-4o human preference data into a Llama-3.2-3B model to enable cost-effective automatic evaluation.
    - **Mechanism**: The Llama-3.2-3B model is SFT-trained on 400K GPT-4o evaluation samples, achieving a Cohen's kappa of 0.70-0.75 with GPT-4o.
    - **Design Motivation**: GPT-4o evaluation is precise but costly ($26/model), whereas Llama-3.2 provides near-zero marginal cost.

### Loss & Training

SFT: LoRA (r=256, alpha=512, dropout=0.05) fine-tuning, LR=1e-4, 1 epoch. DPO: personalized answers as chosen, generic answers as rejected. The combination of SFT and DPO achieves the best performance.

## Key Experimental Results

### Main Results

Llama-3.2 evaluation scores on Test Set 1 (Seen User/Unseen QA):

| Model | Method | Persona | Schedule | Multi-Info | Total |
|------|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 0-shot | 44.7 | 8.8 | 10.8 | 30.4 |
| GPT-4o-mini | 3-shot | 42.6 | 75.4 | 11.4 | 37.5 |
| Llama-3.1-8B | SFT+DPO | **48.1** | **98.1** | 18.4 | **44.6** |
| Qwen-2.5-7B | SFT+DPO | 43.2 | 99.9 | **38.1** | 44.2 |

### Ablation Study

Impact of long context (Gold dialogue vs. Full history):

| Context Type | GPT-4o-mini Persona | Llama Persona | Gap |
|-----------|:---:|:---:|:---:|
| Gold dialogue (~15 words) | 68.0 | 61.6 | — |
| Full history (~17K tokens) | 44.7 | 39.7 | **-23.3** |

### Key Findings

1. **Schedule is the easiest (99.8%)**: Involves structured, explicit answers; **Multi-Info is the hardest (4-38%)**: Requires compositional reasoning.
2. **Long context is a major bottleneck**: The ~17K token history results in a 23.3% performance degradation.
3. **Pure DPO is highly unstable (5.4%)**: Prior SFT initialization is required for proper convergence.
4. **Optimal few-shot is 3**: Using more than 3 examples becomes detrimental.
5. **Inconsistency of BLEU/ROUGE-L with human preferences**: Mistral scores high on BLEU but receives low human preference ratings.

## Highlights & Insights

- **Five-dimensional desiderata** comprehensively construct the key challenges of personalized assistants for the first time.
- **Llama-3.2 proxy evaluator** distilled from GPT-4o preferences provides cost-effective and highly correlated evaluations.
- **"Personalization = adapting to the user" vs. "Personalization = assigning a persona to the LLM"**—HiCUPID explicitly defines the former.
- **Combination of SFT and DPO behaves best** and generalizes to Unseen Users.

## Limitations & Future Work

- GPT-4o synthetic data might suffer from distribution bias.
- Only English is evaluated.
- Determining the optimal level of personalization remains an unresolved sociological question.
- DPO training is highly sensitive to hyperparameters.

## Related Work & Insights

- **vs. LaMP (Salemi et al. 2024)**: Non-conversational personalization—HiCUPID focuses on conversational settings and tests long-context modeling.
- **vs. PersonaChat**: Defines "personalization" as assigning a persona to the LLM—HiCUPID defines it as adapting to the user.
- **Insights**: Current LLMs underperform significantly in extracting implicit information scattered across long contexts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Five-dimensional desiderata definition + proxy evaluation model.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Open/closed-source models + training/inference methods + ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clarified definition of desiderata and transparent dataset construction.
- **Value**: ⭐⭐⭐⭐ Serves as a standard benchmark for personalized assistant research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs + Persona-Plug = Personalized LLMs](llms_persona-plug_personalized_llms.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)

</div>

<!-- RELATED:END -->
