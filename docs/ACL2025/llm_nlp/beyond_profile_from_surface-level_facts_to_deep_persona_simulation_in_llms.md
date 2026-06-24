---
title: >-
  [Paper Note] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs
description: >-
  [ACL 2025][LLM (Other)][Deep Persona Simulation] This paper proposes CharacterBot, which learns the linguistic style and deep cognitive patterns of Lu Xun from his 17 essay collections through four training tasks (Author Perspective Reconstruction pre-training + multiple-choice/generative QA/style transfer fine-tuning) paired with the CharLoRA parameter update mechanism, significantly outperforming various baselines in linguistic accuracy and viewpoint understanding.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Deep Persona Simulation"
  - "CharLoRA"
  - "Lu Xun"
  - "Style Transfer"
  - "Multi-Task Fine-Tuning"
date: 2026-05-08
content_hash: 454b8daffe96ba5a
---

# Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.12988](https://arxiv.org/abs/2502.12988)  
**Code**: [https://github.com/CharacterBot](https://github.com/CharacterBot)  
**Area**: LLM / Role-Playing  
**Keywords**: Deep Persona Simulation, CharLoRA, Lu Xun, Style Transfer, Multi-Task Fine-Tuning

## TL;DR
This paper proposes CharacterBot, which learns the linguistic style and deep cognitive patterns of Lu Xun from his 17 essay collections through four training tasks (Author Perspective Reconstruction pre-training + multiple-choice/generative QA/style transfer fine-tuning) paired with the CharLoRA parameter update mechanism, significantly outperforming various baselines in linguistic accuracy and viewpoint understanding.

## Background & Motivation
**Background**: LLM persona simulation is primarily achieved through memorizing basic personal information or fine-tuning with dialogue data. Prompt engineering methods dynamically inject character descriptions but fail to capture deep traits.

**Limitations of Prior Work**: Existing methods oversimplify persona representations, limiting them to surface-level conversations or basic attribute descriptions. Genuine persona simulation requires capturing deep-seated identity elements such as worldviews, ethical frameworks, and context-dependent viewpoints.

**Key Challenge**: "Books are the mirror of the soul" — a writer's works inherently embody their beliefs, insights, and response patterns, reflecting the essence of a persona far better than résumé-style descriptions. However, how to extract these deep traits from literary works remains a major challenge.

**Goal**: To learn linguistic style and ideological depth from the complete works of an author, constructing deep persona simulations that go beyond surface-level imitation.

**Key Insight**: Taking Lu Xun as a case study, a multi-task framework of pre-training + three-task fine-tuning is designed, which pairs with CharLoRA to maintain cross-task persona consistency.

**Core Idea**: Use Author Perspective Reconstruction (APR) pre-training to learn linguistic style; use multiple-choice questions, question answering, and style transfer fine-tuning to learn ideologies; and use the CharLoRA shared-specific parameter mechanism to maintain persona consistency.

## Method

### Overall Architecture
Pre-training Stage:
- **APR (Author Perspective Reconstruction)**: Converts Lu Xun's first-person essays into third-person ("The author believes/criticizes") to help the model establish attribution links between viewpoints and the author.
- Performs next-token prediction on the original texts + APR versions.

Fine-tuning Stage (3 tasks):
- **Multiple-Choice QA**: Designs 4-option multiple-choice questions derived from the essays to test the understanding of the author's opinions.
- **Generative QA**: Open-ended question answering where outputs must reflect Lu Xun's rhetorical and ideological patterns.
- **Style Transfer**: Rewrites style-free texts into Lu Xun's style while maintaining semantic consistency.

### Key Designs

1. **CharLoRA**:

    - Function: Adapts LoRA for persona simulation optimization — the pre-training stage learns a shared $\mathbf{A}_{pt}$ to encode general style, while during fine-tuning, each task has an independent $\mathbf{B}_i$ to encode task-specific patterns.
    - Forward propagation: $\mathbf{h}_i = W_0 \mathbf{x} + \mathbf{B}_i \mathbf{A}_{pt} \mathbf{x}$
    - During fine-tuning, only the active task's $\mathbf{B}_i$ and the shared $\mathbf{A}_{pt}$ are updated, while other task matrices are frozen.
    - Design Motivation: The shared matrix ensures persona consistency (Lu Xun's core voice penetrates all tasks), while specialized matrices allow task adaptation.

2. **Data Generation**: GPT-4o was utilized to generate training data, consisting of 638 essays $\times$ (3 QA sets + 3 style transfer pairs) per essay, followed by manual quality validation.

## Key Experimental Results

### Main Results

| Model | MCQ Accuracy | Gen QA Content Score | Gen QA Style Score | Style Transfer BLEU | Style Transfer ROUGE-1 | Style Matching |
|------|-----------|------------|------------|------------|---------------|---------|
| Llama3.1-8B | 0.614 | 2.370 | 1.354 | 0.113 | 0.264 | 0.267 |
| Qwen2.5-7B | 0.787 | 2.828 | 2.818 | 0.115 | 0.233 | 0.456 |
| GPT-4o | 0.734 | 3.214 | 2.542 | 0.088 | 0.196 | 0.471 |
| Tongyi Xingchen | 0.788 | 3.172 | 2.823 | 0.101 | 0.187 | 0.534 |
| **CharacterBot** | **0.880** | **3.214** | **2.885** | **0.293** | **0.410** | **0.937** |

### Key Findings
- CharacterBot achieves a style matching score of 0.937 (vs. 0.534 for the second best), demonstrating that deep training captures writing styles far better than generic persona models.
- The multiple-choice accuracy of 0.880 significantly outperforms GPT-4o's 0.734 — proving the model truly understands Lu Xun's perspectives, rather than merely imitating surface features.
- The BLEU/ROUGE scores for style transfer are also substantially ahead, demonstrating that CharLoRA effectively maintains style consistency.

## Highlights & Insights
- **"Works-as-a-Mirror" Persona Simulation Paradigm**: Instead of memorizing personal biographies, the model learns thought processes and styles from the author's complete works — yielding deeper and more authentic simulations.
- **Shared-Specific Separation in CharLoRA**: The design of a general linguistic style expert coupled with task-specific experts is simple yet effective, and generalized easily to other multi-task persona simulations.
- **Ingenuity of APR Pre-training**: Converting first-person to third-person with attribution tags helps the model establish relationships of "who said what."

## Limitations & Future Work
- Only validated on a single author (Lu Xun); generalization to other characters needs to be verified.
- Training data was generated by GPT-4o, meaning the upper bound of quality is constrained by GPT-4o's comprehension of Lu Xun.
- Only supports Chinese; cross-lingual persona simulation has not been tested.
- The evaluation metrics for deep "ideological simulation" remain insufficiently objective — how to quantify "depth of thought" remains an open question.
- Ethical risks: High-fidelity persona simulation could be exploited to mislead or fabricate statements by famous figures.

## Related Work & Insights
- **vs. CharacterGLM**: CharacterGLM is trained based on persona description dialogues and yields poor results (only 0.073 accuracy); the complete works-driven approach in this work is more effective.
- **vs. LuXun-GPT**: Focusing solely on style transfer, whereas this work is more comprehensive (covering both style and ideology) and achieves a vastly superior style matching score of 0.937 vs. 0.387.
- **vs. Prompt-based Role-Playing**: Prompt engineering fails to capture deep-seated traits, whereas CharacterBot achieves a deeper understanding via fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm of learning deep character traits from literary works is highly novel, and the design of CharLoRA is highly creative.
- Experimental Thoroughness: ⭐⭐⭐ Only evaluated on a single author, and some evaluation metrics rely on GPT-4o grading.
- Writing Quality: ⭐⭐⭐⭐ The motivation is well-argued, and the methodology is clearly described.
- Value: ⭐⭐⭐⭐ It opens up a new direction for deep persona simulation, and CharLoRA is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs + Persona-Plug = Personalized LLMs](llms_persona-plug_personalized_llms.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ACL 2025\] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](meaning-beyond-truth-conditions-anaphora-accessibility.md)
- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](../../ACL2026/llm_nlp/personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)

</div>

<!-- RELATED:END -->
