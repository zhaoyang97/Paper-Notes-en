---
title: >-
  [Paper Note] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning
description: >-
  [ACL 2026][Medical NLP][Multilingual Medical Reasoning] The authors construct CureMed-Bench, a medical reasoning dataset covering 13 languages (including low-resource languages like Amharic, Yoruba, and Swahili) with 15…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Multilingual Medical Reasoning"
  - "GRPO"
  - "Curriculum Learning"
  - "Code-switching SFT"
  - "Low-resource Languages"
date: 2026-05-08
content_hash: f6f2a81d24cb7053
---

# CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning

**Conference**: ACL 2026 Oral  
**arXiv**: [2601.13262](https://arxiv.org/abs/2601.13262)  
**Code**: cure_med (Paper link provided, repository address not explicitly in cache)  
**Area**: Medical NLP / Multilingual LLMs / Reinforcement Learning  
**Keywords**: Multilingual Medical Reasoning, GRPO, Curriculum Learning, Code-switching SFT, Low-resource Languages

## TL;DR
The authors construct CureMed-Bench, a medical reasoning dataset covering 13 languages (including low-resource languages like Amharic, Yoruba, and Swahili) with 15,774 open-ended QA pairs. They propose Cure-Med: a two-stage "code-switching aware SFT + Curriculum GRPO" framework that jointly optimizes reasoning accuracy and language consistency. This approach improves language consistency/reasoning accuracy to 85.21% / 54.35% for the 7B model and 94.96% / 70.04% for the 32B model.

## Background & Motivation
**Background**: Mainstream medical LLMs either follow "closed MCQ + supervised fine-tuning" (MedQA / MMedBench / MedMCQA) or perform monolingual open-ended QA (HealthSearchQA). Evaluations remain English-centric, leaving multilingual medical reasoning nearly blank.

**Limitations of Prior Work**: On non-English languages, especially low-resource ones like Amharic / Yoruba / Hausa, LLMs suffer from two types of failures: (1) significant drops in logical accuracy; (2) "language drift" (input is in Swahili, but the intermediate or final answer reverts to English), making them unusable in clinical settings.

**Key Challenge**: To gain the trust of doctors/patients, the system must achieve both "transparent reasoning processes" and "stable output language." Existing SFT often sacrifices reasoning depth, while pure RL suffers from sparse rewards and poor early signals for low-resource languages, making it difficult to optimize both simultaneously.

**Goal**: (i) Provide the community with an open-ended medical reasoning benchmark covering 13 languages; (ii) train a model that optimizes both "logical correctness" and "language fidelity" while remaining robust to low-resource languages.

**Key Insight**: The authors observe that reward signals are more stable for high-resource languages. Therefore, "language resource level" is treated as curriculum difficulty. RL is first stabilized on high-resource languages before gradually introducing mid- and low-resource languages. Simultaneously, intermediate reasoning is allowed to code-switch (thinking in English + clinical terminology, final answer in the target language) to preserve reasoning depth while stabilizing the output language.

**Core Idea**: Utilize a joint optimization of "code-switching SFT cold-start + resource-level curriculum GRPO + composite reward (accuracy + language + format)" for multilingual medical reasoning.

## Method

### Overall Architecture
The pipeline consists of three stages: (A) Data Construction — Clinical materials are pulled from MedlinePlus, GPT-4o generates multilingual MCQs, and a multi-stage filter removes trivial questions solvable by three small models. These are converted to open-ended (retaining reference reasoning chain $r$ and reference answer $y^*$), followed by manual verification by native speakers and medical experts (average score 4.89/5). (B) Cold-Start SFT — SFT is performed on a Qwen2.5-Instruct backbone using long CoT trajectories that allow code-switching. Intermediate reasoning steps can use any $\ell_t \in \mathcal{L}$, but the final answer must be in the target language $\ell$. (C) Curriculum GRPO — Languages are categorized into high/mid/low resource levels and trained in that order. When entering a new level, a proportion of $\alpha=0.85$ from the previous level is retained to prevent forgetting. The composite reward constrains accuracy, language, and format simultaneously.

### Key Designs

1.  **Code-switching Aware SFT Cold-Start**:
    - **Function**: Stabilizes multi-step reasoning capabilities, providing an initial policy for subsequent RL that does not collapse in intermediate steps.
    - **Mechanism**: For each query $x$ in target language $\ell$, a trajectory $\mathbf{r}=\{r_1,\dots,r_T\}$ is constructed where the language $\ell_t$ of $r_t$ can differ from $\ell$ (e.g., French consultation but intermediate English clinical terms). The final answer $y^*$ is forced to $\ell$, with loss defined as $\mathcal{L}_{\text{SFT}}=-\log p_\theta(\mathbf{r}, y^*\mid x)$.
    - **Design Motivation**: Directly forcing the use of low-resource languages throughout causes reasoning quality to collapse (many medical terms lack equivalents in Amharic). Allowing intermediate code-switching is a compromise: "let the model think in the language it knows best and answer in the patient's language," leaving room for RL optimization.

2.  **Composite Verifiable Reward**:
    - **Function**: Deconstructs "correctness," "language consistency," and "format compliance" into separate weighted scores to avoid reward hacking.
    - **Mechanism**: $R = \lambda_{\text{acc}} R_{\text{acc}} + \lambda_{\text{lang}} R_{\text{lang}} + \lambda_{\text{fmt}} R_{\text{fmt}}$. $R_{\text{acc}} \in [0,1]$ is provided by GPT-4.1 as a verifier (exact match for closed questions, partial credit for paraphrased open questions); $R_{\text{lang}}$ is a 0/1 indicator for whether the output strictly uses the target language; $R_{\text{fmt}}$ checks if `<thinking>/<step n>/<answer>` tags are compliant.
    - **Design Motivation**: A single answer reward leads to "correct answer, wrong language," while a single language reward causes the model to abandon reasoning to chase format scores. The three-way reward + a third-party judge model separate from the training verifier suppresses both "drifting language for accuracy" and "faking answers for language consistency."

3.  **Language Resource Level-based Curriculum GRPO**:
    - **Function**: Gradually migrates RL signals from "high-resource languages with abundant positive samples" to "low-resource languages with scarce positive samples."
    - **Mechanism**: Languages are split into high (FR / JA / ES / VI), mid (KO / TH / TR / BN), and low (AM / YO / HA / HI / SW). GRPO trains on "high" until a reward plateau, then expands to "mid," and finally to "low." Each phase samples $\mathcal{D}_i = \alpha \mathcal{D}_{i-1} + (1-\alpha)\mathcal{D}_{L_i}$ with $\alpha=0.85$ to maintain old capabilities. The GRPO update rule $A_{i,k} = R_{i,k} - \text{mean}(\{R_{i,k}\})$ remains unchanged.
    - **Design Motivation**: Mixing all 13 languages in RL directly results in near-zero positive samples for low-resource languages, making advantages near-constant and updates ineffective. The curriculum puts "languages with the stablest reward signals" first. Once the model forms basic multilingual reasoning + consistency, introducing low-resource languages effectively uses high-resource languages to "warm up" the reward surface.

### Loss & Training
The SFT stage maximizes $\log p_\theta(\mathbf{r}, y^*\mid x)$. The RL stage follows the standard GRPO clipped objective, with advantages normalized within groups and KL regularization against the cold-start model. A retention ratio of $\alpha=0.85$ is used across curriculum stages. Scaling is performed from 1.5B to 32B.

## Key Experimental Results

### Main Results
The benchmark reports average Language Consistency / Logical Accuracy (mean ± std) across 13 languages. Selection of 7B results:

| Model | Consistency ↑ | Accuracy ↑ |
|------|---------------|------------|
| Qwen2.5-Instruct-7B (Backbone) | 25.44 ± 0.36 | 29.56 ± 0.42 |
| Mistral-7B | 18.70 ± 1.30 | 15.23 ± 1.20 |
| BioMistral-7B | 7.10 ± 0.90 | 4.80 ± 0.95 |
| MedAlpaca-7B | 3.50 ± 0.90 | 2.47 ± 0.95 |
| HuatuoGPT-o1-8B (Prev. SOTA) | 67.30 ± 0.14 | 46.86 ± 0.09 |
| LLaMA-3.1-Instruct-8B | 36.56 ± 0.31 | 18.91 ± 0.18 |
| **Cure-Med-Qwen2.5-7B (Ours)** | **85.21** | **54.35** |

For the 3B segment (showing effectiveness for small models):

| Model | Consistency ↑ | Accuracy ↑ |
|------|---------------|------------|
| Qwen2.5-Instruct-3B | 8.39 ± 0.42 | 10.83 ± 0.60 |
| LLaMA-3.2-3B | 23.69 ± 0.36 | 10.41 ± 0.38 |
| **Cure-Med-Qwen2.5-3B** | **74.28 ± 0.60** | **42.93 ± 0.60** |

Scaling to 32B further reaches Consistency 94.96 / Accuracy 70.04, demonstrating method scalability.

### Ablation Study

| Configuration | Consistency / Accuracy Trend | Description |
|------|------------------------------|------|
| Full Cure-Med (SFT + Curriculum GRPO + 3-way Reward) | 85.21 / 54.35 (7B) | Full method |
| w/o code-switching SFT (Direct target language SFT) | Reasoning quality drops significantly, especially in low-resource | Code-switch is key for low-resource reasoning |
| w/o curriculum (13 language mixed GRPO) | Low-resource accuracy lags significantly | Curriculum order is vital for low-resource languages |
| w/o $R_{\text{lang}}$ | Accuracy maintained but consistency collapses | Language reward is indispensable |
| w/o $R_{\text{acc}}$ (Only language + format reward) | High consistency but hallucinated answers | Validates necessity of combined rewards |

### Key Findings
- The greatest gain from curriculum RL is not in "high" tiers, but in "low" tiers: while performance gaps are small in high-resource languages, Cure-Med at least doubles the strongest baseline in low-resource ones, indicating that "high-to-low" curriculum effectively "distills" reward signals into low-resource languages.
- Code-switching is an engineering necessity for medical multilingual reasoning: clinical terms are often missing in low-resource languages, and forced translation causes hallucinations. Allowing intermediate English terms while answering in the target language preserves accuracy without ruining patient experience.
- Advantages are maintained on OOD samples (unseen medical questions + unseen languages), suggesting the method learns a general pattern of "getting it right first, then using the right language" rather than question distribution.

## Highlights & Insights
- Using "language resource level" as curriculum difficulty is clever: while typical curriculum learning sorts by problem complexity, this paper sorts by "reward signal stability," essentially using early high-SNR rewards to stabilize the policy. This is applicable to any multilingual RL task with sparse and imbalanced rewards.
- The three-way decoupled reward + third-party judge model is a minimum viable solution to "simultaneously preventing language drift and accuracy drift"; any work aiming for multi-constraint RL can learn from this combination.
- CureMed-Bench is one of the few medical datasets featuring "open-ended + single verifiable answer + low-resource languages + clinical validation," likely becoming a de facto standard benchmark for multilingual medical RL.

## Limitations & Future Work
- Rewards still rely on GPT-4.1 as a verifier, introducing "isomorphic bias" and black-box costs; the verifier itself may be inaccurate in low-resource languages, causing reward noise.
- The "optimal mixing ratio" for code-switching SFT data is empirical and lacks systematic study; best switch patterns likely vary significantly across language pairs.
- Evaluation still emphasizes single-turn QA, missing multi-turn history taking and uncertainty communication in real clinical scenarios; multi-turn dialog capabilities need to be added.
- Cultural and terminological nuances (different drug names in different regions) rely on manual review, limiting scalability.

## Related Work & Insights
- **vs HuatuoGPT-o1 / OpenBioLLM / UltraMedical**: These follow monolingual + domain supervision routes; consistency is near zero in multilingual (especially low-resource) scenarios. The core difference here is treating multilingual fidelity as a first-class optimization objective.
- **vs GRPO / DeepSeekMath**: Shares the GRPO framework; this paper does not change the optimization rules but migrates the algorithm to reward-sparse multilingual medical tasks via "curriculum + composite rewards."
- **vs MMedBench / XMedBench**: These are MCQs that mask intermediate reasoning; CureMed-Bench forces open-ended generation, allowing independent measurement of reasoning and consistency, closer to clinical needs.

## Rating
- Novelty: ⭐⭐⭐⭐ Treating "language resource level = curriculum difficulty" is a simple but significantly effective perspective shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 languages × 3 scales + multiple baselines, though more detailed reward ablation is desired.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline, good coordination between formulas and diagrams.
- Value: ⭐⭐⭐⭐ Both the dataset and training framework directly advance fairness in low-resource medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)

</div>

<!-- RELATED:END -->
