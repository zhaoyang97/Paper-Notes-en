---
title: >-
  [Paper Note] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms
description: >-
  [ACL 2025][LLM (Other)][Behavior Control] This work proposes STA (Steering Target Atoms), which leverages Sparse Autoencoders (SAEs) to disentangle LLM representations into atomic knowledge components. By filtering and manipulating target atoms based on activation magnitude and frequency, STA achieves more robust and fine-grained behavior control compared to prompt engineering. It outperforms existing steering methods on both safety detoxification and reasoning control tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Behavior Control"
  - "steering vector"
  - "SAE"
  - "Sparse Autoencoder"
  - "Safety Alignment"
date: 2026-05-08
content_hash: d9755055a12b2656
---

# Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms

**Conference**: ACL 2025  
**arXiv**: [2505.20322](https://arxiv.org/abs/2505.20322)  
**Code**: [https://github.com/zjunlp/steer-target-atoms](https://github.com/zjunlp/steer-target-atoms)  
**Area**: LLM / Interpretability / AI Safety  
**Keywords**: Behavior Control, steering vector, SAE, Sparse Autoencoder, Safety Alignment

## TL;DR
This work proposes STA (Steering Target Atoms), which leverages Sparse Autoencoders (SAEs) to disentangle LLM representations into atomic knowledge components. By filtering and manipulating target atoms based on activation magnitude and frequency, STA achieves more robust and fine-grained behavior control compared to prompt engineering. It outperforms existing steering methods on both safety detoxification and reasoning control tasks.

## Background & Motivation
**Background**: LLM behavior control primarily relies on prompt engineering (system prompts) and steering vectors (directly modifying hidden states during forward propagation). While steering is more direct than prompting, traditional steering vectors operate within entangled representation spaces, which easily leads to side effects.

**Limitations of Prior Work**: (a) Prompt engineering is highly sensitive and fragile, and minor input variations can lead to unpredictable outputs; (b) traditional steering vectors (such as CAA) operate in dense representation spaces and fail to precisely control specific behavioral directions; (c) existing SAE steering has only been validated on toy tasks (such as entity recognition/tense transformation), leaving open-generation tasks unresolved.

**Key Challenge**: Knowledge is highly entangled (polysemanticity/superposition) in the high-dimensional representation space of LLMs. Directly steering hidden states can inadvertently damage non-target knowledge.

**Goal**: To precisely locate target atomic components in the high-dimensional sparse space disentangled by SAEs, thereby achieving fine-grained behavior control.

**Key Insight**: SAEs are used to project hidden states into a high-dimensional sparse space. The target atoms are located based on the activation differences (dual filtering of magnitude and frequency) between positive and negative samples, and then mapped back to the original space to obtain a refined steering vector.

**Core Idea**: Filtering target atoms in the SAE-disentangled space based on magnitude and frequency yields a more precise behavior control vector compared to direct steering.

## Method

### Overall Architecture
1. Run positive/negative samples through the model to obtain hidden states $\to$ encode them via SAE to obtain sparse activations.
2. Perform dual-filtering of target atoms based on magnitude difference $\Delta\mathbf{a}$ and frequency difference $\Delta\mathbf{f}$.
3. Map the target atoms back to the original representation space through the SAE decoder to obtain $\mathbf{v}_{STA}$.
4. Add $\hat{\mathbf{h}} = \mathbf{h} + \lambda \mathbf{v}_{STA}$ to the designated layer during inference.

### Key Designs

1. **Target Atom Identification (Dual Filtering)**:

    - Magnitude: $\Delta\mathbf{a}_j$ measures the average activation difference of the $j$-th atom between positive and negative samples.
    - Frequency: $\Delta\mathbf{f}_j$ measures the difference in activation frequency of the $j$-th atom between positive and negative samples.
    - An atom is selected as a target atom only when it simultaneously satisfies $\Delta\mathbf{a}_j \geq \alpha$ AND $\Delta\mathbf{f}_j \geq \beta$.
    - **Design Motivation**: Selecting based only on magnitude might capture noisy atoms with occasional high activations. Adding the frequency constraint ensures the selection of highly **consistent** atoms.

2. **SAE Decoder Mapping**:

    - $\mathbf{v}_{STA} = \mathbf{a}_{target} \mathbf{W}_{dec} + \mathbf{b}_{dec}$
    - By retaining only the contributions of the target atoms and setting the rest to zero, a refined steering vector is obtained.

3. **Steering vs Prompting Fair Comparison**:

    - Prompts are converted into equivalent steering interventions via STA to implement a fair comparison.
    - It is found that steering consistently outperforms prompting in robustness and flexibility.

### Application Extensions
- Beyond safety detoxification, this method is also successfully applied to control the CoT (Chain of Thought) length of large reasoning models (e.g., DeepSeek-R1) to manage "over-thinking".

## Key Experimental Results

### Main Results (Gemma-2-9b-it Safety Detoxification)

| Method | SafeEdit ASR↓ | RealToxic ASR↓ | Avg Safety ↑ | MMLU↑ | GSM8K↑ |
|------|-------------|---------------|---------|-------|--------|
| Vanilla | 29.63 | 2.59 | 83.89 | 72.06 | 75.66 |
| Prompt_hand | 21.26 | 1.58 | 88.58 | 71.07 | 74.83 |
| CAA | 8.52 | 1.25 | 95.12 | 70.77 | 75.21 |
| SAE_AXBENCH | 9.26 | 1.58 | 94.58 | 70.89 | 72.63 |
| **STA** | **4.22** | **0.67** | **97.56** | 70.27 | 71.65 |

### Ablation Study: Robustness Comparison (Under Adversarial Attacks)

| Method | Clean Input Safety Rate | Safety Rate Under Adversarial Attack | Drop |
|------|-------------|---------------|---------|
| Prompt | High | Significant Drop | ~20%+ |
| STA | High | Slight Drop | ~5% |

### Key Findings
- STA achieves the highest safety rate of 97.56% with minimal side effects—yielding only a 1-4 percentage point drop on MMLU/GSM8K.
- Steering is more robust against adversarial attacks—since directly modifying activations is unaffected by variations in the input text.
- Effective steering vectors can be extracted using only a small number of samples (even in a few-shot manner).
- Steering has been successfully applied to control the CoT length of reasoning models—opening a new direction.

## Highlights & Insights
- **Dual filtering (magnitude + frequency)** is simple yet crucial—it eliminates noisy atoms with occasional high activations.
- **Systematic validation of Steering > Prompting**: Rather than relying solely on empirical observations, a fair comparison framework is realized via STA.
- **Breakthrough of SAE from toy tasks to practical safety tasks**: This study validates its effectiveness in practical scenarios such as safety detoxification and reasoning control for the first time.

## Limitations & Future Work
- Reliance on pre-trained SAEs—the quality of the SAE directly impacts performance.
- The choice of hyperparameters $\alpha, \beta, \lambda$ lacks theoretical guidance.
- There is a 1-4 point drop in MMLU/GSM8K—the side effects are not completely eliminated.
- Lack of scalability testing on larger models (70B+).
- "Atoms" may not be the minimal steering units; finer-grained decomposition represents a future direction.

## Related Work & Insights
- **vs. CAA (Rimsky et al.)**: While CAA operates in dense spaces, STA operates in the SAE-disentangled space, achieving higher precision—e.g., CAA has an 8.52% ASR on SafeEdit vs. STA's 4.22%.
- **vs. Direct SAE steering**: Prior work only validated this on toy tasks, whereas STA is applied to open-generation safety tasks for the first time.
- **vs. Prompt Engineering**: Prompts are sensitive and fragile, whereas STA is robust and interpretable.
- This work offers valuable insights into the intersection of interpretability and safety alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ The steering scheme with SAE disentanglement and dual filtering is elegant; controlling reasoning models introduces an innovative application scenario.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive experiments across multiple models and tasks, featuring a comprehensive and fair comparison between steering and prompting.
- Writing Quality: ⭐⭐⭐⭐ The motivation for the approach is clear, and the mathematical formulations are concise.
- Value: ⭐⭐⭐⭐⭐ This work possesses direct practical value for the safety control of LLMs, representing a significant advancement in steering research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Steering off Course: Reliability Challenges in Steering Language Models](steering_off_course_reliability_challenges_in_steering_language_models.md)
- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization](unseentimeqa_time-sensitive_question-answering_beyond_llms_memorization.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ICLR 2026\] Beyond Magic Words: Sharpness-Aware Prompt Evolving for Robust Large Language Models with TARE](../../ICLR2026/llm_nlp/beyond_magic_words_sharpness-aware_prompt_evolving_for_robust_large_language_mod.md)

</div>

<!-- RELATED:END -->
