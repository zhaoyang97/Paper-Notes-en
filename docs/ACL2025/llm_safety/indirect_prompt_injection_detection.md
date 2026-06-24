---
title: >-
  [Paper Note] Can Indirect Prompt Injection Attacks Be Detected and Removed?
description: >-
  [ACL 2025][LLM Safety][indirect prompt injection] This paper systematically studies the detection and removal of indirect prompt injection attacks: it constructs an evaluation benchmark, discovers that existing detection models perform poorly against indirect attacks while specially trained models can achieve 99% accuracy, proposes two removal methods (segmentation-based and extraction-based), and combines detection and removal into a filtering pipeline to effectively reduce…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "indirect prompt injection"
  - "detection"
  - "removal"
  - "filtering"
  - "segmentation"
  - "extraction"
  - "over-defense"
date: 2026-05-08
content_hash: 5f881c78bcb4cb33
---

# Can Indirect Prompt Injection Attacks Be Detected and Removed?

**Conference**: ACL 2025  
**arXiv**: [2502.16580](https://arxiv.org/abs/2502.16580)  
**Code**: [GitHub](https://github.com/LukeChen-go/indirect-pia-detection)  
**Institution**: NUS & HKUST  
**Area**: LLM Security  
**Keywords**: indirect prompt injection, detection, removal, filtering, segmentation, extraction, over-defense

## TL;DR

This paper systematically studies the detection and removal of indirect prompt injection attacks: it constructs an evaluation benchmark, discovers that existing detection models perform poorly against indirect attacks while specially trained models can achieve 99% accuracy, proposes two removal methods (segmentation-based and extraction-based), and combines detection and removal into a filtering pipeline to effectively reduce the attack success rate of indirect prompt injection.

## Background & Motivation

### Background

- Prompt injection attacks are categorized into direct attacks (where the user is the attacker) and indirect attacks (where malicious instructions are injected into external data sources).
- Indirect attacks present a more realistic threat: attackers embed malicious instructions into webpages or documents, which misguide LLMs after being retrieved via search engines.
- Indirect attacks can achieve various malicious goals: phishing, ad promotion, public opinion manipulation, etc.
- Existing defense methods mainly focus on direct attacks, leaving the detection and removal of indirect attacks severely under-investigated.

### Limitations of Prior Work

- Existing detection models (such as ProtectAI, Prompt Guard, and Llama Guard) are primarily trained on direct attacks, yielding low detection rates for indirect attacks.
- There is almost no research on what to do after detection—how to remove malicious content while preserving useful information?
- Over-defense issue: detection models misclassify clean documents as injected documents, affecting normal utility.
- The injection location (prefix/middle/suffix) significantly impacts detection performance, but prior methods fail to consider this.

### Key Insight

- Detection of indirect prompt injection requires specialized trained models, as general safety models are insufficient.
- Detection and removal should be combined into a unified filtering pipeline.
- Over-defense primarily occurs on out-of-domain (OOD) documents and is almost non-existent in-domain (ID).
- Different injection locations require distinct removal strategies: segmentation is suitable for prefix/middle, while extraction is suitable for suffix.

## Method

### Overall Architecture

A two-stage filtering pipeline: the first stage uses a detection model to determine whether a document has been injected; the second stage removes the injected content from the detected documents. Finally, the processed clean document is passed to the LLM to execute the user's original task.

### Key Designs

#### Key Design 1: Detection Model Training

- **Classification Model**: Uses DeBERTa, mapping the first hidden state $h_1$ to binary classification logits.
- **Generative Model**: Uses Qwen2-0.5B/1.5B, mapping the last hidden state to the vocabulary logits at "yes"/"no" positions.
- Training Data Construction: 40% clean documents, 15% prefix (head) injection, 30% middle injection, and 15% suffix (tail) injection.
- Training data is constructed using only the simplest Naive Attack, but it generalizes well to other attack types.

#### Key Design 2: Segmentation Removal Method (Segmentation Removal)

- Segment the injected document into multiple parts sentence by sentence.
- Classify each segment individually using the detection model.
- Discard segments predicted to contain injected instructions.
- Reassemble the remaining clean segments into the processed document.
- Advantage: Fully leverages the capability of the trained document-level detection model.

#### Key Design 3: Extraction Removal Method (Extraction Removal)

- Train an extraction model to directly identify and extract injected instructions from the document.
- The training target consists of three loss terms: standard language modeling loss + start token prediction + end token prediction.
- After extraction, calculate the longest common substring (LCS) and remove it from the original document.
- Advantage: Especially effective for suffix injection (the model is proficient at identifying anomalous content at the end of a document).

### Benchmark Dataset Construction

- Construct evaluation benchmarks (Inj-SQuAD, Inj-TriviaQA) based on SQuAD and TriviaQA QA datasets.
- Each sample contains: original prompt $p$, clean document $d$, answer $a$, injected instruction $x$, and probe $y$.
- Injected instructions cover three types of real-world attack purposes: phishing, advertisement, and propaganda (generated using GPT-4o).
- Injection locations cover prefix, middle, and suffix.

## Key Experimental Results

### Detection Performance Comparison

| Model | Inj-SQuAD Avg Accuracy | Inj-TriviaQA Accuracy |
|------|-------|-------|
| Llama3-8B-Instruct | 78.74% | Severe Over-defense |
| Llama-Guard3-8B | Up to 39.11% | — |
| ProtectAI-detector | Only effective against Ignore attacks | — |
| **DeBERTa (ours)** | — | **99.12%** |
| **Qwen2-1.5B (ours)** | — | **97.20%** |

### Removal Performance (Table 2, Inj-TriviaQA)

| Method | Naive Removal Rate | Ignore | Fakecom | Combined |
|------|------------|--------|---------|----------|
| Segment-DeBERTa (Head) | 84.66% | 100% | 79.88% | 100% |
| Segment-Qwen2-1.5B (Head) | 96.33% | 98.22% | 96.66% | 98.11% |
| Extract-Qwen2-1.5B (Tail) | **100%** | **98.44%** | **97.66%** | **94.66%** |

### Over-defense Rate (Table 1)

| Model | In-domain (SQuAD→SQuAD) | Out-of-domain (SQuAD→TriviaQA) |
|------|------------------|---------------------|
| DeBERTa-SQuAD | **0.0%** | 12.44% |
| Qwen2-1.5B-SQuAD | **0.0%** | 11.11% |
| Qwen2-0.5B-SQuAD | **0.0%** | 27.33% |

### Filtering Pipeline vs Baseline Defenses (ASR↓)

| Method | Naive | Ignore | Escape | Fakecom |
|------|-------|--------|--------|---------|
| Sandwich | High | High | High | High |
| StruQ (fine-tune) | Low | Low | Medium | High |
| **Filtering (ours)** | **Low** | **Low** | **Low** | **Medium** |

### Key Findings

1. **Existing models are ineffective against indirect attacks**: Llama Guard achieves a maximum of only 39%, and ProtectAI is only effective against specific attacks.
2. **Specialized training yields notable performance**: DeBERTa trained only on Naive Attacks achieves a 99% detection rate and generalizes well to other attacks.
3. **Over-defense primarily occurs out-of-domain**: The in-domain over-defense rate is 0%, while out-of-domain goes up to 27%; stronger models and more fluent documents are less prone to over-defense.
4. **Complementarity between segmentation and extraction**: Segmentation performs better overall, but extraction achieves a 100% removal rate for suffix injections (the most effective attack location).
5. **Removal does not compromise information utility**: Document corruption caused by over-defense barely affects the accuracy of responses to the original QA tasks.
6. **Injected positions in training data are crucial**: Models trained on only a single position struggle to generalize to other positions.

## Highlights & Insights

- **The first work to systematically study the detection + removal of indirect prompt injections**, filling an important research gap.
- **The filtering pipeline approach** is practical and modular: detection and removal can be upgraded independently.
- **Efficient defense with small models**: DeBERTa (110M) and Qwen2 (1.5B) perform significantly better than 8B-scale Guard models.
- Generalizes well to attacks such as Ignore and Escape, despite being trained only on Naive Attacks.

## Limitations & Future Work

- Limited generalization to Fake Completion Attacks (due to its injection pattern being highly distinct from Naive Attacks).
- Over-defense remains notable in out-of-domain scenarios (12-27%), restricting cross-domain deployment.
- Segmentation removal relies on the sentence-level classification capability of the detection model, whereas that model is trained on document-level tasks.
- Does not consider active evasion of detection by attackers (e.g., dispersing the injection instruction across multiple sentences).
- The benchmark dataset only covers QA scenarios and has not been expanded to multi-turn dialogues, code generation, etc.

## Related Work & Insights

- The same team's prior work (Chen et al., 2024a) focused on direct prompt injection defense, while this paper targets indirect scenarios.
- ProtectAI and Meta Prompt Guard serve as the primary open-source detection baselines.
- Insights: Future work can deploy a detection-removal-defense pipeline as an end-to-end security middleware at the frontend of LLM applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study on the detection and removal of indirect attacks.
- **Technical Depth**: ⭐⭐⭐⭐ — Sound methodological design, comprehensive experimental dimensions.
- **Practicality**: ⭐⭐⭐⭐⭐ — Can be directly deployed as a safety filtering layer for LLM applications.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dimensional evaluation across detection, removal, filtering, and over-defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Defense Against Prompt Injection Attack by Leveraging Attack Techniques](defense_prompt_injection.md)
- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](../../ACL2026/llm_safety/piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](../../ACL2026/llm_safety/robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2025\] Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](llm_watermark_distillation_robustness.md)

</div>

<!-- RELATED:END -->
