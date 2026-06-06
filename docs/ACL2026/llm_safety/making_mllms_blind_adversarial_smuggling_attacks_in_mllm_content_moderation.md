---
title: >-
  [Paper Note] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation
description: >-
  [ACL 2026][LLM Safety][Adversarial Attack] This paper reveals the threat of "Adversarial Smuggling Attacks" (ASA) in multimodal content moderation—encoding harmful content into human-readable but AI-unreadable visual for…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Adversarial Attack"
  - "Content Moderation"
  - "Multimodal Large Language Models"
  - "Perceptual Blindness"
  - "Reasoning Blockade"
date: 2026-05-08
content_hash: c2d8e8d3deeb078c
---

# Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation

**Conference**: ACL 2026  
**arXiv**: [2604.06950](https://arxiv.org/abs/2604.06950)  
**Code**: [https://github.com/lizhiheng2025/SmuggleBench](https://github.com/lizhiheng2025/SmuggleBench)  
**Area**: Multimodal VLM  
**Keywords**: Adversarial Attack, Content Moderation, Multimodal Large Language Models, Perceptual Blindness, Reasoning Blockade

## TL;DR

This paper reveals the threat of "Adversarial Smuggling Attacks" (ASA) in multimodal content moderation—encoding harmful content into human-readable but AI-unreadable visual formats to bypass automatic detection. The authors construct SmuggleBench, a benchmark containing 1,700 samples and 9 attack techniques, finding that SOTA models, including GPT-5, suffer from attack success rates exceeding 90%.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) are being widely deployed as automated content moderators to filter harmful content such as hate speech, violence, and pornography. Models like GPT-5, Gemini 2.5 Pro, and Qwen3-VL have already performed excellently on standard content moderation tasks.

**Limitations of Prior Work**: Existing adversarial attack research primarily focuses on two paradigms: adversarial perturbations (adding imperceptible noise leading to misclassification, "making MLLMs stupid") and adversarial jailbreaking (using malicious instructions to bypass safety guardrails, "making MLLMs bad"). However, both overlook a more covert threat: exploiting the perception gap between humans and AI to disguise harmful content as benign visual formats.

**Key Challenge**: Adversarial smuggling attacks exploit the Human-AI capability gap. Harmful content is presented in visual formats that humans can easily read but AI cannot perceive (e.g., integrating "KILL ALL" into the texture of a forest landscape). This implies that models possess systemic vulnerabilities at both the visual perception and semantic reasoning levels.

**Goal**: (1) Formally define adversarial smuggling attacks and their two attack paths; (2) build the first dedicated evaluation benchmark, SmuggleBench; (3) evaluate the vulnerability of SOTA models and explore mitigation strategies.

**Key Insight**: The MLLM moderation process is decomposed into two stages: perception (text extraction) and reasoning (semantic judgment). Attacks can take effect in either stage: Perceptual Blindness prevents text recognition, while Reasoning Blockade prevents semantic understanding.

**Core Idea**: Adversarial smuggling is a third category of MLLM adversarial threat, independent of perturbations and jailbreaks; it "makes MLLMs blind" rather than stupid or bad. Current SOTA models have almost no resistance to it.

## Method

### Overall Architecture

The work consists of three parts: (1) Formal definition and taxonomy of ASA; (2) construction of the SmuggleBench benchmark (data-driven category discovery + dual-source data collection); (3) systematic evaluation of SOTA models on SmuggleBench and exploration of mitigation strategies. The input is an image containing hidden harmful content, and the output is the model's safety/unsafety judgment.

### Key Designs

1.  **Adversarial Smuggling Attack Taxonomy (ASA Taxonomy)**:
    *   **Function**: Systematizes diverse smuggling strategies into a structured classification.
    *   **Mechanism**: Based on a data-driven discovery process—collecting millions of potential smuggling images from the open web, using Jina-CLIP-v2 to extract visual embeddings and Qwen-VL-Max to extract descriptive keywords. Two-stage unsupervised clustering via BERTopic (visual clustering followed by keyword c-TF-IDF labeling) is performed, with final review by domain experts. This results in 9 attack techniques under Perceptual Blindness (6 types: Tiny Text, Occluded Text, Low Contrast, Handwriting, Artistic Deformation, AI Illusions) and Reasoning Blockade (3 types: Dense Text Masking, Semantic Camouflage, Visual Puzzles).
    *   **Design Motivation**: Traditional manual classification easily misses the diversity of real-world attacks. Data-driven discovery ensures the benchmark covers the actual threat space.

2.  **SmuggleBench Benchmark Construction**:
    *   **Function**: Provides the first standardized test set specifically for evaluating MLLM resistance to smuggling attacks.
    *   **Mechanism**: Employs a dual-source data collection strategy. Automated Synthesis (Syn) is used for Low Contrast and AI Illusions—requiring precise control over visual threshold parameters to balance camouflage effects with human readability. Wild Collection (Wild) is used for the other 7 categories—capturing real-world artifacts like natural occlusion, irregular handwriting, and compression noise. A total of 1,700 samples were verified for human readability via independent labeling by three people (2/3 consensus), ensuring model failure stems from the attack rather than objective unreadability.
    *   **Design Motivation**: Synthetic data alone cannot cover the diversity of real attacks, while wild data alone makes it difficult to control attack parameters. Combining both balances coverage and controllability.

3.  **Two-stage Evaluation Protocol (Perception-Reasoning Diagnostic)**:
    *   **Function**: Accurately diagnoses whether model failure stems from perception or reasoning.
    *   **Mechanism**: A two-step prompting strategy is designed—Step 1 requires the model to explicitly transcribe text in the image (perception), and Step 2 requires evaluating safety (reasoning). Two metrics are used: Attack Success Rate (ASR) for overall vulnerability and Text Extraction Rate (TER) for locating the attack path. Low TER = Perceptual Blindness; high TER + high ASR = Reasoning Blockade.
    *   **Design Motivation**: A single metric cannot distinguish which stage an attack affects. The two-step prompt and dual-metric design enable precise fault diagnosis.

### Loss & Training

This paper is primarily an evaluation work and does not involve training new models. In exploring mitigation strategies: (1) CoT defense uses structured prompts to guide the model through step-by-step visual inspection and semantic decoding; (2) SFT defense involves full-parameter fine-tuning of Qwen2.5-VL-7B-Instruct using a balanced dataset of 1,700 adversarial samples and 1,700 benign samples, with a 50/50 stratified split for training and testing.

## Key Experimental Results

### Main Results

| Model | Perceptual Blindness ASR↓ | Perceptual Blindness TER↑ | Reasoning Blockade ASR↓ | Reasoning Blockade TER↑ | Total ASR↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5 | 98.5% | 9.9% | 98.7% | 45.1% | 98.6% |
| Gemini 2.5 Pro | 84.9% | 22.6% | 83.7% | 64.2% | 84.5% |
| Qwen3-VL-8B | 93.1% | 15.5% | 89.7% | 58.2% | 91.9% |
| Qwen3-VL-32B | 89.7% | 14.7% | 91.3% | 57.9% | 90.2% |
| Qwen3-VL-235B | 90.4% | 16.8% | 90.3% | 59.9% | 90.4% |

### Ablation Study (Mitigation Strategies)

| Defense Strategy | Model | ASR Change | TER Change | FPR Change | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CoT (Inference) | Qwen3-VL-235B | -7.2% | +1.7% | +2.7% | Limited improvement, FPR nearly tripled |
| SFT (Training) | Qwen2.5-VL-7B | -81.5% | +10.1% | +6.6% | Massive ASR reduction but significant FPR increase |

### Key Findings

*   **Model scale cannot defend against ASA**: From Qwen3-VL 8B to 235B, the ASR only dropped from 91.9% to 90.4%; the scaling effect is almost zero. GPT-5’s ASR is as high as 98.6%, making it the most vulnerable among all models.
*   **AI Illusions are the most lethal attacks**: The TER for all models is near 0% (GPT-5 is only 0.3%), with ASR near 100%. After using ControlNet to blend text into visual scenes, models completely fail to perceive the hidden text.
*   **Reasoning Blockade is harder to defend than Perceptual Blindness**: Even when models successfully extract text (TER 50-60%), they still judge the content as safe in 83-98% of cases, indicating a lack of capability to associate extracted text with harmful intent.
*   **CoT cannot compensate for perceptual defects**: For AI Illusions attacks, the reduction in ASR via CoT is 0. Explicit reasoning steps cannot compensate for the fundamental failure of the visual encoder.
*   **SFT is effective but introduces false positives**: SFT reduced ASR from 95% to 13.5%, but the False Positive Rate (FPR) rose from 1.6% to 8.2%. Over-sensitivity leads to many normal samples being misjudged.

## Highlights & Insights

*   **Defined the third category of MLLM adversarial threats**: Clearly distinguishes between adversarial perturbations ("becoming stupid"), adversarial jailbreaking ("becoming bad"), and adversarial smuggling ("becoming blind"), expanding the threat model space for MLLM safety research. This taxonomic framework is a significant conceptual contribution to the safety community.
*   **Perception-Reasoning Diagnostic Framework**: Pinpoints where an attack takes effect using ASR + TER dual metrics, providing a diagnostic basis for targeted defense. This framework can be generalized to the robustness evaluation of any multi-stage AI system.
*   **Data-driven Attack Taxonomy Discovery**: Rather than relying on empirical manual design of attack types, categories are discovered via unsupervised clustering of millions of real-world data points, ensuring the realism and coverage of the benchmark.
*   **GPT-5 is more vulnerable than smaller models (98.6% vs 84.5%)**: A surprising finding, possibly because larger models tend to trust the surface semantics of visual input rather than deeply analyzing potential deceptive intent.

## Limitations & Future Work

*   SmuggleBench currently only covers English harmful content; multilingual smuggling attacks (e.g., using visual similarities of non-Latin characters) are not addressed.
*   SFT defense presents a sharp trade-off between FPR and ASR, necessitating more refined defense strategies (e.g., adaptive thresholds, multi-stage moderation).
*   The joint defense effect of adversarial training and CoT has not been explored.
*   The actual vulnerability of commercial content moderation platforms (e.g., OpenAI Moderation API) has not been evaluated.
*   Fundamental capability bottlenecks of visual encoders (CLIP, SigLIP) are difficult to solve via post-processing and may require innovation at the encoder architecture level.

## Related Work & Insights

*   **vs Adversarial Perturbation**: Adversarial perturbations add imperceptible noise to cause misclassification; ASA does the opposite—adding human-perceptible but AI-imperceptible harmful content. Both exploit different directions of the Human-AI perception gap.
*   **vs Adversarial Jailbreak**: Jailbreak attacks induce harmful output through explicit malicious instructions, whereas ASA does not require the model to generate harmful output; it only requires the model to "not see" the embedded harmful content.
*   **vs Traditional OCR Robustness Research**: Traditional OCR research focuses on the accuracy of text recognition in natural scenes; ASA weaponizes OCR weaknesses as an attack method, revealing the severe consequences of insufficient OCR capabilities in safety scenarios.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First to define and systematically study adversarial smuggling attacks, opening a new direction for MLLM safety.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 6 models (including GPT-5), 9 attack techniques, 1,700 samples, and two defense strategies.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous taxonomic framework, and powerful intuitive visual cases.
*   Value: ⭐⭐⭐⭐⭐ Reveals systemic vulnerabilities in MLLM content moderation, with direct warning significance for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)
- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](../../ICLR2026/llm_safety/expguard_llm_content_moderation_in_specialized_domains.md)
- [\[ACL 2026\] CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](crossguard_safeguarding_mllms_against_joint-modal_implicit_malicious_attacks.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)

</div>

<!-- RELATED:END -->
