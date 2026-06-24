---
title: >-
  [Paper Note] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting
description: >-
  [ECCV2024][Multimodal VLM][MLLM Security] The AdaShield framework is proposed, which comprises a meticulously designed static defense prompt (AdaShield-S) and an LLM-based adaptive iterative optimization framework (AdaShield-A). Without fine-tuning MLLMs or training additional modules, it effectively defends against structure-based jailbreak attacks, reducing the attack success rate from over 75% to below 15% while maintaining normal task performance.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "MLLM Security"
  - "Jailbreak Attack Defense"
  - "Adaptive Prompting"
  - "Structure-based Attack"
  - "Black-box Defense"
date: 2026-05-08
content_hash: ffad4dc25a07ae8c
---

# AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting

**Conference**: ECCV2024  
**arXiv**: [2403.09513](https://arxiv.org/abs/2403.09513)  
**Code**: [https://github.com/rain305f/AdaShield](https://github.com/rain305f/AdaShield)  
**Area**: AI Security / Multimodal Large Language Model Defense  
**Keywords**: [MLLM Security, Jailbreak Attack Defense, Adaptive Prompting, Structure-based Attack, Black-box Defense]

## TL;DR

The AdaShield framework is proposed, which comprises a meticulously designed static defense prompt (AdaShield-S) and an LLM-based adaptive iterative optimization framework (AdaShield-A). Without fine-tuning MLLMs or training additional modules, it effectively defends against structure-based jailbreak attacks, reducing the attack success rate from over 75% to below 15% while maintaining normal task performance.

## Background & Motivation

**Background** Multimodal Large Language Models (MLLMs) have made significant progress in vision-language reasoning, but safety issues are increasingly prominent. Jailbreak attacks fall into two categories: perturbation-based attacks (adding adversarial perturbations to images) and structure-based attacks (embedding malicious contents into images via typography or text layout). Well-established countermeasures exist for the former (e.g., image purification, adversarial training), but traditional adversarial defenses are almost ineffective against the latter because it embeds structured information with semantic meanings.

**Limitations of Prior Work** Alignment-during-training methods (e.g., DRESS) require a large amount of high-quality data and computational resources. Post-processing filtering methods (e.g., MLLMP) require training additional harmful content detectors and introduce significant inference time overhead (16.03s vs 9.40s for normal queries). The simple defense prompt proposed by FigStep has limited effectiveness and lacks scenario adaptability. Furthermore, the harmful content detector of MLLMP exhibits poor generalization, achieving only a 4.34% accuracy on pornographic scenarios in the QR dataset.

**Key Challenge** Effective defense requires tailoring safety rules for each attack scenario, yet manual design cannot cover all scenarios. Concurrently, defense must not be excessive to avoid rejecting benign queries (the over-defense problem).

**Goal** To automatically and adaptively generate defense prompts for MLLMs, achieving both a high defense rate and low over-defense without fine-tuning.

**Key Insight** Leverages the intrinsic capability of LLMs as a defense prompt generator, automatically producing scenario-specific defense prompts through conversational, iterative optimization with the target MLLM.

**Core Idea** Empowers an LLM "defender" to iteratively learn to generate defense prompts tailored to various attack scenarios by observing jailbreak failure cases of the target model, thereby establishing a retrievable defense prompt pool.

## Method

### Overall Architecture

AdaShield is categorized into two versions: AdaShield-S manually designs a general static defense prompt $P_s$ based on four design intuitions and prepends it to the model input; AdaShield-A introduces a collaborative, iterative framework between a "defender" LLM $D$ and the target MLLM $M$ to automatically optimize defense prompts and generate a diversified defense prompt pool $\mathcal{P}$. During inference, the most matching defense prompt is retrieved based on the similarity of the CLIP embedding of the input query.

### Key Designs

1. **Static Defense Prompt Design based on 4 Intuitions (AdaShield-S)**:

    - **Function**: Manually designs a general defense prompt $P_s$ covering a complete inspection process.
    - **Mechanism**: Integrates four design intuitions: (1) thoroughly inspect image content for malicious text/items; (2) utilize Chain-of-Thought (CoT) to step-by-step check whether the instruction is harmful; (3) explicitly specify the response pattern to malicious queries (e.g., "I am sorry"); (4) include instructions for handling benign queries to prevent over-defense. Ablation studies verify that the absence of any intuition leads to an increased ASR, with Intuition 3 being the most critical (missing explicit response instruction in $P_c$ causes the CogVLM ASR to soar from 16% to 75%).
    - **Design Motivation**: The core of structure-based attacks is bypassing safety alignment via images; thus, defense must guide the model to actively examine the text/semantic content in the images.

2. **Adaptive Defense Prompt Automated Optimization Framework (AdaShield-A)**:

    - **Function**: Automatically generates a customized defense prompt pool for different attack scenarios, adaptively retrieving the best prompt during inference.
    - **Mechanism**: 5-step process during the training phase: (1) collect a few malicious samples and feed them into the target model $M$ to get jailbreak responses; (2) the defender $D$ (Vicuna-13B) automatically generates improved prompts based on the failed prompts and jailbreak responses; (3) conduct keyword matching to judge if the new response is still a successful jailbreak; (4) if failed, perform iterative optimization; (5) filter prompts with good generalization on the validation set (ASR threshold $\alpha=0.8$) and rewrite them using GPT-4 to increase diversity. During inference, retrieval is performed by calculating the cosine similarity after concatenating CLIP text + image embeddings: $Q_{\text{best}}, P_{\text{best}} = \{Q_i, P_i | \arg\max_i \cos(z_t, z_i) \text{ and } \max \cos(z_t, z_i) > \beta\}$. If the maximum similarity is below the threshold $\beta=0.7$, it is classified as a benign query and no defense prompt is applied.
    - **Design Motivation**: A single general prompt cannot cover complex attack scenarios such as law, finance, and medicine. The automated framework readily scales to arbitrary scenarios and is applicable to black-box models (MLLM-as-a-Service, MLMaaS).

### Loss & Training

AdaShield does not involve parameter training of models. The optimization of defense prompts is completed through conversational iterations: the defender $D$ receives system prompts + failure cases to generate new prompts, and the target model $M$ evaluates the defense effectiveness. Key hyperparameters: expectation validation set ASR threshold $\alpha=0.8$, retrieval similarity threshold $\beta=0.7$. Evaluation metrics include keyword-based ASR and GPT Recheck ASR.

## Key Experimental Results

### Main Results

| Model | Method | QR ASR↓ | QR Recheck↓ | FigStep ASR↓ | FigStep Recheck↓ | MM-Vet Total↑ |
|------|------|---------|-------------|-------------|-----------------|-------------|
| LLaVA-1.5 | Vanilla | 75.75 | 67.71 | 70.47 | 87.21 | 36.8 |
| LLaVA-1.5 | FSD | 69.50 | 59.38 | 64.88 | 80.93 | 33.1 |
| LLaVA-1.5 | MLLMP | 77.96 | 64.69 | 73.72 | 76.51 | 36.3 |
| LLaVA-1.5 | AdaShield-S | 24.43 | 20.61 | 26.05 | 35.58 | 35.2 |
| LLaVA-1.5 | AdaShield-A | **15.22** | **15.43** | **10.47** | **22.33** | 36.3 |
| CogVLM | AdaShield-A | **1.37** | **1.43** | **0.00** | **0.00** | **51.0** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Random retrieval vs AdaShield-A retrieval | 18.2% vs 15.2% (LLaVA QR ASR) | Adaptive retrieval outperforms random selection |
| $P_s$ vs $P_v$ (removing benign instructions) | 35.2 vs 29.8 (LLaVA MM-Vet) | Verifies Intuition 4 in preventing over-defense |
| $P_a$ (no image inspection) | 39.6-53.6% ASR | Verifies the criticality of Intuition 1 (image inspection) |
| $P_c$ (no explicit response instruction) | 62-81% ASR | Verifies that Intuition 3 (explicit response instruction) is the most critical |

### Key Findings

- AdaShield-A achieves a zero FigStep ASR on CogVLM, while its MM-Vet score (51.0) exceeds that of the Vanilla model (50.0).
- Inference time is significantly optimized: AdaShield-A requires only 1.46s on harmful queries, compared to 16.03s for MLLMP.
- AdaShield-A demonstrates solid generalization to unseen scenarios: Easy training $\rightarrow$ Hard testing yields an ASR of only 27.38%.
- The over-defense problem is effectively mitigated by the similarity threshold $\beta$, as low-similarity queries do not append defense prompts.

## Highlights & Insights

- Concise and elegant idea: leverages the intrinsic capability of LLMs to protect LLMs, where the defense prompt acts as a "safety shield" prepended to the input.
- Four design intuitions are rigorously validated via ablation studies, with clear and quantifiable contributions from each intuition.
- No fine-tuning or training of additional modules is required, and the method is applicable to black-box API services, resulting in extremely low deployment barriers.
- The adaptive retrieval mechanism balances defense effectiveness (matching prompts from the most relevant scenarios) and normal usage (avoiding defense triggering for low-similarity cases).

## Limitations & Future Work

- Only two structure-based attacks (FigStep and QR) are evaluated, leaving more complex hybrid attacks unverified.
- The defense prompt pool is built on limited training samples, which restricts its coverage.
- Relies on keyword matching to determine jailbreak success, which may be bypassed by more subtle harmful responses.
- CLIP embedding retrieval might produce mismatches for queries with similar semantics but different intents.
- On MiniGPT-v2, AdaShield-S causes normal task performance to drop sharply (36.8 $\rightarrow$ 1.4), indicating that the over-defense problem is more severe on weaker models.

## Related Work & Insights

This work pioneers a new "prompt-based defense" path in the MLLM security field, forming a tripartite dynamic alongside the fine-tuning path (e.g., DRESS) and the post-hoc path (e.g., MLLMP). Insights for future work: (1) automated optimization of defense prompts can be extended to adversarial attacks; (2) the collaborative guardian-target dual-model paradigm can be utilized in broader safety alignment scenarios; (3) retrieval-augmented defense strategies are promising for integration with RAG technologies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of adaptive defense prompts is novel, and the guardian-target collaborative framework is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models and attacks, with design choices validated through ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-motivated, with a persuasive exposition of the intuition-driven methodology.
- **Value**: ⭐⭐⭐⭐⭐ Zero-cost deployment, black-box compatibility, and substantial efficacy make it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ACL 2025\] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](../../ACL2025/multimodal_vlm/adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)
- [\[ECCV 2024\] FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)
- [\[ECCV 2024\] Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)

</div>

<!-- RELATED:END -->
