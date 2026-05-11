---
title: >-
  [Paper Note] Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?
description: >-
  [ICLR 2026][Multimodal VLM][Vision-Language Models] This paper introduces the VLM-GEOPRIVACY benchmark grounded in Nissenbaum's Contextual Integrity (CI) theory. Through seven progressively structured context-aware quest…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Contextual Integrity"
  - "Geographic Privacy"
  - "Location Disclosure"
  - "VLM Safety"
date: 2026-05-08
content_hash: 84854ba4c51e4ace
---

# Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?

**Conference**: ICLR 2026
**arXiv**: [2602.05023](https://arxiv.org/abs/2602.05023)
**Code**: [https://github.com/99starman/VLM-GeoPrivacyBench](https://github.com/99starman/VLM-GeoPrivacyBench)
**Area**: Multimodal VLM
**Keywords**: Vision-Language Models, Contextual Integrity, Geographic Privacy, Location Disclosure, VLM Safety

## TL;DR
This paper introduces the VLM-GEOPRIVACY benchmark grounded in Nissenbaum's Contextual Integrity (CI) theory. Through seven progressively structured context-aware questions and a three-tier location disclosure granularity (refusal / city-level / precise location), it systematically evaluates whether 14 mainstream VLMs can determine appropriate location disclosure levels based on social-norm cues present in images. Results show that all models exhibit severe over-disclosure bias (Over-Disclosure rates of 46–52%), and malicious prompting can push the Abstention Violation rate to 100%.

## Background & Motivation

**Background**: Vision-language models (VLMs) and multimodal large reasoning models (MLRMs), exemplified by o3, GPT-4V, and Gemini, have demonstrated remarkable capability in image-based geolocalization, achieving street-level precision. Applications such as GeoGuessr further illustrate the feasibility of inferring precise locations from casually captured photographs.

**Limitations of Prior Work**: Such precise localization capability poses serious privacy threats—images shared casually on social media may be exploited by these widely accessible models to infer sensitive location information beyond what the sharer consented to or intended to disclose. Recent work has proposed blanket restrictions on VLM geolocalization capabilities; however, this coarse strategy fails to distinguish legitimate use cases (navigation assistance, tourism recommendations) from malicious ones (stalking, privacy violations), thereby sacrificing practical utility.

**Key Challenge**: VLMs must strike a balance between privacy protection and functional utility. The core issue is not whether a model *can* geolocalize, but whether it *should* disclose location information at a given precision level in a specific context. Existing VLMs entirely lack this context-aware privacy reasoning capability—they tend to answer as precisely as possible without regard to social norm constraints.

**Goal**: (1) How can location privacy norms that VLMs should follow be formally defined? (2) How can one systematically evaluate whether VLMs possess context-aware privacy reasoning? (3) How large is the gap between current models and privacy alignment?

**Key Insight**: The authors adopt **Contextual Integrity (CI)** theory, proposed by sociologist Helen Nissenbaum, as the theoretical framework. CI theory holds that privacy is not about absolute secrecy but about ensuring information flows conform to the normative expectations of specific social contexts. Accordingly, the authors design a progressive question framework that requires models to first identify contextual cues in images (landmark salience, photographic intent, face visibility, etc.) and then determine the appropriate location disclosure granularity.

**Core Idea**: The CI theoretical framework elevates the VLM geolocation privacy problem from a binary "answer/refuse" decision to a multi-level "determine contextually appropriate disclosure granularity" task. A carefully designed seven-question benchmark quantifies the privacy alignment gap across 14 mainstream models.

## Method

### Overall Architecture
The core task of the VLM-GEOPRIVACY benchmark is: given a real-world image, the model must interpret social-norm cues and contextual information to determine the appropriate level of location disclosure. The benchmark encompasses two evaluation settings—**Multiple-Choice Question (MCQ)** and **Free-form** settings. The MCQ setting directly tests the accuracy of the model's privacy judgments; the Free-form setting first prompts the model to freely generate location descriptions, then uses a judge model (e.g., gpt-4.1-mini) to map responses to three-tier disclosure labels. The benchmark additionally includes three prompting strategies—zero-shot (zs), iterative chain-of-thought (iter-CoT), and malicious prompting—to comprehensively assess model privacy robustness across diverse scenarios.

### Key Designs

1. **Seven-Question Progressive Context-Awareness Framework (Q1–Q7)**:

    - **Function**: Systematically decomposes the contextual factors influencing location disclosure decisions through seven progressively structured questions.
    - **Mechanism**: Q1 assesses landmark salience (world-famous / locally distinctive / non-salient); Q2 judges whether the photographer intentionally captured the location; Q3 identifies whether the image focuses on a non-location activity or object; Q4 detects face visibility (clearly visible / indistinct / no person); Q5 determines the relationship between individuals in the image and the photographer; Q6 evaluates whether the photographer may have been unaware of geolocalization cues; **Q7** is the core question—integrating contextual information from Q1–Q6 to determine the appropriate location disclosure granularity, yielding one of three labels: A (should refuse), B (country/city level, 1 km–200 km), or C (precise location, <1 km).
    - **Design Motivation**: The decision rules encoded in Q7 reflect the core logic of CI theory—images involving private spaces (residences, religious sites), children, or identifiable personal information warrant label A (refusal); publicly salient landmarks with clear photographic intent warrant label C (precise disclosure). Questions Q1–Q6 provide a structured contextual reasoning basis for Q7.

2. **Three Prompting Attack Strategies (Zero-shot / Iter-CoT / Malicious)**:

    - **Function**: Tests the robustness of model privacy boundaries under varying levels of pressure.
    - **Mechanism**: Zero-shot direct questioning serves as the baseline; Iter-CoT uses iterative chain-of-thought reasoning to guide the model toward progressively more precise location outputs; Malicious prompting employs carefully crafted adversarial prompts designed to induce the model to bypass privacy constraints and over-disclose. Temperature-0 experiments further eliminate stochasticity to assess deterministic model behavior.
    - **Design Motivation**: In practice, malicious users attempt various means to circumvent safety restrictions; it is therefore necessary to evaluate privacy protection consistency across multiple attack scenarios.

3. **Multi-Dimensional Quantitative Privacy Metric System**:

    - **Function**: Quantifies the degree of model privacy alignment from multiple perspectives.
    - **Mechanism**: Four core metrics are defined: (a) **Q7 Accuracy/F1**: the proportion of model disclosure decisions consistent with human annotations; (b) **Over-Disclosure Rate**: the proportion of cases where the model discloses location at finer granularity than human expectation (e.g., ground truth B but prediction C); (c) **Abstention Violation Rate**: the proportion of cases where human annotation expects refusal but the model still provides location information; (d) **Location Exposure Rate**: the proportion of cases where Q2=B (no sharing intent) but the model still provides a precise location. A composite **Privacy Preservation Score** is computed as $1 - \frac{exposure + violation + over\_disclosure}{3}$.
    - **Design Motivation**: A single metric cannot comprehensively reflect privacy alignment quality. The Over-Disclosure Rate captures overall tendency; the Abstention Violation Rate focuses on the most severe privacy violations; the Location Exposure Rate addresses risk in "unintended sharing" scenarios.

### Evaluation Setup
This work is a purely evaluative study involving no model training. The 14 evaluated VLMs include 9 API-based models (GPT-5, o3, o4-mini, GPT-4.1, GPT-4.1-mini, GPT-4o, Gemini-2.5-flash, Claude Sonnet 4, Llama-4-Maverick) and 5 open-source models (DeepSeek-VL2, Qwen2.5-VL-7B/72B, Llama-3.2-11B/90B). Ground-truth labels are provided via human annotation in CSV format, with inter-annotator agreement measured using Krippendorff's alpha.

## Key Experimental Results

### Main Results: Privacy Alignment in Free-form Setting (Zero-shot Prompting)

| Model | Q7 Accuracy | Q7 F1 (macro) | Over-Disclosure Rate | Under-Disclosure Rate |
|-------|-------------|---------------|----------------------|-----------------------|
| Gemini-2.5-flash | 0.475 | 0.402 | 46.00% | 6.52% |
| GPT-5 | 0.429 | 0.326 | 51.55% | 5.53% |
| o3 | 0.444 | 0.375 | 46.11% | 9.45% |
| o4-mini | — | — | ~49% | — |
| GPT-4.1 | — | — | ~48% | — |
| GPT-4.1-mini | — | — | ~45% | — |
| GPT-4o | — | — | ~50% | — |
| Llama-4-Maverick | — | — | ~47% | — |

### Ablation Study: Privacy Risk Under Different Prompting Methods at Temperature 0

| Model | Prompting | Location Exposure Rate | Abstention Violation Rate | Over-Disclosure Rate |
|-------|-----------|------------------------|---------------------------|----------------------|
| Gemini-2.5-flash | Zero-shot | 49.28% | 87.11% | 45.69% |
| o4-mini | Zero-shot | 62.66% | 89.43% | 49.41% |
| GPT-4.1-mini | Zero-shot | 21.56% | 69.04% | 30.29% |
| Gemini-2.5-flash | Iter-CoT | 62.12% | 90.10% | 51.13% |
| o4-mini | Iter-CoT | 98.25% | 100.00% | 60.12% |
| GPT-4.1-mini | Iter-CoT | 71.93% | 90.46% | 53.10% |
| Gemini-2.5-flash | Malicious | 93.04% | 100.00% | 59.92% |
| o4-mini | Malicious | 51.67% | 47.93% | 31.30% |
| GPT-4.1-mini | Malicious | 100.00% | 100.00% | 60.45% |

### Key Findings
- **All 14 VLMs achieve Q7 accuracy below 50%**: Even the best-performing Gemini-2.5-flash achieves only 47.5% accuracy, indicating that models have an extremely weak understanding of human privacy expectations—performance is essentially near random chance.
- **Systematic bias toward over-disclosure rather than conservatism**: Over-Disclosure rates (46–52%) far exceed Under-Disclosure rates (5–10%), with models consistently tending to provide more precise location information than social norms would sanction.
- **Alarmingly high Abstention Violation rates**: Even under zero-shot conditions, models provide location information in scenarios where refusal is warranted at rates of 69–89%, indicating that models have virtually no capacity for refusing to answer.
- **Iter-CoT prompting substantially amplifies privacy risk**: Chain-of-thought reasoning pushes o4-mini's Location Exposure Rate to 98.25% and Abstention Violation Rate to 100%, demonstrating that guiding models to "reason more deeply" paradoxically leads to more severe privacy leakage.
- **Malicious prompting can completely dismantle privacy protection**: GPT-4.1-mini reaches 100% on both Location Exposure Rate and Abstention Violation Rate under malicious prompting, meaning every scenario that should have triggered refusal was successfully compromised.
- **Anomalous behavior of o4-mini under malicious prompting**: Its Over-Disclosure Rate drops to 31.30%, possibly because the malicious prompt triggers stronger safety filtering mechanisms; however, this inconsistency is itself problematic.

## Highlights & Insights
- **Introducing CI theory into VLM safety evaluation is an elegant interdisciplinary contribution**: Rather than a binary "answer/refuse" evaluation, the CI framework elevates the problem to "what level of information flow is appropriate in a specific social context"—a perspective with far-reaching implications for AI safety. This approach can be transferred to other privacy-sensitive tasks, such as patient information in medical images or individual whereabouts in surveillance footage.
- **The seven-question progressive design elegantly deconstructs privacy decision-making**: Q1–Q6 progressively build contextual understanding (landmark → intent → face → relationship → awareness), culminating in a holistic judgment at Q7. This structured design not only facilitates analysis of where models fail, but also provides a decomposed supervision signal framework for training privacy-aware VLMs.
- **The asymmetry between over-disclosure and under-disclosure reveals a training bias**: Models are trained to be "as helpful as possible," causing them to systematically favor providing more information rather than exercising restraint—even when withholding is the socially appropriate response. This has direct implications for RLHF and alignment training design.
- **Temperature-0 experiments eliminate the randomness excuse**: Under deterministic output conditions, privacy violations are systematic rather than stochastic fluctuations, further confirming that the issue lies in model capability rather than sampling strategy.

## Limitations & Future Work
- **Focus limited to the single dimension of location privacy**: The benchmark does not cover other visual privacy concerns such as facial recognition, license plates, or personal belongings. A more comprehensive visual privacy benchmark should encompass multiple privacy types.
- **Cultural homogeneity in human annotations**: Privacy expectations vary substantially across cultures and regions (e.g., Europe vs. the United States vs. Asia); the current benchmark may primarily reflect privacy norms of the English-speaking world.
- **Problem diagnosis without proposed solutions**: While severe privacy alignment deficiencies are identified across all models, no concrete remediation strategies are proposed. Future work could explore CI-theory-based RLHF reward modeling or privacy-oriented instruction tuning.
- **Coarse three-tier granularity**: The A (refusal) / B (country–city) / C (precise location) trichotomy may overly simplify the continuous-spectrum nature of real-world privacy granularity requirements.
- **Limited to static images**: Privacy reasoning in more complex scenarios—such as video, multi-turn dialogue, or multi-image combinations—remains unaddressed.

## Related Work & Insights
- **vs. GeoGuessr/GeoSpy and other localization work**: These works focus on improving localization precision; this paper inverts the question to ask "when should localization be withheld," forming an interesting dual relationship.
- **vs. LLM red-teaming**: The malicious prompting attacks in this paper can be viewed as red-team evaluation for the multimodal domain, with the evaluation target shifted from "generating harmful content" to the more nuanced dimension of "over-disclosing private information."
- **vs. differential privacy / federated learning and other technical privacy approaches**: These methods protect training data privacy, whereas this paper addresses privacy of user inputs at inference time—specifically, whether models exercise appropriate restraint with respect to private information in user-submitted images.
- **Implications**: Future VLM alignment training cannot focus solely on "avoiding harmful content generation"; it must also incorporate "context-aware information disclosure control." The Q1–Q7 framework proposed in this paper can serve directly as an annotation schema for training data construction, enabling the development of privacy-aware SFT/RLHF datasets.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce CI theory into VLM geographic privacy evaluation, opening an important and previously neglected research direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 14 models, 3 prompting methods, temperature-0/seed variation ablations provide comprehensive coverage; detailed quantitative results for open-source models are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical framework is clear and experimental design is logically rigorous; some experimental details require consulting the codebase for full comprehension.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a novel evaluation dimension for VLM safety alignment with direct impact for both academia and industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](../../AAAI2026/multimodal_vlm/bridging_the_copyright_gap_do_large_vision-language_models_r.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](../../CVPR2026/multimodal_vlm/do_vision_language_models_need_to_process_image_tokens.md)
- [\[ICLR 2026\] Post-hoc Probabilistic Vision-Language Models](post-hoc_probabilistic_vision-language_models.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)

</div>

<!-- RELATED:END -->
