---
title: >-
  [Paper Note] Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment
description: >-
  [NeurIPS 2025][LLM Safety][Social Bias] The authors construct a benchmark consisting of 1,343 real news image–open-ended question pairs, with each image annotated with social attributes such as age, gender, race, occupation, and sports. Then, using GPT-4o as a judge, they evaluate 15 mainstream VLMs across three dimensions: accuracy, bias, and faithfulness. They find that visual social cues systematically alter model responses, gender and occupational biases are the most seve…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Social Bias"
  - "Vision-Language Models"
  - "Benchmarking"
  - "LLM-as-Judge"
  - "Faithfulness"
date: 2026-05-08
content_hash: 6bbac866aafe54f9
---

# Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment

**Conference**: NeurIPS 2025  
**arXiv**: [2509.19659](https://arxiv.org/abs/2509.19659)  
**Code**: None (The paper promises to open-source prompts / rubric / code, but no specific repository address is provided in the main text; ⚠️ subject to the original text)  
**Area**: AI Safety / Multimodal VLM  
**Keywords**: Social Bias, Vision-Language Models, Benchmarking, LLM-as-Judge, Faithfulness

## TL;DR
The authors construct a benchmark consisting of 1,343 real news image–open-ended question pairs, with each image annotated with social attributes such as age, gender, race, occupation, and sports. Then, using GPT-4o as a judge, they evaluate 15 mainstream VLMs across three dimensions: accuracy, bias, and faithfulness. They find that visual social cues systematically alter model responses, gender and occupational biases are the most severe, and higher faithfulness does not necessarily translate to lower bias.

## Background & Motivation
**Background**: Vision-Language Models (VLMs) couple visual encoders with large language models to jointly process images and text, and have been widely used in tasks such as visual question answering, image-text dialogue, and instruction following. However, images naturally carry "social cues" such as age, gender, race, occupation, and attire, which activate latent associations within the models.

**Limitations of Prior Work**: Most existing fairness benchmarks are text-only (e.g., DecodingTrust), examining bias solely in text prompts; research on how images trigger or amplify stereotypes lags significantly behind. Even existing VLM bias evaluation works (e.g., VisoGender, VL-Stereoset, SocialBias, PAIRS, GenderBias-VL) generally suffer from three common flaws: (i) reliance on text-only or synthetic/captioned setups rather than real images; (ii) a focus on closed-ended tasks like classification and multiple-choice; (iii) evaluating bias in isolation from grounding/faithfulness, failing to clarify the role played by visible social cues in real images.

**Key Challenge**: A model can "faithfully" describe visible evidence in an image while simultaneously injecting demographic assumptions that do not exist in the picture. Bias and faithfulness are actually two independent, sometimes even conflicting, dimensions. Prior evaluations conflated them or only measured one, leading to distorted conclusions.

**Goal**: Decomposed into three research questions—RQ1: How do current VLMs perform overall on real image-text pairs with social cues? RQ2: How does performance vary across different social attributes (age/gender/race/occupation/sports)? RQ3: What is the trade-off between faithfulness and stereotyping bias?

**Key Insight**: Using real news images paired with open-ended questions, along with fine-grained demographic annotations for each image, enables the simultaneous evaluation of both bias and faithfulness on the same dataset. News images are close to real-world deployment scenarios and inherently carry rich social cues, exposing the true tendencies of models better than synthetic images.

**Core Idea**: Replace "synthetic images + closed-ended classification + single-dimensional bias scores" with "real news images + open-ended QA + multi-dimensional LLM-as-Judge rubric" to decouple bias and faithfulness for joint auditing.

## Method

### Overall Architecture
The core of this paper is an evaluation benchmark and protocol, self-contained in two clear phases: the data side (top half)—collecting images from news sources, filtering and deduplicating using a whitelist, manually annotating attributes and ground-truth answers, yielding 1,343 image-text pairs; and the evaluation side (bottom half)—using a unified zero-shot prompt to generate structured JSON outputs (answer + rationale) from 15 VLMs, scoring them via a 6-dimensional rubric with GPT-4o as the judge, followed by human spot-checks. The input to the entire pipeline is a "news image", and the outputs are each model's scores across three core metrics: accuracy, bias, and faithfulness.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Google News RSS<br/>News Images (July 2024)"] --> B["News Image Benchmark Construction<br/>Whitelist Filtering + Deduplication + Attribute Annotation"]
    B --> C["1,343 Image-Text Pairs<br/>with Age/Gender/Race/Occupation/Sports Labels"]
    C --> D["Open-Ended Querying Protocol<br/>Zero-Shot + JSON (answer + rationale)"]
    D -->|15 VLMs (7B~19B)| E["Model Output"]
    E --> F["LLM-as-Judge Evaluation<br/>GPT-4o Six-Dimensional Rubric"]
    F --> G["Accuracy ↑ / Bias ↓ / Faithfulness ↑<br/>+ Human Spot-Checks"]
```

### Key Designs

**1. Real News Image Benchmark Construction: Exposing Stereotypes with Real Images Carrying Social Cues**

To address the limitation of prior works relying on synthetic images/pure text detached from real-world deployment, the authors collected images from Google News RSS feeds in July 2024, covering themes such as healthcare, climate, education, diplomacy, social justice, gun control, inequality, democracy, technology, and environment. To ensure reliability, they only retained sources from mainstream media whitelists (CNN, BBC, NYT, Guardian, Reuters, AP, Bloomberg, USA Today, etc.). After deduplication, each image was paired with an open-ended question designed to assess both scene understanding and social cues. Each image is annotated with five attributes: age (child/adult/elderly), gender (male/female/unknown), race (Black/White/Asian/Indigenous/Other), occupation (doctor/nurse/engineer...), and sports (soccer/basketball...). The Q&A pairs and attribute labels were first drafted by an LLM and then verified by 5 trained annotators, with disagreements resolved by majority voting or arbitration. Using real news images instead of synthetic ones aligns the boundary between "visible cues" and "model assumptions" more closely with practical applications.

**2. Open-Ended Querying Protocol: Stabilizing Outputs and Isolating Grounding Evidence with Structured JSON**

Past closed-ended (classification/multiple-choice) tasks failed to expose models' tendency to make "extrapolative assumptions" in free-form generation. The authors instead adopt a standardized zero-shot prompt: temperature is set to 0 (to ensure deterministic replication), with an additional safety scan run at 0.2, top-$p=1.0$, and a maximum of 128 tokens. The model is required to return a JSON object containing `answer` (string) and `rationale` (2–3 sentences). This format reduces verbosity variance, stabilizes the downstream judge, and forces the model to explicitly state its reasoning. When few-shot exemplars are provided, their rationales are strictly restricted to referencing only **visible evidence**, thereby guiding the model to ground its responses in the image.

**3. LLM-as-Judge Six-Dimensional Rubric: Jointly Scoring Decoupled Bias and Faithfulness**

This is the key design to investigate RQ3 "faithful $\neq$ unbiased". The authors use GPT-4o as a judge to grade outputs based on a 1-100 rubric. The three core dimensions are: **Bias (lower is better)**, **Answer Relevance (higher is better)**, and **Faithfulness (to the image, higher is better)**. The judge is provided with the image, question, and the model's JSON output, and is explicitly instructed to **penalize stereotypical assertions that lack visible evidence support**. Since bias and faithfulness are split into two independent scoring axes, the authors quantitatively observe the phenomenon where a model can faithfully describe evidence in the image while simultaneously injecting demographic assumptions not present in it. The judge's scores are validated via human spot-checks to mitigate the risk of "judge bias".

## Key Experimental Results

### Main Results
In 1,343 image-text pairs, evaluating 15 open-source/commercial VLMs (7B–19B), the overall accuracy/bias/faithfulness (scored by LLM-as-Judge, lower bias is better) are selected as follows:

| Model | Accuracy ↑ | Bias ↓ | Faithfulness ↑ |
|------|---------|--------|---------|
| Gemini 2.0 | 85.97 | 15.19 | 78.96 |
| Aya Vision 8B | 83.76 | **9.84** | 56.78 |
| Janus-Pro 7B | 82.02 | 16.79 | 78.68 |
| Phi-4 | 80.00 | 17.10 | **81.67** |
| InternVL2.5 | 79.98 | 12.97 | 73.50 |
| GLM-4V-9B | 72.47 | 11.96 | 65.71 |
| Qwen2.5-VL | 71.18 | **9.46** | 68.98 |
| Molmo-7B | 63.54 | 13.31 | 56.38 |
| PaliGemma | 58.71 | 19.60 | 67.93 |
| MAGMA | **47.61** | 11.52 | 53.01 |

Key Comparison: Phi-4 has the highest faithfulness (81.67) but its bias remains high at 17.10; Qwen2.5-VL has the lowest bias (9.46) but relatively lower accuracy (71.18). This directly demonstrates that "overall performance cannot be simply attributed to model scale."

### Attribute-Level Analysis
Breaking down accuracy (Acc) / Bias (Bias↓) / Faithfulness (Faith) by five social attributes, selected representative models are as follows:

| Model | Gender Bias↓ | Occupation Bias↓ | Race Bias↓ | Occupation Acc↑ | Race Acc↑ |
|------|-----------|-----------|-----------|----------|----------|
| Gemini 2.0 | 19.2 | 16.2 | 11.9 | 91.6 | 82.0 |
| InternVL2.5 | 15.5 | 29.8 | 5.1 | 91.6 | 73.4 |
| LLaMA 3.2 11B | 21.8 | 30.7 | 9.8 | 80.4 | 74.3 |
| Phi-3.5 Vision | 15.0 | 28.5 | 9.0 | 78.0 | 63.9 |
| Phi-4 | 17.0 | 22.3 | 13.7 | 92.0 | 68.8 |
| Aya Vision 8B | 18.6 | 20.3 | 5.1 | 90.7 | 80.6 |

### Key Findings
- **RQ1 (Overall)**: Newer models such as Gemini, Phi-4, and Aya Vision significantly outperform earlier systems in accuracy and faithfulness. However, **improvements in grounding capability do not always translate to lower bias**, and scale is not a silver bullet.
- **RQ2 (Attribute-Level)**: Accuracy is highest for occupation-related questions (up to 92.0% for Phi-4) and lowest for race (often below 70%). **Bias is most severe in gender and occupation**, indicating these categories easily trigger stereotypical priors. Faithfulness declines in gender-related scenarios, where models frequently extrapolate beyond visible evidence.
- **RQ3 (Faithfulness vs. Bias)**: The two are misaligned. Janus-Pro and Phi-4 can provide faithfully grounded answers while still injecting demographic assumptions about race/gender. Qwen2.5-VL avoids explicit demographic attribution and has low bias, but its answers are less informative. This reveals an inherent tension between "remaining faithful to image evidence" and "avoiding harmful inferences".

## Highlights & Insights
- **Splitting bias and faithfulness into two independent scoring axes** is the most valuable design of this benchmark. It turns the intuition "faithful $\neq$ unbiased" into a quantifiable conclusion (Phi-4 ranks first in faithfulness but exhibits high bias). This decoupled evaluation paradigm can be transferred to any task where correctness and safety compete.
- **Using real news images + open-ended QA** instead of synthetic images/multiple-choice questions makes the evaluation close to real-world deployment and reveals the models' true tendency to over-extrapolate during free-form generation—a behavior that closed-ended tasks fail to capture.
- **Enforcing JSON (answer + rationale) output** is a highly practical engineering trick: it stabilizes inputs for the LLM judge, reduces verbosity variance, and explicitly exposes the model's reasoning process for auditing. This can be directly reused in other LLM-as-Judge evaluations.

## Limitations & Future Work
- The authors acknowledge that the data scale and domain are limited, only covering news images within a one-year window, and lacking breadth across cultures, languages, and visual domains.
- Categorical demographic labels are used for annotation. While convenient for evaluation, they inevitably simplify identities and contexts.
- Although the LLM judge was calibrated and partially validated by humans, it still carries its own pre-trained biases, which may over- or under-estimate subtle harms.
- The analysis is constrained to zero-shot/few-shot prompting and does not capture biased behaviors that might emerge after fine-tuning or reinforcement learning.
- Self-identified limitations: Both bias and faithfulness scores originate from a single judge (GPT-4o), lacking a multi-judge consistency analysis. The seriousness of the "sports" attribute differs significantly from "age/gender/race/occupation", so caution is advised when comparing bias scores horizontally across these attributes (mechanisms triggering bias vary across themes and cannot be directly compared in scale).
- Future work: Expanding to non-Western sources, multilingualism, dynamic social contexts, and introducing alternative evaluation methods such as human-in-the-loop or adversarial probing.

## Related Work & Insights
- **vs VisoGender / VL-Stereoset**: These focus on gender and stereotypical associations but are limited to closed-ended, narrow settings. This work uses real news images + open-ended QA, covering five attribute types and jointly evaluating both bias and faithfulness.
- **vs SocialBias (counterfactual probing)**: SocialBias uses counterfactual prompts to probe demographic attributes, leaning toward text-only/controlled setups. This work measures biases directly on visible cues in real images, aligning better with deployment risks.
- **vs PAIRS / GenderBias-VL**: These demonstrate how models amplify gender/racial stereotypes, but on a smaller scale or in narrower scenarios. This work scales up (1,343 pairs) and incorporates faithfulness as an independent dimension, thereby arriving at the novel conclusion that "faithfulness $\neq$ unbiasedness".

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of jointly auditing decoupled bias and faithfulness on real news images is novel, though individual technical components (LLM-as-Judge, attribute annotation) are mostly combinations of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 15 mainstream VLMs, five attributes, and three core metrics, but relies on a single judge and lacks diverse cultural/linguistic scale.
- Writing Quality: ⭐⭐⭐⭐ The three RQs are clearly structured with straightforward conclusions, but the paper is somewhat short (workshop/short paper style), and method details are relatively brief.
- Value: ⭐⭐⭐⭐ Provides a reproducible multimodal fairness auditing benchmark and a "dual-dimension joint evaluation" paradigm, which has practical reference value for VLM safety evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](../../ICML2025/llm_safety/tamas_benchmarking_adversarial_risks_in_multi-agent_llm_systems.md)
- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](../../ACL2025/llm_safety/real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[ACL 2026\] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](../../ACL2026/llm_safety/ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)
- [\[ACL 2026\] Understanding and Mitigating Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks](../../ACL2026/llm_safety/understanding_and_mitigating_bias_inheritance_in_llm-based_data_augmentation_on_.md)
- [\[ACL 2026\] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](../../ACL2026/llm_safety/generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)

</div>

<!-- RELATED:END -->
