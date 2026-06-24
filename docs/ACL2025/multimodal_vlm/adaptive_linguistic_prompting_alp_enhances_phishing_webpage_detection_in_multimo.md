---
title: >-
  [Paper Note] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models
description: >-
  [ACL 2025][Multimodal VLM][Phishing Webpage Detection] Proposes Adaptive Linguistic Prompting (ALP), an 8-shot structured prompting approach that guides multimodal LLMs to jointly reason across HTML text, screenshots, and URLs to detect phishing webpages. Combined analysis achieves an F1-score of $0.93$ on GPT-4o, outperforming traditional zero-shot baselines.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Phishing Webpage Detection"
  - "Multimodal LLM"
  - "few-shot prompting"
  - "GPT-4o"
  - "Gemini 1.5 Pro"
date: 2026-05-08
content_hash: 9e3a4290888066f6
---

# Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.13357](https://arxiv.org/abs/2507.13357)  
**Code**: [https://github.com/atharvab167/Adaptive-Linguistic-Prompting-ALP-Multimodal-LLM-Phishing-Detection](https://github.com/atharvab167/Adaptive-Linguistic-Prompting-ALP-Multimodal-LLM-Phishing-Detection)  
**Area**: Multimodal VLM / AI Security  
**Keywords**: Phishing Webpage Detection, Multimodal LLM, few-shot prompting, GPT-4o, Gemini 1.5 Pro

## TL;DR
Proposes Adaptive Linguistic Prompting (ALP), an 8-shot structured prompting approach that guides multimodal LLMs to jointly reason across HTML text, screenshots, and URLs to detect phishing webpages. Combined analysis achieves an F1-score of $0.93$ on GPT-4o, outperforming traditional zero-shot baselines.

## Background & Motivation
**Background**: Phishing attacks represent a significant cybersecurity threat, with over 1.2 million mitigated attacks in 2024 alone. Traditional detection methods rely on heuristic URL matching, HTML structure analysis, and blacklist mechanisms.

**Limitations of Prior Work**: Traditional methods perform poorly against zero-day attacks and advanced evasion/obfuscation techniques. Machine-learning-based approaches face challenges from adversarial perturbations and dynamic content, while computer-vision-based brand identification methods require continuous retraining.

**Key Challenge**: While multimodal LLMs (e.g., GPT-4o) possess powerful semantic understanding capabilities, directly using zero-shot prompting for phishing detection lacks structured reasoning guidance, failing to fully exploit multimodal information. While prior work (Lee et al. 2024) demonstrated that LLMs outperform traditional methods in phishing detection, their prompt design remains a simplistic zero-shot brand-and-domain verification.

**Goal**: To enhance the phishing detection capabilities of multimodal LLMs through prompt engineering (without modifying model weights)—specifically, by designing structured few-shot prompts to guide LLMs through step-by-step semantic reasoning.

**Key Insight**: Combine few-shot prompting and chain-of-thought (CoT) reasoning to design modality-specific structured prompt templates, guiding LLMs to separately analyze webpage content, screenshot visual cues, and URL structures.

**Core Idea**: Utilize a meticulously designed 8-shot prompt (ALP) to guide multimodal LLMs in performing structured reasoning across textual deception patterns, urgency cues, and manipulative phrasing. This substantially improves phishing detection performance without modifying the underlying model.

## Method

### Overall Architecture
The system comprises two parallel analysis pipelines and a fusion decision module:
- **Input**: The HTML content, screenshot image, and embedded URL list of the webpage to be detected.
- **Pipeline 1 - MWA (Multimodal Webpage Analysis)**: Simultaneously analyzes the HTML text and screenshots, guiding the LLM via 8-shot ALP prompts to evaluate brand consistency, language compliance, and visual trust.
- **Pipeline 2 - USA (URL Structure Analysis)**: Extracts all URLs from the webpage, guiding the LLM via 8-shot ALP prompts to analyze domain legitimacy, protocol security, and path structure.
- **Fusion Decision**: Combines the outputs of both pipelines for final classification.

### Key Designs

1. **MWA (Multimodal Webpage Analysis) Prompt**:

    - **Function**: Guides the LLM to simultaneously analyze HTML contents and webpage screenshots, outputting brand identification, phrasing analysis, visual analysis, confidence scores, and final classifications.
    - **Mechanism**: Designs a structured 8-shot prompt template, where each exemplar demonstrates reasoning through the sequence: "Brand Identification $\rightarrow$ Phrasing & Grammar Analysis $\rightarrow$ Screenshot Analysis $\rightarrow$ Key Metrics $\rightarrow$ Supporting Evidence $\rightarrow$ Confidence $\rightarrow$ Classification". Upon seeing these examples, the LLM learns to execute the same step-by-step analysis on new webpages.
    - **Design Motivation**: Zero-shot prompting asks the LLM to directly output a "phishing/benign" judgment, which lacks reasoning depth. Displaying structured reasoning processes through few-shot exemplars steers the LLM's attention to subtle cues like brand mismatches, linguistic grammatical irregularities, and visual spoofing.

2. **USA (URL Structure Analysis) Prompt**:

    - **Function**: Analyzes all URLs extracted from the HTML to detect phishing indicators within the domains, protocols, and paths.
    - **Mechanism**: Uses a similar 8-shot structured prompt to guide the LLM in analyzing domain legitimacy (e.g., whether it belongs to a well-known brand), HTTPS utilization, and whether the URL path contains suspicious structures (such as generic TLDs, deceptive paths).
    - **Design Motivation**: URL analysis and content analysis are complementary. Some phishing pages perform exceptionally well visually but expose themselves through their URLs, and vice versa.

3. **"Suspicious-First" Strategy**:

    - **Function**: Integrates a risk-oriented "suspicious-first" instruction into the USA prompt, directing the LLM to lean towards classifying uncertain cases as suspicious.
    - **Mechanism**: Simply appends a risk-averse bias instruction to the prompt template.
    - **Design Motivation**: This simple prompt modification boosted GPT-4o's USA accuracy from $81\%$ to $91\%$, illustrating that the impact of prompt engineering can rival architectural model modifications.

### Fusion Decision Strategy
When the results of MWA and USA are consistent, they are directly adopted. In case of disagreement, if USA classifies the page as phishing or the MWA confidence level is $> 8.5$, it is classified as phishing; otherwise, block as benign. The threshold of $8.5$ was determined on the training set by balancing precision and recall. This risk-averse fusion strategy prioritizes security.

## Key Experimental Results

### Dataset
Using the dataset from Lee et al. (2024): 1,607 benign brands and 289 phishing brands, containing one screenshot and one HTML JSON file per brand. After filtering, 311 benign brands and all 289 phishing brands were retained.

### Main Results

| Model | Method | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| GPT-4o | Baseline (Zero-shot) | 0.91 | 0.91 | 0.91 |
| GPT-4o | MWA (ALP) | 0.80 | 0.89 | 0.84 |
| GPT-4o | USA (ALP) | 0.91 | 0.91 | 0.91 |
| GPT-4o | **Combined** | **0.92** | **0.94** | **0.93** |
| Gemini 1.5 Pro | Baseline | 0.76 | 0.85 | 0.81 |
| Gemini 1.5 Pro | MWA (ALP) | 0.94 | 0.87 | 0.90 |
| Gemini 1.5 Pro | USA (ALP) | 0.88 | 0.85 | 0.87 |
| Gemini 1.5 Pro | **Combined** | **0.91** | **0.92** | **0.91** |

### Ablation Study

| Configuration | F1 (GPT-4o) | Description |
|------|-------------|------|
| Combined (ALP) | 0.93 | Full method |
| USA w/o suspicious-first | 0.81 | USA drops significantly without risk-averse prompt |
| USA w/ suspicious-first | 0.91 | Improves by 10 points when added back |
| Baseline (Zero-shot) | 0.91 | Lee et al.'s brand+domain verification |
| MWA standalone | 0.84 | Webpage content + screenshot analysis only |

### Key Findings
- GPT-4o performs better in URL analysis, while Gemini 1.5 Pro excels in multimodal webpage analysis. Theoretically, the optimal combination is Gemini for MWA and GPT-4o for USA.
- The simple "suspicious-first" prompt modification yielded a $10\%$ improvement in F1-score (from $0.81$ to $0.91$), demonstrating that the impact of prompt engineering can equal that of model improvements.
- Fusing the two pipelines outperforms any single pipeline, as URL analysis and content analysis capture complementary phishing cues.

## Highlights & Insights
- **The Power of Prompt Engineering**: Achieving significant performance boosts without model modifications. The "suspicious-first" strategy yields a $10\%$ F1 improvement with just a single sentence, which is highly practical.
- **Complementary Fusion Design**: Decomposing the problem into two independent sub-analyses (content analysis vs. URL analysis) and fusing them with simple rules is more interpretable and flexible than end-to-end approaches.
- **Cross-Model Complementarity**: Different LLMs demonstrate distinct strengths across modalities, offering potential for building hybrid systems.

## Limitations & Future Work
- The dataset scale is relatively small (600 brands), and lacks coverage of emerging phishing tactics and non-English content.
- Reliance on closed-source commercial LLMs (GPT-4o) incurs high inference costs, hindering large-scale deployment.
- The fusion strategy is overly simplistic (hard threshold of $8.5$), which could be replaced with weighted probabilities or learning-based fusion models.
- Robustness against adversarial evasion strategies (e.g., context-aware rewriting) remains untested.
- A comprehensive comparison with traditional ML methods (non-Transformer architectures) is lacking.

## Related Work & Insights
- **vs. Lee et al. (2024)**: They proposed a multimodal LLM approach based on zero-shot brand-and-domain verification (baseline). This work replaces the zero-shot prompt with an 8-shot ALP prompt, lifting the F1-score from $0.91$ to $0.93$.
- **vs. Koide et al. (2024) ChatSpamDetector**: They also use LLMs for security detection, but ChatSpamDetector targets emails, whereas this work targets webpages and emphasizes multimodal fusion.
- **vs. Visual Methods (PhishPedia, etc.)**: Visual methods require continuous retraining to adapt to new brands, whereas LLM-and-prompting approaches possess inherent zero-shot generalization capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core contribution lies in prompt design. While methodological novelty is moderate, the utility is strong.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Small dataset size; lacks comparison with more baselines and robustness tests.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure and detailed experimental description; the appendix provides complete prompt templates and output examples.
- **Value**: ⭐⭐⭐⭐ Strong practical applicability. The findings on prompt engineering (especially "suspicious-first") offer good reference value, though the overall depth is somewhat limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ECCV 2024\] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](../../ECCV2024/multimodal_vlm/adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)
- [\[ACL 2025\] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)

</div>

<!-- RELATED:END -->
