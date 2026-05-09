---
title: >-
  [Paper Note] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][Multimodal Reasoning Models] This paper presents the first systematic study of privacy leakage risks arising from multimodal large reasoning models (MLRMs) inferring sensitive geographic location information from user-generated images. It proposes a three-tier privacy risk framework, the DoxBench benchmark, and the Glare information-theoretic evaluation metric. The findings demonstrate that MLRMs surpass non-expert humans in geographic inference, significantly lowering the barrier for adversaries to obtain sensitive location information.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Multimodal Reasoning Models
  - Privacy Leakage
  - Geolocalization
  - Benchmark
  - AI Safety
date: 2026-05-08
content_hash: e9a3ab2b4f43659f
---

# Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2504.19373](https://arxiv.org/abs/2504.19373)
**Code**: [https://github.com/SaFo-Lab/DoxBench](https://github.com/SaFo-Lab/DoxBench)
**Area**: LLM Reasoning
**Keywords**: Multimodal Reasoning Models, Privacy Leakage, Geolocalization, Benchmark, AI Safety

## TL;DR

This paper presents the first systematic study of privacy leakage risks arising from multimodal large reasoning models (MLRMs) inferring sensitive geographic location information from user-generated images. It proposes a three-tier privacy risk framework, the DoxBench benchmark, and the Glare information-theoretic evaluation metric. The findings demonstrate that MLRMs surpass non-expert humans in geographic inference, significantly lowering the barrier for adversaries to obtain sensitive location information.

## Background & Motivation

- **State of the Field**: The emergence of multimodal large reasoning models such as OpenAI o3 and Gemini 2.5 Pro has endowed models with the ability to infer high-level semantic information from visual inputs. While valuable for applications such as navigation and augmented reality, this capability introduces serious location-related privacy leakage risks. Under GDPR and CCPA, geolocation data is classified as personal information, and precise geolocation is further classified as "sensitive personal information."
- **Limitations of Prior Work**: Existing research has three primary limitations: (1) it focuses primarily on geolocalization performance rather than privacy leakage risk per se; (2) datasets largely feature "benign" scenes such as public landmarks, lacking privacy-sensitive scenarios; and (3) low-resolution Google Street View images are used, failing to reflect the quality of real user-generated content.
- **Paper Goals**: This paper aims to close this gap by systematically studying the location inference capabilities of MLRMs in privacy-sensitive scenarios.

## Method

### Overall Architecture

The paper constructs a comprehensive research framework comprising a three-tier privacy risk definition, the DoxBench benchmark dataset, a novel evaluation metric (Glare), a clue analysis tool (ClueMiner), and a collaborative attack framework (GeoMiner).

### Key Designs

**Three-Tier Privacy Risk Framework**: Based on the two dimensions of "private space" and "personal appearance," three risk levels are defined:
- Level 1 (Low Risk): Personal appearance present but not in a private space, corresponding to personal transient risk.
- Level 2 (Medium Risk): In a private space but without personal appearance, corresponding to household persistent risk.
- Level 3 (High Risk): Both private space and personal appearance present, with both risk types compounded.

**DoxBench Dataset**: A collection of 500 high-resolution images captured with an iPhone across 6 California cities (San Francisco, San Jose, Sacramento, Los Angeles, Irvine, and San Diego), covering scenarios including selfies and third-person perspectives. Images are divided into 6 categories (including a special Mirror/reflection category) and retain complete EXIF metadata.

**Glare Information-Theoretic Metric**: Unifies the Verifiable Response Rate (VRR), median error distance $d_{50}$, and mean error distance $\bar{d}$ into a single bit-valued measure:

$$\text{Glare} = H(R) + \text{VRR} \cdot \log_2\!\left(\frac{A_0}{\pi \cdot d_{50} \cdot \bar{d}}\right)$$

where $H(R)$ is the entropy of response behavior (Risk Term), and the second term quantifies the degree to which responses narrow the adversary's search area (Leakage Term).

**ClueMiner Clue Analysis Tool**: Employs chain-of-thought (CoT) prompting to elicit the visual clues used during the model's reasoning process, followed by automated classification and frequency statistics to identify whether the model relies on privacy-sensitive visual cues.

**GeoMiner Attack Framework**: Decomposes geographic inference into two stages — a Detector that extracts visual clues, and an Analyzer that conducts reasoning based on these clues — simulating a collaborative attack mode analogous to consulting a human expert.

### Loss & Training

This is an evaluation study and does not involve model training. The core technical contribution lies in the information-theoretic derivation of the Glare metric: starting from mutual information decomposition, a closed-form evaluation metric is derived via a Shannon entropy upper bound, a uniform Earth prior assumption, and a flat-Earth approximation.

## Key Experimental Results

### Main Results

Thirteen models (7 MLRMs + 6 MLLMs) and 268 non-expert humans are evaluated on DoxBench:

| Model | VRR (%) | AED (km) | MED (km) | CCPA Acc. (%) | Glare (bits) |
|-------|---------|----------|----------|---------------|--------------|
| Non-expert Humans | 99.10 | 140.08 | 37.22 | 6.01 | 1309.73 |
| GPT-5 (Top-1) | 78.41 | 11.26 | 4.35 | 17.40 | 1633.87 |
| OpenAI o3 (Top-1) | 80.80 | 13.56 | 5.46 | 14.73 | 1628.50 |
| Gemini 2.5 Pro (Top-1) | 84.53 | 14.75 | 4.63 | 19.73 | 1701.61 |
| Gemini 2.5 Pro (Top-3) | 95.07 | 9.92 | 2.98 | 21.97 | 1987.16 |
| GPT-5 (Top-3) | 74.23 | 6.69 | 2.15 | 22.03 | 1688.66 |

Key findings: Under the Top-1 setting, MLRMs achieve an average CCPA accuracy of 11.61%; under Top-3, this rises to 14.95%. The average Glare of MLRMs exceeds the non-expert human baseline.

### Ablation Study

| Ablation Dimension | Key Result |
|-------------------|------------|
| Risk Level (L1→L3) | CCPA accuracy and Glare decrease monotonically from L1 to L3; Mirror category is most challenging |
| CoT Clue Reasoning | CCPA +4.91% on answered samples; CCPA +11.17% on previously unanswered samples |
| Tool Augmentation (o3) | VRR 84.85%→100%, AED 168.71→42.88 km, Glare +49.45% |
| Manual Blurring Defense | VRR reduced by 16.58%, Glare reduced by 30.6%, yet CCPA accuracy remains at 10.56% |
| Adversarial Noise Defense | Effective against o3 (Glare: 2648→593) but ineffective against Gemini |

### Key Findings

1. **MLRMs surpass non-expert humans**: GPT-5 achieves a Top-3 CCPA accuracy of 22.03%, approximately 3.7× the human baseline.
2. **Clue-driven reasoning mechanism**: In 98% of samples, models follow a clue-driven reasoning pattern; the most frequently used clues are "street layout" and "front yard design."
3. **Tool use substantially amplifies attacks**: o3 equipped with search tools demonstrates significantly enhanced fine-grained localization capability.
4. **Existing defenses are insufficient**: Llama Guard4 classifies all inputs as safe; blurring and adversarial noise offer only limited mitigation.
5. **Reflective surface privacy threat**: The Mirror category reveals a novel threat of indirect location leakage through reflective surfaces.

## Highlights & Insights

- **Precise problem formulation**: This is the first work to systematically define three-tier privacy risks for image-based location leakage from a legal framework (GDPR/CCPA), tightly coupling security research with regulatory compliance.
- **Elegant Glare metric design**: The information-theoretic formulation unifies response behavior and response content into a single quantitative measure, addressing the inability of existing metrics (standalone median distance or VRR) to comprehensively capture privacy risk.
- **High-quality dataset**: Images are captured with an iPhone in real-world scenarios, including innovative categories such as the Mirror class, far exceeding existing low-resolution Street View datasets.
- **Complete attack chain**: The research covers the full pipeline from risk definition → dataset → evaluation → clue analysis → attack framework → defense evaluation.

## Limitations & Future Work

1. The dataset is primarily sourced from California; although 50 images from other U.S. states are included for generalization validation, international scene coverage remains limited.
2. Exploration of defense mechanisms is relatively preliminary, with only three methods tested: Llama Guard4, manual blurring, and adversarial noise.
3. Evaluation is restricted to black-box API access; the internal model mechanisms (e.g., attention visualization) underlying the clue extraction process remain unexplored.
4. Although the Mirror category is innovative, its sample size (46 images) is small, and statistical significance warrants further validation.

## Related Work & Insights

The core distinction from prior geolocalization evaluation work is threefold: (1) the shift from "assessing localization capability" to "quantifying privacy leakage risk"; (2) the use of genuinely privacy-sensitive scenarios rather than public landmarks; and (3) the proposal of a unified information-theoretic metric. This work carries important implications for the AI safety community: models with stronger reasoning capabilities pose greater privacy threats, necessitating the integration of privacy alignment mechanisms at inference time.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic study of location privacy leakage in MLRMs; novel problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 13 models + human baseline with multiple ablation dimensions; defense experiments are somewhat thin.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; effective integration of legal framework and technical metrics; thorough appendix.
- **Value**: ⭐⭐⭐⭐⭐ — Carries significant cautionary implications for AI safety and privacy protection.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)

<!-- RELATED:END -->
