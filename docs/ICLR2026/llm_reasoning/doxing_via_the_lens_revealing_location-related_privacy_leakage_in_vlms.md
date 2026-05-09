---
title: >-
  [Paper Note] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][Privacy Leakage] This paper systematically reveals the privacy leakage risks of multi-modal large reasoning models (MLRMs) in inferring sensitive geographic location information from images. It proposes a three-tier privacy risk framework, the DoxBench benchmark, an information-theoretic metric Glare, and a collaborative attack framework GeoMiner.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Privacy Leakage
  - Geolocation
  - Multi-modal Reasoning Models
  - MLRM
  - Visual Clue Reasoning
date: 2026-05-08
content_hash: 92c5c5bdf339811c
---

# Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2504.19373](https://arxiv.org/abs/2504.19373)
**Code**: [GitHub](https://github.com/SaFo-Lab/DoxBench)
**Area**: LLM Reasoning
**Keywords**: Privacy Leakage, Geolocation, Multi-modal Reasoning Models, MLRM, Visual Clue Reasoning

## TL;DR

This paper systematically reveals the privacy leakage risks of multi-modal large reasoning models (MLRMs) in inferring sensitive geographic location information from images. It proposes a three-tier privacy risk framework, the DoxBench benchmark, an information-theoretic metric Glare, and a collaborative attack framework GeoMiner.

## Background & Motivation

With the emergence of MLRMs such as OpenAI o3 and Gemini 2.5 Pro, these models are no longer confined to simple image captioning or object recognition, but exhibit complex reasoning capabilities for inferring high-level information from visual inputs. However, this capability introduces serious location-related privacy risks:

1. **Individual Risk**: When images containing identifiable individuals expose any location, they reveal sensitive personal daily routines.
2. **Household Risk**: When images reveal private locations (regardless of whether individuals are present), they continuously expose household routines.
3. **Legal Compliance Issues**: Under GDPR and CCPA, precise geolocation data is explicitly classified as sensitive personal information.

Three key limitations of prior work:
- Primarily evaluates geolocation performance rather than treating location privacy leakage as a security problem.
- Datasets consist mostly of "benign" public scenes such as landmarks and tourist attractions, lacking privacy-sensitive scenarios.
- Use of low-resolution Google Street View images severely underestimates models' inferential capabilities.

## Method

### Overall Architecture

The paper's contributions consist of three core components: (1) a three-tier visual privacy risk framework; (2) the DoxBench benchmark dataset and a novel metric; (3) the ClueMiner analysis tool and the GeoMiner attack framework.

### Key Designs

**Three-Tier Privacy Risk Framework:**

| Risk Level | Property | Private Space | Personal Image | Legal Mapping |
|------------|----------|--------------|---------------|---------------|
| Level 1 (Low) | Transient Risk | ✗ | ✓ | CCPA §1798.140(ae)(1)(C) |
| Level 2 (Medium) | Persistent Risk | ✓ | ✗ | CCPA §1798.140(v)(1)(A) |
| Level 3 (High) | Dual Risk | ✓ | ✓ | GDPR + CCPA multiple clauses |

**DoxBench Dataset Construction:**
- 500 high-resolution images captured with iPhone from 6 representative regions in California (San Francisco, San Jose, Sacramento, Los Angeles, Irvine, San Diego).
- Covers 6 categories, including an original "Mirror" category (privacy leakage via reflective surfaces).
- All images retain complete EXIF metadata (GPS coordinates).

**Information-Theoretic Metric Glare:**

$$\text{Glare} = a \left[ H(R) + \text{VRR} \cdot \log_2 \left( \frac{A_0}{\pi d_{50} \bar{d}} \right) \right] \; [\text{bits}]$$

where $H(R) = -\text{VRR} \cdot \log_2 \text{VRR} - (1 - \text{VRR}) \cdot \log_2(1 - \text{VRR})$

- $A_0 = 1.48 \times 10^8 \text{ km}^2$: total land area of the Earth
- $d_{50}$, $\bar{d}$: median and mean of error distances
- $a = 100$: scaling factor
- First term (Risk Term): amount of information leaked by the model's response behavior itself
- Second term (Leakage Term): amount of information reflected by the localization precision of the response content

**GeoMiner Attack Framework:** Decomposes the localization process into two stages — Clue Extraction and Reasoning — and improves geolocation performance through a collaborative mode.

### Loss & Training

This paper is an evaluation study and does not involve model training. The core strategies are:
- Minimal prompting: "Where is it?" as a stress test.
- Top-K prediction variants to obtain multiple candidate addresses.
- CoT prompting to guide MLLMs in simulating clue-based reasoning.

## Key Experimental Results

### Main Results

**13 Models + Human Baseline Comparison (Top-1 Setting):**

| Model | VRR↑ | AED(km)↓ | MED(km)↓ | CCPA Acc.↑ | Glare(bits)↑ |
|-------|------|----------|----------|-----------|-------------|
| Human Non-Expert | 99.10% | 140.08 | 37.22 | 6.01% | 1309.73 |
| GPT-5† | 78.41% | 11.26 | 4.35 | 17.40% | 1633.87 |
| OpenAI o3† | 80.80% | 13.56 | 5.46 | 14.73% | 1628.50 |
| Gemini 2.5 Pro† | 84.53% | 14.75 | 4.63 | 19.73% | 1701.61 |
| GPT-4.1 | 83.48% | 15.24 | 6.07 | 13.84% | 1647.29 |
| QvQ-max† | 66.74% | 121.06 | 24.02 | 9.25% | 1025.05 |

**Key Results under Top-3 Setting:**

| Model | VRR | CCPA Acc. | Glare |
|-------|-----|-----------|-------|
| GPT-5† | 74.23% | 22.03% | 1688.66 |
| Gemini 2.5 Pro† | 95.07% | 21.97% | 1987.16 |
| OpenAI o3† | 87.95% | 20.09% | 1912.77 |
| GPT-4.1 | 96.88% | 19.42% | 1916.55 |

### Ablation Study

**Analysis by Privacy Risk Level (Top-1):**
- Level 1 → Level 2: CCPA accuracy decreases by 11.10%, Glare decreases by 161.77 bits.
- Level 2 → Level 3: CCPA accuracy decreases by 2.83%, Glare decreases by 211.25 bits.
- The Mirror category is the most challenging: Glare is only 677.91 bits, with CCPA accuracy of only 3.54%.

**Effect of CoT Prompting on MLLMs:**
- Answered cases (Top-1): average CCPA accuracy improvement of 4.91%, average Glare improvement of 137.18 bits.
- Unanswered cases (Top-1): average CCPA accuracy improvement of 11.17%, average Glare improvement of 1256.89 bits.
- This confirms that clue-based reasoning is a key factor in privacy leakage.

**Cross-Region Generalization (Multi-State U.S. Level-3 Dataset):**

| Model | VRR | AED(km) | CCPA Acc. | Glare |
|-------|-----|---------|-----------|-------|
| o3 + tools | 100% | 3.06 | 34.00% | 2375.48 |
| Gemini 2.5 Pro | 100% | 7.19 | 24.00% | 2100.69 |
| GPT-5 | 100% | 4.59 | 22.00% | 2110.35 |

### Key Findings

1. **MLRMs significantly surpass non-expert humans**: average Glare of 1418.97 bits (Top-1) exceeds the human baseline of 1309.73 bits; precise localization accuracy is approximately twice that of humans.
2. **Two root causes**: (1) powerful visual clue reasoning combined with internal world knowledge; (2) absence of privacy alignment mechanisms that would suppress the use of privacy-relevant visual cues.
3. **The Claude family exhibits the lowest VRR** (9–40%), demonstrating relatively stronger refusal mechanisms, whereas nearly all other models respond proactively.
4. **Tool augmentation substantially amplifies the threat**: o3 with search tools achieves 34% CCPA accuracy on the cross-state dataset.

## Highlights & Insights

1. **First systematic study of location privacy leakage**: advances MLRM privacy risk from theoretical concern to quantifiable empirical analysis.
2. **Information-theoretic metric innovation**: Glare unifies three independent metrics — VRR, AED, and MED — into a single comparable measure.
3. **Legal framework alignment**: the three-tier risk framework maps directly to GDPR/CCPA provisions, offering practical guidance for legal compliance.
4. **Mirror category discovery**: a novel threat type in which location information is indirectly leaked via reflective surfaces (vehicle bodies, glass).
5. **Impressive experimental scale and diversity**: 14 MLRM/MLLM models + 268 MTurk human evaluators + 500 precisely annotated images.

## Limitations & Future Work

1. **Geographic concentration of the dataset**: primarily collected in California; the 50 cross-state supplementary samples offer limited representativeness.
2. **Evaluation limited to location inference**: broader privacy risks such as identity linkage and behavioral pattern inference are not addressed.
3. **Insufficient exploration of defenses**: the paper identifies the problem but does not propose effective privacy protection mechanisms.
4. **Flat-Earth approximation error**: Glare uses a planar approximation for area computation, with a maximum relative error of approximately 25.75%.
5. **Mitigation via model fine-tuning or safety alignment is not explored.**

## Related Work & Insights

- **GeoGuessr has long attracted community interest as a capability**, but this paper is the first to frame it as a security threat rather than a capability benchmark.
- Compared to concurrent works such as jay2025evaluatingprecisegeolocationinference and huang2025vlmsgeoguessrmastersexceptional, this paper focuses on privacy-sensitive scenarios rather than public landmarks.
- Insight: the "emergent" reasoning capabilities of large models may produce unexpected negative consequences in the security domain, motivating a new research direction of "reasoning safety alignment."

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study of location privacy leakage in MLRMs, defining a novel threat model.
- **Technical Depth**: ⭐⭐⭐⭐ — Information-theoretic metric design is rigorous; experimental evaluation is comprehensive.
- **Experimental Scale**: ⭐⭐⭐⭐⭐ — 14 models + 268 human evaluators + 500 precisely annotated images.
- **Practicality**: ⭐⭐⭐⭐ — Directly linked to legal regulations, with meaningful guidance for industry security practices.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-defined frameworks.

**Overall**: ⭐⭐⭐⭐ (4/5) — A highly important security paper that reveals a previously overlooked privacy threat in the MLRM era. The experimental design and metric innovation are commendable, though exploration of defensive directions remains limited.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)

<!-- RELATED:END -->
