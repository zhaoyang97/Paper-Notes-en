---
title: >-
  [Paper Note] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset
description: >-
  [NeurIPS 2025 (Dataset & Benchmark)][LLM Safety][guardrail benchmark] This paper introduces Poly-Guard, the first **large-scale, multi-domain, policy-grounded** safety guardrail benchmark. It extracts 400+ risk categorie…
tags:
  - "NeurIPS 2025 (Dataset & Benchmark)"
  - "LLM Safety"
  - "guardrail benchmark"
  - "policy-grounded"
  - "multi-domain safety"
  - "adversarial attack"
  - "over-refusal"
date: 2026-05-08
content_hash: 63a83bc77d197854
---

# Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset

**Conference**: NeurIPS 2025 (Dataset & Benchmark)
**arXiv**: [2506.19054](https://arxiv.org/abs/2506.19054)
**Code**: [github.com/AI-secure/PolyGuard](https://github.com/AI-secure/PolyGuard)
**Data**: [huggingface.co/datasets/AI-Secure/PolyGuard](https://huggingface.co/datasets/AI-Secure/PolyGuard)
**Authors**: Mintong Kang, Zhaorun Chen, Chejian Xu, Jiawei Zhang, Chengquan Guo, Minzhou Pan, Ivan Revilla, Yu Sun, Bo Li
**Affiliations**: UIUC, UChicago, CSU, Virtue AI
**Area**: AI Safety / Content Safety / Guardrail Evaluation
**Keywords**: guardrail benchmark, policy-grounded, multi-domain safety, adversarial attack, over-refusal

## TL;DR

This paper introduces Poly-Guard, the first **large-scale, multi-domain, policy-grounded** safety guardrail benchmark. It extracts 400+ risk categories and 1,000+ safety rules from 150+ real-world industry safety policies, generates 100K+ instances spanning 8 safety-critical domains, and systematically evaluates 19 guardrail models, revealing 8 key findings including domain specialization, evolutionary forgetting, scaling stagnation, and adversarial vulnerability.

## Background & Motivation

- **Background**: The widespread deployment of LLMs in high-stakes domains such as finance, law, and healthcare has driven the development of numerous guardrail models (LlamaGuard series, ShieldGemma, WildGuard, Granite Guardian, etc.) and evaluation benchmarks (ToxicChat, HarmBench, SALAD-Bench, etc.). However, existing benchmarks suffer from serious deficiencies in systematicity and real-world alignment.

- **Limitations of Prior Work**:
    1. **Ad hoc risk taxonomies**: Existing benchmarks rely on risk classification schemes designed independently by individual organizations, lacking principled alignment with standardized safety policies (government regulations, platform codes of conduct, industry ethical standards).
    2. **Neglect of domain specificity**: The same risk category (e.g., privacy leakage) carries substantially different meanings across social media, human resources, and financial domains. Existing benchmarks are predominantly general-domain and cannot capture inter-domain variation.
    3. **Scarcity of benign data**: High-quality "hard safe" samples—content that superficially involves sensitive topics but is actually compliant—are lacking, making it impossible to detect over-refusal in guardrail models.
    4. **Insufficient adversarial coverage**: Existing attack-augmented datasets (e.g., JailbreakBench) primarily test jailbreak vulnerabilities in LLMs themselves, rather than the adversarial robustness of guardrail models specifically.

- **Key Challenge**: Guardrail models can only be evaluated in a manner reflective of real deployment conditions when assessed under authentic, cross-domain safety policy frameworks. Yet no such unified benchmark currently exists—one that simultaneously covers a sufficient breadth of domains and policies while maintaining sufficient granularity (at the rule level rather than the category level) to precisely localize model failure points.

- **Goal**: To construct the first large-scale, multi-domain guardrail evaluation benchmark aligned with real-world industry safety policies, and to conduct systematic benchmarking of 19 state-of-the-art guardrail models, revealing their strengths and blind spots.

- **Key Insight**: The pipeline begins from real safety policy documents and proceeds through automated extraction and structuring (policy crawling → two-level risk category/safety rule extraction → rule-conditioned data generation → detoxification pairing → interaction format augmentation → adversarial attack augmentation), forming an end-to-end data construction pipeline.

- **Core Idea**: Using 150+ real-world safety policies as the foundation, Poly-Guard constructs the first policy-grounded cross-domain guardrail benchmark through two-level risk hierarchy extraction, rule-conditioned safe/unsafe paired generation, and adversarial augmentation.

## Method

### Overall Architecture

Poly-Guard's construction follows a **two-stage pipeline** (Figure 2):
1. **Policy → Structured Risk Extraction**: Automated crawling of 150+ official safety policy documents across 8 domains → GPT-4o-based extraction of a two-level hierarchy: high-level risk categories (400+) and fine-grained safety rules (1,000+).
2. **Structured Risk → Dataset Generation**: Rule-conditioned unsafe sample generation using non-safety-aligned/uncensored LLMs → detoxification prompting to generate paired safe samples → interaction format diversification (declarative/instructional/multi-turn dialogue) → adversarial attack augmentation.

The final dataset covers 8 domains, 100K+ instances, 400+ risk categories, and 1,000+ safety rules.

### Key Designs

1. **Policy-Grounded Two-Level Risk Hierarchy Extraction**

    Function: Automatically extracts a structured risk taxonomy from raw safety policy documents.
    Mechanism: A two-stage prompting framework is designed—in the first stage, an LLM acts as a policy analyst to extract atomic, actionable behavioral restriction rules from raw platform safety documents (e.g., "Do not publish or distribute child sexual abuse material"); in the second stage, the extracted rules are deduplicated, semantically clustered, and abstracted to produce a two-level hierarchy of "risk categories → safety rules." Prior to this, a safety policy crawling agent handles the heterogeneous formats (PDF/HTML/Markdown), dispersed sources, and inconsistent structures of policy documents.
    Design Motivation: Existing benchmarks operate only at the category level (e.g., "hate speech"), which is too coarse to precisely localize which specific rule a model fails on. The two-level hierarchy enables rule-level fine-grained evaluation, facilitating targeted improvement.

2. **Detoxification Prompting for Paired Sample Generation**

    Function: Generates high-quality paired safe samples for each unsafe sample to detect over-refusal in guardrail models.
    Mechanism: An asymmetric prompting strategy is employed—the unsafe generation prompt elicits content that explicitly violates a rule (spanning a spectrum from overt to covert violations), while the detoxification prompt reverses the intent through minimal editing, preserving sensitive context and semantics while rendering the content compliant. The generation of disclaimers or excessively sanitized versions is explicitly prohibited, ensuring that "hard safe" samples remain linguistically and semantically challenging.
    Design Motivation: Existing benchmarks lack challenging safe samples (XSTest/OKTest attempt this but rely on manual annotation at scales of only a few hundred instances), making over-refusal detection infeasible. The paired design ensures topical balance between safe and unsafe samples, preventing classifiers from exploiting topic-level shortcuts.

3. **Multi-Strategy Adversarial Attack Augmentation**

    Function: Evaluates the robustness of guardrail models under adversarial conditions.
    Mechanism: Three attack strategies targeting common guardrail vulnerabilities are first designed: ① Risk Category Shifting (fabricating category changes to mislead the model); ② Reasoning Distraction (inserting irrelevant reasoning tasks to divert attention); ③ Instruction Hijacking (exploiting the model's instruction-following tendency to directly manipulate outputs). These strategies serve as seeds, and the PAIR and AutoDAN adversarial prompt optimization algorithms iteratively refine adversarial suffixes.
    Design Motivation: Existing attack-augmented datasets (e.g., JailbreakBench) target LLM jailbreaking rather than guardrail model testing. The attacks in this work are specifically designed around the decision boundaries of guardrail models, making them more representative of real deployment threats.

### Loss & Training

Poly-Guard is an **evaluation benchmark** rather than a training method. Data generation uses non-safety-aligned LLMs for rule-conditioned generation, with GPT-4o assisting in risk hierarchy extraction. The evaluation protocol adopts three metrics—F1, Recall, and FPR—rather than continuous score metrics such as AUPRC, since commercial APIs (e.g., Azure Content Safety, Bedrock Guardrail) do not expose confidence scores.

## Key Experimental Results

### Main Results

F1/Recall evaluation of 19 guardrail models across 8 domains (Table 1, F1 values ×100):

| Model | Social Media (Msg/Comm/Stream) | General Reg (EU/GDPR) | HR (Svc/Cust) | Finance | Law | Education | Code | Cyber |
|------|------|------|------|------|------|------|------|------|
| LlamaGuard 1 | 33.1/38.4/32.7 | 13.0/16.1 | 25.6/17.3 | 23.7 | 11.8 | 15.2 | 28.3 | 61.9 |
| LlamaGuard 2 | 49.7/60.9/55.6 | 47.8/64.4 | 52.5/52.1 | 64.6 | 62.2 | 44.7 | 51.0 | 88.0 |
| LlamaGuard 3 (1B) | 46.7/47.2/46.5 | 50.4/50.9 | 48.2/47.2 | 46.9 | 48.1 | 46.0 | 50.0 | 51.8 |
| LlamaGuard 3 (8B) | 61.2/63.3/63.5 | 37.0/32.7 | 27.4/26.8 | 49.6 | 44.2 | 28.6 | 13.8 | 81.6 |
| **MDJudge 2** | **73.7/75.3/75.9** | **64.0/81.7** | **80.4/75.6** | 76.9 | 65.6 | **77.9** | 56.5 | **89.1** |
| **WildGuard** | 76.0/74.3/76.0 | 56.6/66.4 | 77.0/71.7 | **86.5** | **76.4** | 69.4 | 55.0 | 80.2 |
| **Granite Guardian (3B)** | 71.1/70.5/71.9 | 67.9/78.2 | 80.1/78.7 | **90.4** | **80.2** | **80.0** | **63.8** | 85.0 |
| Granite Guardian (5B) | 69.5/70.3/67.4 | 63.3/80.3 | 84.6/81.6 | 85.0 | 66.8 | 75.8 | 64.0 | 87.7 |
| ShieldGemma (2B) | 4.8/5.5/4.5 | 0.0/0.0 | 8.8/4.4 | 0.0 | 0.0 | 2.2 | 16.5 | 26.8 |
| Azure Content Safety | 20.2/16.6/20.7 | 2.5/0.5 | 4.4/0.8 | 0.0 | 0.6 | 3.3 | 0.3 | 3.3 |

### Ablation Study

**Adversarial Attack Success Rate (ASR)**—attack success rates of the 5 strongest guardrail models across 8 domains (Table 2):

| Model | Social Media | General Reg | HR | Finance | Law | Education | Code | Cyber | Mean |
|------|------|------|------|------|------|------|------|------|------|
| Aegis Defensive | 0.759/0.717/0.767 | 0.559/0.884 | 0.689/0.420 | 0.555 | 0.892 | 0.435 | 0.768 | **0.677** |
| Granite Guardian (5B) | 0.989/0.992/0.994 | 0.674/0.966 | 0.993/0.842 | 0.863 | 0.997 | 0.990 | 0.912 | **0.928** |
| MDJudge 2 | 0.754/0.792/0.729 | 0.641/0.919 | 0.964/0.588 | 0.529 | 0.871 | 0.970 | 0.776 | **0.776** |
| **WildGuard** | **0.183/0.103/0.235** | **0.315/0.356** | **0.347/0.036** | **0.038** | **0.268** | **0.213** | **0.080** | **0.198** |
| LLM Guard | 0.470/0.452/0.608 | 0.781/0.991 | 0.864/0.332 | 0.388 | 0.854 | 0.990 | 0.368 | **0.645** |

**Model Scaling Comparison** (mean F1):

| Comparison | Smaller Model F1 | Larger Model F1 | Finding |
|------|----------|----------|------|
| LlamaGuard 3 (1B) vs. (8B) | **0.485** | 0.423 | Smaller model superior |
| Granite Guardian (3B) vs. (5B) | **0.774** | 0.749 | Smaller model superior |

**LlamaGuard Series Evolution** (mean F1 across 23 risk categories in the Instagram domain):

| Version | Mean F1 | Cybersecurity | Misinformation | Hate Speech |
|------|---------|--------------|----------------|------------|
| LlamaGuard 1 | 0.294 | 0.472 | 0.045 | — |
| LlamaGuard 4 | **0.605** | **0.797** | **0.692** | 0.734 (↓ vs. v3's 0.777) |

### Key Findings

1. **Domain Specialization** (Finding 1): Guardrail models exhibit clear domain specialization—Granite Guardian excels in formal-register domains (HR/Finance/Education), LLM Guard leads on social media, and performance trends are consistent within sub-domains.
2. **Evolutionary Forgetting** (Finding 2): As the LlamaGuard series expands its risk category coverage from v1 to v4 (mean F1: 0.294→0.605), performance on common categories is not guaranteed to improve—Hate Speech actually declines in v4.
3. **Scaling Stagnation** (Finding 3): Smaller models do not necessarily underperform larger ones—LlamaGuard 3 (1B) achieves a higher mean F1 than (8B), and Granite Guardian (3B) outperforms (5B).
4. **Context Gains** (Finding 4): Dialogue formats are more reliably evaluated than single-turn instructions or declarative statements (12–13 of 14 effective models achieve higher F1 on dialogue, with an average improvement >5%).
5. **Adversarial Vulnerability** (Finding 5): All models are highly vulnerable to optimized adversarial attacks—Granite Guardian (5B) achieves a mean ASR of 92.8%, and even the best-performing WildGuard reaches 19.8%.
6. **Severity-Skewed Robustness** (Finding 6): High-severity risk categories (e.g., EU AI Act prohibited AI practices) exhibit significantly greater adversarial robustness than low-severity categories.
7. **Category-Skewed Moderation** (Finding 7): The standard deviation of F1 across risk categories is consistently >10% (e.g., in the Instagram domain, mean F1 for Hate Speech = 0.715 vs. Identity Misrepresentation = 0.273).
8. **Conservative Bias** (Finding 8): Models systematically achieve high precision but low recall (mean precision 0.701 vs. recall 0.479 in the Social Media domain), tending toward missed detections rather than false positives.

## Highlights & Insights

1. **Policy-Grounded Data Construction Paradigm**: The paper pioneers an end-to-end pipeline from real-world safety policy documents to structured evaluation data (policy crawling agent → two-stage prompting extraction → rule-conditioned generation). This not only produces a dataset but also provides a **generalizable framework** extensible to new domains and policies.

2. **"Risk Forgetting During Model Evolution" Phenomenon**: As the LlamaGuard series expands its coverage, performance on common risk categories can degrade—a phenomenon analogous to catastrophic forgetting in continual learning, carrying important implications for the iterative development of safety models.

3. **Scale Does Not Equal Capability**: The finding that LlamaGuard 3 (1B) outperforms (8B) directly challenges the assumption that "larger models are always safer," suggesting that data quality and training strategy may matter more than model scale for guardrail models.

4. **Methodological Value of Detoxification Pairing**: Generating "hard safe" samples by reversing intent through minimal editing is more scalable than XSTest's manual construction, and enables systematic detection of over-refusal—a critical issue for user experience in real deployments.

5. **WildGuard's Defensive Advantage**: Among all evaluated models, WildGuard achieves a mean ASR of 19.8%, far outperforming the second-best Aegis Defensive at 67.7%. The robustness sources within WildGuard's training strategy merit further investigation.

## Limitations & Future Work

1. **Cultural/Geographic Bias**: Safety policies are drawn primarily from Western institutions and global platforms, lacking regulations and cultural norms from non-Western regions (e.g., content safety standards in China and the Middle East differ substantially).
2. **Generative Model Bias**: Using LLMs to generate data may introduce model-specific linguistic pattern biases; despite using uncensored models, the naturalness and diversity of generated samples remain constrained by the capabilities of the generating model.
3. **Text Modality Only**: Multimodal safety (safety issues in images, video, and audio) is not addressed, despite the fact that risks in social media and content creation are frequently multimodal.
4. **Static Policies**: Safety policies are subject to change over time (e.g., the ongoing revision of the EU AI Act); the dataset requires periodic updates to reflect policy evolution.
5. **Evaluation Metric Limitations**: Some commercial APIs do not expose confidence scores, precluding more refined evaluation (e.g., AUPRC); F1 evaluation based on discrete verdicts may obscure differences in model behavior near decision boundaries.

## Related Work & Insights

| Benchmark/Dataset | Scale | Domain Coverage | Policy-Grounded | Benign Data | Attack Augmentation |
|------------|------|---------|---------|---------|---------|
| HarmBench / AdvBench | Medium | General | ✗ | ✗ | ✗ |
| ToxicChat | Small | General Chat | ✗ | Limited | ✗ |
| XSTest / OKTest | ≤Hundreds | General | ✗ | ✓ (Manual) | ✗ |
| AIRBench | Medium | Regulation | Partial | ✗ | ✗ |
| CyberSecEval | Medium | Cybersecurity | Partial | ✗ | ✗ |
| GuardBench | Medium | Aggregated | ✗ | Inherited | ✗ |
| SALAD-Bench | Large | General | ✗ | ✗ | ✓ |
| **Poly-Guard** | **100K+** | **8 Domains** | **✓ (150+ policies)** | **✓ (Detoxification)** | **✓ (PAIR+AutoDAN)** |

**Directions for Future Work**:
- The policy-grounded data construction pipeline can be directly transferred to other safety contexts (e.g., autonomous driving safety policies, financial compliance auditing).
- The "risk forgetting" phenomenon suggests that safety model training should incorporate continual learning strategies to maintain performance on common categories while expanding coverage.
- WildGuard's adversarial robustness advantage warrants further investigation and may serve as a reference paradigm for adversarial training of guardrail models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first multi-domain guardrail benchmark combining real-world safety policies with large-scale data generation; the pipeline of policy crawling agent + two-level risk extraction demonstrates strong methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 19 models × 8 domains; all 8 systematic findings are supported by data, with adversarial evaluation and evolutionary analysis providing rich insights.
- Technical Depth: ⭐⭐⭐⭐ The data construction pipeline is complete and well-designed (policy crawling → rule extraction → conditioned generation → detoxification → adversarial augmentation), though the contribution is fundamentally an engineering pipeline rather than a novel algorithm.
- Practical Value: ⭐⭐⭐⭐⭐ Directly applicable to guardrail development and model selection—findings on domain specialization, scaling stagnation, and evolutionary forgetting have direct implications for industrial guardrail deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)
- [\[NeurIPS 2025\] ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)
- [\[ICCV 2025\] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](../../ICCV2025/llm_safety/asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)
- [\[ICLR 2026\] VeriTrail: Closed-Domain Hallucination Detection with Traceability](../../ICLR2026/llm_safety/veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)

</div>

<!-- RELATED:END -->
