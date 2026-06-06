---
title: >-
  [Paper Note] LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control
description: >-
  [ACL2026][LLM/NLP][Localization evaluation] LoCar proposes 13 deployment-level KPIs for Korean in-vehicle assistants and evaluates 11 models using human-calibrated LLM-as-a-Judge combined with honorific morphological ver…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "Localization evaluation"
  - "Korean honorifics"
  - "In-vehicle assistant"
  - "LLM-as-a-Judge"
  - "Multi-turn dialogue"
date: 2026-05-08
content_hash: 839e983ef9671293
---

# LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control

**Conference**: ACL2026  
**arXiv**: [2605.21086](https://arxiv.org/abs/2605.21086)  
**Code**: No public code; data contains proprietary materials from industry partners, the paper states it cannot be disclosed  
**Area**: LLM Evaluation / Localization / In-vehicle Assistants  
**Keywords**: Localization evaluation, Korean honorifics, In-vehicle assistant, LLM-as-a-Judge, Multi-turn dialogue  

## TL;DR
LoCar proposes 13 deployment-level KPIs for Korean in-vehicle assistants and evaluates 11 models using human-calibrated LLM-as-a-Judge combined with honorific morphological verification. It finds that while general understanding is near saturation, fine-grained honorific control and multi-turn strategic guidance remain significantly unstable.

## Background & Motivation
**Background**: In-vehicle assistants are evolving from fixed-command systems into LLM applications capable of interpreting vehicle manuals, understanding navigation needs, and managing multi-turn dialogues. Existing evaluations primarily focus on general knowledge, reasoning, or English interaction quality, whereas commercial deployment requirements for local markets are more granular. For instance, different honorific levels in Korean directly impact the user's perception of politeness, trust, and professionalism.

**Limitations of Prior Work**: Conventional LLM benchmarks struggle to cover two types of requirements in vehicle scenarios: functional correctness related to vehicle operations and navigation, and sociolinguistic norms for specific language markets. Even if a model answers correctly, it may fail deployment standards regarding honorific levels, conciseness, timing of clarification, or proactive suggestions.

**Key Challenge**: Deployment-level evaluation requires a level of detail that encompasses linguistic-cultural norms and in-car interaction flows. However, human evaluation is costly, and standard LLM judges often confuse similar Korean honorific forms. The authors address how to make evaluation both functional and capable of reliably checking localized linguistic styles while remaining automatable.

**Goal**: To build a Korean in-vehicle assistant evaluation framework covering two core use cases: Car Expert and Navigation; define linguistic style and dialogue capability KPIs; synthesize and augment test data; calibrate the evaluator with human annotations; and analyze the performance gaps of different models relative to local deployment requirements.

**Key Insight**: Instead of treating "Korean capability" as a single aggregate score, the paper decomposes it into actionable KPIs: Conciseness, Hae/Haeyo/Hapsyo honorifics, implicit understanding, context understanding, harmful question response, clarification, retention, refinement, reflection, proactive suggestions, and troubleshooting.

**Core Idea**: Generate in-vehicle assistant data using an industry-specific scene taxonomy, then utilize a hybrid evaluation pipeline consisting of "LLM judge majority voting + Korean sentence-final morphological checking" to turn localized linguistic norms into quantifiable deployment metrics.

## Method
LoCar contributes a complete evaluation system rather than a single model. It defines a task taxonomy for in-vehicle assistants, constructs single-turn and multi-turn samples based on vehicle/navigation manuals and real in-car dialogues, and selects appropriate automated evaluation methods for each KPI.

### Overall Architecture
The framework consists of three steps. Step one is data taxonomy: Car Expert covers vehicle knowledge, operations, and diagnostics derived from owner's manual hierarchies; Navigation covers destination search, route explanation, traffic inquiries, and contextual recommendations. Step two is data construction: Single-turn Q&A is synthesized from manual and navigation taxonomies; multi-turn dialogues are expanded from single-turn seeds into interaction flows requiring state tracking, with three-fold augmentation based on Korean honorifics. Step three is evaluation: Human-calibrated LLM-as-a-Judge with multi-model majority voting is used for general KPIs, while morphological verification of sentence endings is added for honorific KPIs to compensate for the LLM judge's inability to distinguish similar honorific levels.

### Key Designs
1. **Two-tier 13-KPI Evaluation System**:
	- **Function**: Decomposes in-vehicle assistant capabilities into Linguistic Style and Dialogue Capability tiers.
	- **Mechanism**: The Linguistic Style tier includes Conciseness, Hae, Haeyo, and Hapsyo; the Dialogue Capability tier includes single-turn (Implicit Understanding, Context Understanding, Harmful Question Response) and multi-turn (Clarification, Retention, Refinement, Reflection, Proactive Suggestion, Troubleshooting).
	- **Design Motivation**: Failures in in-vehicle assistants are not just wrong answers, but also excessive length, incorrect politeness levels, failure to refuse dangerous requests, or multi-turn state management errors.

2. **Industry Taxonomy Driven Data Construction**:
	- **Function**: Ensures the test set covers real-world vehicle functions rather than generalized chat samples.
	- **Mechanism**: Car Expert parses owner's manuals into 109 categories and 4,395 sub-categories; Navigation is derived from navigation manuals and real dialogues into 7 major categories and 28 sub-categories. Single-turn QA is deduplicated, filtered, and mapped to KPIs; multi-turn data is expanded from seeds into 3-5 round dialogues.
	- **Design Motivation**: Localized evaluation must align with actual product functions; otherwise, high benchmark scores do not guarantee in-car usability.

3. **Hybrid Honorific Evaluation**:
	- **Function**: Improves the reliability of distinguishing fine-grained Korean speech styles such as Hae, Haeyo, and Hapsyo.
	- **Mechanism**: The LLM judge handles contextual semantic judgment, while lightweight morphological checks handle sentence-final honorific suffix matching. The paper reports that human-evaluator consistency for honorific classification improved from 0.69 to 0.94, a **Gain** of 24 percentage points.
	- **Design Motivation**: Korean honorific levels are often marked morphologically; adjacent levels are easily confused by LLM-only judges. Rule-based checks serve as high-precision error filters.

### Loss & Training
This work does not train the evaluated models. Evaluator selection is based on 803 human-annotated calibration samples, each independently labeled by 3 annotators. Candidate judge models were selected based on cross-KPI consistency and overall agreement; the final pipeline uses majority voting between DeepSeek-v3.1, Gemini-2.5-Flash, and GPT-5-mini. For dialogue capability KPIs, 50 test instances were randomly sampled. For multi-turn samples, one target turn was chosen for evaluation, and hae, haeyo, or hapsyo was randomly assigned as the target honorific style.

## Key Experimental Results

### Main Results
| Experiment Item | Setting | Key Data | Conclusion |
|-------|------|---------|------|
| Human Calibration Set | 13 KPIs, single/multi-turn | 803 human-labeled samples, 3 annotators/sample | Provides basis for LLM-as-a-Judge selection and calibration |
| Honorific Hybrid Eval | LLM-only vs. LLM + Morphological | Human-judge consistency 0.69 → 0.94, +24pp | Morphological checks significantly improve fine-grained honorific judgment |
| Single-turn Overall Avg | 11 models | Navigation: Implicit 0.92, Context 0.94, Harmful 0.85; Car Expert: Implicit 0.96, Harmful 0.93 | Single-turn understanding metrics are near saturation |
| Multi-turn Overall Avg | 11 models | Navigation: Clarification 0.58, Proactive 0.78, Retention 0.88; Car Expert: Clarification 0.51, Troubleshooting 0.95 | Strategic clarification is hardest; retention and consistency are more stable |
| Evaluated Model Scale | Evaluated models | 11 models total, including 6 Korean local models and global API models | Framework effectively compares deployment gaps between local and global models |

### Ablation Study
| Analysis Item | Configuration | Key Data | Note |
|------|------|---------|------|
| Honorific Judge Improv. | Gemini-2.5-Flash | Hae +0.06, Haeyo +0.11, Hapsyo +0.19 | Morphological verification helps more for formal hapsyo |
| Honorific Judge Improv. | GPT-5-mini | Hae +0.08, Haeyo +0.18, Hapsyo +0.52 | GPT-5-mini had the worst LLM-only confusion on hapsyo, benefiting most |
| Honorific Judge Improv. | DeepSeek-v3.1 | Hae +0.03, Haeyo +0.08, Hapsyo +0.09 | All three judges showed consistent gains |
| Multi-turn Rep. Model | gpt-5.1 | Nav Clarification 0.84, Proactive 1.00; Car Clarification 0.92 | Leading models are stronger on multi-turn strategic KPIs |
| Multi-turn Low Score | kanana-1.5-15.7b-a3b | Nav Clarification 0.28, Proactive 0.50; Car Clarification 0.22 | Clarification and proactive intervention remain unstable even for local models |
| Inference Latency | 11 models | solar-pro 91.16s, gpt-5.1 50.4s, Qwen3 32.88s | Higher latency does not necessarily correlate with better multi-turn strategy |

### Key Findings
- Single-turn understanding metrics are already very high, suggesting that "knowing the answer" is not the primary bottleneck for in-vehicle Q&A.
- Fine-grained honorific control remains unstable, especially confusion between adjacent politeness levels like haeyo and hapsyo.
- In multi-turn dialogues, Clarification and Proactive KPIs are noticeably more difficult than Retention, Refinement, or Reflection, as they require the model to judge *when* to intervene rather than just continuing the context.
- The evaluator itself needs localization: LLM judges are not naturally reliable for Korean honorifics and must be combined with linguistically-aware morphology.

## Highlights & Insights
- LoCar decomposes "localization" from vague multilingual capability into specific, actionable sociolinguistic metrics, which is closer to real deployment than simply measuring Korean Q&A accuracy.
- The hybrid evaluation design is practical: LLM judges excel at contextual understanding while rule-based morphological checks catch honorific suffixes; the combination is more robust than either alone.
- The paper clearly demonstrates the two tiers of assistant capability: understanding vehicle and navigation knowledge is fundamental, but the true difficulty lies in making correct clarifications, proactive suggestions, and handling safety boundaries in multi-turn contexts.
- Insights are relevant for other language markets. Even if not Korean, many languages have local politeness, dialects, honorifics, registers, or cultural conventions that require specialized evaluation components.

## Limitations & Future Work
- LoCar was developed and validated only for the Korean language and market; honorific detection relies on Korean-specific morphology and cannot be directly migrated to languages where politeness is encoded via vocabulary, word order, or context.
- The data contains proprietary material from industry partners and cannot be released publicly, which limits reproducibility and community expansion.
- The evaluation is conducted in an offline text-based setting, failing to cover real-world in-car ASR errors, TTS presentation, acoustic noise, multimodal screen information, and real-time driving states.
- The paper intentionally excludes OEM-specific metrics to maintain generality, but real deployment requires context involving dynamic weather, location, user history, and tool-calling.
- Future work could extend to cross-lingual LoCar, closed-loop in-car testing, end-to-end voice evaluation, and dynamic evaluation combined with RAG and tool-use.

## Related Work & Insights
- **vs. MT-Bench / Arena-Hard**: General dialogue evaluations focus on overall response quality, whereas LoCar focuses on local linguistic style and task continuity in vehicle deployment.
- **vs. Korean-specific benchmarks**: General Korean benchmarks measure linguistic proficiency, but LoCar further mandates honorific levels, vehicle functionality, and multi-turn management.
- **vs. Pure LLM-as-a-Judge**: Pure LLM judges are unreliable for fine-grained Korean honorifics; LoCar adds morphological verification as a high-precision constraint.
- **vs. Automotive QA Datasets**: Standard automotive QA often only tests knowledge correctness; LoCar simultaneously tests safety responses, clarification, proactivity, and sociolinguistic adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Decoupling in-vehicle assistant localization into 13 KPIs and adding morphological verification is highly applicable.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes 803 calibration samples, 11 models, and single/multi-turn evaluation, though data is not public and only covers Korean.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure; taxonomy, evaluator, and deployment implications are well-linked.
- Value: ⭐⭐⭐⭐☆ Provides direct insights for multilingual assistants, localization evaluation, and enterprise-level LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](../../NeurIPS2025/llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)
- [\[ACL 2026\] Unlocking the Potential of Diffusion Language Models through Template Infilling](unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)
- [\[ACL 2026\] Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing](overcoming_copyright_barriers_in_corpus_distribution_through_non-reversible_hash.md)

</div>

<!-- RELATED:END -->
