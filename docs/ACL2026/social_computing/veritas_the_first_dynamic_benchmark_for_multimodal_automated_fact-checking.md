---
title: >-
  [Paper Note] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking
description: >-
  [ACL2026][Social Computing][Multimodal Fact-Checking] VeriTaS utilizes a quarterly-updated seven-stage automated pipeline to transform real-world multilingual text, image…
tags:
  - "ACL2026"
  - "Social Computing"
  - "Multimodal Fact-Checking"
  - "Dynamic Benchmark"
  - "Data Leakage"
  - "ClaimReview"
  - "Fact-Checking Evaluation"
date: 2026-05-08
content_hash: 8f8f48c1c6e4e978
---

# VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking

**Conference**: ACL2026  
**arXiv**: [2601.08611](https://arxiv.org/abs/2601.08611)  
**Code**: https://veritas.mai.informatik.tu-darmstadt.de  
**Area**: audio_speech  
**Keywords**: Multimodal Fact-Checking, Dynamic Benchmark, Data Leakage, ClaimReview, Fact-Checking Evaluation

## TL;DR
VeriTaS utilizes a quarterly-updated seven-stage automated pipeline to transform real-world multilingual text, image, and video claims from professional fact-checking organizations into a standardized, interpretable, and evaluable multimodal fact-checking benchmark, demonstrating that current state-of-the-art multimodal models remain significantly far from reliable AFC.

## Background & Motivation
**Background**: Automated fact-checking has expanded from pure text claim verification to images, videos, social media posts, and cross-lingual dissemination scenarios. Real-world misinformation is rarely a single sentence; it is a "claim package" composed of text, images, videos, publication dates, original sources, and context. Consequently, evaluation systems must cover multimodal evidence, real dissemination paths, and professional fact-checking verdicts.

**Limitations of Prior Work**: Most existing AFC benchmarks are static. Once model pre-training corpora cover these public claims and verdicts, the test set may degenerate into a memorization task. Another issue is coarse labeling: many datasets provide only single labels like true/false/NEI, failing to distinguish between different error sources such as manipulated images, miscontextualized images, false text statements, or missing critical context.

**Key Challenge**: Fact-checking evaluation requires recent, real, and interpretable data, but manual construction is extremely costly. Relying solely on old data leads to contamination by knowledge cutoff dates and data leakage; relying solely on synthetic data risks authenticity and ethical issues.

**Goal**: The authors aim to achieve four objectives: construct a dynamic benchmark that updates continuously; cover text, images, videos, and multilingual claims; unify heterogeneous fact-checking verdicts into fine-grained scores; and verify the consistency between automated labeling and human judgment while measuring real capabilities with the latest models.

**Key Insight**: The paper leverages ClaimReview, a structured entry point for real-world fact-checking, and utilizes LLMs for article extraction, original appearance retrieval, claim rewriting, verdict standardization, and "intact" claim rectification. This preserves the evidence base of professional fact-checkers while scaling to quarterly updates.

**Core Idea**: Use an automated pipeline to transform the real-world fact-checking ecosystem into a continuously updating, fine-grained, interpretable multimodal AFC evaluation benchmark that is robust against knowledge leakage.

## Method
VeriTaS is not a new fact-checking model but rather a data and annotation framework for model evaluation. The core contribution lies in decomposing chaotic real-world fact-checking materials into automatically processable stages: identifying expert claims from ClaimReview, recovering original dissemination content and media, rewriting claims into self-contained forms, and mapping various agency verdicts to a unified continuous score.

### Overall Architecture
The input consists of public ClaimReview structured records and fact-checking articles, including claim text, ratings, review URLs, publication dates, languages, and some appearance URLs. The pipeline first filters trustworthy publishers and crawls original text, then completes original social media appearances and media files. Subsequently, multimodal LLMs normalize the raw claim into a concise, self-contained statement that does not leak the verdict while retaining necessary media.

The output is the VeriTaS benchmark organized by quarter. Each sample includes the claim, media, date, language, appearance information, overall Integrity score, underlying attribute scores, and a text justification. The current version releases 25K claims covering 25 quarters from Q1 2020 to Q1 2026, with 1K claims per quarter, maintaining a balance between Intact and Compromised samples.

### Key Designs
1. **Seven-Stage Dynamic Construction Pipeline**:

	- **Function**: Converts continuously emerging new fact-checking reviews into evaluable samples, with a commitment to ongoing quarterly expansions.
	- **Mechanism**: Stage 1 collects ~398K reviews from ClaimReview; Stage 2 identifies 848 publishers and retains professional fact-checking organizations, resulting in 335K trusted reviews; Stage 3 crawls article bodies and media, removing noise like ads and cookie prompts; Stage 4 recovers original appearance URLs and archived URLs from articles; Stage 5 filters irrelevant media and rewrites claims into ~72K self-contained statements; Stage 6 performs verdict standardization; Stage 7 generates and verifies "Intact" versions to balance the data.
	- **Design Motivation**: Dynamicity is not merely appending new files but requires stable extraction of usable samples from real fact-checking production chains. The seven-stage design integrates credibility, context, media, and labels into a unified pipeline, reducing manual maintenance costs.

2. **Decoupled Verdict and Integrity Scoring**:

	- **Function**: Avoids collapsing all errors into a coarse true/false label, allowing evaluations to identify whether models err on media, text, or context.
	- **Mechanism**: The paper decomposes judgments into four underlying attributes: Media Authenticity, Media Contextualization, Veracity, and Context Coverage, plus an overall Integrity score representing whether the claim as a whole is acceptable. Each attribute is assigned a continuous score in $[-1, 1]$, where values below $-1/3$ are Negative and above $1/3$ are Positive. Integrity is determined by the worst "compromising property" among attributes (2)-(4). MSE is used as the primary evaluation metric because it strongly penalizes True/False flips while allowing for approximate correctness.
	- **Design Motivation**: Real-world errors are rarely a binary choice. An image might be real but used in the wrong context; text might be basically correct but omit critical context. Decoupling attributes makes the benchmark more reflective of professional fact-checking and provides greater interpretability for error analysis.

3. **LLM Ensemble Labeling, Filtering, and Rectification**:

	- **Function**: Achieves high-consistency verdicts and a balanced Intact/Compromised distribution without complete reliance on manual annotation.
	- **Mechanism**: Stage 6 uses an ensemble of GPT-5.2, Gemini 2.5 Pro, Claude Sonnet 4.5, and Llama 4 Maverick to score each attribute and generate justifications, which are then aggregated by mean; claims are discarded if disagreement among members exceeds 1. Since fact-checking naturally skews toward Compromised samples, Stage 7 uses the justification of the compromising property to generate evidence-grounded "Intact" rewrites, which are then re-verified for shareability, consistency, and Integrity.
	- **Design Motivation**: Professional fact-checking databases contain far more false claims than true ones; direct sampling would cause class imbalance. Unconditional synthesis of true claims would lack realism. Evidence-based rectification strikes a balance between authenticity and data balance.

### Loss & Training
This paper does not train new models. The construction side relies on the specialized calling of multimodal LLMs, ensemble aggregation, and strict filtering. The evaluation side maps model outputs to continuous scores for Integrity and underlying attributes. The primary metric is MSE, supplemented by MAE and 3-bin/7-bin accuracy. All baseline evaluations require the exclusion of evidence published after the claim date to prevent retrieval systems from "peeking" at future information.

## Key Experimental Results

### Main Results
The experiments are divided into three categories: data scale statistics, manual verification, and model baseline evaluation. The most significant conclusion is that the automated pipeline has high human consistency, but current AFC models and general-purpose multimodal models remain unreliable on the most recent quarters.

| Stage / Data Slice | Count or Metric | Description | Conclusion |
|--------|------|------|----------|
| ClaimReview discovery | ~398K reviews | Structured fact-checking records from 2016-01 to 2026-03 | The raw fact-checking ecosystem is sufficiently large |
| Trustworthy publisher filtering | 335K reviews | Retained professional orgs from 848 publishers; discarded ~64K | Source quality is controlled via publisher credibility |
| Article cleaning | 208K reviews | Extracted article body and media | Provides context for justification and claim rewriting |
| Appearance recovery | 94K reviews | Only 13.3% of ClaimReview has appearance; LLM finds extra URLs | Original source recovery is a key bottleneck |
| Claim normalization | 72K claims | Removed verdict leaks, filled media references, restricted length | Obtained self-contained statements |
| Verdict standardization | 36K claims | Filtered those with high ensemble disagreement | Retained high-consistency fine-grained labels |
| Final release | 25K claims | 25 quarters, 1K/quarter, Intact/Compromised balanced | Enables dynamic and longitudinal evaluation |
| Media & Language | 8,692 images, 5,334 videos, 54 languages | English 39.0%, Spanish 10.9%, Hindi 5.8% | Coverage significantly exceeds pure English/text benchmarks |

| Setting | MSE↓ | MAE↓ | 7-bin Acc↑ | 3-bin Acc↑ | Interpretation |
|--------|------|------|----------|------|------|
| VeriTaS Full Pipeline | 0.034 | 0.102 | 69.1 | 97.5 | High consistency with human Integrity judgment |
| Ensemble w/o filtering | 0.035 | 0.105 | 68.6 | 97.6 | Filtering contributes little but improves robustness |
| GPT-5.2 Single Model | 0.076 | 0.184 | 51.2 | 95.7 | Single model error is significantly higher |
| Gemini 3.1 Pro Single Model | 0.071 | 0.099 | 73.4 | 95.7 | High 7-bin but MSE still inferior to ensemble |
| Claude Sonnet 4 Single Model | 0.048 | 0.091 | 72.5 | 97.1 | Close to ensemble but inferior to full pipeline |
| Llama 4 Maverick Single | 0.042 | 0.103 | 66.7 | 97.1 | Open-source model usable, but ensemble is more stable |

On the latest Q1 2026 split, powerful models still show significant errors. Without retrieval, Claude Opus 4.6’s MSE is 0.453, the strongest among general models. With web search, Claude Opus 4.6 drops to 0.183, Gemini 3 Flash to 0.275, and Qwen 3.5 397B to 0.318. DEFAME using Claude Opus 4.6 backbone achieves 0.282, while Loki actually performs significantly worse across multiple backbones, indicating that "specialized systems" are not automatically superior to strong general multimodal models.

### Ablation Study

| Analysis Item | Key Metric | Description |
|------|---------|------|
| Knowledge Cutoff Impact | MSE increases from ~0.6 to >0.8 in longitudinal split | Models perform significantly worse after cutoff; static benchmarks overestimate capability |
| Latest Quarter + Retrieval | Claude Opus 4.6 MSE 0.183 | Retrieval helps, but remains far from the "acceptable" 0.1 threshold |
| Media Subsets | Higher MSE on video claims | Video fact-checking remains a weak point for most models |
| Integrity Subsets | Bias towards "Compromised" labels | Type bias increases errors on Intact samples |
| Rectified Claim Quality | ~5.1% samples have quality concerns | Evidence-grounded rewriting is generally credible but requires audit |

### Key Findings
- The necessity of dynamic benchmarks is empirically confirmed: the rise in MSE after the model's knowledge cutoff indicates that old fact-checking samples are likely contaminated by parametric memory.
- Human verification shows that the Full Pipeline Integrity MSE is only 0.034, indicating that LLM ensembles with filtering can map professional fact-checker verdicts quite reliably.
- Retrieval augmentation is not a silver bullet. Even with search tools, the strongest MSE (0.183) is still notable, and evaluations must restrict evidence sources by claim date.
- Videos, multilingual content, and Intact samples are the primary challenges; many models have an a priori bias towards "false claims," making them prone to labeling real or rectified claims as Compromised.

## Highlights & Insights
- The biggest highlight is treating the data leakage problem as a first-class citizen in benchmark design. Instead of just claiming "future updates," the authors provide a quarterly pipeline and longitudinal cutoff experiments, directly proving that static AFC evaluations will increasingly become distorted.
- The verdict design is highly reusable. Anchoring Integrity in media context, text veracity, and context coverage is better suited for multimodal information pollution than single binary labels and can easily transfer to news, medical, or scientific claim verification.
- The positioning of "rectification" is clever. It does not generate positive examples out of thin air but corrects false claims into shareable true claims based on evidence from fact-checking articles, mitigating class imbalance while maintaining realism.
- The evaluation protocol is highly sensitive to "time." Requiring that retrieved evidence not be later than the claim publication date is crucial for all tool-augmented AFC/agent evaluations; otherwise, systems might use future facts to reverse-engineer answers.

## Limitations & Future Work
- VeriTaS depends on the continuous accessibility of ClaimReview, Data Commons, Google Fact Check Tools, and professional fact-checking organizations; changes in platform policies could affect dynamic updates.
- While verified, rectified Intact claims may still carry LLM rewriting styles, which future models might exploit as style shortcuts.
- Cross-lingual de-duplication is not yet exhaustive; the same factual event may enter the data in multiple languages, though this reflects real dissemination patterns.
- Currently covers text, images, and videos; if audio-only misinformation increases, the pipeline will need expansion for audio transcription, voice spoofing detection, and context judgment.
- The benchmark evaluates verdict scores but not the quality of evidence submitted by the model. Future work could include evaluations for evidence retrieval, justification factuality, and evidence sufficiency.

## Related Work & Insights
- **vs. Static AFC Benchmarks**: Traditional benchmarks are usually one-off releases, good for horizontal comparison but prone to pre-training contamination. VeriTaS centers the evaluation on "newly emerging claims" via quarterly updates.
- **vs. Synthetic Misinformation Datasets**: Purely synthetic claims are controllable but lack realism; VeriTaS claims originate from real fact-checking articles, and rectified samples are constrained by evidence grounding.
- **vs. Single-Label Binary Classification**: Single-label tasks are easy to evaluate but have low explanatory power; VeriTaS breaks down media authenticity, context, text veracity, and coverage, aligning more closely with professional workflows.
- **Insight for Future Work**: Any benchmark targeting real-world, long-cycle cycles should explicitly record sample timestamps, knowledge cutoffs, and tool-available evidence times; otherwise, model capability and memory capability will be conflated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of dynamic, multimodal, and multilingual AFC benchmarks with continuous Integrity scoring is highly complete.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Data statistics, human verification, latest-quarter baselines, and longitudinal cutoff analyses are robust.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, but the data construction phase is information-dense, requiring cross-referencing between the main text and appendix.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for fact-checking evaluation, tool-augmented VLM evaluation, and time-sequenced leakage-proof benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)
- [\[ACL 2026\] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2025\] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](../../ICML2025/social_computing/defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)
- [\[ACL 2026\] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)

</div>

<!-- RELATED:END -->
