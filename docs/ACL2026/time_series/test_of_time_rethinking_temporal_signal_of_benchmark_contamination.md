---
title: >-
  [Paper Note] Test of Time: Rethinking Temporal Signal of Benchmark Contamination
description: >-
  [ACL2026][Time Series][Benchmark contamination] This paper demonstrates that "performance decay after cutoff" is not robust evidence of benchmark contamination: the temporal decay signal can significantly change or even…
tags:
  - "ACL2026"
  - "Time Series"
  - "Benchmark contamination"
  - "temporal signals"
  - "LLM evaluation"
  - "question rephrasing"
  - "influence functions"
date: 2026-05-08
content_hash: 9128ff5994425c61
---

# Test of Time: Rethinking Temporal Signal of Benchmark Contamination

**Conference**: ACL2026  
**arXiv**: [2509.00072](https://arxiv.org/abs/2509.00072)  
**Code**: None  
**Area**: LLM Evaluation / Benchmark Contamination / Temporal Analysis  
**Keywords**: Benchmark contamination, temporal signals, LLM evaluation, question rephrasing, influence functions  

## TL;DR
This paper demonstrates that "performance decay after cutoff" is not robust evidence of benchmark contamination: the temporal decay signal can significantly change or even disappear when the same batch of source documents is shifted from original text cloze tests to LLM-rephrased questions.

## Background & Motivation
**Background**: Large model evaluation increasingly relies on public benchmarks, but public questions, solutions, and discussions are likely to enter training corpora. Since most frontier models do not disclose training data, researchers often use indirect probes to judge contamination. A popular practice is temporal analysis: comparing model performance on questions released before and after the training cutoff. If performance is significantly worse post-cutoff, this post-cutoff performance decay is interpreted as evidence that pre-cutoff questions were memorized.

**Limitations of Prior Work**: This inference seems intuitive but conflates "question release time" with "question construction method." Many temporal benchmarks take original questions directly from web pages, competitions, or papers; others let LLMs generate new questions based on the same source material. While both share the same source material, their surface forms, retrieval cues, and memorability differ completely. Thus, the same model might appear to be memorizing in one format while truly reasoning in another.

**Key Challenge**: Temporal decay signals aim to measure the contamination relationship between training corpora and test questions, but what is actually observed is "whether the model can trace the test input back to text seen during training." If a question is rephrased sufficiently by an LLM, the model may fail to link the current question to the source document even if the source paper is in the training set. Conversely, cloze tests or original snippets expose strong memorization cues.

**Goal**: The authors aim to answer three questions: first, whether LLM-generated arXiv reasoning questions truly lack post-cutoff decay; second, whether this absence is caused by the question generation method rather than a lack of contamination in the source material; third, whether the internal mechanisms of the model can explain why cloze and LLM-generated questions yield different signals.

**Key Insight**: The key control variable of the paper is "same source material, different question phrasing." The authors construct LLM-synthesized QA and cloze QA around the same set of arXiv papers, transfer this idea to LiveCodeBench and Wikipedia current events, and finally perform influence function analysis using the training corpus of the open-data model OLMo2.

**Core Idea**: By using question construction as an intervention variable, it is proven that temporal contamination signals are highly sensitive to phrasing transformations and cannot serve as sufficient evidence for contamination conclusions on their own.

## Method

### Overall Architecture
The paper can be viewed as a three-layer verification framework. The first layer is temporal analysis at the performance level: collecting math/physics arXiv papers over 26 months (May 2023 to June 2025), generating multi-step reasoning QA based on content like theorems, and comparing pre/post cutoff accuracy across 8 frontier models. The second layer is construction intervention: constructing cloze questions on the same source papers and extending similar rephrasing experiments to LiveCodeBench and Wiki-based QA. The third layer is at the mechanistic level: using influence functions on OLMo2-7B-Instruct to check which training documents most influence the model when answering a specific question.

The input is a batch of timestamped public documents or benchmark questions. The output is not a new model but a set of contamination probe results: pre/post cutoff gaps under different construction methods, replication experiments across different fields, and the hit rates of source documents in influence rankings for cloze vs. LLM-generated QA. This decouples "contamination" from a single accuracy drop into two levels: "traceability of phrasing" and "likelihood of source material appearing in training."

### Key Designs
1. **Temporal Evaluation of Same-Source Material**:
	- Function: Replicate and expand RealMath-like settings to check if LLM-generated questions consistently lack post-cutoff performance decay.
	- Mechanism: The authors crawled 20,277 math/physics papers via the arXiv API over 26 months and used o4-mini to generate QA requiring 5+ steps of reasoning from theorems. After GPT-4o deduplication and human inspection, 1,643 questions corresponding to 1,098 papers were retained. Monthly accuracy is normalized as $Accuracy_m=C_m/Q_m$, followed by a pre/post comparison around each model's cutoff.
	- Design Motivation: If LLM-generated questions show no decay across multiple models, fields, and cutoff dates, it indicates that "source material coming from public arXiv" is insufficient to produce temporal decay; one must further examine whether the questions retain memorizable textual cues.

2. **Question Construction Method as an Intervention Variable**:
	- Function: Observe whether temporal signals flip by changing only the question format without altering the answer source or underlying solution.
	- Mechanism: Construct cloze questions for arXiv abstracts (masking 5 semantic key phrases per paper); rephrase variable names, semantic backgrounds, and symbols for 400 LiveCodeBench questions using o4-mini while keeping algorithmic solutions intact; construct dated MCQs for Wikipedia current events and rephrase question statements while keeping options and answers.
	- Design Motivation: This design keeps difficulty, answers, and release times as fixed as possible, varying only "proximity to the original text." If temporal decay appears in original/cloze formats but weakens in LLM-transformed formats, it proves the temporal signal is not a stable contamination indicator but is strongly modulated by benchmark construction.

3. **Influence Functions for Tracing Source Document Traceability**:
	- Function: Explain mechanistically why the same contamination source produces different signals under different phrasings.
	- Mechanism: OLMo2-7B-Instruct was selected as it has public training data, allowing confirmation that certain arXiv papers are indeed in the training set. For 40 known contaminated papers, cloze and LLM-generated QA were constructed, and the top-100 influential documents were ranked among 10,000 training samples. Influence scores were approximated using Kronfluence/EK-FAC, with the core form being $$I_f(z) \approx -\nabla_\theta f(\theta_s)^\top (G+\lambda I)^{-1}\nabla_\theta L(z,\theta_s)$$.
	- Design Motivation: Performance curves only show phenomena; influence functions ask if the model truly treats the source paper as a critical training point when answering. The high hit rate for cloze and low hit rate for LLM-generated QA support the explanation that rephrasing weakens source document traceability.

### Loss & Training
This paper does not propose training a new model but rather an evaluation and analysis pipeline. The question generation phase uses o4-mini with high reasoning effort, while filtering uses GPT-4o to remove duplicate or simple samples. Human inspection ensures deterministic answers, at least 5 intermediate reasoning steps, clarity, and derivability from source material. Model evaluation is run via the OpenRouter API without web search to reduce hidden retrieval interference. Influence function experiments use the public training corpus of OLMo2-7B-Instruct and EK-FAC to approximate inverse-curvature vector products for computable attribution of training points on large models.

## Key Experimental Results

### Main Results
LLM-generated arXiv multi-step QA did not show systematic post-cutoff decay. In the physics domain, most models actually improved slightly after the cutoff; the average change across 16 model-domain observations was +2.19 percentage points (95% CI [+0.61, +3.78], paired t-test $p=0.010$).

| Setting | Model / Statistic | Pre-cutoff | Post-cutoff | Gap (Post-Pre) | Conclusion |
|------|---------------|------------|-------------|---------------|------|
| Physics, LLM-generated QA | DeepSeek-R1 | 21.1 | 22.7 | +1.6 pp | No decay |
| Physics, LLM-generated QA | Gemini-2.0-Flash | 33.3 | 39.2 | +5.9 pp | Higher post-cutoff |
| Physics, LLM-generated QA | Llama-3.3-70B | 15.1 | 15.5 | +0.4 pp | Mostly flat |
| Physics, LLM-generated QA | o4-mini | 36.8 | 40.5 | +3.7 pp | Higher post-cutoff |
| Math + Physics Summary | Mean of 16 obs | - | - | +2.19 pp | Disproves "inevitable decay" |

### Ablation Study
When the same source papers were converted to cloze questions, temporal decay reappeared; when LiveCodeBench or Wiki QA were semantically rephrased by LLMs, the originally more pronounced decay was weakened or removed.

| Intervention | Metric / Model | Original or Cloze Gap | LLM-transformed Gap | Notes |
|------|-------------|------------------|---------------------|------|
| RealMath arXiv cloze | GPT-4o-mini, LLM judge | -3.83 pp | N/A | Decay appears in cloze |
| RealMath arXiv cloze | Llama-3.1-405B, LLM judge| -5.25 pp | N/A | Large models also decay |
| RealMath arXiv cloze | Claude-3.5-Sonnet, BLEU | -6.60 pp | N/A | Literal match also decays |
| Wiki-based QA | GPT-3.5-turbo | -2.65 pp | -0.62 pp | Rephrasing weakens decay |
| Wiki-based QA | GPT-4 | -1.04 pp | +2.81 pp | Becomes gain after rephrasing |
| Wiki-based QA | GPT-4o-mini | -7.59 pp | -4.99 pp | Decay remains but smaller |

Mechanistic experiments further show that cloze questions allow models to more easily trace back to source documents in training, while LLM-generated QA makes this tracking difficult.

| Question Format | Top-1 hit rate | Top-3 hit rate | Sample Size | Meaning |
|----------|----------------|----------------|--------|------|
| Cloze questions | 77.5% | 100.0% | 40 papers | Source papers are often most influential |
| LLM-generated QA | 17.5% | 25.0% | 40 papers | Same source harder to trace after generation |

### Key Findings
- The most critical finding is not "no contamination," but that "no decay does not equal no contamination." Influence function experiments show LLM-generated QA can originate from known training documents even when temporal decay is not evident.
- Question construction is a strong confounding factor. Direct source text/cloze questions act like memorization probes, while LLM-rephrased questions act like semantic transfer or reasoning probes; they have different sensitivities to contamination.
- Cross-domain validation on LiveCodeBench and Wiki QA is important as it shows this phenomenon is not specific to arXiv theorem QA but is a general benchmark transformation effect.

## Highlights & Insights
- Decoupling contamination detection into a causal comparison of "same source, different phrasing" is the most valuable design of this paper. It reminds future benchmark developers to report not only release dates but also how questions were constructed from source materials.
- The use of influence functions is clever: it does not attempt to prove contamination for all black-box models but establishes a mechanistic example on open-data models to show that source documents being in the training set does not guarantee that LLM-rephrased questions will trigger memorized retrieval.
- The implication for evaluation practice is direct: if a benchmark relies on temporal freshness, it is best to test variants including original text, cloze, semantic rephrasing, and structure-preserving versions; looking at a single scalar accuracy gap is prone to over-interpretation.

## Limitations & Future Work
- The arXiv time window is 26 months; while it covers multiple model cutoffs, it may still be affected by changes in paper difficulty, domain popularity, and question generation quality over time.
- Influence function experiments were limited to 40 known contaminated papers due to computational costs; while the mechanism is clear, the statistical scale remains small.
- LLM-generated questions and cloze questions differ not only in "rephrasing" but potentially in difficulty, answer granularity, and grading reliability. Future work could design finer continuous perturbation intensities to quantify the relationship between phrasing distance and temporal signals.
- This paper mainly discusses contamination detection and has not yet provided a complete new metric to replace temporal analysis. More robust directions may involve combining temporal splitting, near-duplicate detection, influence functions, membership inference, and multi-version phrasing consistency tests.

## Related Work & Insights
- **vs Time Travel / LiveCodeBench temporal analysis**: These works treat pre/post cutoff differences as contamination cues. Ours points out that this cue is highly sensitive to construction and is better suited as a warning signal rather than evidence for a standalone conclusion.
- **vs Rephrasing / Perturbation contamination probes**: Prior work often observes performance drops after rephrasing and interprets this as fragile reasoning or contamination. Ours conversely shows that rephrasing can also remove temporal decay, suggesting "score changes after rephrasing" must be interpreted alongside construction mechanisms.
- **vs RealMath**: RealMath found that LLM-generated research-level math QA shows no obvious post-cutoff decay. Ours extends this to a larger window, more models, and physics, and explains the cause using cloze controls.
- **vs Training data auditing**: Direct auditing requires developers to disclose data, which is difficult. Our influence-function experiments on open-data models provide mechanistic references for black-box evaluation but cannot yet replace large-scale data auditing.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Decoupling temporal signals from construction is a sharp problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Provides evidence from arXiv, LiveCodeBench, Wiki, and influence functions, though the mechanistic sample size is small.
- Writing Quality: ⭐⭐⭐⭐☆ Clear argumentation, progressive experimental layers, though some tables are dense.
- Value: ⭐⭐⭐⭐⭐ Direct warning for LLM benchmark freshness, contamination detection, and question generation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](../../NeurIPS2025/time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](../../NeurIPS2025/time_series/syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[ACL 2026\] STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](../../ICLR2026/time_series/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/time_series/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)

</div>

<!-- RELATED:END -->
