---
title: >-
  [Paper Note] Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation
description: >-
  [ICLR 2026][Multimodal VLM][speculative decoding] Inspired by the draft-then-verify paradigm of Speculative Decoding, this paper proposes Speculative Verdict (SV), which employs multiple lightweight VLMs to generate diverse reasoning paths as drafts, while a large model serves as the verdict to synthesize, verify, and correct them. Without any training, SV surpasses GPT-4o by 11.9% on information-intensive VQA and recovers correct answers in 47–53% of minority-correct cases.
tags:
  - ICLR 2026
  - Multimodal VLM
  - speculative decoding
  - visual reasoning
  - information-intensive VQA
  - draft-verdict framework
  - consensus expert selection
date: 2026-05-08
content_hash: 68164f4c019fbb29
---

# Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation

**Conference**: ICLR 2026
**arXiv**: [2510.20812](https://arxiv.org/abs/2510.20812)
**Code**: [https://github.com/Tinaliu0123/speculative-verdict](https://github.com/Tinaliu0123/speculative-verdict)
**Area**: Multimodal VLM
**Keywords**: speculative decoding, visual reasoning, information-intensive VQA, draft-verdict framework, consensus expert selection

## TL;DR
Inspired by the draft-then-verify paradigm of Speculative Decoding, this paper proposes Speculative Verdict (SV), which employs multiple lightweight VLMs to generate diverse reasoning paths as drafts, while a large model serves as the verdict to synthesize, verify, and correct them. Without any training, SV surpasses GPT-4o by 11.9% on information-intensive VQA and recovers correct answers in 47–53% of minority-correct cases.

## Background & Motivation

**Background**: Large VLMs perform well on general VQA but face significant challenges in information-intensive image understanding—such as infographic and chart analysis involving dense visual-textual interleaving with annotations, legends, and complex layouts. Benchmarks like InfographicVQA, ChartMuseum, and ChartQAPro require models to precisely locate and reason over information embedded in complex layouts. The core challenge lies in the coordination of two critical capabilities: precise localization (identifying all relevant regions in dense layouts) and multi-hop reasoning (linking visual and textual evidence scattered across different regions).

**Limitations of Prior Work**: Existing methods primarily improve perception via search-based zoom-in pipelines. Learning-based approaches (e.g., DeepEyes, Pixel-Reasoner) train zoom strategies with reinforcement learning at high cost; training-free methods crop based on attention maps or confidence scores, but these signals correlate poorly with truly relevant regions in dense layouts and often misdirect to visually similar but irrelevant areas. Both paradigms struggle to comprehensively gather scattered evidence required for multi-hop reasoning.

**Key Challenge**: Information-intensive VQA exhibits extreme error sensitivity—any single misread or omission during localization propagates along the reasoning chain, leading to entirely incorrect final answers. A single model cannot simultaneously achieve comprehensive evidence coverage and error-free reasoning at every step. Moreover, simple majority voting completely fails in minority-correct scenarios, where multiple models may make the same error at the same location.

**Goal**: (1) How to improve evidence coverage in information-intensive VQA without training? (2) How to correct errors and recover correct answers from multiple imperfectly correct reasoning paths? (3) How to efficiently select the most reliable draft experts to balance accuracy and inference cost? (4) Can multi-model synthesis surpass the reasoning capability of a single large model?

**Key Insight**: The core insight of Speculative Decoding—draft models rapidly expand coverage while the verifier ensures correctness—maps naturally onto information-intensive visual reasoning: multiple lightweight VLMs can serve as drafts to locate evidence and extract information from different perspectives, while a large model acts as the verdict to synthesize and eliminate contradictions. A key observation is that different VLMs tend to localize different regions and extract different evidence from the same information-dense image, forming natural complementarity.

**Core Idea**: Transfer the draft-then-verify paradigm of Speculative Decoding from token-level inference acceleration to task-level multi-model evidence synthesis and error correction in VQA.

## Method

### Overall Architecture
Given an input image-question pair $(x, q)$, SV operates in two stages: (1) **Draft stage**—from a candidate pool of $k=5$ VLMs, $m=3$ consensus-strongest draft experts are selected via a consensus scoring mechanism, each generating a detailed reasoning path $r_i$ using a CoT prompt; (2) **Verdict stage**—a large model (GPT-4o or Qwen2.5-VL-72B) receives the original image, the question, and all reasoning paths $\{r_i\}_{i=1}^{m}$, and produces the final answer $y = J(x, q, \{r_i\}_{i=1}^{m})$ in a single inference pass by verifying, resolving contradictions, and synthesizing.

### Key Designs

1. **Draft Stage: Multi-Expert Reasoning Path Generation**

   - **Function**: Obtain diverse evidence localization and reasoning paths via multiple lightweight VLMs.
   - **Mechanism**: Each draft expert generates a structured reasoning path using a CoT template with three levels—global scanning and localization proposals (identifying relevant regions, sub-figures, axis titles) → evidence extraction (converting visual/textual elements into structured cues, e.g., reading legends, mapping colors, parsing axis labels) → analysis and reasoning operations (filtering, sorting, computing, cross-referencing). Different experts may localize and extract differently, forming a complementary but noisy evidence pool.
   - **Design Motivation**: A single VLM is prone to misreading or missing content in certain regions of dense images; multiple models reasoning independently substantially improve evidence coverage.
   - **Implementation**: The draft pool comprises 5 VLMs of 7–9B parameters (Qwen2.5-VL-7B, MiMo-VL-7B-RL, InternVL3-8B, GLM-4.1V-9B-Thinking, Ovis2.5-9B), with diverse architectures selected to ensure complementarity.

2. **Consensus Expert Selection**

   - **Function**: Training-free selection of the most reliable draft experts from the candidate pool.
   - **Mechanism**: Each of the $k$ candidate VLMs first generates a candidate answer $y_i$; a global consensus score is then computed for each answer as $s(y_i) = \sum_{j \neq i} |NLL_j(y_i) - NLL_j(y_j)|$, where $NLL_j(y_i)$ denotes the negative log-likelihood of answer $y_i$ under model $M_j$. A lower consensus score indicates greater peer agreement; the $m$ models with the lowest scores are selected as draft experts. This step requires only prefill computation, with each draft decoded only once.
   - **Design Motivation**: Information-intensive VQA has a unique correct answer per question, so inter-model consensus naturally points toward more reliable reasoning paths. Compared to selecting the most divergent experts (maximizing diversity), consensus-based selection proves more effective for this task type.
   - **Computational Efficiency**: Consensus scoring requires only prefill over candidate answers with no additional decoding, adding negligible overhead to total inference time.

3. **Verdict Stage: Synthesis, Verification, and Error Correction**

   - **Function**: Recover the correct answer from multiple potentially incomplete reasoning paths.
   - **Mechanism**: The large model simultaneously receives the original image and all draft reasoning paths as context, acting as a synthesizer rather than a voter—evaluating localization consistency, identifying cross-path contradictions, and integrating consistent cues to generate a coherent prediction. Computation is concentrated in the prefill phase (processing thousands of tokens of reasoning paths), with only a few answer tokens decoded, avoiding the high cost of iterative region-by-region analysis or long-chain autoregressive generation by the large model.
   - **Design Motivation**: Majority voting fails in minority-correct scenarios—when the majority of experts make the same error at the same location, the correct answer is suppressed. By cross-validating factual details across reasoning paths rather than simply tallying votes, the verdict can recover information from the minority of correct paths.
   - **Cost Advantage**: The verdict requires only a single inference call, with computation concentrated in the prefill phase; only a few tokens need to be decoded.

### Loss & Training
SV is entirely training-free and requires no fine-tuning of any model. The draft pool uses 5 open-source VLMs of 7–9B parameters (Qwen2.5-VL-7B, MiMo-VL-7B-RL, InternVL3-8B, GLM-4.1V-9B-Thinking, Ovis2.5-9B), and the verdict uses GPT-4o or Qwen2.5-VL-72B. For information-intensive benchmarks, PP-StructureV3 is additionally applied to convert images into layout-preserving structured formats to assist the verdict model.

## Key Experimental Results

### Main Results

| Model | InfographicVQA (ANLS) | ChartMuseum (Acc) | ChartQAPro (Acc) | HR-Bench 4K (Acc) |
|---|---|---|---|---|
| GPT-4o | 76.5 | 42.7 | 52.6 | 67.4 |
| GLM-4.1V-Thinking (9B) | 84.8 | 48.0 | 56.2 | 72.3 |
| Qwen2.5-VL-72B | 84.2 | 40.7 | 60.7 | 73.1 |
| DeepEyes (7B) | 75.5 | 28.0 | 48.7 | 73.0 |
| Pixel-Reasoner (7B) | 84.0 | 25.9 | 39.3 | — |
| **SV (GPT-4o verdict)** | **88.4** (+11.9) | **49.3** (+6.6) | **64.0** (+11.4) | 71.4 (+4.0) |
| **SV (72B verdict)** | 86.7 (+2.5) | 48.2 (+7.5) | 63.0 (+2.3) | **75.6** (+2.5) |

### Ablation Study

| Dimension | Configuration | InfographicVQA | ChartQAPro | Notes |
|---|---|---|---|---|
| Draft count | m=1 | ~85 | ~59 | Performance scales approximately linearly with m |
| Draft count | m=3 (default) | 88.4 | 64.0 | Optimal accuracy–efficiency trade-off |
| Draft count | m=5 | ~88.5 | ~64 | Saturated; cost grows linearly |
| Verdict input | Final answers only | 73.4 | 59.2 | Severe degradation without reasoning paths |
| Verdict input | Full reasoning paths | 88.4 | 64.0 | +15pp / +4.8pp over answers-only |
| Selection strategy | Consensus selection | 88.4 | 64.0 | Default; optimal |
| Selection strategy | Divergence selection | <single-model baseline | <single-model baseline | Diversity is harmful for this task type |
| Verdict scale | Small verdict (7–9B) | 84.1–85.4 | 57.2–60.3 | Smaller models decode more but perform worse |

### Key Findings
- SV achieves a 47–53% recovery rate on minority-correct cases: even when the majority of drafts provide incorrect answers, the verdict can extract the correct information from the minority of correct paths—a feat impossible under majority voting.
- Recovery rate on zero-correct cases is 2.5–4.5%: even when all drafts and the verdict individually answer incorrectly, SV can recover the correct answer by synthesizing partially correct reasoning steps, demonstrating that the total information in complementary reasoning paths exceeds that of any single path.
- SV outperforms all tool-driven methods: surpassing DeepEyes by 12.9–21.3% and Pixel-Reasoner by 4.4–24.7%, indicating that reasoning path synthesis is superior to region-by-region zoom-in.
- Consensus selection > diversity selection: divergence-based selection even falls below single-model baselines, since information-intensive VQA has unique answers and consensus naturally points toward correctness.
- Reasoning paths are far more important than final answers: passing only answers to the verdict causes a 15pp drop, confirming that intermediate evidence in reasoning processes is critical for error correction.
- $m=3$ is the optimal draft count: performance scales approximately linearly from $m=1$ to $m=3$, saturating beyond $m=3$, while inference cost grows linearly with $m$.
- Generalization gains are also observed on MathVista and TallyQA (+17.8% and +1.5% over GPT-4o, respectively), demonstrating that SV is not limited to information-intensive scenarios.

## Highlights & Insights
- The transfer of Speculative Decoding from token-level acceleration to task-level error correction is conceptually elegant—the core principle of "drafts expand coverage, verifier ensures quality" is preserved but applied at an entirely new level. This paradigm is transferable to any scenario requiring answer synthesis from multiple imperfect information sources (e.g., multi-source document QA, scientific reasoning).
- The consensus scoring mechanism measures inter-model agreement via NLL differences in a clean and computationally efficient manner (prefill only, no additional decoding). The key normalization design—subtracting each model's NLL for its own answer—eliminates cross-model calibration differences, enabling fairer cross-model comparisons.
- The ability to recover minority-correct cases is the fundamental advantage of SV over majority voting. From an information-theoretic perspective, reasoning paths carry far more evidence than final answers, allowing the verdict to make judgments at the granularity of individual reasoning steps.
- Concentrating verdict computation in the prefill rather than the decoding phase is an elegant engineering choice—the large model only needs to process the input context (thousands of tokens of reasoning paths) and output a few answer tokens, avoiding expensive long-sequence autoregressive generation.
- The fully training-free nature makes SV plug-and-play: as stronger open-source VLMs emerge, both the draft pool and the verdict model can be seamlessly replaced for continued gains.

## Limitations & Future Work
- The use of 5 candidate VLMs and 1 large verdict model incurs non-trivial total inference cost, though it is cheaper than having a large model perform iterative region-by-region analysis. Lighter verdict alternatives should be explored for resource-constrained settings.
- The system places high demands on verdict model capability—small verdicts (7–9B) perform substantially worse, indicating a strong dependency on large models.
- The effect of draft pool composition on performance remains unexplored: which model combinations yield the strongest complementarity? Do architecturally or training-objective-diverse combinations outperform homogeneous ones?
- PP-StructureV3 document structure extraction introduces an additional preprocessing step, adding system complexity and potentially being ineffective for non-document images.
- Whether consensus selection remains effective for open-ended tasks with non-unique answers (e.g., image captioning, creative generation) is unclear.
- When all draft models make the same type of error at the same location (e.g., systematic OCR failure), SV cannot recover the correct answer.

## Related Work & Insights
- **vs. DeepEyes / Pixel-Reasoner**: These methods train zoom strategies with RL to progressively magnify regions; SV replaces tool-driven search with multi-model reasoning synthesis. SV's advantages are that it requires no training and achieves more comprehensive coverage, though zoom-based methods may be more efficient when precise localization of a single region suffices.
- **vs. LLaVA-Critic (LMM-as-a-Judge)**: LLaVA-Critic selects the best single answer from candidates; SV synthesizes multiple paths to generate a new answer. SV exceeds it by 4.9–11.9%, because synthesis can repair partial errors within each path, whereas selection can only accept or reject complete paths.
- **vs. Speculative Decoding**: Original Speculative Decoding performs token-level verification to accelerate inference speed; SV performs task-level synthesis to improve reasoning quality—both share the draft-then-verify framework but pursue entirely different objectives.
- **vs. Majority Voting / Self-Consistency**: Majority voting assumes the correct answer is the most common, but in information-intensive reasoning, multiple models may make the same error at the same location, making the majority incorrect. SV overcomes this limitation through reasoning-path-level synthesis rather than answer-level voting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The conceptual transfer of Speculative Decoding to task-level visual reasoning is creative; the NLL-normalized consensus scoring design is clean and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on 7 benchmarks (4 information-intensive + HR-Bench + MathVista + TallyQA); ablation studies cover draft count, selection strategy, verdict input format, and verdict model scale; error-correction capability is quantitatively analyzed.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured; a running case (Figure 3) threads through the paper to aid understanding of the full pipeline, with method description and experimental analysis closely aligned.
- **Value**: ⭐⭐⭐⭐ — The training-free framework is highly practical; gains on information-intensive VQA are substantial and consistent; the reasoning path synthesis paradigm has broad applicability.

<!-- END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HIVE: Query, Hypothesize, Verify — An LLM Framework for Multimodal Reasoning-Intensive Retrieval](../../CVPR2026/multimodal_vlm/hive_query_hypothesize_verify_an_llm_framework_for_multimodal_reasoning-intensiv.md)
- [\[ICLR 2026\] Empowering Small VLMs to Think with Dynamic Memorization and Exploration](empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)
- [\[CVPR 2026\] Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](../../CVPR2026/multimodal_vlm/downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)
- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](liveweb-ie_a_benchmark_for_online_web_information_extraction.md)

</div>

<!-- RELATED:END -->
