---
title: >-
  [Paper Note] SWAN: Semantic Watermarking with Abstract Meaning Representation
description: >-
  [ACL2026][LLM Safety][Semantic Watermarking] SWAN embeds watermarks into the semantic graph structure of sentences using Abstract Meaning Representation (AMR) templates rather than token or embedding regions. Consequentl…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Semantic Watermarking"
  - "AMR"
  - "paraphrase robustness"
  - "S2match"
  - "Text Provenance"
date: 2026-05-08
content_hash: d046131ae5aa42da
---

# SWAN: Semantic Watermarking with Abstract Meaning Representation

**Conference**: ACL2026  
**arXiv**: [2605.04305](https://arxiv.org/abs/2605.04305)  
**Code**: None  
**Area**: LLM Security / Text Watermarking / Semantic Representation  
**Keywords**: Semantic Watermarking, AMR, paraphrase robustness, S2match, Text Provenance  

## TL;DR
SWAN embeds watermarks into the semantic graph structure of sentences using Abstract Meaning Representation (AMR) templates rather than token or embedding regions. Consequently, the watermark remains detectable through AMR parsing, template matching, and proportion z-tests even after meaning-preserving paraphrasing.

## Background & Motivation
**Background**: As LLM-generated text becomes increasingly natural, text watermarking has emerged as a key technology for identifying AI-generated content, tracing sources, and mitigating large-scale misinformation.

**Limitations of Prior Work**: Mainstream token-level watermarks shift token sampling preferences during generation to push more tokens into a secret green list. While simple to implement and easy to detect, these methods lose signals easily when faced with paraphrasing, synonym substitution, or minor rewriting.

**Key Challenge**: A watermark must be stealthy and detectable while surviving semantics-preserving rewrites. Token-level signals are too superficial. While embedding-level semantic watermarks are more robust, detection still degrades if paraphrasing pushes the sentence vector into a different semantic region.

**Goal**: The authors aim to anchor watermarks to a level more stable than tokens and sentence embeddings: the abstract semantic structure of sentences. As long as rewriting does not alter the core semantic relations—"who did what to whom"—the watermark should be retained.

**Key Insight**: Abstract Meaning Representation (AMR) represents sentence semantics as graphs, where nodes represent concepts or events and edges represent semantic roles. Diverse surface-level paraphrases can map to the same or highly similar AMR graphs, which is naturally suited for paraphrase-robust watermarking.

**Core Idea**: Construct a private AMR template bank. During generation, match each sentence to a randomly selected AMR template. During detection, parse the text's AMR and calculate the proportion of sentences that match the private templates.

## Method
The core of SWAN is shifting the "watermark key" from vocabulary hashes or embedding partitions to a private AMR template bank.

It is training-free: it does not require training a watermark model or accessing the target LLM's logits. Instead, it uses prompt guidance and rejection sampling to ensure sentences fall into the target semantic structure.

### Overall Architecture
The system first constructs a template AMR bank from MASSIVE-AMR.

MASSIVE-AMR provides approximately 84K AMR graphs corresponding to 1,685 information-querying utterances.

The authors further abstract the original AMRs into templates—for example, replacing specific named entities with NE, common nouns with N, and unspecified concepts with X.

The template bank only retains AMR patterns with a frequency between 3 and 20 that contain at least 3 concept nodes.

During the generation phase, for each sentence generated, a template AMR is randomly selected from the private bank, and both the current context and the template are included in the LLM prompt.

The LLM is required to generate a sentence that is contextually coherent, satisfies the user's original intent, and conforms as much as possible to the template's semantic structure.

After generation, the system uses an AMR parser to parse the candidate sentence into an AMR graph and calculates its similarity to the target template using S2match.

If the similarity exceeds an injection threshold, the sentence is accepted; otherwise, it is resampled.

If a template fails multiple times in the current context, the system switches to a different template to avoid infinite retries on incompatible semantic structures.

During the detection phase, the system segments candidate paragraphs into sentences, parses the AMR of each sentence, and calculates its maximum S2match with all templates in the private bank.

If the score exceeds a detection threshold, the sentence is counted as watermarked. Finally, a one-proportion z-test is performed on the ratio of watermarked sentences in the paragraph.

### Key Designs
1. **Private AMR Template Bank**:
	- **Function**: Serves as the watermark key, defining which abstract semantic structures belong to the "green zone."
	- **Mechanism**: Extracts graph structures from MASSIVE-AMR, removing specific entities and lexical details while retaining predicates, semantic roles, and conceptual relations. Templates with very low frequency are too rare and difficult to generate, while those with very high frequency are too common and increase false positives; thus, the authors retain medium-frequency templates.
	- **Design Motivation**: While token-level methods use vocabulary partitions and embedding-level methods use vector regions as keys, SWAN uses semantic graph structures. As long as the bank remains private, the detector can verify structural matches while attackers struggle to know which AMR patterns to avoid.

2. **AMR-Guided Rejection Sampling**:
	- **Function**: Pushes the generated sentences toward the target semantic structure without changing model parameters or accessing logits.
	- **Mechanism**: The LLM prompt provides both historical context and a target AMR template, requesting the generation of a natural sentence that instantiates placeholder concepts like NE/N/X. Each candidate sentence is parsed by an AMR parser and compared with the target template using S2match; it is accepted only if $S2match(\hat{g}, g) \geq \theta_{accept}$.
	- **Design Motivation**: Forcing the insertion of keywords destroys fluency and is easily deleted. Allowing the model to generate around an AMR template hides the watermark within the predicate-argument structure, while the surface text remains natural.

3. **Paragraph-level z-test detection**:
	- **Function**: Accumulates sentence-level template matching into a paragraph-level provenance judgment.
	- **Mechanism**: For each sentence in a paragraph, the maximum template similarity to the bank is calculated; if it exceeds $\theta_{detect}$, it is counted toward $k$. Given the total number of sentences $n$ and the random hit rate $\lambda$ under the null hypothesis, $z=(k-\lambda n)/\sqrt{n\lambda(1-\lambda)}$ is used to determine if the hit ratio is abnormally high.
	- **Design Motivation**: Single-sentence AMR parsing may be noisy, and single-sentence false positives are inevitable. Paragraph-level statistics aggregate weak signals, similar to z-score detectors in token watermarking.

### Loss & Training
SWAN has no training loss; key hyperparameters are derived from the generation and detection workflows.

The default AMR bank size is 50, though the authors also tested settings of 100, 500, and 800.

Watermark generation uses DeepSeek-R1-Distill-Qwen-14B with temperature 0.6 and top_p 0.9.

A maximum of 50 attempts are made per sentence: a maximum of 10 templates, with up to 5 generation attempts per template.

Detection uses the `amrlib` `parse_xfm_bart_large` pipeline, a parser based on BART-large trained on AMR-3.

Paraphrase attacks use Pegasus, Parrot, and Claude 3.7 Sonnet.

Text quality is evaluated reference-free by Claude 3.7 across three dimensions: coherence, fluency, and diversity.

## Key Experimental Results

### Main Results
In scenarios without paraphrasing, SWAN's raw detectability is close to strong sentence-level baselines and outperforms the low FPR metrics of the token-level SynthID.

| Method | AUC↑ | TPR@1%↑ | TPR@5%↑ |
|------|------|----------|----------|
| SynthID | 97.0 | 64.8 | 84.8 |
| SemStamp | 99.4 | 96.8 | 100.0 |
| k-SemStamp | 99.1 | 96.8 | 96.4 |
| SWAN | 99.1 | 91.6 | 97.6 |

Regarding the critical metric of paraphrase robustness, SWAN achieves the highest AUC under all three types of attacks, showing significant advantages particularly against strong LLM rewrites like Claude 3.7.

| Method | Pegasus AUC/TPR@1%/TPR@5% | Parrot AUC/TPR@1%/TPR@5% | Claude AUC/TPR@1%/TPR@5% |
|------|----------------------------|---------------------------|---------------------------|
| SemStamp | 97.6 / 87.2 / 97.6 | 94.8 / 69.2 / 97.6 | 84.4 / 36.8 / 84.8 |
| k-SemStamp | 97.3 / 88.8 / 88.4 | 92.8 / 68.0 / 66.8 | 87.6 / 53.6 / 53.2 |
| SWAN | 98.1 / 81.2 / 92.8 | 97.5 / 82.0 / 92.4 | 98.3 / 86.0 / 95.2 |

This table directly demonstrates the value of AMR semantic anchoring: while Claude rewriting significantly weakens SemStamp and k-SemStamp, SWAN's AUC remains at 98.3.

### Ablation Study
The impact of AMR bank size on AUC is minimal, indicating that the method is not sensitive to the scale of the template library.

| AMR bank size | AUC↑ |
|---------------|------|
| 50 | 99.1 |
| 100 | 98.7 |
| 500 | 98.4 |
| 800 | 99.3 |

In terms of sampling efficiency, SWAN is somewhat slower than SemStamp, but most sentences converge quickly.

| Metric | SWAN | Comparison / Notes |
|------|------|-------------|
| Avg. Acceptance Attempts | 17.7 | 13.8 for SemStamp |
| % Accepted within 10 trials | 42% | Indicates many templates are easily satisfied |
| % Accepted within 15 trials | 54% | Over half succeed within a low budget |
| Spike near max budget | 46-50 trials| Indicates some templates are incompatible with context |
| Generation Scale | 1,250 sentences | 250 samples × 5 sentences per paragraph |

### Key Findings
- SWAN does not sacrifice detectability in scenarios without original text modification, with AUC scores generally in the same tier as SemStamp/k-SemStamp.
- The gap widens significantly after paraphrasing; notably, under Claude rewriting, SWAN maintains an AUC of 98.3, whereas SemStamp drops to 84.4 and k-SemStamp to 87.6.
- Expanding the AMR bank from 50 to 800 maintains an AUC of 98+, showing a stable balance between template coverage and false positives.
- Rejection sampling overhead exists, but 42% of sentences succeed within 10 attempts, making the overall overhead acceptable. The true challenge lies in context-aware template selection.
- Text quality assessments show all watermarking methods cause slight degradation, but SWAN is comparable to sentence-level baselines, incurring no significant extra quality cost for robustness.

## Highlights & Insights
- SWAN’s most valuable insight is that "paraphrasing preserves meaning, so watermarks should be written into the meaning representation." This is more natural than chasing paraphrases across tokens or embeddings.
- The AMR template bank turns the watermark into an interpretable structural signal. When detection fails, one can analyze which predicate-argument structure failed to parse, rather than just receiving an opaque embedding hash.
- The training-free and black-box generation nature makes the method easier to integrate with closed-source or API-based models, as it requires no logit access or model weight modifications.
- Paragraph-level z-tests inherit the statistical logic of traditional watermark detectors while replacing token hits with semantic template hits—a clean abstract migration.

## Limitations & Future Work
- Detection is highly dependent on AMR parser quality; parsing errors can lead to missed detections or false positives. AMR parsing may be unstable in low-resource languages, specialized technical texts, or non-news genres.
- The AMR bank serves as a private key; if an attacker guesses or leaks the bank, they could deliberately rewrite semantic structures to evade detection.
- Current evaluations primarily use English RealNews, limiting domain and language coverage.
- Rejection sampling still incurs a cost, with repeated failures occurring when specific templates are incompatible with the context; more intelligent template-context matching is needed.
- SWAN focuses on sentence-level AMR; watermark structures might be reorganized during attacks involving merging, splitting, or cross-sentence rewriting. Future work could explore paragraph-level AMR or AMR subgraph watermarks.

## Related Work & Insights
- **vs SynthID / token-level watermark**: SynthID relies on token distribution perturbation; it provides effective low-FPR detection but is fragile to paraphrasing. SWAN does not alter token preferences but constrains semantic structure, making it more resistant to surface-level rewriting.
- **vs SemStamp**: SemStamp partitions the sentence vector space into green buckets, providing some paraphrase robustness, but strong rewriting can shift embeddings. SWAN uses AMR graph matching, which maintains structural signals under semantically equivalent rewrites.
- **vs k-SemStamp**: k-SemStamp improves semantic region partitioning via clustering but remains within continuous embedding regions. SWAN’s discrete graph structure is more interpretable and closer to the definition of "unchanged meaning."
- **vs PostMark / post-hoc watermark**: PostMark injects signals via paragraph semantics and watermark words. It is practical but may leave lexical traces. SWAN does not rely on a fixed vocabulary; signals are hidden in semantic role combinations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using AMR graph structures for text watermarking is highly novel and distinct from token/embedding routes.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers detection, paraphrasing, bank size, sampling efficiency, and quality; language and domain scope remain somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐☆ The method is clearly explained and experimental tables are direct, though AMR parsing error and threshold selection could be further detailed.
- Value: ⭐⭐⭐⭐⭐ Provides practical insights for robust text watermarking and AI-generated content provenance, particularly suitable for research on semantic-level provenance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](../../ICLR2026/llm_safety/pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)
- [\[ACL 2026\] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering](safeconstellations_mitigating_over-refusals_in_llms_through_task-aware_represent.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)

</div>

<!-- RELATED:END -->
