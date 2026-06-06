---
title: >-
  [Paper Note] MetFuse: Figurative Fusion between Metonymy and Metaphor
description: >-
  [ACL 2026][NLP Understanding][Metonymy] The authors propose a three-stage pipeline (candidate generation → MLM scoring/selection → LLM polishing) to rewrite literal sentences into three figurative variants: metonymy…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Metonymy"
  - "Metaphor"
  - "figurative fusion"
  - "data augmentation"
  - "LLM generation"
date: 2026-05-08
content_hash: da31e7f43e1b516b
---

# MetFuse: Figurative Fusion between Metonymy and Metaphor

**Conference**: ACL 2026  
**arXiv**: [2604.12919](https://arxiv.org/abs/2604.12919)  
**Code**: https://github.com/cincynlp/MetFuse (Yes)  
**Area**: Linguistics / Figurative Language / Dataset Construction  
**Keywords**: Metonymy, Metaphor, figurative fusion, data augmentation, LLM generation

## TL;DR
The authors propose a three-stage pipeline (candidate generation → MLM scoring/selection → LLM polishing) to rewrite literal sentences into three figurative variants: metonymy, metaphor, and hybrid. This results in the first MetFuse dataset (1,000 quadruplets × 4,000 sentences). Empirical findings demonstrate that "the presence of metaphorical verbs makes metonymic nouns in the same sentence more explicit," and data augmentation consistently improves performance across 8 metonymy/metaphor classification benchmarks.

## Background & Motivation

**Background**: Metonymy (single-domain substitution, e.g., "stadium" referring to "fans") and metaphor (cross-domain mapping, e.g., "fans erupted") are the two pillars of figurative language. However, the NLP community has long treated them as independent tasks: metonymy has datasets like ConMeC / RelocaR / WiMCor, while metaphor has VUA / FLUTE / MOH-X, with almost no research addressing their integration.

**Limitations of Prior Work**: (i) Lack of data - Theoretical linguistics (Goossens 1990's metaphtonymy; Barcelona 2003) has long noted their co-occurrence, but no meaning-aligned datasets exist to support computational research; (ii) Lack of generation methods - Directly prompting LLMs to "turn this into metonymy" achieves only a 38.8% success rate because metonymy must adhere to intra-domain constraints, which naive prompting fails to control; (iii) Lack of interaction analysis - No study has systematically quantified how the identifiability of metaphor and metonymy changes when they co-occur.

**Key Challenge**: Metonymy is strictly constrained by contiguity relations (part-whole, container-content, etc.), resulting in a small candidate space. Metaphor enjoys the freedom of cross-domain mapping with a large candidate space. This asymmetry makes "generating both within a unified framework" extremely difficult.

**Goal**: (a) Given a literal sentence, controllably generate semantically aligned triplets of metonymy / metaphor / hybrid variants; (b) Construct the MetFuse dataset using this framework; (c) Empirically answer whether metonymy is easier to identify in hybrid sentences compared to metonymy-only sentences.

**Key Insight**: The authors leverage two asymmetries—first, in the generation stage, using "narrow candidates + MLM scoring" to constrain metonymy, while using "broad freedom + three emotional tones + sentiment-based selection" to release metaphor; second, in the analysis stage, assuming that the strong selectional preference of metaphorical verbs "forces" readers to interpret metonymic nouns as animate agents, thereby making the metonymy more explicit.

**Core Idea**: A three-stage pipeline (LLM candidate generation + MLM/sentiment scoring selection + LLM controlled polishing) generates figurative variants. The hypothesis that "metaphor strengthens metonymy" is validated through hybrid data augmentation, embedding similarity, and zero-shot LLM experiments.

## Method

### Overall Architecture
The pipeline focuses on "subject noun + predicate verb" pairs in SVO structures. It first uses SpaCy dependency parsing to filter literal sentences from Wikipedia where the subject is a human entity with a specific dependency relationship to a verb. Then, two parallel pipelines—**metonymy generation** and **metaphor generation**—are executed. Finally, hybrid sentences are constructed at zero cost by substituting the refined metonymic noun phrase into the refined metaphorical sentence. All three pipelines essentially follow a three-stage process: "i) LLM candidate generation → ii) external scorer selection → iii) LLM controlled polishing," though the scorers and temperature strategies differ significantly.

### Key Designs

1.  **Metonymy Generation: contiguity-prompt + MLM masked LM scoring**:
    - **Function**: Given a literal sentence and a target noun, generate an intra-domain metonymic replacement noun while maintaining the original meaning.
    - **Mechanism**: The authors found that naive prompting fails because "LLMs do not know what contiguity is." They instead use targeted questions about the noun's location / occupants / salient parts (e.g., "Where does a judge work?") with temperature=0.7 to obtain a set of candidates $c$. The original noun is replaced by `[MASK]` and fed into BERT to calculate $\log p(c \mid \text{context})$, selecting $c^* = \arg\max_c \log p_{\text{BERT}}(c)$ as the replacement. Finally, light polishing is done via LLM with temperature=0.4 (low temperature prevents losing the metonymy through secondary rewriting).
    - **Design Motivation**: MLM scoring essentially "picks words under the premise of syntactic/semantic fluency," which filters out out-of-domain candidates generated by the LLM (e.g., replacing "judge" with "briefcase" results in logp=-12.28, leading to automatic rejection). This transforms intra-domain constraints from a prompt engineering problem into a probabilistic scoring problem.

2.  **Metaphor Generation: tone-conditioned candidates + sentiment-based word selection**:
    - **Function**: Given a literal sentence and a target verb, generate cross-domain, hyperbolic yet tone-consistent metaphorical verbs.
    - **Mechanism**: Following Stowe et al. (2021a), the authors believe "controlled generation" is more effective for metaphors. To allow for cross-domain freedom, an LLM generates hyperbolic verb candidates under three tones: positive / negative / neutral (temperature=0.7, top-p=0.9). A TweetNLP sentiment model labels the original sentence, selects the candidate verb with the matching tone, and then the LLM (temperature=0.6) polishes the sentence for fluency.
    - **Design Motivation**: The authors observed that "hyperbole without tone constraints often conflicts with the overall atmosphere of the sentence" (e.g., an ecstatic verb appearing in a sad sentence). Introducing sentiment selection constrains "metaphorical freedom" within the soft boundary of "tone consistency," preserving cross-domain flexibility without breaking semantic alignment.

3.  **Hybrid Zero-cost Splicing + metaphor-forces-metonymy Analysis**:
    - **Function**: Construct hybrid sentences and verify if "metaphor makes metonymy more explicit."
    - **Mechanism**: Since metonymy generation barely alters syntax (only replaces the noun), the "refined metonymic noun phrase" is directly substituted into the subject position of the "refined metaphorical sentence" to create the hybrid, requiring no extra post-processing. Verification uses three types of evidence: (i) 4 LLMs achieve 1.4–4.3 higher F1 on zero-shot metonymy resolution for hybrid vs. metonymy-only sentences; (ii) BERT contextual embeddings show $\text{sim}(N_{\text{lit}}, N_{\text{hyb}}) > \text{sim}(N_{\text{lit}}, N_{\text{mty}})$, indicating that the noun embedding in hybrids is closer to literal usage (i.e., more "explicit"); (iii) Data augmentation using MetFuse's hybrid subset outperforms metonymy-only augmentation on 4 metonymy benchmarks.
    - **Design Motivation**: From a cognitive linguistics perspective, metaphorical verbs (e.g., "butchered") carry strong animate-agent selectional preferences, forcing readers to interpret inanimate nouns like "newsroom" as "the journalists in the newsroom." This uses metaphor as a "forcing device" to disambiguate metonymy. This explanation ties empirical results back to Lakoff–Johnson theory, representing the paper's most elegant contribution.

### Loss & Training
This paper does not involve model training but rather a prompting workflow. For downstream evaluation, BERT-base is fine-tuned for 3 epochs with lr=1e-5 and batch=8. The MetFuse augmentation sample size is fixed at 50% of the original training set. LLM evaluations are entirely zero-shot, including GPT-OSS-20B / Qwen3-30B / Llama-3.1-70B / Gemini-2.5-Flash.

## Key Experimental Results

### Main Results
Human evaluation of the framework (250 sample sentences) shows that the proposed method significantly outperforms the general prompting baseline across all three figurative types:

| Variant Type | General prompt | Ours | Gain |
|--------------|----------------|------|------|
| Metonymy     | 38.8%          | 75.2%| +36.4 pp |
| Metaphor     | 70.8%          | 84.0%| +13.2 pp |
| Hybrid       | 49.2%          | 74.0%| +24.8 pp |

Downstream metonymy classification (70/30 split, BERT fine-tune), using MetFuse for data augmentation:

| Dataset   | Baseline (Train) | +MetFuse Metonymy | +MetFuse Hybrid |
|-----------|------------------|-------------------|------------------|
| ConMeC    | 75.49            | 76.71 (+1.22)     | **79.33 (+3.84)**|
| Pedinotti | 68.42            | 66.92 (-1.50)     | **70.44 (+2.02)**|
| RelocaR   | 67.33            | 69.99 (+2.66)     | **70.67 (+3.34)**|
| WiMCor    | 81.67            | 82.33 (+0.66)     | **82.67 (+1.00)**|

→ Hybrid augmentation consistently outperforms metonymy-only augmentation across 4 datasets, proving that "metaphorical co-occurrence" provides a stronger training signal.

### Ablation Study
LLM zero-shot metonymy resolution (hybrid vs. metonymy-only positive sentences):

| Model          | Metonymy-only F1 | Hybrid F1 | Gain |
|----------------|------------------|------------|------|
| GPT-OSS-20B    | 67.3             | 71.6       | +4.3 |
| Qwen3-30B      | 85.4             | 87.3       | +1.9 |
| Llama-3.1-70B  | 90.4             | 91.3       | +0.9 |
| Gemini-2.5     | 93.9             | 94.7       | +0.8 |

BERT embedding similarity (verifying "metonymy is more explicit in hybrids"): $\text{sim}(N_{\text{lit}}, N_{\text{hyb}}) > \text{sim}(N_{\text{lit}}, N_{\text{mty}})$ holds consistently across 4 models, with a gap of 0.20–1.86 pp, aligning with human metonymicity ratings (hybrid 3.65 vs. metonymy 3.47 on a 5-point scale).

### Key Findings
- **Asymmetry Effect**: The metaphor → metonymy direction is robust (consistent across 4 datasets + 4 LLMs + human ratings), but the metonymy → metaphor direction is unstable (hybrid augmentation only wins on VUA Verb / MOH-X and loses to pure metaphor augmentation on FLUTE / TroFi).
- **Explainable Mechanism**: Surprisal scores show hybrid noun token surprisal=12.81 ≈ metonymy=12.79, and verb surprisal=12.66 ≈ metaphor=11.38, proving that hybrids retain figurative intensity in both dimensions.
- **Framework Insensitivity to LLM**: Running the same framework with Llama-3.1-8B / GPT-OSS-20B / Qwen3-30B / Llama-3.1-70B / GPT-5 consistently yielded metonymy success rates between 72-75%. This indicates that the structural constraints of the pipeline (contiguity prompt + MLM scoring) are the primary drivers, not LLM capacity.

## Highlights & Insights
- **"Using MLM as a domain gate" is a reusable tactic**: Translating the difficult-to-prompt "intra-domain constraint" into BERT token-level log-likelihood functions as a training-free domain classifier, which is much simpler than fine-tuning a semantic relationship model.
- **Forcing-device explanation is elegant**: Elevating "why metonymy is more explicit in hybrids" from empirical observation to cognitive linguistic explanation via selectional preference creates a classic "experimental phenomenon + theoretical closure" loop.
- **Zero-cost hybrid splicing**: By exploiting the natural complementarity (metonymy changes nouns without altering syntax; metaphor changes predicates), the need for independent hybrid generation and alignment is eliminated. This is transferable to other figurative pairs like sarcasm + irony or hyperbole + simile.
- **Clean data augmentation setup**: Using a fixed 50% augmentation ratio + identical BERT hyperparameters + 8 benchmarks ensures the "hybrid > metonymy-only" conclusion is free from cherry-picking suspicions.

## Limitations & Future Work
- The authors admit covering only location-for-people / institution-for-people subject metonymy; object metonymy and part-whole relations are not included. The hybrid success rate (74%) also implies that 26% of generations fail, meaning quality is capped by the LLM pipeline.
- The inconsistency in the "metonymy → metaphor" direction during evaluation was not convincingly explained, with only a placeholder conclusion of "deeper semantic complexities" in Appendix B.
- Lack of explicit domain mapping labels prevents fine-grained research into "which domain pairs are more likely to trigger figurative fusion." Future directions include adding conceptual domain ontologies, expanding to object metonymy, and extending the pipeline to non-English languages (where metaphor/metonymy vary significantly).

## Related Work & Insights
- **vs PRINCIPLES / MERMAID (Metaphor Generation)**: MERMAID uses symbolism + discriminative decoding to control metaphoricity but focuses only on metaphor. Ours applies "controlled + flexible" logic to both metonymy and metaphor.
- **vs ChainNet (Maudslay et al. 2024)**: ChainNet encodes metonymy/metaphor into WordNet relationship chains; Ours provides sentence-level quadruplet data, making the two complementary.
- **vs ConMeC (Ghosh & Jiang 2025)**: The same authors' previous work focused only on common-noun metonymy classification. This paper upgrades to a "generation + analysis + augmentation" trinity, completing the infrastructure.

## Rating
- Novelty: ⭐⭐⭐⭐ First joint metonymy + metaphor dataset + empirical validation of the forcing-device hypothesis, though the pipeline (candidate + scoring + polish) has precedents in figurative generation literature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 downstream benchmarks + 4 LLM zero-shot + 5 LLM framework generalizations + cross-domain experiments + multi-angle validation via surprisal/embeddings.
- Writing Quality: ⭐⭐⭐⭐⭐ The closure between "experimental phenomena → cognitive linguistic explanation" is excellent, and the error analysis (Table 10) is thorough.
- Value: ⭐⭐⭐⭐ Open-sourced dataset and framework provide new infrastructure for figurative language NLP, though the limitation to location metonymy narrows immediate applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](exploring_concreteness_through_a_figurative_lens.md)
- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](../../ICCV2025/nlp_understanding/balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[ACL 2026\] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)

</div>

<!-- RELATED:END -->
