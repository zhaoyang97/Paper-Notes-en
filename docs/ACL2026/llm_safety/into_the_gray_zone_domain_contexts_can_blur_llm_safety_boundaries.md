---
title: >-
  [Paper Note] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries
description: >-
  [ACL 2026][LLM Safety][Jailbreak attack] This paper discovers that domain-specific contexts (e.g., chemistry papers) selectively relax LLM protection for related harmful knowledge (Vertical Unlocking)…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Jailbreak attack"
  - "Safety boundaries"
  - "Domain contexts"
  - "Gray zone"
  - "Multi-turn adversarial"
date: 2026-05-08
content_hash: 81b76e748988b267
---

# Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries

**Conference**: ACL 2026  
**arXiv**: [2604.15717](https://arxiv.org/abs/2604.15717)  
**Code**: [https://github.com/JerryHung1103/JARGON](https://github.com/JerryHung1103/JARGON)  
**Area**: LLM Safety / Alignment  
**Keywords**: Jailbreak attack, Safety boundaries, Domain contexts, Gray zone, Multi-turn adversarial

## TL;DR

This paper discovers that domain-specific contexts (e.g., chemistry papers) selectively relax LLM protection for related harmful knowledge (Vertical Unlocking), while security research contexts trigger broad protection relaxation across all harmful categories (General Unlocking). Based on this, it proposes the Jargon attack framework, achieving a success rate of over 93% on seven frontier models including GPT-5.2 and Claude-4.5.

## Background & Motivation

**Background**: LLM alignment training teaches models "when to refuse" rather than "how to forget"; restricted knowledge remains encoded in model parameters and can be retrieved under appropriate contextual conditions. Early jailbreaking methods have continuously upgraded from adversarial suffixes (GCG) to role-playing (DAN), and then to multi-turn progressive attacks (Crescendo).

**Limitations of Prior Work**: (1) The "researcher" roles used in existing jailbreak methods are too superficial—simply claiming "I am a researcher" lacks the depth of real professional expertise, which modern LLMs have learned to identify; (2) However, LLMs cannot simply reject all professional domain interactions, otherwise it would harm the user experience of legitimate professionals (security researchers need to discuss vulnerabilities, and pharmacologists must cite controlled substances).

**Key Challenge**: LLMs face a fundamental "helpfulness-safety" paradox—the same knowledge can be used to help or harm. The model must infer intent from contextual signals, but this creates exploitable gaps. Professional domain contexts push queries into a "gray zone," where models struggle to judge whether they should provide assistance or refuse.

**Goal**: (1) Systematically study how domain contexts affect LLM safety behavior; (2) Distinguish between Vertical Unlocking (domain-specific) and General Unlocking (cross-domain security research); (3) Design and evaluate a systematic attack framework and defense strategies.

**Key Insight**: During pre-training, LLMs were exposed to a large volume of academic literature where security researchers routinely discuss cross-category threats. This training data distribution causes the model to establish an implicit association of "security research framework $\rightarrow$ sensitivity topics allowed," making it an exploitable vulnerability.

**Core Idea**: Security research contexts occupy a specialized privileged position in LLM safety boundaries—it is both a legitimate professional requirement and naturally involves cross-category threat discussions, thus triggering a broader protection relaxation than ordinary domain contexts.

## Method

### Overall Architecture

Jargon operates in three stages: (1) Establish security research context—presenting real jailbreak paper abstracts and methodology sections; (2) Build trust—consolidate the academic interaction framework through benign academic discussions (e.g., requesting summaries, asking about methodology); (3) Extract harmful knowledge—reconstruct harmful targets as academic case studies, utilizing established trust and context for the attack.

### Key Designs

1.  **Hierarchical Vulnerability Structure of Vertical and General Unlocking**:
    - **Function**: Reveal two mechanisms by which domain contexts influence LLM safety boundaries.
    - **Mechanism**: Vertical Unlocking—domain-specific papers (e.g., chemistry) yield significantly higher attack scores for domain-related harmful queries (e.g., chemical weapons), with heatmaps showing a clear diagonal pattern. General Unlocking—a single jailbreak research paper applied to all 8 threat categories achieves or exceeds diagonal scores across the board, as security research naturally involves discussing cross-category threats.
    - **Design Motivation**: This hierarchical structure explains why security research contexts are more dangerous than ordinary domain contexts—the "legitimacy" of the former covers all harmful categories.

2.  **Multi-turn Trust Building + Query Variation Generation**:
    - **Function**: Systematically exploit General Unlocking for attacks.
    - **Mechanism**: In the first $k$ turns (usually $k=2$), send benign academic questions about the paper to establish a collaborative mode. From round $k+1$, reconstruct the harmful target $g$ as an academic case study. Since refusal decisions are uncertain in the gray zone, generate $V$ query variants (half paraphrases, half reframing) for each attack, evaluating them in parallel to take the response with the highest harm score.
    - **Design Motivation**: The query variation strategy is critical—ablation experiments show that variant generation increases ASR from 54% to 93% on GPT-5.2.

3.  **Trajectory Memory**:
    - **Function**: Accumulate and reuse successful experiences across different attack targets.
    - **Mechanism**: Maintain a buffer of successful attack trajectories. When attacking a new target, retrieve semantically similar successful trajectories via embedding cosine similarity to serve as few-shot demonstrations. The buffer is initialized with seed trajectories and grows with attack successes.
    - **Design Motivation**: Successful attack strategies often have transfer value for semantically similar targets; trajectory memory achieves cumulative improvements in attack efficiency.

### Loss & Training

Regarding defense: (1) Policy-guided protection—designing customized safety policies to guide `gpt-oss-safeguard` to output classification decisions and response guidance; (2) Alignment fine-tuning—constructing a paired dataset (Jargon attacks + safety-guided responses). Fine-tuning on Qwen3-8B reduced ASR from 100% to 66% while maintaining general performance on MMLU, HellaSwag, and GSM8K.

## Key Experimental Results

### Main Results

**Attack Success Rate (ASR %) on Seven Frontier LLMs**

| Method | GPT-5.2 | Claude-4.5 Sonnet | Claude-4.5 Opus | Gemini-3 Flash | DeepSeek-V3.2 | Qwen3-235B | LLaMA-4-Scout | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PAIR | 5 | 0 | 1 | 24 | 68 | 15 | 32 | 20.7 |
| AmpleGCG | 10 | 5 | 1 | 27 | 75 | 22 | 27 | 23.9 |
| Crescendo | 22 | 23 | 11 | 79 | 73 | 52 | 73 | 47.6 |
| FITD | 54 | 48 | 24 | 96 | 95 | 73 | 49 | 62.7 |
| X-Teaming | 59 | 18 | 22 | 94 | 100 | 99 | 97 | 69.9 |
| **Jargon** | **93** | **100** | **100** | **100** | **100** | **100** | **100** | **99.0** |

### Ablation Study

**Impact of Query Variation Generation**

| Model | Full Jargon | w/o Variation |
| :--- | :--- | :--- |
| GPT-5.2 | 93% | 54% |
| Claude-Sonnet-4.5 | 100% | 100% |
| LLaMA-4-Scout | 100% | 79% |

**Defense Effectiveness**

| Configuration | ASR ↓ | MMLU | HellaSwag | GSM8K |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B (Original) | 100% | 0.730 | 0.749 | 0.882 |
| + Policy-guided | 61% | — | — | — |
| + Fine-tuning | 66% | 0.725 | 0.742 | 0.885 |

### Key Findings

- Jargon achieves 93% ASR on the hardest-to-break GPT-5.2 (strongest baseline X-Teaming only 59%) and 100% on the Claude-4.5 series (FITD only 24%).
- Attack effectiveness is insensitive to context types: attack papers, defense papers, and security surveys all reach 96%+ ASR—the key is the security research framework rather than specific content.
- Context length is positively correlated: Full Paper > Abstract+Method > Abstract; longer contexts dilute the attention share of safety signals.
- Activation space analysis reveals the "gray zone": Jargon queries are located between benign and harmful regions in MDS projections, where model refusal decisions are unreliable.
- Attention analysis: After reconstruction with academic context, the attention weights of sensitive tokens decrease significantly, diluting safety detection signals.

## Highlights & Insights

- The "gray zone" concept is profound and intuitive—safety boundaries are not clear decision lines but gradient transition zones, contributing significantly to the theoretical understanding of safety alignment.
- The distinction between Vertical/General Unlocking is highly insightful—explaining why security research contexts are more dangerous than other professional domains.
- The defense strategy's philosophy of "helpful but harmless" is superior to "blanket refusal"—reduced ASR while maintaining general capabilities after fine-tuning.

## Limitations & Future Work

- Defense strategies reduced ASR from 100% to 61-66%, which is still far from solving the problem.
- Only academic papers were tested as context; formats such as technical blogs and industry reports were not verified.
- The Knowledge Purification component might make out-of-context content appear more harmful, leading to inflated harm scores.
- The fundamental issue lies in the LLM's inability to truly distinguish between the intent of "understanding threats for defense" and "understanding threats for attack."

## Related Work & Insights

- **vs Crescendo**: Crescendo attacks via progressive escalation, while Jargon attacks by establishing a real academic context; the latter far exceeds the former on the latest models (47.6% vs 99.0%).
- **vs X-Teaming**: X-Teaming uses multi-agent integration but only reaches 18-22% on the Claude-4.5 series; Jargon's academic context framework is more effective at bypassing frontend safety classifiers.
- **vs PAIR**: PAIR is a single-turn prompt optimization that almost completely fails against safety-enhanced models, whereas Jargon's multi-turn trust-building strategy is more effective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The "gray zone" concept and discovery of the general unlocking mechanism are significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 10 target models, 5 baselines, activation/attention analysis, and defense exploration.
- Writing Quality: ⭐⭐⭐⭐⭐ Perfect narrative logic from motivation to findings, method, and defense.
- Value: ⭐⭐⭐⭐⭐ Provides profound insights into the fundamental challenges of LLM safety alignment, with attack effectiveness remaining valid on the latest frontier models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness](purging_the_gray_zone_latent-geometric_denoising_for_precise_knowledge_boundary_.md)
- [\[ACL 2026\] Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning](unlearners_can_lie_evaluating_and_improving_honesty_in_llm_unlearning.md)
- [\[ACL 2026\] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)
- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)

</div>

<!-- RELATED:END -->
