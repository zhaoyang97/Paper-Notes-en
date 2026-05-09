---
title: >-
  [Paper Note] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries
description: >-
  [ACL 2026][LLM Alignment][Jailbreak Attack] This paper demonstrates that domain-specific contexts (e.g., chemistry papers) selectively relax LLM safeguards on related harmful knowledge (vertical unlocking), while security research contexts trigger broad relaxation across all harmful categories (general unlocking). Based on these findings, the authors propose the Jargon attack framework, achieving over 93% attack success rate (ASR) on seven frontier models including GPT-5.2 and Claude-4.5.
tags:
  - ACL 2026
  - LLM Alignment
  - Jailbreak Attack
  - Safety Boundaries
  - Domain Context
  - Gray Zone
  - Multi-turn Adversarial
date: 2026-05-08
content_hash: 2f1d8eede08d6eee
---

# Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries

**Conference**: ACL 2026
**arXiv**: [2604.15717](https://arxiv.org/abs/2604.15717)
**Code**: [https://github.com/JerryHung1103/JARGON](https://github.com/JerryHung1103/JARGON)
**Area**: LLM Safety / Alignment
**Keywords**: Jailbreak Attack, Safety Boundaries, Domain Context, Gray Zone, Multi-turn Adversarial

## TL;DR

This paper demonstrates that domain-specific contexts (e.g., chemistry papers) selectively relax LLM safeguards on related harmful knowledge (vertical unlocking), while security research contexts trigger broad relaxation across all harmful categories (general unlocking). Based on these findings, the authors propose the Jargon attack framework, achieving over 93% attack success rate (ASR) on seven frontier models including GPT-5.2 and Claude-4.5.

## Background & Motivation

**State of the Field**: LLM alignment training teaches models *when to refuse* rather than *how to forget*; restricted knowledge remains encoded in model parameters and can be retrieved under appropriate contextual conditions. Early jailbreak methods have escalated from adversarial suffixes (GCG) to role-playing (DAN) to multi-turn progressive attacks (Crescendo).

**Limitations of Prior Work**: (1) The "researcher" persona used in existing jailbreak methods is too superficial — merely claiming "I am a researcher" lacks the depth of genuine domain expertise, and modern LLMs have learned to recognize such shallow impersonation. (2) However, LLMs cannot simply reject all domain-expert interactions without degrading the experience of legitimate professionals (security researchers need to discuss vulnerabilities; pharmacologists must reference controlled substances).

**Root Cause**: LLMs face a fundamental helpfulness–safety paradox — the same knowledge can be used to help or to harm, and models must infer intent from contextual signals, which creates exploitable gaps. Professional domain contexts push queries into a "gray zone" where the model struggles to determine whether to assist or refuse.

**Paper Goals**: (1) Systematically investigate how domain contexts influence LLM safety behavior; (2) distinguish between vertical unlocking (domain-specific) and general unlocking (cross-category via security research contexts); (3) design and evaluate a systematic attack framework alongside defensive strategies.

**Starting Point**: During pretraining, LLMs were exposed to extensive academic literature in which security researchers routinely discuss cross-category threats. This training data distribution leads models to form an implicit association — *security research framing → permission to discuss sensitive topics* — which constitutes an exploitable vulnerability.

**Core Idea**: Security research contexts occupy a uniquely privileged position within LLM safety boundaries: they represent both a legitimate professional need and a natural occasion to discuss cross-category threats, thereby triggering broader safety relaxation than ordinary domain contexts.

## Method

### Overall Architecture

Jargon operates in three phases: (1) **Establish a security research context** — present authentic excerpts (abstracts and method sections) from jailbreak papers; (2) **Build trust** — reinforce the academic interaction frame through benign scholarly exchanges (e.g., requesting summaries, asking about methodology); (3) **Extract harmful knowledge** — reframe the harmful target as an academic case study, exploiting the established trust and context.

### Key Designs

1. **Hierarchical Vulnerability Structure: Vertical vs. General Unlocking**

   - **Function**: Reveals two mechanisms by which domain contexts influence LLM safety boundaries.
   - **Mechanism**: *Vertical unlocking* — domain-specific papers (e.g., chemistry) yield significantly higher attack scores on domain-relevant harmful queries (e.g., chemical weapons) than on unrelated categories, producing a pronounced diagonal pattern in the heat map. *General unlocking* — a single jailbreak research paper applied across all 8 threat categories achieves attack scores that universally meet or exceed the diagonal level, because security research inherently involves discussing cross-category threats.
   - **Design Motivation**: This hierarchical structure explains why security research contexts are more dangerous than ordinary domain contexts — the "legitimacy" of the former extends across all harmful categories.

2. **Multi-turn Trust Building + Query Variant Generation**

   - **Function**: Systematically exploit general unlocking to conduct attacks.
   - **Mechanism**: The first $k$ turns (typically $k=2$) consist of benign academic questions about the paper, establishing a cooperative pattern. From turn $k+1$ onward, the harmful target $g$ is reframed as an academic case study. Because the model's refusal decision is unreliable in the gray zone, $V$ query variants are generated for each attack (half paraphrased, half reframed), evaluated in parallel, and the response with the highest harm score is selected.
   - **Design Motivation**: Query variant generation is critical — ablation experiments show it raises ASR on GPT-5.2 from 54% to 93%.

3. **Trajectory Memory**

   - **Function**: Accumulate and reuse successful attack experiences across targets.
   - **Mechanism**: A buffer of successful attack trajectories is maintained. When attacking a new target, semantically similar successful trajectories are retrieved via embedding cosine similarity and used as few-shot demonstrations. The buffer is initialized with seed trajectories and grows as attacks succeed.
   - **Design Motivation**: Successful attack strategies tend to transfer to semantically related targets; trajectory memory enables cumulative improvement in attack efficiency.

### Loss & Training

On the defense side: (1) **Policy-guided safeguarding** — a custom safety policy is designed to guide `gpt-oss-safeguard` in producing classification decisions and response guidance; (2) **Alignment fine-tuning** — a paired dataset (Jargon attacks + safety-guided responses) is constructed, and fine-tuning Qwen3-8B reduces ASR from 100% to 66% while preserving general capabilities on MMLU, HellaSwag, and GSM8K.

## Key Experimental Results

### Main Results

**Attack Success Rate (ASR %) on Seven Frontier LLMs**

| Method | GPT-5.2 | Claude-4.5 Sonnet | Claude-4.5 Opus | Gemini-3 Flash | DeepSeek-V3.2 | Qwen3-235B | LLaMA-4-Scout | Avg. |
|---|---|---|---|---|---|---|---|---|
| PAIR | 5 | 0 | 1 | 24 | 68 | 15 | 32 | 20.7 |
| AmpleGCG | 10 | 5 | 1 | 27 | 75 | 22 | 27 | 23.9 |
| Crescendo | 22 | 23 | 11 | 79 | 73 | 52 | 73 | 47.6 |
| FITD | 54 | 48 | 24 | 96 | 95 | 73 | 49 | 62.7 |
| X-Teaming | 59 | 18 | 22 | 94 | 100 | 99 | 97 | 69.9 |
| **Jargon** | **93** | **100** | **100** | **100** | **100** | **100** | **100** | **99.0** |

### Ablation Study

**Effect of Query Variant Generation**

| Model | Full Jargon | w/o Variants |
|---|---|---|
| GPT-5.2 | 93% | 54% |
| Claude-Sonnet-4.5 | 100% | 100% |
| LLaMA-4-Scout | 100% | 79% |

**Defense Effectiveness**

| Configuration | ASR ↓ | MMLU | HellaSwag | GSM8K |
|---|---|---|---|---|
| Qwen3-8B (original) | 100% | 0.730 | 0.749 | 0.882 |
| + Policy-guided safeguard | 61% | — | — | — |
| + Fine-tuning | 66% | 0.725 | 0.742 | 0.885 |

### Key Findings

- Jargon achieves 93% ASR on GPT-5.2, the hardest model to attack (the strongest baseline, X-Teaming, reaches only 59%), and 100% on the Claude-4.5 series (FITD reaches only 24%).
- Attack effectiveness is insensitive to context type: attack papers, defense papers, and security surveys all yield 96%+ ASR — the security research framing matters, not the specific content.
- Context length correlates positively with effectiveness: Full Paper > Abstract+Method > Abstract; longer contexts dilute the attention weight assigned to safety signals.
- Activation-space analysis confirms the "gray zone": Jargon queries occupy an intermediate region between benign and harmful clusters in MDS projections, where the model's refusal decisions are unreliable.
- Attention analysis reveals that after academic reframing, the attention weights on sensitive tokens decrease substantially, diluting safety detection signals.

## Highlights & Insights

- The "gray zone" concept is both profound and intuitive — safety boundaries are not sharp decision lines but gradual transition regions, which constitutes an important theoretical contribution to safety alignment.
- The vertical/general unlocking distinction is highly insightful, explaining why security research contexts are more dangerous than other professional domains.
- The defensive philosophy of "helpful but harmless" is preferable to blanket refusal — fine-tuning reduces ASR while preserving general capabilities.

## Limitations & Future Work

- The proposed defenses reduce ASR from 100% to 61–66%, which is far from a complete solution.
- Only academic papers are tested as context; formats such as technical blogs and industry reports remain unvalidated.
- The Knowledge Purification component may cause out-of-context content to appear more harmful, potentially inflating harm scores.
- The fundamental challenge is that LLMs cannot reliably distinguish the intent of "understanding threats to defend" from "understanding threats to attack."

## Related Work & Insights

- **vs. Crescendo**: Crescendo attacks via progressively escalating requests; Jargon attacks by establishing a genuine academic context. The latter substantially outperforms the former on the latest models (47.6% vs. 99.0%).
- **vs. X-Teaming**: X-Teaming employs multi-agent ensembling but achieves only 18–22% on the Claude-4.5 series; Jargon's academic context framing is more effective at bypassing front-end safety classifiers.
- **vs. PAIR**: PAIR is a single-turn prompt optimization method that becomes nearly ineffective against safety-enhanced models; Jargon's multi-turn trust-building strategy is substantially more effective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The "gray zone" concept and the discovery of the general unlocking mechanism constitute significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 target models, 5 baselines, activation/attention analysis, and defense exploration; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative logic from motivation to findings to method to defense is flawless.
- Value: ⭐⭐⭐⭐⭐ — Provides deep insight into the fundamental challenges of LLM safety alignment, with attack effectiveness demonstrated on the latest frontier models.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](../../AAAI2026/llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ICLR 2026\] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](../../ICLR2026/llm_alignment/align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)
- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](../../AAAI2026/llm_alignment/safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)
- [\[NeurIPS 2025\] Preference Learning with Lie Detectors can Induce Honesty or Evasion](../../NeurIPS2025/llm_alignment/preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)

<!-- RELATED:END -->
