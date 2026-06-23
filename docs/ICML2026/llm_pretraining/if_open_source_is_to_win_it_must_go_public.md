---
title: >-
  [Paper Note] If open source is to win, it must go public
description: >-
  [ICML 2026][Pretraining][Paper Note] This ICML 2026 position paper argues that "open-source AI" in its current form cannot truly democratize AI access or provide public goods in the same way Linux or PyTorch did. It posits that open source can only succeed if embedded within "Public AI"—infrastructure for compute, inference, post-training, and data provid
tags:
  - ICML 2026
  - Pretraining
date: 2026-05-08
content_hash: 44cc24c36e5bebf6
---
# If open source is to win, it must go public

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2507.09296](https://arxiv.org/abs/2507.09296)  
**Code**: None (Position paper)  
**Area**: Others / AI Governance and Open Source Ecosystem  
**Keywords**: Open-source AI, Public AI, Infrastructure, Governance, Digital Public Goods

## TL;DR
This ICML 2026 position paper argues that "open-source AI" in its current form cannot truly democratize AI access or provide public goods in the same way Linux or PyTorch did. It posits that open source can only succeed if embedded within "Public AI"—infrastructure for compute, inference, post-training, and data provided by governments, national labs, universities, and non-profit institutions.

## Background & Motivation

**Background**: Over the past decade, open source (e.g., PyTorch, HuggingFace Transformers, OpenCLIP, Megatron-LM, lm-eval-harness) has become a cultural and technical norm in the ML community. Community projects (EleutherAI Pile/Pythia, LAION-5B, Stable Diffusion, OLMo, RedPajama, Marin) have reached or exceeded the capabilities of closed-source frontier labs at various points in time.

**Limitations of Prior Work**: The authors argue that the equation "Open Source = Democratizing AI" is breaking down in the foundation model era:

- *Pre-training Costs*: Modern large models require thousands of GPUs for weeks or months, as well as web-scale data and distributed engineering teams, which only a few large corporations or state-level institutions can afford.
- *Post-training Barriers*: Fine-tuning, alignment, tool integration, and prompt orchestration—the stages that make a model truly usable—are often closed-source. RLHF data is siloed by platforms and does not flow back to the community.
- *Inference Costs*: Unlike traditional open-source software, which has near-zero hosting costs, large model inference requires continuous GPU resources, orchestration systems, and cost management.
- *License Fragility*: "Open weights" $\neq$ open source. The LLaMA agreement contains restrictive and revocable clauses; Meta can stop releases or impose stricter limits at any time. OpenAI prohibits using its output to train competing products.
- *Transparency Gaps*: Releasing weights does not equate to releasing source code. Training data, cleaning decisions, RLHF processes, and compute configurations remain undisclosed, preventing external researchers from verifying safety claims or reproducing behaviors.
- *Safety and Governance*: Open-source models are often "research artifacts" rather than "deployment-ready," lacking sustained investment in red-teaming and alignment. Community contributions (evaluations, datasets, fine-tuning tricks) are frequently "co-opted" by closed-source frontier labs.
- *Coding Agent Case*: When open-source developers use subscription-based agents like Claude Code or Codex, prompts, iterative processes, feedback, source code snippets, and API keys are captured as implicit data labor by private harnesses.

**Key Challenge**: The prerequisites of the open-source software era (cycles of contribution-use-redistribution open to all, parity via commodity hardware) have failed in the LLM era. LLMs are essentially "impure public goods" or "club goods" that require scarce private complements (compute, energy, engineering teams) to activate. The authors use a metaphor: the book remains a non-excludable public good, but the catalog is so vast that the common person must hire a "private guide" to find it, rendering access club-like.

**Goal**: To explicitly propose the position that "Open-source AI is insufficient for democratizing AI and must be complemented by Public AI," providing four principles for Public AI, five categories of existing practical cases, and responses to five opposing viewpoints.

**Key Insight**: Drawing from the economics of public goods (Mazzucato, Reiss, Gries & Naudé) and STS/open-source research (Kelty, Weber, Eghbal), AI is reframed as "Digital Public Infrastructure (DPI)" similar to roads, libraries, water, and electricity, rather than just "another software library."

**Core Idea**: Institutional completion through the four principles of "Public AI = Public Support + Public Access + Public Accountability + Private Commitments" to fill the structural gaps in compute, post-training, inference, and governance left by pure open source.

## Method

As a position paper, there are no methodology experiments, but rather a clear argumentative structure organized across eight typical sections. It begins by presenting the tensions of open-source AI (Ideals vs. Commercial vs. New LLM Constraints), reviews the history of success in ML open-source software and AI projects, categorizes three types of challenges (Resources, Licensing, Governance), proposes the core claim and four principles, provides existence proofs via projects like BLOOM/Jean Zay, LAION/JUWELS, EuroLLM, and NDIF, responds to counter-arguments, and concludes with the title's thesis.

### Overall Architecture

The paper constitutes an argumentative chain: it diagnoses why pure open source fails in the LLM era, provides institutional complements via the four principles of Public AI, and assumes the responsibility of debating the five strongest counter-arguments. The methodological weapon throughout is public goods economics—introducing concepts like "impure public goods," "club goods," and lighthouse finance to the ML community, anchored by empirical evidence such as the potential end of the LLaMA family and the capture of user workflows by coding agents.

### Key Designs

**1. Three-dimensional Diagnosis: Why Open-source AI is Insufficient**

Ours structures the failure of pure open source into resource, licensing, and governance layers. In economic terms, "impure public goods / club goods" explain why open weights do not equal public goods; weights require private complements to activate. Specific cases like LLaMA's revocable licenses and OpenAI's restrictions on competitive training solidify the "open weight $\neq$ open source" argument. Finally, the coding agent case demonstrates a new form of co-optation where user contributions are captured by private harnesses as implicit data labor.

**2. Definition of the Four Principles of Public AI**

Ours converges "Public AI" into actionable institutional norms:
- *Public Support*: Public funding and infrastructure should cover not only pre-training but also inference, deployment, post-training, and data.
- *Public Access*: Researchers from the Global South, civic technologists, and local communities outside Big Tech must be able to build, adapt, and use competitive models.
- *Public Accountability*: Models and infrastructure should be provided, hosted, and maintained by institutions accountable to the public (governments, labs, universities, non-profits).
- *Private Commitments*: Encouraging or requiring private entities to make commitments regarding openness, safety, and community control.
This follows the DPI (Digital Public Infrastructure) paradigm already established for identity, payments, and data exchange.

**3. Responses to Five Opposing Viewpoints**

- *View 1: The market is working (OpenAI/Meta leading)*: Response: Access $\neq$ governance $\neq$ sovereignty. The potential discontinuation of open LLaMA versions proves private access can be unilaterally revoked.
- *View 2: Open source will eventually win*: Response: Current top models are mostly pre-trained by capital-rich private firms. Purely non-profit downloads are orders of magnitude lower; the exception (LAION) relies precisely on public supercomputes, proving the necessity of Public AI.
- *View 3: OSS + commercial hosting is enough*: Response: Commercial hosting is revocable; BLOOM on Jean Zay and openCLIP on JUWELS already demonstrate the need for public underwriting.
- *View 4: Regulation is better than public investment*: Response: Regulation curbs harm but does not ensure access or equal participation. Public AI proactively builds capacity.
- *View 5: Public AI will be inefficient or captured*: Response: Successes like GPS, the Internet, CERN, and W3C show that public technical infrastructure can succeed. Public AI can be a multilateral hybrid structure like "Airbus for AI."

## Key Experimental Results

As a position paper, no experiments were conducted, but key data were cited to support the arguments.

### Comparison of Model Downloads (Hugging Face, January 2026)

| Model | Monthly Downloads | Type | Implication |
|------|---------|------|------|
| LLaMA 3.1-8B | 6M | Private Open Source | Commercial labs dominate model distribution |
| EleutherAI Pythia | 900k | Pure Non-profit | Order of magnitude smaller than LLaMA |
| OLMo 3-7B | 170k | Academic Non-profit | ~35x smaller than LLaMA |
| LAION CLAP | 14M | Public Compute + Non-profit | Only counter-example at parity with private firms |
| openCLIP (Single variant) | 1M–2M | Public Compute + Non-profit | Cumulative >60M |

### Scale of Public AI Compute (European OpenEuroLLM Alliance)

| Dimension | Value | Description |
|------|------|------|
| Participating Institutions | 20 European institutions | Consortium scale |
| Compute Quota | >10M GPU-hours | EuroHPC strategic resource |
| Accessed Supercomputers | 4 | Leonardo / LUMI / JUPITER / MareNostrum5 |
| Current Model Quality | Still trails Qwen / DeepSeek / GPT-OSS | Authors admit public output lags behind frontier |

### Key Findings

- *Asymmetry between Capital and Public Investment*: Monthly downloads of a single private model (LLaMA 3.1-8B) roughly equal the cumulative results of all European public AI investments, highlighting the scale disadvantage of pure public paths.
- *Explanatory Power of the LAION Counter-example*: The LAION series proves that non-profits can produce world-class open-source models in multimodal domains if provided with public supercomputing support.
- *License Fragility as Reality*: Reported shifts in LLaMA's release strategy and the closure of free versions of private models serve as contemporary evidence against the "OSS + hosting is enough" stance.

## Highlights & Insights

- **Refining the "open weight $\neq$ open source" concept**: Ours clearly distinguishes between source (the blueprint) and weight (the artifact), clarifying the confusion regarding whether LLaMA or Mistral are "truly open source."
- **Coding agents as "Co-optation 2.0"**: Using late-2025 cases like Claude Code, ours illustrates how developer contributions (prompts, feedback, file system context) are captured as "implicit data labor" without governance mechanisms.
- **Designing AI as DPI**: By migrating existing DPI paradigms (identity, payments) to AI infrastructure, the ideological debate over "government-made models" is transformed into an executable engineering paradigm.
- **Spectrum-based Rebuttals**: The five views cover the full spectrum from market fundamentalism to public failure theories, providing a template for writing position papers.

## Limitations & Future Work

- *Thin Empirical Support*: The paper is positional and lacks systematic data-level evidence for claims such as "opaque safety claims."
- *Vague "Public" Boundaries*: The four principles do not clarify the hierarchy between international agencies, single states, or multilateral alliances. The tensions between models like "Airbus for AI" vs. "CERN for AI" are not fully explored.
- *Weak Global South Perspective*: Despite the Public Access principle, examples are heavily skewed toward European and US supercomputers.
- *Relationship between Governance and Regulation*: The interaction between Public AI and proactive regulation remains abstract and lacks concrete scenario demonstrations.
- *Future Directions*: Proposing a quantifiable "Public AI Index" to score projects based on the four principles and designing defense mechanisms against public failure modes like capture or inefficiency.

## Related Work & Insights

- **vs. Bommasani et al., 2024**: While that work focuses on policy frameworks for governing models, ours argues that governance alone is insufficient without public infrastructure providing the substrate.
- **vs. Widder et al., 2024**: Ours inherits the diagnosis that open source serves large corporations but provides a constructive institutional response: Public AI.
- **vs. Mazzucato's "Entrepreneurial State"**: Ours applies the argument of basic research as a public good to the full stack of AI (compute, data, inference, post-training).
- **Inspirations**: (1) Researchers should prioritize long-term accessibility over short-term free access; (2) Participation in public AI projects preserves research access to model internals; (3) Be wary of implicit data capture in developer workflows.

## Rating
- Novelty: ⭐⭐⭐⭐ Synthesizing "open weight $\neq$ open source," co-optation, and the DPI paradigm is a fresh angle in the ICML context.
- Experimental Thoroughness: ⭐⭐ Typical for a position paper, but quantitative data is relatively thin.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent structure, precise terminology, and a model for spectrum-based argumentation.
- Value: ⭐⭐⭐⭐ Provides actionable insights for researchers, contributors, and policymakers to restructure AI funding toward public interest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](../../ECCV2024/llm_pretraining/plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[ICML 2026\] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)
- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[ECCV 2024\] I Can't Believe It's Not Scene Flow!](../../ECCV2024/llm_pretraining/i_canapost_believe_itaposs_not_scene_flow.md)
- [\[ICML 2026\] Incremental BPE Tokenization](incremental_bpe_tokenization.md)

</div>

<!-- RELATED:END -->
