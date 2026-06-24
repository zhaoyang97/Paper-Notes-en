---
title: >-
  [Paper Note] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment
description: >-
  [AAAI 2026][Multi-Agent][interpretable alignment] This paper proposes ARCANE, a framework that formulates alignment as a multi-agent collaboration problem. A manager agent learns to generate natural-language rubrics (weighted verifiable criterion sets) through dialogue with stakeholders, which serve as interpretable proxy reward functions for a worker agent. Via two-stage SFT+GSPO training, the framework enables test-time configurable alignment…
tags:
  - "AAAI 2026"
  - "Multi-Agent"
  - "interpretable alignment"
  - "rubric learning"
  - "multi-agent collaboration"
  - "GSPO"
  - "test-time alignment"
date: 2026-05-08
content_hash: 89d00b3f69caf6ff
---

# ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment

**Conference**: AAAI 2026
**arXiv**: [2512.06196](https://arxiv.org/abs/2512.06196)  
**Code**: [https://github.com/DeepFlow-research/manager_agent_gym](https://github.com/DeepFlow-research/manager_agent_gym)  
**Area**: LLM Agent
**Keywords**: interpretable alignment, rubric learning, multi-agent collaboration, GSPO, test-time alignment

## TL;DR
This paper proposes ARCANE, a framework that formulates alignment as a multi-agent collaboration problem. A manager agent learns to generate natural-language rubrics (weighted verifiable criterion sets) through dialogue with stakeholders, which serve as interpretable proxy reward functions for a worker agent. Via two-stage SFT+GSPO training, the framework enables test-time configurable alignment, improving mean return from 0.58 to 0.74 (N=8) on the GDPVal benchmark with the GSPO variant.

## Background & Motivation

**Background**: RLHF is the dominant paradigm for LLM alignment, but it encodes preferences statically at training time and cannot adapt to shifting stakeholder preferences. Test-time reward models (e.g., GenRM, GRAM) enable dynamic evaluation but lack transparency—users cannot inspect what criteria are used or how they are weighted.

**Limitations of Prior Work**:
   - **Rigidity of RLHF**: Optimizes fixed training-time preferences; retraining is required when preferences drift. In multi-agent deployments, distributed preferences are even harder to capture statically.
   - **Opacity of test-time methods**: GenRM/GRAM output scalar or textual judgments without revealing which criteria drive the evaluation.
   - **Static nature of existing rubric methods**: Auto-Rubric, RaR, and similar approaches assume rubrics are given a priori rather than learned, and cannot adapt to evolving preferences.

**Key Challenge**: Alignment must simultaneously satisfy interpretability (auditable by stakeholders), configurability (adjustable at test time), and effectiveness (genuinely improving output quality), yet existing methods satisfy at most two of these properties.

**Goal**: Treat rubric generation itself as a policy optimization problem, enabling the manager agent to learn to distill interpretable, verifiable, weighted criterion sets from stakeholder dialogues.

**Key Insight**: Drawing on utility theory, the paper decomposes a stakeholder's latent utility function into a linear combination of weighted verifiable criteria, and interactively "reconstructs" the utility function through manager–stakeholder dialogue.

**Core Idea**: Alignment = manager learning to generate natural-language rubrics + worker executing tasks according to rubrics + stakeholder adjusting weights at test time.

## Method

### Overall Architecture
A three-role architecture: the Stakeholder holds the true utility function $U^*$ → the Manager distills a rubric $R$ through dialogue → the Worker executes the task according to $R$ to generate output $y$. The rubric $R = \{(c_j, w_j)\}_{j=1}^M$ is a set of weighted verifiable criteria, each with a corresponding verifier $\nu_j$. The proxy utility $\hat{u}_\phi(y|x) = \sum_j w_j \nu_j(c_j, x, y)$ serves as an interpretable approximation of $U^*$.

### Key Designs

1. **Rubric Representation and Verifiers**:

    - Function: Decompose preferences into structured, verifiable natural-language criteria.
    - Mechanism: Each criterion $c_j$ is a natural-language description (e.g., "includes citations to recent empirical studies"), with weight $w_j \in [0,1]$ and $\sum w_j = 1$. Verifiers may be rule-based (deterministic checks) or model-based (LLM/classifier evaluation of semantic properties).
    - Design Motivation: Linear weighting renders utility decomposable and auditable; natural-language criteria make rubrics accessible to non-technical stakeholders.

2. **Stakeholder–Manager Collaborative Dialogue**:

    - Function: The manager poses questions to the stakeholder to elicit latent preferences, then synthesizes a rubric.
    - Mechanism: $R = \mathfrak{D}_\phi(x, q_{1:T}, a_{1:T})$, optimizing an objective that incorporates interaction costs: $\max_{\pi_M} \mathbb{E}[U^*(y|x) - \lambda_{\text{clarify}} C_{\text{clarify}} - \lambda_{\text{compute}} C_{\text{compute}}]$
    - Design Motivation: Modeled as a "one-shot cooperative game under partial observability"—the stakeholder exposes limited, noisy information through language, and the manager must infer a faithful structured approximation.

3. **Two-Stage Training (SFT + GSPO)**:

    - **Stage I (SFT)**: A large reasoning model generates synthetic dialogues and reference rubrics; the manager is warm-started with standard language modeling loss.
    - **Stage II (GSPO)**: The manager is treated as a stochastic policy; $K$ rubrics are sampled per task, and the stakeholder utility obtained after worker execution serves as the reward. Sequence-level importance ratios $s_k(\phi)$ (rather than token-level) are used, along with KL regularization, clarification cost $C_{\text{clarify}}$, and computation cost $C_{\text{compute}}$. A prioritized experience replay mechanism focuses training on low-return episodes.
    - Design Motivation: SFT avoids cold-start; GSPO directly optimizes end-to-end utility rather than imitating reference rubrics.

4. **Test-Time Rubric Guidance**:

    - Function: Use learned rubrics to guide the worker at test time without gradient updates.
    - Mechanism: Supports Best-of-K sampling (selecting the best output by rubric score), importance resampling, and tree/beam search. Stakeholders can directly edit criteria $\{c_j\}$ and weights $\{w_j\}$.
    - Design Motivation: The interpretability of rubrics allows humans to directly modify the alignment direction at inference time.

### Loss & Training
- SFT Loss: Standard next-token prediction with system prompts and task inputs masked.
- GSPO Loss: PPO-style clipped objective + KL divergence regularization + clarification cost + computation cost (Equations 12–15).
- Prioritized Experience Replay: At each epoch, episodes in the N-th percentile of lowest return are replayed.

## Key Experimental Results

### Main Results

GDPVal benchmark, 219 tasks (175 train + 44 evaluation), covering multi-step reasoning and tool use:

| Method | Mean Return (N=1) | Mean Return (N=8) |
|--------|-------------------|-------------------|
| No Rubric | 0.58±0.01 | 0.58±0.02 |
| SFT Model | 0.59±0.09 | 0.68±0.03 |
| GSPO Model | 0.62±0.12 | **0.74±0.03** |
| Oracle Rubric | 0.70±0.12 | 0.81±0.03 |

### Ablation Study (Faithfulness — NDCG@8)

| Method | Mean NDCG@8 | Note |
|--------|-------------|------|
| No-Conv (Base) | 0.7998 | Base rubric without stakeholder dialogue |
| SFT Rubric | 0.8103 | Manager trained with SFT |
| GSPO Rubric | **0.8722** | Ranking consistency significantly improved after GSPO training |

### Key Findings
- **GSPO > SFT**: Statistically significant (Wilcoxon p=0.0182); GSPO rubric achieves mean return 0.74 vs. SFT 0.68 at N=8.
- **Parallel scaling curves**: The best-of-N scaling slopes of SFT, GSPO, and Oracle are nearly identical (~+0.03 per doubling of N), indicating that learned rubrics approximate the oracle scoring function.
- **Domain variation**: Subjective/language-intensive tasks benefit most (content/communication +11.5%, legal +12.5%), while operational tasks show a slight decline (−8.1%).
- **Interpretability preserved**: GSPO rubrics contain approximately 12 criteria per rubric with 17–18 tokens per criterion, closely matching the structural profile of Oracle rubrics.

## Highlights & Insights
- **Framing rubric generation as policy optimization**: Rather than assuming rubrics are given, the framework trains an agent to learn how to generate them—a meaningful advance over the "rubric-as-reward" paradigm.
- **Sequence-level importance ratios in GSPO**: More appropriate than GRPO's token-level ratios for structured outputs such as rubrics, mitigating variance issues in long sequences.
- **Cost-aware alignment**: Incorporating stakeholder interaction costs and computation costs into the optimization objective prevents excessive clarification or overly complex rubrics.

## Limitations & Future Work
- **Single-worker evaluation only**: The framework is designed to support multiple workers, but experiments involve only a single worker, leaving multi-worker coordination dynamics unverified.
- **GDPVal consists of discrete episodic tasks**: Continuous long-horizon deployment evaluation is absent.
- **No causal regularization**: The manager may learn criteria that correlate with utility without being causally responsible for it (spurious correlations).
- **No-Rubric baseline uses an RLHF model**: This indicates insufficient training-time alignment, but no direct comparison with test-time methods such as GenRM is provided.

## Related Work & Insights
- **vs. RLHF/DPO**: These methods encode preferences statically at training time and do not support test-time configuration; ARCANE's rubrics can be manually edited or automatically adjusted at inference time.
- **vs. GenRM/GRAM**: These provide dynamic evaluation but lack transparency; ARCANE's rubrics are structured natural language that can be audited and modified.
- **vs. Auto-Rubric/RaR**: These assume statically given rubrics and do not learn from stakeholder interaction; ARCANE learns to generate the rubrics themselves.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Framing rubric generation as a multi-agent RL problem is an original contribution; the theoretical framework (utility theory + bilevel optimization) is rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ RQ1–3 are clearly structured with statistical significance tests, but only 44 evaluation tasks are used and direct comparison with methods such as GenRM is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical formalization is rigorous; the logical chain from problem definition to method to experiments is complete.
- Value: ⭐⭐⭐⭐⭐ The integration of interpretability, configurability, and effectiveness in a single alignment framework carries important implications for real-world LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Efficient and Interpretable Multi-Agent Communication](../../ICLR2026/multi_agent/learning_efficient_and_interpretable_multi-agent_communication.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)
- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)

</div>

<!-- RELATED:END -->
