---
title: >-
  [Paper Note] Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment
description: >-
  [ICLR 2026][IR evaluation] This paper proposes DREAM — a multi-agent, multi-round debate framework with opposing-stance initialization for IR relevance annotation: cases with consensus are automatically labeled…
tags:
  - "ICLR 2026"
  - "IR evaluation"
  - "multi-agent debate"
  - "relevance annotation"
  - "human-AI collaboration"
  - "BRIDGE benchmark"
date: 2026-05-08
content_hash: 17643d770ea3bd3f
---

# Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment

**Conference**: ICLR 2026
**arXiv**: [2602.06526](https://arxiv.org/abs/2602.06526)
**Code**: [GitHub](https://github.com/DISL-Lab/DREAM-ICLR-26)
**Area**: Other
**Keywords**: IR evaluation, multi-agent debate, relevance annotation, human-AI collaboration, BRIDGE benchmark

## TL;DR

This paper proposes DREAM — a multi-agent, multi-round debate framework with opposing-stance initialization for IR relevance annotation: cases with consensus are automatically labeled, while disagreements are escalated to human annotators (aided by debate history). DREAM achieves 95.2% balanced accuracy with only 3.5% human escalation. Based on this framework, the BRIDGE benchmark is constructed, uncovering 29,824 missing relevant annotations absent from existing benchmarks (428% of the original annotations), and correcting ranking bias in retrieval systems as well as retrieval-generation performance misalignment in RAG evaluation.

## Background & Motivation

**Background**: Information retrieval (IR) evaluation relies heavily on manually annotated query-chunk relevance judgments. Due to the high cost of annotation, only a small number of documents are labeled in practice, leaving a large volume of unannotated relevant documents — so-called "holes" — treated as irrelevant by default. These holes introduce systematic bias into evaluation results, causing certain retrievers to be underestimated when they happen to retrieve relevant but unannotated documents.

**Limitations of Prior Work**:

1. **Overconfidence in fully automatic LLM annotation**: LLMJudge (single-agent) achieves only 73.9% balanced accuracy, with severely insufficient recall on the irrelevant class (50.2%) — exhibiting a strong tendency to label documents as "relevant."
2. **Low efficiency of confidence-based human-AI hybrid methods**: Methods such as LARA use LLM token probabilities for uncertainty estimation, but suffer from poor calibration — requiring 50% human escalation to match DREAM's accuracy at 3.5% escalation.
3. **Cascading effects of holes**: Holes in IR benchmarks not only distort retrieval system rankings but also cause retrieval-generation misalignment in RAG evaluation — strong retrievers are misidentified as poor ones, and good generated outputs are incorrectly attributed to the model's internal knowledge.
4. **Fundamental limitations of single-agent judgment**: Regardless of how finely confidence is calibrated, a single model perspective cannot overcome systematic bias.

**Key Challenge**: A high-accuracy, low-human-cost annotation method is needed. Fully automatic approaches are insufficiently accurate (73.9%), while confidence-based hybrid methods suffer from unreliable calibration and still require substantial human effort.

**Goal**: Replace single-agent judgment with multi-agent debate. Two agents are initialized with opposing stances → engage in multi-round mutual critique → consensus yields a high-confidence automatic label (a more reliable signal than single-model confidence) → disagreement is escalated to human annotators (assisted by debate history).

## Method

### Overall Architecture

The DREAM pipeline consists of three stages:

1. **Opposing-stance initialization**: Agent $m_1$ is assigned the "relevant" stance $s_1$, and Agent $m_2$ is assigned the "irrelevant" stance $s_2$.
2. **Multi-round debate with mutual critique**: In each round, both agents review each other's arguments, extract evidence sentences, and generate updated labels and reasoning.
3. **Consensus/escalation decision**: Agreement → adopt the consensus label; persistent disagreement → escalate to human arbitration along with the debate history.

Formally:

$$\text{DREAM}(q,c) = \begin{cases} y_1^j, & \exists j \leq R \text{ s.t. } y_1^j = y_2^j \text{ (consensus reached)} \\ \text{Human}(q, c, h^R), & \text{otherwise (persistent disagreement)} \end{cases}$$

where $R$ is the maximum number of debate rounds (default 2) and $h^R$ is the debate history at the final round.

### Key Design 1: Opposing-Stance Initialization

Forcing the two agents to begin from opposing stances is the core design of DREAM. Its roles are:

- **Preventing premature consensus**: If both agents start from a neutral position, LLMs' overconfidence tendency leads them to quickly converge — potentially on an incorrect answer.
- **Surfacing conflicting evidence**: Opposing stances compel each agent to deeply explore evidence supporting its own position and challenge the other's.
- **Eliminating single-perspective bias**: Ensures that both "relevant" and "irrelevant" possibilities are thoroughly argued.

Experiments confirm that the order of stance initialization does not affect results (no order dependency).

### Key Design 2: Agreement-Based Escalation

This approach is fundamentally distinct from confidence-based escalation strategies such as LARA:

- **Agreement signal vs. confidence score**: Multi-agent consensus is more reliable than a single model's (often poorly calibrated) confidence score.
- **No calibration training required**: There is no need to train a confidence calibration model on human-annotated data.
- **No threshold tuning required**: No manual escalation threshold is needed — consensus yields automatic annotation; disagreement triggers escalation.
- **Precision comparison**: LARA achieves only 82.1% bAcc at 3.5% escalation, while DREAM reaches 95.2% under the same condition.

### Key Design 3: Debate History-Augmented Human Review

When a case is escalated to human annotators, DREAM provides the complete debate history as an auxiliary resource:

- Human annotators receive both agents' arguments, extracted evidence sentences, and reasoning processes.
- There is no need to analyze the original documents from scratch — annotators directly review structured pro/con argumentation.
- Experimental validation: human annotation bAcc improves from 87.3% to 92.0% with debate history, and inter-annotator agreement (Fleiss κ) increases from 0.50 to 0.62.

## Key Experimental Results

### Main Results: Annotation Accuracy and Escalation Rate

| Method | Irrelevant Recall | Relevant Recall | bAcc | Escalation Rate |
|--------|------------------|----------------|------|----------------|
| LLMJudge | 50.2% | 97.5% | 73.9% | 0.0% |
| LARA (3.5%) | 74.5% | 89.6% | 82.1% | 3.5% |
| LARA (12.5%) | 76.1% | 91.6% | 83.9% | 12.5% |
| LARA (50%) | 94.1% | 98.4% | 96.3% | 50.0% |
| Human-Only (MTurk) | 89.9% | 97.8% | 93.8% | 100.0% |
| **DREAM** | **91.9%** | **98.4%** | **95.2%** | **3.5%** |

DREAM achieves 95.2% bAcc with only 3.5% human escalation, surpassing Human-Only (93.8%). LARA requires 50% human escalation to approach this level.

### Ablation Study: Debate Rounds and Arbitration Strategy

| Setting | Arbitrator | Irrelevant Recall | Relevant Recall | bAcc |
|---------|-----------|------------------|----------------|------|
| DREAM (R=1) | LLM | 82.9% | 97.2% | 90.0% |
| DREAM (R=2) | LLM | 90.0% | 96.7% | 93.3% |
| DREAM (R=3) | LLM | 90.8% | 95.7% | 93.2% |
| **DREAM (R=2)** | **Human** | **91.8%** | **98.4%** | **95.1%** |

Two debate rounds suffice for saturation (R=3 yields no additional gain). Human arbitration (95.1%) significantly outperforms LLM arbitration (93.3%), validating the AI-human collaboration strategy.

### BRIDGE Benchmark Construction

| Metric | Value |
|--------|-------|
| Total annotations | 116,622 |
| Automatic annotations (agent consensus) | 112,566 (96.5%) |
| Human annotations (agent disagreement) | 4,056 (3.5%) |
| Discovered missing relevant chunks (holes) | **29,824** |
| Gold chunks in original annotations | 6,976 |
| Holes as proportion of original annotations | **428%** |
| Human annotation cost | ~$506 |
| Cost reduction vs. Human-Only | 200× |
| Speed improvement vs. Human-Only | 3.5–7× |

### Impact of Holes: Retrieval System Re-Ranking

| Metric | Original Benchmark | BRIDGE | Change |
|--------|-------------------|--------|--------|
| Average Hole@10 | 17.1% | Corrected | Eliminated |
| System ranking changes | — | 20/25 systems re-ranked | Significant |
| RAGAlign@10 (average) | 0.70 | **0.84** | +0.14 |
| RAGAlign Pearson correlation | — | **0.985** | Highly aligned |

After correcting holes, retrieval-generation alignment (RAGAlign) improves from 0.70 to 0.84, with a Pearson correlation of 0.985. This demonstrates that retrieval-generation misalignment in prior IR evaluation partly stems from systematic underestimation of retrieval metrics.

## Highlights & Insights

- **Core insight — Agreement > Confidence**: Multi-agent consensus is a more reliable quality signal than single-model confidence. LARA requires 14× the human effort to match DREAM's accuracy; the root cause is the fundamentally unreliable calibration of LLM confidence scores.
- **Dual value of debate history**: It not only enables agents to converge efficiently within two rounds, but also serves as an auxiliary resource that improves human annotation quality from 87.3% to 92.0% — achieving genuine AI-human collaboration rather than a simple "fallback to humans when AI fails" paradigm.
- **The striking scale of 29,824 holes**: The original benchmark contains only 6,976 gold annotations; the missing annotations discovered by DREAM amount to 428% of the original — indicating that mainstream IR benchmark evaluations carry systematic bias.
- **A new explanation for retrieval-generation misalignment**: Previously attributed to "conflicts between external and internal knowledge," this paper reveals an overlooked contributing factor — retrieval performance itself has been systematically underestimated.

## Limitations & Future Work

- Increasing the number of agents actually reduces accuracy (harder to reach consensus on relevant cases).
- The evaluation set of 700 pairs is relatively limited in scale.
- The framework depends on Llama3.3-70B; switching to a different model may require re-validation.
- For highly ambiguous borderline cases, debate may still fail to resolve disagreement.

## Rating

- Novelty: ⭐⭐⭐⭐ Multi-agent debate combined with agreement-based escalation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation + BRIDGE construction + retrieval re-ranking + RAG alignment analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with well-motivated problem definition and methodology
- Value: ⭐⭐⭐⭐⭐ Important methodological advance in IR evaluation + practical impact of the BRIDGE benchmark

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](missing_mass_for_differentially_private_domain_discovery.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](../../AAAI2026/others/area-optimal_control_strategies_for_heterogeneous_multi-agen.md)
- [\[CVPR 2026\] AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](../../CVPR2026/others/assistmimic_physics_grounded_humanoid_assistance.md)
- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](../../ICCV2025/others/edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)

</div>

<!-- RELATED:END -->
