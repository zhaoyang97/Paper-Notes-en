---
title: >-
  [Paper Note] Language Models and Logic Programs for Trustworthy Tax Reasoning
description: >-
  [AAAI 2026][LLM/NLP][Statutory Reasoning] This paper reframes tax law reasoning as a semantic parsing task, where LLMs translate statutory text and case facts into Prolog logic programs that are subsequently executed by a symbolic solver. By combining gold-standard statute translations, retrieval-augmented case examples, and self-consistency checks, the system achieves 86/100 accuracy on the SARA dataset while reducing estimated deployment cost to \$15.78 per person — less than 6% of the average U.S. tax filing cost.
tags:
  - AAAI 2026
  - LLM/NLP
  - Statutory Reasoning
  - Tax Computation
  - Neuro-Symbolic Systems
  - Prolog
  - Semantic Parsing
  - LLM + Symbolic Solver
date: 2026-05-08
content_hash: 816e732b4f1800f8
---

# Language Models and Logic Programs for Trustworthy Tax Reasoning

**Conference**: AAAI 2026
**arXiv**: [2508.21051](https://arxiv.org/abs/2508.21051)
**Code**: [GitHub](https://github.com/wjurayj/legal_logic_programs)
**Area**: NLP Understanding / Legal AI
**Keywords**: Statutory Reasoning, Tax Computation, Neuro-Symbolic Systems, Prolog, Semantic Parsing, LLM + Symbolic Solver

## TL;DR
This paper reframes tax law reasoning as a semantic parsing task, where LLMs translate statutory text and case facts into Prolog logic programs that are subsequently executed by a symbolic solver. By combining gold-standard statute translations, retrieval-augmented case examples, and self-consistency checks, the system achieves 86/100 accuracy on the SARA dataset while reducing estimated deployment cost to \$15.78 per person — less than 6% of the average U.S. tax filing cost.

## Background & Motivation

**Background**: Nearly every adult worldwide must file taxes annually. In the United States, this costs an average of \$270 and 13 hours per person. Tax law reasoning requires the compositional application of overlapping rules combined with numerical computation, where errors incur financial penalties. Although LLMs demonstrate general mathematical reasoning capabilities, tax reasoning requires applying contingent rules provided at inference time rather than general arithmetic learned during training.

**Limitations of Prior Work**:
- **Unreliable direct computation by LLMs**: Even GPT-4/o3 achieves only 56–76% accuracy when directly computing tax liabilities, and errors carry real financial consequences.
- **Lack of auditability**: The causal relationship between an LLM's chain-of-thought (CoT) and its final answer is unreliable and cannot serve as an audit trail.
- **Inability to express uncertainty**: LLMs always produce an answer and do not abstain when uncertain.

**Key Challenge**: Tax law reasoning demands high accuracy, auditability, and the capacity to abstain under uncertainty — precisely the three dimensions where LLMs are weakest.

**Goal**: How can LLMs and symbolic reasoning be integrated so that an automated tax filing system simultaneously satisfies high accuracy, auditability, and cost-effectiveness?

**Key Insight**: Reframe tax law reasoning as semantic parsing — LLMs handle translation from natural language to Prolog code, while a Prolog engine handles precise computation and reasoning traces.

**Core Idea**: LLMs perform translation (natural language → Prolog); a symbolic solver performs reasoning. The two components collaborate with a division of labor, augmented by self-consistency checking and abstention upon failure.

## Method

### Overall Architecture
Three progressively refined approaches: (1) **Direct** — the LLM directly computes tax liability; (2) **Parsed** — the LLM zero-shot translates statutes and case facts into Prolog, executed by SWI-Prolog; (3) **Few-Shot** — gold-standard statute Prolog is provided together with intelligently retrieved example case translations. Self-consistency checking (requiring two independent runs to agree before accepting an answer) can be applied on top of any method.

### Key Designs

1. **Zero-Shot Parsed**:

   - **Function**: The LLM translates statutory text and case facts into a Prolog program without any demonstrations.
   - **Mechanism**: The LLM is provided the full statutory text and case description and instructed to generate a Prolog program that can compute the tax liability of the target individual. SWI-Prolog executes the program; a 10-second timeout is treated as abstention.
   - **Design Motivation**: Prolog execution provides a natural abstention mechanism — a program that fails to execute signals system uncertainty, which is safer than an LLM that always produces an answer. The execution trace itself constitutes an auditable reasoning process.

2. **Few-Shot Parsing with Gold Statutes**:

   - **Function**: The LLM is provided human-translated statute Prolog together with the 5 most relevant already-translated cases as in-context demonstrations.
   - **Mechanism**: An instruction-based retrieval system (prompting o4-mini to rank the other 99 cases by logical-structural similarity) identifies the 5 most relevant precedent cases, whose gold-standard Prolog translations serve as few-shot examples. The task is thereby reduced to: given these examples, translate the facts of the current case into Prolog predicates.
   - **Design Motivation**: (1) Statute translation is a one-time investment reusable across an unlimited number of subsequent cases. (2) Relevant case examples teach the LLM the formalization conventions specific to the statutory system (Neo-Davidsonian event semantics, 61 predicates). (3) The complex joint parsing of statutes and cases is simplified to pure case-fact extraction.

3. **Self-Consistency Check**:

   - **Function**: Requires two independent runs to produce the same answer before accepting it.
   - **Mechanism**: Can be applied within the same method (Parsed+Parsed) or across methods (Direct+Parsed, Direct+Few-Shot). Any disagreement triggers abstention and referral to a human.
   - **Design Motivation**: Trades coverage for accuracy. Since the cost of an error far exceeds the cost of manual filing (\$270), abstention is preferred over incorrect output.

### Loss & Training / Evaluation Methodology
**Break-Even Price**: A novel economic evaluation metric.
- Underreporting actual tax liability by more than $\max(\$5{,}000,\ 10\% \times \text{actual liability})$: penalty = 20% of the underreported amount.
- Overreporting: cost = overreported amount.
- Abstention: cost = \$270 (average U.S. tax filing cost).
- Otherwise: \$0.
- Break-Even Price = average cost across all cases, representing the minimum service price at which deployment breaks even.

## Key Experimental Results

### Main Results

**Without gold statutes (zero-shot parsing and direct computation)**:

| Model | Method | Correct | Wrong | Abstain | Break-Even Price |
|-------|--------|---------|-------|---------|-----------------|
| DeepSeek-R1 | Direct | 74 | 26 | 0 | \$304.29 |
| DeepSeek-R1 | Parsed | 38 | 10 | 52 | \$249.64 |
| DeepSeek-R1 | Direct+Direct | 66 | 12 | 22 | **\$94.20** |
| o3 | Parsed | 75 | 15 | 10 | **\$47.43** |
| GPT-5 | Direct | 76 | 24 | 0 | \$299.11 |

**With gold statutes and retrieval-augmented examples**:

| Model | Method | Correct | Wrong | Abstain | Break-Even Price |
|-------|--------|---------|-------|---------|-----------------|
| GPT-4.1 | Few-Shot | 87 | 8 | 5 | \$247.99 |
| GPT-4.1 | Few-Shot+Few-Shot | 81 | 5 | 14 | **\$40.08** |
| GPT-5 | Few-Shot | **86** | 9 | 5 | **\$15.78** |
| o3 | Few-Shot | 81 | 13 | 6 | \$60.26 |
| DeepSeek-V3 (chat) | Few-Shot | 78 | 18 | 4 | \$468.66 |

### Ablation Study

| Comparison Dimension | Chat Models | Reasoning Models | Notes |
|---------------------|-------------|-----------------|-------|
| Direct Solving | Weaker | Stronger | Reasoning models excel at direct computation |
| Zero-Shot Parsing | Weaker | Stronger | Reasoning models excel at zero-shot translation |
| Few-Shot Parsing | **Stronger** | Weaker | Chat models outperform with demonstrations |

### Key Findings
- **Divergence between chat and reasoning models**: Reasoning models are superior at direct computation and zero-shot parsing, but chat models outperform them at few-shot parsing. This may be because extended CoT interferes with the straightforward pattern-matching nature of translation given demonstrations.
- **GPT-5 Few-Shot is the optimal configuration**: 86/100 correct with a break-even price of only \$15.78, approximately 6% of the average U.S. tax filing cost.
- **Self-consistency substantially reduces cost**: GPT-4.1 Few-Shot+Few-Shot reduces the break-even price from \$247.99 to \$40.08 (an 84% reduction) by sacrificing coverage (87→81) for a significant drop in errors (8→5).
- **Symbolic solver provides natural quality control**: In Parsed methods, many programs fail to execute and are automatically rejected; while the number of correct answers is lower, the number of errors is also minimal.
- **Effect of model scale**: Small models (Qwen-32B, Llama-70B) almost completely fail at zero-shot parsing, whereas large models (o3) achieve the best performance in that setting. Parsing capability improves sharply with scale.

## Highlights & Insights
- **The Break-Even Price metric**: Translating AI accuracy into actual dollar costs provides an intuitive measure of economic feasibility. This approach of directly incorporating statutory penalties into evaluation is highly practical and should be extended to the evaluation of other high-stakes tasks.
- **Prolog execution failure as abstention mechanism**: The paper elegantly repurposes program non-executability as a confidence signal, eliminating the need for a separate uncertainty estimation module.
- **Chat models surpassing reasoning models in few-shot parsing**: This counterintuitive finding offers practical guidance for model selection — not every task benefits from reasoning models.
- **Economics of phased investment**: Gold-standard statute translation is a one-time fixed cost, after which the marginal cost per case is negligible. This aligns closely with the business model of commercial tax software companies.

## Limitations & Future Work
- **Limitations of the SARA dataset**: Only 9 simplified statutory provisions and 100 manually constructed cases are included, far short of the complexity of real tax law. Actual U.S. tax law encompasses thousands of provisions and numerous exceptions.
- **Dependence on gold-standard statute translations**: Best performance requires statutes to be manually pre-translated into Prolog, which represents a substantial engineering effort in real legal systems.
- **Only numerical computation cases evaluated**: Of SARA's 376 cases, only the 100 requiring numerical computation (the harder subset) are evaluated; the remaining 276 binary judgment cases are not assessed.
- **Reliance on closed-source models**: Best results are achieved by GPT-5 and o3, with open-source models lagging considerably.
- **Statutory changes not addressed**: Tax law is amended annually, requiring continuous maintenance of Prolog translations.

## Related Work & Insights
- **vs. GPT-4 Direct (Blair-Stanek et al. 2024)**: That work evaluated GPT-4 on SARA using direct computation only; the present paper substantially improves performance through symbolic solvers and few-shot prompting.
- **vs. Legal AI systems (Sergot et al. 1986)**: Classical legal expert systems rely entirely on hand-coded rules; this paper uses LLMs to automatically parse case facts, significantly reducing manual effort.
- **vs. Catala (Merigoux et al. 2021)**: Catala introduced a domain-specific programming language for encoding French tax law; the present approach using Prolog with LLM-based translation is more general-purpose.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframes tax law reasoning as semantic parsing; the break-even price economic evaluation framework is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8+ models × 5+ methods × multiple combinations; comprehensive ablations and economic analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, methodological progression is well-structured, and the economic perspective is introduced naturally.
- Value: ⭐⭐⭐⭐ Important implications for legal AI and neuro-symbolic systems; the break-even price framework is broadly transferable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](../../ACL2026/llm_nlp/foresight_optimization_for_strategic_reasoning_in_large_language_models.md)

<!-- RELATED:END -->
