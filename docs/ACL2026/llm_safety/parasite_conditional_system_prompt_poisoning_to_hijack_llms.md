---
title: >-
  [Paper Note] PARASITE: Conditional System Prompt Poisoning to Hijack LLMs
description: >-
  [ACL2026][LLM Safety][Conditional Prompt Poisoning] PARASITE formalizes the threat of "system prompts downloaded from public marketplaces containing conditional trigger backdoors" as a new supply chain risk. It utilizes…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Conditional Prompt Poisoning"
  - "System Prompt Security"
  - "Black-box Attack"
  - "Discrete Prompt Optimization"
  - "Defense Evaluation"
date: 2026-05-08
content_hash: bc7718364502da9e
---

# PARASITE: Conditional System Prompt Poisoning to Hijack LLMs

**Conference**: ACL2026  
**arXiv**: [2505.16888](https://arxiv.org/abs/2505.16888)  
**Code**: https://github.com/vietph34/PARASITE  
**Area**: LLM Security / Prompt Security / System Prompt Supply Chain  
**Keywords**: Conditional Prompt Poisoning, System Prompt Security, Black-box Attack, Discrete Prompt Optimization, Defense Evaluation

## TL;DR
PARASITE formalizes the threat of "system prompts downloaded from public marketplaces containing conditional trigger backdoors" as a new supply chain risk. It utilizes global semantic search combined with word-level greedy perturbations to generate highly stealthy system prompts that hijack responses only for target queries under black-box conditions.

## Background & Motivation
**Background**: LLM applications increasingly rely on system prompts to define personas, permission boundaries, and response styles. Many developers do not design prompts from scratch but rather copy "optimized" system prompts from markets like FlowGPT, Hugging Face, or open-source repositories to integrate into their models or APIs.

**Limitations of Prior Work**: This system prompt supply chain has historically been treated as a productivity tool rather than a security boundary. Existing research focuses more on user-side jailbreaks, indirect RAG injection, or model weight/data backdoors. These attacks either require re-injection in every conversation, white-box access, or significantly disrupt the model's overall behavior, failing to explain how a "seemingly normal system prompt can remain dormant long-term."

**Key Challenge**: Attackers want the model to remain functional for normal queries to convince users the prompt is safe, while outputting specific incorrect stances or facts for a few sensitive queries. This is not a standard jailbreak "breaking the rules," but a sparse, discrete, and constrained search problem: the prompt must stay close to the benign semantic manifold while moving toward the malicious target.

**Goal**: The paper aims to answer three questions: First, how to define conditional poisoning threats purely through system prompts; second, whether such "sleeper agent" prompts can be automatically found in black-box API scenarios without access to weights/gradients; and third, whether common perplexity, similarity, grammar correction, or safety audits can detect or mitigate these attacks.

**Key Insight**: The authors view the system prompt as a supply chain object that can be published, reused, and audited, rather than a single-turn input. This perspective is critical because once a malicious prompt is uploaded to a public market, it can persist in various downstream applications and trigger only on specific queries, making it much harder to detect than a one-time jailbreak suffix.

**Core Idea**: A dual-objective black-box optimization framework is employed to search for system prompts that push specific queries toward attacker-specified answers while maintaining benign performance and low suspiciousness.

## Method
PARASITE stands for System Prompt AdveRsarial Attack for Selective Inference-Time Exploitation. It does not involve model training or appending jailbreak suffixes to user inputs. Instead, it embeds a conditional trigger mechanism within the system prompt itself: it acts like a normal assistant until it encounters target semantics.

### Overall Architecture
The paper defines three stakeholders: the attacker (accesses target API but no weights/gradients), the platform (hosts prompts and may use filters like perplexity or safety models), and the victim (downloads prompts for their LLM).

Input includes a target query set $Q_t$ and a benign query set $Q_b$. $Q_t$ contains sensitive questions and target incorrect answers; $Q_b$ contains normal questions and ground truth. The output is an optimized system prompt $p^*$ that induces target responses on $Q_t$ while maintaining benign behavior on $Q_b$.

The objective is formulated as dual-objective optimization: the adversarial loss $L_{adv}(p)$ drives the model toward target answers on sensitive queries, while the benign loss $L_{benign}(p)$ maintains correctness on normal queries. The joint loss $L(p)=L_{adv}(p)+L_{benign}(p)$ is minimized with constraints on semantic similarity and perplexity.

The process consists of two steps: First, a global semantic search uses an LLM rewriter to generate a readable prompt skeleton with an inherent semantic bias toward the target. Second, local greedy refinement applies minor perturbations or synonym substitutions to high-impact words to cross the model’s local decision boundaries while preserving readability.

### Key Designs
1.  **Threat Modeling of Conditional System Prompt Poisoning**:
    - Function: Transforms "malicious system prompts" from vague prompt injection descriptions into an evaluable security problem.
    - Mechanism: Attack success is not just measured by whether target queries are hijacked, but also by whether normal queries remain unaffected. The paper uses $Q_t$ to constrain target triggers and $Q_b$ to constrain benign behavior, calculating the selective attack strength via $Delta F1 = F1_{benign} - F1_{malicious}$.
    - Design Motivation: Traditional jailbreaks seek broad failure, making the prompt or output obvious. The danger here lies in "only failing in one corner," requiring stealth and conditionality to be part of the objective function.

2.  **Adversarial AutoPrompt Global Semantic Search**:
    - Function: Finds a natural, interpretable prompt skeleton that partially meets the attack goal under black-box (no gradient) conditions.
    - Mechanism: AAP utilizes GPT-4o-mini as a prompt rewriter. Each round, it evaluates the current prompt's discrete scores on benign and target sets, analyzes failure cases, and instructs the generator to rewrite based on feedback. Scores are converted to binary signals via token-level F1 to avoid rigid exact matching.
    - Design Motivation: Feasible solutions for conditional poisoning are like sparse islands; direct word-level search easily gets stuck. Using semantic search first brings the prompt into the correct vicinity, reducing the blindness of subsequent local searches.

3.  **Word-level Greedy Refinement and Tolerable Noise**:
    - Function: Searches for fine-grained perturbations on the semantic skeleton that trigger target behavior without damaging benign performance.
    - Mechanism: PARASITE uses leave-one-out to estimate the impact of each word on the joint loss, ranking them by importance. It then applies random splits, character swaps, keyboard proximity replacements, deletions, or synonym swaps to high-impact words, accepting candidates that reduce joint loss.
    - Design Motivation: Real-world prompt markets already contain minor spelling and grammar errors; filters struggle to categorize all errors as malicious. PARASITE exploits this "natural noise background" to hide trigger signals.

### Loss & Training
PARASITE does not train model parameters; it performs discrete optimization based on API queries. For target queries, the attacker wants the model to output $y_{adv}$; for benign queries, the model should output $y_{true}$. Loss is combined using F1 or EM (Exact Match) to evaluate proximity to reference answers.

In Stage 1, the optimization signal is a coarse discrete score: rewards for correct benign answers and penalties for failing to induce target answers. This signal drives large semantic moves by the LLM rewriter.

In Stage 2, optimization becomes a fine-grained word-level greedy search. The algorithm iteratively selects important words, tries black-box perturbations, and queries the target model for new losses. The attack threshold $k$ controls aggressiveness; higher $k$ usually increases malicious success but may decrease benign performance.

The authors emphasize low cost: Stage 1 costs roughly $\$0.003$ per target, and Stage 2 costs roughly $\$1.99$ due to query volume. Total costs of $\approx\$2$ to generate a poisoned prompt for a target objective suggest this is more than a theoretical threat.

## Key Experimental Results

### Main Results
Experiments were conducted across three sets: non-targeted fact hijacking on TriviaQA, targeted high-risk concept hijacking on TruthfulQA, and real-world feasibility on GPT-4o-mini / GPT-3.5-Turbo APIs. TriviaQA targets were pre-filtered to questions the model answers correctly under manual benign prompts to avoid confusion.

Training and testing are strictly separated. Optimization uses only 10 target and 20 benign queries per target but is evaluated on 100 unseen target rewrites and 1000 held-out benign queries.

| Setting | Model / Metric | Manual or AAP | PARASITE | Main Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| TriviaQA Non-targeted | Llama2-7B: Benign F1 / Malicious F1 / Delta F1 | Manual: 73.09 / 54.00 / 19.09; AAP: 66.31 / 79.19 / -12.88 | 63.84 / 33.36 / 30.48 | PARASITE significantly lowers Malicious F1 while preserving high Benign F1 |
| TriviaQA Non-targeted | Llama2-13B: Benign F1 / Malicious F1 / Delta F1 | Manual: 85.00 / 96.50 / -11.50; AAP: 82.14 / 82.46 / -0.32 | 66.77 / 32.66 / 34.11 | Semantic search alone is insufficient; word-level refinement creates selective gaps |
| TriviaQA Non-targeted | DeepSeek-7B: Benign F1 / Malicious F1 / Delta F1 | Manual: 52.11 / 100.00 / -47.89; AAP: 52.49 / 69.71 / -17.22 | 43.99 / 28.15 / 15.84 | Manual prompts fail to hijack; PARASITE establishes stable conditional triggers |

TruthfulQA targeted experiments reflect real-world risks (Politics, Health, Conspiracies). Results show strong generalization from Two-Option optimization to Four-Option and Free-Form formats.

| Setting | Model | Benign F1 | Malicious F1 | Aggregate Psi | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Two-Option | Qwen2.5-7B | 62.76 | 73.03 | 67.89 | Strong target trigger without sacrificing benign performance |
| Two to Four-Option | Qwen2.5-7B | 56.45 | 70.75 | 63.60 | High malicious success even after format change |
| Free-Form | Qwen2.5-7B | 41.55 | 62.18 | 51.81 | Hijack persists in free text, showing conceptual-level influence |

Commercial API experiments show GPT-4o-mini Malicious F1 drops from 99.28 (manual) to 52.44, while Benign F1 remains at 71.44.

### Ablation Study

| Ablation / Analysis | Key Metric | Description |
| :--- | :--- | :--- |
| Remove spelling noise | Malicious F1 on Qwen2.5-7B drops from ~67.9 to 22.7 | Word-level noise is the key freedom for crossing local decision boundaries |
| Initialization: M+Greedy vs A+Greedy | Delta F1 on Llama3.1-8B increases from 12.56 to 17.69 | AAP semantic skeletons aid cross-model and cross-format transfer |
| Model Scale | PARASITE outperforms AAP from Qwen2.5 3B to 32B | Larger models are not immune; stronger instruction following may better preserve benign performance |
| Detection Filters | GPT-5 judge detection rate for ECLIPSE is ~81%, vs <7% for PARASITE | Traditional filters miss noise in natural language but catch gibberish suffixes |

### Key Findings
- Stage 2 word-level refinement is the core contributor. AAP finds semantic regions, but LLM rewriters often "auto-correct" back to natural expressions, lacking the precision to cross local boundaries.
- Failure modes differ from suffix jailbreaks. Methods like ECLIPSE often break model utility overall, whereas PARASITE acts as a selective bias.
- Defense is difficult due to low false positive tolerance. Simple grammar correction cannot eliminate semantic triggers, while aggressive rewriting harms prompt functionality.

## Highlights & Insights
- The most valuable takeaway is viewing system prompts through the lens of supply chain security. Prompts are essentially executable policies; if they come from third parties, they require trust and auditing.
- "Conditional poisoning" is more relevant to real-world abuse than broad jailbreaking. Attackers likely want subtle shifts in topics like voting, medical advice, or history rather than obvious violations.
- Dual-objective evaluation is insightful. Reporting only attack success encourages crude model destruction; reporting both benign preservation and target hijacking reveals the true risk of stealthy attacks.
- Analysis of "tolerable noise" is clever. Spelling errors in prompts are common, making "natural noise" a gray area for both attackers and defenders.

## Limitations & Future Work
- The study focuses on single-turn dialogues. Multi-turn interactions might strengthen attacks through context accumulation or make anomalies more apparent.
- No human perceptibility studies were conducted. While filters fail, it's unknown if human auditors would notice suspicious phrasing.
- Tasks are benchmark-oriented. Real applications involve long contexts and tool calls where system prompts significantly impact tool selection and agent policies.
- Defense discussion is preliminary. Future work should explore behavioral differential testing and provenance for third-party prompts.

## Related Work & Insights
- **vs GCG / AutoDAN**: These optimize user-side adversarial suffixes to break safety boundaries. PARASITE optimizes system prompts with stability constraints for benign queries, making it a supply chain backdoor.
- **vs ECLIPSE**: ECLIPSE is a black-box suffix search prone to visible gibberish; PARASITE maintains readability via semantic skeletons.
- **vs Training Backdoors / Sleeper Agents**: Training backdoors require weight control; PARASITE relies only on API queries and prompt text, lowering the deployment bar.
- **Insight for Defense**: Security cannot rely on static text audits. Platforms should submit third-party prompts to a suite of high-risk semantic probes to detect selective behavioral shifts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[ACL 2026\] Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets](train_in_vain_functionality-preserving_poisoning_to_prevent_unauthorized_use_of_.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)

</div>

<!-- RELATED:END -->
