---
title: >-
  [Paper Note] Mango: Multi-Agent Web Navigation via Global-View Optimization
description: >-
  [ACL2026][Robotics][Web Navigation] Mango constructs a lightweight approximate global structure of a website before navigation and utilizes Thompson Sampling to dynamically allocate a finite navigation budget among candi…
tags:
  - "ACL2026"
  - "Robotics"
  - "Web Navigation"
  - "Global Structure Analysis"
  - "Multi-Armed Bandit"
  - "Thompson Sampling"
  - "Episodic Memory"
date: 2026-05-08
content_hash: 250e56b6ab895d5b
---

# Mango: Multi-Agent Web Navigation via Global-View Optimization

**Conference**: ACL2026  
**arXiv**: [2604.18779](https://arxiv.org/abs/2604.18779)  
**Code**: https://github.com/VichyTong/Mango  
**Area**: Web Agent / LLM Agent / Web Navigation  
**Keywords**: Web Navigation, Global Structure Analysis, Multi-Armed Bandit, Thompson Sampling, Episodic Memory

## TL;DR
Mango constructs a lightweight approximate global structure of a website before navigation and utilizes Thompson Sampling to dynamically allocate a finite navigation budget among candidate URLs. This prevents LLM web agents from searching blindly from the homepage and significantly outperforms baselines such as AgentOccam and WebWalker on WebVoyager and WebWalkerQA.

## Background & Motivation
**Background**: LLM web agents typically start from a website's root URL and seek answers through sequential actions such as clicking, typing, and page reading. Existing works primarily improve browser perception, action space alignment, step-by-step planning, or agentic search to help models make better next-step decisions based on local observations of the current page.

**Limitations of Prior Work**: Real-world websites often feature deep hierarchical structures and vast numbers of pages. If every task starts from the homepage, the agent must traverse numerous irrelevant pages top-down, making it prone to navigation traps, exploring incorrect branches, or failing to reach the target page within a strict action budget. While search strategies like MCTS can explore trajectory trees, the simulation overhead is prohibitive in web scenarios characterized by large branching factors and long horizons.

**Key Challenge**: The bottleneck in web navigation is not only "where to click next" but also "where to start the exploration." An agent with local observations might waste most of its budget due to a poor initial entry point, even if its action selection is sound. Conversely, exhaustively crawling the entire website is impractical.

**Goal**: The authors aim to construct a lightweight global view before navigation to select entry URLs relevant to the user's query and adaptively decide which entry to visit first, whether to continue exploration, or whether to abandon a path under a limited budget.

**Key Insight**: Mango treats candidate URLs as arms in a multi-armed bandit problem and uses the reflection results after a navigation attempt as reward signals. Compared to MCTS, which expands the entire interaction tree, the bandit approach simply balances exploration and exploitation among candidate entries, making it better suited for strict budgets.

**Core Idea**: First, a candidate URL set is formed using lightweight BFS crawling, BM25, and site-specific Google search. Then, Thompson Sampling, initialized with BM25 relevance as a prior, is used to select URLs. After each navigation, a reflection agent assesses whether the path is promising, updating the Beta posterior and episodic memory.

## Method

### Overall Architecture
The input to Mango is a user query $q$ and a root URL $u_r$. The system first performs **Global Structure Analysis**: it crawls reachable pages within the same domain, filters non-HTML and external links, and uses BM25 to identify candidate URLs relevant to the query. For large websites where crawling is difficult, an LLM generates search keywords to supplement candidates via Google `site:` search. Next, the system enters **URL Prioritization and Selection**: the candidate URL set $\mathcal{U}$ is modeled as a multi-armed bandit with a finite lifetime. Thompson Sampling selects the next navigation entry from the active arms. The navigation agent interacts with the browser environment starting from the chosen URL. After an attempt, a reflection agent determines if the answer is sufficient or if the path warrants further exploration, updating the posterior and writing to episodic memory.

In experiments, Mango uses a Playwright-based environment for WebVoyager (aligned with AgentOccam) and a Crawl4AI environment for WebWalkerQA (aligned with WebWalker) to ensure fair execution settings. The navigation budget $b$ per URL and the number of Thompson Sampling iterations are both set to 10.

### Key Designs
1. **Global Structure Analysis for Candidate Entry Generation**:
	- **Function**: Sieve out "promising entry pages" from the massive website structure before actual navigation.
	- **Mechanism**: Mango performs a lightweight BFS crawl starting from the root URL, retaining only same-domain HTML pages up to a maximum page limit $\tau$. Page contents are scored against the query using BM25, and the top 10 are added to the candidate set. If the website is too large or difficult to crawl (e.g., arXiv with millions of pages), an LLM suggests keywords for Google `site:` search to supplement the top 10 results.
	- **Design Motivation**: The homepage is not always the best entry point. Obtaining an approximate global structure transforms navigation from "blind search from the root" into "targeted probing from relevant sub-tree entries." Combining BM25 and Google search balances internal reachability with search engine indexing.

2. **Thompson Sampling-based URL Selection**:
	- **Function**: Dynamically decide the most valuable candidate URL to visit under a finite navigation budget.
	- **Mechanism**: Each URL is an arm with an Active/Exhausted state, maintaining Beta distribution parameters $(\alpha_u, \beta_u)$. Initial parameters are derived from normalized BM25 scores: $\rho_u = (\lambda_u - \min \lambda) / (\max \lambda - \min \lambda + \epsilon)$, with $\alpha_u^{(0)} = 1 + \kappa\rho_u$ and $\beta_u^{(0)} = 1 + \kappa(1 - \rho_u)$. In each step, $\theta_u$ is sampled from the Beta posterior of active arms, and the URL with the highest value is selected. If reflection yields a positive reward, $\alpha$ is increased; otherwise, $\beta$ is increased. If a path is deemed a dead end, it is marked as Exhausted.
	- **Design Motivation**: The BM25 prior utilizes global relevance but is not fully reliable; navigation feedback corrects the prior but is limited in frequency. Thompson Sampling effectively balances the two and is more lightweight than fixed ranking or MCTS simulation.

3. **Reflection Agent and Episodic Memory**:
	- **Function**: Determine if a navigation attempt has completed the task or deserves continuation, and avoid repeating previous mistakes.
	- **Mechanism**: If the navigation agent claims success, the reflection agent verifies if the final answer and action trajectory satisfy the query. If the answer is insufficient but the path remains promising, a positive reward is given, increasing the likelihood of future exploration of that URL. If the budget is exhausted, the reflection agent assesses the current page's relevance; irrelevance earns a negative reward. Trajectories, outputs, and reflections are stored in episodic memory and provided as context if the same URL is revisited.
	- **Design Motivation**: Web navigation often requires multiple probes. Binary success/failure metrics conflate "near-misses" with "complete errors." The reflection agent categorizes states as continuable, abandonable, or completed, using memory to reduce redundant exploration.

### Loss & Training
Mango does not train a new model; it is primarily an inference-time agent pipeline. Experiments utilize five backbones: GPT-5-mini and Qwen3-4B/8B/14B/32B. For Qwen3 models, thinking mode is disabled, with temperature=0.7 and top_p=0.8. Key hyperparameters include a navigation budget $b=10$, 10 Thompson Sampling iterations, and a top-10 selection from candidate sources. Sensitivity analysis indicates $\kappa=3$, $\tau=1000$, and a candidate Top-10 are optimal settings.

## Key Experimental Results

### Main Results
| Benchmark | Backbone | Prev. SOTA SR | Ours SR | Gain | Remarks |
|-----------|----------|-------------|----------|----------|------|
| WebVoyager | GPT-5-mini | AgentOccam 56.25 | 63.57 | +7.32 | Rounded to 63.6%, +7.3% in abstract |
| WebVoyager | Qwen3-32B | AgentOccam 34.11 | 37.98 | +3.87 | Improvement on open-source models |
| WebWalkerQA | GPT-5-mini | WebWalker 25.74 | 52.50 | +26.76 | +26.8% in abstract |
| WebWalkerQA | Qwen3-4B | WebWalker 12.50 | 17.06 | +4.56 | Effective even on small models |
| WebWalkerQA | Qwen3-32B | WebWalker 16.76 | 28.38 | +11.62 | Mango scales monotonically with model size |

In WebWalkerQA, Mango using GPT-5-mini achieved 60.59% on single-source QA Overall and 44.41% on multi-source QA Overall, totaling 52.50%. In comparison, WebWalker achieved 29.41%, 22.06%, and 25.74%, and AgentOccam achieved 19.12%, 21.47%, and 20.29%, respectively.

### Ablation Study
| Benchmark | Backbone | Random URL | Google-only | MCTS | Mango | Key Conclusion |
|-----------|----------|------------|-------------|------|-------|----------|
| WebVoyager | GPT-5-mini | 56.59 | 59.69 | 46.51 | 63.57 | Thompson Sampling significantly outperforms MCTS |
| WebVoyager | Qwen3-32B | 27.13 | 32.56 | 23.26 | 37.98 | Global structure + bandit both contribute |
| WebWalkerQA | GPT-5-mini | 47.50 | 49.41 | 42.21 | 52.50 | Google-only is insufficient; MCTS has high budget costs |
| WebWalkerQA | Qwen3-32B | 19.85 | 25.88 | 16.47 | 28.38 | Mango maintains advantage on open-source models |

### Efficiency & Failure Analysis
| Analysis Item | Key Figure | Explanation |
|--------|----------|------|
| WebVoyager GPT-5-mini action count | Mango 14.18, AgentOccam 9.46, WebWalker 7.38 | Mango is more willing to explore, solving longer tasks |
| WebWalkerQA GPT-5-mini action count | Mango 19.13, AgentOccam 10.09, WebWalker 10.38 | Higher success rate comes with higher action costs |
| Failure Sample Size | 323 WebWalkerQA failure cases | Manual inspection of GPT-5-mini backbone failures |
| Exceed Budget | 52.4% | Budget exhausted due to deep information or candidate errors |
| Locating Wrongly | 24.6% | Misled by ambiguous links to incorrect sub-pages |
| Reasoning Error | 15.4% | Reached correct page but failed in extraction/reasoning |
| Out-of-date Golden Answers | 5.6% | Benchmark ground truth answers were expired |
| Reflection Error | 2.0% | Reflection agent prematurely judged answers as sufficient |

### Key Findings
- Mango's primary gain comes from "pruning the search space before navigation." It does not necessarily make the LLM smarter but provides it with a superior set of starting points.
- MCTS performs poorly under strict budgets because it requires extensive interaction for expansion and value estimation; Thompson Sampling avoids trajectory tree simulation, making it better for candidate URL selection.
- On GPT-5-mini, Mango has a higher action count precisely because it can complete complex tasks where baselines have already plateaued, rather than just being inefficiently long.
- Over half of the failures are due to exceeding the budget, indicating that the global view is still an approximation and deep-rooted long-tail web pages remain challenging.

## Highlights & Insights
- **Shifting from "In-page Decision" to "Entry Selection"**: Many web agent papers assume navigation starts at the homepage; Mango directly challenges this. For large websites, entry selection is half the battle.
- **BM25 Prior + Bandit Posterior is a Practical Combination**: BM25 provides inexpensive global relevance, while reflection rewards provide online feedback. This design is cheaper than full LLM scoring and more robust than fixed ranking against initial estimation errors.
- **Integrating Reflection into URL Posteriors**: Reflection does not just generate logs or natural language summaries; it directly influences the probability distribution of the next URL selection. This allows the reflection module to participate truly in control.
- **Honest Failure Analysis**: The authors distinguish between navigation, localization, reading/reasoning failures, and expired benchmark answers. This demonstrates that Mango addresses exploration efficiency rather than being a panacea for all web QA problems.

## Limitations & Future Work
- The global structure is only a lightweight approximation and cannot cover large, dynamic, or deep websites. If target information is buried too deeply, it may still exceed the budget.
- The quality of the candidate set is critical for subsequent bandit performance. If BM25, LLM keywords, or Google results introduce poor entry points early, posterior adjustments may not occur fast enough under strict budgets.
- Failures can still occur due to LLM reading comprehension or detail extraction errors even after reaching the correct page, which is outside the scope of the navigation strategy itself.
- Mango sometimes achieves higher success rates through more actions, which may not be cost-effective in latency-sensitive or API cost-sensitive scenarios.
- The use of Google search as a supplementary source may be affected by search API availability, region, personalization, and page updates in practical deployment.

## Related Work & Insights
- **vs AgentOccam**: AgentOccam emphasizes aligning action and observation spaces to make browsers easier for LLMs to manipulate; Mango focuses on entry selection and budget allocation before navigation. They are complementary.
- **vs WebWalker**: WebWalker uses an explore-critic paradigm to explore pages incrementally; Mango reduces irrelevant exploration through global structure and bandits.
- **vs MCTS web agent**: MCTS is suitable for simulatable search spaces with controllable branching; web navigation has high interaction costs and complex branching, making Mango's Thompson Sampling more lightweight.
- **Inspiration**: Similar ideas can be applied to codebase navigation, document retrieval, and enterprise knowledge base QA: establish a lightweight global index first, then use bandits/reflection to allocate budgets among candidates.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Strategic focus on global website structure and bandit-based entry selection is clear and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across two benchmarks, five backbones, action counts, ablations, sensitivity, and failure analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Intuitive methodology and sufficient tables, though some notation and algorithmic details are slightly scattered.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for building practical web agents, especially regarding entry selection and reflection control in budget-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](../../NeurIPS2025/robotics/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](../../AAAI2026/robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](../../AAAI2026/robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[AAAI 2026\] Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](../../AAAI2026/robotics/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)

</div>

<!-- RELATED:END -->
