---
title: >-
  [Paper Note] Mango: Multi-Agent Web Navigation via Global-View Optimization
description: >-
  [ACL 2026][Robotics & Embodied AI][Thompson Sampling] Mango constructs a global approximate structure of a website before navigation and uses Thompson Sampling to dynamically allocate a limited navigation budget among candidate URLs. This prevents LLM web agents from blindly exploring from the root URL and significantly outperforms baselines like AgentOccam and WebWalker
tags:
  - ACL 2026
  - Robotics & Embodied AI
  - Thompson Sampling
date: 2026-05-08
content_hash: b1e160c612433e18
---
# Mango: Multi-Agent Web Navigation via Global-View Optimization

**Conference**: ACL2026  
**arXiv**: [2604.18779](https://arxiv.org/abs/2604.18779)  
**Code**: https://github.com/VichyTong/Mango  
**Area**: Web Agent / LLM Agent / Web Navigation  
**Keywords**: Web navigation, global structure analysis, multi-armed bandit, Thompson Sampling, episodic memory

## TL;DR
Mango constructs a global approximate structure of a website before navigation and uses Thompson Sampling to dynamically allocate a limited navigation budget among candidate URLs. This prevents LLM web agents from blindly exploring from the root URL and significantly outperforms baselines like AgentOccam and WebWalker on WebVoyager and WebWalkerQA.

## Background & Motivation
**Background**: LLM web agents typically start from the root URL of a website and find answers progressively through actions like clicking, inputting, and reading pages. Existing work mainly focuses on improving browser perception, action space alignment, step-by-step planning, or agentic search to help models make better next-step decisions under local observations of the current page.

**Limitations of Prior Work**: Real-world websites often have deep hierarchical structures and a vast number of pages. If all tasks start from the homepage, the agent must traverse many irrelevant pages top-down, making it easy to fall into navigation traps, explore incorrect branches, or fail to reach the target page within a strict action budget. While search strategies like MCTS can explore trajectory trees, the simulation overhead is extremely high in scenarios with large branching factors and long horizons typical of the web.

**Key Challenge**: The bottleneck of web navigation is not just "where to click next," but also "where to start exploring." An agent with local observations may waste most of its budget due to a poor initial entry point even if its action selection is reasonable; however, exhaustively crawling the entire website is unrealistic.

**Goal**: The authors aim to construct a lightweight global view before navigation to select entry URLs relevant to the user query, and then adaptively decide which entry to visit first, whether to continue exploring, or whether to abandon a path under a limited budget.

**Key Insight**: Mango treats candidate URLs as arms of a multi-armed bandit (MAB) and the reflection results after a navigation attempt as reward signals. Compared to MCTS which expands the entire interaction tree, a bandit only needs to quickly balance exploration and exploitation among candidate entries, making it more suitable for strict budgets.

**Core Idea**: First, a candidate URL set is formed using lightweight BFS crawling, BM25, and site-specific Google search. Then, Thompson Sampling, initialized with BM25 relevance priors, is used to select URLs. After each navigation attempt, a reflection agent judges whether the path is promising, updating the Beta posterior and episodic memory.

## Method

### Overall Architecture
The input to Mango is a user query $q$ and a root URL $u_r$. The system first performs **Global Structure Analysis**: it crawls reachable webpages within the same domain, filters non-HTML and external links, and uses BM25 to find candidate URLs relevant to the query. For large websites difficult to cover via crawling, it uses an LLM to generate search keywords for Google `site:` retrieval to supplement entry candidates. Subsequently, it enters **URL Prioritization and Selection**: the candidate URL set $\mathcal{U}$ is modeled as a multi-armed bandit with a limited lifetime, using Thompson Sampling to select the next navigation entry from active arms. The navigation agent interacts with the browser environment starting from the selected URL. After an attempt, the reflection agent judges whether the answer is sufficient or if the path is worth continuing, updating the posterior and writing to the episodic memory.

In experiments, Mango uses a Playwright-based environment aligned with AgentOccam for WebVoyager, and a Crawl4AI environment aligned with WebWalker for WebWalkerQA to ensure fair execution settings. The navigation budget $b$ for each URL and the number of Thompson Sampling iterations are both set to 10.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: Query q + root URL"] --> GSA
    subgraph GSA["Global Structure Analysis Generates Candidate Entries"]
        direction TB
        B["BFS crawling same-domain HTML<br/>Max pages τ"] --> C["BM25 scoring with query for Top-10"]
        D["Large site supplement: LLM keywords<br/>Google site: retrieval Top-10"]
    end
    GSA --> E["Candidate URL set 𝒰"]
    E --> TS["URL Selection via Thompson Sampling<br/>BM25 initializes Beta prior, sample to select active arm"]
    TS --> NAV["Navigation agent interacts with browser from selected URL<br/>(Scaffolding)"]
    NAV --> REF["Reflection agent and episodic memory<br/>Determining Continue / Abandon / Finished"]
    REF -->|Hopeful → Pos reward (inc α) / Wrong → Neg reward (inc β)| TS
    REF -->|Dead end: mark Exhausted; Write trajectory to memory| E
    REF -->|Sufficient answer| F["Output Answer"]
```

### Key Designs

**1. Global Structure Analysis: Pruning the search space from the entire website structure before navigation**

Homepages are often not the best entry points, but crawling the entire site is impractical. Mango constructs a lightweight global approximate map. It performs BFS starting from the root URL, retaining only same-domain HTML pages up to a maximum page limit $\tau$. It then uses BM25 to score these pages against the query and selects the Top-10 for the candidate set. For large sites like arXiv with millions of pages, an LLM generates keywords based on the query for Google `site:` searches to provide the Top-10 additional candidates. This transforms navigation from "blind search from the root" to "probing from relevant subtree entries," where BM25 handles internal reachability and Google covers indexed pages.

**2. URL Selection via Thompson Sampling: Dynamically deciding the most valuable entry under a limited budget**

In what order should candidate entries be tried? BM25 relevance cannot be fully trusted, and navigation feedback is limited. Mango models the candidate URL set $\mathcal{U}$ as a multi-armed bandit with a limited lifetime to balance these factors. Each URL is an arm with states of Active/Exhausted, maintaining Beta distribution parameters $(\alpha_u, \beta_u)$. Initial values are derived from normalized BM25 scores $\rho_u=(\lambda_u-\min \lambda)/(\max \lambda-\min \lambda+\epsilon)$, where $\alpha_u^{(0)}=1+\kappa\rho_u$ and $\beta_u^{(0)}=1+\kappa(1-\rho_u)$. In each step, the model samples $\theta_u$ from the Beta posterior of active arms and selects the maximum for navigation. Positive reflections increase $\alpha$, while negative ones increase $\beta$; paths judged as dead ends are marked Exhausted. Compared to fixed rankings or MCTS tree simulations, Thompson Sampling quickly balances exploration and exploitation among entries, proving more robust under strict budgets.

**3. Reflection Agent and Episodic Memory: Determining completion, promise, and avoiding repeated attempts**

Web navigation often requires multiple trials; binary success/failure signals conflate "nearly finished" with "completely wrong." If the navigation agent claims completion, the reflection agent verifies the answer and trajectory against the query. If the answer is insufficient but the path is promising, a positive reward makes the URL more likely to be explored further. If the budget is exhausted, it judges if the current page is still relevant; irrelevance triggers a negative reward. Every trial's trajectory, output, and reflection are recorded in episodic memory, serving as context for the navigation agent if the same URL is revisited. Reflection thus categorizes navigation states into "Continue / Abandon / Finished" and uses memory to reduce redundant exploration—reflection directly drives the posterior of the next URL selection.

### Loss & Training
Mango does not train new models; it is primarily an inference-time agent pipeline. Five backbones are used: GPT-5-mini and Qwen3-4B/8B/14B/32B. Qwen3 models have thinking mode disabled, with temperature=0.7 and top_p=0.8. Key hyperparameters include a navigation budget $b=10$, 10 Thompson Sampling iterations, and Top-10 candidates from each source. Sensitivity analysis shows $\kappa=3$, $\tau=1000$, and Top-10 candidates are optimal settings.

## Key Experimental Results

### Main Results
| Benchmark | Backbone | Prev. SOTA SR | Mango SR | Gain | Notes |
|-----------|----------|-------------|----------|----------|------|
| WebVoyager | GPT-5-mini | AgentOccam 56.25 | 63.57 | +7.32 | Approx 63.6%, +7.3% |
| WebVoyager | Qwen3-32B | AgentOccam 34.11 | 37.98 | +3.87 | Improved on open-source models |
| WebWalkerQA | GPT-5-mini | WebWalker 25.74 | 52.50 | +26.76 | +26.8% in abstract |
| WebWalkerQA | Qwen3-4B | WebWalker 12.50 | 17.06 | +4.56 | Effective on small models |
| WebWalkerQA | Qwen3-32B | WebWalker 16.76 | 28.38 | +11.62 | Monotonic gain with scale |

In WebWalkerQA, Mango with GPT-5-mini achieved 60.59% for single-source QA Overall and 44.41% for multi-source QA Overall, totaling 52.50%. In contrast, WebWalker achieved 29.41%, 22.06%, and 25.74% respectively, while AgentOccam achieved 19.12%, 21.47%, and 20.29%.

### Ablation Study
| Benchmark | Backbone | Random URL | Google-only | MCTS | Mango | Key Conclusion |
|-----------|----------|------------|-------------|------|-------|----------|
| WebVoyager | GPT-5-mini | 56.59 | 59.69 | 46.51 | 63.57 | Thompson Sampling > MCTS |
| WebVoyager | Qwen3-32B | 27.13 | 32.56 | 23.26 | 37.98 | Structure + Bandit both contribute |
| WebWalkerQA | GPT-5-mini | 47.50 | 49.41 | 42.21 | 52.50 | Google alone is insufficient; MCTS is costly |
| WebWalkerQA | Qwen3-32B | 19.85 | 25.88 | 16.47 | 28.38 | Advantage maintained on open-source |

### Efficiency and Failure Analysis
| Item | Key Figure | Explanation |
|--------|----------|------|
| WebVoyager GPT-5-mini action count | Mango 14.18, AgentOccam 9.46, WebWalker 7.38 | Mango continues exploration to solve longer tasks |
| WebWalkerQA GPT-5-mini action count | Mango 19.13, AgentOccam 10.09, WebWalker 10.38 | SR increase comes with higher action costs |
| Failed Samples | 323 WebWalkerQA cases | Manual check on GPT-5-mini backbone failures |
| Exceed Budget | 52.4% | Budget exhausted due to deep info or candidate errors |
| Locating Wrongly | 24.6% | Misled by ambiguous links to incorrect subpages |
| Reasoning Error | 15.4% | Correct page reached, but extraction/reasoning failed |
| Out-of-date Golden Answers | 5.6% | Benchmark ground truth has expired |
| Reflection Error | 2.0% | Reflection agent judged answer sufficient too early |

### Key Findings
- The primary gain of Mango comes from "pruning the search space before navigation." It does not make the LLM smarter but provides it with a better starting set.
- MCTS performs poorly under strict budgets because it requires extensive interaction for expansion and evaluation; Thompson Sampling avoids trajectory tree simulation, making it better for entry selection.
- The higher action count on GPT-5-mini is because Mango successfully completes complex tasks where baselines plateaued, rather than just dragging out simple tasks.
- Over half of the failures are due to exceeding the budget, indicating that the global view is still an approximation and deep-rooted information remains difficult.

## Highlights & Insights
- **From "In-page Decision" to "Entry Selection"**: Many web agent papers assume starting from the homepage. Mango challenges this, asserting that for large sites, entry selection is half the battle.
- **BM25 Prior + Bandit Posterior is a Practical Combo**: BM25 provides cheap global relevance, while reflection rewards provide online feedback. This design is cheaper than pure LLM scoring and more adaptable to initial estimation errors than fixed rankings.
- **Connecting Reflection to URL Posteriors**: Reflection does more than generate logs; it directly influences the probability distribution of the next URL selection, allowing the reflection module to truly participate in control.
- **Honest Failure Analysis**: The authors distinguish between navigation failure, locating failure, reading/reasoning failure, and benchmark expiration, showing that Mango solves exploration efficiency rather than all web QA problems.

## Limitations & Future Work
- Global structure is a lightweight approximation and cannot cover large, dynamic, or deep websites. Information buried very deep may still exceed the budget.
- The quality of the candidate set is critical for the subsequent bandit. If BM25, LLM keywords, or Google results provide incorrect entries early on, posterior correction might not be fast enough under strict budgets.
- Success still depends on LLM reading comprehension or detail extraction even after reaching the correct page, which navigation strategies cannot solve.
- Mango sometimes achieves higher success rates via more actions, which might not be cost-effective in latency-sensitive or API-cost-sensitive scenarios.
- Dependency on Google search as a supplement may be affected by search API availability, region, personalization, and page updates.

## Related Work & Insights
- **vs AgentOccam**: AgentOccam emphasizes aligning actions and observations for easier browser operation; Mango focuses on entry selection and budget allocation before navigation. They are complementary.
- **vs WebWalker**: WebWalker uses an explore-critic paradigm to explore step-by-step; Mango uses global structure and bandits to reduce irrelevant exploration first.
- **vs MCTS web agent**: MCTS is suitable for simulatable spaces with controllable branching; Mango’s Thompson Sampling is lighter for high-cost, complex web branching.
- **Inspiration**: Similar ideas can be applied to codebase navigation, document retrieval, and enterprise knowledge base QA: establish a lightweight global index first, then distribute budget among entries via bandits/reflection.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Approaches web navigation from global structure and bandit entry selection; clear and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive benchmarks, backbones, action counts, ablations, sensitivity, and failure analysis.
- **Writing Quality**: ⭐⭐⭐⭐☆ Method is intuitive with sufficient tables, though some notation details are slightly scattered.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for building real-world web agents, especially regarding entry selection and reflection control in budget-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](../../ICLR2026/robotics/mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[CVPR 2026\] DiffuView: Multi-View Diffusion Pretraining for 3D-Aware Robotic Manipulation](../../CVPR2026/robotics/diffuview_multi-view_diffusion_pretraining_for_3d_aware_robotic_manipulation.md)
- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](../../AAAI2026/robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](../../AAAI2026/robotics/urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)
- [\[CVPR 2025\] CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos](../../CVPR2025/robotics/citywalker_learning_embodied_urban_navigation_from_web-scale_videos.md)

</div>

<!-- RELATED:END -->
