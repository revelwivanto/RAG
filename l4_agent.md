Where L4 genuinely earns its place
1. Constrained sourcing — the replanning loop.
"Cari laptop untuk Design, budget 5 juta" is not a router problem. It's: search internal purchase history → search marketplace listings → score candidates through the 4 models → check policy cap and budget → and if nothing satisfies all constraints, decide what to relax. That last step is the L4 part. A router can't do it; there's no fixed path. The agent has to notice "no Design-adequate machine exists at 5 juta" and choose between widening the budget, dropping to refurbished, suggesting a lower spec tier, or telling the user the budget is unrealistic. That's plan → execute → evaluate → replan, and it's the single most defensible agent in your system because the failure is informative rather than an error.

2. The self-validation loop — which solves the problem we just found.
The "Macbook Core i7" silent-garbage case is exactly what reflection is for. An agent that scores a request, then critiques its own inputs — is this config in the catalogue? how many features did I default? is dept_budget_remaining a real number or the median? — and routes back to the user instead of answering. You agreed to the validation gate in point 1/2; making it a reflection step rather than an if statement is the difference between L2 and L4, and here the added complexity buys something real: the system declines to answer instead of answering confidently wrong.

3. (Optional showcase) Batch pre-screening.
"Review semua request bulan ini, flag yang kemungkinan ditolak dan jelaskan kenapa." Autonomous iteration over many records with per-record tool use and a synthesized report. Strong demo, genuinely useful to a procurement function, and unambiguously agentic.

Where I'd push back
Multi-agent decomposition for its own sake. Splitting into Extractor / Sourcing / Risk / Advisor agents is the obvious "look, multi-agent!" move, but four sub-agents that run in a fixed order with a shared context are a prompt chain wearing a costume — that's L2 with extra latency. Multi-agent earns its keep when sub-agents need independent context windows (so one's noisy retrieval doesn't pollute another's reasoning) or different models. Your sourcing agent chewing through 120 marketplace listings is a real case for context isolation. The Risk agent just calling four .predict() methods is not — that's a tool, not an agent.

The constraint that bounds all of this
Your hardware. Qwen2.5-14B-AWQ on 2×T4 falls back to the slow non-Marlin AWQ kernel (Turing, cc 7.5). An L4 loop multiplies generated tokens by 3-5× versus a single answer — a plan, per-step reasoning, a critique pass, a synthesis. On that setup you're plausibly looking at 30-60s per user turn, and your --max-model-len 8192 limits how much a planner can accumulate before it has to summarize.

That's not a reason to avoid L4, but it argues for one deep agent on the sourcing path rather than agents everywhere — keep single-turn factual RAG ("apa itu treasury") at L1/L2 where it belongs. For the KPI, being able to justify why each path sits at the level it does is a stronger result than maximizing levels; a reviewer will probe the decorative multi-agent design much harder than a well-argued two-level split.

If you want, I can sketch the router's decision boundaries and the sourcing agent's loop before any code gets written.

https://claude.ai/code/artifact/11703afe-444a-467a-96a3-896e08358d3e