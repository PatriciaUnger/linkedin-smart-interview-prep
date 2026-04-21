"""
interview_kb.py
───────────────
Curated knowledge base of example interview answers across roles and skills.

V3 UPDATE (Sprint 5):
    Expanded from 20 to 35 examples. My V2 professor feedback explicitly asked
    to "expand knowledge base". The additions target gaps I identified after
    running the V3 evaluation harness:

    1. More weak examples with varied failure modes — V2 was skewed toward
       strong answers (16 strong vs 4 weak), which meant retrieval for the
       "what to avoid" example often surfaced the same 2-3 docs regardless
       of query.

    2. Failure / learning-from-mistakes — under-covered in V2 (1 strong, 1 weak).
       One of the most common behavioural interview categories.

    3. Debugging / troubleshooting — missing entirely in V2. A staple of
       technical interviews.

    4. Role diversity — V2 was heavily biased toward product_manager,
       software_engineer, data_scientist. Added marketing_analyst and
       sales roles which are common on LinkedIn job postings.

    5. Negotiation and difficult conversations — missing in V2.

    6. Time management / competing deadlines — missing in V2.

    7. Mentoring / developing others — missing in V2.

    8. Customer focus / user empathy — missing in V2.

    The original 20 V2 examples are preserved below the section divider.
"""

KNOWLEDGE_BASE = [

    # =======================================================================
    # V2 KNOWLEDGE BASE — original 20 examples from v2 (unchanged)
    # =======================================================================

    # ── BEHAVIOURAL: Conflict / Stakeholder Management ─────────────────────
    {
        "text": (
            "In my previous role as a product manager, two engineering leads disagreed "
            "on the architecture for a new payments feature. I scheduled a structured "
            "workshop where each side presented their approach with explicit trade-off "
            "analysis on latency, cost, and maintainability. I then facilitated a "
            "decision matrix vote. The team aligned on a hybrid solution within one "
            "session, and we shipped on time. The key was making the trade-offs visible "
            "so the conversation moved from opinion to data."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "conflict resolution",
        "why": "Uses STAR structure, quantifies outcome, shows facilitation skill.",
    },
    {
        "text": (
            "There was a disagreement on my team. I just waited for my manager to "
            "decide because it was not my place to get involved."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "weak",
        "skill": "conflict resolution",
        "why": "Passive, shows no initiative, no outcome described.",
    },
    {
        "text": (
            "During a product launch, the sales director wanted to delay by two weeks "
            "to align with a trade event, while engineering had already committed to "
            "the original date. I set up a 30-minute call with both, presented a "
            "comparison of revenue impact vs. engineering cost of rescheduling, and "
            "proposed a phased release: core feature on schedule, marketing push at "
            "the trade event. Both agreed. Revenue in the first month exceeded the "
            "forecast by 18%."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "stakeholder alignment",
        "why": "Data-driven negotiation, concrete outcome, shows cross-functional leadership.",
    },

    # ── BEHAVIOURAL: Leadership / Initiative ───────────────────────────────
    {
        "text": (
            "I noticed our sprint retrospectives were running over time and producing "
            "repetitive action items that no one owned. I introduced a simple template: "
            "each item had a named owner and a due date, and I added a five-minute "
            "retrospective-of-retrospectives at the start of each sprint to review "
            "completion. Within two sprints, action-item completion went from 40% to "
            "85%, and meeting time dropped by 20 minutes."
        ),
        "role": "software_engineer",
        "type": "behavioural",
        "quality": "strong",
        "skill": "process improvement",
        "why": "Self-initiated, measurable improvement, shows ownership mindset.",
    },
    {
        "text": (
            "I am a very proactive person and always take initiative in my team. "
            "I like to lead by example and motivate my colleagues."
        ),
        "role": "software_engineer",
        "type": "behavioural",
        "quality": "weak",
        "skill": "leadership",
        "why": "Generic claims with no evidence, no specific situation or result.",
    },
    {
        "text": (
            "When our team lead left unexpectedly three weeks before a major delivery, "
            "I volunteered to own the release plan. I created a dependency map, ran "
            "daily 15-minute standups, and escalated two blockers to the CTO early "
            "enough to resolve them. We delivered on the original date with zero "
            "critical bugs in the first two weeks post-launch."
        ),
        "role": "software_engineer",
        "type": "behavioural",
        "quality": "strong",
        "skill": "leadership under pressure",
        "why": "Concrete situation, proactive escalation, quantified delivery outcome.",
    },

    # ── BEHAVIOURAL: Failure / Learning ────────────────────────────────────
    {
        "text": (
            "I once underestimated the complexity of migrating our authentication "
            "system and committed to a two-week timeline publicly. By week one it was "
            "clear we needed four. I immediately informed the product owner, presented "
            "a revised plan with a temporary workaround to unblock downstream teams, "
            "and documented the estimation mistakes for the team. We now use a "
            "three-point estimation approach for all infrastructure work."
        ),
        "role": "software_engineer",
        "type": "behavioural",
        "quality": "strong",
        "skill": "handling failure",
        "why": "Honest about mistake, shows recovery plan, demonstrates systemic learning.",
    },
    {
        "text": (
            "I do not really make big mistakes because I am very careful and always "
            "double-check my work before submitting."
        ),
        "role": "software_engineer",
        "type": "behavioural",
        "quality": "weak",
        "skill": "handling failure",
        "why": "Avoids the question, sounds defensive and unrealistic to interviewers.",
    },

    # ── BEHAVIOURAL: Data-Driven Decision Making ────────────────────────────
    {
        "text": (
            "Our conversion rate on the checkout page had dropped 3% over two months. "
            "Instead of guessing, I ran a funnel analysis in Mixpanel and found 60% of "
            "the drop was concentrated in mobile users on Android. I cross-referenced "
            "with our error logs and found a payment SDK update had introduced a "
            "rendering bug. We rolled back the SDK, conversion recovered within 48 "
            "hours, and I added an automated alert for future SDK-related regressions."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "data-driven decisions",
        "why": "Specific metrics, logical investigation, concrete fix and prevention.",
    },

    # ── TECHNICAL: System Design ────────────────────────────────────────────
    {
        "text": (
            "I would design the notification system as a pub/sub architecture. "
            "Producers publish events to a message broker like Kafka partitioned by "
            "user ID to preserve ordering. A pool of consumer workers reads from the "
            "broker, applies user preference filters, and dispatches via email/SMS "
            "adapters. For retry logic I would use exponential backoff with a "
            "dead-letter queue for failures. This decouples the notification logic "
            "from the core service, scales horizontally, and makes it easy to add "
            "new channels without touching the producers."
        ),
        "role": "software_engineer",
        "type": "technical",
        "quality": "strong",
        "skill": "system design",
        "why": "Covers scalability, fault tolerance, extensibility, and decoupling clearly.",
    },
    {
        "text": (
            "I would use a database table to store notifications and poll it every "
            "minute to send them. It is simple and easy to implement."
        ),
        "role": "software_engineer",
        "type": "technical",
        "quality": "weak",
        "skill": "system design",
        "why": "Polling is inefficient at scale, no mention of fault tolerance or ordering.",
    },

    # ── TECHNICAL: Machine Learning ─────────────────────────────────────────
    {
        "text": (
            "To detect fraudulent transactions I would start with a gradient-boosted "
            "model trained on historical labelled transactions. Key features would "
            "include transaction velocity per user, deviation from spending patterns, "
            "and merchant category risk scores. Because fraud is rare I would use "
            "stratified sampling and optimise for precision-recall AUC rather than "
            "accuracy. I would also build a simple rule-based pre-filter to catch "
            "obvious cases cheaply before hitting the model, and monitor feature "
            "drift monthly to trigger retraining."
        ),
        "role": "data_scientist",
        "type": "technical",
        "quality": "strong",
        "skill": "machine learning",
        "why": "Addresses class imbalance, correct metric choice, mentions monitoring and rules.",
    },
    {
        "text": (
            "I would train a neural network on the transaction data and it would "
            "learn to detect fraud automatically."
        ),
        "role": "data_scientist",
        "type": "technical",
        "quality": "weak",
        "skill": "machine learning",
        "why": "No feature engineering, ignores class imbalance, no evaluation strategy.",
    },

    # ── TECHNICAL: SQL / Analytics ──────────────────────────────────────────
    {
        "text": (
            "To calculate 7-day rolling retention I would use a window function: "
            "COUNT(DISTINCT user_id) over a ROWS BETWEEN 6 PRECEDING AND CURRENT ROW "
            "partition by cohort date. I would pre-aggregate daily actives into a "
            "summary table first to keep the window computation fast, and index on "
            "(cohort_date, event_date). For very large datasets I would push this to "
            "a columnar store like BigQuery or Redshift where window functions are "
            "parallelised."
        ),
        "role": "data_scientist",
        "type": "technical",
        "quality": "strong",
        "skill": "SQL",
        "why": "Correct window function syntax, performance-aware, platform-aware.",
    },

    # ── BEHAVIOURAL: Communication / Presenting to Non-Technical Audiences ──
    {
        "text": (
            "I had to present our A/B test results to the executive team. Instead of "
            "showing confidence intervals and p-values I translated the outcome into "
            "business terms: the new onboarding flow would add approximately 1,400 "
            "paying users per month at current traffic levels. I used a single slide "
            "with one number and a recommended action. The CEO approved the rollout "
            "in the meeting."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "executive communication",
        "why": "Audience-adapted language, single clear recommendation, decisive outcome.",
    },

    # ── BEHAVIOURAL: Cross-functional collaboration ─────────────────────────
    {
        "text": (
            "I was leading a feature that required input from legal, design, and "
            "backend teams simultaneously. I created a shared RACI matrix, held "
            "weekly syncs with all three, and maintained a single shared doc with "
            "open decisions and owners. When legal raised a GDPR concern two weeks "
            "before launch, I had already pre-mapped the data flows so I could "
            "respond with a documented answer within 24 hours. We launched on time "
            "with full compliance."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "cross-functional collaboration",
        "why": "Structured approach, proactive risk management, concrete outcome.",
    },

    # ── TECHNICAL: APIs and Integration ────────────────────────────────────
    {
        "text": (
            "For the REST API design I would follow resource-oriented URLs, use HTTP "
            "verbs semantically (GET for reads, POST for creates, PATCH for partial "
            "updates), and version via the URL path (v1/). I would return standard "
            "HTTP status codes, use pagination via cursor rather than offset for "
            "large collections, and include rate limiting headers. For authentication "
            "I would use short-lived JWTs with refresh tokens rather than long-lived "
            "API keys to limit exposure."
        ),
        "role": "software_engineer",
        "type": "technical",
        "quality": "strong",
        "skill": "API design",
        "why": "Covers versioning, pagination, auth, and standard conventions clearly.",
    },

    # ── BEHAVIOURAL: Prioritisation ─────────────────────────────────────────
    {
        "text": (
            "I use a modified RICE framework for prioritisation. When I inherited the "
            "backlog at my last company it had 140 items. I scored each on reach, "
            "impact, and confidence, and added a strategic alignment column tied to "
            "the three company OKRs. The bottom 60 items scored near zero and were "
            "archived. The top 10 became our next two quarters of roadmap, which I "
            "reviewed monthly with the leadership team. This reduced roadmap debates "
            "in planning from 90 minutes to 20."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "prioritisation",
        "why": "Named framework, quantified backlog reduction, measurable meeting efficiency gain.",
    },
    {
        "text": (
            "I prioritise based on what my manager tells me is most important and "
            "then I just work through the list."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "weak",
        "skill": "prioritisation",
        "why": "Passive, no framework, no independent judgment demonstrated.",
    },

    # ── TECHNICAL: Cloud / DevOps ───────────────────────────────────────────
    {
        "text": (
            "For zero-downtime deployments I use blue-green or canary strategies. "
            "In our CI/CD pipeline (GitHub Actions + ArgoCD) we deploy the new "
            "version to 5% of traffic first, monitor error rate and p99 latency for "
            "15 minutes via Datadog, and auto-promote or auto-rollback based on "
            "thresholds. Database migrations are always backwards-compatible and run "
            "as a separate step before the app deploy so both versions can run "
            "simultaneously during the transition window."
        ),
        "role": "software_engineer",
        "type": "technical",
        "quality": "strong",
        "skill": "DevOps / deployment",
        "why": "Specific toolchain, canary strategy, DB migration safety, metric-based gates.",
    },

    # ── BEHAVIOURAL: Ambiguity / Unstructured Problems ──────────────────────
    {
        "text": (
            "When I joined, the company had no clear definition of 'active user'. "
            "Different teams were using four different definitions, leading to "
            "conflicting reports in board meetings. I interviewed eight stakeholders "
            "to understand each use case, proposed three candidate definitions with "
            "pros and cons, and facilitated a decision with the CPO and CFO. We "
            "aligned on one definition, documented it in our data dictionary, and "
            "I updated the four dashboards within a week. Board reporting conflicts "
            "on this metric dropped to zero."
        ),
        "role": "data_scientist",
        "type": "behavioural",
        "quality": "strong",
        "skill": "working with ambiguity",
        "why": "Structured problem decomposition, stakeholder alignment, measurable cleanup outcome.",
    },

    # =======================================================================
    # V3 KB EXPANSION — 15 new examples added Sprint 5
    # =======================================================================

    # ── BEHAVIOURAL: Failure / Learning (V2 only had 2 — now 4) ────────────
    {
        "text": (
            "I once shipped a pricing experiment without checking that our analytics "
            "SDK was correctly attributing the new variant. Two weeks of data were "
            "unusable. When I realised, I told the VP the same day, paused the "
            "experiment, and wrote a short post-mortem identifying three preventable "
            "root causes. We now have a pre-launch checklist that includes an "
            "attribution dry run. I ran the same experiment correctly a month later "
            "and it shipped with a 6% lift."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "learning from failure",
        "why": "Owns mistake without defensiveness, fixes root cause systemically, shows follow-through.",
    },
    {
        "text": (
            "Honestly, my biggest weakness is that I work too hard and care too much "
            "about my projects. Sometimes I need to remind myself to take breaks."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "weak",
        "skill": "learning from failure",
        "why": "Textbook humble-brag non-answer. Interviewers see through this immediately.",
    },

    # ── BEHAVIOURAL: Negotiation / Difficult Conversations (new category) ──
    {
        "text": (
            "A senior engineer on my team was consistently missing code review "
            "turnaround targets, and junior engineers were blocked on him. Rather "
            "than going to his manager, I had a direct 1:1 where I framed the impact "
            "specifically: three PRs stalled for 4+ days in the last sprint. He "
            "explained he was overloaded with architecture work. We agreed he would "
            "delegate 50% of reviews to a rotation and flag his own capacity weekly. "
            "Median review time dropped from 3.5 days to under 1 day within a month."
        ),
        "role": "engineering_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "difficult conversations",
        "why": "Direct, data-grounded, preserves the relationship, concrete outcome measured.",
    },
    {
        "text": (
            "I avoid confrontation because I think it creates a bad atmosphere on "
            "the team. If someone is underperforming I usually just do their work "
            "to make sure things get done."
        ),
        "role": "engineering_manager",
        "type": "behavioural",
        "quality": "weak",
        "skill": "difficult conversations",
        "why": "Describes conflict avoidance, not management. Signals poor leadership judgment.",
    },

    # ── BEHAVIOURAL: Time Management / Competing Deadlines (new category) ──
    {
        "text": (
            "In my last quarter I had three overlapping deadlines: a board deck, a "
            "customer renewal doc, and a hiring panel I was chairing. I blocked two "
            "hours each morning for the highest-stakes item (the board deck), used "
            "afternoons for the renewal doc where I had input from others, and "
            "delegated the hiring panel coordination to a teammate with context. I "
            "renegotiated the renewal deadline by three days when I realised the "
            "scope had grown. All three landed on time and the renewal closed at "
            "112% of target."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "time management",
        "why": "Concrete prioritisation logic, delegation, proactive renegotiation, quantified outcome.",
    },
    {
        "text": (
            "I am really good at multitasking and I can handle a lot of things at "
            "once. I just work late if I need to get everything done."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "weak",
        "skill": "time management",
        "why": "Mythologises 'multitasking' (which research shows reduces performance) and normalises overwork.",
    },

    # ── TECHNICAL: Debugging / Troubleshooting (new category) ──────────────
    {
        "text": (
            "A customer reported random 500 errors on our API. I started with the "
            "logs and saw the errors clustered around a single endpoint, all with "
            "the same trace ID pattern. The stack traces pointed to a database "
            "timeout, but only on queries that joined three specific tables. "
            "I reproduced it locally with a payload from the failing request and "
            "found the query plan was doing a full scan when the user had over ten "
            "thousand records. I added a composite index, verified the plan changed "
            "to an index seek, and deployed to staging first. Error rate dropped "
            "from 0.8% to 0.01% within an hour of the prod deploy."
        ),
        "role": "software_engineer",
        "type": "technical",
        "quality": "strong",
        "skill": "debugging",
        "why": "Systematic approach: logs then hypothesis then reproduce then fix then verify then deploy safely.",
    },
    {
        "text": (
            "I usually restart the server first because that fixes most issues. "
            "If that does not work I search Stack Overflow for the error message."
        ),
        "role": "software_engineer",
        "type": "technical",
        "quality": "weak",
        "skill": "debugging",
        "why": "No structured investigation, no hypothesis-testing, signals shallow technical depth.",
    },

    # ── TECHNICAL: Data Quality / Pipeline Issues ──────────────────────────
    {
        "text": (
            "Our daily revenue dashboard started showing a 40% dip that did not "
            "match finance's numbers. I traced back through the pipeline: raw events "
            "were correct, the staging table was correct, but the aggregation job "
            "was dropping rows where currency was null. A new payment provider had "
            "started sending events without the currency field. I wrote a default-"
            "to-USD fallback plus a monitoring alert that triggers if more than 1% "
            "of events have null currency. The fix took 2 hours; the monitoring is "
            "what prevents the next version of this bug."
        ),
        "role": "data_scientist",
        "type": "technical",
        "quality": "strong",
        "skill": "data quality",
        "why": "Root-cause investigation up the pipeline, fix AND prevention, systemic thinking.",
    },

    # ── BEHAVIOURAL: Mentoring / Developing Others (new category) ──────────
    {
        "text": (
            "A junior data analyst on my team was great at SQL but struggled to "
            "frame business questions. I paired with her on three projects: we "
            "started each with a stakeholder interview where I modelled the "
            "questions, then I gradually handed that role to her. By month three "
            "she was independently running the intake for our marketing squad. Her "
            "manager later told me two business leads specifically requested her "
            "for their next analysis because her questions surfaced insights they "
            "had not thought of."
        ),
        "role": "data_scientist",
        "type": "behavioural",
        "quality": "strong",
        "skill": "mentoring",
        "why": "Concrete mentoring pattern (model then co-do then hand off), external validation of outcome.",
    },

    # ── BEHAVIOURAL: Customer Focus / User Empathy (new category) ──────────
    {
        "text": (
            "Customer support was flagging a spike in complaints about our export "
            "feature. Rather than just reading the tickets I sat with three support "
            "agents for 90 minutes each and watched them work. The pattern was clear: "
            "customers were running multi-hour exports and the progress bar reset on "
            "browser refresh, so they assumed the export had failed and retried. I "
            "shipped a fix that persisted export state server-side and added an email "
            "notification on completion. Related tickets dropped from 140 per week "
            "to under 10 within a month."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "customer focus",
        "why": "Goes beyond ticket reading, direct observation, fixes root cause not symptom.",
    },

    # ── MARKETING: Analytics / Campaign Measurement (new role) ─────────────
    {
        "text": (
            "To measure our Q3 campaign effectiveness I combined three data sources: "
            "paid media spend from our attribution tool, organic lift from a "
            "geo-holdout test in three markets, and qualitative brand surveys before "
            "and after the campaign. The paid media showed a 3.2x ROAS, but the "
            "geo-holdout showed the true incremental ROAS was closer to 1.8x because "
            "of significant organic cannibalisation. I presented both numbers to the "
            "CMO and we reallocated 20% of Q4 spend toward brand channels where the "
            "surveys showed we were under-indexing."
        ),
        "role": "marketing_analyst",
        "type": "technical",
        "quality": "strong",
        "skill": "marketing measurement",
        "why": "Triangulation of methods, honest about attribution limits, actionable reallocation.",
    },
    {
        "text": (
            "I look at the conversion numbers from Google Analytics and if they are "
            "going up the campaign is working."
        ),
        "role": "marketing_analyst",
        "type": "technical",
        "quality": "weak",
        "skill": "marketing measurement",
        "why": "Ignores attribution, incrementality, seasonality, and confounding factors.",
    },

    # ── SALES: Closing / Negotiation (new role) ────────────────────────────
    {
        "text": (
            "A mid-market deal was stalled in the legal review stage for six weeks. "
            "Their general counsel had flagged three clauses but communication had "
            "gone quiet. I proposed a 30-minute call with their GC plus our legal "
            "lead, specifically framed as walking through the three clauses "
            "together so we could close the remaining gaps the same day. We worked "
            "through two of the three live and agreed redlines on the third within "
            "48 hours. The deal closed at the original 180k ACV the next week."
        ),
        "role": "sales",
        "type": "behavioural",
        "quality": "strong",
        "skill": "deal closing",
        "why": "Proactive unblocking, specific framing that invites progress, concrete outcome.",
    },

    # ── BEHAVIOURAL: Saying No / Setting Boundaries (new category) ─────────
    {
        "text": (
            "Our VP wanted my team to take on a side project that would have delayed "
            "our Q2 roadmap by six weeks. Rather than just refusing, I built a "
            "one-page impact analysis: the side project had an estimated 300k "
            "upside, but the roadmap items we would delay had a combined 1.2M "
            "pipeline attached to them. I proposed a lighter-weight version of the "
            "side project that needed 30% of the effort and captured roughly 60% of "
            "the upside. He agreed. Both landed on their revised timelines."
        ),
        "role": "product_manager",
        "type": "behavioural",
        "quality": "strong",
        "skill": "prioritisation under pressure",
        "why": "Says no with data, offers a constructive alternative, preserves the relationship.",
    },

    # ── TECHNICAL: Code Review / Technical Mentorship ──────────────────────
    {
        "text": (
            "I follow two rules in code review: separate blocking from nitpick "
            "comments explicitly, and always explain the why behind a blocking "
            "comment. For example, on a recent PR a junior engineer used nested "
            "loops on a list that could grow to 100k items. I did not just say "
            "use a hash map. I commented with the O(n squared) vs O(n) impact at "
            "our expected scale and linked a short internal doc. She refactored it, "
            "and now uses the same pattern in her other PRs without being asked."
        ),
        "role": "software_engineer",
        "type": "behavioural",
        "quality": "strong",
        "skill": "code review",
        "why": "Specific review practice, teaches rather than dictates, shows lasting impact on mentee.",
    },

    # ── BEHAVIOURAL: Generic weak — tests weak-retrieval floor ─────────────
    {
        "text": (
            "I think the most important skill for this role is being a team player "
            "and having strong communication skills. I also believe passion and "
            "dedication are really important."
        ),
        "role": "general",
        "type": "behavioural",
        "quality": "weak",
        "skill": "generic platitudes",
        "why": "Pure buzzword soup. Contains no information about the candidate.",
    },
]


def get_knowledge_base() -> list[dict]:
    """Return the full knowledge base as a list of document dicts."""
    return KNOWLEDGE_BASE


def get_roles() -> list[str]:
    """Return sorted list of unique role values in the KB."""
    return sorted(set(d["role"] for d in KNOWLEDGE_BASE))
