"""
Interview Knowledge Base
A curated set of example interview answers (strong and weak) across roles and skills.
These are the documents indexed by the RAG engine.
"""

KNOWLEDGE_BASE = [

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
]


def get_knowledge_base() -> list[dict]:
    """Return the full knowledge base as a list of document dicts."""
    return KNOWLEDGE_BASE


def get_roles() -> list[str]:
    """Return sorted list of unique role values in the KB."""
    return sorted(set(d["role"] for d in KNOWLEDGE_BASE))
