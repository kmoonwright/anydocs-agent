# Specialists

One paragraph per specialist. **The orchestrator's planning prompt is assembled from these
paragraphs**, so a new specialist becomes routable by being described here — do not edit the
orchestrator prompt to add one. Write each paragraph as a job description: what it is for, what
it is good at, and when the orchestrator should choose it. The planner reads nothing else about
them.

## librarian

Answers from documentation already in the store. Rewrites the question into one to three
retrieval queries, searches across the active corpora, reads the passages that come back, and
drafts an answer with inline citations that resolve to a real URL and heading. Reports a
retrieval confidence and lists any sub-question the store could not answer. **Choose the
Librarian for every question that might already be answerable** — it runs first on the hot path,
and its confidence is what decides whether anything else needs to happen. It cannot fetch,
crawl or save.

## scout

Finds documentation the store lacks. Searches the web with site and version hints, judges which
results are a vendor's own documentation versus blogs and forums, and probes the best candidate
for its doc root and page count. Returns ranked source candidates, each with a reason, plus one
recommended root and corpus name — or none, when nothing authoritative turned up. **Choose the
Scout when the Librarian's retrieval confidence is low, or when a user asks the agent to learn a
product it has never seen.** It cannot save anything; it only recommends.

## archivist

Turns pages into durable knowledge. Crawls an approved root, digests each page into a structured
summary with its facts tied to source spans, writes each digest with provenance into a versioned
snapshot, and reports pages added, unchanged, failed and robots-blocked along with the three
questions the new corpus best answers. **Choose the Archivist after a crawl has been approved**,
or to re-ingest a page that changed. It is the only specialist that can write to the store.

## auditor

Checks a draft before it ships. Splits the answer into claims, verifies each against the passage
it was cited from, lints code samples against the cited pages' API surface, checks that version
markers in the answer match the sources, and returns a pass-or-revise verdict with per-claim
verdicts, a confidence score and the reasons behind it. **The Auditor runs on every turn that
produces a draft**, after the draft exists and before the answer is returned — a numeric question
answered by the Analyst never reaches it, because there is no cited passage to verify against. It reads; it never writes.

## analyst

Answers numeric questions about documentation usage — page views, search misses, time to answer
by section — by translating a question into a parameterized query against the analytics store,
formatting the result in human units and stating its assumptions. **Choose the Analyst when the
question asks for a number about the docs themselves rather than for their content** — how often
a page is read, which searches find nothing, how long answers take. It cannot answer from page
content, and it never estimates a number it could not query for.
