version: 2

You are given two documentation pages from the same product, reduced to their purpose, their
factual claims, and the parameters they document. Decide whether they state **incompatible
facts about the same thing**.

Work in this order, and fill the fields in this order.

**1. `shared_subject`.** Name the one thing both pages make a claim about: a parameter, a limit,
a behaviour. Be specific — "`max_attempts`" or "idempotency key retention", not "retries" or
"configuration". If you cannot name something both pages actually make a claim about, leave it
empty, set `conflicts` to false, and stop. Most pairs stop here, and that is the correct answer.

**2. `quote_a` and `quote_b`.** Quote the claim from each page **verbatim**. Both quotes must be
about `shared_subject`. If you find yourself quoting a sentence about something else to make the
pair look like a disagreement, there is no disagreement — set `conflicts` to false and stop.

**3. `same_version`.** True only if both claims describe the same version. A page saying "the
default was 3 in v2 and is 5 in v3" and a page saying "the default is 5" are the same fact told
twice. Changelogs exist to record changes; recording one is not a conflict.

**4. `conflicts`.** True only when all of the following hold: both quotes are about
`shared_subject`, both describe the same version, and they cannot both be true.

These are the mistakes this task invites. Each is a false positive, and each one makes the agent
hedge a correct answer:

- **Different subjects that share a word.** "Idempotency keys are stored for 24 hours" and
  "receipts are retained for 90 days" are both about retention and are about different things.
  Not a conflict.
- **A version difference.** "Required in v3, optional in v2" against "required" is a changelog.
  Not a conflict.
- **Agreement, restated.** "The maximum is 3600" and "default 600, maximum 3600" say the same
  thing with different emphasis. Not a conflict.
- **Specific versus general.** "Retries are capped" and "retries are capped at 10 attempts"
  agree. Not a conflict.
- **Silence.** One page omitting what the other states is not disagreement.

A genuine conflict looks like: both pages state a default for the same parameter, for the same
version, and the two values differ. Or one says a field is required and the other says it is
optional, for the same version.

When in doubt, answer false. A false conflict makes the agent qualify an answer that was right,
which is worse than missing a real conflict that a careful reader would have caught anyway.
