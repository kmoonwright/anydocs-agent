version: 1

Summarize a conversation between a user and a documentation agent, so the agent can keep
answering follow-up questions after the raw turns are dropped.

Write one short paragraph covering only what a follow-up would need:

- What the user is trying to do, in their words where possible.
- Which product, corpus and versions are in play.
- What has already been answered, in one clause each — not the answers themselves.
- Anything they corrected the agent about, or said they did not want. These survive longest and
  are the most expensive to lose.

Leave out: the content of the answers, citations, your own commentary on how the conversation
went, and anything the user did not actually say. This summary replaces the turns it covers, so
a detail you omit is gone — but an invention is worse than an omission, because the next answer
will be built on it.

Write it as plain prose addressed to no one. No preamble, no "the user asked about".
