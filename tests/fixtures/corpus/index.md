# Relayr

Relayr is a message delivery SDK for sending webhooks and notifications with automatic retries,
idempotency and delivery receipts. These docs cover v3; v2 is in maintenance until 2027-01-01.

- Install with `pip install relayr`.
- Every send requires an idempotency key.
- Deliveries are retried automatically on 5xx responses and connection errors.
- Delivery receipts are available for 90 days.

## Where to start

Read [Quickstart](quickstart.md) for a first send, then [Retries](retries.md) for how failures
are handled.
