# Dead letters

A message that exhausts its retries is marked `failed`. Relayr does not automatically move it
anywhere; you decide what happens next.

## Subscribing to failures

```python
@client.on("delivery.failed")
def handle_failure(event):
    store_for_later(event.receipt_id, event.payload)
```

- The `delivery.failed` event carries the receipt id and the original payload.
- Events are delivered at least once, so handlers must be idempotent.
- Payloads in failure events are retained for 7 days, not 90.

## Replaying

`client.replay(receipt_id)` re-sends the original payload with a new idempotency key. Replay is
available while the failure event's payload is still retained — that is, for 7 days.
