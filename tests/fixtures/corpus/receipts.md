# Delivery receipts

A receipt records what happened to one message. Receipts are retained for 90 days.

## Fetching a receipt

```python
receipt = client.get_receipt("rcpt_01H...")
print(receipt.status, receipt.attempts, receipt.last_error)
```

## Fields

- `id` — the receipt id, prefixed `rcpt_`.
- `status` — `queued`, `delivered`, `failed` or `dropped`.
- `attempts` — how many delivery attempts were made.
- `last_error` — the error from the final failed attempt, or null.
- `delivered_at` — timestamp of successful delivery, or null.

## Listing receipts

`client.list_receipts(status="failed", since="2026-01-01")` returns an iterator. The window may
not exceed 90 days, which is the retention period.
