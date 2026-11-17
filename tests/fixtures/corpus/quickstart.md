# Quickstart

Send your first message in three lines. This page applies to v3.

## Install

```bash
pip install relayr
```

## Send a message

```python
from relayr import Client

client = Client(api_key="rk_live_...")
receipt = client.send(
    destination="https://example.com/hook",
    payload={"event": "order.created"},
    idempotency_key="order-1234",
)
print(receipt.id, receipt.status)
```

- `destination` must be an absolute https URL.
- `payload` is serialized as JSON; it must be under 256 KB.
- `idempotency_key` is required. Reusing a key returns the original receipt rather than sending
  again.

## What you get back

`send` returns a `Receipt` with `id`, `status` and `attempts`. Status is one of `queued`,
`delivered`, `failed` or `dropped`.
