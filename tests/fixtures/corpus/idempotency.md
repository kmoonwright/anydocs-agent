# Idempotency

Every `send` requires an `idempotency_key`. Relayr stores the key for 24 hours.

- Sending twice with the same key returns the first receipt and does not deliver again.
- Keys are scoped per API key, not globally.
- A key may be up to 255 characters.
- After 24 hours a reused key is treated as a new send.

```python
receipt_a = client.send(destination=url, payload=data, idempotency_key="order-1234")
receipt_b = client.send(destination=url, payload=data, idempotency_key="order-1234")
assert receipt_a.id == receipt_b.id
```

## Choosing a key

Use something derived from the event you are delivering, not a random value — a random key
defeats the purpose. An order id, an event id or a hash of the payload all work.
