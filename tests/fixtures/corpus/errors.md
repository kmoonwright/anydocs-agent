# Errors

Relayr raises a typed exception per failure class.

```python
from relayr import DeliveryError, RateLimited, InvalidPayload

try:
    client.send(destination=url, payload=data, idempotency_key=key)
except InvalidPayload as exc:
    log.error("payload rejected: %s", exc.field)
except RateLimited as exc:
    sleep(exc.retry_after)
except DeliveryError as exc:
    log.error("delivery failed after %d attempts", exc.attempts)
```

- `InvalidPayload` — the payload is malformed or too large. Not retried.
- `RateLimited` — HTTP 429. Carries `retry_after` in seconds.
- `DeliveryError` — retries were exhausted. Carries `attempts` and `receipt_id`.
- `ConfigurationError` — raised at client construction, e.g. a missing API key.
