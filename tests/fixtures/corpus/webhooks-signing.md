# Verifying webhook signatures

Relayr signs every delivery so the receiver can verify it came from you.

```python
from relayr import verify_signature

verify_signature(
    body=request.body,
    header=request.headers["Relayr-Signature"],
    secret="whsec_...",
    tolerance_seconds=300,
)
```

- The signature header contains a timestamp and an HMAC-SHA256 of `timestamp.body`.
- `tolerance_seconds` rejects deliveries older than the window. Default is 300.
- Verify against the raw request body, not a re-serialized copy.
- `verify_signature` raises `SignatureError` on mismatch; it does not return false.
