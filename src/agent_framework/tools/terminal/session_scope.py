"""Bind terminal labels to their owner, including explicit aliases."""
import hashlib
import re


def scoped_terminal_id(owner, label=None):
    if not isinstance(owner, str) or not owner:
        return label or 'default'
    if not label or label == owner:
        return owner
    # Idempotence is needed at both the manager boundary and the sandbox.
    if re.fullmatch(re.escape(owner) + r'-[a-f0-9]{32}', str(label)):
        return label
    return owner + '-' + hashlib.sha256(str(label).encode()).hexdigest()[:32]
