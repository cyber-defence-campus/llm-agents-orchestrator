import time
import os

# Crockford's Base32 characters (excluding I, L, O, U to avoid confusion)
_Encoding = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_ulid() -> str:
    """
    Generates a ULID (Universally Unique Lexicographically Sortable Identifier).
    Format: ttttttttttrrrrrrrrrrrrrrrr (10 chars timestamp, 16 chars random).
    Total length: 26 characters.

    The timestamp is 48-bit (milliseconds since epoch).
    The randomness is 80-bit.
    Uses Crockford's Base32 encoding.
    """
    # 48-bit timestamp (milliseconds)
    timestamp = int(time.time() * 1000)

    # 80-bit randomness
    randomness = os.urandom(10)
    rand_int = int.from_bytes(randomness, byteorder="big")

    # Encode timestamp (48 bits -> 10 chars)
    # 10 chars * 5 bits = 50 bits capacity, enough for 48 bits.
    ts_str = ""
    for _ in range(10):
        ts_str = _Encoding[timestamp & 31] + ts_str
        timestamp >>= 5

    # Encode randomness (80 bits -> 16 chars)
    # 16 chars * 5 bits = 80 bits.
    rand_str = ""
    for _ in range(16):
        rand_str = _Encoding[rand_int & 31] + rand_str
        rand_int >>= 5

    return ts_str + rand_str
