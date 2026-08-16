"""
Checks current rate-limit/quota status for one or more Groq API keys, by
making one minimal, cheap request per key and reading the rate-limit
headers Groq returns on every response (success or not) — so you can see
where you stand WITHOUT waiting to hit an actual 429 error.

This still costs a small number of tokens per key (unavoidable — you have
to make a real request to get real headers back), but uses a tiny 1x1
pixel image and asks for only 1 output token to keep the cost minimal.

Note: Groq's exact header names/meanings for "tokens per minute" vs
"tokens per day" vs "requests per day" have varied across their docs, and
I can't verify the exact current behavior against a real key from here —
so this script prints EVERY header it gets back that looks rate-limit
related, unfiltered, rather than guessing which one means what. Read the
header names themselves (they're usually self-explanatory, e.g.
"x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens") and compare
against console.groq.com/settings/limits if anything looks unclear.

Setup:
    setx GROQ_API_KEYS "key1,key2"
    (then close and reopen your terminal)

Usage:
    python check_quota.py
"""
import os
import sys
import base64
from groq import Groq

VISION_MODEL = "qwen/qwen3.6-27b"

# A minimal 2x2 white PNG (Groq's vision model rejects images smaller than
# 2x2 pixels in either dimension), to keep the probe request as cheap as possible
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8//8/AwMDEwMDAwMDAwAkBgMB"
    "/DXemwAAAABJRU5ErkJggg=="
)


def check_key(label, api_key):
    client = Groq(api_key=api_key)
    print(f"=== {label} ===")
    try:
        response = client.chat.completions.with_raw_response.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "reply with just: ok"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=1,
        )
        headers = response.headers
        found_any = False
        for name in headers.keys():
            if "ratelimit" in name.lower() or "retry-after" in name.lower():
                print(f"  {name}: {headers[name]}")
                found_any = True
        if not found_any:
            print("  (no rate-limit headers found in this response — Groq's API may "
                  "have changed; check console.groq.com/settings/limits instead)")
    except Exception as e:
        print(f"  Request failed: {e}")
        print("  (if this is a 429, the error message itself usually says how long to wait)")
    print()


def main():
    keys_env = os.environ.get("GROQ_API_KEYS", "")
    api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]
    if not api_keys:
        print("GROQ_API_KEYS environment variable is not set.")
        print('Run: setx GROQ_API_KEYS "key1,key2"')
        print("Then close and reopen your terminal, and try again.")
        sys.exit(1)

    for i, key in enumerate(api_keys, start=1):
        check_key(f"Key #{i}", key)


if __name__ == "__main__":
    main()