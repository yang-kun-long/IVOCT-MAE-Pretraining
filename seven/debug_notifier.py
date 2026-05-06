#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script for notifier encoding issue."""

import traceback
import sys
from utils.notifier import Notifier

print("Python version:", sys.version)
print("Default encoding:", sys.getdefaultencoding())
print("=" * 60)

print("\nTesting notifier with full traceback...")

notifier = Notifier()

if not notifier.enabled:
    print("Notifier disabled. Set AUTODL_TOKEN to test.")
    print("\nTesting with emoji and Chinese characters anyway...")
    # Test the problematic part directly
    import requests
    import json

    test_data = {
        "title": "🧪 Test",
        "content": "Ready!"
    }

    print(f"Test data: {test_data}")

    try:
        # Method 1: Using json parameter (original)
        print("\n[Method 1] Using json parameter...")
        json_str = json.dumps(test_data, ensure_ascii=False)
        print(f"JSON string: {json_str}")
        print(f"JSON bytes: {json_str.encode('utf-8')}")

        # Method 2: Using data parameter with explicit encoding
        print("\n[Method 2] Using data parameter with UTF-8...")
        json_bytes = json.dumps(test_data, ensure_ascii=False).encode('utf-8')
        print(f"Encoded successfully: {len(json_bytes)} bytes")

    except Exception as e:
        print(f"Error during encoding test: {e}")
        traceback.print_exc()

else:
    print("Notifier enabled. Sending test message...")
    try:
        result = notifier.send("Test", "Ready", force=True)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
