"""
Training notification system for AutoDL.
Supports multiple notification backends:
1. AutoDL API (recommended for AutoDL platform)
2. Server酱 (ServerChan) for WeChat notifications

AutoDL API:
    - Get token from: https://www.autodl.com/console/account/info
    - Rate limit: 1 request per minute
    - Free tier: 100 messages per day

Server酱:
    - Get SendKey from: https://sct.ftqq.com/
    - Rate limit: varies by plan

Usage:
    # Method 1: AutoDL API (recommended)
    export AUTODL_TOKEN="your_token"
    notifier = Notifier(backend="autodl")

    # Method 2: Server酱
    export SERVERCHAN_KEY="your_key"
    notifier = Notifier(backend="serverchan")

    # Send notifications
    notifier.send("Training Started", "Epoch 1/200")
"""

import os
import requests
import time
from datetime import datetime
from typing import Optional, Literal


class Notifier:
    """Send training notifications via AutoDL API or Server酱."""

    def __init__(
        self,
        backend: Literal["autodl", "serverchan"] = "autodl",
        token: Optional[str] = None,
        rate_limit: float = 60.0  # seconds between notifications
    ):
        """
        Initialize notifier.

        Args:
            backend: "autodl" or "serverchan"
            token: API token/key. If None, reads from env var.
            rate_limit: Minimum seconds between notifications (default: 60s for AutoDL)
        """
        self.backend = backend
        self.rate_limit = rate_limit
        self.last_send_time = 0

        if backend == "autodl":
            self.token = token or os.environ.get("AUTODL_TOKEN")
            self.api_url = "https://www.autodl.com/api/v1/wechat/message/push"
        elif backend == "serverchan":
            self.token = token or os.environ.get("SERVERCHAN_KEY")
            self.api_url = f"https://sctapi.ftqq.com/{self.token}.send" if self.token else None
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.enabled = self.token is not None

        if not self.enabled:
            print(f"[Notifier] Warning: No token provided for {backend}. Notifications disabled.")
            if backend == "autodl":
                print("[Notifier] Get your token at: https://www.autodl.com/console/account/info")
            else:
                print("[Notifier] Get your key at: https://sct.ftqq.com/")

    def _check_rate_limit(self) -> bool:
        """Check if enough time has passed since last notification."""
        now = time.time()
        elapsed = now - self.last_send_time

        if elapsed < self.rate_limit:
            remaining = self.rate_limit - elapsed
            print(f"[Notifier] Rate limit: wait {remaining:.0f}s before next notification")
            return False

        return True

    def send(self, title: str, content: str = "", force: bool = False) -> bool:
        """
        Send notification.

        Args:
            title: Notification title (required)
            content: Notification content (optional)
            force: Skip rate limit check (use sparingly)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        # Rate limit check (unless forced)
        if not force and not self._check_rate_limit():
            return False

        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if self.backend == "autodl":
                # AutoDL API - use simple ASCII-safe format
                import json

                # Ensure title and content are strings
                title_str = str(title)
                content_str = str(content) if content else ""

                data = {
                    "title": title_str,
                    "content": f"[{timestamp}]\n{content_str}" if content_str else f"[{timestamp}]"
                }

                headers = {
                    "Authorization": "Bearer {}".format(self.token),
                    "Content-Type": "application/json"
                }

                # Encode JSON with UTF-8
                json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')

                response = requests.post(
                    self.api_url,
                    data=json_data,
                    headers=headers,
                    timeout=10
                )

            elif self.backend == "serverchan":
                # Server酱 API
                data = {
                    "title": title,
                    "desp": f"**Time**: {timestamp}\n\n{content}"
                }
                response = requests.post(self.api_url, data=data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if self.backend == "autodl":
                    success = result.get("code") == "Success"
                else:
                    success = result.get("code") == 0

                if success:
                    print(f"[Notifier] ✓ Sent: {title}")
                    self.last_send_time = time.time()
                    return True
                else:
                    error_msg = result.get("message") or result.get("msg") or result
                    print(f"[Notifier] ✗ Failed: {error_msg}")
                    return False
            else:
                print(f"[Notifier] ✗ HTTP {response.status_code}: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"[Notifier] ✗ Error: {e}")
            return False

    def training_started(self, config: dict) -> bool:
        """Notify training started."""
        title = "🚀 MAE Training Started"
        content = f"""Dataset: {config.get('dataset_size', 'N/A')} images
Batch: {config.get('batch_size', 'N/A')}, Epochs: {config.get('epochs', 'N/A')}
Mode: {config.get('mode', 'N/A')}
GPU: {config.get('gpu', 'N/A')}"""
        return self.send(title, content, force=True)

    def epoch_completed(self, epoch: int, total: int, metrics: dict) -> bool:
        """Notify epoch completed (only for milestones to respect rate limit)."""
        # Only notify every 20 epochs or last epoch (to stay within rate limit)
        if epoch % 20 != 0 and epoch != total:
            return False

        title = f"📊 Epoch {epoch}/{total}"
        content = f"""Loss: {metrics.get('total_loss', 'N/A'):.6f}
MSE: {metrics.get('mse_loss', 'N/A'):.6f}
SSIM: {metrics.get('ssim_loss', 'N/A'):.6f}"""
        return self.send(title, content)

    def checkpoint_saved(self, epoch: int, loss: float, is_best: bool = False) -> bool:
        """Notify checkpoint saved (only for best model to respect rate limit)."""
        if not is_best:
            return False  # Skip regular checkpoints

        title = "⭐ New Best Model"
        content = f"Epoch {epoch}, Loss: {loss:.6f}"
        return self.send(title, content)

    def training_completed(self, total_epochs: int, best_loss: float, time_elapsed: str) -> bool:
        """Notify training completed."""
        title = "🎉 Training Completed"
        content = f"""Epochs: {total_epochs}
Best Loss: {best_loss:.6f}
Time: {time_elapsed}"""
        return self.send(title, content, force=True)

    def training_error(self, epoch: int, error: str) -> bool:
        """Notify training error."""
        title = "❌ Training Error"
        content = f"Epoch {epoch} failed\n{error[:100]}"
        return self.send(title, content, force=True)


# Convenience function
def create_notifier(sendkey: Optional[str] = None) -> Notifier:
    """Create a notifier instance."""
    return Notifier(sendkey=sendkey)


if __name__ == "__main__":
    # Test notification
    notifier = Notifier()

    if notifier.enabled:
        print("Testing notification...")
        notifier.send(
            title="🧪 Test Notification",
            content="This is a test from IVOCT MAE training system",
            desp="If you receive this, notifications are working correctly!"
        )
    else:
        print("Please set SERVERCHAN_KEY environment variable to test.")
        print("Get your key at: https://sct.ftqq.com/")
