"""
Sarvam AI Integration Service — CallZenza AI Receptionist.

Provides connectivity and client authentication for Sarvam AI
(Indian multilingual voice, speech, translation, and Samvaad models).

Security Rules:
- API Key is read strictly from environment variables (SARVAM_API_KEY / settings.SARVAM_API_KEY).
- The raw key is NEVER exposed in logs, API responses, console output, or frontend code.
"""

import logging
import os
from typing import Dict, Any, Optional, List
import httpx

from src.config.env import settings

logger = logging.getLogger(__name__)


class SarvamService:
    """Service to interact with Sarvam AI APIs."""

    def __init__(self):
        self._base_url = getattr(settings, "SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")

    @property
    def api_key(self) -> Optional[str]:
        """Dynamically fetch the latest API key from settings or environment."""
        return getattr(settings, "SARVAM_API_KEY", None) or os.getenv("SARVAM_API_KEY")

    def is_configured(self) -> bool:
        """Check whether the Sarvam API key is configured."""
        key = self.api_key
        return bool(key and key.strip())

    def get_masked_key(self) -> str:
        """Return a safely masked representation of the API key."""
        key = self.api_key
        if not key:
            return "NOT_CONFIGURED"
        clean = key.strip()
        if len(clean) <= 8:
            return "***"
        return f"{clean[:6]}...{clean[-4:]}"

    def get_headers(self, auth_type: str = "subscription") -> Dict[str, str]:
        """
        Generate headers for Sarvam AI requests.
        Supports both 'subscription' (api-subscription-key) and 'bearer' (Authorization: Bearer ...).
        """
        key = self.api_key or ""
        headers = {
            "Content-Type": "application/json"
        }
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers["api-subscription-key"] = key
        return headers

    def verify_authentication(self, timeout: float = 8.0) -> Dict[str, Any]:
        """
        Verify that the Sarvam API key authenticates successfully against Sarvam AI.
        Does not leak or display the raw API key in logs or results.
        """
        if not self.is_configured():
            return {
                "success": False,
                "authenticated": False,
                "message": "SARVAM_API_KEY is not configured in environment.",
                "status_code": 0
            }

        key = self.api_key
        masked = self.get_masked_key()

        # Try models endpoint which supports both subscription key and bearer token
        url = f"{self._base_url}/v1/models"
        logger.info(f"[SarvamService] Verifying authentication with key ({masked}) against {url}...")

        # Test with Bearer auth first, fallback to api-subscription-key
        for auth_mode in ["bearer", "subscription"]:
            headers = self.get_headers(auth_type=auth_mode)
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("id") for m in data.get("data", [])]
                        logger.info(f"[SarvamService] Authentication verified successfully via {auth_mode} (HTTP 200).")
                        return {
                            "success": True,
                            "authenticated": True,
                            "auth_mode": auth_mode,
                            "models": models,
                            "status_code": 200,
                            "message": "Sarvam AI API authentication verified successfully."
                        }
                    elif resp.status_code in (401, 403):
                        safe_body = resp.text.replace(key, "[MASKED]") if key else resp.text
                        logger.warning(f"[SarvamService] Auth check returned HTTP {resp.status_code}: {safe_body}")
            except httpx.ConnectError:
                return {
                    "success": False,
                    "authenticated": False,
                    "status_code": 0,
                    "message": "Network connection error while reaching api.sarvam.ai. Check internet access."
                }
            except Exception as exc:
                safe_err = str(exc).replace(key, "[MASKED]") if key else str(exc)
                logger.error(f"[SarvamService] Auth check exception: {safe_err}")
                return {
                    "success": False,
                    "authenticated": False,
                    "status_code": 0,
                    "message": f"Connection error: {safe_err}"
                }

        return {
            "success": False,
            "authenticated": False,
            "status_code": 401,
            "message": "Sarvam AI API key failed authentication check."
        }


# Global singleton
sarvam_service = SarvamService()
