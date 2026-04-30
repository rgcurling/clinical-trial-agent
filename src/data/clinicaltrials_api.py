"""
Async client for ClinicalTrials.gov API v2.

Designed for use inside FastAPI (async) and as a standalone sync wrapper.
All network calls use aiohttp with exponential back-off on 429/5xx.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)


class ClinicalTrialsAPI:
    """
    Async client for ClinicalTrials.gov API v2.

    Usage (inside async context):
        async with ClinicalTrialsAPI() as api:
            trials = await api.search("non-small cell lung cancer", location="Boston, MA")
    """

    def __init__(self, *, timeout: int = 30, max_retries: int = 3):
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "ClinicalTrialsAPI":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()

    # ── public API ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query_text: str,
        location: Optional[str] = None,
        max_distance: int = 50,
        status: str = "RECRUITING",
        limit: int = 50,
    ) -> list[dict]:
        """
        Search for trials matching *query_text*.

        Returns a list of normalized trial dicts with keys:
            nct_id, title, phase, status, summary, eligibility,
            conditions, locations, contact, trial_url
        """
        params = self._build_params(query_text, location, max_distance, status, limit)
        raw = await self._get_with_retry(BASE_URL, params)
        studies = raw.get("studies", [])
        return [t for s in studies if (t := self._parse_study(s)) is not None]

    # ── internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_params(
        query_text: str,
        location: Optional[str],
        max_distance: int,
        status: str,
        limit: int,
    ) -> dict:
        params: dict = {
            "query.cond": query_text,
            "filter.overallStatus": status,
            "pageSize": min(limit, 100),
            "format": "json",
        }
        if location:
            params["filter.geo"] = f"distance({location},{max_distance}mi)"
        return params

    async def _get_with_retry(self, url: str, params: dict) -> dict:
        session = self._session
        if session is None:
            raise RuntimeError("Use ClinicalTrialsAPI as an async context manager")

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status in (429, 500, 502, 503):
                        wait = 2 ** attempt
                        logger.warning(f"HTTP {resp.status} — retrying in {wait}s (attempt {attempt+1})")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientError as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(f"Request error — retrying in {wait}s: {exc}")
                await asyncio.sleep(wait)

        raise RuntimeError(f"ClinicalTrials.gov request failed after {self._max_retries} attempts") from last_exc

    @staticmethod
    def _parse_study(study: dict) -> Optional[dict]:
        try:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            cond_mod = proto.get("conditionsModule", {})
            elig_mod = proto.get("eligibilityModule", {})
            design_mod = proto.get("designModule", {})
            desc_mod = proto.get("descriptionModule", {})
            contacts_mod = proto.get("contactsLocationsModule", {})

            nct_id = id_mod.get("nctId", "")
            if not nct_id:
                return None

            title = id_mod.get("officialTitle") or id_mod.get("briefTitle", "")
            phases = design_mod.get("phases", [])
            phase = phases[0] if phases else None
            overall_status = status_mod.get("overallStatus", "")
            conditions = cond_mod.get("conditions", [])
            eligibility = elig_mod.get("eligibilityCriteria", "")
            summary = desc_mod.get("briefSummary", "") or desc_mod.get("detailedDescription", "")

            locations = []
            primary_contact: dict = {}
            for loc in contacts_mod.get("locations", []):
                city = loc.get("city", "")
                state = loc.get("state", "")
                if city:
                    locations.append(f"{city}, {state}".strip(", "))
                if not primary_contact:
                    for contact in loc.get("contacts", []):
                        if contact.get("name"):
                            primary_contact = {
                                "name": contact.get("name", ""),
                                "email": contact.get("email", ""),
                                "phone": contact.get("phone", ""),
                            }
                            break

            return {
                "nct_id": nct_id,
                "title": title,
                "phase": phase,
                "status": overall_status,
                "summary": summary,
                "eligibility": eligibility,
                "conditions": conditions,
                "locations": locations,
                "contact": primary_contact,
                "trial_url": f"https://clinicaltrials.gov/study/{nct_id}",
            }
        except Exception as exc:
            logger.warning(f"Failed to parse study: {exc}")
            return None


# ── sync convenience wrapper ───────────────────────────────────────────────────

def search_sync(
    query_text: str,
    location: Optional[str] = None,
    status: str = "RECRUITING",
    limit: int = 50,
) -> list[dict]:
    """Blocking wrapper for use outside async contexts."""

    async def _run():
        async with ClinicalTrialsAPI() as api:
            return await api.search(query_text, location=location, status=status, limit=limit)

    return asyncio.run(_run())
