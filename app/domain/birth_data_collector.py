"""
Birth data multi-turn collection state machine.

Manages the multi-turn conversational flow that collects a user's birth
data (name, gender, date of birth, birth hour) before the first Tu Vi
reading can be generated.

Responsibilities
----------------
- ``_PartialBirthData``: accumulate fields across turns, detect completeness.
- ``BirthDataCollector``: own the in-memory ``_pending`` dict and orchestrate
  the collection flow using the LLM for free-text extraction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.core.interfaces import LLMService, TuViEnginePort, VectorStoreRepository
from app.domain.models import BirthData, Gender
from app.domain.prompts import (
    BIRTH_DATA_PROMPT,
    BIRTH_EXTRACTION_SYSTEM,
    FIELD_LABELS,
)

if TYPE_CHECKING:
    pass  # forward refs only

logger = logging.getLogger(__name__)

# Ordered list of double-hour names (index matches birth_hour integer 0–11)
_HOUR_NAMES = [
    "Ty", "Suu", "Dan", "Mao", "Thin", "Ty",
    "Ngo", "Mui", "Than", "Dau", "Tuat", "Hoi",
]


# =============================================================================
#  _PartialBirthData
# =============================================================================


@dataclass
class _PartialBirthData:
    """Accumulates birth data fields across multiple messages."""

    name: str | None = None
    gender: Gender | None = None
    solar_dob: str | None = None
    birth_hour: int | None = None  # None = not yet provided; -1 = unknown

    @property
    def missing_fields(self) -> list[str]:
        """Return human-readable labels for fields still missing."""
        missing: list[str] = []
        if not self.name:
            missing.append(FIELD_LABELS["name"])
        if self.gender is None:
            missing.append(FIELD_LABELS["gender"])
        if not self.solar_dob:
            missing.append(FIELD_LABELS["solar_dob"])
        if self.birth_hour is None:
            missing.append(FIELD_LABELS["birth_hour"])
        return missing

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields) == 0

    def to_birth_data(self) -> BirthData:
        """Convert to validated BirthData. Only call when ``is_complete``."""
        assert self.is_complete, "Cannot convert incomplete birth data"
        return BirthData(
            name=self.name,  # type: ignore[arg-type]
            gender=self.gender,  # type: ignore[arg-type]
            solar_dob=self.solar_dob,  # type: ignore[arg-type]
            birth_hour=self.birth_hour if self.birth_hour is not None else -1,
        )

    def merge(self, other: _PartialBirthData) -> None:
        """Merge fields from *other* into self, keeping existing values."""
        if other.name and not self.name:
            self.name = other.name
        if other.gender is not None and self.gender is None:
            self.gender = other.gender
        if other.solar_dob and not self.solar_dob:
            self.solar_dob = other.solar_dob
        if other.birth_hour is not None and self.birth_hour is None:
            self.birth_hour = other.birth_hour


# =============================================================================
#  BirthDataCollector
# =============================================================================


class BirthDataCollector:
    """
    Multi-turn state machine that collects birth data from a user.

    Owns the in-memory ``_pending`` dict (keyed by ``user_id``).
    Delegates LLM extraction and persistence to injected dependencies.
    """

    def __init__(
        self,
        llm: LLMService,
        store: VectorStoreRepository,
        tuvi: TuViEnginePort,
        generate_answer_fn: Any,  # callable(query, chart_json, short_term, long_term) -> str
        persist_messages_fn: Any,  # callable(user_id, query, answer) -> None
    ) -> None:
        self._llm = llm
        self._store = store
        self._tuvi = tuvi
        self._generate_answer = generate_answer_fn
        self._persist_messages = persist_messages_fn
        self._pending: dict[str, _PartialBirthData] = {}

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def handle(self, user_id: str, query: str) -> str:
        """
        Process a user message during the birth-data collection phase.

        Returns a string to send back to the user — either a prompt for
        more information or the first Tu Vi reading if all data is now
        available.
        """
        accumulated = self._pending.get(user_id)

        # First interaction: check whether the message already has birth data
        if accumulated is None:
            extracted = self._extract(query)
            if extracted is not None and extracted.is_complete:
                birth_data = extracted.to_birth_data()
                self._save_birth_data(user_id, birth_data)
                return self._first_reading(user_id, query, birth_data)

            if extracted is not None and any([
                extracted.name,
                extracted.gender,
                extracted.solar_dob,
                extracted.birth_hour is not None,
            ]):
                self._pending[user_id] = extracted
                return self._reprompt(extracted)

            return BIRTH_DATA_PROMPT

        # Subsequent message: accumulate and check completeness
        extracted = self._extract(query)
        if extracted is not None:
            accumulated.merge(extracted)

        self._pending[user_id] = accumulated

        if accumulated.is_complete:
            birth_data = accumulated.to_birth_data()
            del self._pending[user_id]
            self._save_birth_data(user_id, birth_data)
            return self._first_reading(user_id, query, birth_data)

        return self._reprompt(accumulated)

    def save_birth_data(self, user_id: str, birth_data: BirthData) -> None:
        """Persist birth data for a user (public alias for external callers)."""
        self._save_birth_data(user_id, birth_data)

    def get_birth_data(self, user_id: str) -> BirthData | None:
        """Retrieve the user's stored birth data."""
        try:
            user_doc = self._store.get_user(user_id)
            if user_doc is None:
                return None
            bd = user_doc.get("birth_data")
            if bd is None:
                return None
            return BirthData(**bd)
        except Exception:
            logger.exception("Failed to retrieve birth data for user %s", user_id)
            return None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _save_birth_data(self, user_id: str, birth_data: BirthData) -> None:
        self._store.upsert_user(
            user_id,
            {"birth_data": birth_data.model_dump()},
        )
        logger.info("Saved birth data for user %s: %s", user_id, birth_data.name)

    def _reprompt(self, partial: _PartialBirthData) -> str:
        """Build a friendly message asking for only the missing fields."""
        ack_parts: list[str] = []
        if partial.name:
            ack_parts.append(f"Ho ten: {partial.name}")
        if partial.gender is not None:
            ack_parts.append(f"Gioi tinh: {partial.gender.value}")
        if partial.solar_dob:
            ack_parts.append(f"Ngay sinh: {partial.solar_dob}")
        if partial.birth_hour is not None:
            if partial.birth_hour == -1:
                ack_parts.append("Gio sinh: khong ro")
            else:
                ack_parts.append(
                    f"Gio sinh: gio {_HOUR_NAMES[partial.birth_hour]}"
                )

        ack = ""
        if ack_parts:
            ack = "Da nhan:\n" + "\n".join(f"  - {p}" for p in ack_parts) + "\n\n"

        missing = "\n".join(f"  - {f}" for f in partial.missing_fields)
        return (
            f"{ack}"
            "Toi can them thong tin sau de lap la so Tu Vi:\n"
            f"{missing}\n\n"
            "Vui long cung cap them nhe!"
        )

    def _first_reading(
        self,
        user_id: str,
        query: str,
        birth_data: BirthData,
    ) -> str:
        """Generate the first Tu Vi reading once birth data is complete."""
        chart_json = self._generate_chart(birth_data)

        welcome_query = (
            f"Nguoi dung vua cung cap thong tin ngay sinh: {birth_data.name}, "
            f"{birth_data.gender.value}, sinh ngay {birth_data.solar_dob}. "
            "Hay chao don nguoi dung va cung cap tong quan la so Tu Vi cua ho."
        )

        answer = self._generate_answer(
            query=welcome_query,
            chart_json=chart_json,
            short_term=[],
            long_term=[],
        )

        if not answer:
            answer = (
                f"Da luu thong tin cua ban, {birth_data.name}! "
                "Bay gio ban co the hoi bat cu dieu gi ve la so Tu Vi."
            )

        self._persist_messages(user_id, query, answer)
        return answer

    def _generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        """Generate the Tu Vi chart. Returns empty dict on failure."""
        try:
            return self._tuvi.generate_chart(birth_data)
        except Exception:
            logger.exception("Tu Vi chart generation failed in collector")
            return {}

    def _extract(self, text: str) -> _PartialBirthData | None:
        """
        Use the LLM to extract birth data fields from free-form text.

        Returns a ``_PartialBirthData`` with whatever fields were found,
        or ``None`` if the text contains no birth information at all.
        """
        if not text.strip():
            return None

        try:
            raw = self._llm.extract_structured(
                system_prompt=BIRTH_EXTRACTION_SYSTEM,
                user_message=text,
                max_tokens=256,
            )
        except Exception:
            logger.exception("LLM birth data extraction failed")
            return None

        if raw is None:
            return None

        partial = _PartialBirthData()

        name = raw.get("name")
        if isinstance(name, str) and name.strip():
            partial.name = name.strip()

        gender_raw = str(raw.get("gender", "")).strip().lower()
        if gender_raw in ("nam",):
            partial.gender = Gender.MALE
        elif gender_raw in ("nu", "nu~", "nuu"):
            partial.gender = Gender.FEMALE

        dob = str(raw.get("solar_dob", "")).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dob):
            try:
                datetime.strptime(dob, "%Y-%m-%d")
                partial.solar_dob = dob
            except ValueError:
                pass

        hour_val = raw.get("birth_hour")
        if isinstance(hour_val, int) and -1 <= hour_val <= 11:
            partial.birth_hour = hour_val

        return partial
