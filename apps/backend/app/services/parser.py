"""Document parsing service using markitdown and LLM.

The PDF/DOC/DOCX validation and conversion lives in ``app.document_conversion`` and runs in a child
process (see ``parse_document``); this module owns the slots, the deadline and the LLM parsing.
"""

import asyncio
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio

from app.config import settings
from app.document_conversion import (  # noqa: F401  (re-exported for routers and tests)
    MAX_EXTRACTED_TEXT_BYTES,
    MAX_UNPACKED_DOCUMENT_BYTES,
    DocumentResourceLimitError,
    DocumentValidationError,
)
from app.llm import complete_json, get_llm_config, get_model_name, get_safe_max_tokens
from app.prompts import PARSE_RESUME_PROMPT
from app.prompts.templates import RESUME_SCHEMA_EXAMPLE
from app.schemas import ResumeData

logger = logging.getLogger(__name__)

# Each conversion (pdfminer/markitdown) runs in its own child process, which is killed when it
# outlives its deadline. These slots bound how many children run at once, and a slot is only
# released after its child is dead. Before 2026-11-20 the conversion ran in a thread, which cannot
# be killed: two stuck PDFs kept both slots and every upload answered 504 until the service was
# restarted (2026-10-22 to 2026-11-19, and again 30 minutes after a restart).
# Env: DOCUMENT_CONVERSION_WORKERS (default 2) and DOCUMENT_CONVERSION_TIMEOUT_SECONDS (default 120);
# invalid values fall back to the default (see app/config.py). Each child holds the memory of one
# pdfminer parse (tens to hundreds of MB), so raise the workers only with RAM to spare.
DOCUMENT_CONVERSION_WORKERS = settings.document_conversion_workers
DOCUMENT_CONVERSION_TIMEOUT_SECONDS = settings.document_conversion_timeout_seconds
_DOCUMENT_CONVERSION_LIMITER = anyio.CapacityLimiter(DOCUMENT_CONVERSION_WORKERS)
# The child imports app.document_conversion only: a light module that does not load app.config or
# litellm, so a conversion starts in well under a second. Tests point this at a fake worker.
_WORKER_COMMAND: list[str] = [sys.executable, "-m", "app.document_conversion"]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
# After kill() the child is reaped (wait) so no zombie stays behind; this bounds that wait.
_KILL_WAIT_SECONDS = 10.0
_STDERR_TAIL_BYTES = 2000


class DocumentConversionError(RuntimeError):
    """The conversion process died or produced no usable result (crash, OOM kill)."""


# Matches date ranges like "Jan 2020 - Dec 2023", "May 2021 - Present",
# "January 2020 - Current", and single dates like "Jun 2023".
_MD_DATE_RE = re.compile(
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)"
    r"\.?\s+\d{4})"
    r"(?:\s*[-–—]\s*"
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)"
    r"\.?\s+\d{4}"
    r"|Present|Current|Now|Ongoing))?",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)",
    re.IGNORECASE,
)


def _extract_markdown_dates(markdown: str) -> list[str]:
    """Extract all month-inclusive date ranges from markdown text."""
    return _MD_DATE_RE.findall(markdown)


def restore_dates_from_markdown(
    parsed_data: dict[str, Any],
    markdown: str,
) -> dict[str, Any]:
    """Patch year-only dates in parsed data with month-inclusive dates from markdown.

    The LLM sometimes drops months during parsing (e.g. "Jun 2020 - Aug 2021"
    becomes "2020 - 2021"). This function extracts all month-inclusive dates
    from the raw markdown and replaces year-only entries where a match exists.
    """

    def date_key(value: str) -> str:
        years = re.findall(r"\d{4}", value)
        if not years:
            return ""
        suffix = (
            "current"
            if re.search(r"\b(?:present|current|now|ongoing)\b", value, re.IGNORECASE)
            else ""
        )
        if not suffix and len(years) == 2 and years[0] == years[1]:
            years = years[:1]
        return " - ".join([*years, suffix] if suffix else years)

    occurrences: list[tuple[str, str, str, int]] = []
    lines = markdown.splitlines()
    for line_index, line in enumerate(lines):
        line_context = " ".join(line.casefold().split())
        for match in _MD_DATE_RE.finditer(line):
            full_date = re.sub(r"\s*[-–—]\s*", " - ", match.group(0).strip())
            key = date_key(full_date)
            if key:
                occurrences.append((key, full_date, line_context, line_index))
    if not occurrences:
        return parsed_data

    used: set[int] = set()
    patched = 0
    entry_count = sum(
        sum(isinstance(entry, dict) for entry in parsed_data.get(section, []))
        for section in ("workExperience", "education", "personalProjects")
        if isinstance(parsed_data.get(section), list)
    )
    custom_data = parsed_data.get("customSections")
    if isinstance(custom_data, dict):
        entry_count += sum(
            len(section.get("items", []))
            for section in custom_data.values()
            if isinstance(section, dict)
            and section.get("sectionType") == "itemList"
            and isinstance(section.get("items"), list)
        )

    def restore_entry(entry: Any, identity_fields: tuple[str, ...]) -> None:
        nonlocal patched
        if not isinstance(entry, dict):
            return
        years = entry.get("years", "")
        if not isinstance(years, str) or not years or _MONTH_RE.search(years):
            return
        key = date_key(years)
        candidates = [
            index
            for index, occurrence in enumerate(occurrences)
            if occurrence[0] == key and index not in used
        ]
        if not candidates:
            return
        terms = [
            " ".join(str(entry.get(field, "")).split()).casefold()
            for field in identity_fields
            if str(entry.get(field, "")).strip()
        ]
        scored = []
        for index in candidates:
            line_context = occurrences[index][2]
            line_index = occurrences[index][3]
            matched_terms = 0
            score = 0
            for term in terms:
                term_score = 20 if term in line_context else 0
                for distance in range(1, 6):
                    previous = line_index - distance
                    if previous >= 0 and term in " ".join(
                        lines[previous].casefold().split()
                    ):
                        term_score = max(term_score, 10 - distance)
                for distance in range(1, 4):
                    following = line_index + distance
                    if following < len(lines) and term in " ".join(
                        lines[following].casefold().split()
                    ):
                        term_score = max(term_score, 4 - distance)
                if term_score:
                    matched_terms += 1
                    score += term_score
            scored.append(((matched_terms, score), index))
        best_score = max(score for score, _ in scored)
        best = [index for score, index in scored if score == best_score]
        selected: int | None = None
        if (
            best_score[0] > 0
            and len(best) == 1
            and (entry_count == 1 or best_score[0] == len(terms))
        ):
            selected = best[0]
        elif len(candidates) == 1 and entry_count == 1:
            selected = candidates[0]
        if selected is None:
            logger.info("Date restoration left ambiguous value unchanged: %s", years)
            return
        entry["years"] = occurrences[selected][1]
        used.add(selected)
        patched += 1

    for section_key, identity_fields in (
        ("workExperience", ("company", "title")),
        ("education", ("institution", "degree")),
        ("personalProjects", ("name", "role")),
    ):
        for entry in parsed_data.get(section_key, []):
            restore_entry(entry, identity_fields)

    custom = parsed_data.get("customSections", {})
    if isinstance(custom, dict):
        for section in custom.values():
            if (
                not isinstance(section, dict)
                or section.get("sectionType") != "itemList"
            ):
                continue
            for item in section.get("items", []):
                restore_entry(item, ("title", "subtitle"))

    if patched:
        logger.info("Restored months in %d date fields from raw markdown", patched)

    return parsed_data


_NON_CONTENT_RESUME_KEYS = frozenset(
    {
        "id",
        "sectionType",
        "descriptionStyles",
        "isDefault",
        "isVisible",
        "order",
        "key",
        "displayName",
    }
)
# Depth guard against self-referential or pathological LLM output.  Recursion
# starts at depth 0 on a *top-level section value*, so the deepest user-visible
# value the real ``ResumeData`` schema can produce sits at depth 5:
#
#   customSections(0) -> CustomSection(1) -> items(2) -> CustomSectionItem(3)
#       -> description(4) -> bullet string(5)
#
# Every other content section is shallower: workExperience / personalProjects
# bottom out at depth 3 (list -> Experience -> description -> bullet),
# education and additional at depth 2, personalInfo at depth 1, summary at
# depth 0.  Values are still inspected at depth 9 (the cut-off is ``>= 10``),
# so the limit leaves four full levels of headroom over the schema maximum.
# Nothing that validates as ``ResumeData`` can be misjudged empty here;
# anything deeper is malformed LLM output rather than a resume.  Raise this
# only if the schema itself grows deeper -- see the boundary tests in
# tests/unit/test_parser.py::TestMeaningfulResumeContent.
_MAX_RESUME_CONTENT_RECURSION = 10


def _has_meaningful_resume_value(
    value: Any,
    *,
    depth: int = 0,
    filter_structural_keys: bool = True,
) -> bool:
    """Return whether a value contains non-structural, user-visible text.

    Custom-section identifiers are dictionary keys rather than schema fields,
    so their values are checked without filtering the identifier itself.  Once
    inside a section, normal structural-key filtering resumes.
    """
    if depth >= _MAX_RESUME_CONTENT_RECURSION:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(
            _has_meaningful_resume_value(item, depth=depth + 1) for item in value
        )
    if isinstance(value, dict):
        return any(
            (not filter_structural_keys or key not in _NON_CONTENT_RESUME_KEYS)
            and _has_meaningful_resume_value(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def has_meaningful_resume_content(resume_data: Any) -> bool:
    """Return whether parsed resume data contains any user-facing content.

    ``ResumeData`` intentionally defaults most fields to empty strings/lists.
    That is useful for the builder, but it also means an LLM response such as
    ``{}`` validates successfully.  Treating that response as a parsed resume
    produces a blank PDF and makes every downstream tailoring request operate
    on empty data.
    """

    if not isinstance(resume_data, dict):
        return False

    content_sections = (
        "personalInfo",
        "summary",
        "workExperience",
        "education",
        "personalProjects",
        "additional",
        "customSections",
    )
    return any(
        _has_meaningful_resume_value(
            resume_data.get(section),
            filter_structural_keys=section != "customSections",
        )
        for section in content_sections
    )


def _validate_parsed_resume(result: dict[str, Any]) -> dict[str, Any]:
    """Validate that parsed output is a schema-valid, non-empty resume."""
    parsed_data = ResumeData.model_validate(result).model_dump()
    if not has_meaningful_resume_content(parsed_data):
        raise ValueError("LLM returned an empty structured resume.")
    return parsed_data


def _safe_suffix(filename: str) -> str:
    """The extension picks the container check and the markitdown converter in the child."""
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ""


def _stderr_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - _STDERR_TAIL_BYTES))
            return stream.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    """Kill the child and wait for it, so no zombie or running process survives."""
    try:
        process.kill()
    except ProcessLookupError:
        pass  # already gone
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), _KILL_WAIT_SECONDS)
    except TimeoutError:
        logger.error(
            "Document conversion process %s did not exit %.0fs after kill",
            process.pid,
            _KILL_WAIT_SECONDS,
        )


def _read_conversion_result(
    returncode: int | None, result_path: Path, stderr_path: Path
) -> str:
    """Turn the finished child's result file into text, or the exception the file reports."""
    if returncode != 0:
        raise DocumentConversionError(
            f"Document conversion process exited with code {returncode}: "
            f"{_stderr_tail(stderr_path)}"
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        status = payload["status"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DocumentConversionError(
            "Document conversion process left no readable result"
        ) from exc
    if status == "ok" and isinstance(payload.get("text"), str):
        return payload["text"]
    message = str(payload.get("message", ""))
    kind = payload.get("kind")
    if kind == "resource_limit":
        raise DocumentResourceLimitError(message)
    if kind == "validation":
        raise DocumentValidationError(message)
    raise DocumentConversionError(f"{payload.get('type', 'Error')}: {message}")


async def _run_conversion_process(content: bytes, filename: str, timeout: float) -> str:
    """Convert in a child process that is killed (and reaped) if it outlives ``timeout``.

    The working directory holds the input, the child's JSON result and its stderr; it is removed
    whatever happens (success, error, deadline, cancellation of the request).
    """
    workdir = Path(tempfile.mkdtemp(prefix="rm-convert-"))
    input_path = workdir / f"input{_safe_suffix(filename)}"
    result_path = workdir / "result.json"
    stderr_path = workdir / "stderr.log"
    process: asyncio.subprocess.Process | None = None
    try:
        input_path.write_bytes(content)
        with stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *_WORKER_COMMAND,
                str(input_path),
                str(result_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr,
                cwd=str(_BACKEND_ROOT),
            )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Document conversion killed after exceeding its %.0fs deadline: "
                "%s (%d bytes, pid %s)",
                DOCUMENT_CONVERSION_TIMEOUT_SECONDS,
                filename,
                len(content),
                process.pid,
            )
            raise
        return _read_conversion_result(process.returncode, result_path, stderr_path)
    finally:
        try:
            if process is not None and process.returncode is None:
                await _kill_and_reap(process)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


async def parse_document(content: bytes, filename: str) -> str:
    """Convert a bounded PDF/DOC/DOCX to Markdown without blocking the request event loop.

    The conversion runs in a child process (``python -m app.document_conversion``) so that one
    that does not finish can be killed: the deadline (queue wait plus conversion) raises
    ``TimeoutError`` after killing it, and its slot is free again for the next upload. Cancelling
    the request kills it too.

    Args:
        content: Raw file bytes
        filename: Original filename for extension detection

    Returns:
        Markdown text content

    Raises:
        TimeoutError: the deadline passed (in the queue or converting); the child was killed.
        DocumentValidationError / DocumentResourceLimitError: the file is invalid / too big.
        DocumentConversionError: the child died or the converter failed unexpectedly.
    """
    loop = asyncio.get_running_loop()
    timeout = DOCUMENT_CONVERSION_TIMEOUT_SECONDS
    deadline = loop.time() + timeout
    limiter = _DOCUMENT_CONVERSION_LIMITER
    borrower = object()
    # A queued request that is cancelled or times out never starts a process.
    await asyncio.wait_for(limiter.acquire_on_behalf_of(borrower), timeout=timeout)
    try:
        return await _run_conversion_process(
            content, filename, max(0.0, deadline - loop.time())
        )
    finally:
        limiter.release_on_behalf_of(borrower)


async def parse_resume_to_json(markdown_text: str) -> dict[str, Any]:
    """Parse resume markdown to structured JSON using LLM.

    After LLM parsing, patches any year-only dates with month-inclusive
    dates extracted from the raw markdown. This ensures months are never
    lost regardless of LLM behavior.

    Args:
        markdown_text: Resume content in markdown format

    Returns:
        Structured resume data matching ResumeData schema
    """
    if not markdown_text or not markdown_text.strip():
        raise ValueError("Resume content is empty after text extraction.")

    prompt = PARSE_RESUME_PROMPT.format(
        schema=RESUME_SCHEMA_EXAMPLE,
        resume_text=markdown_text,
    )

    config = get_llm_config()
    model_name = get_model_name(config)
    result = await complete_json(
        prompt=prompt,
        system_prompt="You are a JSON extraction engine. Output only valid JSON, no explanations.",
        max_tokens=get_safe_max_tokens(model_name, config=config),
        retries=3,
        response_validator=_validate_parsed_resume,
    )

    # Patch dates: restore months the LLM may have dropped
    result = restore_dates_from_markdown(result, markdown_text)

    # Validate against schema
    return _validate_parsed_resume(result)
