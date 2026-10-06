import type { ResumeData } from '@/components/dashboard/resume-component';

/**
 * CLIENT VARIANT of a résumé: the same content WITHOUT the candidate's contact info.
 *
 * Qualifinds shows candidates to its clients with name and experience only: no email, phone,
 * street address, links or social handles. `/api/v1/resumes/{id}/pdf?variant=client` renders the
 * print page through this function, so every template (they all read the same `ResumeData`)
 * is covered at one point.
 *
 * Pure and synchronous on purpose: it runs on the server, BEFORE any template renders, so the
 * removed data never reaches the HTML that Chromium prints.
 *
 * What it does:
 *  - personalInfo: drops email, phone, website, linkedin, github; reduces `location` to
 *    city / state / country (no street, number or postal code).
 *  - projects: drops `github` and `website`.
 *  - every string anywhere in the résumé (summary, bullets, custom sections, skills…): removes
 *    emails, URLs, @handles and phone numbers copied verbatim from the original.
 */

const PLACEHOLDER = '';

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
const URL_RE =
  /\b(?:https?:\/\/|www\.)[^\s<>"')\]]+|\b(?:[A-Za-z0-9-]+\.)+(?:com|mx|net|org|io|co|es|dev|app|me)\/[^\s<>"')\]]*/gi;
const HANDLE_RE = /(^|[\s(>])@[A-Za-z0-9._-]{2,}/g;
/** 8+ digits with the separators people really type; the same threshold Qualifinds measured. */
const PHONE_RE = /\+?\d(?:[\s().-]*\d){7,}/g;

const GROUP_SEPARATORS = /[\s\-–—/]+/;
const YEAR_RE = /^(?:19|20)\d{2}$/;
const MONTH_DAY_RE = /^(?:0?[1-9]|[12]\d|3[01])$/;

/** "2019-2021", "2019 - 2021" and "01-2019 - 03-2021" are date ranges, not phone numbers. */
function isDateRange(raw: string): boolean {
  const parts = raw.trim().split(GROUP_SEPARATORS).filter(Boolean);
  if (parts.length < 2) return false;
  return (
    parts.every((p) => YEAR_RE.test(p) || MONTH_DAY_RE.test(p)) && parts.some((p) => YEAR_RE.test(p))
  );
}

/** Removes emails, URLs, handles and phone numbers from free text (they are deleted, not masked). */
export function scrubContactText(text: string): string {
  if (!text) return text;
  const removed = text
    .replace(EMAIL_RE, PLACEHOLDER)
    .replace(URL_RE, PLACEHOLDER)
    .replace(HANDLE_RE, (_m, lead: string) => lead)
    .replace(PHONE_RE, (m) => (isDateRange(m) ? m : PLACEHOLDER));
  // Text with no contact in it comes back untouched; only tidy what a removal left behind
  // ("Contact: " / "Mail:  · Tel:").
  if (removed === text) return text;
  return removed
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\s*[:|·,;–—-]\s*(?=[:|·,;–—-]|$)/g, '')
    .trim();
}

const STREET_WORD_RE =
  /\b(?:calle|av\.?|avenida|blvd\.?|boulevard|col\.?|colonia|fracc\.?|fraccionamiento|c\.p\.?|cp|street|st\.?|road|rd\.?|apt\.?|suite|ste\.?|piso|int\.?|ext\.?|no\.?|num\.?|#)\b/i;

/**
 * City / state / country only. A free-text location may hold a full address
 * ("Calle 5 #123, Col. Centro, Monterrey, NL, México"): segments with digits or street words are
 * dropped, and only the last three of what remains are kept.
 */
export function cityCountryOnly(location: string | undefined): string | undefined {
  if (!location) return location;
  const kept = location
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s !== '' && !/\d/.test(s) && !STREET_WORD_RE.test(s));
  const out = kept.slice(-3).join(', ');
  return out === '' ? undefined : out;
}

function scrubDeep<T>(value: T): T {
  if (typeof value === 'string') return scrubContactText(value) as unknown as T;
  if (Array.isArray(value)) return value.map((v) => scrubDeep(v)) as unknown as T;
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) out[k] = scrubDeep(v);
    return out as T;
  }
  return value;
}

export function toClientResume(data: ResumeData): ResumeData {
  const scrubbed = scrubDeep(data);

  const info = scrubbed.personalInfo;
  const personalInfo = info
    ? {
        name: info.name,
        title: info.title,
        // The Qualifinds client sees the location only as city / state / country.
        location: cityCountryOnly(data.personalInfo?.location),
        // email, phone, website, linkedin and github are intentionally NOT copied.
      }
    : undefined;

  const personalProjects = scrubbed.personalProjects?.map((p) => ({
    ...p,
    github: undefined,
    website: undefined,
  }));

  return { ...scrubbed, personalInfo, personalProjects };
}
