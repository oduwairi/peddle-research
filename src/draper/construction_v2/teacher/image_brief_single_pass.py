"""Single-pass teacher for the image-brief skill.

Mirror of :mod:`draper.construction_v2.teacher.single_pass`, but the
deliverable is an ``<image_brief>...</image_brief>`` region whose body is
FREEFORM ART-DIRECTION PROSE — not JSON. The teacher receives the source
ad's metadata AND a LITERAL, factual VLM caption of the real winning
creative, and re-registers the caption observational -> directive into a
forward-looking art-direction brief (with bindings intact and a single
optional trailing ``Avoid:`` exclusion line).

The brief the teacher authors is image-skill-specific:
``task + objective + product + creative{brand_guidelines, on_creative_text,
key_facts}``. Three fields are NOT authored by the teacher and are injected at
ingest: ``ad_copy`` (verbatim, platform-labeled), ``creative.orientation``
(platform-derived canvas), and ``platform``. The ``creative`` block is the
bridge from the real creative into the brief — style (``brand_guidelines``) plus
the factual content the copy doesn't supply (``on_creative_text`` +
``key_facts``). The deliverable's composition is the writer's job and never
appears in the brief.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, get_args

from draper.construction.batch.types import BatchRequest
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.ingest.response_parser import (
    ParsedResponse,
    ParseRejection,
    parse_response,
)
from draper.construction_v2.platform_labels import platform_group_for, render_labeled_ad
from draper.construction_v2.schemas.image_brief import (
    AdObjective,
    aspect_ratio_for_platform,
)
from draper.construction_v2.teacher.single_pass import (
    _BRIEF_RE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    _strip_fence,
)

# Key in ``SourceAd.raw`` carrying the VLM caption of the real creative.
# Populated by the captioning pipeline (Phase 2) and joined into the
# source-ad load step when ``config.skill == "image_brief"``.
CAPTION_RAW_KEY: str = "image_caption"

# Valid ``objective`` enum values, sourced from the schema so the teacher
# prompt and parser can never drift from :data:`AdObjective`.
_VALID_OBJECTIVES: frozenset[str] = frozenset(get_args(AdObjective))


IMAGE_BRIEF_TEACHER_SYSTEM: str = (
    "You are preparing training data for Draper, a marketing specialist. "
    "This slice trains Draper to produce art-direction image briefs that "
    "downstream feed an image-generation model.\n"
    "\n"
    "This skill runs AFTER the ad copy is already written. You are given one "
    "real, high-performing ad — its FINISHED COPY (verbatim, platform-labeled) "
    "AND a LITERAL, factual caption describing exactly what was visible in the "
    "real creative that ran with that copy. Produce a single response "
    "containing a `<brief>` JSON region, then the canonical Draper assistant "
    "turn: a `<think>` block followed by an `<image_brief>` region carrying "
    "the art-direction prose brief. The image must PAIR with the copy you are "
    "given — that is the whole job of this skill.\n"
    "\n"
    "## What the literal caption is (read this first)\n"
    "\n"
    "The caption is a plain, observational description of the real creative: "
    "people, products, props, setting, composition, lighting, color, any "
    "visible text or logos, and the photographic/design style. It states ONLY "
    "what is literally visible. It does NOT interpret strategy or intent. You "
    "will use it as the exact, accurate ground truth for what the winning "
    "visual looked like — nothing in it is up for reinterpretation.\n"
    "\n"
    "## Grounding contract\n"
    "\n"
    "The STRATEGIC brief fields — `task`, `objective`, `product` — MUST be "
    "derivable from the ad's copy text plus the supplied metadata "
    "(advertiser_name, landing_page_url, platform). The literal caption is NOT "
    "a source for those fields. World knowledge about the brand is FORBIDDEN. "
    "If the ad does not state or clearly imply a fact, leave that field null or "
    "empty.\n"
    "\n"
    "The `creative` block is the bridge from the real creative into the brief. "
    "`brand_guidelines` distills the brand's reusable VISUAL STYLE (medium, "
    "aesthetic register, type feel) at the brand level. `on_creative_text` and "
    "`key_facts` carry the factual CONTENT the campaign is about that the copy "
    "and product do not already supply: `on_creative_text` is verbatim on-image "
    "text; `key_facts` is everything else the creative must get right about the "
    "product, offer, or subject.\n"
    "\n"
    "`key_facts` exists for exactly one job: carry the facts the writer would "
    "NEED to build this exact creative yet could not work out from the brief "
    "alone. Pull candidates from the literal caption and keep one only when "
    "BOTH hold: (1) substituting a different, equally tasteful choice in its "
    "place would change WHAT the ad communicates or claims — not merely how it "
    "looks — and (2) it does not already follow from the copy + product in the "
    "brief. The decisive line is between what the creative merely CONTAINS and "
    "what it DEPENDS on: most of what is visible could be chosen differently "
    "without changing the ad's meaning, and that is the writer's to invent; "
    "only what the ad is genuinely making a claim about — such that a different "
    "choice would be a different claim — is a fact. (As a general illustration: "
    "in a car ad, the coastal road the sedan is posed on is the writer's to "
    "reinvent, but the trim name and the mileage figure the campaign is about "
    "are facts the copy may omit and the writer cannot guess.) Fail either test "
    "and it stays out, however prominent it is in the picture.\n"
    "\n"
    "Both directions cost you at inference, where the agent fills `key_facts` "
    "from the copy and product alone, having never seen the creative. A fact "
    "that passes both tests but is left out cannot be recovered — the writer is "
    "forced to hallucinate it. A detail that is redundant or freely chosen but "
    "bridged anyway teaches the writer to transcribe a picture it was never "
    "shown. State each kept fact as the fact itself, never as the way it is "
    "arranged — the arrangement is the writer's. Keep entries factual and "
    "atomic; many ads need none.\n"
    "\n"
    "## Regions\n"
    "\n"
    "1. `<brief>...</brief>` — the canonical Draper brief shape for the image "
    "skill. Keys:\n"
    "     - `task` (required string): a one-sentence natural-language query a "
    "founder would type to ask Draper for the creative/image brief for THIS "
    "campaign (whose copy is in front of you). Examples: "
    '"Give me the image brief for our payroll-compliance ad on Meta.", '
    '"What should the creative look like for this Pinterest pin?" Infer it '
    "from the ad — do not hardcode.\n"
    "     - `objective` (required string): what the ad is trying to DO — its "
    "marketing purpose, the single best fit from: `awareness` (build brand "
    "familiarity / mood, no hard offer), `promo_offer` (a discount/deal/"
    "limited offer is the point), `launch` (announces a new product or "
    "feature), `social_proof` (leads with a testimonial / UGC / results / "
    "reviews), `conversion` (direct-response — drive a specific action on the "
    "product's value). Read it off the copy.\n"
    "     - `product` (required object). Only `tone_signals` is required "
    "(non-empty string array — voice cues read off the ad's natural voice). "
    "All other product fields are OPTIONAL and should be null/empty when the "
    "ad doesn't support them: `name`, `description`, `category`, "
    "`key_features`, `unique_selling_points`, `price_info`, "
    "`category_context`, `proof_points`, `offer`.\n"
    "     - `creative` (required object) — the visual-direction block. Author "
    "three keys (a fourth, `orientation`, is the canvas and is injected "
    "automatically — do NOT emit it):\n"
    "         - `brand_guidelines` (required string) — the brand's recurring "
    "VISUAL STYLE: its medium / art-style (photographic, illustrated, 3D, flat, "
    "etc.), aesthetic register (e.g. clean and high-key, bold and graphic, warm "
    "and editorial), and type feel. Distill it at the reusable-brand level — how "
    "the brand's work looks across campaigns, derived only from the creative's "
    "visible style and the advertiser identity, never this one image's "
    "composition and never world knowledge of the brand.\n"
    "         - `on_creative_text` (string array) — every text string burned "
    "INTO the creative that is NOT part of the ad copy (overlay headlines, "
    "button labels, item or section labels, words on packaging or a document in "
    "frame). Copy each one VERBATIM; every visible non-copy string belongs here. "
    "The string only — never where it sits. Leave [] when the only text in the "
    "creative is the ad copy itself.\n"
    "         - `key_facts` (string array) — the facts the writer would need to "
    "build this creative but could not work out from the copy + product. Apply "
    "the two-part test (above) to each caption candidate: keep it only if "
    "substituting a different tasteful choice would change what the ad claims "
    "AND it does not already follow from the brief; otherwise leave it out. [] "
    "whenever the copy + product already imply all the content.\n"
    "     - Do NOT emit `ad_copy`, `creative.orientation`, or `platform`. These "
    "are known facts about the ad (the finished copy, the canvas, the "
    "placement) and are attached to the brief automatically; never reproduce, "
    "summarize, or paraphrase the copy inside the `<brief>` JSON, and never "
    "author an orientation or platform.\n"
    "\n"
    "2. `<think>...</think>` — **Persona switch: for this block you ARE "
    "Draper, reading the brief and the campaign's finished copy and deciding "
    "the visual.** First-person decisional voice narrating the visual "
    "decisions a practitioner makes — what the hero subject should be, how the "
    "composition expresses the `brand_guidelines` feel and serves the "
    "`objective`, how the image pairs with and reinforces the specific copy "
    "(the headline's promise, the offer, the hook), and which copy — if any — "
    "belongs ON the creative as text. You MAY reason from the ad copy: it is "
    "part of the brief Draper sees. Anchor every choice to `brand_guidelines`, "
    "the `objective`, a product field, or the copy. Do NOT describe the "
    "supplied caption as an artifact you are looking at — Draper writes from "
    "inside the act of art-directing.\n"
    "\n"
    "3. `<image_brief>...</image_brief>` — a faithful, forward-looking "
    "ART-DIRECTION PROSE brief. This is NOT JSON and NOT a list of labeled "
    "fields. Write it as one or more directive paragraphs an art director "
    "would hand to an illustrator or a photographer.\n"
    "\n"
    "   **Re-register the literal caption, observational -> directive.** Take "
    "every visible fact the caption reports and restate it as an instruction: "
    '"This ad shows a sealed cup of iced boba" becomes "Create a sealed cup '
    'of iced boba". Transform ONLY the register — invent NOTHING the caption '
    "did not state, and drop nothing it did.\n"
    "\n"
    "   Preserve every binding intact — this is the whole point of prose over "
    "a field list:\n"
    "     - **Color in place.** Bind each color to the thing it colors and "
    "where it sits, never as a free-floating palette.\n"
    "     - **On-creative text quoted in situ.** If text is visible in the "
    "creative, quote it verbatim and say where it sits.\n"
    "     - **Named props concrete.** Keep every prop as the specific object the "
    "caption named, never collapsed to a generic category ('storage', "
    "'garnish').\n"
    "     - **Human-subject specifics.** Preserve apparent age, presentation, "
    "skin tone (as a visual fact), wardrobe, expression, and pose exactly as "
    "the caption reports them.\n"
    "     - **Composition, lighting, style.** Carry framing/angle/depth, "
    "lighting quality, and the photographic-vs-illustration-vs-render style "
    "the caption named.\n"
    "\n"
    "   Do NOT editorialize. The image brief is a faithful restatement of the "
    'visual — NOT a "why it works" rationale. All strategic reasoning lives '
    "in `<brief>` and `<think>`; the deliverable carries zero "
    '"this builds trust" / "this draws the eye" prose.\n'
    "\n"
    "   Do NOT instruct the rendering of real trademarked brand logos. If the "
    "caption reports a recognizable brand mark, describe it generically (e.g. "
    '"a plain unbranded label") and, when the goal is to keep it out, list it '
    "under exclusions.\n"
    "\n"
    "   **Exclusions — the one structured element.** If anything must be kept "
    "OUT of the frame, end the region with a single line, as the LAST line, "
    "in exactly this form:\n"
    "     `Avoid: item one; item two; item three`\n"
    "   Items are separated by `; ` (semicolon then space). Use this for real "
    "brand logos, faces/hands when the creative deliberately excluded them, "
    "text overlay, clutter, or anything else the visual deliberately omits. "
    "If nothing must be excluded, omit the `Avoid:` line entirely. There is "
    "NO other structured syntax in the deliverable — no JSON, no bullet "
    "lists, no field labels.\n"
    "\n"
    "### Worked example\n"
    "\n"
    'For a boba milk-tea ad whose finished copy reads *Headline: "50% less '
    'sugar, same boba you love"* and whose literal caption reads: *"A '
    "vertical split "
    "image of an iced milk tea drink. On the left, a sealed clear plastic cup "
    "against a pastel-blue background shows beige milk tea over dark tapioca "
    "pearls with a green straw and a domed lid. On the right, the same cup "
    "against a cream background with milk being poured in and swirling. White "
    "text across the middle reads '50% less sugar'. Bright studio product "
    "photography, shallow depth of field.\"* — the brief's `objective` is "
    "`promo_offer` (the copy leads with a 50%-less-sugar claim) and "
    '`creative.brand_guidelines` might be "Bright, playful, appetizing brand '
    "feel; high-key studio product photography on clean color-blocked fields; "
    'flat modern sans-serif type" (the reusable STYLE — note it says nothing '
    "about the split layout or which side the pour is on). Here "
    '`creative.on_creative_text` is [] (the visible "50% less sugar" text is '
    "part of the headline copy, not a separate overlay) and "
    "`creative.key_facts` is [] (the copy already names the iced boba milk "
    "tea, so the subject is inferable; the split layout, the pour, and the "
    "pastel/cream fields are not needed — the writer invents them. Nothing in "
    "the creative is a non-inferable specific the copy omits, so there is "
    "nothing to bridge). A faithful "
    "`<image_brief>` is:\n"
    "\n"
    "`<image_brief>Create a vertical split-collage product shot of an iced "
    "boba milk tea. On the left half, set the sealed clear cup against a "
    "clean pastel-blue field, the drink filling the frame: pale milk-tea "
    "beige liquid layered over a dense bed of dark-brown tapioca pearls at "
    "the bottom, a wide clear dome lid, and a fat translucent green straw "
    "angled into the cup. On the right half, mirror the same cup mid-pour "
    "against a soft cream field, milk streaming in and swirling the tea so "
    "the two tones marble together. Center a small white sans-serif label "
    'across the seam reading "50% less sugar". Shoot it as bright, high-key '
    "commercial product photography, crisp studio lighting, shallow depth of "
    "field with the cup tack-sharp and the colored fields softly clean behind "
    "it. Mood is fresh, playful, and appetizing — a cold-drink-on-a-hot-day "
    "register. Avoid: real cafe brand logos; human hands; cluttered "
    "background; visible ice melt or condensation drips</image_brief>`\n"
    "\n"
    "## Hard rules\n"
    "- Inside `<think>`, Draper writes from inside the act — he doesn't "
    "reference the supplied caption as something he 'sees'. He MAY reason "
    "from the finished ad copy (it is part of the brief he was handed).\n"
    "- The `creative` block: `brand_guidelines` names the brand's reusable "
    "VISUAL STYLE at the brand level (never quoting or paraphrasing the ad copy, "
    "never world knowledge); `on_creative_text` is verbatim non-copy on-image "
    "text (the string, never its placement); `key_facts` is what the creative "
    "depends on that the copy omits and the writer could not infer — never what "
    "the brief already implies or what the writer is free to choose without "
    "changing the ad's claim. Both content lists are [] when nothing must be "
    "bridged.\n"
    "- `objective` must be exactly one of: "
    f"{', '.join(sorted(_VALID_OBJECTIVES))}.\n"
    "- Do NOT emit `ad_copy`, `creative.orientation`, or `platform` inside "
    "`<brief>` — all three are injected automatically.\n"
    "- `tone_signals` must be non-empty.\n"
    "- The `<image_brief>` region must be prose (no JSON, no field labels). "
    "The only permitted structured syntax is the optional trailing `Avoid:` "
    "line.\n"
    "- The deliverable carries every visible fact from the literal caption "
    "with bindings intact, invents no visual facts, and contains no "
    '"why it works" editorializing.\n'
    "- Never instruct rendering of a real trademarked logo; describe "
    "third-party marks generically.\n"
    "- No content before `<brief>`. No markdown fences around the regions "
    "themselves."
)


_IMAGE_BRIEF_RE = re.compile(r"<image_brief>(.*?)</image_brief>", re.DOTALL | re.IGNORECASE)
_AVOID_RE = re.compile(r"^Avoid:\s*(.+)$", re.IGNORECASE)

# A line consisting solely of a markdown code fence, optionally language-tagged
# (```` ``` ````, ```` ```json ````, ...). Some providers — Gemini in particular —
# wrap the ``<brief>`` JSON in such a fence despite the prompt forbidding it.
_FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def _strip_structural_fences(content: str) -> str:
    """Drop standalone markdown code-fence lines that precede ``<think>``.

    Gemini frequently wraps the ``<brief>`` region in a ```` ```json ```` fence.
    ``_BRIEF_RE`` still matches the tags, but the leftover fence markers land
    between ``</brief>`` and ``<think>`` and trip ``parse_response``'s
    "no noise before <think>" guard (``pre_think_noise``). Only the structural
    prefix up to ``<think>`` is normalized; the freeform deliverable after
    ``</think>`` is never touched, so any fences a teacher legitimately wrote
    into the prose are preserved.
    """
    m = re.search(r"<think\b", content, re.IGNORECASE)
    if m is None:
        return content
    head, tail = content[: m.start()], content[m.start() :]
    kept = [ln for ln in head.splitlines() if not _FENCE_LINE_RE.match(ln)]
    head_clean = "\n".join(kept)
    if head_clean and not head_clean.endswith("\n"):
        head_clean += "\n"
    return head_clean + tail


def _parse_exclusions(prose: str) -> list[str]:
    """Parse the exclusion list from a prose deliverable's trailing ``Avoid:`` line.

    Scans the prose's lines (each stripped of leading whitespace) for the LAST
    line matching ``^Avoid:\\s*(.+)$`` (case-insensitive), splits the capture on
    ``;``, strips each item, and drops empties. Returns ``[]`` when no ``Avoid:``
    line is present. Never raises.
    """
    last_capture: str | None = None
    for raw_line in prose.splitlines():
        m = _AVOID_RE.match(raw_line.strip())
        if m:
            last_capture = m.group(1)
    if last_capture is None:
        return []
    return [item.strip() for item in last_capture.split(";") if item.strip()]


def build_image_brief_user_message(ad: SourceAd, *, caption: str) -> str:
    """Render the user turn for an image-brief teacher request.

    Shows the teacher three things about the real ad: its FINISHED COPY
    (platform-labeled via ``render_labeled_ad`` — exactly the form the
    copywriting skill emits, and the form the pipeline injects back into the
    brief's ``ad_copy`` field), the LITERAL, factual VLM ``caption`` of the
    creative that ran with it, and the TARGET CANVAS (the platform-derived
    orientation the pipeline injects). The teacher derives ``objective`` +
    ``product`` from the copy + metadata and authors the ``creative`` block
    (``brand_guidelines`` style + the ``on_creative_text`` / ``key_facts``
    content bridges) from the caption, then re-registers the caption
    observational -> directive into the prose ``<image_brief>`` deliverable. The
    copy is shown so the visual the teacher designs PAIRS with the specific copy;
    the canvas is shown so the ``<think>`` can compose for it.
    """
    if not caption.strip():
        msg = (
            "image-brief teacher requires a non-empty VLM caption for the "
            "source ad's creative; caption was empty"
        )
        raise ValueError(msg)

    labeled_copy = render_labeled_ad(ad).strip()
    if not labeled_copy:
        msg = (
            f"image-brief teacher requires non-empty ad copy for ad "
            f"{ad.ad_id!r}; render_labeled_ad produced nothing"
        )
        raise ValueError(msg)

    parts: list[str] = [
        "# Campaign",
        "",
        f"- ad_id: {ad.ad_id}",
        f"- platform_hint: {ad.platform}",
    ]
    advertiser = ad.raw.get("advertiser_name") if isinstance(ad.raw, dict) else None
    landing = ad.raw.get("landing_page_url") if isinstance(ad.raw, dict) else None
    if isinstance(advertiser, str) and advertiser:
        parts.append(f"- advertiser_name: {advertiser!r}")
    if isinstance(landing, str) and landing:
        parts.append(f"- landing_page_url: {landing!r}")
    parts.extend(
        [
            "",
            "## Finished ad copy (verbatim — the image must pair with this)",
            "",
            labeled_copy,
            "",
            "## Creative description (the LITERAL VLM caption of the real "
            "creative — the supervision target for the image brief)",
            "",
            caption.strip(),
            "",
            "## Target canvas (attached to the brief automatically)",
            "",
            f"- aspect_ratio: {aspect_ratio_for_platform(platform_group_for(ad.platform).value)}",
            "",
            "Produce the response now. Begin with `<brief>`, then `<think>`, "
            "then the `<image_brief>` region. Derive `objective` + `product` "
            "from the copy + metadata; author the `creative` block from the "
            "caption — `brand_guidelines` (reusable style), `on_creative_text` "
            "(verbatim non-copy on-image text), and `key_facts` (load-bearing "
            "content facts the copy doesn't supply, stated as facts not "
            "pictures); do NOT put the copy, an "
            "orientation, or a platform in the brief — all are attached "
            "automatically. Re-register every visible fact in the caption "
            "observational -> directive into the prose `<image_brief>` "
            "art-direction brief, keeping all bindings intact and inventing "
            "nothing it did not state.",
        ]
    )
    return "\n".join(parts)


def build_image_brief_request(
    ad: SourceAd,
    *,
    model: str,
    caption: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BatchRequest:
    """Build a ``BatchRequest`` for the image-brief single-pass teacher.

    ``caption`` defaults to ``ad.raw[CAPTION_RAW_KEY]`` so callers that
    have already enriched their source ads (the production submit path)
    don't need to pass it explicitly. Smoke scripts that pass captions
    out-of-band can supply the keyword argument directly.
    """
    if caption is None:
        cap = ad.raw.get(CAPTION_RAW_KEY) if isinstance(ad.raw, dict) else None
        if not isinstance(cap, str) or not cap.strip():
            msg = (
                f"image-brief teacher needs a caption for ad {ad.ad_id!r}; "
                f"none found on ad.raw[{CAPTION_RAW_KEY!r}] and none "
                "passed explicitly"
            )
            raise ValueError(msg)
        caption = cap
    return BatchRequest(
        custom_id=f"teacher-{ad.ad_id}",
        system=IMAGE_BRIEF_TEACHER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": build_image_brief_user_message(ad, caption=caption),
            }
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@dataclass(frozen=True)
class ImageBriefParseResult:
    """Outcome of parsing one image-brief teacher response.

    ``brief`` carries the parsed brief JSON. ``deliverable`` is the full
    ``<image_brief>...</image_brief>`` prose body (verbatim, including the
    trailing ``Avoid:`` line) — the student learns to emit the whole region.
    ``exclusions`` carries the parsed items off that ``Avoid:`` line (``[]``
    when none). ``think`` is the reasoning region.
    """

    brief: dict[str, Any] | None
    think: str | None
    exclusions: list[str]
    deliverable: str | None
    errors: list[str] = field(default_factory=list)


def parse_image_brief_response(content: str) -> ImageBriefParseResult:
    """Parse an image-brief teacher response.

    Splits the response into ``<brief>`` (JSON), ``<think>``, and the prose
    ``<image_brief>`` deliverable. The ``<brief>`` JSON parse validates the
    teacher-authored keys (``task``, ``objective``, ``brand_guidelines``) and
    strips any ``ad_copy`` / ``aspect_ratio`` the teacher should not author
    (both are injected at ingest). The ``<image_brief>`` region is treated as
    freeform prose — its body is kept verbatim as ``deliverable`` and the
    trailing ``Avoid:`` line is parsed into ``exclusions``. There is no JSON
    parse for the image-brief region; this function never raises.
    """
    errors: list[str] = []

    if not content or not content.strip():
        return ImageBriefParseResult(
            brief=None,
            think=None,
            exclusions=[],
            deliverable=None,
            errors=["empty content"],
        )

    # Tolerate providers (Gemini) that wrap the structural regions in markdown
    # code fences despite the prompt forbidding them — strip the leftover fence
    # lines before they trip the parser's pre-think-noise guard.
    content = _strip_structural_fences(content)

    brief_payload: dict[str, Any] | None = None
    bm = _BRIEF_RE.search(content)
    if bm:
        try:
            payload = json.loads(_strip_fence(bm.group(1)))
        except json.JSONDecodeError as e:
            errors.append(f"brief JSON parse: {e}")
        else:
            if isinstance(payload, dict):
                if not isinstance(payload.get("task"), str) or not payload["task"].strip():
                    errors.append("brief missing `task` string")
                obj = payload.get("objective")
                if not isinstance(obj, str) or obj not in _VALID_OBJECTIVES:
                    errors.append("brief missing/invalid `objective`")
                creative = payload.get("creative")
                if not isinstance(creative, dict):
                    errors.append("brief missing `creative` object")
                else:
                    bg = creative.get("brand_guidelines")
                    if not isinstance(bg, str) or not bg.strip():
                        errors.append("brief missing `creative.brand_guidelines` string")
                    # `orientation` is the canvas — injected deterministically at
                    # ingest. Drop any stray copy so it can't shadow the injection.
                    for stray in ("orientation", "aspect_ratio"):
                        if stray in creative:
                            creative.pop(stray, None)
                            errors.append(
                                f"brief emitted a `creative.{stray}` field (dropped at ingest)"
                            )
                # The teacher must not author the copy or the canvas — both are
                # injected verbatim/deterministically at ingest. Drop any stray
                # top-level field so it can't shadow the injection.
                for stray in ("copy", "ad_copy", "aspect_ratio", "platform"):
                    if stray in payload:
                        payload.pop(stray, None)
                        errors.append(f"brief emitted a `{stray}` field (dropped at ingest)")
                brief_payload = payload
            else:
                errors.append("brief JSON is not an object")
        tail = content[bm.end() :]
    else:
        errors.append("missing <brief> region")
        tail = content

    parsed = parse_response(tail)
    if isinstance(parsed, ParseRejection):
        errors.append(f"response_parser: {parsed.value}")
        return ImageBriefParseResult(
            brief=brief_payload,
            think=None,
            exclusions=[],
            deliverable=None,
            errors=errors,
        )
    if not isinstance(parsed, ParsedResponse):
        errors.append("unexpected response_parser output")
        return ImageBriefParseResult(
            brief=brief_payload,
            think=None,
            exclusions=[],
            deliverable=None,
            errors=errors,
        )

    deliverable = parsed.deliverable
    ib_match = _IMAGE_BRIEF_RE.search(deliverable)
    exclusions: list[str] = []
    if ib_match:
        exclusions = _parse_exclusions(ib_match.group(1))
    else:
        errors.append("missing <image_brief> region")

    return ImageBriefParseResult(
        brief=brief_payload,
        think=parsed.think,
        exclusions=exclusions,
        deliverable=deliverable,
        errors=errors,
    )


__all__ = [
    "CAPTION_RAW_KEY",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "IMAGE_BRIEF_TEACHER_SYSTEM",
    "ImageBriefParseResult",
    "build_image_brief_request",
    "build_image_brief_user_message",
    "parse_image_brief_response",
]
