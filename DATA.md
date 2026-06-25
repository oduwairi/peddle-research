# Data Availability Statement

This research is grounded in a 55,000-ad corpus collected from **AdFlex**
(https://adflex.io), a commercial, paid multi-platform ad-intelligence API
(Facebook, TikTok, X, Pinterest, Reddit). To respect AdFlex's terms of service and to
protect the collected corpus, the **full raw corpus is not redistributed** in this
repository.

What this means for reproducibility: the raw scrape cannot be re-created by a third
party regardless, because it requires authenticated, paid access to the AdFlex API.
Following standard practice for proprietary-data research, this release provides the
**derived training data** and a **de-identified representative sample**, which are
sufficient to inspect the schema, verify the scoring/tiering logic, and re-run the
fine-tuning and evaluation steps.

## Released (see the companion data package)

| Artifact | Contents | Notes |
|----------|----------|-------|
| `sft_dataset/{train,validation,test}.jsonl` | 4,081 / 226 / 228 chat-format SFT pairs | The instruction-backtranslation training set the student model is actually fine-tuned on. Source ad IDs hashed. |
| `sample/scored_sample_deidentified.jsonl` | 300 scored ads, balanced across tiers (high/medium/low) and platforms | De-identified: advertiser names/IDs, landing-page and creative URLs, targeting (demographics/interests/placements/country), and raw API blobs removed. Retains ad copy, engagement counts, composite score, and tier. |

The companion data package is published separately (release asset / OSF / Zenodo) — see
the repository release page.

## Withheld

| Artifact | Reason |
|----------|--------|
| `raw/adflex_ads.jsonl` — full 55,160-ad raw corpus | AdFlex paid-API terms; harvestable third-party content |
| Full scored corpus (55,160 ads with advertiser/landing-page metadata) | Same as above |

## Available on request

The full scored corpus (without raw API blobs) can be made available to thesis
examiners and bona-fide researchers on reasonable request, subject to AdFlex's terms,
by contacting the author (oduwairi@gmail.com).

## Field-stripping applied to the de-identified sample

Removed: `raw_data`, `advertiser_id`, `advertiser_name`, `advertiser_ad_count`,
`landing_page_url`, `creative_url`, `demographics`, `interests`, `devices`,
`placements`, `country`, `content_safety_*`. Original AdFlex `ad_id` replaced with a
salted hash (`smp_…`). Sampling is stratified (100 ads/tier) with a fixed seed (42).
