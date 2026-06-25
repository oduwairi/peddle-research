# Draper.ai — Training Data Architecture

A structured overview of data sources for building the fine-tuning dataset and marketing reasoning corpus for Draper.ai.

---

## Source 1: Large-Scale Ad Intelligence Scraping (PRIMARY — This Is the Moat)

The core training data. Tens of thousands of real ads scraped at scale, each scored with a composite performance proxy.

### Data Source

**AdFlex API** (primary, sole ad intelligence source), supplemented by free official ad libraries (Meta Ad Library, Google Ads Transparency Center, TikTok Ads Library) for structural data.

- **AdFlex** — adflex.io — €99/month Pro plan, 500K credits/month. 100 credits per search API call, 18 ads per page. Covers 5 platforms with engagement data: Facebook, TikTok, X, Pinterest, Reddit. Also provides Meta (Facebook/Instagram) ad library data at zero credit cost (but without engagement metrics).

### What You Get Per Ad

- Ad copy (title, description/subtitle)
- Creative format and URL (image, video, carousel)
- Platform (Facebook, TikTok, X, Pinterest, Reddit)
- Country/region targeting and placement data
- Active days (longevity signal)
- Engagement metrics (platform-specific):
  - **Facebook:** reactions, comments, shares, views
  - **TikTok:** likes, plays, favorites, comments, shares
  - **X:** reposts, replies, likes, bookmarks
  - **Pinterest:** saves, repins, comments
  - **Reddit:** upvotes, comments
- Advertiser name, verified status, landing page URL
- Raw API response preserved for debugging

### Composite Performance Scoring

Not just longevity — a multi-signal proxy that produces a **continuous performance score** per ad:

| Signal | What It Measures | Type |
|--------|-----------------|------|
| **Longevity** (days running) | Economic signal — advertisers kill unprofitable ads | Positive |
| **Engagement volume** (likes + comments + shares) | Audience resonance | Positive |
| **Engagement velocity** (engagement / days) | Normalized performance — controls for longevity bias | Positive |
| **Re-delivery** (stopped and relaunched) | Deliberate advertiser signal — chose to bring it back | Strong positive |
| **Advertiser persistence** (total active ads from same page) | Budget/success indicator | Positive |
| **Early death** (short-lived ads) | Negative example — ad failed fast | Negative |

**Tiering:** Ads are grouped into performance tiers (e.g. high, average, low) based on score distribution. The exact thresholds are determined empirically during dataset construction. The contrast between tiers is what teaches the model.

### Scale Target

Tens of thousands of ads across diverse verticals: e-commerce, SaaS, D2C, local services, fitness, education, finance, B2B, healthcare, food & beverage, real estate.

### Official Ad Library Supplements (Free, Structural Data Only)

These provide additional structural data but lack engagement metrics. Useful for expanding ad copy coverage and, for political/EU ads, real spend/impression figures:

- **Meta Ad Library** — Ad creative, copy, CTA, start/end dates, targeting, status. EU/UK ads include spend ranges, impression estimates, and demographic distribution. No engagement metrics (likes/comments/shares) for any ads. Free via Apify scraper.
- **Google Ads Transparency Center** — Ad creatives, advertiser info, first/last shown dates, `total_days_shown` field (direct longevity signal), regional data. No engagement metrics. Via SerpApi.
- **TikTok Ads Library** — Video ads, advertiser name, target regions, impression counts (sometimes exposed), CTA, captions. Via Apify scraper.

---

## Source 2: Marketing Knowledge Corpus (SECONDARY — Teaches Reasoning)

This is what teaches the model *why* things work and *how* to think about strategy. The ad scraping teaches what works empirically; this teaches the reasoning behind it.

### Sources

**Agency case studies with real performance data:**
- MarketingProfs Case Studies — marketingprofs.com — Hundreds organized by metrics & ROI
- Single Grain — singlegrain.com — Detailed campaign breakdowns with performance data
- HubSpot Case Studies — hubspot.com/case-studies — Enterprise marketing with ROI data
- SE Ranking — seranking.com — Campaign results with specific metrics

**Expert marketing content:**
- Neil Patel / NP Digital — neilpatel.com — Campaign data and strategy
- AdEspresso — adespresso.com/blog — Facebook/Instagram ad experiments with real metrics
- KlientBoost — klientboost.com/blog — PPC and CRO case studies with numbers
- CXL — cxl.com — Conversion optimization case studies
- Disruptive Advertising — disruptiveadvertising.com — ROAS case studies

**Platform success stories:**
- Meta Business Success Stories — facebook.com/business/success
- Google Ads Customer Stories — ads.google.com/home/resources/
- TikTok Business Case Studies — tiktok.com/business

**Industry reports and academic sources:**
- Marketing Science journal papers on advertising effectiveness
- AdAge industry reports
- Google Scholar: "advertising campaign effectiveness", "digital advertising experiment", "A/B test advertising"

### What Gets Extracted

A frontier LLM extracts and structures:
- Channel selection logic and rationale
- Audience segmentation principles
- Budget allocation heuristics
- Messaging frameworks (AIDA, PAS, BAB, etc.)
- Platform-specific best practices
- Real ROI, ROAS, CPA, conversion metrics tied to specific strategies

### Training Format

Primarily Q&A pairs, strategic reasoning chains, and framework explanations. Supplemented by empirical patterns observed in Source 1.

### Benchmark Data (for Proxy Label Calibration)

Published industry benchmarks used to calibrate performance tiers in Source 1:

- **WordStream / LocaliQ** — wordstream.com — Average CTR, CPC, CVR, CPA by industry for Google and Facebook Ads. 16,000+ US campaigns analyzed annually. Updated yearly since ~2016.
- **LocaliQ 2025 Benchmarks** — localiq.com — Avg CTR 6.66%, avg CPC $5.26 across 20+ industries.
- **Coupler.io PPC Statistics** — blog.coupler.io/ppc-statistics/ — 120+ PPC insights across search, display, social, video.

---

## Source 3: IRA Dataset (VALIDATION — Proxy Score Ground Truth)

3,517 Facebook advertisements with actual spend, impressions, clicks, targeting parameters, demographics, and run dates.

### Purpose

**Not a training source.** Too small (3,517 ads) and too domain-specific (political advertising) to train on meaningfully. Invaluable for two things:

1. **Proxy score validation** — Check whether composite performance scores from Source 1 correlate with real spend/impressions/clicks. If ads scored as "high performing" via proxy signals also have high spend and impressions in IRA data, the proxy methodology is validated. This is a clean experiment and a research contribution.

2. **Structural reference** — One of the only datasets where every field is filled: ad text, images, impressions, clicks, spend, targeting, run dates. Useful for defining what a "complete campaign record" looks like in the training schema.

---

## Source 4: Upworthy Research Archive (SUPPLEMENTARY — Headline Optimization Only)

32,487 experiments, 150,817 headline variations, 538 million participant assignments. Published in Nature Scientific Data.

### Purpose

**Demoted from primary to supplementary.** Provides causal CTR data (randomized A/B tests), but only for one narrow sub-task: which linguistic patterns drive clicks in headlines.

**Not useful for:** Campaign strategy, audience targeting, channel selection, budget allocation, competitive analysis, or anything beyond headline optimization.

### Potential Uses

- **Training (narrow):** Teach the model which headline patterns win A/B tests — useful for the copy generation/variation training format specifically.
- **Evaluation (preferred):** Test whether Draper.ai-generated headlines would win A/B tests against baselines. Better as an eval benchmark than a training source.

---

## Multi-Task Training Example Formats

The same underlying scraped data is constructed into multiple training formats:

### Format 1 — Campaign Generation
- **Input:** Product/brand context (scraped from advertiser's page or inferred from ad)
- **Output:** Full campaign — audience, channels, messaging framework, ad copy variants, CTAs
- **Construction:** Cluster high-performing ads from same advertiser/vertical, use frontier LLM to synthesize into campaign structure

### Format 2 — Ad Critique and Improvement
- **Input:** Here's an ad [low-performing ad from scrape]. What's wrong and how would you improve it?
- **Output:** Analysis of weaknesses + improved version (modeled on what high-performers in same vertical do differently)

### Format 3 — Comparative Analysis
- **Input:** Here are two ads in the same vertical with different performance profiles. Explain why one outperformed the other.
- **Output:** Structural analysis of messaging, CTA, creative choices, audience fit

### Format 4 — Strategic Q&A
- **Input:** Marketing strategy questions (channel selection, budget allocation, audience targeting logic)
- **Output:** Grounded reasoning drawing on empirical patterns from the data
- **Construction:** Primarily from Source 2, supplemented by empirical patterns from Source 1

### Format 5 — Copy Generation and Variation
- **Input:** Generate 5 headline variants for [product] emphasizing [angle]
- **Output:** Variants modeled on linguistic patterns from high-performing ads in that vertical

### Format 6 — Channel/Audience Reasoning
- **Input:** What platform and audience should I target for [product type]?
- **Output:** Recommendation grounded in empirical platform-vertical-demographic patterns from the scrape

---

## Pipeline Flow

```
1. Scrape        → Raw ads with metadata and engagement (Source 1)
2. Score         → Composite performance proxy per ad
3. Filter & Tier → High / medium / low performers
4. Cluster       → Group by advertiser, vertical, platform
5. Construct     → Multiple training formats from same underlying data
6. Reconstruct   → Frontier LLM fills in strategic rationale, audience inference
                    where metadata is incomplete
7. Validate      → Proxy scores against IRA ground truth (Source 3)
8. Quality filter → Remove incoherent or low-quality constructed examples
9. Fine-tune     → QLoRA on 7B base model with multi-task training
```

---

## Budget Considerations

| Item | Cost | Notes |
|------|------|-------|
| AdFlex API | €99/month (Pro plan) | Primary scraping infrastructure. 500K credits/month, 100 credits/call. |
| Apify actors | ~$0.75/1K results | For Meta Ad Library and TikTok Ad Library scraping |
| SerpApi | Variable | For Google Ads Transparency Center |
| Frontier LLM (labeling) | Variable | Claude/GPT-4o for structured extraction |
| A100 GPU | Variable | Cloud or university allocation for fine-tuning |

---

## Key Methodological Concern

The frontier LLM reconstruction step (pipeline step 6) introduces a circularity risk: using GPT-4/Claude to construct training data, then potentially evaluating with the same models as judge.

**Mitigations:**
- Use different models for construction vs. evaluation
- Weight real-world campaign deployment evaluation heavily
- Clearly document which model was used at each stage
- This is a legitimate concern the examiner will raise — address it directly in the thesis
