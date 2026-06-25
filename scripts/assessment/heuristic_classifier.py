"""Vertical classifier for Draper.ai ad corpus.

Infers a real business vertical from advertiser name, landing page URL,
and ad copy using a four-level priority system:

    Level 1 — domain substring match (highest confidence)
    Level 2 — advertiser name regex match
    Level 3 — ad copy (headline + body) regex match
    Level 4 — semantic embedding similarity (fallback for unknowns)

Levels 1-3 are keyword/regex rules: fast, deterministic, zero cost.
Level 4 uses a local sentence-transformer model (all-MiniLM-L6-v2) to
embed the ad text and find the closest vertical via cosine similarity.
It runs only when levels 1-3 return "unknown", so it never fires for
clearly-matched ads. The model is lazy-loaded and cached in memory.

Within each level, verticals are checked in PRIORITY_ORDER so that more
specific verticals (e.g. pharma_medical) beat broader siblings
(e.g. health_wellness) when both would match.

Returns "unknown" when no rule fires and semantic confidence is below
the threshold (default 0.28).
"""

from __future__ import annotations

import logging
import re
from functools import cache
from urllib.parse import urlparse

import numpy as np

from draper.scoring.schemas import ScoredAd

logger = logging.getLogger("draper")

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

VERTICAL_LABELS: list[str] = [
    "pharma_medical",
    "crypto_web3",
    "gambling_betting",
    "nonprofit_charity",
    "gaming",
    "finance",
    "health_wellness",
    "saas_software",
    "b2b_marketing",
    "media_entertainment",
    "fashion_beauty",
    "travel_hospitality",
    "food_beverage",
    "education",
    "real_estate_home",
    "automotive",
    "ecommerce",
    "news_politics",
    "unknown",
]

# The priority order used when checking across verticals at each level.
# More specific / easily confused verticals come first.
PRIORITY_ORDER: list[str] = [
    "pharma_medical",       # before health_wellness
    "crypto_web3",          # before finance & gaming
    "gambling_betting",     # before gaming & finance
    "nonprofit_charity",    # before media/news
    "gaming",               # before media_entertainment
    "finance",
    "health_wellness",
    "saas_software",        # before b2b_marketing & ecommerce
    "b2b_marketing",        # before ecommerce
    "media_entertainment",
    "fashion_beauty",       # before ecommerce
    "travel_hospitality",
    "food_beverage",
    "education",
    "real_estate_home",
    "automotive",
    "ecommerce",
    "news_politics",
]

# ---------------------------------------------------------------------------
# Level 1 — Domain substring fragments
# ---------------------------------------------------------------------------
# Each fragment is checked as a SUBSTRING of the full raw URL string so that
# truncated hostnames like "opify.com" still match "shopify".

_DOMAIN_RULES: dict[str, list[str]] = {
    # --- ecommerce ---
    "ecommerce": [
        "amazon.", "amazon.co",
        "etsy.com", "eettsy.com",
        "temu.com", "emu.com",
        "walmart.com",
        "footlocker.com", "foot-locker.com",
        "farfetch.com",
        "bhphotovideo.com",
        "danburymint.com",
        "bedbathandbeyond.com",
        "ikea.com",
        "gumroad.com",
        "featclothing.com",
        "roadrunnersports.com",
        "olaplex.com",
        "instacart.com",
        "fancyspring.com",
        "namedcollective.com",
        "blue-tomato.com",
        "shineshore.com",
        "projectrepat.com",
        "cusfamilygift.com",
        "bigrigthreads.com",
        "unclesamsshirts.com",
        "robotimonline.com", "rolife.com",
        "helloicejewelry.com",
        "kaokaomi.com",
        "softimply.com",
        "vibzo.com",
        "pandasupps.com", "panda-supps.com",
        "indigo-lune.com",
        "ullpopken.com", "ullapopken.com",
        "kuwala.com", "kulanikinis.com",
        "balibody.com",
        "hexclad.com", "hex-clad.com",
        "target.com",
        "sephora.com",
        "buy.coorslight.com",
        "venderbys.com",
        "marnetic.com",
        "callie.com",
        "indigo-lune.com",
        "four-words.co.nz",
    ],

    # --- saas_software ---
    "saas_software": [
        "shopify.com", "opify.com",         # truncated
        "grammarly.com",
        "clickup.com",
        "canva.com",
        "adobe.com",
        "vercel.com",
        "squarespace.com", "quarespace.com", # truncated
        "elementor.com",
        "hubspot.com", "offers.hubspot.com",
        "kit.com",
        "codemagic.io",
        "beautiful.ai",
        "altitask.com",
        "softplan.com.br",
        "axon.ai",
        "veryfi.com",
        "bluehost.com",
        "wix.com",
        "plan7architect.com",
        "zeroqode.com",
        "freshpaint.io",
        "ovhcloud.com",
        "hpe.com",
        "codefinity.com",
        "bypassgpt.ai",
        "whitebridge.ai",
        "openevidence.com",
        "endel.io",
        "glow.as",
        "knocknoc.com",
        "artfindertcg.com",
        "artfinder.com",
        "notion.so",
        "airtable.com",
        "zapier.com",
        "monday.com",
        "asana.com",
        "figma.com",
        "linear.app",
        "loom.com",
        "intercom.com",
    ],

    # --- gaming ---
    "gaming": [
        "steampowered.com", "store.steampowered.com",
        "plarium.com",
        "hero-wars.com", "herowars.com",
        "warrobots.com", "war-robots.com",
        "eternalfury.com",
        "yuplay.com",
        "chaoscraftgame.com",
        "anzer.quest",
        "ditogames.com",
        "oldschool.runescape.com", "runescape.com",
        "fnafgame.com", "fivenightsatfreddys.com",
        "alliedgaming.gg",
        "game-promo.com",
        "epicgames.com",
        "origin.com",
        "gog.com",
        "itch.io",
        "roblox.com",
        "miniclip.com",
        "gameloft.com",
        "supercell.com",
        "kingdigital.com",
        "ubisoft.com",
    ],

    # --- finance ---
    "finance": [
        "ftmo.com",
        "natwest.com", "nwm.com",
        "fsinsight.com", "fundstrat.com",
        "sofi.com",
        "moneris.com",
        "interactivebrokers.com",
        "iforex.com",
        "mitrade.com",
        "ifinanciamento.it",
        "ekingdombank.com",
        "safeharbor.com",
        "td.com", "tdbank.com",
        "aonaffinity.com", "aon.com",
        "barclays.com",
        "bancociudad.com.ar",
        "fidelity.com",
        "schwab.com",
        "robinhood.com",
        "etrade.com",
        "nerdwallet.com",
        "creditkarma.com",
        "experian.com",
        "equifax.com",
        "paypal.com",
        "stripe.com",
        "wise.com",
        "revolut.com",
        "n26.com",
        "monzo.com",
    ],

    # --- crypto_web3 ---
    "crypto_web3": [
        "bitget.com",
        "mexc.com",
        "zerohash.com",
        "rayls.com",
        "fhenix.io",
        "bcgame.com", "bc.game", "bcgame.co",
        "luma.com",
        "binance.com",
        "coinbase.com",
        "kraken.com",
        "bybit.com",
        "okx.com",
        "kucoin.com",
        "crypto.com",
        "opensea.io",
        "uniswap.org",
        "metamask.io",
        "ledger.com",
        "trezor.io",
        "coingecko.com",
        "coinmarketcap.com",
        "dexscreener.com",
        "pancakeswap.finance",
    ],

    # --- health_wellness ---
    "health_wellness": [
        "goodlifefitness.com",
        "fit4you.com",
        "betterme.world",
        "functionhealth.com",
        "dailyweightadvantage.com",
        "hims.com", "ims.com",           # "ims.com" truncated from hims.com
        "stretchformore.com",
        "pandasupps.com",
        "petwellness.com",
        "beyond-alpha.com",
        "scienceandhumans.com",
        "welleco.com",
        "mindbodyonline.com",
        "calm.com",
        "headspace.com",
        "noom.com",
        "myfitnesspal.com",
        "peloton.com",
        "openfit.com",
        "aaptiv.com",
        "whoop.com",
        "oura.com",
        "8fit.com",
        "centr.com",
        "fiture.com",
        "mirror.co",
        "lesmills.com",
    ],

    # --- pharma_medical ---
    "pharma_medical": [
        "xolair.com",
        "dupixent.com",
        "descovystories.com",
        "carenow.com",
        "ccfatreatment.com",
        "medibank.com.au",
        "pearlevision.com", "earlevision.com",  # truncated
        "archernursing.com",
        "proctolog.com",
        "rxlist.com",
        "drugs.com",
        "webmd.com",
        "healthline.com",
        "medscape.com",
        "cvs.com",
        "walgreens.com",
        "riteaid.com",
        "hims.com",
        "ro.co",
        "forhims.com",
        "nurx.com",
        "sesamecare.com",
        "zocdoc.com",
        "teladoc.com",
        "mdlive.com",
    ],

    # --- nonprofit_charity ---
    "nonprofit_charity": [
        "madinah.com", "cbb.madinah.com", "opefoundation.madinah.com", "onsur.madinah.com",
        "muslimhelpuk.org",
        "matwprojectusa.org", "matwproject.org",
        "spotlightcharity.org",
        "smiletrain.org", "miletrain.org",  # truncated
        "camfed.org",
        "vitalant.org",
        "2t.org",
        "donorsupport.co",
        "wfp.org", "worldfoodprogramme.org",
        "givingcatalog.com",
        "oneummah.org",
        "ummetvakfi.org",
        "redcross.org",
        "unicef.org",
        "savethechildren.org",
        "doctorswithoutborders.org",
        "habitat.org",
        "feedingamerica.org",
        "charitywater.org",
        "stjude.org",
        "cancer.org",
        "diabetes.org",
        "alzfdn.org",
        "nfcr.org",
    ],

    # --- media_entertainment ---
    "media_entertainment": [
        "netflix.com",
        "primevideo.com", "amazon.com/primevideo",
        "foxnation.com",
        "paramountplus.com",
        "gladiatormovie.com",
        "voyo.ro",
        "cravecanada.ca", "crave.ca",
        "njutafilms.se",
        "filmstaden.se",
        "livenation.com",
        "ticketmaster.com",
        "universalaudio.com",
        "boomlibrary.com",
        "youtube.com", "youtu.be",
        "spotify.com",
        "appletv.apple.com",
        "disneyplus.com",
        "hulu.com",
        "hbomax.com", "max.com",
        "peacocktv.com",
        "discoveryplus.com",
        "twitch.tv",
        "soundcloud.com",
        "bandcamp.com",
        "deezer.com",
        "tidal.com",
        "giphy.com",
        "tenor.co",
    ],

    # --- fashion_beauty ---
    "fashion_beauty": [
        "laroche-posay.us", "laroche-posay.com",
        "juicycouture.co.uk", "juicycouture.com",
        "marstheLabel.com",
        "balibody.com",
        "plouise.co.uk",
        "cledepeaubeaute.com",
        "uniwigs.com",
        "kulanikinis.com",
        "jnlnaturals.com",
        "namedcollective.com",
        "helloicejewelry.com",
        "shineshore.com",
        "olaplex.com",
        "barenecessities.com", "bare-necessities.com",
        "comfrt.com",
        "alexandervenacci.com",
        "vessi.com",
        "strass-steentjes.nl",
        "pascaldesign.com",
        "numeris.com",
        "revolve.com",
        "asos.com",
        "zara.com",
        "hm.com",
        "nordstrom.com",
        "macys.com",
        "glossier.com",
        "kylie.com",
        "rhode.com",
        "cerave.com",
        "neutrogena.com",
        "lorealparis.com",
        "maybelline.com",
    ],

    # --- travel_hospitality ---
    "travel_hospitality": [
        "booking.com",
        "esim.holafly.com", "holafly.com",
        "accorplus.com", "accor.com",
        "luxresorts.com",
        "addresshotels.com",
        "go-roadtrip.com",
        "thepointsguy.com", "epointsguy.com",
        "transfergalaxy.com",
        "airback.com",
        "cuballama.com",
        "airbnb.com",
        "expedia.com",
        "tripadvisor.com",
        "hotels.com",
        "kayak.com",
        "skyscanner.com",
        "google.com/flights",
        "united.com",
        "delta.com",
        "aa.com",
        "southwest.com",
        "marriott.com",
        "hilton.com",
        "hyatt.com",
        "ihg.com",
        "trivago.com",
        "agoda.com",
        "hostelworld.com",
        "viator.com",
        "getaround.com",
        "turo.com",
    ],

    # --- food_beverage ---
    "food_beverage": [
        "justfoodfordogs.com",
        "mob.co.uk",
        "buy.coorslight.com", "coorslight.com",
        "majestic.co.uk",
        "stadtsalat.de",
        "coca-cola.com",
        "sweetbay.gr",
        "zenergy.co.nz",
        "doordash.com",
        "ubereats.com",
        "grubhub.com",
        "postmates.com",
        "deliveroo.com",
        "gopuff.com",
        "freshly.com",
        "hungryroot.com",
        "factor75.com",
        "hellofresh.com",
        "homechef.com",
        "sunbasket.com",
        "butcherbox.com",
        "drizly.com",
        "vivino.com",
        "totalwine.com",
        "reservebar.com",
    ],

    # --- education ---
    "education": [
        "online.berklee.edu", "berkleemusic.com",
        "codefinity.com",
        "latrobe.edu.au",
        "uatx.substack.com", "uatx.org",
        "uxarmy.com",
        "coursera.org",
        "udemy.com",
        "skillshare.com",
        "masterclass.com",
        "edx.org",
        "khanacademy.org",
        "duolingo.com",
        "babbel.com",
        "rosettastone.com",
        "codecademy.com",
        "pluralsight.com",
        "linkedin.com/learning",
        "udacity.com",
        "bootcamp.learn.co",
        "generalassemb.ly",
        "flatiron.com",
        "lambdaschool.com",
        "springboard.com",
        "careerkarma.com",
    ],

    # --- real_estate_home ---
    "real_estate_home": [
        "homes.com", "omes.com",           # truncated
        "srmresidential.com",
        "casagrand.in",
        "case-de-lemn.ro",
        "homebuddy.com", "omebuddy.com",   # truncated
        "wickes.co.uk",
        "cozey.ca",
        "sleepcountry.ca", "leepcountry.ca",  # truncated
        "noahome.com", "noa-home.com",
        "caspercanada.ca", "casper.com",
        "woosa.sg",
        "austpek.com.au",
        "sacpoolpros.com",
        "rgplants.co.uk",
        "magnesiacore.com",
        "ecokit.eu",
        "safestore.co.uk",
        "billybyanthem.com", "anthemproperties.com",
        "zillow.com",
        "realtor.com",
        "redfin.com",
        "opendoor.com",
        "offerpad.com",
        "compass.com",
        "homelight.com",
        "wayfair.com",
        "overstock.com",
        "westelm.com",
        "cb2.com",
        "crateandbarrel.com",
        "potterybarn.com",
        "article.com",
    ],

    # --- automotive ---
    "automotive": [
        "buick.ca", "buick.com",
        "mazdausa.com", "mazda.com",
        "kbb.com",
        "slateauto.com", "late.auto",       # "late.auto" truncated
        "wipertech.com",
        "autonation.com",
        "electricmotocross.com",
        "ford.com",
        "chevrolet.com",
        "toyota.com",
        "honda.com",
        "nissan.com",
        "hyundaiusa.com",
        "kia.com",
        "bmwusa.com",
        "mbusa.com",
        "audiusa.com",
        "vw.com",
        "tesla.com",
        "rivian.com",
        "lucidmotors.com",
        "carvana.com",
        "carmax.com",
        "cars.com",
        "autotrader.com",
        "edmunds.com",
        "truecar.com",
    ],

    # --- b2b_marketing ---
    "b2b_marketing": [
        "offers.hubspot.com",
        "affiliateworldconferences.com",
        "sapiosciences.com",
        "rocketdevs.com",
        "usemassive.com",
        "coqli.com",
        "levonterteryan.com",
        "axon.ai",
        "voyagemedia.com",
        "yellow.pro",
        "dwp.gov.uk",
        "usps.com",
        "heavybit.com",
        "kvk.nl",
        "quickbooks.intuit.com", "intuit.com",
        "salesforce.com",
        "marketo.com",
        "pardot.com",
        "mailchimp.com",
        "constantcontact.com",
        "activecampaign.com",
        "klaviyo.com",
        "hootsuite.com",
        "sproutsocial.com",
        "buffer.com",
        "semrush.com",
        "ahrefs.com",
        "moz.com",
        "similarweb.com",
        "drift.com",
        "intercom.com",
        "zendesk.com",
        "freshdesk.com",
        "workday.com",
        "bamboohr.com",
        "greenhouse.io",
        "lever.co",
        "jobvite.com",
        "icims.com",
    ],

    # --- gambling_betting ---
    "gambling_betting": [
        "betnacional.com",
        "gaminggiveaways.co.uk",
        "williamhillvegas.com", "williamhill.com",
        "pulsz.com", "ulsz.com",           # truncated
        "zenarcade.com", "afeplacegaming.com",
        "electricslots.com",
        "bc.game", "bcgame.com", "bcgame.co",
        "hol.org",
        "luckysync.com", "lucksync.com",
        "draftkings.com",
        "fanduel.com",
        "betmgm.com",
        "caesarssports.com",
        "pointsbet.com",
        "betonline.ag",
        "bovada.lv",
        "888casino.com",
        "bet365.com",
        "betway.com",
        "unibet.com",
        "pokerstars.com",
        "wsop.com",
        "worldwinner.com",
        "lucktastic.com",
        "jackpotjoy.com",
        "slotomania.com",
    ],

    # --- news_politics ---
    "news_politics": [
        "foxnation.com",
        "wsj.com",
        "mldiario.com",
        "derechadiario.com.ar",
        "registertovote.london",
        "donwinslow.com",
        "shoutoutuk.co.uk",
        "nytimes.com",
        "washingtonpost.com",
        "theguardian.com",
        "bbc.com", "bbc.co.uk",
        "cnn.com",
        "reuters.com",
        "apnews.com",
        "politico.com",
        "axios.com",
        "thehill.com",
        "breitbart.com",
        "dailywire.com",
        "jacobinmag.com",
        "motherjones.com",
        "vox.com",
        "buzzfeednews.com",
        "huffpost.com",
    ],
}

# ---------------------------------------------------------------------------
# Level 2 — Advertiser name regex patterns
# ---------------------------------------------------------------------------

_NAME_PATTERN_DEFS: dict[str, list[str]] = {
    # --- ecommerce ---
    "ecommerce": [
        r"\bamazon\b",
        r"\betsy\b",
        r"\btemu\b",
        r"\bwalmart\b",
        r"\bfoot\s*locker\b",
        r"\bfarfetch\b",
        r"\bikea\b",
        r"\binstacart\b",
        r"\bolaplex\b",
        r"\bhex\s*clad\b",
        r"\bsephora\b",
        r"\btarget\b",
        r"\bdanbury\s*mint\b",
        r"\bbed\s*bath\s*(and|&|beyond)\b",
        r"\bbig\s*rig\s*threads\b",
        r"\buncle\s*sam'?s\s*shirts\b",
        r"\brolife\b",
        r"\brobotimonline\b",
        r"\bhello\s*ice\s*jewelry\b",
        r"\bkao\s*kaomi\b",
        r"\bshineshore\b",
        r"\bnamed\s*collective\b",
        r"\bproject\s*repat\b",
        r"\bfancy\s*spring\b",
        r"\bindigo\s*lune\b",
        r"\bulla\s*popken\b",
        r"\bkula\s*nikin\b",
        r"\bbali\s*body\b",
        r"\bfour\s*words\b",
        r"\bvibzo\b",
        r"\bpanda\s*supps\b",
        r"\bsoftimply\b",
        r"\bgumroad\b",
        r"\bfeatclothing\b",
        r"\broad\s*runner\s*sports\b",
        r"\bblue[-\s]?tomato\b",
        r"\bmaltesers\b", r"\bmalteasers\b",
        r"\bcoors\s*light\b",
    ],

    # --- saas_software ---
    "saas_software": [
        r"\bshopify\b",
        r"\bgrammarly\b",
        r"\bclickup\b",
        r"\bcanva\b",
        r"\badobe\b",
        r"\bvercel\b",
        r"\bsquarespace\b",
        r"\belementor\b",
        r"\bhubspot\b",
        r"\bwix\b",
        r"\bbluehost\b",
        r"\bbeautiful\.?ai\b",
        r"\baltitask\b",
        r"\bsoftplan\b",
        r"\baxon\.?ai\b",
        r"\bveryfi\b",
        r"\bplan7architect\b",
        r"\bzeroqode\b",
        r"\bfreshpaint\b",
        r"\bovhcloud\b",
        r"\bhpe\b",
        r"\bcodefinity\b",
        r"\bbypass\s*gpt\b",
        r"\bwhitebridge\b",
        r"\bopenevidence\b",
        r"\bendel\b",
        r"\bglow\.as\b",
        r"\bknocknoc\b",
        r"\bnotion\b",
        r"\bairtable\b",
        r"\bzapier\b",
        r"\bmonday\.com\b",
        r"\basana\b",
        r"\bfigma\b",
        r"\blinear\b",
        r"\bloom\b",
        r"\bintercom\b",
        r"\bcodemagic\b",
        r"\bkit\.com\b",
        r"\bkit\s+email\b",
    ],

    # --- gaming ---
    "gaming": [
        r"\bplarium\b",
        r"\braid[\s:-]*shadow\s*legends\b",
        r"\bhero[\s-]*wars\b",
        r"\bwar[\s-]*robots\b",
        r"\beternal\s*fury\b",
        r"\byuplay\b",
        r"\bchaoscraft\b",
        r"\bdito\s*games\b",
        r"\bold\s*school\s*runescape\b",
        r"\bfive\s*nights\s*at\s*freddy'?s\b",
        r"\ballied\s*gaming\b",
        r"\bepic\s*games\b",
        r"\broblox\b",
        r"\bsupercell\b",
        r"\belectronic\s*arts\b",
        r"\bactivision\b",
        r"\bblizzard\b",
        r"\bubisoft\b",
        r"\b2k\s*games\b",
        r"\bnintendo\b",
        r"\bsteam\b",
        r"\bgog\b",
        r"\bitch\.io\b",
        r"\bking\b(?=.*game|\s+game)",
        r"\bgame\s*loft\b", r"\bgameloft\b",
    ],

    # --- finance ---
    "finance": [
        r"\bftmo\b",
        r"\bnatwest\b",
        r"\bfsinsight\b", r"\bfundstrat\b",
        r"\bsofi\b",
        r"\bmoneris\b",
        r"\binteractive\s*brokers\b",
        r"\biforex\b",
        r"\bmitrade\b",
        r"\bsafe\s*harbor\s*financial\b",
        r"\bbanc[oa]\s*ciudad\b",
        r"\btd\s*bank\b",
        r"\baon\s*affinity\b",
        r"\bbarclays\b",
        r"\bfidelity\b",
        r"\bschwab\b",
        r"\brobinhood\b",
        r"\betrade\b",
        r"\bnerdwallet\b",
        r"\bcredit\s*karma\b",
        r"\bexperian\b",
        r"\bpaypal\b",
        r"\bstripe\b",
        r"\bwise\b",
        r"\brevolut\b",
        r"\bn26\b",
        r"\bmonzo\b",
        r"\bchime\b",
        r"\bsofi\b",
        r"\bsynchrony\b",
        r"\bdisc?over\b",
        r"\bamex\b", r"\bamerican\s*express\b",
    ],

    # --- crypto_web3 ---
    "crypto_web3": [
        r"\bbitget\b",
        r"\bmexc\b",
        r"\bzerohash\b",
        r"\brayls\b",
        r"\bfhenix\b",
        r"\bbc\.?game\b",
        r"\bbinance\b",
        r"\bcoinbase\b",
        r"\bkraken\b",
        r"\bbybit\b",
        r"\bokx\b",
        r"\bkucoin\b",
        r"\bcrypto\.com\b",
        r"\bopensea\b",
        r"\buniswap\b",
        r"\bledger\b",
        r"\btrezor\b",
        r"\bdefi\b",
        r"\bnft\s*(platform|market|store|drop)\b",
        r"\bweb3\b",
        r"\bblockchain\b(?=.*platform|.*wallet|.*exchange)",
        r"\bcoinmarketcap\b",
        r"\bcoingecko\b",
        r"\bpancakeswap\b",
    ],

    # --- health_wellness ---
    "health_wellness": [
        r"\bgood\s*life\s*fitness\b",
        r"\bfit4you\b",
        r"\bbetterme\b",
        r"\bfunction\s*health\b",
        r"\bdaily\s*weight\s*advantage\b",
        r"\bgundry\s*md\b",
        r"\bhims\b",
        r"\bstretch\s*for\s*more\b",
        r"\bpanda\s*supps\b",
        r"\bpet\s*wellness\b",
        r"\bbeyond[\s-]*alpha\b",
        r"\bscience\s*(and|&)\s*humans\b",
        r"\bfit\s*hub\b",
        r"\bwelleco\b",
        r"\bcalm\b",
        r"\bheadspace\b",
        r"\bnoom\b",
        r"\bmyfitnesspal\b",
        r"\bpeloton\b",
        r"\bwhoop\b",
        r"\boura\b",
        r"\bragglan\s*gym\b",
        r"\bgymnastics?\b",
        r"\bwellness\s*coach\b",
        r"\bmental\s*health\s*(app|platform|coach)\b",
        r"\bmeditation\s*app\b",
        r"\bsupplement[s]?\s*(brand|company|store)\b",
    ],

    # --- pharma_medical ---
    "pharma_medical": [
        r"\bxolair\b",
        r"\bdupixent\b",
        r"\bdescovy\b",
        r"\bcare\s*now\b",
        r"\bccfa\s*treatment\b",
        r"\bproctolog\b",
        r"\bmedibank\b",
        r"\bpearle\s*vision\b",
        r"\barcher\s*nursing\b",
        r"\bpfizer\b",
        r"\bj&?j\b", r"\bjohnson\s*&?\s*johnson\b",
        r"\babbvie\b",
        r"\belilly\b", r"\beli\s*lilly\b",
        r"\bmerck\b",
        r"\bbristol[\s-]*myers\b",
        r"\bgenentech\b",
        r"\bamgen\b",
        r"\bnovocare\b",
        r"\bnovonordisk\b",
        r"\bastrazeneca\b",
        r"\bgskclinical\b",
        r"\bzocdoc\b",
        r"\bteladoc\b",
        r"\bmdlive\b",
        r"\bnurx\b",
        r"\bro\.?co\b",
        r"\bsesame\s*care\b",
        r"\bwalgreens?\b",
        r"\bcvs\b",
        r"\brite\s*aid\b",
    ],

    # --- nonprofit_charity ---
    "nonprofit_charity": [
        r"\bmadinah\b",
        r"\bmuslim\s*help\s*uk\b",
        r"\bmatw\s*project\b",
        r"\bspotlight\s*(charity|humanity)\b",
        r"\bsmile\s*train\b",
        r"\bcamfed\b",
        r"\bvitalant\b",
        r"\btunnel\s*to\s*towers\b",
        r"\bdonor\s*support\b",
        r"\bworld\s*food\s*programme\b",
        r"\bwfp\b",
        r"\bgov\s*glance\s*foundation\b",
        r"\bgiving\s*catalog\b",
        r"\bone\s*ummah\b",
        r"\bummet\s*vakfi\b",
        r"\bred\s*cross\b",
        r"\bunicef\b",
        r"\bsave\s*the\s*children\b",
        r"\bdoctors\s*without\s*borders\b",
        r"\bhabitat\s*for\s*humanity\b",
        r"\bfeeding\s*america\b",
        r"\bcharity\s*water\b",
        r"\bst\.?\s*jude\b",
        r"\bamerican\s*cancer\s*society\b",
        r"\bnational\s*diabetes\b",
        r"\bmosque\b", r"\bmasjid\b",
        r"\bchurch\b(?=.*give|.*donate|.*charity)",
        r"\bfundraising\s*(org|foundation|charity)\b",
    ],

    # --- media_entertainment ---
    "media_entertainment": [
        r"\bnetflix\b",
        r"\bprime\s*video\b", r"\bamazon\s*prime\b",
        r"\bfox\s*nation\b",
        r"\bparamount\+?\b",
        r"\bvoyo\b",
        r"\bcrave\s*(canada|tv)?\b",
        r"\bnjutafilms\b",
        r"\bfilmstaden\b",
        r"\blive\s*nation\b",
        r"\bticketmaster\b",
        r"\buniversal\s*audio\b",
        r"\bboom\s*library\b",
        r"\bbase\s*entertainment\b",
        r"\byoutube\b",
        r"\bspotify\b",
        r"\bapple\s*tv\+?\b",
        r"\bdisney\+?\b",
        r"\bhulu\b",
        r"\bhbo\s*(max)?\b",
        r"\bmax\b(?=.*streaming|.*hbo)",
        r"\bpeacock\s*(tv)?\b",
        r"\bdiscovery\+?\b",
        r"\btwitch\b",
        r"\bsoundcloud\b",
        r"\bbandcamp\b",
        r"\bdeezer\b",
        r"\btidal\b",
        r"\bbox\s*office\b",
        r"\bcinema\b", r"\bcineplex\b",
        r"\btheater\b", r"\btheatre\b",
        r"\bpodcast\s*(network|studio|app)\b",
    ],

    # --- fashion_beauty ---
    "fashion_beauty": [
        r"\bla\s*roche[\s-]*posay\b",
        r"\bjuicy\s*couture\b",
        r"\bmars\s*the\s*label\b",
        r"\bbali\s*body\b",
        r"\bplouise\b",
        r"\bcle\s*de\s*peau\b",
        r"\buniwigs?\b",
        r"\bkula\s*nikinis\b",
        r"\bjnl\s*naturals\b",
        r"\bnamed\s*collective\b",
        r"\bolaplex\b",
        r"\bbare\s*(minerals|necessities)\b",
        r"\bcomfrt\b",
        r"\balexander\s*venacci\b",
        r"\bvessi\b",
        r"\bstrass[\s-]*steentjes\b",
        r"\bpascal\s*design\b",
        r"\basos\b",
        r"\bzara\b",
        r"\bh&m\b", r"\bh\s*and\s*m\b",
        r"\bnordstrom\b",
        r"\bmacy'?s\b",
        r"\bglossier\b",
        r"\bkylie\s*(cosmetics|skin|jenner)\b",
        r"\brhode\b(?=.*skin|.*beauty)",
        r"\bcerave\b",
        r"\bneutrogena\b",
        r"\bloreal\b", r"\bl'oreal\b",
        r"\bmaybelline\b",
        r"\brevlon\b",
        r"\bmax\s*factor\b",
        r"\burban\s*decay\b",
        r"\bnyx\s*(cosmetics)?\b",
        r"\bcoloropop\b",
        r"\bsmashbox\b",
        r"\btarte\b(?=.*cosmetics|.*beauty)",
        r"\bbenefitcosmetics\b", r"\bbenefit\s*cosmetics\b",
    ],

    # --- travel_hospitality ---
    "travel_hospitality": [
        r"\bbooking\.com\b",
        r"\bholafly\b",
        r"\baccor(\s*plus)?\b",
        r"\blux\s*resorts\b",
        r"\baddress\s*hotels\b",
        r"\bgo[\s-]*roadtrip\b",
        r"\bthe\s*points\s*guy\b",
        r"\btransfer\s*galaxy\b",
        r"\bairback\b",
        r"\bcuballama\b",
        r"\bairbnb\b",
        r"\bexpedia\b",
        r"\btripadvisor\b",
        r"\bhotels\.com\b",
        r"\bkayak\b",
        r"\bskyscanner\b",
        r"\bunited\s*airlines\b",
        r"\bdelta\s*airlines\b",
        r"\bamerican\s*airlines\b",
        r"\bsouthwest\s*airlines\b",
        r"\bmarriott\b",
        r"\bhilton\b",
        r"\bhyatt\b",
        r"\bihg\b",
        r"\btrivago\b",
        r"\bagoda\b",
        r"\bhostelworld\b",
        r"\bviator\b",
        r"\bturo\b",
        r"\bgetaround\b",
        r"\bairalo\b",
        r"\besim\b(?=.*travel|.*international|.*roaming)",
    ],

    # --- food_beverage ---
    "food_beverage": [
        r"\bjust\s*food\s*for\s*dogs\b",
        r"\bmob\s*(kitchen|recipe|meal)?\b",
        r"\bcoors\s*light\b",
        r"\bmajestic\s*wine\b",
        r"\bstadtsalat\b",
        r"\bcoca[\s-]*cola\b",
        r"\bsweetbay\b",
        r"\bz\s*energy\b",
        r"\bdoordash\b",
        r"\buber\s*eats\b",
        r"\bgrubhub\b",
        r"\bdeliveroo\b",
        r"\bgopuff\b",
        r"\bhello\s*fresh\b",
        r"\bhome\s*chef\b",
        r"\bsun\s*basket\b",
        r"\bbutcher\s*box\b",
        r"\bdrizly\b",
        r"\bvivino\b",
        r"\btotal\s*wine\b",
        r"\bknorr\b",
        r"\bminute\s*maid\b",
        r"\bstarbucks\b",
        r"\bmcdonald'?s\b",
        r"\bkfc\b",
        r"\bsubway\b",
        r"\bchipotle\b",
        r"\bdominos?\b",
        r"\bpizza\s*hut\b",
        r"\bdairy\s*queen\b",
        r"\bburger\s*king\b",
        r"\bwendy'?s\b",
        r"\bpanera\b",
        r"\bheineken\b",
        r"\bbudweiser\b",
        r"\bcorona\b(?=.*beer|.*cerveza)",
        r"\bnestl[eé]\b",
        r"\bunilever\b",
        r"\bcraft\s*beer\b", r"\bbrewery\b",
    ],

    # --- education ---
    "education": [
        r"\bberklee\s*online\b",
        r"\bcodefinity\b",
        r"\bla\s*trobe\b",
        r"\buniversity\s*of\s*austin\b",
        r"\bux\s*army\b",
        r"\bcoursera\b",
        r"\budemy\b",
        r"\bskillshare\b",
        r"\bmasterclass\b",
        r"\bedx\b",
        r"\bkhan\s*academy\b",
        r"\bduolingo\b",
        r"\bbabbel\b",
        r"\brosetta\s*stone\b",
        r"\bcodeacademy\b", r"\bcodeacademy\b",
        r"\bpluralsight\b",
        r"\budacity\b",
        r"\bgeneral\s*assembly\b",
        r"\bflatiron\s*(school)?\b",
        r"\blambda\s*school\b",
        r"\bspringboard\b",
        r"\bcareer\s*karma\b",
        r"\bbootcamp\b",
        r"\bonline\s*university\b",
        r"\bonline\s*college\b",
        r"\bdistance\s*learning\b",
        r"\buniversity\b(?=.*online|.*apply|.*enroll)",
        r"\bcollege\b(?=.*online|.*apply|.*enroll)",
        r"\btaylors?\s*university\b",
        r"\baustinscholar\b",
    ],

    # --- real_estate_home ---
    "real_estate_home": [
        r"\bsrm\s*residential\b",
        r"\bcasagrand\b",
        r"\bcostantini\s*case\b",
        r"\bcase[\s-]*de[\s-]*lemn\b",
        r"\bhomebuddy\b",
        r"\bwickes\b",
        r"\bcozey\b",
        r"\bsleep\s*country\b",
        r"\bnoa\s*home\b",
        r"\bcasper\s*(sleep|canada)?\b",
        r"\bwoosa\b",
        r"\baustpek\b",
        r"\bsac\s*pool\s*pros\b",
        r"\brg\s*plants\b",
        r"\bmagnesiacore\b",
        r"\becokit\b",
        r"\bsafestore\b",
        r"\banthems?\s*properties\b",
        r"\bzillow\b",
        r"\brealtor\b",
        r"\bredfin\b",
        r"\bopendoor\b",
        r"\bcompass\b(?=.*real\s*estate|.*realty|.*homes)",
        r"\bhomelight\b",
        r"\bwayfair\b",
        r"\boverstock\b",
        r"\bwest\s*elm\b",
        r"\bcb2\b",
        r"\bcrate\s*(and|&)\s*barrel\b",
        r"\bpottery\s*barn\b",
        r"\barticle\b(?=.*furniture|.*sofa|.*home)",
        r"\bfirst\s*home\s*specialists\b",
    ],

    # --- automotive ---
    "automotive": [
        r"\bbuick\b",
        r"\bmazda\b",
        r"\bkelley\s*blue\s*book\b", r"\bkbb\b",
        r"\bslate\s*auto\b",
        r"\bwiper\s*tech\b",
        r"\bautonation\b",
        r"\belectric\s*motocross\b",
        r"\bford\b(?=.*motor|.*vehicle|.*truck|.*cars)",
        r"\bchevrolet\b", r"\bchevy\b",
        r"\btoyota\b",
        r"\bhonda\b",
        r"\bnissan\b",
        r"\bhyundai\b",
        r"\bkia\b",
        r"\bbmw\b",
        r"\bmercedes[\s-]*benz\b",
        r"\baudi\b",
        r"\bvolkswagen\b", r"\bvw\b",
        r"\btesla\b",
        r"\brivian\b",
        r"\blucid\s*motors\b",
        r"\bcarvana\b",
        r"\bcarmax\b",
        r"\bautotrader\b",
        r"\bedmunds\b",
        r"\btruecar\b",
        r"\bgoodyear\b",
        r"\bmichelin\b",
        r"\bbridgestone\b",
        r"\bmidas\b",
        r"\bjiffy\s*lube\b",
        r"\bpep\s*boys\b",
    ],

    # --- b2b_marketing ---
    "b2b_marketing": [
        r"\baffiliate\s*world\s*conferences\b",
        r"\bsapio\s*sciences\b",
        r"\brocke?t\s*devs\b",
        r"\busemassive\b",
        r"\bcoqli\b",
        r"\blevon\s*terteryan\b",
        r"\bvoyage\s*media\b",
        r"\byellow\.pro\b",
        r"\bheavybit\b",
        r"\bkamer\s*van\s*koophandel\b",
        r"\bquickbooks\b",
        r"\bintuit\b",
        r"\bsalesforce\b",
        r"\bmarketo\b",
        r"\bmailchimp\b",
        r"\bconstant\s*contact\b",
        r"\bactivecampaign\b",
        r"\bklaviyo\b",
        r"\bhootsuite\b",
        r"\bsprout\s*social\b",
        r"\bbuffer\b(?=.*social|.*marketing|.*schedule)",
        r"\bsemrush\b",
        r"\bahrefs\b",
        r"\bmoz\b(?=.*seo|.*marketing)",
        r"\bsimilarweb\b",
        r"\bdrift\b(?=.*marketing|.*sales|.*chatbot)",
        r"\bzendesk\b",
        r"\bfreshdesk\b",
        r"\bworkday\b",
        r"\bbamboohr\b",
        r"\bgreenhouse\b(?=.*recruit|.*hire|.*talent)",
        r"\bjobvite\b",
        r"\bicims\b",
        r"\brecruiting\s*(platform|software|tool)\b",
        r"\bhr\s*(software|platform|solution)\b",
    ],

    # --- gambling_betting ---
    "gambling_betting": [
        r"\bbetnacional\b",
        r"\bgaming\s*giveaways\b",
        r"\bwilliam\s*hill\b",
        r"\bpulsz\b",
        r"\bzenarcade\b",
        r"\belectric\s*slots\b",
        r"\bbc\.?game\b",
        r"\blucky\s*sync\b", r"\blucksync\b",
        r"\bdraftkings\b",
        r"\bfanduel\b",
        r"\bbetmgm\b",
        r"\bcaesars\s*(sports|casino|sportsbook)\b",
        r"\bpointsbet\b",
        r"\bbet365\b",
        r"\bbetway\b",
        r"\bunibet\b",
        r"\bpokerstars\b",
        r"\bworld\s*series\s*of\s*poker\b",
        r"\bjackpotjoy\b",
        r"\bslotomania\b",
        r"\blucktastic\b",
        r"\bworldwinner\b",
        r"\bsweepstakes\s*casino\b",
        r"\bsocial\s*casino\b",
    ],

    # --- news_politics ---
    "news_politics": [
        r"\bfox\s*nation\b",
        r"\bwall\s*street\s*journal\b", r"\bwsj\b",
        r"\bmundo\s*libre\b",
        r"\bla\s*derecha\s*diario\b",
        r"\bdon\s*winslow\b",
        r"\bshout\s*out\s*uk\b",
        r"\bnew\s*york\s*times\b", r"\bnytimes\b",
        r"\bwashington\s*post\b",
        r"\bthe\s*guardian\b",
        r"\bbbc\s*(news|world)?\b",
        r"\bcnn\b",
        r"\breuters\b",
        r"\bap\s*news\b", r"\bassociated\s*press\b",
        r"\bpolitico\b",
        r"\baxios\b",
        r"\bthe\s*hill\b",
        r"\bbreitbart\b",
        r"\bdaily\s*wire\b",
        r"\bjacobinmag\b",
        r"\bvox\b(?=.*news|.*media|.*politics)",
        r"\bhuffpost\b",
        r"\bdemocracy\s*of\s*hope\b",
        r"\bstef?an\s*radu\s*oprea\b",
        r"\bnews\s*(outlet|network|channel)\b",
        r"\bpolitical\s*(party|org|movement)\b",
        r"\badvocacy\s*(org|group|campaign)\b",
        r"\bcampaign\b(?=.*vote|.*elect|.*political)",
    ],
}

# --- Level 3 — Ad copy regex patterns ---

_COPY_PATTERN_DEFS: dict[str, list[str]] = {
    # --- ecommerce ---
    "ecommerce": [
        r"\bshop\s*now\b", r"\bshop\s*online\b",
        r"\badd\s*to\s*cart\b",
        r"\bfree\s*shipping\b",
        r"\b(limited\s*time\s*)?offer[s]?\b(?=.*shop|.*buy|.*order)",
        r"\bsale\s*(ends|off|today)\b",
        r"\bexclusive\s*deal[s]?\b",
        r"\bbuy\s*(now|one|two|three|get)\b",
        r"\borders?\s*ship\s*(today|fast|free|worldwide)\b",
        r"\btracked\s*delivery\b",
        r"\bmarket\s*place\b",
        r"\bphysical\s*goods\b",
        r"\bonline\s*store\b",
        r"\bonline\s*retail\b",
        r"\bd2c\b", r"\bdirect[\s-]to[\s-]consumer\b",
        r"\bproduct\s*launch\b",
        r"\bnew\s*collection\b",
        r"\bunboxing\b",
        r"\bwarranty\b(?=.*product|.*buy|.*shop)",
        r"\bmade[\s-]to[\s-]order\b",
        r"\bhandmade\b",
        r"\bartisan\b",
        r"\bcustom\s*(print|shirt|gift|jewelry|mug)\b",
        r"\breturn\s*policy\b",
        r"\b30[\s-]*day\s*(return|money\s*back)\b",
        r"\bgift\s*(idea|for|wrap|card)\b",
        r"\bsize\s*(guide|chart)\b",
        r"\bin\s*stock\b",
        r"\bonly\s+\d+\s*left\b",
        r"\bflash\s*sale\b",
        r"\bclearance\b",
    ],

    # --- saas_software ---
    "saas_software": [
        r"\bfree\s*trial\b",
        r"\bsign\s*up\s*free\b",
        r"\bstart\s*for\s*free\b",
        r"\b(monthly|annual|yearly)\s*subscription\b",
        r"\bper\s*(user|seat|month)\b",
        r"\bsaas\b",
        r"\bdashboard\b",
        r"\bworkflow\s*automation\b",
        r"\bno[\s-]*code\b",
        r"\blow[\s-]*code\b",
        r"\bintegrat\w+\b(?=.*tool|.*app|.*platform|.*software)",
        r"\bapi\b(?=.*developer|.*connect|.*integration)",
        r"\bcloud[\s-]*based\b",
        r"\bproductivity\s*(tool|app|platform|software)\b",
        r"\bproject\s*management\b",
        r"\btask\s*management\b",
        r"\bteam\s*collaboration\b",
        r"\bremote\s*(team|work)\b",
        r"\bcode\s*(editor|platform|tool)\b",
        r"\bwebsite\s*builder\b",
        r"\bdrag[\s-]*and[\s-]*drop\b",
        r"\btemplate[s]?\b(?=.*website|.*design|.*app)",
        r"\bplugin[s]?\b",
        r"\bextension[s]?\b(?=.*chrome|.*browser|.*app)",
        r"\bai[\s-]*powered\b(?=.*tool|.*software|.*app|.*platform)",
        r"\bwriting\s*(assistant|tool|ai)\b",
        r"\bgrammar\s*(check|tool|fix)\b",
        r"\bseo\s*(tool|platform|software)\b",
        r"\bemail\s*marketing\s*(tool|platform|software)\b",
        r"\bautomat\w+\b(?=.*email|.*marketing|.*workflow)",
    ],

    # --- gaming ---
    "gaming": [
        r"\bplay\s*(now|free|online|today)\b",
        r"\bdownload\s*(now|free|the\s*game)\b",
        r"\bjoin\s*(the\s*battle|the\s*fight|millions\s*of\s*players)\b",
        r"\blevel\s*(up|system|design)\b",
        r"\bin[\s-]*game\b",
        r"\bgame\s*(play|mode|update|event|server)\b",
        r"\braid\b(?=.*boss|.*dungeon|.*legendary|.*hero)",
        r"\bguild\b", r"\bclan\b",
        r"\bmultiplayer\b", r"\bpvp\b", r"\bpve\b",
        r"\bquest[s]?\b", r"\bmission[s]?\b(?=.*complete|.*game)",
        r"\bhero[s]?\b(?=.*battle|.*game|.*warrior)",
        r"\bcharacter\b(?=.*game|.*unlock|.*upgrade)",
        r"\bskill\s*tree\b",
        r"\bloot\b", r"\bdrop\s*rate\b",
        r"\bgameplay\b",
        r"\bgaming\s*(gear|setup|peripheral)\b",
        r"\bpc\s*gaming\b",
        r"\bconsole\s*(game|gaming|exclusive)\b",
        r"\bmobile\s*game\b",
        r"\bapp\s*store\b(?=.*game|.*download)",
        r"\bgoogle\s*play\b(?=.*game|.*download)",
        r"\bgame\s*key[s]?\b",
        r"\bsteam\s*key\b",
        r"\bepic\s*games\s*store\b",
        r"\besports\b",
        r"\btournament\b(?=.*game|.*prize|.*esport)",
        r"\bstreamer\b", r"\btwitch\s*streamer\b",
    ],

    # --- finance ---
    "finance": [
        r"\binvest\w+\b",
        r"\btrading\s*(platform|account|signal)\b",
        r"\bstock[s]?\s*(market|broker|trade)\b",
        r"\bforex\b",
        r"\bportfolio\b(?=.*invest|.*manage|.*grow)",
        r"\bcapital\s*(gain[s]?|market|fund)\b",
        r"\bsavings?\s*(account|rate|goal|plan)\b",
        r"\bhigh[\s-]*yield\s*savings\b",
        r"\binterest\s*rate\b",
        r"\b(personal|home|auto)\s*loan\b",
        r"\bmortgage\b",
        r"\binsurance\s*(plan|quote|policy)\b",
        r"\blife\s*insurance\b",
        r"\bhealth\s*insurance\b(?=.*plan|.*quote|.*enroll)",
        r"\bretirement\s*(plan|account|savings)\b",
        r"\b(roth\s*)?ira\b",
        r"\b401k\b",
        r"\bfinancial\s*(advisor|planning|freedom)\b",
        r"\bcredit\s*(card|score|report|limit)\b",
        r"\bdebt\s*(management|consolidation|free)\b",
        r"\bbudget(ing|ing\s*app|ing\s*tool)?\b",
        r"\bpayment\s*(processing|gateway|solution)\b",
        r"\bmoney\s*transfer\b",
        r"\bremittance\b",
        r"\bcurrency\s*exchange\b",
        r"\bprop\s*trading\b",
        r"\bfunded\s*account\b",
        r"\btrading\s*challenge\b",
        r"\bbanking\s*(app|platform|solution)\b",
    ],

    # --- crypto_web3 ---
    "crypto_web3": [
        r"\bcryptocurrenc\w+\b",
        r"\bbitcoin\b", r"\bethereum\b",
        r"\bblockchain\b",
        r"\bdefi\b",
        r"\bnft[s]?\b",
        r"\bweb3\b",
        r"\btoken[s]?\b(?=.*crypto|.*blockchain|.*defi|.*launch|.*mint)",
        r"\bmint\s*(nft|token|pass)\b",
        r"\bcrypto\s*(wallet|exchange|trading|investment|market)\b",
        r"\bdecentralized\s*(finance|exchange|app)\b",
        r"\bsmart\s*contract[s]?\b",
        r"\bliquidity\s*(pool|mining|farming)\b",
        r"\bstaking\b(?=.*crypto|.*eth|.*token|.*reward)",
        r"\byield\s*farming\b",
        r"\bgas\s*fee[s]?\b",
        r"\balt\s*coin[s]?\b", r"\baltcoin[s]?\b",
        r"\bhodl\b",
        r"\bsatoshi\b",
        r"\bmetaverse\b",
        r"\bdao\b(?=.*govern|.*token|.*vote)",
        r"\bplay[\s-]to[\s-]earn\b",
        r"\bmove[\s-]to[\s-]earn\b",
        r"\bcrypto\s*presale\b",
        r"\btoken\s*launch\b", r"\btge\b",
        r"\bexchange\s*(listing|launch|platform)\b(?=.*token|.*coin|.*crypto)",
        r"\bspot\s*trading\b(?=.*crypto|.*btc|.*exchange)",
        r"\bfutures\s*trading\b(?=.*crypto|.*btc|.*exchange)",
    ],

    # --- health_wellness ---
    "health_wellness": [
        r"\bweight\s*(loss|management|goal)\b",
        r"\blose\s*(weight|fat|belly)\b",
        r"\bfit(ness)?\s*(goal[s]?|journey|challenge)\b",
        r"\bwork\s*out\b", r"\bworkout\s*(plan|routine|app)\b",
        r"\bmental\s*(health|wellness|wellbeing)\b",
        r"\bstress\s*(relief|reduction|management)\b",
        r"\bmeditat\w+\b",
        r"\bmindfulness\b",
        r"\bsleep\s*(better|quality|improvement|tracker)\b",
        r"\bsupplement[s]?\b(?=.*health|.*wellness|.*fitness|.*muscle|.*energy)",
        r"\bprotein\s*(shake|powder|bar|supplement)\b",
        r"\bpre[\s-]*workout\b",
        r"\bcollagen\b",
        r"\bprobiotics?\b",
        r"\bvitamin[s]?\b(?=.*health|.*wellness|.*supplement)",
        r"\bpersonal\s*trainer\b",
        r"\bgymnastics?\b", r"\bgyms?\b(?=.*member|.*join|.*near)",
        r"\byoga\b",
        r"\bpilates\b",
        r"\bmarathon\b", r"\b5k\b(?=.*train|.*run|.*race)",
        r"\bwellness\s*(app|program|journey|coach)\b",
        r"\bhealth\s*(tracking|coach|app|program)\b",
        r"\bbody\s*(transform|composition|fat|mass)\b",
        r"\bhormone\b(?=.*health|.*therapy|.*balance)",
        r"\btestosterone\b",
        r"\bguthealth\b", r"\bgut\s*health\b",
    ],

    # --- pharma_medical ---
    "pharma_medical": [
        r"\bprescription\b",
        r"\bclinical\s*trial[s]?\b",
        r"\bfda[\s-]approved\b",
        r"\bside\s*effect[s]?\b",
        r"\bdosage\b",
        r"\bmedication[s]?\b",
        r"\bdrug[s]?\b(?=.*prescription|.*fda|.*approved|.*clinical)",
        r"\btreatment\s*(option[s]?|plan|center)\b",
        r"\bsymptom[s]?\b(?=.*treat|.*relieve|.*manage|.*reduce)",
        r"\bdiagnos\w+\b",
        r"\bdoctor\s*(visit|consult|recommend)\b",
        r"\bspecialist\b(?=.*medical|.*refer|.*consult)",
        r"\bsurgery\b",
        r"\brehabilitation\b", r"\brehab\s*center\b",
        r"\burgent\s*care\b",
        r"\bemergency\s*(room|care|service)\b",
        r"\bmedical\s*(device|center|clinic|service)\b",
        r"\bhospital\b",
        r"\bpharmacy\b",
        r"\bchronic\s*(condition|disease|pain|illness)\b",
        r"\ballerg\w+\b(?=.*treatment|.*relief|.*medication|.*shot)",
        r"\basthma\b",
        r"\bdermatolog\w+\b",
        r"\boptometr\w+\b", r"\bvision\s*(care|center|clinic)\b",
        r"\bdental\s*(care|clinic|implant)\b",
        r"\bnursing\b(?=.*career|.*school|.*program|.*jobs?)",
        r"\bmental\s*health\s*(therapy|treatment|clinic|disorder)\b",
    ],

    # --- nonprofit_charity ---
    "nonprofit_charity": [
        r"\bdonate\s*(now|today|to|here)\b",
        r"\byour\s*donation\b",
        r"\bfundraising\b",
        r"\bcharity\b",
        r"\bnonprofit\b", r"\bnon[\s-]profit\b",
        r"\bngo\b",
        r"\brelief\s*(fund|effort|mission|aid)\b",
        r"\bhumanitarian\b",
        r"\baid\s*(mission|organization|workers?)\b",
        r"\bhelp\s*(children|families|refugees|the\s*poor|the\s*hungry)\b",
        r"\bfeed\s*(the\s*hungry|families|children|people)\b",
        r"\bclean\s*water\s*(project|access|mission)\b",
        r"\borphan\b",
        r"\brefugee[s]?\b",
        r"\bsadaqah\b", r"\bzakat\b", r"\bwaqf\b",
        r"\bummah\b",
        r"\bblood\s*donation\b", r"\bdonate\s*blood\b",
        r"\bstem\s*cell\s*donation\b",
        r"\breligious\s*(giving|charity|aid)\b",
        r"\bspiritual\s*(giving|donation|campaign)\b",
        r"\bscholarship\s*(fund|program)\b(?=.*charity|.*donate|.*give)",
        r"\bsave\s*(lives|children|families|the\s*world)\b",
        r"\bevery\s*dollar\s*(counts|helps|makes)\b",
        r"\btax[\s-]*deductible\s*donation\b",
    ],

    # --- media_entertainment ---
    "media_entertainment": [
        r"\bwatch\s*(now|online|free|anywhere)\b",
        r"\bstream\s*(now|online|free|anywhere)\b",
        r"\bstreaming\s*(service|platform|subscription)\b",
        r"\bnew\s*(episode[s]?|season|series|movie|film)\b",
        r"\bnow\s*streaming\b",
        r"\boriginal\s*(series|content|film|show)\b",
        r"\bexclusive\s*(content|series|footage)\b",
        r"\bticket[s]?\s*(on\s*sale|available|book)\b",
        r"\blive\s*(event|show|concert|performance|stream)\b",
        r"\bconcert\s*(tour|tickets?|venue)\b",
        r"\bfestival\s*(lineup|tickets?|pass)\b",
        r"\bpodcast\b",
        r"\bmovie\s*(trailer|tickets?|premiere)\b",
        r"\bfilm\s*(festival|premiere|screening)\b",
        r"\bmusic\s*(video|release|album|playlist)\b",
        r"\bnew\s*album\b", r"\bsingle\s*out\s*now\b",
        r"\blistening\s*(party|session)\b",
        r"\bvideo[\s-]on[\s-]demand\b",
        r"\bvod\b",
        r"\bsubscription\s*(plan|box|service)\b(?=.*watch|.*stream|.*listen)",
        r"\bbinge[\s-]watch\b",
        r"\bseries\s*finale\b",
        r"\bdocumentar\w+\b",
    ],

    # --- fashion_beauty ---
    "fashion_beauty": [
        r"\bskincare\b",
        r"\bmoisturizer\b",
        r"\bserum\b",
        r"\bspf\b", r"\bsunscreen\b",
        r"\banti[\s-]aging\b",
        r"\bwrinkle\b",
        r"\bfoundation\b(?=.*makeup|.*shade|.*skin)",
        r"\bmascara\b", r"\beyeliner\b", r"\blipstick\b",
        r"\bcontour\b(?=.*makeup|.*palette|.*kit)",
        r"\bperfume\b", r"\bfragrance\b",
        r"\bhair\s*(care|color|growth|loss|extension|wig)\b",
        r"\bwig[s]?\b",
        r"\bcollection\b(?=.*fashion|.*clothing|.*dress|.*style|.*wear)",
        r"\boutfit\b",
        r"\bdress\b(?=.*new|.*style|.*wear|.*summer|.*winter)",
        r"\bjewelry\b", r"\bjewellery\b",
        r"\baccesories?\b",
        r"\bhandba[gy]\b",
        r"\bsneaker[s]?\b",
        r"\bheels?\b(?=.*shoes?|.*style|.*fashion)",
        r"\bboutique\b",
        r"\bnew\s*season\b(?=.*fashion|.*collection|.*style)",
        r"\bstyle\s*(guide|tip[s]?|refresh)\b",
        r"\blook\s*(book|good|great|amazing)\b",
        r"\bbeauty\s*(routine|tip[s]?|hack[s]?|product[s]?)\b",
        r"\bnaturals?\s*(makeup|skincare|beauty|hair)\b",
        r"\bcruelty[\s-]free\b",
        r"\bvegan\s*(beauty|makeup|skincare)\b",
        r"\bsustainable\s*(fashion|clothing|wear)\b",
    ],

    # --- travel_hospitality ---
    "travel_hospitality": [
        r"\bbook\s*(now|your|a\s*trip|a\s*flight|a\s*hotel)\b",
        r"\btravel\b(?=.*deal|.*tip|.*package|.*destination|.*plan)",
        r"\bflight[s]?\b(?=.*book|.*deal|.*cheap|.*search|.*find)",
        r"\bhotel[s]?\b(?=.*book|.*deal|.*stay|.*resort|.*night)",
        r"\bresort[s]?\b",
        r"\bvacation\s*(deal[s]?|package[s]?|planner)\b",
        r"\bholiday\s*(deal[s]?|package[s]?|destination)\b",
        r"\btour\s*(package[s]?|operator|guide)\b",
        r"\bdestination[s]?\b",
        r"\bitinerary\b",
        r"\bcruise\b",
        r"\bvisas?\b(?=.*travel|.*application|.*process)",
        r"\bpassport\b(?=.*apply|.*renew|.*visa)",
        r"\besim\b(?=.*travel|.*data|.*international|.*roaming)",
        r"\bdata\s*plan\b(?=.*travel|.*international|.*roaming)",
        r"\broaming\b",
        r"\bloyalty\s*(points?|program|reward[s]?)\b(?=.*hotel|.*flight|.*travel)",
        r"\bfrequent\s*flyer\b",
        r"\bairbnb\b",
        r"\broad\s*trip\b",
        r"\bbucket\s*list\b(?=.*travel|.*destination|.*trip)",
        r"\bcabin\b(?=.*crew|.*rent|.*stay|.*retreat)",
        r"\bhostel\b",
        r"\bexplore\b(?=.*destination|.*world|.*country|.*city)",
        r"\badventure\s*(travel|tour|trip|holiday)\b",
    ],

    # --- food_beverage ---
    "food_beverage": [
        r"\brecipe[s]?\b",
        r"\bmeal\s*(plan|prep|delivery|kit)\b",
        r"\bfood\s*(delivery|order|truck)\b",
        r"\border\s*(food|pizza|takeout|takeaway)\b",
        r"\brestaurant\b",
        r"\bcaf[eé]\b",
        r"\bbarista\b",
        r"\bcraft\s*beer\b", r"\bbeer\s*(tasting|flight|flight)\b",
        r"\bwine\s*(tasting|pairing|delivery|shop)\b",
        r"\bwhiskey\b", r"\bwhisky\b", r"\bspirits\b(?=.*drink|.*bar|.*cocktail)",
        r"\bcocktail[s]?\b",
        r"\bvegan\s*(recipe|food|meal|diet)\b",
        r"\bplant[\s-]based\s*(diet|meal|food|protein)\b",
        r"\bgluten[\s-]free\b(?=.*food|.*recipe|.*meal|.*bak)",
        r"\bketo\b(?=.*recipe|.*food|.*meal|.*diet)",
        r"\bsnack[s]?\b(?=.*brand|.*healthy|.*organic|.*natural)",
        r"\bgrocery\s*(delivery|shop|store)\b",
        r"\bfresh\s*(produce|ingredient[s]?|food)\b",
        r"\borganic\s*(food|grocery|produce|farm)\b",
        r"\bsupermarket\b", r"\bgrocery\b",
        r"\bcpg\b",
        r"\bdrink\s*(brand|mix|recipe)\b",
        r"\bbeverage[s]?\b",
        r"\benergy\s*drink[s]?\b",
        r"\bpet\s*(food|nutrition|diet)\b(?=.*dog|.*cat|.*animal)",
    ],

    # --- education ---
    "education": [
        r"\benroll\s*(now|today|free)\b",
        r"\blearn\s*(online|free|to\s*code|a\s*language|from\s*home)\b",
        r"\bonline\s*(course[s]?|class\w*|learning|degree|certificate)\b",
        r"\bcertificate\s*(program|course|of\s*completion)\b",
        r"\bdegree\s*(program|online|bachelor|master)\b",
        r"\bscholarship\b(?=.*apply|.*eligible|.*program)",
        r"\btuition\b",
        r"\buniversity\s*(degree|application|campus|program)\b",
        r"\bcollege\s*(degree|application|campus|admission)\b",
        r"\badmission[s]?\b",
        r"\bacademic\b",
        r"\bcurriculum\b",
        r"\bsyllabus\b",
        r"\binstructor\b", r"\bmentor\b(?=.*learn|.*course|.*program)",
        r"\btutor\b",
        r"\bstudent[s]?\b(?=.*learn|.*enroll|.*course|.*program)",
        r"\bstudying\b",
        r"\bcoding\s*(bootcamp|class|course)\b",
        r"\bprogramming\s*(course|lesson|skill)\b",
        r"\bdata\s*science\s*(course|program|bootcamp)\b",
        r"\bux\s*(design|research|course|certification)\b",
        r"\blanguage\s*(learning|course|app|lesson)\b",
        r"\bskill\s*(development|building|up|learn)\b",
        r"\bprofessional\s*(development|certification|training)\b",
        r"\beLearning\b", r"\be[\s-]learning\b",
    ],

    # --- real_estate_home ---
    "real_estate_home": [
        r"\breal\s*estate\b",
        r"\bhomes?\s*for\s*(sale|rent)\b",
        r"\bproperty\s*(for\s*sale|listing[s]?|search|investment)\b",
        r"\bbuy\s*(a\s*home|your\s*home|property)\b",
        r"\bmortgage\s*(rate|pre[\s-]approval|refinance)\b",
        r"\bdown\s*payment\b",
        r"\bhome\s*(improvement|renovation|remodel|decor|design)\b",
        r"\binterior\s*(design|decor|decorator)\b",
        r"\bfurniture\b",
        r"\bsofa\b", r"\bcouch\b",
        r"\bbedroom\s*(furniture|set|decor)\b",
        r"\bmattress\b",
        r"\bsleep\s*(system|solution|mattress)\b",
        r"\bbathroom\s*(renovation|design|vanity)\b",
        r"\bkitchen\s*(renovation|design|cabinet)\b",
        r"\bhardwood\s*floor\b",
        r"\btile[s]?\b(?=.*floor|.*bathroom|.*kitchen|.*install)",
        r"\brental\s*(property|income|apartment|unit)\b",
        r"\bapartment\s*(for\s*rent|listing|search|finder)\b",
        r"\bcondo\s*(for\s*sale|listing|buy)\b",
        r"\bnew\s*(home|development|build|construction)\b",
        r"\bmodular\s*(home|house|build)\b",
        r"\bstorage\s*(unit|facility|solution)\b",
        r"\blandscaping\b",
        r"\bpool\s*(installation|maintenance|contractor)\b",
        r"\bhome\s*warranty\b",
        r"\bwood\s*(house|cabin|log\s*home)\b",
    ],

    # --- automotive ---
    "automotive": [
        r"\btest\s*drive\b",
        r"\bcar\s*(deal[s]?|lease|loan|payment|buy|sell|search)\b",
        r"\bnew\s*(car|truck|suv|vehicle)\b",
        r"\bused\s*(car[s]?|truck[s]?|vehicle[s]?)\b",
        r"\belectric\s*vehicle\b", r"\bev\b(?=.*car|.*truck|.*charge|.*range)",
        r"\belectric\s*(car|truck|motorcycle|motorbike)\b",
        r"\bhybrid\s*(car|vehicle|suv)\b",
        r"\bcar\s*rental\b",
        r"\bauto\s*(insurance|loan|lease|repair|part[s]?)\b",
        r"\bvehicle\s*(insurance|loan|wrap|wrap[s]?|history)\b",
        r"\bmpg\b", r"\bfuel\s*(efficiency|economy|saving)\b",
        r"\bhorsepower\b",
        r"\b0\s*to\s*60\b",
        r"\boff[\s-]road\b",
        r"\btow\s*(capacity|rating|truck)\b",
        r"\btruck\s*(bed|payload|capacity|accessory)\b",
        r"\bwiper\s*(blade[s]?|replacement)\b",
        r"\btire[s]?\b(?=.*change|.*rotate|.*buy|.*replace)",
        r"\bwheel[s]?\b(?=.*rim[s]?|.*alloy|.*upgrade|.*custom)",
        r"\bengine\s*(repair|upgrade|swap|rebuild)\b",
        r"\bcar\s*detailing\b",
        r"\bvin\s*(report|check|history)\b",
        r"\bcar\s*history\b",
        r"\bvehicle\s*history\b",
        r"\bcertified\s*pre[\s-]owned\b",
        r"\bmotorcycle[s]?\b",
        r"\bcharg(e|ing)\s*(station|point|network)\b(?=.*ev|.*electric|.*vehicle)",
    ],

    # --- b2b_marketing ---
    "b2b_marketing": [
        r"\bb2b\b",
        r"\blead\s*generation\b",
        r"\bdemand\s*generation\b",
        r"\binbound\s*marketing\b",
        r"\bsales\s*(funnel|pipeline|enablement)\b",
        r"\bcrm\b",
        r"\bmarketing\s*(automation|platform|strategy|software)\b",
        r"\bseo\s*(strategy|service|agency|tool)\b",
        r"\bppc\b", r"\bpaid\s*(search|media|advertising)\b",
        r"\bcontent\s*marketing\b",
        r"\baffiliate\s*(marketing|program|network)\b",
        r"\binfluencer\s*marketing\b",
        r"\bgrowth\s*(hacking|strategy|marketing)\b",
        r"\bdigital\s*(marketing|agency|strategy)\b",
        r"\bhr\s*(software|platform|solution|tool)\b",
        r"\brecruit\s*(platform|software|service)\b",
        r"\btalent\s*(acquisition|management|sourcing)\b",
        r"\bpayroll\s*(software|solution|service)\b",
        r"\baccounting\s*(software|solution|service)\b",
        r"\binvoicing\s*(software|tool|solution)\b",
        r"\bbusiness\s*(intelligence|analytics|dashboard)\b",
        r"\bcloud\s*(infrastructure|hosting|server|solution)\b",
        r"\bdevops\b",
        r"\bci/cd\b", r"\bcontinuous\s*integration\b",
        r"\bapi\s*(management|gateway|platform)\b",
        r"\benterprise\s*(solution|software|plan)\b",
        r"\bwhite[\s-]label\b",
        r"\bsaas\s*(platform|product|company)\b",
        r"\bstartup\s*(tool|resource|community)\b",
        r"\bdeveloper\s*(community|tool|conference)\b",
    ],

    # --- gambling_betting ---
    "gambling_betting": [
        r"\bbet\s*(now|online|today|\$|\d+)\b",
        r"\bsports\s*betting\b",
        r"\bsportsbook\b",
        r"\bodds\b(?=.*bet|.*win|.*sport)",
        r"\bcasino\b",
        r"\bonline\s*casino\b",
        r"\bslot[s]?\s*(machine[s]?|game[s]?)?\b",
        r"\bpoker\b",
        r"\broulette\b",
        r"\bblackjack\b",
        r"\bfantasy\s*(sports|football|basketball|baseball)\b",
        r"\bdaily\s*fantasy\b",
        r"\bfree\s*(spins?|chips?|coins?|play)\b(?=.*casino|.*slot|.*game)",
        r"\bbonus\s*(offer|code|chip)\b(?=.*casino|.*bet|.*gambling)",
        r"\bsweepstakes\b",
        r"\bjackpot\b",
        r"\bwager\b",
        r"\bgambl\w+\b",
        r"\blottery\b",
        r"\braffle\b",
        r"\bprize\s*(pool|draw|giveaway)\b(?=.*win|.*enter|.*lucky)",
        r"\bwin\s*big\b",
        r"\bplay\s*(and\s*win|to\s*win|for\s*real|for\s*money)\b",
        r"\bcash\s*prize[s]?\b",
        r"\b18\+\b", r"\b21\+\b",
        r"\bgamble\s*responsibly\b",
        r"\bbegambleaware\b",
    ],

    # --- news_politics ---
    "news_politics": [
        r"\bbreaking\s*news\b",
        r"\blatest\s*news\b",
        r"\btop\s*stories\b",
        r"\bnews\s*(alert|update|brief)\b",
        r"\bheadlines?\b(?=.*today|.*daily|.*news)",
        r"\bdaily\s*(news|brief|digest)\b",
        r"\bjournalism\b",
        r"\binvestigative\s*report\b",
        r"\bpolitical\s*(commentary|analysis|news|opinion)\b",
        r"\belection\b",
        r"\bvote\b(?=.*register|.*now|.*today|.*election)",
        r"\bregister\s*to\s*vote\b",
        r"\bpolitical\s*(party|candidate|rally)\b",
        r"\bcampaign\s*(ad|political|election)\b",
        r"\badvocacy\b",
        r"\bopinion\s*(piece|column|editorial)\b",
        r"\beditorial\b",
        r"\bsubscribe\s*(to\s*our\s*newsletter|for\s*updates)\b(?=.*news|.*politics|.*journal)",
        r"\bnewsletter\b(?=.*politics|.*news|.*analysis|.*daily)",
        r"\bpublic\s*policy\b",
        r"\bcivic\s*(engagement|action|education)\b",
        r"\bfact[\s-]check\b",
        r"\bopposition\b(?=.*party|.*leader|.*candidate)",
        r"\bdemocracy\b(?=.*vote|.*rights|.*defend|.*protect)",
    ],
}

# ---------------------------------------------------------------------------
# Pre-compile all regex patterns at module load time
# ---------------------------------------------------------------------------

_NAME_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    vertical: [re.compile(p, re.IGNORECASE) for p in patterns]
    for vertical, patterns in _NAME_PATTERN_DEFS.items()
}

_COPY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    vertical: [re.compile(p, re.IGNORECASE) for p in patterns]
    for vertical, patterns in _COPY_PATTERN_DEFS.items()
}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _extract_hostname(url: str) -> str:
    """Extract hostname from a potentially malformed URL.

    Falls back to returning the raw string if urlparse cannot identify a
    netloc, so substring searches still work on truncated hostnames like
    "opify.com" (truncated "shopify.com").
    """
    if not url:
        return ""
    raw = url.strip()
    # urlparse needs a scheme to extract netloc reliably
    raw_with_scheme = "https://" + raw if "://" not in raw else raw
    try:
        parsed = urlparse(raw_with_scheme)
        hostname = parsed.hostname or ""
        # Return both the parsed hostname and the original string so
        # callers can do substring checks against either.
        return hostname
    except ValueError:
        return ""


def _match_domain(url: str, fragments: list[str]) -> bool:
    """Return True if any fragment matches the URL's hostname on a domain boundary.

    A fragment matches when:
      - the hostname equals the fragment exactly, OR
      - the hostname ends with ``"." + fragment`` (i.e. fragment is a proper
        parent-domain suffix), OR
      - the fragment contains a ``/`` (path-qualified) AND appears as a
        substring of the full URL — used for a handful of subpath rules
        like ``"amazon.com/primevideo"``.

    This is strict enough to avoid false positives like the 5-char fragment
    ``"ro.co"`` (Ro telehealth) accidentally matching inside
    ``smartgyro.com``, while still catching the truncated hostnames in our
    corpus (``"opify.com"`` → shopify, ``"emu.com"`` → temu) because those
    truncations are listed as their own fragments.
    """
    if not url:
        return False
    hostname = _extract_hostname(url).lower()
    url_lower = url.lower()
    for fragment in fragments:
        frag = fragment.lower()
        if "/" in frag:
            # Path-qualified rule — fall back to URL-string substring match.
            if frag in url_lower:
                return True
            continue
        if not hostname:
            continue
        if hostname == frag or hostname.endswith("." + frag):
            return True
    return False


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Level 4 — Semantic embedding fallback
# ---------------------------------------------------------------------------

# Rich anchor text per vertical. These are the "prototype" texts that the
# model embeds to represent each vertical. Richer = better separation.
_VERTICAL_ANCHORS: dict[str, str] = {
    "ecommerce": (
        "Online retail store selling physical products with free shipping. "
        "Shop now, add to cart, buy online, order today, flash sale, discount code, "
        "limited time offer, product launch, new collection, returns policy, "
        "e-commerce marketplace, direct-to-consumer brand, product reviews."
    ),
    "saas_software": (
        "Software as a service, productivity app, cloud platform, free trial, "
        "monthly subscription, per seat pricing, workflow automation, no-code tool, "
        "API integration, project management, team collaboration, website builder, "
        "drag and drop editor, dashboard, developer tools, SaaS product."
    ),
    "gaming": (
        "Video game, mobile game, online multiplayer, play now, download free, "
        "level up, in-game items, raid, quest, hero, battle, guild, esports, "
        "RPG, strategy game, game keys, Steam, console gaming, PC gaming, "
        "game update, tournament, streamer, loot box."
    ),
    "finance": (
        "Investment platform, stock trading, forex trading, portfolio management, "
        "savings account, high yield, interest rate, personal loan, mortgage, "
        "insurance plan, retirement account, 401k, financial advisor, credit card, "
        "banking app, money transfer, prop trading, funded account."
    ),
    "crypto_web3": (
        "Cryptocurrency exchange, Bitcoin, Ethereum, blockchain, DeFi, NFT, "
        "Web3, token launch, crypto wallet, decentralized finance, smart contract, "
        "staking rewards, yield farming, liquidity pool, altcoin, metaverse, "
        "crypto trading, token mint, DAO governance."
    ),
    "health_wellness": (
        "Fitness app, weight loss, workout routine, gym membership, mental health, "
        "meditation, mindfulness, sleep tracker, supplements, protein shake, "
        "personal trainer, yoga, wellness coaching, health tracking, gut health, "
        "hormone balance, body transformation, wellness journey."
    ),
    "pharma_medical": (
        "Prescription medication, FDA approved drug, clinical trial, side effects, "
        "dosage, treatment options, chronic condition, urgent care, medical device, "
        "pharmacy, doctor consultation, specialist referral, surgery, rehabilitation, "
        "dermatology, optometry, dental care, nursing career."
    ),
    "nonprofit_charity": (
        "Donate now, fundraising, charity, nonprofit, humanitarian aid, relief fund, "
        "help families in need, clean water project, orphan support, refugees, "
        "zakat, sadaqah, blood donation, save lives, every dollar counts, "
        "tax-deductible donation, religious giving, foundation."
    ),
    "media_entertainment": (
        "Streaming service, watch now, new episode, new season, original series, "
        "exclusive content, movie trailer, concert tickets, live event, festival, "
        "podcast, music video, album release, subscribe to watch, binge watch, "
        "documentary, box office, cinema, video on demand."
    ),
    "fashion_beauty": (
        "Skincare routine, moisturizer, serum, SPF, anti-aging, foundation, "
        "mascara, lipstick, perfume, hair care, wig, fashion collection, outfit, "
        "jewelry, accessories, sneakers, new season, beauty tips, cruelty-free, "
        "vegan beauty, sustainable fashion, style guide."
    ),
    "travel_hospitality": (
        "Book flights, hotel deals, vacation package, resort stay, travel deals, "
        "holiday destination, tour operator, cruise, eSIM for travel, international "
        "roaming, loyalty points, frequent flyer, road trip, adventure travel, "
        "airbnb, hostel, itinerary, passport, visa application."
    ),
    "food_beverage": (
        "Recipe, meal planning, food delivery, restaurant, café, craft beer, "
        "wine tasting, cocktails, vegan recipe, plant-based diet, gluten free, "
        "grocery delivery, fresh ingredients, organic food, beverage brand, "
        "meal kit, pet food, nutrition, cooking."
    ),
    "education": (
        "Online course, learn coding, enroll now, certificate program, university "
        "degree, scholarship, tuition, online learning, bootcamp, programming course, "
        "data science, language learning, tutoring, skill development, eLearning, "
        "academic curriculum, student enrollment."
    ),
    "real_estate_home": (
        "Homes for sale, property listing, buy a home, mortgage rate, down payment, "
        "home improvement, interior design, furniture, mattress, sleep system, "
        "bathroom renovation, kitchen remodel, rental property, apartment for rent, "
        "new development, storage unit, landscaping, pool installation."
    ),
    "automotive": (
        "New car, test drive, car loan, car lease, electric vehicle, hybrid SUV, "
        "car rental, auto insurance, vehicle history, fuel efficiency, horsepower, "
        "off-road, truck accessories, wiper blades, tire change, motorcycle, "
        "EV charging station, certified pre-owned."
    ),
    "b2b_marketing": (
        "B2B software, lead generation, sales pipeline, CRM, marketing automation, "
        "SEO strategy, content marketing, affiliate marketing, digital agency, "
        "HR software, recruiting platform, payroll solution, accounting software, "
        "cloud infrastructure, DevOps, enterprise software, developer community."
    ),
    "gambling_betting": (
        "Sports betting, sportsbook, casino, online slots, poker, roulette, "
        "blackjack, fantasy sports, bet now, odds, jackpot, free spins, bonus offer, "
        "sweepstakes, wager, lottery, raffle, win big, cash prize, 18+."
    ),
    "news_politics": (
        "Breaking news, latest headlines, journalism, political commentary, election, "
        "register to vote, political party, campaign, advocacy, opinion column, "
        "editorial, subscribe to newsletter, investigative report, public policy, "
        "civic engagement, fact check, democracy."
    ),
}

_SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_SEMANTIC_THRESHOLD = 0.28  # min cosine similarity to assign a vertical


class _SemanticClassifier:
    """Lazy-loaded sentence-transformer classifier for vertical inference.

    The model and anchor embeddings are computed once on first use and
    cached in memory for the lifetime of the process.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._anchor_matrix: np.ndarray | None = None  # shape (n_verticals, dim)
        self._anchor_verticals: list[str] = []

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            logger.info("Loading sentence-transformer model '%s' …", _SEMANTIC_MODEL_NAME)
            model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
            anchors = list(_VERTICAL_ANCHORS.items())
            self._anchor_verticals = [v for v, _ in anchors]
            texts = [t for _, t in anchors]
            embeddings: np.ndarray = model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self._anchor_matrix = embeddings
            self._model = model
            logger.info("Semantic classifier ready (%d verticals).", len(anchors))
        except Exception as exc:  # pragma: no cover
            logger.warning("Semantic classifier unavailable: %s", exc)
            self._model = None

    def classify(self, text: str, threshold: float = _SEMANTIC_THRESHOLD) -> str:
        """Return the closest vertical if cosine similarity exceeds *threshold*."""
        self._ensure_loaded()
        if self._model is None or self._anchor_matrix is None:
            return "unknown"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            model: SentenceTransformer = self._model  # type: ignore[assignment]
            vec: np.ndarray = model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )
            sims: np.ndarray = self._anchor_matrix @ vec[0]
            best_idx: int = int(np.argmax(sims))
            best_score: float = float(sims[best_idx])
            if best_score >= threshold:
                return self._anchor_verticals[best_idx]
        except Exception as exc:  # pragma: no cover
            logger.debug("Semantic classification error: %s", exc)
        return "unknown"


# Module-level singleton — allocated lazily on first semantic call.
_semantic_clf: _SemanticClassifier | None = None


@cache
def _get_semantic_clf() -> _SemanticClassifier:
    """Return the process-level singleton, creating it if needed."""
    return _SemanticClassifier()


def classify_ad(
    advertiser_name: str,
    landing_page_url: str,
    headline: str,
    body: str,
    *,
    use_semantic: bool = True,
    semantic_threshold: float = _SEMANTIC_THRESHOLD,
) -> str:
    """Classify an ad into one of the 18 business verticals (or 'unknown').

    Four-level priority:
      1. Domain substring match against landing_page_url
      2. Regex match against lowercased advertiser_name
      3. Regex match against lowercased headline + body
      4. Semantic cosine similarity via local sentence-transformer (fallback)

    Args:
        advertiser_name: The ad's advertiser name.
        landing_page_url: The ad's destination URL (may be malformed).
        headline: Ad headline copy.
        body: Ad body copy.
        use_semantic: Whether to run the semantic fallback (Level 4) when
            levels 1-3 return "unknown". Defaults to True. Set False for
            fast batch runs where accuracy on edge cases is less critical.
        semantic_threshold: Minimum cosine similarity for semantic assignment.

    Returns:
        The matched vertical string, or "unknown".
    """
    # Level 1: domain match
    for vertical in PRIORITY_ORDER:
        fragments = _DOMAIN_RULES.get(vertical, [])
        if fragments and _match_domain(landing_page_url, fragments):
            return vertical

    # Level 2: advertiser name match
    name_lower = advertiser_name.lower()
    for vertical in PRIORITY_ORDER:
        patterns = _NAME_PATTERNS.get(vertical, [])
        for pat in patterns:
            if pat.search(name_lower):
                return vertical

    # Level 3: ad copy match
    copy_lower = (headline + " " + body).lower()
    for vertical in PRIORITY_ORDER:
        patterns = _COPY_PATTERNS.get(vertical, [])
        for pat in patterns:
            if pat.search(copy_lower):
                return vertical

    # Level 4: semantic fallback
    if use_semantic:
        text = f"{advertiser_name} {headline} {body}".strip()
        if text:
            return _get_semantic_clf().classify(text, threshold=semantic_threshold)

    return "unknown"


def infer_vertical(ad: ScoredAd, *, use_semantic: bool = True) -> str:
    """Infer the true business vertical for a ScoredAd.

    Thin wrapper over :func:`classify_ad` that unpacks the relevant fields
    from the nested ``ScoredAd.ad`` (``RawAd``) object.

    Args:
        ad: A scored advertisement.
        use_semantic: Passed through to :func:`classify_ad`. Set False to
            skip the embedding model and run keyword-only.

    Returns:
        The inferred vertical string (one of :data:`VERTICAL_LABELS`).
    """
    return classify_ad(
        advertiser_name=ad.ad.advertiser_name,
        landing_page_url=ad.ad.landing_page_url,
        headline=ad.ad.ad_copy.headline,
        body=ad.ad.ad_copy.body,
        use_semantic=use_semantic,
    )
