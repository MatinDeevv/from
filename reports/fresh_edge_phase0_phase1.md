# Fresh XAU Edge Discovery - Phase 0 and Phase 1

## Phase 0 - Data Documentation

This fresh run uses only deterministic mechanism-family symbols, not the old full 50-symbol fishing universe.

Source files:
- Yahoo leader bars from `C:\Users\marti\from\data\cache\yahoo`
- XAU execution bars from `C:\Users\marti\Documents\Quant Lab\Data\data\MASTER_DATASET_FILLED_PASS2.parquet`
- Forward XAU holdout from `C:\Users\marti\from\data\derived\xauusd_dukascopy_5m_forward_20260522_20260618.csv`

Data profile files:
- `C:\Users\marti\from\reports\fresh_edge_5m_mechanism_data_profile.csv`
- `C:\Users\marti\from\reports\fresh_edge_60m_mechanism_data_profile.csv`

Gap rule: missing bars are counted as null close values or duplicate timestamps inside the vendor-provided bar stream. Normal exchange closures are not counted as missing bars. Under that rule, all 16 planned symbols have 0.0% missing/duplicate bars in both 5m and 60m cache files. This does not prove a true pre-forward availability snapshot; it only makes the local data assumption explicit.

Universe construction rule: include only symbols named by the six predeclared mechanism families below. This rule is deterministic and could be written before the forward window: energy commodities, US risk sectors, silver, long-duration rates, safe-haven FX, and crypto liquidity/risk impulse.

Symbols:
- Energy / commodity complex: `UNG`, `USO`, `CL=F`
- US sector risk appetite: `XLF`, `XLV`, `XLY`, `XLK`
- Silver leads gold: `SI=F`
- Long-duration rates: `TLT`, `IEF`, `ZB=F`
- FX safe-haven divergence: `USDJPY=X`, `DX-Y.NYB`, `EURUSD=X`
- New mechanism, crypto liquidity/risk impulse: `BTC-USD`, `ETH-USD`

Hard-coded split dates:
- 5m train: `2026-03-24 08:00:00+00:00` to `2026-04-23 23:59:59+00:00`
- 5m validation: `2026-04-24 00:00:00+00:00` to `2026-05-08 23:59:59+00:00`
- 5m old test: `2026-05-09 00:00:00+00:00` to `2026-05-22 13:50:00+00:00`
- 5m forward: `2026-05-22 13:55:00+00:00` to `2026-06-17 23:55:00+00:00`
- 60m train: `2023-07-21 08:00:00+00:00` to `2025-07-01 00:00:00+00:00`
- 60m validation: `2025-07-01 00:00:00+00:00` to `2026-01-01 00:00:00+00:00`
- 60m old test: `2026-01-01 00:00:00+00:00` to `2026-05-22 12:59:59+00:00`
- 60m forward: `2026-05-22 13:00:00+00:00` to `2026-06-17 23:00:00+00:00`

Known limitation: the 5m forward window remains short and was already used to design stricter gates. Any survivor is in-sample under the new gate until a future untouched window or longer reconstructed leader history is available.

## Phase 1 - Mechanism-First Hypotheses

### 1. Energy / Commodity Complex Leads Gold

Lead-lag story: energy ETFs and crude futures can reprice global inflation/liquidity shocks before XAU fully reacts, especially around London hours when global commodity desks are active. If commodity vol-adjusted momentum is strong, gold may follow the broader commodity impulse.

Expected regime: inflation scare, commodity squeeze, or liquidity impulse. Expected strongest in London. It should be weaker in quiet risk-neutral sessions.

Expected sign: follow. A positive vol-adjusted energy move should imply positive XAU drift over the next several 5m bars; a negative energy move should imply negative XAU drift.

Falsification: forward PnL positive only in one late subperiod, bootstrap lower bound <= 0, or reverse-rule PnL not materially negative.

### 2. US Sector Risk-Appetite Fades Gold

Lead-lag story: US cyclicals/defensives can reflect equity-risk appetite that competes with defensive gold demand. When sectors surge on risk appetite, XAU can lag then fade; when sectors dump, XAU can catch a safe-haven bid.

Expected regime: equity risk-on/risk-off transitions, strongest in London/NY overlap or US hours. Sector-specific versions may differ: financials and consumer discretionary should be cleaner risk appetite proxies than health care.

Expected sign: fade. Positive sector vol-adjusted impulse should predict lower/lagging XAU; negative sector impulse should predict higher XAU.

Falsification: follow beats fade across the forward window, or fade only works because of one subperiod/concentrated event.

### 3. Silver Leads Gold

Lead-lag story: silver and gold share precious-metal drivers, but silver can move faster during commodity/risk liquidity impulses. CME silver momentum can lead short-horizon XAU repricing.

Expected regime: broad metals movement, inflation repricing, commodity liquidity shocks. All sessions are plausible because futures trade nearly around the clock.

Expected sign: follow. Strong SI=F volz should predict same-direction XAU movement.

Falsification: SI=F follow has fewer than 100 forward trades, bootstrap lower bound <= 0, or fade performs symmetrically well.

### 4. Long-Duration Rates Fade Gold

Lead-lag story: duration instruments react to real-rate and growth expectations. A bond rally can initially coexist with risk-off flows, but if the prior strict screen finds fade, the possible mechanism is short-term overreaction in duration relative to gold or rate-driven USD/liquidity repricing that gold absorbs inversely after a lag.

Expected regime: macro/rates repricing, not single-stock risk. All sessions are plausible for futures; ETF versions should be more US-session sensitive.

Expected sign: fade for TLT/IEF/ZB=F, matching the strict-gate survivor prior.

Falsification: effective signal requires fewer than 100 forward trades, only works in ETF hours with no futures confirmation, or follow and fade are both profitable.

### 5. FX Safe-Haven Divergence

Lead-lag story: USD and JPY moves can precede gold when safe-haven or dollar-liquidity flows hit FX first. DXY strength often pressures gold, while USDJPY can capture dollar/risk and yen safe-haven behavior. EURUSD is an inverse dollar proxy.

Expected regime: dollar liquidity, safe-haven FX shocks, macro data windows. London and NY should matter most.

Expected sign: DXY follow for USD strength should often imply XAU fade/down; EURUSD strength should often imply XAU follow/up if dollar weakness drives gold. USDJPY is ambiguous, so both follow and fade are tested.

Falsification: no direction has stable subperiods, or latency/reverse behavior suggests the signal is just timestamp alignment noise.

### 6. Crypto Liquidity / Risk Impulse

Lead-lag story: BTC/ETH trade continuously and can react quickly to global liquidity/risk impulses outside ETF market hours. If gold is repricing the same liquidity shock more slowly, crypto vol-adjusted momentum or cross-asset z-spread could lead XAU.

Expected regime: broad liquidity/risk-on shocks, weekend-to-Asia handoff, and dollar-liquidity moves. This is the new family because the prior strict survivors were concentrated in commodities/sectors/rates, leaving portfolio independence weak.

Expected sign: test both follow and fade, but the primary story is follow for liquidity risk-on and possible fade during stress deleveraging.

Falsification: high forward PnL but failed bootstrap/subperiod stability, or no improvement from the new momentum-ratio / EW z-spread features.
