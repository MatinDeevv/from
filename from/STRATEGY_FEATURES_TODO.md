# 50 Trading-Logic Features — Progress + TODO

Goal: add 50 opt-in trading-logic knobs (sizing / risk / entry-filter / exit / regime)
on top of the model signal. All default OFF → today's walk-forward numbers unchanged
until a knob is enabled. Plan file: `~/.claude/plans/iridescent-prancing-dragonfly.md`.

## Architecture
One shared inline header drives BOTH sim sites (same pattern as `wf_metrics.hpp`) so the
honest walk-forward path and the quick backtest can't diverge.

- `include/strategy/strategy.hpp` — `StrategyConfig` (50 knobs + `from(cfg,args)`),
  `StrategyState` (equity/dd/streaks/cooldown/day/kelly), `EntryContext`, `decide()`,
  `on_close()`.
- Decision is CAUSAL: reads only entry-time inputs (probs, entry-tick features, time,
  state). Realized `ret` used only after take, shaped by fixed-at-entry exit clamps.

## The 50 (groups)
- A. Direction & gating (1-10): conf_gate, margin_gate, neutral_mass_max, prob_floor,
  long_only, short_only, invert_signal, signal_ema_confirm, agreement_run, flip_block.
- B. Sizing (11-22): fixed, conf_weighted, frac_kelly, vol_target, inv_spread, size_cap,
  size_floor, equity_compound, dd_derisk, anti_martingale, martingale, recovery_size.
- C. Entry-context filters (23-32): max_spread, min_tick_rate, vol_min, vol_max,
  trend_filter, trend_block, min_expected_edge, cost_skip_mult, session_mask, env_mask.
- D. Exit/barrier (33-40): barrier_tp_mult, barrier_sl_mult, barrier_cost_mult, tp_clamp,
  sl_clamp, net_loss_cap, net_win_cap, extra_cost.
- E. Risk/portfolio state (41-50): max_dd_halt, cooldown_after_loss, cooldown_after_win,
  loss_streak_halt, win_streak_cap, daily_loss_limit, daily_trade_cap, min_trade_spacing,
  equity_floor_halt, equity_target_halt.

## DONE
- [x] `include/strategy/strategy.hpp` — full 50-knob config + state + decide() pipeline.
- [x] `include/data/windower.hpp` — added `Sample::entry_time_ms` + set from `time_[window_]`
      (needed for session/day knobs #31, #46, #47).

## TODO (remaining)
- [ ] **wfdeep.cpp wiring** (honest path) — IN PROGRESS, not finished:
  - [ ] `#include "strategy/strategy.hpp"`.
  - [ ] Parse `StrategyConfig sc = StrategyConfig::from(cfg, args)` near line 131.
  - [ ] Add `std::vector<int64_t> all_time_ms;` parallel to `all_t` (decl ~line 165).
  - [ ] Cache format bump **FTD1 → FTD2**: read + write `all_time_ms` (read ~line 200-212,
        write ~line 305-323, magic at 186 & 305). Old caches fail magic → auto-regenerate.
  - [ ] Producer loop (~line 273-281): `all_time_ms.push_back(s.entry_time_ms);`
  - [ ] `eval_block` (lines 621-665): build `EntryContext` per trade from entry-tick row
        `tickfeat[(entry_off[g] + window - 1) * FEAT + featidx]` using FROM_FEAT_SPREAD(5),
        FROM_FEAT_RV_64(15), FROM_FEAT_TICK_RATE(12), FROM_FEAT_MID_VELOCITY(9); set
        `ret=all_ret[g]`, `cost=all_cost[g]*(1+slippage_mult)+commission`, `t_index=all_t[g]`,
        `time_ms=all_time_ms[g]`. Fresh `StrategyState` per eval_block call (folds stay
        independent). Replace decision body with `Decision d = strat::decide(sc,st,ec); if(d.take){ out_nets.push_back(d.net); ...trade_sink... }`.
  - [ ] NOTE: barrier knobs #33-35 in wfdeep map to EXISTING `--barrier-k`/`--cost-mult`
        (they're baked into the cache key at line 172-173). Do NOT double-apply strategy
        barrier mults here → would desync cache. Leave wfdeep barriers as-is.

- [ ] **backtest.cpp wiring** (cli/backtest.cpp:130-163) — NOT STARTED:
  - [ ] `#include "strategy/strategy.hpp"`, parse `StrategyConfig::from`.
  - [ ] Wire #33-35 to `Windower::set_barriers(sc.barrier_tp_mult, sc.barrier_sl_mult,
        sc.barrier_cost_mult)` at line 56 (no cache here → safe).
  - [ ] Build `EntryContext` from `samples[i+b].X` last row + `entry_spread` + `entry_time_ms`;
        feed ret/cost in pips (caller unit). Replace trade block with `decide()`.
  - [ ] Accumulate `d.net`; size affects pip PnL.

- [ ] **config.toml** — add documented `[strategy]` section, all 50 keys = default (off)
      values + one-line comment each. Note unit-sensitive keys (clamps, net caps,
      equity/day limits) = caller pnl unit (return-units in wfdeep, pips in backtest).

- [ ] **Build + verify**:
  - [ ] `cmake --build build` (or RUN.cmd) clean in both TUs.
  - [ ] Defaults no-op regression: wfdeep on small slice = byte-identical trades/edge/PF.
  - [ ] Per-group toggle smoke: e.g. `--strategy.size_mode vol_target`,
        `--strategy.max_spread ...`, `--strategy.session_mask ...`, `--strategy.frac_kelly 0.5`.

## Gotchas / decisions
- `all_t` / `t_index` = tick ORDINAL, not ms. Wall-clock comes from new `entry_time_ms`
  (#31/#46/#47 inert when time_ms==0).
- decide() unit-agnostic; honest path = return-units (the one that matters for the report).
- Overfit risk (50 knobs): rely on existing purged WF + deflated Sharpe in wf_metrics.hpp;
  enable knobs in small groups, judge only on untouched holdout. (Per request: no extra
  gating added in code — guidance only.)
