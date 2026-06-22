#pragma once
// ============================================================================
// strategy.hpp — 50 opt-in trading-logic features for the XAUUSD engine.
//
// ONE shared decision layer, included by BOTH simulation sites so they can never
// diverge (same pattern as wf_metrics.hpp):
//   - cli/wfdeep.cpp   eval_block()  (honest purged walk-forward, return units)
//   - cli/backtest.cpp run loop      (single-pass backtest, pip units)
//
// Design contract:
//   * decide() is CAUSAL — it reads ONLY entry-time inputs (class probs, entry-tick
//     features, wall-clock time, and accumulated portfolio state). The realized
//     future return (EntryContext::ret) is NOT read by the entry/sizing logic; it
//     is only used AFTER the take decision to compute net, and is shaped by exit
//     clamps that are fixed at entry (a stop/target known before the trade).
//   * Every feature is OFF by default -> decide() reproduces today's behavior
//     (argmax + conf_gate + fixed unit size), so existing walk-forward numbers are
//     unchanged until a knob is turned on.
//   * Unit-agnostic: caller passes ret/cost in its own unit (return-units in
//     wfdeep, pips in backtest). Ratio/count/streak knobs are unit-free; the few
//     unit-sensitive knobs (clamps, net caps, equity/day limits) are interpreted
//     in the caller's pnl unit (see config.toml comments).
//
// Header-only / inline / self-contained.
// ============================================================================

#include "commands.hpp"
#include "common.h"
#include "utils/config_parser.hpp"
#include "wf_metrics.hpp"  // wfm::arg_float / wfm::arg_size

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>

namespace from {
namespace strat {

// ---------------------------------------------------------------------------
// Position-sizing mode (feature group B selector).
// ---------------------------------------------------------------------------
enum class SizeMode { Fixed, ConfWeighted, FracKelly, VolTarget, InvSpread };

inline SizeMode parse_size_mode(const std::string& s) {
    if (s == "conf" || s == "conf_weighted") return SizeMode::ConfWeighted;
    if (s == "kelly" || s == "frac_kelly")   return SizeMode::FracKelly;
    if (s == "vol"   || s == "vol_target")   return SizeMode::VolTarget;
    if (s == "inv_spread")                   return SizeMode::InvSpread;
    return SizeMode::Fixed;
}

// ===========================================================================
// StrategyConfig — all 50 knobs. Defaults = neutral (feature inert).
// ===========================================================================
struct StrategyConfig {
    // --- A. Direction & signal gating (1-10) ---
    float  conf_gate        = 0.50f;  // 1  min top-class prob to trade
    float  margin_gate      = 0.0f;   // 2  min (p_top - p_second)
    float  neutral_mass_max = 1.0f;   // 3  skip if p_neutral above this
    float  prob_floor       = 0.0f;   // 4  min prob of the chosen direction
    bool   long_only        = false;  // 5  take longs only
    bool   short_only       = false;  // 6  take shorts only
    bool   invert_signal    = false;  // 7  contrarian: trade opposite of model
    bool   signal_ema_confirm = false;// 8  require signed-prob EMA to agree w/ dir
    float  signal_ema_alpha = 0.2f;   // 8  EMA smoothing for the confirm filter
    int    agreement_run    = 0;      // 9  require N consecutive same-dir preds
    int    flip_block       = 0;      // 10 block reversal for N trades after entry

    // --- B. Position sizing (11-22) ---
    SizeMode size_mode      = SizeMode::Fixed;  // 11/12/13/14/15 selector
    float  base_size        = 1.0f;   // 11 baseline unit size
    float  kelly_frac       = 0.5f;   // 13 fraction of rolling Kelly to deploy
    int    kelly_window     = 200;    // 13 trailing trades for live Kelly est.
    float  vol_target       = 0.0f;   // 14 target vol; size = vt/entry_rv (0=off)
    float  size_cap         = 1e9f;   // 16 max size multiplier
    float  size_floor       = 0.0f;   // 17 min size (below -> skip if >0 and clipped)
    bool   equity_compound  = false;  // 18 scale size by current equity/base_equity
    float  base_equity      = 1.0f;   // 18 equity denominator for compounding
    float  dd_derisk        = 0.0f;   // 19 shrink size by (1 - dd/dd_derisk), 0=off
    float  anti_martingale  = 1.0f;   // 20 size *= this^win_streak (>1 ramps winners)
    float  martingale       = 1.0f;   // 21 size *= this^loss_streak (>1 ramps losers)
    float  recovery_size    = 1.0f;   // 22 size multiplier while equity < peak

    // --- C. Entry-context filters (23-32), all causal entry-time ---
    float  max_spread       = 1e9f;   // 23 skip if entry spread above this
    float  min_tick_rate    = -1e9f;  // 24 skip if entry tick_rate below this
    float  vol_min          = -1e9f;  // 25 skip if entry rv below this
    float  vol_max          = 1e9f;   // 26 skip if entry rv above this
    int    trend_filter     = 0;      // 27 +1: longs only w/ up-momentum & shorts w/ down; -1: opposite; 0 off
    float  trend_block      = 0.0f;   // 28 skip if |entry momentum| > this AND against dir
    float  min_expected_edge= -1e9f;  // 29 skip if (p_up-p_down)*edge_k*rv - cost < this
    float  cost_skip_mult   = 0.0f;   // 30 skip if cost > mult*edge_k*rv (0=off)
    uint32_t session_mask   = 0xFFFFFFFFu; // 31 allowed UTC hours bitmask (bit h)
    uint32_t env_mask       = 0xFFFFFFFFu; // 32 allowed env_id bitmask (recomputed regime)
    float  edge_k           = 1.0f;   // shared scale for #29/#30 expected-move proxy

    // --- D. Exit / barrier shaping (33-40) ---
    float  barrier_tp_mult  = 1.0f;   // 33 -> Windower::set_barriers k_tp
    float  barrier_sl_mult  = 1.0f;   // 34 -> Windower::set_barriers k_sl
    float  barrier_cost_mult= 1.5f;   // 35 -> Windower::set_barriers cost_mult
    float  tp_clamp         = 1e9f;   // 36 cap directional gross pnl at +tp_clamp
    float  sl_clamp         = 1e9f;   // 37 floor directional gross pnl at -sl_clamp
    float  net_loss_cap     = 1e9f;   // 38 cap per-trade net loss at -net_loss_cap
    float  net_win_cap      = 1e9f;   // 39 cap per-trade net win at +net_win_cap
    float  extra_cost       = 0.0f;   // 40 extra fixed cost/trade (exec realism)

    // --- E. Risk / portfolio state (41-50) ---
    float  max_dd_halt      = 1e9f;   // 41 pause trading while drawdown > this
    int    cooldown_after_loss = 0;   // 42 skip N trades after a loss
    int    cooldown_after_win  = 0;   // 43 skip N trades after a win
    int    loss_streak_halt = 0;      // 44 permanently halt after N straight losses
    int    win_streak_cap   = 0;      // 45 pause (cooldown) after N straight wins
    float  daily_loss_limit = 1e9f;   // 46 stop for the (UTC) day after losing this
    int    daily_trade_cap  = 1<<30;  // 47 max trades per (UTC) day
    long long min_trade_spacing = 0;  // 48 min tick-index gap between entries
    float  equity_floor_halt= -1e9f;  // 49 permanently halt if equity < this
    float  equity_target_halt = 1e9f; // 50 permanently halt once equity >= this

    // Construct from config.toml [strategy] with CLI --strategy.<key> override.
    static StrategyConfig from(const Config& cfg, const CliArgs& args) {
        StrategyConfig c;
        auto F = [&](const char* key, float def) -> float {
            std::string cli = std::string("--strategy.") + key;
            float v = cfg.get_float(std::string("strategy.") + key, def);
            return wfm::arg_float(args, cli, v);
        };
        auto I = [&](const char* key, long long def) -> long long {
            std::string cli = std::string("--strategy.") + key;
            long long v = static_cast<long long>(
                cfg.get_size(std::string("strategy.") + key, static_cast<size_t>(def)));
            std::string a = args.get(cli, "");
            return a.empty() ? v : static_cast<long long>(std::stoll(a));
        };
        auto B = [&](const char* key, bool def) -> bool {
            std::string cli = std::string("--strategy.") + key;
            bool v = cfg.get_bool(std::string("strategy.") + key, def);
            if (args.has(cli)) {                       // presence = true unless =false
                std::string a = args.get(cli, "true");
                return !(a == "false" || a == "0" || a == "no");
            }
            return v;
        };
        auto S = [&](const char* key, const std::string& def) -> std::string {
            std::string cli = std::string("--strategy.") + key;
            std::string v = cfg.get_string(std::string("strategy.") + key, def);
            return args.get(cli, v);
        };

        c.conf_gate         = F("conf_gate", c.conf_gate);
        c.margin_gate       = F("margin_gate", c.margin_gate);
        c.neutral_mass_max  = F("neutral_mass_max", c.neutral_mass_max);
        c.prob_floor        = F("prob_floor", c.prob_floor);
        c.long_only         = B("long_only", c.long_only);
        c.short_only        = B("short_only", c.short_only);
        c.invert_signal     = B("invert_signal", c.invert_signal);
        c.signal_ema_confirm= B("signal_ema_confirm", c.signal_ema_confirm);
        c.signal_ema_alpha  = F("signal_ema_alpha", c.signal_ema_alpha);
        c.agreement_run     = static_cast<int>(I("agreement_run", c.agreement_run));
        c.flip_block        = static_cast<int>(I("flip_block", c.flip_block));

        c.size_mode         = parse_size_mode(S("size_mode", "fixed"));
        c.base_size         = F("base_size", c.base_size);
        c.kelly_frac        = F("kelly_frac", c.kelly_frac);
        c.kelly_window      = static_cast<int>(I("kelly_window", c.kelly_window));
        c.vol_target        = F("vol_target", c.vol_target);
        c.size_cap          = F("size_cap", c.size_cap);
        c.size_floor        = F("size_floor", c.size_floor);
        c.equity_compound   = B("equity_compound", c.equity_compound);
        c.base_equity       = F("base_equity", c.base_equity);
        c.dd_derisk         = F("dd_derisk", c.dd_derisk);
        c.anti_martingale   = F("anti_martingale", c.anti_martingale);
        c.martingale        = F("martingale", c.martingale);
        c.recovery_size     = F("recovery_size", c.recovery_size);

        c.max_spread        = F("max_spread", c.max_spread);
        c.min_tick_rate     = F("min_tick_rate", c.min_tick_rate);
        c.vol_min           = F("vol_min", c.vol_min);
        c.vol_max           = F("vol_max", c.vol_max);
        c.trend_filter      = static_cast<int>(I("trend_filter", c.trend_filter));
        c.trend_block       = F("trend_block", c.trend_block);
        c.min_expected_edge = F("min_expected_edge", c.min_expected_edge);
        c.cost_skip_mult    = F("cost_skip_mult", c.cost_skip_mult);
        c.edge_k            = F("edge_k", c.edge_k);
        c.session_mask      = static_cast<uint32_t>(I("session_mask", c.session_mask));
        c.env_mask          = static_cast<uint32_t>(I("env_mask", c.env_mask));

        c.barrier_tp_mult   = F("barrier_tp_mult", c.barrier_tp_mult);
        c.barrier_sl_mult   = F("barrier_sl_mult", c.barrier_sl_mult);
        c.barrier_cost_mult = F("barrier_cost_mult", c.barrier_cost_mult);
        c.tp_clamp          = F("tp_clamp", c.tp_clamp);
        c.sl_clamp          = F("sl_clamp", c.sl_clamp);
        c.net_loss_cap      = F("net_loss_cap", c.net_loss_cap);
        c.net_win_cap       = F("net_win_cap", c.net_win_cap);
        c.extra_cost        = F("extra_cost", c.extra_cost);

        c.max_dd_halt       = F("max_dd_halt", c.max_dd_halt);
        c.cooldown_after_loss = static_cast<int>(I("cooldown_after_loss", c.cooldown_after_loss));
        c.cooldown_after_win  = static_cast<int>(I("cooldown_after_win", c.cooldown_after_win));
        c.loss_streak_halt  = static_cast<int>(I("loss_streak_halt", c.loss_streak_halt));
        c.win_streak_cap    = static_cast<int>(I("win_streak_cap", c.win_streak_cap));
        c.daily_loss_limit  = F("daily_loss_limit", c.daily_loss_limit);
        c.daily_trade_cap   = static_cast<int>(I("daily_trade_cap", c.daily_trade_cap));
        c.min_trade_spacing = I("min_trade_spacing", c.min_trade_spacing);
        c.equity_floor_halt = F("equity_floor_halt", c.equity_floor_halt);
        c.equity_target_halt= F("equity_target_halt", c.equity_target_halt);
        return c;
    }
};

// ===========================================================================
// EntryContext — everything decide() may read at entry time (causal).
// ===========================================================================
struct EntryContext {
    float p_up = 0.0f, p_neutral = 1.0f, p_down = 0.0f;  // class probs
    float ret  = 0.0f;   // signed realized return entry->exit (caller's unit) — NOT read pre-take
    float cost = 0.0f;   // per-trade cost (caller's unit, >=0)
    long long t_index = 0;   // entry tick ordinal (for spacing/dedup)
    long long time_ms = 0;   // entry wall-clock ms (0 = unknown -> session/day knobs inert)
    float spread    = 0.0f;  // entry spread (raw or norm; same unit as max_spread)
    float rv        = 0.0f;  // entry realized vol (e.g. RV64)
    float tick_rate = 0.0f;  // entry tick rate
    float momentum  = 0.0f;  // signed entry momentum proxy (mid velocity)
};

struct Decision {
    bool  take = false;
    int   dir  = 0;     // +1 long / -1 short (post bias/inversion)
    float size = 0.0f;  // position size multiplier actually used
    float net  = 0.0f;  // realized net pnl after sizing + clamps + costs
};

// UTC hour [0,23] of a wall-clock ms timestamp.
inline int utc_hour(long long time_ms) {
    long long ms_day = 24LL * 60 * 60 * 1000;
    long long t = ((time_ms % ms_day) + ms_day) % ms_day;
    return static_cast<int>(t / (60LL * 60 * 1000));
}

// Recompute a coarse regime id from entry rv + session — mirrors
// Windower::environment_for so #32 env_mask is meaningful in both paths.
inline int regime_id(const EntryContext& e) {
    int hour = (e.time_ms > 0) ? utc_hour(e.time_ms) : -1;
    if (hour >= 7 && hour < 12) return 0;
    if (hour >= 13 && hour < 17) return 1;
    if (hour >= 23 || (hour >= 0 && hour < 4)) return 2;
    if (e.rv > 0.001f) return 3;
    if (e.rv < 0.0001f) return 4;
    return 5;
}

// ===========================================================================
// StrategyState — mutable portfolio memory across one time-ordered sim block.
// Fresh instance per fold / per holdout so blocks stay independent.
// ===========================================================================
struct StrategyState {
    double equity = 0.0, peak = 0.0, max_dd = 0.0;
    int win_streak = 0, loss_streak = 0;
    int cooldown_left = 0;            // trades to skip (cooldown / win-streak pause)
    long long last_trade_t = -1;      // tick ordinal of last taken entry
    int last_dir = 0;                 // direction of last taken entry
    int flip_block_left = 0;          // trades during which reversal is blocked
    long long cur_day = -1;           // current UTC day bucket
    double day_pnl = 0.0;             // pnl accumulated this day
    int day_trades = 0;               // trades taken this day
    bool day_locked = false;          // daily limit/cap hit -> no more this day
    float sig_ema = 0.0f;             // EMA of signed prob (p_up - p_down)
    int agree_run = 0, agree_dir = 0; // consecutive same-dir prediction run
    std::deque<float> kelly_hist;     // trailing per-trade net for live Kelly
    bool halted = false;              // permanent halt (loss-streak / equity floor/target)

    // Rolling fractional-Kelly fraction from trailing net history.
    float kelly_fraction(int window) const {
        if (kelly_hist.empty()) return 0.0f;
        size_t n = kelly_hist.size();
        size_t start = (window > 0 && n > static_cast<size_t>(window)) ? n - window : 0;
        double gp = 0.0, gl = 0.0; size_t w = 0, l = 0;
        for (size_t i = start; i < n; ++i) {
            float v = kelly_hist[i];
            if (v > 0.0f) { gp += v; ++w; } else { gl += -static_cast<double>(v); ++l; }
        }
        double tot = static_cast<double>(w + l);
        if (tot < 1.0 || l == 0) return 0.0f;
        double winrate = static_cast<double>(w) / tot;
        double avg_win = (w > 0) ? gp / static_cast<double>(w) : 0.0;
        double avg_loss = (l > 0) ? gl / static_cast<double>(l) : 0.0;
        if (avg_loss <= 1e-12) return 0.0f;
        double b = avg_win / avg_loss;
        double k = winrate - (1.0 - winrate) / (b + 1e-12);
        return static_cast<float>(std::max(0.0, k));
    }
};

// ===========================================================================
// decide() — the whole 50-feature pipeline.
// Updates signal-tracking state on every call; updates portfolio state only on
// a taken trade (via on_close, called by decide on take).
// ===========================================================================
inline void on_close(const StrategyConfig& c, StrategyState& st, const Decision& d) {
    st.equity += d.net;
    if (st.equity > st.peak) st.peak = st.equity;
    double dd = st.peak - st.equity;
    if (dd > st.max_dd) st.max_dd = dd;

    bool win = d.net > 0.0f;
    if (win) { st.win_streak++; st.loss_streak = 0; }
    else     { st.loss_streak++; st.win_streak = 0; }

    // cooldowns triggered by outcome
    if (!win && c.cooldown_after_loss > 0) st.cooldown_left = std::max(st.cooldown_left, c.cooldown_after_loss);
    if ( win && c.cooldown_after_win  > 0) st.cooldown_left = std::max(st.cooldown_left, c.cooldown_after_win);
    if (c.win_streak_cap > 0 && st.win_streak >= c.win_streak_cap) st.cooldown_left = std::max(st.cooldown_left, 1);
    if (c.loss_streak_halt > 0 && st.loss_streak >= c.loss_streak_halt) st.halted = true;

    // permanent equity halts
    if (st.equity < c.equity_floor_halt)  st.halted = true;
    if (st.equity >= c.equity_target_halt) st.halted = true;

    // day bookkeeping
    st.day_pnl += d.net;
    st.day_trades++;
    if (st.day_pnl <= -static_cast<double>(c.daily_loss_limit)) st.day_locked = true;
    if (st.day_trades >= c.daily_trade_cap) st.day_locked = true;

    // live-kelly history
    st.kelly_hist.push_back(d.net);
    if (c.kelly_window > 0 && st.kelly_hist.size() > static_cast<size_t>(c.kelly_window) * 2)
        st.kelly_hist.pop_front();

    st.last_dir = d.dir;
    if (c.flip_block > 0) st.flip_block_left = c.flip_block;
}

inline Decision decide(const StrategyConfig& c, StrategyState& st, const EntryContext& e) {
    Decision d;

    // ---- signal tracking (updated every call, pre-gate) ----
    float signed_prob = e.p_up - e.p_down;
    st.sig_ema = c.signal_ema_alpha * signed_prob + (1.0f - c.signal_ema_alpha) * st.sig_ema;

    // raw model prediction (argmax over up/neutral/down)
    int raw = 1;  // neutral
    float mx = e.p_neutral;
    if (e.p_up   > mx) { mx = e.p_up;   raw = 0; }
    if (e.p_down > mx) { mx = e.p_down; raw = 2; }
    int pred_dir = (raw == 0) ? 1 : (raw == 2 ? -1 : 0);

    // agreement-run tracking
    if (pred_dir != 0 && pred_dir == st.agree_dir) st.agree_run++;
    else { st.agree_dir = pred_dir; st.agree_run = (pred_dir != 0) ? 1 : 0; }

    // decrement per-trade-ordinal counters that gate this opportunity
    bool in_cooldown = st.cooldown_left > 0;
    if (st.cooldown_left > 0) st.cooldown_left--;
    bool flip_active = st.flip_block_left > 0;
    if (st.flip_block_left > 0) st.flip_block_left--;

    // ---- day rollover (UTC) ----
    if (e.time_ms > 0) {
        long long ms_day = 24LL * 60 * 60 * 1000;
        long long day = e.time_ms / ms_day;
        if (day != st.cur_day) { st.cur_day = day; st.day_pnl = 0.0; st.day_trades = 0; st.day_locked = false; }
    }

    if (st.halted) return d;                 // 44/49/50 permanent stop
    if (st.day_locked) return d;             // 46/47 daily stop
    if (in_cooldown) return d;               // 42/43/45 cooldown
    if (pred_dir == 0) return d;             // model says neutral

    // ---- direction & inversion (5/6/7) ----
    int dir = pred_dir;
    if (c.invert_signal) dir = -dir;
    if (c.long_only  && dir != 1)  return d;
    if (c.short_only && dir != -1) return d;

    // ---- A. signal gating (1/2/3/4/8/9/10) ----
    if (mx < c.conf_gate) return d;                                  // 1
    float p_top = mx;
    float p_second = std::max({ (raw==0?std::max(e.p_neutral,e.p_down):0.0f),
                                (raw==1?std::max(e.p_up,e.p_down):0.0f),
                                (raw==2?std::max(e.p_up,e.p_neutral):0.0f) });
    if ((p_top - p_second) < c.margin_gate) return d;               // 2
    if (e.p_neutral > c.neutral_mass_max) return d;                 // 3
    float p_dir = (dir == 1) ? e.p_up : e.p_down;
    if (p_dir < c.prob_floor) return d;                             // 4
    if (c.signal_ema_confirm) {                                     // 8
        if ((dir == 1 && st.sig_ema <= 0.0f) || (dir == -1 && st.sig_ema >= 0.0f)) return d;
    }
    if (c.agreement_run > 0 && (st.agree_dir != pred_dir || st.agree_run < c.agreement_run)) return d;  // 9
    if (flip_active && dir != st.last_dir && st.last_dir != 0) return d;  // 10

    // ---- C. entry-context filters (23-32), causal ----
    if (e.spread    > c.max_spread)    return d;                    // 23
    if (e.tick_rate < c.min_tick_rate) return d;                   // 24
    if (e.rv < c.vol_min || e.rv > c.vol_max) return d;            // 25/26
    if (c.trend_filter != 0) {                                     // 27
        int want = c.trend_filter * dir;  // +1: align dir w/ momentum sign
        if (want > 0 && e.momentum * dir <= 0.0f) return d;
        if (want < 0 && e.momentum * dir >= 0.0f) return d;
    }
    if (c.trend_block > 0.0f && std::abs(e.momentum) > c.trend_block
        && e.momentum * dir < 0.0f) return d;                     // 28
    float move_proxy = c.edge_k * e.rv;                            // expected |move|
    float exp_edge = signed_prob * move_proxy - e.cost;           // 29 (signed by model)
    if (dir == -1) exp_edge = (-signed_prob) * move_proxy - e.cost;
    if (exp_edge < c.min_expected_edge) return d;                 // 29
    if (c.cost_skip_mult > 0.0f && e.cost > c.cost_skip_mult * move_proxy) return d;  // 30
    if (e.time_ms > 0) {                                          // 31 session
        int h = utc_hour(e.time_ms);
        if (!((c.session_mask >> h) & 1u)) return d;
    }
    if (c.env_mask != 0xFFFFFFFFu) {                              // 32 regime gate
        int rid = regime_id(e);
        if (rid >= 0 && rid < 32 && !((c.env_mask >> rid) & 1u)) return d;
    }

    // ---- spacing / overlap (48) ----
    if (c.min_trade_spacing > 0 && st.last_trade_t >= 0
        && (e.t_index - st.last_trade_t) < c.min_trade_spacing) return d;  // 48

    // ---- max-drawdown pause (41) ----
    if (static_cast<float>(st.max_dd) > c.max_dd_halt) return d;  // 41 (resumes on recovery)

    // ---- B. sizing (11-22) ----
    float size = c.base_size;
    switch (c.size_mode) {
        case SizeMode::ConfWeighted: {                            // 12
            float denom = std::max(1e-6f, 1.0f - c.conf_gate);
            size = c.base_size * std::clamp((p_top - c.conf_gate) / denom, 0.0f, 1.0f);
            break;
        }
        case SizeMode::FracKelly:                                 // 13
            size = c.base_size * c.kelly_frac * st.kelly_fraction(c.kelly_window);
            break;
        case SizeMode::VolTarget:                                 // 14
            size = (e.rv > 1e-9f && c.vol_target > 0.0f) ? c.base_size * (c.vol_target / e.rv) : c.base_size;
            break;
        case SizeMode::InvSpread:                                 // 15
            size = c.base_size / (1.0f + std::max(0.0f, e.spread));
            break;
        case SizeMode::Fixed: default: size = c.base_size; break; // 11
    }
    if (c.equity_compound && c.base_equity > 1e-9f)               // 18
        size *= std::max(0.0f, static_cast<float>(st.equity) / c.base_equity + 1.0f);
    if (c.dd_derisk > 0.0f)                                       // 19
        size *= std::clamp(1.0f - static_cast<float>(st.max_dd) / c.dd_derisk, 0.0f, 1.0f);
    if (c.anti_martingale != 1.0f) size *= std::pow(c.anti_martingale, static_cast<float>(st.win_streak));   // 20
    if (c.martingale      != 1.0f) size *= std::pow(c.martingale,      static_cast<float>(st.loss_streak));  // 21
    if (st.equity < st.peak) size *= c.recovery_size;            // 22
    size = std::min(size, c.size_cap);                           // 16
    if (size < c.size_floor) return d;                          // 17 (too small -> skip)
    if (size <= 0.0f) return d;

    // ---- D. exit shaping on realized directional pnl (33-40) ----
    // (33/34/35 are applied to the Windower at sim setup, not here.)
    float pnl = dir * e.ret;                  // directional gross pnl (caller's unit)
    pnl = std::min(pnl, c.tp_clamp);          // 36
    pnl = std::max(pnl, -c.sl_clamp);         // 37
    float cost = e.cost + c.extra_cost;       // 40
    float net = size * (pnl - cost);
    net = std::max(net, -c.net_loss_cap);     // 38
    net = std::min(net,  c.net_win_cap);      // 39

    d.take = true; d.dir = dir; d.size = size; d.net = net;
    st.last_trade_t = e.t_index;
    on_close(c, st, d);
    return d;
}

}  // namespace strat
}  // namespace from
