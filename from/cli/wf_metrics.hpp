#pragma once
// ============================================================================
// wf_metrics.hpp — shared after-cost trade metrics for the walk-forward family.
//
// Extracted (verbatim) from walkforward.cpp so wfdeep.cpp reuses the SAME honest
// metric machinery (N_eff / CI / deflation), guaranteeing byte-identical stats
// logic across the summary and raw-window paths. walkforward.cpp is intentionally
// left untouched (it keeps its own private copy); only wfdeep.cpp includes this.
//
// Everything here is `inline` and self-contained.
// ============================================================================

#include "commands.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace from {
namespace wfm {

inline size_t arg_size(const CliArgs& args, const std::string& key, size_t def) {
    std::string v = args.get(key, "");
    return v.empty() ? def : static_cast<size_t>(std::stoull(v));
}

inline float arg_float(const CliArgs& args, const std::string& key, float def) {
    std::string v = args.get(key, "");
    return v.empty() ? def : std::stof(v);
}

// ----------------------------------------------------------------------------
// After-cost trade metrics computed from a vector of net per-trade returns.
// NaN-safe: zero trades -> all-zero struct.
// ----------------------------------------------------------------------------
struct TradeStats {
    size_t trades = 0;
    double winrate = 0.0;
    double edge = 0.0;
    double profit_factor = 0.0;
    double kelly = 0.0;
    double max_drawdown = 0.0;
    double t_stat = 0.0;
    double sharpe = 0.0;
    double n_eff = 0.0;
    double se = 0.0;
    double ci_lo = 0.0;
    double ci_hi = 0.0;
};

// nets must already be in time order. `block` is the overlap factor (reach/stride):
// consecutive samples share future path, so N_eff = trades / block is used for
// se / t_stat / CI (NOT the raw count).
inline TradeStats compute_stats(const std::vector<float>& nets, double block) {
    TradeStats st;
    st.trades = nets.size();
    if (st.trades == 0) return st;

    double sum = 0.0, gross_profit = 0.0, gross_loss = 0.0;
    size_t wins = 0;
    for (float v : nets) {
        sum += v;
        if (v > 0.0f) { ++wins; gross_profit += v; }
        else          { gross_loss += -static_cast<double>(v); }
    }
    double n = static_cast<double>(st.trades);
    double mean = sum / n;
    st.edge = mean;
    st.winrate = static_cast<double>(wins) / n;

    if (block < 1.0) block = 1.0;
    st.n_eff = n / block;
    if (st.n_eff < 1.0) st.n_eff = 1.0;

    double var = 0.0;
    for (float v : nets) {
        double d = static_cast<double>(v) - mean;
        var += d * d;
    }
    var /= n;
    double sd = std::sqrt(var);
    if (sd > 1e-12 && st.trades >= 2) {
        st.sharpe = mean / sd;
        st.se = sd / std::sqrt(st.n_eff);
        st.t_stat = (st.se > 1e-300) ? (mean / st.se) : 0.0;
    } else {
        st.sharpe = 0.0;
        st.se = 0.0;
        st.t_stat = 0.0;
    }
    st.ci_lo = mean - 1.96 * st.se;
    st.ci_hi = mean + 1.96 * st.se;

    if (gross_loss > 1e-12) st.profit_factor = gross_profit / gross_loss;
    else                    st.profit_factor = (gross_profit > 0.0) ? 999.0 : 0.0;

    double losses = n - static_cast<double>(wins);
    double avg_win  = (wins   > 0)   ? gross_profit / static_cast<double>(wins)   : 0.0;
    double avg_loss = (losses > 0.0) ? gross_loss   / losses                      : 0.0;
    if (avg_loss > 1e-12) {
        double b = avg_win / avg_loss;
        st.kelly = st.winrate - (1.0 - st.winrate) / (b + 1e-12);
    } else {
        st.kelly = 0.0;
    }

    double equity = 0.0, peak = 0.0, max_dd = 0.0;
    for (float v : nets) {
        equity += v;
        if (equity > peak) peak = equity;
        double dd = peak - equity;
        if (dd > max_dd) max_dd = dd;
    }
    st.max_drawdown = max_dd;
    return st;
}

inline void print_stats_row(std::ostream& os, const std::string& name, const TradeStats& s) {
    os << std::left << std::setw(14) << name << std::right
       << std::setw(9)  << s.trades
       << std::setw(10) << std::fixed << std::setprecision(1) << s.n_eff
       << std::setw(10) << std::setprecision(3) << s.winrate
       << std::setw(12) << std::setprecision(6) << s.edge
       << std::setw(9)  << std::setprecision(2) << s.profit_factor
       << std::setw(9)  << std::setprecision(3) << s.kelly
       << std::setw(11) << std::setprecision(6) << s.max_drawdown
       << std::setw(11) << std::setprecision(6) << s.se
       << std::setw(9)  << std::setprecision(2) << s.t_stat
       << std::setw(9)  << std::setprecision(3) << s.sharpe
       << std::setw(13) << std::setprecision(6) << s.ci_lo
       << std::setw(13) << std::setprecision(6) << s.ci_hi
       << "\n";
}

// ----------------------------------------------------------------------------
// Robustness: Probabilistic & Deflated Sharpe Ratio (Bailey / Lopez de Prado).
//
// A Sharpe estimated from a finite, non-normal, OVERLAPPING sample is noisy and,
// when it is the best of many trials, upward-biased. PSR answers "what is the
// probability the TRUE Sharpe exceeds a benchmark, given skew/kurtosis and the
// effective sample size". DSR sets that benchmark to the Sharpe you would expect
// to see by luck alone after `n_trials` independent backtests -> the honest test.
// All inputs are per-trade (not annualized); n_eff is the overlap-deflated count.
// ----------------------------------------------------------------------------
inline double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Acklam's inverse-normal-CDF approximation (probit), abs err < 1.2e-9.
inline double normal_ppf(double p) {
    if (p <= 0.0) return -1e9;
    if (p >= 1.0) return 1e9;
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    double q, r;
    if (p < 0.02425) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 0.97575) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

struct SharpeRobustness {
    double sharpe = 0.0;   // per-trade
    double skew = 0.0;
    double kurt = 3.0;     // non-excess (normal = 3)
    double psr = 0.0;      // P(true SR > 0)
    double dsr = 0.0;      // P(true SR > expected-max-under-n_trials)
    double sr_star = 0.0;  // the deflation benchmark used by DSR
};

// nets in time order; n_eff = overlap-deflated trade count; n_trials = number of
// configurations tried (folds * hyperparam grid). n_trials<=1 => DSR==PSR.
inline SharpeRobustness sharpe_robustness(const std::vector<float>& nets, double n_eff,
                                          size_t n_trials = 1) {
    SharpeRobustness r;
    size_t n = nets.size();
    if (n < 4 || n_eff < 4.0) return r;
    double mean = 0.0;
    for (float v : nets) mean += v;
    mean /= static_cast<double>(n);
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (float v : nets) {
        double d = static_cast<double>(v) - mean;
        double d2 = d * d;
        m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    }
    m2 /= n; m3 /= n; m4 /= n;
    double sd = std::sqrt(m2);
    if (sd < 1e-12) return r;
    r.sharpe = mean / sd;
    r.skew = m3 / (sd * sd * sd);
    r.kurt = m4 / (m2 * m2);

    // Asymptotic SE of the Sharpe estimator under non-normality (Mertens / Lo).
    double denom = 1.0 - r.skew * r.sharpe + (r.kurt - 1.0) / 4.0 * r.sharpe * r.sharpe;
    if (denom < 1e-9) denom = 1e-9;
    double se_sr = std::sqrt(denom / (n_eff - 1.0));

    r.psr = normal_cdf((r.sharpe - 0.0) / se_sr);

    // Expected max Sharpe of n_trials independent strategies with true SR=0.
    if (n_trials > 1) {
        const double gamma = 0.5772156649015329;  // Euler-Mascheroni
        double Nt = static_cast<double>(n_trials);
        double e_max = (1.0 - gamma) * normal_ppf(1.0 - 1.0 / Nt) +
                       gamma * normal_ppf(1.0 - 1.0 / (Nt * std::exp(1.0)));
        r.sr_star = se_sr * e_max;  // benchmark Sharpe expected from luck alone
    }
    r.dsr = normal_cdf((r.sharpe - r.sr_star) / se_sr);
    return r;
}

inline void print_stats_header(std::ostream& os) {
    os << std::left << std::setw(14) << "segment" << std::right
       << std::setw(9)  << "trades"
       << std::setw(10) << "N_eff"
       << std::setw(10) << "winrate"
       << std::setw(12) << "edge"
       << std::setw(9)  << "PF"
       << std::setw(9)  << "kelly"
       << std::setw(11) << "maxDD"
       << std::setw(11) << "se"
       << std::setw(9)  << "t_stat"
       << std::setw(9)  << "sharpe"
       << std::setw(13) << "ci95_lo"
       << std::setw(13) << "ci95_hi"
       << "\n";
    os << std::string(128, '-') << "\n";
}

}  // namespace wfm
}  // namespace from
