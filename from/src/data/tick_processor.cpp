#include "data/tick_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define ALIGNED_MALLOC(size, align) _aligned_malloc(size, align)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <x86intrin.h>
#define ALIGNED_MALLOC(size, align) aligned_alloc(align, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

namespace from {

// OPTIMIZED: Process ticks in batches with SIMD
FeatureChunk TickProcessor::process(const TickChunk& chunk) {
    FeatureChunk out;
    out.features = Tensor<float>({chunk.size, FROM_MAX_FEATURES});
    out.mid = chunk.mid;
    out.spread.resize(chunk.size);
    out.time_ms = chunk.time_ms;
    out.size = chunk.size;

    if (chunk.size == 0) return out;

    // Pre-allocate buffers (reuse across calls)
    static thread_local std::vector<double> ret_buf;
    static thread_local std::vector<double> dmid_buf;
    static thread_local std::vector<int> dir_buf;
    static thread_local std::vector<double> ofi_buf;
    static thread_local std::vector<double> micro_delta_buf;
    static thread_local std::vector<double> mid_delta_buf;

    ret_buf.reserve(chunk.size);
    dmid_buf.reserve(chunk.size);
    dir_buf.reserve(chunk.size);
    ofi_buf.reserve(chunk.size);
    micro_delta_buf.reserve(chunk.size);
    mid_delta_buf.reserve(chunk.size);

    // BATCH PROCESS: Compute all basic features in one pass (cache-friendly!)
    #pragma omp parallel for if(chunk.size > 10000)
    for (int i = 0; i < static_cast<int>(chunk.size); ++i) {
        double ask = chunk.ask[i];
        double bid = chunk.bid[i];
        double mid = chunk.mid[i] != 0.0 ? chunk.mid[i] : (ask + bid) * 0.5;
        double av = chunk.ask_vol[i];
        double bv = chunk.bid_vol[i];
        double spread = ask - bid;
        double norm_spread = spread / (mid + 1e-5);
        double ofi = (av - bv) / (av + bv + 1e-8);

        // Compute time delta
        double dt = 0.1;
        if (i > 0) {
            dt = std::max(0.001, (chunk.time_ms[i] - chunk.time_ms[i - 1]) / 1000.0);
        }

        // Compute velocity (use previous mid from global state for i=0)
        double prev_m = i > 0 ? chunk.mid[i-1] : prev_mid_;
        double dmid = mid - prev_m;
        double velocity = dmid / dt;
        double rate = std::clamp(1.0 / dt, 0.0, 1000.0);
        double log_rate = std::log(rate + 1.0);
        double micro = (ask * bv + bid * av) / (av + bv + 1e-8);
        double micro_dev = micro - mid;
        double amihud = std::abs(dmid) / (av + bv + 1e-8);

        // Store basic features (no dependencies)
        float* row = &out.features.at(i, 0);
        row[FROM_FEAT_ASK] = static_cast<float>(ask);
        row[FROM_FEAT_BID] = static_cast<float>(bid);
        row[FROM_FEAT_MID] = static_cast<float>(mid);
        row[FROM_FEAT_ASK_VOL] = static_cast<float>(av);
        row[FROM_FEAT_BID_VOL] = static_cast<float>(bv);
        row[FROM_FEAT_SPREAD] = static_cast<float>(spread);
        row[FROM_FEAT_NORM_SPREAD] = static_cast<float>(norm_spread);
        row[FROM_FEAT_OFI] = static_cast<float>(ofi);
        row[FROM_FEAT_MID_VELOCITY] = static_cast<float>(velocity);
        row[FROM_FEAT_TICK_RATE] = static_cast<float>(rate);
        row[FROM_FEAT_LOG_TICK_RATE] = static_cast<float>(log_rate);
        row[FROM_FEAT_MICRO_DEV] = static_cast<float>(micro_dev);
        row[FROM_FEAT_AMIHUD] = static_cast<float>(amihud);

        out.spread[i] = spread;
    }

    // Sequential features (depend on previous state)
    double prev_ofi = prev_ofi_;
    double prev_vel = prev_velocity_;
    int prev_dir = prev_dir_;

    for (size_t i = 0; i < chunk.size; ++i) {
        double mid = chunk.mid[i] != 0.0 ? chunk.mid[i] : (chunk.ask[i] + chunk.bid[i]) * 0.5;
        double av = chunk.ask_vol[i];
        double bv = chunk.bid_vol[i];
        double ofi = (av - bv) / (av + bv + 1e-8);
        double norm_spread = out.spread[i] / (mid + 1e-5);

        double prev_m = i > 0 ? chunk.mid[i-1] : prev_mid_;
        double dmid = mid - prev_m;
        double dt = i > 0 ? std::max(0.001, (chunk.time_ms[i] - chunk.time_ms[i-1]) / 1000.0) : 0.1;
        double velocity = dmid / dt;
        double accel = velocity - prev_vel;
        double bounce = (velocity * prev_vel < 0.0 && std::abs(dmid) < out.spread[i]) ? 1.0 : 0.0;
        int dir = dmid > 0.0 ? 1 : (dmid < 0.0 ? -1 : prev_dir);
        double rate = std::clamp(1.0 / dt, 0.0, 1000.0);

        float* row = &out.features.at(i, 0);
        row[FROM_FEAT_D_OFI] = static_cast<float>(ofi - prev_ofi);
        row[FROM_FEAT_MID_ACCEL] = static_cast<float>(accel);
        row[FROM_FEAT_BID_ASK_BOUNCE] = static_cast<float>(bounce);

        // Update rolling windows
        double ret = prev_m > 0.0 ? std::log((mid + 1e-8) / (prev_m + 1e-8)) : 0.0;
        auto push_sq = [](std::deque<double>& q, double& sum, double value, size_t cap) {
            sum += value;
            q.push_back(value);
            if (q.size() > cap) {
                sum -= q.front();
                q.pop_front();
            }
        };
        double ret_sq = ret * ret;
        push_sq(ret_sq_window_16_, ret_sq_sum_16_, ret_sq, 16);
        push_sq(ret_sq_window_64_, ret_sq_sum_64_, ret_sq, 64);
        push_sq(ret_sq_window_256_, ret_sq_sum_256_, ret_sq, 256);

        dmid_window_.push_back(dmid);
        if (dmid_window_.size() > 128) dmid_window_.pop_front();

        dir_window_.push_back(dir);
        dir_sum_32_ += dir;
        if (dir_window_.size() > 32) {
            dir_sum_32_ -= dir_window_.front();
            dir_window_.pop_front();
        }

        double rv16 = ret_sq_window_16_.size() >= 16 ? std::sqrt(ret_sq_sum_16_) : 0.0;
        double rv64 = ret_sq_window_64_.size() >= 64 ? std::sqrt(ret_sq_sum_64_) : 0.0;
        double rv256 = ret_sq_window_256_.size() >= 256 ? std::sqrt(ret_sq_sum_256_) : 0.0;

        row[FROM_FEAT_RV_16] = static_cast<float>(rv16);
        row[FROM_FEAT_RV_64] = static_cast<float>(rv64);
        row[FROM_FEAT_RV_256] = static_cast<float>(rv256);

        // Roll spread
        double cov = 0.0;
        if (dmid_window_.size() >= 2) {
            for (size_t k = 1; k < dmid_window_.size(); ++k) {
                cov += dmid_window_[k] * dmid_window_[k - 1];
            }
            cov /= static_cast<double>(dmid_window_.size() - 1);
        }
        double roll = 2.0 * std::sqrt(std::max(-cov, 0.0));
        row[FROM_FEAT_ROLL_SPREAD] = static_cast<float>(roll);

        // Lee-Ready sum
        row[FROM_FEAT_LEE_READY_SUM_32] = static_cast<float>(dir_sum_32_);

        // Kyle-Hasbrouck lambda: price impact per unit signed volume
        double signed_vol = av - bv;
        kh_dprice_.push_back(dmid);
        kh_svol_.push_back(signed_vol);
        if (kh_dprice_.size() > 64) { kh_dprice_.pop_front(); kh_svol_.pop_front(); }

        double kh_lambda = 0.0;
        if (kh_dprice_.size() >= 16) {
            double sv_mean = 0.0, dp_mean = 0.0;
            for (size_t k = 0; k < kh_dprice_.size(); ++k) {
                sv_mean += kh_svol_[k]; dp_mean += kh_dprice_[k];
            }
            double n_kh = static_cast<double>(kh_dprice_.size());
            sv_mean /= n_kh; dp_mean /= n_kh;
            double cov_sum = 0.0, var_sum = 0.0;
            for (size_t k = 0; k < kh_dprice_.size(); ++k) {
                double dv = kh_svol_[k] - sv_mean;
                cov_sum += dv * (kh_dprice_[k] - dp_mean);
                var_sum += dv * dv;
            }
            kh_lambda = var_sum > 1e-10 ? cov_sum / var_sum : 0.0;
        }
        row[FROM_FEAT_KYLE_HASBROUCK] = static_cast<float>(kh_lambda);

        // Feature 22: Trade Imbalance Ratio (32-tick)
        float bvol_contrib = (dir > 0) ? static_cast<float>(av) : 0.0f;
        float svol_contrib = (dir < 0) ? static_cast<float>(bv) : 0.0f;
        buy_vol_32_.push_back(bvol_contrib);
        sell_vol_32_.push_back(svol_contrib);
        buy_sum_32_f_ += bvol_contrib;
        sell_sum_32_f_ += svol_contrib;
        if (buy_vol_32_.size() > 32) {
            buy_sum_32_f_ -= buy_vol_32_.front(); buy_vol_32_.pop_front();
            sell_sum_32_f_ -= sell_vol_32_.front(); sell_vol_32_.pop_front();
        }
        float tir_denom = buy_sum_32_f_ + sell_sum_32_f_ + 1e-8f;
        row[FROM_FEAT_TRADE_IMBALANCE] = (buy_sum_32_f_ - sell_sum_32_f_) / tir_denom;

        // Feature 23: Volatility Regime Indicator (RV16/RV256)
        double vri = (rv256 > 1e-10) ? rv16 / rv256 : 1.0;
        vri = std::min(vri, 10.0);
        row[FROM_FEAT_VOL_REGIME] = static_cast<float>(vri);

        // Feature 24: Spread Compression Signal (z-score of norm_spread over 64 ticks)
        ns_window_64_.push_back(norm_spread);
        ns_sum_64_ += norm_spread;
        ns_sq_sum_64_ += norm_spread * norm_spread;
        if (ns_window_64_.size() > 64) {
            double old_ns = ns_window_64_.front(); ns_window_64_.pop_front();
            ns_sum_64_ -= old_ns;
            ns_sq_sum_64_ -= old_ns * old_ns;
        }
        double ns_mean = ns_sum_64_ / static_cast<double>(ns_window_64_.size());
        double ns_var = ns_sq_sum_64_ / static_cast<double>(ns_window_64_.size()) - ns_mean * ns_mean;
        double ns_std = std::sqrt(std::max(ns_var, 0.0) + 1e-10);
        double scs = (norm_spread - ns_mean) / ns_std;
        row[FROM_FEAT_SPREAD_COMPRESSION] = static_cast<float>(std::clamp(scs, -5.0, 5.0));

        // Feature 25: Autocorrelation of returns — lag 1, 32-tick window
        ac_returns_.push_back(ret);
        if (ac_returns_.size() > 33) ac_returns_.pop_front();
        double ac1 = 0.0;
        if (ac_returns_.size() >= 32) {
            size_t n_ac = ac_returns_.size() - 1;
            double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
            double sum_x2 = 0.0, sum_y2 = 0.0;
            for (size_t k = 0; k < n_ac; ++k) {
                double x = ac_returns_[k];
                double y = ac_returns_[k + 1];
                sum_x += x; sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x; sum_y2 += y * y;
            }
            double fn = static_cast<double>(n_ac);
            double cov_ac = (sum_xy - sum_x * sum_y / fn) / fn;
            double sx = std::sqrt(std::max(0.0, (sum_x2 - sum_x * sum_x / fn) / fn) + 1e-10);
            double sy = std::sqrt(std::max(0.0, (sum_y2 - sum_y * sum_y / fn) / fn) + 1e-10);
            ac1 = cov_ac / (sx * sy + 1e-10);
        }
        row[FROM_FEAT_AUTOCORR_LAG1] = static_cast<float>(std::clamp(ac1, -1.0, 1.0));

        // Feature 26: Volume Clock Deviation
        ema_tick_rate_ = EMA_ALPHA * rate + (1.0 - EMA_ALPHA) * ema_tick_rate_;
        ema_tick_sq_ = EMA_ALPHA * rate * rate + (1.0 - EMA_ALPHA) * ema_tick_sq_;
        double ema_var = ema_tick_sq_ - ema_tick_rate_ * ema_tick_rate_;
        double ema_std = std::sqrt(std::max(ema_var, 0.0) + 1e-10);
        double vcd = (rate - ema_tick_rate_) / ema_std;
        row[FROM_FEAT_VOL_CLOCK_DEV] = static_cast<float>(std::clamp(vcd, -5.0, 5.0));

        prev_ofi = ofi;
        prev_vel = velocity;
        prev_dir = dir;
        prev_mid_ = mid;
    }

    // Update global state
    if (chunk.size > 0) {
        prev_ofi_ = (chunk.ask_vol[chunk.size-1] - chunk.bid_vol[chunk.size-1]) /
                    (chunk.ask_vol[chunk.size-1] + chunk.bid_vol[chunk.size-1] + 1e-8);
        prev_velocity_ = prev_vel;
        prev_dir_ = prev_dir;
    }

    return out;
}

}  // namespace from
