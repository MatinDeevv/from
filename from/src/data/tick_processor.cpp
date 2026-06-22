#include "data/tick_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

    // -------------------------------------------------------------------------
    // DEFENSIVE INGESTION GUARD (data-no-nan-inf-guard)
    // Freshly-downloaded Dukascopy ticks can contain malformed rows (non-finite
    // or non-positive ask/bid/mid, negative/NaN volumes). A single bad tick
    // poisons log-returns -> Welford normalizer -> ALL subsequent features.
    // Repair each tick by carrying forward the last valid value; clamp volumes.
    // For valid data the sanitized buffers are bit-identical to the raw input,
    // so valid-data behavior is unchanged.
    // -------------------------------------------------------------------------
    std::vector<double> s_ask(chunk.size);
    std::vector<double> s_bid(chunk.size);
    std::vector<double> s_mid(chunk.size);
    std::vector<float>  s_av(chunk.size);
    std::vector<float>  s_bv(chunk.size);

    double last_ask = 0.0, last_bid = 0.0, last_mid = 0.0;
    bool have_valid = false;
    size_t repaired_ticks = 0;

    for (size_t i = 0; i < chunk.size; ++i) {
        double ask = chunk.ask[i];
        double bid = chunk.bid[i];
        double mid = chunk.mid[i];
        // Derive mid from ask/bid if absent (matches original (ask+bid)*0.5 path).
        if (!(std::isfinite(mid) && mid > 0.0)) {
            mid = (ask + bid) * 0.5;
        }

        bool ask_ok = std::isfinite(ask) && ask > 0.0;
        bool bid_ok = std::isfinite(bid) && bid > 0.0;
        bool mid_ok = std::isfinite(mid) && mid > 0.0;
        bool repaired = false;

        if (!ask_ok) { ask = have_valid ? last_ask : 0.0; repaired = true; }
        if (!bid_ok) { bid = have_valid ? last_bid : 0.0; repaired = true; }
        if (!mid_ok) {
            // Try to rebuild mid from any now-valid ask/bid, else carry forward.
            double rebuilt = (ask + bid) * 0.5;
            mid = (std::isfinite(rebuilt) && rebuilt > 0.0)
                      ? rebuilt
                      : (have_valid ? last_mid : 0.0);
            repaired = true;
        }

        // Volumes: clamp NaN/Inf/negative to 0 (these never carry forward).
        float av = chunk.ask_vol[i];
        float bv = chunk.bid_vol[i];
        if (!(std::isfinite(av) && av >= 0.0f)) { av = 0.0f; repaired = true; }
        if (!(std::isfinite(bv) && bv >= 0.0f)) { bv = 0.0f; repaired = true; }

        if (repaired) ++repaired_ticks;

        // A tick is "valid" for carry-forward once all three prices are positive.
        if (ask > 0.0 && bid > 0.0 && mid > 0.0) {
            last_ask = ask;
            last_bid = bid;
            last_mid = mid;
            have_valid = true;
        }

        s_ask[i] = ask;
        s_bid[i] = bid;
        s_mid[i] = mid;
        s_av[i] = av;
        s_bv[i] = bv;
    }

    if (repaired_ticks > 0) {
        std::fprintf(stderr,
                     "[tick_processor] repaired %zu/%zu malformed tick(s) in chunk\n",
                     repaired_ticks, chunk.size);
    }

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
        double ask = s_ask[i];
        double bid = s_bid[i];
        double mid = s_mid[i];
        double av = s_av[i];
        double bv = s_bv[i];
        double spread = ask - bid;
        double norm_spread = spread / (mid + 1e-5);
        double ofi = (av - bv) / (av + bv + 1e-8);

        // Compute time delta
        double dt = 0.1;
        if (i > 0) {
            dt = std::max(0.001, (chunk.time_ms[i] - chunk.time_ms[i - 1]) / 1000.0);
        }

        // Compute velocity (use previous mid from global state for i=0)
        double prev_m = i > 0 ? s_mid[i-1] : prev_mid_;
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
        double mid = s_mid[i];
        double av = s_av[i];
        double bv = s_bv[i];
        double ofi = (av - bv) / (av + bv + 1e-8);
        double norm_spread = out.spread[i] / (mid + 1e-5);

        double prev_m = i > 0 ? s_mid[i-1] : prev_mid_;
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

        // ============================================================
        // 100-FEATURE EXPANSION (features 27..126). Strictly causal: every
        // value derives from ticks <= i. A bounded backward scan over hist_
        // (<=1024 rows) collects windowed aggregates; EMA/RSI/run-length use
        // persistent running state that survives across chunks. One-time cost
        // at preprocessing (results are cached downstream).
        // ============================================================
        {
            double signed_vol_now = av - bv;          // order-flow imbalance
            double vol_now        = av + bv;
            double micro_dev_now  = row[FROM_FEAT_MICRO_DEV];
            double amihud_now     = row[FROM_FEAT_AMIHUD];
            double kyle_now       = row[FROM_FEAT_KYLE_HASBROUCK];
            double spread_now     = out.spread[i];
            double inv_mid        = 1.0 / (mid + 1e-9);

            // ---- persistent EMA-of-mid (spans 16/64/256/1024) ----
            static const double ema_span[4] = {16.0, 64.0, 256.0, 1024.0};
            if (!ema_init_) { for (int s = 0; s < 4; ++s) ema_mid_[s] = mid; ema_init_ = true; }
            for (int s = 0; s < 4; ++s) {
                double a = 2.0 / (ema_span[s] + 1.0);
                ema_mid_[s] += a * (mid - ema_mid_[s]);
            }
            row[FROM_FEAT_EMADEV_16]     = static_cast<float>((mid - ema_mid_[0]) * inv_mid);
            row[FROM_FEAT_EMADEV_64]     = static_cast<float>((mid - ema_mid_[1]) * inv_mid);
            row[FROM_FEAT_EMADEV_256]    = static_cast<float>((mid - ema_mid_[2]) * inv_mid);
            row[FROM_FEAT_EMADEV_1024]   = static_cast<float>((mid - ema_mid_[3]) * inv_mid);
            row[FROM_FEAT_EMAX_16_64]    = static_cast<float>((ema_mid_[0] - ema_mid_[1]) * inv_mid);
            row[FROM_FEAT_EMAX_64_256]   = static_cast<float>((ema_mid_[1] - ema_mid_[2]) * inv_mid);
            row[FROM_FEAT_EMAX_256_1024] = static_cast<float>((ema_mid_[2] - ema_mid_[3]) * inv_mid);

            // ---- Wilder RSI (periods 14/32/64), scaled to [0,1] ----
            static const double rsi_p[3] = {14.0, 32.0, 64.0};
            double gain = dmid > 0.0 ? dmid : 0.0;
            double loss = dmid < 0.0 ? -dmid : 0.0;
            if (!rsi_init_) { for (int s = 0; s < 3; ++s) { rsi_ag_[s] = gain; rsi_al_[s] = loss; } rsi_init_ = true; }
            for (int s = 0; s < 3; ++s) {
                double a = 1.0 / rsi_p[s];
                rsi_ag_[s] += a * (gain - rsi_ag_[s]);
                rsi_al_[s] += a * (loss - rsi_al_[s]);
            }
            auto rsi_val = [](double ag, double al) { double rs = ag / (al + 1e-12); return 1.0 - 1.0 / (1.0 + rs); };
            row[FROM_FEAT_RSI_14] = static_cast<float>(rsi_val(rsi_ag_[0], rsi_al_[0]));
            row[FROM_FEAT_RSI_32] = static_cast<float>(rsi_val(rsi_ag_[1], rsi_al_[1]));
            row[FROM_FEAT_RSI_64] = static_cast<float>(rsi_val(rsi_ag_[2], rsi_al_[2]));

            // ---- direction run length + accel sign/jerk ----
            if (dir > 0)      dir_run_ = dir_run_ > 0 ? dir_run_ + 1 : 1;
            else if (dir < 0) dir_run_ = dir_run_ < 0 ? dir_run_ - 1 : -1;
            row[FROM_FEAT_DIRRUN]  = static_cast<float>(dir_run_);
            row[FROM_FEAT_ACCSIGN] = static_cast<float>((accel > 0.0) - (accel < 0.0));
            row[FROM_FEAT_JERK]    = static_cast<float>(accel - prev_accel_);
            prev_accel_ = accel;

            // ---- push current tick into the history ring ----
            HistRow hr;
            hr.ret = ret; hr.absret = std::abs(ret); hr.mid = mid; hr.ofi = ofi;
            hr.svol = signed_vol_now; hr.vol = vol_now; hr.vel = velocity; hr.accel = accel;
            hr.spread = spread_now; hr.microdev = micro_dev_now; hr.amihud = amihud_now;
            hr.kyle = kyle_now; hr.rate = rate; hr.bounce = static_cast<float>(bounce); hr.dir = dir;
            hist_.push_back(hr);
            if (hist_.size() > HIST_CAP) hist_.pop_front();

            const size_t H = hist_.size();
            const size_t last = H - 1;

            // running accumulators for the backward scan
            double a_ret=0,a_ret2=0,a_ret3=0,a_ret4=0,a_abs=0;
            double a_ofi=0,a_ofi2=0,a_svol=0,a_vol=0,a_vol2=0;
            double a_spr=0,a_spr2=0,a_vel=0,a_vel2=0,a_acc=0;
            double a_mid=0,a_mid2=0,a_amihud=0,a_kyle=0,a_mdev=0,a_mdev2=0;
            double a_rate=0,a_bounce=0,a_volmid=0,a_buy=0,a_sell=0;
            int    a_dirsum=0,a_up=0;
            double maxmid=mid,minmid=mid;

            // window snapshots: idx 0..7 -> {8,16,32,64,128,256,512,1024}
            static const size_t WIN[8] = {8,16,32,64,128,256,512,1024};
            double s_ret[8]={0},s_ret2[8]={0},s_ret3[8]={0},s_ret4[8]={0},s_abs[8]={0};
            double s_mid[8]={0},s_mid2[8]={0},s_ofi[8]={0},s_ofi2[8]={0},s_svol[8]={0};
            double s_vol[8]={0},s_vol2[8]={0},s_spr[8]={0},s_spr2[8]={0},s_vel[8]={0},s_vel2[8]={0};
            double s_acc[8]={0},s_amihud[8]={0},s_kyle[8]={0},s_mdev[8]={0},s_mdev2[8]={0};
            double s_rate[8]={0},s_bounce[8]={0},s_volmid[8]={0},s_maxmid[8]={0},s_minmid[8]={0};
            double s_buy[8]={0},s_sell[8]={0};
            int    s_dirsum[8]={0},s_up[8]={0};
            size_t s_n[8]={0};
            bool   snapped[8]={false};

            for (size_t k = 1; k <= H; ++k) {
                const HistRow& e = hist_[last - (k - 1)];
                a_ret += e.ret; double r2 = e.ret*e.ret;
                a_ret2 += r2; a_ret3 += r2*e.ret; a_ret4 += r2*r2; a_abs += e.absret;
                a_ofi += e.ofi; a_ofi2 += e.ofi*e.ofi; a_svol += e.svol;
                a_vol += e.vol; a_vol2 += e.vol*e.vol;
                a_spr += e.spread; a_spr2 += e.spread*e.spread;
                a_vel += e.vel; a_vel2 += e.vel*e.vel; a_acc += e.accel;
                a_mid += e.mid; a_mid2 += e.mid*e.mid;
                a_amihud += e.amihud; a_kyle += e.kyle;
                a_mdev += e.microdev; a_mdev2 += e.microdev*e.microdev;
                a_rate += e.rate; a_bounce += e.bounce; a_volmid += e.vol*e.mid;
                a_dirsum += e.dir; if (e.dir > 0) ++a_up;
                if (e.dir > 0) a_buy += e.vol; else if (e.dir < 0) a_sell += e.vol;
                if (e.mid > maxmid) maxmid = e.mid;
                if (e.mid < minmid) minmid = e.mid;
                for (int w = 0; w < 8; ++w) {
                    if (!snapped[w] && k == WIN[w]) {
                        s_ret[w]=a_ret; s_ret2[w]=a_ret2; s_ret3[w]=a_ret3; s_ret4[w]=a_ret4; s_abs[w]=a_abs;
                        s_mid[w]=a_mid; s_mid2[w]=a_mid2; s_ofi[w]=a_ofi; s_ofi2[w]=a_ofi2; s_svol[w]=a_svol;
                        s_vol[w]=a_vol; s_vol2[w]=a_vol2; s_spr[w]=a_spr; s_spr2[w]=a_spr2;
                        s_vel[w]=a_vel; s_vel2[w]=a_vel2; s_acc[w]=a_acc;
                        s_amihud[w]=a_amihud; s_kyle[w]=a_kyle; s_mdev[w]=a_mdev; s_mdev2[w]=a_mdev2;
                        s_rate[w]=a_rate; s_bounce[w]=a_bounce; s_volmid[w]=a_volmid;
                        s_maxmid[w]=maxmid; s_minmid[w]=minmid;
                        s_buy[w]=a_buy; s_sell[w]=a_sell; s_dirsum[w]=a_dirsum; s_up[w]=a_up;
                        s_n[w]=k; snapped[w]=true;
                    }
                }
            }
            // windows longer than available history: fall back to full totals
            for (int w = 0; w < 8; ++w) {
                if (!snapped[w]) {
                    s_ret[w]=a_ret; s_ret2[w]=a_ret2; s_ret3[w]=a_ret3; s_ret4[w]=a_ret4; s_abs[w]=a_abs;
                    s_mid[w]=a_mid; s_mid2[w]=a_mid2; s_ofi[w]=a_ofi; s_ofi2[w]=a_ofi2; s_svol[w]=a_svol;
                    s_vol[w]=a_vol; s_vol2[w]=a_vol2; s_spr[w]=a_spr; s_spr2[w]=a_spr2;
                    s_vel[w]=a_vel; s_vel2[w]=a_vel2; s_acc[w]=a_acc;
                    s_amihud[w]=a_amihud; s_kyle[w]=a_kyle; s_mdev[w]=a_mdev; s_mdev2[w]=a_mdev2;
                    s_rate[w]=a_rate; s_bounce[w]=a_bounce; s_volmid[w]=a_volmid;
                    s_maxmid[w]=maxmid; s_minmid[w]=minmid;
                    s_buy[w]=a_buy; s_sell[w]=a_sell; s_dirsum[w]=a_dirsum; s_up[w]=a_up;
                    s_n[w]=H;
                }
            }

            auto MEAN = [&](double sum, int w) { return s_n[w] ? sum / static_cast<double>(s_n[w]) : 0.0; };
            auto STDV = [&](double sum, double sum2, int w) {
                if (!s_n[w]) return 0.0;
                double m = sum / static_cast<double>(s_n[w]);
                double v = sum2 / static_cast<double>(s_n[w]) - m * m;
                return std::sqrt(v > 0.0 ? v : 0.0);
            };

            // 64-tick lagged / conditional stats (one short loop)
            size_t n64 = H < 64 ? H : 64;
            double bipow=0,c2=0,c5=0,sumr=0,sumo=0,sumv=0,srr=0,soo=0,svv=0,sro=0,srv=0,ret2up=0,ret2dn=0;
            for (size_t j = 0; j < n64; ++j) {
                const HistRow& e = hist_[last - j];
                double r = e.ret;
                sumr += r; sumo += e.ofi; sumv += e.vol;
                srr += r*r; soo += e.ofi*e.ofi; svv += e.vol*e.vol;
                sro += r*e.ofi; srv += r*e.vol;
                if (r > 0.0) ret2up += r*r; else ret2dn += r*r;
                if (j + 1 < n64) bipow += std::abs(r) * std::abs(hist_[last-(j+1)].ret);
                if (j + 2 < n64) c2 += r * hist_[last-(j+2)].ret;
                if (j + 5 < n64) c5 += r * hist_[last-(j+5)].ret;
            }
            double var64 = srr;
            double fn64 = static_cast<double>(n64 ? n64 : 1);
            auto corr = [&](double sxy, double sx, double sy, double sxx, double syy) {
                double cov = fn64*sxy - sx*sy;
                double dx  = fn64*sxx - sx*sx;
                double dy  = fn64*syy - sy*sy;
                double den = std::sqrt((dx > 0 ? dx : 0.0) * (dy > 0 ? dy : 0.0));
                return den > 1e-12 ? std::clamp(cov/den, -1.0, 1.0) : 0.0;
            };

            // 128-tick max drawdown (oldest -> newest)
            size_t n128 = H < 128 ? H : 128;
            double peak = -1e18, mdd = 0.0;
            for (size_t j = n128; j-- > 0; ) {
                double m = hist_[last - j].mid;
                if (m > peak) peak = m;
                double dd = (peak - m) / (peak + 1e-9);
                if (dd > mdd) mdd = dd;
            }

            // ---- moments helper (skew/kurt) ----
            auto skew_w = [&](int w) {
                double n = static_cast<double>(s_n[w] ? s_n[w] : 1);
                double m1=s_ret[w]/n, m2=s_ret2[w]/n, m3=s_ret3[w]/n;
                double var=m2-m1*m1, cm3=m3-3*m1*m2+2*m1*m1*m1;
                return var>1e-18 ? cm3/std::pow(var,1.5) : 0.0;
            };
            auto kurt_w = [&](int w) {
                double n = static_cast<double>(s_n[w] ? s_n[w] : 1);
                double m1=s_ret[w]/n, m2=s_ret2[w]/n, m3=s_ret3[w]/n, m4=s_ret4[w]/n;
                double var=m2-m1*m1, cm4=m4-4*m1*m3+6*m1*m1*m2-3*m1*m1*m1*m1;
                return var>1e-18 ? cm4/(var*var)-3.0 : 0.0;
            };

            // ===== write all expansion columns =====
            row[FROM_FEAT_MOM_8]    = static_cast<float>(s_ret[0]);
            row[FROM_FEAT_MOM_16]   = static_cast<float>(s_ret[1]);
            row[FROM_FEAT_MOM_32]   = static_cast<float>(s_ret[2]);
            row[FROM_FEAT_MOM_64]   = static_cast<float>(s_ret[3]);
            row[FROM_FEAT_MOM_128]  = static_cast<float>(s_ret[4]);
            row[FROM_FEAT_MOM_256]  = static_cast<float>(s_ret[5]);
            row[FROM_FEAT_MOM_512]  = static_cast<float>(s_ret[6]);
            row[FROM_FEAT_MOM_1024] = static_cast<float>(s_ret[7]);

            row[FROM_FEAT_RV_8]    = static_cast<float>(std::sqrt(s_ret2[0]));
            row[FROM_FEAT_RV_32]   = static_cast<float>(std::sqrt(s_ret2[2]));
            row[FROM_FEAT_RV_128]  = static_cast<float>(std::sqrt(s_ret2[4]));
            row[FROM_FEAT_RV_512]  = static_cast<float>(std::sqrt(s_ret2[6]));
            row[FROM_FEAT_RV_1024] = static_cast<float>(std::sqrt(s_ret2[7]));

            row[FROM_FEAT_ABSRET_8]   = static_cast<float>(MEAN(s_abs[0],0));
            row[FROM_FEAT_ABSRET_16]  = static_cast<float>(MEAN(s_abs[1],1));
            row[FROM_FEAT_ABSRET_32]  = static_cast<float>(MEAN(s_abs[2],2));
            row[FROM_FEAT_ABSRET_64]  = static_cast<float>(MEAN(s_abs[3],3));
            row[FROM_FEAT_ABSRET_128] = static_cast<float>(MEAN(s_abs[4],4));

            row[FROM_FEAT_RSTD_64]  = static_cast<float>(STDV(s_ret[3],s_ret2[3],3));
            row[FROM_FEAT_RSTD_256] = static_cast<float>(STDV(s_ret[5],s_ret2[5],5));

            row[FROM_FEAT_SKEW_64]  = static_cast<float>(std::clamp(skew_w(3),-10.0,10.0));
            row[FROM_FEAT_SKEW_256] = static_cast<float>(std::clamp(skew_w(5),-10.0,10.0));
            row[FROM_FEAT_KURT_64]  = static_cast<float>(std::clamp(kurt_w(3),-10.0,50.0));
            row[FROM_FEAT_KURT_256] = static_cast<float>(std::clamp(kurt_w(5),-10.0,50.0));

            row[FROM_FEAT_SHARPE_16]  = static_cast<float>(MEAN(s_ret[1],1)/(STDV(s_ret[1],s_ret2[1],1)+1e-9));
            row[FROM_FEAT_SHARPE_64]  = static_cast<float>(MEAN(s_ret[3],3)/(STDV(s_ret[3],s_ret2[3],3)+1e-9));
            row[FROM_FEAT_SHARPE_256] = static_cast<float>(MEAN(s_ret[5],5)/(STDV(s_ret[5],s_ret2[5],5)+1e-9));

            auto donch = [&](int w){ double rng=s_maxmid[w]-s_minmid[w]; return rng>1e-12?(mid-s_minmid[w])/rng:0.5; };
            row[FROM_FEAT_DONCH_32]  = static_cast<float>(donch(2));
            row[FROM_FEAT_DONCH_128] = static_cast<float>(donch(4));
            row[FROM_FEAT_DONCH_512] = static_cast<float>(donch(6));

            auto midz = [&](int w){ double m=MEAN(s_mid[w],w); double sd=STDV(s_mid[w],s_mid2[w],w); return sd>1e-9?std::clamp((mid-m)/sd,-6.0,6.0):0.0; };
            row[FROM_FEAT_MIDZ_64]   = static_cast<float>(midz(3));
            row[FROM_FEAT_MIDZ_256]  = static_cast<float>(midz(5));
            row[FROM_FEAT_MIDZ_1024] = static_cast<float>(midz(7));

            row[FROM_FEAT_OFIM_8]    = static_cast<float>(MEAN(s_ofi[0],0));
            row[FROM_FEAT_OFIM_32]   = static_cast<float>(MEAN(s_ofi[2],2));
            row[FROM_FEAT_OFIM_128]  = static_cast<float>(MEAN(s_ofi[4],4));
            row[FROM_FEAT_OFISD_32]  = static_cast<float>(STDV(s_ofi[2],s_ofi2[2],2));
            row[FROM_FEAT_OFISD_128] = static_cast<float>(STDV(s_ofi[4],s_ofi2[4],4));

            row[FROM_FEAT_SVOLSUM_32]  = static_cast<float>(s_svol[2]);
            row[FROM_FEAT_SVOLSUM_128] = static_cast<float>(s_svol[4]);

            row[FROM_FEAT_VOLM_32]   = static_cast<float>(MEAN(s_vol[2],2));
            row[FROM_FEAT_VOLM_128]  = static_cast<float>(MEAN(s_vol[4],4));
            row[FROM_FEAT_VOLSD_32]  = static_cast<float>(STDV(s_vol[2],s_vol2[2],2));
            row[FROM_FEAT_VOLSD_128] = static_cast<float>(STDV(s_vol[4],s_vol2[4],4));

            double sprm64 = MEAN(s_spr[3],3), sprsd64 = STDV(s_spr[3],s_spr2[3],3);
            row[FROM_FEAT_SPRM_64]  = static_cast<float>(sprm64);
            row[FROM_FEAT_SPRSD_64] = static_cast<float>(sprsd64);
            row[FROM_FEAT_SPRZ_64]  = static_cast<float>(sprsd64>1e-12?std::clamp((spread_now-sprm64)/sprsd64,-6.0,6.0):0.0);

            row[FROM_FEAT_VELM_32]  = static_cast<float>(MEAN(s_vel[2],2));
            row[FROM_FEAT_VELM_128] = static_cast<float>(MEAN(s_vel[4],4));
            row[FROM_FEAT_VELSD_32] = static_cast<float>(STDV(s_vel[2],s_vel2[2],2));
            row[FROM_FEAT_VELSD_128]= static_cast<float>(STDV(s_vel[4],s_vel2[4],4));

            row[FROM_FEAT_ACCM_32] = static_cast<float>(MEAN(s_acc[2],2));

            row[FROM_FEAT_FRACUP_64] = static_cast<float>(s_n[3]?static_cast<double>(s_up[3])/s_n[3]:0.5);

            row[FROM_FEAT_AC2_64] = static_cast<float>(var64>1e-12?std::clamp(c2/var64,-1.0,1.0):0.0);
            row[FROM_FEAT_AC5_64] = static_cast<float>(var64>1e-12?std::clamp(c5/var64,-1.0,1.0):0.0);

            double ms8 = MEAN(s_ret2[0],0), ms64 = MEAN(s_ret2[3],3);
            row[FROM_FEAT_VARRATIO] = static_cast<float>(ms64>1e-18?std::clamp(ms8/ms64,0.0,10.0):1.0);

            row[FROM_FEAT_AMIHUDM_64] = static_cast<float>(MEAN(s_amihud[3],3));
            row[FROM_FEAT_KYLEM_64]   = static_cast<float>(MEAN(s_kyle[3],3));
            row[FROM_FEAT_MDEVM_32]   = static_cast<float>(MEAN(s_mdev[2],2));
            row[FROM_FEAT_MDEVSD_32]  = static_cast<float>(STDV(s_mdev[2],s_mdev2[2],2));

            double bv64 = 1.5707963267948966 * bipow;  // (pi/2) * bipower variation
            row[FROM_FEAT_JUMP_64]   = static_cast<float>(var64>1e-18?std::clamp((var64-bv64)/var64,0.0,1.0):0.0);
            row[FROM_FEAT_SEMIUP_64] = static_cast<float>(std::sqrt(ret2up));
            row[FROM_FEAT_SEMIDN_64] = static_cast<float>(std::sqrt(ret2dn));

            // ---- time-of-day cyclical + session flags ----
            const int64_t ms_day = 86400000LL;
            int64_t tod = ((chunk.time_ms[i] % ms_day) + ms_day) % ms_day;
            double hour = tod / 3600000.0;
            double minute = (tod % 3600000LL) / 60000.0;
            const double TWO_PI = 6.283185307179586;
            row[FROM_FEAT_TOD_SIN_H] = static_cast<float>(std::sin(TWO_PI*hour/24.0));
            row[FROM_FEAT_TOD_COS_H] = static_cast<float>(std::cos(TWO_PI*hour/24.0));
            row[FROM_FEAT_TOD_SIN_M] = static_cast<float>(std::sin(TWO_PI*minute/60.0));
            row[FROM_FEAT_TOD_COS_M] = static_cast<float>(std::cos(TWO_PI*minute/60.0));
            row[FROM_FEAT_SESS_ASIA]   = (hour >= 23.0 || hour < 7.0)  ? 1.0f : 0.0f;
            row[FROM_FEAT_SESS_LONDON] = (hour >= 7.0  && hour < 16.0) ? 1.0f : 0.0f;
            row[FROM_FEAT_SESS_NY]     = (hour >= 13.0 && hour < 22.0) ? 1.0f : 0.0f;

            row[FROM_FEAT_DSPR_8]  = static_cast<float>(spread_now - MEAN(s_spr[0],0));
            row[FROM_FEAT_DSPR_32] = static_cast<float>(spread_now - MEAN(s_spr[2],2));

            double pu = s_n[3] ? static_cast<double>(s_up[3])/s_n[3] : 0.5;
            double ent = 0.0;
            if (pu > 1e-9 && pu < 1.0-1e-9) ent = -(pu*std::log2(pu) + (1.0-pu)*std::log2(1.0-pu));
            row[FROM_FEAT_SIGNENT_64] = static_cast<float>(ent);

            row[FROM_FEAT_MDD_128] = static_cast<float>(mdd);

            auto vwapd = [&](int w){ double vw=s_vol[w]>1e-12?s_volmid[w]/s_vol[w]:mid; return (mid-vw)*inv_mid; };
            row[FROM_FEAT_VWAPD_64]  = static_cast<float>(vwapd(3));
            row[FROM_FEAT_VWAPD_256] = static_cast<float>(vwapd(5));

            row[FROM_FEAT_CORR_RO_64] = static_cast<float>(corr(sro,sumr,sumo,srr,soo));
            row[FROM_FEAT_CORR_RV_64] = static_cast<float>(corr(srv,sumr,sumv,srr,svv));

            row[FROM_FEAT_RVR_8_64]   = static_cast<float>(std::sqrt(s_ret2[0])/(std::sqrt(s_ret2[3])+1e-12));
            row[FROM_FEAT_RVR_64_512] = static_cast<float>(std::sqrt(s_ret2[3])/(std::sqrt(s_ret2[6])+1e-12));

            row[FROM_FEAT_LR_64]  = static_cast<float>(s_dirsum[3]);
            row[FROM_FEAT_LR_128] = static_cast<float>(s_dirsum[4]);

            auto tir = [&](int w){ double d=s_buy[w]+s_sell[w]; return d>1e-12?(s_buy[w]-s_sell[w])/d:0.0; };
            row[FROM_FEAT_TIR_64]  = static_cast<float>(tir(3));
            row[FROM_FEAT_TIR_128] = static_cast<float>(tir(4));

            row[FROM_FEAT_TRM_64]     = static_cast<float>(MEAN(s_rate[3],3));
            row[FROM_FEAT_BOUNCEF_64] = static_cast<float>(MEAN(s_bounce[3],3));
            row[FROM_FEAT_NRANGE_64]  = static_cast<float>((s_maxmid[3]-s_minmid[3])*inv_mid);
        }

        // FINITE GUARD: clamp every feature column of this row before it is
        // consumed by the Welford normalizer. For valid data all values are
        // already finite, so this is a no-op (no valid-data behavior change).
        for (size_t f = 0; f < FROM_MAX_FEATURES; ++f) {
            if (!std::isfinite(row[f])) row[f] = 0.0f;
        }

        prev_ofi = ofi;
        prev_vel = velocity;
        prev_dir = dir;
        prev_mid_ = mid;
    }

    // Update global state
    if (chunk.size > 0) {
        prev_ofi_ = (s_av[chunk.size-1] - s_bv[chunk.size-1]) /
                    (s_av[chunk.size-1] + s_bv[chunk.size-1] + 1e-8);
        prev_velocity_ = prev_vel;
        prev_dir_ = prev_dir;
    }

    return out;
}

}  // namespace from
