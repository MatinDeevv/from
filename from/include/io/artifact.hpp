#pragma once
// ============================================================================
// artifact.hpp — committee-ready model artifact writers (walk-forward family).
//
// Both cli/walkforward.cpp (243-dim summary MLP) and cli/wfdeep.cpp (raw-window
// deep MLP) call these inline helpers AFTER the FINAL HOLDOUT model is trained
// and holdout / OOF TradeStats are computed. Each run produces a self-contained
// directory:
//
//   <model_dir>/<id>/
//     model.from / deep.bin   weights (arch-specific; written by the driver)
//     norm1.bin               first-pass Welford Normalizer state (MLP; fixes A1)
//     meta.json               hand-written JSON: config + nested holdout/oof metrics
//     report.txt              the full text report (also still written to --output)
//   <model_dir>/manifest.csv  one appended CSV row per run (header on create)
//
// id = "<arch>_h<horizon>_cm<cost_mult>_bk<barrier_k>_cg<conf_gate>_s<seed>"
// arch = "mlp" | "deep".
//
// These helpers take plain args / small POD structs (NOT either driver's private
// TradeStats type) so both drivers can call them without coupling. Metrics are
// copied field-by-field into ArtifactMetrics at the call site.
//
// No external JSON / serialization libs: everything is hand-written and meant to
// compile cleanly on the GNU / L4 toolchain (<filesystem> already used in both
// drivers). Windows-only bits are guarded.
// ============================================================================

#include "data/normalizer.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace from {
namespace io {

// ----------------------------------------------------------------------------
// Plain metric snapshot (one per segment: holdout, oof). Field-for-field copy
// of the relevant TradeStats members; decoupled from either driver's type.
// ----------------------------------------------------------------------------
struct ArtifactMetrics {
    size_t trades = 0;
    double n_eff = 0.0;
    double winrate = 0.0;
    double edge = 0.0;
    double profit_factor = 0.0;
    double kelly = 0.0;
    double max_drawdown = 0.0;
    double t_stat = 0.0;
    double sharpe = 0.0;
    double ci_lo = 0.0;
    double ci_hi = 0.0;
};

// ----------------------------------------------------------------------------
// Run configuration + identity for meta.json / manifest. Strings are plain.
// `layers` is the deep hidden-layer spec (e.g. "2048,1024,512"); empty for MLP.
// ----------------------------------------------------------------------------
struct ArtifactMeta {
    std::string id;
    std::string arch;            // "mlp" | "deep"
    std::string data;
    size_t window = 0;
    size_t stride = 0;
    size_t horizon = 0;
    double direction_threshold = 0.0;
    double barrier_k = 0.0;
    double cost_mult = 0.0;
    double conf_gate = 0.0;
    uint32_t seed = 0;
    std::string layers;          // deep only ("" for mlp)
    size_t max_samples = 0;
    size_t folds = 0;
    double holdout_frac = 0.0;
    double embargo_frac = 0.0;
    double commission = 0.0;
    double slippage_mult = 0.0;

    ArtifactMetrics holdout;
    ArtifactMetrics oof;
    bool edge_verdict = false;   // holdout ci_lo>0 && kelly>0 && n_eff>=30 && pf>1
};

// ----------------------------------------------------------------------------
// Build the canonical artifact id. Floats are rendered with fixed precision so
// the id is stable / filesystem-safe (no scientific notation, no stray '.').
// ----------------------------------------------------------------------------
inline std::string make_artifact_id(const std::string& arch, size_t horizon,
                                    double cost_mult, double barrier_k,
                                    double conf_gate, uint32_t seed) {
    std::ostringstream os;
    os << arch
       << "_h" << horizon
       << "_cm" << std::fixed << std::setprecision(2) << cost_mult
       << "_bk" << std::setprecision(2) << barrier_k
       << "_cg" << std::setprecision(2) << conf_gate
       << "_s" << seed;
    return os.str();
}

// mkdir -p (no throw on already-exists). Returns true if the dir exists after.
inline bool ensure_dir(const std::filesystem::path& dir) {
    std::error_code ec;
    if (std::filesystem::exists(dir, ec)) return std::filesystem::is_directory(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return std::filesystem::is_directory(dir, ec);
}

// ----------------------------------------------------------------------------
// Save the FIRST-PASS Welford Normalizer state -> dir/"norm1.bin" (magic NRM1).
// Layout (little-endian native, matching serializer.hpp value layout):
//   char[4] "NRM1"
//   uint64  dims
//   double  mean[dims]
//   double  m2[dims]
//   uint64  count[dims]
//   uint8   frozen
// This fixes audit A1: a reloaded MLP can re-derive RAW->normalized inputs.
// Returns false on write failure.
// ----------------------------------------------------------------------------
inline bool save_norm1(const std::filesystem::path& path, const Normalizer& norm) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const char magic[4] = {'N', 'R', 'M', '1'};
    out.write(magic, 4);
    uint64_t dims = static_cast<uint64_t>(norm.mean().size());
    out.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    for (double v : norm.mean())  out.write(reinterpret_cast<const char*>(&v), sizeof(double));
    for (double v : norm.m2())    out.write(reinterpret_cast<const char*>(&v), sizeof(double));
    for (uint64_t v : norm.count()) out.write(reinterpret_cast<const char*>(&v), sizeof(uint64_t));
    uint8_t frozen = norm.frozen() ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&frozen), sizeof(frozen));
    return static_cast<bool>(out);
}

// ----------------------------------------------------------------------------
// Load a FIRST-PASS Welford Normalizer state written by save_norm1 (magic NRM1)
// into `norm` via set_state. Byte-for-byte inverse of save_norm1; mirrors the
// inline reader in cli/walkforward.cpp so deployment (cli/infer.cpp) re-derives
// the SAME raw->normalized mapping the model trained on. Without this, inference
// feeds a default (mean 0 / var 1) normalizer => raw-scale inputs => garbage.
// Returns false if the file is missing, has a bad magic, or is truncated; `norm`
// is left untouched on failure.
// ----------------------------------------------------------------------------
inline bool load_norm1(const std::filesystem::path& path, Normalizer& norm) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4];
    in.read(magic, 4);
    if (!in || magic[0] != 'N' || magic[1] != 'R' || magic[2] != 'M' || magic[3] != '1') return false;
    uint64_t dims = 0;
    in.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    if (!in || dims == 0 || dims > 1024) return false;
    std::vector<double>   mean(dims), m2(dims);
    std::vector<uint64_t> count(dims);
    for (double& v : mean)    in.read(reinterpret_cast<char*>(&v), sizeof(double));
    for (double& v : m2)      in.read(reinterpret_cast<char*>(&v), sizeof(double));
    for (uint64_t& v : count) in.read(reinterpret_cast<char*>(&v), sizeof(uint64_t));
    uint8_t frozen = 0;
    in.read(reinterpret_cast<char*>(&frozen), sizeof(frozen));
    if (!in) return false;
    norm.set_state(std::move(mean), std::move(m2), std::move(count), frozen != 0);
    return true;
}

// ----------------------------------------------------------------------------
// JSON helpers (hand-written; no lib). Escape strings, no trailing commas.
// ----------------------------------------------------------------------------
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Render a double safely for JSON (no NaN/Inf — JSON has no such literals; emit 0).
inline std::string json_num(double v) {
    if (!(v == v) || v == std::numeric_limits<double>::infinity() ||
        v == -std::numeric_limits<double>::infinity()) {
        return "0";
    }
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

// Serialize one metrics block as a nested JSON object. `indent` is the leading
// whitespace for each member line; the object braces use one-less indent level.
inline std::string metrics_json(const ArtifactMetrics& m, const std::string& indent) {
    std::ostringstream os;
    os << "{\n"
       << indent << "  \"trades\": "        << m.trades                 << ",\n"
       << indent << "  \"n_eff\": "         << json_num(m.n_eff)        << ",\n"
       << indent << "  \"winrate\": "       << json_num(m.winrate)      << ",\n"
       << indent << "  \"edge\": "          << json_num(m.edge)         << ",\n"
       << indent << "  \"profit_factor\": " << json_num(m.profit_factor) << ",\n"
       << indent << "  \"kelly\": "         << json_num(m.kelly)        << ",\n"
       << indent << "  \"max_drawdown\": "  << json_num(m.max_drawdown) << ",\n"
       << indent << "  \"t_stat\": "        << json_num(m.t_stat)       << ",\n"
       << indent << "  \"sharpe\": "        << json_num(m.sharpe)       << ",\n"
       << indent << "  \"ci_lo\": "         << json_num(m.ci_lo)        << ",\n"
       << indent << "  \"ci_hi\": "         << json_num(m.ci_hi)        << "\n"
       << indent << "}";
    return os.str();
}

// ----------------------------------------------------------------------------
// Write meta.json into `path`. Valid JSON: strings escaped, no trailing commas.
// Returns false on write failure.
// ----------------------------------------------------------------------------
inline bool write_meta_json(const std::filesystem::path& path, const ArtifactMeta& m) {
    std::ofstream out(path);
    if (!out) return false;
    out << "{\n"
        << "  \"id\": \""                  << json_escape(m.id)   << "\",\n"
        << "  \"arch\": \""                << json_escape(m.arch) << "\",\n"
        << "  \"data\": \""                << json_escape(m.data) << "\",\n"
        << "  \"window\": "                << m.window            << ",\n"
        << "  \"stride\": "                << m.stride            << ",\n"
        << "  \"horizon\": "               << m.horizon           << ",\n"
        << "  \"direction_threshold\": "   << json_num(m.direction_threshold) << ",\n"
        << "  \"barrier_k\": "             << json_num(m.barrier_k)  << ",\n"
        << "  \"cost_mult\": "             << json_num(m.cost_mult)  << ",\n"
        << "  \"conf_gate\": "             << json_num(m.conf_gate)  << ",\n"
        << "  \"seed\": "                  << m.seed              << ",\n"
        << "  \"layers\": \""              << json_escape(m.layers) << "\",\n"
        << "  \"max_samples\": "           << m.max_samples       << ",\n"
        << "  \"folds\": "                 << m.folds             << ",\n"
        << "  \"holdout_frac\": "          << json_num(m.holdout_frac) << ",\n"
        << "  \"embargo_frac\": "          << json_num(m.embargo_frac) << ",\n"
        << "  \"commission\": "            << json_num(m.commission)   << ",\n"
        << "  \"slippage_mult\": "         << json_num(m.slippage_mult) << ",\n"
        << "  \"metrics\": {\n"
        << "    \"holdout\": " << metrics_json(m.holdout, "    ") << ",\n"
        << "    \"oof\": "     << metrics_json(m.oof, "    ")     << "\n"
        << "  },\n"
        << "  \"edge_verdict\": \"" << (m.edge_verdict ? "yes" : "no") << "\"\n"
        << "}\n";
    return static_cast<bool>(out);
}

// ----------------------------------------------------------------------------
// Append one CSV row to <model_dir>/manifest.csv (creating it with the header
// if missing). Concurrent fleet workers each append a single line; the line is
// built fully in memory and written with one write() + flush so interleaving is
// line-atomic in practice. `artifact_dir` is the artifact's own directory path.
//
// Columns:
//   id,arch,horizon,cost_mult,barrier_k,conf_gate,seed,
//   holdout_pf,holdout_kelly,holdout_edge,holdout_t,holdout_neff,holdout_ci_lo,
//   oof_pf,oof_t,edge_verdict,dir
// ----------------------------------------------------------------------------
inline bool append_manifest_row(const std::filesystem::path& manifest_path,
                                const ArtifactMeta& m,
                                const std::string& artifact_dir) {
    static const char* kHeader =
        "id,arch,horizon,cost_mult,barrier_k,conf_gate,seed,"
        "holdout_pf,holdout_kelly,holdout_edge,holdout_t,holdout_neff,holdout_ci_lo,"
        "oof_pf,oof_t,edge_verdict,dir\n";

    std::error_code ec;
    bool need_header = !std::filesystem::exists(manifest_path, ec);

    std::ofstream out(manifest_path, std::ios::out | std::ios::app);
    if (!out) return false;
    if (need_header) out << kHeader;

    auto num = [](double v) { return json_num(v); };  // reuse NaN-safe renderer

    std::ostringstream row;
    row << m.id << ','
        << m.arch << ','
        << m.horizon << ','
        << num(m.cost_mult) << ','
        << num(m.barrier_k) << ','
        << num(m.conf_gate) << ','
        << m.seed << ','
        << num(m.holdout.profit_factor) << ','
        << num(m.holdout.kelly) << ','
        << num(m.holdout.edge) << ','
        << num(m.holdout.t_stat) << ','
        << num(m.holdout.n_eff) << ','
        << num(m.holdout.ci_lo) << ','
        << num(m.oof.profit_factor) << ','
        << num(m.oof.t_stat) << ','
        << (m.edge_verdict ? "yes" : "no") << ','
        << artifact_dir << '\n';

    const std::string line = row.str();
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
    out.flush();
    return static_cast<bool>(out);
}

}  // namespace io
}  // namespace from
