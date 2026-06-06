#pragma once

#include "initializers.hpp"
#include "layers/layer_base.hpp"

#include <cmath>
#include <sstream>

namespace from {

class Conv1D : public ILayer {
    size_t in_channels_;
    size_t out_channels_;
    std::vector<size_t> kernels_{3, 5, 8, 13, 21, 34, 55, 89};
    std::vector<size_t> dilations_{1, 2, 4, 8};

public:
    std::vector<Tensor<float>> weights;
    std::vector<Tensor<float>> grad_weights;
    Tensor<float> bias;
    Tensor<float> grad_bias;

    Conv1D(size_t in_channels = FROM_MAX_FEATURES, size_t out_channels = 64)
        : in_channels_(in_channels),
          out_channels_(out_channels),
          bias(Tensor<float>::zeros({out_channels})),
          grad_bias(Tensor<float>::zeros({out_channels})) {
        size_t per = std::max<size_t>(1, out_channels / kernels_.size());
        size_t made = 0;
        for (size_t k : kernels_) {
            size_t oc = std::min(per, out_channels - made);
            if (k == kernels_.back()) {
                oc = out_channels - made;
            }
            Tensor<float> w({oc, in_channels, k});
            for (size_t o = 0; o < oc; ++o) {
                for (size_t c = 0; c < in_channels; ++c) {
                    for (size_t i = 0; i < k; ++i) {
                        w.at(o, c, i) = 1.0f / static_cast<float>(k * in_channels);
                    }
                }
            }
            weights.push_back(w);
            grad_weights.push_back(Tensor<float>::zeros({oc, in_channels, k}));
            made += oc;
            if (made >= out_channels) {
                break;
            }
        }
    }

    Tensor<float> forward(const Tensor<float>& x, bool training = true) override {
        (void)training;
        require(x.shape().size() == 3, "Conv1D expects [batch, seq, channels]");
        size_t batch = x.shape()[0], seq = x.shape()[1], channels = x.shape()[2];
        require(channels == in_channels_, "Conv1D channel mismatch");
        Tensor<float> out({batch, seq, out_channels_});
        size_t oc_base = 0;
        for (size_t bi = 0; bi < weights.size(); ++bi) {
            const Tensor<float>& w = weights[bi];
            size_t oc = w.shape()[0];
            size_t k = w.shape()[2];
            size_t dilation = dilations_[bi % dilations_.size()];
            for (size_t b = 0; b < batch; ++b) {
                for (size_t t = 0; t < seq; ++t) {
                    for (size_t o = 0; o < oc; ++o) {
                        float acc = bias.at(oc_base + o);
                        for (size_t c = 0; c < channels; ++c) {
                            for (size_t kk = 0; kk < k; ++kk) {
                                size_t lag = (k - 1 - kk) * dilation;
                                if (t >= lag) {
                                    acc += x.at(b, t - lag, c) * w.at(o, c, kk);
                                }
                            }
                        }
                        out.at(b, t, oc_base + o) = acc;
                    }
                }
            }
            oc_base += oc;
        }
        return out;
    }

    std::vector<ParameterRef> parameters() override {
        std::vector<ParameterRef> p;
        for (size_t i = 0; i < weights.size(); ++i) {
            p.push_back({"kernel_" + std::to_string(i), &weights[i], &grad_weights[i], false});
        }
        p.push_back({"bias", &bias, &grad_bias, false});
        return p;
    }

    std::string inspect_kernels() const {
        std::ostringstream os;
        size_t idx = 0;
        for (const auto& w : weights) {
            os << "Kernel bank " << idx << " size=" << w.shape()[2] << "\n";
            for (size_t kk = 0; kk < w.shape()[2]; ++kk) {
                float v = w.at(0, 0, kk);
                int bars = static_cast<int>(std::round(std::abs(v) * 80.0f));
                os << "  " << kk << " " << v << " ";
                for (int b = 0; b < bars; ++b) os << (v >= 0 ? '#' : '-');
                os << "\n";
            }
            os << "  closest classical indicator: SMA/EMA family candidate\n";
            ++idx;
        }
        return os.str();
    }

    std::string name() const override { return "conv1d"; }
};

}  // namespace from
