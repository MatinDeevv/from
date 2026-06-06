#pragma once

#include "common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace from {

template <class T>
class Tensor {
    std::shared_ptr<std::vector<T>> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;
    std::shared_ptr<std::mutex> grad_mutex_;

    static size_t product(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    }

    static std::vector<size_t> make_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size(), 1);
        if (!shape.empty()) {
            for (size_t i = shape.size() - 1; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
            }
        }
        return strides;
    }

    size_t physical_offset(size_t flat_index) const {
        if (is_contiguous()) {
            return offset_ + flat_index;
        }
        size_t rem = flat_index;
        size_t off = offset_;
        for (size_t d = 0; d < shape_.size(); ++d) {
            size_t idx = rem / contiguous_stride(d);
            rem %= contiguous_stride(d);
            off += idx * strides_[d];
        }
        return off;
    }

    size_t contiguous_stride(size_t dim) const {
        size_t s = 1;
        for (size_t i = dim + 1; i < shape_.size(); ++i) {
            s *= shape_[i];
        }
        return s;
    }

    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t nd = std::max(a.size(), b.size());
        std::vector<size_t> out(nd, 1);
        for (size_t r = 0; r < nd; ++r) {
            size_t ai = (r < nd - a.size()) ? 1 : a[r - (nd - a.size())];
            size_t bi = (r < nd - b.size()) ? 1 : b[r - (nd - b.size())];
            if (ai != bi && ai != 1 && bi != 1) {
                throw std::invalid_argument("Tensor broadcast shape mismatch");
            }
            out[r] = std::max(ai, bi);
        }
        return out;
    }

    T broadcast_value_at(const std::vector<size_t>& out_index, const std::vector<size_t>& out_shape) const {
        (void)out_shape;
        size_t nd = out_index.size();
        size_t rank = shape_.size();
        size_t off = offset_;
        for (size_t d = 0; d < rank; ++d) {
            size_t out_d = nd - rank + d;
            size_t idx = shape_[d] == 1 ? 0 : out_index[out_d];
            off += idx * strides_[d];
        }
        return (*storage_)[off];
    }

    static std::vector<size_t> unravel(size_t flat, const std::vector<size_t>& shape) {
        std::vector<size_t> idx(shape.size(), 0);
        for (size_t d = 0; d < shape.size(); ++d) {
            size_t stride = 1;
            for (size_t k = d + 1; k < shape.size(); ++k) {
                stride *= shape[k];
            }
            idx[d] = flat / stride;
            flat %= stride;
        }
        return idx;
    }

    Tensor binary_op(const Tensor& other, const std::function<T(T, T)>& fn) const {
        std::vector<size_t> out_shape = broadcast_shape(shape_, other.shape_);
        Tensor out(out_shape);
        size_t n = out.numel();
        for (size_t i = 0; i < n; ++i) {
            auto idx = unravel(i, out_shape);
            out[i] = fn(broadcast_value_at(idx, out_shape), other.broadcast_value_at(idx, out_shape));
        }
        return out;
    }

public:
    bool requires_grad = false;
    std::shared_ptr<Tensor<T>> grad;
    std::function<void()> grad_fn;
    std::vector<std::shared_ptr<Tensor<T>>> parents;

    Tensor()
        : storage_(std::make_shared<std::vector<T>>()), grad_mutex_(std::make_shared<std::mutex>()) {}

    explicit Tensor(const std::vector<size_t>& shape)
        : storage_(std::make_shared<std::vector<T>>(product(shape), T{})),
          shape_(shape),
          strides_(make_strides(shape)),
          grad_mutex_(std::make_shared<std::mutex>()) {}

    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
        : storage_(std::make_shared<std::vector<T>>(data)),
          shape_(shape),
          strides_(make_strides(shape)),
          grad_mutex_(std::make_shared<std::mutex>()) {
        require(data.size() == product(shape), "Tensor data size does not match shape");
    }

    Tensor(std::shared_ptr<std::vector<T>> storage, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
        : storage_(std::move(storage)),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          offset_(offset),
          grad_mutex_(std::make_shared<std::mutex>()) {}

    static Tensor zeros(const std::vector<size_t>& shape) { return Tensor(shape); }

    static Tensor ones(const std::vector<size_t>& shape) {
        Tensor t(shape);
        std::fill(t.storage_->begin(), t.storage_->end(), T{1});
        return t;
    }

    static Tensor full(const std::vector<size_t>& shape, T value) {
        Tensor t(shape);
        std::fill(t.storage_->begin(), t.storage_->end(), value);
        return t;
    }

    static Tensor arange(size_t n) {
        Tensor t({n});
        for (size_t i = 0; i < n; ++i) {
            t[i] = static_cast<T>(i);
        }
        return t;
    }

    static Tensor linspace(T start, T end, size_t n) {
        Tensor t({n});
        if (n == 1) {
            t[0] = start;
            return t;
        }
        for (size_t i = 0; i < n; ++i) {
            T a = static_cast<T>(i) / static_cast<T>(n - 1);
            t[i] = start * (T{1} - a) + end * a;
        }
        return t;
    }

    static Tensor randn(const std::vector<size_t>& shape, T mean = T{}, T stddev = T{1}, uint64_t seed = 1) {
        Tensor t(shape);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(static_cast<double>(mean), static_cast<double>(stddev));
        for (size_t i = 0; i < t.numel(); ++i) {
            t[i] = static_cast<T>(dist(rng));
        }
        return t;
    }

    static Tensor rand_uniform(const std::vector<size_t>& shape, T low = T{}, T high = T{1}, uint64_t seed = 1) {
        Tensor t(shape);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(static_cast<double>(low), static_cast<double>(high));
        for (size_t i = 0; i < t.numel(); ++i) {
            t[i] = static_cast<T>(dist(rng));
        }
        return t;
    }

    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    size_t numel() const { return product(shape_); }
    bool empty() const { return numel() == 0; }

    bool is_contiguous() const {
        return strides_ == make_strides(shape_);
    }

    Tensor contiguous() const {
        Tensor out(shape_);
        for (size_t i = 0; i < numel(); ++i) {
            out[i] = (*this)[i];
        }
        return out;
    }

    T* data_ptr() {
        if (!is_contiguous()) {
            throw std::runtime_error("data_ptr requires contiguous tensor");
        }
        return storage_->data() + offset_;
    }

    const T* data_ptr() const {
        if (!is_contiguous()) {
            throw std::runtime_error("data_ptr requires contiguous tensor");
        }
        return storage_->data() + offset_;
    }

    T& operator[](size_t flat_index) {
        return (*storage_)[physical_offset(flat_index)];
    }

    const T& operator[](size_t flat_index) const {
        return (*storage_)[physical_offset(flat_index)];
    }

    T& at(size_t i) { return (*storage_)[offset_ + i * strides_[0]]; }
    const T& at(size_t i) const { return (*storage_)[offset_ + i * strides_[0]]; }

    T& at(size_t i, size_t j) { return (*storage_)[offset_ + i * strides_[0] + j * strides_[1]]; }
    const T& at(size_t i, size_t j) const { return (*storage_)[offset_ + i * strides_[0] + j * strides_[1]]; }

    T& at(size_t i, size_t j, size_t k) { return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]]; }
    const T& at(size_t i, size_t j, size_t k) const { return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2]]; }

    T& at(size_t i, size_t j, size_t k, size_t l) {
        return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }
    const T& at(size_t i, size_t j, size_t k, size_t l) const {
        return (*storage_)[offset_ + i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    Tensor reshape(const std::vector<size_t>& new_shape) const {
        require(product(new_shape) == numel(), "reshape element count mismatch");
        require(is_contiguous(), "reshape requires contiguous tensor");
        return Tensor(storage_, new_shape, make_strides(new_shape), offset_);
    }

    Tensor view(const std::vector<size_t>& new_shape) const { return reshape(new_shape); }

    Tensor squeeze(size_t dim) const {
        require(dim < shape_.size() && shape_[dim] == 1, "squeeze dimension must have size 1");
        auto sh = shape_;
        auto st = strides_;
        sh.erase(sh.begin() + static_cast<long long>(dim));
        st.erase(st.begin() + static_cast<long long>(dim));
        return Tensor(storage_, sh, st, offset_);
    }

    Tensor unsqueeze(size_t dim) const {
        require(dim <= shape_.size(), "unsqueeze dimension out of range");
        auto sh = shape_;
        sh.insert(sh.begin() + static_cast<long long>(dim), 1);
        return Tensor(storage_, sh, make_strides(sh), offset_);
    }

    Tensor permute(const std::vector<size_t>& order) const {
        require(order.size() == shape_.size(), "permute rank mismatch");
        std::vector<size_t> sh(order.size()), st(order.size());
        std::vector<int> seen(order.size(), 0);
        for (size_t i = 0; i < order.size(); ++i) {
            require(order[i] < order.size() && !seen[order[i]], "invalid permute order");
            seen[order[i]] = 1;
            sh[i] = shape_[order[i]];
            st[i] = strides_[order[i]];
        }
        return Tensor(storage_, sh, st, offset_);
    }

    Tensor T2() const {
        require(shape_.size() == 2, "T2 requires 2D tensor");
        return permute({1, 0});
    }

    Tensor slice(size_t dim, size_t start, size_t end) const {
        require(dim < shape_.size() && start <= end && end <= shape_[dim], "invalid slice");
        auto sh = shape_;
        sh[dim] = end - start;
        return Tensor(storage_, sh, strides_, offset_ + start * strides_[dim]);
    }

    Tensor select(size_t dim, size_t index) const {
        require(dim < shape_.size() && index < shape_[dim], "invalid select");
        auto sh = shape_;
        auto st = strides_;
        size_t off = offset_ + index * strides_[dim];
        sh.erase(sh.begin() + static_cast<long long>(dim));
        st.erase(st.begin() + static_cast<long long>(dim));
        return Tensor(storage_, sh, st, off);
    }

    Tensor operator+(const Tensor& other) const { return binary_op(other, [](T a, T b) { return a + b; }); }
    Tensor operator-(const Tensor& other) const { return binary_op(other, [](T a, T b) { return a - b; }); }
    Tensor operator*(const Tensor& other) const { return binary_op(other, [](T a, T b) { return a * b; }); }
    Tensor operator/(const Tensor& other) const { return binary_op(other, [](T a, T b) { return a / (b + static_cast<T>(FROM_EPS_D)); }); }

    Tensor operator+(T s) const { return unary([s](T a) { return a + s; }); }
    Tensor operator-(T s) const { return unary([s](T a) { return a - s; }); }
    Tensor operator*(T s) const { return unary([s](T a) { return a * s; }); }
    Tensor operator/(T s) const { return unary([s](T a) { return a / (s + static_cast<T>(FROM_EPS_D)); }); }

    Tensor unary(const std::function<T(T)>& fn) const {
        Tensor out(shape_);
        for (size_t i = 0; i < numel(); ++i) {
            out[i] = fn((*this)[i]);
        }
        return out;
    }

    Tensor neg() const { return unary([](T x) { return -x; }); }
    Tensor abs() const { return unary([](T x) { return std::abs(x); }); }
    Tensor sqrt() const { return unary([](T x) { return std::sqrt(std::max<T>(x, T{}) + static_cast<T>(FROM_EPS_D)); }); }
    Tensor exp() const { return unary([](T x) { return std::exp(x); }); }
    Tensor log() const { return unary([](T x) { return std::log(std::max<T>(x, T{}) + static_cast<T>(FROM_EPS_D)); }); }
    Tensor pow(T p) const { return unary([p](T x) { return std::pow(x, p); }); }
    Tensor clamp(T lo, T hi) const { return unary([lo, hi](T x) { return clamp_value(x, lo, hi); }); }

    Tensor reduce(size_t dim, const std::function<T(const std::vector<T>&)>& fn) const {
        require(dim < shape_.size(), "reduce dimension out of range");
        std::vector<size_t> out_shape = shape_;
        size_t red = out_shape[dim];
        out_shape.erase(out_shape.begin() + static_cast<long long>(dim));
        if (out_shape.empty()) {
            out_shape.push_back(1);
        }
        Tensor out(out_shape);
        for (size_t i = 0; i < out.numel(); ++i) {
            auto out_idx = unravel(i, out_shape);
            std::vector<T> vals(red);
            for (size_t r = 0; r < red; ++r) {
                std::vector<size_t> full(shape_.size(), 0);
                for (size_t d = 0, od = 0; d < shape_.size(); ++d) {
                    if (d == dim) {
                        full[d] = r;
                    } else {
                        full[d] = out_idx[od++];
                    }
                }
                size_t off = offset_;
                for (size_t d = 0; d < shape_.size(); ++d) {
                    off += full[d] * strides_[d];
                }
                vals[r] = (*storage_)[off];
            }
            out[i] = fn(vals);
        }
        return out;
    }

    Tensor sum(size_t dim) const {
        return reduce(dim, [](const std::vector<T>& v) {
            return std::accumulate(v.begin(), v.end(), T{});
        });
    }

    Tensor mean(size_t dim) const {
        return reduce(dim, [](const std::vector<T>& v) {
            return std::accumulate(v.begin(), v.end(), T{}) / static_cast<T>(v.size());
        });
    }

    Tensor max(size_t dim) const {
        return reduce(dim, [](const std::vector<T>& v) {
            return *std::max_element(v.begin(), v.end());
        });
    }

    Tensor min(size_t dim) const {
        return reduce(dim, [](const std::vector<T>& v) {
            return *std::min_element(v.begin(), v.end());
        });
    }

    Tensor var(size_t dim) const {
        return reduce(dim, [](const std::vector<T>& v) {
            T m = std::accumulate(v.begin(), v.end(), T{}) / static_cast<T>(v.size());
            T acc = T{};
            for (T x : v) {
                T d = x - m;
                acc += d * d;
            }
            return acc / static_cast<T>(std::max<size_t>(1, v.size() - 1));
        });
    }

    Tensor std(size_t dim) const { return var(dim).sqrt(); }

    Tensor sum() const {
        Tensor out({1});
        out[0] = accumulate_values();
        return out;
    }

    T accumulate_values() const {
        T s = T{};
        for (size_t i = 0; i < numel(); ++i) {
            s += (*this)[i];
        }
        return s;
    }

    Tensor mean() const {
        Tensor out({1});
        out[0] = accumulate_values() / static_cast<T>(std::max<size_t>(1, numel()));
        return out;
    }

    Tensor max() const {
        Tensor out({1});
        out[0] = (*this)[0];
        for (size_t i = 1; i < numel(); ++i) {
            out[0] = std::max(out[0], (*this)[i]);
        }
        return out;
    }

    Tensor min() const {
        Tensor out({1});
        out[0] = (*this)[0];
        for (size_t i = 1; i < numel(); ++i) {
            out[0] = std::min(out[0], (*this)[i]);
        }
        return out;
    }

    Tensor kahan_sum() const {
        Tensor c = contiguous();
        Tensor out({1});
        if constexpr (std::is_same_v<T, float>) {
            out[0] = from_kahan_sum_f32(c.data_ptr(), c.numel());
        } else if constexpr (std::is_same_v<T, double>) {
            out[0] = from_kahan_sum_f64(c.data_ptr(), c.numel());
        } else {
            out[0] = c.accumulate_values();
        }
        return out;
    }

    Tensor kahan_mean() const {
        Tensor out = kahan_sum();
        out[0] /= static_cast<T>(std::max<size_t>(1, numel()));
        return out;
    }

    Tensor matmul(const Tensor& b) const {
        require(shape_.size() == 2 && b.shape_.size() == 2, "matmul requires 2D tensors");
        size_t m = shape_[0], k = shape_[1], n = b.shape_[1];
        require(k == b.shape_[0], "matmul dimension mismatch");
        Tensor a_cont = contiguous();
        Tensor b_cont = b.contiguous();
        Tensor out({m, n});
        if constexpr (std::is_same_v<T, float>) {
            from_gemm_tile_f32(a_cont.data_ptr(), b_cont.data_ptr(), out.data_ptr(), m, k, n, k, n, n);
        } else {
            for (size_t i = 0; i < m; ++i) {
                for (size_t p = 0; p < k; ++p) {
                    T av = a_cont.at(i, p);
                    for (size_t j = 0; j < n; ++j) {
                        out.at(i, j) += av * b_cont.at(p, j);
                    }
                }
            }
        }
        return out;
    }

    Tensor bmm(const Tensor& b) const {
        require(shape_.size() == 3 && b.shape_.size() == 3, "bmm requires 3D tensors");
        require(shape_[0] == b.shape_[0] && shape_[2] == b.shape_[1], "bmm dimension mismatch");
        Tensor out({shape_[0], shape_[1], b.shape_[2]});
        for (size_t batch = 0; batch < shape_[0]; ++batch) {
            Tensor a2 = select(0, batch).contiguous();
            Tensor b2 = b.select(0, batch).contiguous();
            Tensor c = a2.matmul(b2);
            for (size_t i = 0; i < shape_[1]; ++i) {
                for (size_t j = 0; j < b.shape_[2]; ++j) {
                    out.at(batch, i, j) = c.at(i, j);
                }
            }
        }
        return out;
    }

    Tensor outer(const Tensor& v) const {
        require(shape_.size() == 1 && v.shape_.size() == 1, "outer requires 1D tensors");
        Tensor out({shape_[0], v.shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < v.shape_[0]; ++j) {
                out.at(i, j) = at(i) * v.at(j);
            }
        }
        return out;
    }

    Tensor dot(const Tensor& v) const {
        require(shape_.size() == 1 && v.shape_.size() == 1 && shape_[0] == v.shape_[0], "dot requires equal 1D tensors");
        Tensor out({1});
        T acc = T{};
        for (size_t i = 0; i < shape_[0]; ++i) {
            acc += at(i) * v.at(i);
        }
        out[0] = acc;
        return out;
    }

    Tensor norm(int p = 2) const {
        Tensor out({1});
        T acc = T{};
        if (p == 1) {
            for (size_t i = 0; i < numel(); ++i) {
                acc += std::abs((*this)[i]);
            }
            out[0] = acc;
        } else {
            for (size_t i = 0; i < numel(); ++i) {
                acc += (*this)[i] * (*this)[i];
            }
            out[0] = std::sqrt(acc + static_cast<T>(FROM_EPS_D));
        }
        return out;
    }

    Tensor softmax(size_t dim) const {
        require(dim < shape_.size(), "softmax dimension out of range");
        Tensor out(shape_);
        size_t inner = 1;
        for (size_t i = dim + 1; i < shape_.size(); ++i) {
            inner *= shape_[i];
        }
        size_t outer = numel() / (shape_[dim] * inner);
        for (size_t o = 0; o < outer; ++o) {
            for (size_t in = 0; in < inner; ++in) {
                T mx = -std::numeric_limits<T>::infinity();
                for (size_t d = 0; d < shape_[dim]; ++d) {
                    mx = std::max(mx, (*this)[o * shape_[dim] * inner + d * inner + in]);
                }
                T den = T{};
                for (size_t d = 0; d < shape_[dim]; ++d) {
                    den += std::exp((*this)[o * shape_[dim] * inner + d * inner + in] - mx);
                }
                for (size_t d = 0; d < shape_[dim]; ++d) {
                    out[o * shape_[dim] * inner + d * inner + in] =
                        std::exp((*this)[o * shape_[dim] * inner + d * inner + in] - mx) /
                        (den + static_cast<T>(FROM_EPS_D));
                }
            }
        }
        return out;
    }

    Tensor log_softmax(size_t dim) const {
        Tensor sm = softmax(dim);
        return sm.log();
    }

    Tensor layer_norm(size_t dim, T eps = static_cast<T>(1e-5)) const {
        require(dim < shape_.size(), "layer_norm dimension out of range");
        Tensor out(shape_);
        size_t inner = 1;
        for (size_t i = dim + 1; i < shape_.size(); ++i) {
            inner *= shape_[i];
        }
        size_t outer = numel() / (shape_[dim] * inner);
        for (size_t o = 0; o < outer; ++o) {
            for (size_t in = 0; in < inner; ++in) {
                T mean_v = T{};
                for (size_t d = 0; d < shape_[dim]; ++d) {
                    mean_v += (*this)[o * shape_[dim] * inner + d * inner + in];
                }
                mean_v /= static_cast<T>(shape_[dim]);
                T var_v = T{};
                for (size_t d = 0; d < shape_[dim]; ++d) {
                    T x = (*this)[o * shape_[dim] * inner + d * inner + in] - mean_v;
                    var_v += x * x;
                }
                var_v /= static_cast<T>(shape_[dim]);
                T inv = T{1} / std::sqrt(var_v + eps);
                for (size_t d = 0; d < shape_[dim]; ++d) {
                    out[o * shape_[dim] * inner + d * inner + in] =
                        ((*this)[o * shape_[dim] * inner + d * inner + in] - mean_v) * inv;
                }
            }
        }
        return out;
    }

    Tensor pad(size_t pad_size, T value = T{}) const {
        require(shape_.size() >= 2, "pad expects at least 2D tensor");
        std::vector<size_t> sh = shape_;
        size_t seq_dim = sh.size() - 2;
        sh[seq_dim] += pad_size;
        Tensor out = Tensor::full(sh, value);
        for (size_t i = 0; i < numel(); ++i) {
            auto idx = unravel(i, shape_);
            idx[seq_dim] += pad_size;
            size_t off = 0;
            auto st = make_strides(sh);
            for (size_t d = 0; d < sh.size(); ++d) {
                off += idx[d] * st[d];
            }
            out[off] = (*this)[i];
        }
        return out;
    }

    Tensor unfold(size_t kernel_size, size_t stride) const {
        require(shape_.size() == 2, "unfold expects [seq, channels]");
        size_t seq = shape_[0], channels = shape_[1];
        require(kernel_size <= seq && stride > 0, "invalid unfold parameters");
        size_t windows = 1 + (seq - kernel_size) / stride;
        Tensor out({windows, kernel_size * channels});
        for (size_t w = 0; w < windows; ++w) {
            for (size_t k = 0; k < kernel_size; ++k) {
                for (size_t c = 0; c < channels; ++c) {
                    out.at(w, k * channels + c) = at(w * stride + k, c);
                }
            }
        }
        return out;
    }

    void accumulate_grad(const Tensor& g) {
        std::lock_guard<std::mutex> lock(*grad_mutex_);
        if (!grad) {
            grad = std::make_shared<Tensor<T>>(shape_);
        }
        require(g.numel() == grad->numel(), "gradient shape mismatch");
        for (size_t i = 0; i < g.numel(); ++i) {
            (*grad)[i] += g[i];
        }
    }

    void zero_grad() {
        std::lock_guard<std::mutex> lock(*grad_mutex_);
        if (grad) {
            for (size_t i = 0; i < grad->numel(); ++i) {
                (*grad)[i] = T{};
            }
        }
    }

    void backward() {
        require(numel() == 1, "backward requires scalar tensor");
        accumulate_grad(Tensor::ones(shape_));
        if (grad_fn) {
            grad_fn();
        }
    }
};

template <class T>
Tensor<T> operator+(T s, const Tensor<T>& t) { return t + s; }

template <class T>
Tensor<T> operator*(T s, const Tensor<T>& t) { return t * s; }

}  // namespace from
