// Range coder implementation, based on [1].
//
// [1] G. N. N. Martin, "Range coding: an algorithm for removing redundancy from
// a digitised message", presented to the Video & Data Recording Conference,
// held in Southampton, July 24-27, 1979.
//

#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <limits>
#include <tuple>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <numeric>
#include <iterator>

#include <bitset>

using cdf_t = int32_t;
/** Encapsulates a pointer to a CDF tensor */
class CDF_package
{
private:
    cdf_t *cdf_ptr;
    int Lp;

public:
    CDF_package(const at::Tensor &cdf)
    {
        TORCH_CHECK(!cdf.is_cuda(), "cdf must be on CPU!");
        const auto s = cdf.sizes();
        TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected NLp");

        const auto cdf_acc = cdf.accessor<cdf_t, 2>();
        cdf_ptr = (cdf_t *)cdf_acc.data();
        Lp = s[1];
    };

    uint32_t index(int idx, int value) const
    {
        return cdf_ptr[idx * Lp + value];
    }

    std::vector<cdf_t> slice(int idx, int len = -1) const
    {
        int offset = idx * Lp;
        if (len < 1)
            len = Lp;
        std::vector<cdf_t> slice_(cdf_ptr + offset, cdf_ptr + offset + len);
        return slice_;
    }

    ~CDF_package()
    {
        cdf_ptr = nullptr;
    }
};
// -----------------------------------------------------------------------------

/** Class to save output bit by bit to a byte string */
class RangeEncoder
{
private:
    uint32_t base_ = 0;
    uint32_t size_minus1_ = std::numeric_limits<uint32_t>::max();
    uint64_t delay_ = 0;
    std::string out = "";

public:
    void Encode(uint32_t lower, uint32_t upper, int precision, int i, bool eq = true)
    {
        // const uint64_t span = static_cast<uint64_t>(size_minus1_) + 1;
        // if (i < 16 or eq)
        // {
        //     printf("{%d} (%llu %u %u)\n", i, span, lower, upper);
        // }
        TORCH_CHECK(precision > 0 && precision <= 16, "Precision must in (0, 16]");
        TORCH_CHECK(0 <= lower && lower < upper && upper <= (1 << precision), "Expect 0 <= lower < upper <= 2^precision");
        const uint64_t size = static_cast<uint64_t>(size_minus1_) + 1;
        TORCH_CHECK((size >> 16) != 0, "Can not encode zero probability");

        const uint32_t a = (size * static_cast<uint64_t>(lower)) >> precision;
        const uint32_t b = ((size * static_cast<uint64_t>(upper)) >> precision) - 1;
        TORCH_CHECK(a < b, "cumFreq contains negative frequencies");

        base_ += a;
        size_minus1_ = b - a;
        const bool base_overflow = (base_ < a);

        if ((base_ + size_minus1_) < base_)
        {
            TORCH_CHECK((((base_ - a) + size)) >> 32 != 0, "tfc172");
            TORCH_CHECK((delay_ & 0xFFFF) != 0, "tfc173");
            if (size_minus1_ >> 16 == 0)
            {
                TORCH_CHECK((base_ >> 16) == 0xFFFF, "tfc197");
                base_ <<= 16;
                size_minus1_ <<= 16;
                size_minus1_ |= 0xFFFF;
                // TODO(ssjhv): It is possible that for very long input, delay
                // overflow during below. If overflow is detected, this delay is too
                // long the encoder should forcefully move to state 0. In such case,
                // base can be raised to 2^32 (force case #2), or (base + size) can be
                // lowered to 2^32 (force case #3), depending on which transition
                // keeps size larger.
                TORCH_CHECK(delay_ < (static_cast<uint64_t>(1) << 62), "tfc207");
                delay_ += 0x20000; // Two more bytes of zeros. Check overflow?
            }
            return;
        }

        // If reached here, the current state is 0.
        // First handle the case when the previous state was state 1.
        if (delay_ != 0)
        {
            // In case #2 or #3, the encoder state changes to state 0. Recall that when
            // the encoder state changed from state 0 to state 1, the top 16 bits of
            // (base + size - 1) was temporarily stored in `delay`, because the output
            // could be either (delay - 1) or (delay).
            //
            // And from above, the delayed value encoded in `delay` is
            //   delay" <- delay[16:0] * 2^(8 * delay[MAX:16])
            //
            // In case #2, the interval moved below 2^32. So (delay" - 1) is the
            // converged value after interval refinements. Write out
            // (delay[16:0] - 1) and write (8 * delay[MAX:16]) bytes of 0xFF.
            //
            // In case #3, the interval moved above 2^32. So delay" is the converged
            // value after interval refinement. Write out delay[16:0] and write
            // (8 * delay[MAX:16]) bytes of 0x00.
            if (base_overflow)
            {
                // Case #2.
                TORCH_CHECK(((static_cast<uint64_t>(base_ - a) + a) >> 32) != 0, "tfc233");
                out.push_back(static_cast<char>(delay_ >> 8));
                out.push_back(static_cast<char>(delay_ >> 0));
                out.append(delay_ >> 16, static_cast<char>(0));
            }
            else
            {
                // Case #3.
                TORCH_CHECK((static_cast<uint64_t>(base_ + size_minus1_) >> 32) == 0, "tfc239");
                --delay_;
                out.push_back(static_cast<char>(delay_ >> 8));
                out.push_back(static_cast<char>(delay_ >> 0));
                out.append(delay_ >> 16, static_cast<char>(0xFF));
            }
            // Reset to state 0.
            delay_ = 0;
        }

        if (size_minus1_ >> 16 == 0)
        {
            const uint32_t top = base_ >> 16;

            base_ <<= 16;
            size_minus1_ <<= 16;
            size_minus1_ |= 0xFFFF;

            if (base_ <= base_ + size_minus1_)
            {
                // Still in state 0. Write the top 16 bits.
                out.push_back(static_cast<char>(top >> 8));
                out.push_back(static_cast<char>(top));
            }
            else
            {
                // New state is 1.
                TORCH_CHECK(top < 0xFFFF, "tfc262");
                delay_ = top + 1;
            }
        }
    }

    void Finalize()
    {
        if (delay_ != 0)
        {
            // The last state was state 1. Since base < 2^32 < base + size, pick 2^32
            // (state 1, case #3).
            // NOTE: It is a bit difficult to trigger this code path on purpose.
            // TODO(ssjhv): Find a way to trigger this code path for test coverage.
            out.push_back(static_cast<char>(delay_ >> 8));
            if ((delay_ & 0xFF) != 0)
            {
                out.push_back(static_cast<char>(delay_));
            }
        }
        else if (base_ != 0)
        {
            // If base == 0, then pick 0 from [base, base + size) and no zeros are
            // explicitly written.
            //
            // Otherwise, pick (base + (2^16 - base[16:0])), i.e., round up base to the
            // next multiple of 2^16. As 2^16 < size, this value should be in the
            // interval [base, base + size).
            const uint32_t mid = ((base_ - 1) >> 16) + 1;
            TORCH_CHECK((mid & 0xFFFF) == mid, "tfc291");
            out.push_back(static_cast<char>(mid >> 8));
            if ((mid & 0xFF) != 0)
            {
                out.push_back(static_cast<char>(mid >> 0));
            }
        }

        base_ = 0;
        size_minus1_ = std::numeric_limits<uint32_t>::max();
        delay_ = 0;
    };

    void print()
    {
        std::cout << out << std::endl;
    }

    py::bytes tobytes()
    {
        return py::bytes(out);
    }

    ~RangeEncoder()
    {   
        out = "";
    }
};

/** Encode symbols `sym` with CDF represented by `cdf_ptr`. NOTE: this is not exposted to python. */
py::bytes unbounded_index_range_encode_kernel(
    const at::Tensor &data,
    const at::Tensor &index,
    const at::Tensor &cdf_data,
    const at::Tensor &cdf_size,
    int precision, int overflow_width)
{

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
    RangeEncoder encoder;

    const CDF_package cdf(cdf_data);
    auto cdf_size_ = cdf_size.accessor<int32_t, 1>();

    const int N = at::numel(index);
    const auto idx_reshaped = at::reshape(index, {N});
    auto idx_ = idx_reshaped.accessor<int32_t, 1>();
    const auto data_reshaped = at::reshape(data, {N});
    auto data_ = data_reshaped.accessor<int32_t, 1>();

    const uint32_t max_overflow = (1 << overflow_width) - 1;

    for (int i = 0; i < N; i++)
    {
        const int32_t idx = idx_[i];
        const int32_t max_value = cdf_size_[idx];
        int32_t value = data_[i];
        uint32_t overflow = 0;
        // if (i < 16)
        // {
        //     printf("<%d> %d %d\n", i, value, max_value);
        // }

        if (value < 0)
        {
            overflow = -2 * value - 1;
            value = max_value;
        }
        else if (value >= max_value)
        {
            overflow = 2 * (value - max_value);
            value = max_value;
        }
        // if (i < 16)
        // {
        //     printf("<%d> %d %d\n", i, value, max_value);
        // }
        encoder.Encode(cdf.index(idx, value), value == max_value ? 0x10000U : cdf.index(idx, value + 1), precision, i, value == max_value);

        // Encode overflow using variable length code.
        if (value == max_value)
        {
            int32_t widths = 0;
            while (overflow >> (widths * overflow_width) != 0)
            {
                ++widths;
            }
            uint32_t val = widths;
            while (val >= max_overflow)
            {
                encoder.Encode(max_overflow, max_overflow + 1, overflow_width, i);
                val -= max_overflow;
            }
            encoder.Encode(val, val + 1, overflow_width, i);
            for (int j = 0; j < widths; ++j)
            {
                const uint32_t val = (overflow >> (j * overflow_width)) & max_overflow;
                encoder.Encode(val, val + 1, overflow_width, i);
            }
        }
        // encoder.print();
    }
    encoder.Finalize();
    // encoder.print();

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;
#endif

    return encoder.tobytes();
}

/** See torchac.py */
py::bytes unbounded_index_range_encode(
    const at::Tensor &data,
    const at::Tensor &index,
    const at::Tensor &cdf, /* NLp */
    const at::Tensor &cdf_size,
    int precision = 16, int overflow_width = 4)
{
    TORCH_CHECK(cdf.dim() == 2 && cdf.size(0) == cdf_size.size(0), "cdf check");
    return unbounded_index_range_encode_kernel(data.cpu(), index.cpu(), cdf.cpu(), cdf_size.cpu(), precision, overflow_width);
}

//------------------------------------------------------------------------------

/** Class to read byte string bit by bit */
class RangeDecoder
{
private:
    uint32_t base_ = 0;
    uint32_t size_minus1_ = std::numeric_limits<uint32_t>::max();
    uint64_t value_ = 0;
    std::string::const_iterator current_;
    std::string::const_iterator end_;

public:
    RangeDecoder(const std::string &source) : current_(source.begin()), end_(source.end())
    {
        Read16BitValue();
        Read16BitValue();
    };

    int32_t Decode(std::vector<int32_t> cdf, int precision, int i)
    {
        TORCH_CHECK(precision > 0 && precision <= 16, "Precision must in (0, 16]");

        const uint64_t size = static_cast<uint64_t>(size_minus1_) + 1;
        const uint64_t offset = ((static_cast<uint64_t>(value_ - base_) + 1) << precision) - 1;
        const uint64_t target = offset / size;
        // After the binary search, `pv` points to the smallest number v that
        // satisfies offset < (size * v) / 2^precision.

        if (i < 16)
        {
            printf("c(%d)[", i);
            for (auto it = cdf.begin(); it != cdf.end(); it++)
            {
                printf("%d ", *it);
            }
            printf("]\n");
        }

        // Assumes that cdf[0] == 0. Therefore (size * cdf[0]) / 2^precision is always
        // less than or equal to offset.
        const int32_t *pv = cdf.data() + 1;
        // `len` can be cdf.size() - 2 if there is guarantee that the last element of
        // cdf is 2^precision.
        auto len = cdf.size() - 1;
        TORCH_CHECK(len > 0, "CDF len must great than 0");
        if (i < 16)
        {
            printf("(%d)1 <%lu %d %d>\n", i, len, *pv, *cdf.data());
        }

        int c = 0;
        do
        {
            const auto half = len / 2;
            const int32_t *mid = pv + half;
            TORCH_CHECK(0 <= *mid && *mid <= (1 << precision), "Expect 0 <= *mid <= 2^precision");
            if (i == 4)
            {
                printf("(%d)1 state <%lu %lu %d %d>\n", i, len, half, *mid, *pv);
                printf("(%d)2 state <%llu %llu %llu %llu>\n", i, size, static_cast<uint64_t>(*mid), size * static_cast<uint64_t>(*mid), offset);
            }
            if (size * static_cast<uint64_t>(*mid) <= offset)
            {
                if (i == 4)
                {
                    printf("(%d) <%d> shift to <%d>\n", i, *pv, *(mid + 1));
                }
                pv = mid + 1;
                len -= half + 1;
            }
            else
            {
                len = half;
            }
            if (i == 4)
            {
                printf("(%d) state <%lu %lu %d %d>\n", i, len, half, *mid, *pv);
                if (*pv == *(pv + 1))
                {
                    printf("pv eq\n");
                    // break;
                }
                if (len < half)
                {
                    printf("le lt\n");
                    // break;
                }
            }
            c++;
            if (c > (int)cdf.size())
                break;
        } while (len > 0);

        // int32_t left = 0;
        // int32_t right = cdf.size() - 1;
        // int ans

        // while (left + 1 < right)
        // { // ?
        //     // left and right will be < 0x10000 in practice, so left+right fits in uint16_t...
        //     const auto m = (left + right) / 2;
        //     const auto v = cdf[m];

        //     if (v < target)
        //     {
        //         left = m;
        //     }
        //     else if (v > offset)
        //     {
        //         right = m;
        //     }
        //     else
        //     {
        //         return m;
        //     }
        // }
        // return left;

        // If (size * v) / 2^precision <= offset for all v in cdf, then pv points to
        // one after the last element of cdf. That is a decoding error.
        //
        // TODO(ssjhv): Consider returning -1 to indicate error. Or start len =
        // cdf.size() - 2 instead and give up detecting this error.
        TORCH_CHECK(pv < (cdf.data() + cdf.size()));

        const uint32_t a = (size * static_cast<uint64_t>(*(pv - 1))) >> precision;
        const uint32_t b = ((size * static_cast<uint64_t>(*pv)) >> precision) - 1;
        auto shifted = offset >> precision;
        TORCH_CHECK(a <= shifted and shifted <= b)

        base_ += a;
        size_minus1_ = b - a;

        if (size_minus1_ >> 16 == 0)
        {
            base_ <<= 16;
            size_minus1_ <<= 16;
            size_minus1_ |= 0xFFFF;

            Read16BitValue();
        }
        auto ret = pv - cdf.data() - 1;

        if (i < 16)
        {
            printf("(%d) ret=<%ld %d %d>\n", i, ret, *pv, *cdf.data());
        }

        return ret;
    }

    void Read16BitValue()
    {
        value_ <<= 8;
        if (current_ != end_)
        {
            value_ |= static_cast<uint8_t>(*current_++);
        }
        value_ <<= 8;
        if (current_ != end_)
        {
            value_ |= static_cast<uint8_t>(*current_++);
        }
    }
};

at::Tensor unbounded_index_range_decode_kernel(
    const std::string &encoded,
    const at::Tensor &index,
    const at::Tensor &cdf_data, /* NLp */
    const at::Tensor &cdf_size,
    int precision = 16, int overflow_width = 4)
{

#ifdef VERBOSE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
    RangeDecoder decoder(encoded);

    const CDF_package cdf(cdf_data);
    auto cdf_size_ = cdf_size.accessor<int32_t, 1>();

    const int N = at::numel(index);
    const auto idx_reshaped = at::reshape(index, {N});
    auto idx_ = idx_reshaped.accessor<int32_t, 1>();
    auto out = torch::empty({N}, at::kInt);
    auto out_ = out.accessor<int32_t, 1>();

    const uint32_t max_overflow = (1 << overflow_width) - 1;
    const int32_t overflow_cdf_size = (1 << overflow_width) + 1;
    std::vector<int32_t> overflow_cdf(overflow_cdf_size);
    std::iota(overflow_cdf.begin(), overflow_cdf.end(), 0);

    for (int i = 0; i < N; i++)
    {
        const int idx = idx_[i];
        const int max_value = cdf_size_[idx];

        int32_t value = decoder.Decode(cdf.slice(idx, max_value + 2), precision, i);

        // Decode overflow using variable length code.
        if (value == max_value)
        {
            int32_t widths = 0;
            uint32_t val;
            do
            {
                val = decoder.Decode(overflow_cdf, overflow_width, i);
                widths += val;
            } while (val == max_overflow);
            uint32_t overflow = 0;
            for (int32_t j = 0; j < widths; ++j)
            {
                const uint32_t val = decoder.Decode(overflow_cdf, overflow_width, i);
                TORCH_CHECK(val <= max_overflow, "tfc350")
                overflow |= val << (j * overflow_width);
            }
            // Map positive values back to integer values.
            value = overflow >> 1;
            if (overflow & 1)
            {
                value = -value - 1;
            }
            else
            {
                value += max_value;
            }
        }

        out_[i] = (int32_t)value;
    }

#ifdef VERBOSE
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;
#endif

    return out.reshape_as(index);
}

/** See torchac.py */
at::Tensor unbounded_index_range_decode(
    const std::string &encoded,
    const at::Tensor &index,
    const at::Tensor &cdf, /* NLp */
    const at::Tensor &cdf_size,
    int precision = 16, int overflow_width = 4)
{
    TORCH_CHECK(cdf.dim() == 2 && cdf.size(0) == cdf_size.size(0), "cdf check");
    return unbounded_index_range_decode_kernel(encoded, index.cpu(), cdf.cpu(), cdf_size.cpu(), precision, overflow_width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("unbounded_index_range_encode", &unbounded_index_range_encode, "Encode from CDF");
    m.def("unbounded_index_range_decode", &unbounded_index_range_decode, "Decode from CDF");
#ifdef COMPILE_CUDA
    m.def("cuda_supported", []() { return true; });
#else
    m.def("cuda_supported", []() { return false; });
#endif
}
