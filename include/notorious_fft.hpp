/*
	NotoriousFFT - C++17 convenience wrapper
	SPDX-License-Identifier: MIT

	Features:
	- RAII wrappers for aux data
	- std::vector support for all transforms
	- std::complex integration
	- Iterator-based interfaces
	- Template helpers for compile-time transform sizes
	- Type-safe complex number handling
*/

#ifndef NOTORIOUS_FFT_HPP
#define NOTORIOUS_FFT_HPP

#include "notorious_fft.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <complex>
#include <type_traits>
#include <utility>
#include <array>
#include <numeric>
#include <algorithm>

namespace notorious_fft {

/* ============================================================================
   Type Traits and Type Aliases
   ============================================================================ */

using real_t = notorious_fft_real;
using cmpl_t = notorious_fft_cmpl;

/* Get the underlying scalar type (float or double) */
using scalar_t = std::conditional_t<std::is_same_v<real_t, float>, float, double>;

/* Complex type from std::complex */
using complex_t = std::complex<scalar_t>;

/* ============================================================================
   RAII Wrapper for Auxiliary Data
   ============================================================================ */

struct aux_deleter {
	void operator()(notorious_fft_aux *a) const noexcept { 
		if (a) notorious_fft_free_aux(a); 
	}
};

using aux_ptr = std::unique_ptr<notorious_fft_aux, aux_deleter>;

/* Helper to create aux_ptr with null check */
[[nodiscard]] inline aux_ptr make_aux(notorious_fft_aux *raw) {
	if (!raw) throw std::runtime_error("notorious_fft: failed to allocate aux data");
	return aux_ptr(raw);
}

/* ============================================================================
   Aux Data Creation - 1D/2D/3D and arbitrary dimensions
   ============================================================================ */

// Complex DFT
[[nodiscard]] inline aux_ptr mkaux_dft_1d(int N) { 
	return make_aux(notorious_fft_mkaux_dft_1d(N)); 
}

[[nodiscard]] inline aux_ptr mkaux_dft_2d(int N1, int N2) { 
	return make_aux(notorious_fft_mkaux_dft_2d(N1, N2)); 
}

[[nodiscard]] inline aux_ptr mkaux_dft_3d(int N1, int N2, int N3) { 
	return make_aux(notorious_fft_mkaux_dft_3d(N1, N2, N3)); 
}

[[nodiscard]] inline aux_ptr mkaux_dft(int d, const int *Ns) { 
	return make_aux(notorious_fft_mkaux_dft(d, const_cast<int*>(Ns))); 
}

[[nodiscard]] inline aux_ptr mkaux_dft(std::initializer_list<int> dims) {
	std::vector<int> ns(dims);
	return make_aux(notorious_fft_mkaux_dft(static_cast<int>(ns.size()), ns.data()));
}

// Real DFT
[[nodiscard]] inline aux_ptr mkaux_realdft_1d(int N) { 
	return make_aux(notorious_fft_mkaux_realdft_1d(N)); 
}

[[nodiscard]] inline aux_ptr mkaux_realdft_2d(int N1, int N2) { 
	return make_aux(notorious_fft_mkaux_realdft_2d(N1, N2)); 
}

[[nodiscard]] inline aux_ptr mkaux_realdft_3d(int N1, int N2, int N3) { 
	return make_aux(notorious_fft_mkaux_realdft_3d(N1, N2, N3)); 
}

[[nodiscard]] inline aux_ptr mkaux_realdft(int d, const int *Ns) { 
	return make_aux(notorious_fft_mkaux_realdft(d, const_cast<int*>(Ns))); 
}

[[nodiscard]] inline aux_ptr mkaux_realdft(std::initializer_list<int> dims) {
	std::vector<int> ns(dims);
	return make_aux(notorious_fft_mkaux_realdft(static_cast<int>(ns.size()), ns.data()));
}

// DCT-2/DST-2 and DCT-3/DST-3 (Type 2/3 transforms)
[[nodiscard]] inline aux_ptr mkaux_t2t3_1d(int N) { 
	return make_aux(notorious_fft_mkaux_t2t3_1d(N)); 
}

[[nodiscard]] inline aux_ptr mkaux_t2t3_2d(int N1, int N2) { 
	return make_aux(notorious_fft_mkaux_t2t3_2d(N1, N2)); 
}

[[nodiscard]] inline aux_ptr mkaux_t2t3_3d(int N1, int N2, int N3) { 
	return make_aux(notorious_fft_mkaux_t2t3_3d(N1, N2, N3)); 
}

[[nodiscard]] inline aux_ptr mkaux_t2t3(int d, const int *Ns) { 
	return make_aux(notorious_fft_mkaux_t2t3(d, const_cast<int*>(Ns))); 
}

[[nodiscard]] inline aux_ptr mkaux_t2t3(std::initializer_list<int> dims) {
	std::vector<int> ns(dims);
	return make_aux(notorious_fft_mkaux_t2t3(static_cast<int>(ns.size()), ns.data()));
}

// DCT-4/DST-4 (Type 4 transforms)
[[nodiscard]] inline aux_ptr mkaux_t4_1d(int N) { 
	return make_aux(notorious_fft_mkaux_t4_1d(N)); 
}

[[nodiscard]] inline aux_ptr mkaux_t4_2d(int N1, int N2) { 
	return make_aux(notorious_fft_mkaux_t4_2d(N1, N2)); 
}

[[nodiscard]] inline aux_ptr mkaux_t4_3d(int N1, int N2, int N3) { 
	return make_aux(notorious_fft_mkaux_t4_3d(N1, N2, N3)); 
}

[[nodiscard]] inline aux_ptr mkaux_t4(int d, const int *Ns) { 
	return make_aux(notorious_fft_mkaux_t4(d, const_cast<int*>(Ns))); 
}

[[nodiscard]] inline aux_ptr mkaux_t4(std::initializer_list<int> dims) {
	std::vector<int> ns(dims);
	return make_aux(notorious_fft_mkaux_t4(static_cast<int>(ns.size()), ns.data()));
}

/* ============================================================================
   Low-level Transform Wrappers (pointer-based)
   ============================================================================ */

inline void dft(const notorious_fft_cmpl *x, notorious_fft_cmpl *y, const aux_ptr &a) {
	notorious_fft_dft(const_cast<notorious_fft_cmpl*>(x), y, a.get());
}

inline void invdft(const notorious_fft_cmpl *x, notorious_fft_cmpl *y, const aux_ptr &a) {
	notorious_fft_invdft(const_cast<notorious_fft_cmpl*>(x), y, a.get());
}

inline void realdft(const real_t *x, notorious_fft_cmpl *y, const aux_ptr &a) {
	notorious_fft_realdft(const_cast<real_t*>(x), y, a.get());
}

inline void invrealdft(const notorious_fft_cmpl *x, real_t *y, const aux_ptr &a) {
	notorious_fft_invrealdft(const_cast<notorious_fft_cmpl*>(x), y, a.get());
}

inline void dct2(const real_t *x, real_t *y, const aux_ptr &a) {
	notorious_fft_dct2(const_cast<real_t*>(x), y, a.get());
}

inline void dst2(const real_t *x, real_t *y, const aux_ptr &a) {
	notorious_fft_dst2(const_cast<real_t*>(x), y, a.get());
}

inline void dct3(const real_t *x, real_t *y, const aux_ptr &a) {
	notorious_fft_dct3(const_cast<real_t*>(x), y, a.get());
}

inline void dst3(const real_t *x, real_t *y, const aux_ptr &a) {
	notorious_fft_dst3(const_cast<real_t*>(x), y, a.get());
}

inline void dct4(const real_t *x, real_t *y, const aux_ptr &a) {
	notorious_fft_dct4(const_cast<real_t*>(x), y, a.get());
}

inline void dst4(const real_t *x, real_t *y, const aux_ptr &a) {
	notorious_fft_dst4(const_cast<real_t*>(x), y, a.get());
}

/* ============================================================================
   std::complex <-> notorious_fft_cmpl Conversion Helpers
   ============================================================================ */

namespace detail {

/* Convert std::complex array to notorious_fft_cmpl array (in-place compatible) */
inline void to_notorious_fft_cmpl(const complex_t *src, notorious_fft_cmpl *dst, size_t n) {
	static_assert(sizeof(complex_t) == sizeof(notorious_fft_cmpl), 
		"std::complex and notorious_fft_cmpl must have same size");
	if (reinterpret_cast<const void*>(src) != reinterpret_cast<void*>(dst)) {
		std::memcpy(dst, src, n * sizeof(notorious_fft_cmpl));
	}
}

/* Convert notorious_fft_cmpl array to std::complex array */
inline void from_notorious_fft_cmpl(const notorious_fft_cmpl *src, complex_t *dst, size_t n) {
	if (reinterpret_cast<const void*>(src) != reinterpret_cast<void*>(dst)) {
		std::memcpy(dst, src, n * sizeof(complex_t));
	}
}

/* Helper to get pointer from vector (handles both value and pointer vectors) */
template<typename T>
T* data_ptr(std::vector<T> &v) { return v.data(); }

template<typename T>
const T* data_ptr(const std::vector<T> &v) { return v.data(); }

} /* namespace detail */

/* ============================================================================
   High-level STL Container Interface
   ============================================================================ */

/* ---- Complex DFT with std::vector<std::complex<T>> ---- */

[[nodiscard]] inline std::vector<complex_t> dft(const std::vector<complex_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dft: input size must be positive");
	
	auto a = mkaux_dft_1d(N);
	std::vector<complex_t> y(N);
	
	std::vector<notorious_fft_cmpl> lx(N), ly(N);
	detail::to_notorious_fft_cmpl(x.data(), lx.data(), N);
	notorious_fft_dft(lx.data(), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), y.data(), N);
	
	return y;
}

[[nodiscard]] inline std::vector<complex_t> invdft(const std::vector<complex_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("invdft: input size must be positive");
	
	auto a = mkaux_dft_1d(N);
	std::vector<complex_t> y(N);
	
	std::vector<notorious_fft_cmpl> lx(N), ly(N);
	detail::to_notorious_fft_cmpl(x.data(), lx.data(), N);
	notorious_fft_invdft(lx.data(), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), y.data(), N);
	
	return y;
}

/* In-place complex DFT */
inline void dft_in_place(std::vector<complex_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dft_in_place: input size must be positive");
	
	auto a = mkaux_dft_1d(N);
	
	std::vector<notorious_fft_cmpl> lx(N), ly(N);
	detail::to_notorious_fft_cmpl(x.data(), lx.data(), N);
	notorious_fft_dft(lx.data(), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), x.data(), N);
}

inline void invdft_in_place(std::vector<complex_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("invdft_in_place: input size must be positive");
	
	auto a = mkaux_dft_1d(N);
	
	std::vector<notorious_fft_cmpl> lx(N), ly(N);
	detail::to_notorious_fft_cmpl(x.data(), lx.data(), N);
	notorious_fft_invdft(lx.data(), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), x.data(), N);
}

/* ---- Real DFT ---- */

[[nodiscard]] inline std::vector<complex_t> realdft(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("realdft: input size must be positive");
	
	auto a = mkaux_realdft_1d(N);
	const int out_size = N / 2 + 1;
	std::vector<complex_t> y(out_size);
	
	std::vector<notorious_fft_cmpl> ly(out_size);
	notorious_fft_realdft(const_cast<real_t*>(x.data()), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), y.data(), out_size);
	
	return y;
}

[[nodiscard]] inline std::vector<real_t> invrealdft(const std::vector<complex_t> &z, int N) {
	if (N <= 0) throw std::invalid_argument("invrealdft: output size must be positive");
	if (static_cast<int>(z.size()) != N / 2 + 1) {
		throw std::invalid_argument("invrealdft: input size must be N/2+1");
	}
	
	auto a = mkaux_realdft_1d(N);
	std::vector<real_t> y(N);
	
	std::vector<notorious_fft_cmpl> lz(z.size());
	detail::to_notorious_fft_cmpl(z.data(), lz.data(), z.size());
	notorious_fft_invrealdft(lz.data(), y.data(), a.get());
	
	return y;
}

/* Convenience: N inferred from output complex vector size */
[[nodiscard]] inline std::vector<real_t> invrealdft(const std::vector<complex_t> &z) {
	/* z has size N/2+1, so N = 2*(size-1) */
	const int out_size = static_cast<int>(z.size());
	const int N = 2 * (out_size - 1);
	return invrealdft(z, N);
}

/* ---- DCT/DST ---- */

[[nodiscard]] inline std::vector<real_t> dct2(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dct2: input size must be positive");
	
	auto a = mkaux_t2t3_1d(N);
	std::vector<real_t> y(N);
	notorious_fft_dct2(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

[[nodiscard]] inline std::vector<real_t> dst2(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dst2: input size must be positive");
	
	auto a = mkaux_t2t3_1d(N);
	std::vector<real_t> y(N);
	notorious_fft_dst2(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

[[nodiscard]] inline std::vector<real_t> dct3(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dct3: input size must be positive");
	
	auto a = mkaux_t2t3_1d(N);
	std::vector<real_t> y(N);
	notorious_fft_dct3(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

[[nodiscard]] inline std::vector<real_t> dst3(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dst3: input size must be positive");
	
	auto a = mkaux_t2t3_1d(N);
	std::vector<real_t> y(N);
	notorious_fft_dst3(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

[[nodiscard]] inline std::vector<real_t> dct4(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dct4: input size must be positive");
	
	auto a = mkaux_t4_1d(N);
	std::vector<real_t> y(N);
	notorious_fft_dct4(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

[[nodiscard]] inline std::vector<real_t> dst4(const std::vector<real_t> &x) {
	const int N = static_cast<int>(x.size());
	if (N <= 0) throw std::invalid_argument("dst4: input size must be positive");
	
	auto a = mkaux_t4_1d(N);
	std::vector<real_t> y(N);
	notorious_fft_dst4(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

/* ============================================================================
   Multi-dimensional Transforms with std::vector
   ============================================================================ */

namespace detail {

/* Calculate total size from dimensions */
inline size_t total_size(std::initializer_list<int> dims) {
	return std::accumulate(dims.begin(), dims.end(), 
		static_cast<size_t>(1), 
		std::multiplies<size_t>());
}

} /* namespace detail */

/* ---- 2D Complex DFT ---- */

[[nodiscard]] inline std::vector<complex_t> dft_2d(
		const std::vector<complex_t> &x, int N1, int N2) {
	const size_t expected_size = static_cast<size_t>(N1) * N2;
	if (x.size() != expected_size) {
		throw std::invalid_argument("dft_2d: input size must be N1*N2");
	}
	
	auto a = mkaux_dft_2d(N1, N2);
	std::vector<complex_t> y(x.size());
	
	std::vector<notorious_fft_cmpl> lx(x.size()), ly(y.size());
	detail::to_notorious_fft_cmpl(x.data(), lx.data(), x.size());
	notorious_fft_dft(lx.data(), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), y.data(), y.size());
	
	return y;
}

/* ---- 3D Complex DFT ---- */

[[nodiscard]] inline std::vector<complex_t> dft_3d(
		const std::vector<complex_t> &x, int N1, int N2, int N3) {
	const size_t expected_size = static_cast<size_t>(N1) * N2 * N3;
	if (x.size() != expected_size) {
		throw std::invalid_argument("dft_3d: input size must be N1*N2*N3");
	}
	
	auto a = mkaux_dft_3d(N1, N2, N3);
	std::vector<complex_t> y(x.size());
	
	std::vector<notorious_fft_cmpl> lx(x.size()), ly(y.size());
	detail::to_notorious_fft_cmpl(x.data(), lx.data(), x.size());
	notorious_fft_dft(lx.data(), ly.data(), a.get());
	detail::from_notorious_fft_cmpl(ly.data(), y.data(), y.size());
	
	return y;
}

/* ---- 2D Real DCT ---- */

[[nodiscard]] inline std::vector<real_t> dct2_2d(
		const std::vector<real_t> &x, int N1, int N2) {
	const size_t expected_size = static_cast<size_t>(N1) * N2;
	if (x.size() != expected_size) {
		throw std::invalid_argument("dct2_2d: input size must be N1*N2");
	}
	
	auto a = mkaux_t2t3_2d(N1, N2);
	std::vector<real_t> y(x.size());
	notorious_fft_dct2(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

[[nodiscard]] inline std::vector<real_t> dct3_2d(
		const std::vector<real_t> &x, int N1, int N2) {
	const size_t expected_size = static_cast<size_t>(N1) * N2;
	if (x.size() != expected_size) {
		throw std::invalid_argument("dct3_2d: input size must be N1*N2");
	}
	
	auto a = mkaux_t2t3_2d(N1, N2);
	std::vector<real_t> y(x.size());
	notorious_fft_dct3(const_cast<real_t*>(x.data()), y.data(), a.get());
	return y;
}

/* ============================================================================
   Reusable Transform Objects (for repeated transforms of same size)
   ============================================================================ */

class dft_plan {
	aux_ptr aux_;
	int N_;
	mutable std::vector<notorious_fft_cmpl> tmp_in_, tmp_out_;
	
public:
	explicit dft_plan(int N) : aux_(mkaux_dft_1d(N)), N_(N), 
		tmp_in_(N), tmp_out_(N) {}
	
	[[nodiscard]] std::vector<complex_t> execute(const std::vector<complex_t> &x) const {
		if (static_cast<int>(x.size()) != N_) {
			throw std::invalid_argument("dft_plan: input size mismatch");
		}
		
		detail::to_notorious_fft_cmpl(x.data(), tmp_in_.data(), N_);
		notorious_fft_dft(tmp_in_.data(), tmp_out_.data(), aux_.get());
		
		std::vector<complex_t> y(N_);
		detail::from_notorious_fft_cmpl(tmp_out_.data(), y.data(), N_);
		return y;
	}
	
	void execute_in_place(std::vector<complex_t> &x) const {
		if (static_cast<int>(x.size()) != N_) {
			throw std::invalid_argument("dft_plan: input size mismatch");
		}
		
		detail::to_notorious_fft_cmpl(x.data(), tmp_in_.data(), N_);
		notorious_fft_dft(tmp_in_.data(), tmp_out_.data(), aux_.get());
		detail::from_notorious_fft_cmpl(tmp_out_.data(), x.data(), N_);
	}
	
	[[nodiscard]] int size() const noexcept { return N_; }
};

class realdft_plan {
	aux_ptr aux_;
	int N_;
	int out_size_;
	mutable std::vector<notorious_fft_cmpl> tmp_out_;
	
public:
	explicit realdft_plan(int N) : aux_(mkaux_realdft_1d(N)), N_(N),
		out_size_(N / 2 + 1), tmp_out_(out_size_) {}
	
	[[nodiscard]] std::vector<complex_t> execute(const std::vector<real_t> &x) const {
		if (static_cast<int>(x.size()) != N_) {
			throw std::invalid_argument("realdft_plan: input size mismatch");
		}
		
		notorious_fft_realdft(const_cast<real_t*>(x.data()), tmp_out_.data(), aux_.get());
		
		std::vector<complex_t> y(out_size_);
		detail::from_notorious_fft_cmpl(tmp_out_.data(), y.data(), out_size_);
		return y;
	}
	
	[[nodiscard]] int size() const noexcept { return N_; }
	[[nodiscard]] int output_size() const noexcept { return out_size_; }
};

class dct2_plan {
	aux_ptr aux_;
	int N_;
	
public:
	explicit dct2_plan(int N) : aux_(mkaux_t2t3_1d(N)), N_(N) {}
	
	[[nodiscard]] std::vector<real_t> execute(const std::vector<real_t> &x) const {
		if (static_cast<int>(x.size()) != N_) {
			throw std::invalid_argument("dct2_plan: input size mismatch");
		}
		
		std::vector<real_t> y(N_);
		notorious_fft_dct2(const_cast<real_t*>(x.data()), y.data(), aux_.get());
		return y;
	}
	
	void execute(const real_t *x, real_t *y) const {
		notorious_fft_dct2(const_cast<real_t*>(x), y, aux_.get());
	}
	
	[[nodiscard]] int size() const noexcept { return N_; }
};

/* ============================================================================
   Iterator-based Interface (for custom containers)
   ============================================================================ */

/* Transform using forward iterators - complex DFT */
template<typename InIter, typename OutIter>
void dft(InIter first, InIter last, OutIter out, const aux_ptr &a) {
	const int N = static_cast<int>(std::distance(first, last));
	std::vector<notorious_fft_cmpl> lx(N), ly(N);
	
	for (int i = 0; i < N; ++i, ++first) {
		complex_t c = *first;
		reinterpret_cast<real_t*>(&lx[i])[0] = c.real();
		reinterpret_cast<real_t*>(&lx[i])[1] = c.imag();
	}
	
	notorious_fft_dft(lx.data(), ly.data(), a.get());
	
	for (int i = 0; i < N; ++i, ++out) {
		real_t *p = reinterpret_cast<real_t*>(&ly[i]);
		*out = complex_t(p[0], p[1]);
	}
}

/* Transform using forward iterators - real DFT */
template<typename InIter, typename OutIter>
void realdft(InIter first, InIter last, OutIter out, const aux_ptr &a) {
	const int N = static_cast<int>(std::distance(first, last));
	std::vector<real_t> x(N);
	
	for (int i = 0; i < N; ++i, ++first) {
		x[i] = static_cast<real_t>(*first);
	}
	
	const int out_size = N / 2 + 1;
	std::vector<notorious_fft_cmpl> ly(out_size);
	notorious_fft_realdft(x.data(), ly.data(), a.get());
	
	for (int i = 0; i < out_size; ++i, ++out) {
		real_t *p = reinterpret_cast<real_t*>(&ly[i]);
		*out = complex_t(p[0], p[1]);
	}
}

/* ============================================================================
   Range-based Interface (C++20 style, but works in C++17)
   ============================================================================ */

template<typename Range>
[[nodiscard]] auto dft(const Range &range) 
	-> std::vector<complex_t> {
	return dft(std::vector<complex_t>(range.begin(), range.end()));
}

/* ============================================================================
   Utilities
   ============================================================================ */

/* Check if a size is valid for notorious_fft (must be power of 2) */
[[nodiscard]] constexpr bool is_valid_size(int N) noexcept {
	return N > 0 && (N & (N - 1)) == 0;
}

/* Get the next power of 2 >= N */
[[nodiscard]] constexpr int next_power_of_2(int N) noexcept {
	if (N <= 1) return 1;
	N--;
	N |= N >> 1;
	N |= N >> 2;
	N |= N >> 4;
	N |= N >> 8;
	N |= N >> 16;
	return N + 1;
}

/* ============================================================================
   Multi-dimensional Array View (simple wrapper for flat data)
   ============================================================================ */

template<typename T, int Dims>
class md_view {
	T *data_;
	std::array<int, Dims> shape_;
	
public:
	md_view(T *data, std::array<int, Dims> shape) 
		: data_(data), shape_(shape) {}
	
	[[nodiscard]] T& operator()(std::array<int, Dims> idx) {
		size_t offset = 0;
		size_t stride = 1;
		for (int i = Dims - 1; i >= 0; --i) {
			offset += idx[i] * stride;
			stride *= shape_[i];
		}
		return data_[offset];
	}
	
	[[nodiscard]] const T& operator()(std::array<int, Dims> idx) const {
		return const_cast<md_view*>(this)->operator()(idx);
	}
	
	[[nodiscard]] int size(int dim) const noexcept { return shape_[dim]; }
	[[nodiscard]] T* data() noexcept { return data_; }
	[[nodiscard]] const T* data() const noexcept { return data_; }
};

/* Factory function */
template<typename T, int Dims>
[[nodiscard]] auto make_md_view(T *data, std::array<int, Dims> shape) {
	return md_view<T, Dims>(data, shape);
}

} /* namespace notorious_fft */

#endif /* NOTORIOUS_FFT_HPP */
