// GF(2) bit-vector helpers for Pauli strings and parity rows.
//
// A Pauli string on n qubits is 2*nw 64-bit words: nw words of X bits followed by nw
// words of Z bits (the layout cuPauliProp uses). A parity row is a mask of the same shape;
// its value on a string is the parity of (string AND mask).
#pragma once

#include <cstdint>
#include <vector>

using BV = std::vector<uint64_t>;

struct Layout {
  int n = 0;    // qubits
  int nw = 0;   // words per X (or Z) part
  int words() const { return 2 * nw; }
};

inline bool getX(const uint64_t* v, int q) { return (v[q / 64] >> (q % 64)) & 1ULL; }
inline bool getZ(const uint64_t* v, const Layout& L, int q) { return (v[L.nw + q / 64] >> (q % 64)) & 1ULL; }
inline void flipX(uint64_t* v, int q) { v[q / 64] ^= (1ULL << (q % 64)); }
inline void flipZ(uint64_t* v, const Layout& L, int q) { v[L.nw + q / 64] ^= (1ULL << (q % 64)); }

inline int dotParity(const uint64_t* a, const uint64_t* b, int words) {
  int p = 0;
  for (int w = 0; w < words; w++) p ^= __builtin_popcountll(a[w] & b[w]) & 1;
  return p;
}

// Symplectic product: 1 if the two Pauli strings anticommute.
inline int symplectic(const uint64_t* a, const uint64_t* b, const Layout& L) {
  int p = 0;
  for (int w = 0; w < L.nw; w++)
    p ^= (__builtin_popcountll(a[w] & b[L.nw + w]) ^ __builtin_popcountll(a[L.nw + w] & b[w])) & 1;
  return p;
}

// Bit index i in [0, 2n): i < n is X bit i, otherwise Z bit i - n.
inline bool getBit(const uint64_t* v, const Layout& L, int i) { return i < L.n ? getX(v, i) : getZ(v, L, i - L.n); }
inline void flipBit(uint64_t* v, const Layout& L, int i) { if (i < L.n) flipX(v, i); else flipZ(v, L, i - L.n); }

// Row-reduced basis of a subspace of GF(2)^(2n).
struct Span {
  Layout L;
  std::vector<BV> basis;
  std::vector<int> pivot;   // pivot bit index of each basis vector

  int dim() const { return (int)basis.size(); }

  // Would v enlarge the span? Leaves the reduced remainder in v.
  bool reduce(BV& v) const {
    for (size_t i = 0; i < basis.size(); i++)
      if (getBit(v.data(), L, pivot[i]))
        for (int w = 0; w < L.words(); w++) v[w] ^= basis[i][w];
    for (int w = 0; w < L.words(); w++) if (v[w]) return true;
    return false;
  }

  // Adds v if independent; keeps the basis fully reduced. Returns true if the dimension grew.
  bool add(BV v) {
    if (!reduce(v)) return false;
    int p = -1;
    for (int i = 0; i < 2 * L.n; i++) if (getBit(v.data(), L, i)) { p = i; break; }
    for (auto& b : basis)
      if (getBit(b.data(), L, p))
        for (int w = 0; w < L.words(); w++) b[w] ^= v[w];
    basis.push_back(v);
    pivot.push_back(p);
    return true;
  }

  // Basis of all rows r with r . v = 0 for every v in the span (dimension 2n - dim).
  std::vector<BV> annihilator() const {
    std::vector<char> isPivot(2 * L.n, 0);
    for (int p : pivot) isPivot[p] = 1;
    std::vector<BV> out;
    for (int f = 0; f < 2 * L.n; f++) {
      if (isPivot[f]) continue;
      BV r(L.words(), 0);
      flipBit(r.data(), L, f);
      for (size_t i = 0; i < basis.size(); i++)
        if (getBit(basis[i].data(), L, f)) flipBit(r.data(), L, pivot[i]);
      out.push_back(r);
    }
    return out;
  }
};
