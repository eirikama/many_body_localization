// heisenberg_diag.cpp
// Exact diagonalization of the Heisenberg spin chain with random on-site fields.
// Usage: ./heisenberg_diag <P> <W> <duplicate_number>

#include "lapacke.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXd;

// ─── Types ─────────────────────────────────────────────────────────────────

using SpinBasis = std::vector<int>;
using TripletList = std::vector<Triplet<double>>;

// ─── Bit utilities ─────────────────────────────────────────────────────────

// Extract the j-th bit (MSB first) of the integer `state` given `P` sites.
inline bool bit(int state, int site, int P) {
    return (state >> (P - 1 - site)) & 1;
}

// Count the number of set bits (popcount).
inline int popcount(int x) {
    return __builtin_popcount(x);
}

// Return true if x has exactly 2 set bits that are adjacent (cyclically).
bool isAdjacentPair(int x, int P) {
    if (popcount(x) != 2) return false;

    // Build per-site bitmask and check adjacency including wrap-around.
    for (int i = 0; i < P; ++i) {
        if (bit(x, i, P) && bit(x, (i + 1) % P, P)) return true;
    }
    return false;
}

// Return true if log2(x) is an integer (x is a power of two, x != 0).
inline bool isPowerOfTwo(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

// ─── Spin-sector filtering ──────────────────────────────────────────────────

// Return true if |state⟩ lives in the Sz = S/2 sector.
bool inSpinSector(int S, int state, int P) {
    int up = 0;
    for (int i = 0; i < P; ++i)
        up += bit(state, i, P) ? 1 : -1;
    return up == S;
}

SpinBasis buildBasis(int S, int P) {
    SpinBasis basis;
    const int N = 1 << P;
    for (int i = 0; i < N; ++i)
        if (inSpinSector(S, i, P))
            basis.push_back(i);
    return basis;
}

// ─── Hamiltonian matrix elements ────────────────────────────────────────────

// Diagonal ZZ term: H_diag = J * sum_<ij> Sz_i Sz_j  (periodic boundary).
double diagElement(int state, double J, int P) {
    double elem = 0.0;
    for (int i = 0; i < P; ++i) {
        bool si = bit(state, i, P);
        bool sj = bit(state, (i + 1) % P, P);
        elem += (si == sj) ? J / 4.0 : -J / 4.0;
    }
    return elem;
}

// Build sparse Hamiltonian (without disorder) in the given spin sector.
SparseMatrix<double> buildHamiltonian(const SpinBasis& basis, double J, int P) {
    const int dim = static_cast<int>(basis.size());
    TripletList triplets;
    triplets.reserve(5 * dim); // rough upper bound

    for (int j = 0; j < dim; ++j) {
        // Diagonal entry.
        triplets.emplace_back(j, j, diagElement(basis[j], J, P));

        for (int i = 0; i < dim; ++i) {
            if (i == j) continue;

            int diff = basis[i] ^ basis[j];
            int lo   = std::abs(basis[i] - basis[j]);

            if (isAdjacentPair(diff, P)) {
                int mid = diff - lo;
                if (lo > 0 && isPowerOfTwo(lo) &&
                    mid > 0 && isPowerOfTwo(mid)) {
                    triplets.emplace_back(i, j, J * 0.5);
                }
            }
        }
    }

    SparseMatrix<double> H(dim, dim);
    H.setFromTriplets(triplets.begin(), triplets.end());
    return H;
}

// ─── Random disorder ────────────────────────────────────────────────────────

// Generate P uniform random fields in [-W, W] using a Mersenne twister.
std::vector<double> randomFields(double W, unsigned seed, int P) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-W, W);

    std::vector<double> fields(P);
    for (double& h : fields) h = dist(rng);
    return fields;
}

// Diagonal contribution of disorder for a given basis state.
double disorderElement(const std::vector<double>& fields, int state, int P) {
    double elem = 0.0;
    for (int i = 0; i < P; ++i)
        elem += bit(state, i, P) ? +0.5 * fields[i] : -0.5 * fields[i];
    return elem;
}

// ─── I/O helpers ────────────────────────────────────────────────────────────

void ensureResultsDir() {
    const char* dir = "results";
    struct stat info{};
    if (stat(dir, &info) != 0) {
        if (mkdir(dir, 0777) == -1)
            std::cerr << "Warning: could not create 'results' directory.\n";
    } else if (!(info.st_mode & S_IFDIR)) {
        throw std::runtime_error("'results' exists but is not a directory.");
    }
}

void writeMatrix(const std::string& path, const MatrixXd& M) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open file: " + path);
    out << M << '\n';
}

void writeVector(const std::string& path, const VectorXd& v) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open file: " + path);
    out << v << '\n';
}

std::string resultPath(const std::string& prefix, int P, double W, int dup) {
    std::ostringstream ss;
    ss << "results/" << prefix
       << "_P" << P
       << "_W" << static_cast<int>(1000 * W)
       << "_dupli" << dup << ".txt";
    return ss.str();
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <P> <W> <duplicate_number>\n";
        return 1;
    }

    const int    P   = std::atoi(argv[1]);
    const double W   = std::atof(argv[2]);
    const int    dup = std::atoi(argv[3]);

    constexpr int    S = 0;   // Sz = 0 sector
    constexpr double J = 1.0; // Heisenberg coupling

    // ── Build basis and Hamiltonian ──
    const SpinBasis basis = buildBasis(S, P);
    const int dim = static_cast<int>(basis.size());

    SparseMatrix<double> H = buildHamiltonian(basis, J, P);

    // ── Add disorder ──
    std::srand(std::time(nullptr));
    const auto fields = randomFields(W, static_cast<unsigned>(std::rand()), P);
    for (int i = 0; i < dim; ++i)
        H.coeffRef(i, i) += disorderElement(fields, basis[i], P);

    // ── Convert upper triangle for LAPACK ──
    MatrixXd A = H.triangularView<Eigen::Upper>();

    // ── Diagonalize with LAPACK dsyev ──
    int n   = dim;
    int lda = n;
    int info;
    double wkopt;
    int lwork = -1;
    double* w = new double[n];
    double* L = A.data();

    LAPACK_dsyev("V", "U", &n, L, &lda, w, &wkopt, &lwork, &info);
    lwork = static_cast<int>(wkopt);
    double* work = new double[lwork];
    LAPACK_dsyev("V", "U", &n, L, &lda, w, work, &lwork, &info);

    if (info > 0) {
        std::cerr << "LAPACK dsyev failed to converge (info=" << info << ").\n";
        delete[] w; delete[] work;
        return 1;
    }

    VectorXd eigvals = Map<VectorXd>(w, n);
    MatrixXd eigvecs = Map<MatrixXd>(L, n, n);

    delete[] w;
    delete[] work;

    // ── Write results ──
    ensureResultsDir();
    writeVector(resultPath("eigenvalues",  P, W, dup), eigvals);
    writeMatrix(resultPath("eigenvectors", P, W, dup), eigvecs);

    return 0;
}