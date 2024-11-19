#include "lapacke.h"
#include <Eigen/Sparse>
#include <bitset>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace std;
using namespace Eigen;

void createResultFolder();
std::vector<bool> createBitset(int P, int i);
bool is_integer(double k);
double inter_diag_elem(int i, double J, int P);
bool check_dyad(int a, int P);
bool spin_tot(int S, int i, int P);
double rand_elem(vector<double> ran_vec, int i, int P);
vector<double> gen_ran_vec(double W, double seed, int N);

int main(int argc, char *argv[]) {

  const int P = std::atoi(argv[1]);          // Number of Particles
  double W = std::atof(argv[2]);             // Number of Particles
  int duplicate_number = std::atoi(argv[3]); // Number of Particles

  const int N = pow(2.0, P); // Dimensionality of Problem
  int S = 0;                 // Our choice of Spin Sector
  double J = 1;              // Strengths of Interaction
  typedef Triplet<double> T;
  vector<T> tripletList;
  vector<double> r;
  vector<int> Sz_sec;
  srand(time(NULL));

  // Finding the Basis - States in Given Spin - Sector
  for (int i = 0; i < N; i++) {
    if (spin_tot(S, i, P)) {
      Sz_sec.push_back(i);
    }
  }

  // Calculating the matrix elements
  double a, b;
  int i, j = 0;
  for (vector<int>::iterator it1 = Sz_sec.begin(); it1 != Sz_sec.end(); ++it1) {
    i = 0;
    for (vector<int>::iterator it2 = Sz_sec.begin(); it2 != Sz_sec.end();
         ++it2) {
      if (i == j)
        tripletList.push_back(T(j, i, inter_diag_elem(*it1, J, P)));
      a = *it1 ^ *it2;
      if (check_dyad(a, P)) {
        b = a - (abs(*it1 - *it2));
        if (is_integer(log2(b)) && b != 0) {
          tripletList.push_back(T(i, j, J * 0.5));
        }
      }
      i++;
    }
    j++;
    if (j >= Sz_sec.size())
      break;
  }

  int N_sector = Sz_sec.size();
  SparseMatrix<double> H(N_sector, N_sector);
  H.setFromTriplets(tripletList.begin(), tripletList.end());

  // Adding the random Fields
  r = gen_ran_vec(W, (double)rand(), P);
  for (int i = 0; i < N_sector; i++)
    H.coeffRef(i, i) += rand_elem(r, Sz_sec[i], P);

  Eigen::MatrixXd A = H.triangularView<Eigen::Upper>();

  // std::cout << "A:\n" << A << "\n";

  // Diagonalizing
  int n = N_sector;
  int lda = n;
  int info;
  double wkopt;
  double *w = (double *)malloc(n * sizeof(double));
  double *L = A.data(); // Direct pointer to matrix data

  // Workspace query
  int lw = -1;
  LAPACK_dsyev("V", "U", &n, L, &lda, w, &wkopt, &lw, &info);

  // Allocate optimal workspace
  lw = (int)wkopt;
  double *work = (double *)malloc(lw * sizeof(double));

  // Actual dsyev call
  LAPACK_dsyev("V", "U", &n, L, &lda, w, work, &lw, &info);

  if (info > 0) {
    std::cerr << "The algorithm failed to compute eigenvalues.\n";
    free(w);
    free(work);
    return 1;
  }

  // Map the results to Eigen objects
  Eigen::VectorXd eig_vals = Eigen::Map<Eigen::VectorXd>(w, N_sector);
  Eigen::MatrixXd eig_vecs = Eigen::Map<Eigen::MatrixXd>(L, N_sector, N_sector);

  // Output results
  // std::cout << "Eigenvalues:\n" << eig_vals << "\n";
  // std::cout << "Eigenvectors:\n" << eig_vecs << "\n";

  createResultFolder();

  std::ostringstream eigvec_filename;
  eigvec_filename << "results/eigenvectors_P" << P << "_W" << (int)(1000 * W)
                  << "_dupli" << duplicate_number << ".txt";
  std::ofstream eigvec_outfile(eigvec_filename.str());
  if (eigvec_outfile.is_open()) {
    eigvec_outfile << eig_vecs << "\n";
    eigvec_outfile.close();
    // std::cout << "Eigenvectors saved to " << eigvec_filename.str() << "\n";
  } else {
    std::cerr << "Failed to open file " << eigvec_filename.str() << "\n";
  }

  std::ostringstream eigval_filename;
  eigval_filename << "results/eigenvalues_P" << P << "_W" << (int)(1000 * W)
                  << "_dupli" << duplicate_number << ".txt";
  std::ofstream eigval_outfile(eigval_filename.str());
  if (eigval_outfile.is_open()) {
    eigval_outfile << eig_vals << "\n";
    eigval_outfile.close();
    // std::cout << "Eigenvalues saved to " << eigval_filename.str() << "\n";
  } else {
    std::cerr << "Failed to open file " << eigval_filename.str() << "\n";
  }

  // Free allocated memory
  free(w);
  free(work);
}

// Ensure the "result" folder exists
void createResultFolder() {
  const char *folder_name = "results";

  // Check if the folder exists; if not, create it
  struct stat info;
  if (stat(folder_name, &info) != 0) { // Folder doesn't exist
    if (mkdir(folder_name, 0777) == -1) {
      std::cerr << "Error creating directory 'result'.\n";
    }
  } else if (!(info.st_mode & S_IFDIR)) { // File named 'result' exists but it's
                                          // not a directory
    std::cerr << "'result' exists but is not a directory.\n";
  }
}

// Function to create a bitset using std::vector<bool>
std::vector<bool> createBitset(int P, int i) {
  std::vector<bool> bitset(
      P, false); // Initialize a vector of size P with all bits set to false

  for (int j = 0; j < P; ++j) {
    // Extract the j-th bit of 'i' and store it in the bitset
    bitset[j] = (i >> j) & 1;
  }

  // Reverse the vector to match bitset's most-significant bit first style, if
  // needed.
  std::reverse(bitset.begin(), bitset.end());

  return bitset;
}

// Function for Checking if a Number is an Integer
bool is_integer(double k) { return floor(k) == k; }

// Calculating the Diagonal Part of the Interaction
double inter_diag_elem(int i, double J, int P) {
  // bitset <P> a(i) ;
  std::vector<bool> a = createBitset(P, i);
  double elem = 0;
  for (int i = 0; i < P - 1; i++) {
    if (a[i] == a[i + 1])
      elem += J / 4;
    else
      elem -= J / 4;
  }
  if (a[0] == a[P - 1])
    elem += J / 4;
  else
    elem -= J / 4;

  return elem;
}

// Checking if a binary Number has two adjacent 1 ’ s and the rest 0 ’ s
bool check_dyad(int a, int P) {
  // bitset <P> b(a) ;
  std::vector<bool> b = createBitset(P, a);
  int dyad = 0;
  int count = 0;
  int last_bit = 1;
  for (int i = 0; i < P; i++)
    if (b[i] == 1)
      count += 1;
  if (count == 2) {
    for (int i = 0; i < P; i++) {
      if (b[i] == 1 && i != P - 1) {
        if (b[i + 1] == 1) {
          dyad = 1;
          break;
        }
      }
    }
    if (b[0] == 1 && b[P - 1] == 1)
      dyad = 1;
  }
  return dyad;
}

// Checking if Product State has total Spin equal to given S
bool spin_tot(int S, int i, int P) {

  std::vector<bool> spins = createBitset(P, i);
  bool is_spin_tot = 0;
  int count = 0;

  for (int i = 0; i < P; i++) {
    if (spins[i] == 1)
      count++;
    else
      count--;
  }

  if (count == S)
    is_spin_tot = 1;

  return is_spin_tot;
}

// Generating the random Part of the i ’ th diagonal Element
double rand_elem(vector<double> ran_vec, int i, int P) {
  std::vector<bool> spins = createBitset(P, i);
  double elem = 0;
  for (int i = 0; i < P; i++) {
    if (spins[i])
      elem += ran_vec[i] * 0.5;
    else
      elem -= ran_vec[i] * 0.5;
  }
  return elem;
}

// Generating a Vector of uniformly distributed random Numbers on [ -W , W ]
vector<double> gen_ran_vec(double W, double seed, int N) {
  vector<double> R;
  srand(seed);
  for (int i = 0; i < N; i++)
    R.push_back(2 * W * (rand() / double(RAND_MAX) - 0.5));
  return R;
}
