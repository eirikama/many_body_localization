# include <iostream>
# include <bitset>
# include <vector>
# include <math.h>
# include <cstdlib>
# include <ctime>
# include <Eigen/Sparse>
# include <stdio.h>
# include "lapacke.h"

using namespace std ; 
using namespace Eigen ;


std::vector<bool> createBitset(int P, int i);
bool is_integer( double k );
double inter_diag_elem( int i , double J, int P);
bool check_dyad( int a, int P );
bool create_bitset( int a );
bool spin_tot ( int S, int i, int P);
double rand_elem ( vector < double > ran_vec , int i, int P );
vector <double> gen_ran_vec ( double W , double seed, int N);


int main(){

const int P = 4; // Number of Particles
const int N = pow (2.0 , P ) ; // Dimensionality of Problem
int S = 0; // Our choice of Spin Sector
double J = 1; // Strengths of Interaction
double W = 3; // Strength of Disorder
typedef Triplet < double > T ;
vector <T> tripletList ;
vector < double > r ;
vector < int > Sz_sec ;
srand ( time ( NULL ) ) ;


// Finding the Basis - States in Given Spin - Sector
for ( int i = 0; i < N ; i ++) if ( spin_tot(S, i, P)) Sz_sec.push_back(i);

// Calculating the matrix elements
double a , b ; 
int i, j = 0;
for ( vector<int>::iterator it1 = Sz_sec.begin(); it1 != Sz_sec.end(); ++it1 ){
	i = 0;
	for ( vector<int>::iterator it2 = Sz_sec.begin(); it2 != Sz_sec.end(); ++it2){
		if ( i == j ) tripletList.push_back( T(j, i, inter_diag_elem(*it1, J, P)));
		a = * it1 ^* it2 ;
		if ( check_dyad ( a, P ) ){
			b = a - ( abs (* it1 - * it2 ) ) ;
			if ( is_integer ( log2 ( b ) ) && b !=0){
				tripletList.push_back(T(i ,j , J*0.5)) ;
			}
		}
		i++;
	}
	j++;
	if ( j >= Sz_sec.size() ) break ;
}

int N_sector = Sz_sec.size();
SparseMatrix <double> H(N_sector, N_sector);
H.setFromTriplets( tripletList.begin() , tripletList.end() ) ;


// Adding the random Fields
r = gen_ran_vec(W , (double) rand(),  P) ;
for ( int i =0; i < N_sector ; i ++) H.coeffRef(i, i) += rand_elem(r , Sz_sec[i], P) ;


Eigen::MatrixXd A = H.triangularView<Eigen::Upper>();

    std::cout << "A:\n" << A << "\n";



    // Diagonalizing
    int n = N_sector;
    int lda = n;
    int info;
    double wkopt;
    double* w = (double*)malloc(n * sizeof(double));
    double* L = A.data(); // Direct pointer to matrix data

    // Workspace query
    int lw = -1;
    LAPACK_dsyev("V", "U", &n, L, &lda, w, &wkopt, &lw, &info);

    // Allocate optimal workspace
    lw = (int)wkopt;
    double* work = (double*)malloc(lw * sizeof(double));

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
    std::cout << "Eigenvalues:\n" << eig_vals << "\n";
    std::cout << "Eigenvectors:\n" << eig_vecs << "\n";

    // Free allocated memory
    free(w);
    free(work);


}



// Function to create a bitset using std::vector<bool>
std::vector<bool> createBitset(int P, int i) {
    std::vector<bool> bitset(P, false); // Initialize a vector of size P with all bits set to false
    
    for (int j = 0; j < P; ++j) {
        // Extract the j-th bit of 'i' and store it in the bitset
        bitset[j] = (i >> j) & 1;
    }
    
    // Reverse the vector to match bitset's most-significant bit first style, if needed.
    std::reverse(bitset.begin(), bitset.end());
    
    return bitset;
}


// Function for Checking if a Number is an Integer
bool is_integer ( double k ) { 
	return floor ( k ) == k ; 
}


// Calculating the Diagonal Part of the Interaction
double inter_diag_elem ( int i , double J, int P){
	//bitset <P> a(i) ;
	std::vector<bool> a = createBitset(P, i);
	double elem = 0;
	for ( int i = 0; i < P - 1; i ++){
		if ( a[ i ] == a[ i + 1]) elem += J / 4;
		else elem -= J / 4;
	}
	if ( a[0] == a[ P - 1]) elem += J / 4;
	else elem -= J / 4;

	return elem ;
}


// Checking if a binary Number has two adjacent 1 ’ s and the rest 0 ’ s
bool check_dyad ( int a, int P){
	//bitset <P> b(a) ;
	std::vector<bool> b = createBitset(P, a);
	int dyad = 0;
	int count = 0;
	int last_bit = 1;
	for ( int i = 0; i < P ; i ++) if ( b[ i ] == 1) count += 1;
	if (count == 2){
		for ( int i = 0; i < P ; i ++){
			if ( b[ i ] == 1 && i != P - 1){
				if ( b[ i + 1] == 1){
					dyad = 1;
					break;
				}
			}
		}
		if ( b[0] == 1 && b[ P - 1] == 1) dyad = 1;
	}
	return dyad ;
}


// Checking if Product State has total Spin equal to given S
bool spin_tot ( int S, int i, int P){
	
	std::vector<bool> spins = createBitset(P, i);
	bool is_spin_tot = 0;
	int count = 0;
	
	for ( int i = 0; i < P ; i ++){
		if ( spins [ i ] == 1) count ++;
		else count--;
	}
	
	if ( count == S ) is_spin_tot = 1;
	
	return is_spin_tot ;
}


// Generating the random Part of the i ’ th diagonal Element
double rand_elem ( vector < double > ran_vec , int i, int P ){
	std::vector<bool> spins = createBitset(P, i);
	double elem = 0;
	for ( int i = 0; i < P ; i ++){
		if ( spins [ i ]) elem += ran_vec [ i ]*0.5;
		else elem -= ran_vec [ i ]*0.5;
	}
	return elem ;
}

// Generating a Vector of uniformly distributed random Numbers on [ -W , W ]
vector <double> gen_ran_vec ( double W , double seed, int N){
	vector <double> R;
	srand(seed);
	for ( int i = 0; i < N ; i++) R.push_back(2*W*(rand() / double (RAND_MAX)-0.5));
	return R ;
}

