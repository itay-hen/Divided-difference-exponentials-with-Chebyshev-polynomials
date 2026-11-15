// Runs the PMR-QMC update scheme of Barash, Babakhani & Hen (PRR 6, 013281,
// 2024) on the Hamiltonian in hamiltonian.hpp and records the stream of
// divided-difference operations of the weight object (divdiff_trace.hpp).
// Usage: pmr_trace beta Tsteps steps seed trace_file
#include<iostream>
#include<fstream>
#include<complex>
#include<random>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<bitset>
#include"divdiff_trace.hpp"
#include"hamiltonian.hpp"
#include"params_trace.hpp"
#ifdef EXHAUSTIVE_CYCLE_SEARCH
	#define rmin 0
	#define rmax cycle_max_len
	#define lmin r
	#define lmax cycle_max_len
#else
	#define rmin (cycle_min_len-1)/2
	#define rmax (cycle_max_len+1)/2
	#define lmin 2*r-1
	#define lmax 2*r+1
#endif
static std::mt19937 rng;
static std::uniform_int_distribution<> dice2(0,1);
static std::uniform_int_distribution<> diceN(0,N-1);
static std::uniform_int_distribution<> diceNop(0,Nop-1);
static std::uniform_real_distribution<> val(0.0,1.0);
static std::geometric_distribution<> geometric_int(GAPS_GEOMETRIC_PARAMETER);
double factorial[qmax];
int cycle_len[Ncycles];
int cycle_min_len, cycle_max_len;
std::bitset<Ncycles> P_in_cycles[Nop];
#define REALW std::real
ExExFloat zero;
int save_data_flag = 0, mpi_rank = 0, mpi_size = 0;
#include"replica_core.hpp"
Replica replica;
int main(int argc, char** argv){
	if(argc < 6){ fprintf(stderr,"usage: beta Tsteps steps seed trace\n"); return 1; }
	const double beta = atof(argv[1]); const long Tst = atol(argv[2]), st = atol(argv[3]);
	rng.seed(atoi(argv[4]));
	divdiff_init();
	double curr=1; factorial[0]=curr; zero -= zero;
	for(int k=1;k<qmax;k++){ curr*=k; factorial[k]=curr; }
	for(int i=0;i<Ncycles;i++) cycle_len[i] = cycles[i].count();
	cycle_min_len = 64; for(int i=0;i<Ncycles;i++) cycle_min_len = std::min(cycle_min_len,cycle_len[i]);
	cycle_max_len = 0; for(int i=0;i<Ncycles;i++) cycle_max_len = std::max(cycle_max_len,cycle_len[i]);
	for(int i=0;i<Nop;i++) for(int j=0;j<Ncycles;j++) if(cycles[j].test(i)) P_in_cycles[i].set(j);
	replica.init(beta);
	for(long s=0;s<Tst;s++) replica.update();
	g_trace = fopen(argv[5],"w"); g_trace_target = &replica.d;
	fprintf(g_trace,"# beta %.17g N %d Nop %d\n", beta, N, Nop);
	replica.d.trace_snapshot();
	double qsum=0; int qmx=0;
	for(long s=0;s<st;s++){ replica.update(); qsum += replica.q; qmx = std::max(qmx,replica.q); }
	fclose(g_trace);
	printf("beta=%g <q>=%.1f max q=%d\n", beta, qsum/st, qmx);
	return 0;
}
