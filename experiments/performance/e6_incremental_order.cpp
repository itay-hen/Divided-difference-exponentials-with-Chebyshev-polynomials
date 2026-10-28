// =============================================================================
// e6_incremental_order.cpp -- Experiment E6c: does node ORDER (and the fixed
// interval) affect the accuracy of IncrementalExpDivDiff?
//
// (A) Large-q failures (q >= ~250 at c = 5, q >= ~400 at c = 50): initialize
//     with random / ascending / descending order.  Result: order-independent;
//     the failure is the double range of the UNSCALED incremental table
//     (R_n(c) underflows to 0 while D_q^(n) overflows -> NaN), not ordering.
// (B) Accuracy on wide intervals: c in {5, 20, 50}, q in {5..100}, 100 uniform
//     random node sets in [-c, c], incremental object on [-c, c] (fixed),
//     built either by initialize(all nodes) or by initialize({x_0}) + q
//     appends, in random / ascending / descending order.  Error vs MNP
//     (MNP is accurate to ~1e-14 for these q).  Also the batch evaluator on
//     ascending input with internal sorting on (default) / off.
// Output: stdout and e6_incremental_order.csv
// =============================================================================
#include "chebyshev_exp_divdiff.hpp"
#include "incremental_chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
using Inc = chebyshev_incremental::IncrementalExpDivDiff<double>;
int main(int argc, char** argv){
  const std::string outdir = (argc > 1) ? argv[1] : ".";
  FILE* f = std::fopen((outdir + "/e6_incremental_order.csv").c_str(), "w");
  std::fprintf(f, "c,q,n_sets,build,order,worst_rel_err,throws\n");
  // (A) large-q range failure vs order
  for (double c : {5.0, 50.0}) for (int q : {200, 250, 400}) {
    std::mt19937_64 g(1); std::uniform_real_distribution<double> U(-c,c);
    std::vector<double> x(q+1); for(auto&v:x) v=U(g);
    for (int o=0;o<3;o++){
      auto y=x; if(o==1) std::sort(y.begin(),y.end()); if(o==2) std::sort(y.begin(),y.end(),std::greater<double>());
      Inc inc(-c,c,q+10);
      try{ double S=inc.initialize_normalized(y); printf("A c=%g q=%d order=%s S=%.3e\n",c,q,o==0?"rand":o==1?"asc":"desc",S);}
      catch(std::exception&e){printf("A c=%g q=%d order=%s THROW %.60s\n",c,q,o==0?"rand":o==1?"asc":"desc",e.what());}
    }
  }
  // (B) accuracy vs order on wide intervals, vs MNP (accurate ~1e-14 here)
  for (double c : {5.0, 20.0, 50.0}) for (int q : {5, 10, 20, 50, 100}) {
    double worst[3][2]={{0}}; int thr[3][2]={{0}}; int n=100;
    double wb[2]={0,0}; int bref[2]={0,0};
    std::mt19937_64 g(7+q); std::uniform_real_distribution<double> U(-c,c);
    for(int s=0;s<n;s++){
      std::vector<double> x(q+1); for(auto&v:x) v=U(g);
      double m=mccurdy_templated::ExpDivDiff<double>::compute(x);
      for(int o=0;o<3;o++){
        auto y=x; if(o==1) std::sort(y.begin(),y.end()); if(o==2) std::sort(y.begin(),y.end(),std::greater<double>());
        for(int mode=0;mode<2;mode++){ // 0: initialize all; 1: init with y0 then append
          Inc inc(-c,c,q+10);
          try{ double v;
            if(mode==0) v=inc.initialize(y);
            else { inc.initialize({y[0]}); for(int k=1;k<=q;k++) v=inc.add_node(y[k]); }
            worst[o][mode]=std::max(worst[o][mode], std::fabs(v-m)/std::fabs(m));
          }catch(...){thr[o][mode]++;}
        }
      }
      for(int so=0;so<2;so++){ chebyshev::ExpDivDiff<double> E(-c,c); E.set_sort_nodes(so==0);
        auto y=x; std::sort(y.begin(),y.end()); // ascending input: worst case for unsorted
        try{ double v=E.evaluate(y); wb[so]=std::max(wb[so],std::fabs(v-m)/std::fabs(m)); }catch(...){bref[so]++;} }
    }
    printf("B c=%-3g q=%-3d inc worst err (throws) init:rand %.1e(%d) asc %.1e(%d) desc %.1e(%d) | append:rand %.1e(%d) asc %.1e(%d) desc %.1e(%d) | batch asc-input sort:on %.1e(%d) off %.1e(%d)\n",
      c,q,worst[0][0],thr[0][0],worst[1][0],thr[1][0],worst[2][0],thr[2][0],worst[0][1],thr[0][1],worst[1][1],thr[1][1],worst[2][1],thr[2][1],wb[0],bref[0],wb[1],bref[1]);
    const char* on[3]={"random","ascending","descending"};
    for(int mode=0;mode<2;mode++) for(int o=0;o<3;o++)
      std::fprintf(f,"%g,%d,%d,%s,%s,%.3e,%d\n",c,q,n,mode==0?"initialize":"appends",on[o],worst[o][mode],thr[o][mode]);
    std::fprintf(f,"%g,%d,%d,batch_sort_on,ascending,%.3e,%d\n",c,q,n,wb[0],bref[0]);
    std::fprintf(f,"%g,%d,%d,batch_sort_off,ascending,%.3e,%d\n",c,q,n,wb[1],bref[1]);
    fflush(stdout);
  }
  std::fclose(f);
}
