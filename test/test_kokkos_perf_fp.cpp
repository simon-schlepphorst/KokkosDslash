#include "kokkos_dslash_config.h"
#include "gtest/gtest.h"
//#include "../mock_nodeinfo.h"
#include "qdpxx_utils.h"
#include "dslashm_w.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "./kokkos_types.h"
#include "./kokkos_defaults.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

#include <chrono>
#include <ctime>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestKokkos, TestDslashFP) {
  std::vector<IndexType> xyz_values = {4, 8, 16, 32, 48, 56};
  std::vector<IndexType> t_values   = {8, 16, 32, 64, 128};

  for (auto const& x : xyz_values) {
    for (auto const& t : t_values) {
      IndexArray latdims = {{x, x, x, t}};
      int iters          = 100;

      initQDPXXLattice(latdims);
      LatticeInfo info(latdims, 4, 3, NodeInfo());
      KokkosFineGaugeField<MGComplex<REAL32>> kokkos_gauge(info);

      {
        multi1d<LatticeColorMatrixF> gauge_in(n_dim);
        for (int mu = 0; mu < n_dim; ++mu) {
          gaussian(gauge_in[mu]);
          reunit(gauge_in[mu]);
        }

        // Import gauge field
        QDPGaugeFieldToKokkosGaugeField(gauge_in, kokkos_gauge);
        // QDP Gauge field ought to go away here
      }

      KokkosCBFineSpinor<MGComplex<REAL32>, 4> kokkos_spinor_in(info, EVEN);
      KokkosCBFineSpinor<MGComplex<REAL32>, 4> kokkos_spinor_out(info, ODD);
      {
        LatticeFermionF psi_in;
        gaussian(psi_in);

        // Import Spinor
        QDPLatticeFermionToKokkosCBSpinor(psi_in, kokkos_spinor_in);
        // QDP++ LatticeFermionF should go away here.
      }

      // for(int sites_per_team=8; sites_per_team < 8192; sites_per_team *=2) {
      int sites_per_team = 32;

      KokkosDslash<MGComplex<REAL32>, MGComplex<REAL32>, MGComplex<REAL32>> D(
          info, sites_per_team);

      for (int rep = 0; rep < 10; ++rep) {
        // for(int isign=-1; isign < 2; isign+=2) {
        int isign = 1;
        MasterLog(INFO, "Timing Dslash: isign == %d", isign);
        // double start_time = omp_get_wtime();
        //  auto start_time = std::clock();
        // auto start_time = std::chrono::high_resolution_clock::now();
        Kokkos::Timer timer;
        timer.reset();

        for (int i = 0; i < iters; ++i) {
          D(kokkos_spinor_in, kokkos_gauge, kokkos_spinor_out, isign);
          Kokkos::fence();
        }
        double time_taken = timer.seconds();

        double rfo       = 1.0;
        double num_sites = static_cast<double>((latdims[0] / 2) * latdims[1] *
                                               latdims[2] * latdims[3]);
        // footprint dont't need iters and 8, rfo
        // neighbors Vector + Sites
        double bytes_in = static_cast<double>(
            (8 * 4 * 3 * 2 * sizeof(REAL32) + 8 * 3 * 3 * 2 * sizeof(REAL32)) *
            num_sites * iters);
        // returnvector
        double bytes_out =
            (1.0 + rfo) *
            static_cast<double>(4 * 3 * 2 * sizeof(REAL32) * num_sites * iters);
        double flops = static_cast<double>(1320.0 * num_sites * iters);

        MasterLog(INFO,
                  "sites_per_team=%d time per iter = %lf (usec) Performance: "
                  "%lf GFLOPS",
                  sites_per_team, time_taken * 1.0e6 / (double)(iters),
                  flops / (time_taken * 1.0e9));
        MasterLog(INFO, "sites_per_team=%d Effective BW: %lf GB/sec",
                  sites_per_team,
                  (bytes_in + bytes_out) / (time_taken * 1.0e9));

        // } // isign
      }
      //  } // -- sites_per_team
    }
  }
}