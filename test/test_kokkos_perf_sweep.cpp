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

TEST(TestKokkos, TestDslashSweep) {
  const IndexType L_start = 4;
  const IndexType L_max   = 64;
  const int default_iters = 10;  // iterations per timing
#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
  const int reps = 20;  // repetitons of measurements
#else
  const int reps = 50;
#endif

  for (IndexType L = L_start; L <= L_max; L += 2) {
    IndexArray latdims = {{L, L, L, L}};

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

    // Flops and Bandwidth calculations 1
    double num_sites = static_cast<double>((latdims[0] / 2) * latdims[1] *
                                           latdims[2] * latdims[3]);

    MasterLog(
        INFO,
        "Memory Footprint:\n"
        "     - gauge field: %16.0lf bytes\n"
        "     - spinor in:   %16.0lf bytes\n"
        "     - spinor out:  %16.0lf bytes\n"
        "     - total:       %16.0lf bytes",
        static_cast<double>(3 * 3 * 2 * sizeof(REAL32) * num_sites * 2),
        static_cast<double>(4 * 3 * 2 * sizeof(REAL32) * num_sites),
        static_cast<double>(4 * 3 * 2 * sizeof(REAL32) * num_sites),
        static_cast<double>(
            2 * (3 * 3 * 2 * sizeof(REAL32) + 4 * 3 * 2 * sizeof(REAL32)) *
            num_sites));

#if defined(MG_KOKKOS_USE_MDRANGE) || defined(MG_FLAT_PARALLEL_DSLASH)
    // sites_per_team arbitrary since it's not used anyway
    int sites_per_team = 32;
    {
#else
    for (int sites_per_team = 1; sites_per_team < 8192; sites_per_team *= 2) {
#endif

#if defined(MG_FLAT_PARALLEL_DSLASH)
      MasterLog(INFO, "Timing Dslash:");
#elif defined(MG_KOKKOS_USE_MDRANGE)
#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
      IndexArray best_blocks = {16, 16, 1, 1};
#else
      IndexArray best_blocks = {4, 2, 2, 16};
#endif
      MasterLog(INFO, "Timing Dslash: (Bx,By,Bz,Bt)=(%d,%d,%d,%d)",
                best_blocks[0], best_blocks[1], best_blocks[2], best_blocks[3]);
#else
    MasterLog(INFO, "Timing Dslash: sites_per_team: %d", sites_per_team);
#endif

      KokkosDslash<MGComplex<REAL32>, MGComplex<REAL32>, MGComplex<REAL32>> D(
          info, sites_per_team);

      Kokkos::Timer timer;
      timer.reset();
      int iters = 0;
      while (timer.seconds() < 0.5) {
        int isign = 1;
#if defined(MG_KOKKOS_USE_MDRANGE)
        D(kokkos_spinor_in, kokkos_gauge, kokkos_spinor_out, isign,
          best_blocks);
#else
        D(kokkos_spinor_in, kokkos_gauge, kokkos_spinor_out, isign);
#endif
        Kokkos::fence();
        iters += 1;
      }
      if (iters < default_iters) {
        iters = default_iters;
      }
      MasterLog(INFO, "Iterations per timing: %d", iters);

      // Flops and Bandwidth calculations 2
      double rfo      = 1.0;
      double bytes_in = static_cast<double>(
          (8 * 4 * 3 * 2 * sizeof(REAL32) + 8 * 3 * 3 * 2 * sizeof(REAL32)) *
          num_sites * iters);
      double bytes_out =
          static_cast<double>(4 * 3 * 2 * sizeof(REAL32) * num_sites * iters);
      double rfo_bytes_out = (1.0 + rfo) * bytes_out;
      double flops         = static_cast<double>(1320.0 * num_sites * iters);

      for (int rep = 0; rep < reps; ++rep) {
        // for(int isign=-1; isign < 2; isign+=2) {
        int isign = 1;

        timer.reset();

        for (int i = 0; i < iters; ++i) {
#if defined(MG_KOKKOS_USE_MDRANGE)
          D(kokkos_spinor_in, kokkos_gauge, kokkos_spinor_out, isign,
            best_blocks);
#else
          D(kokkos_spinor_in, kokkos_gauge, kokkos_spinor_out, isign);
#endif
          Kokkos::fence();
        }
        double time_taken = timer.seconds();

        MasterLog(INFO,
                  "time = %lf (msec) "
                  "time per iter = %lf (usec) Performance: "
                  "%lf GFLOPS Effective BW (RFO=1): %lf GB/sec",
                  time_taken * 1.0e3, time_taken * 1.0e6 / (double)(iters),
                  flops / (time_taken * 1.0e9),
                  (bytes_in + rfo_bytes_out) / (time_taken * 1.0e9));

        // } // isign
      }
    }  // -- sites_per_team
  }
}