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

#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
constexpr static int V = 16;
#elif defined(MG_USE_AVX512) || defined(MG_USE_SVE512)
constexpr static int V = 8;
#elif defined(MG_USE_AVX2)
constexpr static int V = 4;
#else
constexpr static int V = MG_VECLEN_SP;
#endif

TEST(TestKokkos, TestDslashVecSweep) {
  const IndexType L_start = 16;
  const IndexType L_max   = 64;
  const int iters         = 10;  // iterations per timing
#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
  const int reps = 20;  // repetitons of measurements
#else
  const int reps = 50;
#endif

  for (IndexType L = L_start; L <= L_max; L += 2) {
    IndexArray latdims = {{L, L, L, 4 * L}};

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

    KokkosCBFineSpinor<SIMDComplex<REAL32, V>, 4> kokkos_spinor_in(info, EVEN);
    KokkosCBFineSpinor<SIMDComplex<REAL32, V>, 4> kokkos_spinor_out(info, ODD);
    {
      multi1d<LatticeFermionF> psi_in(V);
      for (int v = 0; v < V; ++v) {
        gaussian(psi_in[v]);
      }
      // Import Spinor
      QDPLatticeFermionToKokkosCBSpinor(psi_in, kokkos_spinor_in);
      // QDP++ LatticeFermionF should go away here.
    }

    // Flops and Bandwidth calculations
    double rfo       = 1.0;
    double num_sites = static_cast<double>((latdims[0] / 2) * latdims[1] *
                                           latdims[2] * latdims[3]);
    double bytes_in  = static_cast<double>(
        (8 * 4 * 3 * 2 * sizeof(REAL32) * V + 8 * 3 * 3 * 2 * sizeof(REAL32)) *
        num_sites * iters);
    double bytes_out =
        static_cast<double>(V * 4 * 3 * 2 * sizeof(REAL32) * num_sites * iters);
    double rfo_bytes_out = (1.0 + rfo) * bytes_out;
    double flops         = static_cast<double>(1320.0 * V * num_sites * iters);

    MasterLog(
        INFO,
        "Memory Footprint:\n"
        "     - gauge field: %16.0lf bytes\n"
        "     - spinor in:   %16.0lf bytes\n"
        "     - spinor out:  %16.0lf bytes\n"
        "     - total:       %16.0lf bytes",
        static_cast<double>(3 * 3 * 2 * sizeof(REAL32) * num_sites * 2),
        static_cast<double>(4 * 3 * 2 * sizeof(REAL32) * V * num_sites),
        static_cast<double>(4 * 3 * 2 * sizeof(REAL32) * V * num_sites),
        static_cast<double>(
            2 * (3 * 3 * 2 * sizeof(REAL32) + 4 * 3 * 2 * sizeof(REAL32) * V) *
            num_sites));

#if defined(MG_KOKKOS_USE_MDRANGE) || defined(MG_FLAT_PARALLEL_DSLASH)
    // sites_per_team arbitrary since it's not used anyway
    int sites_per_team = 32;
    {
#else
    for (int sites_per_team = 1; sites_per_team < 8192; sites_per_team *= 2) {
#endif

#if defined(MG_FLAT_PARALLEL_DSLASH)
      MasterLog(INFO, "Timing Dslash: Vector size: %d", V);
#elif defined(MG_KOKKOS_USE_MDRANGE)
#if defined(MG_USE_CUDA) || defined(MG_USE_HIP)
      IndexArray best_blocks = {16, 16, 1, 1};
#else
      IndexArray best_blocks = {4, 2, 2, 16};
#endif
      MasterLog(INFO,
                "Timing Dslash: Vector size: %d (Bx,By,Bz,Bt)=(%d,%d,%d,%d)", V,
                best_blocks[0], best_blocks[1], best_blocks[2], best_blocks[3]);
#else
    MasterLog(INFO, "Timing Dslash: Vector size: %d sites_per_team: %d", V,
              sites_per_team);
#endif

      KokkosDslash<MGComplex<REAL32>, SIMDComplex<REAL32, V>,
                   ThreadSIMDComplex<REAL32, V>>
          D(info, sites_per_team);
      for (int rep = 0; rep < reps; ++rep) {
        // for(int isign=-1; isign < 2; isign+=2) {
        int isign = 1;

        Kokkos::Timer timer;
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
                  "time per iter = %lf (usec) Performance: "
                  "%lf GFLOPS Effective BW (RFO=1): %lf GB/sec",
                  time_taken * 1.0e6 / (double)(iters),
                  flops / (time_taken * 1.0e9),
                  (bytes_in + rfo_bytes_out) / (time_taken * 1.0e9));

        // } // isign
      }
    }  // -- sites_per_team
  }
}
