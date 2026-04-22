// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include "accel-sim.h"

#include <cstdlib>
#include <omp.h>

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary
 * format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

int main(int argc, const char **argv) {
  // Default to 1 OpenMP thread unless user set OMP_NUM_THREADS explicitly.
  // Rationale: on large-core hosts (e.g. 112-core NUMA server), OpenMP
  // defaults to nproc threads and per-cycle barrier contention can stall
  // the simulator for hours (observed 4 CTA/sec on 112-thread default vs
  // ~100+ CTA/sec on 1 thread).  Users who want parallelism set
  // OMP_NUM_THREADS=<N> in the environment and the explicit value wins.
  if (std::getenv("OMP_NUM_THREADS") == nullptr) {
    omp_set_num_threads(1);
  }

  accel_sim_framework accel_sim(argc, argv);
  accel_sim.simulation_loop();

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}
