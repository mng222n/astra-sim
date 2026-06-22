#ifndef OVERLAP_HIERARCHICAL_ALLREDUCE_H
#define OVERLAP_HIERARCHICAL_ALLREDUCE_H

#include "astra-sim/system/collective/Algorithm.h"
#include "astra-sim/system/Sys.h"
#include <vector>

namespace AstraSim {

// The strict progression of the OHAR pipeline
enum class OharPhase {
    INIT,
    LRS,  // Local Reduce-Scatter (Intra-node)
    IRS,  // Inter-node Reduce-Scatter (Global)
    IAG,  // Inter-node All-Gather (Global)
    LAG,  // Local All-Gather (Intra-node)
    DONE
};

class OverlapHierarchicalAllReduce : public Algorithm {
private:
    int num_chunks;
    uint64_t chunk_size;
    double local_mem_bw_GBps;
    
    // Topology awareness
    int local_ring_size;
    int global_ring_size;
    int local_id;
    int global_id;

    // Independent state trackers for each dataset split
    std::vector<OharPhase> chunk_states;
    int chunks_completed;

    // Core state machine drivers
    void advance_chunk(int chunk_id);
    void inject_traffic(int chunk_id, OharPhase phase);
    void simulate_memory_delay(int chunk_id);

public:
    OverlapHierarchicalAllReduce(std::string name, Sys* sys, uint64_t total_message_size, 
                                 int preferred_dataset_splits, double local_mem_bw);
    virtual ~OverlapHierarchicalAllReduce() = default;

    virtual void run() override;
    
    // Callback invoked by the network backend when a chunk arrives
    void on_message_arrival(int chunk_id);
};

} // namespace AstraSim

#endif // OVERLAP_HIERARCHICAL_ALLREDUCE_H