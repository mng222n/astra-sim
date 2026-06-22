#include "OverlapHierarchicalAllReduce.h"

namespace AstraSim {

OverlapHierarchicalAllReduce::OverlapHierarchicalAllReduce(
    std::string name, Sys* sys, uint64_t total_message_size, 
    int preferred_dataset_splits, double local_mem_bw)
    : Algorithm(name, sys), 
      num_chunks(preferred_dataset_splits),
      local_mem_bw_GBps(local_mem_bw),
      chunks_completed(0) 
{
    this->chunk_size = total_message_size / num_chunks;
    chunk_states.resize(num_chunks, OharPhase::INIT);

    // Extract topology data from the ASTRA-sim system instance
    this->local_ring_size = sys->get_nodes_in_ring("local");
    this->global_ring_size = sys->get_nodes_in_ring("global");
    this->local_id = sys->get_ring_id("local");
    this->global_id = sys->get_ring_id("global");
}

void OverlapHierarchicalAllReduce::run() {
    // Pipeline initialization: Push all chunks into the first state
    for (int i = 0; i < num_chunks; ++i) {
        advance_chunk(i);
    }
}

void OverlapHierarchicalAllReduce::advance_chunk(int chunk_id) {
    OharPhase current = chunk_states[chunk_id];

    switch (current) {
        case OharPhase::INIT:
            chunk_states[chunk_id] = OharPhase::LRS;
            inject_traffic(chunk_id, OharPhase::LRS);
            break;

        case OharPhase::LRS:
            chunk_states[chunk_id] = OharPhase::IRS;
            inject_traffic(chunk_id, OharPhase::IRS);
            break;

        case OharPhase::IRS:
            chunk_states[chunk_id] = OharPhase::IAG;
            inject_traffic(chunk_id, OharPhase::IAG);
            break;

        case OharPhase::IAG:
            chunk_states[chunk_id] = OharPhase::LAG;
            inject_traffic(chunk_id, OharPhase::LAG);
            break;

        case OharPhase::LAG:
            chunk_states[chunk_id] = OharPhase::DONE;
            chunks_completed++;
            if (chunks_completed == num_chunks) {
                exit(); // Notify ASTRA-sim that the collective is finished
            }
            break;
            
        case OharPhase::DONE:
            break;
    }
}

void OverlapHierarchicalAllReduce::inject_traffic(int chunk_id, OharPhase phase) {
    int dest_id;
    std::string network_level;

    // Determine routing based on the phase
    if (phase == OharPhase::LRS || phase == OharPhase::LAG) {
        // Intra-node routing (Local Ring)
        dest_id = (local_id + 1) % local_ring_size;
        network_level = "local";
    } else {
        // Inter-node routing (Global Ring)
        dest_id = (global_id + 1) % global_ring_size;
        network_level = "global";
    }

    // Abstract ASTRA-sim injection call (maps to SimSend/SimRecv generation)
    // When GASM is active, it intercepts this generation to prioritize chunks
    sys->generate_traffic(
        network_level, 
        dest_id, 
        chunk_size, 
        this,    // Reference to this algorithm to trigger callbacks
        chunk_id // Passing chunk_id as the message tag
    );
}

void OverlapHierarchicalAllReduce::on_message_arrival(int chunk_id) {
    OharPhase current = chunk_states[chunk_id];

    // LRS and IRS are reduction operations, meaning they require math operations 
    // on the GPU/NPU memory before proceeding. We apply the local-mem-bw delay here.
    if (current == OharPhase::LRS || current == OharPhase::IRS) {
        simulate_memory_delay(chunk_id);
    } else {
        // IAG and LAG are pure memory copies (concatenation). 
        // We advance immediately, or apply a smaller memcopy delay if desired.
        advance_chunk(chunk_id);
    }
}

void OverlapHierarchicalAllReduce::simulate_memory_delay(int chunk_id) {
    // Convert 50 GB/s to nanosecond delay per byte
    double delay_ns = (double)chunk_size / local_mem_bw_GBps;
    
    // Register the delay with the ASTRA-sim event queue.
    // Use a lambda function to advance the chunk once the memory delay completes.
    sys->register_event(EventType::Workload, delay_ns, [this, chunk_id]() {
        this->advance_chunk(chunk_id);
    });
}

} // namespace AstraSim