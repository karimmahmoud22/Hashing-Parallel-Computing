%%writefile hash_kernal.cu

#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "algorithm"
#include "random"
#include "stdint.h"
#include "stdio.h"
#include "vector"
#include "chrono"
#include <cuda_runtime.h>

#define HASH_TABLE_CAPACITY 1 * (2 * 1024 * 1024)  // 1 million pairs
#define NUM_OF_PAIRS (HASH_TABLE_CAPACITY / 2)   // divided by two because each entry in the hash table will store a key-value pair.
#define EMPTY_SLOT_VALUE 0xffffffff  // -1
#define SEARCH_PAIRS_COUNT 50
constexpr int NUM_STREAMS = 4;

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

cudaStream_t streams[NUM_STREAMS];
struct TableItem
{
    uint32_t key;
    uint32_t value;
};


__device__ uint32_t hashFunction(uint32_t key)
{
    // This XOR operation with a shifted version of k helps in spreading out the input bits across the entire 32-bit word.
    key ^= key >> 16;
    // Multiplication is used to further mix the bits of the input, leading to better dispersion of hash values
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    // Finally, another round of XOR and bit shifting is performed to ensure the final hash value is well-distributed and within the desired range.
    // This final mixing step ensures that the hash value is well-distributed across the entire range of 32-bit integers.
    key ^= key >> 16;
    // The final hash value is masked with HASH_TABLE_CAPACITY - 1 to ensure that it falls within the range of the hash table's capacity.
    // This is typically done to ensure that the hash value maps to a valid index within the hash table array.
    key &= (HASH_TABLE_CAPACITY - 1);
    return key;
}

// Create a hash table. For linear probing, this is just an array of KeyValues
TableItem* init_hashtable() 
{
    // Allocate memory
    TableItem* table;
    cudaMalloc(&table, sizeof(TableItem) * HASH_TABLE_CAPACITY);

    // Initializes the allocated memory block with the value 0xff, which corresponds to the EMPTY_SLOT_VALUE value.
    // This effectively sets all entries in the hash table to empty.
    cudaMemset(table, 0xff, sizeof(TableItem) * HASH_TABLE_CAPACITY);

    return table;
}

// Insert the key/values in kvs into the hashtable
__global__ void insertKernal(TableItem* table_ptr, const TableItem* pairs_to_insert, unsigned int size)
{
    unsigned int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (t_idx < size)
    {
        uint32_t key = pairs_to_insert[t_idx].key;
        uint32_t value = pairs_to_insert[t_idx].value;
        uint32_t insert_location = hashFunction(key);

        while (true)
        {
            uint32_t location_found = atomicCAS(&table_ptr[insert_location].key, EMPTY_SLOT_VALUE, key);
            if (location_found == EMPTY_SLOT_VALUE || location_found == key)
            {
                table_ptr[insert_location].value = value;
                return;
            }

            insert_location = (insert_location + 1) & (HASH_TABLE_CAPACITY-1); // linear proboing to handle the collision
        }
    }
}
 
void insertPairs(TableItem* table_ptr, const TableItem* pairs_to_insert, uint32_t size)
{
    // Copy the keyvalues to the GPU
    TableItem* d_pairs;
    cudaMalloc(&d_pairs, sizeof(TableItem) * size);
    cudaMemcpy(d_pairs, pairs_to_insert, sizeof(TableItem) * size, cudaMemcpyHostToDevice);

    int min_grid;
    int threads_per_block;
    // Using a CUDA runtime API function to determine the optimal block size for launching the kernel function
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid,
        &threads_per_block,
        insertKernal,
        0,
        0 
        );


    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table ( streaming )
    int inserting_batches = 16;
    uint32_t insert_batch_size = size / inserting_batches;
    for (uint32_t i = 0; i < inserting_batches; i++) {
        int streamIndex = i % NUM_STREAMS;  // Use round-robin scheduling for streams
        insertKernal<<<((insert_batch_size + threads_per_block - 1) / threads_per_block), threads_per_block, 0, streams[streamIndex]>>>(table_ptr, d_pairs + i * insert_batch_size, insert_batch_size);
    }

    // Wait for all streams to finish
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }


    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("Inserted %d pairs in %f ms\n", size, milliseconds);

    cudaFree(d_pairs);
}

// Lookup keys in the hashtable, and return the values
__global__ void searchKernal(TableItem* table_ptr, TableItem* target, unsigned int size)
{
    unsigned int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < size)
    {
        uint32_t key = target[t_idx].key;
        uint32_t search_location = hashFunction(key);

        while (true)
        {
            if (table_ptr[search_location].key == key)
            {
                target[t_idx].value = table_ptr[search_location].value;
                return;
            }
            // if empty search_location, so it is not exist, return empty value ( -1 )
            if (table_ptr[search_location].key == EMPTY_SLOT_VALUE)
            {
                target[t_idx].value = EMPTY_SLOT_VALUE;
                return;
            }
            search_location = (search_location + 1) & (HASH_TABLE_CAPACITY - 1);
        }
    }
}

void searchPairs(TableItem* table_ptr, TableItem* target, uint32_t size)
{
    // Copy the keyvalues to the GPU
    TableItem* d_pairs;
    cudaMalloc(&d_pairs, sizeof(TableItem) * size);
    cudaMemcpy(d_pairs, target, sizeof(TableItem) * size, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int min_grid;
    int threads_per_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &threads_per_block, searchKernal, 0, 0);
    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Insert all the keys into the hash table
    searchKernal<<<((uint32_t)size + threads_per_block - 1) / threads_per_block, threads_per_block >>> (table_ptr, d_pairs, (uint32_t)size);
    // copy results to device
    TableItem* result;
    cudaMemcpy(result, d_pairs, sizeof(TableItem) * size, cudaMemcpyDeviceToHost);

    // print the pair found d_pairs
    printf("\n ====== Result of Searching ====== \n");
    for (int i = 0; i < size; i++)
    {
        printf("key: %d, value: %d\n", result[i].key, result[i].value);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("Searched for %d items in %f ms\n", size, milliseconds);

    cudaFree(d_pairs);
}

__global__ void removeKernal(TableItem* table_ptr, const TableItem* pairs_to_remove, unsigned int size)
{
    unsigned int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < size)
    {
        uint32_t key = pairs_to_remove[t_idx].key;
        uint32_t slot = hashFunction(key);

        while (true)
        {
            if (table_ptr[slot].key == key)
            {
                table_ptr[slot].value = EMPTY_SLOT_VALUE;
                return;
            }
            if (table_ptr[slot].key == EMPTY_SLOT_VALUE)
            {
                return;
            }
            slot = (slot + 1) & (HASH_TABLE_CAPACITY - 1);
        }
    }
}

void removePairs(TableItem* table_ptr, const TableItem* pairs_to_remove, uint32_t size)
{
    // Copy the keyvalues to the GPU
    TableItem* d_pairs;
    cudaMalloc(&d_pairs, sizeof(TableItem) * size);
    cudaMemcpy(d_pairs, pairs_to_remove, sizeof(TableItem) * size, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int min_grid;
    int threads_per_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &threads_per_block, removeKernal, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table 
    removeKernal<<<((uint32_t)size + threads_per_block - 1) / threads_per_block, threads_per_block >>>(table_ptr, d_pairs, (uint32_t)size);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("Removed %d pairs in %f ms\n", size, milliseconds);

    cudaFree(d_pairs);
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__ void fetchPairsKernal(TableItem* table_ptr, TableItem* result, uint32_t* size)
{
    unsigned int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < HASH_TABLE_CAPACITY) 
    {
        if (table_ptr[t_idx].key != EMPTY_SLOT_VALUE) 
        {
            uint32_t value = table_ptr[t_idx].value;
            if (value != EMPTY_SLOT_VALUE)
            {
                uint32_t count = atomicAdd(size, 1);
                result[count] = table_ptr[t_idx];
            }
        }
    }
}

std::vector<TableItem> fetchPairs(TableItem* table_ptr)
{
    uint32_t* d_pairs_size;
    cudaMalloc(&d_pairs_size, sizeof(uint32_t));
    cudaMemset(d_pairs_size, 0, sizeof(uint32_t));

    TableItem* d_pairs;
    cudaMalloc(&d_pairs, sizeof(TableItem) * NUM_OF_PAIRS);

    int min_grid;
    int threads_per_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &threads_per_block, fetchPairsKernal, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    fetchPairsKernal<<<(HASH_TABLE_CAPACITY + threads_per_block - 1) / threads_per_block, threads_per_block>>>(table_ptr, d_pairs, d_pairs_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    

    uint32_t size;
    cudaMemcpy(&size, d_pairs_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<TableItem>result;
    result.resize(size);

    cudaMemcpy(result.data(), d_pairs, sizeof(TableItem) * size, cudaMemcpyDeviceToHost);

    
    printf("Fetched %d pairs in %f ms\n", size, milliseconds);


    cudaFree(d_pairs);
    cudaFree(d_pairs_size);

    return result;
}


std::vector<TableItem> getRandomPairs(std::mt19937& rnd, uint32_t size)
{
    std::uniform_int_distribution<uint32_t> dis(0, EMPTY_SLOT_VALUE - 1);

    std::vector<TableItem> pairs;
    pairs.reserve(size);

    for (uint32_t i = 0; i < size; i++)
    {
        uint32_t rand_key = dis(rnd);
        uint32_t rand_val = dis(rnd);
        pairs.push_back(TableItem{rand_key, rand_val});
    }

    return pairs;
}

std::vector<TableItem> shuffling(std::mt19937& rnd, std::vector<TableItem>pairs, uint32_t count)
{
    std::shuffle(pairs.begin(), pairs.end(), rnd);

    std::vector<TableItem>result;
    result.resize(count);

    std::copy(pairs.begin(), pairs.begin() + count, result.begin());

    return result;
}

double calulateTime(Time start_time) 
{
    Time end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d = end_time - start_time;
    std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds>(d);
    return ms.count() / 1000.0f;
}


int main() 
{
    std::random_device rd;
    // uint32_t seed = rd();
    uint32_t seed = 2793116369;
    std::mt19937 rnd(seed);


    printf("Starting to generate random pairs \n");
    

    std::vector<TableItem> pairs_to_insert = getRandomPairs(rnd, NUM_OF_PAIRS);
    std::vector<TableItem> pairs_to_remove = shuffling(rnd, pairs_to_insert, NUM_OF_PAIRS / 2);

    // pick random pairs to search for them in the hash table
    std::vector<TableItem> pairs_to_search = shuffling(rnd, pairs_to_insert, SEARCH_PAIRS_COUNT);

    // print the pairs_to_search
    printf("====== Randomly Choosen %d pairs for searching ====== \n\n", SEARCH_PAIRS_COUNT);
    for (int i = 0; i < SEARCH_PAIRS_COUNT; i++)
    {
        printf("key: %d, value: %d\n", pairs_to_search[i].key, pairs_to_search[i].value);
    }


    // Initialize CUDA streams
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    printf("\n ====== Testing insertion of %d elements ====== \n\n", (uint32_t)pairs_to_insert.size());

    Time timer = std::chrono::high_resolution_clock::now();

    TableItem* table_ptr = init_hashtable();

    // Insert testing
    const uint32_t inserting_batches = 1;
    uint32_t insert_batch_size = (uint32_t)pairs_to_insert.size() / inserting_batches;
    for (uint32_t i = 0; i < inserting_batches; i++)
    {
        insertPairs(table_ptr, pairs_to_insert.data() + i * insert_batch_size, insert_batch_size);
    }

    
    // remove testing
    const uint32_t removing_batches = 1;
    uint32_t remove_batch_size = (uint32_t)pairs_to_remove.size() / removing_batches;
    for (uint32_t i = 0; i < removing_batches; i++)
    {
        removePairs(table_ptr, pairs_to_remove.data() + i * remove_batch_size, remove_batch_size);
    }

    // search testing
    printf("\n======   Searching for the chosen %d elements after we removed some elements ======", (uint32_t)pairs_to_search.size());
    searchPairs(table_ptr, pairs_to_search.data(), SEARCH_PAIRS_COUNT);

    // Get all the pairs from the hash table
    printf("\n====== Testing Fetching all date from the table ====== \n\n");
    std::vector<TableItem>fetched_pairs = fetchPairs(table_ptr);

    // Free CUDA streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(table_ptr);

    // Summarize results
    double milliseconds = calulateTime(timer);
    double seconds = milliseconds / 1000.0f;
    printf("\n====== Total time for the whole process : %f ms  \n", milliseconds);


    printf("===================================  Finsidhed ================================== \n");

    return 0;
}
