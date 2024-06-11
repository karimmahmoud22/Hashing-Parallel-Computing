%%writefile cpu.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <random>

#define HASH_TABLE_CAPACITY 1 * 2 * 1024 * 1024
#define NUM_OF_PAIRS (HASH_TABLE_CAPACITY / 2)   // divided by two because each entry in the hash table will store a key-value pair.
#define EMPTY_SLOT_VALUE 0xffffffff  // -1
#define SEARCH_PAIRS_COUNT 1 * 1024 * 1024

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct TableItem
{
    uint32_t key;
    uint32_t value;
};

uint32_t hashFunction(uint32_t key)
{
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    key &= (HASH_TABLE_CAPACITY - 1);
    return key;
}

std::vector<TableItem> init_hashtable()
{
    std::vector<TableItem> table(HASH_TABLE_CAPACITY, {EMPTY_SLOT_VALUE, EMPTY_SLOT_VALUE});
    return table;
}

void insertPairs(std::vector<TableItem>& table, const std::vector<TableItem>& pairs_to_insert)
{
    int i = 0;
    for (const auto& pair : pairs_to_insert)
    {
        uint32_t insert_location = hashFunction(pair.key);
        while (true)
        {
            if (table[insert_location].key == EMPTY_SLOT_VALUE)
            {
                table[insert_location] = pair;
                break;
            }
            insert_location = (insert_location + 1) & (HASH_TABLE_CAPACITY-1); // Linear probing to handle collisions
            i++;
        }
    }
}

void searchPairs(const std::vector<TableItem>& table, std::vector<TableItem>& target, uint32_t size)
{
    for (auto& item : target)
    {
        uint32_t search_location = hashFunction(item.key);
        while (true)
        {
            if (table[search_location].key == item.key)
            {
                item.value = table[search_location].value;
                break;
            }
            if (table[search_location].key == EMPTY_SLOT_VALUE)
            {
                item.value = EMPTY_SLOT_VALUE;
                break;
            }
            search_location = (search_location + 1) & (HASH_TABLE_CAPACITY-1);
        }
    }
}

void removePairs(std::vector<TableItem>& table, const std::vector<TableItem>& pairs_to_remove)
{
    for (const auto& pair : pairs_to_remove)
    {
        uint32_t slot = hashFunction(pair.key);
        while (true)
        {
            if (table[slot].key == pair.key)
            {
                table[slot].value = EMPTY_SLOT_VALUE;
                break;
            }
            if (table[slot].key == EMPTY_SLOT_VALUE)
            {
                break;
            }
            slot = (slot + 1) & (HASH_TABLE_CAPACITY-1);
        }
    }
}

std::vector<TableItem> fetchPairs(const std::vector<TableItem>& table)
{
    std::vector<TableItem> result;
    for (const auto& item : table)
    {
        if (item.key != EMPTY_SLOT_VALUE && item.value != EMPTY_SLOT_VALUE)
        {
            result.push_back(item);
        }
    }
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
        pairs.push_back(TableItem{ rand_key, rand_val });
    }

    return pairs;
}

std::vector<TableItem> shuffling(std::mt19937& rnd, std::vector<TableItem> pairs, uint32_t count)
{
    std::shuffle(pairs.begin(), pairs.end(), rnd);

    std::vector<TableItem> result;
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

void printTime(const std::string& operation, const std::chrono::steady_clock::time_point& start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken for " << operation << ": " << duration.count() << " ms" << std::endl;
}

int main()
{
    std::random_device rd;
    // uint32_t seed = rd();
    uint32_t seed = 2793116369;
    std::mt19937 rnd(seed);

    std::cout << "Random seed = " << seed << std::endl;

    std::cout << "Starting to generate random pairs" << std::endl;

    std::vector<TableItem> pairs_to_insert = getRandomPairs(rnd, NUM_OF_PAIRS);
    std::vector<TableItem> pairs_to_remove = shuffling(rnd, pairs_to_insert, NUM_OF_PAIRS);
    std::vector<TableItem> pairs_to_search = shuffling(rnd, pairs_to_insert, NUM_OF_PAIRS);

    // std::cout << "====== Printing the pairs chosen for searching ======" << std::endl;
    // for (const auto& item : pairs_to_search)
    // {
    //     std::cout << "key: " << item.key << ", value: " << item.value << std::endl;
    // }

    std::cout << "\n ====== Testing insertion of " << pairs_to_insert.size() << " elements ======" << std::endl;

    Time timer = std::chrono::high_resolution_clock::now();

    std::vector<TableItem> table = init_hashtable();

    // Insert testing
    auto insertStart = std::chrono::steady_clock::now();
    insertPairs(table, pairs_to_insert);
    std::cout << "Insertion completed" << std::endl;
    printTime("insertPairs", insertStart);

    // Search testing
    std::cout << "\n ====== Testing searching for " << pairs_to_search.size() << " elements ======" << std::endl;
    auto searchStart = std::chrono::steady_clock::now();
    searchPairs(table, pairs_to_search, NUM_OF_PAIRS);
    std::cout << "Searching completed" << std::endl;
    printTime("searchPairs", searchStart);

    // Fetch testing
    std::cout << "\n ====== Testing Fetching all data from the table ======" << std::endl;
    auto fetchStart = std::chrono::steady_clock::now();
    std::vector<TableItem> fetched_pairs = fetchPairs(table);
    std::cout << "Fetched pairs size: " << fetched_pairs.size() << std::endl;
    printTime("fetchPairs", fetchStart);

    // Remove testing
    std::cout << "\n ====== Testing removing " << pairs_to_remove.size() << " elements ======" << std::endl;
    auto removeStart = std::chrono::steady_clock::now();
    removePairs(table, pairs_to_remove);
    std::cout << "Removing completed" << std::endl;
    printTime("removePairs", removeStart);


    // Summarize results
    double milliseconds = calulateTime(timer);
    std::cout << "\n====== Total time for the whole process : " << milliseconds << " ms ======" << std::endl;

    std::cout << "=================================== Finished ==================================" << std::endl;

    return 0;
}
