// ***********************************************************************************
// SEARCH n VALUES BY mr
// ***********************************************************************************
//
// Author: Javier Hernandez
//
// Email:  271314@pm.me
// 
// Description:
//   High-performance parallel search engine for finding n values that belong to 
//   a specific mr class. A number n belongs to mr class when its Collatz sequence
//   encounters the pseudocycle associated with mr as the first complete pseudocycle
//   (both ni and nj values must be found in order).
//
//   In other words: "Searching for n values belonging to the mr class"
//
//   Companion computational tools and step-by-step visualizations available at:
//   https://github.com/hhvvjj/a-new-algebraic-framework-for-the-collatz-conjecture
//
// Features:
//   - Ordered insertion maintains sorted output (no post-processing needed)
//   - Dynamic file creation (only creates files for taxonomies that have results)
//   - Smart taxonomy detection (only A+B or A+C, never all three)
//   - Single-line JSON output format
//   - In-memory sorted buffer for efficient ordered writing
//   - Requires BOTH ni AND nj for complete pseudocycle detection
//   - Thread-safe file creation without deadlocks
//   - Taxonomy classification based on GLOBAL maximum position
//   - Continues sequence to end to find true global maximum
//   - Automatic checkpoint/resume capability for long-running searches
//   - Graceful interruption handling (Ctrl+C) with state preservation
//
// Usage:
//   ./search_n_values_by_mr <exponent> <target_mr>
//   Example: ./search_n_values_by_mr 25 1821
//            (finds all n < 2^25 that belong to mr class 1821)
//
// Output:
//   - JSON file: results_exp_<exponent>_mr_<target_mr>_<taxonomy>.json`
//   - Console: Real-time progress and comprehensive summary report
//   - Checkpoint: checkpoint.bin (auto-saved every 5 minutes, auto-removed on completion)
//
// License:
//   CC-BY-NC-SA 4.0 International 
//   For additional details, visit:
//   https://creativecommons.org/licenses/by-nc-sa/4.0/
//
//   For full details, visit 
//   https://github.com/hhvvjj/search-n-values-by-mr/blob/main/LICENSE
//
// Research Reference:
//   Based on the tuple-based transform methodology described in:
//   https://doi.org/10.5281/zenodo.15546925

// ***********************************************************************************
// 1. HEADERS, DEFINES, TYPEDEFS & GLOBAL VARIABLES
// ***********************************************************************************

// =============================
// SYSTEM HEADERS
// =============================
#include <signal.h>     // Signal handling for Ctrl+C interrupt detection
#include <stdio.h>      // Standard I/O for file operations and console output
#include <stdlib.h>     // Memory allocation, string conversion, program exit
#include <string.h>     // Memory operations (memset, memmove) and string handling
#include <stdint.h>     // Fixed-width integer types (uint64_t, uint32_t)
#include <stdbool.h>    // Boolean type and true/false constants
#include <unistd.h>     // POSIX file operations (access, rename)
#include <omp.h>        // OpenMP parallel programming (threads, locks, timing)     

// =============================
// CHECKPOINT CONFIGURATION
// =============================

#define CHECKPOINT_INTERVAL 300.0           // Seconds between automatic checkpoint saves (5 minutes)
#define CHECKPOINT_FILE "checkpoint.bin"    // Primary checkpoint filename for search state persistence
#define CHECKPOINT_BACKUP "checkpoint.bak"  // Backup checkpoint for atomic write safety
#define CHECKPOINT_MAGIC 0x4E425952         // Magic number "NBYR" for file validation
#define CHECKPOINT_VERSION 1                // Checkpoint format version for compatibility checking
                                            // Note: Exponents > 45 are computationally impractical

// =============================
// SAFETY AND PERFORMANCE LIMITS
// =============================

#define MAX_SEQUENCE_LENGTH 50000           // Maximum Collatz sequence iterations before forced termination
#define MAX_SAFE_ITERATIONS 10000           // Iteration threshold for divergence detection
#define DIVERGENCE_MULTIPLIER 1000          // Factor for detecting exponential growth (current > n * 1000)

#define PROGRESS_UPDATE_INTERVAL 10.0       // Seconds between console progress reports
#define PROGRESS_CHECK_FREQUENCY 1000000    // Iterations between progress update checks (thread 0 only)
#define PROGRESS_ETA_THRESHOLD 0.01         // Minimum progress percentage before showing ETA (0.01%)

#define TAXONOMY_STRING_SIZE 8              // Buffer size for taxonomy character storage
#define DEFAULT_CHUNK_SIZE 1000             // Default work chunk size for parallel scheduling
#define LOCAL_COUNTER_UPDATE_FREQUENCY 100  // Iterations between thread-local to global counter sync

#define MIN_EXPONENT 1                      // Minimum valid exponent (2^1 = 2, smallest meaningful range)
#define MAX_EXPONENT 63                     // Maximum valid exponent (2^63, prevents uint64_t overflow)

// =============================
// HASH TABLE CONFIGURATION
// =============================

#define HASH_SIZE 8192                      // Hash table bucket count (power of 2 for efficient modulo)
#define HASH_MASK (HASH_SIZE - 1)           // Bitmask for fast modulo via bitwise AND

// =============================
// SORTED BUFFER CONFIGURATION
// =============================

#define INITIAL_BUFFER_CAPACITY 100000      // Initial allocation size for sorted buffers (grows exponentially)
#define BUFFER_FLUSH_THRESHOLD 50000        // Buffer element count triggering automatic flush to disk

// =============================
// SIGNAL HANDLING
// =============================

volatile sig_atomic_t checkpoint_signal_received = 0;  // Flag set by SIGINT handler (Ctrl+C), checked by thread 0

// =============================
// CORE DATA STRUCTURES
// =============================

/**
 * @brief Represents a single pseudocycle entry in the Collatz conjecture dictionary.
 * 
 * Stores the complete mathematical definition of a pseudocycle:
 * - target_mr: Unique identifier for this pseudocycle class
 * - ni: First boundary value encountered when entering the pseudocycle
 * - nj: Second boundary value that completes the pseudocycle detection
 * - distance: Separation between ni and nj in the sequence trajectory
 * 
 * Special cases: mr=0 (trivial cycle with ni=nj=1), single-value pseudocycles (ni==nj)
 */
typedef struct {
    uint64_t target_mr;
    uint64_t ni;
    uint64_t nj;
    int distance;
} dictionary_entry_t;

/**
 * @brief Results of Collatz sequence analysis for taxonomy classification.
 * 
 * Contains complete analysis of a number's Collatz trajectory:
 * - max_value: Global maximum value reached in the sequence
 * - ni_position: Position where first pseudocycle boundary (ni) was found
 * - nj_position: Position where second pseudocycle boundary (nj) was found
 * - max_position: Position where the global maximum occurred
 * - taxonomy: Classification (A/B/C) based on maximum position relative to pseudocycle
 * - found_pseudocycle: Whether both ni and nj were successfully detected
 * 
 * Taxonomy rules: A (max before ni), B (max between ni and nj inclusive), C (max after nj)
 */
typedef struct {
    uint64_t max_value;
    int ni_position;
    int nj_position;
    int max_position;
    char taxonomy;
    bool found_pseudocycle;
    char padding[6];
} collatz_analysis_t;

/**
 * @brief Thread-safe sorted buffer for maintaining ordered n values in memory.
 * 
 * Provides efficient in-memory storage with automatic ordering:
 * - numbers: Dynamically allocated array of sorted uint64_t values
 * - count: Current number of elements in the buffer
 * - capacity: Total allocated capacity (grows exponentially when needed)
 * - lock: OpenMP lock for thread-safe concurrent insertions
 * 
 * Supports binary search insertion (O(log n) lookup + O(n) shift) and periodic
 * flushing to disk when threshold is reached.
 */
typedef struct {
    uint64_t* numbers;
    size_t count;
    size_t capacity;
    omp_lock_t lock;
} sorted_buffer_t;

/**
 * @brief Manages dynamic file creation and streaming output for taxonomy results.
 * 
 * Handles lazy file creation and thread-safe writing for up to 2 taxonomy files:
 * - file_1, file_2: File pointers for the two active taxonomy files
 * - taxonomy_1, taxonomy_2: Which taxonomies ('A', 'B', or 'C') each file represents
 * - file_X_created: Flags tracking which files have been created
 * - first_write_X: Flags for JSON array formatting (comma handling)
 * - creation_lock: Ensures thread-safe file creation without deadlocks
 * - buffer_A, buffer_B, buffer_C: In-memory sorted buffers for each taxonomy
 * - filename_X: Pre-formatted filenames for each possible taxonomy output
 * - exponent, target_mr: Search parameters for filename generation
 * 
 * Only creates files for taxonomies that actually have results (A+B or A+C, never all three).
 * For mr=0, only taxonomy A is used.
 */
typedef struct {
    FILE* file_1;
    FILE* file_2;
    char taxonomy_1;
    char taxonomy_2;
    bool file_1_created;
    bool file_2_created;
    bool first_write_1;
    bool first_write_2;
    omp_lock_t creation_lock;
    sorted_buffer_t buffer_A;
    sorted_buffer_t buffer_B;
    sorted_buffer_t buffer_C;
    char filename_A[256];
    char filename_B[256];
    char filename_C[256];
    int exponent;
    uint64_t target_mr;
} streaming_writer_t;

/**
 * @brief Tracks search progress for real-time reporting and statistics.
 * 
 * Maintains thread-safe counters for progress monitoring:
 * - processed: Total numbers processed so far
 * - found_count: Total valid n values found belonging to target mr class
 * - last_n_found: Most recently discovered n value (for reporting)
 * - last_update_time: Timestamp of last progress report (for throttling)
 * - lock: OpenMP lock for thread-safe counter updates
 * 
 * Updated atomically by worker threads and read periodically for progress display.
 */
typedef struct {
    uint64_t processed;
    uint64_t found_count;
    uint64_t last_n_found;
    double last_update_time;
    omp_lock_t lock;
} progress_tracker_t;

/**
 * @brief Checkpoint file header for resume capability.
 * 
 * Stores minimal state needed to resume interrupted searches:
 * - magic_number: File validation marker (0x4E425952 = "NBYR")
 * - version: Checkpoint format version for compatibility checking
 * - exponent: Search range exponent (must match on resume)
 * - target_mr: Target pseudocycle mr value (must match on resume)
 * - last_n: Last number successfully processed before interruption
 * - total_processed: Cumulative count of processed numbers
 * - found_count: Cumulative count of matches found
 * 
 * Written to disk every 5 minutes and on graceful termination (Ctrl+C).
 */
typedef struct {
    uint64_t magic_number;
    uint32_t version;
    int exponent;
    uint64_t target_mr;
    uint64_t last_n;
    uint64_t total_processed;
    uint64_t found_count;
} checkpoint_header_t;

/**
 * @brief Complete search context bundling all state for parallel execution.
 * 
 * Aggregates all necessary components for the search operation:
 * - max_n: Upper bound of search range (2^exponent)
 * - target_mr: The pseudocycle mr class being searched for
 * - dict_entry: Pointer to the pseudocycle definition (ni, nj values)
 * - start_time: Search start timestamp for elapsed time calculation
 * - exponent: Power of 2 defining the search range
 * - writer: Output file management system
 * - progress: Progress tracking and reporting system
 * 
 * Passed to all worker functions to provide unified access to search state.
 */
typedef struct {
    uint64_t max_n;
    uint64_t target_mr;
    const dictionary_entry_t* dict_entry;
    double start_time;
    int exponent;
    char padding1[4];
    streaming_writer_t* writer;
    progress_tracker_t* progress;
} search_context_t;

/**
 * @brief Hash table node for O(1) pseudocycle boundary lookup.
 * 
 * Implements chained hash table with support for multiple pseudocycles sharing values:
 * - value: The ni or nj value being hashed
 * - entries: Array of pointers to dictionary entries containing this value
 * - entry_count: Number of pseudocycles that use this value as ni or nj
 * - next: Next node in the collision chain
 * 
 * Enables fast detection of which pseudocycles are encountered during sequence traversal.
 * Multiple entries exist because different mr classes can share the same ni/nj values.
 */
typedef struct hash_node {
    uint64_t value;
    const dictionary_entry_t** entries;
    int entry_count;
    struct hash_node* next;
} hash_node_t;

// =============================
// PSEUDOCYCLE DICTIONARY
// =============================

/**
 * @brief Complete dictionary of all 42 known Collatz pseudocycles.
 * 
 * Each entry defines a pseudocycle with: target_mr (identifier), ni (first boundary),
 * nj (second boundary), distance (separation between ni and nj). Entry at index 0
 * represents the trivial cycle (mr=0, ni=1, nj=1).
 */
static const dictionary_entry_t PSEUDOCYCLE_DICTIONARY[] = {
    {0, 1, 1, 1},
    {1, 3, 4, 1},
    {2, 6, 5, 1},
    {3, 7, 8, 1},
    {6, 14, 13, 1},
    {7, 15, 16, 1},
    {8, 18, 17, 1},
    {9, 19, 20, 1},
    {12, 25, 26, 1},
    {16, 33, 34, 1},
    {19, 39, 40, 1},
    {25, 51, 52, 1},
    {45, 91, 92, 1},
    {53, 108, 107, 1},
    {60, 121, 122, 75},
    {79, 159, 160, 1},
    {91, 183, 184, 1},
    {121, 243, 244, 1},
    {125, 252, 251, 1},
    {141, 284, 283, 1},
    {166, 333, 334, 1},
    {188, 378, 377, 1},
    {205, 411, 412, 1},
    {243, 487, 488, 1},
    {250, 501, 502, 1},
    {324, 649, 650, 1},
    {333, 667, 668, 1},
    {432, 865, 866, 1},
    {444, 889, 890, 1},
    {487, 975, 976, 1},
    {576, 1153, 1154, 1},
    {592, 1185, 1186, 1},
    {649, 1299, 1300, 1},
    {667, 1335, 1336, 1},
    {683, 1368, 1367, 1},
    {865, 1731, 1732, 1},
    {889, 1779, 1780, 1},
    {1153, 2307, 2308, 1},
    {1214, 2430, 2429, 1},
    {1821, 3643, 3644, 1},
    {2428, 4857, 4858, 1},
    {3643, 7287, 7288, 1}
};

#define DICTIONARY_SIZE (sizeof(PSEUDOCYCLE_DICTIONARY) / sizeof(PSEUDOCYCLE_DICTIONARY[0]))

// =============================
// GLOBAL HASH TABLE STATE
// =============================

static hash_node_t* hash_table[HASH_SIZE];
static bool hash_table_initialized = false;

// =============================
// COMPILE-TIME SAFETY CHECKS
// =============================

_Static_assert(sizeof(collatz_analysis_t) <= 40, "collatz_analysis_t should be 40 bytes or less");

// =============================
// SIGNAL HANDLER
// =============================

static void checkpoint_signal_handler(int sig) {
    if (sig == SIGINT) {
        checkpoint_signal_received = 1;
    }
}

/* 
 ***********************************************************************************
 * 2. MEMORY MANAGEMENT UTILITIES
 ***********************************************************************************
 */

 /**
 * @brief Safely allocates memory with automatic error handling and immediate program termination on failure.
 * 
 * This function provides a wrapper around the standard malloc() call with comprehensive
 * error checking and standardized failure handling. It ensures consistent behavior across
 * the application when memory allocation fails, eliminating the need for repetitive null
 * pointer checks at every allocation site while providing descriptive error messages
 * for debugging purposes.
 * 
 * The function implements a fail-fast strategy: if memory allocation fails, it immediately
 * prints an informative error message to stderr and terminates the program with exit code 1.
 * This approach is suitable for applications where memory allocation failure represents
 * an unrecoverable error condition that should halt execution rather than attempting
 * graceful degradation or recovery.
 * 
 * Error Handling Strategy:
 * - Immediate detection of malloc() failure through null pointer check
 * - Descriptive error message including context information for debugging
 * - Graceful program termination with standard error exit code
 * - No possibility of returning null pointers to calling code
 * 
 * This centralized allocation strategy simplifies error handling throughout the codebase
 * and ensures that memory allocation failures are always detected and reported consistently,
 * preventing potential segmentation faults from null pointer dereferences.
 * 
 * @param size The number of bytes to allocate. Must be greater than 0 for meaningful
 *             allocation. The function passes this value directly to malloc() without
 *             modification. Typical values range from sizeof(small_struct) to
 *             millions of bytes for large arrays.
 * 
 * @param context A descriptive string identifying the purpose of the allocation.
 *                This string is included in error messages to aid debugging and
 *                should describe what the memory is intended for. The string should
 *                be a literal or stable pointer as it may be used after the function
 *                returns (in error cases).
 *                Examples: "hash node", "sorted buffer", "streaming writer"
 * 
 * @return A valid pointer to the allocated memory block. This function never returns
 *         NULL because it terminates the program if allocation fails. The returned
 *         memory is uninitialized and may contain arbitrary data.
 * 
 * @note This function calls exit(1) on allocation failure, making it unsuitable
 *       for applications that need to recover from memory allocation failures or
 *       perform cleanup operations before termination.
 * 
 * @note The returned memory is uninitialized. Use memset() or similar functions
 *       if zero-initialization is required for the allocated block.
 * 
 * @note The context parameter should be a string literal or stable string to
 *       ensure it remains valid during error reporting. Passing a temporary
 *       string may result in undefined behavior during error output.
 * 
 * @warning This function terminates the program on failure, so it should only be
 *          used in contexts where immediate termination is acceptable. For library
 *          code or recoverable scenarios, use standard malloc() with manual checking.
 * 
 * @warning Memory allocated by this function must be freed with standard free()
 *          when no longer needed to prevent memory leaks.
 * 
 * @complexity O(1) - constant time wrapper around malloc() with simple error checking.
 *             The actual allocation complexity depends on the system malloc() implementation.
 * 
 * @see malloc(3) for underlying allocation mechanism and performance characteristics
 * @see exit(3) for program termination behavior and exit code semantics
 * @see free(3) for deallocating memory allocated by this function
 * 
 * @example
 * ```c
 * // Allocate space for a hash node structure
 * hash_node_t* node = allocate_memory_safe(sizeof(hash_node_t), "hash node");
 * 
 * // Allocate space for an array of 100,000 uint64_t values
 * uint64_t* buffer = allocate_memory_safe(100000 * sizeof(uint64_t), "sorted buffer");
 * 
 * // Allocate space for a streaming writer structure
 * streaming_writer_t* writer = allocate_memory_safe(sizeof(streaming_writer_t), 
 *                                                     "streaming writer");
 * 
 * // Error case (simulated - when system is out of memory):
 * // If malloc() fails, program prints to stderr:
 * // "[*] ERROR: Memory allocation failed for sorted buffer"
 * // and exits with code 1, preventing further execution
 * 
 * // Remember to free allocated memory when done:
 * free(node);
 * free(buffer);
 * free(writer);
 * ```
 */
static void* allocate_memory_safe(size_t size, const char* context) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "\n[*] ERROR: Memory allocation failed for %s\n", context);
        exit(1);
    }
    return ptr;
}

/*
 ***********************************************************************************
 * 3. CORE ALGORITHM FUNCTIONS
 ***********************************************************************************
 */

 /**
 * @brief Applies one iteration of the Collatz function with overflow protection.
 * 
 * Performs a single step of the Collatz sequence transformation with built-in
 * overflow detection to prevent undefined behavior from arithmetic overflow.
 * The function modifies the input value in-place according to the Collatz rules:
 * - If n is odd: n = 3n + 1 (with overflow check)
 * - If n is even: n = n / 2 (using bit shift for performance)
 * 
 * Overflow Detection Strategy:
 * - For odd numbers, checks if n > UINT64_MAX/3 before multiplication
 * - Returns false if overflow would occur, allowing caller to handle gracefully
 * - Even numbers cannot overflow (division always reduces value)
 * 
 * The in-place modification design minimizes memory operations and allows for
 * efficient sequence iteration without temporary variables or return value overhead.
 * 
 * @param n Pointer to the number to transform. The value is modified in-place.
 *          Must not be NULL. The pointed value should be > 0 for meaningful
 *          Collatz sequence behavior.
 * 
 * @return true if the transformation was successful, false if overflow would occur.
 *         A false return indicates the sequence cannot continue safely and should
 *         be terminated to prevent undefined behavior.
 * 
 * @note This function uses bitwise operations for parity checking (n & 1) and
 *       division by 2 (n >> 1) for optimal performance in tight iteration loops.
 * 
 * @note The overflow check is conservative: it prevents any multiplication that
 *       would exceed UINT64_MAX, ensuring all arithmetic stays within defined bounds.
 * 
 * @note This function does not validate that the input pointer is non-NULL.
 *       Passing NULL results in undefined behavior (segmentation fault).
 * 
 * @warning The caller must check the return value. Ignoring a false return and
 *          continuing iteration may lead to incorrect results or infinite loops.
 * 
 * @complexity O(1) - constant time for both odd and even cases
 * 
 * @see verify_target_mr_is_first_pseudocycle_fast() for primary usage context
 * @see analyze_collatz_sequence() for sequence analysis using this function
 * 
 * @example
 * ```c
 * uint64_t n = 27;
 * 
 * // First iteration: 27 is odd → 3*27+1 = 82
 * if (apply_collatz_function(&n)) {
 *     printf("%lu\n", n);  // Output: 82
 * }
 * 
 * // Second iteration: 82 is even → 82/2 = 41
 * if (apply_collatz_function(&n)) {
 *     printf("%lu\n", n);  // Output: 41
 * }
 * 
 * // Overflow case:
 * uint64_t large = UINT64_MAX / 2;  // Very large odd number
 * if (!apply_collatz_function(&large)) {
 *     printf("Overflow detected, sequence terminated\n");
 * }
 * 
 * // Typical usage in a loop:
 * uint64_t current = 27;
 * while (current != 1) {
 *     if (!apply_collatz_function(&current)) {
 *         // Handle overflow - sequence diverged
 *         break;
 *     }
 * }
 * ```
 */
static inline bool apply_collatz_function(uint64_t* n) {
    if (*n & 1) {
        if (*n > UINT64_MAX / 3) {
            return false;
        }
        *n = 3 * (*n) + 1;
    } else {
        *n = *n >> 1;
    }
    return true;
}

/* 
 ***********************************************************************************
 * 4. HASH TABLE SYSTEM
 ***********************************************************************************
 */

 /**
 * @brief Computes hash value for uint64_t keys using multiplicative hashing.
 * 
 * Implements a fast multiplicative hash function optimized for 64-bit integers.
 * Uses multiplicative constant 2654435761 ≈ 2^32 × φ (golden ratio φ = (√5-1)/2)
 * which provides excellent distribution properties for hash tables. The function
 * multiplies the input by this constant, extracts the upper 32 bits via right
 * shift, and applies a bitmask to constrain the result to the hash table size.
 * 
 * Hash Distribution Properties:
 * - Uniform distribution across the hash space for sequential and random inputs
 * - Minimal clustering due to golden ratio multiplication
 * - Fast computation using only multiplication, shift, and bitwise AND
 * - Deterministic: same input always produces same hash value
 * 
 * The hash table size (HASH_SIZE = 8192) is a power of 2, allowing the mask
 * operation (& HASH_MASK) to efficiently perform modulo without division.
 * 
 * @param value The uint64_t value to hash (typically ni or nj from pseudocycle dictionary)
 * 
 * @return Hash value in range [0, HASH_SIZE-1] suitable for array indexing
 * 
 * @note This function is declared inline for performance in tight lookup loops
 * 
 * @note The golden ratio constant provides better distribution than simple modulo
 *       or XOR-based hashing for the pseudocycle value patterns
 * 
 * @complexity O(1) - constant time multiplication and bitwise operations
 * 
 * @see add_to_hash_table() for hash table insertion using this function
 * @see verify_target_mr_is_first_pseudocycle_fast() for lookup usage
 */
static inline uint32_t hash_function(uint64_t value) {
    return (uint32_t)((value * 2654435761ULL) >> 32) & HASH_MASK;
}

/**
 * @brief Adds a pseudocycle dictionary entry to the hash table for fast lookup.
 * 
 * Inserts a mapping from a boundary value (ni or nj) to its corresponding
 * dictionary entry, enabling O(1) lookup during sequence verification. Handles
 * hash collisions using chaining and supports multiple dictionary entries
 * mapping to the same value (when different pseudocycles share boundary points).
 * 
 * Collision Handling Strategy:
 * - Searches collision chain for existing node with matching value
 * - If found, expands the entries array to include new dictionary reference
 * - If not found, creates new node and prepends to chain (O(1) insertion)
 * 
 * This design allows multiple pseudocycles to share the same ni or nj values,
 * which is essential because different mr classes can have overlapping boundaries.
 * 
 * @param value The boundary value (ni or nj) to use as hash key
 * @param entry Pointer to the dictionary entry to associate with this value
 * 
 * @note Calls exit(1) if realloc fails during entries array expansion
 * 
 * @note This function is NOT thread-safe; must be called during single-threaded
 *       initialization phase before parallel search begins
 * 
 * @complexity O(k) where k is collision chain length (typically very small)
 * 
 * @see initialize_hash_table() for bulk insertion of all dictionary entries
 * @see hash_function() for hash value computation
 */
static void add_to_hash_table(uint64_t value, const dictionary_entry_t* entry) {
    uint32_t hash = hash_function(value);
    hash_node_t* node = hash_table[hash];
    
    while (node) {
        if (node->value == value) {
            node->entry_count++;
            node->entries = realloc(node->entries, 
                                   node->entry_count * sizeof(dictionary_entry_t*));
            if (!node->entries) {
                fprintf(stderr, "\n[*] ERROR: Memory reallocation failed for hash node entries\n");
                exit(1);
            }
            node->entries[node->entry_count - 1] = entry;
            return;
        }
        node = node->next;
    }
    
    node = allocate_memory_safe(sizeof(hash_node_t), "hash node");
    node->value = value;
    node->entry_count = 1;
    node->entries = allocate_memory_safe(sizeof(dictionary_entry_t*), "entry array");
    node->entries[0] = entry;
    node->next = hash_table[hash];
    hash_table[hash] = node;
}

/**
 * @brief Initializes the global hash table with all pseudocycle dictionary entries.
 * 
 * Populates the hash table by iterating through PSEUDOCYCLE_DICTIONARY and
 * inserting both ni and nj values from each entry (when they differ). This
 * one-time initialization enables O(1) pseudocycle boundary lookups during
 * parallel search execution.
 * 
 * Initialization Process:
 * - Checks if already initialized to prevent redundant work
 * - Clears hash table array to NULL (ensures clean state)
 * - Inserts all ni values from dictionary
 * - Inserts all nj values (skipping when nj == ni for single-value pseudocycles)
 * - Sets initialized flag to prevent re-initialization
 * 
 * Thread Safety:
 * - Uses initialized flag to support idempotent calls
 * - Must complete before parallel region begins
 * - Not thread-safe during initialization itself
 * 
 * @note Called automatically by execute_search_with_guided_scheduling if not
 *       already initialized, ensuring hash table is ready before search begins
 * 
 * @note After initialization, the hash table is read-only during parallel search,
 *       making lookups thread-safe without locking
 * 
 * @complexity O(n) where n is DICTIONARY_SIZE (42 entries, effectively O(1))
 * 
 * @see cleanup_hash_table() for deallocation of hash table resources
 * @see add_to_hash_table() for individual entry insertion
 */
static void initialize_hash_table(void) {
    if (hash_table_initialized) {
        return;
    }
    memset(hash_table, 0, sizeof(hash_table));
    for (size_t i = 0; i < DICTIONARY_SIZE; i++) {
        const dictionary_entry_t* entry = &PSEUDOCYCLE_DICTIONARY[i];
        add_to_hash_table(entry->ni, entry);
        if (entry->nj != entry->ni) {
            add_to_hash_table(entry->nj, entry);
        }
    }
    hash_table_initialized = true;
}

/**
 * @brief Deallocates all hash table memory and resets to uninitialized state.
 * 
 * Frees all dynamically allocated memory used by the hash table, including
 * collision chain nodes and their entry arrays. Resets the global hash table
 * to a clean state, allowing re-initialization if needed. This function ensures
 * proper cleanup and prevents memory leaks at program termination.
 * 
 * Cleanup Process:
 * - Checks initialized flag (safe to call even if not initialized)
 * - Iterates through all hash table buckets
 * - For each bucket, traverses collision chain and frees nodes
 * - Frees dynamic entries array in each node
 * - Resets all bucket pointers to NULL
 * - Clears initialized flag
 * 
 * @note Idempotent: safe to call multiple times. After cleanup, hash table can be
 *       reinitialized with initialize_hash_table() for subsequent searches.
 * 
 * @note Must be called after parallel search completes, not during execution
 * 
 * @note After cleanup, initialize_hash_table() can be called again to rebuild
 *       the table for subsequent searches
 * 
 * @complexity O(m) where m is total number of nodes across all buckets
 *             (typically small due to good hash distribution)
 * 
 * @see initialize_hash_table() for hash table construction
 * @see cleanup_search_context() for cleanup orchestration
 */
static void cleanup_hash_table(void) {
    if (!hash_table_initialized) {
        return;
    }
    for (int i = 0; i < HASH_SIZE; i++) {
        hash_node_t* node = hash_table[i];
        while (node) {
            hash_node_t* next = node->next;
            free(node->entries);
            free(node);
            node = next;
        }
        hash_table[i] = NULL;
    }
    hash_table_initialized = false;
}

/* 
 ***********************************************************************************
 * 5. DICTIONARY LOOKUP FUNCTIONS
 ***********************************************************************************
*/
/**
 * @brief Searches the pseudocycle dictionary for an entry matching the target mr value.
 * 
 * Performs a linear search through the PSEUDOCYCLE_DICTIONARY array to find the
 * dictionary entry corresponding to the specified mr (mathematical residue) value.
 * This function is used during initialization and validation to retrieve the
 * complete pseudocycle definition (ni, nj, distance) for a given mr class.
 * 
 * Search Strategy:
 * - Linear iteration through all 42 dictionary entries
 * - Returns pointer to first matching entry (mr values are unique)
 * - Returns NULL if target_mr not found in dictionary
 * 
 * Performance Considerations:
 * - O(n) linear search is acceptable because DICTIONARY_SIZE is small (42)
 * - Called infrequently (initialization and argument validation only)
 * - Not used in hot paths where hash table lookups are preferred
 * 
 * The returned pointer references static const data in PSEUDOCYCLE_DICTIONARY,
 * so the data remains valid for the program's lifetime and should not be modified.
 * 
 * @param target_mr The mr value to search for (0 to 3643 for known pseudocycles)
 * 
 * @return Pointer to the matching dictionary_entry_t if found, NULL otherwise.
 *         The returned pointer is const and references static data.
 * 
 * @note The returned pointer should never be freed (points to static const array)
 * 
 * @note This function does not validate that target_mr is in a reasonable range;
 *       it simply searches for exact matches
 * 
 * @complexity O(n) where n = DICTIONARY_SIZE = 42 (effectively O(1) constant time)
 * 
 * @see PSEUDOCYCLE_DICTIONARY for the complete array of known pseudocycles
 * @see validate_and_parse_arguments() for usage in command-line validation
 * 
 * @example
 * ```c
 * // Look up the pseudocycle for mr=1821
 * const dictionary_entry_t* entry = lookup_pseudocycle(1821);
 * if (entry) {
 *     printf("Found: ni=%lu, nj=%lu\n", entry->ni, entry->nj);
 *     // Output: "Found: ni=3643, nj=3644"
 * }
 * 
 * // Look up non-existent mr value
 * const dictionary_entry_t* invalid = lookup_pseudocycle(9999);
 * if (!invalid) {
 *     printf("mr=9999 not found in dictionary\n");
 * }
 * 
 * // Typical usage in main():
 * const dictionary_entry_t* dict_entry = lookup_pseudocycle(target_mr);
 * if (!dict_entry) {
 *     printf("ERROR: Invalid mr value\n");
 *     return 1;
 * }
 * ```
 */
static const dictionary_entry_t* lookup_pseudocycle(uint64_t target_mr) {
    for (size_t i = 0; i < DICTIONARY_SIZE; i++) {
        if (PSEUDOCYCLE_DICTIONARY[i].target_mr == target_mr) {
            return &PSEUDOCYCLE_DICTIONARY[i];
        }
    }
    return NULL;
}

/* 
 ***********************************************************************************
 * 6. SEQUENCE VERIFICATION FUNCTIONS
 ***********************************************************************************
 */

 /**
 * @brief Fast verification that target_mr is the first complete pseudocycle encountered in sequence.
 * 
 * This is the core filtering function that determines whether a number n belongs to
 * the target mr class. It verifies two critical conditions:
 * 1. The sequence encounters the target_mr pseudocycle (both ni and nj in order)
 * 2. No other pseudocycle is completed before the target (first pseudocycle criterion)
 * 
 * The function uses bitfield tracking to efficiently detect when pseudocycles are
 * completed: each bit in found_ni_bits represents whether that pseudocycle's ni
 * has been seen. When the corresponding nj is encountered, the pseudocycle is
 * complete. The function immediately returns false if any non-target pseudocycle
 * completes first.
 * 
 * Special Case Handling:
 * - mr=0 (trivial cycle): Verifies sequence reaches 1 without encountering other
 *   complete pseudocycles first. Checks all positions including initial value.
 * - mr>0: Checks initial value for ni matches, then iterates sequence looking for
 *   ni→nj completion patterns
 * 
 * Hash Table Integration:
 * - Uses O(1) hash lookups to check if current value is a pseudocycle boundary
 * - Handles multiple pseudocycles sharing the same ni/nj values
 * - Efficiently tracks up to 64 pseudocycles simultaneously using bitfields
 * 
 * Divergence Protection:
 * - Limits iteration to MAX_SEQUENCE_LENGTH to prevent infinite loops
 * - Detects exponential growth (current > n * DIVERGENCE_MULTIPLIER)
 * - Returns false on overflow or suspected divergence
 * 
 * @param n The number whose Collatz sequence is being verified
 * @param target_mr The mr value that must be the first complete pseudocycle
 * 
 * @return true if target_mr is the first complete pseudocycle encountered,
 *         false if another pseudocycle completes first, overflow occurs, or
 *         sequence diverges without finding target
 * 
 * @note This function is performance-critical (called for every candidate n)
 *       and is optimized for speed with inline hash lookups and bitfield operations
 * 
 * @note Requires hash_table_initialized to be true before calling
 * 
 * @note For mr=0, the function verifies that 1 is reached before any other
 *       complete pseudocycle (other than the trivial cycle itself)
 * 
 * @warning Does not perform full sequence analysis (no taxonomy classification)
 *          Use analyze_collatz_sequence() for complete analysis of passing sequences
 * 
 * @complexity O(k) where k is sequence length to first pseudocycle or MAX_SEQUENCE_LENGTH
 * 
 * @see analyze_collatz_sequence() for detailed sequence analysis after verification
 * @see process_single_number() for typical usage in parallel search
 * @see initialize_hash_table() for required hash table setup
 * 
 * @example
 * ```c
 * // Verify n=27 belongs to mr=9 class
 * if (verify_target_mr_is_first_pseudocycle_fast(27, 9)) {
 *     printf("27 belongs to mr=9 class\n");
 *     // Sequence: 27 → ... → 19 (ni) → 20 (nj) completes first
 * }
 * 
 * // Verify mr=0 (trivial cycle)
 * if (verify_target_mr_is_first_pseudocycle_fast(1, 0)) {
 *     printf("1 reaches trivial cycle first\n");
 * }
 * 
 * // Typical usage in search loop:
 * for (uint64_t n = 1; n < max_n; n++) {
 *     if (verify_target_mr_is_first_pseudocycle_fast(n, target_mr)) {
 *         // n belongs to target_mr class, proceed with full analysis
 *         collatz_analysis_t analysis = analyze_collatz_sequence(n, dict_entry);
 *     }
 * }
 * ```
 */
static bool verify_target_mr_is_first_pseudocycle_fast(uint64_t n, uint64_t target_mr) {
    uint64_t current = n;
    uint64_t found_ni_bits = 0;
    
    if (target_mr == 0) {
        for (int pos = 0; pos <= MAX_SEQUENCE_LENGTH && current != 1; pos++) {
            if (pos > 0 && !apply_collatz_function(&current)) {
                return false;
            }
            
            // Check initial value for ni matches
            uint32_t hash = hash_function(current);
            hash_node_t* node = hash_table[hash];
            
            while (node) {
                if (node->value == current) {
                    for (int i = 0; i < node->entry_count; i++) {
                        const dictionary_entry_t* entry = node->entries[i];
                        if (entry->target_mr == 0) continue; // Skip trivial cycle self-detection
                        
                        int entry_idx = entry - PSEUDOCYCLE_DICTIONARY;
                        
                        if (current == entry->ni) {
                            // Mark this pseudocycle's ni as seen
                            found_ni_bits |= (1ULL << entry_idx);
                        }
                        else if (current == entry->nj && (found_ni_bits & (1ULL << entry_idx))) {
                            return false;  // Another pseudocycle completed first - n doesn't belong to mr=0
                        }
                    }
                    break;
                }
                node = node->next;
            }
            
            if (pos > MAX_SAFE_ITERATIONS && current > n * DIVERGENCE_MULTIPLIER) {
                return false;
            }
        }
        return (current == 1);
    }
    
    uint32_t hash = hash_function(current);
    hash_node_t* node = hash_table[hash];
    
    while (node) {
        if (node->value == current) {
            for (int i = 0; i < node->entry_count; i++) {
                const dictionary_entry_t* entry = node->entries[i];
                int entry_idx = entry - PSEUDOCYCLE_DICTIONARY;
                
                if (current == entry->ni) {
                    found_ni_bits |= (1ULL << entry_idx);
                }
            }
            break;
        }
        node = node->next;
    }
    
    for (int pos = 1; pos <= MAX_SEQUENCE_LENGTH && current != 1; pos++) {
        if (!apply_collatz_function(&current)) {
            return false;
        }
        
        if (pos > MAX_SAFE_ITERATIONS && current > n * DIVERGENCE_MULTIPLIER) {
            return false;
        }
        
        hash = hash_function(current);
        node = hash_table[hash];
        
        while (node) {
            if (node->value == current) {
                for (int i = 0; i < node->entry_count; i++) {
                    const dictionary_entry_t* entry = node->entries[i];
                    int entry_idx = entry - PSEUDOCYCLE_DICTIONARY;
                    
                    if (current == entry->ni) {
                        found_ni_bits |= (1ULL << entry_idx);
                    }
                    else if (current == entry->nj && (found_ni_bits & (1ULL << entry_idx))) {
                        // First complete pseudocycle found - check if it's the target
                        return (entry->target_mr == target_mr);
                    }
                }
                break;
            }
            node = node->next;
        }
    }
    
    return false;
}

/*
 ***********************************************************************************
 * 7. SEQUENCE ANALYSIS FUNCTIONS
 ***********************************************************************************
 */

 /**
 * @brief Performs complete Collatz sequence analysis for taxonomy classification.
 * 
 * This function conducts a comprehensive analysis of a number's Collatz trajectory,
 * tracking the global maximum value, its position, and the positions where the
 * target pseudocycle boundaries (ni and nj) are encountered. The analysis results
 * are used to classify the number into one of three taxonomies (A, B, or C) based
 * on the relative position of the maximum value within the pseudocycle detection.
 * 
 * Taxonomy Classification Rules:
 * - Taxonomy A: Global maximum occurs before ni is encountered
 * - Taxonomy B: Global maximum occurs between ni and nj (inclusive of nj position)
 * - Taxonomy C: Global maximum occurs after nj is encountered
 * 
 * Special Case: mr=0 (Trivial Cycle)
 * - For n=1: Immediately returns with max=1, taxonomy=A (already in pseudocycle)
 * - For n>1: Iterates until reaching 1, always classifies as taxonomy A because
 *   the maximum necessarily occurs before reaching the cycle point (1)
 * - This special handling prevents the sequence from continuing past 1 into the
 *   cycle (1→4→2→1) which would incorrectly suggest taxonomy B
 * 
 * Standard Case: mr>0
 * - Checks if n itself is ni (handles sequences starting in pseudocycle)
 * - Iterates sequence tracking max_value and max_position continuously
 * - Detects ni and nj in order (nj only valid after ni found)
 * - Continues to sequence end or limit to find true global maximum
 * - Classifies taxonomy based on final max_position relative to ni/nj positions
 * 
 * Single-Value Pseudocycles:
 * - When ni==nj, finding ni immediately completes pseudocycle detection
 * - Still tracks maximum position for accurate taxonomy classification
 * 
 * Divergence Protection:
 * - Limits iteration to MAX_SEQUENCE_LENGTH
 * - Monitors for exponential growth (current > n * DIVERGENCE_MULTIPLIER)
 * - Terminates early on overflow or suspected divergence
 * 
 * @param n The number whose Collatz sequence is being analyzed. Must be > 0.
 * @param dict_entry Pointer to the pseudocycle dictionary entry containing ni, nj,
 *                   and target_mr values. Must not be NULL.
 * 
 * @return collatz_analysis_t structure containing:
 *         - max_value: Highest value reached in sequence
 *         - max_position: Position where maximum occurred (1-indexed)
 *         - ni_position: Position where ni was found (-1 if not found)
 *         - nj_position: Position where nj was found (-1 if not found)
 *         - taxonomy: 'A', 'B', or 'C' classification
 *         - found_pseudocycle: true if both ni and nj were detected in order
 * 
 * @note This function should only be called after verify_target_mr_is_first_pseudocycle_fast()
 *       returns true, as it assumes the target pseudocycle will be found
 * 
 * @note Position values are 1-indexed (initial value n is at position 1)
 * 
 * @note For mr=0, taxonomy is always 'A' by mathematical definition
 * 
 * @note If pseudocycle is not found (shouldn't happen after verification), returns
 *       with found_pseudocycle=false and taxonomy='A' as default
 * 
 * @complexity O(k) where k is sequence length to convergence or MAX_SEQUENCE_LENGTH
 * 
 * @see verify_target_mr_is_first_pseudocycle_fast() for pre-filtering before analysis
 * @see write_result_immediately() for using taxonomy classification in output
 * @see process_single_number() for typical usage workflow
 * 
 * @example
 * ```c
 * @example
 * ```c
 * // Analyze n=27 for mr=60 (ni=121, nj=122)
 * const dictionary_entry_t* entry = lookup_pseudocycle(60);
 * collatz_analysis_t result = analyze_collatz_sequence(27, entry);
 * 
 * // Result:
 * // max_value = 9232 (peak of sequence)
 * // ni_position = 17 (found 121 at step 17)
 * // nj_position = 91 (found 122 at step 91)
 * // max_position = 78 (maximum occurs between ni and nj)
 * // taxonomy = 'B' (max between ni and nj inclusive)
 * // found_pseudocycle = true
 * 
 * printf("Taxonomy: %c\n", result.taxonomy);  // Output: B
 * `
 * // Special case: mr=0, n=1
 * const dictionary_entry_t* trivial = lookup_pseudocycle(0);
 * collatz_analysis_t trivial_result = analyze_collatz_sequence(1, trivial);
 * // Returns immediately: max=1, taxonomy='A', already in cycle
 * 
 * // Special case: mr=0, n=5
 * // Sequence: 5 → 16 → 8 → 4 → 2 → 1
 * // max_value=16 at position=2, reaches 1 at position=6
 * // taxonomy='A' (maximum occurs before detecting ni=1, sequence stops there)
 * ```
 */
static collatz_analysis_t analyze_collatz_sequence(uint64_t n, const dictionary_entry_t* dict_entry) {
    collatz_analysis_t result = {
        .max_value = n,
        .ni_position = -1,
        .nj_position = -1,
        .max_position = 1,
        .taxonomy = 'A',
        .found_pseudocycle = false
    };
   
    if (dict_entry->target_mr == 0) {
        if (n == 1) {
            result.max_value = 1;
            result.max_position = 1;
            result.ni_position = 1;
            result.nj_position = 1;
            result.found_pseudocycle = true;
            result.taxonomy = 'A';  // Degenerate case: n=1 is already at ni=nj=max (all at position 1)
            return result;
        }
        
        uint64_t current = n;
        int position = 1;
        
        for (int step = 0; step < MAX_SEQUENCE_LENGTH - 1 && current != 1; step++) {
            if (!apply_collatz_function(&current)) {
                break;
            }
            position++;
            
            if (current > result.max_value) {
                result.max_value = current;
                result.max_position = position;
            }
            
            if (step > MAX_SAFE_ITERATIONS && current > n * DIVERGENCE_MULTIPLIER) {
                break;
            }
        }
        
        if (current == 1) {
            result.ni_position = position;
            result.nj_position = position;
            result.found_pseudocycle = true;
            result.taxonomy = 'A';  // For mr=0, always A (stops at ni=1 without exploring cycle 1→4→2→1)
        }
        
        return result;
    }
    
    uint64_t current = n;
    int position = 1;
    uint64_t target_ni = dict_entry->ni;
    uint64_t target_nj = dict_entry->nj;
    
    bool single_value_pseudocycle = (target_ni == target_nj);
    
    if (current == target_ni) {
        result.ni_position = position;
        if (single_value_pseudocycle) {
            result.nj_position = position;
            result.found_pseudocycle = true;
        }
    }
    
    for (int step = 0; step < MAX_SEQUENCE_LENGTH - 1 && current != 1; step++) {
        if (!apply_collatz_function(&current)) {
            break;
        }
        position++;
        
        if (current > result.max_value) {
            result.max_value = current;
            result.max_position = position;
        }
        
        if (current == target_ni && result.ni_position == -1) {
            result.ni_position = position;
        }
        
        if (current == target_nj && result.nj_position == -1 && result.ni_position != -1) {
            result.nj_position = position;
            result.found_pseudocycle = true;
        }
        
        if (step > MAX_SAFE_ITERATIONS && current > n * DIVERGENCE_MULTIPLIER) {
            break;
        }
    }
    
    if (result.ni_position != -1 && result.nj_position != -1) {
        result.found_pseudocycle = true;
        if (result.max_position < result.ni_position) {
            result.taxonomy = 'A';  // Maximum occurred before ni
        } else if (result.max_position <= result.nj_position) {
            result.taxonomy = 'B';  // Maximum occurred between ni and nj (inclusive)
        } else {
            result.taxonomy = 'C';  // Maximum occurred after nj
        }
    } else {
        result.found_pseudocycle = false;
        result.taxonomy = 'A';
    }
    
    return result;
}

/*
 ***********************************************************************************
 * 8. CHECKPOINT SYSTEM
 ***********************************************************************************
 */

 /**
 * @brief Saves current search state to checkpoint file for resume capability.
 * 
 * Creates a binary checkpoint file containing the minimal state needed to resume
 * an interrupted search from the last processed position. The checkpoint includes
 * search parameters (exponent, target_mr), progress counters, and the last
 * successfully processed n value. This enables graceful interruption handling
 * (Ctrl+C) and automatic recovery from crashes or system failures.
 * 
 * Checkpoint Strategy:
 * - Atomic write with backup: renames existing checkpoint to .bak before writing new one
 * - Binary format for compact size and fast I/O
 * - Magic number validation (0x4E425952 = "NBYR") for corruption detection
 * - Version number for compatibility checking across code updates
 * - Thread-safe: acquires progress lock before accessing shared counters
 * 
 * Safety Features:
 * - Backup file preserved until new checkpoint successfully written
 * - Safe to call multiple times (overwrites previous checkpoint)
 * - Gracefully handles write failures (silently returns without crashing)
 * - Lock-protected access to progress counters prevents data races
 * 
 * Checkpoint Contents:
 * - Magic number and version for validation
 * - Search parameters (exponent, target_mr) for matching on resume
 * - Last processed n value for continuation point
 * - Cumulative counters (total_processed, found_count) for progress tracking
 * 
 * @param ctx Pointer to search context containing progress tracker and search parameters
 * @param last_n The last n value that was completely processed before checkpoint
 * 
 * @note This function is called periodically (every 5 minutes) during search execution
 *       and on graceful termination (Ctrl+C or completion)
 * 
 * @note Silently fails if file creation fails (does not crash or report error)
 *       This is intentional to allow search to continue even if checkpointing fails
 * 
 * @note The checkpoint file is deleted automatically upon successful search completion
 * 
 * @note Thread-safe: uses progress->lock to ensure atomic read of counters
 * 
 * @complexity O(1) - constant time write of fixed-size header structure
 * 
 * @see load_checkpoint() for reading and validating checkpoint on program start
 * @see checkpoint_signal_handler() for triggering checkpoint on Ctrl+C
 * @see execute_search_with_guided_scheduling() for periodic checkpoint calls
 * 
 * @example
 * ```c
 * // Save checkpoint at n=1000000 during search
 * save_checkpoint(&ctx, 1000000);
 * // Creates checkpoint.bin with state, renames old to checkpoint.bak
 * 
 * // Automatic periodic checkpointing in search loop:
 * if (current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL) {
 *     save_checkpoint(ctx, checkpoint_position);
 *     last_checkpoint_time = current_time;
 * }
 * 
 * // On Ctrl+C signal:
 * if (checkpoint_signal_received) {
 *     save_checkpoint(ctx, safe_pos);
 *     printf("Interrupted! Checkpoint saved\n");
 *     exit(0);
 * }
 * ```
 */
static void save_checkpoint(const search_context_t* ctx, uint64_t last_n) {
    omp_set_lock(&ctx->progress->lock);
    
    if (access(CHECKPOINT_FILE, F_OK) == 0) {
        rename(CHECKPOINT_FILE, CHECKPOINT_BACKUP);
    }
    
    FILE* fp = fopen(CHECKPOINT_FILE, "wb");
    if (!fp) {
        omp_unset_lock(&ctx->progress->lock);
        return;
    }
    
    checkpoint_header_t header;
    header.magic_number = CHECKPOINT_MAGIC;
    header.version = CHECKPOINT_VERSION;
    header.exponent = ctx->exponent;
    header.target_mr = ctx->target_mr;
    header.last_n = last_n;
    header.total_processed = ctx->progress->processed;
    header.found_count = ctx->progress->found_count;
    
    fwrite(&header, sizeof(checkpoint_header_t), 1, fp);
    fclose(fp);
    omp_unset_lock(&ctx->progress->lock);
}

/**
 * @brief Loads and validates checkpoint file to resume interrupted search.
 * 
 * Attempts to load a previously saved checkpoint file and restore the search state,
 * allowing seamless continuation from the last saved position. Performs extensive
 * validation to ensure the checkpoint matches the current search parameters and
 * is not corrupted. Falls back to fresh start if checkpoint is invalid or missing.
 * 
 * Validation Checks:
 * - File existence (tries both checkpoint.bin and checkpoint.bak backup)
 * - Magic number match (detects corrupted or invalid files)
 * - Version compatibility (prevents loading incompatible checkpoint formats)
 * - Parameter matching (exponent and target_mr must match current search)
 * - All failures result in warning message and fresh start (safe fallback)
 * 
 * Resume Process:
 * - Reads checkpoint header from binary file
 * - Validates all fields against current search context
 * - Sets start_n to last_n + 1 (resume from next number)
 * - Restores progress counters (processed, found_count)
 * - Displays resume information with completion percentage
 * 
 * Fallback Behavior:
 * - Missing file: silently returns false (normal fresh start)
 * - Corrupted file: prints warning, returns false
 * - Parameter mismatch: prints warning with details, returns false
 * - All failures are non-fatal and allow clean fresh start
 * 
 * @param ctx Pointer to search context (modified with restored progress counters)
 * @param start_n Pointer to starting n value (set to last_n + 1 if checkpoint loaded)
 * 
 * @return true if checkpoint successfully loaded and validated, false if no valid
 *         checkpoint found (caller should start fresh from n=1)
 * 
 * @note Tries checkpoint.bak as fallback if checkpoint.bin is missing or corrupted
 * 
 * @note All validation failures print informative warning messages to help user
 *       understand why checkpoint was rejected
 * 
 * @note Safe to call even if no checkpoint exists (returns false, no side effects)
 * 
 * @note Modifies ctx->progress counters only if checkpoint successfully validates
 * 
 * @complexity O(1) - constant time read and validation of fixed-size header
 * 
 * @see save_checkpoint() for checkpoint creation
 * @see validate_and_parse_arguments() for initial parameter validation
 * @see main() for typical usage at search initialization
 * 
 * @example
 * ```c
 * // Typical usage in main():
 * search_context_t ctx = { ... };
 * uint64_t start_n = 1;
 * 
 * if (load_checkpoint(&ctx, &start_n)) {
 *     // Checkpoint loaded successfully
 *     // start_n now set to resume position
 *     // ctx.progress->processed and found_count restored
 * } else {
 *     // No valid checkpoint, start from beginning
 *     // start_n remains 1
 * }
 * 
 * execute_parallel_search(&ctx, start_n, ...);
 * 
 * // Example output when resuming:
 * // "[*] CHECKPOINT LOADED"
 * // "    - Last processed: n = 5000000"
 * // "    - Resuming from: n = 5000001"
 * // "    - Completed: 14.9011611938% | Remaining: 85.0988388062%"
 * 
 * // Example output when checkpoint invalid:
 * // "[*] WARNING: Checkpoint exponent mismatch (checkpoint=25, current=30). Starting fresh."
 * ```
 */
static bool load_checkpoint(search_context_t* ctx, uint64_t* start_n) {
    FILE* fp = fopen(CHECKPOINT_FILE, "rb");
    if (!fp) {
        fp = fopen(CHECKPOINT_BACKUP, "rb");
        if (!fp) return false;
    }
    
    checkpoint_header_t header;
    if (fread(&header, sizeof(checkpoint_header_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    if (header.magic_number != CHECKPOINT_MAGIC) {
        fclose(fp);
        printf("\n[*] WARNING: Checkpoint file corrupted. Starting fresh.\n");
        return false;
    }
    
    if (header.version != CHECKPOINT_VERSION) {
        fclose(fp);
        printf("\n[*] WARNING: Checkpoint version mismatch (checkpoint=%u, current=%u). Starting fresh.\n",
               header.version, CHECKPOINT_VERSION);
        return false;
    }
    
    if (header.exponent != ctx->exponent) {
        fclose(fp);
        printf("\n[*] WARNING: Checkpoint exponent mismatch (checkpoint=%d, current=%d). Starting fresh.\n",
               header.exponent, ctx->exponent);
        return false;
    }
    
    if (header.target_mr != ctx->target_mr) {
        fclose(fp);
        printf("\n[*] WARNING: Checkpoint target_mr mismatch (checkpoint=%lu, current=%lu). Starting fresh.\n",
               header.target_mr, ctx->target_mr);
        return false;
    }
    
    *start_n = header.last_n + 1;
    fclose(fp);
    
    ctx->progress->processed = header.total_processed;
    ctx->progress->found_count = header.found_count;

    double completed_pct = (header.total_processed * 100.0) / (ctx->max_n - 1);
    double remaining_pct = 100.0 - completed_pct;
    
    printf("\n[*] CHECKPOINT LOADED\n");
    printf("\t - Last processed: n = %lu\n", header.last_n);
    printf("\t - Resuming from: n = %lu\n", *start_n);
    printf("\t - Completed: %.10f%% | Remaining: %.10f%%\n", completed_pct, remaining_pct);
    
    return true;
}

/*
 ***********************************************************************************
 * 9. SORTED BUFFER SYSTEM
 ***********************************************************************************
 */

 /**
 * @brief Initializes a sorted buffer with default capacity and thread-safe lock.
 * 
 * Allocates and configures a new sorted buffer for maintaining an ordered collection
 * of uint64_t values in memory. The buffer supports dynamic growth, binary search
 * insertion, and thread-safe concurrent access from multiple OpenMP threads.
 * 
 * Initialization Process:
 * - Allocates initial capacity (INITIAL_BUFFER_CAPACITY = 100,000 elements)
 * - Sets count to 0 (empty buffer)
 * - Initializes OpenMP lock for thread-safe operations
 * - Allocates backing array for storing sorted values
 * 
 * Memory Management:
 * - Initial allocation size chosen to minimize reallocations for typical workloads
 * - Buffer grows exponentially (doubles) when capacity reached
 * - Memory allocated via allocate_memory_safe (exits on allocation failure)
 * 
 * @param buffer Pointer to sorted_buffer_t structure to initialize. Must not be NULL.
 *               Structure members will be modified in-place.
 * 
 * @note Must call free_sorted_buffer() when done to prevent memory leaks
 * 
 * @note The buffer is empty after initialization (count=0)
 * 
 * @note Lock must be destroyed with omp_destroy_lock when buffer freed
 * 
 * @complexity O(1) - constant time allocation and initialization
 * 
 * @see insert_sorted() for adding values to initialized buffer
 * @see free_sorted_buffer() for cleanup and deallocation
 * @see create_streaming_writer() for typical usage context
 * 
 * @example
 * ```c
 * sorted_buffer_t buffer;
 * init_sorted_buffer(&buffer);
 * 
 * // Buffer now ready for use:
 * // buffer.capacity = 100000
 * // buffer.count = 0
 * // buffer.numbers = allocated array
 * // buffer.lock = initialized
 * 
 * // Later cleanup:
 * free_sorted_buffer(&buffer);
 * ```
 */
static void init_sorted_buffer(sorted_buffer_t* buffer) {
    buffer->capacity = INITIAL_BUFFER_CAPACITY;
    buffer->numbers = allocate_memory_safe(buffer->capacity * sizeof(uint64_t), "sorted buffer");
    buffer->count = 0;
    omp_init_lock(&buffer->lock);
}

/**
 * @brief Finds insertion position for a value in sorted array using binary search.
 * 
 * Performs a binary search to determine the correct index at which to insert a new
 * value while maintaining sorted order (ascending). Returns the leftmost position
 * where the value should be inserted, ensuring stability for duplicate values.
 * 
 * Search Algorithm:
 * - Standard binary search with lower/upper bound tracking
 * - Returns position where value should be inserted (not position of existing value)
 * - For duplicates, returns position maintaining stable sort (insertion after existing)
 * - Handles empty array (returns 0)
 * 
 * The function guarantees that after insertion at the returned position and shifting
 * elements right, the array remains sorted in ascending order.
 * 
 * @param array Pointer to sorted uint64_t array to search. Must be sorted ascending.
 * @param count Number of elements currently in array (may be 0 for empty array)
 * @param value The value to find insertion position for
 * 
 * @return Index where value should be inserted (0 to count inclusive).
 *         Returns 0 if array is empty or value is smallest.
 *         Returns count if value is larger than all existing elements.
 * 
 * @note Array must already be sorted in ascending order for correct results
 * 
 * @note Return value is always in range [0, count], suitable for array insertion
 * 
 * @note This is a pure search function; it does not modify the array
 * 
 * @complexity O(log n) - binary search logarithmic time
 * 
 * @see insert_sorted() for usage in actual insertion operation
 * 
 * @example
 * ```c
 * uint64_t arr[] = {10, 20, 30, 40, 50};
 * size_t count = 5;
 * 
 * // Find position for value smaller than all
 * size_t pos = binary_search_insert_pos(arr, count, 5);
 * // Returns: 0 (insert at beginning)
 * 
 * // Find position for value in middle
 * pos = binary_search_insert_pos(arr, count, 35);
 * // Returns: 3 (insert between 30 and 40)
 * 
 * // Find position for value larger than all
 * pos = binary_search_insert_pos(arr, count, 60);
 * // Returns: 5 (insert at end)
 * 
 * // Find position for duplicate
 * pos = binary_search_insert_pos(arr, count, 30);
 * // Returns: 3 (insert after existing 30)
 * 
 * // Empty array
 * pos = binary_search_insert_pos(arr, 0, 25);
 * // Returns: 0 (only valid position)
 * ```
 */
static size_t binary_search_insert_pos(uint64_t* array, size_t count, uint64_t value) {
    if (count == 0) return 0;
    
    size_t left = 0;
    size_t right = count;
    
    while (left < right) {
        // Overflow-safe midpoint calculation
        size_t mid = left + (right - left) / 2;  
        if (array[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}


/**
 * @brief Thread-safely inserts a value into sorted buffer maintaining ascending order.
 * 
 * Adds a new value to the sorted buffer in the correct position to maintain ascending
 * order. Automatically handles capacity expansion when needed (doubling strategy) and
 * uses binary search for efficient position lookup. All operations are protected by
 * an OpenMP lock for thread-safe concurrent insertions from parallel regions.
 * 
 * Insertion Process:
 * - Acquires lock for exclusive access
 * - Checks capacity and expands if needed (doubles size)
 * - Binary searches for correct insertion position
 * - Shifts existing elements right to make space (memmove)
 * - Inserts new value at calculated position
 * - Increments count
 * - Releases lock
 * 
 * Capacity Management:
 * - Exponential growth (doubles) when capacity reached
 * - Minimizes number of reallocations for growing datasets
 * - Exits program on realloc failure (unrecoverable error)
 * 
 * Thread Safety:
 * - Lock acquisition ensures only one thread modifies buffer at a time
 * - Safe for concurrent calls from multiple OpenMP threads
 * - Lock contention possible under high insertion rate
 * 
 * @param buffer Pointer to sorted_buffer_t to insert into. Must be initialized.
 * @param value The uint64_t value to insert in sorted position
 * 
 * @note Exits program with error message if realloc fails during expansion
 * 
 * @note Thread-safe: acquires buffer->lock for entire operation
 * 
 * @note Maintains strict ascending order (duplicates allowed)
 * 
 * @note After insertion, buffer->count is incremented by 1
 * 
 * @complexity O(log n + n) - O(log n) for binary search, O(n) for memmove shift
 *             Amortized O(log n + n) for insertions with occasional O(n) realloc
 * 
 * @see binary_search_insert_pos() for position calculation
 * @see write_result_immediately() for typical usage inserting discovered n values
 * 
 * @example
 * ```c
 * sorted_buffer_t buffer;
 * init_sorted_buffer(&buffer);
 * 
 * // Thread-safe insertions from parallel region
 * #pragma omp parallel
 * {
 *     insert_sorted(&buffer, 42);
 *     insert_sorted(&buffer, 17);
 *     insert_sorted(&buffer, 99);
 * }
 * 
 * // Buffer now contains: [17, 42, 99] in sorted order
 * // Count = 3
 * 
 * // Capacity expansion example:
 * // If count = 100000 (at capacity), next insertion:
 * // - Reallocates to capacity = 200000
 * // - Copies existing data
 * // - Inserts new value in sorted position
 * ```
 */
static void insert_sorted(sorted_buffer_t* buffer, uint64_t value) {
    omp_set_lock(&buffer->lock);
    
    if (buffer->count >= buffer->capacity) {
        buffer->capacity *= 2;  // Exponential growth to amortize reallocation cost
        buffer->numbers = realloc(buffer->numbers, buffer->capacity * sizeof(uint64_t));
        if (!buffer->numbers) {
            fprintf(stderr, "\n[*] ERROR: Failed to resize sorted buffer\n");
            exit(1);
        }
    }
    
    size_t pos = binary_search_insert_pos(buffer->numbers, buffer->count, value);
    
    if (pos < buffer->count) {
        memmove(&buffer->numbers[pos + 1], &buffer->numbers[pos], 
                (buffer->count - pos) * sizeof(uint64_t));
    }
    
    buffer->numbers[pos] = value;
    buffer->count++;
    
    omp_unset_lock(&buffer->lock);
}

/**
 * @brief Flushes sorted buffer contents to file in JSON array format.
 * 
 * Writes all values from the sorted buffer to the specified file in JSON array
 * format with proper comma separation. Handles the first_write flag to avoid
 * leading commas in the JSON output. After writing, resets the buffer count to 0,
 * making it ready for new insertions without reallocation.
 * 
 * Output Format:
 * - Values written as comma-separated list: "val1, val2, val3"
 * - No comma before first value if first_write is true
 * - Commas added before subsequent values
 * - Updates first_write flag after first value written
 * - File flushed after write to ensure data persisted
 * 
 * Thread Safety:
 * - Acquires buffer lock for exclusive access during flush
 * - Safe to call concurrently with insert_sorted (lock coordination)
 * - File writing itself not protected (caller must ensure file access safety)
 * 
 * Buffer State After Flush:
 * - count reset to 0 (buffer logically empty)
 * - capacity unchanged (memory retained for reuse)
 * - numbers array still allocated and ready for new insertions
 * 
 * @param buffer Pointer to sorted_buffer_t containing values to write
 * @param file File pointer to write to (must be open for writing)
 * @param first_write Pointer to boolean flag indicating if this is first write to file.
 *                    Set to false after first value written. Must not be NULL.
 * 
 * @note Does nothing if file is NULL or buffer count is 0
 * @warning Passing NULL buffer results in undefined behavior (segmentation fault)
 * 
 * @note Buffer count is reset to 0 after successful flush (reuses allocated memory)
 * 
 * @note File is flushed (fflush) after write to ensure data persistence
 * 
 * @note Thread-safe: acquires buffer->lock during operation
 * 
 * @complexity O(n) where n is buffer->count (linear write of all elements)
 * 
 * @see close_streaming_writer() for final flush before file closure
 * @see write_result_immediately() for triggering periodic flushes
 * 
 * @example
 * ```c
 * // Setup
 * sorted_buffer_t buffer;
 * init_sorted_buffer(&buffer);
 * FILE* file = fopen("output.json", "w");
 * fprintf(file, "[");  // JSON array start
 * bool first_write = true;
 * 
 * // Add some values
 * insert_sorted(&buffer, 10);
 * insert_sorted(&buffer, 20);
 * insert_sorted(&buffer, 30);
 * 
 * // First flush
 * flush_buffer_to_file(&buffer, file, &first_write);
 * // File now contains: "[10, 20, 30"
 * // first_write = false
 * // buffer.count = 0 (ready for reuse)
 * 
 * // Add more values
 * insert_sorted(&buffer, 40);
 * insert_sorted(&buffer, 50);
 * 
 * // Second flush
 * flush_buffer_to_file(&buffer, file, &first_write);
 * // File now contains: "[10, 20, 30, 40, 50"
 * 
 * // Finalize
 * fprintf(file, "]");
 * fclose(file);
 * // Final file: "[10, 20, 30, 40, 50]"
 * ```
 */
static void flush_buffer_to_file(sorted_buffer_t* buffer, FILE* file, bool* first_write) {
    if (!file || buffer->count == 0) return;
    
    omp_set_lock(&buffer->lock);
    
    for (size_t i = 0; i < buffer->count; i++) {
        if (!(*first_write)) {
            fprintf(file, ", ");
        } else {
            *first_write = false;
        }
        fprintf(file, "%lu", buffer->numbers[i]);
    }
    fflush(file);
    
    buffer->count = 0;
    
    omp_unset_lock(&buffer->lock);
}


/**
 * @brief Frees all memory associated with a sorted buffer and destroys its lock.
 * 
 * Deallocates the dynamic array backing the buffer and destroys the OpenMP lock,
 * preventing memory leaks and lock resource leaks. Resets the numbers pointer to
 * NULL to prevent dangling pointer access after deallocation.
 * 
 * Cleanup Process:
 * - Checks if numbers array is allocated (safe for already-freed buffers)
 * - Frees numbers array with standard free()
 * - Sets numbers pointer to NULL (defensive programming)
 * - Destroys OpenMP lock to release lock resources
 * 
 * @param buffer Pointer to sorted_buffer_t to deallocate. Must not be NULL.
 * 
 * @note Safe to call even if numbers is already NULL (idempotent for numbers)
 * 
 * @note After calling, buffer->numbers is NULL and buffer should not be used
 *       without re-initialization via init_sorted_buffer()
 * 
 * @note Must be called for every buffer initialized with init_sorted_buffer()
 *       to prevent memory and lock resource leaks
 * 
 * @note Does not free the buffer structure itself (only its internal allocations)
 * 
 * @complexity O(1) - constant time deallocation and lock destruction
 * 
 * @see init_sorted_buffer() for buffer initialization
 * @see close_streaming_writer() for cleanup of writer buffers
 * 
 * @example
 * ```c
 * sorted_buffer_t buffer;
 * init_sorted_buffer(&buffer);
 * 
 * // Use buffer...
 * insert_sorted(&buffer, 42);
 * 
 * // Cleanup
 * free_sorted_buffer(&buffer);
 * // buffer.numbers now NULL
 * // buffer.lock destroyed
 * 
 * // Safe to call again (numbers already NULL)
 * free_sorted_buffer(&buffer);  // No-op for numbers, still destroys lock
 * ```
 */
static void free_sorted_buffer(sorted_buffer_t* buffer) {
    if (buffer->numbers) {
        free(buffer->numbers);
        buffer->numbers = NULL;
    }
    omp_destroy_lock(&buffer->lock);
}

/*
 ***********************************************************************************
 * 10. STREAMING WRITER SYSTEM
 ***********************************************************************************
 */

/**
 * @brief Creates and initializes a streaming writer for dynamic taxonomy file output.
 * 
 * Allocates and configures a streaming writer system that manages up to three sorted
 * buffers (one per taxonomy A/B/C) and lazily creates output files only for taxonomies
 * that actually contain results. Pre-formats filenames for all possible taxonomy outputs
 * and initializes thread-safe locks for deadlock-free concurrent file creation.
 * 
 * Initialization Process:
 * - Allocates streaming_writer_t structure
 * - Sets all file pointers to NULL (lazy creation)
 * - Initializes creation_lock for thread-safe file creation
 * - Pre-formats three filenames with pattern: n_values_for_mr_{mr}_on_range_1_to_2pow{exp}_and_taxonomy_{A|B|C}.json
 * - Initializes three sorted buffers (buffer_A, buffer_B, buffer_C)
 * - Stores exponent and target_mr for filename generation and output management
 * 
 * File Creation Strategy:
 * - Files created on-demand when first value of that taxonomy is found
 * - Maximum 2 files created (A+B or A+C, never all three)
 * - For mr=0, only taxonomy A file is ever created
 * - Prevents creation of empty files for unused taxonomies
 * 
 * Memory Management:
 * - Must call close_streaming_writer() to free resources and close files
 * - Exits program if allocation fails (unrecoverable error)
 * 
 * @param exponent Power of 2 defining search range (used in filename)
 * @param target_mr Pseudocycle mr value being searched (used in filename)
 * 
 * @return Pointer to initialized streaming_writer_t ready for use.
 *         Never returns NULL (exits on allocation failure).
 * 
 * @note Caller must call close_streaming_writer() to prevent resource leaks
 * 
 * @note No files are created during initialization (lazy creation on first write)
 * 
 * @note All three buffers initialized even though max 2 taxonomies can occur
 *       (simpler code, minimal memory overhead)
 * 
 * @complexity O(1) - constant time allocation and initialization
 * 
 * @see close_streaming_writer() for cleanup and file finalization
 * @see write_result_immediately() for adding values to writer
 * @see get_or_create_file() for lazy file creation mechanism
 * 
 * @example
 * ```c
 * // Create writer for exponent=25, mr=1821
 * streaming_writer_t* writer = create_streaming_writer(25, 1821);
 * 
 * // Writer state after creation:
 * // - file_1 = NULL, file_2 = NULL (no files created yet)
 * // - filename_A = "n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_A.json"
 * // - filename_B = "n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_B.json"
 * // - filename_C = "n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_C.json"
 * // - buffer_A, buffer_B, buffer_C all initialized and empty
 * // - creation_lock initialized
 * 
 * // Use writer...
 * write_result_immediately(writer, 27, &analysis);
 * 
 * // Cleanup
 * close_streaming_writer(writer);
 * ```
 */
static streaming_writer_t* create_streaming_writer(int exponent, uint64_t target_mr) {
    streaming_writer_t* writer = allocate_memory_safe(sizeof(streaming_writer_t), "streaming writer");
    
    writer->file_1 = NULL;
    writer->file_2 = NULL;
    writer->taxonomy_1 = '\0';
    writer->taxonomy_2 = '\0';
    writer->file_1_created = false;
    writer->file_2_created = false;
    writer->first_write_1 = true;
    writer->first_write_2 = true;
    writer->exponent = exponent;
    writer->target_mr = target_mr;
    
    omp_init_lock(&writer->creation_lock);
   
    snprintf(writer->filename_A, sizeof(writer->filename_A), 
             "n_values_for_mr_%lu_on_range_1_to_2pow%d_and_taxonomy_A.json", target_mr, exponent);
    snprintf(writer->filename_B, sizeof(writer->filename_B), 
             "n_values_for_mr_%lu_on_range_1_to_2pow%d_and_taxonomy_B.json", target_mr, exponent);
    snprintf(writer->filename_C, sizeof(writer->filename_C), 
             "n_values_for_mr_%lu_on_range_1_to_2pow%d_and_taxonomy_C.json", target_mr, exponent);
    
    init_sorted_buffer(&writer->buffer_A);
    init_sorted_buffer(&writer->buffer_B);
    init_sorted_buffer(&writer->buffer_C);
    
    return writer;
}


/**
 * @brief Thread-safely retrieves or creates output file for specified taxonomy.
 * 
 * Implements lazy file creation with deadlock-free thread coordination. Returns an
 * existing file pointer if the taxonomy file has already been created, or creates
 * a new file on first access. Manages up to 2 simultaneous output files (taxonomies
 * are mutually exclusive: only A+B or A+C combinations occur, never all three).
 * 
 * File Creation Logic:
 * - Checks if requested taxonomy already has an open file (returns immediately)
 * - If not, creates new file in first available slot (file_1 or file_2)
 * - Writes JSON array opening bracket "[" to new file
 * - Records which taxonomy the file represents (taxonomy_1 or taxonomy_2)
 * - Sets creation flag and initializes first_write flag for JSON formatting
 * - Returns file pointer and sets first_write_ptr for caller
 * 
 * Thread Safety:
 * - All operations protected by creation_lock (critical section)
 * - Lock held only during file creation logic (minimal contention)
 * - Multiple threads can safely request different taxonomies concurrently
 * - Deadlock-free: lock released before returning in all paths
 * 
 * Slot Allocation:
 * - Uses two slots (file_1, file_2) to support maximum 2 taxonomies
 * - Allocates slots in order: file_1 first, then file_2
 * - Returns NULL if both slots already used (shouldn't happen in practice)
 * 
 * @param writer Pointer to streaming_writer_t managing output files
 * @param taxonomy Character specifying taxonomy ('A', 'B', or 'C')
 * @param first_write_ptr Output parameter: set to point to first_write flag for this file.
 *                        Used by caller for JSON comma formatting. Must not be NULL.
 * 
 * @return FILE* pointer to opened file for this taxonomy, or NULL if:
 *         - Invalid taxonomy character (not A/B/C)
 *         - Both file slots already allocated to other taxonomies (shouldn't happen)
 *         - File creation fails (prints error and exits program)
 * 
 * @note Exits program with error message if fopen() fails (unrecoverable I/O error)
 * 
 * @note Thread-safe: uses creation_lock for all file creation operations
 * 
 * @note Files are opened in write mode ("w"), truncating any existing content
 * 
 * @note Writes "[" to newly created files as JSON array opening
 * 
 * @complexity O(1) - constant time file creation and lookup
 * 
 * @see create_streaming_writer() for writer initialization
 * @see write_result_immediately() for typical usage requesting files
 * @see close_streaming_writer() for file closure and cleanup
 * 
 * @example
 * ```c
 * streaming_writer_t* writer = create_streaming_writer(25, 1821);
 * 
 * // First call for taxonomy A
 * bool* first_write;
 * FILE* file_a = get_or_create_file(writer, 'A', &first_write);
 * // Creates file, writes "[", returns file pointer
 * // *first_write points to writer->first_write_1 (true)
 * // writer->file_1 = file_a
 * // writer->taxonomy_1 = 'A'
 * 
 * // Second call for taxonomy A (same taxonomy)
 * FILE* file_a2 = get_or_create_file(writer, 'A', &first_write);
 * // Returns existing file_a immediately (no file creation)
 * // file_a2 == file_a
 * 
 * // First call for taxonomy B (different taxonomy)
 * FILE* file_b = get_or_create_file(writer, 'B', &first_write);
 * // Creates second file, writes "[", returns file pointer
 * // writer->file_2 = file_b
 * // writer->taxonomy_2 = 'B'
 * 
 * // Now both slots occupied:
 * // file_1 → taxonomy A
 * // file_2 → taxonomy B
 * 
 * // Third different taxonomy would return NULL (shouldn't happen)
 * FILE* file_c = get_or_create_file(writer, 'C', &first_write);
 * // Returns: NULL (both slots already used)
 * ```
 */
static FILE* get_or_create_file(streaming_writer_t* writer, char taxonomy, bool** first_write_ptr) {
    omp_set_lock(&writer->creation_lock);
    
    if (writer->taxonomy_1 == taxonomy && writer->file_1_created) {
        *first_write_ptr = &writer->first_write_1;
        FILE* f = writer->file_1;
        omp_unset_lock(&writer->creation_lock);
        return f;
    }
    if (writer->taxonomy_2 == taxonomy && writer->file_2_created) {
        *first_write_ptr = &writer->first_write_2;
        FILE* f = writer->file_2;
        omp_unset_lock(&writer->creation_lock);
        return f;
    }
    
    const char* filename = NULL;
    if (taxonomy == 'A') filename = writer->filename_A;
    else if (taxonomy == 'B') filename = writer->filename_B;
    else if (taxonomy == 'C') filename = writer->filename_C;
    else {
        omp_unset_lock(&writer->creation_lock);
        return NULL;
    }
    
    FILE** file_ptr;
    bool* created_flag;
    char* taxonomy_slot;
    
    if (!writer->file_1_created) {
        file_ptr = &writer->file_1;
        created_flag = &writer->file_1_created;
        taxonomy_slot = &writer->taxonomy_1;
        *first_write_ptr = &writer->first_write_1;
    } else if (!writer->file_2_created) {
        file_ptr = &writer->file_2;
        created_flag = &writer->file_2_created;
        taxonomy_slot = &writer->taxonomy_2;
        *first_write_ptr = &writer->first_write_2;
    } else {
        omp_unset_lock(&writer->creation_lock);
        return NULL;
    }
    
    *file_ptr = fopen(filename, "w");
    if (!*file_ptr) {
        fprintf(stderr, "\n[*] ERROR: Could not create output file %s\n", filename);
        exit(1);
    }
    
    fprintf(*file_ptr, "[");
    *created_flag = true;
    *taxonomy_slot = taxonomy;
    
    fflush(stdout);
    
    FILE* result = *file_ptr;
    omp_unset_lock(&writer->creation_lock);
    return result;
}


/**
 * @brief Writes discovered n value to appropriate sorted buffer and displays progress.
 * 
 * This is the primary output function that receives each discovered n value belonging
 * to the target mr class, adds it to the appropriate taxonomy-specific sorted buffer,
 * and reports the discovery to the console. Handles first-time progress header display
 * and periodic buffer flushing to disk when threshold reached.
 * 
 * Processing Flow:
 * - Determines target buffer based on taxonomy (A, B, or C)
 * - Inserts n value into sorted buffer (maintains ascending order)
 * - Displays progress header on first discovery (static first_report flag)
 * - Reports each discovery with n value, taxonomy, and target mr
 * - Triggers buffer flush to file when BUFFER_FLUSH_THRESHOLD (50,000) reached
 * 
 * Console Output:
 * - First call displays: "Progress: (0.0%) | Processed: 0 | Found: 0 | ..."
 * - Each call displays: "Match found: n = {value} (Taxonomy {A|B|C}) generates mr = {mr}"
 * - Output protected by critical section for clean display from parallel threads
 * 
 * Buffer Management:
 * - Values accumulate in memory (sorted buffer) for efficient batch writes
 * - Automatic flush when buffer reaches 50,000 elements
 * - Final flush occurs in close_streaming_writer()
 * - Files created lazily on first flush for each taxonomy
 * 
 * Thread Safety:
 * - Critical section (discovery_report) protects console output
 * - insert_sorted() uses internal locking for buffer thread safety
 * - first_report flag safely shared (only transitions false→true once)
 * 
 * @param writer Pointer to streaming_writer_t managing output files and buffers
 * @param n The discovered n value belonging to target mr class
 * @param analysis Pointer to collatz_analysis_t containing taxonomy classification
 * 
 * @note Uses static first_report flag for one-time header display
 * 
 * @note Thread-safe: critical section for console, locked buffer operations
 * 
 * @note May trigger file creation via get_or_create_file() during flush
 * 
 * @note Invalid taxonomy characters (not A/B/C) are silently ignored (early return)
 * 
 * @complexity O(log n + n) - dominated by insert_sorted binary search and shift
 *             Plus O(m) for periodic flush where m = buffer size
 * 
 * @see insert_sorted() for buffer insertion mechanism
 * @see flush_buffer_to_file() for disk write mechanism
 * @see process_single_number() for typical calling context
 * 
 * @example
 * ```c
 * streaming_writer_t* writer = create_streaming_writer(25, 1821);
 * 
 * // First discovery (taxonomy B)
 * collatz_analysis_t analysis1 = {.taxonomy = 'B', ...};
 * write_result_immediately(writer, 27, &analysis1);
 * // Console output:
 * // "    - Progress: (0.0%) | Processed: 0 | Found: 0 | 0.0 nums/sec | ETA: -- min"
 * // "        - Match found: n = 27 (Taxonomy B) generates mr = 1821"
 * // Value added to buffer_B (no file created yet)
 * 
 * // Second discovery (taxonomy A)
 * collatz_analysis_t analysis2 = {.taxonomy = 'A', ...};
 * write_result_immediately(writer, 51, &analysis2);
 * // Console output:
 * // "        - Match found: n = 51 (Taxonomy A) generates mr = 1821"
 * // Value added to buffer_A
 * 
 * // ... many more discoveries ...
 * 
 * // 50,000th value in buffer_B triggers flush:
 * write_result_immediately(writer, 999999, &analysis1);
 * // Creates file "n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_B.json"
 * // Writes all 50,000 sorted values to file
 * // Resets buffer_B.count to 0 for continued use
 * ```
 */
static void write_result_immediately(streaming_writer_t* writer, 
                                     uint64_t n, 
                                     const collatz_analysis_t* analysis) {
    static bool first_report = true;
    
    char taxonomy = analysis->taxonomy;
    sorted_buffer_t* buffer = NULL;
    
    if (taxonomy == 'A') {
        buffer = &writer->buffer_A;
    } else if (taxonomy == 'B') {
        buffer = &writer->buffer_B;
    } else if (taxonomy == 'C') {
        buffer = &writer->buffer_C;
    } else {
        return;
    }
    
    insert_sorted(buffer, n);
    
    #pragma omp critical(discovery_report)
    {
        if (first_report) {
            printf("\t - Progress: (0.0%%) | Processed: 0 | Found: 0 | 0.0 nums/sec | ETA: -- min\n");
            fflush(stdout);
            first_report = false;
        }
        printf("\t\t - Match found: n = %lu (Taxonomy %c) generates mr = %lu\n", 
               n, taxonomy, writer->target_mr);
        fflush(stdout);
    }
    
    if (buffer->count >= BUFFER_FLUSH_THRESHOLD) {
        bool* first_write = NULL;
        FILE* file = get_or_create_file(writer, taxonomy, &first_write);
        if (file && first_write) {
            flush_buffer_to_file(buffer, file, first_write);
        }
    }
}

/**
 * @brief Finalizes output by flushing remaining buffers and closing all files.
 * 
 * Completes the output process by writing any remaining buffered values to disk,
 * properly closing JSON arrays with closing brackets, and deallocating all resources
 * associated with the streaming writer. Ensures no data loss and properly formatted
 * JSON output files.
 * 
 * Finalization Process:
 * - Prints output files section header to console
 * - Flushes all three buffers (A, B, C) if they contain data
 * - Creates files on-demand during flush if not yet created
 * - Closes JSON arrays by writing "]" to all open files
 * - Closes all file handles
 * - Frees all sorted buffer memory
 * - Destroys creation lock
 * - Frees writer structure itself
 * 
 * Special Handling for mr=0:
 * - All output logic works correctly (only buffer_A will have data)
 * - Automatically handles case where only taxonomy A exists
 * - No special-case code needed (general logic handles it)
 * 
 * Buffer Flushing:
 * - Each buffer flushed independently if count > 0
 * - Creates file if not already created (handles case of small result sets)
 * - Maintains proper JSON comma formatting via first_write flags
 * 
 * @param writer Pointer to streaming_writer_t to finalize and deallocate.
 *               Can be NULL (safe no-op).
 * 
 * @note Safe to call with NULL writer (returns immediately)
 * 
 * @note After calling, writer pointer is invalid and should not be used
 * 
 * @note Ensures all buffered data written to disk before files closed
 * 
 * @note All allocated memory freed (prevents leaks)
 * 
 * @note Files are properly closed even if buffers empty (writes "[]" for empty files)
 * 
 * @complexity O(n) where n is total elements across all buffers (final flush)
 * 
 * @see create_streaming_writer() for writer initialization
 * @see flush_buffer_to_file() for buffer writing mechanism
 * @see cleanup_search_context() for typical calling context
 * 
 * @example
 * ```c
 * streaming_writer_t* writer = create_streaming_writer(25, 1821);
 * 
 * // Write many values...
 * write_result_immediately(writer, 27, &analysis);
 * write_result_immediately(writer, 51, &analysis);
 * // ... more writes ...
 * 
 * // Finalization
 * close_streaming_writer(writer);
 * 
 * // Console output:
 * // "[*] OUTPUT FILES"
 * 
 * // File operations:
 * // 1. Flush buffer_A remaining values (if any) to file
 * // 2. Flush buffer_B remaining values (if any) to file
 * // 3. Write "]" to close JSON arrays in both files
 * // 4. Close both file handles
 * 
 * // Result files contain:
 * // "n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_A.json": [51, ...]
 * // "n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_B.json": [27, ...]
 * 
 * // All memory freed, writer invalid
 * ```
 */
static void close_streaming_writer(streaming_writer_t* writer) {
    if (!writer) return;
    
    printf("\n[*] OUTPUT FILES\n");
    fflush(stdout);
    
    if (writer->buffer_A.count > 0) {
        bool* first_write = NULL;
        FILE* file = get_or_create_file(writer, 'A', &first_write);
        if (file && first_write) {
            flush_buffer_to_file(&writer->buffer_A, file, first_write);
        }
    }
    
    if (writer->buffer_B.count > 0) {
        bool* first_write = NULL;
        FILE* file = get_or_create_file(writer, 'B', &first_write);
        if (file && first_write) {
            flush_buffer_to_file(&writer->buffer_B, file, first_write);
        }
    }
    
    if (writer->buffer_C.count > 0) {
        bool* first_write = NULL;
        FILE* file = get_or_create_file(writer, 'C', &first_write);
        if (file && first_write) {
            flush_buffer_to_file(&writer->buffer_C, file, first_write);
        }
    }
    
    if (writer->file_1) {
        fprintf(writer->file_1, "]");
        fclose(writer->file_1);
    }
    
    if (writer->file_2) {
        fprintf(writer->file_2, "]");
        fclose(writer->file_2);
    }
    
    free_sorted_buffer(&writer->buffer_A);
    free_sorted_buffer(&writer->buffer_B);
    free_sorted_buffer(&writer->buffer_C);
    
    omp_destroy_lock(&writer->creation_lock);
    
    free(writer);
}

/* 
 ***********************************************************************************
 * 11. PROGRESS TRACKING SYSTEM
 ***********************************************************************************
 */

 /**
 * @brief Creates and initializes a progress tracker for monitoring search execution.
 * 
 * Allocates and configures a progress tracking structure that maintains thread-safe
 * counters for monitoring search progress in real-time. Initializes all counters to
 * starting values and sets up OpenMP lock for atomic updates from parallel threads.
 * 
 * Initialization:
 * - Allocates progress_tracker_t structure
 * - Sets processed counter to initial_processed (for checkpoint resume support)
 * - Zeros found_count and last_n_found counters
 * - Initializes last_update_time to 0.0 (triggers immediate first update)
 * - Creates OpenMP lock for thread-safe counter access
 * 
 * @param initial_processed Starting value for processed counter (0 for fresh start,
 *                          non-zero when resuming from checkpoint)
 * 
 * @return Pointer to initialized progress_tracker_t ready for use.
 *         Never returns NULL (exits on allocation failure).
 * 
 * @note Caller must destroy lock with omp_destroy_lock when done
 * 
 * @note Exits program if allocation fails (unrecoverable error)
 * 
 * @complexity O(1) - constant time allocation and initialization
 * 
 * @see update_progress_if_needed() for progress reporting using tracker
 * @see load_checkpoint() for setting initial_processed from checkpoint
 * 
 * @example
 * ```c
 * // Fresh start
 * progress_tracker_t* tracker = create_progress_tracker(0);
 * // tracker->processed = 0
 * // tracker->found_count = 0
 * 
 * // Resume from checkpoint
 * progress_tracker_t* tracker2 = create_progress_tracker(5000000);
 * // tracker2->processed = 5000000 (continuing from checkpoint)
 * ```
 */
static progress_tracker_t* create_progress_tracker(uint64_t initial_processed) {
    progress_tracker_t* tracker = allocate_memory_safe(sizeof(progress_tracker_t), "progress tracker");
    tracker->processed = initial_processed;
    tracker->found_count = 0;
    tracker->last_n_found = 0;
    tracker->last_update_time = 0.0;
    omp_init_lock(&tracker->lock);
    return tracker;
}

/**
 * @brief Displays periodic progress updates with statistics and ETA estimation.
 * 
 * Checks if sufficient time has elapsed since the last progress report and, if so,
 * displays comprehensive search statistics including completion percentage, processing
 * rate, and estimated time to completion. Throttles updates to avoid excessive console
 * output while providing regular feedback during long-running searches.
 * 
 * Update Strategy:
 * - Checks elapsed time since last update (throttle at PROGRESS_UPDATE_INTERVAL = 10 seconds)
 * - Captures atomic snapshot of progress counters under lock
 * - Calculates statistics: percentage, rate, ETA
 * - Displays formatted progress line to console
 * - Updates last_update_time to prevent immediate re-trigger
 * 
 * Output Format:
 * - Early stages (< 0.01%): "Progress: (X%) | Processed: N | Found: M | R nums/sec"
 * - Later stages (≥ 0.01%): Same format plus "| ETA: X.X min"
 * - ETA suppressed at start to avoid wildly inaccurate estimates
 * 
 * Statistics Computed:
 * - progress_percent: (processed / max_n) * 100
 * - rate: processed / elapsed_time (numbers per second)
 * - eta: elapsed * (100 - progress) / progress (remaining time in seconds)
 * 
 * Thread Safety:
 * - Acquires tracker lock for entire update operation
 * - Takes atomic snapshot of counters to ensure consistency
 * - Lock held during calculation and console output (minimal contention)
 * 
 * @param ctx Pointer to search context containing progress tracker and search parameters
 * 
 * @note Only displays update if PROGRESS_UPDATE_INTERVAL seconds have elapsed
 * 
 * @note Thread-safe: acquires progress->lock for atomic counter access
 * 
 * @note ETA calculation suppressed until progress ≥ 0.01% to avoid early inaccuracy
 * 
 * @note Console output flushed immediately for real-time visibility
 * 
 * @complexity O(1) - constant time statistics calculation and output
 * 
 * @see create_progress_tracker() for tracker initialization
 * @see execute_search_with_guided_scheduling() for periodic calling
 * 
 * @example
 * ```c
 * search_context_t ctx = { ... };
 * 
 * // Early in search (0.005% complete):
 * update_progress_if_needed(&ctx);
 * // Output: "    - Progress: (0.0050000000%) | Processed: 1677 | Found 12 | 5234.2 nums/sec"
 * // (no ETA shown yet)
 * 
 * // Later in search (15.234% complete):
 * update_progress_if_needed(&ctx);
 * // Output: "    - Progress: (15.2340000000%) | Processed: 5112345 | Found 156 | 425678.3 nums/sec | ETA: 1.2 min"
 * 
 * // Called every ~10 seconds during search:
 * // Time 0s: (no output, last_update_time = 0)
 * // Time 10s: Progress update displayed
 * // Time 15s: (no output, < 10s since last)
 * // Time 20s: Progress update displayed
 * // ...
 * ```
 */
static void update_progress_if_needed(const search_context_t* ctx) {
    progress_tracker_t* tracker = ctx->progress;
    
    omp_set_lock(&tracker->lock);
    
    double current_time = omp_get_wtime();
    if (current_time - tracker->last_update_time >= PROGRESS_UPDATE_INTERVAL) {
        tracker->last_update_time = current_time;
        uint64_t processed_snapshot = tracker->processed;
        uint64_t found_snapshot = tracker->found_count;
        
        double elapsed = current_time - ctx->start_time;
        double progress_percent = (double)processed_snapshot / ctx->max_n * 100.0;
        double rate = (elapsed > 0.0) ? processed_snapshot / elapsed : 0.0;

        if (progress_percent >= PROGRESS_ETA_THRESHOLD) {
            double eta = elapsed * (100.0 - progress_percent) / progress_percent;
            printf("\t - Progress: (%.10f%%) | Processed: %lu | Found %lu | %.1f nums/sec | ETA: %.1f min\n",
                   progress_percent, processed_snapshot, found_snapshot, rate, eta/60.0);
        } else {
            printf("\t - Progress: (%.10f%%) | Processed: %lu | Found %lu | %.1f nums/sec\n",
                   progress_percent, processed_snapshot, found_snapshot, rate);
        }
        fflush(stdout);
    }
    
    omp_unset_lock(&tracker->lock);
}

/* 
 ***********************************************************************************
 * 12. CORE PROCESSING ENGINE
 ***********************************************************************************
 */

 /**
 * @brief Processes a single number through verification and analysis pipeline.
 * 
 * This is the core per-number processing function that determines whether a given n
 * belongs to the target mr class and, if so, performs full sequence analysis and
 * outputs the result. Implements the complete two-stage verification workflow:
 * fast filtering followed by detailed analysis for matching numbers.
 * 
 * Processing Pipeline:
 * 1. Fast verification: Check if target_mr is first complete pseudocycle
 * 2. If verification fails: Increment processed counter, return early
 * 3. If verification passes: Perform full sequence analysis
 * 4. Check if pseudocycle was found (both ni and nj detected)
 * 5. If found: Increment found counter, write result to output
 * 6. Always increment processed counter
 * 
 * Two-Stage Design Rationale:
 * - verify_target_mr_is_first_pseudocycle_fast() quickly filters out ~99.99% of numbers
 * - analyze_collatz_sequence() only called for numbers passing verification
 * - Minimizes expensive full sequence analysis to only relevant numbers
 * - Local counters reduce atomic operation overhead in tight loop
 * 
 * Counter Management:
 * - Uses local counters (thread-private) for performance
 * - Caller aggregates local counters via OpenMP reduction
 * - Global found_count updated atomically for real-time reporting
 * 
 * @param n The number to process and potentially classify
 * @param ctx Pointer to search context containing target_mr, dictionary entry, and writer
 * @param local_found Pointer to thread-local counter of found values (incremented if match)
 * @param local_processed Pointer to thread-local counter of processed values (always incremented)
 * 
 * @note Always increments local_processed regardless of whether n matches
 * 
 * @note Only increments local_found if pseudocycle actually detected in analysis
 * 
 * @note Updates global ctx->progress->found_count atomically for progress reporting
 * 
 * @note Calls write_result_immediately() which handles output and console reporting
 * 
 * @complexity O(k) where k is Collatz sequence length (dominated by verification step)
 * 
 * @see verify_target_mr_is_first_pseudocycle_fast() for first-stage filtering
 * @see analyze_collatz_sequence() for second-stage analysis
 * @see write_result_immediately() for output handling
 * @see execute_search_with_guided_scheduling() for parallel calling context
 * 
 * @example
 * ```c
 * search_context_t ctx = { .target_mr = 1821, ... };
 * uint64_t local_found = 0, local_processed = 0;
 * 
 * // Process n=27 (belongs to different mr class)
 * process_single_number(27, &ctx, &local_found, &local_processed);
 * // verify returns false → early exit
 * // local_processed = 1, local_found = 0
 * 
 * // Process n=51 (belongs to mr=25, not target)
 * process_single_number(51, &ctx, &local_found, &local_processed);
 * // verify returns false → early exit
 * // local_processed = 2, local_found = 0
 * 
 * // Process n=3643 (belongs to mr=1821, the target)
 * process_single_number(3643, &ctx, &local_found, &local_processed);
 * // verify returns true → analyze_collatz_sequence called
 * // analysis.found_pseudocycle = true
 * // local_found = 1, local_processed = 3
 * // Result written to output file
 * // Console: "Match found: n = 3643 (Taxonomy A) generates mr = 1821"
 * 
 * // After processing many numbers in parallel:
 * // Thread's local counters aggregated via OpenMP reduction
 * ```
 */
static void process_single_number(uint64_t n, search_context_t* ctx, 
                                  uint64_t* local_found, uint64_t* local_processed) {
    
    if (!verify_target_mr_is_first_pseudocycle_fast(n, ctx->target_mr)) {
        (*local_processed)++;
        return;
    }
    
    collatz_analysis_t analysis = analyze_collatz_sequence(n, ctx->dict_entry);
    
    if (analysis.found_pseudocycle) {
        (*local_found)++;
        
        write_result_immediately(ctx->writer, n, &analysis);
        
        #pragma omp atomic
        ctx->progress->found_count++;
    }
    
    (*local_processed)++;
}

/*
 ***********************************************************************************
 * 13. PARALLEL SCHEDULING SYSTEM
 ***********************************************************************************
 */

/**
 * @brief Executes parallel search with guided scheduling, checkpointing, and progress tracking.
 * 
 * This is the main parallel execution engine that distributes the search workload across
 * multiple OpenMP threads using guided scheduling for optimal load balancing. Implements
 * automatic checkpointing every 5 minutes, graceful interrupt handling (Ctrl+C), periodic
 * progress updates, and efficient counter aggregation via OpenMP reductions.
 * 
 * Parallel Execution Strategy:
 * - OpenMP parallel region with guided scheduling (dynamic load balancing)
 * - Each thread maintains local counters (reduced overhead vs. atomic operations)
 * - Thread 0 designated as coordinator for progress/checkpoint duties
 * - Reduction clauses aggregate thread-local counters at parallel region end
 * 
 * Checkpointing System:
 * - Automatic save every CHECKPOINT_INTERVAL (300 seconds = 5 minutes)
 * - Manual save on Ctrl+C signal (checkpoint_signal_received flag)
 * - Final checkpoint saved at completion
 * - Checkpoint files cleaned up after successful completion
 * 
 * Progress Tracking:
 * - Thread-local counters updated every iteration
 * - Global counter synchronized every LOCAL_COUNTER_UPDATE_FREQUENCY (100) iterations
 * - Progress display updated every PROGRESS_CHECK_FREQUENCY (1,000,000) iterations
 * - Thread 0 exclusively handles progress reporting to avoid contention
 * 
 * Interrupt Handling:
 * - Checks checkpoint_signal_received flag periodically (thread 0 only)
 * - On interrupt: saves checkpoint, displays resume instructions, exits cleanly
 * - Ensures no data loss on premature termination
 * 
 * Thread Coordination:
 * - Thread 0: handles progress, checkpointing, and interrupt detection
 * - All threads: process numbers, update local counters, participate in reductions
 * - Atomic operations minimize lock contention
 * 
 * @param ctx Pointer to search context with all search parameters and state
 * @param start_n First number to process (1 for fresh start, >1 for checkpoint resume)
 * @param total_found Output parameter: set to total number of matching n values found
 * @param total_processed Output parameter: set to total number of n values processed
 * 
 * @note Initializes hash table if not already initialized (idempotent)
 * 
 * @note Uses OpenMP guided scheduling for better load balancing than static chunks
 * 
 * @note Thread 0 bears additional responsibility for coordination tasks
 * 
 * @note Saves final checkpoint even on completion (cleaned up immediately after)
 * 
 * @note Automatically removes checkpoint files after successful completion
 * 
 * @warning Calls exit(0) on Ctrl+C interrupt after saving checkpoint
 * 
 * @complexity O(n/p * k) where n is search range size, p is thread count, k is average
 *             sequence length (near-linear scaling with thread count)
 * 
 * @see process_single_number() for per-number processing logic
 * @see save_checkpoint() for checkpoint persistence mechanism
 * @see update_progress_if_needed() for progress display
 * @see execute_parallel_search() for typical calling context
 * 
 * @example
 * ```c
 * search_context_t ctx = { .max_n = 33554432, .target_mr = 1821, ... };
 * uint64_t found = 0, processed = 0;
 * 
 * // Fresh start from n=1
 * execute_search_with_guided_scheduling(&ctx, 1, &found, &processed);
 * 
 * // Example execution timeline:
 * // T=0s:   Search starts, hash table initialized
 * // T=10s:  First progress update displayed
 * // T=300s: Automatic checkpoint saved at n=X
 * // T=310s: Progress update
 * // T=600s: Another automatic checkpoint
 * // ...
 * // T=Xs:   User presses Ctrl+C
 * //         → Checkpoint saved immediately
 * //         → "Interrupted! Checkpoint saved at n = Y"
 * //         → "Run again with same parameters to resume"
 * //         → exit(0)
 * 
 * // If allowed to complete:
 * // Final checkpoint saved
 * // Checkpoint files removed
 * // Console: "Final checkpoint saved"
 * //          "Checkpoint files cleaned...search completed"
 * // found = total matches, processed = 33554431
 * ```
 */
static void execute_search_with_guided_scheduling(search_context_t* ctx, uint64_t start_n, 
                                                   uint64_t* total_found, uint64_t* total_processed) {
    double last_checkpoint_time = omp_get_wtime();
    uint64_t local_found = 0, local_processed = 0;
    uint64_t checkpoint_position = start_n;
    
    if (!hash_table_initialized) {
        initialize_hash_table();
    }
    
    #pragma omp parallel reduction(+:local_found, local_processed)
    {
        uint64_t thread_found = 0, thread_processed = 0;
        int thread_num = omp_get_thread_num();
        
        #pragma omp for schedule(guided)
        for (uint64_t n = start_n; n < ctx->max_n; n++) {
            process_single_number(n, ctx, &thread_found, &thread_processed);
            
            if (thread_processed % LOCAL_COUNTER_UPDATE_FREQUENCY == 0) {
                #pragma omp atomic
                // Periodic sync to reduce atomic operation overhead
                ctx->progress->processed += LOCAL_COUNTER_UPDATE_FREQUENCY;
            }
            
            if (thread_num == 0) {
                #pragma omp atomic write
                // Track safe resume point for checkpointing
                checkpoint_position = n;  
                
                if (thread_processed % PROGRESS_CHECK_FREQUENCY == 0) {
                    update_progress_if_needed(ctx);
                    
                    double current_time = omp_get_wtime();
                    if (current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL) {
                        uint64_t safe_pos;
                        #pragma omp atomic read
                        safe_pos = checkpoint_position;
                        
                        save_checkpoint(ctx, safe_pos);
                        last_checkpoint_time = current_time;
                        printf("\t - [Autosaving checkpoint at n = %lu]\n", safe_pos);
                        fflush(stdout);
                    }
                }
                
                if (checkpoint_signal_received) {
                    uint64_t safe_pos;
                    #pragma omp atomic read
                    safe_pos = checkpoint_position;
                    
                    save_checkpoint(ctx, safe_pos);
                    printf("\n\t - Interrupted! Checkpoint saved at n = %lu\n", safe_pos);
                    printf("\t - Run again with same parameters to resume\n");
                    fflush(stdout);
                    exit(0);
                }
            }
        }
        
        local_found += thread_found;
        local_processed += thread_processed;
    }
    
    save_checkpoint(ctx, ctx->max_n - 1);
    printf("\t - Final checkpoint saved\n");
    
    remove(CHECKPOINT_FILE);
    remove(CHECKPOINT_BACKUP);
    printf("\t - Checkpoint files cleaned...search completed\n");
    
    *total_found = local_found;
    *total_processed = local_processed;
}

/**
 * @brief High-level wrapper that initiates parallel search with appropriate messaging.
 * 
 * Provides a clean interface for starting the parallel search process with user-friendly
 * console output indicating whether this is a fresh start or checkpoint resume. Delegates
 * actual search execution to execute_search_with_guided_scheduling() while handling the
 * presentation layer.
 * 
 * Functionality:
 * - Prints search process header to console
 * - Displays starting position (fresh start vs. resume)
 * - Calls core parallel search engine
 * - Returns aggregated found/processed counts to caller
 * 
 * Console Output Examples:
 * - Fresh start: "Starting from n=1"
 * - Resume: "Resuming from n=5000001"
 * 
 * @param ctx Pointer to search context with all search parameters and state
 * @param start_n First number to process (1 for fresh, >1 for resume)
 * @param found_count Output parameter: receives total matches found
 * @param processed_count Output parameter: receives total numbers processed
 * 
 * @note This is a thin wrapper around execute_search_with_guided_scheduling()
 * 
 * @note Primarily handles user-facing messaging and delegates computation
 * 
 * @complexity O(n/p * k) - same as execute_search_with_guided_scheduling()
 * 
 * @see execute_search_with_guided_scheduling() for actual search implementation
 * @see main() for typical calling context
 * 
 * @example
 * ```c
 * search_context_t ctx = { ... };
 * uint64_t found = 0, processed = 0;
 * 
 * // Fresh start
 * execute_parallel_search(&ctx, 1, &found, &processed);
 * // Console output:
 * // "[*] SEARCH PROCESS"
 * // "    - Starting from n=1"
 * // [progress updates follow...]
 * 
 * // Resume from checkpoint
 * execute_parallel_search(&ctx, 5000001, &found, &processed);
 * // Console output:
 * // "[*] SEARCH PROCESS"
 * // "    - Resuming from n=5000001"
 * // [progress updates follow...]
 * ```
 */  
static void execute_parallel_search(search_context_t* ctx, uint64_t start_n, 
                                    uint64_t* found_count, uint64_t* processed_count) {
    printf("\n[*] SEARCH PROCESS\n");
    
    if (start_n == 1) {
        printf("\t - Starting from n=1\n");
    } else {
        printf("\t - Resuming from n=%lu\n", start_n);
    }
    
    execute_search_with_guided_scheduling(ctx, start_n, found_count, processed_count);
}

/*
 ***********************************************************************************
 * 14. CLEANUP AND VALIDATION
 ***********************************************************************************
 */

 /**
 * @brief Cleans up all resources associated with search context to prevent memory leaks.
 * 
 * Performs comprehensive cleanup of all dynamically allocated resources and system
 * resources (locks, file handles) used during search execution. Ensures proper
 * deallocation order to prevent memory leaks and resource leaks, leaving the system
 * in a clean state after search completion or failure.
 * 
 * Cleanup Order:
 * 1. Close streaming writer (flushes buffers, closes files, frees writer structure)
 * 2. Destroy progress tracker lock and free progress structure
 * 3. Clean up hash table (frees all nodes and entry arrays)
 * 
 * Safety Features:
 * - Null-pointer safe: checks if components exist before cleanup
 * - Sets pointers to NULL after freeing (prevents double-free)
 * - Idempotent for hash table cleanup (safe to call multiple times)
 * 
 * Resource Types Cleaned:
 * - Heap memory: writer, progress tracker, hash table nodes, sorted buffers
 * - File handles: all open output files
 * - OpenMP locks: writer creation_lock, buffer locks, progress lock
 * 
 * @param ctx Pointer to search_context_t to clean up. Can be NULL (safe no-op).
 * 
 * @note Safe to call with NULL or partially initialized context
 * 
 * @note After calling, ctx->writer and ctx->progress are NULL and should not be used
 * 
 * @note Hash table is reset to uninitialized state (can be reinitialized if needed)
 * 
 * @note Does not free the ctx structure itself (typically stack-allocated)
 * 
 * @complexity O(m + b) where m is hash table nodes and b is total buffer elements
 * 
 * @see close_streaming_writer() for writer cleanup details
 * @see cleanup_hash_table() for hash table deallocation
 * @see main() for typical calling context at program termination
 * 
 * @example
 * ```c
 * search_context_t ctx = {
 *     .writer = create_streaming_writer(25, 1821),
 *     .progress = create_progress_tracker(0),
 *     ...
 * };
 * 
 * // Use context for search...
 * execute_parallel_search(&ctx, 1, &found, &processed);
 * 
 * // Cleanup all resources
 * cleanup_search_context(&ctx);
 * 
 * // After cleanup:
 * // - All output files flushed and closed
 * // - All buffers freed
 * // - All locks destroyed
 * // - Hash table deallocated
 * // - ctx.writer = NULL
 * // - ctx.progress = NULL
 * // - No memory leaks
 * 
 * // Safe to call multiple times (no-op after first call)
 * cleanup_search_context(&ctx);
 * ```
 */
static void cleanup_search_context(search_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->writer) {
        close_streaming_writer(ctx->writer);
        ctx->writer = NULL;
    }
    
    if (ctx->progress) {
        omp_destroy_lock(&ctx->progress->lock);
        free(ctx->progress);
    }
    
    cleanup_hash_table();
}

/**
 * @brief Displays program header banner to console.
 * 
 * Prints a formatted banner introducing the program and its author. Provides
 * visual separation and clear identification of program output. Called once
 * at program start before argument validation.
 * 
 * Output Format:
 * - Top border line with asterisks
 * - Program description centered
 * - Author and year right-aligned
 * - Bottom border line
 * 
 * @note Pure output function with no side effects
 * 
 * @note Always outputs to stdout (not stderr)
 * 
 * @complexity O(1) - constant time console output
 * 
 * @see validate_and_parse_arguments() for typical calling context
 * 
 * @example
 * ```c
 * print_program_header();
 * 
 * // Console output:
 * // ************************************************************************************
 * // * High-performance search engine to find n values belonging to a specific mr class *
 * // *                                                            Javier Hernandez 2026 *
 * // ************************************************************************************
 * ```
 */
static void print_program_header() {
    printf("\n************************************************************************************\n");
    printf("* High-performance search engine to find n values belonging to a specific mr class *\n");
    printf("*                                                            Javier Hernandez 2026 *\n");
    printf("************************************************************************************\n");
}


/**
 * @brief Validates command-line arguments and parses search parameters.
 * 
 * Comprehensive validation and parsing of command-line arguments with detailed error
 * messages and usage examples. Validates exponent range, target_mr format and dictionary
 * membership, and provides helpful feedback for all error cases. Computes derived values
 * (max_n) and returns parsed parameters via output pointers.
 * 
 * Validation Checks:
 * 1. Argument count (must be exactly 3: program name, exponent, target_mr)
 * 2. Exponent format (valid integer string)
 * 3. Exponent range (MIN_EXPONENT=1 to MAX_EXPONENT=63)
 * 4. Target_mr format (valid non-negative integer string)
 * 5. Target_mr dictionary membership (must exist in PSEUDOCYCLE_DICTIONARY)
 * 
 * Error Handling Strategy:
 * - Prints program header before any validation
 * - Displays usage instructions on argument count error
 * - Provides specific error messages with examples for each validation failure
 * - Lists all valid mr values if target_mr not found in dictionary
 * - Returns false on any validation failure (caller should exit)
 * 
 * Output Parameters:
 * - exponent: Parsed power of 2 for search range
 * - max_n: Computed as 2^exponent (search range upper bound)
 * - target_mr: Parsed target pseudocycle mr value
 * 
 * Help Output Includes:
 * - Usage syntax
 * - Example command lines
 * - Recommended exponent values with ranges
 * - Complete list of available mr values (42 total)
 * 
 * @param argc Argument count from main()
 * @param argv Argument vector from main()
 * @param exponent Output parameter: receives parsed exponent value
 * @param max_n Output parameter: receives computed 2^exponent value
 * @param target_mr Output parameter: receives parsed target_mr value
 * 
 * @return true if all arguments valid and parsed successfully, false otherwise
 * 
 * @note Always calls print_program_header() before validation
 * 
 * @note On false return, detailed error message already printed to console
 * 
 * @note max_n computed as (1UL << exponent) after validation
 * 
 * @note Uses strtol/strtoull for robust integer parsing with error detection
 * 
 * @complexity O(n) where n is DICTIONARY_SIZE for mr lookup (effectively O(1))
 * 
 * @see lookup_pseudocycle() for dictionary membership check
 * @see main() for typical calling context
 * 
 * @example
 * ```c
 * // Valid invocation
 * // ./search_n_values_by_mr 25 1821
 * int exponent;
 * uint64_t max_n, target_mr;
 * if (validate_and_parse_arguments(argc, argv, &exponent, &max_n, &target_mr)) {
 *     // exponent = 25
 *     // max_n = 33554432
 *     // target_mr = 1821
 * }
 * 
 * // Invalid exponent
 * // ./search_n_values_by_mr 99 1821
 * // Output:
 * // "[*] ERROR: Exponent 99 is out of valid range. Must be between 1 and 63."
 * // "    - Note: Maximum is 63 to prevent overflow (2^64 causes undefined behavior)"
 * 
 * // Invalid target_mr
 * // ./search_n_values_by_mr 25 9999
 * // Output:
 * // "[*] ERROR: Target mr=9999 not found in pseudocycle dictionary."
 * // "    - Known mr values (42 total):"
 * // "        0, 1, 2, 3, 6, 7, 8, 9, 12, 16,"
 * // "        [... all 42 values listed ...]"
 * 
 * // Wrong argument count
 * // ./search_n_values_by_mr 25
 * // Output:
 * // "[*] USAGE:"
 * // "    ./search_n_values_by_mr <exponent> <target_mr>"
 * // "[*] EXAMPLES:"
 * // "    ./search_n_values_by_mr 25 1821  (search n < 2^25 for mr=1821)"
 * // [... complete usage instructions ...]
 * ```
 */
static bool validate_and_parse_arguments(int argc, char* argv[], int* exponent, uint64_t* max_n, uint64_t* target_mr) {
    
    print_program_header();

    if (argc != 3) {
        printf("\n[*] USAGE:");
        printf("\n\t%s <exponent> <target_mr>\n", argv[0]);
        printf("\n[*] EXAMPLES:");
        printf("\n\t%s 25 1821  (search n < 2^25 for mr=1821)\n", argv[0]);
        printf("\n\t%s 20 0     (search n < 2^20 for mr=0, trivial cycle)\n", argv[0]);
        printf("\n\t%s 30 3643  (search n < 2^30 for mr=3643)\n", argv[0]);
        printf("\n[*] RECOMMENDED EXPONENTS:");
        printf("\n\t20 -> 2^20 = 1,048,576 (quick test)");
        printf("\n\t25 -> 2^25 = 33,554,432 (default)");
        printf("\n\t30 -> 2^30 = 1,073,741,824 (intensive use)");
        printf("\n[*] AVAILABLE MR VALUES:");
        printf("\n\t0, 1, 2, 3, 6, 7, 8, 9, 12, 16, 19, 25, 45, 53, 60, 79,");
        printf("\n\t91, 121, 125, 141, 166, 188, 205, 243, 250, 324, 333, 432,");
        printf("\n\t444, 487, 576, 592, 649, 667, 683, 865, 889, 1153, 1214,");
        printf("\n\t1821, 2428, 3643");
        printf("\n\n");
        return false;
    }
    
    // Validate exponent
    char* endptr;
    long parsed_exponent = strtol(argv[1], &endptr, 10);
    
    if (endptr == argv[1] || *endptr != '\0') {
        printf("\n[*] ERROR: Invalid exponent '%s'. Must be a valid integer.\n", argv[1]);
        printf("\t- Examples of valid input: 20, 25, 30\n");
        printf("\t- Examples of invalid input: 32x, abc, 25.5, 30 40\n\n");
        return false;
    }
    
    *exponent = (int)parsed_exponent;
    
    if (*exponent < MIN_EXPONENT || *exponent > MAX_EXPONENT) {
        printf("\n[*] ERROR: Exponent %d is out of valid range. Must be between %d and %d.\n", 
                *exponent, MIN_EXPONENT, MAX_EXPONENT);
        printf("\t- Note: Maximum is %d to prevent overflow (2^64 causes undefined behavior)\n\n", 
                MAX_EXPONENT);
        return false;
    }
    
    // Validate target_mr
    char* endptr_mr;
    unsigned long long parsed_mr = strtoull(argv[2], &endptr_mr, 10);
    
    if (endptr_mr == argv[2] || *endptr_mr != '\0') {
        printf("\n[*] ERROR: Invalid target_mr '%s'. Must be a valid integer.\n", argv[2]);
        printf("\t- Examples of valid input: 0, 1821, 3643\n");
        printf("\t- Examples of invalid input: abc, 1821x, 25.5, 30 40\n\n");
        return false;
    }
    
    if (argv[2][0] == '-') {
        printf("\n[*] ERROR: Invalid target_mr '%s'. Must be a non-negative integer.\n", argv[2]);
        printf("\t- Target mr cannot be negative\n\n");
        return false;
    }
    
    *target_mr = (uint64_t)parsed_mr;
    
    // Verify MR in dictionary
    const dictionary_entry_t* dict_check = lookup_pseudocycle(*target_mr);
    if (!dict_check) {
        printf("\n[*] ERROR: Target mr=%lu not found in pseudocycle dictionary.\n", *target_mr);
        printf("\n\t- Known mr values (%zu total):\n\t\t", DICTIONARY_SIZE);
        for (size_t i = 0; i < DICTIONARY_SIZE; i++) {
            printf("%lu", PSEUDOCYCLE_DICTIONARY[i].target_mr);
            if (i < DICTIONARY_SIZE - 1) {
                printf(", ");
                if ((i + 1) % 10 == 0 && i < DICTIONARY_SIZE - 1) printf("\n\t\t");
            }
        }
        printf("\n\n");
        return false;
    }
    
    *max_n = 1UL << *exponent;
    return true;
}

/* 
 ***********************************************************************************
 * 15. MAIN FUNCTION
 ***********************************************************************************
 */

 /**
 * @brief Main entry point orchestrating the complete search workflow from initialization to cleanup.
 * 
 * Coordinates the entire search process including argument validation, checkpoint loading,
 * parallel search execution, results reporting, and resource cleanup. Implements the complete
 * workflow for finding all n values belonging to a specific mr class within the range [1, 2^exponent).
 * 
 * Execution Workflow:
 * 1. Validate and parse command-line arguments (exponent, target_mr)
 * 2. Install Ctrl+C signal handler for graceful interruption
 * 3. Look up pseudocycle dictionary entry for target_mr
 * 4. Display algorithm setup information (threads, range, pseudocycle)
 * 5. Create streaming writer for output file management
 * 6. Initialize search context with all components
 * 7. Attempt to load checkpoint (resume if found, start fresh otherwise)
 * 8. Execute parallel search across all available threads
 * 9. Display comprehensive results summary (counts, timing, rate)
 * 10. Clean up all resources (files, memory, locks)
 * 11. List generated output files with sizes
 * 12. Handle mr=0 special case (only show taxonomy A file)
 * 
 * Special Handling:
 * - mr=0 (trivial cycle): Only displays taxonomy A output file
 * - mr>0: Displays up to 2 taxonomy files (A+B or A+C combinations)
 * - Checkpoint resume: Seamlessly continues from last saved position
 * - Ctrl+C: Saves checkpoint and exits cleanly (can resume later)
 * 
 * Output Summary Includes:
 * - Total numbers processed in range
 * - Total execution time (seconds)
 * - Processing speed (numbers/second)
 * - Count of valid n values found
 * - Match rate percentage
 * - Generated output files with sizes
 * 
 * File Output Protection:
 * - For mr=0: Only acknowledges taxonomy A file, warns about unexpected B/C files
 * - For mr>0: Lists all created taxonomy files (typically A and one of B or C)
 * - Helps users identify stale files from previous runs with different parameters
 * 
 * @param argc Argument count from command line
 * @param argv Argument vector: [program_name, exponent, target_mr]
 * 
 * @return 0 on successful completion, 1 on error (invalid arguments or dictionary lookup failure)
 * 
 * @note Never returns on Ctrl+C interrupt (exit(0) called in signal handler after checkpoint)
 * 
 * @note Installs SIGINT handler that saves checkpoint and exits on Ctrl+C
 * 
 * @note All dynamically allocated resources freed before return (no leaks)
 * 
 * @note Output files remain on disk after program completion
 * 
 * @complexity O(n/p * k) where n is search range (2^exponent), p is thread count, 
 *             k is average Collatz sequence length
 * 
 * @see validate_and_parse_arguments() for argument parsing
 * @see execute_parallel_search() for main computation
 * @see cleanup_search_context() for resource deallocation
 * 
 * @example
 * ```c
 * // Example invocation and output:
 * // $ ./search_n_values_by_mr 25 1821
 * 
 * // ************************************************************************************
 * // * High-performance search engine to find n values belonging to a specific mr class *
 * // *                                                            Javier Hernandez 2026 *
 * // ************************************************************************************
 * //
 * // [*] ALGORITHM SETUP
 * //     - Using 16 threads
 * //     - Exploring range from 1 to 2^25 - 1 = 33554431
 * //     - Target mr: 1821
 * //     - Pseudocycle: [3643, 3644]
 * //
 * // [*] SEARCH PROCESS
 * //     - Starting from n=1
 * //     - Progress: (0.0%) | Processed: 0 | Found: 0 | 0.0 nums/sec | ETA: -- min
 * //         - Match found: n = 3643 (Taxonomy A) generates mr = 1821
 * //         - Match found: n = 7286 (Taxonomy A) generates mr = 1821
 * //     - Progress: (15.234%) | Processed: 5112345 | Found 156 | 425678.3 nums/sec | ETA: 1.2 min
 * //     [... more progress updates ...]
 * //     - Final checkpoint saved
 * //     - Checkpoint files cleaned...search completed
 * //
 * // [*] SEARCH RESULTS
 * //     - Total numbers processed: 33554431
 * //     - Total time: 78.456 seconds
 * //     - Speed: 427623.12 numbers/second
 * //     - Valid numbers found: 1245
 * //     - Match rate: 0.003712%
 * //
 * // [*] OUTPUT FILES
 * //     - File 1: 'n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_A.json' (15234 bytes)
 * //     - File 2: 'n_values_for_mr_1821_on_range_1_to_2pow25_and_taxonomy_B.json' (8756 bytes)
 * 
 * // Example with mr=0:
 * // $ ./search_n_values_by_mr 20 0
 * // [... similar output ...]
 * // [*] OUTPUT FILES
 * //     - File 1: 'n_values_for_mr_0_on_range_1_to_2pow20_and_taxonomy_A.json' (54321 bytes)
 * //     - WARNING: Ignoring unexpected file 'n_values_for_mr_0_on_range_1_to_2pow20_and_taxonomy_B.json' (mr=0 only produces taxonomy A)
 * //     ^ (if stale file from previous run exists)
 * 
 * // Example with checkpoint resume:
 * // $ ./search_n_values_by_mr 30 3643
 * // [... setup ...]
 * // [*] CHECKPOINT LOADED
 * //     - Last processed: n = 500000000
 * //     - Resuming from: n = 500000001
 * //     - Completed: 46.5661287308% | Remaining: 53.4338712692%
 * // [*] SEARCH PROCESS
 * //     - Resuming from n=500000001
 * //     [... continues from checkpoint ...]
 * ```
 */
int main(int argc, char* argv[]) {
    int exponent;
    uint64_t max_n, target_mr;

    if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n, &target_mr)) {
        return 1;
    }
    
    signal(SIGINT, checkpoint_signal_handler);

    const dictionary_entry_t* dict_entry = lookup_pseudocycle(target_mr);
    if (!dict_entry) {
        printf("\n[*] ERROR: mr=%lu not found in the dictionary\n", target_mr);
        return 1;
    }

    printf("\n[*] ALGORITHM SETUP\n");
    printf("\t - Using %d threads\n", omp_get_max_threads());
    printf("\t - Exploring range from 1 to 2^%d - 1 = %lu\n", exponent, max_n - 1);
    printf("\t - Target mr: %lu\n", target_mr);
    printf("\t - Pseudocycle: [%lu, %lu]\n", dict_entry->ni, dict_entry->nj);
    fflush(stdout);
    
    streaming_writer_t* writer = create_streaming_writer(exponent, target_mr);
    
    search_context_t ctx = {
        .max_n = max_n,
        .target_mr = target_mr,
        .exponent = exponent,
        .dict_entry = dict_entry,
        .writer = writer,
        .progress = create_progress_tracker(0),
        .start_time = omp_get_wtime()
    };
    
    uint64_t start_n = 1;
    load_checkpoint(&ctx, &start_n);
    
    uint64_t found_count = 0;
    uint64_t processed_count = 0;
    
    execute_parallel_search(&ctx, start_n, &found_count, &processed_count);
    
    double total_time = omp_get_wtime() - ctx.start_time;
    
    printf("\n[*] SEARCH RESULTS\n");
    printf("\t - Total numbers processed: %lu\n", processed_count);
    printf("\t - Total time: %.3f seconds\n", total_time);
    printf("\t - Speed: %.2f numbers/second\n", (double)processed_count / total_time);
    printf("\t - Valid numbers found: %lu\n", found_count);
    if (processed_count > 0) {
        printf("\t - Match rate: %.6f%%\n", (double)found_count / processed_count * 100.0);
    }
    fflush(stdout);
    
    cleanup_search_context(&ctx);
   
    char filename_A[256], filename_B[256], filename_C[256];
    
    snprintf(filename_A, sizeof(filename_A), "n_values_for_mr_%lu_on_range_1_to_2pow%d_and_taxonomy_A.json", target_mr, exponent);
    snprintf(filename_B, sizeof(filename_B), "n_values_for_mr_%lu_on_range_1_to_2pow%d_and_taxonomy_B.json", target_mr, exponent);
    snprintf(filename_C, sizeof(filename_C), "n_values_for_mr_%lu_on_range_1_to_2pow%d_and_taxonomy_C.json", target_mr, exponent);
    
    
    FILE* test_A = fopen(filename_A, "r");
    FILE* test_B = fopen(filename_B, "r");
    FILE* test_C = fopen(filename_C, "r");
       

    if (test_A) {
        fseek(test_A, 0, SEEK_END);
        long size = ftell(test_A);
        fclose(test_A);
        printf("\t - File 1: '%s' (%ld bytes)\n", filename_A, size);
    }

    if (target_mr > 0) {
        if (test_B) {
            fseek(test_B, 0, SEEK_END);
            long size = ftell(test_B);
            fclose(test_B);
            printf("\t - File 2: '%s' (%ld bytes)\n", filename_B, size);
        }
        
        if (test_C) {
            fseek(test_C, 0, SEEK_END);
            long size = ftell(test_C);
            fclose(test_C);
            printf("\t - File 2: '%s' (%ld bytes)\n", filename_C, size);
        }
    } else {  // mr=0 only produces taxonomy A - warn about unexpected B/C files
        if (test_B) {
            fclose(test_B);
            printf("\t - WARNING: Ignoring unexpected file '%s' (mr=0 only produces taxonomy A)\n", filename_B);
        }
        if (test_C) {
            fclose(test_C);
            printf("\t - WARNING: Ignoring unexpected file '%s' (mr=0 only produces taxonomy A)\n", filename_C);
        }
    }

    fflush(stdout);

    return 0;

}
