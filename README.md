# Search n values by $m_r$

[![Research](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15546925-blue.svg)](https://doi.org/10.5281/zenodo.15546925)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GCC](https://img.shields.io/badge/GCC-7.0+%20Required-red.svg?style=flat&logo=gnu)](https://gcc.gnu.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-Parallel-yellow.svg?style=flat)](https://www.openmp.org/)

High-performance search engine for discovering all starting numbers that share a specific pseudocycle in Collatz sequences using Tuple-based Transform methodology.

## Mathematical Foundation

### Tuple-Based Transform

The reversible algorithm which transforms Collatz sequence values using the tuple $[p, f(p), m, q]$. Companion computational tools and step-by-step visualizations available [here](https://github.com/hhvvjj/a-new-algebraic-framework-for-the-collatz-conjecture). The complete theoretical framework is detailed in this [article](http://dx.doi.org/10.5281/zenodo.15546925)

### Reverse Pseudocycle Discovery

This tool, given a specific $m_r$ value, identifies **all starting numbers $n$** in a range that generate that particular pseudocycle pattern.

### Known $m_r$ Values

From comprehensive analysis of range $1$ to $2^{40} - 1$, the complete set of 42 known $m_r$ values is:
```
0, 1, 2, 3, 6, 7, 8, 9, 12, 16, 19, 25, 45, 53, 60, 79, 91, 121, 125, 141, 166, 188, 205, 243, 250, 324, 333, 432, 444, 487, 576, 592, 649, 667, 683, 865, 889, 1153, 1214, 1821, 2428, 3643
```

## Installation

### System Requirements

- GCC compiler with OpenMP support
- C99 standard support
- Minimum 4GB RAM (8GB+ recommended for large searches)

### Package Installation

#### Red Hat-based Systems (RHEL, CentOS, Fedora, Rocky Linux or AlmaLinux)
```bash
# RHEL/CentOS/Rocky/AlmaLinux 8+
sudo dnf groupinstall "Development Tools"

# RHEL/CentOS 7
sudo yum groupinstall "Development Tools"

# Fedora
sudo dnf groupinstall "Development Tools"

# Verify OpenMP support
gcc -fopenmp --version
```

#### Debian-based Systems (Ubuntu, Debian or Linux Mint)
```bash
# Ubuntu/Debian/Mint
sudo apt update
sudo apt install build-essential manpages-dev

# Verify OpenMP support
gcc -fopenmp --version
```

### Compilation

#### Obtain the source code
```bash
# Clone the repository
git clone https://github.com/hhvvjj/search-n-values-by-mr.git
cd search-n-values-by-mr
```

#### Build Commands
```bash
# Standard compilation
gcc -fopenmp -O3 -std=c99 -Wall -Wextra search_n_values_by_mr.c -o search_n_values_by_mr

# Debug build
gcc -fopenmp -O0 -g -std=c99 -Wall -Wextra -DDEBUG search_n_values_by_mr.c -o search_n_values_by_mr_debug

# Optimized build (recommended)
gcc -O3 -march=native -fopenmp search_n_values_by_mr.c -o search_n_values_by_mr
```

## Usage
```
./search_n_values_by_mr <exponent> <target_mr>
```

### Parameters

- **exponent**: Integer between 1 and 63, defines search range as [1, 2^exp)
- **target_mr**: The specific $m_r$ value to search for (non-negative integer)

### Examples
```bash
# Quick test (1.048.575 numbers)
./search_n_values_by_mr 20 25        # Range: [1, 2^20)

# Standard analysis (33.554.431 numbers)
./search_n_values_by_mr 25 25        # Range: [1, 2^25)

# Large scale (1.073.741.823 numbers)
./search_n_values_by_mr 30 25       # Range: [1, 2^30)

# Research scale (1.099.511.627.775 numbers)
./search_n_values_by_mr 40 25       # Range: [1, 2^40)
```

## Output

### Console Output
```
************************************************************************************
* High-performance search engine to find n values belonging to a specific mr class *
*                                                            Javier Hernandez 2026 *
************************************************************************************

[*] ALGORITHM SETUP
	 - Using 8 threads
	 - Exploring range from 1 to 2^25 - 1 = 33554431
	 - Target mr: 25
	 - Pseudocycle: [51, 52]

[*] SEARCH PROCESS
	 - Starting from n=1
	 - Progress: (0.0%) | Processed: 0 | Found: 0 | 0.0 nums/sec | ETA: -- min
		 - Match found: n = 51 (Taxonomy B) generates mr = 25
		 - Match found: n = 102 (Taxonomy B) generates mr = 25
		 - Match found: n = 204 (Taxonomy B) generates mr = 25
		 - Match found: n = 408 (Taxonomy A) generates mr = 25
		 - Match found: n = 816 (Taxonomy A) generates mr = 25
		 - Match found: n = 1632 (Taxonomy A) generates mr = 25
		 - Match found: n = 3264 (Taxonomy A) generates mr = 25
		 - Match found: n = 6528 (Taxonomy A) generates mr = 25
		 - Match found: n = 13056 (Taxonomy A) generates mr = 25
		 - Match found: n = 26112 (Taxonomy A) generates mr = 25
		 - Match found: n = 52224 (Taxonomy A) generates mr = 25
		 - Match found: n = 104448 (Taxonomy A) generates mr = 25
		 - Match found: n = 208896 (Taxonomy A) generates mr = 25
		 - Match found: n = 417792 (Taxonomy A) generates mr = 25
		 - Match found: n = 835584 (Taxonomy A) generates mr = 25
	 - Progress: (25.6180763245%) | Processed: 8596000 | Found 15 | 17897316.6 nums/sec | ETA: 0.0 min
		 - Match found: n = 1671168 (Taxonomy A) generates mr = 25
		 - Match found: n = 6684672 (Taxonomy A) generates mr = 25
		 - Match found: n = 13369344 (Taxonomy A) generates mr = 25
		 - Match found: n = 3342336 (Taxonomy A) generates mr = 25
		 - Match found: n = 26738688 (Taxonomy A) generates mr = 25
	 - Final checkpoint saved
	 - Checkpoint files cleaned...search completed

[*] SEARCH RESULTS
	 - Total numbers processed: 33554431
	 - Total time: 1.966 seconds
	 - Speed: 17071186.06 numbers/second
	 - Valid numbers found: 20
	 - Match rate: 0.000060%

[*] OUTPUT FILES
	 - File 1: 'n_values_for_mr_25_on_range_1_to_2pow_25_and_taxonomy_A.json' (128 bytes)
	 - File 2: 'n_values_for_mr_25_on_range_1_to_2pow_25_and_taxonomy_B.json' (14 bytes)
```

### JSON Output Format

Results are exported to a JSON file with the naming convention:

`n_values_for_mr_<target_mr>_on_range_1_to_2pow<exponent>_and_taxonomy_<taxonomy>.json`
```json
[975, 1950, 3900, 7800]
```

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{hernandez2026searchnvaluesbymr,
  title={Search n values by mr},
  author={Hernandez, Javier},
  year={2026},
  url={https://github.com/hhvvjj/search-n-values-by-mr},
  note={Implementation based on research DOI: 10.5281/zenodo.15546925}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](https://github.com/hhvvjj/search-n-values-by-mr/blob/main/LICENSE) file.

## Author

**Javier Hernández**  
Independent Researcher  
Email: 271314@pm.me  
GitHub: [@hhvvjj](https://github.com/hhvvjj)