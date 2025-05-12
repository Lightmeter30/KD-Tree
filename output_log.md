# output

test data: 4D $10^4$

```shell
Performing performance analysis for 4D data...
function build_tree took 2.318812 seconds

Performance Analysis Results:
--------------------------------------------------

Nearest Neighbor Search:
KD-Tree avg time: 0.006747 seconds
Brute Force avg time: 0.257286 seconds
Speedup: 38.13x
Correctness: 100/100 (100.0%)

K-Nearest Neighbors Search:
KD-Tree avg time: 0.008567 seconds
Brute Force avg time: 0.267855 seconds
Speedup: 31.27x
Correctness: 100/100 (100.0%)

Range Search:
KD-Tree avg time: 0.000536 seconds
Brute Force avg time: 0.217947 seconds
Speedup: 406.53x
Correctness: 100/100 (100.0%)
```

test data:10D $10^5$
```shell
Performing performance analysis for 10D data...
function build_tree took 4.253347 seconds

Performance Analysis Results:
--------------------------------------------------

Nearest Neighbor Search:
KD-Tree avg time: 0.348755 seconds
Brute Force avg time: 0.550117 seconds
Speedup: 1.58x
Correctness: 100/100 (100.0%)

K-Nearest Neighbors Search:
KD-Tree avg time: 0.451607 seconds
Brute Force avg time: 0.574066 seconds
Speedup: 1.27x
Correctness: 100/100 (100.0%)

Range Search:
KD-Tree avg time: 0.000280 seconds
Brute Force avg time: 0.410730 seconds
Speedup: 1467.62x
Correctness: 100/100 (100.0%)
```