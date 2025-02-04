import main
import time

start_vanilla = time.time()
main.prime_finder_vanilla(40000)
end_vanilla = time.time()
print(end_vanilla - start_vanilla)

start_cython = time.time()
main.prime_finder_optimized(40000)
end_cython = time.time()
print(end_cython - start_cython)
