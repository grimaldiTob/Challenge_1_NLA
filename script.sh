mpirun -n 4 ./test1 A2.mtx w.mtx sol.mtx hist.txt -i cg -tol 1.0e-14;
cp sol.mtx ../../../Challenge_1-20250922
