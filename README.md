# CHALLENGE 1 NLA
 TODO: Write complete documentation

 ## PART 9
 LIS' `test1` output: 
 ```
 mpirun -n 4 ./test1 A2.mtx w.mtx sol.mtx hist.txt -i cg -tol 1.0e-14

number of processes = 4
matrix size = 228000 x 228000 (1138088 nonzero entries)

initial vector x      : all components set to 0
precision             : double
linear solver         : CG
preconditioner        : none
convergence condition : ||b-Ax||_2 <= 1.0e-14 * ||b-Ax_0||_2
matrix storage format : CSR
linear solver status  : normal end

CG: number of iterations = 65
CG:   double             = 65
CG:   quad               = 0
CG: elapsed time         = 1.664273e-01 sec.
CG:   preconditioner     = 9.953400e-03 sec.
CG:     matrix creation  = 2.580000e-07 sec.
CG:   linear solver      = 1.564739e-01 sec.
CG: relative residual    = 6.602933e-15

 ```
Before loading `sol.mtx` into Eigen, it is necessary to convert it from Vector-style MatrixMarket to Array-style MatrixMarket. This can be achieve through the following bash command: 
```
awk 'NR==1{print "%%MatrixMarket matrix array real general"} NR==2{print $1, "1"} NR>2{print $2}' sol.mtx > sol_eigen.mtx

```