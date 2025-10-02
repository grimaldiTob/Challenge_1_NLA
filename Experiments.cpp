#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/ArpackSupport>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdR;
using namespace std;


/* The idea behind the following function is to return an mn*mn Sparse matrix
    which has a maximum number of non-zero diagonal equal to the total entries
    of a filter matrix passed as reference in the function. */
SparseMatrix<double> matrix_formation(const MatrixXd& filter, int m, int n) {
    int mn = m * n;

    int fh = filter.rows();  // filter height
    int fw = filter.cols();  // filter width
    int h_center = fh / 2;   // kernel center (rows)
    int w_center = fw / 2;   // kernel center (cols)

    int nnz_filter = (filter.array() != 0.0).count(); // filter could have zeros so we get the number of non-zeros

    std::vector<Triplet<double>> triplets;
    triplets.reserve(nnz_filter * mn); // we reserve less space if there are zeros in filter --> even better if the filter gets big and sparse
    // we allocate just the space for the nnz elements in the filter *mn (could be updated even more knowing the number of indexes out of bound but I guess it's ok)

    auto idx = [n](int i, int j) { return i * n + j; }; // lambda function to convert 
    // 2D indices to 1D index (don't ask more than this, I found it on StackOverflow...)

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int row = idx(i, j); // retrieve row-major index from i and j (row and col indexes of the image)
            for (int di = 0; di < fh; ++di) {
                for (int dj = 0; dj < fw; ++dj) { // loop over the filter entries
                    double w = filter(di, dj);   // weight of the filter entry
                    if (w == 0.0) continue; // ignore the zero entries in the filter

                    int ni = i + (di - h_center); // corresponding row index in the image
                    int nj = j + (dj - w_center); // corresponding column index in the image

                    // zero-padding: skip if outside image
                    if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;

                    int col = idx(ni, nj); // retrieve column index in the mn x mn matrix
                    triplets.emplace_back(row, col, w); // add the entry to the triplet list
                }
            }
        }
    }

    SparseMatrix<double> A(mn, mn);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

/* Function that accepts an image vector as an argument and it saves an image into memory.
    It also accepts parameters of width and heigth and a string which will be the name of the
    image file. */
void save_image(const VectorXd& image_v, int width, int height, const string out_image_path){
    unsigned char* output_data= new unsigned char[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j; // row-major order
            output_data[index] = static_cast<unsigned char>(image_v[index]);
        }
    }
    if (stbi_write_png(out_image_path.c_str(), width, height, 1,
                     output_data, width) == 0) 
    {
        std::cerr << "Error: Could not save output image" << std::endl;
        delete[] output_data;
        return;
    }
    delete[] output_data;
}

int main(int argc, char* argv[]) 
{
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    // ================= Matrices and Constant definitions ===========================
    const char* input_image_path = argv[1];

    MatrixXdR H_av1(3,3);
    H_av1 << 1, 1, 0,
             1, 2, 1,
             0, 1, 1;
    H_av1 = (1.0/8.0) * H_av1;

    /*MatrixXd H_av2(5,5);
    H_av2(0,0)=0; H_av2(0,1)=1; H_av2(0,2)=2; H_av2(0,3)=1; H_av2(0,4)=0;
    H_av2(1,0)=1; H_av2(1,1)=2; H_av2(1,2)=4; H_av2(1,3)=2; H_av2(1,4)=1;
    H_av2(2,0)=2; H_av2(2,1)=4; H_av2(2,2)=8; H_av2(2,3)=4; H_av2(2,4)=2;
    H_av2(3,0)=1; H_av2(3,1)=2; H_av2(3,2)=4; H_av2(3,3)=2; H_av2(3,4)=1;
    H_av2(4,0)=0; H_av2(4,1)=1; H_av2(4,2)=2; H_av2(4,3)=1; H_av2(4,4)=0;
    H_av2 = (1.0/80.0) * H_av2;*/ // In the pdf, but not actually needed for the challenge

    MatrixXdR H_sh1(3,3);
    H_sh1 <<  0, -2,  0,
             -2,  9, -2,
              0, -2,  0;    


    MatrixXdR H_ed2(3,3);
    H_ed2 <<  -1, -2,  -1,
              0,  0, 0,
              1, 2,  1;    

    // PART 1
    //Load the image using stb_image
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) 
    {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }
    
    // Convert image data to Eigen matrix
    MatrixXdR image_matrix(height, width);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            int index = (i * width + j) * channels;  // linear index to acces the matrix
            image_matrix(i, j) = static_cast<unsigned char>(image_data[index]);
        }
    }
    stbi_image_free(image_data); // Free the image memory
    //std::cout << "Converted matrix:" << std::endl << image_matrix << std::endl;

    // ================================== REQUEST 1 ====================================
    std::cout << "Matrix size: " << image_matrix.rows() << "x" << image_matrix.cols() << std::endl;

    
    // ================================= REQUEST 2 ===================================

    // Adding noise to the image
    MatrixXdR noise = MatrixXdR::Random(image_matrix.rows(), image_matrix.cols()) * 40; // noise in range [-40, 40]

    // Sum and clamp to [0,255]
    MatrixXdR noisy_image = image_matrix + noise;
    noisy_image = noisy_image.cwiseMax(0.0).cwiseMin(255.0);
    VectorXd w = Map<VectorXd>(noisy_image.data(), noisy_image.size());
    save_image(w, width, height, "noisy_image.png");

    // ================================ REQUEST 3 ====================================

    // Reshape images into a nm x 1 vector in row-major order
    VectorXd v = Map<VectorXd>(image_matrix.data(), image_matrix.size());
    //VectorXd w = Map<VectorXd>(noisy_image.data(), noisy_image.size());
    std::cout << "Image vector size: " << v.size() << std::endl;
    std::cout << "Noisy image vector size: " << w.size() << "\n" << std::endl;
    std::cout << "Euclidean norm of v: " << v.norm() << std::endl;
    std::cout << "Euclidean norm of w: " << w.norm() << "\n" << std::endl; // not necessary
    
    // =============================== REQUEST 4 ==================================
    
    // Smooth the noisy image trough the smoothing kernel H_{av1}

    SparseMatrix<double> A1 = matrix_formation(H_av1, height, width);
    cout << "The number of nnz in A1 is " << A1.nonZeros() << endl << endl;

    // =============================== REQUEST 5 =================================

    VectorXd smoothed_noisy_image = A1 * w;
    smoothed_noisy_image = smoothed_noisy_image.cwiseMax(0.0).cwiseMin(255.0);
    
    save_image(smoothed_noisy_image, width, height, "smoothed_noisy_image.png");
    
    // =============================== REQUEST 6 =================================

    SparseMatrix<double> A2 = matrix_formation(H_sh1, height, width);

    bool A2_sym = A2.isApprox(A2.transpose());
    
    Eigen::SimplicialLLT<SparseMatrix<double>> chol(A2);
    cout << "A2 is SPD: " << ((A2_sym && chol.info() == Success) ? "Yes" : "No") << endl  << endl;
        
    cout << "The number of nnz in A2 is " << A2.nonZeros() << endl;

    // =============================== REQUEST 7 =================================

    VectorXd sharpened_original_image_2 = A2 * v; // sharpening the original image
    // Clamp values to [0, 255]
    sharpened_original_image_2 = sharpened_original_image_2.cwiseMax(0.0).cwiseMin(255.0);
    
    save_image(sharpened_original_image_2, width, height, "sharpened_original.png");

    // ================================= REQUEST 8 ================================
    // saveMarket(A2, "./A2.mtx"); method doesn't work
    // saveMarketVector(w.transpose(), "./w.mtx");

    int r = A2.rows(); 
    int c = A2.cols();
    int nnz = A2.nonZeros();
    FILE* outA2 = fopen("A2.mtx", "w");
    fprintf(outA2,"%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(outA2,"%d %d %d\n", r, c, nnz);
    for (int k=0; k<A2.outerSize(); ++k)
        for (SparseMatrix<double>::InnerIterator it(A2, k); it; ++it)
            fprintf(outA2, "%d %d %f\n", it.row(), it.col(), it.value());
    fclose(outA2);
    cout << "A2 matrix saved in A2.mtx" << endl;

    int n = w.size();
    FILE* outW = fopen("w.mtx", "w");
    fprintf(outW,"%%%%MatrixMarket matrix array real general\n");
    fprintf(outW,"%d %d\n", n, 1);
    for (int i=0; i<n; i++) {
        fprintf(outW,"%d %f\n", i, w(i)); // TODO: understand if the index should be i or i++
    }
    fclose(outW);
    cout << "w vector saved in w.mtx" << endl;

    // ================================== REQUEST 9 ================================

    /* Da quanto ho capito si dovrebbe prendere il sol.mtx che riceviamo da LIS e spostarlo
        nella cartella Challenge e a quel punto runnando sto codice ti salva anche la nuova immagine.
        Si potrebbe fare tipo un check dove magari se viene trovato un file sol.mtx allora esegue sta
        routine sennò la skippa. Penso si possa fare almeno finchè non troviamo una soluzione.*/


    //VectorXd x(A2.rows());
    //loadMarket(x, "./sol.mtx")
    
    //save_image(x, width, height, "immagine_a_caso.png");


    // ================================== REQUEST 10 ================================

    SparseMatrix<double> A3 = matrix_formation(H_ed2, height, width);

    bool A3_sym = A3.isApprox(A3.transpose());
    cout << "The number of nnz in A3 is " << A3.nonZeros() << endl;
    cout << "A3 is symmetric: " << ((A3_sym) ? "Yes" : "No") << endl  << endl;

    // =================================== REQUEST 11 ===============================

    VectorXd sobel_filter_image = A3*v;
    sobel_filter_image = sobel_filter_image.cwiseMax(0.0).cwiseMin(255.0);

    save_image(sobel_filter_image, width, height, "sobel_filtered_image.png");

    // ==================================== REQUEST 12 ==============================

    SparseMatrix<double> I_A3(A3.rows(), A3.cols());
    I_A3.setIdentity();
    I_A3 *= 3;
    I_A3 += A3;
    double tol = 1.e-8;
    int iterations;
    VectorXd y(I_A3.rows());

    Eigen::BiCGSTAB<SparseMatrix<double>> bicgstab;
    bicgstab.setTolerance(tol);
    bicgstab.setMaxIterations(1000); 
    
    bicgstab.compute(I_A3);
    y = bicgstab.solve(w);
    if (bicgstab.info() != Success) {
        std::cerr << "Solver did not converge!" << std::endl;
    }

    cout << "Relative residual is: " << bicgstab.error() << endl;
    cout << "#iterations: " << bicgstab.iterations() << endl;

    save_image(y, width, height, "y_image.png");    


    cout << "Process finished with no errors (so far...)" << endl;
    return 0;
}