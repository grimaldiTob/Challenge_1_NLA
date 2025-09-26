#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
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

int main(int argc, char* argv[]) 
{
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    // ================= Matrices and Constant definitions ===========================
    const char* input_image_path = argv[1];

    MatrixXd H_av1(3,3);
    H_av1 << 1, 1, 0,
             1, 2, 1,
             0, 1, 1;
    H_av1 = (1.0/8.0) * H_av1;

    MatrixXd H_av2(5,5);
    H_av2(0,0)=0; H_av2(0,1)=1; H_av2(0,2)=2; H_av2(0,3)=1; H_av2(0,4)=0;
    H_av2(1,0)=1; H_av2(1,1)=2; H_av2(1,2)=4; H_av2(1,3)=2; H_av2(1,4)=1;
    H_av2(2,0)=2; H_av2(2,1)=4; H_av2(2,2)=8; H_av2(2,3)=4; H_av2(2,4)=2;
    H_av2(3,0)=1; H_av2(3,1)=2; H_av2(3,2)=4; H_av2(3,3)=2; H_av2(3,4)=1;
    H_av2(4,0)=0; H_av2(4,1)=1; H_av2(4,2)=2; H_av2(4,3)=1; H_av2(4,4)=0;
    H_av2 = (1.0/80.0) * H_av2;


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
    MatrixXd image_matrix(height, width);
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

    // ================================== REQUEST NUMBER 1 ====================================
    std::cout << "Matrix size: " << image_matrix.rows() << "x" << image_matrix.cols() << std::endl;

    // PART 2
    // Adding noise to the image
    MatrixXd noise = MatrixXd::Random(image_matrix.rows(), image_matrix.cols()) * 40; // noise in range [-40, 40]

    // Sum and clamp to [0,255]
    MatrixXd noisy_image = image_matrix + noise;
    noisy_image = noisy_image.cwiseMax(0.0).cwiseMin(255.0);

    // Convert back to unsigned char for saving
    unsigned char* output_data = new unsigned char[width * height];
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int index = i * width + j;
            output_data[index] = static_cast<unsigned char>(noisy_image(i, j));
        }
    }

    // ================================= REQUEST NUMBER 2 ===================================
    // Save the noisy image
    const std::string output_image_path2 = "noisy.png";
    if (stbi_write_png(output_image_path2.c_str(), width, height, 1,
                     output_data, width) == 0) 
    {
        std::cerr << "Error: Could not save output image" << std::endl;
        delete[] output_data;
        return 1;
    }
    delete[] output_data;

    // ================================ REQUEST NUMBER 3 ====================================
    // PART 3
    // Reshape images into a nm x 1 vector in row-major order
    VectorXd v = Map<VectorXd>(image_matrix.data(), image_matrix.size());
    VectorXd w = Map<VectorXd>(noisy_image.data(), noisy_image.size());
    std::cout << "Image vector size: " << v.size() << std::endl;
    std::cout << "Noisy image vector size: " << w.size() << "\n" << std::endl;
    std::cout << "Euclidean norm of v: " << v.norm() << std::endl;
    std::cout << "Euclidean norm of w: " << w.norm() << "\n" << std::endl; // not necessary
    
    // =============================== REQUEST NUMBER 4 ==================================
    
    // PART 4 
    // Smooth the image trough the smoothing kernel H_{av1}

    SparseMatrix<double> A1 = matrix_formation(H_av1, height, width);
    cout << "The number of nnz in A1 is " << A1.nonZeros() << endl << endl;

    // PART 5 
    VectorXd smoothed_noisy_image = A1 * w;
    smoothed_noisy_image = smoothed_noisy_image.cwiseMax(0.0).cwiseMin(255.0);
    unsigned char* output_data_2 = new unsigned char[width * height];
    for (int j = 0; j < width; j++) {
        output_data_2[j] = static_cast<unsigned char>(smoothed_noisy_image[j]);
    }
    const std::string output_image_path3 = "smoothed_noisy_2.png";
    if (stbi_write_png(output_image_path3.c_str(), width, height, 1,
                     output_data_2, width) == 0) 
    {
        std::cerr << "Error: Could not save output image" << std::endl;
        delete[] output_data_2;
        return 1;
    }
    delete[] output_data_2;
    
    // PART 6 

    SparseMatrix<double> A2 = matrix_formation(H_av2, height, width);
    cout << "The number of nnz in A2 is " << A2.nonZeros() << endl;
    cout << "A2 is symmetric: " << (A2.isApprox(A2.transpose()) ? "Yes" : "No") << endl  << endl;

    // PART 7

    VectorXd smoothed_noisy_image_2 = A2 * w;
    smoothed_noisy_image_2 = smoothed_noisy_image_2.cwiseMax(0.0).cwiseMin(255.0);
    unsigned char* output_data_3 = new unsigned char[width * height];
    for (int j = 0; j < width; j++) {
        output_data_3[j] = static_cast<unsigned char>(smoothed_noisy_image_2[j]);
    }
    const std::string output_image_path4 = "smoothed_noisy_3.png";
    if (stbi_write_png(output_image_path4.c_str(), width, height, 1,
                     output_data_3, width) == 0) 
    {
        std::cerr << "Error: Could not save output image" << std::endl;
        delete[] output_data_3;
        return 1;
    }
    delete[] output_data_3;

    // PART 8
    
    cout << "Process finished with no errors (so far...)" << endl;
    
    return 0;
}