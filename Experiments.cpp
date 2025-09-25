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
SparseMatrix<double> matrix_formation(MatrixXd& filter, int mn)
{
    vector<Triplet<double>> triplets;
    triplets.reserve(mn * filter.size()); 

    VectorXd filter_v = Map<VectorXd>(filter.data(), filter.size());
    int center = filter.size() / 2;

    for (int i = -center; i < center; i++){
        int idx = i + center;
        int diagLen = mn - abs(i);

        if(diagLen > 0) {
            for(int k = 0; k < diagLen; k++){
                int row = (i >= 0) ? k : k-i;
                int col = row + i;
                triplets.emplace_back(row, col, filter_v(idx));
            }
            //VectorXd d = VectorXd::Constant(diagLen, filter_v(idx));
            //A.diagonal(i) = d;
        }
    }

    SparseMatrix<double> sparse(mn,mn);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    return sparse;
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

    // ==================================REQUEST NUMBER 1 ====================================
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

    // PART 3
    // Reshape images into a nm x 1 vector
    VectorXd v = Map<VectorXd>(image_matrix.data(), image_matrix.size());
    VectorXd w = Map<VectorXd>(noisy_image.data(), noisy_image.size());
    std::cout << "Image vector size: " << v.size() << std::endl;
    std::cout << "Noisy image vector size: " << w.size() << "\n" << std::endl;
    std::cout << "Euclidean norm of v: " << v.norm() << std::endl;
    std::cout << "Euclidean norm of w: " << w.norm() << "\n" << std::endl; // not necessary
    
    // PART 4 
    // Smooth the image trough the smoothing kernel H_{av1}

    SparseMatrix<double> A1 = matrix_formation(H_av1, v.size());
    cout << "The number of nnz in A1 is " << A1.nonZeros() << endl;

    // PART 5 
    VectorXd smoothed_noisy_image = A1 * w;
    smoothed_noisy_image = smoothed_noisy_image.cwiseMax(0.0).cwiseMin(255.0);
    unsigned char* output_data_2 = new unsigned char[width * height];
    for (int i = 0; i < width*height; i++){
        output_data_2[i] = static_cast<unsigned char>(smoothed_noisy_image[i]);
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
    
    cout << "Process finished with no errors (so far...)" << endl;

    return 0;
}