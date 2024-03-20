#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

void generate_data_files(int matrix_size) {
    srand(time(NULL));
    char filename[10];
    sprintf(filename, "%d.txt", matrix_size);
    FILE *ptr = fopen(filename, "w");
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        fprintf(ptr, "%d ", rand() % 100);
    }
    fclose(ptr);
}

void read_data(int matrix_size, int input_matrix[][matrix_size], const char *filename) {
    FILE *ptr = fopen(filename, "r");
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            fscanf(ptr, "%d", &input_matrix[i][j]);
        }
    }
    fclose(ptr);
}

void multiply_matrices(int matrix_size, int input_matrix1[][matrix_size], int input_matrix2[][matrix_size], int output_matrix[][matrix_size]) {
    double start_time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            int sum = 0;
            for (int k = 0; k < matrix_size; k++) {
                sum += input_matrix1[i][k] * input_matrix2[k][j];
            }
            output_matrix[i][j] = sum;
        }
    }
    double time_taken = omp_get_wtime() - start_time;
    printf("Time taken for %dx%d matrix multiplication: %lf seconds\n", matrix_size, matrix_size, time_taken);
}

int main() {
    const int matrix_size1 = 128;
    const int matrix_size2 = 256;

    generate_data_files(matrix_size1);
    generate_data_files(matrix_size2);

    int matrix_a[matrix_size1][matrix_size1], matrix_b[matrix_size1][matrix_size1], product_ab[matrix_size1][matrix_size1];
    read_data(matrix_size1, matrix_a, "128.txt");
    read_data(matrix_size1, matrix_b, "128.txt"); // Assuming matrices are square for multiplication

    multiply_matrices(matrix_size1, matrix_a, matrix_b, product_ab);

    int matrix_c[matrix_size2][matrix_size2], matrix_d[matrix_size2][matrix_size2], product_cd[matrix_size2][matrix_size2];
    read_data(matrix_size2, matrix_c, "256.txt");
    read_data(matrix_size2, matrix_d, "256.txt"); // Assuming matrices are square for multiplication

    multiply_matrices(matrix_size2, matrix_c, matrix_d, product_cd);

    return 0;
}
