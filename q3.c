#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>
#include<omp.h>

int main()
{
    int matrix1_128[128][128], matrix2_128[128][128], product_128[128][128];
    int matrix1_256[256][256], matrix2_256[256][256], product_256[256][256];
    FILE *ptr;
    
    // Read data from file for the first matrix (128x128)
    ptr = fopen("128.txt","r");
    for(int i=0; i<128; i++) {
        for(int j=0; j<128; j++) {
            int temp;
            fscanf(ptr, "%d", &temp);
            char t = fgetc(ptr);
            matrix1_128[i][j] = temp;
            matrix2_128[i][j] = temp;
            product_128[i][j] = 0;
        }
    }
    fclose(ptr);

    // Perform matrix multiplication for the first matrix (128x128)
    double time_128 = omp_get_wtime();
    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            for(int i=0; i<128/4; ++i) {
                for(int j=0; j<128/4; ++j) {
                    for(int k=0; k<128/4; ++k)
                        product_128[i][j] += matrix1_128[i][k] * matrix2_128[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=128/4; i<128/2; ++i) {
                for(int j=0; j<128/2; ++j) {
                    for(int k=0; k<128; ++k)
                        product_128[i][j] += matrix1_128[i][k] * matrix2_128[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=128/2; i<3*128/4; ++i) {
                for(int j=0; j<128; ++j) {
                    for(int k=0; k<128; ++k)
                        product_128[i][j] += matrix1_128[i][k] * matrix2_128[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=3*128/4; i<128; ++i) {
                for(int j=0; j<128; ++j) {
                    for(int k=0; k<128; ++k)
                        product_128[i][j] += matrix1_128[i][k] * matrix2_128[k][j];
                }
            }
        }
    }
    printf("Time taken for 128x128 matrix: %lf\n", omp_get_wtime()-time_128);

    // Read data from file for the second matrix (256x256)
    ptr = fopen("256.txt","r");
    for(int i=0; i<256; i++) {
        for(int j=0; j<256; j++) {
            int temp;
            fscanf(ptr, "%d", &temp);
            char t = fgetc(ptr);
            matrix1_256[i][j] = temp;
            matrix2_256[i][j] = temp;
            product_256[i][j] = 0;
        }
    }
    fclose(ptr);

    // Perform matrix multiplication for the second matrix (256x256)
    double time_256 = omp_get_wtime();
    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            for(int i=0; i<256/4; ++i) {
                for(int j=0; j<256; ++j) {
                    for(int k=0; k<256; ++k)
                        product_256[i][j] += matrix1_256[i][k] * matrix2_256[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=256/4; i<256/2; ++i) {
                for(int j=0; j<256; ++j) {
                    for(int k=0; k<256; ++k)
                        product_256[i][j] += matrix1_256[i][k] * matrix2_256[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=256/2; i<3*256/4; ++i) {
                for(int j=0; j<256; ++j) {
                    for(int k=0; k<256; ++k)
                        product_256[i][j] += matrix1_256[i][k] * matrix2_256[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=3*256/4; i<256; ++i) {
                for(int j=0; j<256; ++j) {
                    for(int k=0; k<256; ++k)
                        product_256[i][j] += matrix1_256[i][k] * matrix2_256[k][j];
                }
            }
        }
    }
    printf("Time taken for 256x256 matrix: %lf\n", omp_get_wtime()-time_256);

    return 0;
}

