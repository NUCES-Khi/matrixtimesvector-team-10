#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>
#include<omp.h>

int main()
{
    // Declaration of variables
    int matrix1[128][128], matrix2[128][128], product[128][128];
    FILE *file_ptr;
    double start_time;

    // Read data from file for the first matrix (128x128)
    file_ptr = fopen("matrix_128.txt","r");
    for(int i=0;i<128;i++){
        for(int j=0;j<128;j++){
            int temp;
            fscanf(file_ptr,"%d",&temp);
            char t = fgetc(file_ptr);
            matrix1[i][j]=temp;
            matrix2[i][j]=temp;
            product[i][j]=0;         
        }
    }
    fclose(file_ptr);

    // Matrix multiplication for the first matrix (128x128)
    start_time = omp_get_wtime(); // Start time measurement
    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            for(int i=0;i<128/4;++i)
            {
                for(int j=0;j<128/4;++j)
                {
                    for(int k=0;k<128/4;++k)
                        product[i][j] += matrix1[i][k]*matrix2[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=(128/4);i<128/2;++i)
            {
                for(int j=0;j<128/2;++j)
                {
                    for(int k=0;k<128;++k)
                        product[i][j] += matrix1[i][k]*matrix2[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=(128/2);i<3*128/4;++i)
            {
                for(int j=0;j<128;++j)
                {
                    for(int k=0;k<128;++k)
                        product[i][j] += matrix1[i][k]*matrix2[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=3*128/4;i<128;++i)
            {
                for(int j=0;j<128;++j)
                {
                    for(int k=0;k<128;++k)
                        product[i][j] += matrix1[i][k]*matrix2[k][j];
                }
            }
        }   
    }
    printf("Time taken for 128x128 matrix: %lf seconds\n", omp_get_wtime()-start_time); // Print execution time

    // Read data from file for the second matrix (256x256)
    file_ptr = fopen("matrix_256.txt","r");
    int matrix3[256][256], matrix4[256][256], product2[256][256];
    for(int i=0;i<256;i++){
        for(int j=0;j<256;j++){
            int temp;
            fscanf(file_ptr,"%d",&temp);
            char t = fgetc(file_ptr);
            matrix3[i][j]=temp;
            matrix4[i][j]=temp;
            product2[i][j]=0;
        }
    }
    fclose(file_ptr);

    // Matrix multiplication for the second matrix (256x256)
    start_time = omp_get_wtime(); // Start time measurement
    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            for(int i=0;i<256/4;++i)
            {
                for(int j=0;j<256;++j)
                {
                    for(int k=0;k<256;++k)
                        product2[i][j] += matrix3[i][k]*matrix4[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=(256/4);i<256/2;++i)
            {
                for(int j=0;j<256;++j)
                {
                    for(int k=0;k<256;++k)
                        product2[i][j] += matrix3[i][k]*matrix4[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=(256/2);i<3*256/4;++i)
            {
                for(int j=0;j<256;++j)
                {
                    for(int k=0;k<256;++k)
                        product2[i][j] += matrix3[i][k]*matrix4[k][j];
                }
            }
        }
        #pragma omp section
        {
            for(int i=3*256/4;i<256;++i)
            {
                for(int j=0;j<256;++j)
                {
                    for(int k=0;k<256;++k)
                        product2[i][j] += matrix3[i][k]*matrix4[k][j];
                }
            }
        }   
    }
    printf("Time taken for 256x256 matrix: %lf seconds\n", omp_get_wtime()-start_time); // Print execution time

    return 0;
}

