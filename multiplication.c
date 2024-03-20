#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<time.h>
#define size 10

int main()
{
    MPI_Init(NULL,NULL); 
    int totalprocs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalprocs);
    srand(time(NULL));
    
    int matrix_a[size][size], matrix_b[size][size], product[size][size];
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            matrix_a[i][j] = rand() % 10;
            matrix_b[i][j] = rand() % 10;
        }
    }   

    int row_sum[size];
    if(rank != 0) {
        for(int j=0; j<size; j++) {
            row_sum[j] = 0;
            for(int k=0; k<size; ++k) {
                row_sum[j] += matrix_a[rank-1][k] * matrix_b[k][j];
            }
        }
        MPI_Send(row_sum, size, MPI_INT, 0, 1, MPI_COMM_WORLD);
    } else {
        for(int j=1; j<totalprocs; j++) {
            MPI_Recv(row_sum, size, MPI_INT, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int k=0; k<size; ++k)
                product[j-1][k] = row_sum[k];
        }
        printf("Product matrix:\n");
        for(int i=0; i<size; ++i) {
            for(int j=0; j<size; ++j)
                printf("%d ", product[i][j]);
            printf("\n");
        }
    }
    MPI_Finalize(); 
    return 0;
}

