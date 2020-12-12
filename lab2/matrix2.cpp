#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
int N = 2000;
#define ROOT 0

// Process = 4
// n = 20

double *A;
double *B;
double *C;

void PrintMatrix(double *matr, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%7.4f ", matr[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void RandInit(double *matr, int N, int value) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matr[i * N + j] = value;
        }
    }
}

void InitMatrices(double *A, double *B, double *C, int N, int a_value, int b_value) {
    srand((unsigned) time(NULL));
    RandInit(A, N, a_value);
    RandInit(B, N, b_value);
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0;
    }
}

void LocalMultiplication(double *a, double *b, double *c, int Process, int Rank, int IterNum, int Rows, int N) {
    int BlockNumInC = (Rank + IterNum) % Process;
    int Offset = BlockNumInC * Rows;
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Rows; ++j) {
            c[i * N + j + Offset] = 0.0;
            for (int k = 0; k < N; ++k) {
                c[i * N + j + Offset] += a[i * N + k] * b[j * N + k];
            }
        }
    }
}


int main(int argc, char *argv[]) {
    double start = MPI_Wtime();

    int Process, Rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &Process);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

    int MaxProcRank = Process - 1;

    int Rows = N / Process;
    int Elements = N * Rows;

    double *bufA = (double *) malloc(N * Rows * sizeof(double));
    double *bufB = (double *) malloc(N * Rows * sizeof(double));
    double *bufC = (double *) malloc(N * Rows * sizeof(double));

    MPI_Datatype COLUMN_TYPE;
    MPI_Datatype RESULT_COLUMN_TYPE;
    if (Rank == 0) {
        A = (double *) malloc(N * N * sizeof(double));
        B = (double *) malloc(N * N * sizeof(double));
        C = (double *) malloc(N * N * sizeof(double));

        InitMatrices(A, B, C, N, 1, 2);
        // printf("Matrix A\n");
        // PrintMatrix(A, N);
        // printf("Matrix B\n");
        // PrintMatrix(B, N);

        MPI_Type_vector(N, 1, N, MPI_DOUBLE, &COLUMN_TYPE);
        MPI_Type_commit(&COLUMN_TYPE);
        MPI_Type_create_resized(COLUMN_TYPE, 0, 1 * sizeof(double), &RESULT_COLUMN_TYPE);
        MPI_Type_commit(&RESULT_COLUMN_TYPE);
    }

    MPI_Scatter(A, Elements, MPI_DOUBLE, bufA, Elements, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(B, Rows, RESULT_COLUMN_TYPE, bufB, Elements, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    int CurrIterNum = 0;
    LocalMultiplication(bufA, bufB, bufC, Process, Rank, CurrIterNum, Rows, N);

    int NextProcRank = Rank == MaxProcRank ? ROOT : Rank + 1;
    int PrevProcRank = Rank == ROOT ? MaxProcRank : Rank - 1;

    MPI_Status Status;

    int TotalIterCount = Process;
    for (CurrIterNum = 1; CurrIterNum < TotalIterCount; CurrIterNum++) {
        MPI_Sendrecv_replace(bufB, Elements, MPI_DOUBLE, NextProcRank, 0, PrevProcRank, 0, MPI_COMM_WORLD,
                             &Status);
        LocalMultiplication(bufA, bufB, bufC, Process, Rank, CurrIterNum, Rows, N);
    }

    MPI_Gather(bufC, Elements, MPI_DOUBLE, C, Elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(bufA);
    free(bufB);
    free(bufC);

    if (Rank == 0) {
        fprintf(stdout,"Time = %.6f\n\n", MPI_Wtime()-start);
        // printf("Matrix C\n");
        // PrintMatrix(C, N);
        // printf("TEST\n");
        
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}