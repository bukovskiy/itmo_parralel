#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#define N               1000       /* number of rows and columns in matrix */

MPI_Status status;

double a[N][N];
double b[N][N];
double c[N][N];

int sequential_matmul(){
    
    int i,j,k;
    double start = MPI_Wtime();

    for(i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 1.0;
        b[i][j]= 2.0;
      }
    }

    
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++){
            c[i][j] = 0;
            for (k=0; k<N; k++){
                c[i][j] += a[i][k]*b[k][j];
            }   
        }
    }

    // printf("Sequential result matrix:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", c[i][j]);
    //   printf ("\n");
    // }

    

    fprintf(stdout,"Sequential Time = %.6f\n\n", MPI_Wtime()-start);
    return 0;
}

int mpi_matmul(int argc, char **argv)
{
  int numtasks,taskid,numworkers,source,dest,rows,offset,i,j,k;

  double start = MPI_Wtime();

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  numworkers = numtasks-1;

  /*---------------------------- master ----------------------------*/
  if (taskid == 0) {
    //printf("%d", numtasks);
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 1.0;
        b[i][j]= 2.0;
      }
    }

    // printf("Here is the start matrix a:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", a[i][j]);
    //   printf ("\n");
    // }

    //  printf("Here is the start matrix b:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", b[i][j]);
    //   printf ("\n");
    // }

    /* send matrix data to the worker tasks */
    rows = N/numworkers;
    offset = 0;

    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b, N*N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      offset = offset + rows;
    }

    /* wait for results from all worker tasks */
    for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }


    // printf("Here is the result matrix:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", c[i][j]);
    //   printf ("\n");
    // }

    fprintf(stdout,"Time = %.6f\n\n", MPI_Wtime()-start);

  }

  /*---------------------------- worker----------------------------*/
  if (taskid > 0) {
    source = 0;
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

    /* Matrix multiplication */
    for (k=0; k<N; k++)
      for (i=0; i<rows; i++) {
        c[i][k] = 0.0;
        for (j=0; j<N; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
          //printf("rank is %d, c is %f", taskid, c[i][k]);
      }


    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}

int mpi_matmul_column(int argc, char **argv)
{
  int numtasks,taskid,numworkers,source,dest,rows,columns,offset,i,j,k;

  double start = MPI_Wtime();

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Datatype COLUMN_TYPE;
  MPI_Datatype RESULT_COLUMN_TYPE;
  MPI_Type_vector(N, 1, N, MPI_DOUBLE, &COLUMN_TYPE);
  MPI_Type_commit(&COLUMN_TYPE);
  MPI_Type_create_resized(COLUMN_TYPE, 0, 1 * sizeof(double), &RESULT_COLUMN_TYPE);
  MPI_Type_commit(&RESULT_COLUMN_TYPE);


  numworkers = numtasks-1;

  /*---------------------------- master ----------------------------*/
  if (taskid == 0) {
    //printf("%d", numtasks);
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 1.0;
        b[i][j]= 2.0;
      }
    }


    // int a[N][N]={{1,2,3,4},{5,6,7,8},{9,1,2,3},{4,5,6,7,}};
    // int b[N][N]={{1,2,3,4},{5,6,7,8},{9,1,2,3},{4,5,6,7,}};

    // printf("Here is the start matrix a:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", a[i][j]);
    //   printf ("\n");
    // }

    //  printf("Here is the start matrix b:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", b[i][j]);
    //   printf ("\n");
    // }

    /* send matrix data to the worker tasks */
    rows = N/numworkers;
    columns = N/numworkers;
    offset = 0;

    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b[0][0], N, RESULT_COLUMN_TYPE, dest, 1, MPI_COMM_WORLD);
      offset = offset + rows;
    }

    /* wait for results from all worker tasks */
    for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }


    // printf("Parallel 2result matrix:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", c[i][j]);
    //   printf ("\n");
    // }

    fprintf(stdout,"Time = %.6f\n\n", MPI_Wtime()-start);

  }

  /*---------------------------- worker----------------------------*/
  if (taskid > 0) {
    source = 0;
    double d[N][N];
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b[0][0], N, RESULT_COLUMN_TYPE, source, 1, MPI_COMM_WORLD, &status);

    // if (taskid==1){
    // for(int i=0;i<N;i++){
    //     for (int j=0; j<N; j++){
    //         printf("%6.2f", b[i][j]);
    //         }
    //     printf("\n");
    //     }

    // for(int i=0;i<rows;i++){
    //     for (int j=0; j<N; j++){
    //         printf("%6.2f", a[i][j]);
    //         }
    //     printf("\n");
    //     }
    //}
    //   printf("\n");
    /* Matrix multiplication */
    for (k=0; k<N; k++)
      for (i=0; i<rows; i++) {
        c[i][k] = 0.0;
        for (j=0; j<N; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
          //printf("rank is %d, c is %f\n", taskid, b[j][k]);
      }


    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}

int main(int argc, char **argv){

mpi_matmul(argc, argv);
sequential_matmul();
//mpi_matmul_column(argc,argv);

return 0;
}