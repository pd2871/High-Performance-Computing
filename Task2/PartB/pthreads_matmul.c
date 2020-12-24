/**
 * Matrix (2000*2000) multiplication with multiple threads.
 
 * Compile the program as   cc -o pthreads_matmul pthreads_matmul.c -pthread 
 
 * For 2 threads use   ./pthreads_matmul 2    (no. of threads must equally divide the rows of matrix A which is 2000)
 
 * Threads will run parallely for different rows of matrix A. For eg:
   Suppose matrix A and matrix B are of size 2000*2000 and 2 threads are being used then 
   1st thread will use 1000 rows of matrix A and 2nd thread will use remaining 1000 rows of matrix A
   All the columns of matrix A are used by those threads for multiplication with matrix B.

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

int num_threads;
int **A, **B, **C;
int N,P,M;

int ** allocate_matrix(int row, int column)
{
  int **mat = (int **)malloc(row * sizeof(int*));
  for(int i = 0; i < row; i++) {
    mat[i] = (int *)malloc(column * sizeof(int));
  }
  return mat;
}


void init_matrix( int **matrix, int row, int column )
{
  int i, j;

  for (i = 0; i < row; i++) {
    for (j = 0; j < column; j++) {
      matrix[ i ][ j ] = rand() % 10;  //creates matrix of elements in range of 0 to 9
    }
  }
}


void print_matrix( int **matrix, int row, int column )
{
  int i, j;

  for (i = 0; i < row; i++) {
    for (j = 0; j < column-1; j++) {
      printf( "%d, ", matrix[ i ][ j ] );
    }
    printf( "%d", matrix[ i ][ j ] );
    putchar( '\n' );
  }
  putchar('\n');
}


void * matrix_multiplication( void *arg )
{
  int i, j, k, tid, portion, row_start, row_end;
  
  tid = *(int *)(arg); // get the thread ID assigned sequentially.
  portion = N / num_threads;   //making portions for matrix A
  row_start = tid * portion;
  row_end = (tid+1) * portion;

  for (i=row_start; i<row_end; i++) { // hold row index of 'matrixA'
    for (j=0; j<M; j++) { // hold column index of 'matrixB'
      C[i][j] = 0;
      for (k=0; k<P; k++) { 
	C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
 
}


//Calculating time

int time_difference(struct timespec *start, struct timespec *finish, long long int *difference)
{
    long long int ds = finish->tv_sec - start->tv_sec;
    long long int dn = finish->tv_nsec - start->tv_nsec;
    if (dn < 0)
    {
	ds--;
	dn += 1000000000;
    }
    *difference = ds *1000000000 + dn;
    return !(*difference > 0);
}


int main(int argc, char *argv[])
{
  int i;
  
  struct timespec start, finish;
  long long int time_elapsed;

  pthread_t * threads;

  num_threads = atoi(argv[1]);  //no. of threads is passed during run time
  
  N = 2000;  //row for matrix A and C
  P = 2000;  //row for matrix B and column of matrix A
  M = 2000;  //Column for matrix B and C


  //no. of threads must divide the rows of Matrix A
  if (N % num_threads != 0) {
    fprintf( stderr, "N = %d must be a multiple of num of threads = %d\n",
	     N, num_threads );
    return -1;
  }
  
  if (num_threads <= 0) {
    fprintf(stderr, "Thread numbers must be greater 0 \n");
    return -1;
  }

  threads = (pthread_t *) malloc( num_threads * sizeof(pthread_t) );

  //memory allocation of matrices
  A = allocate_matrix(N,P);
  B = allocate_matrix(P,M);
  C = allocate_matrix(N,M);
  
  init_matrix(A, N, P);  //initialize the matrix A with numbers in range 0 to 9
  init_matrix(B, P, M);  //initialize the matrix B with numbers in range 0 to 9
  
  if ((N <= 10) && (P<=10) && (M<=10)) {
    printf( "\nMatrix A:\n" );
    print_matrix(A, N, P);
    printf( "Matrix B:\n" );
    print_matrix(B, P, M);
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  
  for (i = 0; i < num_threads; i++) {
    int *tid;
    tid = (int *) malloc( sizeof(int) );
    *tid = i;
    pthread_create( &threads[i], NULL, matrix_multiplication, (void *)tid );  //creating threads
  }

  for (i = 0; i < num_threads; i++) {
    pthread_join( threads[i], NULL );
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  
  if ((N <= 10) && (P<=10) && (M<=10)) {
    printf( "Matrix C:\n" );
    print_matrix(C, N, M);
  }
  
  time_difference(&start, &finish, &time_elapsed);
  printf( "Number of threads: %d\n",
          num_threads);
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));

  return 0;
}

