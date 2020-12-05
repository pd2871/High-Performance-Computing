/*

- Matrix multiplication without any threads

- compile the program using cc -o improved_matmul improved_matmul.c 

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

int main()
{
	struct timespec start, finish;
        long long int time_elapsed;
		
 	N=2000;
 	P=2000;
 	M=2000;
 	
	int i, j, k;
	
	if ((N < 1) && (P<1) && (M<1)) {
          fprintf( stderr, "Matrix size must be greater than 0\n");
        return -1;
        }
         
        //memory allocation of matrices
 	A = allocate_matrix(N,P);
  	B = allocate_matrix(P,M);
  	C = allocate_matrix(N,M);
  
  	init_matrix(A, N, P);  //initialize the matrix A with numbers in range 0 to 9
  	init_matrix(B, P, M);  //initialize the matrix B with numbers in range 0 to 9
  	
  	if((N <= 10) && (P<=10) && (M<=10)) {
    	  printf( "\nMatrix A:\n" );
    	  print_matrix(A, N, P);
    	  printf( "Matrix B:\n" );
    	  print_matrix(B, P, M);
  	}
	
  	clock_gettime(CLOCK_MONOTONIC, &start);  //starting the timer
	
	for (int i = 0; i < N; i++){
          for (int k = 0; k < P; k++){
            for (int j = 0; j < M; j++) { // swapped order
              C[i][j] += A[i][k] * B[k][j];
            }
          }
        }
	
	clock_gettime(CLOCK_MONOTONIC, &finish);  //finishing the timer
	
	if((N <= 10) && (P<=10) && (M<=10)) {
    	  printf( "Matrix C:\n" );
    	  print_matrix(C, N, M);
  	}
  
  	time_difference(&start, &finish, &time_elapsed);

  	printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));

	return 0;
}

