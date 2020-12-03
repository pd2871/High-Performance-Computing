/**
 * Matrix (N*N) multiplication with multiple threads.
 
 * Compile the program as   cc -o matmul matrixmul.c -pthread 
 
 * For 1 thread use   ./matmul 1    (no. of threads must be divisible by matrix size which is 2000)
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

int size, num_threads;
int **matrix1, **matrix2, **matrix3;

int ** allocate_matrix( int size )
{
  /* Allocate 'size' * 'size' doubles contiguously. */
  int * vals = (int *) malloc( size * size * sizeof(int) );

  /* Allocate array of int* with size 'size' */
  int ** ptrs = (int **) malloc( size * sizeof(int*) );

  int i;
  for (i = 0; i < size; ++i) {
    ptrs[ i ] = &vals[ i * size ];
  }

  return ptrs;
}

void init_matrix( int **matrix, int size )
{
  int i, j;

  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      matrix[ i ][ j ] = rand() % 9;  //creates matrix of elements in range of 0 to 9
    }
  }
}

void print_matrix( int **matrix, int size )
{
  int i, j;

  for (i = 0; i < size; ++i) {
    for (j = 0; j < size-1; ++j) {
      printf( "%d, ", matrix[ i ][ j ] );
    }
    printf( "%d", matrix[ i ][ j ] );
    putchar( '\n' );
  }
}

void * matrix_multiplication( void *arg )
{
  int i, j, k, tid, portion_size, row_start, row_end;
  int sum;
  
  tid = *(int *)(arg); // get the thread ID assigned sequentially.
  portion_size = size / num_threads;
  row_start = tid * portion_size;
  row_end = (tid+1) * portion_size;

  for (i = row_start; i < row_end; ++i) { // hold row index of 'matrix1'
    for (j = 0; j < size; ++j) { // hold column index of 'matrix2'
      sum = 0;
      for (k = 0; k < size; ++k) { 
	sum += matrix1[ i ][ k ] * matrix2[ k ][ j ];
      }
      matrix3[ i ][ j ] = sum;
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


int main( int argc, char *argv[] )
{
  int i;
  int sum = 0;
  
  struct timespec start, finish;
  long long int time_elapsed;

  pthread_t * threads;

  num_threads = atoi( argv[1] );  //no. of threads is passed during run time
  size = 2000;  //2000*2000 sized matrix

  if ( size % num_threads != 0 ) {
    fprintf( stderr, "size %d must be a multiple of num of threads %d\n",
	     size, num_threads );
    return -1;
  }
  
  if ( num_threads <= 0 ) {
    fprintf(stderr, "Thread numbers must be greater 0 \n");
    return -1;
  }

  threads = (pthread_t *) malloc( num_threads * sizeof(pthread_t) );

  matrix1 = allocate_matrix( size );
  matrix2 = allocate_matrix( size );
  matrix3 = allocate_matrix( size );
  
  init_matrix( matrix1, size );
  init_matrix( matrix2, size );

  if ( size <= 10 ) {
    printf( "Matrix 1:\n" );
    print_matrix( matrix1, size );
    printf( "Matrix 2:\n" );
    print_matrix( matrix2, size );
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  
  for ( i = 0; i < num_threads; ++i ) {
    int *tid;
    tid = (int *) malloc( sizeof(int) );
    *tid = i;
    pthread_create( &threads[i], NULL, matrix_multiplication, (void *)tid );
  }

  for ( i = 0; i < num_threads; ++i ) {
    pthread_join( threads[i], NULL );
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  
  time_difference(&start, &finish, &time_elapsed);
  printf( "Number of threads: %d\n",
          num_threads);
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));

  return 0;
}
