#include <pthread.h>
#include <stdio.h>

/*

    Compile using   cc -o rd RemoveDependencies.c

*/


int main() { 

  int A;
  int B;
  int C;
  int D;
  int Btemp;

  
  printf("Enter A value\n");
  scanf("%d", &A);
  
  printf("\nEnter D value\n");
  scanf("%d", &D);
  
  //removing output dependency by using temporary variable
  Btemp = A + D;
  C = Btemp + D;
  B = C + D;
  
  printf("\nThe value of B is %d\n", B);
  printf("The value of C is %d\n", C);
  
  return 0;
  
}
