#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

/*

   Compile using   cc -o 2threads 2threads.c -lcrypt -pthread

*/

//multithread password cracking

pthread_mutex_t lock; 

void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

void mainThread()
{
    pthread_t ti, tii;
    
    void *kernel_function_1();
    void *kernel_function_2();
    
    pthread_create(&ti, NULL, kernel_function_1,      
    "$6$AS$8pcKmkYTPSlXF7rrQWymrzSf0msE12EZaavSOst1A3pwdO/k7JHyFhdi9Xg8JjkTJ8vWSrSx7IeYqoy3ZIvI8/");  //passing encrypted password as arguments
    
    pthread_create(&tii, NULL, kernel_function_2, 
    "$6$AS$8pcKmkYTPSlXF7rrQWymrzSf0msE12EZaavSOst1A3pwdO/k7JHyFhdi9Xg8JjkTJ8vWSrSx7IeYqoy3ZIvI8/");  //passing encrypted password as arguments
    
    pthread_join(ti, NULL);
    pthread_join(tii, NULL);
    pthread_mutex_destroy(&lock); 

}

void *kernel_function_1(char *salt_and_encrypted)
{
  int x, y, z;     // Loop counters
  char salt[7];    
  char plain[7];   // The combination of letters currently being checked 
  char *enc;       // Pointer to the encrypted password
  int count = 0;

  substr(salt, salt_and_encrypted, 0, 6);
  
  pthread_mutex_lock(&lock); 

  for(x='A'; x<='M'; x++){  //searching from A to M
    for(y='A'; y<='Z'; y++){
      for(z=0; z<=99; z++){
        sprintf(plain, "%c%c%02d", x, y, z); 
          
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(salt_and_encrypted, enc) == 0){
	  printf("#%-8d%s %s\n", count, plain, enc);
		//return;
        } 
      }
    }
  }
  
  printf("%d solutions explored\n", count);
    
  pthread_mutex_unlock(&lock); 
}


void *kernel_function_2(char *salt_and_encrypted)
{
    int x, y, z;	
    char salt[7];	
    char plain[7];	
    char *enc;	
    int count = 0;	

    substr(salt, salt_and_encrypted, 0, 6);
    
    pthread_mutex_lock(&lock);

    for(x='N'; x<='Z'; x++){  //searching from N to Z 
      for(y='A'; y<='Z'; y++){
        for(z=0; z<=99; z++){
          sprintf(plain, "%c%c%02d", x, y, z); 
          enc = (char *) crypt(plain, salt);
          count++;
          if(strcmp(salt_and_encrypted, enc) == 0){
	    printf("#%-8d%s %s\n", count, plain, enc);
		//return;	
          } 
        }
      }
    }
    
    printf("%d solutions explored\n", count);
    
    pthread_mutex_unlock(&lock); 
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

    struct timespec start, finish;
    long long int time_elapsed;
    
    if (pthread_mutex_init(&lock, NULL) != 0) { 
        printf("\n mutex init has failed\n"); 
        return 1; 
    } 
    
    clock_gettime(CLOCK_MONOTONIC, &start);

    mainThread();

    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    time_difference(&start, &finish, &time_elapsed);
    printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));
        
    return 0;
}
