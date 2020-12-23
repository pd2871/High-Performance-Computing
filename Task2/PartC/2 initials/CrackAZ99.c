#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <time.h>


/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer.

  Compile with:
    cc -o CrackAZ99 CrackAZ99.c -lcrypt

  If you want to analyse the output then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:
    ./CrackAZ99 > output.txt
******************************************************************************/

int count=0;     // A counter used to track the number of combinations explored so far


void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}


void crack(char *salt_and_encrypted){
  int p, d, l;     // Loop counters
  char salt[7];    
  char plain[7];   // The combination of letters currently being checked 
  char *enc;       // Pointer to the encrypted password

  substr(salt, salt_and_encrypted, 0, 6);

  for(p='A'; p<='Z'; p++){
    for(d='A'; d<='Z'; d++){
      for(l=0; l<=99; l++){
      printf("%d",p);
        sprintf(plain, "%c%c%02d", p, d, l); 
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(salt_and_encrypted, enc) == 0){
	    printf("#%-8d%s %s\n", count, plain, enc);
		//return;	
        } 
      }
    }
  }
}

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

int main(int argc, char *argv[]){
    struct timespec start, finish;
    long long int time_elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    crack("$6$AS$8pcKmkYTPSlXF7rrQWymrzSf0msE12EZaavSOst1A3pwdO/k7JHyFhdi9Xg8JjkTJ8vWSrSx7IeYqoy3ZIvI8/");	
    printf("%d solutions explored\n", count);
  
    clock_gettime(CLOCK_MONOTONIC, &finish);

    time_difference(&start, &finish, &time_elapsed);

    printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,(time_elapsed / 1.0e9));

    return 0;
}

