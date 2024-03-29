#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <time.h>


/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 3 uppercase
  letters and a 2 digit integer.

  Compile with:
    cc -o CrackAZZ99 CrackAZZ99.c -lcrypt

  If you want to analyse the output then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:
    ./CrackAZZ99 > output.txt

******************************************************************************/

int count=0;     // A counter used to track the number of combinations explored so far


void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

void crack(char *salt_and_encrypted){
  int x, y, p, z;     // Loop counters
  char salt[7];    
  char plain[7];   // The combination of letters currently being checked 
  char *enc;       // Pointer to the encrypted password

  substr(salt, salt_and_encrypted, 0, 6);

  for(x='A'; x<='Z'; x++){
    for(y='A'; y<='Z'; y++){
      for(p='A'; p<='Z'; p++){
        for(z=0; z<=99; z++){
          sprintf(plain, "%c%c%c%02d", x, y, p, z); 
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
    
    crack("$6$AS$yfyiZXQNoN.uALrNqz4BHm9cnS3AIwOw8KoUZoGucCYqM9BoHnTp6QUof04Gr37vdIuee3.f9LR.wQ2f0B4bK.");	//PDL28 encrypted form
    printf("%d solutions explored\n", count);
  
    clock_gettime(CLOCK_MONOTONIC, &finish);

    time_difference(&start, &finish, &time_elapsed);

    printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,(time_elapsed / 1.0e9));

    return 0;
}

