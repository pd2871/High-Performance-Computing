#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <crypt.h>
#include <unistd.h>
#include <time.h>

//__global__ --> GPU function which can be launched by many blocks and threads
//__device__ --> GPU function or variables
//__host__ --> CPU function or variables

__device__ int count=0;     // A counter used to track the number of combinations explored so far

__device__ char* CudaCrypt(char* rawPassword){

	char * newPassword = (char *) malloc(sizeof(char) * 11);

	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for(int i =0; i<10; i++){
		if(i >= 0 && i < 6){ //checking all lower case letter limits
			if(newPassword[i] > 122){
				newPassword[i] = (newPassword[i] - 122) + 97;
			}else if(newPassword[i] < 97){
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		}else{ //checking number section
			if(newPassword[i] > 57){
				newPassword[i] = (newPassword[i] - 57) + 48;
			}else if(newPassword[i] < 48){
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}
	return newPassword;
}

//it is device function which will compare two strings and returns 0 if both strings match
__device__ int my_strcmp (const char * s1, const char * s2) {
    for(; *s1 == *s2; ++s1, ++s2)
        if(*s1 == 0)
            return 0;
    return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
}


//this device function will crack the encrypted password
__global__ void crack(char * alphabet, char * numbers){

    char *enc;
    char genRawPass[4];
    
    char * myEncryptedPassword = (char *)"rnqgac4034";  //this is the encrypted password that will be searched using CUDA

    
    //encrypting all the raw passwords
    genRawPass[0] = alphabet[blockIdx.x];
    genRawPass[1] = alphabet[blockIdx.y];

    genRawPass[2] = numbers[threadIdx.x];
    genRawPass[3] = numbers[threadIdx.y];

    count++;

    //firstLetter - 'a' - 'z' (26 characters)
    //secondLetter - 'a' - 'z' (26 characters)
    //firstNum - '0' - '9' (10 characters)
    //secondNum - '0' - '9' (10 characters)

    //Idx --> gives current index of the block or thread


    //printf("%c %c %c %c = %s\n", genRawPass[0],genRawPass[1],genRawPass[2],genRawPass[3], CudaCrypt(genRawPass));


    enc = (char *) CudaCrypt(genRawPass); //encrypts all the raw passwords
    
    if (my_strcmp(myEncryptedPassword, enc) == 0){  //compares all the encrypted passwords with my encrypted password
    
      printf("#%-8d%s %s\n", count, genRawPass, enc); //prints the found raw password with its encrypted form
      
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



int main(int argc, char ** argv){

struct timespec start, finish;
long long int time_elapsed;

char cpuAlphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
char cpuNumbers[26] = {'0','1','2','3','4','5','6','7','8','9'};

char * gpuAlphabet;
cudaMalloc( (void**) &gpuAlphabet, sizeof(char) * 26); 
cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

char * gpuNumbers;
cudaMalloc( (void**) &gpuNumbers, sizeof(char) * 26); 
cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 26, cudaMemcpyHostToDevice);

clock_gettime(CLOCK_MONOTONIC, &start);

crack<<< dim3(26,26,1), dim3(10,10,1) >>>( gpuAlphabet, gpuNumbers);
cudaDeviceSynchronize();

clock_gettime(CLOCK_MONOTONIC, &finish);

time_difference(&start, &finish, &time_elapsed);

printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));
return 0;
}



//"rnqgac4034 - PD28"









