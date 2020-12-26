#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lodepng.h"


//compile with c++ lodepng file

/*

  Compile using   nvcc GaussianBlurFiltering.cu lodepng.cpp -o cu_filter
  
  use  ./cu_filter img.png     for gaussian filtering the image using cuda
  
*/


__device__ unsigned char getRed(unsigned char *image, unsigned int row, unsigned int col, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4);
  return image[i];
}

__device__ unsigned char getGreen(unsigned char *image, unsigned int row, unsigned int col, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4) +1;
  return image[i];
}

__device__ unsigned char getBlue(unsigned char *image, unsigned int row, unsigned int col, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4) +2;
  return image[i];
}

__device__ unsigned char getAlpha(unsigned char *image, unsigned int row, unsigned int col, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4) +3;
  return image[i];
}

__device__ void setRed(unsigned char *image, unsigned int row, unsigned int col, unsigned char red, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4);
  image[i] = red;
}

__device__ void setGreen(unsigned char *image, unsigned int row, unsigned int col, unsigned char green, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4) +1;
  image[i] = green;
}

__device__ void setBlue(unsigned char *image, unsigned int row, unsigned int col, unsigned char blue, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4) +2;
  image[i] = blue;
}

__device__ void setAlpha(unsigned char *image, unsigned int row, unsigned int col, unsigned char alpha, unsigned int width){
  unsigned int i = (row * width * 4) + (col * 4) +3;
  image[i] = alpha;
}


__global__ void gaussian_blur_filtering(unsigned char * gpu_imageOuput, unsigned char * gpu_imageInput, unsigned int * width){

	unsigned int iwidth = *width;
	
	unsigned redTL,redTC, redTR;
	unsigned redL, redC, redR;
	unsigned redBL,redBC, redBR;
	unsigned newRed;

        unsigned greenTL,greenTC, greenTR;
        unsigned greenL, greenC, greenR;
        unsigned greenBL,greenBC, greenBR;
        unsigned newGreen;

	unsigned blueTL,blueTC, blueTR;
	unsigned blueL, blueC, blueR;
	unsigned blueBL,blueBC, blueBR;
	unsigned newBlue;
	
	
	  
	float filter[3][3] = {
	  { 1.0/16, 2.0/16, 1.0/16 },
	  { 2.0/16, 4.0/16, 2.0/16 },
	  { 1.0/16, 2.0/16, 1.0/16 }};

	unsigned int row = blockIdx.x+1;  //starts from index 1
	unsigned int col = threadIdx.x+1;   //starts from index 1
	
	
	setGreen(gpu_imageOuput, row, col, getGreen(gpu_imageInput, row, col, iwidth), iwidth);
        setBlue(gpu_imageOuput, row, col, getBlue(gpu_imageInput, row, col, iwidth), iwidth);
        setAlpha(gpu_imageOuput, row, col, 255, iwidth);

        redTL = getRed(gpu_imageInput, row-1, col-1, iwidth);
        redTC = getRed(gpu_imageInput, row-1, col, iwidth);
        redTR = getRed(gpu_imageInput, row-1, col+1, iwidth);

        redL = getRed(gpu_imageInput, row, col-1, iwidth);
        redC = getRed(gpu_imageInput, row, col, iwidth);
        redR = getRed(gpu_imageInput, row, col+1, iwidth);

        redBL = getRed(gpu_imageInput, row+1, col-1, iwidth);
        redBC = getRed(gpu_imageInput, row+1, col, iwidth);
        redBR = getRed(gpu_imageInput, row+1, col+1, iwidth);

        newRed = redTL*filter[0][0] + redTC*filter[0][1] + redTR*filter[0][2]
	     + redL*filter[1][0]  + redC*filter[1][1]  + redR*filter[1][2]
	     + redBL*filter[2][0] + redBC*filter[2][1] + redBR*filter[2][2];
 
        setRed(gpu_imageOuput, row, col, newRed, iwidth);

        greenTL = getGreen(gpu_imageInput, row-1, col-1, iwidth);
        greenTC = getGreen(gpu_imageInput, row-1, col, iwidth);
        greenTR = getGreen(gpu_imageInput, row-1, col+1, iwidth);
  
        greenL = getGreen(gpu_imageInput, row, col-1, iwidth);
        greenC = getGreen(gpu_imageInput, row, col, iwidth);
        greenR = getGreen(gpu_imageInput, row, col+1, iwidth);

        greenBL = getGreen(gpu_imageInput, row+1, col-1, iwidth);
        greenBC = getGreen(gpu_imageInput, row+1, col, iwidth);
        greenBR = getGreen(gpu_imageInput, row+1, col+1, iwidth);

        newGreen = greenTL*filter[0][0] + greenTC*filter[0][1] + greenTR*filter[0][2]
	     + greenL*filter[1][0]  + greenC*filter[1][1]  + greenR*filter[1][2]
	     + greenBL*filter[2][0] + greenBC*filter[2][1] + greenBR*filter[2][2];
 
        setGreen(gpu_imageOuput, row, col, newGreen, iwidth);

        blueTL = getBlue(gpu_imageInput, row-1, col-1, iwidth);
        blueTC = getBlue(gpu_imageInput, row-1, col, iwidth);
        blueTR = getBlue(gpu_imageInput, row-1, col+1, iwidth);

        blueL = getBlue(gpu_imageInput, row, col-1, iwidth);
        blueC = getBlue(gpu_imageInput, row, col, iwidth);
        blueR = getBlue(gpu_imageInput, row, col+1, iwidth);

        blueBL = getBlue(gpu_imageInput, row+1, col-1, iwidth);
        blueBC = getBlue(gpu_imageInput, row+1, col, iwidth);
        blueBR = getBlue(gpu_imageInput, row+1, col+1, iwidth);

        newBlue = blueTL*filter[0][0] + blueTC*filter[0][1] + blueTR*filter[0][2]
	     + blueL*filter[1][0]  + blueC*filter[1][1]  + blueR*filter[1][2]
	     + blueBL*filter[2][0] + blueBC*filter[2][1] + blueBR*filter[2][2];
 
        setBlue(gpu_imageOuput, row, col, newBlue, iwidth);
        
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


int main(int argc, char **argv){

	struct timespec start, finish;
  	long long int time_elapsed;
  	
  	unsigned int * img_width;
	unsigned int * img_height;
	
	const int N = 4000*4000*4;
	const int INT_BYTE = N *sizeof(unsigned int);
  	
  	img_width = (unsigned int*) malloc(INT_BYTE);
  	img_height = (unsigned int*) malloc(INT_BYTE);
  	
	unsigned int error;
	unsigned int encError;
	unsigned char* image;
	const char* filename = argv[1];
	const char* newFileName = "cu_filtered.png";
	
	error = lodepng_decode32_file(&image, img_width, img_height, filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}
	
	
	printf("Image width = %d height = %d\n", *img_width, *img_height);

	const int ARRAY_SIZE = (*img_width)* (*img_height)*4;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);
	const int INT_BYTES = ARRAY_SIZE *sizeof(unsigned int);

	unsigned char host_imageInput[ARRAY_SIZE * 4];
	unsigned char host_imageOutput[ARRAY_SIZE * 4];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		host_imageInput[i] = image[i];
	}

	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;
	unsigned int * width;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);
	cudaMalloc((void**) &width, INT_BYTES);

	cudaMemcpy(d_in, host_imageInput, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	cudaMemcpy(width, img_width, INT_BYTES, cudaMemcpyHostToDevice);

	clock_gettime(CLOCK_MONOTONIC, &start);	

	// launch the kernel
	gaussian_blur_filtering<<< *img_height-1-1, *img_width-1-1 >>>(d_out, d_in, width);   

	cudaDeviceSynchronize();	
	
	// copy back the result array to the CPU
	cudaMemcpy(host_imageOutput, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	encError = lodepng_encode32_file(newFileName, host_imageOutput, *img_width, *img_height);
	if(encError){
		printf("error %u: %s\n", error, lodepng_error_text(encError));
	}
	
	clock_gettime(CLOCK_MONOTONIC, &finish);
	
	time_difference(&start, &finish, &time_elapsed);
	
	time_difference(&start, &finish, &time_elapsed);
	
	printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
