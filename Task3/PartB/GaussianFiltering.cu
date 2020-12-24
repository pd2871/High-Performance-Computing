#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"

//change width and height values according to new image
#define width 600
#define height 315

//compile with c++ lodepng file

/*

  Compile using   nvcc GaussianFiltering.cu lodepng.cpp -o filter
  
  use  ./filter img.png     for gaussian filtering the image using cuda
  
*/

unsigned int img_width;
unsigned int img_height;

__device__ unsigned char getRed(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4);
  return image[i];
}

__device__ unsigned char getGreen(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4) +1;
  return image[i];
}

__device__ unsigned char getBlue(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4) +2;
  return image[i];
}

__device__ unsigned char getAlpha(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4) +3;
  return image[i];
}

__device__ void setRed(unsigned char *image, unsigned int row, unsigned int col, unsigned char red){
  unsigned int i = (row * width * 4) + (col * 4);
  image[i] = red;
}

__device__ void setGreen(unsigned char *image, unsigned int row, unsigned int col, unsigned char green){
  unsigned int i = (row * width * 4) + (col * 4) +1;
  image[i] = green;
}

__device__ void setBlue(unsigned char *image, unsigned int row, unsigned int col, unsigned char blue){
  unsigned int i = (row * width * 4) + (col * 4) +2;
  image[i] = blue;
}

__device__ void setAlpha(unsigned char *image, unsigned int row, unsigned int col, unsigned char alpha){
  unsigned int i = (row * width * 4) + (col * 4) +3;
  image[i] = alpha;
}


__global__ void gaussian_filtering(unsigned char * gpu_imageOuput, unsigned char * gpu_imageInput){

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

	int row = blockIdx.x+1;  //starts from index 1
	int col = threadIdx.x+1;   //starts from index 1
	
	
	setGreen(gpu_imageOuput, row, col, getGreen(gpu_imageInput, row, col));
        setBlue(gpu_imageOuput, row, col, getBlue(gpu_imageInput, row, col));
        setAlpha(gpu_imageOuput, row, col, 255);

        redTL = getRed(gpu_imageInput, row-1, col-1);
        redTC = getRed(gpu_imageInput, row-1, col);
        redTR = getRed(gpu_imageInput, row-1, col+1);

        redL = getRed(gpu_imageInput, row, col-1);
        redC = getRed(gpu_imageInput, row, col);
        redR = getRed(gpu_imageInput, row, col+1);

        redBL = getRed(gpu_imageInput, row+1, col-1);
        redBC = getRed(gpu_imageInput, row+1, col);
        redBR = getRed(gpu_imageInput, row+1, col+1);

        newRed = redTL*filter[0][0] + redTC*filter[0][1] + redTR*filter[0][2]
	     + redL*filter[1][0]  + redC*filter[1][1]  + redR*filter[1][2]
	     + redBL*filter[2][0] + redBC*filter[2][1] + redBR*filter[2][2];
 
        setRed(gpu_imageOuput, row, col, newRed);

        greenTL = getGreen(gpu_imageInput, row-1, col-1);
        greenTC = getGreen(gpu_imageInput, row-1, col);
        greenTR = getGreen(gpu_imageInput, row-1, col+1);
  
        greenL = getGreen(gpu_imageInput, row, col-1);
        greenC = getGreen(gpu_imageInput, row, col);
        greenR = getGreen(gpu_imageInput, row, col+1);

        greenBL = getGreen(gpu_imageInput, row+1, col-1);
        greenBC = getGreen(gpu_imageInput, row+1, col);
        greenBR = getGreen(gpu_imageInput, row+1, col+1);

        newGreen = greenTL*filter[0][0] + greenTC*filter[0][1] + greenTR*filter[0][2]
	     + greenL*filter[1][0]  + greenC*filter[1][1]  + greenR*filter[1][2]
	     + greenBL*filter[2][0] + greenBC*filter[2][1] + greenBR*filter[2][2];
 
        setGreen(gpu_imageOuput, row, col, newGreen);

        blueTL = getBlue(gpu_imageInput, row-1, col-1);
        blueTC = getBlue(gpu_imageInput, row-1, col);
        blueTR = getBlue(gpu_imageInput, row-1, col+1);

        blueL = getBlue(gpu_imageInput, row, col-1);
        blueC = getBlue(gpu_imageInput, row, col);
        blueR = getBlue(gpu_imageInput, row, col+1);

        blueBL = getBlue(gpu_imageInput, row+1, col-1);
        blueBC = getBlue(gpu_imageInput, row+1, col);
        blueBR = getBlue(gpu_imageInput, row+1, col+1);

        newBlue = blueTL*filter[0][0] + blueTC*filter[0][1] + blueTR*filter[0][2]
	     + blueL*filter[1][0]  + blueC*filter[1][1]  + blueR*filter[1][2]
	     + blueBL*filter[2][0] + blueBC*filter[2][1] + blueBR*filter[2][2];
 
        setBlue(gpu_imageOuput, row, col, newBlue);
        
}

int main(int argc, char **argv){

	unsigned int error;
	unsigned int encError;
	unsigned char* image;
	const char* filename = argv[1];
	const char* newFileName = "gaussian_filtered.png";
	
	
	error = lodepng_decode32_file(&image, &img_width, &img_height, filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}
	
	
	printf("Image width = %d height = %d\n", img_width, img_height);

	const int ARRAY_SIZE = img_width*img_height*4;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

	unsigned char host_imageInput[ARRAY_SIZE * 4];
	unsigned char host_imageOutput[ARRAY_SIZE * 4];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		host_imageInput[i] = image[i];
	}

	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	cudaMemcpy(d_in, host_imageInput, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	gaussian_filtering<<< img_height-1-1, img_width-1-1 >>>(d_out, d_in);   

	cudaDeviceSynchronize();	
	
	// copy back the result array to the CPU
	cudaMemcpy(host_imageOutput, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	encError = lodepng_encode32_file(newFileName, host_imageOutput, img_width, img_height);
	if(encError){
		printf("error %u: %s\n", error, lodepng_error_text(encError));
	}

	//free(image);
	//free(host_imageInput);
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
