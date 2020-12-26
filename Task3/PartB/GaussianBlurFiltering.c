#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*

  Compile using   cc GaussianBlurFiltering.c lodepng.c -o c_filter
  
  use  ./c_filter img.png     for gaussian filtering the image using cuda
  
*/

unsigned int width;
unsigned int height;

unsigned char getRed(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4);
  return image[i];
}

unsigned char getGreen(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4) +1;
  return image[i];
}

unsigned char getBlue(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4) +2;
  return image[i];
}

unsigned char getAlpha(unsigned char *image, unsigned int row, unsigned int col){
  unsigned int i = (row * width * 4) + (col * 4) +3;
  return image[i];
}

void setRed(unsigned char *image, unsigned int row, unsigned int col, unsigned char red){
  unsigned int i = (row * width * 4) + (col * 4);
  image[i] = red;
}

void setGreen(unsigned char *image, unsigned int row, unsigned int col, unsigned char green){
  unsigned int i = (row * width * 4) + (col * 4) +1;
  image[i] = green;
}

void setBlue(unsigned char *image, unsigned int row, unsigned int col, unsigned char blue){
  unsigned int i = (row * width * 4) + (col * 4) +2;
  image[i] = blue;
}

void setAlpha(unsigned char *image, unsigned int row, unsigned int col, unsigned char alpha){
  unsigned int i = (row * width * 4) + (col * 4) +3;
  image[i] = alpha;
}


float filter[3][3] = {
  { 1.0/16, 2.0/16, 1.0/16 },
  { 2.0/16, 4.0/16, 2.0/16 },
  { 1.0/16, 2.0/16, 1.0/16 }};


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
  
  unsigned char *image;
  const char* filename = argv[1];
  const char* newFileName = "c_filtered.png";
  unsigned char *newImage;

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

  lodepng_decode32_file(&image, &width, &height, filename);
  newImage = malloc(height * width * 4 * sizeof(unsigned char));

  printf("Image width = %d height = %d\n", width, height);

  clock_gettime(CLOCK_MONOTONIC, &start);

  for(int row = 1; row < height-1; row++){
    for(int col = 1; col < width-1; col++){
      setGreen(newImage, row, col, getGreen(image, row, col));
      setBlue(newImage, row, col, getBlue(image, row, col));
      setAlpha(newImage, row, col, 255);

      redTL = getRed(image, row-1, col-1);
      redTC = getRed(image, row-1, col);
      redTR = getRed(image, row-1, col+1);

      redL = getRed(image, row, col-1);
      redC = getRed(image, row, col);
      redR = getRed(image, row, col+1);

      redBL = getRed(image, row+1, col-1);
      redBC = getRed(image, row+1, col);
      redBR = getRed(image, row+1, col+1);

      newRed = redTL*filter[0][0] + redTC*filter[0][1] + redTR*filter[0][2]
             + redL*filter[1][0]  + redC*filter[1][1]  + redR*filter[1][2]
             + redBL*filter[2][0] + redBC*filter[2][1] + redBR*filter[2][2];
 
      setRed(newImage, row, col, newRed);

      greenTL = getGreen(image, row-1, col-1);
      greenTC = getGreen(image, row-1, col);
      greenTR = getGreen(image, row-1, col+1);

      greenL = getGreen(image, row, col-1);
      greenC = getGreen(image, row, col);
      greenR = getGreen(image, row, col+1);

      greenBL = getGreen(image, row+1, col-1);
      greenBC = getGreen(image, row+1, col);
      greenBR = getGreen(image, row+1, col+1);

      newGreen = greenTL*filter[0][0] + greenTC*filter[0][1] + greenTR*filter[0][2]
             + greenL*filter[1][0]  + greenC*filter[1][1]  + greenR*filter[1][2]
             + greenBL*filter[2][0] + greenBC*filter[2][1] + greenBR*filter[2][2];
 
      setGreen(newImage, row, col, newGreen);

      blueTL = getBlue(image, row-1, col-1);
      blueTC = getBlue(image, row-1, col);
      blueTR = getBlue(image, row-1, col+1);

      blueL = getBlue(image, row, col-1);
      blueC = getBlue(image, row, col);
      blueR = getBlue(image, row, col+1);

      blueBL = getBlue(image, row+1, col-1);
      blueBC = getBlue(image, row+1, col);
      blueBR = getBlue(image, row+1, col+1);

      newBlue = blueTL*filter[0][0] + blueTC*filter[0][1] + blueTR*filter[0][2]
             + blueL*filter[1][0]  + blueC*filter[1][1]  + blueR*filter[1][2]
             + blueBL*filter[2][0] + blueBC*filter[2][1] + blueBR*filter[2][2];
 
      setBlue(newImage, row, col, newBlue);

    }
  }

  
  lodepng_encode32_file(newFileName, newImage, width, height);
  
  clock_gettime(CLOCK_MONOTONIC, &finish);
  
  time_difference(&start, &finish, &time_elapsed);
  
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed,
        (time_elapsed / 1.0e9));

  free(image);

  return 0;
}

