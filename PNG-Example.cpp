//Compiles with: gcc -o PNG-Example PNG-Example.cpp PNG-helper.hpp -lm -lpng -lstdc++
//Runs with: ./PNG-Example Input.png Output.png

#include "png.hpp"

int main(int argc, char **argv){

//Reads the given image
   PNG img_in(argv[1]);
   img_in.read_png_file();


//Example how to create a new yellow image of same dimension as input image
#ifndef USEINT
   unsigned char* out_img;
   out_img=(unsigned char*)malloc(img_in.getWidth()*img_in.getHeight()*4);
   for(int i=0;i<img_in.getWidth()*img_in.getHeight();i++){
	out_img[4*i+0]=(unsigned char)255; //Red
	out_img[4*i+1]=(unsigned char)255; //Green
	out_img[4*i+2]=(unsigned char)0;   //Blue
	out_img[4*i+3]=(unsigned char)255; //Alpha
	}
#else
   unsigned int* out_img;
   out_img=(unsigned int*)malloc(img_in.getWidth()*img_in.getHeight()*sizeof(unsigned int));
#endif
   PNG img_out(argv[2],img_in.getWidth(),img_in.getHeight(),out_img);
   img_out.write_png_file();

   return 0;
}
