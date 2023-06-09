#include "png.hpp"
#include <cmath>
#include <cstdlib>
#include <png.h>
#include <stdlib.h>
#include <unistd.h>

static char *INPUT = strdup("cactus.png");

PNG readInput() {
  PNG img_in(INPUT);
  img_in.read_png_file();
  return img_in;
}

int copy() {
  PNG img_in(INPUT);
  img_in.read_png_file();
  unsigned int *raw = img_in.getRawImg();
  unsigned int *out_img;
  out_img = (unsigned int *)malloc(img_in.getWidth() * img_in.getHeight() *
                                   sizeof(unsigned int));
  PNG img_out(strdup(".copy.png"), img_in.getWidth(), img_in.getHeight(),
              out_img);
  img_out.setRawImg(raw);
  img_out.write_png_file();

  return 0;
}

__device__ int brighten_pixel(int p, int amount) {

  unsigned int r = PNG_GETR(p) + amount;
  unsigned int g = PNG_GETG(p) + amount;
  unsigned int b = PNG_GETB(p) + amount;
  unsigned int a = PNG_GETA(p) + amount;
  return PNG_OUTPUT(r, g, b, a);
}

__global__ void brighten_device(unsigned int *in, unsigned int *out, int length,
                                int threads, int amount) {
  for (int i = threadIdx.x; i < length; i += threads) {
    out[i] = brighten_pixel(in[i], amount);
  }
}

__host__ void brighten_cuda() {
  PNG input = readInput();
  int size = input.getHeight() * input.getWidth();
  int threads = 1024;
  unsigned int *raw_in = input.getRawImg();
  unsigned int *raw_out = (unsigned int *)malloc(size * sizeof(unsigned int));
  unsigned int *dev_in, dev_out;
  cudaMalloc((void **)&dev_in, size * sizeof(unsigned int));
  cudaMalloc((void **)&dev_out, size * sizeof(unsigned int));
  cudaMemcpy(dev_in, raw_in, size * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  brighten_device<<<1, threads>>> size(dev_in, dev_out, size, threads, 20);

  cudaMemcpy(raw_out, dev_out, size * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  PNG img_out(strdup(".bright.png"), input.getWidth(), input.getHeight(),
              raw_out);
  img_out.write_png_file();
  cudaFree(dev_in);
  cudaFree(dev_out);
  free(raw_in);
  free(raw_out);
}

int brighten(int amount) {
  PNG input = readInput();
  int size = input.getHeight() * input.getWidth();
  unsigned int *raw_in = input.getRawImg();
  unsigned int *raw_out = (unsigned int *)malloc(size * sizeof(unsigned int));
  for (int i = 0; i < input.getHeight() * input.getWidth(); i++) {
    unsigned int p = raw_in[i];
    unsigned int r = PNG_GETR(p) + amount;
    unsigned int g = PNG_GETG(p) + amount;
    unsigned int b = PNG_GETB(p) + amount;
    unsigned int a = PNG_GETA(p) + amount;
    raw_out[i] = PNG_OUTPUT(r, g, b, a);
  }
  PNG img_out(strdup(".bright.png"), input.getWidth(), input.getHeight(),
              raw_out);
  img_out.write_png_file();
  return 0;
}

int main(int argc, char **argv) {
  copy();
  brighten(50);
  return 0;
}