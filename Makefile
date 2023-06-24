all:
	make build -s && make run -s
build:
	gcc $F -fopenmp -g -o $F.bin -lm
run:
	./$F.bin

la_1:
	F=la_1.c make -s
omp_1:
	F=omp_1.c make -s
pi_int:
	F=pi_int.c make -s
pi_mc:
	F=pi_mc.c make -s
image:
	gcc image.cpp png.hpp -lm -lpng -lstdc++
clean:
	rm -f *.bin
