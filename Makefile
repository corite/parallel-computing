all:
	make build -s && make run -s
build:
	gcc $F -fopenmp -g -o $F.bin
run:
	./$F.bin

la_1:
	F=la_1.c make -s
omp_1:
	F=omp_1.c make -s

clean:
	rm -f *.bin
