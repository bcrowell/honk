tests: tests.c oscillator.cl
	cc -DRUN_ON_CPU -o tests tests.c -x c -std=c99 oscillator.cl -lm
	./tests


cpu_c.so: cpu_c.c constants.h
	gcc -c -Wall -Werror -fpic cpu_c.c
	gcc -shared -o cpu_c.so cpu_c.o
