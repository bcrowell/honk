tests: tests.c oscillator.cl
	cc -DRUN_ON_CPU -o tests tests.c -x c -std=c99 oscillator.cl -lm
	./tests


