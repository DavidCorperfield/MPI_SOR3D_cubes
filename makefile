#This makefile create the executable file FractalModel
CC    = CC
CXX   = mpicxx
RUN   = mpiexec
MAKE  = MAKE

#FLAGS = -Os -g -Wall -std=gnu99
FLAGS = -O3 -ffast-math -Wall
CXXFLAGS = $(FLAGS)

all: all-redirect

PROGRAMS = test

SOURCES = Matrix.cpp ResGrid.cpp SizeGrid.cpp SizeDomain.cpp Sources.cpp MpiFunctions.cpp Input.cpp SorSolution.cpp MpiInput.cpp BoundaryConditions.cpp

OBJECTS = Matrix.o ResGrid.o SizeGrid.o SizeDomain.o Sources.o MpiFunctions.o Input.o SorSolution.o MpiInput.o BoundaryConditions.o

PGM_SOURCES = ${SOURCES} main.cpp

${PROGRAMS}: main.o ${OBJECTS}
	${CXX} $(CXXFLAGS)  -lm -o ${PROGRAMS} main.o ${OBJECTS}

all-redirect: ${PROGRAMS}
	#Automatically execute the compiled code
	rm -rf results/*.dat
	${RUN} -n 8 ./${PROGRAMS}
	
clear:
	rm -f *.o ${PROGRAMS} *~ PI* results/*dat results/*eps results/*avi

.SUFFIXES: .o .c .f .cc .f90
.c:
	${CC} ${FLAGS} -o $* $< 
.c.o:
	${CC} ${FLAGS} -c $<
.cc:
	${CXX} ${FLAGS} -o $* $<  

sources:
	@echo ${PGM_SOURCES}