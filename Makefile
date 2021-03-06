GCC 	 = g++
CC       = mpicxx
CFLAGS   = -DEIGEN_USE_LAPACKE --std=c++14 -g -O3 -DEIGEN_NO_DEBUG
CFLAGS1   = -DEIGEN_USE_LAPACKE -DTTOR_SHARED --std=c++14 -g -O3 -DEIGEN_NO_DEBUG
INCLUDE  = -I../src/
SRCDIR   = ../src
OBJDIR   = ../build
LFLAGS   = -llapack -lblas -lpthread

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

OBJ1  = $(OBJDIR)/serialization.o $(OBJDIR)/util.o

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(GCC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(OBJ)
	$(GCC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

tuto: tuto.cpp $(OBJ1)
	$(GCC) $(CFLAGS1) -o $@ $^ $(INCLUDE) $(LFLAGS)

MultiLapack: MultiLapack.cpp $(OBJ1)
	$(GCC) $(CFLAGS1) -o $@ $^ $(INCLUDE) $(LFLAGS)

ttor_distri: ttor_distri.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

cholesky: cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

run: tuto
	mpirun -mca shmem posix -mca btl ^tcp -n 2 ./tuto	

clean:
	rm -f tuto
	rm -f $(OBJDIR)/*.o
