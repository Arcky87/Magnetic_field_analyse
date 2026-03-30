comp:=gfortran
pattern:=*.f95
paral:=-fopenmp
opt:=-O2
source:=$(wildcard $(pattern))
   obj:=$(patsubst %.f95, %.o, $(source))
prog: $(obj)
	$(comp) $(opt) $(paral) $^ -o $@
%.o %.mod : %.f95
	$(comp) -c $(opt) $(paral) $<
field_curve_modern_fortran.o : magnetic_model.mod
clear:
	rm -f *.o
clean:
	rm -f *.o *.mod prog
make_python_bib: prog magnetic_model.f95
	f2py -c magnetic_model.f95 --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -m magnetic_model
	mv magnetic_model.cpython-312-x86_64-linux-gnu.so ../magnetic_model.cpython-312-x86_64-linux-gnu.so