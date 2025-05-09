To compile the figures, run:
```bash
make all_generated # default target
```
this will generate a bunch of output in `./generated`.

Note that if this is the first time you run make, you'll have to run it again to generate the constants table.

To complie the PDF:
```bash
make all
```
note that the PDF should be able to be complied from just the stuff tracked in git; this means the output of `make all_generated` needs to be tracked.


To continuously run latexmk:
```bash
latexmk -pdf -shell-escape -pvc -interaction=batchmode neurips_2025.tex
```


to visualize the makefile:
```
make -Bnd | make2graph | dot -Tpng -o makefile_graph.png
```
