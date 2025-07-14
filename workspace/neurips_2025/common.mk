# Disable built-in rules and variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
.ONESHELL:
SHELL:=/bin/bash
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh; conda activate

#output_filetype:=pdf
vector_output_filetype:=svg
raster_output_filetype:=png

#python_command := python
python_command := coverage run --append
#python_command := python -m ipdb -c "c"

script_path :=.
generated_path:=./generated
pdf_path:=./generated


prediction_methods := kf bw vjf
dim_red_methods := prosvd sjpca mmica

ifndef meta_dependency_graph_guard_variable
meta_dependency_graph_guard_variable := true
meta_dependency_graph:
	make -nd all_generated | make2graph | sed 's/label="[^"]*\/\([^"]*\)"/label="\1"/' | dot -Tpng -o makefile_graph.png
	# make -Bnd | make2graph | dot -Tpng -o makefile_graph.png
endif