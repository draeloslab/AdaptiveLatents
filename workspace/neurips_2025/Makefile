.DEFAULT_GOAL := all_generated

include common.mk
include half_turn_plots/Makefile
include dim_red_plots/Makefile
include simulation_plots/Makefile
include paralell_compare_plots/Makefile

all_generated: half_turn_plots dim_red_plots simulation_plots paralell_compare_plots

pdf: all_generated FORCE
	cd $(pdf_path)
	latexmk -pdf -shell-escape neurips_2025.tex

all: pdf

clean:
	trash $(generated_path)/*.$(vector_output_filetype)
	trash $(generated_path)/*.yaml
	trash $(generated_path)/*.tex
	trash $(pdf_path)/aux_dir/*
	trash makefile_graph.png
	cd $(pdf_path)
	latexmk -C
