# Disable built-in rules and variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables



.ONESHELL:
SHELL:=/bin/bash

CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh; conda activate

.DEFAULT_GOAL := all_generated

.PHONY: FORCE

output_filetype := pdf

python_command := python
#python_command := python -m ipdb -c "c"

generated_path := /tmp/
script_path := .

include $(script_path)/*.mk

half_turn_prefix := $(generated_path)/hturn_log_pred_p_
half_turn_plots := $(foreach method, bw vjf kf, $(half_turn_prefix)$(method).$(output_filetype))
$(half_turn_plots): $(half_turn_prefix)%.$(output_filetype): $(script_path)/half_turn_log_pred_heatmaps.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/half_turn_log_pred_heatmaps.py \
		--output $@ \
		--pred_type $*


dim_red_methods := prosvd sjpca mmica

by_step_prefix := $(generated_path)/profile_by_step_
profile_by_step_plots := $(foreach method, $(dim_red_methods), $(by_step_prefix)$(method).$(output_filetype))
$(profile_by_step_plots): $(by_step_prefix)%.$(output_filetype): $(script_path)/%_code.py $(script_path)/dimension_reduction_plots.py $(script_path)/make_step_time_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/dimension_reduction_plots.py \
	    --type-of-plot time \
		--output $@ \
		--transformer $*

stability_prefix := $(generated_path)/stability_by_step_
stability_by_step_plots := $(foreach method, $(dim_red_methods), $(stability_prefix)$(method).$(output_filetype))
$(stability_by_step_plots): $(stability_prefix)%.$(output_filetype): $(script_path)/%_code.py $(script_path)/dimension_reduction_plots.py $(script_path)/make_stability_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/dimension_reduction_plots.py \
	    --type-of-plot stability \
		--output $@ \
		--transformer $*


nearness_to_offline_prefix := $(generated_path)/nearness_to_offline_by_step_
nearness_to_offline_by_step_plots := $(foreach method, $(dim_red_methods), $(nearness_to_offline_prefix)$(method).$(output_filetype))
$(nearness_to_offline_by_step_plots): $(nearness_to_offline_prefix)%.$(output_filetype): $(script_path)/%_code.py $(script_path)/dimension_reduction_plots.py $(script_path)/make_nearness_to_offline_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/dimension_reduction_plots.py \
	    --type-of-plot nearness_to_offline \
		--output $@ \
		--transformer $*

$(generated_path)/show_toy_dataset.pdf: $(script_path)/show_toy_dataset.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/show_toy_dataset.py \
		--output $@

$(script_path)/learn_s_hat_ss.py $(script_path)/learn_s_hat_toy.py: $(script_path)/learn_s_hat_plots.py
$(script_path)/learn_s_hat_ss.py: $(script_path)/sim_stim.py
$(script_path)/sim_stim.py $(script_path)/learn_s_hat_toy.py: $(script_path)/learn_s_hat_plots.py # finalize_log import

%.py::
	touch $@

$(generated_path)/learn_s_hat_toy_1step.pdf: $(script_path)/learn_s_hat_toy.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/learn_s_hat_toy.py \
		--output $@ \
	    --type-of-plot 1-step-prediction

$(generated_path)/learn_s_hat_table_toy.tex: $(script_path)/learn_s_hat_toy.py $(script_path)/make_constants_file.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/learn_s_hat_toy.py \
		--output $@ \
	    --type-of-plot 1-step-prediction-table

$(generated_path)/learn_s_hat_toy_manifold_error.pdf: $(script_path)/learn_s_hat_toy.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/learn_s_hat_toy.py \
		--output $@ \
	    --type-of-plot manifold-error


$(generated_path)/learn_s_hat_ss_1step.pdf: $(script_path)/learn_s_hat_ss.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/learn_s_hat_ss.py \
		--output $@ \
	    --type-of-plot 1-step-prediction



$(generated_path)/delay_table_ss.tex: $(script_path)/learn_s_hat_ss.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/learn_s_hat_ss.py \
		--output $@ \
	    --type-of-plot delay-table


generated_yamls = $(shell find $(generated_path) -name '*.yaml')
$(generated_yamls): $(generated_path)/generated_by_%.yaml: $(script_path)/%.py $(script_path)/common.py ;

#constant_input_files = adaptive_latents_config.yaml $(generated_yamls)
constants.tex: FORCE # $(script_path)/make_constants_file.py $(constant_input_files)
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(script_path)/make_constants_file.py \
		--output $@ \
		--input $(wildcard $(generated_path)/*.yaml) $(wildcard $(generated_path)/*.tex)

half_turn_plots: $(half_turn_plots)
profile_by_step_plots: $(profile_by_step_plots)
stability_by_step_plots: $(stability_by_step_plots)
nearness_to_offline_by_step_plots: $(nearness_to_offline_by_step_plots)
toy_dataset: $(generated_path)/show_toy_dataset.pdf
learn_s_hat_toy: $(generated_path)/learn_s_hat_toy_1step.pdf $(generated_path)/learn_s_hat_toy_manifold_error.pdf $(generated_path)/learn_s_hat_table_toy.tex
learn_s_hat_ss: $(generated_path)/learn_s_hat_ss_1step.pdf  $(generated_path)/delay_table_ss.tex
constants: constants.tex

all_generated: half_turn_plots profile_by_step_plots stability_by_step_plots nearness_to_offline_by_step_plots toy_dataset learn_s_hat_toy learn_s_hat_ss constants

pdf: all_generated FORCE
	latexmk -pdf -shell-escape neurips_2025.tex

all: pdf

meta_dependency_graph:
	make -Bnd | make2graph | dot -Tpng -o makefile_graph.png

clean:
	trash $(generated_path)/*.$(output_filetype)
	trash $(generated_path)/*.yaml
	trash $(generated_path)/*.tex
	trash constants.tex
	trash aux_dir/*
	trash makefile_graph.png
	latexmk -C
