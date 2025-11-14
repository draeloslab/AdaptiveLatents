simulation_plots: toy_dataset learn_s_hat_toy learn_s_hat_toy_spin learn_s_hat_ss optim zong_stim $(generated_path)/optim_col_vs_rand_with_high_d_rand.$(vector_output_filetype) $(generated_path)/optim_col_vs_rand_with_high_d_rand_closed.$(vector_output_filetype)

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(realpath $(ROOT_DIR)/../common.mk)
-include $(realpath $(ROOT_DIR)/../local.mk)
sim_script_path:=$(ROOT_DIR)


$(sim_script_path)/learn_s_hat_toy.py: $(sim_script_path)/learn_s_hat_plots.py
	touch $@

$(sim_script_path)/learn_s_hat_ss.py: $(sim_script_path)/learn_s_hat_plots.py $(sim_script_path)/sim_stim.py
	touch $@

$(sim_script_path)/sim_stim.py: $(sim_script_path)/learn_s_hat_plots.py
	touch $@



$(generated_path)/show_toy_dataset.$(vector_output_filetype): $(sim_script_path)/show_toy_dataset.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/show_toy_dataset.py \
		--output $@


$(generated_path)/learn_s_hat_toy_1step.$(vector_output_filetype) $(generated_path)/toy_curvy.$(vector_output_filetype) $(generated_path)/toy_curvy_spun.$(vector_output_filetype): $(sim_script_path)/learn_s_hat_toy.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/learn_s_hat_toy.py \
		--output $@ \
	    --type-of-plot 1-step-prediction


$(generated_path)/learn_s_hat_toy_manifold_error.$(vector_output_filetype): $(sim_script_path)/learn_s_hat_toy.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/learn_s_hat_toy.py \
		--output $@ \
	    --type-of-plot manifold-error


ss_1step_prefix := $(generated_path)/ss_1step_
ss_1step_plots := $(foreach method, $(prediction_methods), $(ss_1step_prefix)$(method).$(raster_output_filetype))
$(ss_1step_plots): $(ss_1step_prefix)%.$(raster_output_filetype): $(sim_script_path)/learn_s_hat_ss.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/learn_s_hat_ss.py \
		--output $@ \
	    --type-of-plot 1-step-prediction \
		--type-of-predictor $*

$(generated_path)/optim_col_vs_rand.$(vector_output_filetype): $(sim_script_path)/optimization_comparison.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/optimization_comparison.py \
		--output $@ \
	    --type-of-plot optim_col_vs_rand


$(generated_path)/optim_col_vs_rand_with_high_d_rand.$(vector_output_filetype): $(sim_script_path)/optimization_comparison.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/optimization_comparison.py \
		--output $@ \
	    --type-of-plot optim_col_vs_rand_with_high_d_rand

$(generated_path)/optim_col_vs_rand_with_high_d_rand_closed.$(vector_output_filetype): $(sim_script_path)/optimization_comparison.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/optimization_comparison.py \
		--output $@ \
	    --type-of-plot optim_col_vs_rand_with_high_d_rand_closed


#$(generated_path)/optim_open_vs_closed.$(vector_output_filetype): $(script_path)/optimization_comparison.py $(script_path)/sim_stim.py
#	$(CONDA_ACTIVATE) adaptive_latents
#	$(python_command) $(script_path)/optimization_comparison.py \
#		--output $@ \
#	    --type-of-plot optim_open_vs_closed


$(generated_path)/optim_open_vs_closed_toy.$(vector_output_filetype): $(sim_script_path)/optimization_comparison.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/optimization_comparison.py \
		--output $@ \
	    --type-of-plot optim_open_vs_closed_toy \
		--type-of-dim-red prosvd # you could do more here



open_vs_closed_by_dimred_prefix := $(generated_path)/open_vs_closed_by_dimred_
open_vs_closed_by_dimred_plots := $(foreach method, $(dim_red_methods), $(open_vs_closed_by_dimred_prefix)$(method).$(vector_output_filetype))
$(open_vs_closed_by_dimred_plots): $(open_vs_closed_by_dimred_prefix)%.$(vector_output_filetype): $(sim_script_path)/optimization_comparison.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/optimization_comparison.py \
		--output $@ \
	    --type-of-plot optim_open_vs_closed \
		--type-of-dim-red $*

$(generated_path)/zong_stim.$(vector_output_filetype): $(sim_script_path)/zong_stim.py $(sim_script_path)/sim_stim.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(sim_script_path)/zong_stim.py --output $@


$(generated_path)/learn_s_hat_toy_1step_spin.$(vector_output_filetype): $(sim_script_path)/learn_s_hat_toy_1step_spin.ipynb $(sim_script_path)/learn_s_hat_toy.py
	$(CONDA_ACTIVATE) adaptive_latents && cd $(sim_script_path)
	papermill -p output $@ $< /tmp/output.ipynb


open_vs_closed_by_dimred: $(open_vs_closed_by_dimred_plots)
toy_dataset: $(generated_path)/show_toy_dataset.$(vector_output_filetype)
learn_s_hat_toy: $(generated_path)/learn_s_hat_toy_1step.$(vector_output_filetype) $(generated_path)/learn_s_hat_toy_manifold_error.$(vector_output_filetype) #$(generated_path)/learn_s_hat_table_toy.tex
learn_s_hat_toy_spin: $(generated_path)/learn_s_hat_toy_1step_spin.$(vector_output_filetype)
learn_s_hat_ss: $(ss_1step_plots)  # $(generated_path)/delay_table_ss.tex
optim: $(generated_path)/optim_col_vs_rand.$(vector_output_filetype)  $(generated_path)/optim_open_vs_closed_toy.$(vector_output_filetype) open_vs_closed_by_dimred # $(generated_path)/optim_open_vs_closed.$(vector_output_filetype)
zong_stim: $(generated_path)/zong_stim.$(vector_output_filetype)