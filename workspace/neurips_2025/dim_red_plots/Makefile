# NOTE: you should probably not use this makefile, but the one at the root of the project

dim_red_plots: profile_by_step_plots stability_by_step_plots nearness_to_offline_by_step_plots native_nearness_to_offline_by_step_plots

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(realpath $(ROOT_DIR)/../common.mk)
-include $(realpath $(ROOT_DIR)/../local.mk)
dim_red_script_path:=$(ROOT_DIR)

by_step_prefix := $(generated_path)/profile_by_step_
profile_by_step_plots := $(foreach method, $(dim_red_methods), $(by_step_prefix)$(method).$(vector_output_filetype))
$(profile_by_step_plots): $(by_step_prefix)%.$(vector_output_filetype): $(dim_red_script_path)/%_code.py $(dim_red_script_path)/dimension_reduction_plots.py $(dim_red_script_path)/make_step_time_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(dim_red_script_path)/dimension_reduction_plots.py \
	    --type-of-plot time \
		--output $@ \
		--transformer $*
profile_by_step_plots: $(profile_by_step_plots)


stability_prefix := $(generated_path)/stability_by_step_
stability_by_step_plots := $(foreach method, $(dim_red_methods), $(stability_prefix)$(method).$(vector_output_filetype))
$(stability_by_step_plots): $(stability_prefix)%.$(vector_output_filetype): $(dim_red_script_path)/%_code.py $(dim_red_script_path)/dimension_reduction_plots.py $(dim_red_script_path)/make_stability_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(dim_red_script_path)/dimension_reduction_plots.py \
	    --type-of-plot stability \
		--output $@ \
		--transformer $*
stability_by_step_plots: $(stability_by_step_plots)


nearness_to_offline_prefix := $(generated_path)/nearness_to_offline_by_step_
nearness_to_offline_by_step_plots := $(foreach method, $(dim_red_methods), $(nearness_to_offline_prefix)$(method).$(vector_output_filetype))
$(nearness_to_offline_by_step_plots): $(nearness_to_offline_prefix)%.$(vector_output_filetype): $(dim_red_script_path)/%_code.py $(dim_red_script_path)/dimension_reduction_plots.py $(dim_red_script_path)/make_nearness_to_offline_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(dim_red_script_path)/dimension_reduction_plots.py \
	    --type-of-plot nearness_to_offline \
		--output $@ \
		--transformer $*
nearness_to_offline_by_step_plots: $(nearness_to_offline_by_step_plots)

native_nearness_to_offline_prefix := $(generated_path)/native_nearness_to_offline_by_step_
native_nearness_to_offline_by_step_plots := $(foreach method, $(dim_red_methods), $(native_nearness_to_offline_prefix)$(method).$(vector_output_filetype))
$(native_nearness_to_offline_by_step_plots): $(native_nearness_to_offline_prefix)%.$(vector_output_filetype): $(dim_red_script_path)/%_code.py $(dim_red_script_path)/dimension_reduction_plots.py $(dim_red_script_path)/make_native_nearness_to_offline_figure.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(dim_red_script_path)/dimension_reduction_plots.py \
	    --type-of-plot native_nearness_to_offline \
		--output $@ \
		--transformer $*
native_nearness_to_offline_by_step_plots: $(native_nearness_to_offline_by_step_plots)