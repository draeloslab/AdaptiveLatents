half_turn_plots: htps

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(realpath $(ROOT_DIR)/../common.mk)
-include $(realpath $(ROOT_DIR)/../local.mk)
htp_script_path:=$(ROOT_DIR)


half_turn_prefix := $(generated_path)/hturn_log_pred_p_
half_turn_plots_list := $(foreach method, $(prediction_methods), $(half_turn_prefix)$(method).$(raster_output_filetype))
$(half_turn_plots_list): $(half_turn_prefix)%.$(raster_output_filetype): $(htp_script_path)/half_turn_log_pred_heatmaps.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(htp_script_path)/half_turn_log_pred_heatmaps.py \
		--output $@ \
		--pred_type $*

htps: $(half_turn_plots_list)