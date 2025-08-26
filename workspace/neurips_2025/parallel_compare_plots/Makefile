parallel_compare_plots: pcps

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(realpath $(ROOT_DIR)/../common.mk)
-include $(realpath $(ROOT_DIR)/../local.mk)
pcp_script_path:=$(ROOT_DIR)

$(generated_path)/paralell_compare.$(raster_output_filetype): $(pcp_script_path)/parallel_compare.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(pcp_script_path)/parallel_compare.py \
		--output $@ \

pcps: $(generated_path)/paralell_compare.$(raster_output_filetype)
