.ONESHELL:
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(realpath $(ROOT_DIR)/../common.mk)
-include $(realpath $(ROOT_DIR)/../local.mk)
benchmark_script_path:=$(ROOT_DIR)

benchmarking: benchmark_csvs $(generated_path)/benchmark_heatmap_table.$(vector_output_filetype) $(generated_path)/benchmark_single_trace.$(vector_output_filetype)

csv_prefix := $(generated_path)/benchmark_
define CSV_RULE
$(csv_prefix)$(dmethod)_$(pmethod).csv: $(benchmark_script_path)/record_benchmark_run.py
	source $$$$(conda info --base)/etc/profile.d/conda.sh; conda activate adaptive_latents
	$(python_command) $(benchmark_script_path)/record_benchmark_run.py --output $$@ --dimred_type $(dmethod) --pred_type $(pmethod)
endef
$(foreach dmethod, $(dim_red_methods), \
	$(foreach pmethod, $(prediction_methods), \
		$(eval $(CSV_RULE)) \
	) \
)
benchmark_csvs := \
				$(foreach dmethod, $(dim_red_methods), \
					$(foreach pmethod, $(prediction_methods), \
						$(csv_prefix)$(dmethod)_$(pmethod).csv \
					) \
				)
benchmark_csvs: $(benchmark_csvs)

$(generated_path)/benchmark_heatmap_table.$(vector_output_filetype): benchmark_csvs $(benchmark_script_path)/benchmark_heatmap_table.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(benchmark_script_path)/benchmark_heatmap_table.py --output $@

$(generated_path)/benchmark_single_trace.$(vector_output_filetype): $(csv_prefix)sjpca_bw.csv $(benchmark_script_path)/benchmark_single_trace.py
	$(CONDA_ACTIVATE) adaptive_latents
	$(python_command) $(benchmark_script_path)/benchmark_single_trace.py --input $< --output $@
