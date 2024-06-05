from scripts.main import main


def test_run_main(outdir):
    main(output_directory=outdir, steps_to_run=100)
