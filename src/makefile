.PHONY: run

run:
	julia -p 1 main.jl $(ARGS)

run_mp:
	julia -p $(NWORKERS) main.jl --nworkers $(NWORKERS) $(ARGS)

%:
	@:
