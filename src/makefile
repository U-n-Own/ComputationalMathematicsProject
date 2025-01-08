.PHONY: run
.PHONY: run_mp
.PHONY: example

run:
	julia -p 1 main.jl $(ARGS)

run_mp:
	julia -p $(NWORKERS) main.jl --nworkers $(NWORKERS) $(ARGS)

example:
	julia -p 1 main.jl compute ./dataset/net10_8_1.dmx_out.npz --precond --iterations 200
%:
	@:
