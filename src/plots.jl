using Plots

function run_GMRES(GMRES_Method, A, D, E, y, n, precond, restart, step)
    
    residual_in_time = []
    residual_over_iterations = []
    times = []
    x = zeros(size(y))

    for i = 1:step:n
        tick = time()
        if precond
            if restart
                println("now solving precond problem with restarted algo:")
                @time x = GMRES_Restarted(GMRES_Method, A, y, (D, E), i, 2000, 10E-10)
            else
                println("now solving precond problem:")
                @time x = GMRES_Method(D, E, y, i)
            end
        else
            if restart
                println("now solving non-precond problem with restarted algo:")
                @time x = GMRES_Restarted(GMRES_Method, A, y, (A,), i, 900, 10E-10)
            else
                println("now solving non-precond problem:")
                @time x = GMRES_Method(A, y, i)
            end
        end
        tock = time()

        push!(residual_in_time, norm(A * x - y))
        push!(times, tock - tick)
        push!(residual_over_iterations, norm(A * x - y))

        # if current residual is not different from previous one, we stop
        if i > 1 && abs(residual_over_iterations[end] - residual_over_iterations[end-1]) < 10E-10
            # since we stopped, we only take for plotting the values until now
            residual_in_time = residual_in_time[1:end-1]
            residual_over_iterations = residual_over_iterations[1:end-1]
            times = times[1:end-1]
            break
        end
    end


    return x, residual_in_time, residual_over_iterations, times
end

function GMRES_Experiments(GMRES_Method, A, D, E, y, n, precond, restart)
    """
    GMRES experiments with and without preconditioner

    GMRES_type: GMRES_IncrementalQR or GMRES_IncrementalQR_precond
    A: SparseMatrixCSC the matrix of the problem
    D: Diagonal the diagonal matrix of the problem
    E: SparseMatrixCSC the node arc incidence matrix of the problem
    y: Vector the right hand side of the problem
    n: Int the number of iterations
    precond: if we want to use preconditioner
    restart: if we want to use restarted GMRES

    Returns:
    x: Vector the solution of the problem
    plot_residual: Plots the residual over the iterations
    plot_residual_time: Plots the residual over the time
    """
    step = round(Int, n/50)
    x, residual_in_time, residual_over_iterations, times = run_GMRES(GMRES_Method, A, D, E, y, n, precond, restart, step)

    # n has to be max the lenght of the residual_over_iterations if we converged before the steps gives
    n = length(residual_over_iterations)
    
    plot_residual = plot(1:n, residual_over_iterations, label="Residual over iterations", 
    xlabel="Iterations", ylabel="Residual", title="Residual over iterations", scatter! = true, yaxis=:log)
    # with scatter     
    # plot residual over time
    plot_residual_time = plot(times, residual_in_time, label="Residual over time", 
    xlabel="Time", ylabel="Residual", title="Residual over time", scatter! = true, yaxis=:log)


    return x, plot_residual, plot_residual_time
end

