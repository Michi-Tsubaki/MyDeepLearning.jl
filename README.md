# MyDeepLearning.jl

## How to use?

If you are new to Julia or Flux.jl, you can check the [official web page](https://fluxml.ai/Flux.jl/stable/) first.

For experienced users with the latest stable Julia properly installed:

1. Clone this project.
1. Start the Julia REPL inside the project folder.
1. Activate and instantiate the environment
    1. `import Pkg`
    2. `Pkg.activate(".")`
    3. `Pkg.instantiate()`
3. Start [Pluto.jl](https://github.com/fonsp/Pluto.jl)
    1. `import Pluto`
    1. `Pluto.run()`
4. Now you can see the Pluto page is opened in your browser. Paste
   `notebooks/Flux_simple_MNIST.jl` (or any other file under the `notebooks` folder) into
   the input box and click the `Open` button.
