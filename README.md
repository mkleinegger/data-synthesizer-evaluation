# ReadMe

## General

This project is used to experiment with different data synthesizers and check their fidelity, utility and privacy. For this reason, several different data synthesizers are used in a workflow, which gathers different metrics to evaluate the different aspects.

More information about the data synthesizers and metrics used can be found under https://mkleinegger.github.io/data-synthesizer-evaluation/report.pdf or `./docs/report.pdf`. Furthermore it includes the results and a conclusion about the different data synthesizers used.

The `./notebook.ipynb` contains the workflow aswell as the code to get the metrics and create the blocks. However, the workflow aswell as the results are only discussed briefly and are more explained in the `./report.pdf`.

The original dataset can be found in the directory `./compas` and all the synthetic data is located in `./data`.
Additionaly we saved all the models within the respective directory in `./data`.

## Setup
The notebook installs all the necessary dependencies itself. However they can be also installed via

    pip3 install -r ./requirements
