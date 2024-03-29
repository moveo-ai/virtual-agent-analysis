# Virtual Agent Analysis

Analyze the behavior and performance of Moveo.AI virtual agents.

## Prerequisites

- [Python 3.10](https://www.python.org/downloads/)
- [pipenv](https://pipenv.kennethreitz.org/en/latest/): Manages a virtual environment and pip dependencies

## Getting Started

1. Install Python 3.10.

   ```bash
   pyenv install 3.10
   pyenv global 3.10
   ```

2. Clone the repo.

   ```bash
   git clone git@github.com:moveo-ai/virtual-agent-analysis.git
   cd virtual-agent-analysis
   ```

3. Install dependencies.

   ```bash
   make install-dev
   ```

4. Generate a `.env` file at the project's root by duplicating the contents of `.env.example`. Then, populate the file with your specific values.
5. Start JupyterLab

   ```bash
   make jupyter
   ```

6. Select the notebook you are interested in.
7. Add your data (only CSV format currently supported) in the `/data` directory.
8. Run the notebook.

## Used for creating the graphs

https://plotly.com/python/

## Contributing

### To ensure that your commits [exclude any notebook outputs](https://gist.github.com/33eyes/431e3d432f73371509d176d0dfb95b6e) while contributing to the project, execute the following command within your terminal, in the project's root directory:

```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

### The project uses a Makefile.

To see all the available commands:

```bash
make help
```
