# Tests

This folder contains tests for the different routines implemented in the project, namely:

- Simulation paremeters reading (SPR).
- Molecular dynamics reading (MDR).
- Diffusive information extractions (DIE).

In order to find all tests from the home directory of the project (parent directory of tests) just run:

```bash
python3 -m unittest discover -v
```

and to run, for instance SPR tests, execute:

```bash
python3 -m unittest tests.test_SPR -v
```
