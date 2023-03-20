# Tests

This folder contains tests for the different routines implemented in the project, namely:

- Identification of diffusive paths (ID).
- Analysis of correlations (AC).
- Analysis of atomistic descriptors (AD).

being each subroutine tested in the corresponding file.

In order to find all tests from the home directory of the project (parent directory of Tests) just run:

```bash
python3 -m unittest discover -v
```

and to run, for instance ID tests, execute:

```bash
python3 -m unittest tests.test_ID -v
```
