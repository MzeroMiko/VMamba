Install the Cauchy and Vandermonde CUDA kernels:
```
python setup.py install
```

(Optional) Test the extensions
```
pytest -q -s test_cauchy.py
pytest -q -s test_vandermonde.py
```
