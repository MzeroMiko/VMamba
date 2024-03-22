this is a fork of [s4nd](https://github.com/state-spaces/s4)

```bash
conda create -n s4 --clone vmamba
pip install pytorch_lightning timm==0.5.4 hydra-core
cd extensions/kernels/ && python setup.py install
```

change convnexts4nd/src/models/sequence/modules/s4nd.py#233:
    "int(l_i if l_k is None else min(l_i, round(l_k / rate))) for l_i, l_k in zip(L_input, self.l_max)"

