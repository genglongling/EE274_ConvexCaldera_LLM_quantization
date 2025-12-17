## Convex-CALDERA: LLM Weight Compression via Convex Optimization with Low-Rank and Low-Precision Factor

**This code is based on the Lambda Labs platform's Launch Instance (GPU: NVIDIA H100 PCIe, Image: Lambda Stack 22.04).**

#### SCL Baselines

In the `scl` directory, there are three scripts: `scl_llama2_7b_quant.py`, `scl_llama2_13b_quant.py`, and `scl_llama3_8b_quant.py`, each corresponding to a different model. Taking the `llama2-7b` model as an example, to run

##### SCL–Scalar Quant (8-bit) 

```shell
python scl_llama2_7b_quant.py --method scalar_uniform
```

##### SCL–Lloyd–Max Quant (8-bit)

```shell
python scl_llama2_7b_quant.py \
  --method lloyd_max \
  --sample_size 200000 \
  --num_iters 25
```

##### SCL–Vector Quant (8-bit VQ)

```shell
python scl_llama2_7b_quant.py \
  --method vector_vq \
  --block_size 4 \
  --sample_size 200000 \
  --num_iters 25
```

The above commands will create a folder, e.g., `llama2_7b_scl_scalar8/`, and then run the following command to evaluate:

```shell
python eval_scl_llama2_7b.py --model_dir llama2_7b_scl_scalar8/
```

Note: Replace the folder name after `--model_dir` accordingly. Meanwhile, in `eval_scl_llama2_7b.py`, within the `TASKS` list:

```python
TASKS = [
    "wikitext",
    "c4",
    "winogrande",
    "rte",
    "piqa",
    "arc_challenge",
]
```

Select the metrics you wish to evaluate. If you choose `c4`, it is recommended to add the parameter `limit=1000` in the `evaluator.simple_evaluate()` call. For other metrics, this parameter can typically be omitted. This adjustment prevents the evaluation of `c4` from becoming excessively slow.

#### Convex-CALDERA (rank-4096)

In the `repo` directory, run

```shell
python convex_caldera_example.py
```

Note: In `convex_caldera_example.py`, select the corresponding model.

It will output the relevant compression information in the console, and generate the `llama2_7b_convex_caldera_quant` folder (as an example), as well as produce images within the `plots` folder. Afterward, copy the `llama2_7b_convex_caldera_quant/` folder into the `scl` directory, or modify the path accordingly, and use the previous `eval_scl_llama2_7b.py` method to evaluate.