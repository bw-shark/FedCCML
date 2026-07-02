$ErrorActionPreference = "Stop"

python -m fedccml.prepare_data `
  --dataset Cifar100 `
  --data-root data `
  --num-clients 20 `
  --alpha 0.1 `
  --seed 1

python -m fedccml.main `
  --dataset Cifar100 `
  --data-root data `
  --output-root results `
  --model CNN `
  --global-rounds 300 `
  --local-epochs 5 `
  --batch-size 32 `
  --local-learning-rate 0.005 `
  --server-learning-rate 0.005 `
  --alpha 20 `
  --beta 1 `
  --eval-gap 10 `
  --goal dir01
