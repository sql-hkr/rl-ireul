# RL-Ireul

## Implemented

### Model free

- [x] Vanilla DQN
- [x] Double DQN
- [x] Dueling DQN
- [ ] Prioritized DQN
- [x] Noisy DQN
- [ ] Categorical DQN
- [ ] Rainbow DQN

### Model Based

### Imitation

### Multi Agent

## Dependencies

- pytorch-1.12.0
- tensorboard-2.6.0
- tensorboardx-2.2
- gym[atari,accept-rom-license]==0.25.0

## Docker Image

```
docker build -t ireul:0.1 .
docker run --gpus all --rm -it -v $PWD:/workspace --network=host ireul:0.1
```

### jupyter lab

```shell
jupyter lab --allow-root
```

```shell
xvfb-run -s "-screen 0 1400x900x24" jupyter lab
```

## Usage

```shell
python setup.py develop
# or
python setup.py install
```
