# keras-rl

### prerequisite

```python
pip install keras-rl2
pip install gym
pip install h5py
```

- for atari

```python
pip install Pillow
pip install gym[atari]
```

### have-to-do things

It should be added in import line or not correctly be implemented.

```python
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
```

### reference

- in gym environment, 1 episode consists of 10000 steps
- [pre-built environment set](http://gym.openai.com/envs/#classic_control)

- [keras-rl2]()

