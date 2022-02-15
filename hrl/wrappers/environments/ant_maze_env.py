# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from .maze_env import MazeEnv
from .ant import AntEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv

    @property
    def env(self):
        return self.wrapped_env

    def get_dataset(self):
        return {
            "observations": [
                (-10., -10.), (10., 10.)
            ]
        }

    def step(self, action):
        return super().step(action*30.)