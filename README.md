# pFTA
probabilistic First-Take-All feature for human activity recognition from sensor signals. This is the matlab source code for the pFTA feature and the probabilistic Temporal Order Encoding Algorithm proposed in the paper

"Learning Compact Features for Human Activity Recognition via Probabilistic First-Take-All."

# Instruction

A preprocessed Smpartphone-based Human Activity Dataset is provided to repeat the experiment presented in the paper.
The original Dataset can be found at https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones


Run the scriptRepeatExperimentResult.m to get pFTA classification results based on the optimized projections.

Run the scriptAutoTestRand.m to get the pFTA classification results based on random projection.

To repeat the results on the random projection, please set the rand seed as "default".

To train your own projections, run ScriptAutoTrain.m


# Copyright

Copyright (c) Jun Ye. 2016.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
