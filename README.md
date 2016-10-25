# pFTA
probabilistic First-Take-All feature for human activity recognition from sensor signals. This is the matlab source code for the pFTA feature and the probabilistic Temporal Order Encoding Algorithm proposed in the paper

"Learning Compact Features for Human Activity Recognition via Probabilistic First-Take-All."

# Instruction

1) A preprocessed Smpartphone-based Human Activity Dataset is provided to repeat the experiment presented in the paper.The original Dataset can be found at https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones In the experiment, the 30 subjects of the dataset are evenly split into 5 groups and the subject-based Leave-one-out cross-validation is employed for the performance evaluation. The final result is the average of the results of the above 5 runs.  Subject grouping policy is as follows,

Group     Subjects

-------------------------

Group1: 1 2 3 4 5 6

Group2: 7 8 9 10 11 12

Group3: 13 14 15 16 17 18

Group4: 19 20 21 22 23 24

Group5: 25 26 27 28 29 30


2) Run the scriptRepeatExperimentResult.m to get pFTA classification results based on the optimized projections.

3) Run the scriptAutoTestRand.m to get the pFTA classification results based on random projection.

4) To repeat the results on the random projection, please set the rand seed as "default".

5) To train your own projections, run ScriptAutoTrain.m

6) The mtimesx tool is needed to run the algorithm to compute the multiplication of ND Array in matlab. I have enclosed this package in the source.



# Copyright

Copyright (c) Jun Ye. 2016.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
