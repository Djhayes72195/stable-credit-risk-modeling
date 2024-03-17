### Overview

This file will be used to record my thoughts and objectives during the development
process. This section might feel somewhat disorganized. It exists mostly to track my thoughts
and questions. 

### Feature selection/engineering

There are 466 features available to me. Feature selection
will be important. The starter notebook which I used to begin this project
utilized only static_0_0, static_0_1, person_1 and credit_bureau_b_2. It
is unclear why these particular tables were selected. A systematic method
for performing feature selection would be ideal.

Methods:
- Domain knowledge: I should explore the literature to gather a sense of
which features are certainly important and which can be discarded.
I think it would be best to first gather a list of features to look
for, then track them down in the data.
- Foundational data analysis: Build a script that will examine
each feature in isolation and calculate some simple statistics,
mean, median,  # of missing values, variance, etc. Build a report
to the effect to guide me through the feature selection process.
- More advanced methods: If necessary, try using models with
built in feature selection (LASSO) or Recursive feature elimination.
- Revisit as necessary: Once I have built a model, return
to feature selection and see if adding/removing features will help.


