### Overview

In order to better approach the problem, I will spend some time reviewing
existing literature on credit risk modeling. At this point, I am particularly interested
in feature selection. We include notes relating to the domain of credit risk modeling
as well as feature selection in a more general sense.



### Resource: Credit Scoring by Natasha Mashanovich

https://medium.com/datadriveninvestor/credit-scoring-choose-the-modeling-methodology-right-part-2-13b5abad8ab6

Credit scoring is achieved by many methods. Recent techniques include the utilization of hundreds or thousands of models, ensemble techniques, etc. however, there is a tandout model called the credit scorecard model, based on logistic regression. Easy to build and execute. In fact, scorecard models are sometimes mandaded by regulators  in some countries for their explainability.

A scorecard model conssits of a set of attributes. Within an attribute, weighted points or scores are assigned to each attribute value. The sum of these scores is the final credit score.

Example: Take "age" as the attribute in question. For 0-25, the score is 10, for 26-40 the score is 25, etc.
This value will be added to that of the other attributes to obtain the final score.

External data, such as credit bureau data, dominate internal data for new customers.
Behavioral data, which is gathered from existing customers, has better predictive power than application scorecards.

Different scorecards are used throughout the customers time with the org as more data becomes available.
The new scorecards can be used to set credit limits or interest rates, etc.

Aside: I believe that I have both behavioral and credit data.

![Credit score development steps](CreditScoreDevelopment.png)

Garbage in, garbage out: Data preparation is key. It is also the most challenging and time-consuming part of the process.

- Aggregations, combination of different sources, transformations, cleansing.

Data should be relevant, accruate, consistent and complete, with sufficient and diverse volume.

Data exploration includes both univariate and bivariate analysis and ranges from univariate statistics and frequency distributions to correlations, cross-tabulation, and characteristic analysis.

cross-tabulation: A table, usually 2D, that shows the count of records have some combination of characteristics.

Ex                          | Where do you live?
Favorite Football team      | Kansas City | Denver  |
                Chiefs      |       15    |    3    |
                Broncos     |       5     |    20   |

Characteristic Analysis: Broad term, means lots of things.

Strategies for mitigating bias in feature selection:
    - Collaboration with domain experts.
    - Awareness of any problems in relation to data sources, reliability
    or mismeasurement.
    - Cleaning the data.
    - Using control variables to account for banned features or 
    specific events such as economic drift.

Feature selection is an iterative process that starts before you have built a model and continues
through the whole project. 

In credit risk modeling, two commonly used techs are information value and stepwise selection.

Information value: 

- Put constinuous variables into bins
- Calc porportion of defaults in each bin.
- Calc Weight of Evidence (WOE) for each bin
    - WOE = ln((Proportion of no-default)/(Porportion of default))
- Calc information value:
    - IV = sum_{all_bins}{(Prop of Goods - Prop of bads)(WOE)}

Note: Add IV to my feature breakdown.


Model building:

Note: This article treats logistic regression as the default model.

Since logistic regression is an additive model, special variable transformations are required.

- Fine classing
- Coarse classing
- Dummy coding or WOE transformation

These transformation enhance model explainability and
assist in acheiving a linear relationship between the features
and the target.

Binning is sometimes required due to regulators requirement of model explainability.
There is a thing called optimal binning which seeks bins that maximize predictive power.

Dummy coding: The process of creating binary features that map to our categorical variables.
A large number of dummy variables costs compute and may lead to overfitting and
interpretability issues. 

WOE transformation: More favored alternative to dummy coding. Substitutes
each coarse class with a risk value, then collapses all risk values into a single numerical variable.
Particularly well suited for logistic regression as both are based on log-odds.

Note: Optimal binning, dummy coding and WOE transformation are a time consuming process.
It is best to find a package that can do this for me.

scorecardpy can do this for me.

After transformation are carried out, it is good practice to check if
you features are still good. Good feature are ones that:

 (i) have higher information value (usually between 0.1 and 0.5), (ii) have a linear relationship with the dependent variable, (iii) have good coverage across all categories, (iv) have a normal distribution, (v) contain a notable overall contribution, and (vi) are relevant to the business.

Later, you scale the model so that the outputs fall into a particular range and have particular meaning.

For instance, you might set a base score of 500, and say that for every 50 points above that the risk of default halves.


Once the model is built we calculate validation metrics: Accuracy, complexity,
error rate, ROC curve.

ROC curve: Plot sensitivity (recall) at different thresholds.
Area under ROC curve is a useful measure. .75 is industry standard.
