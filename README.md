1) Each boolean gets 5% chance of being chosen, the Shap value pushes this up or down based on custom logic
2) All mappings etc leave the original features alone
3) For algorithms it is True/False too and if there are many Trues, then they are averaged together
4) There are also booleans for the type of ensemble combination technique (forced False if there is only one algorithm chosen)
5) Mappings etc are only ever done on the original features

(removal techniques: KBest, dropout - one challenge is might need as many columns as there are new features)
















JUST IDEAS BELOW
================

scaling, outliers, missing values, removing, compressing, expanding, combining, categorical encoding, unioning, ordering, reframing, sampling, generating, algorithm choice, and algorithm parameters

one of each
null allowed (requires checking and logically acting on null options)
instead of algorithm choice as a parameter, instead, use the hyperparameter field (no true/false using algorithm)
why not allow a lot of each one (it got a worse score, why? Takes more of a search space?)
quickest v1 is lots of booleans right?
If it could work quickly by doing one step at a time it can lock in and save the transformation steps.
Maybe random scaling only, then lock in, then do the same for outlier removal, etc
if null error etc, then in case of error set to average value and try again
The search space becomes too large too fast
A random search should give a good enough answer, especially if improving over time
*******
- I liked the bayes idea.  Each gets small chance of being chosen and that goes up and down based on shap results each batch
- deal with removing at the same time
- New variables for each True
- compressing always done with original features (internally do standard scale before pca etc)
- Implement all the scalng ones
- Allow nulls? yes.   Allow Only true/false?  Yes.
*******
xgboost is good no worries and batch it more

which are allowed to be null?
booleans or one of each type?
one shouldn't destroy another
can some be done univariately?
Simple all together? probably


END:
it should choose right combination that gives the best score
What would give the best combination?
Would it be 2 types of scalings or one?
Would it have booleans only or numerics?



# shap_scaling_experiments

### Question



### Hypothesis



### Data Sources



### Data Descriptions



### Steps



### Conclusion




