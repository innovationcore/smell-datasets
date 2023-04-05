# smell-datasets

## Raw Data
Raw measurements from the Smell Annotator are found in the "raw_data" directory.

## Dataset Parser
The dataset parser combines the raw files, eliminates unused channels, and prepares a dataset for model training.
```
python dataset_gen.py 
```

### Data Visualization
The dataset visualization code implements t-distributed stochastic neighbor embedding (t-SNE) and plots information on a scatter plot.
```
python visual_gen.py 
```

![Data_Visualization](smell_dataset_viz.png)


## Models 
Pretrained models can be found in the "models" directory

