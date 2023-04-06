# smell-datasets

## Raw Data
Raw measurements from the Smell Annotator are found in the "raw_data" directory.

## Dataset Parser
The dataset parser combines the raw files, eliminates unused channels, and prepares a dataset for model training.

Output support has been added to product timeseries datasets.
```
python dataset_gen.py 
```

### Data Visualization
The dataset visualization code implements t-distributed stochastic neighbor embedding (t-SNE) and plots information on a scatter plot.
```
python visual_gen.py 
```

![Data_Visualization](smell_dataset_viz.png)

```
{
    "0": "ambient air",
    "1": "ambient room",
    "2": "cocacola",
    "3": "cocacola cold",
    "4": "dasani water",
    "5": "dasani water cold",
    "6": "motts applejuice",
    "7": "motts applejuice cold",
    "8": "pureleaf sweettea",
    "9": "pureleaf sweettea cold",
    "10": "redbull",
    "11": "redbull cold",
    "12": "starbucks dark coffee cold",
    "13": "starbucks dark roast coffee",
    "14": "starbucks dark roast coffee hot"
    
}
```

## Models 
Pretrained models can be found in the "models" directory

