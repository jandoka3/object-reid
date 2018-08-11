# Identity tracking in multi-view camera surveillance system 

The provided code was developed using the following software:
- Python 3.4
- tensorflow-gpu 1.6.0rc1
- scikit-learn 0.19.1

An MSI GeForce GTX 1070Ti and an MSI GeForce GTX 1060 OC GPU were used for training and testing the models.

# Datasets and trained models

For your convenience, all datasets, files and model weights which are needed to evaluate our models are zipped into a single folder. The file can be downloaded from [this link](https://drive.google.com/open?id=1ZqpSeq-8r5_WcH8In99wwPOclGY56JQA).

To ensure the code will run smoothly, please extract the above file in the same folder as the `test.py` file. Your folder structure should then look like this:

    .
    ├── ...               
    ├── datasets                
    │   ├── Cuhk03_detected          
    │   ├── Cuhk03_labeled 
    │   ├── Market1501
    │   ├── VehicleReId
    │   └── VeRi 
    ├── excluders                
    ├── nets               
    ├── test.py                
    ├── evaluate.py             
    └── ...


# Running the evaluation

Once the zip file has been downloaded and extracted, the evaluation code is ready to run. All models can be evaluated by running the `test.py` file with different arguments. We explain in detail:

- dataset: This argument represents the model that was trained on a specific dataset. The following values are available:
  - Cuhk03_detected
  - Cuhk03_labeled
  - Market1501
  - VehicleReId
  - VeRi
  
  Please make sure that the argument is capitalised as above.

- re_rank: This argument is used for performing re-ranking as a post-processing step. It holds no other value and can be omitted in order to obtain the results without re-ranking. Please note that re-ranking is only available for image-to-image re-identification.

- image_to_track: This argument is used for image-to-track testing and is only available for the `VeRi` and `VehicleReId` datasets. Similarly as above, it holds no other value and should be omitted for image-to-image evaluation.

- selector: This argument is used to define the metric that will be used for selecting a representative image of a track. It is only available for image-to-track evaluation and it holds the following values: `mean`, `max`. Please note that in case you omit this argument while `image_to_track` is chosen, the `mean` parameter will be selected by default.

- gpu: Optional argument for running the script on a specific GPU. Permitted values include every positive integer. However if no GPU is matched with the given number, the CPU will be used instead. 

## Examples

If you wish to perform an image-to-image evaluation on the `Market-1501` dataset with re-ranking, run the `test.py` script as follows:

```
python test.py \
    --dataset Market1501 \
    --re_rank
```

For evaluating our trained model on the `VeRi` dataset in the image-to-track scenario with the image of maximum similarity being chosen as the track's representative sample, you can run the following:

```
python test.py \
    --dataset VeRi \
    --image_to_track \
    --selector max
```

# Output files

Upon termination, `test.py` will print the final results on the console and also create two files in a folder named `Results`. That folder will be located inside the corresponding dataset. For example, for the `Market-1501` dataset, the output files will be located under `./datasets/Market1501/results/`.
The files will be named according to the arguments that were given, but will always have the following endings:
- `*_evaluation.json`: This file includes the mAP metric and the CMC metrics.
- `*_ranked_images.csv`: This file contains the paths of the first 10 ranked images for every query. More specifically, it has as many rows as query images and 11 columns. The first column represents the path to the query image and the rest of the columns show the paths to each of the rank-10 images respectively.

Finally, an image will be produced for qualitative evaluation. That image comprises a set of different sub-figures as follows: The first column of each row depicts the probe image and the following 10 columns show the corresponding top 10 ranked images. Each row represents a query image of a unique identity to avoid showing duplicates when multiple queries of the same identity are considered. Green borders represent ground truths while red borders represent an identity different from that of the probe. 
 
# Performance discrepancies 

Please note that the calculation of the mAP score is based on the scikit-learn library and performance may deviate on other versions.
