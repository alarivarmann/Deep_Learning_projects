## Introduction and Performance

This is the demonstration notebook for E-Commerce Data (Re)-Mapper developed by Alari Varmann from Integrify.
How to run : To see the application running without human intervention, the Docker container has been built, in which you can choose between 3 models.
Right now, this will be asked from the user in the terminal, but ArgumentParser could also be used.
The models may predict wrong sometimes many in a row even, but it has been double validated on the existing data that the **validation set performance** of all of them should be **above 92%**, while the ones using the processed features have performance of around **98%+** for all the most common category tree paths.



### Performance
Right now, there may be some performance slowdown issues, depdendent on the environment. 
The runtime speed of predictions in Local conda environment has been around 3 seconds per iteration, or around 0.3 iterations per second.
However on `ml.p2.xlarge` instance on `AWS`, the typical performance should be between 4-10 iterations per second, so at least 10 times faster.
This is strange because the machine tested on locally is Octa-core machine, and the container uses the CPU version of Pytorch.
For the sake of simplicity and saving of the resources, more experiments on runtime environments should be conducted.

UPDATE : The Docker container has been run on `ml.p2.xlarge` instance and the speed was around 2-3 iterations per second. Thus, a potential gain of around two times could be achieved.
In case of wishing higher performance, one potential bug in the original source library should try to be fixed (please check the end of the document).

## How to use the Predict Function 
The predict function used in this case is a local method for `RNNLearner` which is a Recurrent Neural Network model class.
It is documented [here](https://docs.fast.ai/text.learner.html#LanguageLearner.predict)
```python
predict(text:str, n_words:int=1, no_unk:bool=True, temperature:float=1.0, min_p:float=None, sep:str=' ', decoder='decode_spec_tokens')
``` 

### Run all cells up to the cell test_data_full.head() 
#### Predictions from Model 1 : 6_full (6 full processed features, version 1)
**and then come back here to run**:


```python
featurerow = test_data_full.iloc[0]
test_data_full.iloc[0]


```




    combined_name_description    mille notti-oceano aluslakana 270x270cm, hiekk...
    cleaned_description          oceano aluslakana 270x270cm, hiekka från mille...
    brand                                                              Mille Notti
    original_tree                Tekstiilit & Matot|Makuuhuonetekstiilit|Alusla...
    provider                                                                 Rum21
    summary                                                                       
    Name: 440838, dtype: object



```python
def get_one_prediction(model, feature_row):
    """This function returns one prediction per Pandas Dataframe row"""
    one_prediction = model.predict(feature_row)[0].obj
    return one_prediction
``` 


```python
print("Test data for FULL model  read and loaded!")
#final_learner_2.data.add_test(featurerow)
print("Test data added to the final model")



get_one_prediction(model=final_learner_2, feature_row=featurerow)
```

    Test data for FULL model  read and loaded!
    Test data added to the final model





    569




```python
final_learner_2.predict(featurerow)[0]
```




    Category 569




```python
final_learner_2.predict(featurerow)[0].obj
```




    569



**This is how the predict function can be used**

### Step-by-Step Guide to Process The Data and Get Predictions

> 0. Imports and Set Raw (Products) Data Location Definition
```python
import pandas as pd
import os
from fastai import *
from fastai.text import *
ROOT = os.getcwd()
products_path = os.path.join(ROOT,"data","products.csv")
```

> 1. Run the Utility Functions Cell (Next cell)



```python
!pip install tqdm
```

### 1.1 Getting Models. 

Download the Databunch only if you wish to validate the predictions on the whole validation sets yourself



### IMPORTANT : FIRST CREATE "data" , "models" and "bunches_full" folders to the root of the project and download the products.csv file into data folder
### Then download all the models and name them this way:
ROOT = os.getcwd()

LOCAL_MODEL_PATH = os.path.join(ROOT,"models")

cl_bunch_path = os.path.join(ROOT,"bunches_full","latest_cl_bunch_FULL_bs48")

cl_bunch_path_remote_path="https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/databunches/latest_cl_bunch_FULL_bs48"

final_model_path_2= os.path.join(LOCAL_MODEL_PATH,"final_model_6_full")
final_model_path_2_remote = "https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/e_commerce_predictor/models/6_features_80pc_final.pkl"
 
 
final_model_path_1= os.path.join(LOCAL_MODEL_PATH,"pre_model_6_full")
final_model_path_1_remote = "https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/e_commerce_predictor/models/6_features_80pc.pkl"
   
   
raw_model_path = os.path.join(LOCAL_MODEL_PATH,"6_raw_features_85pc") 
raw_model_path_remote = "https://filedn.com/lK1VhM9GbBxVlERr9KFjD4B/ecommerce/e_commerce_predictor/models/6_raw_features_85pc.pkl"


### 1.2 RUN Utility Functions for Preparing Data for Prediction


```python
from tqdm import tqdm, tqdm_notebook
import ntpath
import pandas as pd
import os
from fastai import *
from fastai.text import *
#ROOT = "/home/ec2-user/SageMaker/e_commerce_hierarchical/e_commerce_project/"



ROOT = os.getcwd()
products_path = os.path.join(ROOT,"data","products.csv")
LOCAL_MODEL_PATH = os.path.join(ROOT,"models")



def path_leaf(path):
    """This function gives the leaf of a path"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def subtract(a, b):
    """This function subtracts two paths, used for loading purposes"""
    return "".join(a.rsplit(b))

def produce_doublet_from_path(path):
    """Function where the last two functions are used. It splits one path into two
    from the leaf. """
    name = path_leaf(path)
    base = subtract(path,name)
    print(f"Base path is {base}")
    print(f"Object name is {name}")
    return name,base


def replace_global_missing_features_with_empty(df, feature_names):
    """This function replaces missing, NAN or empty values in Pandas dataframe with empty string"""
    df.loc[:, feature_names] = df.loc[:, feature_names].replace(np.nan, '', regex=True).fillna("")
    return df

def select_globally_defined_features_and_target(data, label_name,feature_names):
    """Selects features and labels in Pandas Dataframe by their name(s)"""
    labels = data.loc[:, label_name]
    features = data.loc[:, feature_names].reset_index(drop=True)
    return features, labels


def drop_if_exists(df, cols):
    """Drops a column in Pandas dataframe only if it exists there"""
    for col in cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df

def text_cleaner(text):
    """This is the only actual preprocessing function used. It's a list of regex rules,
    each rule is applied one-by-one, then whitespace is trimmed and text is lower cased."""
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.rstrip()
    return text.lower()

def preprocessing_for_prediction(df, target_category, model_ver):
    """ This function generates data, target_category name and feature_names list \
    for to be used in the prediction function later. All features are of string data type!"""
    df = df.loc[:, ["name", "description", "summary", "original_tree", "brand", "provider", target_category]]
    # the features above are all the features defined in the 6_raw model
    if model_ver == "6_raw":
        feature_names = df.columns.tolist()
        feature_names.remove(target_category)
        print(f"Feature names are {feature_names}")
        df.loc[:, target_category] = df.loc[:, target_category].astype(int)
        print("Production mode running on trivial NAN -> "" preprocessing.")
        print(f"Before exiting step 1, the columns are {df.columns.tolist()}")
        print(f"""Check : the features have to be the following : \n 
        name, description, summary, original, tree, brand, provider""")
    elif model_ver == "6_full":
        df["combined_name_description"] = df.progress_apply(
            lambda row: str(row["name"]).lower() + str(row["description"]), axis=1)
        df["combined_name_description"] = df.progress_apply(lambda row: text_cleaner(str(
            row["combined_name_description"]).lower()), axis=1)
        df = df.rename(columns={"description": "cleaned_description"}, inplace=False)
        df = drop_if_exists(df, cols=["name"])
        df["cleaned_description"] = df.progress_apply(lambda row: text_cleaner(str(row["cleaned_description"]).lower()),
                                                      axis=1)
        df = df.loc[:,
             ["combined_name_description", "cleaned_description", "brand", "original_tree", "provider", "summary",
              target_category]]
        feature_names = df.columns.tolist()
        feature_names.remove(target_category)
    df = replace_global_missing_features_with_empty(df, feature_names=feature_names)
    print(f"The 6 feature model has columns like {df.columns.tolist()}")

    return {"data": df, "target_category": target_category, "feature_names": feature_names}



```

> 2. Use the function `load_test_data` to prepare the data from model version and raw data path. This is defined in the next cell.

 ```python
def load_test_data(model_ver,path=products_path):
    """The idea of this function is to prepare all the data for testing the prediction model.
    The input arguments are model_ver (model version, either 6_full or 6_raw and
    path of the raw Pandas dataframe with the products.
    It returns both the data to be tested (Pandas Dataframe) and the Series of the target column (mapped_id)""")
    #if amountlines != -1:
    #    rows_to_keep = list(range(amountlines))
    #    df = pd.read_csv(path, skiprows=lambda x: x not in rows_to_keep)
    df = pd.read_csv(path)
    test_data = df.sample(1000)
    print("Test data sampled, starting preprocessing")
    tqdm_notebook().pandas()
    preproc_dict=preprocessing_for_prediction(df=test_data, target_category="mapped_id", model_ver=model_ver)
    test_data = preproc_dict["data"]
    print("Test data preprocessed")
    test_target = test_data.loc[:,"mapped_id"]
    test_data = test_data.drop("mapped_id",axis=1) #drop_if_exists(test_data,"mapped_id")
    return test_data,test_target

```



```python
import pandas as pd
import os
from fastai import *
from fastai.text import *
from tqdm import tqdm, tqdm_notebook

#ROOT = "/home/ec2-user/SageMaker/e_commerce_hierarchical/e_commerce_project/app"
ROOT = os.getcwd()
products_path = os.path.join(ROOT,"data","products.csv")

def load_test_data(model_ver,path=products_path,subsample=None):
    """The idea of this function is to prepare all the data for testing the prediction model.
    The input arguments are model_ver (model version, either 6_full or 6_raw and
    path of the raw Pandas dataframe with the products.
    It returns both the data to be tested (Pandas Dataframe) and the Series of the target column (mapped_id)"""
    #if amountlines != -1:
    #    rows_to_keep = list(range(amountlines))
    #    df = pd.read_csv(path, skiprows=lambda x: x not in rows_to_keep)
    df = pd.read_csv(path)
    if subsample is not None:
        test_data = df.sample(subsample)
    print("Test data sampled, starting preprocessing")
    tqdm_notebook().pandas()
    preproc_dict=preprocessing_for_prediction(df=test_data, target_category="mapped_id", model_ver=model_ver)
    test_data = preproc_dict["data"]
    print("Test data preprocessed")
    test_target = test_data.loc[:,"mapped_id"]
    test_data = test_data.drop("mapped_id",axis=1) #drop_if_exists(test_data,"mapped_id")
    return test_data,test_target

test_data_full,test_target = load_test_data(model_ver="6_full",path=products_path,subsample=1000)


```

    /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2903: DtypeWarning: Columns (1,4) have mixed types. Specify dtype option on import or set low_memory=False.
      if self.run_code(code, result):


    Test data sampled, starting preprocessing



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=1000), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=1000), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=1000), HTML(value='')))


    
    The 6 feature model has columns like ['combined_name_description', 'cleaned_description', 'brand', 'original_tree', 'provider', 'summary', 'mapped_id']
    Test data preprocessed


>2. Initialize the paths and Load the fast.AI prediction model

```python
def test_learner_loading(path=final_model_path):
    """Given the path for the fast.AI prediction model, it loads and returns it"""
    name,base = produce_doublet_from_path(path)
    loaded_model = load_learner(path=base,file=f"{name}.pkl")
    return loaded_model
```




```python
import os
from fastai.text import *

def test_learner_loading(path):
    """Given the path for the fast.AI prediction model, it loads and returns it"""
    name,base = produce_doublet_from_path(path)
    loaded_model = load_learner(path=base,file=f"{name}.pkl")
    return loaded_model

LOCAL_MODEL_PATH = os.path.join(ROOT,"models")

final_model_path= os.path.join(LOCAL_MODEL_PATH,"final_model_6_full")
final_model_path_2= os.path.join(LOCAL_MODEL_PATH,"final_model_6_full")
final_model_path_1= os.path.join(LOCAL_MODEL_PATH,"pre_model_6_full")
raw_model_path = os.path.join(LOCAL_MODEL_PATH,"6_raw_features_85pc") 

print("databunch loaded")
final_learner_2 = test_learner_loading(path=final_model_path_2)
final_learner_1 = test_learner_loading(path=final_model_path_1)
raw_feature_learner = test_learner_loading(path=raw_model_path)

print("All models loaded")

```

    All models loaded


>3. Load the test data and use the following function to obtain the predictions
```python
obtain_test_predictions(model=final_learner_2, features=test_data_full, how_many_preds=1000)
```


```python
import os
from tqdm import tqdm


def get_one_prediction(model, feature_row):
    """This function returns one prediction per Pandas Dataframe row"""
    one_prediction = model.predict(feature_row)[0].obj
    return one_prediction


def obtain_test_predictions(model, features, how_many_preds=1000):
    """This function applies the prediction function to the features Dataframe, row by row"""
    featurecount = features.shape[1]
    features = features.reset_index(drop=True)
    for i, feature_row in tqdm(features.iterrows(), total=features.shape[0]):
        prediction = get_one_prediction(model, feature_row)
        print(f"Prediction {i} is {prediction}")
        if i == how_many_preds:
            break

```

#### Predictions from Model 1 : 6_full (6 full processed features, version 1)


```python
test_data_full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>combined_name_description</th>
      <th>cleaned_description</th>
      <th>brand</th>
      <th>original_tree</th>
      <th>provider</th>
      <th>summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>440838</th>
      <td>mille notti-oceano aluslakana 270x270cm, hiekk...</td>
      <td>oceano aluslakana 270x270cm, hiekka från mille...</td>
      <td>Mille Notti</td>
      <td>Tekstiilit &amp; Matot|Makuuhuonetekstiilit|Alusla...</td>
      <td>Rum21</td>
      <td></td>
    </tr>
    <tr>
      <th>49873</th>
      <td>sheer tint moisture spf20 light 40 mluv-suojan...</td>
      <td>uv-suojan sisältävä sheer tint moisture öljytö...</td>
      <td>Dermalogica</td>
      <td>IHONHOITO|Kasvot|Ihotyypit|Normaali iho</td>
      <td>Cocopanda</td>
      <td></td>
    </tr>
    <tr>
      <th>549708</th>
      <td>1:35 bm-21 grad multiple rocket launcherthe bm...</td>
      <td>the bm-21 grad is a russian truck-mounted 122 ...</td>
      <td>Trumpeter</td>
      <td>Pienoismallit|Pienoismalli maakalusto|Maakalus...</td>
      <td>Hobbylinna</td>
      <td>The BM-21 Grad is a Russian truck-mounted 122 ...</td>
    </tr>
    <tr>
      <th>556448</th>
      <td>sl amr 4.7 nx eagle 1x12 19, täysjousitettu ma...</td>
      <td>tehokas täysjousitus ja kevyt alumiinirunko. 1...</td>
      <td>Ghost</td>
      <td>Pyöräily|Polkupyörät|Maastopyörät</td>
      <td>XXL</td>
      <td></td>
    </tr>
    <tr>
      <th>67703</th>
      <td>pvc-housut or28507701021kuvaushousuissa on pai...</td>
      <td>kuvaushousuissa on painonappi, vetoketju, vyön...</td>
      <td>Black Level</td>
      <td>PVC Housut|Black Level</td>
      <td>Antishop.fi</td>
      <td>Kapealahkeiset pvc-housut</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Test data for FULL model  read and loaded!")
final_learner_2.data.add_test(test_data_full)
print("Test data added to the final model")

obtain_test_predictions(model=final_learner_2, features=test_data_full, how_many_preds=10)
```

      0%|          | 1/1000 [00:00<02:18,  7.22it/s]

    Test data added to the final model
    Prediction 0 is 5422


      0%|          | 3/1000 [00:00<02:33,  6.50it/s]

    Prediction 1 is 5598
    Prediction 2 is 2580


      0%|          | 5/1000 [00:00<02:11,  7.54it/s]

    Prediction 3 is 778
    Prediction 4 is 5322
    Prediction 5 is 5598


      1%|          | 7/1000 [00:00<02:04,  7.95it/s]

    Prediction 6 is 5181
    Prediction 7 is 2901


      1%|          | 9/1000 [00:01<02:06,  7.80it/s]

    Prediction 8 is 201
    Prediction 9 is 187


      1%|          | 9/1000 [00:01<02:31,  6.55it/s]

    Prediction 10 is 187


    


#### Predictions from Model 2 : 6_full (6 full processed features, version 2)


```python
print("Test data for FULL model read and loaded!")
final_learner_1.data.add_test(test_data_full)
print("Test data added to the final model")

obtain_test_predictions(model=final_learner_1, features=test_data_full, how_many_preds=10)
```

      0%|          | 1/1000 [00:00<02:20,  7.11it/s]

    Test data added to the final model
    Prediction 0 is 5422


      0%|          | 3/1000 [00:00<02:34,  6.45it/s]

    Prediction 1 is 5598
    Prediction 2 is 2580


      0%|          | 5/1000 [00:00<02:12,  7.50it/s]

    Prediction 3 is 778
    Prediction 4 is 5322
    Prediction 5 is 5598


      1%|          | 7/1000 [00:00<02:04,  7.98it/s]

    Prediction 6 is 5181
    Prediction 7 is 2901


      1%|          | 9/1000 [00:01<02:06,  7.86it/s]

    Prediction 8 is 201
    Prediction 9 is 187


      1%|          | 9/1000 [00:01<02:29,  6.61it/s]

    Prediction 10 is 187


    


### And the Right Answers Are 


```python
test_target[0:11].values.tolist()
```




    [5422, 5598, 2580, 778, 5322, 5598, 5181, 5609, 201, 187, 187]



*Explanation : Since we don't have any new test data, it is possible that the model has seen these or similar data points during training*.

Test the model with data points that are similar to the ones used in the dataset to get realistic estimates.
Although the model will always give predictions, even if some columns are empty, it should be taken into account that if the predictions are off in this case, it might be partially caused by the missing column values, for which case they could be tried to be imputed.

Sometimes even simple imputations can work, in other times even deep learning can be used for imputations.


(*I have built full imputation pipelines before over nonrelational databases*)
    

#### Predictions from Model 3 : 6_raw (6 raw features without preprocessing, except Nan->"" )


```python
test_data_raw,test_target2 = load_test_data(model_ver="6_raw")
print("Test data for RAW model read and loaded!")
test_data_raw.head()
```

    /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2903: DtypeWarning: Columns (1,4) have mixed types. Specify dtype option on import or set low_memory=False.
      if self.run_code(code, result):


    Test data sampled, starting preprocessing



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    Feature names are ['name', 'description', 'summary', 'original_tree', 'brand', 'provider']
    Production mode running on trivial NAN ->  preprocessing.
    Before exiting step 1, the columns are ['name', 'description', 'summary', 'original_tree', 'brand', 'provider', 'mapped_id']
    Check : the features have to be the following : 
     
            name, description, summary, original, tree, brand, provider
    The 6 feature model has columns like ['name', 'description', 'summary', 'original_tree', 'brand', 'provider', 'mapped_id']
    Test data preprocessed
    Test data for RAW model read and loaded!





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>description</th>
      <th>summary</th>
      <th>original_tree</th>
      <th>brand</th>
      <th>provider</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>329725</th>
      <td>Jack &amp; Jones Jean Jacket Cj 077 Farkkutakki Me...</td>
      <td>- Farkkutakki kahdella rintataskulla. Denim ei...</td>
      <td></td>
      <td>Clothing|Jackets|Denim Jacket</td>
      <td>Jack &amp; Jones</td>
      <td>Jack &amp; Jones</td>
    </tr>
    <tr>
      <th>211611</th>
      <td>Elliot Shirt Paita Bisnes Harmaa Bruuns Bazaar</td>
      <td>Bruuns Bazaar Elliot Shirt</td>
      <td></td>
      <td>Men|Shirts</td>
      <td>Bruuns Bazaar</td>
      <td>Boozt</td>
    </tr>
    <tr>
      <th>429815</th>
      <td>System Professional - Sp Refined Texture Model...</td>
      <td></td>
      <td></td>
      <td>Hiustuotteet|Muotoilu|Muotoiluvoiteet</td>
      <td>Wella</td>
      <td>Nordicfeel</td>
    </tr>
    <tr>
      <th>392760</th>
      <td>Beamz Uskomaton Discosieni Kaukosäätimellä</td>
      <td>Tässä sellainen sieni, jota ei tule vastaa sie...</td>
      <td></td>
      <td>Discolaitteet</td>
      <td>Beamz</td>
      <td>Mulle Toi</td>
    </tr>
    <tr>
      <th>17574</th>
      <td>Kengät Les Tropéziennes Par M Belarbi  Gloss</td>
      <td>Kengät les tropéziennes par m belarbi gloss bl...</td>
      <td></td>
      <td>Naisten|Kengät|Bootsit</td>
      <td>Les Tropéziennes par M Belarbi</td>
      <td>Spartoo</td>
    </tr>
  </tbody>
</table>
</div>




```python

raw_feature_learner.data.add_test(test_data_raw)
print("Test data added to the final model")

obtain_test_predictions(model=raw_feature_learner, features=test_data_raw, how_many_preds=10)
```

      0%|          | 1/1000 [00:00<02:07,  7.86it/s]

    Test data added to the final model
    Prediction 0 is 5598


      0%|          | 3/1000 [00:00<01:54,  8.73it/s]

    Prediction 1 is 212
    Prediction 2 is 1901
    Prediction 3 is 408


      1%|          | 7/1000 [00:00<01:41,  9.76it/s]

    Prediction 4 is 187
    Prediction 5 is 187
    Prediction 6 is 1604


      1%|          | 9/1000 [00:01<02:25,  6.82it/s]

    Prediction 7 is 204
    Prediction 8 is 6305


      1%|          | 10/1000 [00:01<02:19,  7.07it/s]

    Prediction 9 is 188


      1%|          | 10/1000 [00:02<03:33,  4.64it/s]

    Prediction 10 is 778


    


### The right answers in this case are


```python
test_target2[0:10].values
```




    array([5598,  212, 1901,  408,  187,  187, 1604,  204, 6305,  188])



## Comparison of Models and Suggestion for Use

So we see that the answers are mostly right in both cases, but the models on processed data do better than the model trained on the raw data. 

*Important note : Even the presence of Nan's at prediction time did not seem to be a problem for the model. Just to be sure if possible, better always to remove NaNs*.

Test rounds on small data showed model 1 performing the best, followed by model 2. Model 3 (on raw features) was off more often, behind the models 1 and 2.

However, the validation accuracy of the raw features model should be somewhere around 90%+, but use this model only if it is really not possible to use processed features.

Thus, 

**Recommendation right now is to use model 1. Probably there is no real difference either if model 2 is used, because both of them have cross-entropy loss of around 0.2 and validation accuracy over 96%.**


```python

```


```python

```


```python

```


```python

```


```python

```

### There's Additional Stuff Below that is not needed for Production, but it's about
> 1. Validation of predictions of the whole validation sets (classification bunches)
> 2. How to make Predictions Faster (Acknowledging a possible bug // limitation in the source library)


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

## Running Validations on Full Validation Datasets (Fast.AI Databunches)

### First, the databunches have to be either created or loaded


```python



#cl_bunch_path = os.path.join(ROOT,"code","bunches_full","latest_cl_bunch_FULL_bs48")



def load_databunch(path=cl_bunch_path):
    """Given fast.AI classifier databunch path, it loads and returns it """
    cl_name, cl_base_path = produce_doublet_from_path(cl_bunch_path)
    classifier_bunch = load_data(cl_base_path, file=cl_name) #,**{"num_workers":0}
    return classifier_bunch

def validate_on_validation_set(model,path=cl_bunch_path):
    validation_bunch = load_databunch(path)
    """Given a Fast.AI prediction model and Fast.AI databunch, \
    this function finds the validation metrics (loss and accuracy) on validation data\
    of the databunch """
    model.data.train_dl = databunch.train_dl
    model.data.valid_dl = databunch.valid_dl
    print("VALIDATING THE MODEL. THE RESULT IS")
    validation_result = model.validate(model.data.valid_dl)
    return validation_result


```

    Base path is /home/ec2-user/SageMaker/e_commerce_hierarchical/e_commerce_project/code/bunches_full/
    Object name is latest_cl_bunch_FULL_bs48



```python

```


```python
validation_result_2 = validate_on_validation_set(model=final_learner_2,databunch=validation_bunch)
print(f"The result of the last FULL model training is {validation_result_2})
validation_result_1 = validate_on_validation_set(model=final_learner_1,databunch=validation_bunch)
print(f"The result of the FIRST FULL model training is {validation_result_1})



```

    [0.20045866, tensor(0.9645)]


### The first number above is the cross-entropy Validation Set loss and the other number is predictive accuracy (accuracy derived from the validation set, not the training set)

#### Let's see how the Model Trained on only raw features does on the validation Databunch


```python
raw_validation_result = validate_on_validation_set(model=raw_feature_learner,databunch=validation_bunch)
print(f"The result of the RAW FEATURES model training is {raw_validation_result}")
```

    The result of the RAW FEATURES model training is [7.4402742, tensor(0.0519)]


#### This shows that the raw features model cannot be tested on a databunch that was created from processed data, but original databunch has to be used!



```python

```


```python

```

## Speeding Up Test Predictions : Have to Fix Fast.AI Bug
These predictions are all wrong. For Some Reason, Fast.AI Is Predicting a wrong column


```python
#test_probabilities,_ = final_learner_2.get_preds(DatasetType.Test,ordered=True)
#test_predictions = torch.argmax(test_probabilities, dim=1)
print(test_predictions[0:10])
#data_pred = TextDataBunch.from_df(
#  path, test_df, valid_df, test_df=None,
#  vocab=data_clas.vocab, classes=data_clas.classes,
#  text_cols=DEFAULT_TEXT_COLS, label_cols=DEFAULT_LABEL_COL,
#)
#
##model.show_results(ds_type=DatasetType.Train)
##model.data.add_test(test_data_full)
#orig_classes, pred_classes = [], []
#for b in progress_bar(model.test_dl):
#    pred_probs = model.pred_batch(batch=b)
#    #orig_classes += [data_clas.classes[x] for x in b[1]]
#    #pred_classes += [data_clas.classes[x.argmax()] for x in pred_probs]
```

    tensor([  71,  163,  938,  223,  311,   71,   71, 1030,  317,  951])



```python
test_target[0:10].values
```




    array([ 212,  482, 5181,  598,  762,  212,  212, 5886,  774, 5322])



### If we drop the prediction indices, we see that Fast.AI is predicting 2 columns for some reason.
It should predict only the Category column and not give the tensor out


```python
obtain_test_predictions(model=final_learner_1, features=test_data_full, how_many_preds=1000)
```

      0%|          | 1/1000 [00:00<01:43,  9.66it/s]

    Prediction 0 is (Category 212, tensor(71), tensor([2.5375e-09, 5.2089e-08, 2.4279e-08,  ..., 1.9809e-10, 8.7067e-09,
            8.4734e-09]))


      0%|          | 3/1000 [00:00<02:37,  6.33it/s]

    Prediction 1 is (Category 482, tensor(163), tensor([2.7041e-07, 2.0830e-06, 6.7330e-08,  ..., 2.8553e-09, 1.6552e-06,
            1.4680e-06]))
    Prediction 2 is (Category 5181, tensor(938), tensor([7.3933e-09, 1.9553e-07, 7.7729e-08,  ..., 7.4974e-11, 6.4989e-10,
            3.5885e-10]))


      0%|          | 5/1000 [00:00<02:15,  7.36it/s]

    Prediction 3 is (Category 598, tensor(223), tensor([4.0987e-11, 2.0054e-09, 1.6541e-07,  ..., 1.2306e-10, 6.6816e-07,
            8.7567e-08]))
    Prediction 4 is (Category 762, tensor(311), tensor([1.2036e-10, 1.3761e-08, 1.0309e-12,  ..., 2.9363e-13, 1.3206e-09,
            1.3574e-11]))


      1%|          | 7/1000 [00:01<02:21,  7.03it/s]

    Prediction 5 is (Category 212, tensor(71), tensor([1.0216e-09, 1.2874e-08, 4.8393e-09,  ..., 8.3432e-11, 6.1998e-09,
            2.1025e-09]))
    Prediction 6 is (Category 212, tensor(71), tensor([8.0099e-09, 1.1881e-07, 1.1596e-07,  ..., 4.1671e-10, 2.0704e-08,
            1.7880e-08]))


      1%|          | 9/1000 [00:01<02:39,  6.23it/s]

    Prediction 7 is (Category 5886, tensor(1030), tensor([8.4273e-10, 5.9291e-10, 1.5412e-08,  ..., 1.6924e-11, 3.6951e-09,
            4.5356e-10]))
    Prediction 8 is (Category 774, tensor(317), tensor([1.2387e-09, 1.0815e-07, 7.0596e-09,  ..., 6.8684e-10, 6.5068e-09,
            5.0885e-09]))


      1%|          | 11/1000 [00:01<02:35,  6.34it/s]

    Prediction 9 is (Category 5322, tensor(951), tensor([1.9032e-09, 1.8962e-08, 3.1860e-07,  ..., 2.9642e-11, 2.7985e-10,
            1.6801e-09]))
    Prediction 10 is (Category 338, tensor(109), tensor([3.0170e-05, 1.5516e-04, 1.4880e-04,  ..., 1.7925e-06, 6.4803e-05,
            1.3253e-04]))


      1%|▏         | 13/1000 [00:01<02:16,  7.24it/s]

    Prediction 11 is (Category 6262, tensor(1069), tensor([5.6247e-09, 3.9610e-07, 2.2790e-10,  ..., 1.7851e-11, 2.0815e-08,
            1.8166e-10]))
    Prediction 12 is (Category 1604, tensor(440), tensor([6.4382e-10, 1.0106e-07, 4.6156e-08,  ..., 3.1862e-10, 2.5501e-09,
            5.8772e-09]))
    Prediction 13 is (Category 187, tensor(55), tensor([6.4318e-11, 4.8625e-10, 2.3674e-10,  ..., 4.5229e-11, 1.8094e-10,
            4.1479e-10]))


      2%|▏         | 15/1000 [00:02<02:05,  7.86it/s]

    Prediction 14 is (Category 187, tensor(55), tensor([3.9690e-10, 2.4529e-09, 3.5681e-09,  ..., 2.0043e-10, 8.0289e-10,
            3.8611e-09]))
    Prediction 15 is (Category 1604, tensor(440), tensor([1.2016e-11, 1.1969e-09, 5.9463e-10,  ..., 1.0156e-11, 1.3133e-10,
            1.1694e-10]))


      2%|▏         | 19/1000 [00:02<01:47,  9.10it/s]

    Prediction 16 is (Category 567, tensor(200), tensor([4.1701e-07, 2.0200e-07, 2.9024e-08,  ..., 1.7551e-09, 5.7990e-08,
            7.8653e-08]))
    Prediction 17 is (Category 187, tensor(55), tensor([2.1139e-10, 1.1197e-09, 8.9729e-10,  ..., 1.2048e-10, 7.1356e-10,
            1.0140e-09]))
    Prediction 18 is (Category 567, tensor(200), tensor([4.0549e-08, 9.2300e-08, 1.3723e-08,  ..., 1.8751e-10, 1.7643e-09,
            1.4545e-08]))


      2%|▏         | 21/1000 [00:02<01:39,  9.80it/s]

    Prediction 19 is (Category 1253, tensor(403), tensor([8.0549e-06, 4.8097e-05, 1.4370e-02,  ..., 3.4594e-07, 8.6861e-05,
            3.8863e-05]))
    Prediction 20 is (Category 2980, tensor(638), tensor([4.6539e-06, 2.6183e-04, 3.7588e-04,  ..., 5.7173e-07, 4.9386e-06,
            1.3559e-05]))


      2%|▏         | 23/1000 [00:02<02:05,  7.79it/s]

    Prediction 21 is (Category 2441, tensor(519), tensor([6.4841e-09, 1.9929e-08, 3.1188e-08,  ..., 1.2187e-09, 4.6714e-07,
            9.8371e-09]))
    Prediction 22 is (Category 2425, tensor(518), tensor([1.3892e-11, 1.5046e-10, 9.5565e-10,  ..., 1.8310e-12, 9.2062e-10,
            8.7336e-09]))


      3%|▎         | 26/1000 [00:03<01:48,  8.96it/s]

    Prediction 23 is (Category 2541, tensor(530), tensor([3.2938e-09, 7.9447e-08, 5.8372e-11,  ..., 1.1090e-10, 8.4458e-12,
            2.3384e-09]))
    Prediction 24 is (Category 567, tensor(200), tensor([6.7017e-08, 1.9958e-07, 4.2756e-08,  ..., 3.2173e-10, 5.7056e-09,
            7.8529e-08]))
    Prediction 25 is (Category 588, tensor(217), tensor([4.6708e-08, 8.3342e-07, 2.8303e-07,  ..., 8.8186e-10, 3.6298e-10,
            1.3937e-09]))


      3%|▎         | 27/1000 [00:03<01:59,  8.12it/s]

    Prediction 26 is (Category 2271, tensor(498), tensor([6.1477e-09, 1.5463e-08, 5.2608e-09,  ..., 1.8229e-10, 1.7188e-08,
            4.4088e-09]))
    Prediction 27 is (Category 212, tensor(71), tensor([3.2490e-09, 5.8936e-08, 2.7642e-08,  ..., 3.8399e-10, 2.7430e-08,
            1.9385e-09]))


      3%|▎         | 30/1000 [00:03<02:07,  7.60it/s]

    Prediction 28 is (Category 201, tensor(64), tensor([1.0854e-10, 5.3455e-10, 4.6623e-10,  ..., 2.4858e-12, 1.2582e-10,
            4.7358e-11]))
    Prediction 29 is (Category 203, tensor(65), tensor([2.5704e-08, 5.2228e-06, 1.1441e-07,  ..., 8.5825e-10, 2.1039e-09,
            1.1448e-08]))


      3%|▎         | 32/1000 [00:04<02:07,  7.58it/s]

    Prediction 30 is (Category 2271, tensor(498), tensor([1.3939e-08, 3.5268e-08, 4.6092e-08,  ..., 4.2224e-10, 7.1456e-08,
            1.6395e-08]))
    Prediction 31 is (Category 594, tensor(219), tensor([8.7369e-10, 3.3760e-07, 5.0545e-08,  ..., 1.9205e-10, 6.3895e-08,
            4.3259e-08]))


      3%|▎         | 33/1000 [00:04<02:37,  6.15it/s]

    Prediction 32 is (Category 2923, tensor(621), tensor([3.4393e-10, 2.4077e-09, 7.9917e-07,  ..., 7.8501e-11, 1.0150e-08,
            1.7691e-10]))


      3%|▎         | 34/1000 [00:04<03:36,  4.45it/s]

    Prediction 33 is (Category 525, tensor(180), tensor([3.0558e-08, 1.6317e-06, 2.8342e-07,  ..., 3.4243e-09, 9.4310e-08,
            3.9092e-08]))


      4%|▎         | 36/1000 [00:05<03:21,  4.78it/s]

    Prediction 34 is (Category 2425, tensor(518), tensor([6.0899e-11, 5.8235e-10, 5.1802e-09,  ..., 5.8095e-12, 1.6284e-09,
            3.0549e-08]))
    Prediction 35 is (Category 3686, tensor(759), tensor([4.9515e-08, 6.3808e-06, 6.7513e-07,  ..., 7.6439e-09, 8.7976e-10,
            5.6784e-09]))


      4%|▍         | 38/1000 [00:05<02:47,  5.76it/s]

    Prediction 36 is (Category 187, tensor(55), tensor([1.2404e-09, 8.4103e-09, 1.1487e-08,  ..., 4.8620e-10, 4.0104e-09,
            6.8272e-09]))
    Prediction 37 is (Category 5598, tensor(990), tensor([7.6785e-10, 4.3867e-09, 4.6488e-09,  ..., 4.6035e-11, 1.2639e-09,
            2.6194e-09]))
    Prediction 38 is (Category 212, tensor(71), tensor([4.1647e-09, 1.2398e-07, 6.5129e-08,  ..., 3.6469e-10, 2.4545e-08,
            1.7404e-08]))


      4%|▍         | 41/1000 [00:05<03:00,  5.30it/s]

    Prediction 39 is (Category 398, tensor(122), tensor([2.1481e-05, 2.3269e-04, 5.6144e-05,  ..., 1.0645e-06, 2.8555e-05,
            6.1816e-05]))
    Prediction 40 is (Category 775, tensor(318), tensor([1.0715e-08, 2.9368e-07, 4.4682e-09,  ..., 2.7089e-09, 1.3865e-09,
            8.0249e-08]))


      4%|▍         | 42/1000 [00:06<02:54,  5.50it/s]

    Prediction 41 is (Category 2425, tensor(518), tensor([1.2353e-11, 9.7242e-11, 9.1979e-10,  ..., 1.2514e-12, 6.5678e-10,
            7.4247e-09]))


      4%|▍         | 44/1000 [00:06<03:33,  4.48it/s]

    Prediction 42 is (Category 2425, tensor(518), tensor([1.6032e-10, 1.1213e-09, 9.7212e-09,  ..., 1.6518e-11, 3.3693e-09,
            4.9469e-08]))
    Prediction 43 is (Category 212, tensor(71), tensor([3.9086e-09, 1.3894e-07, 5.4743e-08,  ..., 3.9327e-10, 1.5640e-08,
            8.4106e-09]))


      4%|▍         | 45/1000 [00:06<02:26,  6.53it/s]


    Prediction 44 is (Category 212, tensor(71), tensor([2.2943e-08, 2.4633e-07, 8.4883e-08,  ..., 1.8434e-09, 9.0445e-08,
            3.3441e-08]))



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-129-1ac2381cdd39> in <module>()
    ----> 1 obtain_test_predictions(model=final_learner_1, features=test_data_full, how_many_preds=1000)
    

    <ipython-input-128-9198e15cfd23> in obtain_test_predictions(model, features, how_many_preds)
         23     features = features.reset_index(drop=True)
         24     for i, feature_row in tqdm(features.iterrows(), total=features.shape[0]):
    ---> 25         prediction = get_one_prediction(model, feature_row)
         26         print(f"Prediction {i} is {prediction}")
         27         if i == how_many_preds:


    <ipython-input-128-9198e15cfd23> in get_one_prediction(model, feature_row)
         14 
         15 def get_one_prediction(model, feature_row):
    ---> 16     one_prediction = model.predict(feature_row)
         17     # print(f"Model Prediction value is: {one_prediction}")
         18     return one_prediction


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/basic_train.py in predict(self, item, return_x, batch_first, with_dropout, **kwargs)
        368     def predict(self, item:ItemBase, return_x:bool=False, batch_first:bool=True, with_dropout:bool=False, **kwargs):
        369         "Return predicted class, label and probabilities for `item`."
    --> 370         batch = self.data.one_item(item)
        371         res = self.pred_batch(batch=batch, with_dropout=with_dropout)
        372         raw_pred,x = grab_idx(res,0,batch_first=batch_first),batch[0]


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/basic_data.py in one_item(self, item, detach, denorm, cpu)
        178         "Get `item` into a batch. Optionally `detach` and `denorm`."
        179         ds = self.single_ds
    --> 180         with ds.set_item(item):
        181             return self.one_batch(ds_type=DatasetType.Single, detach=detach, denorm=denorm, cpu=cpu)
        182 


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/contextlib.py in __enter__(self)
         79     def __enter__(self):
         80         try:
    ---> 81             return next(self.gen)
         82         except StopIteration:
         83             raise RuntimeError("generator didn't yield") from None


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/data_block.py in set_item(self, item)
        609     def set_item(self,item):
        610         "For inference, will briefly replace the dataset with one that only contains `item`."
    --> 611         self.item = self.x.process_one(item)
        612         yield None
        613         self.item = None


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/data_block.py in process_one(self, item, processor)
         89         if processor is not None: self.processor = processor
         90         self.processor = listify(self.processor)
    ---> 91         for p in self.processor: item = p.process_one(item)
         92         return item
         93 


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/text/data.py in process_one(self, item)
        289 
        290     def process_one(self, item):
    --> 291         return self.tokenizer._process_all_1(_join_texts([item], self.mark_fields, self.include_bos, self.include_eos))[0]
        292 
        293     def process(self, ds):


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/text/transform.py in _process_all_1(self, texts)
        110     def _process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        111         "Process a list of `texts` in one process."
    --> 112         tok = self.tok_func(self.lang)
        113         if self.special_cases: tok.add_special_cases(self.special_cases)
        114         return [self.process_text(str(t), tok) for t in texts]


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/text/transform.py in __init__(self, lang)
         23     "Wrapper around a spacy tokenizer to make it a `BaseTokenizer`."
         24     def __init__(self, lang:str):
    ---> 25         self.tok = spacy.blank(lang, disable=["parser","tagger","ner"])
         26 
         27     def tokenizer(self, t:str) -> List[str]:


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/spacy/__init__.py in blank(name, **kwargs)
         30 def blank(name, **kwargs):
         31     LangClass = util.get_lang_class(name)
    ---> 32     return LangClass(**kwargs)
         33 
         34 


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/spacy/language.py in __init__(self, vocab, make_doc, max_length, meta, **kwargs)
        166         if make_doc is True:
        167             factory = self.Defaults.create_tokenizer
    --> 168             make_doc = factory(self, **meta.get("tokenizer", {}))
        169         self.tokenizer = make_doc
        170         self.pipeline = []


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/spacy/language.py in create_tokenizer(cls, nlp)
         78             suffix_search=suffix_search,
         79             infix_finditer=infix_finditer,
    ---> 80             token_match=token_match,
         81         )
         82 


    tokenizer.pyx in spacy.tokenizer.Tokenizer.__init__()


    tokenizer.pyx in spacy.tokenizer.Tokenizer.add_special_case()


    vocab.pyx in spacy.vocab.Vocab.make_fused_token()


    vocab.pyx in spacy.vocab.Vocab.get_by_orth()


    vocab.pyx in spacy.vocab.Vocab._new_lexeme()


    ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/spacy/lang/lex_attrs.py in lower(string)
        175 
        176 
    --> 177 def lower(string):
        178     return string.lower()
        179 


    KeyboardInterrupt: 


### One can see clearly that the fast.AI is for some reason predicting the 2nd column (e.g. prediction 0 is tensor(71), not 212 as is the mapped id). The solution has not been found, but should definitely exist


```python
## One solution may be to change the source code since the predict function is working
## Second option would be to get pred_batch to work properly
path="/home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai"
!grep -Rn {path} -e "def predict"
#--include=\*.{py,ipynb,md} 
```

    /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/basic_train.py:372:    def predict(self, item:ItemBase, return_x:bool=False, batch_first:bool=True, with_dropout:bool=False, **kwargs):
    /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/basic_train.py:429:    def predict_with_mc_dropout(self, item:ItemBase, with_dropout:bool=True, n_times=10, **kwargs):
    /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/text/learner.py:118:    def predict(self, text:str, n_words:int=1, no_unk:bool=True, temperature:float=1., min_p:float=None, sep:str=' ',
    /home/ec2-user/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/fastai/data_block.py:622:    def predict(self, res):



```python

```
