# recode-with-mistral-finetune

`mistral-finetune` is a light-weight codebase that enables memory-efficient and performant finetuning of Mistral's models.
It is based on [LoRA](https://arxiv.org/abs/2106.09685), a training paradigm where most weights are frozen and only 1-2% of additional weights in the form of low-rank matrix perturbations are trained. 

This repo is a fork of the original [`mistral-finetune` ](https://github.com/mistralai/mistral-finetune) adapted to the training of models dedicated to ICD-10 coding from clinical notes. 

The purpurse of this fork is to help information medical teams to finetune Mistral model on the ICD-10 coding task (in french) with so called annotated data :
- data = clinical notes (1 note or the concatenation of all the notes available for the patient in EMR). The restriction is that the model can only take a fixed number of token as entry.
- annotation = ICD-10 codes of the PMSI resume. 2 formats are possible
  * when using classification : lits of code (ex [C509, I10,...]) - work in progress
  * when using generative model : definition of the code (code) (ex : Hypertention artérielle primitive (I10)). - examples in this projet.

For the second, we want to finetune of a generative model. We can use 2 paradigms :
- Next token prediction : you give the model a long text, and the task is to prodict next word. For ICD-10 coding, we train the model will on a text which is the concatenation of the note and ICD-10 coding. This task will help the model to learn contextualised reprensentations of medical words from the clinical notes and of the of ICD-10 definitions and codes
- Instruction prediction : the model here is seen as an assistant. You give to the assistant a context (medical ICD-10 coding from clinical notes) and a question (what codes will you choose for the following clinical note), and the assistant will give a correct answer (the ICD-10 codes).

The project gives examples for this finetuning that can run of google collab and in your own environements. To help get to grips with the code we also provide a synthtic dataset with the method to produce it.

So the project contains 3 notebooks (`tutorials/`):
- <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/generate_fictives_notes.ipynb">
   generate_fictives_notes.ipynb </a>  : code to produce fictionnal data with Mistral AI API. Some fsample data (annoted clinical notes with ICD-10 codes) have been generated and are available in the sample_data folder :
   
   * <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/text.csv">Texts </a>
   * <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/codes.csv">Codes </a>
- <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/prepare_data_for_generative_finetuning.ipynb">
  prepare_data_for_generative_finetuning.ipynb </a>  : In order to feed the model, data need reformating (see below). This notebook aims to prepare the data produced at the previous step. The preprocessing has already been done on the fictional sample data and are also avaiblable in sample_data folder :
  
   * Next token prediction (text)
      
      - <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/text.csv">train_text.jsonl</a>
      - <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/codes.csv">val_text.jsonl</a> 
      - <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/codes.csv">test_text.jsonl</a> 
    
    * Instruction prediction (instruct)
      
      - <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/text.csv">train_instruct.jsonl</a>
      - <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/codes.csv">val_instruct.jsonl</a> 
      - <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/sample_data/codes.csv">test_instruct.jsonl</a> 

- <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb">mistral_finetune_7b.ipynb</a>  : finnaly perform training and eveluation of the model (ex Mistral-7B). The notebook produce its own config file with all necessary parameters (see below).


## Prepare dataset 

To ensure effective training, `mistral-finetune` has strict 
requirements for how the training data has to be formatted.

All data files must be stored in jsonl format files.

You can build two types of data files:

### _Pretrain_:

Pretrain data corresponds to plain text data stored in the `"text"` key. E.g:

```jsonl
{"text": "Text contained in clinical note n°1. ICD-10 codes : definition 1 (code 1),..."}
{"text": "Text contained in clinical note n°2. ICD-10 codes : definition 1 (code 1),..."}
```

In the pretrain paradigm models are funetune with the next token prediction task.
### _Instruct_:

Currently two different types of instruction following data are supported:

- _Instruct_: In the conversational data stored in the `"messages"` key in the form of a list. Each list item is a dictionary containing the `"content"` and `"role"` keys. `"role"` is a string being one of 
  * "system" :  task contextualization
  * "user" : question the assistant will answer
  * "assistant" : expected result from the assistant
The loss will only be computed if "role" == "assistant". 

For ICD-10 coding in French we have adopted the following conventions :
- system : Vous êtes un modèle de langage en française spécialisé dans le codage des diagnostics selon la classification internationale des maladies version 10 (CIM-10) pour les résumés standardisés de sortie du programme de médicalisation des systèmes d'information français (PMSI). A partir des comptes rendus d'hospitalisation vous donnerez les codes diagnostics CIM-10 que l'on peut retenir pour le séjours en distiguant diagnostic principal, diagnostic relié et diagnostics associés.
- user : Générez le codage CIM-10 du résumé strandisé de sortie PMSI à partir du compte rendu d'hospitalisation suivant : texte du compte rendu
- assistant : Codes CIM 10 retenus pour le résumé strandisé de sortie PMSI : diagnostic principal : définition du code (code), diagnistic relié : aucun, diagnostic associé : définition diagnostic 1 (code 1),...
```jsonl
{
  "messages": [
     {
      "role": "system",
      "content": "Vous êtes un modèle de langage en française spécialisé dans le codage des diagnostics selon la classification internationale des maladies version 10 (CIM-10) pour les résumés standardisés de sortie du programme de médicalisation des systèmes d'information français (PMSI). A partir des comptes rendus d'hospitalisation vous donnerez les codes diagnostics CIM-10 que l'on peut retenir pour le séjours en distiguant diagnostic principal, diagnostic relié et diagnostics associés."
    },
    {
      "role": "user",
      "content": "Générez le codage CIM-10 du résumé strandisé de sortie PMSI à partir du compte rendu d'hospitalisation suivant : texte du compte rendu n°1"
    },
    {
      "role": "assistant",
      "content": "Codes CIM 10 retenus pour le résumé strandisé de sortie PMSI : diagnostic principal : définition du code (code), diagnistic relié : aucun, diagnostic associé : définition diagnostic 1 (code 1),..."
    }
  ]
}
{
  "messages": [
     {
      "role": "system",
      "content": "Vous êtes un modèle de langage en française spécialisé dans le codage des diagnostics selon la classification internationale des maladies version 10 (CIM-10) pour les résumés standardisés de sortie du programme de médicalisation des systèmes d'information français (PMSI). A partir des comptes rendus d'hospitalisation vous donnerez les codes diagnostics CIM-10 que l'on peut retenir pour le séjours en distiguant diagnostic principal, diagnostic relié et diagnostics associés."
    },
    {
      "role": "user",
      "content": "Générez le codage CIM-10 du résumé strandisé de sortie PMSI à partir du compte rendu d'hospitalisation suivant : texte du compte rendu n°2"
    },
    {
      "role": "assistant",
      "content": "Codes CIM 10 retenus pour le résumé strandisé de sortie PMSI : diagnostic principal : définition du code (code), diagnistic relié : aucun, diagnostic associé : définition diagnostic 1 (code 1),..."
    }
  ]
}
```


 The notebook ```prepare_data_for_generative_finetuning```  will show how to prepare data step by step. <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/prepare_data_for_generative_finetuning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>


## Verify dataset

Before starting a training run you should verify that your dataset is correctly formatted and get an 
estimation of the training time. You can do so by using the [./utils/validate_data](https://github.com/mistralai/mistral-finetune/blob/main/utils/validate_data.py) script.

Note that this step is crucial to ensure that the data is correctly formatted.

```
cd $HOME/mistral-finetune
python -m utils.validate_data --train_yaml example/7B.yaml
```

You should get a summary of the data input and training parameters:

```
Train States
 --------------------
{
   "expected": {
       "eta": "00:52:44",
       "data_tokens": 25169147,
       "train_tokens": 131072000,
       "epochs": "5.21",
       "max_steps": 500,
       "data_tokens_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "25169147.0"
       },
       "train_tokens_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "131072000.0"
       },
       "epochs_per_dataset": {
           "/Users/johndoe/data/ultrachat_chunk_train.jsonl": "5.2"
       }
   },
}
```

Having `max_steps` set to 500 would lead to iterating through the dataset roughly 5 times which is reasonable, but might 
be a bit too much. A recommended setting is shown below which would only take 30min on a 8xH100 cluster.

See notebook mistral_finetune_7b <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab mistral_finetune_7b"/></a>


## Training
The train is performed with the train script with a simple command line :
```
cd mistral-finetune
torchrun --nproc-per-node 1 -m train example/instruct_icd_v1.yaml
```

The notobook mistral_finetune_7b gives you a full example <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

All the parameters of the training procedure are stored in yaml config file (see example/7B.yaml). Modify your training yaml to include the ultrachat dataset and verify the yaml

The example `mistral-finetune/examples/7B` defines reasonable parameters for learning rate, weight decay, etc... but you are advised to 
customize these settings for your use case.

Generally, a training configuration should fill the following parameters:

- `model_id_or_path` defines the model to start training from. This can be a path to a pre-trained model or a local model directory.
- `run_dir` defines the directory where training checkpoints and metrics are stored.
- `seq_len` defines the sequence length for training. This is the maximum length of input sequences the model will process. Samples are packed to reach a length of `seq_len` for maximum training efficiency.
- `batch_size` defines the number of training examples used per GPU. **Note**: The overall effective batch_size (in tokens) across all GPUs equals `num_gpus` x `batch_size` x `seq_len`.
- `max_steps` defines the maximum number of training steps. This is the total number of iterations the training process will run. It can be adjusted based on the specific needs of your training scenario. Total number of tokens seen during training is `max_steps` x `num_gpus` x `batch_size` x `seq_len`.
- `optim.lr` defines the learning rate. This is the initial learning rate for the optimizer.
- `optim.weight_decay` defines weight decay. Weight decay is a regularization technique used to prevent overfitting by penalizing large weights. We recommend leaving it at 0.1.
- `optim.pct_start` defines the percentage of the total training steps used for the learning rate warm-up phase before it starts to decrease. It corresponds to pct_start of PyTorch's OneCycleLR.
- `lora.rank` defines the size of the LoRA (Low-Rank Adaptation) adapters. We recommend 64 or less, which adjusts the rank of the low-rank decomposition used in LoRA.
- `seed` defines the random seed for initialization and data shuffling/sampling. Setting a seed ensures reproducibility of results.
- `log_freq` defines the logging frequency. This specifies how often (in steps) to log training metrics.
- `data.instruct_data` is the path to the instruction data used for training. This field has to be filled with one or multiple data sources in the format as explained above. Each data source should either be a path to a jsonl file or a path to a directory containing jsonl files followed by a weighting to define the importance of this dataset: `<path/to/data_source>:<weight>`. E.g.: `data.instruct_data: "/path/to/data1.jsonl:5.,/path/to/data2.jsonl:1.,/path/to/dir_of_jsonls:1."`
- `data.data` is an optional path to additional pretraining data in the format as explained above. Note that this field can be left blank.
- `data.eval_instruct_data` is an optional path to evaluation instruction data to run cross-validation at every `eval_freq` steps. Cross-validation metrics are displayed as `loss` and `perplexity`.
- `eval_freq` defines how often (in steps) to evaluate the model. This specifies the interval at which the model is evaluated on the validation set.
- `no_eval` is a flag to enable or disable intermediate evaluation. Setting it to False enables periodic evaluation during training.
- `ckpt_freq` defines how often (in steps) to save checkpoints. This specifies the interval at which the model's state is saved.
- `save_adapters` defines whether to only save the trained LoRA checkpoints or whether the trained LoRA should directly be merged into the base model and saved. **Note**: When setting `save_adapters=False` make sure that you have enough CPU and GPU memory to save the full model on a single process (this is usually only possible for the 7B model).
- `wandb.key` is used to pass your Weights & Biases (wandb) API key for logging. This allows you to log training metrics to the wandb dashboard.
- `wandb.project` defines the wandb project name. This is where the training run will be logged in the wandb interface.


## Inference

Once your model is trained, you should try it out in inference. We recommend using [mistral-inference](https://github.com/mistralai/mistral-inference). 

See mistral_finetune_7b gives you a full example <a target="_blank" href="https://github.com/24p11/recode-with-mistral-finetune/blob/main/tutorials/mistral_finetune_7b.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Parameters optimisation

**Mixed precision**

Models are mainly a dictionary of parameters (a lots 7 Billions in our example). 
Parameters are numeric values stored in a specific format, usually numpy float32 which uses a lot of GPU memory. 
Some other format can be used (See : [NVIDIA blog](https://developer.nvidia.com/blog/getting-immediate-speedups-with-a100-tf32/))

<img src="images/precision.png" alt="Mixed precision"/></a>

For our training, the format to use is hard coded in the ```train.py``` file, line 175
- bflot16 and tf32 can be use with novel GPU ampere format (A100, H100)
- float16 is use instead (with GPU V100 for example)

If you have a non ampere GPU you will also need to change precision for inference (see mistral_finetune_7b.ipynb)

**Learning rate**

By default the training is configure to use [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) scheduler which as 2 period
- increase : begin with LR (initial_lr) = max_lr/div_factor to reach  LR = max_lr
- decrease : when reach  LR = max_lr decrease to  min_lr = initial_lr/final_div_factor
Mistral recommandation for max_lr is 1.e-4.
I have only experience long training with float16. With this format, the training is more instable and which have for consequence to increase the loos greatly, until sometimes an NA value (see experiments logs [logs_RF-Finetune_M7B-v0.3_12_2024.md](experiments/logs_RF-Finetune_M7B-v0.3_12_2024.md)).

I recommand to use the following values when training is performed with flaot16 (not implemented in the actual config file):

- optim:
  * lr: 1.e-5
  * div_factor : 5
  * final_div_factor : 10
  * pct_start: 0.3


**Batching strategy for training data**

The load_dataset functions of mistral-fine, use by default a full in memory loading of the data which can cause memory errors for large dataset.
I recommand to set to ```False``` this option in the data Args file (/finetune/data/args.py) : 
- line 11 :  shuffle: bool = False
- line 11 :  dynamic_chunk_fn_call: bool = False
This wil force lazy loading, more adapted for large datasets.

**LoRA**

Mistral-finetuning library uses the LoRA, a technique that accelerates the fine-tuning of large models while consuming less memory.

<img src="images/lora_diagram.png" alt="Mixed precision"/></a>

See [Lora Hugginface blog post](https://huggingface.co/docs/peft/main/conceptual_guides/lora)

The consequence is that there are 2 possible behaviour for saving the finetuned model  (checkpointing):
- Save LoRA matrices (the left side of the figure above)
- Merge LoRA matrices with the intial model to save a new model like (left side of the figure)

This behaviour is controled by the ```save_adapters``` arguments (see above). For testing the procedure the first choice is possible, but to learn a model you intend to use in production, it should be better to build a full autonomus model.

## Share tricks and results

In the folder [experiments](experiments/) you can sahre logs of your experiments and some comments on yours results.