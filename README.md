# gpt-from-scratch
Implementation of a light version of GPT from scratch, following [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy.

I developed this implementation to get a deeper understanding about the attention mechanism and how GPT works.

The models are trained with the [Tiny Sheakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) for character-level text generation in the style of Sheakespeare. 

Two models are implemented: a simple baseline Bigram model and a Transformer based GPT model.

## Project
The project is organized as follows:
```
gpt-from-scratch/
    ├── config/
    ├── data/
    ├── models/
    ├── notebooks/
    ├── outputs/
    └── src/
        ├── baseline/
        ├── gpt/
        └── utils/
```
- `config/` contains the project configuration file.
- `data/` contains the raw data, train and test splits, and other artifacts.
- `models/` contains the serialized and already trained models.
- `notebooks/` contains th jupyter notebook used for development.
- `src/` contains the code for data loading in `utils/` and the model, training and text generation for both the baseline and gpt models, in `baseline/` and `gpt/` respectively.

## Execution

The source code execution should be as explained below:

### Data
For data downloading from source, data preprocessing and train-val split, execute:
```shell
python -m src.utils.load_data
```

### Training
To train the baseline model, run:
```shell
python -m src.baseline.train
```

For GPT model training, execute:
```shell
python -m src.gpt.train
```

### Generation

To generate text using the baseline model, execute:
```shell
python -m src.baseline.generate
```

To generate text using the GPT model, run:
```shell
python -m src.gpt.generate
```

By default, as it is stated in the config file, both models generate up to 1000 characters, but the numbers of characters can be modified overriding the `generate.max_new_tokens` configuration by executing:
```shell
python -m src.gpt.generate generate.max_new_tokens=5000
```
This example will generate 5000 characters using the GPT model.

## Results

Examples of the generated text are stored in the `outputs` directory. For 1000 new characters, an example of the obtained results with GPT model is shown below:

GPT generated text in the style of Sheakespeare:
```
See, to any o' fall of any crien that lolds
Come pleace to primpliant fightingryme,
Plantage fair a braves, whose arty pity
Though thousand of Buckingham, and accused ixage
Of us lordship on me, and were my letter his bosom
Constain to before he is boref he begin. Then
If the gods noble less raise.

DUKE OF YORK:
Why bridge shall we in all executempt: but the king
his mortal ends. Wiston of loyal hate her,
for thereby the other heart, in brothers aught shame
theyselves and a thingle womb's shall teeld and be:
I coulded thy will borswill ignocences of her:
Let us no my unjoin tell which to out his heir?

ISABELLA:
Harry, that use the battend and torm them, but heaven;
put since in than here did love to straight
The impative distreme of the plant, Hastessand. Will, an his majesty:
And too party: mark him sucklant with e'er her curse.
By thumb and rooks against their he wing,
Femal to catckle will our but thou reased,
The king of strike any king: though we myself,
That be proudent to the 
```

The results show imperfections that could be improved by scaling the model training. In this case, the training was carried out on a small NVIDIA GeForce GTX 1050 GPU.

However, the model is able to generate words that in most cases make sense in the English language and follow the structure present in the training data.