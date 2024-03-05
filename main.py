from fastai.text.all import *
from IPython import display
from IPython.core.display import HTML

def subword(sz, txts, txt):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])

if __name__ == "__main__":
    path = untar_data(URLs.IMDB)


    get_imdb = partial(get_text_files, folders= ['test', 'train', 'unsup'])
    dls_lm = DataBlock(
        blocks = TextBlock.from_folder(path, is_lm=True),
        get_items = get_imdb, 
        splitter = RandomSplitter(0.1),
    ).dataloaders(path, path=path, bs=128, seq_len=80)

    
    learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy,Perplexity()]).to_fp16()

    learn.fit_one_cycle(1, 2e-2)