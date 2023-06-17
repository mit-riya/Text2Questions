import numpy as np
import streamlit as st
from transformers import BertTokenizer, BertModel
import pandas as pd
import nltk
import torch
from keybert import KeyBERT
import random
nltk.download('wordnet')
from nltk.corpus import wordnet 
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine

# Loading the pre-trained BERT model
# Embeddings will be derived from
# the outputs of this model
model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True,
                                      )

# Setting up the tokenizer
# This is the same tokenizer that
# was used in the model to generate 
# embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors
    
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def get_option(options,find_word,find_sen):
    # Getting embeddings for the target
    # word in all given contexts
    target_word_embeddings = []
    for text in options:
      tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text.lower(), tokenizer)
      list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
      
      # Find the position 'bank' in list of tokens
      word_index = tokenized_text.index(find_word)
      # Get the embedding for bank
      word_embedding = list_token_embeddings[word_index]

      target_word_embeddings.append(word_embedding)
    list_of_distances = []
    # Calculating the distance between the
    # embeddings of 'bank' in all the
    # given contexts of the word
    for text1, embed1 in zip(find_sen, target_word_embeddings):
      for text2, embed2 in zip(options, target_word_embeddings):
          cos_dist = 1 - cosine(embed1, embed2)
          list_of_distances.append([text1, text2, cos_dist])

    distances_df = pd.DataFrame(list_of_distances, columns=['text1', 'text2', 'distance'])
    min_dist=2
    min_ind=0
    for ind in distances_df.index:
      if(distances_df['distance'][ind]<min_dist):
        min_dist=distances_df['distance'][ind]
        min_ind=ind
    ans=distances_df['text2'][min_ind]
    n=len(find_word)
    ans=ans[n+4:]
    return ans

def get_mcq(paragraph):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(paragraph,top_n=8,diversity=0.8)
    ans=[]
    list=[]
    for word,prob in keywords:
      if(not wordnet.synsets(word)):
        continue
      syns = wordnet.synsets(word)
      find_word=word
      options=[]
      for syn in syns:
        options.append(syn.definition())
      sentences=[]
      sentences=sent_tokenize(paragraph)
      find_sen=[]
      for i in sentences:
        fl=0
        if find_word in i.lower():
          tem=i.split("\n")
          for k in tem:
            if word in k.lower():
              find_sen.append(k)
              fl=1
              break
          if fl==1:
            break
      for i in range(len(options)):
        options[i]=find_word+" is "+options[i]
      list.append(word)
      # print(find_sen)
      t=get_option(options,find_word,find_sen)
      ans.append(t)
    n=len(ans)
    ls1=[]
    ls2=[]
    for i in range(n):
      ls1.append(list[i])
      ls2.append(ans[i])
    dict2 = {'Column 1': ls1, 'Column 2': ls2}
    random.shuffle(ans)
    mcq=[]
    mcq_ans=[]
    for i in range(n):
      mcq.append(list[i])
      mcq_ans.append(ans[i])
    dict = {'Column 1': mcq, 'Column 2': mcq_ans}
    df1=pd.DataFrame(dict)
    df2=pd.DataFrame(dict2)
    return [df1,df2]
  
    
if __name__ == '__main__':
    st.set_page_config(page_title="Text2Questions", page_icon="🧠")
    original_title = '<h1 style="font-family:Playfair Display; color:Red; font-size: 40px; background-color:White; border-radius: 5px ;padding:10px; margin-bottom:10px">Text2Match</h1>'
    st.markdown(original_title, unsafe_allow_html = True)
    # # getting the input data from the user
    doc = st.text_area('Enter text')
    # # # code for Prediction
    output=[]
    # # creating a button for Prediction
    if st.button('Generate Questions'):
        output = get_mcq(doc)
        ques=output[0]
        ans=output[1]
        st.write(ques)
        st.write("Answer")
        st.write(ans)
