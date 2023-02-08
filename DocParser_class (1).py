# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:24:25 2022

@author: bhati
"""

import os
import streamlit as st
import pdfplumber
#!pip install pdfplumber

import streamlit as st
import haystack
import PyPDF2
#from haystack.preprocessor.preprocessor import PreProcessor
from haystack.nodes import PreProcessor

import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
import base64
import os
#launching elasticsearch
from haystack.utils import launch_es
launch_es()

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

import PyPDF2
import re
import glob
import pandas as pd
from pprint import pprint
import requests
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import streamlit as st

class Document(object):
    
    def __init__(self,pdf_file):
        self.pdf_file = pdf_file
    
        
    def extract_content(pdf_file):
        this_loc=1
        df = pd.DataFrame(columns =('name','content'))
       
        pdfobj = open(f'{pdf_file}','rb') 
        #pdfobj = open(f'{pdf_file.name}','rb')
        base64_pdf = PyPDF2.PdfFileReader(pdfobj)
        this_doc=''
        for page in range(base64_pdf.numPages):
          pageObj = base64_pdf.getPage(page)
          text = pageObj.extractText()
          this_doc+=text
          df.loc[this_loc]=pdf_file,this_doc
          this_loc=this_loc+1
          #return df   
        df_index=df.to_dict('records')                
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="word",
            split_length=100,
            split_respect_sentence_boundary=True)
        
        preprocessed_docs = preprocessor.process(df_index)
        return preprocessed_docs
    

class Corpus(Document):
    def __init__(self):
        pass

    def new_corpus(self,name,preprocessed_docs,lst=[]):
        self.name = name
        self.preprocessed_docs = preprocessed_docs
        document_store_new = ElasticsearchDocumentStore(host="localhost", index=f'{name}',similarity="dot_product")  
        lst.append(name)              
        #document_store_new.delete_documents()
        document_store_new.write_documents(preprocessed_docs)
        return lst 
        return document_store_new
        
        
    def existing_elastic_db_store(self,preprocessed_docs):
        self.preprocessed_docs = preprocessed_docs
        document_store = ElasticsearchDocumentStore(host="localhost", index='database1',similarity="dot_product")                
        #document_store.delete_documents()
        document_store.write_documents(preprocessed_docs)
        return document_store


# def print_answers(results):
#     fields = ["answer", "score"]  # "context",
#     answers = results["answers"]
#     filtered_answers = []
    
#     for ans in answers:
#         filtered_ans = {
#             field: getattr(ans, field)
#             for field in fields
#             if getattr(ans, field) is not None
#         }
#         filtered_answers.append(filtered_ans)

#     return filtered_answers

class Query:
    def __init__(self):
        pass
    
    def retrieval(self,document_store):
        self.document_store=document_store
        retriever_es = DensePassageRetriever(document_store)
        document_store.update_embeddings(
            retriever=retriever_es)
        reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True, num_processes=0)
        pipe = ExtractiveQAPipeline(reader, retriever_es)
        return pipe
    
    def print_answer(results):
        fields = ["answer", "score"]  # "context",
        answers = results["answers"]
        filtered_answers = []
        
        for ans in answers:
            filtered_ans = {
                field: getattr(ans, field)
                for field in fields
                if getattr(ans, field) is not None
            }
            filtered_answers.append(filtered_ans)

        return filtered_answers
      
    def ask_ques(self,question,pipe):    
        self.question = question
        self.pipe = pipe
        
        result = pipe.run(query=question, params={
            "Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})
        ans = Query.print_answer(result)
        return ans
       
        
    
document_obj=Document.extract_content("C:/Users/bhati/Downloads/pdfs/document.pdf")
document_obj

corpus_obj = Corpus()

corpus_obj2 = corpus_obj.new_corpus('test2', document_obj)
corpus_obj2
corpus_obj2 = corpus_obj.new_corpus('test4', document_obj)
corpus_obj2
corpus_obj_test = corpus_obj.new_corpus('test1', document_obj)
corpus_obj_test
corpus_obj3 = corpus_obj.existing_elastic_db_store(document_obj)
    
query_obj = Query().retrieval(corpus_obj2)

ans_obj = Query().ask_ques("What is Machine Learning", query_obj)
print(ans_obj)

###################################################################################
class Driver:
    def __init__(self):
        pass
    
    def upload_a_file_into_new_db(self,name,pdf_file):
        
        self.pdf_file=pdf_file
        self.name=name

        document_obj=Document.extract_content(pdf_file)
        corpus_obj=Corpus().new_corpus(f'{name}', document_obj)
        return corpus_obj
    
    def upload_a_file_into_existing_db(self,pdf_file):
        
        self.pdf_file=pdf_file
        
        document_obj=Document.extract_content(pdf_file)
        corpus_obj=Corpus().existing_elastic_db_store(document_obj)
        return corpus_obj
    
    
    def answers(self,question,corpus_obj):
        
        self.question=question
        self.corpus_obj=corpus_obj

        query_obj = Query().retrieval(corpus_obj)
        ans_obj = Query().ask_ques(question, query_obj)
        return ans_obj
    
driver = Driver()
test = driver.upload_a_file_into_existing_db("C:/Users/bhati/Downloads/pdfs/document.pdf")
test2 = driver.upload_a_file_into_new_db('testing',"C:/Users/bhati/Downloads/pdfs/document.pdf")
test2
get_answer = driver.answers('what is BERT?', test2)





#####################
