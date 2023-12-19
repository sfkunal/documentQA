import os
import time
from llmware.configs import LLMWareConfig
from llmware.library import Library
from llmware.retrieval import Query
from llmware.prompts import HumanInTheLoop, Prompt
import re
import logging

# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)

libraries = {
    "research_papers": "/Users/sfkunal/Code/documentQA/data/research_papers",
    "cs_lectures": "/Users/sfkunal/Code/documentQA/data/cs_lectures",
    "technical_specs": "/Users/sfkunal/Code/documentQA/data/technical_specs",
}

def run(model_name, semantic_search=True):
    print(f"\n > Loading Model: {model_name}...")

    t0 = time.time()
    prompter = Prompt().load_model(model_name, from_hf=False, api_key="")
    t1 = time.time()
    print(f"\n > Model {model_name} load time: {t1 - t0:.3f} seconds \n")

    library_name = ''
    while(library_name not in libraries.keys()):
        print(' > Which library would you like to use? Options:', list(libraries.keys()))
        library_name = input(' > Enter library name: ')

    t2 = time.time()
    library_path = libraries[library_name]
    lib = Library().create_new_library(library_name)
    lib.add_files(library_path)
    output = lib.export_library_to_jsonl_file(output_fp="/Users/sfkunal/Code/documentQA/output", output_fn="documents")
    t3 = time.time()
    print(f"\n > Library {library_name} load time: {t3-t2} seconds")

    query = input("\n > Ask something about your library: ") # Tell me about Varro's backend technical stack

    t4 = time.time()
    print (f"\n > Generating embedding vectors using FAISS ...")
    embedding_model = "mini-lm-sbert"
    lib.install_new_embedding(embedding_model_name=embedding_model, vector_db="faiss")
    t5 = time.time()
    print(f"\n > Embedding time: {t5-t4:.3f} seconds")


    t6 = time.time()
    print (f"\n > Performing a semantic query...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if semantic_search:
        query_results = Query(lib).semantic_query(query, result_count=10)
    else:
        query_results = Query(lib).text_query(query, result_count=10)
    t7 = time.time()
    print(f"\n > Semantic Query time: {t7-t6:.3f} seconds \n")


    t8 = time.time()
    print (f"\n > Prompting LLM with '{query}'")
    sources = prompter.add_source_query_results(query_results)
    responses = prompter.prompt_with_source(query, prompt_name="default_with_context", temperature=0.3)
    t9 = time.time()
    print(f" > LLM Prompt time: {t9-t8:.3f} seconds \n")
    for response in responses:
        print (" > LLM response\n" + response["llm_response"])
    
    
    print (f"\n > Generating CSV report...")
    report_data = prompter.send_to_human_for_review("/Users/sfkunal/Code/documentQA/output", "report.csv")
    print ("File: " + report_data["report_fp"] + "\n")

    return 0


if __name__ == "__main__":

    model_list = ["llmware/bling-1b-0.1",
                  "gpt-3.5-turbo-instruct",
                  "claude-instant-v1"
                  ]

    run(model_list[1], semantic_search=False)