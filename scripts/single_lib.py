import os
import time
from llmware.configs import LLMWareConfig
from llmware.library import Library
from llmware.prompts import HumanInTheLoop, Prompt
import re
import logging

logging.basicConfig(level=logging.INFO)

def custom(model_name, from_hf=False):
    print(f"\n > Loading Model: {model_name}...")

    t0 = time.time()
    prompter = Prompt().load_model(model_name, from_hf=from_hf, api_key="")
    t1 = time.time()

    print(f"\n > Model {model_name} load time: {t1-t0} seconds \n")

    t2 = time.time()
    library_path = "/Users/sfkunal/Code/documentQA/data/research_papers"
    lib = Library().create_new_library("my_lib")
    lib.add_files(library_path)
    t3 = time.time()
    print(f"\n > Library {library_path} load time: {t3-t2} seconds \n")

    output = lib.export_library_to_jsonl_file(output_fp="/Users/sfkunal/Code/documentQA/output", output_fn="documents")

    t4 = time.time()
    for i, doc in enumerate(os.listdir(library_path)):
        if doc != ".DS_Store":
            print(f" > Indexing Document {i+1}: {doc}")
            source = prompter.add_source_document(library_path, doc)
    t5 = time.time()
    print(f"\n > Document Indexing time: {t5-t4} seconds \n")

    query = input("\n > Enter a query: ")


    responses = prompter.prompt_with_source(query, prompt_name="just_the_facts", temperature=0.3)
    for r, response in enumerate(responses):
        print("Response:", re.sub("[\n]", " ", response["llm_response"]).strip())
    prompter.clear_source_materials()

    print("\nPrompt state saved at: ", os.path.join(LLMWareConfig.get_prompt_path(), prompter.prompt_id))
    prompter.save_state()
    csv_output = HumanInTheLoop(prompter).export_current_interaction_to_csv()
    print("csv output - ", csv_output)

    return 0


if __name__ == "__main__":

    model_list = ["llmware/bling-1b-0.1",
                  "gpt-3.5-turbo-instruct",
                  "gpt-4"
                  ]

    custom(model_list[0])
