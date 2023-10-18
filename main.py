import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time
import psutil
import multiprocessing

app = FastAPI()
print(torch.version.cuda)
device = torch.device("cpu")
assert torch.cuda.is_available(), "CUDA is not available. Ensure you're running this on a machine with a CUDA-capable GPU."

# Define the request body model
class Text(BaseModel):
    content: str

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')

@app.post("/get_embedding/")
def get_embedding(text: Text, model):
    try:
        t1 = time.time()
        # Tokenize the user input
        batch_dict = tokenizer([text.content], max_length=512, padding=True, truncation=True, return_tensors='pt')

        # Get model outputs
        outputs = model(**batch_dict)

        # Pooling
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Extract embeddings for the input text
        embeddings = embeddings.tolist()[0]
        print(time.time() - t1)
        return JSONResponse(content={"embedding": embeddings})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




text = """
 Actors do not represent a specific user—they represent roles that
 users adopt. If a user has adopted the respective role, this user is autho
rized to execute the use cases associated with this role. Specific users
 can adopt and set aside multiple roles simultaneously. For example, a
 person can be involved in the submission of a certain assignment as an
 assistant and in another assignment as a student. The role concept is
 also used in other types of UML diagrams, such as the class diagram
 (see Chapter 4), the sequence diagram (see Chapter 6), and the activity
 diagram (see Chapter 7).
 3.4 Relationships between Actors
 Synonyms:
 • Generalization
 • Inheritance
 Generalization for actors
 X
 Figure 3.6
 Example of generalization
 for actors
 Actors often have common properties and some use cases can be used
 by various actors. For example, it is possible that not only professors
 but also assistants (i.e., the entire research personnel) are permitted to
 view student data. To express this, actors may be depicted in an inher
itance relationship (generalization) with one another. When an actor Y
 (sub-actor) inherits from an actor X (super-actor), Y is involved with all
 use cases with which X is involved. In simple terms, generalization ex
presses an “is a” relationship. It is represented with a line from the sub
Y
 Student Administration
 Query
 student data
 Issue
 certificate
 Create
 course
 0..1
 Research
 Associate
 Professor
 Publish
 task
 Assistant
3.4 Relationships between Actors
 29
 actor to the super-actor with a large triangular arrowhead at the super
actor end. In the example in Figure 3.6, the actors Professor and Assis
tant inherit from the actor Research Associate, which means that every
 professor and every assistant is a research associate. Every research as
sociate can execute the use case Query student data. Only professors can
 create a new course (use case Create course); in contrast, tasks can only
 be published by assistants (use case Publish task). To execute the use
 case Issue certificate in Figure 3.6, an actor Professor is required; in ad
dition, an actor Assistant can be involved optionally, which is expressed
 by the multiplicity 0..1.
 There is a great difference between two actors participating in a use
 case themselves and two actors having a common super-actor that par
ticipates in the use case. In the first case, both actors must participate in
 the use case (see Fig. 3.7(a)); in the second case, each of them inherits
 the association. Then each actor participates in the use case individually
"""




def run_test():
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
    counter = 0
    for _ in range(5):  # Running the function 25 times, for example.
        counter += 1
        print(get_embedding(Text(content=f"Hello world {counter} {text}"), model).body)

if __name__ == "__main__":
    # Create a new process to run the test
    test_process = multiprocessing.Process(target=run_test)

    # Start the process
    test_process.start()

    # Set the process's CPU affinity to the first CPU core using psutil
    process = psutil.Process(test_process.pid)
    process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Wait for the process to complete
    test_process.join()