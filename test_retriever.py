# retrieval tests, no key needed
from retriever import Retriever
from knowledge import DOCUMENTS


questions_refused = 0;

TEST_QUESTIONS = [
    "What time does the library close on friday?",
    "How do I pay my tuition?",
    "Tell me any information about the study rooms?",
    "How many peaple can go to the study room?",
    "Where can I register for classes?",
    "How can I see my grades for my classes?",
    "Where can I buy school supplies?",
    "Any information for career services?",
    "Any open career fairs happening this week?",
    "Any information on where to check for my grades for my classes?"
]

my_retriever = Retriever(DOCUMENTS, 0.5);
for question in TEST_QUESTIONS:
    hits = my_retriever.search(question);
    print();
    print("Question : " + question)
    if (len(hits) > 0):
         print(my_retriever.build_context(hits))
    else:
         print("REJECTED")
         questions_refused = questions_refused + 1;


print(f"Questions refused : {questions_refused}")