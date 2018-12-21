from __future__ import unicode_literals
import spacy
import numpy as np
import ast
import sys


def main(input_file, output_file, dataset='en_core_web_md'):
    nlp = spacy.load(dataset)
    word2vec = lambda word: nlp.vocab[word].vector

    questions = []
    answers = []

    with open(input_file, 'r') as f:
        for line in f:
            qa_dict = ast.literal_eval(line.rstrip())

            question = qa_dict['question']
            question = list(map(word2vec, question.split()))
            question = np.sum(question, axis=0)
            questions.append(question)

            answer = qa_dict['answer']
            if '.' in answer:
                answer = answer[:answer.index('.')]
            answer = list(map(word2vec, answer.split()))
            answers.append(answer)

    question_answer_sets = []
    for question, partial_answer in zip(questions, answers):
        t_question = question
        for i in range(1, len(partial_answer)):
            t_question = np.sum([t_question, partial_answer[i - 1]], axis=0)
            question_answer_sets.append((t_question, partial_answer[i]))

    with open(output_file, 'w') as f:
        for question, answer in question_answer_sets:
            f.write(np.array_str(question) + '\n')
            f.write(np.array_str(answer) + '\n')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], dataset=sys.argv[3])
    else:
        print('incorrect number of args. Usage: python word_vectors.py input_file.json output_file.txt [dataset]')
