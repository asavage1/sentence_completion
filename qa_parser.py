import ast
import sys


def main(input_file, output_file, delimiter='\n'):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as w:
            for line in f:
                qa_dict = ast.literal_eval(line.rstrip())

                question = qa_dict['question']
                answer = qa_dict['answer']
                # answer = answer[:answer.index('.')]

                w.write(question + delimiter + answer + '\n')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], delimiter=sys.argv[3])
    else:
        print('incorrect number of args. Usage: python qa_parser.py input_file.json output_file.txt [delimiter]')
